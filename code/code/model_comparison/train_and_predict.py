from __future__ import annotations

import argparse
import gc
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from model_workflow_common import (
    ALL_DEEP_MODELS,
    CANONICAL,
    CANONICAL_LOCK,
    CANONICAL_PREDICTIONS,
    FIXED,
    PREDICTION_FILENAMES,
    RUN_ROOT,
    SEEDS,
    SLUGS,
    build_model,
    canonical_key_frame,
    load_config,
    load_evaluation_data,
    load_pre_evaluation_data,
    log,
    make_prediction_frame,
    now_iso,
    parameter_count,
    predict_deep,
    read_json,
    sha256_file,
    train_deep,
    validate_public_config,
    write_json,
)


DEFAULT_CONFIG = str(Path(__file__).resolve().parents[2] / "configs" / "config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit comparison models before Evaluation access, then generate "
            "Evaluation predictions in a separate locked stage."
        )
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("fit", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def checkpoint_path(model_name: str, seed: int) -> Path:
    return RUN_ROOT / "models" / SLUGS[model_name] / f"seed_{seed}" / "checkpoint.pt"


def selected_hp(lock: dict, model_name: str, config) -> tuple[int, float, str]:
    if model_name in ("No Attention", "No CNN", "No BiLSTM"):
        return (
            int(config.training["hidden_size"]),
            float(config.training["learning_rate"]),
            "fixed canonical optimization values",
        )
    selected = lock["selected_hyperparameters"][model_name]
    return int(selected["width"]), float(selected["learning_rate"]), "validation-only grid"


def train_or_reuse_checkpoint(
    data, hp_lock: dict, hp_lock_sha: str, model_name: str, seed: int, config
):
    width, lr, tuning = selected_hp(hp_lock, model_name, config)
    path = checkpoint_path(model_name, seed)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        required = (
            saved.get("model_name") == model_name
            and saved.get("seed") == seed
            and saved.get("width") == width
            and saved.get("learning_rate") == lr
            and saved.get("hp_lock_sha256") == hp_lock_sha
            and saved.get("data_hashes") == data.data_hashes()
        )
        if not required:
            raise RuntimeError(f"Existing checkpoint identity mismatch: {path}")
        log(f"FINAL CHECKPOINT REUSE model={model_name} seed={seed}")
        return saved

    log(f"FINAL TRAIN START model={model_name} seed={seed} width={width} lr={lr}")
    result = train_deep(
        data,
        model_name=model_name,
        width=width,
        learning_rate=lr,
        seed=seed,
        max_epochs=int(config.training["max_epochs"]),
        patience=int(config.training["early_stopping_patience"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["history"]).to_csv(
        path.parent / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    saved = {
        "stage": "workflow_final_validation_selected_checkpoint",
        "created_at": now_iso(),
        "model_name": model_name,
        "seed": seed,
        "width": width,
        "learning_rate": lr,
        "tuning_method": tuning,
        "max_epochs": int(config.training["max_epochs"]),
        "early_stopping_patience": int(config.training["early_stopping_patience"]),
        "best_epoch": result["best_epoch"],
        "epochs_run": result["epochs_run"],
        "best_val_mse_scaled": result["best_val_mse_scaled"],
        "best_val_mse_gC2": result["best_val_mse_gC2"],
        "runtime_seconds": result["runtime_seconds"],
        "parameter_count": result["parameter_count"],
        "model_state_dict": result["state_dict"],
        "hp_lock_sha256": hp_lock_sha,
        "data_identity": data.data_identity,
        "data_hashes": data.data_hashes(),
        "evaluation_evaluated": False,
    }
    torch.save(saved, path)
    log(
        f"FINAL TRAIN DONE model={model_name} seed={seed} "
        f"best_epoch={saved['best_epoch']} val_mse_gC2={saved['best_val_mse_gC2']:.17g}"
    )
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def assert_evaluation_key_order(evaluation_data, canonical: pd.DataFrame) -> None:
    expected = canonical.loc[
        canonical["seed"] == 42, ["point_id", "Year", "Month"]
    ].reset_index(drop=True)
    actual = evaluation_data.evaluation_index[
        ["point_id", "Year", "Month"]
    ].reset_index(drop=True)
    if not expected.equals(actual):
        raise RuntimeError("Evaluation data order/keys differ from canonical predictions")


def _paths() -> tuple[Path, Path]:
    return (
        RUN_ROOT / "locks" / "workflow_selected_hyperparameters_lock.json",
        RUN_ROOT / "locks" / "workflow_evaluation_start_lock.json",
    )


def _load_hp_lock(config):
    hp_lock_path, _ = _paths()
    if not hp_lock_path.exists():
        raise FileNotFoundError(
            f"Hyperparameter lock not found: {hp_lock_path}. Run Validation-only tuning first."
        )
    hp_lock = read_json(hp_lock_path)
    if hp_lock.get("selection_data_scope") != "train_validation_only":
        raise RuntimeError("HP lock does not identify the Train/Validation selection scope")
    data = load_pre_evaluation_data()
    if data.data_hashes() != hp_lock["data_hashes"] or data.data_identity != hp_lock["data_identity"]:
        raise RuntimeError("HP lock data lineage mismatch")
    return hp_lock_path, hp_lock, sha256_file(hp_lock_path), data


def _validate_evaluation_lock(lock: dict, hp_lock_sha: str) -> None:
    if lock.get("stage") != "workflow_evaluation_start_lock":
        raise RuntimeError("Evaluation-start lock has the wrong stage identity")
    if lock.get("hp_lock_sha256") != hp_lock_sha:
        raise RuntimeError("Evaluation-start lock hyperparameter lineage mismatch")
    if lock.get("selection_data_scope") != "train_validation_only":
        raise RuntimeError("Evaluation-start lock does not identify the Train/Validation selection scope")
    if lock.get("deep_seed_completeness") != list(SEEDS):
        raise RuntimeError("Evaluation-start lock seed set mismatch")

    for record in lock.get("all_deep_checkpoints", []):
        path = Path(record["path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Locked checkpoint changed or is missing: {path}")
    expected_count = len(ALL_DEEP_MODELS) * len(SEEDS)
    if len(lock.get("all_deep_checkpoints", [])) != expected_count:
        raise RuntimeError("Evaluation-start lock does not contain all deep checkpoints")

    for path_key, hash_key in (
        ("rf_model", "rf_model_sha256"),
        ("rf_metadata", "rf_metadata_sha256"),
        ("climatology_means", "climatology_means_sha256"),
    ):
        path = Path(lock[path_key])
        if not path.exists() or sha256_file(path) != lock[hash_key]:
            raise RuntimeError(f"Locked artifact changed or is missing: {path}")


def _write_model_size_and_runtime(config, hp_lock: dict, rf_meta: dict, runtime_rows: list[dict]) -> None:
    full_count = parameter_count(
        build_model("Full canonical", config.input_features, int(config.training["hidden_size"]))
    )
    canonical_count = parameter_count(
        CANONICAL.CNNBiLSTMAttention(config.input_features, int(config.training["hidden_size"]))
    )
    if full_count != canonical_count:
        raise RuntimeError("Local full-architecture parameter count differs from the canonical architecture")
    size_rows = [
        {
            "model": "Full canonical",
            "parameter_count": full_count,
            "model_size": "",
            "source": "canonical W36 architecture",
        }
    ]
    for model_name in ALL_DEEP_MODELS:
        width, _, _ = selected_hp(hp_lock, model_name, config)
        size_rows.append(
            {
                "model": model_name,
                "parameter_count": parameter_count(build_model(model_name, config.input_features, width)),
                "model_size": "",
                "source": "exact trainable parameters",
            }
        )
    size_rows.extend(
        [
            {
                "model": "Random Forest",
                "parameter_count": "N/A",
                "model_size": (
                    f"{rf_meta['tree_count']} trees; "
                    f"{rf_meta['total_fitted_nodes']} total fitted nodes"
                ),
                "source": "fitted deterministic model",
            },
            {
                "model": "point-month climatology",
                "parameter_count": "N/A",
                "model_size": "2,964 stored point-month means",
                "source": "training-period means",
            },
        ]
    )
    pd.DataFrame(size_rows).to_csv(
        RUN_ROOT / "results" / "workflow_parameter_counts_and_model_size.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grid_runtime = pd.read_csv(RUN_ROOT / "models" / "validation_grid_complete.csv")
    pd.concat([grid_runtime, pd.DataFrame(runtime_rows)], ignore_index=True).to_csv(
        RUN_ROOT / "results" / "workflow_training_epochs_and_runtime.csv",
        index=False,
        encoding="utf-8-sig",
    )


def fit_stage(config) -> None:
    """Train/final-fit every comparison model without loading Evaluation values."""
    for name in ("models", "predictions", "results", "locks", "logs"):
        (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)

    hp_lock_path, hp_lock, hp_lock_sha, data = _load_hp_lock(config)
    _, eval_lock_path = _paths()

    runtime_rows: list[dict] = []
    checkpoint_records: list[dict] = []
    for model_name in ALL_DEEP_MODELS:
        for seed in SEEDS:
            saved = train_or_reuse_checkpoint(data, hp_lock, hp_lock_sha, model_name, seed, config)
            path = checkpoint_path(model_name, seed)
            checkpoint_records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "best_epoch": saved["best_epoch"],
                    "best_val_mse_gC2": saved["best_val_mse_gC2"],
                }
            )
            runtime_rows.append(
                {
                    "stage": "final_deep_training",
                    "model": model_name,
                    "seed": seed,
                    "width": saved["width"],
                    "learning_rate": saved["learning_rate"],
                    "best_epoch": saved["best_epoch"],
                    "epochs_run": saved["epochs_run"],
                    "runtime_seconds": saved["runtime_seconds"],
                }
            )

    rf_hp = hp_lock["selected_hyperparameters"]["Random Forest"]
    rf_path = RUN_ROOT / "models" / "baseline_random_forest" / "random_forest.joblib"
    rf_meta_path = rf_path.with_suffix(".json")
    if rf_path.exists() and rf_meta_path.exists():
        rf_meta = read_json(rf_meta_path)
        if rf_meta.get("hp_lock_sha256") != hp_lock_sha or rf_meta.get("data_hashes") != data.data_hashes():
            raise RuntimeError("Existing RF model lineage mismatch")
        rf = joblib.load(rf_path)
        log("FINAL RF REUSE")
    else:
        log(f"FINAL RF FIT START selected_hp={rf_hp}")
        X_train_flat = np.asarray(data.X_train, dtype=np.float32).reshape(len(data.X_train), -1)
        started = time.perf_counter()
        rf = RandomForestRegressor(
            n_estimators=int(rf_hp["n_estimators"]),
            max_depth=rf_hp["max_depth"],
            min_samples_leaf=int(rf_hp["min_samples_leaf"]),
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        )
        rf.fit(X_train_flat, data.y_train)
        rf_runtime = time.perf_counter() - started
        rf_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, rf_path, compress=3)
        rf_meta = {
            "stage": "workflow_final_rf",
            "created_at": now_iso(),
            "selected_hyperparameters": rf_hp,
            "max_features_fixed_project_definition": "sqrt",
            "runtime_seconds": rf_runtime,
            "tree_count": len(rf.estimators_),
            "total_fitted_nodes": int(sum(tree.tree_.node_count for tree in rf.estimators_)),
            "hp_lock_sha256": hp_lock_sha,
            "data_hashes": data.data_hashes(),
            "evaluation_evaluated": False,
        }
        write_json(rf_meta_path, rf_meta)
        log(f"FINAL RF FIT DONE trees={rf_meta['tree_count']} nodes={rf_meta['total_fitted_nodes']}")
        del X_train_flat
        gc.collect()
    runtime_rows.append(
        {
            "stage": "final_deterministic_fit",
            "model": "Random Forest",
            "seed": "N/A",
            "width": "",
            "learning_rate": "",
            "best_epoch": "",
            "epochs_run": "",
            "runtime_seconds": rf_meta["runtime_seconds"],
        }
    )

    sample_index = pd.read_csv(data._paths["Sample_index_aligned.csv"])
    train_index = sample_index.loc[
        sample_index["split"].astype(str).str.lower() == "train"
    ].reset_index(drop=True)
    if len(train_index) != len(data.y_train):
        raise RuntimeError("Climatology train-index alignment mismatch")
    clim_train = train_index[["point_id", "Month"]].copy()
    clim_train["y_g"] = data.y_scaled_to_g(data.y_train)
    climatology = (
        clim_train.groupby(["point_id", "Month"], sort=True, as_index=False)["y_g"]
        .mean()
        .rename(columns={"y_g": "point_month_mean_gC_m2_month"})
    )
    if len(climatology) != 2964 or climatology["point_month_mean_gC_m2_month"].isna().any():
        raise RuntimeError("Climatology does not contain exactly 2,964 finite means")
    clim_path = RUN_ROOT / "models" / "baseline_point_month_climatology" / "means.csv"
    clim_path.parent.mkdir(parents=True, exist_ok=True)
    climatology.to_csv(clim_path, index=False, encoding="utf-8-sig")

    eval_lock = {
        "stage": "workflow_evaluation_start_lock",
        "created_at": now_iso(),
        "hp_lock": str(hp_lock_path.resolve()),
        "hp_lock_sha256": hp_lock_sha,
        "all_deep_checkpoints": checkpoint_records,
        "rf_model": str(rf_path.resolve()),
        "rf_model_sha256": sha256_file(rf_path),
        "rf_metadata": str(rf_meta_path.resolve()),
        "rf_metadata_sha256": sha256_file(rf_meta_path),
        "climatology_means": str(clim_path.resolve()),
        "climatology_means_sha256": sha256_file(clim_path),
        "deep_seed_completeness": list(SEEDS),
        "selection_data_scope": "train_validation_only",
                "data_identity": data.data_identity,
        "data_hashes": data.data_hashes(),
    }
    if eval_lock_path.exists():
        existing = read_json(eval_lock_path)
        _validate_evaluation_lock(existing, hp_lock_sha)
        # Require exact checkpoint identities together with the HP lock.
        if existing.get("all_deep_checkpoints") != checkpoint_records:
            raise RuntimeError("Existing evaluation-start lock does not match current checkpoints")
        log("EVALUATION START LOCK REUSED")
    else:
        write_json(eval_lock_path, eval_lock)
        log(f"EVALUATION START LOCK CREATED sha256={sha256_file(eval_lock_path)}")

    _write_model_size_and_runtime(config, hp_lock, rf_meta, runtime_rows)
    log("PRE-EVALUATION COMPARISON FIT STAGE COMPLETE")


def evaluate_stage(config) -> None:
    """Generate Evaluation predictions only after all comparison decisions are locked."""
    for name in ("predictions", "results", "locks", "logs"):
        (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)

    hp_lock_path, eval_lock_path = _paths()
    if not hp_lock_path.exists() or not eval_lock_path.exists():
        raise FileNotFoundError("Run Validation-only tuning and the comparison fit stage before Evaluation.")
    hp_lock = read_json(hp_lock_path)
    hp_lock_sha = sha256_file(hp_lock_path)
    eval_lock = read_json(eval_lock_path)
    _validate_evaluation_lock(eval_lock, hp_lock_sha)

    # The canonical Evaluation is itself locked and is first accessed here.
    evaluation_data = load_evaluation_data()
    if evaluation_data.data_hashes() != eval_lock["data_hashes"] or evaluation_data.data_identity != eval_lock["data_identity"]:
        raise RuntimeError("Evaluation data do not match the pre-Evaluation comparison lock")
    canonical = canonical_key_frame()
    assert_evaluation_key_order(evaluation_data, canonical)

    full_out = RUN_ROOT / "predictions" / PREDICTION_FILENAMES["Full canonical"]
    if not full_out.exists():
        shutil.copy2(CANONICAL_PREDICTIONS, full_out)
    canonical_lock = read_json(CANONICAL_LOCK)
    recorded_hash = (canonical_lock.get("evaluation_outputs") or {}).get("predictions_all_seeds_sha256")
    if not recorded_hash or sha256_file(full_out) != recorded_hash:
        raise RuntimeError("Copied canonical full predictions do not match the current canonical run lock")

    records = {(row["model"], int(row["seed"])): row for row in eval_lock["all_deep_checkpoints"]}
    for model_name in ALL_DEEP_MODELS:
        frames = []
        width, _, _ = selected_hp(hp_lock, model_name, config)
        for seed in SEEDS:
            record = records[(model_name, seed)]
            path = Path(record["path"])
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(model_name, evaluation_data.X_evaluation.shape[2], width)
            model.load_state_dict(saved["model_state_dict"])
            pred_scaled = predict_deep(model, evaluation_data.X_evaluation)
            frames.append(make_prediction_frame(model_name, seed, evaluation_data, pred_scaled))
            del model, pred_scaled, saved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        output = pd.concat(frames, ignore_index=True)
        output.to_csv(
            RUN_ROOT / "predictions" / PREDICTION_FILENAMES[model_name],
            index=False,
            encoding="utf-8-sig",
        )
        log(f"EVALUATION PREDICTIONS WRITTEN model={model_name} rows={len(output)}")

    rf = joblib.load(Path(eval_lock["rf_model"]))
    X_eval_flat = np.asarray(evaluation_data.X_evaluation, dtype=np.float32).reshape(
        len(evaluation_data.X_evaluation), -1
    )
    rf_pred_scaled = rf.predict(X_eval_flat)
    rf_frame = make_prediction_frame("Random Forest", None, evaluation_data, rf_pred_scaled)
    rf_frame.to_csv(
        RUN_ROOT / "predictions" / PREDICTION_FILENAMES["Random Forest"],
        index=False,
        encoding="utf-8-sig",
    )

    climatology = pd.read_csv(Path(eval_lock["climatology_means"]))
    clim_eval = evaluation_data.evaluation_index[["point_id", "Month"]].merge(
        climatology, on=["point_id", "Month"], how="left", validate="many_to_one"
    )
    if clim_eval["point_month_mean_gC_m2_month"].isna().any():
        raise RuntimeError("Missing climatology prediction")
    clim_scaled = (
        clim_eval["point_month_mean_gC_m2_month"].to_numpy(dtype=float) / 1000.0
        - evaluation_data.y_mean
    ) / evaluation_data.y_scale
    clim_frame = make_prediction_frame("point-month climatology", None, evaluation_data, clim_scaled)
    clim_frame.to_csv(
        RUN_ROOT / "predictions" / PREDICTION_FILENAMES["point-month climatology"],
        index=False,
        encoding="utf-8-sig",
    )
    log("COMPARISON EVALUATION PREDICTIONS WRITTEN")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_public_config(config)
    if args.stage == "fit":
        fit_stage(config)
    else:
        evaluate_stage(config)


if __name__ == "__main__":
    main()
