"""Reproduce the five-fold unseen-site retraining and OOF prediction workflow.

This module implements the coordinate-clustered unseen-site training and
Evaluation protocol.

Protocol
--------
* Read the fixed point_id-to-fold mapping and join the 247-site coordinates.
* For each held-out fold, exclude those sites from both Train (2005--2018) and
  Validation (2019--2021).
* Refit X and y scalers separately for each fold using training sites only.
  The X scaler uses the ``MAX48-fit-then-W36-slice`` protocol:
  fit population mean/SD on the first nine continuous features of every
  training sample's 48-month history, then train/evaluate on the final 36
  months of those histories. Categorical and calendar features retain their
  original coding.
* Train the fixed CNN--BiLSTM--Attention specification for seeds
  42, 2024 and 3407 with Validation-only early stopping.
* The ``fit`` stage uses Train and Validation records. The ``evaluate`` stage
  loads the locked checkpoints, predicts each held-out fold,
  concatenates the 35,568 OOF records and writes the standard summaries.

The workflow reads public site coordinates locally and writes predictions
and checkpoints to the configured runtime directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CODE_ROOT.parent
MODEL_COMPARISON_DIR = CODE_ROOT / "model_comparison"
EVALUATION_DIR = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, MODEL_COMPARISON_DIR, EVALUATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import canonical_pipeline as canonical  # noqa: E402
from model_workflow_common import load_config, validate_public_config  # noqa: E402
from unseen_site_evaluation import (  # noqa: E402
    FOLD_SITE_COUNTS,
    SEEDS,
    load_fixed_folds,
    summarise_predictions,
)


WINDOW = 36
MAX_WINDOW = 48
CONTINUOUS_FEATURE_COUNT = 9
EXPECTED_ROWS = 62244
EXPECTED_SITES = 247
EXPECTED_SPLIT_COUNTS = {"train": 41496, "val": 8892, "test": 11856}
EXPECTED_FEATURES = [
    "LST_Day_1km",
    "NDVI",
    "Precipitation_mm",
    "VPD",
    "surface_solar_radiation_downwards_sum",
    "temperature_2m",
    "volumetric_soil_water_layer_1",
    "elevation",
    "slope",
    "Aspect_Sin",
    "Aspect_Cos",
    "Veg_Coniferous",
    "Veg_Shrub",
    "Veg_Meadow_497",
    "Veg_Meadow_499",
    "Veg_Meadow_504",
    "Veg_Sparse",
    "Target_Month_Sin",
    "Target_Month_Cos",
]


@dataclass
class FoldTrainingBundle:
    """Train/Validation-only data view validated by the canonical trainer."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_mean: float
    y_scale: float

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[2])


@dataclass
class PreparedInputs:
    data_dir: Path
    sample_index: pd.DataFrame
    x48: np.ndarray
    y_raw: np.ndarray


def _normalise_point_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)


def _load_site_table(path: Path, assignment_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"point_id": str})
    return load_fixed_folds(frame, assignment_path)


def _load_prepared(data_dir: Path) -> PreparedInputs:
    data_dir = Path(data_dir).resolve()
    required = {
        "sample_index": data_dir / "Sample_index_aligned.csv",
        "x48": data_dir / "X_tensor_aligned_48.npy",
        "y_raw": data_dir / "Y_tensor_raw.npy",
        "metadata": data_dir / "preprocessing_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared unseen-site inputs are missing: {missing}")

    index = pd.read_csv(required["sample_index"], dtype={"point_id": str})
    index["point_id"] = _normalise_point_id(index["point_id"])
    for column in ("sample_id", "Year", "Month", "Lon", "Lat"):
        index[column] = pd.to_numeric(index[column], errors="raise")
    if len(index) != EXPECTED_ROWS or index["point_id"].nunique() != EXPECTED_SITES:
        raise ValueError("Prepared sample-index row/site identity does not match the fixed workflow")
    if index["sample_id"].astype(int).tolist() != list(range(EXPECTED_ROWS)):
        raise ValueError("sample_id must be the contiguous prepared-array row index 0..62243")
    if index.duplicated(["point_id", "Year", "Month"]).any():
        raise ValueError("Prepared sample index contains duplicate point_id-Year-Month keys")
    observed_split_counts = index["split"].astype(str).str.lower().value_counts().to_dict()
    if observed_split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Prepared split counts differ from the fixed workflow: {observed_split_counts}"
        )

    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    if metadata.get("feature_names") != EXPECTED_FEATURES:
        raise ValueError("Prepared feature order differs from the fixed 19-feature definition")

    x48 = np.load(required["x48"], mmap_mode="r")
    y_raw = np.load(required["y_raw"], mmap_mode="r")
    if x48.shape != (EXPECTED_ROWS, MAX_WINDOW, len(EXPECTED_FEATURES)):
        raise ValueError(f"Unexpected X48 shape: {x48.shape}")
    if y_raw.shape != (EXPECTED_ROWS,):
        raise ValueError(f"Unexpected raw-target shape: {y_raw.shape}")
    return PreparedInputs(data_dir=data_dir, sample_index=index, x48=x48, y_raw=y_raw)


def _validate_site_coordinates(index: pd.DataFrame, folds: pd.DataFrame) -> None:
    sample_coords = (
        index.groupby("point_id", as_index=False, observed=True)[["Lon", "Lat"]].first()
    )
    coordinate_counts = index.groupby("point_id", observed=True)[["Lon", "Lat"]].nunique()
    if not (coordinate_counts == 1).all().all():
        raise ValueError("Prepared sample index contains within-site coordinate drift")
    merged = folds.merge(
        sample_coords,
        on="point_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_site", "_prepared"),
    )
    if len(merged) != EXPECTED_SITES or merged.isna().any().any():
        raise ValueError("Public site table and prepared sample index do not cover the same 247 sites")
    for name in ("Lon", "Lat"):
        if not np.allclose(
            merged[f"{name}_site"].to_numpy(float),
            merged[f"{name}_prepared"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{name} differs between public site table and prepared data")


def _fold_masks(index: pd.DataFrame, folds: pd.DataFrame, fold_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heldout = set(folds.loc[folds["fold_id"] == fold_id, "point_id"])
    training = set(folds["point_id"]) - heldout
    point_id = index["point_id"]
    year = index["Year"].to_numpy(dtype=int)
    train = point_id.isin(training).to_numpy() & ((year >= 2005) & (year <= 2018))
    val = point_id.isin(training).to_numpy() & ((year >= 2019) & (year <= 2021))
    evaluation = point_id.isin(heldout).to_numpy() & ((year >= 2022) & (year <= 2025))

    numeric_id = int(fold_id.split("_")[-1])
    heldout_count = FOLD_SITE_COUNTS[numeric_id]
    training_count = EXPECTED_SITES - heldout_count
    expected = {
        "train": training_count * 14 * 12,
        "val": training_count * 3 * 12,
        "evaluation": heldout_count * 4 * 12,
    }
    observed = {
        "train": int(train.sum()),
        "val": int(val.sum()),
        "evaluation": int(evaluation.sum()),
    }
    if observed != expected:
        raise ValueError(f"{fold_id} split counts differ from the fixed protocol: {observed} vs {expected}")
    if np.any(train & point_id.isin(heldout).to_numpy()) or np.any(val & point_id.isin(heldout).to_numpy()):
        raise ValueError(f"{fold_id} held-out sites leaked into Train/Validation")
    return train, val, evaluation


def _fit_fold_scalers(prepared: PreparedInputs, train_ids: np.ndarray) -> dict[str, Any]:
    # Fit training-site scaler statistics over the 48-month input histories.
    z_fit = np.asarray(
        prepared.x48[train_ids, :, :CONTINUOUS_FEATURE_COUNT], dtype=np.float32
    ).reshape(-1, CONTINUOUS_FEATURE_COUNT)
    mean_z = z_fit.mean(axis=0, dtype=np.float64)
    std_z = z_fit.std(axis=0, dtype=np.float64)
    if not np.isfinite(mean_z).all() or not np.isfinite(std_z).all() or np.any(std_z <= 0):
        raise ValueError("Fold-specific X scaler is non-finite or degenerate")

    y_fit = np.asarray(prepared.y_raw[train_ids], dtype=np.float64)
    y_mean = float(y_fit.mean())
    y_std = float(y_fit.std(ddof=0))
    if not np.isfinite(y_mean) or not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("Fold-specific y scaler is non-finite or degenerate")
    return {
        "x_mean_in_global_z": mean_z,
        "x_std_in_global_z": std_z,
        "y_mean": y_mean,
        "y_std": y_std,
        "x_fit_sample_count": int(len(train_ids)),
        "x_fit_timestep_count": int(len(train_ids) * MAX_WINDOW),
    }


def _transform_x(prepared: PreparedInputs, row_ids: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    # Apply fold-specific scaling to the final 36 months.
    values = np.asarray(prepared.x48[row_ids, -WINDOW:, :], dtype=np.float32).copy()
    mean_z = np.asarray(scaler["x_mean_in_global_z"], dtype=np.float32).reshape(1, 1, -1)
    std_z = np.asarray(scaler["x_std_in_global_z"], dtype=np.float32).reshape(1, 1, -1)
    values[:, :, :CONTINUOUS_FEATURE_COUNT] = (
        values[:, :, :CONTINUOUS_FEATURE_COUNT] - mean_z
    ) / std_z
    if not np.isfinite(values).all():
        raise ValueError("Non-finite fold-specific X values")
    return values


def _scaled_y(prepared: PreparedInputs, row_ids: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    values = np.asarray(prepared.y_raw[row_ids], dtype=np.float64)
    return ((values - scaler["y_mean"]) / scaler["y_std"]).astype(np.float32)


def _json_scaler(scaler: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "MAX48-fit-then-W36-slice",
        "continuous_feature_names": EXPECTED_FEATURES[:CONTINUOUS_FEATURE_COUNT],
        "continuous_indices": list(range(CONTINUOUS_FEATURE_COUNT)),
        "x_mean_in_global_z": np.asarray(scaler["x_mean_in_global_z"], dtype=float).tolist(),
        "x_std_in_global_z": np.asarray(scaler["x_std_in_global_z"], dtype=float).tolist(),
        "x_fit_sample_count": int(scaler["x_fit_sample_count"]),
        "x_fit_timestep_count": int(scaler["x_fit_timestep_count"]),
        "y_mean": float(scaler["y_mean"]),
        "y_std": float(scaler["y_std"]),
        "fit_scope": "training sites only; target years 2005-2018; population SD (ddof=0)",
        "heldout_site_contribution": 0,
        "validation_contribution": 0,
        "evaluation_contribution": 0,
    }


def _checkpoint_path(output_dir: Path, fold_id: str, seed: int) -> Path:
    return output_dir / "models" / fold_id / f"seed_{seed}" / "checkpoint.pt"


def _write_fold_map(output_dir: Path, folds: pd.DataFrame) -> None:
    path = output_dir / "fold_map_point_id_only.csv"
    expected = folds[["point_id", "fold_id"]].sort_values("point_id", kind="mergesort").reset_index(drop=True)
    if path.exists():
        current = pd.read_csv(path, dtype={"point_id": str})
        current["point_id"] = _normalise_point_id(current["point_id"])
        if not current.equals(expected):
            raise RuntimeError(f"Existing fold map conflicts with current public coordinates: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    expected.to_csv(path, index=False)


def fit_stage(config_path: Path, site_table: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    validate_public_config(config)
    prepared = _load_prepared(config.path("processed_data"))
    folds = _load_site_table(site_table, config.path("unseen_site_folds"))
    _validate_site_coordinates(prepared.sample_index, folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_fold_map(output_dir, folds)

    manifest: dict[str, Any] = {
        "workflow": "coordinate_clustered_unseen_site_oof",
        "stage": "fit_complete",
        "window": WINDOW,
        "max_window_scaler_fit": MAX_WINDOW,
        "seeds": list(SEEDS),
        "hidden_size": int(config.training["hidden_size"]),
        "learning_rate": float(config.training["learning_rate"]),
        "train_years": [2005, 2018],
        "validation_years": [2019, 2021],
        "evaluation_years": [2022, 2025],
        "folds": {},
        "evaluation_metrics_used_for_training_or_selection": False,
    }

    for numeric_fold in range(1, 6):
        fold_id = f"fold_{numeric_fold}"
        train_mask, val_mask, _ = _fold_masks(prepared.sample_index, folds, fold_id)
        train_ids = prepared.sample_index.loc[train_mask, "sample_id"].to_numpy(dtype=np.int64)
        val_ids = prepared.sample_index.loc[val_mask, "sample_id"].to_numpy(dtype=np.int64)
        scaler = _fit_fold_scalers(prepared, train_ids)
        x_train = _transform_x(prepared, train_ids, scaler)
        x_val = _transform_x(prepared, val_ids, scaler)
        y_train = _scaled_y(prepared, train_ids, scaler)
        y_val = _scaled_y(prepared, val_ids, scaler)
        bundle = FoldTrainingBundle(
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            y_mean=float(scaler["y_mean"]),
            y_scale=float(scaler["y_std"]),
        )
        canonical.assert_pre_evaluation_isolation(bundle)

        fold_record = {
            "heldout_site_count": FOLD_SITE_COUNTS[numeric_fold],
            "training_site_count": EXPECTED_SITES - FOLD_SITE_COUNTS[numeric_fold],
            "train_rows": int(len(train_ids)),
            "validation_rows": int(len(val_ids)),
            "scaler": _json_scaler(scaler),
            "checkpoints": [],
        }
        for seed in SEEDS:
            path = _checkpoint_path(output_dir, fold_id, seed)
            if path.exists():
                saved = torch.load(path, map_location="cpu")
                if (
                    saved.get("fold_id") != fold_id
                    or int(saved.get("seed")) != seed
                    or int(saved.get("window")) != WINDOW
                    or int(saved.get("hidden_size")) != int(config.training["hidden_size"])
                    or float(saved.get("learning_rate")) != float(config.training["learning_rate"])
                ):
                    raise RuntimeError(f"Existing unseen-site checkpoint identity mismatch: {path}")
            else:
                result = canonical.train_model(
                    bundle,
                    hidden_size=int(config.training["hidden_size"]),
                    learning_rate=float(config.training["learning_rate"]),
                    seed=seed,
                    max_epochs=int(config.training["max_epochs"]),
                    patience=int(config.training["early_stopping_patience"]),
                    deterministic=True,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(result["history"]).to_csv(
                    path.parent / "training_history.csv", index=False, encoding="utf-8"
                )
                saved = {
                    "workflow": "coordinate_clustered_unseen_site_oof",
                    "stage": "validation_selected_checkpoint",
                    "fold_id": fold_id,
                    "seed": seed,
                    "window": WINDOW,
                    "scaler_fit_window": MAX_WINDOW,
                    "input_dim": len(EXPECTED_FEATURES),
                    "hidden_size": int(config.training["hidden_size"]),
                    "learning_rate": float(config.training["learning_rate"]),
                    "max_epochs": int(config.training["max_epochs"]),
                    "early_stopping_patience": int(config.training["early_stopping_patience"]),
                    "best_epoch": int(result["best_epoch"]),
                    "best_val_mse_scaled": float(result["best_val_mse_scaled"]),
                    "best_val_rmse_gC_m2_month": float(result["best_val_rmse_gC_m2_month"]),
                    "fold_scaler": _json_scaler(scaler),
                    "model_state_dict": result["best_state"],
                    "evaluation_metrics_used_for_selection": False,
                }
                torch.save(saved, path)
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fold_record["checkpoints"].append(
                {
                    "seed": seed,
                    "path": str(path.relative_to(PACKAGE_ROOT)),
                    "best_epoch": int(saved["best_epoch"]),
                    "best_val_mse_scaled": float(saved["best_val_mse_scaled"]),
                }
            )
        manifest["folds"][fold_id] = fold_record
        del x_train, x_val, y_train, y_val, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "unseen_site_fit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed 5-fold x 3-seed unseen-site fitting under {output_dir}")


def evaluate_stage(config_path: Path, site_table: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    validate_public_config(config)
    prepared = _load_prepared(config.path("processed_data"))
    folds = _load_site_table(site_table, config.path("unseen_site_folds"))
    _validate_site_coordinates(prepared.sample_index, folds)
    _write_fold_map(output_dir, folds)

    fit_manifest = output_dir / "unseen_site_fit_manifest.json"
    if not fit_manifest.exists():
        raise FileNotFoundError("Run the unseen-site fit stage before Evaluation")
    fit_state = json.loads(fit_manifest.read_text(encoding="utf-8"))
    if fit_state.get("stage") != "fit_complete" or fit_state.get("seeds") != list(SEEDS):
        raise RuntimeError("Unseen-site fit manifest is incomplete or has the wrong seed identity")

    frames: list[pd.DataFrame] = []
    for numeric_fold in range(1, 6):
        fold_id = f"fold_{numeric_fold}"
        _, _, eval_mask = _fold_masks(prepared.sample_index, folds, fold_id)
        eval_ids = prepared.sample_index.loc[eval_mask, "sample_id"].to_numpy(dtype=np.int64)
        expected_eval_rows = FOLD_SITE_COUNTS[numeric_fold] * 4 * 12
        if len(eval_ids) != expected_eval_rows:
            raise RuntimeError(f"{fold_id} Evaluation row count mismatch")

        for seed in SEEDS:
            path = _checkpoint_path(output_dir, fold_id, seed)
            if not path.exists():
                raise FileNotFoundError(f"Missing locked unseen-site checkpoint: {path}")
            checkpoint = torch.load(path, map_location="cpu")
            if (
                checkpoint.get("fold_id") != fold_id
                or int(checkpoint.get("seed")) != seed
                or int(checkpoint.get("window")) != WINDOW
                or int(checkpoint.get("hidden_size")) != int(config.training["hidden_size"])
                or float(checkpoint.get("learning_rate")) != float(config.training["learning_rate"])
                or checkpoint.get("evaluation_metrics_used_for_selection") is not False
            ):
                raise RuntimeError(f"Checkpoint protocol identity mismatch: {path}")

            scaler_record = checkpoint["fold_scaler"]
            scaler = {
                "x_mean_in_global_z": np.asarray(scaler_record["x_mean_in_global_z"], dtype=np.float64),
                "x_std_in_global_z": np.asarray(scaler_record["x_std_in_global_z"], dtype=np.float64),
                "y_mean": float(scaler_record["y_mean"]),
                "y_std": float(scaler_record["y_std"]),
            }
            x_eval = _transform_x(prepared, eval_ids, scaler)
            model = canonical.CNNBiLSTMAttention(
                len(EXPECTED_FEATURES), int(config.training["hidden_size"]), canonical.FIXED
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            device = canonical.get_device()
            model = model.to(device)
            loader = canonical.make_x_loader(x_eval, int(config.training["batch_size"]), device)
            pred_scaled = canonical.predict(model, loader, device)
            y_true = np.asarray(prepared.y_raw[eval_ids], dtype=np.float64) * 1000.0
            y_pred = (
                pred_scaled.astype(np.float64) * float(scaler["y_std"])
                + float(scaler["y_mean"])
            ) * 1000.0
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                raise RuntimeError(f"{fold_id} seed {seed} produced non-finite Evaluation values")

            pred = prepared.sample_index.loc[
                eval_mask, ["point_id", "Lon", "Lat", "Year", "Month"]
            ].reset_index(drop=True).copy()
            pred.insert(0, "seed", seed)
            pred.insert(0, "fold_id", fold_id)
            pred["y_true"] = y_true
            pred["y_pred"] = y_pred
            pred["residual"] = y_pred - y_true
            frames.append(pred)
            del x_eval, model, loader, pred_scaled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    oof = pd.concat(frames, ignore_index=True).sort_values(
        ["seed", "fold_id", "point_id", "Year", "Month"], kind="mergesort"
    ).reset_index(drop=True)
    if len(oof) != 35568:
        raise RuntimeError(f"OOF total must be 35,568 rows, observed {len(oof)}")
    if oof.groupby("seed").size().to_dict() != {42: 11856, 2024: 11856, 3407: 11856}:
        raise RuntimeError("OOF rows per seed must be 11,856")
    if not (oof.groupby(["seed", "point_id"]).size() == 48).all():
        raise RuntimeError("Each seed-site OOF combination must contain 48 Evaluation months")
    if oof.duplicated(["seed", "point_id", "Year", "Month"]).any():
        raise RuntimeError("OOF predictions contain duplicate seed-point-month keys")

    prediction_dir = output_dir / "predictions"
    result_dir = output_dir / "results"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    oof_path = prediction_dir / "unseen_site_oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    seed_metrics, fold_summary, pooled = summarise_predictions(oof)
    seed_metrics.to_csv(result_dir / "unseen_site_per_fold_per_seed_metrics.csv", index=False)
    fold_summary.to_csv(result_dir / "five_fold_three_seed_summary.csv", index=False)
    pooled.to_csv(result_dir / "unseen_site_pooled_oof_metrics_per_seed.csv", index=False)
    print(f"Wrote 35,568 unseen-site OOF predictions and summaries under {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    default_config = PACKAGE_ROOT / "configs" / "config.yaml"
    default_output = PACKAGE_ROOT / "runtime" / "unseen_site"
    for name in ("fit", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=default_config)
        command.add_argument(
            "--site-table",
            required=True,
            type=Path,
            help="Public 247-site CSV containing point_id, Lon and Lat.",
        )
        command.add_argument("--output-dir", type=Path, default=default_output)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "fit":
        fit_stage(args.config.resolve(), args.site_table.resolve(), args.output_dir.resolve())
    else:
        evaluate_stage(args.config.resolve(), args.site_table.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
