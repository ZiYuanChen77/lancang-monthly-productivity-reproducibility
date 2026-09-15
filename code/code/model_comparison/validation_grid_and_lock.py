from __future__ import annotations

import argparse
import itertools
import gc
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from model_workflow_common import (
    DATA_DIR,
    DEEP_GRID_MODELS,
    CANONICAL_LOCK,
    CANONICAL_PIPELINE,
    CANONICAL_WINDOW_MANIFEST,
    RUN_ROOT,
    load_config,
    load_pre_evaluation_data,
    validate_public_config,
    log,
    now_iso,
    read_json,
    sha256_file,
    train_deep,
    write_json,
)


LEARNING_RATES = (0.0001, 0.0003, 0.001)
WIDTHS = (64, 128)
RF_ESTIMATORS = (300, 600)
RF_DEPTHS = (None, 20, 40)
RF_MIN_LEAF = (1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed Validation-only hyperparameter grids for comparison models."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "config.yaml"),
        help="Public configuration path (used by the top-level dispatcher).",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    for name in ("models", "results", "locks", "logs"):
        (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)


def save_partial(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(
        RUN_ROOT / "models" / "validation_grid_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_public_config(config)
    ensure_output_dirs()
    lock_path = RUN_ROOT / "locks" / "workflow_selected_hyperparameters_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Refusing to overwrite HP lock: {lock_path}")
    data = load_pre_evaluation_data()
    partial_path = RUN_ROOT / "models" / "validation_grid_partial.csv"
    rows: list[dict] = (
        pd.read_csv(partial_path).replace({np.nan: ""}).to_dict(orient="records")
        if partial_path.exists()
        else []
    )

    for model_name in DEEP_GRID_MODELS:
        for width, learning_rate in itertools.product(WIDTHS, LEARNING_RATES):
            already_done = any(
                row.get("model") == model_name
                and int(row.get("width")) == width
                and float(row.get("learning_rate")) == learning_rate
                for row in rows
            )
            if already_done:
                log(f"GRID REUSE COMPLETED model={model_name} width={width} lr={learning_rate}")
                continue
            log(
                f"GRID START model={model_name} width={width} lr={learning_rate} "
                "seed=42 max_epochs=25 patience=5"
            )
            result = train_deep(
                data,
                model_name=model_name,
                width=width,
                learning_rate=learning_rate,
                seed=42,
                max_epochs=25,
                patience=5,
            )
            rows.append(
                {
                    "stage": "validation_grid",
                    "model": model_name,
                    "seed": 42,
                    "width": width,
                    "learning_rate": learning_rate,
                    "validation_mse_scaled": result["best_val_mse_scaled"],
                    "validation_mse_gC2": result["best_val_mse_gC2"],
                    "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"],
                    "runtime_seconds": result["runtime_seconds"],
                    "max_epochs": 25,
                    "patience": 5,
                }
            )
            log(
                f"GRID DONE model={model_name} width={width} lr={learning_rate} "
                f"val_mse_gC2={result['best_val_mse_gC2']:.17g}"
            )
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_partial(rows)

    X_train = np.asarray(data.X_train, dtype=np.float32).reshape(len(data.X_train), -1)
    X_val = np.asarray(data.X_val, dtype=np.float32).reshape(len(data.X_val), -1)
    for n_estimators, max_depth, min_samples_leaf in itertools.product(
        RF_ESTIMATORS, RF_DEPTHS, RF_MIN_LEAF
    ):
        already_done = any(
            row.get("model") == "Random Forest"
            and int(row.get("n_estimators")) == n_estimators
            and str(row.get("max_depth")) == ("None" if max_depth is None else str(max_depth))
            and int(row.get("min_samples_leaf")) == min_samples_leaf
            for row in rows
        )
        if already_done:
            log(
                "RF GRID REUSE COMPLETED "
                f"trees={n_estimators} depth={max_depth} min_leaf={min_samples_leaf}"
            )
            continue
        log(
            "RF GRID START "
            f"trees={n_estimators} depth={max_depth} min_leaf={min_samples_leaf}"
        )
        started = time.perf_counter()
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        )
        rf.fit(X_train, data.y_train)
        val_pred = rf.predict(X_val)
        val_mse_scaled = float(mean_squared_error(data.y_val, val_pred))
        runtime = time.perf_counter() - started
        rows.append(
            {
                "stage": "validation_grid",
                "model": "Random Forest",
                "seed": 42,
                "width": "",
                "learning_rate": "",
                "n_estimators": n_estimators,
                "max_depth": "None" if max_depth is None else max_depth,
                "min_samples_leaf": min_samples_leaf,
                "validation_mse_scaled": val_mse_scaled,
                "validation_mse_gC2": val_mse_scaled * (data.y_scale * 1000.0) ** 2,
                "best_epoch": "",
                "epochs_run": "",
                "runtime_seconds": runtime,
                "max_epochs": "",
                "patience": "",
            }
        )
        log(
            "RF GRID DONE "
            f"trees={n_estimators} depth={max_depth} min_leaf={min_samples_leaf} "
            f"val_mse_gC2={rows[-1]['validation_mse_gC2']:.17g}"
        )
        del rf, val_pred
        gc.collect()
        save_partial(rows)

    grid = pd.DataFrame(rows)
    selected: dict[str, dict] = {}
    for model_name in DEEP_GRID_MODELS:
        candidates = grid.loc[grid["model"] == model_name].copy()
        candidates = candidates.sort_values(
            ["validation_mse_gC2", "width", "learning_rate"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        winner = candidates.iloc[0]
        selected[model_name] = {
            "width": int(winner["width"]),
            "learning_rate": float(winner["learning_rate"]),
            "validation_mse_scaled": float(winner["validation_mse_scaled"]),
            "validation_mse_gC2": float(winner["validation_mse_gC2"]),
            "tie_breaking_status": "not required"
            if candidates["validation_mse_gC2"].nunique() == len(candidates)
            else "applied: smaller width, then lower learning rate",
        }

    rf_candidates = grid.loc[grid["model"] == "Random Forest"].copy()
    rf_candidates["trees_rank"] = pd.to_numeric(rf_candidates["n_estimators"])
    rf_candidates["depth_rank"] = rf_candidates["max_depth"].map(
        lambda value: float("inf") if str(value) == "None" else float(value)
    )
    rf_candidates["leaf_rank"] = -pd.to_numeric(rf_candidates["min_samples_leaf"])
    rf_candidates = rf_candidates.sort_values(
        ["validation_mse_gC2", "trees_rank", "depth_rank", "leaf_rank"],
        ascending=True,
        kind="mergesort",
    )
    rf_winner = rf_candidates.iloc[0]
    selected["Random Forest"] = {
        "n_estimators": int(rf_winner["n_estimators"]),
        "max_depth": (
            None if str(rf_winner["max_depth"]) == "None" else int(rf_winner["max_depth"])
        ),
        "min_samples_leaf": int(rf_winner["min_samples_leaf"]),
        "max_features": "sqrt",
        "random_state": 42,
        "validation_mse_scaled": float(rf_winner["validation_mse_scaled"]),
        "validation_mse_gC2": float(rf_winner["validation_mse_gC2"]),
        "tie_breaking_status": "not required"
        if rf_candidates["validation_mse_gC2"].nunique() == len(rf_candidates)
        else "applied: fewer trees, shallower finite depth, larger min_samples_leaf",
    }

    selected_rows = []
    for model_name, hp in selected.items():
        selected_rows.append({"model": model_name, **hp})
    selected_rows.append(
        {
            "model": "point-month climatology",
            "tuning": "none",
            "definition": "training-period point_id × month mean",
        }
    )
    pd.DataFrame(selected_rows).to_csv(
        RUN_ROOT / "results" / "workflow_selected_hyperparameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grid.to_csv(
        RUN_ROOT / "models" / "validation_grid_complete.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lock = {
        "stage": "workflow_selected_hyperparameters_lock",
        "created_at": now_iso(),
        "window": 36,
        "train": "2005-2018",
        "validation": "2019-2021",
        "evaluation": "2022-2025",
        "tuning_seed": 42,
        "criterion": "full-precision Validation MSE",
        "deep_grid": {
            "learning_rate": list(LEARNING_RATES),
            "width": list(WIDTHS),
            "max_epochs": 25,
            "early_stopping_patience": 5,
            "tie_break": "lower MSE; smaller width; lower learning rate",
        },
        "rf_grid": {
            "n_estimators": list(RF_ESTIMATORS),
            "max_depth": list(RF_DEPTHS),
            "min_samples_leaf": list(RF_MIN_LEAF),
            "max_features": "sqrt",
            "random_state": 42,
            "tie_break": (
                "lower MSE; fewer trees; shallower finite depth; larger min_samples_leaf"
            ),
        },
        "all_validation_results": rows,
        "selected_hyperparameters": selected,
        "selection_data_scope": "train_validation_only",
        "data_identity": data.data_identity,
        "data_hashes": data.data_hashes(),
        "lineage": {
            "canonical_lock": str(CANONICAL_LOCK),
            "canonical_lock_sha256": sha256_file(CANONICAL_LOCK),
            "canonical_window_manifest": str(CANONICAL_WINDOW_MANIFEST),
            "canonical_window_manifest_sha256": sha256_file(CANONICAL_WINDOW_MANIFEST),
            "canonical_pipeline": str(CANONICAL_PIPELINE),
            "canonical_pipeline_sha256": sha256_file(CANONICAL_PIPELINE),
            "this_script_sha256": sha256_file(Path(__file__)),
            "common_script_sha256": sha256_file(Path(__file__).parent / "model_workflow_common.py"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
    }
    write_json(lock_path, lock)
    log(f"HP LOCK CREATED sha256={sha256_file(lock_path)} path={lock_path}")


if __name__ == "__main__":
    main()

