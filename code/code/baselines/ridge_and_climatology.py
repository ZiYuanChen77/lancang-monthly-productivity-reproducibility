"""Validation-lock and Evaluation stages for Ridge W36 and monthly climatology.

The ``select`` stage uses only Train (2005--2018) and Validation (2019--2021):
Ridge alpha is chosen on Validation, the selected Ridge is fitted on Train, and
calendar-month climatology means are computed from Train.  Those artifacts are
hashed into a pre-Evaluation lock.

The ``evaluate`` stage first validates that lock and the current canonical run
lineage, then and only then reads the 2022--2025 Evaluation slice and writes the
two deterministic prediction files in manuscript units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


MODEL_COMPARISON_DIR = Path(__file__).resolve().parents[1] / "model_comparison"
if str(MODEL_COMPARISON_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_COMPARISON_DIR))

from model_workflow_common import validate_canonical_state  # noqa: E402


ALPHAS = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
)
EXPECTED_SPLITS = {"train": 41496, "val": 8892, "test": 11856}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    e = p - y
    sse = float(np.sum(e**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "R2": float(1.0 - sse / sst) if sst else np.nan,
        "Bias": float(np.mean(e)),
        "Pearson_r": (
            float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else np.nan
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("select", "evaluate"):
        p = sub.add_parser(stage)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--artifact-dir", type=Path, required=True)
        p.add_argument("--prediction-dir", type=Path, required=True)
        p.add_argument("--window", type=int, default=36)
    return parser


def _load_index(data_dir: Path) -> pd.DataFrame:
    index = pd.read_csv(data_dir / "Sample_index_aligned.csv")
    required = {"point_id", "Year", "Month", "split"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Sample index is missing columns: {missing}")
    index = index.copy()
    index["split"] = index["split"].astype(str).str.lower()
    counts = index["split"].value_counts().to_dict()
    for split, expected in EXPECTED_SPLITS.items():
        if int(counts.get(split, 0)) != expected:
            raise ValueError(
                f"Prepared split {split!r} has {counts.get(split, 0)} rows; expected {expected}."
            )
    if len(index) != sum(EXPECTED_SPLITS.values()):
        raise ValueError("Prepared sample index has an unexpected total row count.")
    return index


def _positions(index: pd.DataFrame, split: str) -> np.ndarray:
    return np.flatnonzero(index["split"].eq(split).to_numpy())


def _artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "ridge_model": artifact_dir / "ridge_w36.joblib",
        "monthly_means": artifact_dir / "monthly_climatology_means.csv",
        "search": artifact_dir / "ridge_alpha_validation_search.csv",
        "validation_predictions": artifact_dir / "baseline_validation_predictions.csv",
        "validation_metrics": artifact_dir / "baseline_validation_metrics.csv",
        "lock": artifact_dir / "baseline_pre_evaluation_lock.json",
    }


def prediction_frame(
    index: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
    model: str,
    alpha: float | None,
) -> pd.DataFrame:
    out = index[["point_id", "Year", "Month", "split"]].copy().reset_index(drop=True)
    out["y_true_gC_m2_month"] = np.asarray(y_true, dtype=float)
    out["y_pred_gC_m2_month"] = np.asarray(pred, dtype=float)
    out["model"] = model
    if alpha is not None:
        out["selected_alpha"] = float(alpha)
    return out


def _validate_lock(lock: dict, artifact_dir: Path) -> None:
    if lock.get("stage") != "ridge_monthly_pre_evaluation_lock":
        raise RuntimeError("Deterministic-baseline lock has the wrong stage identity")
    if lock.get("selection_data_scope") != "train_validation_only":
        raise RuntimeError("Deterministic-baseline lock does not identify the Train/Validation selection scope")
    paths = _artifact_paths(artifact_dir)
    for key, hash_key in (
        ("ridge_model", "ridge_model_sha256"),
        ("monthly_means", "monthly_means_sha256"),
        ("search", "search_sha256"),
        ("validation_predictions", "validation_predictions_sha256"),
        ("validation_metrics", "validation_metrics_sha256"),
    ):
        path = paths[key]
        if not path.exists() or sha256_file(path) != lock.get(hash_key):
            raise RuntimeError(f"Locked baseline artifact changed or is missing: {path}")


def select_stage(data_dir: Path, artifact_dir: Path, prediction_dir: Path) -> None:
    # The selection stage uses the artifact directory; prediction output is
    # written by the evaluation stage.
    _ = prediction_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(artifact_dir)
    if paths["lock"].exists():
        existing = json.loads(paths["lock"].read_text(encoding="utf-8"))
        _validate_lock(existing, artifact_dir)
        print(f"Reused locked deterministic-baseline selection at {artifact_dir}")
        return

    canonical_lock = validate_canonical_state(require_evaluation=False)
    if canonical_lock.get("state") != "NOT_EVALUATED":
        raise RuntimeError(
            "Ridge/monthly baseline selection must be locked before canonical Evaluation."
        )
    index = _load_index(data_dir)
    train_pos = _positions(index, "train")
    val_pos = _positions(index, "val")

    # Memory maps allow the select stage to materialise Train/Validation slices
    # This stage reads only the Train and Validation slices.
    y_map = np.load(data_dir / "Y_tensor_raw.npy", mmap_mode="r")
    x_map = np.load(data_dir / "X_tensor_aligned_36.npy", mmap_mode="r")
    if len(y_map) != len(index) or x_map.shape[0] != len(index):
        raise ValueError("Prepared tensors and sample index are not aligned.")

    y_train = np.asarray(y_map[train_pos], dtype=np.float64) * 1000.0
    y_val = np.asarray(y_map[val_pos], dtype=np.float64) * 1000.0
    x_train = np.asarray(x_map[train_pos], dtype=np.float64).reshape(len(train_pos), -1)
    x_val = np.asarray(x_map[val_pos], dtype=np.float64).reshape(len(val_pos), -1)
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError("Train/Validation inputs contain non-finite values.")

    train_index = index.iloc[train_pos].reset_index(drop=True)
    val_index = index.iloc[val_pos].reset_index(drop=True)
    month_means = (
        pd.DataFrame({"Month": train_index["Month"].to_numpy(), "y": y_train})
        .groupby("Month", sort=True)["y"]
        .mean()
    )
    if len(month_means) != 12 or not np.isfinite(month_means.to_numpy()).all():
        raise ValueError("Training data do not contain twelve finite calendar-month means.")
    monthly_val = val_index["Month"].map(month_means).to_numpy(dtype=float)

    search: list[dict] = []
    for candidate_id, alpha in enumerate(ALPHAS, start=1):
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-4,
            max_iter=10000,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        search.append(
            {"candidate_id": candidate_id, "alpha": alpha, **metric_values(y_val, pred)}
        )
    history = (
        pd.DataFrame(search)
        .sort_values(["RMSE", "MAE", "candidate_id"])
        .reset_index(drop=True)
    )
    selected_alpha = float(history.iloc[0]["alpha"])
    history.to_csv(paths["search"], index=False, encoding="utf-8-sig")

    ridge = Ridge(
        alpha=selected_alpha,
        fit_intercept=True,
        solver="lsqr",
        tol=1e-4,
        max_iter=10000,
    )
    ridge.fit(x_train, y_train)
    ridge_val = ridge.predict(x_val)
    joblib.dump(ridge, paths["ridge_model"], compress=3)
    month_means.rename("mean_gC_m2_month").reset_index().to_csv(
        paths["monthly_means"], index=False, encoding="utf-8-sig"
    )

    validation = pd.concat(
        [
            prediction_frame(
                val_index, y_val, monthly_val, "Monthly climatology", None
            ),
            prediction_frame(
                val_index, y_val, ridge_val, "Ridge W36", selected_alpha
            ),
        ],
        ignore_index=True,
    )
    validation.to_csv(
        paths["validation_predictions"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "model": "Monthly climatology",
                "split": "validation",
                **metric_values(y_val, monthly_val),
            },
            {
                "model": "Ridge W36",
                "split": "validation",
                "selected_alpha": selected_alpha,
                **metric_values(y_val, ridge_val),
            },
        ]
    ).to_csv(paths["validation_metrics"], index=False, encoding="utf-8-sig")

    lock = {
        "stage": "ridge_monthly_pre_evaluation_lock",
        "created_at": now_iso(),
        "window": 36,
        "selection_split": "2019-2021 validation",
        "fit_split": "2005-2018 train",
        "selected_alpha": selected_alpha,
        "alpha_grid": list(ALPHAS),
        "selection_data_scope": "train_validation_only",
        "canonical_data_identity": canonical_lock.get("data_identity"),
        "canonical_data_hashes": canonical_lock.get("data_hashes"),
        "ridge_model_sha256": sha256_file(paths["ridge_model"]),
        "monthly_means_sha256": sha256_file(paths["monthly_means"]),
        "search_sha256": sha256_file(paths["search"]),
        "validation_predictions_sha256": sha256_file(paths["validation_predictions"]),
        "validation_metrics_sha256": sha256_file(paths["validation_metrics"]),
    }
    paths["lock"].write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(
        f"Locked deterministic-baseline selection at {artifact_dir}; "
        f"selected Ridge alpha={selected_alpha:g}."
    )


def evaluate_stage(data_dir: Path, artifact_dir: Path, prediction_dir: Path) -> None:
    paths = _artifact_paths(artifact_dir)
    if not paths["lock"].exists():
        raise FileNotFoundError(
            "Run the deterministic-baseline select stage before Evaluation."
        )
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    _validate_lock(lock, artifact_dir)

    canonical_lock = validate_canonical_state(require_evaluation=True)
    if (
        canonical_lock.get("data_identity") != lock.get("canonical_data_identity")
        or canonical_lock.get("data_hashes") != lock.get("canonical_data_hashes")
    ):
        raise RuntimeError(
            "Current canonical Evaluation data do not match the baseline pre-Evaluation lock."
        )

    index = _load_index(data_dir)
    test_pos = _positions(index, "test")
    y_map = np.load(data_dir / "Y_tensor_raw.npy", mmap_mode="r")
    x_map = np.load(data_dir / "X_tensor_aligned_36.npy", mmap_mode="r")
    if len(y_map) != len(index) or x_map.shape[0] != len(index):
        raise ValueError("Prepared tensors and sample index are not aligned.")
    y_test = np.asarray(y_map[test_pos], dtype=np.float64) * 1000.0
    x_test = np.asarray(x_map[test_pos], dtype=np.float64).reshape(len(test_pos), -1)
    test_index = index.iloc[test_pos].reset_index(drop=True)

    ridge = joblib.load(paths["ridge_model"])
    ridge_test = ridge.predict(x_test)
    month_means = pd.read_csv(paths["monthly_means"]).set_index("Month")["mean_gC_m2_month"]
    monthly_test = test_index["Month"].map(month_means).to_numpy(dtype=float)
    if not np.isfinite(monthly_test).all():
        raise RuntimeError("Monthly climatology is missing an Evaluation month.")

    prediction_dir.mkdir(parents=True, exist_ok=True)
    monthly_frame = prediction_frame(
        test_index, y_test, monthly_test, "Monthly climatology", None
    )
    ridge_frame = prediction_frame(
        test_index,
        y_test,
        ridge_test,
        "Ridge W36",
        float(lock["selected_alpha"]),
    )
    monthly_frame.to_csv(
        prediction_dir / "baseline_monthly_climatology.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ridge_frame.to_csv(
        prediction_dir / "baseline_ridge_w36.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "model": "Monthly climatology",
                "split": "evaluation",
                **metric_values(y_test, monthly_test),
            },
            {
                "model": "Ridge W36",
                "split": "evaluation",
                "selected_alpha": float(lock["selected_alpha"]),
                **metric_values(y_test, ridge_test),
            },
        ]
    ).to_csv(
        artifact_dir / "baseline_evaluation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Wrote deterministic Evaluation predictions to {prediction_dir}")


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    if args.window != 36:
        raise ValueError("The deterministic baseline comparison uses W36.")
    if args.stage == "select":
        select_stage(data_dir, artifact_dir, prediction_dir)
    else:
        evaluate_stage(data_dir, artifact_dir, prediction_dir)


if __name__ == "__main__":
    main()
