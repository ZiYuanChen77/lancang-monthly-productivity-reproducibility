"""Dependence-aware point-cluster bootstrap for prediction metrics.

Rows within a ``point_id`` are resampled together, preserving repeated
site-month dependence.  The module computes percentile 95% intervals for
single-model metrics and paired candidate-minus-reference contrasts.  It is a
stand-alone utility and does not require model weights or raw observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("RMSE", "MAE", "R2", "Bias", "Pearson_r")


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    error = p - y
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - sse / sst) if sst else np.nan,
        "Bias": float(np.mean(error)),
        "Pearson_r": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else np.nan,
    }


def cluster_sufficient_statistics(frame: pd.DataFrame, cluster: str) -> tuple[np.ndarray, np.ndarray]:
    required = {cluster, "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    work = frame.copy()
    work["sqe"] = (work["y_pred"] - work["y_true"]) ** 2
    work["abe"] = np.abs(work["y_pred"] - work["y_true"])
    work["err"] = work["y_pred"] - work["y_true"]
    work["y2"] = work["y_true"] ** 2
    work["p2"] = work["y_pred"] ** 2
    work["yp"] = work["y_true"] * work["y_pred"]
    grouped = work.groupby(cluster, sort=True).agg(
        n=("y_true", "size"), sy=("y_true", "sum"), sy2=("y2", "sum"),
        sp=("y_pred", "sum"), sp2=("p2", "sum"), syp=("yp", "sum"),
        sse=("sqe", "sum"), sae=("abe", "sum"), se=("err", "sum"),
    )
    return grouped.index.to_numpy(), grouped.to_numpy(dtype=float)


def metrics_from_sums(sums: np.ndarray) -> dict[str, np.ndarray]:
    n, sy, sy2, sp, sp2, syp, sse, sae, se = sums.T
    y_centered = sy2 - sy * sy / n
    p_centered = sp2 - sp * sp / n
    covariance = syp - sy * sp / n
    return {
        "RMSE": np.sqrt(sse / n),
        "MAE": sae / n,
        "R2": 1.0 - sse / y_centered,
        "Bias": se / n,
        "Pearson_r": covariance / np.sqrt(y_centered * p_centered),
    }


def bootstrap(frame: pd.DataFrame, *, cluster: str = "point_id", n_resamples: int = 5000, seed: int = 20240724) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels, stats = cluster_sufficient_statistics(frame, cluster)
    counts = rng.multinomial(len(labels), np.full(len(labels), 1.0 / len(labels)), size=n_resamples)
    values = metrics_from_sums(counts @ stats)
    rows = []
    for index in range(n_resamples):
        rows.append({"replicate": index + 1, **{name: float(values[name][index]) for name in METRICS}})
    return pd.DataFrame(rows)


def paired_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame, *, cluster: str = "point_id", n_resamples: int = 5000, seed: int = 20240724) -> pd.DataFrame:
    key = [cluster] + [column for column in ("Year", "Month") if column in candidate.columns and column in reference.columns]
    left = candidate[key + ["y_true", "y_pred"]].rename(columns={"y_pred": "candidate_pred"})
    right = reference[key + ["y_true", "y_pred"]].rename(columns={"y_true": "reference_y_true", "y_pred": "reference_pred"})
    joined = left.merge(right, on=key, validate="one_to_one")
    if not np.allclose(joined["y_true"], joined["reference_y_true"], rtol=0, atol=1e-12):
        raise ValueError("Candidate and reference truth values do not match.")
    rng = np.random.default_rng(seed)
    labels = np.sort(joined[cluster].unique())
    left_stats = cluster_sufficient_statistics(joined.rename(columns={"candidate_pred": "y_pred"})[[cluster, "y_true", "y_pred"]], cluster)[1]
    right_stats = cluster_sufficient_statistics(joined.rename(columns={"reference_pred": "y_pred"})[[cluster, "y_true", "y_pred"]], cluster)[1]
    counts = rng.multinomial(len(labels), np.full(len(labels), 1.0 / len(labels)), size=n_resamples)
    left_values = metrics_from_sums(counts @ left_stats)
    right_values = metrics_from_sums(counts @ right_stats)
    return pd.DataFrame({"replicate": np.arange(1, n_resamples + 1), **{f"delta_{name}": left_values[name] - right_values[name] for name in METRICS}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20240724)
    parser.add_argument("--cluster", default="point_id")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    result = bootstrap(frame, cluster=args.cluster, n_resamples=args.n_resamples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} clustered bootstrap replicates to {args.output}")


if __name__ == "__main__":
    main()
