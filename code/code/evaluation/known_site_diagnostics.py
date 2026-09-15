"""Summarize three-seed known-site Evaluation predictions by month and year.

This utility operates on the canonical prediction table produced by
``canonical_pipeline.py evaluate``.  It preserves the manuscript distinction
between training-seed stochasticity (mean +/- sample SD across seeds) and
other uncertainty procedures.  It does not create bootstrap confidence
intervals; use ``code/uncertainty/clustered_bootstrap.py`` for those.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 2024, 3407)
METRICS = ("RMSE", "MAE", "R2", "Bias", "Pearson_r")
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(y) == 0 or not np.isfinite(np.column_stack([y, p])).all():
        raise ValueError("Metric input must contain finite prediction/reference pairs.")
    error = p - y
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "Bias": float(np.mean(error)),
        "Pearson_r": (
            float(np.corrcoef(y, p)[0, 1])
            if np.std(y) > 0 and np.std(p) > 0 else np.nan
        ),
    }


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "y_true_gC_m2_month": "y_true",
        "y_pred_gC_m2_month": "y_pred",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    required = {"seed", "point_id", "Year", "Month", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["point_id"] = pd.to_numeric(frame["point_id"], errors="raise").astype(int)
    frame["Year"] = pd.to_numeric(frame["Year"], errors="raise").astype(int)
    frame["Month"] = pd.to_numeric(frame["Month"], errors="raise").astype(int)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(float)
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="raise").astype(float)
    if tuple(sorted(frame["seed"].unique())) != SEEDS:
        raise ValueError("Prediction table must contain exactly seeds 42, 2024 and 3407.")
    if frame.duplicated(["seed", "point_id", "Year", "Month"]).any():
        raise ValueError("Prediction table contains duplicate seed-site-month keys.")
    counts = frame["seed"].value_counts().sort_index().to_dict()
    if counts != {42: 11856, 2024: 11856, 3407: 11856}:
        raise ValueError(f"Expected 11,856 Evaluation records per seed; got {counts}.")
    return frame


def summarize_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for group_value, group in frame.groupby(group_col, sort=True):
        per_seed = []
        for seed in SEEDS:
            part = group.loc[group["seed"] == seed]
            values = metric_values(part["y_true"], part["y_pred"])
            per_seed.append(values)
        row: dict[str, object] = {group_col: group_value}
        first = group.loc[group["seed"] == SEEDS[0]]
        row["records_per_seed"] = int(len(first))
        row["sites"] = int(first["point_id"].nunique())
        for metric in METRICS:
            values = np.asarray([item[metric] for item in per_seed], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def annual_site_sums(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["seed", "point_id", "Year"], sort=True, as_index=False)
        .agg(
            n_months=("Month", "size"),
            reference_annual_gC_m2=("y_true", "sum"),
            prediction_annual_gC_m2=("y_pred", "sum"),
        )
    )
    if not (grouped["n_months"] == 12).all():
        raise ValueError("Annual aggregation requires exactly 12 months per site-year-seed.")
    grouped["residual_annual_gC_m2"] = (
        grouped["prediction_annual_gC_m2"] - grouped["reference_annual_gC_m2"]
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frame = read_predictions(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    monthly = summarize_group(frame, "Month")
    monthly.insert(1, "calendar_month", monthly["Month"].map(MONTH_LABELS))
    yearly = summarize_group(frame, "Year")
    annual = annual_site_sums(frame)

    monthly.to_csv(args.output_dir / "known_site_monthly_three_seed_diagnostics.csv", index=False)
    yearly.to_csv(args.output_dir / "known_site_year_specific_three_seed_performance.csv", index=False)
    annual.to_csv(args.output_dir / "known_site_annual_site_sums_by_seed.csv", index=False)
    print(f"Wrote known-site month/year/annual diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
