"""Attach published site folds and summarize three-seed OOF predictions.

Prediction columns: fold_id, seed, point_id, Year, Month, y_true, y_pred.
Prediction units: g C m^-2 month^-1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 2024, 3407)
FOLD_SITE_COUNTS = {1: 36, 2: 44, 3: 34, 4: 74, 5: 59}


def load_fixed_folds(site_table: pd.DataFrame, assignment_path: Path) -> pd.DataFrame:
    """Join study-site coordinates to the published point_id-to-fold mapping."""
    required = ["point_id", "Lon", "Lat"]
    sites = site_table[required].copy()
    folds = pd.read_csv(assignment_path, dtype={"point_id": str, "fold_id": str})
    folds = folds[["point_id", "fold_id"]].copy()
    for frame in (sites, folds):
        frame["point_id"] = frame["point_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
        if len(frame) != 247 or frame["point_id"].nunique() != 247:
            raise ValueError("Expected 247 unique point IDs in sites and folds")
    if set(sites["point_id"]) != set(folds["point_id"]):
        raise ValueError("Study sites and fold assignment have different point IDs")
    expected_counts = {f"fold_{fold}": n for fold, n in FOLD_SITE_COUNTS.items()}
    if folds["fold_id"].value_counts().to_dict() != expected_counts:
        raise ValueError("Fold IDs or held-out site counts mismatch")
    sites[["Lon", "Lat"]] = sites[["Lon", "Lat"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(sites[["Lon", "Lat"]].to_numpy(float)).all():
        raise ValueError("Nonfinite study-site coordinates")
    return sites.merge(folds, on="point_id", validate="one_to_one").sort_values("point_id").reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    e = p - y
    sse = float(np.sum(e**2))
    sst = float(np.sum((y - y.mean())**2))
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "R2": float(1.0 - sse / sst) if sst else np.nan,
        "Bias": float(np.mean(e)),
        "Pearson_r": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else np.nan,
    }



def summarise_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    aliases = {
        "y_true_gC_m2_month": "y_true",
        "y_pred_gC_m2_month": "y_pred",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    required = {"fold_id", "seed", "point_id", "Year", "Month", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    fold_text = frame["fold_id"].astype(str).str.strip()
    fold_numeric = fold_text.str.extract(r"(?:fold[_ -]?)?(\d+)$", expand=False)
    if fold_numeric.isna().any():
        raise ValueError("fold_id must be numeric 1--5 or labels fold_1--fold_5")
    frame["fold_id"] = pd.to_numeric(fold_numeric, errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if tuple(sorted(frame["seed"].unique())) != SEEDS:
        raise ValueError("Prediction table must contain exactly seeds 42, 2024 and 3407")
    if set(frame["fold_id"]) != set(FOLD_SITE_COUNTS):
        raise ValueError("Prediction table must contain fold IDs 1--5")
    if frame.duplicated(["fold_id", "seed", "point_id", "Year", "Month"]).any():
        raise ValueError("Prediction table contains duplicate OOF keys")
    if frame["point_id"].nunique() != 247:
        raise ValueError("Prediction table must cover exactly 247 sites")
    if (frame.groupby("point_id", observed=True)["fold_id"].nunique() != 1).any():
        raise ValueError("Each point_id must belong to exactly one held-out fold")
    if sorted(pd.to_numeric(frame["Year"], errors="raise").astype(int).unique().tolist()) != [2022, 2023, 2024, 2025]:
        raise ValueError("Unseen-site OOF predictions must cover Evaluation years 2022--2025 only")
    if not (frame.groupby(["seed", "point_id"], observed=True).size() == 48).all():
        raise ValueError("Each seed-point_id combination must contain exactly 48 Evaluation months")
    truth_views = []
    for seed in SEEDS:
        truth_views.append(
            frame.loc[frame["seed"] == seed, ["point_id", "Year", "Month", "y_true"]]
            .sort_values(["point_id", "Year", "Month"], kind="mergesort")
            .reset_index(drop=True)
        )
    if not truth_views[0].equals(truth_views[1]) or not truth_views[0].equals(truth_views[2]):
        raise ValueError("OOF y_true values/keys must be identical across the three seeds")
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all():
        raise ValueError("Prediction table contains non-finite values")

    fold_rows, seed_rows = [], []
    for fold_id, part in frame.groupby("fold_id", sort=True):
        n_sites = part["point_id"].nunique()
        if n_sites != FOLD_SITE_COUNTS[int(fold_id)]:
            raise ValueError(
                f"Fold {fold_id} has {n_sites} sites; expected {FOLD_SITE_COUNTS[int(fold_id)]}"
            )
        expected_rows = n_sites * 48
        for seed, seed_part in part.groupby("seed", sort=True):
            if len(seed_part) != expected_rows:
                raise ValueError(
                    f"Fold {fold_id} seed {seed} has {len(seed_part)} records; expected {expected_rows}"
                )
            values = metrics(seed_part["y_true"], seed_part["y_pred"])
            seed_rows.append(
                {
                    "fold_id": f"fold_{int(fold_id)}",
                    "seed": int(seed),
                    "heldout_site_count": n_sites,
                    "evaluation_row_count": len(seed_part),
                    **values,
                }
            )
        per_seed = [
            metrics(seed_part["y_true"], seed_part["y_pred"])
            for _, seed_part in part.groupby("seed", sort=True)
        ]
        row = {
            "fold_id": f"fold_{int(fold_id)}",
            "heldout_site_count": n_sites,
            "evaluation_row_count_per_seed": expected_rows,
        }
        for name in per_seed[0]:
            row[f"{name}_mean"] = float(np.mean([item[name] for item in per_seed]))
            row[f"{name}_sample_sd"] = float(np.std([item[name] for item in per_seed], ddof=1))
        fold_rows.append(row)

    pooled_rows = []
    for seed, seed_part in frame.groupby("seed", sort=True):
        if len(seed_part) != 11856 or seed_part["point_id"].nunique() != 247:
            raise ValueError("Pooled OOF support must be 247 sites and 11,856 records per seed")
        pooled_rows.append(
            {
                "seed": int(seed),
                "rows": 11856,
                "sites": 247,
                **metrics(seed_part["y_true"], seed_part["y_pred"]),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(fold_rows), pd.DataFrame(pooled_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    assign = sub.add_parser("assign-folds", help="Attach published coordinate-cluster fold IDs")
    assign.add_argument("--site-table", required=True, type=Path)
    assign.add_argument("--fold-assignment", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data_documentation/unseen_site_fold_assignment.csv")
    assign.add_argument("--output", required=True, type=Path)

    summary = sub.add_parser("summarize", help="Validate and summarize unseen-site OOF predictions")
    summary.add_argument("--predictions", required=True, type=Path)
    summary.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "assign-folds":
        sites = load_fixed_folds(pd.read_csv(args.site_table, dtype={"point_id": str}), args.fold_assignment)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sites.to_csv(args.output, index=False)
        print(f"Wrote published fold assignments to {args.output}")
        return

    frame = pd.read_csv(args.predictions)
    seed, fold, pooled = summarise_predictions(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed.to_csv(args.output_dir / "unseen_site_seed_metrics.csv", index=False)
    fold.to_csv(args.output_dir / "five_fold_three_seed_summary.csv", index=False)
    pooled.to_csv(args.output_dir / "unseen_site_pooled_oof_metrics_per_seed.csv", index=False)
    print(f"Wrote unseen-site summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
