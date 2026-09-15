"""Grouped bootstrap for known-site, unseen-site OOF and paired protocol contrasts.

The implementation follows the released Supplementary method description:
complete ``point_id`` histories are the primary resampling units; ``point_id``
by year blocks are the secondary sensitivity units.  The same resample
multiplicities are reused for known-site and unseen-site predictions, all three
training seeds, and all five metrics.  Metrics are computed per seed inside each
replicate and then averaged across seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 2024, 3407)
METRICS = ("RMSE", "MAE", "R2", "Bias", "Pearson_r")


def _normalise(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    aliases = {
        "y_true_gC_m2_month": "y_true",
        "y_pred_gC_m2_month": "y_pred",
    }
    for source, target in aliases.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    required = {"seed", "point_id", "Year", "Month", "y_true", "y_pred"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"{label} table is missing columns: {missing}")
    for col in ("seed", "point_id", "Year", "Month"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    for col in ("y_true", "y_pred"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
    if tuple(sorted(out["seed"].unique())) != SEEDS:
        raise ValueError(f"{label} must contain exactly seeds {SEEDS}.")
    if out.duplicated(["seed", "point_id", "Year", "Month"]).any():
        raise ValueError(f"{label} contains duplicate seed-site-month keys.")
    if not np.isfinite(out[["y_true", "y_pred"]].to_numpy(float)).all():
        raise ValueError(f"{label} contains non-finite prediction/reference values.")
    counts = out["seed"].value_counts().sort_index().to_dict()
    if counts != {42: 11856, 2024: 11856, 3407: 11856}:
        raise ValueError(f"{label} must contain 11,856 rows per seed; got {counts}.")
    if out["point_id"].nunique() != 247:
        raise ValueError(f"{label} must contain 247 sites.")
    return out


def _metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
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


def _stats(frame: pd.DataFrame, cluster_cols: list[str], cluster_index: pd.MultiIndex | pd.Index) -> np.ndarray:
    work = frame.copy()
    e = work["y_pred"] - work["y_true"]
    work["y2"] = work["y_true"] ** 2
    work["p2"] = work["y_pred"] ** 2
    work["yp"] = work["y_true"] * work["y_pred"]
    work["sqe"] = e**2
    work["abe"] = np.abs(e)
    work["err"] = e
    grouped = work.groupby(cluster_cols, sort=True).agg(
        n=("y_true", "size"),
        sy=("y_true", "sum"),
        sy2=("y2", "sum"),
        sp=("y_pred", "sum"),
        sp2=("p2", "sum"),
        syp=("yp", "sum"),
        sse=("sqe", "sum"),
        sae=("abe", "sum"),
        se=("err", "sum"),
    )
    grouped = grouped.reindex(cluster_index)
    if grouped.isna().any().any():
        raise ValueError("Known/unseen cluster support differs.")
    return grouped.to_numpy(dtype=float)


def _metrics_from_sums(sums: np.ndarray) -> dict[str, np.ndarray]:
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


def _cluster_index(frame: pd.DataFrame, cluster_cols: list[str]):
    if len(cluster_cols) == 1:
        return pd.Index(sorted(frame[cluster_cols[0]].unique()), name=cluster_cols[0])
    return pd.MultiIndex.from_frame(
        frame[cluster_cols].drop_duplicates().sort_values(cluster_cols).reset_index(drop=True)
    )


def _validate_truth_alignment(known: pd.DataFrame, unseen: pd.DataFrame) -> None:
    keys = ["seed", "point_id", "Year", "Month"]
    left = known[keys + ["y_true"]].sort_values(keys).reset_index(drop=True)
    right = unseen[keys + ["y_true"]].sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Known and unseen protocols do not share the same Evaluation keys.")
    if not np.allclose(left["y_true"], right["y_true"], rtol=0, atol=1e-10):
        raise ValueError("Known and unseen protocols do not share the same reference values.")


def _protocol_point_estimates(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    per_seed = {
        seed: _metric_values(
            frame.loc[frame["seed"] == seed, "y_true"].to_numpy(float),
            frame.loc[frame["seed"] == seed, "y_pred"].to_numpy(float),
        )
        for seed in SEEDS
    }
    means = {m: float(np.mean([per_seed[s][m] for s in SEEDS])) for m in METRICS}
    sds = {m: float(np.std([per_seed[s][m] for s in SEEDS], ddof=1)) for m in METRICS}
    return means, sds


def _bootstrap_protocols(
    known: pd.DataFrame,
    unseen: pd.DataFrame,
    cluster_cols: list[str],
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    index = _cluster_index(known.loc[known["seed"] == SEEDS[0]], cluster_cols)
    n_clusters = len(index)
    expected = 247 if cluster_cols == ["point_id"] else 988
    if n_clusters != expected:
        raise ValueError(
            f"Expected {expected} resampling units for {cluster_cols}; got {n_clusters}."
        )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n_clusters,
        np.repeat(1.0 / n_clusters, n_clusters),
        size=n_resamples,
    ).astype(float)

    known_seed_draws: dict[int, dict[str, np.ndarray]] = {}
    unseen_seed_draws: dict[int, dict[str, np.ndarray]] = {}
    for model_seed in SEEDS:
        k = known.loc[known["seed"] == model_seed]
        u = unseen.loc[unseen["seed"] == model_seed]
        known_seed_draws[model_seed] = _metrics_from_sums(counts @ _stats(k, cluster_cols, index))
        unseen_seed_draws[model_seed] = _metrics_from_sums(counts @ _stats(u, cluster_cols, index))

    known_draws = {
        metric: np.mean(np.vstack([known_seed_draws[s][metric] for s in SEEDS]), axis=0)
        for metric in METRICS
    }
    unseen_draws = {
        metric: np.mean(np.vstack([unseen_seed_draws[s][metric] for s in SEEDS]), axis=0)
        for metric in METRICS
    }
    paired = {metric: unseen_draws[metric] - known_draws[metric] for metric in METRICS}
    return known_draws, unseen_draws, paired


def _ci(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975], method="linear")
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-predictions", required=True, type=Path)
    parser.add_argument("--unseen-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20240724)
    args = parser.parse_args()

    known = _normalise(pd.read_csv(args.known_predictions), "known-site")
    unseen = _normalise(pd.read_csv(args.unseen_predictions), "unseen-site")
    _validate_truth_alignment(known, unseen)

    known_point, known_sd = _protocol_point_estimates(known)
    unseen_point, unseen_sd = _protocol_point_estimates(unseen)

    site_k, site_u, site_d = _bootstrap_protocols(
        known, unseen, ["point_id"], args.n_resamples, args.seed
    )
    year_k, year_u, year_d = _bootstrap_protocols(
        known, unseen, ["point_id", "Year"], args.n_resamples, args.seed
    )

    known_rows, unseen_rows, diff_rows = [], [], []
    for metric in METRICS:
        sk_lo, sk_hi = _ci(site_k[metric])
        sy_lo, sy_hi = _ci(year_k[metric])
        su_lo, su_hi = _ci(site_u[metric])
        uy_lo, uy_hi = _ci(year_u[metric])
        sd_lo, sd_hi = _ci(site_d[metric])
        yd_lo, yd_hi = _ci(year_d[metric])

        known_rows.append({
            "protocol": "known-site chronological evaluation",
            "metric": metric,
            "point_estimate": known_point[metric],
            "three_seed_sample_sd_ddof_1": known_sd[metric],
            "site_cluster_ci_lower_2_5pct": sk_lo,
            "site_cluster_ci_upper_97_5pct": sk_hi,
            "site_year_ci_lower_2_5pct": sy_lo,
            "site_year_ci_upper_97_5pct": sy_hi,
            "training_stochasticity": "3-seed sample SD",
            "sampling_dependence_uncertainty": "95% two-sided percentile grouped bootstrap CI",
        })
        unseen_rows.append({
            "protocol": "coordinate-clustered unseen-site pooled OOF evaluation",
            "metric": metric,
            "point_estimate": unseen_point[metric],
            "three_seed_sample_sd_ddof_1": unseen_sd[metric],
            "site_cluster_ci_lower_2_5pct": su_lo,
            "site_cluster_ci_upper_97_5pct": su_hi,
            "site_year_ci_lower_2_5pct": uy_lo,
            "site_year_ci_upper_97_5pct": uy_hi,
            "training_stochasticity": "3-seed sample SD",
            "sampling_dependence_uncertainty": "95% two-sided percentile grouped bootstrap CI",
        })
        delta = unseen_point[metric] - known_point[metric]
        diff_rows.append({
            "metric": metric,
            "contrast": "unseen-site - known-site protocol contrast",
            "point_estimate": delta,
            "site_cluster_ci_lower_2_5pct": sd_lo,
            "site_cluster_ci_upper_97_5pct": sd_hi,
            "site_cluster_includes_zero": bool(sd_lo <= 0 <= sd_hi),
            "site_year_ci_lower_2_5pct": yd_lo,
            "site_year_ci_upper_97_5pct": yd_hi,
            "site_year_includes_zero": bool(yd_lo <= 0 <= yd_hi),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(known_rows).to_csv(args.output_dir / "known_site_cluster_bootstrap_summary.csv", index=False)
    pd.DataFrame(unseen_rows).to_csv(args.output_dir / "unseen_site_cluster_bootstrap_summary.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(args.output_dir / "unseen_minus_known_paired_difference_summary.csv", index=False)
    print(f"Wrote known/unseen grouped-bootstrap summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
