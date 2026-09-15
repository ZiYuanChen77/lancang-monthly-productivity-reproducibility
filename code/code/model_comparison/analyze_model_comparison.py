from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_workflow_common import ExperimentConfig, load_config, validate_public_config


KEYS = ["point_id", "Year", "Month"]
METRICS = ["RMSE", "MAE", "R2", "Bias", "Pearson_r"]
REFERENCE_MODEL = "Full canonical"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    prediction_file: str
    stochastic: bool
    role: str
    architecture: str
    tuning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dual-domain model metrics and bootstrap intervals.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def model_specs(config: ExperimentConfig) -> list[ModelSpec]:
    return [ModelSpec(**item) for item in config.models]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    error = p - y
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    bias = float(np.mean(error))
    pearson_r = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "Bias": bias,
        "Pearson_r": pearson_r,
    }


def read_prediction(config: ExperimentConfig, spec: ModelSpec) -> pd.DataFrame:
    path = config.path("predictions_dir") / spec.prediction_file
    frame = pd.read_csv(path)
    required = {*KEYS, "y_true_gC_m2_month", "y_pred_gC_m2_month"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.prediction_file} is missing columns: {missing}")

    columns = KEYS + ["y_true_gC_m2_month", "y_pred_gC_m2_month"]
    output = frame[columns].rename(
        columns={
            "y_true_gC_m2_month": "y_true",
            "y_pred_gC_m2_month": "y_pred",
        }
    )
    output["point_id"] = pd.to_numeric(output["point_id"], errors="raise").astype(int)
    output["Year"] = pd.to_numeric(output["Year"], errors="raise").astype(int)
    output["Month"] = pd.to_numeric(output["Month"], errors="raise").astype(int)
    output["y_true"] = pd.to_numeric(output["y_true"], errors="raise").astype(float)
    output["y_pred"] = pd.to_numeric(output["y_pred"], errors="raise").astype(float)

    if spec.stochastic:
        if "seed" not in frame.columns:
            raise ValueError(f"{spec.name} requires a seed column.")
        output["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        if tuple(sorted(output["seed"].unique())) != config.seeds:
            raise ValueError(f"{spec.name} does not contain the configured seed set.")
    else:
        output["seed"] = "deterministic"

    if output.duplicated(KEYS + ["seed"]).any():
        raise ValueError(f"{spec.name} contains duplicate key-by-seed rows.")
    if not np.isfinite(output[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError(f"{spec.name} contains non-finite values.")

    output["model"] = spec.name
    return output


def add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    index = year * 12 + month - 1 + offset
    return index // 12, index % 12 + 1


def excluded_target_keys(config: ExperimentConfig) -> set[tuple[int, int, int]]:
    definition = config.raw["clean_common_domain"]
    point_id = int(definition["contaminated_point_id"])
    year = int(definition["contaminated_input_year"])
    month = int(definition["contaminated_input_month"])
    window_length = int(definition["window_length"])
    return {
        (point_id, *add_months(year, month, offset))
        for offset in range(1, window_length + 1)
    }


def filter_domain(
    frame: pd.DataFrame,
    domain: str,
    excluded_keys: set[tuple[int, int, int]],
) -> pd.DataFrame:
    if domain == "canonical_full_domain":
        return frame.copy()
    keys = list(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
    keep = np.fromiter((key not in excluded_keys for key in keys), dtype=bool)
    return frame.loc[keep].copy()


def validate_common_support(
    data: dict[str, pd.DataFrame],
    specs: list[ModelSpec],
    seeds: tuple[int, ...],
) -> None:
    reference = data[REFERENCE_MODEL]
    reference_truth = (
        reference.loc[reference["seed"] == seeds[0], KEYS + ["y_true"]]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    stochastic = {spec.name for spec in specs if spec.stochastic}
    for name, frame in data.items():
        current = frame.loc[frame["seed"] == seeds[0]] if name in stochastic else frame
        current = current[KEYS + ["y_true"]].sort_values(KEYS).reset_index(drop=True)
        if not current[KEYS].equals(reference_truth[KEYS]):
            raise ValueError(f"Evaluation keys differ between {name} and {REFERENCE_MODEL}.")
        if not np.allclose(current["y_true"], reference_truth["y_true"], rtol=0, atol=1e-10):
            raise ValueError(f"Reference values differ between {name} and {REFERENCE_MODEL}.")


def site_sufficient_statistics(frame: pd.DataFrame, point_ids: np.ndarray) -> np.ndarray:
    work = frame.copy()
    error = work["y_pred"] - work["y_true"]
    work["y2"] = work["y_true"] ** 2
    work["p2"] = work["y_pred"] ** 2
    work["yp"] = work["y_true"] * work["y_pred"]
    work["sqe"] = error**2
    work["abe"] = np.abs(error)
    work["e"] = error
    grouped = (
        work.groupby("point_id", sort=True)
        .agg(
            n=("y_true", "size"),
            sy=("y_true", "sum"),
            sy2=("y2", "sum"),
            sp=("y_pred", "sum"),
            sp2=("p2", "sum"),
            syp=("yp", "sum"),
            sse=("sqe", "sum"),
            sae=("abe", "sum"),
            se=("e", "sum"),
        )
        .reindex(point_ids)
    )
    if grouped.isna().any().any():
        raise ValueError("Cluster support differs between models.")
    return grouped.to_numpy(dtype=np.float64)


def metrics_from_sums(sums: np.ndarray) -> dict[str, np.ndarray]:
    n, sy, sy2, sp, sp2, syp, sse, sae, se = sums.T
    y_centered = sy2 - sy * sy / n
    p_centered = sp2 - sp * sp / n
    covariance = syp - sy * sp / n
    bias = se / n
    return {
        "RMSE": np.sqrt(sse / n),
        "MAE": sae / n,
        "R2": 1.0 - sse / y_centered,
        "Bias": bias,
        "Pearson_r": covariance / np.sqrt(y_centered * p_centered),
    }


def bootstrap_draws(frame: pd.DataFrame, point_ids: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    return metrics_from_sums(counts @ site_sufficient_statistics(frame, point_ids))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_public_config(config)
    specs = model_specs(config)
    data = {spec.name: read_prediction(config, spec) for spec in specs}
    validate_common_support(data, specs, config.seeds)

    results_dir = config.path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    domains = ("clean_common_domain", "canonical_full_domain")
    excluded_keys = excluded_target_keys(config)
    stochastic_models = {spec.name for spec in specs if spec.stochastic}

    filtered: dict[str, dict[str, pd.DataFrame]] = {domain: {} for domain in domains}
    per_seed_rows: list[dict] = []
    summary_rows: list[dict] = []

    for domain in domains:
        for spec in specs:
            part = filter_domain(data[spec.name], domain, excluded_keys)
            filtered[domain][spec.name] = part
            if spec.stochastic:
                seed_records = []
                for seed in config.seeds:
                    seed_frame = part.loc[part["seed"] == seed]
                    values = metric_values(seed_frame["y_true"], seed_frame["y_pred"])
                    record = {
                        "domain": domain,
                        "model": spec.name,
                        "seed": seed,
                        "n_rows": len(seed_frame),
                        "n_point_ids": seed_frame["point_id"].nunique(),
                        **values,
                    }
                    per_seed_rows.append(record)
                    seed_records.append(record)

                summary = {
                    "domain": domain,
                    "model": spec.name,
                    "runs": len(config.seeds),
                    "deterministic": False,
                    "n_rows": seed_records[0]["n_rows"],
                    "role": spec.role,
                    "architecture": spec.architecture,
                    "tuning": spec.tuning,
                }
                for metric in METRICS:
                    values = np.asarray([record[metric] for record in seed_records], dtype=float)
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"{metric}_seed_SD"] = float(values.std(ddof=1))
            else:
                values = metric_values(part["y_true"], part["y_pred"])
                per_seed_rows.append(
                    {
                        "domain": domain,
                        "model": spec.name,
                        "seed": "N/A",
                        "n_rows": len(part),
                        "n_point_ids": part["point_id"].nunique(),
                        **values,
                    }
                )
                summary = {
                    "domain": domain,
                    "model": spec.name,
                    "runs": 1,
                    "deterministic": True,
                    "n_rows": len(part),
                    "role": spec.role,
                    "architecture": spec.architecture,
                    "tuning": spec.tuning,
                }
                for metric in METRICS:
                    summary[f"{metric}_mean"] = float(values[metric])
                    summary[f"{metric}_seed_SD"] = np.nan
            summary_rows.append(summary)

    per_seed = pd.DataFrame(per_seed_rows)
    summary = pd.DataFrame(summary_rows)
    per_seed.to_csv(results_dir / "workflow_per_seed_metrics_by_domain.csv", index=False)
    summary.to_csv(results_dir / "workflow_13_model_dual_domain_metrics.csv", index=False)

    bootstrap_cfg = config.raw["bootstrap"]
    rng = np.random.default_rng(int(bootstrap_cfg["random_seed"]))
    point_ids = np.asarray(sorted(data[REFERENCE_MODEL]["point_id"].unique()))
    counts = rng.multinomial(
        len(point_ids),
        np.repeat(1.0 / len(point_ids), len(point_ids)),
        size=int(bootstrap_cfg["n_resamples"]),
    ).astype(float)

    bootstrap_rows: list[dict] = []
    for domain in domains:
        reference = filtered[domain][REFERENCE_MODEL]
        reference_draws = {}
        reference_points = {}
        for seed in config.seeds:
            frame = reference.loc[reference["seed"] == seed]
            reference_draws[seed] = bootstrap_draws(frame, point_ids, counts)
            reference_points[seed] = metric_values(frame["y_true"], frame["y_pred"])

        for spec in specs:
            if spec.name == REFERENCE_MODEL:
                continue
            candidate = filtered[domain][spec.name]
            for metric in METRICS:
                draws_by_seed = []
                points_by_seed = []
                for seed in config.seeds:
                    frame = candidate.loc[candidate["seed"] == seed] if spec.name in stochastic_models else candidate
                    candidate_draws = bootstrap_draws(frame, point_ids, counts)
                    candidate_point = metric_values(frame["y_true"], frame["y_pred"])
                    draws_by_seed.append(candidate_draws[metric] - reference_draws[seed][metric])
                    points_by_seed.append(candidate_point[metric] - reference_points[seed][metric])

                delta_draws = np.mean(np.vstack(draws_by_seed), axis=0)
                low, high = np.quantile(delta_draws, [0.025, 0.975])
                bootstrap_rows.append(
                    {
                        "domain": domain,
                        "candidate_model": spec.name,
                        "reference_model": REFERENCE_MODEL,
                        "metric": metric,
                        "delta_definition": "candidate - reference",
                        "delta_point_estimate": float(np.mean(points_by_seed)),
                        "ci_low_2.5": float(low),
                        "ci_high_97.5": float(high),
                        "zero_status": "included_zero" if low <= 0 <= high else "excluded_zero",
                        "n_matched_seeds": len(config.seeds),
                        "n_resamples": int(bootstrap_cfg["n_resamples"]),
                        "bootstrap_seed": int(bootstrap_cfg["random_seed"]),
                        "cluster_variable": str(bootstrap_cfg["cluster_variable"]),
                    }
                )

    pd.DataFrame(bootstrap_rows).to_csv(
        results_dir / "workflow_ablation_bootstrap_vs_full.csv",
        index=False,
    )
    print(f"Wrote model-comparison results to {results_dir}")


if __name__ == "__main__":
    main()

