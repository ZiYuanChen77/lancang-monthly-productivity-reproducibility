#!/usr/bin/env python3
"""Prepare water-energy GAMM inputs from monthly, MOD17 and MOD13A1 tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASELINE_START = 2005
BASELINE_END = 2018
EVAL_START = 2022
EVAL_END = 2025
SENTINEL = -9999.0
SUPPORT_RADIUS = 0.10

MONTHLY_COLUMNS = [
    "point_id", "Year", "Month", "daima", "NPP", "NDVI", "Precipitation_mm",
    "VPD", "surface_solar_radiation_downwards_sum", "temperature_2m",
    "volumetric_soil_water_layer_1",
]

MODEL1_FIELDS = [
    "npp_anomaly_mean_g", "precip_rank_baseline", "radiation_rank_baseline",
    "precip_antecedent3_sum_mm", "radiation_antecedent3_mean_MJ_m2",
    "soilwater_lag1", "ndvi_lag1_canonical",
]


def _normalize_point_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").astype(int).map(lambda x: f"{x:03d}")


def _clean_numeric(series: pd.Series, *, sentinel_to_nan: bool = True) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    if sentinel_to_nan:
        out = out.mask(np.isclose(out, SENTINEL, rtol=0.0, atol=0.0))
    return out


def _midrank_against_history(x: float, baseline: np.ndarray) -> float:
    b = np.asarray(baseline, dtype=float)
    b = np.sort(b[np.isfinite(b)])
    if not np.isfinite(x) or b.size == 0:
        return np.nan
    left = np.searchsorted(b, x, side="left")
    right = np.searchsorted(b, x, side="right")
    ties = right - left
    # Midrank of the target observation in the historical reference augmented
    # by that target observation. This yields (less + 0.5) / (n + 1) when
    # there are no historical ties.
    return (left + 0.5 * (ties + 1.0)) / (b.size + 1.0)


def _historical_group_maps(history: pd.DataFrame, column: str) -> dict[tuple[str, int], np.ndarray]:
    return {
        (pid, int(month)): group[column].to_numpy(dtype=float)
        for (pid, month), group in history.groupby(["point_id", "Month"], sort=False)
    }


def _rank_column(eval_df: pd.DataFrame, history: pd.DataFrame, source: str) -> np.ndarray:
    maps = _historical_group_maps(history, source)
    result = np.empty(len(eval_df), dtype=float)
    for i, row in enumerate(eval_df[["point_id", "Month", source]].itertuples(index=False, name=None)):
        pid, month, value = row
        result[i] = _midrank_against_history(float(value), maps[(pid, int(month))])
    return result


def _same_site_month_baseline(history: pd.DataFrame, source: str, statistic: str = "mean") -> pd.Series:
    grouped = history.groupby(["point_id", "Month"], sort=False)[source]
    if statistic == "mean":
        return grouped.mean()
    if statistic == "median":
        return grouped.median()
    raise ValueError(statistic)


def _calendar_rolling_features(full: pd.DataFrame) -> pd.DataFrame:
    out = full.sort_values(["point_id", "Year", "Month"]).copy()
    group = out.groupby("point_id", sort=False)
    out["soilwater_lag1"] = group["volumetric_soil_water_layer_1"].shift(1)
    out["ndvi_lag1_canonical"] = group["NDVI_clean"].shift(1)
    out["precip_antecedent3_sum_mm"] = (
        group["Precipitation_mm"].shift(1)
        + group["Precipitation_mm"].shift(2)
        + group["Precipitation_mm"].shift(3)
    )
    out["radiation_antecedent3_mean_MJ_m2"] = (
        group["surface_solar_radiation_downwards_sum"].shift(1)
        + group["surface_solar_radiation_downwards_sum"].shift(2)
        + group["surface_solar_radiation_downwards_sum"].shift(3)
    ) / 3.0
    return out


def _read_monthly(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    missing = sorted(set(MONTHLY_COLUMNS) - set(d.columns))
    if missing:
        raise ValueError(f"monthly_model_input.csv is missing columns: {missing}")
    d = d.copy()
    d["point_id"] = _normalize_point_id(d["point_id"])
    d["Year"] = pd.to_numeric(d["Year"], errors="raise").astype(int)
    d["Month"] = pd.to_numeric(d["Month"], errors="raise").astype(int)
    for col in [
        "NPP", "NDVI", "Precipitation_mm", "VPD",
        "surface_solar_radiation_downwards_sum", "temperature_2m",
        "volumetric_soil_water_layer_1",
    ]:
        d[col] = _clean_numeric(d[col], sentinel_to_nan=(col != "NPP"))
    # Source NPP: kg C m^-2; analysis response: g C m^-2.
    d["NPP"] *= 1000.0
    d["NDVI_clean"] = d["NDVI"]
    return d.sort_values(["point_id", "Year", "Month"]).reset_index(drop=True)


def _build_dayweighted_monthly(appeears: pd.DataFrame) -> pd.DataFrame:
    a = appeears.copy()
    a["point_id"] = _normalize_point_id(a["point_id"])
    a["date"] = pd.to_datetime(a["composite_start_date"], errors="raise")
    # AppEEARS-scaled PsnNet: kg C m^-2 per composite.
    a["PsnNet"] = pd.to_numeric(a["PsnNet"], errors="coerce") * 1000.0
    starts = pd.Index(sorted(a["date"].dropna().unique()))
    if starts.empty:
        raise ValueError("AppEEARS table contains no composite dates")
    next_dates: dict[pd.Timestamp, pd.Timestamp] = {}
    for left, right in zip(starts[:-1], starts[1:]):
        left_ts = pd.Timestamp(left)
        right_ts = pd.Timestamp(right)
        delta = (right_ts - left_ts).days
        next_dates[left_ts] = right_ts if 1 <= delta <= 8 else left_ts + pd.Timedelta(days=8)
    last = pd.Timestamp(starts[-1])
    next_dates[last] = min(last + pd.Timedelta(days=8), pd.Timestamp("2026-01-01"))

    pieces: list[tuple[str, int, int, float]] = []
    for pid, date, value in a[["point_id", "date", "PsnNet"]].itertuples(index=False, name=None):
        if not np.isfinite(value):
            continue
        start = pd.Timestamp(date)
        end = next_dates[start]
        length = (end - start).days
        if length <= 0 or length > 8:
            end = start + pd.Timedelta(days=8)
            length = 8
        cursor = start
        while cursor < end:
            next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
            stop = min(end, next_month)
            days = (stop - cursor).days
            if days > 0 and 2001 <= cursor.year <= 2025:
                pieces.append((pid, cursor.year, cursor.month, float(value) * days / length))
            cursor = stop

    part = pd.DataFrame(pieces, columns=["point_id", "Year", "Month", "dayweighted_target_g"])
    return part.groupby(["point_id", "Year", "Month"], as_index=False, sort=False)["dayweighted_target_g"].sum()


def _add_appeears_fields(master: pd.DataFrame, monthly: pd.DataFrame, path: Path) -> pd.DataFrame:
    a = pd.read_csv(path)
    required = {"point_id", "composite_start_date", "PsnNet", "psn_good_usable"}
    missing = sorted(required - set(a.columns))
    if missing:
        raise ValueError(f"AppEEARS table is missing columns: {missing}")
    a["point_id"] = _normalize_point_id(a["point_id"])
    a["date"] = pd.to_datetime(a["composite_start_date"], errors="raise")
    a["Year"] = a["date"].dt.year.astype(int)
    a["Month"] = a["date"].dt.month.astype(int)
    a["psn_good_usable"] = pd.to_numeric(a["psn_good_usable"], errors="coerce")

    q = a.groupby(["point_id", "Year", "Month"], as_index=False, sort=False).agg(
        mod17_good_fraction=("psn_good_usable", "mean"),
        mod17_composite_count=("psn_good_usable", "size"),
    )
    q["mod17_QA1"] = (q["mod17_good_fraction"] >= 0.75).astype(int)
    q["mod17_QA2"] = (q["mod17_good_fraction"] >= 1.0).astype(int)

    day = _build_dayweighted_monthly(a)
    history_day = day[(day["Year"] >= BASELINE_START) & (day["Year"] <= BASELINE_END)].copy()
    day_mean = history_day.groupby(["point_id", "Month"])["dayweighted_target_g"].mean()
    day_median = history_day.groupby(["point_id", "Month"])["dayweighted_target_g"].median()
    day_active = (day_median > 0).rename("historically_active_dayweighted")
    day_eval = day[(day["Year"] >= EVAL_START) & (day["Year"] <= EVAL_END)].copy()
    keys = pd.MultiIndex.from_frame(day_eval[["point_id", "Month"]])
    day_eval["dayweighted_baseline_mean_g"] = day_mean.reindex(keys).to_numpy()
    day_eval["dayweighted_baseline_median_g"] = day_median.reindex(keys).to_numpy()
    day_eval["dayweighted_anomaly_mean_g"] = day_eval["dayweighted_target_g"] - day_eval["dayweighted_baseline_mean_g"]
    day_eval["historically_active_dayweighted"] = day_active.reindex(keys).fillna(False).astype(int).to_numpy()

    out = master.merge(q, on=["point_id", "Year", "Month"], how="left", validate="one_to_one")
    out = out.merge(
        day_eval[[
            "point_id", "Year", "Month", "dayweighted_target_g",
            "dayweighted_baseline_mean_g", "dayweighted_baseline_median_g",
            "dayweighted_anomaly_mean_g", "historically_active_dayweighted",
        ]],
        on=["point_id", "Year", "Month"], how="left", validate="one_to_one",
    )
    out["PsnNet_dayweighted_gC_m2_month"] = out["dayweighted_target_g"]
    out["mod17_QA1"] = out["mod17_QA1"].fillna(0).astype(int)
    out["mod17_QA2"] = out["mod17_QA2"].fillna(0).astype(int)
    out["eligible_mod17_QA1"] = (out["eligible_primary_model1"].astype(bool) & out["mod17_QA1"].astype(bool)).astype(int)
    out["eligible_mod17_QA2"] = (out["eligible_primary_model1"].astype(bool) & out["mod17_QA2"].astype(bool)).astype(int)
    complete = np.logical_and.reduce([np.isfinite(out[c].to_numpy(dtype=float)) for c in MODEL1_FIELDS])
    out["eligible_dayweighted"] = (
        out["historically_active_dayweighted"].fillna(0).astype(bool).to_numpy()
        & np.isfinite(out["dayweighted_anomaly_mean_g"].to_numpy(dtype=float))
        & complete
    ).astype(int)
    return out


def _add_optional_ndvi_qa(master: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    out = master.copy()
    fields = [
        "any_snow_ice_flag", "ndvi_total_composite_count", "snow_ice_qa_known",
        "target_month_any_snow_ice", "eligible_cold_snow",
    ]
    if path is None:
        for field in fields:
            out[field] = np.nan
        return out

    q = pd.read_csv(path)
    required = {"point_id", "Year", "Month", "any_snow_ice_flag", "ndvi_total_composite_count"}
    missing = sorted(required - set(q.columns))
    if missing:
        raise ValueError(f"MOD13A1 monthly QA table is missing columns: {missing}")
    q = q.copy()
    q["point_id"] = _normalize_point_id(q["point_id"])
    q["Year"] = pd.to_numeric(q["Year"], errors="raise").astype(int)
    q["Month"] = pd.to_numeric(q["Month"], errors="raise").astype(int)
    q["any_snow_ice_flag"] = pd.to_numeric(q["any_snow_ice_flag"], errors="coerce")
    q["ndvi_total_composite_count"] = pd.to_numeric(q["ndvi_total_composite_count"], errors="coerce")
    q = q[["point_id", "Year", "Month", "any_snow_ice_flag", "ndvi_total_composite_count"]]
    out = out.merge(q, on=["point_id", "Year", "Month"], how="left", validate="one_to_one")
    known = out["any_snow_ice_flag"].isin([0, 1]) & (out["ndvi_total_composite_count"] > 0)
    out["snow_ice_qa_known"] = known.astype(int)
    out["target_month_any_snow_ice"] = np.where(known, out["any_snow_ice_flag"], np.nan)
    out["eligible_cold_snow"] = (
        out["eligible_primary_model1"].astype(bool)
        & np.isfinite(out["temperature_2m"])
        & (out["temperature_2m"] > 0)
        & known
        & (out["target_month_any_snow_ice"] == 0)
    ).astype(int)
    return out


def _build_master(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = _calendar_rolling_features(monthly)
    history = full[(full["Year"] >= BASELINE_START) & (full["Year"] <= BASELINE_END)].copy()
    evaluation = full[(full["Year"] >= EVAL_START) & (full["Year"] <= EVAL_END)].copy()

    group_index = pd.MultiIndex.from_frame(evaluation[["point_id", "Month"]])

    npp_mean = _same_site_month_baseline(history, "NPP", "mean")
    npp_median = _same_site_month_baseline(history, "NPP", "median")
    historically_active = (npp_median > 0).rename("historically_active_canonical")

    baseline_sources = {
        "Precipitation_mm": "precipitation_mm_baseline_mean",
        "surface_solar_radiation_downwards_sum": "radiation_MJ_m2_baseline_mean",
        "temperature_2m": "temperature_2m_baseline_mean",
        "VPD": "VPD_baseline_mean",
        "volumetric_soil_water_layer_1": "soilwater_baseline_mean",
        "NDVI_clean": "NDVI_baseline_mean",
    }
    baseline_means: dict[str, pd.Series] = {}
    for source, dest in baseline_sources.items():
        baseline_means[dest] = _same_site_month_baseline(history, source, "mean")

    evaluation["calendar_month_index"] = evaluation["Year"] * 12 + evaluation["Month"]
    evaluation["npp_baseline_mean_g"] = npp_mean.reindex(group_index).to_numpy()
    evaluation["npp_baseline_median_g"] = npp_median.reindex(group_index).to_numpy()
    evaluation["npp_anomaly_mean_g"] = evaluation["NPP"] - evaluation["npp_baseline_mean_g"]
    evaluation["npp_anomaly_median_g"] = evaluation["NPP"] - evaluation["npp_baseline_median_g"]
    evaluation["historically_active_canonical"] = historically_active.reindex(group_index).fillna(False).astype(int).to_numpy()

    for dest, series in baseline_means.items():
        evaluation[dest] = series.reindex(group_index).to_numpy()

    evaluation["precip_anomaly_mm"] = evaluation["Precipitation_mm"] - evaluation["precipitation_mm_baseline_mean"]
    evaluation["radiation_anomaly_MJ_m2"] = (
        evaluation["surface_solar_radiation_downwards_sum"] - evaluation["radiation_MJ_m2_baseline_mean"]
    )
    evaluation["temperature_anomaly"] = evaluation["temperature_2m"] - evaluation["temperature_2m_baseline_mean"]
    evaluation["VPD_anomaly"] = evaluation["VPD"] - evaluation["VPD_baseline_mean"]
    evaluation["soilwater_anomaly"] = evaluation["volumetric_soil_water_layer_1"] - evaluation["soilwater_baseline_mean"]
    evaluation["NDVI_anomaly"] = evaluation["NDVI_clean"] - evaluation["NDVI_baseline_mean"]

    evaluation["precip_rank_baseline"] = _rank_column(evaluation, history, "Precipitation_mm")
    evaluation["radiation_rank_baseline"] = _rank_column(
        evaluation, history, "surface_solar_radiation_downwards_sum"
    )

    complete_model1 = np.logical_and.reduce([
        np.isfinite(evaluation[c].to_numpy(dtype=float)) for c in MODEL1_FIELDS
    ])
    evaluation["eligible_primary_model1"] = (
        evaluation["historically_active_canonical"].astype(bool).to_numpy() & complete_model1
    ).astype(int)
    # Domain flags describe the requested calendar scope; eligibility flags add
    # the complete-case requirement used by the sensitivity models.
    evaluation["domain_all12"] = 1
    evaluation["domain_apr_sep"] = evaluation["Month"].between(4, 9).astype(int)
    evaluation["eligible_all12"] = complete_model1.astype(int)
    evaluation["eligible_apr_sep"] = (
        complete_model1 & evaluation["Month"].between(4, 9).to_numpy()
    ).astype(int)
    evaluation["baseline_median_npp_g"] = evaluation["npp_baseline_median_g"]
    evaluation["eligible_median_response"] = evaluation["eligible_primary_model1"]
    evaluation["eligible_raw_exposure"] = evaluation["eligible_primary_model1"]

    canonical_baseline = historically_active.reset_index()
    canonical_baseline["historically_active_canonical"] = canonical_baseline["historically_active_canonical"].astype(int)

    covar_idx = npp_mean.index
    covariate_baseline = pd.DataFrame(index=covar_idx).reset_index()
    covariate_baseline["precipitation_mm_baseline_mean"] = baseline_means["precipitation_mm_baseline_mean"].reindex(covar_idx).to_numpy()
    covariate_baseline["radiation_MJ_m2_baseline_mean"] = baseline_means["radiation_MJ_m2_baseline_mean"].reindex(covar_idx).to_numpy()
    counts = history.groupby(["point_id", "Month"])[["Precipitation_mm", "surface_solar_radiation_downwards_sum"]].count()
    covariate_baseline["precipitation_mm_baseline_n"] = counts["Precipitation_mm"].reindex(covar_idx).to_numpy()
    covariate_baseline["radiation_MJ_m2_baseline_n"] = counts["surface_solar_radiation_downwards_sum"].reindex(covar_idx).to_numpy()

    keep_cols = [
        "point_id", "Year", "Month", "calendar_month_index", "daima",
        "NPP", "Precipitation_mm", "surface_solar_radiation_downwards_sum",
        "temperature_2m", "VPD", "volumetric_soil_water_layer_1", "NDVI_clean",
        "npp_baseline_mean_g", "npp_baseline_median_g", "baseline_median_npp_g",
        "npp_anomaly_mean_g", "npp_anomaly_median_g", "historically_active_canonical",
        "precip_rank_baseline", "radiation_rank_baseline",
        "precip_antecedent3_sum_mm", "radiation_antecedent3_mean_MJ_m2",
        "soilwater_lag1", "ndvi_lag1_canonical",
        "temperature_anomaly", "VPD_anomaly", "soilwater_anomaly", "NDVI_anomaly",
        "precip_anomaly_mm", "radiation_anomaly_MJ_m2",
        "eligible_primary_model1", "domain_all12", "domain_apr_sep",
        "eligible_all12", "eligible_apr_sep", "eligible_median_response",
        "eligible_raw_exposure",
    ]
    evaluation = evaluation[keep_cols].rename(columns={"NDVI_clean": "NDVI"})
    return evaluation, covariate_baseline, canonical_baseline


def _support_table(master: pd.DataFrame) -> pd.DataFrame:
    primary = master[master["eligible_primary_model1"] == 1].copy()
    rows = []
    for label, p, r in [("A", 0.75, 0.25), ("B", 0.75, 0.50)]:
        dist = np.sqrt((primary["precip_rank_baseline"] - p) ** 2 + (primary["radiation_rank_baseline"] - r) ** 2)
        keep = np.isfinite(dist) & (dist <= SUPPORT_RADIUS)
        sub = primary.loc[keep]
        rows.append({
            "contrast_point": label,
            "precip_rank_target": p,
            "radiation_rank_target": r,
            "radius": SUPPORT_RADIUS,
            "n_records": int(len(sub)),
            "n_sites": int(sub["point_id"].nunique()),
            "n_years": int(sub["Year"].nunique()),
        })
    return pd.DataFrame(rows)


def _find_public_file(root: Path, relative: str) -> Path | None:
    candidates = [
        root / relative,
        root / "public_data_v1.1" / relative,
        root / "public_data" / relative,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _summary(master: pd.DataFrame, support: pd.DataFrame, appeears_used: bool, ndvi_qa_used: bool) -> dict[str, object]:
    primary = master[master["eligible_primary_model1"] == 1]
    result: dict[str, object] = {
        "evaluation_rows": int(len(master)),
        "primary_rows": int(len(primary)),
        "primary_sites": int(primary["point_id"].nunique()),
        "primary_ar_sections": int(
            np.r_[True,
                  (primary.sort_values(["point_id", "calendar_month_index"])["point_id"].to_numpy()[1:] !=
                   primary.sort_values(["point_id", "calendar_month_index"])["point_id"].to_numpy()[:-1]) |
                  (np.diff(primary.sort_values(["point_id", "calendar_month_index"])["calendar_month_index"].to_numpy()) > 1)
            ].sum()
        ),
        "support": support.to_dict(orient="records"),
        "appeears_mod17_used": appeears_used,
        "mod13a1_monthly_qa_used": ndvi_qa_used,
    }
    for field in ["eligible_mod17_QA1", "eligible_mod17_QA2", "eligible_dayweighted", "eligible_cold_snow"]:
        if field in master.columns and master[field].notna().any():
            result[field] = int(pd.to_numeric(master[field], errors="coerce").fillna(0).sum())
    return result


def _assert_reference_invariants(master: pd.DataFrame, support: pd.DataFrame) -> None:
    primary = master[master["eligible_primary_model1"] == 1]
    expected_support = {
        "A": (401, 215, 4),
        "B": (339, 182, 4),
    }
    if len(master) != 11856:
        raise RuntimeError(f"Evaluation reconstruction mismatch: n={len(master)}")
    if len(primary) != 6992 or primary["point_id"].nunique() != 247:
        raise RuntimeError(
            f"Primary reconstruction mismatch: n={len(primary)}, sites={primary['point_id'].nunique()}"
        )
    ordered = primary.sort_values(["point_id", "calendar_month_index"])
    ids = ordered["point_id"].to_numpy()
    tt = ordered["calendar_month_index"].to_numpy(dtype=int)
    ar_sections = int(np.r_[True, (ids[1:] != ids[:-1]) | (np.diff(tt) > 1)].sum())
    if ar_sections != 988:
        raise RuntimeError(f"Primary AR-section reconstruction mismatch: {ar_sections}")
    domain_counts = (
        int(master["domain_all12"].sum()),
        int(master["domain_apr_sep"].sum()),
        int(master["eligible_all12"].sum()),
        int(master["eligible_apr_sep"].sum()),
    )
    if domain_counts != (11856, 5928, 11855, 5928):
        raise RuntimeError(f"Domain reconstruction mismatch: {domain_counts}")
    for row in support.itertuples(index=False):
        got = (int(row.n_records), int(row.n_sites), int(row.n_years))
        if got != expected_support[row.contrast_point]:
            raise RuntimeError(
                f"Support reconstruction mismatch at {row.contrast_point}: {got} != {expected_support[row.contrast_point]}"
            )
    if "eligible_mod17_QA1" in master and master["eligible_mod17_QA1"].notna().any():
        qa1 = int(master["eligible_mod17_QA1"].sum())
        qa2 = int(master["eligible_mod17_QA2"].sum())
        day = int(master["eligible_dayweighted"].sum())
        primary_flag = master["eligible_primary_model1"].eq(1)
        day_flag = master["eligible_dayweighted"].eq(1)
        overlap = (
            int((primary_flag & day_flag).sum()),
            int((primary_flag & ~day_flag).sum()),
            int((~primary_flag & day_flag).sum()),
        )
        if (qa1, qa2, day) != (5280, 3358, 6992):
            raise RuntimeError(f"MOD17/day-weighted reconstruction mismatch: {(qa1, qa2, day)}")
        if overlap != (6988, 4, 4):
            raise RuntimeError(f"Day-weighted domain overlap mismatch: {overlap}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-data-root", type=Path, help="Root of public_data_v1.1 (or its parent).")
    p.add_argument("--monthly-input", type=Path, help="Path to monthly_model_input.csv.")
    p.add_argument("--appeears-mod17", type=Path, help="Optional public AppEEARS MOD17A2HGF point-sample CSV.")
    p.add_argument("--ndvi-qa", type=Path, help="Optional MOD13A1 monthly QA CSV for cold/snow sensitivity.")
    p.add_argument("--output-dir", type=Path, default=Path("data/water_energy"))
    return p


def main() -> None:
    args = parser().parse_args()
    root = args.public_data_root.resolve() if args.public_data_root else None
    monthly_path = args.monthly_input.resolve() if args.monthly_input else None
    appeears_path = args.appeears_mod17.resolve() if args.appeears_mod17 else None

    if monthly_path is None and root is not None:
        monthly_path = _find_public_file(root, "model_input/monthly_model_input.csv")
    if appeears_path is None and root is not None:
        appeears_path = _find_public_file(root, "mod17_quality/appeears_MOD17A2HGF_point_sample.csv")
    if monthly_path is None or not monthly_path.exists():
        raise FileNotFoundError("monthly_model_input.csv was not found; use --public-data-root or --monthly-input")
    if appeears_path is not None and not appeears_path.exists():
        raise FileNotFoundError(appeears_path)
    ndvi_qa = args.ndvi_qa.resolve() if args.ndvi_qa else None
    if ndvi_qa is not None and not ndvi_qa.exists():
        raise FileNotFoundError(ndvi_qa)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    monthly = _read_monthly(monthly_path)
    master, covariate_baseline, canonical_baseline = _build_master(monthly)
    if appeears_path is not None:
        master = _add_appeears_fields(master, monthly, appeears_path)
    else:
        for field in [
            "mod17_good_fraction", "mod17_composite_count", "mod17_QA1", "mod17_QA2",
            "eligible_mod17_QA1", "eligible_mod17_QA2", "dayweighted_target_g",
            "PsnNet_dayweighted_gC_m2_month", "dayweighted_baseline_mean_g",
            "dayweighted_baseline_median_g", "dayweighted_anomaly_mean_g",
            "historically_active_dayweighted", "eligible_dayweighted",
        ]:
            master[field] = np.nan
    master = _add_optional_ndvi_qa(master, ndvi_qa)
    master = master.sort_values(["point_id", "Year", "Month"]).reset_index(drop=True)
    support = _support_table(master)
    _assert_reference_invariants(master, support)

    master.to_csv(output / "water_energy_analysis_ready_2022_2025.csv", index=False)
    support.to_csv(output / "primary_contrast_support.csv", index=False)
    covariate_baseline.to_csv(output / "covariate_baseline_site_month_2005_2018.csv", index=False)
    canonical_baseline.to_csv(output / "canonical_baseline_site_month_2005_2018.csv", index=False)
    # Local raw-exposure input; NPP is in g C m^-2 month^-1.
    monthly.drop(columns=["NDVI_clean"], errors="ignore").to_csv(
        output / "monthly_model_input.csv", index=False
    )
    if ndvi_qa is not None:
        ndvi_export = pd.read_csv(ndvi_qa)
        ndvi_export.to_csv(output / "ndvi_monthly_qa_2001_2025.csv", index=False)

    summary = _summary(master, support, appeears_path is not None, ndvi_qa is not None)
    summary.update({
        "monthly_input": str(monthly_path),
        "appeears_mod17": str(appeears_path) if appeears_path else None,
        "ndvi_qa": str(ndvi_qa) if ndvi_qa else None,
    })
    (output / "PREPARATION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
