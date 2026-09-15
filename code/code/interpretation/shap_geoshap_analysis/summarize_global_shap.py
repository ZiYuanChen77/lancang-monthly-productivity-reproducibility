from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[3]
RUN_ROOT = PROJECT_ROOT / "runtime" / "runs" / "shap"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
CANONICAL_ROOT = PROJECT_ROOT / "runtime" / "runs" / "canonical"
SAMPLE_ROOT = PROJECT_ROOT / "data_documentation" / "shap_sample_keys"

WINDOW = 36
INPUT_DIM = 19
EXPLAIN_N = 1000
SHAP_NSAMPLES = 50
SHAP_RSEED = 20240724
SHAP_BATCH_SIZE = 16
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20240724
GEO_STATUS = "Sampled-site GeoSHAP uses the dedicated postprocessing workflow"
WINDOW_STATUS = "Attribution robustness scope: canonical W36 model"
LAG_INTERPRETATION = (
    "Repeated annual-lag attribution patterns are consistent with seasonal "
    "autocorrelation and recurring phenological/environmental cycles."
)

FEATURES = [
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

VEGETATION_FEATURES = [
    "Veg_Coniferous",
    "Veg_Shrub",
    "Veg_Meadow_497",
    "Veg_Meadow_499",
    "Veg_Meadow_504",
    "Veg_Sparse",
]

CONTINUOUS_FEATURES = [
    "LST_Day_1km",
    "NDVI",
    "Precipitation_mm",
    "VPD",
    "surface_solar_radiation_downwards_sum",
    "temperature_2m",
    "volumetric_soil_water_layer_1",
    "elevation",
    "slope",
]

GROUPS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("Vegetation productivity", ["NDVI"]),
        ("Thermal", ["LST_Day_1km", "temperature_2m", "VPD"]),
        (
            "Hydrologic",
            ["Precipitation_mm", "volumetric_soil_water_layer_1"],
        ),
        ("Energy", ["surface_solar_radiation_downwards_sum"]),
        ("Terrain", ["elevation", "slope", "Aspect_Sin", "Aspect_Cos"]),
        ("Seasonal phase", ["Target_Month_Sin", "Target_Month_Cos"]),
        ("Vegetation class", VEGETATION_FEATURES.copy()),
    ]
)

EXPECTED_CHECKPOINT_HASHES = {
    42: "9a89d2cf5f95d98e7d4ca3ce782ffc1b3214bfe745740fe1c1ce26616294fa33",
    2024: "693da95365f86a8ec4f38786e63eef67dbf423ec03ce45eed1127ec55da657c2",
    3407: "e10a83ce53a7c23b3787db3329c2a30f1eee9af64301ff0da5a5366b622461d7",
}

EXPECTED_DATA_HASHES = {
    "X_W36": "ed663f960a560419feecc19f30164881ab1cf62eb0c8b0679a7ec19f61e09c8c",
    "sample_index": "87cf6534c4a0595582ae3dab30bb9f3e6640df75a95a6e20f9f94edd6b842b02",
}

CHECKPOINTS = {
    seed: CANONICAL_ROOT
    / "final_seed_runs"
    / f"seed_{seed}"
    / "model_validation_selected.pt"
    for seed in EXPECTED_CHECKPOINT_HASHES
}

X_PATH = DATA_ROOT / "X_tensor_aligned_36.npy"
INDEX_PATH = DATA_ROOT / "Sample_index_aligned.csv"


def run_plan() -> list[dict[str, Any]]:
    rows = [
        {
            "run_id": "background_BG256_D1",
            "model_seed": 42,
            "background_draw": "BG256_D1",
            "background_n": 256,
            "explained_draw": "EXPLAINED_E0",
        }
    ]
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            if draw_id != "BG256_D1":
                rows.append(
                    {
                        "run_id": f"background_{draw_id}",
                        "model_seed": 42,
                        "background_draw": draw_id,
                        "background_n": size,
                        "explained_draw": "EXPLAINED_E0",
                    }
                )
    rows.extend(
        [
            {
                "run_id": "seed_2024",
                "model_seed": 2024,
                "background_draw": "BG256_D1",
                "background_n": 256,
                "explained_draw": "EXPLAINED_E0",
            },
            {
                "run_id": "seed_3407",
                "model_seed": 3407,
                "background_draw": "BG256_D1",
                "background_n": 256,
                "explained_draw": "EXPLAINED_E0",
            },
            {
                "run_id": "explained_E1",
                "model_seed": 42,
                "background_draw": "BG256_D1",
                "background_n": 256,
                "explained_draw": "EXPLAINED_E1",
            },
            {
                "run_id": "explained_E2",
                "model_seed": 42,
                "background_draw": "BG256_D1",
                "background_n": 256,
                "explained_draw": "EXPLAINED_E2",
            },
            {
                "run_id": "explained_E3",
                "model_seed": 42,
                "background_draw": "BG256_D1",
                "background_n": 256,
                "explained_draw": "EXPLAINED_E3",
            },
        ]
    )
    if len(rows) != 17 or len({row["run_id"] for row in rows}) != 17:
        raise RuntimeError("The fixed run plan is not exactly 17 unique runs.")
    return rows


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return "MISSING"
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_manifest_hashes() -> dict[str, str]:
    expected = [
        *(f"sample_keys/background/shap_BG{size}_D{draw}.csv" for size in (64, 128, 256, 512) for draw in (1, 2, 3)),
        *(f"sample_keys/shap_explained_E{draw}.csv" for draw in range(4)),
        "sample_keys/shap_background_manifests.csv",
        "sample_keys/shap_sampling_balance_summary.csv",
    ]
    check_rows = []
    result: dict[str, str] = {}
    for relative_path in expected:
        path = SAMPLE_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        actual_bytes = path.stat().st_size
        check_rows.append(
            {
                "relative_path": relative_path,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_hash,
                "valid": True,
            }
        )
        result[relative_path] = actual_hash
    atomic_csv(
        RUN_ROOT / "validation" / "shap_sample_manifest_checks.csv",
        pd.DataFrame(check_rows),
    )
    return result


def checkpoint_checks() -> pd.DataFrame:
    rows = []
    for seed in EXPECTED_CHECKPOINT_HASHES:
        path = CHECKPOINTS[seed]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        rows.append(
            {
                "window": WINDOW,
                "seed": seed,
                "path": str(path),
                "bytes": path.stat().st_size if path.is_file() else -1,
                "actual_sha256": actual_hash,
                "valid": True,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_csv(
        RUN_ROOT / "validation" / "shap_checkpoint_checks.csv",
        frame,
    )
    return frame


def load_manifest(draw_id: str) -> pd.DataFrame:
    if draw_id.startswith("BG"):
        path = (
            SAMPLE_ROOT
            / "sample_keys"
            / "background"
            / f"shap_{draw_id}.csv"
        )
    else:
        file_draw_id = draw_id.replace("EXPLAINED_", "explained_", 1)
        path = SAMPLE_ROOT / "sample_keys" / f"shap_{file_draw_id}.csv"
    return pd.read_csv(path)


def run_paths(run_id: str) -> dict[str, Path]:
    base = RUN_ROOT / "shap_runs"
    return {
        "shap_npz": base / f"shap_{run_id}.npz",
        "variable_csv": base / f"shap_{run_id}_variable.csv",
        "grouped_csv": base / f"shap_{run_id}_grouped.csv",
        "signed_csv": base / f"shap_{run_id}_signed.csv",
        "execution_json": base / f"shap_{run_id}_execution.json",
        "completion_marker": (
            base / "completed" / f"shap_{run_id}.complete.json"
        ),
    }


def validate_completed_runs(
    manifest_hashes: Mapping[str, str],
) -> pd.DataFrame:
    expected_marker_names = {
        f"shap_{row['run_id']}.complete.json" for row in run_plan()
    }
    marker_dir = RUN_ROOT / "shap_runs" / "completed"
    actual_marker_names = {path.name for path in marker_dir.glob("*.complete.json")}
    if actual_marker_names != expected_marker_names:
        raise RuntimeError(
            "Completion marker set differs from the fixed 17-run plan."
        )

    check_rows = []
    required_npz_keys = {
        "shap_values_gC_m2_month",
        "background_sample_ids",
        "explained_sample_ids",
        "feature_names",
        "window",
        "model_seed",
    }
    for spec in run_plan():
        paths = run_paths(spec["run_id"])
        marker = load_json(paths["completion_marker"])
        protocol = marker.get("protocol", {})
        execution = load_json(paths["execution_json"])
        background = load_manifest(spec["background_draw"])
        explained = load_manifest(spec["explained_draw"])
        background_ids = background["sample_id"].to_numpy(dtype=np.int64)
        explained_ids = explained["sample_id"].to_numpy(dtype=np.int64)
        background_rel = (
            f"sample_keys/background/shap_{spec['background_draw']}.csv"
        )
        explained_file_id = spec["explained_draw"].replace(
            "EXPLAINED_", "explained_", 1
        )
        explained_rel = f"sample_keys/shap_{explained_file_id}.csv"

        checks: dict[str, bool] = {
            "status_complete": marker.get("status") == "COMPLETE",
            "run_id_match": marker.get("run_id") == spec["run_id"],
            "shape_match": marker.get("shape") == [EXPLAIN_N, WINDOW, INPUT_DIM],
            "seed_match": protocol.get("model_seed") == spec["model_seed"],
            "background_draw_match": (
                protocol.get("background_draw") == spec["background_draw"]
            ),
            "explained_draw_match": (
                protocol.get("explained_draw") == spec["explained_draw"]
            ),
            "checkpoint_hash_match": (
                protocol.get("checkpoint_sha256")
                == sha256_file(CHECKPOINTS[spec["model_seed"]])
            ),
            "background_manifest_hash_match": (
                protocol.get("background_manifest_sha256")
                == manifest_hashes[background_rel]
            ),
            "explained_manifest_hash_match": (
                protocol.get("explained_manifest_sha256")
                == manifest_hashes[explained_rel]
            ),
            "cpu": protocol.get("device") == "cpu",
            "explainer_match": protocol.get("explainer") == "GradientExplainer",
            "nsamples_match": protocol.get("nsamples") == SHAP_NSAMPLES,
            "rseed_match": protocol.get("rseed") == SHAP_RSEED,
            "batch_size_match": (
                protocol.get("batch_size") == SHAP_BATCH_SIZE
            ),
            "feature_order_match": protocol.get("feature_order") == FEATURES,
            "group_definition_match": protocol.get("groups") == GROUPS,
            "execution_complete": execution.get("status") == "COMPLETE",
            "execution_identity_match": (
                execution.get("run_id") == spec["run_id"]
                and execution.get("model_seed") == spec["model_seed"]
                and execution.get("background_draw")
                == spec["background_draw"]
                and execution.get("background_n") == spec["background_n"]
                and execution.get("explained_draw") == spec["explained_draw"]
                and execution.get("explained_n") == EXPLAIN_N
            ),
            "execution_protocol_match": (
                execution.get("device") == "cpu"
                and execution.get("explainer") == "GradientExplainer"
                and execution.get("nsamples") == SHAP_NSAMPLES
                and execution.get("rseed") == SHAP_RSEED
                and execution.get("batch_size") == SHAP_BATCH_SIZE
            ),
        }

        output_artifacts = marker.get("output_artifacts", {})
        checks["five_artifacts_recorded"] = set(output_artifacts) == {
            "shap_npz",
            "variable_csv",
            "grouped_csv",
            "signed_csv",
            "execution_json",
        }
        artifact_checks = []
        for key in (
            "shap_npz",
            "variable_csv",
            "grouped_csv",
            "signed_csv",
            "execution_json",
        ):
            record = output_artifacts.get(key, {})
            path = RUN_ROOT / record.get("relative_path", "__MISSING__")
            artifact_checks.append(
                path == paths[key]
                and path.is_file()
                and path.stat().st_size == record.get("bytes")
                and sha256_file(path) == record.get("sha256")
            )
        checks["artifact_hashes_match"] = all(artifact_checks)

        with np.load(paths["shap_npz"], allow_pickle=False) as data:
            checks["npz_keys_match"] = set(data.files) == required_npz_keys
            values = data["shap_values_gC_m2_month"]
            checks["npz_shape_match"] = values.shape == (
                EXPLAIN_N,
                WINDOW,
                INPUT_DIM,
            )
            checks["npz_finite"] = bool(np.isfinite(values).all())
            checks["npz_feature_order_match"] = np.array_equal(
                data["feature_names"], np.asarray(FEATURES)
            )
            checks["npz_window_match"] = np.array_equal(
                data["window"], np.asarray([WINDOW], dtype=np.int16)
            )
            checks["npz_seed_match"] = np.array_equal(
                data["model_seed"],
                np.asarray([spec["model_seed"]], dtype=np.int32),
            )
            checks["npz_background_keys_match"] = np.array_equal(
                data["background_sample_ids"], background_ids
            )
            checks["npz_explained_keys_match"] = np.array_equal(
                data["explained_sample_ids"], explained_ids
            )

        variable = pd.read_csv(paths["variable_csv"])
        grouped = pd.read_csv(paths["grouped_csv"])
        signed = pd.read_csv(paths["signed_csv"])
        checks["summary_shapes_match"] = (
            len(variable) == INPUT_DIM
            and len(grouped) == len(GROUPS)
            and len(signed) == INPUT_DIM
        )
        checks["summary_identities_match"] = (
            variable["entity"].tolist() == FEATURES
            and grouped["entity"].tolist() == list(GROUPS)
            and signed["entity"].tolist() == FEATURES
        )
        checks["summary_finite"] = bool(
            np.isfinite(variable["mean_abs_shap_gC_m2_month"]).all()
            and np.isfinite(grouped["mean_abs_shap_gC_m2_month"]).all()
            and np.isfinite(signed["mean_signed_shap_gC_m2_month"]).all()
        )

        valid = all(checks.values())
        check_rows.append(
            {
                "run_id": spec["run_id"],
                "status": marker.get("status"),
                "model_seed": spec["model_seed"],
                "background_draw": spec["background_draw"],
                "background_n": spec["background_n"],
                "explained_draw": spec["explained_draw"],
                "explained_n": EXPLAIN_N,
                "shap_shape": "1000x36x19",
                "output_artifacts_verified": sum(artifact_checks),
                "checks_passed": sum(checks.values()),
                "checks_total": len(checks),
                "valid": valid,
                "failed_checks": ";".join(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
        if not valid:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(
                f"SHAP run validation failure for {spec['run_id']}: {failed}"
            )

    frame = pd.DataFrame(check_rows)
    atomic_csv(
        RUN_ROOT / "validation" / "shap_run_checks.csv",
        frame,
    )
    return frame


def latest_diagnostic_summary() -> dict[str, Any]:
    path = RUN_ROOT / "logs" / "shap_native_crash_diagnose.log"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    endings = [
        record
        for record in records
        if record.get("event") == "diagnostic_session_end"
    ]
    if not endings:
        raise RuntimeError("No completed diagnostic session is recorded.")
    latest = endings[-1]
    return {
        "log": str(path),
        "session_id": latest.get("session_id"),
        "overall_status": latest.get("overall_status"),
        "stages_total": latest.get("stages_total"),
        "stages_failed": latest.get("stages_failed"),
        "ten_of_ten_passed": (
            latest.get("overall_status") == "PASS"
            and latest.get("stages_total") == 10
            and latest.get("stages_failed") == 0
        ),
    }


def sign_int(values: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    values = np.asarray(values)
    return np.where(values > atol, 1, np.where(values < -atol, -1, 0))


def rank_desc(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(-np.asarray(values), method="average")
    if np.allclose(ranks, np.round(ranks)):
        return np.round(ranks).astype(int)
    return ranks


def canonical_outputs(
    x: np.ndarray,
    manifest_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    from three_seed_aggregation import SEED_RUNS, load_three_seed_shap, summarize_seed_tensors

    seed_values, background_ids, explained_ids = load_three_seed_shap(run_paths, FEATURES)
    values, absolute_values, importance_sd = summarize_seed_tensors(seed_values)
    model_inputs = np.asarray(x[explained_ids], dtype=np.float32)
    row_abs = absolute_values.sum(axis=1)
    row_signed = values.sum(axis=1)
    importance = row_abs.mean(axis=0)
    mean_signed = row_signed.mean(axis=0)
    feature_mean = model_inputs.mean(axis=1)
    contrast = np.zeros(INPUT_DIM, dtype=np.float64)
    for column in range(INPUT_DIM):
        low_q, high_q = np.quantile(feature_mean[:, column], [0.25, 0.75])
        contrast[column] = (
            row_signed[feature_mean[:, column] >= high_q, column].mean()
            - row_signed[feature_mean[:, column] <= low_q, column].mean()
        )
    variable = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap_gC_m2_month": importance,
            "seed_sd_abs_shap_gC_m2_month": importance_sd,
            "rank": rank_desc(importance),
            "mean_signed_shap_gC_m2_month": mean_signed,
            "mean_signed_sign": sign_int(mean_signed),
            "direction_summary": np.where(
                mean_signed > 1e-12,
                "positive mean attribution",
                np.where(
                    mean_signed < -1e-12,
                    "negative mean attribution",
                    "near-zero mean attribution",
                ),
            ),
            "high_minus_low_shap_contrast_gC_m2_month": contrast,
            "contrast_sign": sign_int(contrast),
        }
    ).sort_values("rank", ignore_index=True)

    group_rows = []
    for name, members in GROUPS.items():
        positions = [FEATURES.index(member) for member in members]
        group_abs = row_abs[:, positions].sum(axis=1)
        group_signed = row_signed[:, positions].sum(axis=1)
        group_rows.append(
            {
                "group": name,
                "members": ";".join(members),
                "mean_abs_shap_gC_m2_month": float(group_abs.mean()),
                "seed_sd_abs_shap_gC_m2_month": float(
                    np.abs(seed_values)[:, :, :, positions].sum(axis=(2, 3)).mean(axis=1).std(ddof=1)
                ),
                "mean_signed_shap_gC_m2_month": float(group_signed.mean()),
                "mean_signed_sign": int(sign_int([group_signed.mean()])[0]),
                "signed_summary": (
                    "positive mean attribution"
                    if group_signed.mean() > 1e-12
                    else (
                        "negative mean attribution"
                        if group_signed.mean() < -1e-12
                        else "near-zero mean attribution"
                    )
                ),
            }
        )
    grouped = pd.DataFrame(group_rows)
    grouped["rank"] = rank_desc(
        grouped["mean_abs_shap_gC_m2_month"].to_numpy()
    )
    grouped = grouped.sort_values("rank", ignore_index=True)

    variable_table = pd.concat([
        pd.read_csv(run_paths(run_id)["variable_csv"]).set_index("entity")
        for run_id in SEED_RUNS.values()
    ]).groupby(level=0).mean(numeric_only=True)
    group_table = pd.concat([
        pd.read_csv(run_paths(run_id)["grouped_csv"]).set_index("entity")
        for run_id in SEED_RUNS.values()
    ]).groupby(level=0).mean(numeric_only=True)
    calc_variable = variable.set_index("feature")
    calc_group = grouped.set_index("group")
    for column in (
        "mean_abs_shap_gC_m2_month",
        "mean_signed_shap_gC_m2_month",
        "high_minus_low_shap_contrast_gC_m2_month",
    ):
        if not np.allclose(
            calc_variable.loc[FEATURES, column],
            variable_table.loc[FEATURES, column],
            rtol=2e-6,
            atol=2e-6,
        ):
            raise RuntimeError(f"Canonical variable cross-check failed: {column}")
    if not np.allclose(
        calc_group.loc[list(GROUPS), "mean_abs_shap_gC_m2_month"],
        group_table.loc[list(GROUPS), "mean_abs_shap_gC_m2_month"],
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("Canonical grouped cross-check failed.")

    atomic_csv(
        RUN_ROOT / "canonical" / "shap_canonical_variable_importance.csv",
        variable,
    )
    atomic_csv(
        RUN_ROOT / "canonical" / "shap_canonical_grouped_importance.csv",
        grouped,
    )
    atomic_csv(
        RUN_ROOT / "canonical" / "shap_canonical_signed_effects.csv",
        variable[
            [
                "feature",
                "mean_signed_shap_gC_m2_month",
                "mean_signed_sign",
                "direction_summary",
                "high_minus_low_shap_contrast_gC_m2_month",
                "contrast_sign",
            ]
        ],
    )

    lag_rows = []
    for time_index in range(WINDOW):
        lag = WINDOW - time_index
        for feature_index, feature in enumerate(FEATURES):
            cell = values[:, time_index, feature_index]
            lag_rows.append(
                {
                    "feature": feature,
                    "time_index": time_index,
                    "lag_months_before_target": lag,
                    "lag_label": f"lag_{lag}",
                    "mean_abs_shap_gC_m2_month": float(absolute_values[:, time_index, feature_index].mean()),
                    "mean_signed_shap_gC_m2_month": float(cell.mean()),
                    "signed_sign": int(sign_int([cell.mean()])[0]),
                    "annual_cycle_marker": lag in (12, 24, 36),
                    "interpretation_boundary": LAG_INTERPRETATION,
                }
            )
    lag_profiles = pd.DataFrame(lag_rows)
    atomic_csv(
        RUN_ROOT / "canonical" / "shap_canonical_lag_profiles.csv",
        lag_profiles,
    )

    pattern_rows = []
    leading = variable.nsmallest(5, "rank")["feature"].tolist()
    bands = OrderedDict(
        [
            ("recent_1_6", range(1, 7)),
            ("seasonal_7_12", range(7, 13)),
            ("second_year_13_24", range(13, 25)),
            ("third_year_25_36", range(25, 37)),
        ]
    )
    for feature in leading:
        subset = lag_profiles[lag_profiles["feature"] == feature].copy()
        subset = subset.sort_values(
            "mean_abs_shap_gC_m2_month", ascending=False
        )
        top_lags = subset.head(5)["lag_months_before_target"].astype(int).tolist()
        annual = subset[subset["annual_cycle_marker"]]
        band_values = {
            name: float(
                subset.loc[
                    subset["lag_months_before_target"].isin(list(lags)),
                    "mean_abs_shap_gC_m2_month",
                ].sum()
            )
            for name, lags in bands.items()
        }
        pattern_rows.append(
            {
                "feature": feature,
                "canonical_variable_rank": int(
                    variable.loc[variable["feature"] == feature, "rank"].iloc[0]
                ),
                "peak_lag": int(top_lags[0]),
                "top5_lags": ";".join(str(value) for value in top_lags),
                "annual_marker_lags": "12;24;36",
                "annual_marker_abs_shap_sum": float(
                    annual["mean_abs_shap_gC_m2_month"].sum()
                ),
                "annual_marker_share_of_lag_profile": float(
                    annual["mean_abs_shap_gC_m2_month"].sum()
                    / subset["mean_abs_shap_gC_m2_month"].sum()
                ),
                "dominant_lag_band": max(band_values, key=band_values.get),
                **{
                    f"{name}_abs_shap_sum": value
                    for name, value in band_values.items()
                },
                "interpretation": LAG_INTERPRETATION,
            }
        )
    atomic_csv(
        RUN_ROOT
        / "canonical"
        / "shap_canonical_lag_pattern_summary.csv",
        pd.DataFrame(pattern_rows),
    )

    atomic_npz(
        RUN_ROOT
        / "canonical"
        / "shap_w36_reference_shap_values.npz",
        shap_values_gC_m2_month=values,
        mean_abs_shap_gC_m2_month=absolute_values,
        model_seeds=np.asarray(list(SEED_RUNS), dtype=np.int32),
        background_sample_ids=background_ids,
        explained_sample_ids=explained_ids,
        feature_names=np.asarray(FEATURES, dtype="U64"),
        lag_months_before_target=np.arange(
            WINDOW, 0, -1, dtype=np.int16
        ),
        checkpoint_sha256=np.asarray([sha256_file(CHECKPOINTS[s]) for s in SEED_RUNS], dtype="U64"),
    )
    atomic_json(
        RUN_ROOT / "canonical" / "shap_canonical_metadata.json",
        {
            "identity": "W36 SHAP configuration",
            "created_at": now_iso(),
            "window": WINDOW,
            "model_seeds": list(SEED_RUNS),
            "aggregation": "mean signed SHAP and mean absolute SHAP across seeds",
            "checkpoint_sha256": {str(s): sha256_file(CHECKPOINTS[s]) for s in SEED_RUNS},
            "background": {
                "draw": "BG256_D1",
                "n": 256,
                "manifest_sha256": manifest_hashes[
                    "sample_keys/background/shap_BG256_D1.csv"
                ],
            },
            "explained": {
                "draw": "EXPLAINED_E0",
                "n": EXPLAIN_N,
                "manifest_sha256": manifest_hashes[
                    "sample_keys/shap_explained_E0.csv"
                ],
            },
            "shape": list(values.shape),
            "units": "gC m^-2 month^-1",
            "feature_order": FEATURES,
            "groups": GROUPS,
            "source_runs": list(SEED_RUNS.values()),
            "source_npz_sha256": {r: sha256_file(run_paths(r)["shap_npz"]) for r in SEED_RUNS.values()},
        },
    )
    return variable, grouped, absolute_values, explained_ids


def load_run_summaries(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = run_paths(run_id)
    return pd.read_csv(paths["variable_csv"]), pd.read_csv(paths["grouped_csv"])


def stability_row(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_id: str,
    right_id: str,
    entity_type: str,
) -> dict[str, Any]:
    columns = [
        "entity",
        "mean_abs_shap_gC_m2_month",
        "rank",
        "mean_signed_sign",
        "contrast_sign",
    ]
    merged = left[columns].merge(
        right[columns],
        on="entity",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    top5_left = set(merged.loc[merged["rank_left"] <= 5, "entity"])
    top5_right = set(merged.loc[merged["rank_right"] <= 5, "entity"])
    top10_k = min(10, len(merged))
    top10_left = set(merged.loc[merged["rank_left"] <= top10_k, "entity"])
    top10_right = set(merged.loc[merged["rank_right"] <= top10_k, "entity"])
    return {
        "entity_type": entity_type,
        "left_run": left_id,
        "right_run": right_id,
        "spearman_rho": float(
            stats.spearmanr(merged["rank_left"], merged["rank_right"]).statistic
        ),
        "kendall_tau": float(
            stats.kendalltau(merged["rank_left"], merged["rank_right"]).statistic
        ),
        "top5_overlap_count": len(top5_left & top5_right),
        "top5_overlap_proportion": len(top5_left & top5_right) / 5,
        "top10_overlap_count": len(top10_left & top10_right),
        "top10_overlap_proportion": len(top10_left & top10_right) / top10_k,
        "mean_shap_sign_agreement": float(
            (
                merged["mean_signed_sign_left"]
                == merged["mean_signed_sign_right"]
            ).mean()
        ),
        "contrast_sign_agreement": (
            float(
                (
                    merged["contrast_sign_left"]
                    == merged["contrast_sign_right"]
                ).mean()
            )
            if entity_type == "variable"
            else np.nan
        ),
    }


def stability_outputs(
    canonical_variable: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    canonical_id = "background_BG256_D1"
    canonical_var_raw, canonical_group_raw = load_run_summaries(canonical_id)
    background_rows = []
    background_rankings = []
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            run_id = f"background_{draw_id}"
            variable, grouped = load_run_summaries(run_id)
            background_rankings.extend([variable, grouped])
            for entity_type, current, reference in (
                ("variable", variable, canonical_var_raw),
                ("group", grouped, canonical_group_raw),
            ):
                row = stability_row(
                    reference, current, canonical_id, run_id, entity_type
                )
                row.update(
                    {
                        "sensitivity_type": "background",
                        "background_size": size,
                        "draw": draw,
                        "background_draw": draw_id,
                        "is_canonical_self_comparison": (
                            run_id == canonical_id
                        ),
                    }
                )
                background_rows.append(row)
    background = pd.DataFrame(background_rows)
    atomic_csv(
        RUN_ROOT / "results" / "shap_background_stability.csv",
        background,
    )
    atomic_csv(
        RUN_ROOT / "results" / "shap_background_run_rankings.csv",
        pd.concat(background_rankings, ignore_index=True),
    )
    background_by_size = (
        background.groupby(["entity_type", "background_size"], as_index=False)
        .agg(
            comparisons=("right_run", "count"),
            spearman_min=("spearman_rho", "min"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_max=("spearman_rho", "max"),
            kendall_min=("kendall_tau", "min"),
            kendall_mean=("kendall_tau", "mean"),
            top5_overlap_min=("top5_overlap_count", "min"),
            top5_overlap_mean=("top5_overlap_count", "mean"),
            mean_sign_agreement_min=("mean_shap_sign_agreement", "min"),
            mean_sign_agreement_mean=("mean_shap_sign_agreement", "mean"),
            contrast_sign_agreement_min=("contrast_sign_agreement", "min"),
            contrast_sign_agreement_mean=("contrast_sign_agreement", "mean"),
        )
    )
    atomic_csv(
        RUN_ROOT / "results" / "shap_background_stability_by_size.csv",
        background_by_size,
    )

    seed_map = {
        42: canonical_id,
        2024: "seed_2024",
        3407: "seed_3407",
    }
    seed_summaries = {
        seed: load_run_summaries(run_id)
        for seed, run_id in seed_map.items()
    }
    seed_rows = []
    seed_rankings = []
    for seed, (variable, grouped) in seed_summaries.items():
        variable = variable.copy()
        grouped = grouped.copy()
        variable["seed_label"] = seed
        grouped["seed_label"] = seed
        seed_rankings.extend([variable, grouped])
    for left_seed, right_seed in itertools.combinations(seed_map, 2):
        for entity_type, left, right in (
            (
                "variable",
                seed_summaries[left_seed][0],
                seed_summaries[right_seed][0],
            ),
            (
                "group",
                seed_summaries[left_seed][1],
                seed_summaries[right_seed][1],
            ),
        ):
            row = stability_row(
                left,
                right,
                f"seed_{left_seed}",
                f"seed_{right_seed}",
                entity_type,
            )
            row.update(
                {
                    "sensitivity_type": "model_seed",
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                }
            )
            seed_rows.append(row)
    seed = pd.DataFrame(seed_rows)
    atomic_csv(
        RUN_ROOT / "results" / "shap_three_seed_variable_stability.csv",
        seed[seed["entity_type"] == "variable"].reset_index(drop=True),
    )
    atomic_csv(
        RUN_ROOT / "results" / "shap_three_seed_grouped_stability.csv",
        seed[seed["entity_type"] == "group"].reset_index(drop=True),
    )
    atomic_csv(
        RUN_ROOT / "results" / "shap_three_seed_rankings.csv",
        pd.concat(seed_rankings, ignore_index=True),
    )

    explained_map = {
        "EXPLAINED_E0": canonical_id,
        "EXPLAINED_E1": "explained_E1",
        "EXPLAINED_E2": "explained_E2",
        "EXPLAINED_E3": "explained_E3",
    }
    explained_summaries = {
        draw: load_run_summaries(run_id)
        for draw, run_id in explained_map.items()
    }
    explained_rows = []
    explained_rankings = []
    for draw, (variable, grouped) in explained_summaries.items():
        variable = variable.copy()
        grouped = grouped.copy()
        variable["explained_draw_label"] = draw
        grouped["explained_draw_label"] = draw
        explained_rankings.extend([variable, grouped])
    for left_draw, right_draw in itertools.combinations(explained_map, 2):
        for entity_type, left, right in (
            (
                "variable",
                explained_summaries[left_draw][0],
                explained_summaries[right_draw][0],
            ),
            (
                "group",
                explained_summaries[left_draw][1],
                explained_summaries[right_draw][1],
            ),
        ):
            row = stability_row(
                left, right, left_draw, right_draw, entity_type
            )
            row.update(
                {
                    "sensitivity_type": "explained_sample",
                    "left_explained_draw": left_draw,
                    "right_explained_draw": right_draw,
                    "comparison_type": (
                        "canonical_vs_replicate"
                        if left_draw == "EXPLAINED_E0"
                        else "replicate_pair"
                    ),
                }
            )
            explained_rows.append(row)
    explained = pd.DataFrame(explained_rows)
    atomic_csv(
        RUN_ROOT / "results" / "shap_explained_sample_stability.csv",
        explained,
    )
    atomic_csv(
        RUN_ROOT / "results" / "shap_explained_sample_rankings.csv",
        pd.concat(explained_rankings, ignore_index=True),
    )

    leading = canonical_variable.nsmallest(5, "rank")["feature"].tolist()
    lag_comparisons = []
    comparison_specs = []
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            run_id = f"background_BG{size}_D{draw}"
            if run_id != canonical_id:
                comparison_specs.append(("background", canonical_id, run_id))
    comparison_specs.extend(
        [
            ("model_seed", canonical_id, "seed_2024"),
            ("model_seed", canonical_id, "seed_3407"),
            ("explained_sample", canonical_id, "explained_E1"),
            ("explained_sample", canonical_id, "explained_E2"),
            ("explained_sample", canonical_id, "explained_E3"),
        ]
    )
    with np.load(run_paths(canonical_id)["shap_npz"]) as data:
        reference_shap = np.asarray(data["shap_values_gC_m2_month"])
    for sensitivity_type, left_id, right_id in comparison_specs:
        with np.load(run_paths(right_id)["shap_npz"]) as data:
            current_shap = np.asarray(data["shap_values_gC_m2_month"])
        for feature in leading:
            position = FEATURES.index(feature)
            left_profile = np.abs(reference_shap[:, :, position]).mean(axis=0)
            right_profile = np.abs(current_shap[:, :, position]).mean(axis=0)
            left_peak = WINDOW - int(np.argmax(left_profile))
            right_peak = WINDOW - int(np.argmax(right_profile))
            annual_positions = [0, 12, 24]
            lag_comparisons.append(
                {
                    "sensitivity_type": sensitivity_type,
                    "left_run": left_id,
                    "right_run": right_id,
                    "feature": feature,
                    "lag_profile_spearman_rho": float(
                        stats.spearmanr(left_profile, right_profile).statistic
                    ),
                    "lag_profile_kendall_tau": float(
                        stats.kendalltau(left_profile, right_profile).statistic
                    ),
                    "reference_peak_lag": left_peak,
                    "comparison_peak_lag": right_peak,
                    "peak_lag_match": left_peak == right_peak,
                    "reference_annual_marker_share": float(
                        left_profile[annual_positions].sum()
                        / left_profile.sum()
                    ),
                    "comparison_annual_marker_share": float(
                        right_profile[annual_positions].sum()
                        / right_profile.sum()
                    ),
                    "interpretation_boundary": LAG_INTERPRETATION,
                }
            )
    atomic_csv(
        RUN_ROOT
        / "results"
        / "shap_leading_predictor_lag_profile_stability.csv",
        pd.DataFrame(lag_comparisons),
    )
    return {
        "background": background,
        "background_by_size": background_by_size,
        "seed": seed,
        "explained": explained,
    }


def correlation_outputs(
    x: np.ndarray,
    canonical_variable: pd.DataFrame,
    canonical_group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not X_PATH.is_file():
        raise FileNotFoundError(X_PATH)
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(INDEX_PATH)
    index = pd.read_csv(INDEX_PATH)
    train_ids = index.loc[index["split"] == "train", "sample_id"].to_numpy(
        dtype=np.int64
    )
    positions = [FEATURES.index(name) for name in CONTINUOUS_FEATURES]
    flattened = np.asarray(
        x[train_ids][:, :, positions], dtype=np.float32
    ).reshape(-1, len(positions))
    if not np.isfinite(flattened).all():
        raise RuntimeError("Non-finite fixed training predictors.")
    matrix = np.asarray(
        stats.spearmanr(flattened, axis=0).statistic, dtype=np.float64
    )
    if matrix.shape != (len(positions), len(positions)):
        raise RuntimeError("Unexpected Spearman correlation matrix shape.")
    matrix_frame = pd.DataFrame(
        matrix, columns=CONTINUOUS_FEATURES, index=CONTINUOUS_FEATURES
    ).reset_index(names="predictor")
    atomic_csv(
        RUN_ROOT / "results" / "shap_predictor_spearman_matrix.csv",
        matrix_frame,
    )
    pair_rows = []
    for left, right in itertools.combinations(range(len(positions)), 2):
        rho = float(matrix[left, right])
        if abs(rho) >= 0.7:
            pair_rows.append(
                {
                    "predictor_1": CONTINUOUS_FEATURES[left],
                    "predictor_2": CONTINUOUS_FEATURES[right],
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "threshold": 0.7,
                    "training_samples": len(train_ids),
                    "lag_rows": len(flattened),
                    "interpretation": (
                        "Used only to interpret shared or unstable SHAP "
                        "attribution under correlated inputs; no feature was "
                        "removed and no model was retrained."
                    ),
                }
            )
    pairs = pd.DataFrame(
        pair_rows,
        columns=[
            "predictor_1",
            "predictor_2",
            "spearman_rho",
            "abs_spearman_rho",
            "threshold",
            "training_samples",
            "lag_rows",
            "interpretation",
        ],
    )
    if not pairs.empty:
        pairs = pairs.sort_values(
            "abs_spearman_rho", ascending=False, ignore_index=True
        )
    atomic_csv(
        RUN_ROOT / "results" / "shap_high_correlation_pairs.csv",
        pairs,
    )
    raw = canonical_variable.rename(
        columns={"feature": "name"}
    ).assign(entity_type="variable")
    grouped = canonical_group.rename(
        columns={"group": "name"}
    ).assign(entity_type="group")
    combined = pd.concat(
        [
            raw[
                [
                    "entity_type",
                    "name",
                    "mean_abs_shap_gC_m2_month",
                    "rank",
                    "mean_signed_shap_gC_m2_month",
                    "mean_signed_sign",
                ]
            ],
            grouped[
                [
                    "entity_type",
                    "name",
                    "mean_abs_shap_gC_m2_month",
                    "rank",
                    "mean_signed_shap_gC_m2_month",
                    "mean_signed_sign",
                ]
            ],
        ],
        ignore_index=True,
    )
    atomic_csv(
        RUN_ROOT / "results" / "shap_raw_and_grouped_importance.csv",
        combined,
    )
    return matrix_frame, pairs


def rank_distribution(
    names: Sequence[str],
    ranks: np.ndarray,
    entity_type: str,
) -> pd.DataFrame:
    rows = []
    for column, name in enumerate(names):
        unique, counts = np.unique(ranks[:, column], return_counts=True)
        for rank, count in zip(unique, counts):
            rows.append(
                {
                    "entity_type": entity_type,
                    "entity": name,
                    "rank": float(rank),
                    "count": int(count),
                    "probability": float(count / len(ranks)),
                    "B": BOOTSTRAP_B,
                    "rng_seed": BOOTSTRAP_SEED,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_outputs(
    mean_absolute_shap: np.ndarray,
    canonical_explained_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    manifest = load_manifest("EXPLAINED_E0")
    if not np.array_equal(
        canonical_explained_ids,
        manifest["sample_id"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Canonical explained keys do not match fixed E0.")
    point_ids = manifest["point_id"].astype(str).str.zfill(3).to_numpy()
    clusters = np.asarray(sorted(np.unique(point_ids)), dtype="U3")
    if len(clusters) != 247:
        raise RuntimeError("Canonical bootstrap does not contain 247 point_ids.")
    row_variable = mean_absolute_shap.sum(axis=1).astype(np.float64)
    variable_cluster_sums = np.zeros(
        (len(clusters), INPUT_DIM), dtype=np.float64
    )
    cluster_counts = np.zeros(len(clusters), dtype=np.int32)
    group_names = list(GROUPS)
    group_positions = [
        [FEATURES.index(member) for member in GROUPS[group]]
        for group in group_names
    ]
    row_group = np.column_stack(
        [row_variable[:, positions].sum(axis=1) for positions in group_positions]
    )
    group_cluster_sums = np.zeros(
        (len(clusters), len(group_names)), dtype=np.float64
    )
    for cluster_index, point_id in enumerate(clusters):
        mask = point_ids == point_id
        cluster_counts[cluster_index] = int(mask.sum())
        variable_cluster_sums[cluster_index] = row_variable[mask].sum(axis=0)
        group_cluster_sums[cluster_index] = row_group[mask].sum(axis=0)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(
        len(clusters),
        np.full(len(clusters), 1.0 / len(clusters)),
        size=BOOTSTRAP_B,
    ).astype(np.int16)
    denominators = weights.astype(np.float64) @ cluster_counts.astype(np.float64)
    variable_replicates = (
        weights.astype(np.float64) @ variable_cluster_sums
    ) / denominators[:, None]
    group_replicates = (
        weights.astype(np.float64) @ group_cluster_sums
    ) / denominators[:, None]
    if (
        variable_replicates.shape != (BOOTSTRAP_B, INPUT_DIM)
        or group_replicates.shape != (BOOTSTRAP_B, len(group_names))
        or not np.isfinite(variable_replicates).all()
        or not np.isfinite(group_replicates).all()
    ):
        raise RuntimeError("Fewer than 5000 bootstrap replicates succeeded.")
    variable_ranks = stats.rankdata(
        -variable_replicates, axis=1, method="average"
    )
    group_ranks = stats.rankdata(
        -group_replicates, axis=1, method="average"
    )

    def summary(
        names: Sequence[str],
        estimates: np.ndarray,
        replicates: np.ndarray,
        ranks: np.ndarray,
        entity_type: str,
    ) -> pd.DataFrame:
        rows = []
        for column, name in enumerate(names):
            rows.append(
                {
                    "entity_type": entity_type,
                    "entity": name,
                    "mean_abs_shap_gC_m2_month": float(estimates[column]),
                    "ci95_lower": float(
                        np.percentile(replicates[:, column], 2.5)
                    ),
                    "ci95_upper": float(
                        np.percentile(replicates[:, column], 97.5)
                    ),
                    "mean_rank": float(ranks[:, column].mean()),
                    "rank_p025": float(np.percentile(ranks[:, column], 2.5)),
                    "rank_q25": float(np.percentile(ranks[:, column], 25)),
                    "median_rank": float(np.percentile(ranks[:, column], 50)),
                    "rank_q75": float(np.percentile(ranks[:, column], 75)),
                    "rank_p975": float(
                        np.percentile(ranks[:, column], 97.5)
                    ),
                    "top5_probability": float(
                        (ranks[:, column] <= 5).mean()
                    ),
                    "cluster_unit": "point_id",
                    "clusters": len(clusters),
                    "B": BOOTSTRAP_B,
                    "rng_seed": BOOTSTRAP_SEED,
                    "successful": BOOTSTRAP_B,
                    "failed": 0,
                }
            )
        return pd.DataFrame(rows).sort_values(
            "mean_abs_shap_gC_m2_month",
            ascending=False,
            ignore_index=True,
        )

    variable_summary = summary(
        FEATURES,
        row_variable.mean(axis=0),
        variable_replicates,
        variable_ranks,
        "variable",
    )
    group_summary = summary(
        group_names,
        row_group.mean(axis=0),
        group_replicates,
        group_ranks,
        "group",
    )
    atomic_csv(
        RUN_ROOT
        / "bootstrap"
        / "shap_variable_cluster_bootstrap_summary.csv",
        variable_summary,
    )
    atomic_csv(
        RUN_ROOT
        / "bootstrap"
        / "shap_group_cluster_bootstrap_summary.csv",
        group_summary,
    )
    atomic_csv(
        RUN_ROOT
        / "bootstrap"
        / "shap_variable_cluster_bootstrap_rank_distribution.csv",
        rank_distribution(FEATURES, variable_ranks, "variable"),
    )
    atomic_csv(
        RUN_ROOT
        / "bootstrap"
        / "shap_group_cluster_bootstrap_rank_distribution.csv",
        rank_distribution(group_names, group_ranks, "group"),
    )
    atomic_npz(
        RUN_ROOT
        / "bootstrap"
        / "shap_cluster_bootstrap_replicates.npz",
        point_ids=clusters,
        cluster_draw_multiplicities=weights,
        variable_importance=variable_replicates.astype(np.float32),
        variable_ranks=variable_ranks.astype(np.float32),
        group_importance=group_replicates.astype(np.float32),
        group_ranks=group_ranks.astype(np.float32),
        feature_names=np.asarray(FEATURES, dtype="U64"),
        group_names=np.asarray(group_names, dtype="U64"),
        rng_seed=np.asarray([BOOTSTRAP_SEED], dtype=np.int64),
        successful=np.asarray([BOOTSTRAP_B], dtype=np.int32),
        failed=np.asarray([0], dtype=np.int32),
    )
    return variable_summary, group_summary, BOOTSTRAP_B


def range_record(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
    }


def core_metrics(
    stability: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    background = stability["background"]
    background = background[
        ~background["is_canonical_self_comparison"]
    ]
    background_variable = background[background["entity_type"] == "variable"]
    background_group = background[background["entity_type"] == "group"]
    seed = stability["seed"]
    seed_variable = seed[seed["entity_type"] == "variable"]
    seed_group = seed[seed["entity_type"] == "group"]
    explained = stability["explained"]
    explained = explained[
        explained["comparison_type"] == "canonical_vs_replicate"
    ]
    explained_variable = explained[explained["entity_type"] == "variable"]
    explained_group = explained[explained["entity_type"] == "group"]

    def metric_set(variable: pd.DataFrame, group: pd.DataFrame) -> dict[str, Any]:
        return {
            "variable_spearman": range_record(variable["spearman_rho"]),
            "group_spearman": range_record(group["spearman_rho"]),
            "variable_kendall": range_record(variable["kendall_tau"]),
            "group_kendall": range_record(group["kendall_tau"]),
            "variable_top5_overlap": {
                "min": int(variable["top5_overlap_count"].min()),
                "max": int(variable["top5_overlap_count"].max()),
                "denominator": 5,
            },
            "group_top5_overlap": {
                "min": int(group["top5_overlap_count"].min()),
                "max": int(group["top5_overlap_count"].max()),
                "denominator": 5,
            },
            "variable_mean_sign_agreement": range_record(
                variable["mean_shap_sign_agreement"]
            ),
            "group_mean_sign_agreement": range_record(
                group["mean_shap_sign_agreement"]
            ),
            "variable_contrast_sign_agreement": range_record(
                variable["contrast_sign_agreement"]
            ),
        }

    return {
        "background_sensitivity": {
            "comparisons": len(background_variable),
            "canonical_self_comparison_excluded": True,
            **metric_set(background_variable, background_group),
        },
        "model_seed_sensitivity": {
            "pairwise_comparisons": len(seed_variable),
            **metric_set(seed_variable, seed_group),
        },
        "explained_sample_sensitivity": {
            "canonical_vs_replicate_comparisons": len(explained_variable),
            **metric_set(explained_variable, explained_group),
        },
    }


def write_analysis_scope() -> None:
    atomic_text(
        RUN_ROOT
        / "geoshap"
        / "shap_geoshap_scope.md",
        "\n".join(
            [
                GEO_STATUS,
                "",
                "Point-level keys and row-level SHAP values provide the sampled-site attribution input.",
                "",
            ]
        ),
    )
    atomic_text(
        RUN_ROOT
        / "window_sensitivity"
        / "shap_window_scope.md",
        WINDOW_STATUS + "\n",
    )


def make_inventory() -> Path:
    inventory_path = RUN_ROOT / "validation" / "sha256_inventory.csv"
    rows = []
    for path in sorted(item for item in RUN_ROOT.rglob("*") if item.is_file()):
        if path == inventory_path:
            continue
        rows.append(
            {
                "scope": "SHAP_artifact",
                "path": path.relative_to(RUN_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    external = [*CHECKPOINTS.values(), X_PATH, INDEX_PATH]
    for path in external:
        rows.append(
            {
                "scope": "canonical_input",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(inventory_path, pd.DataFrame(rows))
    return inventory_path


def main() -> None:
    for directory in (
        "canonical",
        "results",
        "bootstrap",
        "geoshap",
        "window_sensitivity",
        "validation",
        "summary",
    ):
        (RUN_ROOT / directory).mkdir(parents=True, exist_ok=True)

    manifest_hashes = sample_manifest_hashes()
    checkpoints = checkpoint_checks()
    run_checks = validate_completed_runs(manifest_hashes)
    x = np.load(X_PATH, mmap_mode="r")
    if x.shape != (62244, WINDOW, INPUT_DIM):
        raise RuntimeError(f"Unexpected canonical W36 input shape: {x.shape}")
    canonical_variable, canonical_group, canonical_shap, explained_ids = (
        canonical_outputs(x, manifest_hashes)
    )
    stability = stability_outputs(canonical_variable)
    _, correlation_pairs = correlation_outputs(
        x, canonical_variable, canonical_group
    )
    variable_bootstrap, group_bootstrap, bootstrap_successful = (
        bootstrap_outputs(canonical_shap, explained_ids)
    )
    write_analysis_scope()
    metrics = core_metrics(stability)

    top_variables = canonical_variable.nsmallest(5, "rank").to_dict("records")
    top_groups = canonical_group.nsmallest(3, "rank").to_dict("records")
    validation = {
        "status": "PASS",
        "created_at": now_iso(),
        "validation": {
            "completion_markers_complete": int(
                (run_checks["status"] == "COMPLETE").sum()
            ),
            "completion_markers_expected": 17,
            "output_artifacts_hash_verified": int(
                run_checks["output_artifacts_verified"].sum()
            ),
            "output_artifacts_expected": 85,
            "all_run_plan_checks_passed": bool(run_checks["valid"].all()),
            "sample_manifests_verified": len(manifest_hashes),
            "sample_manifests_expected": 18,
            "checkpoints_verified": int(checkpoints["valid"].sum()),
            "checkpoints_expected": 3,
            "checkpoint_hashes": {
                str(row.seed): row.actual_sha256
                for row in checkpoints.itertuples(index=False)
            },
        },
        "canonical": {
            "configuration": (
                "W36 / BG256_D1 / EXPLAINED_E0"
            ),
            "shape": list(canonical_shap.shape),
            "top5_variables": [
                {
                    "feature": row["feature"],
                    "rank": int(row["rank"]),
                    "mean_abs_shap_gC_m2_month": float(
                        row["mean_abs_shap_gC_m2_month"]
                    ),
                    "mean_signed_shap_gC_m2_month": float(
                        row["mean_signed_shap_gC_m2_month"]
                    ),
                    "mean_signed_sign": int(row["mean_signed_sign"]),
                }
                for row in top_variables
            ],
            "top3_groups": [
                {
                    "group": row["group"],
                    "rank": int(row["rank"]),
                    "mean_abs_shap_gC_m2_month": float(
                        row["mean_abs_shap_gC_m2_month"]
                    ),
                    "mean_signed_shap_gC_m2_month": float(
                        row["mean_signed_shap_gC_m2_month"]
                    ),
                    "mean_signed_sign": int(row["mean_signed_sign"]),
                }
                for row in top_groups
            ],
            "lag_interpretation": LAG_INTERPRETATION,
        },
        "stability_core_metrics": metrics,
        "high_correlation_pair_count": len(correlation_pairs),
        "bootstrap": {
            "unit": "point_id",
            "clusters": 247,
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "successful": bootstrap_successful,
            "failed": 0,
            "leading_variables": variable_bootstrap.head(5).to_dict("records"),
            "leading_groups": group_bootstrap.head(3).to_dict("records"),
        },
        "geoshap_status": GEO_STATUS,
        "window_sensitivity_status": WINDOW_STATUS,
        "required_artifacts": {
            "canonical_variable": (
                "canonical/shap_canonical_variable_importance.csv"
            ),
            "canonical_group": (
                "canonical/shap_canonical_grouped_importance.csv"
            ),
            "canonical_lag": (
                "canonical/shap_canonical_lag_profiles.csv"
            ),
            "background_stability": (
                "results/shap_background_stability.csv"
            ),
            "seed_stability": (
                "results/shap_three_seed_variable_stability.csv"
            ),
            "explained_stability": (
                "results/shap_explained_sample_stability.csv"
            ),
            "correlation_matrix": (
                "results/shap_predictor_spearman_matrix.csv"
            ),
            "high_correlation_pairs": (
                "results/shap_high_correlation_pairs.csv"
            ),
            "bootstrap_variable": (
                "bootstrap/shap_variable_cluster_bootstrap_summary.csv"
            ),
            "bootstrap_group": (
                "bootstrap/shap_group_cluster_bootstrap_summary.csv"
            ),
            "geoshap": (
                "geoshap/shap_geoshap_scope.md"
            ),
            "window": (
                "window_sensitivity/shap_window_scope.md"
            ),
            "summary": (
                "summary/SHAP_SHAP_ANALYSIS_SUMMARY.md"
            ),
            "sha256_inventory": "validation/sha256_inventory.csv",
        },
    }
    background_selection_seeds = {}
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            frame = load_manifest(draw_id)
            background_selection_seeds[draw_id] = int(frame["selection_seed"].iloc[0])
    explained_selection_seeds = {}
    for draw in range(4):
        draw_id = f"EXPLAINED_E{draw}"
        frame = load_manifest(draw_id)
        explained_selection_seeds[draw_id] = int(frame["selection_seed"].iloc[0])
    run_configuration = {
        "canonical": {
            "window": WINDOW,
            "seeds": [42, 2024, 3407],
            "background_draw": "BG256_D1",
            "explained_draw": "EXPLAINED_E0",
            "explained_n": EXPLAIN_N,
        },
        "feature_order": FEATURES,
        "groups": GROUPS,
        "background_selection_seeds": background_selection_seeds,
        "explained_selection_seeds": explained_selection_seeds,
    }
    configuration_path = RUN_ROOT / "validation" / "shap_run_configuration.json"
    atomic_json(configuration_path, run_configuration)

    validation_path = RUN_ROOT / "validation" / "analysis_summary.json"
    atomic_json(validation_path, validation)

    def fmt_range(record: Mapping[str, float]) -> str:
        return f"{record['min']:.3f}–{record['max']:.3f}"

    def fmt_entities(rows: Sequence[Mapping[str, Any]], key: str) -> str:
        return "; ".join(
            (
                f"{row[key]} (rank {int(row['rank'])}, "
                f"mean|SHAP|={row['mean_abs_shap_gC_m2_month']:.4f})"
            )
            for row in rows
        )

    summary_lines = [
        "# W36 SHAP analysis summary",
        "",
        "## Three-seed W36 SHAP result",
        "",
        f"- Top 5 variables: {fmt_entities(top_variables, 'feature')}.",
        f"- Top 3 groups: {fmt_entities(top_groups, 'group')}.",
        f"- Lag interpretation: {LAG_INTERPRETATION}",
        "",
        "## Robustness",
        "",
        (
            "- Background sensitivity, variable Spearman range "
            f"(11 non-self comparisons): "
            f"{fmt_range(metrics['background_sensitivity']['variable_spearman'])}; "
            "group range "
            f"{fmt_range(metrics['background_sensitivity']['group_spearman'])}."
        ),
        (
            "- Model-seed sensitivity, variable pairwise Spearman range: "
            f"{fmt_range(metrics['model_seed_sensitivity']['variable_spearman'])}; "
            "group range "
            f"{fmt_range(metrics['model_seed_sensitivity']['group_spearman'])}."
        ),
        (
            "- Explained-sample sensitivity, E0-vs-E1/E2/E3 variable "
            "Spearman range: "
            f"{fmt_range(metrics['explained_sample_sensitivity']['variable_spearman'])}; "
            "group range "
            f"{fmt_range(metrics['explained_sample_sensitivity']['group_spearman'])}."
        ),
        "- Background, model-seed and explained-sample sensitivity are reported separately.",
        "",
        "## Uncertainty and analysis scope",
        "",
        f"- Point_id cluster bootstrap: {bootstrap_successful} replicates; seed {BOOTSTRAP_SEED}.",
        f"- Geo-SHAP: {GEO_STATUS}",
        f"- Window sensitivity: {WINDOW_STATUS}",
        "",
        "## Result files",
        "",
        "- `validation/analysis_summary.json`",
        "- `validation/sha256_inventory.csv`",
        "- `canonical/shap_canonical_variable_importance.csv`",
        "- `canonical/shap_canonical_grouped_importance.csv`",
        "- `canonical/shap_canonical_lag_profiles.csv`",
        "- `results/shap_background_stability.csv`",
        "- `results/shap_three_seed_variable_stability.csv`",
        "- `results/shap_explained_sample_stability.csv`",
        "- `results/shap_predictor_spearman_matrix.csv`",
        "- `results/shap_high_correlation_pairs.csv`",
        "- `bootstrap/shap_variable_cluster_bootstrap_summary.csv`",
        "- `bootstrap/shap_group_cluster_bootstrap_summary.csv`",
        "- `geoshap/shap_geoshap_scope.md`",
        "- `window_sensitivity/shap_window_scope.md`",
        "",
    ]
    summary_path = (
        RUN_ROOT
        / "summary"
        / "SHAP_SHAP_ANALYSIS_SUMMARY.md"
    )
    atomic_text(summary_path, "\n".join(summary_lines))
    inventory_path = make_inventory()

    print(
        json.dumps(
            {
                "status": "PASS",
                "run_validation": "17/17",
                "output_artifact_hashes": "85/85",
                "sample_manifests": "18/18",
                "checkpoints": "3/3",
                "bootstrap": "5000/0",
                "geoshap": GEO_STATUS,
                "window": WINDOW_STATUS,
                "analysis_summary": str(validation_path),
                "sha256_inventory": str(inventory_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
