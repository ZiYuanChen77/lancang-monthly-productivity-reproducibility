#!/usr/bin/env python3
"""Canonical W36 sampled-site GeoSHAP post-processing.

This program reads the fixed W36 SHAP artifacts and writes sampled-site
GeoSHAP summaries below ``geoshap_analysis``. The two phases are:

1. ``validate``: input and consistency checks.
2. ``run``: site-level aggregation, robustness, bootstrap, figures, and summary.

Validation precedes statistical aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
SHAP_ROOT = PROJECT_ROOT / "runtime" / "runs" / "shap"
GEOSHAP_ROOT = PROJECT_ROOT / "runtime" / "runs" / "geoshap"
SAMPLE_ROOT = PROJECT_ROOT / "data_documentation" / "shap_sample_keys"

VALIDATION_DIR = GEOSHAP_ROOT / "validation"
CANONICAL_DIR = GEOSHAP_ROOT / "canonical"
ROBUSTNESS_DIR = GEOSHAP_ROOT / "robustness"
VEGETATION_DIR = GEOSHAP_ROOT / "vegetation"
FIGURES_DIR = GEOSHAP_ROOT / "figures"
SUMMARY_DIR = GEOSHAP_ROOT / "summary"

PROJECT_ID = "lancang_monthly_productivity"
ANALYSIS_NAME = "GeoSHAP"
WINDOW = 36
EXPLAINED_N = 1000
INPUT_DIM = 19
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20240724

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

GROUPS = {
    "Vegetation productivity": ["NDVI"],
    "Thermal": ["LST_Day_1km", "temperature_2m", "VPD"],
    "Hydrologic": ["Precipitation_mm", "volumetric_soil_water_layer_1"],
    "Energy": ["surface_solar_radiation_downwards_sum"],
    "Terrain": ["elevation", "slope", "Aspect_Sin", "Aspect_Cos"],
    "Seasonal phase": ["Target_Month_Sin", "Target_Month_Cos"],
    "Vegetation class": [
        "Veg_Coniferous",
        "Veg_Shrub",
        "Veg_Meadow_497",
        "Veg_Meadow_499",
        "Veg_Meadow_504",
        "Veg_Sparse",
    ],
}

ENV_GROUPS = [
    "Energy",
    "Vegetation productivity",
    "Thermal",
    "Hydrologic",
    "Terrain",
]
CORE_GROUPS = ["Energy", "Vegetation productivity"]
VEGETATION_CLASSES = [
    "Veg_Coniferous",
    "Veg_Shrub",
    "Veg_Meadow_497",
    "Veg_Meadow_499",
    "Veg_Meadow_504",
    "Veg_Sparse",
]
NONSEASONAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in {"Target_Month_Sin", "Target_Month_Cos"}
]

REQUIRED_MANIFEST_COLUMNS = [
    "draw_id",
    "selection_seed",
    "row_order",
    "sample_id",
    "point_id",
    "Lon",
    "Lat",
    "Year",
    "Month",
    "split",
    "target_original_row_index",
    "veg_class",
]

CONTROL_FILES = [
    "validation/analysis_summary.json",
    "validation/sha256_inventory.csv",
    "validation/shap_run_configuration.json",
]

REQUESTED_OUTPUTS = [
    "validation/geoshap_input_inventory.csv",
    "validation/geoshap_provenance.md",
    "canonical/geoshap_canonical_site_variable_abs_shap.csv",
    "canonical/geoshap_canonical_site_group_abs_shap.csv",
    "canonical/geoshap_canonical_dominant_environmental_group.csv",
    "canonical/geoshap_dominant_nonseasonal_predictor.csv",
    "robustness/geoshap_background_spatial_stability.csv",
    "robustness/geoshap_background_spatial_stability_summary.csv",
    "robustness/geoshap_three_seed_spatial_stability.csv",
    "robustness/geoshap_three_seed_spatial_stability_summary.csv",
    "robustness/geoshap_explained_spatial_stability.csv",
    "robustness/geoshap_explained_spatial_stability_summary.csv",
    "vegetation/geoshap_vegetation_group_summary.csv",
    "vegetation/geoshap_vegetation_group_bootstrap_ci.csv",
    "vegetation/geoshap_vegetation_dominant_group_counts.csv",
    "vegetation/geoshap_vegetation_dominant_group_bootstrap_ci.csv",
    "figures/geoshap_figure_A_canonical_dominant_environmental_group_map.png",
    "figures/geoshap_figure_B_canonical_NDVI_mean_abs_shap_map.png",
    "figures/geoshap_figure_C_canonical_solar_radiation_mean_abs_shap_map.png",
    "figures/geoshap_figure_D_vegetation_environmental_group_attribution_heatmap.png",
    "figures/geoshap_figure_E_spatial_spearman_robustness_summary.png",
    "summary/w36_geoshap_validation.json",
    "summary/W36_GEOSHAP_ANALYSIS_SUMMARY.md",
]


class ValidationError(RuntimeError):
    """Raised for any fail-closed scientific or consistency mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def posix_rel(path: Path) -> str:
    return path.relative_to(SHAP_ROOT).as_posix()


def safe_output(relative_path: str) -> Path:
    candidate = (GEOSHAP_ROOT / relative_path).resolve()
    root = GEOSHAP_ROOT.resolve()
    require(
        os.path.commonpath([str(candidate), str(root)]) == str(root),
        f"Output path escapes geoshap_analysis: {candidate}",
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def atomic_write_text(path: Path, text: str) -> None:
    require(
        os.path.commonpath([str(path.resolve()), str(GEOSHAP_ROOT.resolve())])
        == str(GEOSHAP_ROOT.resolve()),
        f"Refusing write outside geoshap_analysis: {path}",
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(relative_path: str, payload: Mapping[str, Any]) -> Path:
    path = safe_output(relative_path)
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return path


def write_csv(relative_path: str, frame: pd.DataFrame) -> Path:
    path = safe_output(relative_path)
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    )
    atomic_write_text(path, buffer.getvalue())
    return path


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(path: Path, *, require_unique_sample_id: bool = True) -> pd.DataFrame:
    require(path.is_file(), f"Missing fixed manifest: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "draw_id": "string",
            "selection_seed": "int64",
            "row_order": "int64",
            "sample_id": "int64",
            "point_id": "string",
            "Lon": "float64",
            "Lat": "float64",
            "Year": "int64",
            "Month": "int64",
            "split": "string",
            "target_original_row_index": "int64",
            "veg_class": "string",
        },
        keep_default_na=False,
    )
    require(
        list(frame.columns) == REQUIRED_MANIFEST_COLUMNS,
        f"Manifest column mismatch: {path}",
    )
    require(not frame.isna().any().any(), f"Missing values in manifest: {path}")
    if require_unique_sample_id:
        require(frame["sample_id"].is_unique, f"Duplicate sample_id in manifest: {path}")
    else:
        require(
            not frame.duplicated(["draw_id", "sample_id"]).any(),
            f"Duplicate (draw_id, sample_id) in consolidated manifest: {path}",
        )
    if require_unique_sample_id:
        require(
            frame["row_order"].tolist() == list(range(len(frame))),
            f"row_order mismatch in manifest: {path}",
        )
    else:
        for draw_id, subset in frame.groupby("draw_id", sort=False, observed=True):
            require(
                subset["row_order"].tolist() == list(range(len(subset))),
                f"Within-draw row_order mismatch in consolidated manifest: {path}/{draw_id}",
            )
    require(
        np.isfinite(frame[["Lon", "Lat"]].to_numpy()).all(),
        f"Non-finite coordinates in manifest: {path}",
    )
    require(
        frame["Month"].between(1, 12).all(),
        f"Invalid month in manifest: {path}",
    )
    require(
        set(frame["veg_class"].astype(str)).issubset(set(VEGETATION_CLASSES)),
        f"Unexpected vegetation class in manifest: {path}",
    )
    return frame


def manifest_path(draw_id: str) -> Path:
    if draw_id.startswith("BG"):
        return SAMPLE_ROOT / "sample_keys" / "background" / f"shap_{draw_id}.csv"
    require(
        draw_id.startswith("EXPLAINED_E"),
        f"Unrecognized manifest draw id: {draw_id}",
    )
    file_id = draw_id.replace("EXPLAINED_", "explained_", 1)
    return SAMPLE_ROOT / "sample_keys" / f"shap_{file_id}.csv"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model_seed: int | None
    background_draw: str
    background_n: int
    explained_draw: str


def run_plan() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            specs.append(
                RunSpec(
                    run_id=f"background_{draw_id}",
                    model_seed=42,
                    background_draw=draw_id,
                    background_n=size,
                    explained_draw="EXPLAINED_E0",
                )
            )
    specs.extend(
        [
            RunSpec("seed_2024", 2024, "BG256_D1", 256, "EXPLAINED_E0"),
            RunSpec("seed_3407", 3407, "BG256_D1", 256, "EXPLAINED_E0"),
            RunSpec("explained_E1", 42, "BG256_D1", 256, "EXPLAINED_E1"),
            RunSpec("explained_E2", 42, "BG256_D1", 256, "EXPLAINED_E2"),
            RunSpec("explained_E3", 42, "BG256_D1", 256, "EXPLAINED_E3"),
        ]
    )
    require(len(specs) == 17, "Internal run plan is not exactly 17 runs")
    return specs


def run_paths(spec: RunSpec) -> dict[str, Path]:
    base = SHAP_ROOT / "shap_runs"
    return {
        "shap_npz": base / f"shap_{spec.run_id}.npz",
        "variable_csv": base / f"shap_{spec.run_id}_variable.csv",
        "grouped_csv": base / f"shap_{spec.run_id}_grouped.csv",
        "signed_csv": base / f"shap_{spec.run_id}_signed.csv",
        "execution_json": base / f"shap_{spec.run_id}_execution.json",
        "completion_marker": (
            base / "completed" / f"shap_{spec.run_id}.complete.json"
        ),
    }


def original_tree_digest() -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    paths = sorted(
        (
            path
            for path in SHAP_ROOT.rglob("*")
            if path.is_file() and GEOSHAP_ROOT not in path.parents
        ),
        key=lambda path: path.relative_to(SHAP_ROOT).as_posix(),
    )
    for path in paths:
        rel = path.relative_to(SHAP_ROOT).as_posix()
        size = path.stat().st_size
        item_hash = sha256_file(path)
        digest.update(f"{rel}\0{size}\0{item_hash}\n".encode("utf-8"))
        count += 1
        total_bytes += size
    return {
        "algorithm": "sha256(path\\0bytes\\0file_sha256\\n)",
        "file_count": count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "excluded_subtree": "geoshap_analysis",
    }


def check_code_boundary() -> dict[str, bool]:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    prohibited_patterns = {
        "no_shap_import": r"^\s*(?:from\s+shap\b|import\s+shap\b)",
        "no_explainer_constructor": r"GradientExplainer\s*\(",
        "no_torch_import": r"^\s*(?:from\s+torch\b|import\s+torch\b)",
        "no_model_fit_call": r"\.fit\s*\(",
    }
    checks = {
        label: re.search(pattern, source, flags=re.MULTILINE) is None
        for label, pattern in prohibited_patterns.items()
    }
    return checks


def validate_input_files() -> dict[str, Any]:
    """Validate every validated input without any attribution aggregation."""

    tree_before = original_tree_digest()
    for relative_path in CONTROL_FILES:
        require((SHAP_ROOT / relative_path).is_file(), f"Missing control: {relative_path}")

    validation = load_json(SHAP_ROOT / "validation" / "analysis_summary.json")
    configuration = load_json(SHAP_ROOT / "validation" / "shap_run_configuration.json")

    require(validation.get("status") == "PASS", "SHAP validation is not PASS")
    integrity = validation.get("validation", {})
    require(integrity.get("completion_markers_complete") == 17, "Validation marker count mismatch")
    require(integrity.get("completion_markers_expected") == 17, "Validation expected marker count mismatch")
    require(integrity.get("output_artifacts_hash_verified") == 85, "Validation run-output hash count mismatch")
    require(integrity.get("output_artifacts_expected") == 85, "Validation expected run-output count mismatch")
    require(integrity.get("sample_manifests_verified") == 18, "Validation sample-manifest count mismatch")
    require(integrity.get("sample_manifests_expected") == 18, "Validation expected sample-manifest count mismatch")
    require(integrity.get("all_run_plan_checks_passed") is True, "SHAP run plan was not validated")

    canonical_cfg = configuration.get("canonical", {})
    require(canonical_cfg.get("window") == WINDOW, "Canonical window is not W36")
    require(canonical_cfg.get("seeds") == [42, 2024, 3407], "Three-seed configuration mismatch")
    require(canonical_cfg.get("background_draw") == "BG256_D1", "Canonical background mismatch")
    require(canonical_cfg.get("explained_draw") == "EXPLAINED_E0", "Canonical explained draw mismatch")
    require(canonical_cfg.get("explained_n") == EXPLAINED_N, "Canonical explained n mismatch")
    require(configuration.get("feature_order") == FEATURES, "Fixed feature order mismatch")
    require(configuration.get("groups") == GROUPS, "Fixed group definition mismatch")
    sha_inventory_path = SHAP_ROOT / "validation" / "sha256_inventory.csv"
    sha_inventory = pd.read_csv(sha_inventory_path, dtype=str, keep_default_na=False)
    require(
        list(sha_inventory.columns) == ["scope", "path", "bytes", "sha256"],
        "SHAP sha256 inventory schema mismatch",
    )
    require(sha_inventory["path"].is_unique, "Duplicate path in SHAP sha256 inventory")
    inventory_records = sha_inventory.set_index("path").to_dict(orient="index")

    expected_manifest_paths = {
        f"sample_keys/background/shap_BG{size}_D{draw}.csv"
        for size in (64, 128, 256, 512)
        for draw in (1, 2, 3)
    }
    expected_manifest_paths.update(
        {f"sample_keys/shap_explained_E{draw}.csv" for draw in range(4)}
    )
    expected_manifest_paths.update(
        {
            "sample_keys/shap_background_manifests.csv",
            "sample_keys/shap_sampling_balance_summary.csv",
        }
    )
    manifest_records = {}
    for relative_path in expected_manifest_paths:
        path = SAMPLE_ROOT / relative_path
        require(path.is_file(), f"Missing sample manifest: {relative_path}")
        manifest_records[relative_path] = {
            "bytes": str(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    hash_cache: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    def verify_main_inventory(path: Path) -> None:
        relative = posix_rel(path)
        require(relative in inventory_records, f"Input missing from SHAP sha256 inventory: {relative}")
        record = inventory_records[relative]
        require(int(record["bytes"]) == path.stat().st_size, f"Byte-size mismatch: {relative}")
        require(record["sha256"] == cached_hash(path), f"SHA256 mismatch: {relative}")

    for relative_path in (
        "validation/analysis_summary.json",
        "validation/shap_run_configuration.json",
    ):
        verify_main_inventory(SHAP_ROOT / relative_path)

    for relative_path, record in manifest_records.items():
        path = SAMPLE_ROOT / relative_path
        require(path.is_file(), f"Missing sample-manifest file: {relative_path}")
        require(path.stat().st_size == int(record["bytes"]), f"Sample-manifest byte mismatch: {relative_path}")
        require(cached_hash(path) == record["sha256"], f"Sample-manifest hash mismatch: {relative_path}")

    background_frames: list[pd.DataFrame] = []
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            frame = read_manifest(manifest_path(draw_id))
            require(len(frame) == size, f"Background manifest size mismatch: {draw_id}")
            require(frame["draw_id"].eq(draw_id).all(), f"Background draw identity mismatch: {draw_id}")
            expected_seed = configuration["background_selection_seeds"][draw_id]
            require(frame["selection_seed"].eq(expected_seed).all(), f"Background selection seed mismatch: {draw_id}")
            require(frame["split"].eq("train").all(), f"Non-training row in background manifest: {draw_id}")
            background_frames.append(frame)

    combined_background = read_manifest(
        SAMPLE_ROOT / "sample_keys" / "shap_background_manifests.csv",
        require_unique_sample_id=False,
    )
    expected_combined = pd.concat(background_frames, ignore_index=True)
    pd.testing.assert_frame_equal(
        combined_background.reset_index(drop=True),
        expected_combined.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )

    explained_frames: dict[str, pd.DataFrame] = {}
    for draw in range(4):
        draw_id = f"EXPLAINED_E{draw}"
        frame = read_manifest(manifest_path(draw_id))
        require(len(frame) == EXPLAINED_N, f"Explained manifest size mismatch: {draw_id}")
        require(frame["draw_id"].eq(draw_id).all(), f"Explained draw identity mismatch: {draw_id}")
        expected_seed = configuration["explained_selection_seeds"][draw_id]
        require(frame["selection_seed"].eq(expected_seed).all(), f"Explained selection seed mismatch: {draw_id}")
        require(frame["split"].eq("test").all(), f"Non-test row in explained manifest: {draw_id}")
        explained_frames[draw_id] = frame

    metadata_union = pd.concat(
        [
            frame[["point_id", "Lon", "Lat", "veg_class"]]
            for frame in explained_frames.values()
        ],
        ignore_index=True,
    )
    for column in ("Lon", "Lat", "veg_class"):
        require(
            metadata_union.groupby("point_id", observed=True)[column].nunique().le(1).all(),
            f"Point metadata conflict across explained manifests: {column}",
        )

    marker_dir = SHAP_ROOT / "shap_runs" / "completed"
    expected_marker_names = {
        f"shap_{spec.run_id}.complete.json" for spec in run_plan()
    }
    actual_marker_names = {path.name for path in marker_dir.glob("*.complete.json")}
    require(actual_marker_names == expected_marker_names, "Completion marker set is not the validated 17-run set")

    checkpoint_hashes = {
        int(seed): value
        for seed, value in validation["validation"]["checkpoint_hashes"].items()
    }
    required_npz_keys = {
        "shap_values_gC_m2_month",
        "background_sample_ids",
        "explained_sample_ids",
        "feature_names",
        "window",
        "model_seed",
    }
    run_checks: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    output_hash_verified = 0

    def append_inventory_row(
        path: Path,
        input_type: str,
        *,
        run_id: str = "",
        model_seed: str | int = "",
        background_draw: str = "",
        explained_draw: str = "",
        shape: str = "",
        window: str | int = "",
        n_explained: str | int = "",
        feature_order_match: str = "NOT_APPLICABLE",
        sample_id_match: str = "NOT_APPLICABLE",
        background_sample_id_match: str = "NOT_APPLICABLE",
        fixed_manifest_hash_match: str = "NOT_APPLICABLE",
    ) -> None:
        if path.resolve().is_relative_to(SAMPLE_ROOT.resolve()):
            relative = path.resolve().relative_to(SAMPLE_ROOT.resolve()).as_posix()
        else:
            relative = posix_rel(path)
        inventory_match = "NOT_LISTED_SELF"
        if relative in inventory_records:
            record = inventory_records[relative]
            inventory_match = "YES" if (
                int(record["bytes"]) == path.stat().st_size
                and record["sha256"] == cached_hash(path)
            ) else "NO"
        inventory_rows.append(
            {
                "file": path.name,
                "relative_path": relative,
                "input_type": input_type,
                "bytes": path.stat().st_size,
                "sha256": cached_hash(path),
                "run_id": run_id,
                "model_seed": model_seed,
                "background_draw": background_draw,
                "explained_draw": explained_draw,
                "shape": shape,
                "window": window,
                "n_explained": n_explained,
                "feature_order_match": feature_order_match,
                "sample_id_match": sample_id_match,
                "background_sample_id_match": background_sample_id_match,
                "fixed_manifest_hash_match": fixed_manifest_hash_match,
                "sha256_inventory_match": inventory_match,
                "validation_status": "PASS",
            }
        )

    for relative_path in CONTROL_FILES:
        append_inventory_row(SHAP_ROOT / relative_path, "control")

    referencing_runs: dict[str, list[RunSpec]] = {path: [] for path in expected_manifest_paths}
    for spec in run_plan():
        referencing_runs[
            f"sample_keys/background/shap_{spec.background_draw}.csv"
        ].append(spec)
        explained_file_id = spec.explained_draw.replace("EXPLAINED_", "explained_", 1)
        referencing_runs[f"sample_keys/shap_{explained_file_id}.csv"].append(spec)

    for relative_path in sorted(expected_manifest_paths):
        refs = referencing_runs.get(relative_path, [])
        append_inventory_row(
            SAMPLE_ROOT / relative_path,
            "fixed_manifest",
            run_id=";".join(spec.run_id for spec in refs),
            model_seed=";".join(str(seed) for seed in sorted({spec.model_seed for spec in refs})),
            background_draw=";".join(sorted({spec.background_draw for spec in refs})),
            explained_draw=";".join(sorted({spec.explained_draw for spec in refs})),
            n_explained=(EXPLAINED_N if "explained_E" in relative_path else ""),
            sample_id_match=("YES" if refs else "NOT_APPLICABLE"),
            background_sample_id_match=(
                "YES" if refs and "/background/" in relative_path else "NOT_APPLICABLE"
            ),
            fixed_manifest_hash_match="YES",
        )

    for spec in run_plan():
        paths = run_paths(spec)
        for path in paths.values():
            require(path.is_file(), f"Missing validated run artifact: {path}")

        marker = load_json(paths["completion_marker"])
        execution = load_json(paths["execution_json"])
        protocol = marker.get("protocol", {})
        background = read_manifest(manifest_path(spec.background_draw))
        explained = explained_frames[spec.explained_draw]
        background_ids = background["sample_id"].to_numpy(dtype=np.int64)
        explained_ids = explained["sample_id"].to_numpy(dtype=np.int64)
        background_rel = f"sample_keys/background/shap_{spec.background_draw}.csv"
        explained_file_id = spec.explained_draw.replace("EXPLAINED_", "explained_", 1)
        explained_rel = f"sample_keys/shap_{explained_file_id}.csv"

        checks = {
            "status_complete": marker.get("status") == "COMPLETE",
            "run_id_match": marker.get("run_id") == spec.run_id,
            "shape_match": marker.get("shape") == [EXPLAINED_N, WINDOW, INPUT_DIM],
            "window_match": protocol.get("window") == WINDOW,
            "seed_match": protocol.get("model_seed") == spec.model_seed,
            "background_draw_match": protocol.get("background_draw") == spec.background_draw,
            "explained_draw_match": protocol.get("explained_draw") == spec.explained_draw,
            "checkpoint_hash_match": protocol.get("checkpoint_sha256") == checkpoint_hashes[spec.model_seed],
            "background_manifest_hash_match": protocol.get("background_manifest_sha256") == manifest_records[background_rel]["sha256"],
            "explained_manifest_hash_match": protocol.get("explained_manifest_sha256") == manifest_records[explained_rel]["sha256"],
            "feature_order_match": protocol.get("feature_order") == FEATURES,
            "group_definition_match": protocol.get("groups") == GROUPS,
            "execution_complete": execution.get("status") == "COMPLETE",
            "execution_identity_match": (
                execution.get("run_id") == spec.run_id
                and execution.get("model_seed") == spec.model_seed
                and execution.get("background_draw") == spec.background_draw
                and execution.get("background_n") == spec.background_n
                and execution.get("explained_draw") == spec.explained_draw
                and execution.get("explained_n") == EXPLAINED_N
            ),
            "execution_settings_match": (
                execution.get("device") == "cpu"
                and execution.get("explainer") == "GradientExplainer"
                and execution.get("nsamples") == 50
                and execution.get("rseed") == BOOTSTRAP_SEED
                and execution.get("batch_size") == 16
            ),
        }

        output_artifacts = marker.get("output_artifacts", {})
        expected_output_keys = {
            "shap_npz",
            "variable_csv",
            "grouped_csv",
            "signed_csv",
            "execution_json",
        }
        checks["five_artifacts_recorded"] = set(output_artifacts) == expected_output_keys
        for key in sorted(expected_output_keys):
            record = output_artifacts.get(key, {})
            recorded_path = SHAP_ROOT / record.get("relative_path", "__MISSING__")
            require(recorded_path == paths[key], f"Run-output path mismatch: {spec.run_id}/{key}")
            require(recorded_path.is_file(), f"Missing run-output: {recorded_path}")
            require(recorded_path.stat().st_size == record.get("bytes"), f"Run-output byte mismatch: {recorded_path}")
            require(cached_hash(recorded_path) == record.get("sha256"), f"Run-output marker hash mismatch: {recorded_path}")
            verify_main_inventory(recorded_path)
            output_hash_verified += 1

        verify_main_inventory(paths["completion_marker"])

        with np.load(paths["shap_npz"], allow_pickle=False) as data:
            checks["npz_keys_match"] = set(data.files) == required_npz_keys
            require(checks["npz_keys_match"], f"NPZ key mismatch: {spec.run_id}")
            values = data["shap_values_gC_m2_month"]
            checks["npz_shape_match"] = values.shape == (EXPLAINED_N, WINDOW, INPUT_DIM)
            checks["npz_finite"] = bool(np.isfinite(values).all())
            checks["npz_feature_order_match"] = np.array_equal(data["feature_names"], np.asarray(FEATURES))
            checks["npz_window_match"] = np.array_equal(data["window"], np.asarray([WINDOW], dtype=np.int16))
            checks["npz_seed_match"] = np.array_equal(data["model_seed"], np.asarray([spec.model_seed], dtype=np.int32))
            checks["npz_background_keys_match"] = np.array_equal(data["background_sample_ids"], background_ids)
            checks["npz_explained_keys_match"] = np.array_equal(data["explained_sample_ids"], explained_ids)
            shape_text = "x".join(str(value) for value in values.shape)

        failed_checks = [name for name, passed in checks.items() if not passed]
        require(not failed_checks, f"Run identity/consistency mismatch for {spec.run_id}: {failed_checks}")
        run_checks.append(
            {
                "run_id": spec.run_id,
                "model_seed": spec.model_seed,
                "background_draw": spec.background_draw,
                "explained_draw": spec.explained_draw,
                "shape": shape_text,
                "window": WINDOW,
                "n_explained": EXPLAINED_N,
                "feature_order_match": True,
                "background_sample_id_match": True,
                "sample_id_match": True,
                "all_checks_pass": True,
            }
        )
        append_inventory_row(
            paths["shap_npz"],
            "source_shap_npz",
            run_id=spec.run_id,
            model_seed=spec.model_seed,
            background_draw=spec.background_draw,
            explained_draw=spec.explained_draw,
            shape=shape_text,
            window=WINDOW,
            n_explained=EXPLAINED_N,
            feature_order_match="YES",
            sample_id_match="YES",
            background_sample_id_match="YES",
        )
        append_inventory_row(
            paths["execution_json"],
            "execution_json",
            run_id=spec.run_id,
            model_seed=spec.model_seed,
            background_draw=spec.background_draw,
            explained_draw=spec.explained_draw,
            window=WINDOW,
            n_explained=EXPLAINED_N,
        )
        append_inventory_row(
            paths["completion_marker"],
            "completion_marker",
            run_id=spec.run_id,
            model_seed=spec.model_seed,
            background_draw=spec.background_draw,
            explained_draw=spec.explained_draw,
            shape=shape_text,
            window=WINDOW,
            n_explained=EXPLAINED_N,
            feature_order_match="YES",
        )

    require(output_hash_verified == 85, "Did not verify exactly 85 run-output hashes")
    require(len(run_checks) == 17, "Did not validate exactly 17 configured runs")

    canonical_ids = explained_frames["EXPLAINED_E0"]["sample_id"].to_numpy(dtype=np.int64)
    for spec in run_plan():
        if spec.explained_draw == "EXPLAINED_E0":
            with np.load(run_paths(spec)["shap_npz"], allow_pickle=False) as data:
                require(
                    np.array_equal(data["explained_sample_ids"], canonical_ids),
                    f"E0 sample identity differs for {spec.run_id}",
                )

    boundary_checks = check_code_boundary()
    require(all(boundary_checks.values()), f"Post-processing code boundary check failed: {boundary_checks}")
    tree_after = original_tree_digest()
    require(tree_after == tree_before, "Input SHAP tree changed during validation")

    input_inventory = pd.DataFrame(inventory_rows).sort_values(
        ["input_type", "relative_path"], kind="stable"
    )
    require(
        input_inventory["validation_status"].eq("PASS").all(),
        "Input inventory contains a non-PASS row",
    )
    inventory_output_path = write_csv(
        "validation/geoshap_input_inventory.csv", input_inventory
    )

    historic_statement = (
        SHAP_ROOT / "geoshap" / "shap_geoshap_scope.md"
    )
    require(historic_statement.is_file(), "GeoSHAP scope record is missing")

    input_validation = {
        "project_id": PROJECT_ID,
        "work_package": ANALYSIS_NAME,
        "phase": "validate",
        "status": "PASS",
        "validated_at": now_iso(),
        "statistics_computed": False,
        "configured_runs_validated": len(run_checks),
        "output_artifact_hashes_verified": output_hash_verified,
        "sample_manifests_validated": len(manifest_records),
        "input_inventory_rows": len(input_inventory),
        "input_inventory_relative_path": posix_rel(inventory_output_path),
        "input_inventory_sha256": sha256_file(inventory_output_path),
        "canonical_e0_site_count": int(explained_frames["EXPLAINED_E0"]["point_id"].nunique()),
        "explained_manifest_site_counts": {
            draw_id: int(frame["point_id"].nunique())
            for draw_id, frame in explained_frames.items()
        },
        "input_shap_tree_before": tree_before,
        "input_shap_tree_after_validation": tree_after,
        "historic_geoshap_statement_sha256": sha256_file(historic_statement),
        "code_boundary_checks": boundary_checks,
        "run_checks": run_checks,
    }
    input_validation_path = write_json(
        "validation/geoshap_input_validation.json", input_validation
    )
    return {
        "status": "PASS",
        "input_inventory": str(inventory_output_path),
        "validation_input_validation": str(input_validation_path),
        "input_inventory_rows": len(input_inventory),
        "canonical_site_count": input_validation["canonical_e0_site_count"],
    }


def feature_output_column(feature: str) -> str:
    return f"{feature}_mean_abs_shap_gC_m2_month"


def safe_group_name(group: str) -> str:
    return group.replace(" ", "_")


def group_output_column(group: str) -> str:
    return f"{safe_group_name(group)}_abs_shap_gC_m2_month"


def ratio_output_column(group: str) -> str:
    return f"{safe_group_name(group)}_ratio"


@dataclass
class AggregatedRun:
    spec: RunSpec
    site_variable: pd.DataFrame
    site_group: pd.DataFrame
    site_dominant: pd.DataFrame


def aggregate_run(spec: RunSpec) -> AggregatedRun:
    """Perform the fixed sample -> site -> group absolute-attribution workflow."""

    explained = read_manifest(manifest_path(spec.explained_draw))
    npz_path = run_paths(spec)["shap_npz"]
    with np.load(npz_path, allow_pickle=False) as data:
        sample_ids = np.asarray(data["explained_sample_ids"], dtype=np.int64)
        values = np.asarray(data["shap_values_gC_m2_month"], dtype=np.float64)
        feature_names = data["feature_names"].astype(str).tolist()
        window = int(np.asarray(data["window"]).reshape(-1)[0])
        model_seed = int(np.asarray(data["model_seed"]).reshape(-1)[0])

    require(values.shape == (EXPLAINED_N, WINDOW, INPUT_DIM), f"Shape changed after validation: {spec.run_id}")
    require(feature_names == FEATURES, f"Feature order changed after validation: {spec.run_id}")
    require(window == WINDOW, f"Window changed after validation: {spec.run_id}")
    require(model_seed == spec.model_seed, f"Model seed changed after validation: {spec.run_id}")
    require(np.isfinite(values).all(), f"Non-finite SHAP values: {spec.run_id}")

    sample_abs = np.abs(values).sum(axis=1, dtype=np.float64)
    sample_attribution = pd.DataFrame(sample_abs, columns=FEATURES)
    sample_attribution.insert(0, "sample_id", sample_ids)
    joined = sample_attribution.merge(
        explained[
            ["sample_id", "point_id", "Lon", "Lat", "Year", "Month", "veg_class"]
        ],
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    require(len(joined) == EXPLAINED_N, f"Joined sample count mismatch: {spec.run_id}")
    require(not joined[["point_id", "Lon", "Lat", "Year", "Month", "veg_class"]].isna().any().any(), f"Unmatched sample id: {spec.run_id}")
    require(joined["sample_id"].tolist() == sample_ids.tolist(), f"Join order changed sample identity: {spec.run_id}")

    for column in ("Lon", "Lat", "veg_class"):
        require(
            joined.groupby("point_id", observed=True)[column].nunique().le(1).all(),
            f"Within-point metadata conflict for {column}: {spec.run_id}",
        )

    metadata = (
        joined.groupby("point_id", sort=True, observed=True)
        .agg(
            Lon=("Lon", "first"),
            Lat=("Lat", "first"),
            veg_class=("veg_class", "first"),
            n_explained_samples=("sample_id", "size"),
        )
        .reset_index()
    )
    feature_means = (
        joined.groupby("point_id", sort=True, observed=True)[FEATURES]
        .mean()
        .reset_index()
    )
    site_variable = metadata.merge(
        feature_means, on="point_id", how="inner", validate="one_to_one"
    )
    require(int(site_variable["n_explained_samples"].sum()) == EXPLAINED_N, f"Site sample counts do not sum to 1000: {spec.run_id}")
    require(np.isfinite(site_variable[FEATURES].to_numpy()).all(), f"Non-finite site attribution: {spec.run_id}")

    return group_site_attributions(spec, site_variable)


def group_site_attributions(spec: RunSpec, site_variable: pd.DataFrame) -> AggregatedRun:
    """Group site-level contributions and derive environmental shares."""
    site_group = site_variable[
        ["point_id", "Lon", "Lat", "veg_class", "n_explained_samples"]
    ].copy()
    for group, members in GROUPS.items():
        site_group[group] = site_variable[members].sum(axis=1)
    require(np.isfinite(site_group[list(GROUPS)].to_numpy()).all(), f"Non-finite group attribution: {spec.run_id}")

    denominator = site_group[ENV_GROUPS].sum(axis=1)
    require((denominator > 0).all(), f"Zero environmental attribution denominator: {spec.run_id}")
    environmental_values = site_group[ENV_GROUPS].to_numpy(dtype=np.float64)
    ties = np.isclose(
        environmental_values,
        environmental_values.max(axis=1, keepdims=True),
        rtol=0.0,
        atol=0.0,
    ).sum(axis=1)
    require((ties == 1).all(), f"Tied dominant environmental group: {spec.run_id}")
    dominant_indices = environmental_values.argmax(axis=1)
    site_dominant = site_group[
        ["point_id", "Lon", "Lat", "veg_class", "n_explained_samples"]
    ].copy()
    site_dominant["dominant_environmental_group"] = [
        ENV_GROUPS[index] for index in dominant_indices
    ]
    site_dominant["dominant_group_abs_shap"] = environmental_values[
        np.arange(len(site_group)), dominant_indices
    ]
    for group in ENV_GROUPS:
        site_dominant[f"{group}_ratio"] = site_group[group] / denominator
    site_dominant["dominant_group_ratio"] = site_dominant[
        [f"{group}_ratio" for group in ENV_GROUPS]
    ].max(axis=1)
    return AggregatedRun(spec, site_variable, site_group, site_dominant)


def aggregate_three_seeds(aggregated: Mapping[str, AggregatedRun]) -> AggregatedRun:
    """Average aligned per-site absolute contributions across three seeds."""
    from three_seed_aggregation import SEED_RUNS

    runs = [aggregated[run_id] for run_id in SEED_RUNS.values()]
    metadata = ["point_id", "Lon", "Lat", "veg_class", "n_explained_samples"]
    frames = [run.site_variable.sort_values("point_id").reset_index(drop=True) for run in runs]
    for frame in frames[1:]:
        require(frames[0][metadata].equals(frame[metadata]), "Three-seed site metadata mismatch")
    result = frames[0].copy()
    result[FEATURES] = np.stack([frame[FEATURES].to_numpy(float) for frame in frames]).mean(axis=0)
    spec = RunSpec("three_seed_aggregate", None, "BG256_D1", 256, "EXPLAINED_E0")
    return group_site_attributions(spec, result)


def spatial_class(rho: float) -> str:
    if rho >= 0.80:
        return "STRONG"
    if rho >= 0.60:
        return "MODERATE"
    return "WEAK"


def dominant_class(agreement: float) -> str:
    if agreement >= 0.70:
        return "STRONG"
    if agreement >= 0.50:
        return "MODERATE"
    return "WEAK"


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    left = np.asarray(a, dtype=str)
    right = np.asarray(b, dtype=str)
    require(left.shape == right.shape and left.size > 0, "Invalid arrays for Cohen kappa")
    observed = float(np.mean(left == right))
    expected = 0.0
    for category in ENV_GROUPS:
        expected += float(np.mean(left == category) * np.mean(right == category))
    denominator = 1.0 - expected
    require(denominator > np.finfo(float).eps, "Cohen kappa undefined because expected agreement is 1")
    return (observed - expected) / denominator


def compare_runs(
    family: str,
    comparison_id: str,
    reference: AggregatedRun,
    comparison: AggregatedRun,
) -> list[dict[str, Any]]:
    left = reference.site_group.set_index("point_id")
    right = comparison.site_group.set_index("point_id")
    matched_ids = sorted(set(left.index).intersection(right.index))
    require(matched_ids, f"No matched sites for {comparison_id}")
    for column in ("Lon", "Lat", "veg_class"):
        a = left.loc[matched_ids, column].to_numpy()
        b = right.loc[matched_ids, column].to_numpy()
        if column in ("Lon", "Lat"):
            require(np.array_equal(a.astype(float), b.astype(float)), f"Coordinate mismatch for matched sites: {comparison_id}/{column}")
        else:
            require(np.array_equal(a.astype(str), b.astype(str)), f"Vegetation-class mismatch for matched sites: {comparison_id}")

    left_dom = reference.site_dominant.set_index("point_id").loc[
        matched_ids, "dominant_environmental_group"
    ].astype(str)
    right_dom = comparison.site_dominant.set_index("point_id").loc[
        matched_ids, "dominant_environmental_group"
    ].astype(str)
    agreement = float(np.mean(left_dom.to_numpy() == right_dom.to_numpy()))
    kappa = float(cohen_kappa(left_dom, right_dom))

    rows: list[dict[str, Any]] = []
    for group in ENV_GROUPS:
        x = left.loc[matched_ids, group].to_numpy(dtype=np.float64)
        y = right.loc[matched_ids, group].to_numpy(dtype=np.float64)
        rho = float(spearmanr(x, y).statistic)
        tau = float(kendalltau(x, y, variant="b").statistic)
        require(math.isfinite(rho) and math.isfinite(tau), f"Non-finite correlation: {comparison_id}/{group}")
        rows.append(
            {
                "family": family,
                "comparison_id": comparison_id,
                "reference_run_id": reference.spec.run_id,
                "comparison_run_id": comparison.spec.run_id,
                "environmental_group": group,
                "matched_site_n": len(matched_ids),
                "spearman_rho": rho,
                "kendall_tau": tau,
                "spatial_attribution_class": spatial_class(rho),
                "dominant_environmental_group_agreement": agreement,
                "cohen_kappa": kappa,
            }
        )
    return rows


def summarize_family(detail: pd.DataFrame) -> pd.DataFrame:
    require(set(detail["environmental_group"]) == set(ENV_GROUPS), "Missing group in robustness detail")
    unique_comparisons = detail[
        [
            "comparison_id",
            "matched_site_n",
            "dominant_environmental_group_agreement",
            "cohen_kappa",
        ]
    ].drop_duplicates()
    agreement_values = unique_comparisons[
        "dominant_environmental_group_agreement"
    ].to_numpy(dtype=float)
    kappa_values = unique_comparisons["cohen_kappa"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    family = str(detail["family"].iloc[0])
    for group in ENV_GROUPS:
        subset = detail.loc[detail["environmental_group"] == group]
        rho = subset["spearman_rho"].to_numpy(dtype=float)
        tau = subset["kendall_tau"].to_numpy(dtype=float)
        rows.append(
            {
                "family": family,
                "environmental_group": group,
                "comparisons_n": len(subset),
                "matched_site_n_min": int(subset["matched_site_n"].min()),
                "matched_site_n_max": int(subset["matched_site_n"].max()),
                "spearman_rho_min": float(np.min(rho)),
                "spearman_rho_median": float(np.median(rho)),
                "spearman_rho_max": float(np.max(rho)),
                "kendall_tau_min": float(np.min(tau)),
                "kendall_tau_median": float(np.median(tau)),
                "kendall_tau_max": float(np.max(tau)),
                "median_spatial_attribution_class": spatial_class(float(np.median(rho))),
                "dominant_agreement_min": float(np.min(agreement_values)),
                "dominant_agreement_median": float(np.median(agreement_values)),
                "dominant_agreement_max": float(np.max(agreement_values)),
                "dominant_group_stability_class": dominant_class(float(np.median(agreement_values))),
                "cohen_kappa_min": float(np.min(kappa_values)),
                "cohen_kappa_median": float(np.median(kappa_values)),
                "cohen_kappa_max": float(np.max(kappa_values)),
            }
        )
    return pd.DataFrame(rows)


def vegetation_bootstrap(
    canonical: AggregatedRun,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    def calculate_once() -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        group_rows: list[dict[str, Any]] = []
        dominant_rows: list[dict[str, Any]] = []
        for veg_class in VEGETATION_CLASSES:
            sites = canonical.site_group.loc[
                canonical.site_group["veg_class"].astype(str) == veg_class
            ].reset_index(drop=True)
            require(len(sites) > 0, f"No canonical sites for vegetation class: {veg_class}")
            dominant = canonical.site_dominant.set_index("point_id").loc[
                sites["point_id"], "dominant_environmental_group"
            ].astype(str).to_numpy()
            values = sites[ENV_GROUPS].to_numpy(dtype=float)
            indices = rng.integers(0, len(sites), size=(BOOTSTRAP_B, len(sites)))
            replicate_means = values[indices].mean(axis=1)
            replicate_dominant = dominant[indices]
            for group_index, group in enumerate(ENV_GROUPS):
                lower, upper = np.percentile(
                    replicate_means[:, group_index], [2.5, 97.5], method="linear"
                )
                group_rows.append(
                    {
                        "veg_class": veg_class,
                        "environmental_group": group,
                        "n_sites": len(sites),
                        "B": BOOTSTRAP_B,
                        "seed": BOOTSTRAP_SEED,
                        "mean_group_attribution": float(values[:, group_index].mean()),
                        "lower95": float(lower),
                        "upper95": float(upper),
                    }
                )
                proportions = np.mean(replicate_dominant == group, axis=1)
                prop_lower, prop_upper = np.percentile(
                    proportions, [2.5, 97.5], method="linear"
                )
                dominant_rows.append(
                    {
                        "veg_class": veg_class,
                        "dominant_environmental_group": group,
                        "n_sites": len(sites),
                        "B": BOOTSTRAP_B,
                        "seed": BOOTSTRAP_SEED,
                        "proportion": float(np.mean(dominant == group)),
                        "lower95": float(prop_lower),
                        "upper95": float(prop_upper),
                    }
                )
        return pd.DataFrame(group_rows), pd.DataFrame(dominant_rows)

    first_group, first_dominant = calculate_once()
    second_group, second_dominant = calculate_once()
    reproducible = first_group.equals(second_group) and first_dominant.equals(second_dominant)
    require(reproducible, "Fixed-seed point_id bootstrap did not reproduce exactly")
    return first_group, first_dominant, reproducible


def build_vegetation_outputs(
    canonical: AggregatedRun,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    summary_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for veg_class in VEGETATION_CLASSES:
        sites = canonical.site_group.loc[
            canonical.site_group["veg_class"].astype(str) == veg_class
        ].copy()
        require(len(sites) > 0, f"Missing vegetation class: {veg_class}")
        means = sites[ENV_GROUPS].mean(axis=0)
        denominator = float(means.sum())
        require(denominator > 0, f"Zero class attribution denominator: {veg_class}")
        dominant_group = str(means.idxmax())
        row: dict[str, Any] = {
            "veg_class": veg_class,
            "n_sites": len(sites),
            "n_explained_samples": int(sites["n_explained_samples"].sum()),
        }
        for group in ENV_GROUPS:
            row[f"{safe_group_name(group)}_mean_abs_shap_gC_m2_month"] = float(means[group])
            row[f"{safe_group_name(group)}_share"] = float(means[group] / denominator)
        row["dominant_environmental_group"] = dominant_group
        summary_rows.append(row)

        site_dominant = canonical.site_dominant.loc[
            canonical.site_dominant["veg_class"].astype(str) == veg_class,
            "dominant_environmental_group",
        ].astype(str)
        require(len(site_dominant) == len(sites), f"Vegetation site count mismatch: {veg_class}")
        for group in ENV_GROUPS:
            count = int((site_dominant == group).sum())
            count_rows.append(
                {
                    "veg_class": veg_class,
                    "n_sites": len(sites),
                    "dominant_environmental_group": group,
                    "count": count,
                    "proportion": count / len(sites),
                }
            )
    group_ci, dominant_ci, reproducible = vegetation_bootstrap(canonical)
    return (
        pd.DataFrame(summary_rows),
        group_ci,
        pd.DataFrame(count_rows),
        dominant_ci,
        reproducible,
    )


def save_figure(fig: plt.Figure, relative_path: str) -> Path:
    path = safe_output(relative_path)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": relative_path,
            "Description": "Sampled-site model attribution diagnostic; no spatial interpolation.",
        },
    )
    plt.close(fig)
    require(path.is_file() and path.stat().st_size > 10_000, f"Figure output is missing or unexpectedly small: {path}")
    return path


def make_figures(
    canonical: AggregatedRun,
    vegetation_summary: pd.DataFrame,
    family_details: Mapping[str, pd.DataFrame],
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    outputs: list[Path] = []
    site_dom = canonical.site_dominant.copy()
    category_colors = {
        "Energy": "#E69F00",
        "Vegetation productivity": "#009E73",
        "Thermal": "#D55E00",
        "Hydrologic": "#0072B2",
        "Terrain": "#7A5195",
    }
    mean_lat = float(site_dom["Lat"].mean())
    geographic_aspect = 1.0 / math.cos(math.radians(mean_lat))

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    for group in ENV_GROUPS:
        subset = site_dom.loc[site_dom["dominant_environmental_group"] == group]
        ax.scatter(
            subset["Lon"],
            subset["Lat"],
            s=34,
            c=category_colors[group],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.92,
            label=f"{group} (n={len(subset)})",
        )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(geographic_aspect)
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.set_title(
        "A | Canonical sampled-site dominant environmental attribution group",
        pad=34,
    )
    ax.text(
        0.0,
        1.012,
        "W36 · BG256_D1 · E0",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.legend(title="Model attribution group", loc="best", frameon=True)
    outputs.append(
        save_figure(
            fig,
            "figures/geoshap_figure_A_canonical_dominant_environmental_group_map.png",
        )
    )

    def continuous_site_map(
        feature: str,
        panel: str,
        title: str,
        relative_path: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
        values = canonical.site_variable[feature].to_numpy(dtype=float)
        scatter = ax.scatter(
            canonical.site_variable["Lon"],
            canonical.site_variable["Lat"],
            c=values,
            cmap="viridis",
            s=36,
            edgecolors="white",
            linewidths=0.3,
        )
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect(geographic_aspect)
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.7)
        ax.set_title(
            f"{panel} | {title}",
            pad=34,
        )
        ax.text(
            0.0,
            1.012,
            "W36 · BG256_D1 · E0",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.045)
        colorbar.set_label("Mean |SHAP| (gC m$^{-2}$ month$^{-1}$)")
        return save_figure(fig, relative_path)

    outputs.append(
        continuous_site_map(
            "NDVI",
            "B",
            "Canonical site-level NDVI mean |SHAP|",
            "figures/geoshap_figure_B_canonical_NDVI_mean_abs_shap_map.png",
        )
    )
    outputs.append(
        continuous_site_map(
            "surface_solar_radiation_downwards_sum",
            "C",
            "Canonical site-level solar-radiation mean |SHAP|",
            "figures/geoshap_figure_C_canonical_solar_radiation_mean_abs_shap_map.png",
        )
    )

    heat_values = vegetation_summary.set_index("veg_class").loc[
        VEGETATION_CLASSES,
        [f"{safe_group_name(group)}_mean_abs_shap_gC_m2_month" for group in ENV_GROUPS],
    ].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    mesh = ax.pcolormesh(
        np.arange(len(ENV_GROUPS) + 1),
        np.arange(len(VEGETATION_CLASSES) + 1),
        heat_values,
        cmap="YlGnBu",
        shading="flat",
        edgecolors="white",
        linewidth=1.0,
    )
    ax.set_xticks(np.arange(len(ENV_GROUPS)) + 0.5, ENV_GROUPS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(VEGETATION_CLASSES)) + 0.5, VEGETATION_CLASSES)
    ax.invert_yaxis()
    threshold = float((heat_values.min() + heat_values.max()) / 2)
    for row_index in range(heat_values.shape[0]):
        for col_index in range(heat_values.shape[1]):
            value = heat_values[row_index, col_index]
            ax.text(
                col_index + 0.5,
                row_index + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    ax.set_title("D | Vegetation-class environmental-group model attribution")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    colorbar.set_label("Site-weighted mean |SHAP| (gC m$^{-2}$ month$^{-1}$)")
    outputs.append(
        save_figure(
            fig,
            "figures/geoshap_figure_D_vegetation_environmental_group_attribution_heatmap.png",
        )
    )

    family_order = ["background", "seed", "explained"]
    family_titles = {
        "background": "Background family",
        "seed": "Three-seed family",
        "explained": "Explained-sample family",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7), sharey=True, constrained_layout=True)
    for ax, family in zip(axes, family_order):
        detail = family_details[family]
        for group_index, group in enumerate(ENV_GROUPS):
            values = detail.loc[
                detail["environmental_group"] == group, "spearman_rho"
            ].to_numpy(dtype=float)
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                group_index + offsets,
                values,
                s=25,
                alpha=0.75,
                color=category_colors[group],
                edgecolors="white",
                linewidths=0.25,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot(
                [group_index - 0.2, group_index + 0.2],
                [median, median],
                color="#222222",
                linewidth=1.8,
                zorder=4,
            )
        ax.axhline(0.80, color="#555555", linestyle="--", linewidth=0.8)
        ax.axhline(0.60, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_xticks(range(len(ENV_GROUPS)), ["Energy", "Vegetation\nproductivity", "Thermal", "Hydrologic", "Terrain"], rotation=25, ha="right")
        ax.set_title(family_titles[family])
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.45)
        ax.set_ylim(min(0.0, float(detail["spearman_rho"].min()) - 0.05), 1.02)
    axes[0].set_ylabel("Spatial Spearman ρ")
    fig.suptitle("E | Site-level model-attribution spatial robustness")
    outputs.append(
        save_figure(
            fig,
            "figures/geoshap_figure_E_spatial_spearman_robustness_summary.png",
        )
    )
    return outputs


def family_range_dict(summary: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ENV_GROUPS:
        row = summary.loc[summary["environmental_group"] == group].iloc[0]
        result[group] = {
            "spearman_rho": {
                "min": float(row["spearman_rho_min"]),
                "median": float(row["spearman_rho_median"]),
                "max": float(row["spearman_rho_max"]),
            },
            "kendall_tau": {
                "min": float(row["kendall_tau_min"]),
                "median": float(row["kendall_tau_median"]),
                "max": float(row["kendall_tau_max"]),
            },
            "matched_site_n": {
                "min": int(row["matched_site_n_min"]),
                "max": int(row["matched_site_n_max"]),
            },
            "median_spatial_attribution_class": str(row["median_spatial_attribution_class"]),
        }
    first = summary.iloc[0]
    result["dominant_environmental_group"] = {
        "agreement": {
            "min": float(first["dominant_agreement_min"]),
            "median": float(first["dominant_agreement_median"]),
            "max": float(first["dominant_agreement_max"]),
        },
        "cohen_kappa": {
            "min": float(first["cohen_kappa_min"]),
            "median": float(first["cohen_kappa_median"]),
            "max": float(first["cohen_kappa_max"]),
        },
        "stability_class": str(first["dominant_group_stability_class"]),
    }
    return result


def format_range(minimum: float, median: float, maximum: float) -> str:
    return f"min={minimum:.3f}, median={median:.3f}, max={maximum:.3f}"


def build_summary(
    validation: Mapping[str, Any],
    family_summaries: Mapping[str, pd.DataFrame],
    adverse_results: Sequence[str],
) -> str:
    site_count = int(validation["site_count"])
    coverage = "YES" if validation["coverage_247_point_ids"] else "NO"
    robust = validation["ROBUST_SPATIAL_ATTRIBUTION"]
    dominant_overall = validation["dominant_group_stability_class"]

    def group_sentence(group: str) -> str:
        pieces = []
        for family in ("background", "seed", "explained"):
            row = family_summaries[family].loc[
                family_summaries[family]["environmental_group"] == group
            ].iloc[0]
            pieces.append(
                f"{family}: {format_range(float(row['spearman_rho_min']), float(row['spearman_rho_median']), float(row['spearman_rho_max']))} ({row['median_spatial_attribution_class']})"
            )
        return "; ".join(pieces)

    stability_rows = []
    for group in ENV_GROUPS:
        cells = [group]
        for family in ("background", "seed", "explained"):
            row = family_summaries[family].loc[
                family_summaries[family]["environmental_group"] == group
            ].iloc[0]
            cells.append(
                f"{float(row['spearman_rho_min']):.3f} / {float(row['spearman_rho_median']):.3f} / {float(row['spearman_rho_max']):.3f} ({row['median_spatial_attribution_class']})"
            )
        stability_rows.append("| " + " | ".join(cells) + " |")

    dominant_rows = []
    for family in ("background", "seed", "explained"):
        row = family_summaries[family].iloc[0]
        dominant_rows.append(
            "| "
            + " | ".join(
                [
                    family,
                    f"{float(row['dominant_agreement_min']):.3f}",
                    f"{float(row['dominant_agreement_median']):.3f}",
                    f"{float(row['dominant_agreement_max']):.3f}",
                    str(row["dominant_group_stability_class"]),
                    f"{float(row['cohen_kappa_median']):.3f}",
                ]
            )
            + " |"
        )

    adverse_text = (
        "\n".join(f"- {item}" for item in adverse_results)
        if adverse_results
        else "- All predefined stability criteria were met."
    )
    vegetation_counts = ", ".join(
        f"{row['veg_class']}={row['n_sites']}"
        for row in validation["vegetation_classes_and_n_sites"]
    )
    explained_counts = ", ".join(
        f"{draw_id.replace('EXPLAINED_', '')}={count}"
        for draw_id, count in validation["explained_manifest_site_counts"].items()
    )

    return f"""# W36 sampled-site GeoSHAP summary

Generated: {validation['created_at']}

## Analysis

W36 primary attribution averages absolute contributions across seeds 42, 2024 and 3407 using BG256_D1 / EXPLAINED_E0, then averages within `point_id`. Site count = {site_count}; E0/E1/E2/E3 manifest site counts = {explained_counts}. Vegetation summaries weight sites equally ({vegetation_counts}). Point_id-cluster bootstrap intervals use B={BOOTSTRAP_B} and seed={BOOTSTRAP_SEED}.

## Five-group spatial Spearman stability

| Environmental group | Background min / median / max | Seed min / median / max | Explained min / median / max |
|---|---:|---:|---:|
{chr(10).join(stability_rows)}

## Dominant environmental-group agreement

| Family | Min agreement | Median agreement | Max agreement | Class | Median Cohen kappa |
|---|---:|---:|---:|---|---:|
{chr(10).join(dominant_rows)}

`dominant_group_stability_class` summarizes the three family median-agreement classes; the pooled median is recorded in the validation JSON.

## Stability results

{adverse_text}

## Attribution definition

Absolute SHAP values are aggregated from sample to site. Environmental dominance uses the five groups Energy, Vegetation productivity, Thermal, Hydrologic, and Terrain. Seasonal phase and vegetation class are summarized separately.
"""


def build_provenance_markdown(
    input_validation: Mapping[str, Any],
    end_tree: Mapping[str, Any],
    statistics_started_at: str,
    output_counts: Mapping[str, Any],
) -> str:
    return f"""# W36 GeoSHAP input provenance

Primary inputs: `background_BG256_D1`, `seed_2024`, `seed_3407`;
common explained keys: `EXPLAINED_E0`.

Contributions: sum absolute SHAP over 36 lags, average across three seeds,
then average within point_id. Group contributions sum their member predictors.
Environmental shares use Energy, Vegetation productivity, Thermal, Hydrologic,
and Terrain in the denominator. Vegetation summaries weight sites equally.

Bootstrap: {BOOTSTRAP_B} point_id resamples within each vegetation class.
Site rows: {output_counts['canonical_site_rows']}.
Results: `canonical/`, `vegetation/`, and `robustness/`.
"""


def run_postprocessing() -> dict[str, Any]:
    input_validation_path = VALIDATION_DIR / "geoshap_input_validation.json"
    input_inventory_path = VALIDATION_DIR / "geoshap_input_inventory.csv"
    require(input_validation_path.is_file(), "Input validation is missing; run --phase validate first")
    require(input_inventory_path.is_file(), "Input inventory is missing; run --phase validate first")
    input_validation = load_json(input_validation_path)
    require(input_validation.get("status") == "PASS", "Input validation is not PASS")
    require(input_validation.get("statistics_computed") is False, "Input validation does not precede statistics")
    require(sha256_file(input_inventory_path) == input_validation["input_inventory_sha256"], "Input inventory changed after validation")
    pre_statistics_tree = original_tree_digest()
    require(pre_statistics_tree == input_validation["input_shap_tree_before"], "Input SHAP tree changed before statistics")
    boundary_checks = check_code_boundary()
    require(all(boundary_checks.values()), "Code boundary check failed before statistics")
    statistics_started_at = now_iso()

    aggregated = {spec.run_id: aggregate_run(spec) for spec in run_plan()}
    canonical = aggregate_three_seeds(aggregated)
    sensitivity_reference = aggregated["background_BG256_D1"]
    canonical_site_count = len(canonical.site_variable)

    variable_output = canonical.site_variable[
        ["point_id", "Lon", "Lat", "veg_class", "n_explained_samples"] + FEATURES
    ].rename(columns={feature: feature_output_column(feature) for feature in FEATURES})
    write_csv(
        "canonical/geoshap_canonical_site_variable_abs_shap.csv",
        variable_output,
    )

    group_output = canonical.site_group[
        ["point_id", "Lon", "Lat", "veg_class", "n_explained_samples"]
        + list(GROUPS)
    ].rename(columns={group: group_output_column(group) for group in GROUPS})
    write_csv(
        "canonical/geoshap_canonical_site_group_abs_shap.csv",
        group_output,
    )

    dominant_columns = [
        "point_id",
        "Lon",
        "Lat",
        "veg_class",
        "dominant_environmental_group",
        "dominant_group_abs_shap",
        "dominant_group_ratio",
    ] + [f"{group}_ratio" for group in ENV_GROUPS]
    dominant_output = canonical.site_dominant[dominant_columns].rename(
        columns={f"{group}_ratio": ratio_output_column(group) for group in ENV_GROUPS}
    )
    write_csv(
        "canonical/geoshap_canonical_dominant_environmental_group.csv",
        dominant_output,
    )

    nonseasonal_values = canonical.site_variable[NONSEASONAL_FEATURES].to_numpy(dtype=float)
    nonseasonal_ties = np.isclose(
        nonseasonal_values,
        nonseasonal_values.max(axis=1, keepdims=True),
        rtol=0.0,
        atol=0.0,
    ).sum(axis=1)
    require((nonseasonal_ties == 1).all(), "Tied dominant nonseasonal predictor")
    nonseasonal_indices = nonseasonal_values.argmax(axis=1)
    continuity = canonical.site_variable[
        ["point_id", "Lon", "Lat", "veg_class"]
    ].copy()
    continuity["dominant_nonseasonal_predictor"] = [
        NONSEASONAL_FEATURES[index] for index in nonseasonal_indices
    ]
    continuity["dominant_nonseasonal_predictor_abs_shap"] = nonseasonal_values[
        np.arange(len(continuity)), nonseasonal_indices
    ]
    write_csv(
        "canonical/geoshap_dominant_nonseasonal_predictor.csv",
        continuity,
    )

    background_rows: list[dict[str, Any]] = []
    for spec in run_plan():
        if spec.run_id.startswith("background_") and spec.run_id != sensitivity_reference.spec.run_id:
            background_rows.extend(
                compare_runs(
                    "background",
                    f"{sensitivity_reference.spec.run_id}_vs_{spec.run_id}",
                    sensitivity_reference,
                    aggregated[spec.run_id],
                )
            )
    background_detail = pd.DataFrame(background_rows)
    require(background_detail["comparison_id"].nunique() == 11, "Background comparison count is not 11")
    background_summary = summarize_family(background_detail)
    write_csv(
        "robustness/geoshap_background_spatial_stability.csv",
        background_detail,
    )
    write_csv(
        "robustness/geoshap_background_spatial_stability_summary.csv",
        background_summary,
    )

    seed_pairs = [
        ("42_vs_2024", sensitivity_reference.spec.run_id, "seed_2024"),
        ("42_vs_3407", sensitivity_reference.spec.run_id, "seed_3407"),
        ("2024_vs_3407", "seed_2024", "seed_3407"),
    ]
    seed_rows: list[dict[str, Any]] = []
    for comparison_id, left_id, right_id in seed_pairs:
        seed_rows.extend(
            compare_runs(
                "seed",
                comparison_id,
                aggregated[left_id],
                aggregated[right_id],
            )
        )
    seed_detail = pd.DataFrame(seed_rows)
    require(seed_detail["comparison_id"].nunique() == 3, "Seed comparison count is not 3")
    seed_summary = summarize_family(seed_detail)
    write_csv(
        "robustness/geoshap_three_seed_spatial_stability.csv",
        seed_detail,
    )
    write_csv(
        "robustness/geoshap_three_seed_spatial_stability_summary.csv",
        seed_summary,
    )

    explained_rows: list[dict[str, Any]] = []
    for draw in (1, 2, 3):
        run_id = f"explained_E{draw}"
        explained_rows.extend(
            compare_runs(
                "explained",
                f"E0_vs_E{draw}",
                sensitivity_reference,
                aggregated[run_id],
            )
        )
    explained_detail = pd.DataFrame(explained_rows)
    require(explained_detail["comparison_id"].nunique() == 3, "Explained comparison count is not 3")
    explained_summary = summarize_family(explained_detail)
    write_csv(
        "robustness/geoshap_explained_spatial_stability.csv",
        explained_detail,
    )
    write_csv(
        "robustness/geoshap_explained_spatial_stability_summary.csv",
        explained_summary,
    )

    (
        vegetation_summary,
        vegetation_group_ci,
        vegetation_counts,
        vegetation_dominant_ci,
        bootstrap_reproducible,
    ) = build_vegetation_outputs(canonical)
    write_csv(
        "vegetation/geoshap_vegetation_group_summary.csv",
        vegetation_summary,
    )
    write_csv(
        "vegetation/geoshap_vegetation_group_bootstrap_ci.csv",
        vegetation_group_ci,
    )
    write_csv(
        "vegetation/geoshap_vegetation_dominant_group_counts.csv",
        vegetation_counts,
    )
    write_csv(
        "vegetation/geoshap_vegetation_dominant_group_bootstrap_ci.csv",
        vegetation_dominant_ci,
    )

    family_details = {
        "background": background_detail,
        "seed": seed_detail,
        "explained": explained_detail,
    }
    family_summaries = {
        "background": background_summary,
        "seed": seed_summary,
        "explained": explained_summary,
    }
    figure_paths = make_figures(canonical, vegetation_summary, family_details)

    core_medians: dict[str, dict[str, float]] = {}
    core_minimums: dict[str, dict[str, float]] = {}
    for group in CORE_GROUPS:
        core_medians[group] = {}
        core_minimums[group] = {}
        for family in ("background", "seed", "explained"):
            row = family_summaries[family].loc[
                family_summaries[family]["environmental_group"] == group
            ].iloc[0]
            core_medians[group][family] = float(row["spearman_rho_median"])
            core_minimums[group][family] = float(row["spearman_rho_min"])

    condition_a = all(core_medians[group]["background"] >= 0.80 for group in CORE_GROUPS)
    condition_b = all(core_medians[group]["seed"] >= 0.80 for group in CORE_GROUPS)
    condition_c = all(core_medians[group]["explained"] >= 0.80 for group in CORE_GROUPS)
    all_detail = pd.concat(family_details.values(), ignore_index=True)
    core_detail = all_detail.loc[all_detail["environmental_group"].isin(CORE_GROUPS)]
    condition_d = bool((core_detail["spearman_rho"] >= 0.70).all())
    robust_spatial_attribution = "YES" if all([condition_a, condition_b, condition_c, condition_d]) else "NO"

    family_dominant_classes = {
        family: str(summary.iloc[0]["dominant_group_stability_class"])
        for family, summary in family_summaries.items()
    }
    class_rank = {"WEAK": 0, "MODERATE": 1, "STRONG": 2}
    overall_dominant_class = min(
        family_dominant_classes.values(), key=lambda value: class_rank[value]
    )
    pooled_dominant = all_detail[
        ["family", "comparison_id", "dominant_environmental_group_agreement"]
    ].drop_duplicates()["dominant_environmental_group_agreement"].to_numpy(dtype=float)

    adverse_results: list[str] = []
    for family, summary in family_summaries.items():
        for _, row in summary.iterrows():
            if row["median_spatial_attribution_class"] != "STRONG":
                adverse_results.append(
                    f"{family} / {row['environmental_group']}: median spatial Spearman rho={float(row['spearman_rho_median']):.3f} ({row['median_spatial_attribution_class']}); minimum={float(row['spearman_rho_min']):.3f}."
                )
        first = summary.iloc[0]
        if first["dominant_group_stability_class"] != "STRONG":
            adverse_results.append(
                f"{family} dominant-group agreement: median={float(first['dominant_agreement_median']):.3f} ({first['dominant_group_stability_class']}); minimum={float(first['dominant_agreement_min']):.3f}."
            )
    if not condition_d:
        failing = core_detail.loc[core_detail["spearman_rho"] < 0.70]
        for _, row in failing.iterrows():
            adverse_results.append(
                f"Locked core-surface individual comparison below 0.70: {row['comparison_id']} / {row['environmental_group']} rho={float(row['spearman_rho']):.3f}."
            )
    if robust_spatial_attribution == "NO":
        adverse_results.append(
            "ROBUST_SPATIAL_ATTRIBUTION status: criterion below the predefined threshold."
        )

    end_tree = original_tree_digest()
    require(end_tree == input_validation["input_shap_tree_before"], "Input SHAP tree changed during post-processing")
    historic_statement = SHAP_ROOT / "geoshap" / "shap_geoshap_scope.md"
    require(sha256_file(historic_statement) == input_validation["historic_geoshap_statement_sha256"], "Historic Geo-SHAP statement changed")

    vegetation_site_counts = [
        {
            "veg_class": str(row["veg_class"]),
            "n_sites": int(row["n_sites"]),
            "n_explained_samples": int(row["n_explained_samples"]),
        }
        for _, row in vegetation_summary.iterrows()
    ]
    validation_payload: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "work_package": ANALYSIS_NAME,
        "created_at": now_iso(),
        "canonical_configuration": {
            "window": "W36",
            "model_seeds": [42, 2024, 3407],
            "background_draw": "BG256_D1",
            "explained_draw": "EXPLAINED_E0",
            "run_id": "three_seed_aggregate",
            "shap_array_shape": [EXPLAINED_N, WINDOW, INPUT_DIM],
            "aggregation": "sample_abs=mean_seed(sum_lag(abs(SHAP))); site_abs=mean_within_point_id",
        },
        "site_count": canonical_site_count,
        "coverage_247_point_ids": canonical_site_count == 247,
        "explained_manifest_site_counts": input_validation["explained_manifest_site_counts"],
        "background_spatial_stability_ranges": family_range_dict(background_summary),
        "seed_spatial_stability_ranges": family_range_dict(seed_summary),
        "explained_spatial_stability_ranges": family_range_dict(explained_summary),
        "core_surface_median_correlations": core_medians,
        "core_surface_minimum_correlations": core_minimums,
        "robust_spatial_attribution_conditions": {
            "A_background_core_medians_ge_0_80": condition_a,
            "B_seed_core_medians_ge_0_80": condition_b,
            "C_explained_core_medians_ge_0_80": condition_c,
            "D_all_core_individual_comparisons_ge_0_70": condition_d,
        },
        "dominant_group_agreement_summaries": {
            family: family_range_dict(summary)["dominant_environmental_group"]
            for family, summary in family_summaries.items()
        },
        "dominant_group_stability_definition": "weakest of the three family median-agreement classes",
        "dominant_group_stability_class": overall_dominant_class,
        "pooled_dominant_group_median_agreement": float(np.median(pooled_dominant)),
        "pooled_dominant_group_stability_class": dominant_class(float(np.median(pooled_dominant))),
        "vegetation_classes_and_n_sites": vegetation_site_counts,
        "vegetation_bootstrap": {
            "unit": "point_id within vegetation class",
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "interval": "two-sided 95% percentile",
            "fixed_seed_exact_repeat_check": bootstrap_reproducible,
            "successful": True,
        },
        "prospective_thresholds": {
            "continuous_STRONG": "rho >= 0.80",
            "continuous_MODERATE": "0.60 <= rho < 0.80",
            "continuous_WEAK": "rho < 0.60",
            "dominant_STRONG": "median agreement >= 0.70",
            "dominant_MODERATE": "0.50 <= median agreement < 0.70",
            "dominant_WEAK": "median agreement < 0.50",
        },
        "adverse_or_unstable_results": adverse_results,
        "ROBUST_SPATIAL_ATTRIBUTION": robust_spatial_attribution,
        "ANALYSIS_STATUS": "PASS",
        "provenance": {
            "input_validation_sha256": sha256_file(input_validation_path),
            "input_inventory_sha256": sha256_file(input_inventory_path),
            "input_shap_tree_sha256_before": input_validation["input_shap_tree_before"]["tree_sha256"],
            "input_shap_tree_sha256_after": end_tree["tree_sha256"],
            "input_shap_tree_integrity": True,
            "historic_geoshap_statement_preserved": True,
        },
    }
    validation_path = write_json(
        "summary/w36_geoshap_validation.json",
        validation_payload,
    )

    summary_text = build_summary(
        validation_payload, family_summaries, adverse_results
    )
    summary_path = safe_output(
        "summary/W36_GEOSHAP_ANALYSIS_SUMMARY.md"
    )
    atomic_write_text(summary_path, summary_text)

    output_counts = {
        "canonical_site_rows": canonical_site_count,
        "background_comparisons": int(background_detail["comparison_id"].nunique()),
        "seed_comparisons": int(seed_detail["comparison_id"].nunique()),
        "explained_comparisons": int(explained_detail["comparison_id"].nunique()),
        "vegetation_classes": len(vegetation_summary),
        "figures": len(figure_paths),
    }
    provenance_text = build_provenance_markdown(
        input_validation,
        end_tree,
        statistics_started_at,
        output_counts,
    )
    provenance_path = safe_output("validation/geoshap_provenance.md")
    atomic_write_text(provenance_path, provenance_text)

    for relative_path in REQUESTED_OUTPUTS:
        path = SHAP_ROOT / "geoshap_analysis" / relative_path
        require(path.is_file(), f"Required analysis output missing: {relative_path}")
        require(path.stat().st_size > 0, f"Required analysis output empty: {relative_path}")

    final_tree = original_tree_digest()
    require(final_tree == input_validation["input_shap_tree_before"], "Input SHAP tree changed after summary generation")

    output_inventory_rows = []
    output_inventory_path = GEOSHAP_ROOT / "validation" / "geoshap_output_inventory.csv"
    for path in sorted(
        (item for item in GEOSHAP_ROOT.rglob("*") if item.is_file() and item != output_inventory_path),
        key=lambda item: item.relative_to(GEOSHAP_ROOT).as_posix(),
    ):
        output_inventory_rows.append(
            {
                "relative_path": path.relative_to(GEOSHAP_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(
        "validation/geoshap_output_inventory.csv",
        pd.DataFrame(output_inventory_rows),
    )

    return {
        "ANALYSIS_STATUS": "PASS",
        "ROBUST_SPATIAL_ATTRIBUTION": robust_spatial_attribution,
        "site_count": canonical_site_count,
        "dominant_group_stability_class": overall_dominant_class,
        "validation_json": str(validation_path),
        "summary": str(summary_path),
        "provenance": str(provenance_path),
        "figures": [str(path) for path in figure_paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("validate", "run"),
        help="validate must complete before run",
    )
    args = parser.parse_args()
    try:
        if args.phase == "validate":
            result = validate_input_files()
        else:
            result = run_postprocessing()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "GEOSHAP_TECHNICAL_STATUS": "FAIL",
                    "phase": args.phase,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "fail_closed": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
