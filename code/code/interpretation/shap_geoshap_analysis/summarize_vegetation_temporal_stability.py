#!/usr/bin/env python3
"""Summarize W36 vegetation-conditioned and temporal SHAP patterns."""

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
from scipy.stats import kendalltau, pearsonr, spearmanr


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
SHAP_ROOT = PROJECT_ROOT / "runtime" / "runs" / "shap"
GEOSHAP_ROOT = PROJECT_ROOT / "runtime" / "runs" / "geoshap"
TEMPORAL_ROOT = PROJECT_ROOT / "runtime" / "runs" / "vegetation_temporal"
SAMPLE_ROOT = PROJECT_ROOT / "data_documentation" / "shap_sample_keys"

VALIDATION_DIR = TEMPORAL_ROOT / "validation"
RESULTS_DIR = TEMPORAL_ROOT / "results"
FIGURES_DIR = TEMPORAL_ROOT / "figures"
CODE_DIR = TEMPORAL_ROOT / "code"
SUMMARY_DIR = TEMPORAL_ROOT / "summary"

PROJECT_ID = "lancang_monthly_productivity"
ANALYSIS_NAME = "Vegetation temporal stability"
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

RESULT_FILES = [
    "results/vegetation_temporal_explained_vegetation_group_summary.csv",
    "results/vegetation_temporal_explained_vegetation_composition_stability.csv",
    "results/vegetation_temporal_explained_vegetation_class_stability.csv",
    "results/vegetation_temporal_seed_vegetation_composition_stability.csv",
    "results/vegetation_temporal_background_vegetation_composition_stability.csv",
    "results/vegetation_temporal_background_vegetation_stability_summary.csv",
    "results/vegetation_temporal_nonseasonal_predictor_dominance_by_vegetation_E0_E3.csv",
    "results/vegetation_temporal_nonseasonal_predictor_dominance_stability.csv",
    "results/vegetation_temporal_explained_set_temporal_composition.csv",
    "results/vegetation_temporal_explained_set_pairwise_overlap.csv",
    "results/vegetation_temporal_within_site_temporal_overlap.csv",
    "results/vegetation_temporal_within_site_temporal_overlap_summary.csv",
    "results/vegetation_temporal_explained_vegetation_bootstrap_ci.csv",
]

FIGURE_FILES = [
    "figures/vegetation_temporal_figure_A_E0_E3_vegetation_group_share_heatmaps.png",
    "figures/vegetation_temporal_figure_B_vegetation_group_share_changes.png",
    "figures/vegetation_temporal_figure_C_E0_composition_comparisons.png",
    "figures/vegetation_temporal_figure_D_nonseasonal_predictor_composition.png",
    "figures/vegetation_temporal_figure_E_within_site_temporal_overlap.png",
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


def shap_rel(path: Path) -> str:
    return path.resolve().relative_to(SHAP_ROOT.resolve()).as_posix()


def geoshap_relative(path: Path) -> str:
    return path.resolve().relative_to(TEMPORAL_ROOT.resolve()).as_posix()


def safe_output(relative_path: str) -> Path:
    candidate = (TEMPORAL_ROOT / relative_path).resolve()
    root = TEMPORAL_ROOT.resolve()
    require(
        os.path.commonpath([str(candidate), str(root)]) == str(root),
        f"Output path escapes vegetation temporal stability root: {candidate}",
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def atomic_write_text(path: Path, text: str) -> None:
    root = TEMPORAL_ROOT.resolve()
    require(
        os.path.commonpath([str(path.resolve()), str(root)]) == str(root),
        f"Refusing write outside vegetation temporal stability root: {path}",
    )
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"Unexpected temporary file exists: {temporary}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(relative_path: str, payload: Mapping[str, Any]) -> Path:
    path = safe_output(relative_path)
    require(not path.exists(), f"Refusing to overwrite existing vegetation temporal stability file: {path}")
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return path


def write_csv(relative_path: str, frame: pd.DataFrame) -> Path:
    path = safe_output(relative_path)
    require(not path.exists(), f"Refusing to overwrite existing vegetation temporal stability file: {path}")
    require(len(frame.columns) > 0, f"CSV has no columns: {relative_path}")
    require((frame.columns.astype(str).str.len() > 0).all(), f"CSV has blank header: {relative_path}")
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
    require(path.is_file(), f"Missing sample manifest: {path}")
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
    require(list(frame.columns) == REQUIRED_MANIFEST_COLUMNS, f"Manifest schema mismatch: {path}")
    require(not frame.isna().any().any(), f"Missing values in manifest: {path}")
    if require_unique_sample_id:
        require(frame["sample_id"].is_unique, f"Duplicate sample_id: {path}")
        require(
            frame["row_order"].tolist() == list(range(len(frame))),
            f"row_order mismatch: {path}",
        )
    else:
        require(
            not frame.duplicated(["draw_id", "sample_id"]).any(),
            f"Duplicate draw/sample identity: {path}",
        )
        for draw_id, subset in frame.groupby("draw_id", sort=False, observed=True):
            require(
                subset["row_order"].tolist() == list(range(len(subset))),
                f"Within-draw row_order mismatch: {path}/{draw_id}",
            )
    require(np.isfinite(frame[["Lon", "Lat"]].to_numpy()).all(), f"Non-finite coordinates: {path}")
    require(frame["Month"].between(1, 12).all(), f"Invalid month: {path}")
    require(
        set(frame["veg_class"].astype(str)).issubset(set(VEGETATION_CLASSES)),
        f"Unexpected vegetation class: {path}",
    )
    return frame


def manifest_path(draw_id: str) -> Path:
    if draw_id.startswith("BG"):
        return SAMPLE_ROOT / "sample_keys" / "background" / f"shap_{draw_id}.csv"
    require(draw_id.startswith("EXPLAINED_E"), f"Unrecognized draw id: {draw_id}")
    return SAMPLE_ROOT / "sample_keys" / f"shap_{draw_id.replace('EXPLAINED_', 'explained_', 1)}.csv"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model_seed: int
    background_draw: str
    background_n: int
    explained_draw: str


def run_plan() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            specs.append(RunSpec(f"background_{draw_id}", 42, draw_id, size, "EXPLAINED_E0"))
    specs.extend(
        [
            RunSpec("seed_2024", 2024, "BG256_D1", 256, "EXPLAINED_E0"),
            RunSpec("seed_3407", 3407, "BG256_D1", 256, "EXPLAINED_E0"),
            RunSpec("explained_E1", 42, "BG256_D1", 256, "EXPLAINED_E1"),
            RunSpec("explained_E2", 42, "BG256_D1", 256, "EXPLAINED_E2"),
            RunSpec("explained_E3", 42, "BG256_D1", 256, "EXPLAINED_E3"),
        ]
    )
    require(len(specs) == 17, "Run plan is not the validated 17-run set")
    return specs


def run_paths(spec: RunSpec) -> dict[str, Path]:
    base = SHAP_ROOT / "shap_runs"
    return {
        "shap_npz": base / f"shap_{spec.run_id}.npz",
        "variable_csv": base / f"shap_{spec.run_id}_variable.csv",
        "grouped_csv": base / f"shap_{spec.run_id}_grouped.csv",
        "signed_csv": base / f"shap_{spec.run_id}_signed.csv",
        "execution_json": base / f"shap_{spec.run_id}_execution.json",
        "completion_marker": base / "completed" / f"shap_{spec.run_id}.complete.json",
    }


def input_tree_digest() -> dict[str, Any]:
    """Digest every pre-existing SHAP file while excluding only vegetation temporal stability outputs."""

    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    analysis_root = TEMPORAL_ROOT.resolve()
    paths = sorted(
        (
            path
            for path in SHAP_ROOT.rglob("*")
            if path.is_file() and analysis_root not in path.resolve().parents
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
        "excluded_subtree": shap_rel(TEMPORAL_ROOT),
    }


def geoshap_tree_digest() -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    analysis_root = TEMPORAL_ROOT.resolve()
    paths = sorted(
        (
            path
            for path in GEOSHAP_ROOT.rglob("*")
            if path.is_file() and analysis_root not in path.resolve().parents
        ),
        key=lambda path: path.relative_to(GEOSHAP_ROOT).as_posix(),
    )
    for path in paths:
        rel = path.relative_to(GEOSHAP_ROOT).as_posix()
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
        "excluded_subtree": geoshap_relative(TEMPORAL_ROOT),
    }


def check_code_boundary() -> dict[str, bool]:
    texts = []
    for path in sorted(CODE_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".py", ".mjs", ".js"}:
            texts.append(path.read_text(encoding="utf-8"))
    source = "\n".join(texts)
    checks = {
        "no_shap_import": re.search(r"^\s*(?:from\s+shap\b|import\s+shap\b)", source, re.MULTILINE) is None,
        "no_explainer_constructor": re.search(r"GradientExplainer\s*\(", source) is None,
        "no_torch_import": re.search(r"^\s*(?:from\s+torch\b|import\s+torch\b)", source, re.MULTILINE) is None,
        "no_tensorflow_import": re.search(r"^\s*(?:from\s+tensorflow\b|import\s+tensorflow\b)", source, re.MULTILINE) is None,
        "no_model_fit_call": re.search(r"\.fit\s*\(", source) is None,
        "no_subprocess": re.search(r"^\s*(?:from\s+subprocess\b|import\s+subprocess\b)", source, re.MULTILINE) is None,
    }
    return checks


def validate_inputs() -> dict[str, Any]:
    """Fail closed before any statistical aggregation."""

    for child in TEMPORAL_ROOT.iterdir():
        if child.is_file():
            require(False, f"Unexpected file at vegetation temporal stability root before validation: {child}")
    allowed_existing = {SCRIPT_PATH.name}
    for path in TEMPORAL_ROOT.rglob("*"):
        if path.is_file():
            require(path.parent == CODE_DIR and path.name in allowed_existing, f"Unexpected pre-existing vegetation temporal stability file: {path}")

    tree_before = input_tree_digest()
    geoshap_tree_before = geoshap_tree_digest()
    hash_cache: dict[Path, str] = {}
    input_roles: dict[Path, set[str]] = {}

    def cached_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_file(resolved)
        return hash_cache[resolved]

    def add_input(path: Path, role: str) -> None:
        require(path.is_file(), f"Missing required input: {path}")
        input_roles.setdefault(path.resolve(), set()).add(role)
        cached_hash(path)

    for relative_path in CONTROL_FILES:
        add_input(SHAP_ROOT / relative_path, "SHAP validated control / identity record")

    validation = load_json(SHAP_ROOT / "validation" / "analysis_summary.json")
    configuration = load_json(SHAP_ROOT / "validation" / "shap_run_configuration.json")
    require(validation.get("status") == "PASS", "SHAP final validation is not PASS")
    integrity = validation.get("validation", {})
    require(integrity.get("completion_markers_complete") == 17, "SHAP completion-marker count mismatch")
    require(integrity.get("output_artifacts_hash_verified") == 85, "SHAP run-output count mismatch")
    require(integrity.get("sample_manifests_verified") == 18, "SHAP sample-manifest count mismatch")
    require(integrity.get("all_run_plan_checks_passed") is True, "SHAP run plan not validated")
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
    require(list(sha_inventory.columns) == ["scope", "path", "bytes", "sha256"], "SHAP SHA inventory schema mismatch")
    require(sha_inventory["path"].is_unique, "Duplicate path in SHAP SHA inventory")
    inventory_records = sha_inventory.set_index("path").to_dict(orient="index")

    expected_input_paths = {
        f"sample_keys/background/shap_BG{size}_D{draw}.csv"
        for size in (64, 128, 256, 512)
        for draw in (1, 2, 3)
    }
    expected_input_paths.update({f"sample_keys/shap_explained_E{draw}.csv" for draw in range(4)})
    expected_input_paths.update(
        {"sample_keys/shap_background_manifests.csv", "sample_keys/shap_sampling_balance_summary.csv"}
    )
    input_records = {}
    for relative_path in expected_input_paths:
        path = SAMPLE_ROOT / relative_path
        require(path.is_file(), f"Missing sample manifest: {relative_path}")
        input_records[relative_path] = {
            "bytes": str(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    def verify_main_inventory(path: Path) -> None:
        relative = shap_rel(path)
        require(relative in inventory_records, f"File missing from SHAP SHA inventory: {relative}")
        record = inventory_records[relative]
        require(int(record["bytes"]) == path.stat().st_size, f"SHAP inventory byte mismatch: {relative}")
        require(record["sha256"] == cached_hash(path), f"SHAP inventory SHA256 mismatch: {relative}")

    for relative_path in (
        "validation/analysis_summary.json",
        "validation/shap_run_configuration.json",
    ):
        verify_main_inventory(SHAP_ROOT / relative_path)

    for relative_path, record in input_records.items():
        path = SAMPLE_ROOT / relative_path
        add_input(path, "sample manifest / sampling provenance")
        require(path.stat().st_size == int(record["bytes"]), f"Sample manifest byte mismatch: {relative_path}")
        require(cached_hash(path) == record["sha256"], f"Sample manifest SHA256 mismatch: {relative_path}")

    background_frames: dict[str, pd.DataFrame] = {}
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            frame = read_manifest(manifest_path(draw_id))
            require(len(frame) == size, f"Background manifest size mismatch: {draw_id}")
            require(frame["draw_id"].eq(draw_id).all(), f"Background draw id mismatch: {draw_id}")
            require(frame["selection_seed"].eq(configuration["background_selection_seeds"][draw_id]).all(), f"Background seed mismatch: {draw_id}")
            require(frame["split"].eq("train").all(), f"Non-training background row: {draw_id}")
            background_frames[draw_id] = frame

    combined_background = read_manifest(
        SAMPLE_ROOT / "sample_keys" / "shap_background_manifests.csv",
        require_unique_sample_id=False,
    )
    expected_combined = pd.concat(list(background_frames.values()), ignore_index=True)
    pd.testing.assert_frame_equal(combined_background.reset_index(drop=True), expected_combined.reset_index(drop=True), check_exact=True)

    explained_frames: dict[str, pd.DataFrame] = {}
    for draw in range(4):
        draw_id = f"EXPLAINED_E{draw}"
        frame = read_manifest(manifest_path(draw_id))
        require(len(frame) == EXPLAINED_N, f"Explained manifest size mismatch: {draw_id}")
        require(frame["draw_id"].eq(draw_id).all(), f"Explained draw id mismatch: {draw_id}")
        require(frame["selection_seed"].eq(configuration["explained_selection_seeds"][draw_id]).all(), f"Explained seed mismatch: {draw_id}")
        require(frame["split"].eq("test").all(), f"Non-test explained row: {draw_id}")
        require(set(frame["veg_class"].astype(str)) == set(VEGETATION_CLASSES), f"Missing vegetation class: {draw_id}")
        explained_frames[draw_id] = frame

    metadata_union = pd.concat(
        [frame[["point_id", "Lon", "Lat", "veg_class"]] for frame in explained_frames.values()],
        ignore_index=True,
    )
    for column in ("Lon", "Lat", "veg_class"):
        require(metadata_union.groupby("point_id", observed=True)[column].nunique().le(1).all(), f"Point metadata conflict: {column}")

    marker_dir = SHAP_ROOT / "shap_runs" / "completed"
    expected_marker_names = {f"shap_{spec.run_id}.complete.json" for spec in run_plan()}
    actual_marker_names = {path.name for path in marker_dir.glob("*.complete.json")}
    require(actual_marker_names == expected_marker_names, "Completion marker set mismatch")

    checkpoint_hashes = {int(seed): value for seed, value in validation["validation"]["checkpoint_hashes"].items()}
    required_npz_keys = {
        "shap_values_gC_m2_month",
        "background_sample_ids",
        "explained_sample_ids",
        "feature_names",
        "window",
        "model_seed",
    }
    output_hashes_verified = 0
    run_checks: list[dict[str, Any]] = []
    for spec in run_plan():
        paths = run_paths(spec)
        for key, path in paths.items():
            add_input(path, f"validated configured run artifact: {spec.run_id}/{key}")
            require(path.is_file(), f"Missing validated run artifact: {path}")

        marker = load_json(paths["completion_marker"])
        execution = load_json(paths["execution_json"])
        protocol = marker.get("protocol", {})
        background = background_frames[spec.background_draw]
        explained = explained_frames[spec.explained_draw]
        background_rel = f"sample_keys/background/shap_{spec.background_draw}.csv"
        explained_rel = f"sample_keys/shap_{spec.explained_draw.replace('EXPLAINED_', 'explained_', 1)}.csv"
        checks = {
            "status_complete": marker.get("status") == "COMPLETE",
            "run_id_match": marker.get("run_id") == spec.run_id,
            "shape_match": marker.get("shape") == [EXPLAINED_N, WINDOW, INPUT_DIM],
            "window_match": protocol.get("window") == WINDOW,
            "seed_match": protocol.get("model_seed") == spec.model_seed,
            "background_draw_match": protocol.get("background_draw") == spec.background_draw,
            "explained_draw_match": protocol.get("explained_draw") == spec.explained_draw,
            "checkpoint_hash_match": protocol.get("checkpoint_sha256") == checkpoint_hashes[spec.model_seed],
            "background_manifest_hash_match": protocol.get("background_manifest_sha256") == input_records[background_rel]["sha256"],
            "explained_manifest_hash_match": protocol.get("explained_manifest_sha256") == input_records[explained_rel]["sha256"],
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
            ),
        }

        output_artifacts = marker.get("output_artifacts", {})
        expected_output_keys = {"shap_npz", "variable_csv", "grouped_csv", "signed_csv", "execution_json"}
        checks["five_artifacts_recorded"] = set(output_artifacts) == expected_output_keys
        for key in sorted(expected_output_keys):
            record = output_artifacts.get(key, {})
            recorded = SHAP_ROOT / record.get("relative_path", "__MISSING__")
            require(recorded == paths[key], f"Run-output path mismatch: {spec.run_id}/{key}")
            require(recorded.stat().st_size == record.get("bytes"), f"Run-output byte-count mismatch: {recorded}")
            require(cached_hash(recorded) == record.get("sha256"), f"Run-output SHA mismatch: {recorded}")
            verify_main_inventory(recorded)
            output_hashes_verified += 1
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
            checks["npz_background_keys_match"] = np.array_equal(data["background_sample_ids"], background["sample_id"].to_numpy(np.int64))
            checks["npz_explained_keys_match"] = np.array_equal(data["explained_sample_ids"], explained["sample_id"].to_numpy(np.int64))
        failed = [name for name, passed in checks.items() if not passed]
        require(not failed, f"Run identity/consistency mismatch for {spec.run_id}: {failed}")
        run_checks.append(
            {
                "run_id": spec.run_id,
                "model_seed": spec.model_seed,
                "background_draw": spec.background_draw,
                "explained_draw": spec.explained_draw,
                "shape": [EXPLAINED_N, WINDOW, INPUT_DIM],
                "feature_order_match": True,
                "sample_id_match": True,
                "all_checks_pass": True,
            }
        )

    require(output_hashes_verified == 85, "Did not verify exactly 85 run-outputs")

    geoshap_validation_path = GEOSHAP_ROOT / "summary" / "w36_geoshap_validation.json"
    geoshap_record_path = GEOSHAP_ROOT / "validation" / "geoshap_input_validation.json"
    geoshap_input_inventory_path = GEOSHAP_ROOT / "validation" / "geoshap_input_inventory.csv"
    geoshap_provenance_path = GEOSHAP_ROOT / "validation" / "geoshap_provenance.md"
    geoshap_output_inventory_path = GEOSHAP_ROOT / "validation" / "geoshap_output_inventory.csv"
    geoshap_explained_stability_path = GEOSHAP_ROOT / "robustness" / "geoshap_explained_spatial_stability.csv"

    geoshap_output_inventory = pd.read_csv(geoshap_output_inventory_path, dtype=str, keep_default_na=False)
    require(list(geoshap_output_inventory.columns) == ["relative_path", "bytes", "sha256"], "GeoSHAP output inventory schema mismatch")
    require((not geoshap_output_inventory.empty) and geoshap_output_inventory["relative_path"].is_unique, "GeoSHAP output inventory is empty or contains duplicate paths")
    for row in geoshap_output_inventory.itertuples(index=False):
        path = GEOSHAP_ROOT / row.relative_path
        add_input(path, "input GeoSHAP validated output")
        require(path.stat().st_size == int(row.bytes), f"GeoSHAP output byte mismatch: {row.relative_path}")
        require(cached_hash(path) == row.sha256, f"GeoSHAP output SHA mismatch: {row.relative_path}")
    add_input(geoshap_output_inventory_path, "GeoSHAP output hash inventory")

    actual_geoshap_files = {
        path.relative_to(GEOSHAP_ROOT).as_posix()
        for path in GEOSHAP_ROOT.rglob("*")
        if path.is_file() and TEMPORAL_ROOT.resolve() not in path.resolve().parents
    }
    expected_geoshap_files = set(geoshap_output_inventory["relative_path"].astype(str)) | {"validation/geoshap_output_inventory.csv"}
    require(actual_geoshap_files == expected_geoshap_files, "GeoSHAP input file set differs from the recorded output inventory")

    geoshap_validation = load_json(geoshap_validation_path)
    geoshap_record = load_json(geoshap_record_path)
    for path, role in (
        (geoshap_validation_path, "GeoSHAP machine-readable validation"),
        (geoshap_record_path, "GeoSHAP validation identity record"),
        (geoshap_input_inventory_path, "GeoSHAP validated input inventory"),
        (geoshap_provenance_path, "GeoSHAP provenance record"),
        (geoshap_explained_stability_path, "GeoSHAP explained-sample spatial Spearman result"),
    ):
        add_input(path, role)
        require(geoshap_validation.get("ROBUST_SPATIAL_ATTRIBUTION") == "NO", "GeoSHAP spatial decision is not NO")
    require(geoshap_validation.get("ANALYSIS_STATUS") == "PASS", "GeoSHAP analysis status is not PASS")
    require(geoshap_record.get("status") == "PASS" and geoshap_record.get("statistics_computed") is False, "GeoSHAP input validation record mismatch")
    require(sha256_file(geoshap_record_path) == geoshap_validation["provenance"]["input_validation_sha256"], "GeoSHAP record hash mismatch")
    require(sha256_file(geoshap_input_inventory_path) == geoshap_validation["provenance"]["input_inventory_sha256"], "GeoSHAP input inventory hash mismatch")

    boundary = check_code_boundary()
    require(all(boundary.values()), f"Code boundary check failed: {boundary}")
    require(input_tree_digest() == tree_before, "Input SHAP tree changed during validation")
    require(geoshap_tree_digest() == geoshap_tree_before, "GeoSHAP tree changed during validation")

    inventory_rows = [
        {
            "relative_path": shap_rel(path),
            "bytes": path.stat().st_size,
            "sha256": cached_hash(path),
            "scientific_role": "; ".join(sorted(roles)),
        }
        for path, roles in sorted(input_roles.items(), key=lambda item: shap_rel(item[0]))
    ]
    input_inventory = pd.DataFrame(inventory_rows)
    input_path = write_csv("validation/vegetation_temporal_input_inventory.csv", input_inventory)
    record = {
        "project_id": PROJECT_ID,
        "work_package": ANALYSIS_NAME,
        "phase": "validate",
        "status": "PASS",
        "validated_at": now_iso(),
        "statistics_computed": False,
        "configured_runs_validated": len(run_checks),
        "output_artifact_hashes_verified": output_hashes_verified,
        "sample_manifests_validated": len(input_records),
        "geoshap_outputs_validated": len(expected_geoshap_files),
        "input_inventory_rows": len(input_inventory),
        "input_inventory_sha256": sha256_file(input_path),
        "input_shap_tree_before": tree_before,
        "input_shap_tree_after_validation": input_tree_digest(),
        "geoshap_tree_before": geoshap_tree_before,
        "geoshap_tree_after_validation": geoshap_tree_digest(),
        "code_boundary_checks": boundary,
        "run_checks": run_checks,
    }
    record_path = write_json("validation/vegetation_temporal_input_validation.json", record)
    return {
        "status": "PASS",
        "input_inventory": str(input_path),
        "validation_record": str(record_path),
        "input_inventory_rows": len(input_inventory),
        "configured_runs_validated": len(run_checks),
    }


@dataclass
class AggregatedRun:
    spec: RunSpec
    sample_joined: pd.DataFrame
    site_variable: pd.DataFrame
    site_group: pd.DataFrame


def load_input_validation() -> dict[str, Any]:
    record_path = VALIDATION_DIR / "vegetation_temporal_input_validation.json"
    inventory_path = VALIDATION_DIR / "vegetation_temporal_input_inventory.csv"
    record = load_json(record_path)
    require(record.get("status") == "PASS", "vegetation temporal stability input validation record is not PASS")
    require(record.get("statistics_computed") is False, "Input validation record unexpectedly reports statistics")
    require(inventory_path.is_file(), "vegetation temporal stability input inventory is missing")
    require(sha256_file(inventory_path) == record.get("input_inventory_sha256"), "vegetation temporal stability input inventory changed")
    require(input_tree_digest() == record.get("input_shap_tree_before"), "Input SHAP tree changed after validation")
    require(geoshap_tree_digest() == record.get("geoshap_tree_before"), "GeoSHAP outputs changed after validation")
    return record


def aggregate_run(spec: RunSpec) -> AggregatedRun:
    """Apply the fixed GeoSHAP sample -> site absolute-attribution aggregation."""

    manifest = read_manifest(manifest_path(spec.explained_draw))
    with np.load(run_paths(spec)["shap_npz"], allow_pickle=False) as data:
        values = np.asarray(data["shap_values_gC_m2_month"], dtype=np.float64)
        sample_ids = np.asarray(data["explained_sample_ids"], dtype=np.int64)
        feature_names = data["feature_names"].astype(str).tolist()
        window = int(np.asarray(data["window"]).reshape(-1)[0])
        model_seed = int(np.asarray(data["model_seed"]).reshape(-1)[0])

    require(values.shape == (EXPLAINED_N, WINDOW, INPUT_DIM), f"SHAP shape changed: {spec.run_id}")
    require(np.isfinite(values).all(), f"Non-finite SHAP values: {spec.run_id}")
    require(feature_names == FEATURES, f"Feature order changed: {spec.run_id}")
    require(window == WINDOW, f"Window changed: {spec.run_id}")
    require(model_seed == spec.model_seed, f"Model seed changed: {spec.run_id}")
    require(np.array_equal(sample_ids, manifest["sample_id"].to_numpy(np.int64)), f"Explained sample identity changed: {spec.run_id}")

    sample_abs = np.abs(values).sum(axis=1, dtype=np.float64)
    sample_frame = pd.DataFrame(sample_abs, columns=FEATURES)
    sample_frame.insert(0, "sample_id", sample_ids)
    joined = sample_frame.merge(
        manifest[["sample_id", "point_id", "Lon", "Lat", "Year", "Month", "veg_class"]],
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    require(len(joined) == EXPLAINED_N, f"Joined sample count mismatch: {spec.run_id}")
    require(joined["sample_id"].tolist() == sample_ids.tolist(), f"Join order changed: {spec.run_id}")
    require(not joined[["point_id", "Lon", "Lat", "Year", "Month", "veg_class"]].isna().any().any(), f"Unmatched sample identity: {spec.run_id}")
    for column in ("Lon", "Lat", "veg_class"):
        require(joined.groupby("point_id", observed=True)[column].nunique().le(1).all(), f"Within-point metadata conflict: {spec.run_id}/{column}")

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
    feature_means = joined.groupby("point_id", sort=True, observed=True)[FEATURES].mean().reset_index()
    site_variable = metadata.merge(feature_means, on="point_id", how="inner", validate="one_to_one")
    require(int(site_variable["n_explained_samples"].sum()) == EXPLAINED_N, f"Site counts do not sum to 1000: {spec.run_id}")
    require(set(site_variable["veg_class"].astype(str)) == set(VEGETATION_CLASSES), f"Run lacks a vegetation class: {spec.run_id}")
    require(np.isfinite(site_variable[FEATURES].to_numpy()).all(), f"Non-finite site attribution: {spec.run_id}")

    site_group = site_variable[["point_id", "Lon", "Lat", "veg_class", "n_explained_samples"]].copy()
    for group, members in GROUPS.items():
        site_group[group] = site_variable[members].sum(axis=1)
    require(np.isfinite(site_group[list(GROUPS)].to_numpy()).all(), f"Non-finite grouped attribution: {spec.run_id}")
    require((site_group[ENV_GROUPS].sum(axis=1) > 0).all(), f"Zero environmental attribution denominator: {spec.run_id}")
    return AggregatedRun(spec, joined, site_variable, site_group)


def run_label(run_id: str) -> str:
    mapping = {
        "background_BG256_D1": "E0",
        "explained_E1": "E1",
        "explained_E2": "E2",
        "explained_E3": "E3",
        "seed_2024": "seed2024",
        "seed_3407": "seed3407",
    }
    return mapping.get(run_id, run_id.replace("background_", ""))


def vegetation_summary(aggregated: AggregatedRun, analysis_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for veg_class in VEGETATION_CLASSES:
        sites = aggregated.site_group.loc[aggregated.site_group["veg_class"].astype(str) == veg_class].copy()
        require(len(sites) > 0, f"No sites for vegetation class: {aggregated.spec.run_id}/{veg_class}")
        means = sites[ENV_GROUPS].mean(axis=0)
        denominator = float(means.sum())
        require(denominator > 0, f"Zero class attribution denominator: {aggregated.spec.run_id}/{veg_class}")
        shares = means / denominator
        ranks = shares.rank(method="min", ascending=False).astype(int)
        distinct_shares = sorted({float(shares[group]) for group in ENV_GROUPS}, reverse=True)
        require(len(distinct_shares) >= 2, f"Fewer than two distinct group shares: {aggregated.spec.run_id}/{veg_class}")
        top1_share, top2_share = distinct_shares[:2]
        top1_ties = [group for group in ENV_GROUPS if float(shares[group]) == top1_share]
        top2_ties = [group for group in ENV_GROUPS if float(shares[group]) == top2_share]
        top1_label = ";".join(top1_ties)
        top2_label = ";".join(top2_ties)
        for group in ENV_GROUPS:
            rows.append(
                {
                    "analysis_label": analysis_label,
                    "run_id": aggregated.spec.run_id,
                    "model_seed": aggregated.spec.model_seed,
                    "background_draw": aggregated.spec.background_draw,
                    "explained_draw": aggregated.spec.explained_draw,
                    "vegetation_class": veg_class,
                    "n_sites": len(sites),
                    "n_explained_samples": int(sites["n_explained_samples"].sum()),
                    "environmental_group": group,
                    "mean_abs_shap_gC_m2_month": float(means[group]),
                    "group_share": float(shares[group]),
                    "rank": int(ranks[group]),
                    "top1_environmental_group": top1_label,
                    "top1_share": top1_share,
                    "top1_tie_n": len(top1_ties),
                    "top2_environmental_group": top2_label,
                    "top2_share": top2_share,
                    "top2_tie_n": len(top2_ties),
                    "top1_minus_top2_margin": top1_share - top2_share,
                }
            )
    frame = pd.DataFrame(rows)
    require(len(frame) == len(VEGETATION_CLASSES) * len(ENV_GROUPS), f"Vegetation summary row count mismatch: {aggregated.spec.run_id}")
    share_sums = frame.groupby("vegetation_class", observed=True)["group_share"].sum()
    require(np.allclose(share_sums.to_numpy(), 1.0, rtol=0.0, atol=1e-12), f"Vegetation shares do not sum to one: {aggregated.spec.run_id}")
    return frame


def summary_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    matrix = summary.pivot(index="vegetation_class", columns="environmental_group", values="group_share")
    return matrix.loc[VEGETATION_CLASSES, ENV_GROUPS]


def top_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "vegetation_class",
        "n_sites",
        "n_explained_samples",
        "top1_environmental_group",
        "top1_share",
        "top2_environmental_group",
        "top2_share",
        "top1_minus_top2_margin",
    ]
    table = summary.drop_duplicates("vegetation_class").set_index("vegetation_class").loc[VEGETATION_CLASSES].reset_index()
    return table[columns]


def compare_vegetation_summaries(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    family: str,
    comparison_id: str,
    reference_label: str,
    comparison_label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    left = summary_matrix(reference)
    right = summary_matrix(comparison)
    x = left.to_numpy(dtype=float).reshape(-1)
    y = right.to_numpy(dtype=float).reshape(-1)
    spearman = float(spearmanr(x, y).statistic)
    kendall = float(kendalltau(x, y).statistic)
    pearson = float(pearsonr(x, y).statistic)
    require(np.isfinite([spearman, kendall, pearson]).all(), f"Non-finite composition correlation: {comparison_id}")

    left_top = top_table(reference).set_index("vegetation_class")
    right_top = top_table(comparison).set_index("vegetation_class")
    same = left_top["top1_environmental_group"].astype(str) == right_top["top1_environmental_group"].astype(str)
    overall = {
        "family": family,
        "comparison_id": comparison_id,
        "reference_label": reference_label,
        "comparison_label": comparison_label,
        "n_vegetation_classes": len(VEGETATION_CLASSES),
        "n_environmental_groups": len(ENV_GROUPS),
        "n_flattened_share_cells": len(x),
        "spearman_rho_30cell": spearman,
        "kendall_tau_30cell": kendall,
        "pearson_r_30cell": pearson,
        "n_same_top1": int(same.sum()),
        "top1_agreement_fraction": float(same.mean()),
    }

    class_rows: list[dict[str, Any]] = []
    for veg_class in VEGETATION_CLASSES:
        delta = right.loc[veg_class].to_numpy(dtype=float) - left.loc[veg_class].to_numpy(dtype=float)
        left_row = left_top.loc[veg_class]
        right_row = right_top.loc[veg_class]
        class_rows.append(
            {
                **overall,
                "vegetation_class": veg_class,
                "n_sites_reference": int(left_row["n_sites"]),
                "n_sites_comparison": int(right_row["n_sites"]),
                "n_explained_samples_reference": int(left_row["n_explained_samples"]),
                "n_explained_samples_comparison": int(right_row["n_explained_samples"]),
                "reference_top1_group": str(left_row["top1_environmental_group"]),
                "comparison_top1_group": str(right_row["top1_environmental_group"]),
                "same_top1": bool(same.loc[veg_class]),
                "reference_top1_share": float(left_row["top1_share"]),
                "comparison_top1_share": float(right_row["top1_share"]),
                "reference_top2_group": str(left_row["top2_environmental_group"]),
                "comparison_top2_group": str(right_row["top2_environmental_group"]),
                "reference_top2_share": float(left_row["top2_share"]),
                "comparison_top2_share": float(right_row["top2_share"]),
                "reference_top1_minus_top2_margin": float(left_row["top1_minus_top2_margin"]),
                "comparison_top1_minus_top2_margin": float(right_row["top1_minus_top2_margin"]),
                "top1_minus_top2_margin_change": float(right_row["top1_minus_top2_margin"] - left_row["top1_minus_top2_margin"]),
                "absolute_margin_change": abs(float(right_row["top1_minus_top2_margin"] - left_row["top1_minus_top2_margin"])),
                "mean_absolute_share_difference": float(np.abs(delta).mean()),
                "maximum_absolute_share_difference": float(np.abs(delta).max()),
                "l1_share_distance": float(np.abs(delta).sum()),
            }
        )
    return overall, pd.DataFrame(class_rows)


def build_temporal_composition(manifests: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    years = sorted({int(year) for frame in manifests.values() for year in frame["Year"].unique()})
    rows: list[dict[str, Any]] = []
    for label in ("E0", "E1", "E2", "E3"):
        frame = manifests[f"EXPLAINED_{label}"]
        site_counts = frame.groupby("point_id", observed=True).size()
        year_month = frame["Year"].astype(str) + "-" + frame["Month"].astype(str).str.zfill(2)
        row: dict[str, Any] = {
            "explained_set": label,
            "draw_id": f"EXPLAINED_{label}",
            "n_samples": len(frame),
            "n_unique_point_id": int(frame["point_id"].nunique()),
            "n_unique_year_month": int(year_month.nunique()),
            "mean_samples_per_site": float(site_counts.mean()),
            "median_samples_per_site": float(site_counts.median()),
            "min_samples_per_site": int(site_counts.min()),
            "max_samples_per_site": int(site_counts.max()),
        }
        for year in years:
            row[f"year_{year}_count"] = int((frame["Year"] == year).sum())
        for month in range(1, 13):
            row[f"month_{month:02d}_count"] = int((frame["Month"] == month).sum())
        for veg_class in VEGETATION_CLASSES:
            row[f"{veg_class}_count"] = int((frame["veg_class"].astype(str) == veg_class).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_overlap(manifests: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    years = sorted({int(year) for frame in manifests.values() for year in frame["Year"].unique()})
    pairs = [("E0", "E1"), ("E0", "E2"), ("E0", "E3"), ("E1", "E2"), ("E1", "E3"), ("E2", "E3")]
    rows: list[dict[str, Any]] = []
    for left_label, right_label in pairs:
        left = manifests[f"EXPLAINED_{left_label}"]
        right = manifests[f"EXPLAINED_{right_label}"]
        left_samples = set(left["sample_id"].astype(int))
        right_samples = set(right["sample_id"].astype(int))
        shared_samples = left_samples & right_samples
        left_pym = set(zip(left["point_id"].astype(str), left["Year"].astype(int), left["Month"].astype(int)))
        right_pym = set(zip(right["point_id"].astype(str), right["Year"].astype(int), right["Month"].astype(int)))
        shared_pym = left_pym & right_pym
        left_month_counts = np.array([(left["Month"] == month).sum() for month in range(1, 13)], dtype=float)
        right_month_counts = np.array([(right["Month"] == month).sum() for month in range(1, 13)], dtype=float)
        tvd = 0.5 * float(np.abs(left_month_counts / len(left) - right_month_counts / len(right)).sum())
        row: dict[str, Any] = {
            "comparison_id": f"{left_label}_vs_{right_label}",
            "set_A": left_label,
            "set_B": right_label,
            "n_samples_A": len(left_samples),
            "n_samples_B": len(right_samples),
            "exact_sample_id_n_shared": len(shared_samples),
            "exact_sample_id_proportion_of_A": len(shared_samples) / len(left_samples),
            "exact_sample_id_proportion_of_B": len(shared_samples) / len(right_samples),
            "n_unique_point_year_month_A": len(left_pym),
            "n_unique_point_year_month_B": len(right_pym),
            "exact_point_year_month_n_shared": len(shared_pym),
            "exact_point_year_month_proportion_of_A": len(shared_pym) / len(left_pym),
            "exact_point_year_month_proportion_of_B": len(shared_pym) / len(right_pym),
            "month_of_year_total_variation_distance": tvd,
        }
        for month in range(1, 13):
            count_a = int(left_month_counts[month - 1])
            count_b = int(right_month_counts[month - 1])
            row[f"month_{month:02d}_count_A"] = count_a
            row[f"month_{month:02d}_count_B"] = count_b
            row[f"month_{month:02d}_difference_B_minus_A"] = count_b - count_a
        for year in years:
            count_a = int((left["Year"] == year).sum())
            count_b = int((right["Year"] == year).sum())
            row[f"year_{year}_count_A"] = count_a
            row[f"year_{year}_count_B"] = count_b
            row[f"year_{year}_difference_B_minus_A"] = count_b - count_a
        rows.append(row)
    return pd.DataFrame(rows)


def build_within_site_overlap(
    manifests: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    site_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for label in ("E0", "E1", "E2", "E3"):
        frame = manifests[f"EXPLAINED_{label}"]
        site_sets[label] = {}
        for point_id, subset in frame.groupby("point_id", sort=True, observed=True):
            ym_set = {(int(row.Year), int(row.Month)) for row in subset.itertuples(index=False)}
            month_set = {month for _, month in ym_set}
            year_set = {year for year, _ in ym_set}
            veg_values = subset["veg_class"].astype(str).unique().tolist()
            require(len(veg_values) == 1, f"Within-site vegetation conflict: {label}/{point_id}")
            site_sets[label][str(point_id)] = {
                "year_month": ym_set,
                "month": month_set,
                "year": year_set,
                "veg_class": veg_values[0],
                "n_samples": len(subset),
            }
            detail_rows.append(
                {
                    "record_type": "ESET_SITE_COMPOSITION",
                    "explained_set": label,
                    "set_A": "",
                    "set_B": "",
                    "point_id": str(point_id),
                    "vegetation_class": veg_values[0],
                    "n_samples": len(subset),
                    "n_unique_months_of_year": len(month_set),
                    "n_unique_years": len(year_set),
                    "n_unique_year_month": len(ym_set),
                    "year_month_keys": ";".join(f"{year:04d}-{month:02d}" for year, month in sorted(ym_set)),
                    "n_samples_A": "",
                    "n_samples_B": "",
                    "n_year_month_A": "",
                    "n_year_month_B": "",
                    "exact_year_month_overlap_count": "",
                    "year_month_jaccard": "",
                    "n_months_of_year_A": "",
                    "n_months_of_year_B": "",
                    "month_of_year_overlap_count": "",
                    "month_of_year_jaccard": "",
                }
            )

    pairs = [("E0", "E1"), ("E0", "E2"), ("E0", "E3"), ("E1", "E2"), ("E1", "E3"), ("E2", "E3")]
    summary_rows: list[dict[str, Any]] = []
    for left_label, right_label in pairs:
        matched = sorted(set(site_sets[left_label]) & set(site_sets[right_label]))
        require(len(matched) > 0, f"No matched point_id: {left_label}/{right_label}")
        pair_metrics: list[dict[str, Any]] = []
        for point_id in matched:
            left = site_sets[left_label][point_id]
            right = site_sets[right_label][point_id]
            require(left["veg_class"] == right["veg_class"], f"Cross-set vegetation conflict: {point_id}")
            ym_intersection = left["year_month"] & right["year_month"]
            ym_union = left["year_month"] | right["year_month"]
            month_intersection = left["month"] & right["month"]
            month_union = left["month"] | right["month"]
            metrics = {
                "exact_year_month_overlap_count": len(ym_intersection),
                "year_month_jaccard": len(ym_intersection) / len(ym_union),
                "month_of_year_overlap_count": len(month_intersection),
                "month_of_year_jaccard": len(month_intersection) / len(month_union),
            }
            pair_metrics.append(metrics)
            detail_rows.append(
                {
                    "record_type": "PAIRWISE_SITE_OVERLAP",
                    "explained_set": "",
                    "set_A": left_label,
                    "set_B": right_label,
                    "point_id": point_id,
                    "vegetation_class": left["veg_class"],
                    "n_samples": "",
                    "n_unique_months_of_year": "",
                    "n_unique_years": "",
                    "n_unique_year_month": "",
                    "year_month_keys": "",
                    "n_samples_A": left["n_samples"],
                    "n_samples_B": right["n_samples"],
                    "n_year_month_A": len(left["year_month"]),
                    "n_year_month_B": len(right["year_month"]),
                    "exact_year_month_overlap_count": metrics["exact_year_month_overlap_count"],
                    "year_month_jaccard": metrics["year_month_jaccard"],
                    "n_months_of_year_A": len(left["month"]),
                    "n_months_of_year_B": len(right["month"]),
                    "month_of_year_overlap_count": metrics["month_of_year_overlap_count"],
                    "month_of_year_jaccard": metrics["month_of_year_jaccard"],
                }
            )

        metric_frame = pd.DataFrame(pair_metrics)
        summary: dict[str, Any] = {
            "comparison_id": f"{left_label}_vs_{right_label}",
            "set_A": left_label,
            "set_B": right_label,
            "matched_site_n": len(matched),
        }
        for column in (
            "exact_year_month_overlap_count",
            "year_month_jaccard",
            "month_of_year_overlap_count",
            "month_of_year_jaccard",
        ):
            values = metric_frame[column].to_numpy(dtype=float)
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            summary[f"{column}_median"] = float(median)
            summary[f"{column}_q1"] = float(q1)
            summary[f"{column}_q3"] = float(q3)
            summary[f"{column}_iqr"] = float(q3 - q1)
            summary[f"{column}_min"] = float(values.min())
            summary[f"{column}_max"] = float(values.max())
        exact = metric_frame["exact_year_month_overlap_count"].to_numpy(dtype=float)
        summary["fraction_sites_zero_exact_year_month_overlap"] = float((exact == 0).mean())
        summary["fraction_sites_le_1_exact_shared_year_month"] = float((exact <= 1).mean())
        summary["fraction_sites_ge_2_exact_shared_year_month"] = float((exact >= 2).mean())
        summary_rows.append(summary)
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def bootstrap_group_shares(aggregated: AggregatedRun, analysis_label: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for veg_class in VEGETATION_CLASSES:
        sites = aggregated.site_group.loc[aggregated.site_group["veg_class"].astype(str) == veg_class]
        values = sites[ENV_GROUPS].to_numpy(dtype=np.float64)
        n_sites = len(values)
        require(n_sites > 0, f"No sites for bootstrap: {analysis_label}/{veg_class}")
        indices = rng.integers(0, n_sites, size=(BOOTSTRAP_B, n_sites), endpoint=False)
        bootstrap_means = values[indices].mean(axis=1)
        denominators = bootstrap_means.sum(axis=1, keepdims=True)
        require((denominators > 0).all(), f"Zero bootstrap denominator: {analysis_label}/{veg_class}")
        bootstrap_shares = bootstrap_means / denominators
        point_means = values.mean(axis=0)
        point_shares = point_means / point_means.sum()
        lower = np.percentile(bootstrap_shares, 2.5, axis=0)
        upper = np.percentile(bootstrap_shares, 97.5, axis=0)
        for index, group in enumerate(ENV_GROUPS):
            rows.append(
                {
                    "explained_set": analysis_label,
                    "run_id": aggregated.spec.run_id,
                    "vegetation_class": veg_class,
                    "n_sites": n_sites,
                    "n_explained_samples": int(sites["n_explained_samples"].sum()),
                    "environmental_group": group,
                    "mean_group_share": float(point_shares[index]),
                    "lower95": float(lower[index]),
                    "upper95": float(upper[index]),
                    "bootstrap_unit": "point_id within vegetation class",
                    "B": BOOTSTRAP_B,
                    "seed": BOOTSTRAP_SEED,
                    "interval": "two-sided 95% percentile",
                }
            )
    return pd.DataFrame(rows)


def build_nonseasonal_outputs(explained_runs: Mapping[str, AggregatedRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    proportions: dict[tuple[str, str, str], float] = {}
    top_predictors: dict[tuple[str, str], str] = {}
    solar_feature = "surface_solar_radiation_downwards_sum"
    for label in ("E0", "E1", "E2", "E3"):
        aggregated = explained_runs[label]
        values = aggregated.site_variable[NONSEASONAL_FEATURES].to_numpy(dtype=float)
        ties = np.isclose(values, values.max(axis=1, keepdims=True), rtol=0.0, atol=0.0).sum(axis=1)
        require((ties == 1).all(), f"Tied dominant nonseasonal predictor: {label}")
        indices = values.argmax(axis=1)
        site = aggregated.site_variable[["point_id", "veg_class"]].copy()
        site["dominant_predictor"] = [NONSEASONAL_FEATURES[index] for index in indices]
        for veg_class in VEGETATION_CLASSES:
            subset = site.loc[site["veg_class"].astype(str) == veg_class]
            n_sites = len(subset)
            counts = subset["dominant_predictor"].value_counts().sort_index()
            max_count = int(counts.max())
            top = sorted(counts[counts == max_count].index.astype(str).tolist())
            top_predictors[(label, veg_class)] = ";".join(top)
            for predictor, count in counts.items():
                detail_rows.append(
                    {
                        "explained_set": label,
                        "run_id": aggregated.spec.run_id,
                        "vegetation_class": veg_class,
                        "n_sites": n_sites,
                        "record_type": "predictor_detail",
                        "dominant_predictor_or_category": str(predictor),
                        "n_sites_dominant": int(count),
                        "proportion": float(count / n_sites),
                    }
                )
            categories = {
                "Solar radiation": int((subset["dominant_predictor"] == solar_feature).sum()),
                "NDVI": int((subset["dominant_predictor"] == "NDVI").sum()),
                "Other predictor": int((~subset["dominant_predictor"].isin([solar_feature, "NDVI"])).sum()),
            }
            require(sum(categories.values()) == n_sites, f"Predictor category counts do not sum: {label}/{veg_class}")
            for category, count in categories.items():
                proportion = count / n_sites
                proportions[(label, veg_class, category)] = proportion
                detail_rows.append(
                    {
                        "explained_set": label,
                        "run_id": aggregated.spec.run_id,
                        "vegetation_class": veg_class,
                        "n_sites": n_sites,
                        "record_type": "continuity_category",
                        "dominant_predictor_or_category": category,
                        "n_sites_dominant": count,
                        "proportion": proportion,
                    }
                )

    stability_rows: list[dict[str, Any]] = []
    for veg_class in VEGETATION_CLASSES:
        solar = [proportions[(label, veg_class, "Solar radiation")] for label in ("E0", "E1", "E2", "E3")]
        ndvi = [proportions[(label, veg_class, "NDVI")] for label in ("E0", "E1", "E2", "E3")]
        top = [top_predictors[(label, veg_class)] for label in ("E0", "E1", "E2", "E3")]
        stability_rows.append(
            {
                "vegetation_class": veg_class,
                "n_sites_E0": int((explained_runs["E0"].site_variable["veg_class"].astype(str) == veg_class).sum()),
                "n_sites_E1": int((explained_runs["E1"].site_variable["veg_class"].astype(str) == veg_class).sum()),
                "n_sites_E2": int((explained_runs["E2"].site_variable["veg_class"].astype(str) == veg_class).sum()),
                "n_sites_E3": int((explained_runs["E3"].site_variable["veg_class"].astype(str) == veg_class).sum()),
                "solar_proportion_min": min(solar),
                "solar_proportion_max": max(solar),
                "solar_proportion_range": max(solar) - min(solar),
                "NDVI_proportion_min": min(ndvi),
                "NDVI_proportion_max": max(ndvi),
                "NDVI_proportion_range": max(ndvi) - min(ndvi),
                "top_dominant_predictor_E0": top[0],
                "top_dominant_predictor_E1": top[1],
                "top_dominant_predictor_E2": top[2],
                "top_dominant_predictor_E3": top[3],
                "top_dominant_predictor_consistent": len(set(top)) == 1,
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(stability_rows)


def save_figure(fig: plt.Figure, relative_path: str) -> Path:
    path = safe_output(relative_path)
    require(not path.exists(), f"Refusing to overwrite figure: {path}")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    require(path.is_file() and path.stat().st_size > 10_000, f"Figure output failed: {path}")
    return path


def make_figures(
    explained_summaries: Mapping[str, pd.DataFrame],
    explained_overall: pd.DataFrame,
    nonseasonal_detail: pd.DataFrame,
    within_detail: pd.DataFrame,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 13,
            "legend.fontsize": 8,
        }
    )
    outputs: list[Path] = []
    group_short = {
        "Energy": "Energy",
        "Vegetation productivity": "Vegetation\nproductivity",
        "Thermal": "Thermal",
        "Hydrologic": "Hydrologic",
        "Terrain": "Terrain",
    }

    matrices = {label: summary_matrix(explained_summaries[label]) for label in ("E0", "E1", "E2", "E3")}
    vmax = max(float(matrix.to_numpy().max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    image = None
    for axis, label in zip(axes.flat, ("E0", "E1", "E2", "E3")):
        values = matrices[label].to_numpy(dtype=float)
        image = axis.imshow(values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(ENV_GROUPS)), [group_short[group] for group in ENV_GROUPS], rotation=25, ha="right")
        axis.set_yticks(range(len(VEGETATION_CLASSES)), VEGETATION_CLASSES)
        axis.set_title(f"{label}: vegetation-conditioned model-attribution share")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                color = "white" if values[row, col] > vmax * 0.55 else "black"
                axis.text(col, row, f"{values[row, col]:.3f}", ha="center", va="center", color=color, fontsize=8)
    require(image is not None, "Heatmap image was not created")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Environmental-group share of mean |SHAP|")
    fig.suptitle("W36 E0-E3 vegetation-conditioned model-attribution composition")
    outputs.append(save_figure(fig, FIGURE_FILES[0]))

    colors = dict(zip(ENV_GROUPS, plt.get_cmap("tab10").colors[: len(ENV_GROUPS)]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    for axis, veg_class in zip(axes.flat, VEGETATION_CLASSES):
        for group in ENV_GROUPS:
            shares = [float(matrices[label].loc[veg_class, group]) for label in ("E0", "E1", "E2", "E3")]
            axis.plot(("E0", "E1", "E2", "E3"), shares, marker="o", linewidth=1.6, markersize=4, label=group, color=colors[group])
        axis.set_title(veg_class)
        axis.set_ylabel("Attribution share")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E0-E3 environmental-group model-attribution shares by vegetation class")
    outputs.append(save_figure(fig, FIGURE_FILES[1]))

    all_values = np.concatenate([matrix.to_numpy().reshape(-1) for matrix in matrices.values()])
    minimum = max(0.0, float(all_values.min()) - 0.02)
    maximum = float(all_values.max()) + 0.02
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    comparison_labels = ("E1", "E2", "E3")
    for axis, label in zip(axes, comparison_labels):
        for group in ENV_GROUPS:
            x = matrices["E0"][group].to_numpy(dtype=float)
            y = matrices[label][group].to_numpy(dtype=float)
            axis.scatter(x, y, s=32, alpha=0.85, color=colors[group], label=group, edgecolors="none")
        axis.plot([minimum, maximum], [minimum, maximum], color="0.35", linewidth=1, linestyle="--")
        metric = explained_overall.loc[explained_overall["comparison_label"] == label].iloc[0]
        axis.set_title(f"E0 vs {label}: 30 cells\nSpearman={metric['spearman_rho_30cell']:.3f}")
        axis.set_xlabel("E0 environmental-group share")
        axis.set_ylabel(f"{label} environmental-group share")
        axis.set_xlim(minimum, maximum)
        axis.set_ylim(minimum, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("E0 versus E1-E3 vegetation-conditioned model-attribution composition")
    outputs.append(save_figure(fig, FIGURE_FILES[2]))

    continuity = nonseasonal_detail.loc[nonseasonal_detail["record_type"] == "continuity_category"].copy()
    categories = ["Solar radiation", "NDVI", "Other predictor"]
    category_colors = {
        "Solar radiation": "#E69F00",
        "NDVI": "#009E73",
        "Other predictor": "#7A7A7A",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, veg_class in zip(axes.flat, VEGETATION_CLASSES):
        subset = continuity.loc[continuity["vegetation_class"] == veg_class]
        bottom = np.zeros(4, dtype=float)
        for category in categories:
            values = [
                float(subset.loc[(subset["explained_set"] == label) & (subset["dominant_predictor_or_category"] == category), "proportion"].iloc[0])
                for label in ("E0", "E1", "E2", "E3")
            ]
            axis.bar(("E0", "E1", "E2", "E3"), values, bottom=bottom, label=category, color=category_colors[category], width=0.72)
            bottom += np.asarray(values)
        axis.set_title(veg_class)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Proportion of sites")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E0-E3 nonseasonal dominant-predictor continuity composition")
    outputs.append(save_figure(fig, FIGURE_FILES[3]))

    pairwise = within_detail.loc[within_detail["record_type"] == "PAIRWISE_SITE_OVERLAP"].copy()
    pairwise["comparison_id"] = pairwise["set_A"].astype(str) + " vs " + pairwise["set_B"].astype(str)
    pair_order = ["E0 vs E1", "E0 vs E2", "E0 vs E3", "E1 vs E2", "E1 vs E3", "E2 vs E3"]
    panels = [
        ("exact_year_month_overlap_count", "Exact Year-Month overlap count"),
        ("year_month_jaccard", "Year-Month Jaccard"),
        ("month_of_year_overlap_count", "Month-of-year overlap count"),
        ("month_of_year_jaccard", "Month-of-year Jaccard"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, panels):
        arrays = [pd.to_numeric(pairwise.loc[pairwise["comparison_id"] == pair, column]).to_numpy(dtype=float) for pair in pair_order]
        axis.boxplot(arrays, labels=pair_order, showfliers=False, patch_artist=True, boxprops={"facecolor": "#56B4E9", "alpha": 0.65})
        axis.set_title(title)
        axis.set_ylabel("Count" if column.endswith("count") else "Jaccard")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("E0-E3 within-site temporal sampling overlap diagnostics")
    outputs.append(save_figure(fig, FIGURE_FILES[4]))
    require(len(outputs) == 5, "Did not create exactly five diagnostic figures")
    return outputs


def run_postprocessing() -> dict[str, Any]:
    validation = load_input_validation()
    for directory in (RESULTS_DIR, FIGURES_DIR, SUMMARY_DIR):
        require(not any(directory.iterdir()), f"Output directory is not empty before run: {directory}")
    allowed_validation = {"vegetation_temporal_input_inventory.csv", "vegetation_temporal_input_validation.json"}
    require({path.name for path in VALIDATION_DIR.iterdir() if path.is_file()} == allowed_validation, "Unexpected validation file before run")

    aggregated = {spec.run_id: aggregate_run(spec) for spec in run_plan()}
    explained_runs = {
        "E0": aggregated["background_BG256_D1"],
        "E1": aggregated["explained_E1"],
        "E2": aggregated["explained_E2"],
        "E3": aggregated["explained_E3"],
    }
    seed_runs = {
        "seed42": aggregated["background_BG256_D1"],
        "seed2024": aggregated["seed_2024"],
        "seed3407": aggregated["seed_3407"],
    }

    all_summaries = {run_id: vegetation_summary(run, run_label(run_id)) for run_id, run in aggregated.items()}
    explained_summaries = {
        label: vegetation_summary(run, label) for label, run in explained_runs.items()
    }
    explained_summary_output = pd.concat(
        [explained_summaries[label] for label in ("E0", "E1", "E2", "E3")],
        ignore_index=True,
    )

    explained_overall_rows: list[dict[str, Any]] = []
    explained_class_frames: list[pd.DataFrame] = []
    for label in ("E1", "E2", "E3"):
        overall, class_frame = compare_vegetation_summaries(
            explained_summaries["E0"],
            explained_summaries[label],
            family="explained",
            comparison_id=f"E0_vs_{label}",
            reference_label="E0",
            comparison_label=label,
        )
        explained_overall_rows.append(overall)
        explained_class_frames.append(class_frame)
    explained_overall = pd.DataFrame(explained_overall_rows)
    explained_class = pd.concat(explained_class_frames, ignore_index=True)

    seed_summary_map = {
        label: vegetation_summary(run, label) for label, run in seed_runs.items()
    }
    seed_pairs = [
        ("seed42", "seed2024", "42_vs_2024"),
        ("seed42", "seed3407", "42_vs_3407"),
        ("seed2024", "seed3407", "2024_vs_3407"),
    ]
    seed_frames: list[pd.DataFrame] = []
    for left, right, comparison_id in seed_pairs:
        _, class_frame = compare_vegetation_summaries(
            seed_summary_map[left],
            seed_summary_map[right],
            family="seed",
            comparison_id=comparison_id,
            reference_label=left,
            comparison_label=right,
        )
        seed_frames.append(class_frame)
    seed_output = pd.concat(seed_frames, ignore_index=True)

    canonical_summary = all_summaries["background_BG256_D1"]
    background_specs = [
        spec for spec in run_plan() if spec.run_id.startswith("background_") and spec.run_id != "background_BG256_D1"
    ]
    background_frames: list[pd.DataFrame] = []
    background_overall_rows: list[dict[str, Any]] = []
    for spec in background_specs:
        overall, class_frame = compare_vegetation_summaries(
            canonical_summary,
            all_summaries[spec.run_id],
            family="background",
            comparison_id=f"BG256_D1_vs_{spec.background_draw}",
            reference_label="BG256_D1",
            comparison_label=spec.background_draw,
        )
        background_overall_rows.append(overall)
        background_frames.append(class_frame)
    background_output = pd.concat(background_frames, ignore_index=True)
    background_overall = pd.DataFrame(background_overall_rows)
    background_summary_row: dict[str, Any] = {
        "family": "background",
        "n_comparisons": len(background_overall),
    }
    for source_column, prefix in (
        ("spearman_rho_30cell", "spearman_rho_30cell"),
        ("kendall_tau_30cell", "kendall_tau_30cell"),
        ("pearson_r_30cell", "pearson_r_30cell"),
        ("top1_agreement_fraction", "top1_agreement_fraction"),
    ):
        values = background_overall[source_column].to_numpy(dtype=float)
        background_summary_row[f"{prefix}_min"] = float(values.min())
        background_summary_row[f"{prefix}_median"] = float(np.median(values))
        background_summary_row[f"{prefix}_max"] = float(values.max())
    comparison_l1 = background_output.groupby("comparison_id", observed=True)["l1_share_distance"].mean()
    background_summary_row["mean_class_l1_distance_min"] = float(comparison_l1.min())
    background_summary_row["mean_class_l1_distance_median"] = float(comparison_l1.median())
    background_summary_row["mean_class_l1_distance_max"] = float(comparison_l1.max())
    background_summary = pd.DataFrame([background_summary_row])

    manifests = {f"EXPLAINED_E{draw}": read_manifest(manifest_path(f"EXPLAINED_E{draw}")) for draw in range(4)}
    temporal_composition = build_temporal_composition(manifests)
    pairwise_overlap = build_pairwise_overlap(manifests)
    within_detail, within_summary = build_within_site_overlap(manifests)
    nonseasonal_detail, nonseasonal_stability = build_nonseasonal_outputs(explained_runs)

    bootstrap_frames: list[pd.DataFrame] = []
    bootstrap_repeat_checks: dict[str, bool] = {}
    for label in ("E0", "E1", "E2", "E3"):
        first = bootstrap_group_shares(explained_runs[label], label)
        second = bootstrap_group_shares(explained_runs[label], label)
        repeat = first.equals(second)
        require(repeat, f"Fixed-seed bootstrap repeat check failed: {label}")
        bootstrap_repeat_checks[label] = repeat
        bootstrap_frames.append(first)
    bootstrap_output = pd.concat(bootstrap_frames, ignore_index=True)

    output_frames = {
        RESULT_FILES[0]: explained_summary_output,
        RESULT_FILES[1]: explained_overall,
        RESULT_FILES[2]: explained_class,
        RESULT_FILES[3]: seed_output,
        RESULT_FILES[4]: background_output,
        RESULT_FILES[5]: background_summary,
        RESULT_FILES[6]: nonseasonal_detail,
        RESULT_FILES[7]: nonseasonal_stability,
        RESULT_FILES[8]: temporal_composition,
        RESULT_FILES[9]: pairwise_overlap,
        RESULT_FILES[10]: within_detail,
        RESULT_FILES[11]: within_summary,
        RESULT_FILES[12]: bootstrap_output,
    }
    for relative_path, frame in output_frames.items():
        require(not frame.empty, f"Empty required result: {relative_path}")
        write_csv(relative_path, frame)

    figure_paths = make_figures(explained_summaries, explained_overall, nonseasonal_detail, within_detail)
    require(input_tree_digest() == validation["input_shap_tree_before"], "Input SHAP tree changed during post-processing")
    require(geoshap_tree_digest() == validation["geoshap_tree_before"], "GeoSHAP outputs changed during post-processing")

    record = {
        "project_id": PROJECT_ID,
        "work_package": ANALYSIS_NAME,
        "phase": "run",
        "status": "PASS",
        "completed_at": now_iso(),
        "analysis_scope": "vegetation-temporal summaries from W36 sample-level SHAP artifacts and manifests",
        "aggregation": "sample_abs=sum_lag(abs(SHAP)); site_abs=mean_within_point_id; vegetation summary=equal-weight mean across sites",
        "environmental_groups": ENV_GROUPS,
        "vegetation_classes": VEGETATION_CLASSES,
        "explained_composition_metrics": explained_overall.to_dict(orient="records"),
        "background_summary": background_summary.iloc[0].to_dict(),
        "seed_composition_metrics": seed_output.drop_duplicates("comparison_id")[
            ["comparison_id", "spearman_rho_30cell", "kendall_tau_30cell", "pearson_r_30cell", "n_same_top1", "top1_agreement_fraction"]
        ].to_dict(orient="records"),
        "temporal_composition": temporal_composition.to_dict(orient="records"),
        "within_site_overlap_summary": within_summary.to_dict(orient="records"),
        "bootstrap": {
            "unit": "point_id within vegetation class",
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "interval": "two-sided 95% percentile",
            "sets_run_separately": True,
            "fixed_seed_repeat_checks": bootstrap_repeat_checks,
            "rows": len(bootstrap_output),
        },
        "result_row_counts": {relative_path: len(frame) for relative_path, frame in output_frames.items()},
        "figures_created": [geoshap_relative(path) for path in figure_paths],
        "input_shap_tree_after_run": input_tree_digest(),
        "geoshap_tree_after_run": geoshap_tree_digest(),
    }
    record_path = write_json("validation/vegetation_temporal_run_summary.json", record)
    summary_path = SUMMARY_DIR / "W36_VEGETATION_TEMPORAL_ATTRIBUTION_SUMMARY.md"
    atomic_write_text(summary_path, build_summary())
    inventory_path = VALIDATION_DIR / "vegetation_temporal_output_inventory.csv"
    paths = sorted(
        [path for directory in (SUMMARY_DIR, RESULTS_DIR, FIGURES_DIR) for path in directory.rglob("*") if path.is_file()],
        key=geoshap_relative,
    )
    inventory = pd.DataFrame([
        {"relative_path": geoshap_relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ])
    write_csv("validation/vegetation_temporal_output_inventory.csv", inventory)
    return {
        "status": "PASS",
        "result_files": len(output_frames),
        "figure_files": len(figure_paths),
        "run_summary": str(record_path),
        "summary": str(summary_path),
        "output_inventory": str(inventory_path),
        "bootstrap_rows": len(bootstrap_output),
    }


def metric_lines(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        lines.append(
            ", ".join(
                f"{column}={values[column]:.3f}" if isinstance(values[column], (float, np.floating)) else f"{column}={values[column]}"
                for column in columns
            )
        )
    return "; ".join(lines)


def truth_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().eq("true")


def build_summary() -> str:
    explained_overall = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_vegetation_composition_stability.csv")
    explained_class = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_vegetation_class_stability.csv")
    seed = pd.read_csv(RESULTS_DIR / "vegetation_temporal_seed_vegetation_composition_stability.csv")
    background = pd.read_csv(RESULTS_DIR / "vegetation_temporal_background_vegetation_composition_stability.csv")
    background_summary = pd.read_csv(RESULTS_DIR / "vegetation_temporal_background_vegetation_stability_summary.csv").iloc[0]
    nonseasonal_stability = pd.read_csv(RESULTS_DIR / "vegetation_temporal_nonseasonal_predictor_dominance_stability.csv")
    temporal = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_set_temporal_composition.csv")
    pairwise = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_set_pairwise_overlap.csv")
    within = pd.read_csv(RESULTS_DIR / "vegetation_temporal_within_site_temporal_overlap_summary.csv")
    bootstrap = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_vegetation_bootstrap_ci.csv")
    geoshap_spatial = pd.read_csv(GEOSHAP_ROOT / "robustness" / "geoshap_explained_spatial_stability.csv")

    primary_rows = []
    for row in explained_overall.itertuples(index=False):
        primary_rows.append(
            f"{row.comparison_id}: Spearman={row.spearman_rho_30cell:.3f}, Kendall={row.kendall_tau_30cell:.3f}, "
            f"Pearson={row.pearson_r_30cell:.3f}, top1 agreement={int(row.n_same_top1)}/6 ({row.top1_agreement_fraction:.3f})"
        )
    primary_text = "; ".join(primary_rows)

    consistency_lines = []
    for veg_class in VEGETATION_CLASSES:
        subset = explained_class.loc[explained_class["vegetation_class"] == veg_class]
        same_flags = truth_series(subset["same_top1"])
        same_n = int(same_flags.sum())
        comparisons_changed = subset.loc[~same_flags, "comparison_id"].astype(str).tolist()
        consistency_lines.append(
            f"{veg_class}: {same_n}/3 E0 comparisons retained top1"
            + (f"; changed in {', '.join(comparisons_changed)}" if comparisons_changed else "; unchanged in all three")
        )
    consistency_text = "\n".join(f"   - {line}" for line in consistency_lines)

    class_l1 = explained_class.groupby("vegetation_class", observed=True)["l1_share_distance"].mean().reindex(VEGETATION_CLASSES)
    min_l1 = float(class_l1.min())
    max_l1 = float(class_l1.max())
    most_stable = class_l1[np.isclose(class_l1, min_l1)].index.astype(str).tolist()
    least_stable = class_l1[np.isclose(class_l1, max_l1)].index.astype(str).tolist()

    seed_unique = seed.drop_duplicates("comparison_id")
    seed_lines = [
        f"{row.comparison_id}: Spearman={row.spearman_rho_30cell:.3f}, Kendall={row.kendall_tau_30cell:.3f}, "
        f"Pearson={row.pearson_r_30cell:.3f}, top1 agreement={row.top1_agreement_fraction:.3f}"
        for row in seed_unique.itertuples(index=False)
    ]

    nonseasonal_flags = truth_series(nonseasonal_stability["top_dominant_predictor_consistent"])
    nonseasonal_consistent = nonseasonal_stability.loc[nonseasonal_flags, "vegetation_class"].astype(str).tolist()
    nonseasonal_inconsistent = nonseasonal_stability.loc[~nonseasonal_flags, "vegetation_class"].astype(str).tolist()
    nonseasonal_judgment = (
        "Fully consistent across all six classes"
        if not nonseasonal_inconsistent
        else f"Mixed consistency; {len(nonseasonal_inconsistent)}/6 classes changed top predictor"
    )
    nonseasonal_ranges = []
    for row in nonseasonal_stability.itertuples(index=False):
        nonseasonal_ranges.append(
            f"{row.vegetation_class}: solar range={row.solar_proportion_min:.3f}-{row.solar_proportion_max:.3f}; "
            f"NDVI range={row.NDVI_proportion_min:.3f}-{row.NDVI_proportion_max:.3f}; "
            f"top predictors E0-E3={row.top_dominant_predictor_E0}/{row.top_dominant_predictor_E1}/{row.top_dominant_predictor_E2}/{row.top_dominant_predictor_E3}"
        )

    samples_per_site = "; ".join(
        f"{row.explained_set}={row.mean_samples_per_site:.3f} (median {row.median_samples_per_site:.1f}, range {int(row.min_samples_per_site)}-{int(row.max_samples_per_site)})"
        for row in temporal.itertuples(index=False)
    )
    e0_pairs = pairwise.loc[pairwise["set_A"] == "E0"]
    pair_overlap_text = "; ".join(
        f"{row.comparison_id}: global shared point_id-Year-Month={int(row.exact_point_year_month_n_shared)} "
        f"({row.exact_point_year_month_proportion_of_A:.3f} of E0; {row.exact_point_year_month_proportion_of_B:.3f} of comparison), "
        f"month TVD={row.month_of_year_total_variation_distance:.3f}"
        for row in e0_pairs.itertuples(index=False)
    )
    e0_within = within.loc[within["set_A"] == "E0"]
    within_text = "; ".join(
        f"{row.comparison_id}: median exact overlap={row.exact_year_month_overlap_count_median:.3f}, "
        f"IQR={row.exact_year_month_overlap_count_iqr:.3f}, zero-overlap n={int(round(row.fraction_sites_zero_exact_year_month_overlap * row.matched_site_n))}/{int(row.matched_site_n)} "
        f"(fraction={row.fraction_sites_zero_exact_year_month_overlap:.3f}), "
        f"<=1 fraction={row.fraction_sites_le_1_exact_shared_year_month:.3f}, >=2 fraction={row.fraction_sites_ge_2_exact_shared_year_month:.3f}"
        for row in e0_within.itertuples(index=False)
    )

    geoshap_core = geoshap_spatial.loc[
        geoshap_spatial["environmental_group"].isin(["Energy", "Vegetation productivity"]),
        ["comparison_id", "environmental_group", "spearman_rho"],
    ]
    geoshap_core_text = "; ".join(
        f"{row.comparison_id}/{row.environmental_group} rho={row.spearman_rho:.3f}"
        for row in geoshap_core.itertuples(index=False)
    )

    top_changes = explained_class.loc[~truth_series(explained_class["same_top1"]), ["comparison_id", "vegetation_class", "reference_top1_group", "comparison_top1_group"]]
    adverse_items = []
    if len(top_changes):
        adverse_items.append(
            "Top1 group changes occurred: "
            + "; ".join(
                f"{row.comparison_id}/{row.vegetation_class}: {row.reference_top1_group} -> {row.comparison_top1_group}"
                for row in top_changes.itertuples(index=False)
            )
        )
    if nonseasonal_inconsistent:
        adverse_items.append("Dominant predictor changed across E0-E3 for: " + ", ".join(nonseasonal_inconsistent))
    min_primary = explained_overall.loc[explained_overall["spearman_rho_30cell"].idxmin()]
    adverse_items.append(
        f"Lowest primary 30-cell Spearman: {min_primary.spearman_rho_30cell:.3f} ({min_primary.comparison_id})."
    )
    max_zero = e0_within.loc[e0_within["fraction_sites_zero_exact_year_month_overlap"].idxmax()]
    adverse_items.append(
        f"Largest E0-pair zero exact Year-Month fraction was {max_zero.fraction_sites_zero_exact_year_month_overlap:.3f} ({max_zero.comparison_id})."
    )
    adverse_text = "\n".join(f"- {item}" for item in adverse_items)

    return f"""# W36 vegetation and temporal attribution summary

Generated: {now_iso()}

## Vegetation-conditioned composition

E0 versus E1/E2/E3 30-cell correlations: {primary_text}.

Top environmental-group consistency by vegetation class:
{consistency_text}

Using per-class L1 share distance averaged across E0 versus E1/E2/E3, the smallest mean L1 was {min_l1:.3f} ({', '.join(most_stable)}) and the largest was {max_l1:.3f} ({', '.join(least_stable)}).

Background perturbation stability across 11 comparisons: 30-cell Spearman min/median/max = {background_summary['spearman_rho_30cell_min']:.3f}/{background_summary['spearman_rho_30cell_median']:.3f}/{background_summary['spearman_rho_30cell_max']:.3f}; top1 vegetation-class agreement min/median/max = {background_summary['top1_agreement_fraction_min']:.3f}/{background_summary['top1_agreement_fraction_median']:.3f}/{background_summary['top1_agreement_fraction_max']:.3f}. Kendall and Pearson are included in the result table.

Three-seed stability: {'; '.join(seed_lines)}.

Dominant nonseasonal predictor consistency across the four E-sets: {nonseasonal_judgment}. Consistent classes: {len(nonseasonal_consistent)}/6 ({', '.join(nonseasonal_consistent) if nonseasonal_consistent else 'none'}). Other classes: {', '.join(nonseasonal_inconsistent) if nonseasonal_inconsistent else 'none'}. {'; '.join(nonseasonal_ranges)}.

Explained samples per site: {samples_per_site}.

E0 versus E1/E2/E3 exact Year-Month overlap: {pair_overlap_text}. Within matched sites: {within_text}.

Vegetation bootstrap: {len(bootstrap)} E-set × vegetation-class × environmental-group rows; point_id resampling within vegetation class; B={BOOTSTRAP_B}; seed={BOOTSTRAP_SEED}; two-sided 95% percentile intervals.

## Stability results

{adverse_text}
"""


def render_figure_c_layout() -> dict[str, Any]:
    """Render Figure C from vegetation-composition result tables."""

    target = TEMPORAL_ROOT / FIGURE_FILES[2]
    require(target.is_file(), f"Figure C is missing: {target}")
    summary = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_vegetation_group_summary.csv")
    metrics = pd.read_csv(RESULTS_DIR / "vegetation_temporal_explained_vegetation_composition_stability.csv")
    summaries = {label: summary.loc[summary["analysis_label"] == label].copy() for label in ("E0", "E1", "E2", "E3")}
    matrices = {label: summary_matrix(summaries[label]) for label in summaries}
    colors = dict(zip(ENV_GROUPS, plt.get_cmap("tab10").colors[: len(ENV_GROUPS)]))
    all_values = np.concatenate([matrix.to_numpy().reshape(-1) for matrix in matrices.values()])
    minimum = max(0.0, float(all_values.min()) - 0.02)
    maximum = float(all_values.max()) + 0.02
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.76, wspace=0.30)
    for axis, label in zip(axes, ("E1", "E2", "E3")):
        for group in ENV_GROUPS:
            axis.scatter(
                matrices["E0"][group].to_numpy(dtype=float),
                matrices[label][group].to_numpy(dtype=float),
                s=32,
                alpha=0.85,
                color=colors[group],
                label=group,
                edgecolors="none",
            )
        axis.plot([minimum, maximum], [minimum, maximum], color="0.35", linewidth=1, linestyle="--")
        row = metrics.loc[metrics["comparison_label"] == label].iloc[0]
        axis.set_title(f"E0 vs {label}: 30 cells\nSpearman={row['spearman_rho_30cell']:.3f}")
        axis.set_xlabel("E0 environmental-group share")
        axis.set_ylabel(f"{label} environmental-group share")
        axis.set_xlim(minimum, maximum)
        axis.set_ylim(minimum, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("E0 versus E1-E3 vegetation-conditioned model-attribution composition", y=0.96)
    temporary = target.with_name(target.name + ".tmp")
    require(not temporary.exists(), f"Unexpected temporary figure exists: {temporary}")
    fig.savefig(temporary, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    os.replace(temporary, target)
    require(target.stat().st_size > 10_000, "Rendered Figure C is implausibly small")
    return {"status": "PASS", "scope": "figure rendering", "figure": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "run", "render-figure-c"))
    args = parser.parse_args()
    try:
        if args.phase == "validate":
            result = validate_inputs()
        elif args.phase == "run":
            result = run_postprocessing()
        else:
            result = render_figure_c_layout()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "phase": args.phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
