#!/usr/bin/env python
"""
One-configuration-at-a-time W36 SHAP runner.

The runner executes one configured SHAP analysis at a time:

* verifies the 18 fixed sample manifests;
* loads the fixed W36 checkpoints;
* executes exactly one configuration per invocation;
* checkpoints every SHAP batch atomically;
* writes results and a completion marker after all batches finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[3]
RUN_ROOT = PROJECT_ROOT / "runtime" / "runs" / "shap"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
CANONICAL_ROOT = PROJECT_ROOT / "runtime" / "runs" / "canonical"
SAMPLE_ROOT = PROJECT_ROOT / "data_documentation" / "shap_sample_keys"
LOG_PATH = RUN_ROOT / "logs" / "global_shap.log"

WINDOW = 36
INPUT_DIM = 19
EXPLAIN_N = 1000
SHAP_NSAMPLES = 50
SHAP_RSEED = 20240724
SHAP_BATCH_SIZE = 16

FEATURE_NAMES = [
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

GROUPS: "OrderedDict[str, List[str]]" = OrderedDict(
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
    "X": "ed663f960a560419feecc19f30164881ab1cf62eb0c8b0679a7ec19f61e09c8c",
    "Y_scaler": "f533971fa9842ac1f2f9162d095683ac7fdee0e3da524f6caf6187120636a4ec",
    "X_scaler": "8500e47d5a98b59ef0f028881b5adb09e21051394c6e673642fdc9cf96443367",
    "metadata": "1fec78d21f9ff064b4f0468c4d89dcd95b7e8fe01ba6055e243c836e630c1f8d",
}

X_PATH = DATA_ROOT / "X_tensor_aligned_36.npy"
Y_SCALER_PATH = DATA_ROOT / "Y_scaler_params.npy"
X_SCALER_PATH = DATA_ROOT / "X_scaler_params.npz"
METADATA_PATH = (
    DATA_ROOT / "preprocessing_metadata.json"
)
CHECKPOINTS = {
    seed: (
        CANONICAL_ROOT
        / "final_seed_runs"
        / f"seed_{seed}"
        / "model_validation_selected.pt"
    )
    for seed in (42, 2024, 3407)
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model_seed: int
    background_draw: str
    explained_draw: str


def build_run_plan() -> List[RunSpec]:
    plan = [
        RunSpec(
            "background_BG256_D1", 42, "BG256_D1", "EXPLAINED_E0"
        )
    ]
    for size in (64, 128, 256, 512):
        for draw in (1, 2, 3):
            draw_id = f"BG{size}_D{draw}"
            if draw_id != "BG256_D1":
                plan.append(
                    RunSpec(
                        f"background_{draw_id}",
                        42,
                        draw_id,
                        "EXPLAINED_E0",
                    )
                )
    plan.extend(
        [
            RunSpec("seed_2024", 2024, "BG256_D1", "EXPLAINED_E0"),
            RunSpec("seed_3407", 3407, "BG256_D1", "EXPLAINED_E0"),
            RunSpec("explained_E1", 42, "BG256_D1", "EXPLAINED_E1"),
            RunSpec("explained_E2", 42, "BG256_D1", "EXPLAINED_E2"),
            RunSpec("explained_E3", 42, "BG256_D1", "EXPLAINED_E3"),
        ]
    )
    if len(plan) != 17 or len({item.run_id for item in plan}) != 17:
        raise RuntimeError("Internal run plan must contain 17 unique runs.")
    return plan


RUN_PLAN = build_run_plan()
RUN_BY_ID = {item.run_id: item for item in RUN_PLAN}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(
        timespec="milliseconds"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def log_event(event: str, **fields) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": now_iso(), "event": event, **fields}
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.12g")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def isolate_orphan_temps(partial_dir: Path) -> List[str]:
    orphans = sorted(partial_dir.glob("*.tmp"))
    if not orphans:
        return []
    quarantine = partial_dir / (
        "orphaned_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    quarantine.mkdir(parents=True, exist_ok=False)
    moved = []
    for source in orphans:
        destination = quarantine / source.name
        os.replace(source, destination)
        moved.append(str(destination))
    return moved


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cnn_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                input_dim,
                cnn_channels,
                kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                cnn_channels,
                cnn_channels,
                kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, checkpoint: Mapping[str, object]):
        super().__init__()
        fixed = checkpoint["fixed_model_config"]
        input_dim = int(checkpoint["input_dim"])
        hidden_size = int(checkpoint["hidden_size"])
        cnn_channels = int(fixed["cnn_channels"])
        dropout = float(fixed["dropout"])
        self.conv = ConvBlock(
            input_dim,
            cnn_channels,
            int(fixed["kernel_size"]),
            dropout,
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = hidden_size * 2
        self.attention = TemporalAttention(
            out_dim,
            int(fixed["attention_dim"]),
            dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        out, _ = self.bilstm(x)
        context, _ = self.attention(out)
        return self.regressor(context).squeeze(-1)


class SHAPOutputWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        output = self.base_model(x)
        return output.unsqueeze(-1) if output.ndim == 1 else output


def verify_groups() -> None:
    flattened = [name for fields in GROUPS.values() for name in fields]
    if len(flattened) != 19 or set(flattened) != set(FEATURE_NAMES):
        raise RuntimeError("Locked groups do not exactly cover 19 features.")
    if GROUPS["Vegetation class"] != VEGETATION_FEATURES:
        raise RuntimeError("Vegetation group is not the fixed six fields.")


def verify_sample_manifests() -> Dict[str, str]:
    expected = [
        *(f"sample_keys/background/shap_BG{size}_D{draw}.csv" for size in (64, 128, 256, 512) for draw in (1, 2, 3)),
        *(f"sample_keys/shap_explained_E{draw}.csv" for draw in range(4)),
        "sample_keys/shap_background_manifests.csv",
        "sample_keys/shap_sampling_balance_summary.csv",
    ]
    verified = {}
    for relative_path in expected:
        path = SAMPLE_ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        verified[relative_path] = actual
    return verified


def verify_prepared_data() -> None:
    paths = {
        "X": X_PATH,
        "Y_scaler": Y_SCALER_PATH,
        "X_scaler": X_SCALER_PATH,
        "metadata": METADATA_PATH,
    }
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    scaler = np.load(X_SCALER_PATH, allow_pickle=False)
    metadata_features = list(metadata["feature_names"])
    scaler_features = [str(value) for value in scaler["feature_columns"]]
    if metadata_features != FEATURE_NAMES or scaler_features != FEATURE_NAMES:
        raise RuntimeError("Prepared feature/scaler order mismatch.")


def checkpoint_path(seed: int) -> Path:
    path = CHECKPOINTS[seed]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_fixed_model(seed: int, device: torch.device):
    path = checkpoint_path(seed)
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )
    if (
        int(checkpoint["window"]) != WINDOW
        or int(checkpoint["seed"]) != seed
        or int(checkpoint["input_dim"]) != INPUT_DIM
        or str(checkpoint["model_name"]) != "CNN-BiLSTM-Attention"
    ):
        raise RuntimeError("Checkpoint architecture/identity mismatch.")
    model = CNNBiLSTMAttention(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def manifest_path(draw_id: str) -> Path:
    if draw_id.startswith("BG"):
        return (
            SAMPLE_ROOT
            / "sample_keys"
            / "background"
            / f"shap_{draw_id}.csv"
        )
    return (
        SAMPLE_ROOT
        / "sample_keys"
        / f"shap_explained_{draw_id.split('_')[-1]}.csv"
    )


def load_manifest(draw_id: str) -> pd.DataFrame:
    path = manifest_path(draw_id)
    frame = pd.read_csv(path, dtype={"point_id": str})
    frame["point_id"] = frame["point_id"].str.zfill(3)
    expected_split = "train" if draw_id.startswith("BG") else "test"
    if set(frame["split"].astype(str)) != {expected_split}:
        raise RuntimeError(
            f"{draw_id} contains rows outside {expected_split}."
        )
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"{draw_id} contains duplicate sample_ids.")
    return frame


def deterministic_seeds() -> None:
    # SHAP sampling RNG seeds.
    random.seed(SHAP_RSEED)
    np.random.seed(SHAP_RSEED)
    torch.manual_seed(SHAP_RSEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SHAP_RSEED)


def output_paths(run_id: str) -> Dict[str, Path]:
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


def completion_marker_valid(run_id: str) -> bool:
    paths = output_paths(run_id)
    marker_path = paths["completion_marker"]
    if not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE":
        return False
    for key, item in marker.get("output_artifacts", {}).items():
        path = RUN_ROOT / item["relative_path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(
                f"Completion marker artifact mismatch: {key} / {path}"
            )
    return True


def select_run_id(requested: Optional[str], resume: bool) -> str:
    if requested:
        if requested not in RUN_BY_ID:
            raise RuntimeError(
                f"Unknown run-id {requested!r}. Use --list-runs."
            )
        return requested
    if not resume:
        return "background_BG256_D1"
    for spec in RUN_PLAN:
        if not completion_marker_valid(spec.run_id):
            return spec.run_id
    raise RuntimeError("All 17 configurations already have valid markers.")


def rank_desc(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(-np.asarray(values), method="average")


def sign_int(values: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    values = np.asarray(values)
    return np.where(values > atol, 1, np.where(values < -atol, -1, 0))


def summarize_shap(
    values: np.ndarray,
    model_inputs: np.ndarray,
    spec: RunSpec,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_abs = np.abs(values).sum(axis=1)
    row_signed = values.sum(axis=1)
    importance = row_abs.mean(axis=0)
    mean_signed = row_signed.mean(axis=0)
    mean_feature_value = model_inputs.mean(axis=1)
    contrast = np.zeros(INPUT_DIM, dtype=float)
    for index in range(INPUT_DIM):
        low_q, high_q = np.quantile(
            mean_feature_value[:, index], [0.25, 0.75]
        )
        low = row_signed[
            mean_feature_value[:, index] <= low_q, index
        ]
        high = row_signed[
            mean_feature_value[:, index] >= high_q, index
        ]
        contrast[index] = float(high.mean() - low.mean())
    variable = pd.DataFrame(
        {
            "run_id": spec.run_id,
            "model_seed": spec.model_seed,
            "background_draw": spec.background_draw,
            "explained_draw": spec.explained_draw,
            "entity_type": "variable",
            "entity": FEATURE_NAMES,
            "mean_abs_shap_gC_m2_month": importance,
            "rank": rank_desc(importance),
            "mean_signed_shap_gC_m2_month": mean_signed,
            "mean_signed_sign": sign_int(mean_signed),
            "high_minus_low_shap_contrast_gC_m2_month": contrast,
            "contrast_sign": sign_int(contrast),
        }
    )
    grouped_rows = []
    for group, fields in GROUPS.items():
        positions = [FEATURE_NAMES.index(field) for field in fields]
        group_abs = row_abs[:, positions].sum(axis=1)
        group_signed = row_signed[:, positions].sum(axis=1)
        grouped_rows.append(
            {
                "run_id": spec.run_id,
                "model_seed": spec.model_seed,
                "background_draw": spec.background_draw,
                "explained_draw": spec.explained_draw,
                "entity_type": "group",
                "entity": group,
                "mean_abs_shap_gC_m2_month": float(group_abs.mean()),
                "mean_signed_shap_gC_m2_month": float(
                    group_signed.mean()
                ),
                "mean_signed_sign": int(
                    sign_int(np.asarray([group_signed.mean()]))[0]
                ),
                "high_minus_low_shap_contrast_gC_m2_month": np.nan,
                "contrast_sign": 0,
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    grouped["rank"] = rank_desc(
        grouped["mean_abs_shap_gC_m2_month"].to_numpy()
    )
    grouped = grouped[variable.columns]
    signed = variable[
        [
            "run_id",
            "model_seed",
            "background_draw",
            "explained_draw",
            "entity",
            "mean_signed_shap_gC_m2_month",
            "mean_signed_sign",
            "high_minus_low_shap_contrast_gC_m2_month",
            "contrast_sign",
        ]
    ].copy()
    return variable, grouped, signed


def expected_protocol(
    spec: RunSpec,
    device: str,
    background_manifest_hash: str,
    explained_manifest_hash: str,
) -> dict:
    return {
        "run_id": spec.run_id,
        "window": WINDOW,
        "model_seed": spec.model_seed,
        "checkpoint_sha256": sha256_file(CHECKPOINTS[spec.model_seed]),
        "background_draw": spec.background_draw,
        "background_manifest_sha256": background_manifest_hash,
        "explained_draw": spec.explained_draw,
        "explained_manifest_sha256": explained_manifest_hash,
        "device": device,
        "explainer": "GradientExplainer",
        "nsamples": SHAP_NSAMPLES,
        "rseed": SHAP_RSEED,
        "batch_size": SHAP_BATCH_SIZE,
        "feature_order": FEATURE_NAMES,
        "groups": GROUPS,
    }


def load_or_create_progress(
    partial_dir: Path, protocol: dict, resume: bool
) -> Tuple[Path, dict]:
    progress_path = partial_dir / "progress.json"
    if partial_dir.exists():
        moved = isolate_orphan_temps(partial_dir)
        if moved:
            log_event(
                "ORPHAN_TEMPS_ISOLATED",
                run_id=protocol["run_id"],
                files=moved,
            )
    if progress_path.exists():
        if not resume:
            raise RuntimeError(
                f"Partial progress exists for {protocol['run_id']}; "
                "rerun with --resume."
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("protocol") != protocol:
            raise RuntimeError(
                "Resume refused: stored protocol differs from this invocation."
            )
        return progress_path, progress
    partial_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "RUNNING",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol,
        "completed_rows": 0,
        "completed_batches": 0,
        "current_batch": None,
        "batches": {},
        "run_complete": False,
    }
    atomic_write_json(progress_path, progress)
    return progress_path, progress


def validate_reusable_batch(
    partial_dir: Path,
    batch_key: str,
    entry: Mapping[str, object],
    expected_shape: Tuple[int, int, int],
) -> np.ndarray:
    path = partial_dir / str(entry["file"])
    if not path.exists():
        raise RuntimeError(f"Recorded batch file is missing: {path}")
    if sha256_file(path) != str(entry["sha256"]):
        raise RuntimeError(f"Recorded batch hash mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if tuple(values.shape) != expected_shape:
        raise RuntimeError(
            f"Recorded batch shape mismatch for {batch_key}: {values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"Recorded batch contains NaN/Inf: {path}")
    return values


def run_one_configuration(
    spec: RunSpec, device_name: str, resume: bool
) -> int:
    if completion_marker_valid(spec.run_id):
        if resume:
            log_event(
                "RUN_SKIPPED_ALREADY_COMPLETE", run_id=spec.run_id
            )
            print(
                f"{spec.run_id} already has a valid completion marker; skipped."
            )
            return 0
        raise RuntimeError(
            f"{spec.run_id} is already complete. Use --resume to skip safely."
        )

    verify_groups()
    manifest_hashes = verify_sample_manifests()
    verify_prepared_data()
    background_path = manifest_path(spec.background_draw)
    explained_path = manifest_path(spec.explained_draw)
    background_hash_key = background_path.relative_to(SAMPLE_ROOT).as_posix()
    explained_hash_key = explained_path.relative_to(SAMPLE_ROOT).as_posix()
    if (
        background_hash_key not in manifest_hashes
        or explained_hash_key not in manifest_hashes
    ):
        raise RuntimeError("Selected manifest is absent from sample-manifest inventory.")
    background_manifest = load_manifest(spec.background_draw)
    explained_manifest = load_manifest(spec.explained_draw)
    if len(explained_manifest) != EXPLAIN_N:
        raise RuntimeError("Explained manifest size is not 1,000.")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    device = torch.device(device_name)
    protocol = expected_protocol(
        spec,
        device_name,
        manifest_hashes[background_hash_key],
        manifest_hashes[explained_hash_key],
    )
    partial_dir = (
        RUN_ROOT / "shap_runs" / "_partial" / spec.run_id
    )
    progress_path, progress = load_or_create_progress(
        partial_dir, protocol, resume
    )
    log_event(
        "RUN_START",
        run_id=spec.run_id,
        device=device_name,
        resume=resume,
        completed_rows=progress["completed_rows"],
        protocol=protocol,
    )

    # Import SHAP only after the durable RUN_START/progress record exists.
    import shap

    deterministic_seeds()
    x = np.load(X_PATH, mmap_mode="r")
    if x.shape != (62244, WINDOW, INPUT_DIM):
        raise RuntimeError(f"Unexpected canonical X shape: {x.shape}")
    y_scaler = np.load(Y_SCALER_PATH).astype(np.float64)
    if y_scaler.shape != (2,) or y_scaler[1] <= 0:
        raise RuntimeError("Unexpected canonical Y scaler.")
    output_scale_to_g = float(y_scaler[1] * 1000.0)
    model, checkpoint = load_fixed_model(spec.model_seed, device)
    wrapper = SHAPOutputWrapper(model).to(device)
    wrapper.eval()
    # Validated compatibility setting. One-layer LSTM has no internal dropout.
    wrapper.base_model.bilstm.train()
    background_ids = background_manifest["sample_id"].to_numpy(
        dtype=np.int64
    )
    explained_ids = explained_manifest["sample_id"].to_numpy(
        dtype=np.int64
    )
    background_tensor = torch.from_numpy(
        np.asarray(x[background_ids], dtype=np.float32)
    ).to(device)

    with torch.backends.cudnn.flags(enabled=False):
        explainer = shap.GradientExplainer(wrapper, background_tensor)
        for start in range(0, len(explained_ids), SHAP_BATCH_SIZE):
            stop = min(start + SHAP_BATCH_SIZE, len(explained_ids))
            batch_key = f"{start:04d}_{stop:04d}"
            expected_shape = (stop - start, WINDOW, INPUT_DIM)
            existing = progress["batches"].get(batch_key)
            if existing is not None:
                validate_reusable_batch(
                    partial_dir, batch_key, existing, expected_shape
                )
                log_event(
                    "BATCH_REUSED",
                    run_id=spec.run_id,
                    batch=batch_key,
                    completed_rows=stop,
                )
                continue

            progress["status"] = "RUNNING"
            progress["current_batch"] = {
                "batch": batch_key,
                "start": start,
                "stop": stop,
                "started_at": now_iso(),
            }
            progress["updated_at"] = now_iso()
            atomic_write_json(progress_path, progress)
            log_event(
                "BATCH_START",
                run_id=spec.run_id,
                batch=batch_key,
                start=start,
                stop=stop,
            )
            batch_tensor = torch.from_numpy(
                np.asarray(
                    x[explained_ids[start:stop]], dtype=np.float32
                )
            ).to(device)
            values = explainer.shap_values(
                batch_tensor,
                nsamples=SHAP_NSAMPLES,
                rseed=SHAP_RSEED,
            )
            if isinstance(values, list):
                values = values[0]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            values = np.asarray(values)
            if values.ndim == 4 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.ndim == 4 and values.shape[0] == 1:
                values = values[0]
            if tuple(values.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected SHAP batch shape: {values.shape}"
                )
            values = (
                values.astype(np.float32)
                * np.float32(output_scale_to_g)
            )
            if not np.isfinite(values).all():
                raise RuntimeError("NaN/Inf in SHAP batch.")
            batch_path = partial_dir / f"batch_{batch_key}.npy"
            atomic_save_npy(batch_path, values)
            entry = {
                "file": batch_path.name,
                "start": start,
                "stop": stop,
                "shape": list(values.shape),
                "bytes": batch_path.stat().st_size,
                "sha256": sha256_file(batch_path),
                "saved_at": now_iso(),
            }
            progress["batches"][batch_key] = entry
            progress["completed_batches"] = len(progress["batches"])
            progress["completed_rows"] = sum(
                int(item["stop"]) - int(item["start"])
                for item in progress["batches"].values()
            )
            progress["current_batch"] = None
            progress["updated_at"] = now_iso()
            atomic_write_json(progress_path, progress)
            log_event(
                "BATCH_SAVED",
                run_id=spec.run_id,
                batch=batch_key,
                completed_rows=progress["completed_rows"],
                total_rows=len(explained_ids),
                batch_file=str(batch_path),
                batch_sha256=entry["sha256"],
            )
            del batch_tensor, values
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_batch_count = math.ceil(
        len(explained_ids) / SHAP_BATCH_SIZE
    )
    if (
        progress["completed_rows"] != len(explained_ids)
        or progress["completed_batches"] != expected_batch_count
    ):
        raise RuntimeError("Not all SHAP batches are durably complete.")

    ordered_values = []
    for start in range(0, len(explained_ids), SHAP_BATCH_SIZE):
        stop = min(start + SHAP_BATCH_SIZE, len(explained_ids))
        batch_key = f"{start:04d}_{stop:04d}"
        ordered_values.append(
            validate_reusable_batch(
                partial_dir,
                batch_key,
                progress["batches"][batch_key],
                (stop - start, WINDOW, INPUT_DIM),
            )
        )
    shap_values = np.concatenate(ordered_values, axis=0)
    if shap_values.shape != (EXPLAIN_N, WINDOW, INPUT_DIM):
        raise RuntimeError(
            f"Final SHAP shape mismatch: {shap_values.shape}"
        )
    model_inputs = np.asarray(x[explained_ids], dtype=np.float32)
    variable, grouped, signed = summarize_shap(
        shap_values, model_inputs, spec
    )
    paths = output_paths(spec.run_id)

    atomic_save_npz(
        paths["shap_npz"],
        shap_values_gC_m2_month=shap_values.astype(np.float32),
        background_sample_ids=background_ids,
        explained_sample_ids=explained_ids,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U64"),
        window=np.asarray([WINDOW], dtype=np.int16),
        model_seed=np.asarray([spec.model_seed], dtype=np.int32),
    )
    atomic_write_csv(paths["variable_csv"], variable)
    atomic_write_csv(paths["grouped_csv"], grouped)
    atomic_write_csv(paths["signed_csv"], signed)
    execution = {
        "status": "COMPLETE",
        "completed_at": now_iso(),
        "run_id": spec.run_id,
        "model_seed": spec.model_seed,
        "checkpoint": str(CHECKPOINTS[spec.model_seed]),
        "checkpoint_sha256": sha256_file(CHECKPOINTS[spec.model_seed]),
        "background_draw": spec.background_draw,
        "background_n": len(background_ids),
        "explained_draw": spec.explained_draw,
        "explained_n": len(explained_ids),
        "device": device_name,
        "explainer": "GradientExplainer",
        "nsamples": SHAP_NSAMPLES,
        "rseed": SHAP_RSEED,
        "batch_size": SHAP_BATCH_SIZE,
        "output_units": "gC m^-2 month^-1",
        "strict_checkpoint_load": True,
        "torch_version": torch.__version__,
        "shap_version": shap.__version__,
    }
    atomic_write_json(paths["execution_json"], execution)
    output_artifacts = {}
    for key in (
        "shap_npz",
        "variable_csv",
        "grouped_csv",
        "signed_csv",
        "execution_json",
    ):
        path = paths[key]
        output_artifacts[key] = {
            "relative_path": path.relative_to(RUN_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    completion_marker = {
        "status": "COMPLETE",
        "run_id": spec.run_id,
        "completed_at": now_iso(),
        "protocol": protocol,
        "shape": list(shap_values.shape),
        "completed_batches": expected_batch_count,
        "completed_rows": len(explained_ids),
        "output_artifacts": output_artifacts,
    }
    # The marker is written last. Its existence is the sole completion signal.
    atomic_write_json(paths["completion_marker"], completion_marker)
    progress["status"] = "COMPLETE_SAVED"
    progress["run_complete"] = True
    progress["completion_marker"] = str(paths["completion_marker"])
    progress["updated_at"] = now_iso()
    atomic_write_json(progress_path, progress)
    log_event(
        "RUN_COMPLETE",
        run_id=spec.run_id,
        completion_marker=str(paths["completion_marker"]),
        output_artifacts=output_artifacts,
    )
    print(f"COMPLETE: {spec.run_id}")
    print(f"Marker: {paths['completion_marker']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one W36 SHAP configuration.",
        epilog=(
            "Example:\n"
            "  python run_global_shap.py "
            "--device cpu --run-id background_BG256_D1 --resume"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Compute device (default: cpu).",
    )
    parser.add_argument(
        "--run-id",
        help="Configuration ID; use --list-runs to show valid IDs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a configuration from saved batch checkpoints.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List the 17 valid run IDs and exit without computation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_runs:
        for index, spec in enumerate(RUN_PLAN, start=1):
            print(
                f"{index:02d}  {spec.run_id}  seed={spec.model_seed}  "
                f"background={spec.background_draw}  "
                f"explained={spec.explained_draw}"
            )
        return 0

    run_id = select_run_id(args.run_id, args.resume)
    spec = RUN_BY_ID[run_id]
    try:
        return run_one_configuration(spec, args.device, args.resume)
    except KeyboardInterrupt:
        log_event(
            "RUN_INTERRUPTED",
            run_id=run_id,
            device=args.device,
            traceback="KeyboardInterrupt",
        )
        print(
            f"Interrupted. Re-run with --run-id {run_id} --resume.",
            file=sys.stderr,
        )
        return 130
    except Exception:
        log_event(
            "RUN_FAILED",
            run_id=run_id,
            device=args.device,
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
