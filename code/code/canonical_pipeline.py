# -*- coding: utf-8 -*-
"""W36 model selection, three-seed training and chronological evaluation.

Window and hyperparameters are selected using Validation performance.
Training seeds: 42, 2024, 3407. Evaluation: 2022–2025.
Commands and paths are listed in the repository README.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Fixed study protocol
# -----------------------------------------------------------------------------

PROTOCOL_VERSION = "monthly-productivity-W36-public-v1.0"
SCHEMA_VERSION = "monthly-productivity-W36-schema-v1"
DEFAULT_SEEDS = (42, 2024, 3407)
DEFAULT_WINDOWS = (6, 12, 18, 24, 36, 48)
WINDOW_SELECTION_RULE = (
    "lowest full-precision mean Validation MSE",
    "only if means are exactly equal: lowest full-precision median Validation MSE",
    "only if medians are exactly equal: shorter historical window",
)
EVALUATION_CONFIRMATION_TOKEN = "I_ACCEPT_FINAL_EVALUATION"
REPRODUCIBILITY_STATUS = "seed-controlled stochastic reproducibility"

PREPARED_DATA_DIR_NAME = "data/processed"
PREPARED_METADATA_NAME = "preprocessing_metadata.json"
PREPARED_DATA_VERSION = "prepared_month_inputs_v1"
EXPECTED_TOTAL_SAMPLES = 62244
EXPECTED_SPLIT_COUNTS = {"train": 41496, "val": 8892, "test": 11856}
EXPECTED_YEAR_SETS = {
    "train": tuple(range(2005, 2019)),
    "val": tuple(range(2019, 2022)),
    "test": tuple(range(2022, 2026)),
}
EXPECTED_FEATURE_DIM = 19
EXPECTED_POINT_COUNT = 247

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PREPARED_DATA_PROJECT_RELATIVE_PATH = Path(PREPARED_DATA_DIR_NAME)
PREPARED_DATA_DIR = (PROJECT_ROOT / PREPARED_DATA_PROJECT_RELATIVE_PATH).resolve()
RUNS_BASE = (PROJECT_ROOT / "runtime" / "runs").resolve()
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_PATH_PARTS = {"archive"}


@dataclass(frozen=True)
class FixedModelConfig:
    cnn_channels: int = 64
    kernel_size: int = 3
    attention_dim: int = 64
    dropout: float = 0.30
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 150
    early_stopping_patience: int = 20
    min_delta: float = 1e-6
    grad_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 6
    scheduler_min_lr: float = 1e-6
    num_workers: int = 0


FIXED = FixedModelConfig()


@dataclass(frozen=True)
class RIMEConfig:
    search_seed: int = 42
    rime_epochs: int = 5
    population_size: int = 10
    candidate_max_epochs: int = 25
    candidate_patience: int = 5
    log10_lr_min: float = -4.0
    log10_lr_max: float = -2.0
    hidden_options: Tuple[int, ...] = (32, 64, 128)


RIME_FIXED = RIMEConfig()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def json_converter(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def save_json(obj: Any, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing critical output: {path}")
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary output: {tmp}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_converter)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_new(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing critical output: {path}")
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest(actual_device: Optional[torch.device] = None) -> Dict[str, Any]:
    cuda_name = None
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = "CUDA device available"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "torch": torch.__version__,
        "mealpy": package_version("mealpy"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": cuda_name,
        "actual_device": str(actual_device) if actual_device is not None else str(get_device()),
        "reproducibility_status": REPRODUCIBILITY_STATUS,
    }


def validate_run_root(path: Path) -> Path:
    """Return a safe runtime/runs/<run_id> root or fail closed."""
    resolved = Path(path).resolve()
    forbidden = {part.lower() for part in resolved.parts} & RESERVED_PATH_PARTS
    if forbidden:
        raise ValueError(f"Output path targets a reserved location: {sorted(forbidden)}")
    if resolved.parent != RUNS_BASE:
        raise ValueError(
            "Run outputs must use a direct child run root under "
            f"{RUNS_BASE}, for example runtime/runs/2026-07-23_run01"
        )
    if not RUN_ID_RE.fullmatch(resolved.name):
        raise ValueError("Run ID must use only letters, digits, '.', '_' or '-' and may not be empty")
    return resolved


def assert_stage_can_start(run_root: Path, stage: str, critical_paths: Sequence[Path]) -> None:
    run_root = validate_run_root(run_root)
    lock_path = run_root / "pre_evaluation_lock_manifest.json"
    if lock_path.exists():
        lock = load_json(lock_path)
        if lock.get("state") == "EVALUATED":
            raise RuntimeError(
                f"Run {run_root.name} is EVALUATED; refusing {stage}. Use a fresh run ID."
            )
    conflicts = [str(p) for p in critical_paths if p.exists()]
    if conflicts:
        raise FileExistsError(
            f"Refusing {stage} because outputs already exist: {conflicts}. "
            "Use a fresh run ID; do not overwrite a prior run."
        )


def assert_exact_seeds(values: Iterable[int], context: str) -> Tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if seeds != DEFAULT_SEEDS:
        raise ValueError(
            f"{context} must contain exactly the ordered fixed seeds "
            f"{list(DEFAULT_SEEDS)}; got {list(seeds)}"
        )
    return seeds


def validate_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def require_prepared_data_dir(path: Path) -> Path:
    """Resolve and enforce the single project-relative prepared input location."""
    resolved = Path(path).resolve()
    normalized = os.path.normcase(os.path.normpath(str(resolved)))
    prepared_normalized = os.path.normcase(os.path.normpath(str(PREPARED_DATA_DIR)))
    if normalized != prepared_normalized:
        raise ValueError(
            "Prepared data directory must resolve to the project-relative input location "
            f"{PREPARED_DATA_PROJECT_RELATIVE_PATH.as_posix()!r} under {PROJECT_ROOT}; "
            f"got {resolved}"
        )
    return resolved


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {
            "n": 0, "RMSE": np.nan, "MAE": np.nan,
            "R2": np.nan, "Bias": np.nan, "Pearson_r": np.nan,
        }
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias_v = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    r = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    )
    return {
        "n": int(n), "RMSE": rmse, "MAE": mae,
        "R2": r2, "Bias": bias_v, "Pearson_r": r,
    }


# -----------------------------------------------------------------------------
# Model: matches the canonical W36 prepared-input architecture
# -----------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, cfg: FixedModelConfig):
        super().__init__()
        padding = cfg.kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, cfg.cnn_channels, cfg.kernel_size, padding=padding),
            nn.BatchNorm1d(cfg.cnn_channels),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.cnn_channels, cfg.cnn_channels, cfg.kernel_size, padding=padding),
            nn.BatchNorm1d(cfg.cnn_channels),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, cfg: FixedModelConfig = FIXED):
        super().__init__()
        self.conv = ConvBlock(input_dim, cfg)
        self.bilstm = nn.LSTM(
            input_size=cfg.cnn_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = hidden_size * 2
        self.attention = TemporalAttention(out_dim, cfg.attention_dim, cfg.dropout)
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        out, _ = self.bilstm(x)
        context, _ = self.attention(out)
        return self.regressor(context).squeeze(-1)


# -----------------------------------------------------------------------------
# Data management
# -----------------------------------------------------------------------------

REQUIRED_DATA_FILES = (
    "Y_tensor_aligned.npy",
    "Y_tensor_raw.npy",
    "Y_scaler_params.npy",
    "X_scaler_params.npz",
    "Sample_index_aligned.csv",
    PREPARED_METADATA_NAME,
)
CRITICAL_DATA_HASH_KEYS = (
    "X_selected_window",
    "Y",
    "Y_raw",
    "Y_scaler",
    "X_scaler",
    "Sample_index",
    "prepared_metadata",
)
PRE_EVALUATION_FORBIDDEN_ATTRIBUTES = (
    "y_raw",
    "sample_index",
    "_test_idx",
    "test_idx",
    "evaluation_idx",
    "X_evaluation",
    "y_evaluation",
    "y_raw_evaluation",
    "evaluation_index",
)


def _load_and_validate_data_identity(
    data_dir: Path,
    window: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Path]]:
    data_dir = require_prepared_data_dir(data_dir)
    if int(window) not in DEFAULT_WINDOWS:
        raise ValueError(f"Window must be one of {list(DEFAULT_WINDOWS)}; got {window}")

    paths = {name: data_dir / name for name in REQUIRED_DATA_FILES}
    paths["X"] = data_dir / f"X_tensor_aligned_{int(window)}.npy"
    for selected_window in DEFAULT_WINDOWS:
        paths[f"X_W{selected_window}"] = data_dir / f"X_tensor_aligned_{selected_window}.npy"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared input package is incomplete: {missing}")

    metadata_path = paths[PREPARED_METADATA_NAME]
    metadata = load_json(metadata_path)
    expected_metadata = {
        "version": PREPARED_DATA_VERSION,
        "output_dir": "data/processed",
        "max_window": 48,
        "sample_count": EXPECTED_TOTAL_SAMPLES,
        "point_count": EXPECTED_POINT_COUNT,
        "year_range": [2005, 2025],
        "train_end_year": 2018,
        "val_end_year": 2021,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Prepared metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}")
    if metadata.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Prepared metadata split_counts mismatch: {metadata.get('split_counts')!r}"
        )
    if tuple(metadata.get("all_window_sizes_recommended", ())) != DEFAULT_WINDOWS:
        raise ValueError("Prepared metadata does not identify exactly the six candidate windows")

    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != EXPECTED_FEATURE_DIM:
        raise ValueError("Prepared metadata must provide exactly 19 ordered feature_names")
    if len(set(feature_names)) != EXPECTED_FEATURE_DIM or not all(
        isinstance(name, str) and name for name in feature_names
    ):
        raise ValueError("Prepared metadata feature_names must be 19 unique non-empty strings")

    index_df = pd.read_csv(paths["Sample_index_aligned.csv"])
    required_columns = {"point_id", "Year", "split"}
    if not required_columns.issubset(index_df.columns):
        raise ValueError(f"Sample index missing columns: {sorted(required_columns - set(index_df.columns))}")
    if len(index_df) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_SAMPLES} samples, got {len(index_df)}")

    split = index_df["split"].astype(str).str.lower()
    if set(split.unique()) != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError(f"Unexpected split labels: {sorted(split.unique())}")
    actual_counts = {name: int((split == name).sum()) for name in EXPECTED_SPLIT_COUNTS}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Split counts mismatch: {actual_counts}")

    years = pd.to_numeric(index_df["Year"], errors="raise")
    if not np.isfinite(years.to_numpy(dtype=float)).all() or not (years == years.astype(int)).all():
        raise ValueError("Sample-index years must be finite integers")
    for split_name, expected_years in EXPECTED_YEAR_SETS.items():
        actual_years = tuple(sorted(set(years[split == split_name].astype(int).tolist())))
        if actual_years != expected_years:
            raise ValueError(
                f"{split_name} years mismatch: expected {expected_years}, got {actual_years}"
            )
    if int(index_df["point_id"].nunique()) != EXPECTED_POINT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_POINT_COUNT} fixed points, got {index_df['point_id'].nunique()}"
        )

    for selected_window in DEFAULT_WINDOWS:
        tensor_shape = np.load(paths[f"X_W{selected_window}"], mmap_mode="r").shape
        expected_shape = (
            EXPECTED_TOTAL_SAMPLES,
            selected_window,
            EXPECTED_FEATURE_DIM,
        )
        if tensor_shape != expected_shape:
            raise ValueError(
                f"W{selected_window} tensor shape mismatch: {tensor_shape} != {expected_shape}"
            )
    x_shape = np.load(paths["X"], mmap_mode="r").shape
    y_shape = np.load(paths["Y_tensor_aligned.npy"], mmap_mode="r").shape
    y_raw_shape = np.load(paths["Y_tensor_raw.npy"], mmap_mode="r").shape
    if x_shape != (EXPECTED_TOTAL_SAMPLES, int(window), EXPECTED_FEATURE_DIM):
        raise ValueError(f"Selected-window tensor shape mismatch: {x_shape}")
    if y_shape[0] != EXPECTED_TOTAL_SAMPLES or y_raw_shape[0] != EXPECTED_TOTAL_SAMPLES:
        raise ValueError(f"Target-array sample count mismatch: y={y_shape}, y_raw={y_raw_shape}")

    identity = {
        "resolved_prepared_data_path": str(data_dir),
        "prepared_data_project_relative_path": PREPARED_DATA_PROJECT_RELATIVE_PATH.as_posix(),
        "metadata_file": str(metadata_path),
        "metadata_identity": metadata["version"],
        "ordered_feature_names": feature_names,
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "point_count": EXPECTED_POINT_COUNT,
        "total_samples": EXPECTED_TOTAL_SAMPLES,
        "split_counts": {
            "Train": EXPECTED_SPLIT_COUNTS["train"],
            "Validation": EXPECTED_SPLIT_COUNTS["val"],
            "Evaluation": EXPECTED_SPLIT_COUNTS["test"],
        },
        "year_splits": {
            "Train": "2005-2018",
            "Validation": "2019-2021",
            "Evaluation": "2022-2025",
        },
        "evaluation_terminology": "chronologically held-out evaluation period (2022–2025)",
        "source_split_label_for_evaluation": "test",
        "selected_window": int(window),
    }
    return identity, index_df, paths


def _critical_data_hashes(paths: Dict[str, Path]) -> Dict[str, str]:
    return {
        "X_selected_window": sha256_file(paths["X"]),
        "Y": sha256_file(paths["Y_tensor_aligned.npy"]),
        "Y_raw": sha256_file(paths["Y_tensor_raw.npy"]),
        "Y_scaler": sha256_file(paths["Y_scaler_params.npy"]),
        "X_scaler": sha256_file(paths["X_scaler_params.npz"]),
        "Sample_index": sha256_file(paths["Sample_index_aligned.csv"]),
        "prepared_metadata": sha256_file(paths[PREPARED_METADATA_NAME]),
    }


def _load_current_data_lineage(
    data_dir: Path,
    window: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    identity, _, paths = _load_and_validate_data_identity(data_dir, window)
    return identity, _critical_data_hashes(paths)


def _validate_prepared_data_record(
    manifest: Dict[str, Any],
    selected_window: int,
) -> None:
    identity = manifest.get("prepared_data_identity")
    hashes = manifest.get("prepared_data_hashes")
    if not isinstance(identity, dict) or not isinstance(hashes, dict):
        raise ValueError("Window manifest lacks a recorded data identity/hash lock")
    if require_prepared_data_dir(Path(identity.get("resolved_prepared_data_path", ""))) != PREPARED_DATA_DIR:
        raise ValueError("Data identity does not use the prepared input path")
    if identity.get("prepared_data_project_relative_path") != PREPARED_DATA_PROJECT_RELATIVE_PATH.as_posix():
        raise ValueError("Data identity has the wrong project-relative prepared input path")
    if Path(identity.get("metadata_file", "")).resolve() != (
        PREPARED_DATA_DIR / PREPARED_METADATA_NAME
    ).resolve():
        raise ValueError("Data identity points to the wrong prepared metadata file")
    if identity.get("metadata_identity") != PREPARED_DATA_VERSION:
        raise ValueError("Data identity has the wrong metadata version")
    feature_names = identity.get("ordered_feature_names")
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != EXPECTED_FEATURE_DIM
        or len(set(feature_names)) != EXPECTED_FEATURE_DIM
        or not all(isinstance(name, str) and name for name in feature_names)
    ):
        raise ValueError("Data identity lacks 19 valid ordered feature names")
    expected_identity = {
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "point_count": EXPECTED_POINT_COUNT,
        "total_samples": EXPECTED_TOTAL_SAMPLES,
        "split_counts": {
            "Train": EXPECTED_SPLIT_COUNTS["train"],
            "Validation": EXPECTED_SPLIT_COUNTS["val"],
            "Evaluation": EXPECTED_SPLIT_COUNTS["test"],
        },
        "year_splits": {
            "Train": "2005-2018",
            "Validation": "2019-2021",
            "Evaluation": "2022-2025",
        },
        "selected_window": int(selected_window),
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(
                f"Data identity mismatch for {key}: "
                f"{identity.get(key)!r} != {expected!r}"
            )
    if tuple(hashes) != CRITICAL_DATA_HASH_KEYS:
        raise ValueError("Data hash record has missing, extra, or reordered entries")
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
        raise ValueError("Data hash record contains an invalid SHA-256 value")


def validate_prepared_data_lineage(
    data_manifest: Dict[str, Any],
    current_identity: Dict[str, Any],
    current_hashes: Dict[str, str],
    context: str,
) -> None:
    selected_window = int(data_manifest.get("selected_window", -1))
    _validate_prepared_data_record(data_manifest, selected_window)
    if current_identity != data_manifest["prepared_data_identity"]:
        raise RuntimeError(f"{context} data identity does not match the window manifest")
    if current_hashes != data_manifest["prepared_data_hashes"]:
        raise RuntimeError(f"{context} data hashes do not match the window manifest")


def validate_prepared_data_reference(
    record: Dict[str, Any],
    data_manifest: Dict[str, Any],
    data_manifest_path: Path,
    context: str,
) -> None:
    recorded_path = Path(record.get("prepared_data_manifest", "")).resolve()
    if recorded_path != data_manifest_path.resolve():
        raise ValueError(f"{context} points to a different data manifest")
    validate_hash(
        data_manifest_path,
        record.get("prepared_data_manifest_sha256", ""),
        f"{context} data manifest",
    )
    if record.get("data_identity") != data_manifest.get("prepared_data_identity"):
        raise RuntimeError(f"{context} data identity diverges from the data manifest")
    if record.get("data_hashes") != data_manifest.get("prepared_data_hashes"):
        raise RuntimeError(f"{context} data hashes diverge from the data manifest")


class PreEvaluationDataBundle:
    """Train/Validation-only view; evaluation targets and indices are discarded."""

    def __init__(self, data_dir: Path, window: int):
        self.data_dir = Path(data_dir).resolve()
        self.window = int(window)
        self.data_identity, index_df, paths = _load_and_validate_data_identity(
            self.data_dir, self.window
        )
        self._paths = paths

        split = index_df["split"].astype(str).str.lower().to_numpy()
        train_mask = split == "train"
        val_mask = split == "val"
        x_all = np.load(paths["X"], mmap_mode="r")
        y_all = np.load(paths["Y_tensor_aligned.npy"], mmap_mode="r")
        self.X_train = np.asarray(x_all[train_mask], dtype=np.float32)
        self.y_train = np.asarray(y_all[train_mask], dtype=np.float32)
        self.X_val = np.asarray(x_all[val_mask], dtype=np.float32)
        self.y_val = np.asarray(y_all[val_mask], dtype=np.float32)
        scaler = np.load(paths["Y_scaler_params.npy"]).astype(np.float64)
        self.y_mean = float(scaler[0])
        self.y_scale = float(scaler[1])

        if len(self.X_train) != EXPECTED_SPLIT_COUNTS["train"]:
            raise ValueError("Train-only view count mismatch")
        if len(self.X_val) != EXPECTED_SPLIT_COUNTS["val"]:
            raise ValueError("Validation-only view count mismatch")
        assert_pre_evaluation_isolation(self)

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[2])

    def data_hashes(self) -> Dict[str, str]:
        return _critical_data_hashes(self._paths)

    def y_scaled_to_g(self, values: np.ndarray) -> np.ndarray:
        y_raw_kg = np.asarray(values, dtype=np.float64) * self.y_scale + self.y_mean
        return y_raw_kg * 1000.0


def assert_pre_evaluation_isolation(data: Any) -> None:
    exposed = [name for name in PRE_EVALUATION_FORBIDDEN_ATTRIBUTES if hasattr(data, name)]
    if exposed:
        raise AssertionError(f"Pre-evaluation object exposes forbidden attributes: {exposed}")
    required = ("X_train", "y_train", "X_val", "y_val")
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise AssertionError(f"Pre-evaluation object lacks Train/Validation views: {missing}")


class EvaluationDataBundle:
    """Evaluation-only high-level view, loaded explicitly inside the evaluate stage."""

    def __init__(self, data_dir: Path, window: int):
        self.data_dir = Path(data_dir).resolve()
        self.window = int(window)
        self.data_identity, index_df, paths = _load_and_validate_data_identity(
            self.data_dir, self.window
        )
        self._paths = paths
        evaluation_mask = index_df["split"].astype(str).str.lower().to_numpy() == "test"
        x_all = np.load(paths["X"], mmap_mode="r")
        y_raw_all = np.load(paths["Y_tensor_raw.npy"], mmap_mode="r")
        self.X_evaluation = np.asarray(x_all[evaluation_mask], dtype=np.float32)
        self.y_raw_evaluation = np.asarray(y_raw_all[evaluation_mask], dtype=np.float64)
        self.evaluation_index = index_df.loc[evaluation_mask].reset_index(drop=True).copy()
        scaler = np.load(paths["Y_scaler_params.npy"]).astype(np.float64)
        self.y_mean = float(scaler[0])
        self.y_scale = float(scaler[1])
        if len(self.X_evaluation) != EXPECTED_SPLIT_COUNTS["test"]:
            raise ValueError("Evaluation-only view count mismatch")

    def data_hashes(self) -> Dict[str, str]:
        return _critical_data_hashes(self._paths)

    def y_scaled_to_g(self, values: np.ndarray) -> np.ndarray:
        y_raw_kg = np.asarray(values, dtype=np.float64) * self.y_scale + self.y_mean
        return y_raw_kg * 1000.0


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=FIXED.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def make_x_loader(
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=FIXED.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        total += float(criterion(pred, yb).item())
        n += int(yb.numel())
    return total / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_model(
    data: PreEvaluationDataBundle,
    hidden_size: int,
    learning_rate: float,
    seed: int,
    max_epochs: int,
    patience: int,
    deterministic: bool = True,
    save_history_path: Optional[Path] = None,
) -> Dict[str, Any]:
    assert_pre_evaluation_isolation(data)
    set_seed(seed, deterministic=deterministic)
    device = get_device()

    train_loader = make_loader(
        data.X_train, data.y_train, FIXED.batch_size, True, seed, device
    )
    val_loader = make_loader(
        data.X_val, data.y_val, FIXED.batch_size, False, seed, device
    )

    model = CNNBiLSTMAttention(data.input_dim, hidden_size, FIXED).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=FIXED.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=FIXED.scheduler_factor,
        patience=FIXED.scheduler_patience,
        min_lr=FIXED.scheduler_min_lr,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FIXED.grad_clip_norm)
            optimizer.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)

        train_mse = total / max(n, 1)
        val_mse = evaluate_mse(model, val_loader, device)
        scheduler.step(val_mse)
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_mse < (best_val - FIXED.min_delta)
        if improved:
            best_val = float(val_mse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1

        history.append({
            "epoch": epoch,
            "train_mse_scaled": train_mse,
            "val_mse_scaled": val_mse,
            "best_val_mse_scaled": best_val,
            "learning_rate": lr_now,
            "improved": bool(improved),
            "patience_counter": wait,
        })

        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid best checkpoint")

    model.load_state_dict(best_state)
    best_val_rmse_g = math.sqrt(best_val) * data.y_scale * 1000.0

    if save_history_path is not None:
        ensure_dir(save_history_path.parent)
        pd.DataFrame(history).to_csv(save_history_path, index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_mse_scaled": best_val,
        "best_val_rmse_gC_m2_month": float(best_val_rmse_g),
        "history": history,
    }


# -----------------------------------------------------------------------------
# Stage 1: validation-only window selection
# -----------------------------------------------------------------------------

def _exact_integer_series(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} values must be finite integers")
    return numeric.astype(int)


def validate_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Window", "seed", "best_val_mse_scaled"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Window summary is missing columns: {sorted(missing)}")

    work = df.loc[:, ["Window", "seed", "best_val_mse_scaled"]].copy()
    work["Window"] = _exact_integer_series(work["Window"], "Window")
    work["seed"] = _exact_integer_series(work["seed"], "seed")
    work["best_val_mse_scaled"] = pd.to_numeric(
        work["best_val_mse_scaled"], errors="raise"
    ).astype(float)
    if not np.isfinite(work["best_val_mse_scaled"].to_numpy()).all():
        raise ValueError("best_val_mse_scaled must contain only finite values")
    if work.duplicated(["Window", "seed"]).any():
        duplicates = work.loc[
            work.duplicated(["Window", "seed"], keep=False), ["Window", "seed"]
        ].to_dict(orient="records")
        raise ValueError(f"Duplicate window-seed pairs are forbidden: {duplicates}")

    expected_pairs = {(window, seed) for window in DEFAULT_WINDOWS for seed in DEFAULT_SEEDS}
    actual_pairs = set(zip(work["Window"].tolist(), work["seed"].tolist()))
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)
    if missing_pairs or extra_pairs or len(work) != len(expected_pairs):
        raise ValueError(
            "Window summary must contain exactly six windows x three fixed seeds; "
            f"missing={missing_pairs}, extra={extra_pairs}, rows={len(work)}"
        )
    return work.sort_values(["Window", "seed"]).reset_index(drop=True)


def select_window_by_exact_rule(work: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    stats = (
        work.groupby("Window", as_index=False)
        .agg(
            n_seeds=("seed", "size"),
            mean_val_mse_scaled=("best_val_mse_scaled", "mean"),
            median_val_mse_scaled=("best_val_mse_scaled", "median"),
            std_val_mse_scaled=("best_val_mse_scaled", "std"),
            min_val_mse_scaled=("best_val_mse_scaled", "min"),
            max_val_mse_scaled=("best_val_mse_scaled", "max"),
        )
        .sort_values(
            ["mean_val_mse_scaled", "median_val_mse_scaled", "Window"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if tuple(sorted(stats["Window"].astype(int))) != DEFAULT_WINDOWS:
        raise AssertionError("Internal error: window statistics are incomplete")
    if not (stats["n_seeds"] == len(DEFAULT_SEEDS)).all():
        raise AssertionError("Internal error: each window must have exactly three seeds")
    return stats, int(stats.iloc[0]["Window"])


def validate_window_manifest(manifest: Dict[str, Any], manifest_path: Path) -> int:
    required = {
        "protocol_version",
        "schema_version",
        "stage",
        "candidate_windows",
        "fixed_seeds",
        "selection_rule",
        "source_file",
        "source_summary_sha256",
        "selected_window",
        "all_validation_stats",
        "prepared_data_identity",
        "prepared_data_hashes",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Window manifest missing fields: {sorted(missing)}")
    if manifest["protocol_version"] != PROTOCOL_VERSION or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Window manifest protocol/schema version does not use the canonical protocol")
    if manifest["stage"] != "validation_only_window_selection":
        raise ValueError("Window manifest stage is not validation_only_window_selection")
    if tuple(manifest["candidate_windows"]) != DEFAULT_WINDOWS:
        raise ValueError("Window manifest candidate_windows mismatch")
    if tuple(manifest["fixed_seeds"]) != DEFAULT_SEEDS:
        raise ValueError("Window manifest fixed_seeds mismatch")
    if tuple(manifest["selection_rule"]) != WINDOW_SELECTION_RULE:
        raise ValueError("Window manifest selection_rule mismatch")
    source = Path(manifest["source_file"])
    validate_hash(source, manifest["source_summary_sha256"], "Window-summary source")
    window = int(manifest["selected_window"])
    if window not in DEFAULT_WINDOWS:
        raise ValueError(f"Window manifest selected an window outside the candidate set: {window}")
    _validate_prepared_data_record(manifest, window)
    source_work = validate_window_summary(pd.read_csv(source))
    recomputed_stats, recomputed_window = select_window_by_exact_rule(source_work)
    if window != recomputed_window:
        raise RuntimeError(
            f"Window manifest selected W{window}, but its locked source recomputes to W{recomputed_window}"
        )
    manifest_stats = manifest["all_validation_stats"]
    if len(manifest_stats) != len(DEFAULT_WINDOWS):
        raise ValueError("Window manifest does not contain six per-window statistics")
    by_window = {int(row["Window"]): row for row in manifest_stats}
    if tuple(sorted(by_window)) != DEFAULT_WINDOWS:
        raise ValueError("Window manifest per-window statistics do not cover the candidate windows")
    for row in recomputed_stats.to_dict(orient="records"):
        recorded = by_window[int(row["Window"])]
        for key in (
            "n_seeds",
            "mean_val_mse_scaled",
            "median_val_mse_scaled",
            "std_val_mse_scaled",
            "min_val_mse_scaled",
            "max_val_mse_scaled",
        ):
            if recorded.get(key) != row[key]:
                raise RuntimeError(
                    f"Window manifest statistic mismatch for W{int(row['Window'])} field {key}"
                )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Window manifest missing: {manifest_path}")
    return window


def cmd_select_window(args: argparse.Namespace) -> None:
    src = Path(args.window_seed_summary)
    out = validate_run_root(Path(args.out_dir))
    stats_path = out / "window_selection_validation_stats.csv"
    manifest_path = out / "window_selection_manifest.json"
    assert_stage_can_start(out, "select-window", [stats_path, manifest_path])
    df = pd.read_csv(src)
    work = validate_window_summary(df)
    stats, selected_window = select_window_by_exact_rule(work)
    winner = stats.iloc[0]
    data_identity, data_hashes = _load_current_data_lineage(
        Path(args.data_dir), selected_window
    )
    ensure_dir(out)
    write_csv_new(stats, stats_path)
    source_hash = sha256_file(src)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "stage": "validation_only_window_selection",
        "created_at": now_iso(),
        "source_file": str(src.resolve()),
        "source_summary_sha256": source_hash,
        "candidate_windows": list(DEFAULT_WINDOWS),
        "fixed_seeds": list(DEFAULT_SEEDS),
        "selection_rule": list(WINDOW_SELECTION_RULE),
        "tie_equality": "exact floating-point equality; no tolerance and no np.isclose",
        "selection_data_scope": {
            "model_selection": ["Train", "Validation"],
            "Evaluation": "held out until final evaluation",
        },
        "selected_window": selected_window,
        "selected_window_stats": winner.to_dict(),
        "all_validation_stats": stats.to_dict(orient="records"),
        "prepared_data_identity": data_identity,
        "prepared_data_hashes": data_hashes,
        "prepared_data_stage": "validation_only_window_selection",
        "source_summary_hash": source_hash,
        "environment": environment_manifest(),
    }
    save_json(manifest, manifest_path)
    print(stats.to_string(index=False))
    print(f"\nSELECTED WINDOW (validation-only rule): W={selected_window}")
    print(f"Manifest: {manifest_path}")


# -----------------------------------------------------------------------------
# Stage 2: fixed-window RIME search on Train/Validation only
# -----------------------------------------------------------------------------

def cmd_rime_search(args: argparse.Namespace) -> None:
    run_root = validate_run_root(Path(args.out_dir))
    out = run_root / "rime_search"
    assert_stage_can_start(run_root, "rime-search", [out])

    try:
        from mealpy import FloatVar, IntegerVar
        from mealpy.physics_based.RIME import OriginalRIME
    except Exception as exc:
        raise RuntimeError(
            "The runtime environment does not currently provide mealpy; refusing RIME. "
            "Provisioning the environment requires separate approval."
        ) from exc
    selection_path = Path(args.window_selection).resolve()
    if selection_path != (run_root / "window_selection_manifest.json").resolve():
        raise ValueError("RIME accepts only the window manifest inside the same run root")
    selection = load_json(selection_path)
    window = validate_window_manifest(selection, selection_path)
    data = PreEvaluationDataBundle(Path(args.data_dir), window=window)
    assert_pre_evaluation_isolation(data)
    data_hashes = data.data_hashes()
    validate_prepared_data_lineage(
        selection, data.data_identity, data_hashes, "RIME"
    )
    ensure_dir(out)
    started_rime = time.time()

    history: List[Dict[str, Any]] = []
    cache: Dict[Tuple[int, float], float] = {}
    eval_counter = 0

    def decode(solution: Sequence[float]) -> Dict[str, Any]:
        log10_lr = float(np.clip(float(solution[0]), RIME_FIXED.log10_lr_min, RIME_FIXED.log10_lr_max))
        hidden_idx = int(np.clip(int(round(float(solution[1]))), 0, len(RIME_FIXED.hidden_options) - 1))
        return {
            "window": window,
            "log10_lr": log10_lr,
            "learning_rate": float(10 ** log10_lr),
            "hidden_idx": hidden_idx,
            "hidden_size": int(RIME_FIXED.hidden_options[hidden_idx]),
        }

    def objective(solution: Sequence[float]) -> float:
        nonlocal eval_counter
        hp = decode(solution)
        # Cache at 8-decimal log10 precision to avoid repeated identical discrete/near-identical candidates.
        key = (hp["hidden_size"], round(hp["log10_lr"], 8))
        if key in cache:
            return cache[key]

        eval_counter += 1
        started = time.time()
        result = train_model(
            data=data,
            hidden_size=hp["hidden_size"],
            learning_rate=hp["learning_rate"],
            seed=RIME_FIXED.search_seed,
            max_epochs=RIME_FIXED.candidate_max_epochs,
            patience=RIME_FIXED.candidate_patience,
            deterministic=True,
        )
        fitness = float(result["best_val_rmse_gC_m2_month"])
        row = {
            "eval_id": eval_counter,
            **hp,
            "search_seed": RIME_FIXED.search_seed,
            "best_epoch": result["best_epoch"],
            "best_val_mse_scaled": result["best_val_mse_scaled"],
            "best_val_rmse_gC_m2_month": fitness,
            "runtime_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "rime_search_history_partial.csv", index=False, encoding="utf-8-sig")
        cache[key] = fitness
        print(
            f"Eval {eval_counter:03d}: hidden={hp['hidden_size']} "
            f"lr={hp['learning_rate']:.8f} | Val RMSE={fitness:.4f} gC m^-2 month^-1"
        )
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fitness

    bounds = [
        FloatVar(lb=RIME_FIXED.log10_lr_min, ub=RIME_FIXED.log10_lr_max, name="log10_lr"),
        IntegerVar(lb=0, ub=len(RIME_FIXED.hidden_options) - 1, name="hidden_idx"),
    ]
    problem = {"bounds": bounds, "minmax": "min", "obj_func": objective}

    set_seed(RIME_FIXED.search_seed, deterministic=True)
    optimizer = OriginalRIME(epoch=RIME_FIXED.rime_epochs, pop_size=RIME_FIXED.population_size)
    best_agent = optimizer.solve(problem, seed=RIME_FIXED.search_seed)
    best_hp = decode(best_agent.solution)
    best_hp["best_validation_fitness_gC_m2_month"] = float(best_agent.target.fitness)
    best_hp["selection_data"] = "Train/Validation only; evaluation data excluded"
    best_hp["protocol_version"] = PROTOCOL_VERSION
    best_hp["schema_version"] = SCHEMA_VERSION
    best_hp["stage"] = "fixed_window_rime_best_hyperparameters"
    best_hp["window_selection_manifest"] = str(selection_path)
    best_hp["window_selection_manifest_sha256"] = sha256_file(selection_path)
    best_hp["data_hashes"] = data_hashes

    hist_df = pd.DataFrame(history).sort_values("best_val_rmse_gC_m2_month")
    write_csv_new(hist_df, out / "rime_search_history.csv")
    best_hp_path = out / "best_hyperparameters.json"
    save_json(best_hp, best_hp_path)
    rime_runtime = time.time() - started_rime
    save_json({
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "stage": "fixed_window_rime_search",
        "created_at": now_iso(),
        "parent_window_manifest": str(selection_path),
        "parent_window_manifest_sha256": sha256_file(selection_path),
        "prepared_data_manifest": str(selection_path),
        "prepared_data_manifest_sha256": sha256_file(selection_path),
        "window_selection_manifest": selection,
        "fixed_model_config": asdict(FIXED),
        "rime_config": asdict(RIME_FIXED),
        "search_space": {
            "historical_window": [window],
            "learning_rate_log10": [RIME_FIXED.log10_lr_min, RIME_FIXED.log10_lr_max],
            "hidden_size": list(RIME_FIXED.hidden_options),
        },
        "fitness": "minimum validation RMSE; target transformed back to g C m^-2 month^-1",
        "evaluation_usage": "none; pre-evaluation object exposes Train/Validation only",
        "pre_evaluation_isolation_asserted": True,
        "data_identity": data.data_identity,
        "data_hashes": data_hashes,
        "mealpy_rng": {
            "mealpy_version": package_version("mealpy"),
            "search_seed": RIME_FIXED.search_seed,
            "explicit_mealpy_seed_api": "Optimizer.solve(problem, seed=RIME_FIXED.search_seed)",
            "seed_argument_passed": True,
            "controls": [
                "mealpy Optimizer.solve seed",
                "random.seed",
                "numpy.random.seed",
                "torch.manual_seed",
                "torch.cuda.manual_seed_all",
            ],
            "reproducibility_status": REPRODUCIBILITY_STATUS,
        },
        "environment": environment_manifest(get_device()),
        "rime_runtime_seconds": rime_runtime,
        "best_hyperparameters_file": str(best_hp_path.resolve()),
        "best_hyperparameters_sha256": sha256_file(best_hp_path),
        "best_hyperparameters": best_hp,
        "n_unique_candidate_evaluations": len(history),
    }, out / "rime_search_manifest.json")

    print("\nRIME SEARCH COMPLETE")
    print(json.dumps(best_hp, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# Stage 3: final multi-seed training, validation only
# -----------------------------------------------------------------------------

def cmd_train_seeds(args: argparse.Namespace) -> None:
    run_root = validate_run_root(Path(args.out_dir))
    out = run_root / "final_seed_runs"
    assert_stage_can_start(run_root, "train-seeds", [out])
    hp_path = Path(args.best_hp).resolve()
    rime_manifest_path = Path(args.rime_manifest).resolve()
    expected_rime_dir = (run_root / "rime_search").resolve()
    if hp_path.parent != expected_rime_dir or rime_manifest_path.parent != expected_rime_dir:
        raise ValueError("train-seeds accepts only RIME outputs inside the same run root")
    hp = load_json(hp_path)
    rime_manifest = load_json(rime_manifest_path)
    if hp.get("protocol_version") != PROTOCOL_VERSION or hp.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Best-hyperparameter file protocol/schema version does not match this public pipeline")
    if hp.get("stage") != "fixed_window_rime_best_hyperparameters":
        raise ValueError("Best-hyperparameter file has an invalid stage")
    if (
        rime_manifest.get("protocol_version") != PROTOCOL_VERSION
        or rime_manifest.get("schema_version") != SCHEMA_VERSION
        or rime_manifest.get("stage") != "fixed_window_rime_search"
    ):
        raise ValueError("RIME manifest lineage does not use the canonical protocol")
    validate_hash(hp_path, rime_manifest["best_hyperparameters_sha256"], "Best hyperparameters")
    if Path(rime_manifest["best_hyperparameters_file"]).resolve() != hp_path:
        raise ValueError("RIME manifest points to a different best-hyperparameter file")
    parent_window_path = Path(rime_manifest["parent_window_manifest"]).resolve()
    validate_hash(
        parent_window_path,
        rime_manifest["parent_window_manifest_sha256"],
        "Parent window manifest",
    )
    parent_window = load_json(parent_window_path)
    selected_window = validate_window_manifest(parent_window, parent_window_path)
    if parent_window_path != (run_root / "window_selection_manifest.json").resolve():
        raise ValueError("RIME lineage points outside the same run's window manifest")
    validate_prepared_data_reference(
        rime_manifest, parent_window, parent_window_path, "RIME manifest"
    )
    window = int(hp["window"])
    if window != selected_window:
        raise ValueError("Best hyperparameters do not use the selected window")
    if rime_manifest.get("best_hyperparameters") != hp:
        raise RuntimeError("RIME manifest and best-hyperparameter file disagree")
    hidden_size = int(hp["hidden_size"])
    learning_rate = float(hp["learning_rate"])
    seeds = assert_exact_seeds((int(x) for x in args.seeds.split(",")), "train-seeds")

    data = PreEvaluationDataBundle(Path(args.data_dir), window=window)
    assert_pre_evaluation_isolation(data)
    data_hashes = data.data_hashes()
    validate_prepared_data_lineage(
        parent_window, data.data_identity, data_hashes, "Three-seed training"
    )
    if data_hashes != rime_manifest.get("data_hashes") or data.data_identity != rime_manifest.get("data_identity"):
        raise RuntimeError("Current data do not match the locked RIME data lineage")
    ensure_dir(out)
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = out / f"seed_{seed}"
        ensure_dir(seed_dir)
        started = time.time()
        result = train_model(
            data=data,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            seed=seed,
            max_epochs=FIXED.max_epochs,
            patience=FIXED.early_stopping_patience,
            deterministic=True,
            save_history_path=seed_dir / "training_history.csv",
        )

        checkpoint = {
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "model_name": "CNN-BiLSTM-Attention",
            "window": window,
            "seed": seed,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "fixed_model_config": asdict(FIXED),
            "best_epoch": result["best_epoch"],
            "best_val_mse_scaled": result["best_val_mse_scaled"],
            "best_val_rmse_gC_m2_month": result["best_val_rmse_gC_m2_month"],
            "input_dim": data.input_dim,
            "y_mean": data.y_mean,
            "y_scale": data.y_scale,
            "data_identity": data.data_identity,
            "data_hashes": data_hashes,
            "model_state_dict": result["best_state"],
            "evaluation_evaluated": False,
        }
        ckpt_path = seed_dir / "model_validation_selected.pt"
        if ckpt_path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {ckpt_path}")
        torch.save(checkpoint, ckpt_path)

        row = {
            "seed": seed,
            "window": window,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "best_epoch": result["best_epoch"],
            "best_val_mse_scaled": result["best_val_mse_scaled"],
            "best_val_rmse_gC_m2_month": result["best_val_rmse_gC_m2_month"],
            "runtime_seconds": time.time() - started,
            "checkpoint": str(ckpt_path.resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "evaluation_evaluated": False,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "seed_validation_summary_partial.csv", index=False, encoding="utf-8-sig")
        print(
            f"seed={seed}: Val RMSE={row['best_val_rmse_gC_m2_month']:.4f} "
            f"gC m^-2 month^-1 (epoch {row['best_epoch']})"
        )
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows).sort_values("seed")
    summary_path = out / "seed_validation_summary.csv"
    write_csv_new(summary, summary_path)
    save_json({
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "stage": "final_multiseed_validation",
        "created_at": now_iso(),
        "parent_rime_manifest": str(rime_manifest_path),
        "parent_rime_manifest_sha256": sha256_file(rime_manifest_path),
        "prepared_data_manifest": str(parent_window_path),
        "prepared_data_manifest_sha256": sha256_file(parent_window_path),
        "best_hyperparameters_file": str(hp_path),
        "best_hyperparameters_sha256": sha256_file(hp_path),
        "hyperparameters": hp,
        "seeds": list(seeds),
        "evaluation_usage": "none; pre-evaluation object exposes Train/Validation only",
        "pre_evaluation_isolation_asserted": True,
        "data_identity": data.data_identity,
        "data_hashes": data_hashes,
        "seed_summary_file": str(summary_path.resolve()),
        "seed_summary_sha256": sha256_file(summary_path),
        "environment": environment_manifest(get_device()),
        "seed_results": rows,
    }, out / "seed_training_manifest.json")


# -----------------------------------------------------------------------------
# Stage 4: validate and lock all three seed checkpoints
# -----------------------------------------------------------------------------

def validate_seed_summary_for_lock(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "best_val_rmse_gC_m2_month",
        "checkpoint",
        "checkpoint_sha256",
        "window",
        "hidden_size",
        "learning_rate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Seed summary missing columns: {sorted(missing)}")
    work = df.copy()
    work["seed"] = _exact_integer_series(work["seed"], "seed")
    work["best_val_rmse_gC_m2_month"] = pd.to_numeric(
        work["best_val_rmse_gC_m2_month"], errors="raise"
    ).astype(float)
    if not np.isfinite(work["best_val_rmse_gC_m2_month"].to_numpy()).all():
        raise ValueError("All three Validation RMSE values must be finite")
    if work["seed"].duplicated().any():
        raise ValueError("Duplicate seeds are forbidden in the seed summary")
    if tuple(sorted(work["seed"].tolist())) != tuple(sorted(DEFAULT_SEEDS)) or len(work) != 3:
        raise ValueError(f"Seed summary must contain exactly seeds {list(DEFAULT_SEEDS)}")
    for column in ("window", "hidden_size", "learning_rate"):
        if work[column].nunique(dropna=False) != 1:
            raise ValueError(f"All three seed runs must share the same {column}")
    return work.sort_values("seed").reset_index(drop=True)


def cmd_lock_seeds(args: argparse.Namespace) -> None:
    run_root = validate_run_root(Path(args.out_dir))
    lock_path = run_root / "pre_evaluation_lock_manifest.json"
    assert_stage_can_start(run_root, "lock-seeds", [lock_path])
    ensure_dir(run_root)

    src = Path(args.seed_summary).resolve()
    training_manifest_path = Path(args.seed_training_manifest).resolve()
    expected_seed_dir = (run_root / "final_seed_runs").resolve()
    if src.parent != expected_seed_dir or training_manifest_path.parent != expected_seed_dir:
        raise ValueError("lock-seeds accepts only seed outputs inside the same run root")

    training_manifest = load_json(training_manifest_path)
    if (
        training_manifest.get("protocol_version") != PROTOCOL_VERSION
        or training_manifest.get("schema_version") != SCHEMA_VERSION
        or training_manifest.get("stage") != "final_multiseed_validation"
    ):
        raise ValueError("Seed-training manifest lineage does not use the canonical protocol")
    assert_exact_seeds(training_manifest.get("seeds", ()), "seed-training manifest")
    validate_hash(src, training_manifest["seed_summary_sha256"], "Seed validation summary")
    validate_hash(
        Path(training_manifest["parent_rime_manifest"]),
        training_manifest["parent_rime_manifest_sha256"],
        "Parent RIME manifest",
    )

    data_manifest_path = Path(training_manifest["prepared_data_manifest"]).resolve()
    if data_manifest_path != (run_root / "window_selection_manifest.json").resolve():
        raise ValueError("Seed-training lineage points outside the same run's window manifest")
    data_manifest = load_json(data_manifest_path)
    selected_window = validate_window_manifest(data_manifest, data_manifest_path)
    validate_prepared_data_reference(
        training_manifest, data_manifest, data_manifest_path, "Seed-training manifest"
    )
    current_identity, current_hashes = _load_current_data_lineage(
        Path(args.data_dir), selected_window
    )
    validate_prepared_data_lineage(
        data_manifest, current_identity, current_hashes, "Seed lock"
    )

    work = validate_seed_summary_for_lock(pd.read_csv(src))
    checkpoint_records: List[Dict[str, Any]] = []
    for row in work.itertuples():
        checkpoint_path = Path(str(row.checkpoint)).resolve()
        expected_checkpoint_path = (
            expected_seed_dir / f"seed_{int(row.seed)}" / "model_validation_selected.pt"
        ).resolve()
        if checkpoint_path != expected_checkpoint_path:
            raise ValueError(
                f"Seed {int(row.seed)} checkpoint is outside its expected run directory"
            )
        expected_hash = str(row.checkpoint_sha256)
        validate_hash(checkpoint_path, expected_hash, f"Seed {int(row.seed)} checkpoint")
        checkpoint_records.append({
            "seed": int(row.seed),
            "path": str(checkpoint_path),
            "sha256": expected_hash,
            "validation_rmse_gC_m2_month": float(row.best_val_rmse_gC_m2_month),
        })

    lock_identity, lock_hashes = _load_current_data_lineage(
        Path(args.data_dir), selected_window
    )
    validate_prepared_data_lineage(
        data_manifest, lock_identity, lock_hashes, "Pre-evaluation lock creation"
    )
    hp = training_manifest["hyperparameters"]
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "stage": "pre_evaluation_lock",
        "created_at": now_iso(),
        "run_id": run_root.name,
        "run_root": str(run_root),
        "state": "NOT_EVALUATED",
        "final_selected_window": int(hp["window"]),
        "rime_hyperparameters": {
            "learning_rate": float(hp["learning_rate"]),
            "log10_lr": float(hp["log10_lr"]),
            "hidden_size": int(hp["hidden_size"]),
        },
        "architecture": {
            "model_name": "CNN-BiLSTM-Attention",
            "fixed_model_config": asdict(FIXED),
        },
        "seeds": list(DEFAULT_SEEDS),
        "checkpoints": checkpoint_records,
        "data_identity": training_manifest["data_identity"],
        "data_hashes": training_manifest["data_hashes"],
        "code_file": str(SCRIPT_PATH),
        "code_sha256": sha256_file(SCRIPT_PATH),
        "seed_summary": str(src),
        "seed_summary_sha256": sha256_file(src),
        "seed_training_manifest": str(training_manifest_path),
        "seed_training_manifest_sha256": sha256_file(training_manifest_path),
        "parent_rime_manifest": training_manifest["parent_rime_manifest"],
        "parent_rime_manifest_sha256": training_manifest["parent_rime_manifest_sha256"],
        "prepared_data_manifest": str(data_manifest_path),
        "prepared_data_manifest_sha256": sha256_file(data_manifest_path),
        "evaluation_period": "chronologically held-out evaluation period (2022–2025)",
        "main_performance_reporting": "3-seed mean +/- sample SD (ddof=1)",
    }
    save_json(lock, lock_path)
    print("THREE SEED CHECKPOINTS LOCKED")
    print(f"Pre-evaluation lock: {lock_path}")


# -----------------------------------------------------------------------------
# Stage 5: explicit evaluation of all three locked checkpoints
# -----------------------------------------------------------------------------

def validate_pre_evaluation_lock(
    lock: Dict[str, Any],
    lock_path: Path,
    run_root: Path,
) -> List[Dict[str, Any]]:
    if (
        lock.get("protocol_version") != PROTOCOL_VERSION
        or lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("stage") != "pre_evaluation_lock"
    ):
        raise ValueError("Pre-evaluation lock protocol/schema/stage is invalid")
    if lock.get("state") != "NOT_EVALUATED":
        raise RuntimeError(
            f"Run state is {lock.get('state')!r}; evaluation requires NOT_EVALUATED"
        )
    if Path(lock.get("run_root", "")).resolve() != run_root:
        raise ValueError("Lock manifest run_root does not match --out-dir")
    if lock_path.resolve() != (run_root / "pre_evaluation_lock_manifest.json").resolve():
        raise ValueError("Evaluation accepts only the lock manifest inside the same run root")
    assert_exact_seeds(lock.get("seeds", ()), "pre-evaluation lock")
    if int(lock.get("final_selected_window", -1)) not in DEFAULT_WINDOWS:
        raise ValueError("Locked window is outside the candidate set")

    validate_hash(Path(lock["code_file"]), lock["code_sha256"], "Main pipeline code")
    validate_hash(Path(lock["seed_summary"]), lock["seed_summary_sha256"], "Seed summary")
    validate_hash(
        Path(lock["seed_training_manifest"]),
        lock["seed_training_manifest_sha256"],
        "Seed-training manifest",
    )
    validate_hash(
        Path(lock["parent_rime_manifest"]),
        lock["parent_rime_manifest_sha256"],
        "RIME manifest",
    )
    data_manifest_path = Path(lock["prepared_data_manifest"]).resolve()
    if data_manifest_path != (run_root / "window_selection_manifest.json").resolve():
        raise ValueError("Pre-evaluation lock points outside the same run's window manifest")
    data_manifest = load_json(data_manifest_path)
    selected_window = validate_window_manifest(data_manifest, data_manifest_path)
    if selected_window != int(lock["final_selected_window"]):
        raise ValueError("Pre-evaluation lock window diverges from the canonical manifest")
    validate_prepared_data_reference(
        lock, data_manifest, data_manifest_path, "Pre-evaluation lock"
    )

    checkpoints = lock.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError("Lock must contain exactly three checkpoints")
    if tuple(int(item["seed"]) for item in checkpoints) != DEFAULT_SEEDS:
        raise ValueError("Locked checkpoints must be ordered as seeds 42, 2024, 3407")
    for item in checkpoints:
        validate_hash(
            Path(item["path"]),
            item["sha256"],
            f"Seed {int(item['seed'])} checkpoint",
        )
    return checkpoints


def cmd_evaluate(args: argparse.Namespace) -> None:
    if args.confirm_evaluation != EVALUATION_CONFIRMATION_TOKEN:
        raise RuntimeError(
            "Chronologically held-out evaluation is locked. To rerun, create a new run directory and use:\n"
            f"  --confirm-evaluation {EVALUATION_CONFIRMATION_TOKEN}"
        )

    run_root = validate_run_root(Path(args.out_dir))
    out = run_root / "evaluation"
    assert_stage_can_start(run_root, "evaluate", [out])
    lock_path = Path(args.lock_manifest).resolve()
    lock = load_json(lock_path)
    checkpoints = validate_pre_evaluation_lock(lock, lock_path, run_root)
    lock_hash_before_evaluation = sha256_file(lock_path)

    window = int(lock["final_selected_window"])
    data = EvaluationDataBundle(Path(args.data_dir), window=window)
    data_hashes = data.data_hashes()
    data_manifest_path = Path(lock["prepared_data_manifest"]).resolve()
    data_manifest = load_json(data_manifest_path)
    validate_prepared_data_lineage(
        data_manifest, data.data_identity, data_hashes, "Evaluation"
    )
    if data_hashes != lock["data_hashes"]:
        raise RuntimeError("Evaluation data hashes do not match the pre-evaluation lock")
    if data.data_identity != lock["data_identity"]:
        raise RuntimeError("Evaluation data identity does not match the pre-evaluation lock")

    ensure_dir(out)
    device = get_device()
    evaluation_loader = make_x_loader(data.X_evaluation, FIXED.batch_size, device)
    y_true_g = data.y_raw_evaluation * 1000.0
    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    for locked in checkpoints:
        seed = int(locked["seed"])
        checkpoint = torch.load(Path(locked["path"]), map_location="cpu", weights_only=False)
        if int(checkpoint["seed"]) != seed or int(checkpoint["window"]) != window:
            raise RuntimeError(f"Seed {seed} checkpoint metadata conflicts with the lock")
        locked_hp = lock["rime_hyperparameters"]
        if (
            checkpoint.get("protocol_version") != PROTOCOL_VERSION
            or checkpoint.get("schema_version") != SCHEMA_VERSION
            or int(checkpoint["hidden_size"]) != int(locked_hp["hidden_size"])
            or float(checkpoint["learning_rate"]) != float(locked_hp["learning_rate"])
            or checkpoint.get("fixed_model_config") != lock["architecture"]["fixed_model_config"]
            or checkpoint.get("data_hashes") != lock["data_hashes"]
            or checkpoint.get("data_identity") != lock["data_identity"]
            or int(checkpoint["input_dim"]) != EXPECTED_FEATURE_DIM
            or float(checkpoint["y_mean"]) != data.y_mean
            or float(checkpoint["y_scale"]) != data.y_scale
        ):
            raise RuntimeError(f"Seed {seed} checkpoint metadata/lineage conflicts with the lock")
        model = CNNBiLSTMAttention(
            input_dim=int(checkpoint["input_dim"]),
            hidden_size=int(checkpoint["hidden_size"]),
            cfg=FIXED,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        pred_scaled = predict(model, evaluation_loader, device)
        y_pred_g = data.y_scaled_to_g(pred_scaled)
        seed_metrics = metrics(y_true_g, y_pred_g)
        metric_rows.append({
            "seed": seed,
            **{name: seed_metrics[name] for name in ("RMSE", "MAE", "R2", "Bias", "Pearson_r")},
        })
        pred_df = data.evaluation_index.copy()
        pred_df.insert(0, "seed", seed)
        pred_df["y_true_gC_m2_month"] = y_true_g
        pred_df["y_pred_gC_m2_month"] = y_pred_g
        pred_df["residual_gC_m2_month"] = y_pred_g - y_true_g
        pred_df["y_pred_scaled"] = pred_scaled
        prediction_frames.append(pred_df)
        del model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metric_rows).sort_values("seed").reset_index(drop=True)
    assert_exact_seeds(metrics_df["seed"].tolist(), "evaluation metrics")
    metric_names = ["RMSE", "MAE", "R2", "Bias", "Pearson_r"]
    aggregate = {
        name: {
            "mean": float(metrics_df[name].mean()),
            "sample_sd_ddof_1": float(metrics_df[name].std(ddof=1)),
        }
        for name in metric_names
    }
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_path = out / "evaluation_metrics_by_seed.csv"
    predictions_path = out / "evaluation_predictions_all_seeds.csv"
    summary_path = out / "evaluation_summary.json"
    write_csv_new(metrics_df, metrics_path)
    write_csv_new(predictions, predictions_path)

    evaluation_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "stage": "three_seed_chronological_evaluation",
        "created_at": now_iso(),
        "evaluation_period": "chronologically held-out evaluation period (2022–2025)",
        "source_split_label": "test",
        "selection_locked_before_evaluation": True,
        "pre_evaluation_lock_sha256": lock_hash_before_evaluation,
        "main_performance_reporting": "3-seed mean +/- sample SD (ddof=1)",
        "window": window,
        "metrics_unit": "g C m^-2 month^-1 except R2 and Pearson_r",
        "per_seed_metrics": metric_rows,
        "three_seed_aggregate": aggregate,
        "sample_sd_ddof": 1,
        "metrics_file": str(metrics_path.resolve()),
        "predictions_file": str(predictions_path.resolve()),
        "data_hashes": data_hashes,
        "environment": environment_manifest(device),
    }
    save_json(evaluation_summary, summary_path)

    lock["state"] = "EVALUATED"
    lock["evaluated_at"] = now_iso()
    lock["evaluation_outputs"] = {
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "metrics_by_seed": str(metrics_path.resolve()),
        "metrics_by_seed_sha256": sha256_file(metrics_path),
        "predictions_all_seeds": str(predictions_path.resolve()),
        "predictions_all_seeds_sha256": sha256_file(predictions_path),
    }
    save_json(lock, lock_path, overwrite=True)

    print("THREE-SEED CHRONOLOGICALLY HELD-OUT EVALUATION COMPLETE")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Predictions: {predictions_path}")


# -----------------------------------------------------------------------------
# Safety/data smoke test
# -----------------------------------------------------------------------------

def cmd_smoke_test(args: argparse.Namespace) -> None:
    selection_path = Path(args.window_selection).resolve()
    selection = load_json(selection_path)
    window = validate_window_manifest(selection, selection_path)
    data = PreEvaluationDataBundle(Path(args.data_dir), window=window)
    assert_pre_evaluation_isolation(data)
    validate_prepared_data_lineage(
        selection, data.data_identity, data.data_hashes(), "Smoke test"
    )
    device = get_device()
    model = CNNBiLSTMAttention(data.input_dim, hidden_size=32, cfg=FIXED).to(device)
    sample_n = min(8, len(data.X_train))
    X = torch.from_numpy(np.asarray(data.X_train[:sample_n], dtype=np.float32)).to(device)
    with torch.no_grad():
        y = model(X)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "window": window,
        "input_shape": list(X.shape),
        "output_shape": list(y.shape),
        "train_n": len(data.X_train),
        "val_n": len(data.X_val),
        "evaluation_attributes_exposed": [
            name for name in PRE_EVALUATION_FORBIDDEN_ATTRIBUTES if hasattr(data, name)
        ],
        "device": str(device),
        "status": "PASS" if y.shape == (sample_n,) else "FAIL",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise RuntimeError("Smoke test failed")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validation-only W36 monthly productivity model pipeline"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select-window", help="Select historical window using validation metrics only")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--window-seed-summary", required=True, help="CSV containing Window, seed, best_val_mse_scaled")
    s.add_argument("--out-dir", required=True, help="Fresh runtime/runs/<run_id> root")
    s.set_defaults(func=cmd_select_window)

    s = sub.add_parser("rime-search", help="Run fixed-window RIME on Train/Validation only")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--window-selection", required=True)
    s.add_argument("--out-dir", required=True, help="Same fresh runtime/runs/<run_id> root")
    s.set_defaults(func=cmd_rime_search)

    s = sub.add_parser("train-seeds", help="Train final hyperparameters across seeds; validation only")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--best-hp", required=True)
    s.add_argument("--rime-manifest", required=True)
    s.add_argument("--seeds", default="42,2024,3407")
    s.add_argument("--out-dir", required=True, help="Same fresh runtime/runs/<run_id> root")
    s.set_defaults(func=cmd_train_seeds)

    s = sub.add_parser(
        "lock-seeds",
        help="Validate and lock the three fitted seed checkpoints",
    )
    s.add_argument("--seed-summary", required=True)
    s.add_argument("--seed-training-manifest", required=True)
    s.add_argument("--data-dir", required=True)
    s.add_argument("--out-dir", required=True, help="Same fresh runtime/runs/<run_id> root")
    s.set_defaults(func=cmd_lock_seeds)

    s = sub.add_parser(
        "evaluate",
        help="Evaluate all three locked checkpoints on the chronologically held-out evaluation period (2022–2025)",
    )
    s.add_argument("--data-dir", required=True)
    s.add_argument("--lock-manifest", required=True)
    s.add_argument("--out-dir", required=True, help="Same locked runtime/runs/<run_id> root")
    s.add_argument("--confirm-evaluation", default="")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser(
        "smoke-test",
        help="Check data/model compatibility with a Train/Validation-only high-level view",
    )
    s.add_argument("--data-dir", required=True)
    s.add_argument("--window-selection", required=True)
    s.set_defaults(func=cmd_smoke_test)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
