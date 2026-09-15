from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import yaml
except ImportError:  # pragma: no cover - reported clearly when a CLI is used
    yaml = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PACKAGE_ROOT / "runtime"
PROJECT_ROOT = PACKAGE_ROOT
DATA_DIR = PACKAGE_ROOT / "data" / "processed"
CANONICAL_RUN_ROOT = PACKAGE_ROOT / "runtime" / "runs" / "canonical"
CANONICAL_PREDICTIONS = CANONICAL_RUN_ROOT / "evaluation" / "evaluation_predictions_all_seeds.csv"
CANONICAL_LOCK = CANONICAL_RUN_ROOT / "pre_evaluation_lock_manifest.json"
CANONICAL_WINDOW_MANIFEST = CANONICAL_RUN_ROOT / "window_selection_manifest.json"
CANONICAL_PIPELINE = PACKAGE_ROOT / "code" / "canonical_pipeline.py"
SEEDS = (42, 2024, 3407)
DEEP_GRID_MODELS = ("No RIME", "CNN", "LSTM", "BiLSTM", "CNN-LSTM")
ALL_DEEP_MODELS = (
    "No Attention",
    "No CNN",
    "No BiLSTM",
    "No RIME",
    "CNN",
    "LSTM",
    "BiLSTM",
    "CNN-LSTM",
)
PREDICTION_FILENAMES = {
    "Full canonical": "full_model_canonical.csv",
    "No Attention": "no_attention.csv",
    "No CNN": "no_cnn.csv",
    "No BiLSTM": "no_bilstm.csv",
    "No RIME": "no_rime.csv",
    "CNN": "baseline_cnn.csv",
    "LSTM": "baseline_lstm.csv",
    "BiLSTM": "baseline_bilstm.csv",
    "CNN-LSTM": "baseline_cnn_lstm.csv",
    "Random Forest": "baseline_random_forest.csv",
    "Ridge W36": "baseline_ridge_w36.csv",
    "Monthly climatology": "baseline_monthly_climatology.csv",
    "point-month climatology": "baseline_point_month_climatology.csv",
}
SLUGS = {
    "No Attention": "no_attention",
    "No CNN": "no_cnn",
    "No BiLSTM": "no_bilstm",
    "No RIME": "no_rime",
    "CNN": "baseline_cnn",
    "LSTM": "baseline_lstm",
    "BiLSTM": "baseline_bilstm",
    "CNN-LSTM": "baseline_cnn_lstm",
}


class ExperimentConfig:
    """Small, explicit wrapper around the public YAML configuration."""

    def __init__(self, raw: dict[str, Any], source: Path):
        self.raw = raw
        self.source = Path(source).resolve()
        self.root = self.source.parent.parent
        self.models = list(raw.get("models", []))
        experiment = raw.get("experiment", {})
        self.seeds = tuple(int(value) for value in experiment.get("seeds", SEEDS))
        self.input_features = int(experiment.get("input_features", 19))
        self.training = dict(raw.get("training", {}))

    def path(self, key: str) -> Path:
        value = self.raw.get("paths", {}).get(key)
        if value is None:
            raise KeyError(f"Missing paths.{key} in {self.source}")
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the configuration file.")
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    return ExperimentConfig(raw, source)




def validate_public_config(config: ExperimentConfig) -> None:
    """Require the configuration to match the fixed public W36 workflow.

    The configuration records the W36 analysis specification used in the
    manuscript. Validation checks keep the YAML aligned with that specification.
    """
    if config.root.resolve() != PACKAGE_ROOT.resolve():
        raise ValueError(
            f"Configuration must live under {PACKAGE_ROOT / 'configs'} so all public "
            "workflow paths resolve within the package root."
        )
    exp = config.raw.get("experiment", {})
    expected = {
        "input_features": 19,
        "window_length": 36,
        "seeds": [42, 2024, 3407],
        "train_years": [2005, 2018],
        "validation_years": [2019, 2021],
        "evaluation_years": [2022, 2025],
    }
    for key, value in expected.items():
        if exp.get(key) != value:
            raise ValueError(f"experiment.{key} must equal the fixed value {value!r}.")

    training_expected = {
        "hidden_size": 128,
        "learning_rate": 0.0001542960100551973,
        "max_epochs": 150,
        "early_stopping_patience": 20,
        "batch_size": 256,
        "weight_decay": 0.0001,
        "grad_clip_norm": 1.0,
        "scheduler_factor": 0.5,
        "scheduler_patience": 6,
        "scheduler_min_lr": 0.000001,
    }
    for key, value in training_expected.items():
        if config.training.get(key) != value:
            raise ValueError(f"training.{key} must equal the fixed value {value!r}.")

    bootstrap = config.raw.get("bootstrap", {})
    if bootstrap.get("n_resamples") != 5000:
        raise ValueError("bootstrap.n_resamples must be 5000.")
    if bootstrap.get("random_seed") != 20240724:
        raise ValueError("bootstrap.random_seed must be 20240724.")
    if bootstrap.get("cluster_variable") != "point_id":
        raise ValueError("bootstrap.cluster_variable must be 'point_id'.")

    expected_paths = {
        "processed_data": DATA_DIR,
        "predictions_dir": RUN_ROOT / "predictions",
        "checkpoints_dir": RUN_ROOT / "models",
        "results_dir": RUN_ROOT / "results",
        "workbook": RUN_ROOT / "results" / "model_comparison.xlsx",
    }
    for key, value in expected_paths.items():
        if config.path(key).resolve() != value.resolve():
            raise ValueError(
                f"paths.{key} must resolve to the public workflow location {value}."
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=json_default)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def log(message: str) -> None:
    stamp = now_iso()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    path = RUN_ROOT / "logs" / "workflow_execution.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_canonical_module():
    spec = importlib.util.spec_from_file_location("srs_canonical_pipeline", CANONICAL_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load canonical pipeline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CANONICAL = load_canonical_module()
FIXED = CANONICAL.FIXED


def validate_canonical_state(*, require_evaluation: bool) -> dict[str, Any]:
    """Validate model identity and, for evaluation, the recorded prediction hash."""
    if not CANONICAL_LOCK.exists():
        raise FileNotFoundError(
            f"Canonical lock not found: {CANONICAL_LOCK}. Run canonical_pipeline.py "
            "through lock-seeds first."
        )
    lock = read_json(CANONICAL_LOCK)
    if (
        lock.get("final_selected_window") != 36
        or lock.get("seeds") != list(SEEDS)
        or lock.get("rime_hyperparameters", {}).get("hidden_size") != 128
        or lock.get("rime_hyperparameters", {}).get("learning_rate")
        != 0.0001542960100551973
    ):
        raise RuntimeError("Canonical W36 model identity mismatch")
    if require_evaluation:
        if lock.get("state") != "EVALUATED":
            raise RuntimeError(
                "Canonical Evaluation has not been opened and locked for this run."
            )
        if not CANONICAL_PREDICTIONS.exists():
            raise FileNotFoundError(
                f"Canonical evaluation predictions not found: {CANONICAL_PREDICTIONS}. "
                "Run the canonical evaluate stage first."
            )
        recorded = (lock.get("evaluation_outputs") or {}).get("predictions_all_seeds_sha256")
        if not recorded:
            raise RuntimeError("Canonical lock does not record the current evaluation prediction hash")
        actual = sha256_file(CANONICAL_PREDICTIONS)
        if actual != recorded:
            raise RuntimeError(
                "Canonical predictions do not match the evaluation output recorded by the current run lock"
            )
    return lock

def load_pre_evaluation_data():
    lock = validate_canonical_state(require_evaluation=False)
    if lock.get("state") != "NOT_EVALUATED":
        raise RuntimeError(
            "Pre-Evaluation tuning/fitting is closed because the canonical run is already EVALUATED. "
            "Use a fresh run directory to repeat model selection."
        )
    data = CANONICAL.PreEvaluationDataBundle(DATA_DIR, window=36)
    if data.data_identity != lock["data_identity"] or data.data_hashes() != lock["data_hashes"]:
        raise RuntimeError("Canonical W36 data identity/hash mismatch")
    CANONICAL.assert_pre_evaluation_isolation(data)
    return data


def load_evaluation_data():
    lock = validate_canonical_state(require_evaluation=True)
    data = CANONICAL.EvaluationDataBundle(DATA_DIR, window=36)
    if data.data_identity != lock["data_identity"] or data.data_hashes() != lock["data_hashes"]:
        raise RuntimeError("Evaluation W36 data identity/hash mismatch")
    return data


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class CNNOnly(nn.Module):
    def __init__(self, input_dim: int, channels: int):
        super().__init__()
        self.conv = ConvBlock(input_dim, channels)
        self.regressor = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.conv(x).mean(dim=1)).squeeze(-1)


class CNNAttention(nn.Module):
    """CNN followed by temporal attention, with no recurrent layer."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.conv = ConvBlock(input_dim, 64)
        self.attention = TemporalAttention(64)
        self.regressor = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.attention(self.conv(x))
        return self.regressor(context).squeeze(-1)


class RecurrentOnly(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        bidirectional: bool,
        attention: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=1, batch_first=True, bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim) if attention else None
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.attention(out) if self.attention is not None else out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class CNNRecurrent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        bidirectional: bool,
        attention: bool,
    ):
        super().__init__()
        self.conv = ConvBlock(input_dim, 64)
        self.lstm = nn.LSTM(
            64, hidden, num_layers=1, batch_first=True, bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim) if attention else None
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.conv(x))
        pooled = self.attention(out) if self.attention is not None else out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


def build_model(model_name: str, input_dim: int, width: int) -> nn.Module:
    if model_name == "No Attention":
        return CNNRecurrent(input_dim, 128, bidirectional=True, attention=False)
    if model_name == "No CNN":
        return RecurrentOnly(input_dim, 128, bidirectional=True, attention=True)
    if model_name == "No BiLSTM":
        return CNNAttention(input_dim, 128)
    if model_name == "No RIME":
        return CNNRecurrent(input_dim, width, bidirectional=True, attention=True)
    if model_name == "CNN":
        return CNNOnly(input_dim, channels=width)
    if model_name == "LSTM":
        return RecurrentOnly(input_dim, width, bidirectional=False)
    if model_name == "BiLSTM":
        return RecurrentOnly(input_dim, width, bidirectional=True)
    if model_name == "CNN-LSTM":
        return CNNRecurrent(input_dim, width, bidirectional=False, attention=False)
    if model_name == "Full canonical":
        return CNNRecurrent(input_dim, 128, bidirectional=True, attention=True)
    raise ValueError(f"Unknown model: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def architecture_signature(model_name: str) -> dict[str, bool]:
    """Return the declared component signature used by the structure-validation script."""
    signatures = {
        "Full canonical": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": True},
        "No Attention": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": False},
        "No CNN": {"uses_cnn": False, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": True},
        "No BiLSTM": {"uses_cnn": True, "uses_recurrent": False, "uses_bilstm": False, "uses_attention": True},
        "No RIME": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": True},
        "CNN": {"uses_cnn": True, "uses_recurrent": False, "uses_bilstm": False, "uses_attention": False},
        "LSTM": {"uses_cnn": False, "uses_recurrent": True, "uses_bilstm": False, "uses_attention": False},
        "BiLSTM": {"uses_cnn": False, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": False},
        "CNN-LSTM": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": False, "uses_attention": False},
    }
    if model_name not in signatures:
        raise ValueError(f"No architecture signature is defined for {model_name!r}")
    return signatures[model_name]


@torch.no_grad()
def evaluate_mse(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        total += float(criterion(model(xb), yb).item())
        n += int(yb.numel())
    return total / max(n, 1)


def train_deep(
    data,
    model_name: str,
    width: int,
    learning_rate: float,
    seed: int,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    CANONICAL.assert_pre_evaluation_isolation(data)
    set_seed(seed)
    device = CANONICAL.get_device()
    train_loader = CANONICAL.make_loader(
        data.X_train, data.y_train, FIXED.batch_size, True, seed, device
    )
    val_loader = CANONICAL.make_loader(
        data.X_val, data.y_val, FIXED.batch_size, False, seed, device
    )
    model = build_model(model_name, data.input_dim, width).to(device)
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
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
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
        improved = val_mse < best_val - FIXED.min_delta
        if improved:
            best_val = float(val_mse)
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            wait = 0
        else:
            wait += 1
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": train_mse,
                "val_mse_scaled": val_mse,
                "best_val_mse_scaled": best_val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": bool(improved),
                "patience_counter": wait,
            }
        )
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{model_name} seed {seed}: no valid checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "best_val_mse_scaled": float(best_val),
        "best_val_mse_gC2": float(best_val * (data.y_scale * 1000.0) ** 2),
        "runtime_seconds": float(time.perf_counter() - started),
        "history": history,
        "parameter_count": parameter_count(model),
    }


@torch.no_grad()
def predict_deep(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = CANONICAL.get_device()
    model = model.to(device)
    loader = CANONICAL.make_x_loader(X, FIXED.batch_size, device)
    return CANONICAL.predict(model, loader, device)


def canonical_key_frame() -> pd.DataFrame:
    full = pd.read_csv(CANONICAL_PREDICTIONS)
    required = {"seed", "point_id", "Year", "Month"}
    if not required.issubset(full.columns):
        raise RuntimeError("Canonical predictions lack required key fields")
    if len(full) != 35568 or full["seed"].value_counts().to_dict() != {
        42: 11856,
        2024: 11856,
        3407: 11856,
    }:
        raise RuntimeError("Canonical full predictions row/seed identity mismatch")
    keys = full[["seed", "point_id", "Year", "Month"]]
    if keys.duplicated().any():
        raise RuntimeError("Canonical full predictions contain duplicate keys")
    return full


def make_prediction_frame(
    model_name: str,
    seed: int | None,
    evaluation_data,
    pred_scaled: np.ndarray,
) -> pd.DataFrame:
    frame = evaluation_data.evaluation_index.copy()
    frame.insert(0, "model", model_name)
    frame.insert(1, "seed", seed)
    frame["y_true_gC_m2_month"] = evaluation_data.y_raw_evaluation * 1000.0
    frame["y_pred_gC_m2_month"] = evaluation_data.y_scaled_to_g(pred_scaled)
    frame["split"] = "test"
    if len(frame) != 11856 or not np.isfinite(
        frame[["y_true_gC_m2_month", "y_pred_gC_m2_month"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"{model_name}: invalid prediction rows or non-finite values")
    return frame


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    values = CANONICAL.metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    required = ("RMSE", "MAE", "R2", "Bias", "Pearson_r")
    if not np.isfinite([values[name] for name in required]).all():
        raise RuntimeError("Non-finite metric")
    return {name: float(values[name]) for name in required}
