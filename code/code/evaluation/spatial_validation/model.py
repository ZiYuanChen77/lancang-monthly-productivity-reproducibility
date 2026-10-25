"""CNN–BiLSTM–Attention training and prediction."""
from __future__ import annotations
import copy
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass(frozen=True)
class FixedModelConfig:
    cnn_channels: int = 64
    kernel_size: int = 3
    attention_dim: int = 64
    dropout: float = 0.3
    weight_decay: float = 0.0001
    batch_size: int = 256
    max_epochs: int = 150
    early_stopping_patience: int = 20
    min_delta: float = 1e-06
    grad_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 6
    scheduler_min_lr: float = 1e-06
    num_workers: int = 0
FIXED = FixedModelConfig()

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int, deterministic: bool=True) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalAttention(nn.Module):

    def __init__(self, input_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(attention_dim, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return (context, weights)

class ConvBlock(nn.Module):

    def __init__(self, input_dim: int, cfg: FixedModelConfig):
        super().__init__()
        padding = cfg.kernel_size // 2
        self.net = nn.Sequential(nn.Conv1d(input_dim, cfg.cnn_channels, cfg.kernel_size, padding=padding), nn.BatchNorm1d(cfg.cnn_channels), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Conv1d(cfg.cnn_channels, cfg.cnn_channels, cfg.kernel_size, padding=padding), nn.BatchNorm1d(cfg.cnn_channels), nn.ReLU(), nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)

class CNNBiLSTMAttention(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, cfg: FixedModelConfig=FIXED):
        super().__init__()
        self.conv = ConvBlock(input_dim, cfg)
        self.bilstm = nn.LSTM(input_size=cfg.cnn_channels, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        out_dim = hidden_size * 2
        self.attention = TemporalAttention(out_dim, cfg.attention_dim, cfg.dropout)
        self.regressor = nn.Sequential(nn.Linear(out_dim, hidden_size), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        out, _ = self.bilstm(x)
        context, _ = self.attention(out)
        return self.regressor(context).squeeze(-1)
PRE_EVALUATION_FORBIDDEN_ATTRIBUTES = ('y_raw', 'sample_index', '_test_idx', 'test_idx', 'evaluation_idx', 'X_evaluation', 'y_evaluation', 'y_raw_evaluation', 'evaluation_index')

def assert_pre_evaluation_isolation(data: Any) -> None:
    exposed = [name for name in PRE_EVALUATION_FORBIDDEN_ATTRIBUTES if hasattr(data, name)]
    if exposed:
        raise AssertionError(f'Pre-evaluation object exposes forbidden attributes: {exposed}')
    required = ('X_train', 'y_train', 'X_val', 'y_val')
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise AssertionError(f'Pre-evaluation object lacks Train/Validation views: {missing}')

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=FIXED.num_workers, pin_memory=device.type == 'cuda', drop_last=False, generator=generator if shuffle else None)

def make_x_loader(X: np.ndarray, batch_size: int, device: torch.device) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=FIXED.num_workers, pin_memory=device.type == 'cuda', drop_last=False)

@torch.no_grad()
def evaluate_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
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

def train_model(data, hidden_size: int, learning_rate: float, seed: int, max_epochs: int, patience: int, deterministic: bool=True, save_history_path: Optional[Path]=None) -> Dict[str, Any]:
    assert_pre_evaluation_isolation(data)
    set_seed(seed, deterministic=deterministic)
    device = get_device()
    train_loader = make_loader(data.X_train, data.y_train, FIXED.batch_size, True, seed, device)
    val_loader = make_loader(data.X_val, data.y_val, FIXED.batch_size, False, seed, device)
    model = CNNBiLSTMAttention(data.input_dim, hidden_size, FIXED).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=FIXED.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FIXED.scheduler_factor, patience=FIXED.scheduler_patience, min_lr=FIXED.scheduler_min_lr)
    best_val = float('inf')
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
        lr_now = float(optimizer.param_groups[0]['lr'])
        improved = val_mse < best_val - FIXED.min_delta
        if improved:
            best_val = float(val_mse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
        history.append({'epoch': epoch, 'train_mse_scaled': train_mse, 'val_mse_scaled': val_mse, 'best_val_mse_scaled': best_val, 'learning_rate': lr_now, 'improved': bool(improved), 'patience_counter': wait})
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError('Training produced no valid best checkpoint')
    model.load_state_dict(best_state)
    best_val_rmse_g = math.sqrt(best_val) * data.y_scale * 1000.0
    if save_history_path is not None:
        ensure_dir(save_history_path.parent)
        pd.DataFrame(history).to_csv(save_history_path, index=False, encoding='utf-8-sig')
    return {'model': model, 'best_state': best_state, 'best_epoch': best_epoch, 'best_val_mse_scaled': best_val, 'best_val_rmse_gC_m2_month': float(best_val_rmse_g), 'history': history}
