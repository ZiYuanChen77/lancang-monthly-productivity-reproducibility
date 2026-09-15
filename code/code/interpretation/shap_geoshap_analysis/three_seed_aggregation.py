"""Aligned three-seed SHAP summaries."""

from pathlib import Path

import numpy as np

SEED_RUNS = {42: "background_BG256_D1", 2024: "seed_2024", 3407: "seed_3407"}


def load_three_seed_shap(paths, features):
    """Return seed tensors with identical feature, background and explained keys."""
    tensors = []
    reference_keys = None
    for seed, run_id in SEED_RUNS.items():
        with np.load(Path(paths(run_id)["shap_npz"]), allow_pickle=False) as data:
            if int(np.asarray(data["model_seed"]).item()) != seed:
                raise ValueError(f"Model seed mismatch: {run_id}")
            if data["feature_names"].astype(str).tolist() != list(features):
                raise ValueError(f"Feature order mismatch: {run_id}")
            keys = tuple(np.asarray(data[key], dtype=np.int64) for key in
                         ("background_sample_ids", "explained_sample_ids"))
            if reference_keys is None:
                reference_keys = keys
            elif any(not np.array_equal(a, b) for a, b in zip(reference_keys, keys)):
                raise ValueError(f"SHAP sample keys mismatch: {run_id}")
            values = np.asarray(data["shap_values_gC_m2_month"], dtype=np.float64)
            if values.ndim != 3 or values.shape[1:] != (36, len(features)):
                raise ValueError(f"SHAP tensor shape mismatch: {run_id}")
            if values.shape[0] != len(keys[1]) or not np.isfinite(values).all():
                raise ValueError(f"Invalid SHAP values: {run_id}")
            tensors.append(values)
    return np.stack(tensors), reference_keys[0], reference_keys[1]


def summarize_seed_tensors(tensors):
    """Mean signed SHAP, mean absolute SHAP, and variable-importance seed SD."""
    values = np.asarray(tensors, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected three finite seed tensors")
    absolute = np.abs(values)
    return (values.mean(axis=0), absolute.mean(axis=0),
            absolute.sum(axis=2).mean(axis=1).std(axis=0, ddof=1))
