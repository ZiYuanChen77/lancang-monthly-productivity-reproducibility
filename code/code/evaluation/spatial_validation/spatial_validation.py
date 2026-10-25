"""Five-fold spatial validation with fold-specific window and RIME selection."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[3]
INPUTS = SOURCE / "public_data/model_input"
RUN = SOURCE / "code/runtime/spatial_validation"
SUPPORT = ROOT
COMMIT = "d877751e7c8e799db1e881a0e3f60a0a0b6a55e7"
WINDOWS = (6, 12, 18, 24, 36, 48)
SEEDS = (42, 2024, 3407)
FOLD_COUNTS = {1: 36, 2: 44, 3: 34, 4: 74, 5: 59}
POLICY = {
    "version": "spatial-fold-window-specific-rime-v1",
    "source_commit": COMMIT,
    "windows": list(WINDOWS), "seeds": list(SEEDS),
    "train_years": [2005, 2018], "validation_years": [2019, 2021],
    "evaluation_years": [2022, 2025],
    "rime": {"population": 10, "iterations": 5, "seed": 42,
             "hidden_sizes": [32, 64, 128], "log10_lr": [-4, -2],
             "max_epochs": 25, "patience": 5},
    "window_ranking": "mean_validation_mse_then_exact_tie_median_then_shorter",
    "final_max_epochs": 150, "final_patience": 20,
    "scaler": "raw_training_only_48_month_histories_population_sd",
    "common_evaluation_domain": "union_invalid_histories_over_all_six_windows_11818",
    "baseline": {"window": 36, "hidden_size": 128,
                 "learning_rate": 0.0001542960100551973},
    "bootstrap": {"replicates": 5000, "seed": 20240724, "cluster": "point_id"},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def module(name, rel):
    support_file = SUPPORT / Path(rel).name
    spec = importlib.util.spec_from_file_location(name, support_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical():
    return module("spatial_canonical", "model.py")


def normalize(ids):
    return ids.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)


def invalid_history(index, bad, window):
    times = index.Year.to_numpy(int) * 12 + index.Month.to_numpy(int) - 1
    out = np.zeros(len(index), bool)
    for r in bad.itertuples(index=False):
        delta = times - (int(r.Year) * 12 + int(r.Month) - 1)
        out |= (index.point_id.to_numpy() == str(r.point_id).zfill(3)) & (delta >= 1) & (delta <= window)
    return out


def masks(index, folds, fold):
    held = set(folds.loc[folds.fold_id == f"fold_{fold}", "point_id"])
    other = set(folds.point_id) - held
    years = index.Year.to_numpy(int)
    inside = index.point_id.isin(other).to_numpy()
    train = inside & (years >= 2005) & (years <= 2018)
    val = inside & (years >= 2019) & (years <= 2021)
    test = index.point_id.isin(held).to_numpy() & (years >= 2022) & (years <= 2025)
    expected = (247 - FOLD_COUNTS[fold])
    assert len(held) == FOLD_COUNTS[fold]
    assert int(train.sum()) == expected * 168
    assert int(val.sum()) == expected * 36
    assert int(test.sum()) == FOLD_COUNTS[fold] * 48
    assert not (set(index.loc[train | val, "point_id"]) & held)
    assert not np.any(train & val) and not np.any((train | val) & test)
    return np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test), sorted(held)


def fit_scaler(x48, y, ids):
    block = np.asarray(x48[ids, :, :9], dtype=np.float64).reshape(-1, 9)
    mean, std = block.mean(0), block.std(0, ddof=0)
    target = np.asarray(y[ids], np.float64)
    record = {"x_mean": mean.tolist(), "x_std": std.tolist(),
              "y_mean": float(target.mean()), "y_std": float(target.std(ddof=0)),
              "fit_rows": len(ids), "fit_history_months": 48}
    assert np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0)
    assert math.isfinite(record["y_mean"]) and record["y_std"] > 0
    return record


def transform(x48, ids, window, scaler):
    out = np.asarray(x48[ids, -window:, :], dtype=np.float32).copy()
    # Transform raw floats in float64, round once when assigning into model input.
    out[:, :, :9] = ((out[:, :, :9].astype(np.float64) - np.asarray(scaler["x_mean"])) /
                     np.asarray(scaler["x_std"]))
    assert np.isfinite(out).all()
    return out


@dataclass
class FitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_mean: float
    y_scale: float

    @property
    def input_dim(self):
        return self.X_train.shape[2]


def bundle(x, y, train, val, window, scaler):
    return FitData(transform(x, train, window, scaler),
                   ((np.asarray(y[train], np.float64) - scaler["y_mean"]) / scaler["y_std"]).astype(np.float32),
                   transform(x, val, window, scaler),
                   ((np.asarray(y[val], np.float64) - scaler["y_mean"]) / scaler["y_std"]).astype(np.float32),
                   scaler["y_mean"], scaler["y_std"])


def prepare():
    target = RUN / "prepared"
    if (target / "preparation.json").exists():
        log("Existing prepared inputs found; verifying source identities")
        manifest = read(target / "preparation.json")
        for rel, expected in manifest["source_sha256"].items():
            assert sha(INPUTS / rel) == expected, rel
        for name, expected in manifest["support_sha256"].items():
            assert sha(SUPPORT / name) == expected, name
        for name, expected in manifest["prepared_sha256"].items():
            assert sha(target / name) == expected, name
        return
    assert not target.exists(), "Incomplete input preparation directory"
    prep = module("spatial_preprocessor", "inputs.py")
    source_rel = "monthly_model_input.csv"
    raw = pd.read_csv(INPUTS / source_rel, dtype={"point_id": str})
    raw.point_id = normalize(raw.point_id)
    assert len(raw) == 74100 and raw.point_id.nunique() == 247
    bad = raw.loc[~np.isfinite(raw.NDVI) | raw.NDVI.eq(-9999), ["point_id", "Year", "Month", "NDVI"]]
    assert list(bad[["point_id", "Year", "Month"]].itertuples(index=False, name=None)) == [("233", 2022, 10)]
    log("Building raw histories using released site-wise preprocessing; no scaler is fitted")
    frame = prep.build_frame(INPUTS / source_rel, max_forward_fill=2)
    arrays, y, _, _, index = prep.build_samples(frame)
    index.point_id = normalize(index.point_id)
    assert len(index) == 62244 and arrays[48].shape == (62244, 48, 19)
    assert index.split.value_counts().to_dict() == {"train": 41496, "test": 11856, "val": 8892}
    x48 = arrays[48]
    del arrays
    folds = pd.read_csv(INPUTS / "unseen_site_fold_assignment.csv", dtype=str)
    folds.point_id = normalize(folds.point_id)
    assert len(folds) == 247 and folds.point_id.nunique() == 247
    index["invalid_w36"] = invalid_history(index, bad, 36)
    index["invalid_w48"] = invalid_history(index, bad, 48)
    assert index.invalid_w36.sum() == 36 and index.invalid_w48.sum() == 38
    assert not index.loc[index.split != "test", "invalid_w48"].any()
    index["common_clean_evaluation"] = index.split.eq("test") & ~index.invalid_w48
    assert int(index.common_clean_evaluation.sum()) == 11818
    target.mkdir(parents=True)
    np.save(target / "x_raw_48.npy", x48)
    np.save(target / "y_raw.npy", y)
    index.to_csv(target / "index.csv", index=False)
    folds.to_csv(target / "folds.csv", index=False)
    bad.to_csv(target / "invalid_inputs.csv", index=False)
    index.loc[index.split.eq("test") & index.invalid_w48].to_csv(target / "excluded_common_38.csv", index=False)
    index.loc[index.common_clean_evaluation].to_csv(target / "common_evaluation_11818.csv", index=False)
    for fold in range(1, 6):
        tr, va, te, held = masks(index, folds, fold)
        scaler = fit_scaler(x48, y, tr)
        directory = target / f"fold_{fold}"
        directory.mkdir()
        index.iloc[tr].to_csv(directory / "train_keys.csv", index=False)
        index.iloc[va].to_csv(directory / "validation_keys.csv", index=False)
        index.iloc[te].to_csv(directory / "outer_evaluation_keys.csv", index=False)
        dump(directory / "scaler.json", scaler)
        dump(directory / "site_scope.json", {"held_out": held,
             "training_and_validation": sorted(set(index.iloc[tr].point_id)),
             "train_count": len(tr), "validation_count": len(va),
             "outer_evaluation_count": len(te), "selection_heldout_contribution": 0})
    sources = [source_rel, "unseen_site_fold_assignment.csv",
               "study_sites_247.csv"]
    names = ["x_raw_48.npy", "y_raw.npy", "index.csv", "folds.csv", "invalid_inputs.csv"]
    dump(target / "preparation.json", {"source_commit": COMMIT, "source_sha256": {p: sha(INPUTS / p) for p in sources},
         "support_sha256": {p: sha(SUPPORT / p) for p in ["model.py", "inputs.py"]},
         "prepared_sha256": {p: sha(target / p) for p in names},
         "sample_count": len(index), "common_evaluation_records": 11818,
         "ndvi_w36_excluded": 36, "ndvi_w48_union_excluded": 38,
         "model_training_run": False, "global_scaler_fitted": False,
         "features": list(prep.FEATURES)})
    log("Preparation complete: raw arrays, five disjoint site scopes, 11818 common clean evaluation keys")


def load_inputs():
    p = RUN / "prepared"
    return (np.load(p / "x_raw_48.npy", mmap_mode="r"), np.load(p / "y_raw.npy", mmap_mode="r"),
            pd.read_csv(p / "index.csv", dtype={"point_id": str}), pd.read_csv(p / "folds.csv", dtype=str))


def protocol_gate():
    manifest = {"policy": POLICY, "script_sha256": sha(__file__),
                "protocol_sha256": sha(ROOT / "protocol.md"),
                "preparation_sha256": sha(RUN / "prepared/preparation.json"),
                "public_trainer_sha256": sha(SUPPORT / "model.py")}
    path = RUN / "protocol_lock.json"
    if path.exists():
        assert read(path) == manifest, "Run configuration or input identity mismatch"
    else:
        dump(path, manifest)
    return sha(path)


def train_save(c, data, hp, seed, directory, identity):
    done = directory / "complete.json"
    if done.exists():
        record = read(done)
        assert record["identity"] == identity
        assert sha(directory / "model.pt") == record["checkpoint_sha256"]
        assert sha(directory / "history.csv") == record["history_sha256"]
        return record
    assert not directory.exists(), f"Incomplete training output: {directory}"
    directory.mkdir(parents=True)
    log(f"Final-budget fit {identity['arm']} fold={identity['fold']} W{identity['window']} seed={seed}")
    started = time.monotonic()
    result = c.train_model(data, hidden_size=hp["hidden_size"], learning_rate=hp["learning_rate"],
                           seed=seed, max_epochs=150, patience=20, deterministic=True)
    pd.DataFrame(result["history"]).to_csv(directory / "history.csv", index=False)
    checkpoint = {"identity": identity, "hp": hp, "best_epoch": result["best_epoch"],
                  "model_state_dict": result["best_state"]}
    torch.save(checkpoint, directory / "model.pt")
    record = {"identity": identity, "hp": hp, "best_epoch": int(result["best_epoch"]),
              "best_val_mse_scaled": float(result["best_val_mse_scaled"]),
              "best_val_rmse_gC_m2_month": float(result["best_val_rmse_gC_m2_month"]),
              "runtime_seconds": time.monotonic() - started,
              "checkpoint_sha256": sha(directory / "model.pt"),
              "history_sha256": sha(directory / "history.csv")}
    dump(done, record)
    log(f"Fit saved; validation MSE={record['best_val_mse_scaled']:.8g}, epochs={len(result['history'])}")
    del result
    gc.collect()
    torch.cuda.empty_cache()
    return record


def rime_search(c, data, fold, window, identity):
    from mealpy import FloatVar, IntegerVar
    from mealpy.physics_based.RIME import OriginalRIME
    directory = RUN / f"nested/fold_{fold}/W{window}/rime"
    complete = directory / "complete.json"
    if complete.exists():
        record = read(complete)
        assert record["identity"] == identity
        assert sha(directory / "candidates.jsonl") == record["candidate_log_sha256"]
        return record["best_hp"]
    directory.mkdir(parents=True, exist_ok=True)
    cache, records = {}, []
    history = directory / "candidates.jsonl"
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["identity"] == identity
            assert sha(directory / record["history_file"]) == record["history_sha256"]
            cache[(record["hp"]["hidden_size"], round(record["hp"]["log10_lr"], 8))] = record["fitness"]
            records.append(record)

    def decode(solution):
        llr = float(np.clip(solution[0], -4, -2))
        h = int(np.clip(round(float(solution[1])), 0, 2))
        return {"window": window, "hidden_size": [32, 64, 128][h], "learning_rate": float(10 ** llr), "log10_lr": llr}

    def objective(solution):
        hp = decode(solution)
        key = (hp["hidden_size"], round(hp["log10_lr"], 8))
        if key in cache:
            return cache[key]
        number = len(records) + 1
        started = time.monotonic()
        log(f"RIME fold={fold} W{window} candidate={number} hidden={hp['hidden_size']} lr={hp['learning_rate']:.7g}")
        result = c.train_model(data, hidden_size=hp["hidden_size"], learning_rate=hp["learning_rate"],
                               seed=42, max_epochs=25, patience=5, deterministic=True)
        histname = f"candidate_{number:03d}_history.csv"
        pd.DataFrame(result["history"]).to_csv(directory / histname, index=False)
        fitness = float(result["best_val_rmse_gC_m2_month"])
        record = {"identity": identity, "candidate": number, "hp": hp, "fitness": fitness,
                  "best_val_mse_scaled": float(result["best_val_mse_scaled"]),
                  "best_epoch": int(result["best_epoch"]), "epochs": len(result["history"]),
                  "history_file": histname, "history_sha256": sha(directory / histname),
                  "runtime_seconds": time.monotonic() - started}
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        cache[key] = fitness
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return fitness

    c.set_seed(42, deterministic=True)
    bounds = [FloatVar(lb=-4, ub=-2, name="log10_lr"), IntegerVar(lb=0, ub=2, name="hidden_idx")]
    optimizer = OriginalRIME(epoch=5, pop_size=10)
    best = optimizer.solve({"bounds": bounds, "minmax": "min", "obj_func": objective}, seed=42)
    hp = decode(best.solution)
    dump(complete, {"identity": identity, "best_hp": hp, "fitness": float(best.target.fitness),
                    "unique_candidate_count": len(records), "budget": POLICY["rime"],
                    "candidate_log_sha256": sha(history), "outer_scores_used": False})
    return hp


def rank_windows(records):
    table = pd.DataFrame(records)
    assert set(table.window) == set(WINDOWS)
    assert not table.duplicated(["window", "seed"]).any()
    assert all(set(g.seed) == set(SEEDS) for _, g in table.groupby("window"))
    ranked = table.groupby("window").best_val_mse_scaled.agg(["mean", "median", "std"]).reset_index()
    return ranked.sort_values(["mean", "median", "window"], kind="stable").reset_index(drop=True)


def fit_fold(fold):
    protocol = protocol_gate()
    c = canonical()
    x, y, index, folds = load_inputs()
    train, val, _, held = masks(index, folds, fold)
    scaler = read(RUN / f"prepared/fold_{fold}/scaler.json")
    assert scaler == fit_scaler(x, y, train)
    records = []
    for window in WINDOWS:
        data = bundle(x, y, train, val, window, scaler)
        c.assert_pre_evaluation_isolation(data)
        identity = {"protocol": protocol, "fold": fold, "window": window, "arm": "nested",
                    "heldout_sites": held}
        hp = rime_search(c, data, fold, window, identity)
        for seed in SEEDS:
            result = train_save(c, data, hp, seed, RUN / f"nested/fold_{fold}/W{window}/seed_{seed}",
                                {**identity, "seed": seed})
            records.append({"window": window, "seed": seed, "best_val_mse_scaled": result["best_val_mse_scaled"]})
        del data
        gc.collect()
    ranked = rank_windows(records)
    out = RUN / f"nested/fold_{fold}"
    pd.DataFrame(records).to_csv(out / "window_validation_by_seed.csv", index=False)
    ranked.to_csv(out / "window_ranking.csv", index=False)
    selected = int(ranked.iloc[0].window)
    models = [{"seed": seed, "path": f"nested/fold_{fold}/W{selected}/seed_{seed}/model.pt",
               "sha256": sha(RUN / f"nested/fold_{fold}/W{selected}/seed_{seed}/model.pt")} for seed in SEEDS]
    dump(out / "fold_lock.json", {"protocol": protocol, "fold": fold, "selected_window": selected,
         "selection_scores": ranked.to_dict("records"), "models": models, "outer_scores_used": False})
    log(f"Fold {fold} nested selection LOCKED: W{selected}; no outer scores calculated")
    data = bundle(x, y, train, val, 36, scaler)
    for seed in SEEDS:
        identity = {"protocol": protocol, "fold": fold, "window": 36, "arm": "fixed_baseline", "seed": seed,
                    "heldout_sites": held}
        train_save(c, data, POLICY["baseline"], seed, RUN / f"baseline/fold_{fold}/seed_{seed}", identity)
    dump(RUN / f"baseline/fold_{fold}/fold_lock.json", {"protocol": protocol, "fold": fold, "selected_window": 36,
         "models": [{"seed": seed, "path": f"baseline/fold_{fold}/seed_{seed}/model.pt",
                      "sha256": sha(RUN / f"baseline/fold_{fold}/seed_{seed}/model.pt")} for seed in SEEDS],
         "outer_scores_used": False})


def metrics(y, pred):
    err = pred - y
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
            "R2": float(1 - np.sum(err ** 2) / sst), "Bias": float(np.mean(err)),
            "Pearson_r": float(np.corrcoef(y, pred)[0, 1])}


def metrics_from_sums(totals):
    n, sy, sp, syy, spp, syp, se, sae, sse = np.moveaxis(totals, -1, 0)
    return np.stack([np.sqrt(sse/n), sae/n, 1-sse/(syy-sy*sy/n), se/n,
                     (syp-sy*sp/n)/np.sqrt((syy-sy*sy/n)*(spp-sp*sp/n))], axis=-1)


def evaluation_lock():
    protocol = protocol_gate()
    locks = []
    for arm in ("nested", "baseline"):
        for fold in range(1, 6):
            path = RUN / f"{arm}/fold_{fold}/fold_lock.json"
            state = read(path)
            assert state["protocol"] == protocol and not state["outer_scores_used"]
            for model in state["models"]:
                assert sha(RUN / model["path"]) == model["sha256"]
            locks.append({"path": str(path.relative_to(RUN)), "sha256": sha(path)})
    state = {"protocol": protocol, "all_ten_arm_fold_locks": locks,
             "all_30_selected_models_locked": True, "outer_evaluation_opened_after_lock": True}
    path = RUN / "all_folds_pre_evaluation_lock.json"
    if path.exists():
        assert read(path) == state
    else:
        dump(path, state)


def evaluate():
    evaluation_lock()
    c = canonical()
    x, y, index, folds = load_inputs()
    directory = RUN / "evaluation"
    directory.mkdir(exist_ok=True)
    frames, rows = [], []
    for arm in ("baseline", "nested"):
        for fold in range(1, 6):
            lock = read(RUN / f"{arm}/fold_{fold}/fold_lock.json")
            _, _, ids, _ = masks(index, folds, fold)
            ids = ids[index.iloc[ids].common_clean_evaluation.to_numpy(bool)]
            window = lock["selected_window"]
            assert not index.iloc[ids].invalid_w48.any()
            scaler = read(RUN / f"prepared/fold_{fold}/scaler.json")
            xx = transform(x, ids, window, scaler)
            truth = np.asarray(y[ids], np.float64) * 1000
            for entry in lock["models"]:
                state = torch.load(RUN / entry["path"], map_location="cpu", weights_only=False)
                model = c.CNNBiLSTMAttention(19, int(state["hp"]["hidden_size"]), c.FIXED).to(c.get_device())
                model.load_state_dict(state["model_state_dict"])
                pred = c.predict(model, c.make_x_loader(xx, 256, c.get_device()), c.get_device()).astype(np.float64)
                pred = (pred * scaler["y_std"] + scaler["y_mean"]) * 1000
                frame = index.iloc[ids][["sample_id", "point_id", "Year", "Month"]].copy()
                frame["arm"], frame["fold"], frame["seed"], frame["window"] = arm, fold, entry["seed"], window
                frame["y_true"], frame["y_pred"] = truth, pred
                frames.append(frame)
                rows.append({"arm": arm, "fold": fold, "seed": entry["seed"], "window": window,
                             "n": len(ids), **metrics(truth, pred)})
                del model
                torch.cuda.empty_cache()
    predictions = pd.concat(frames, ignore_index=True)
    for (arm, seed), part in predictions.groupby(["arm", "seed"]):
        assert len(part) == 11818 and part.point_id.nunique() == 247
        assert not part.duplicated(["point_id", "Year", "Month"]).any()
    predictions.to_csv(directory / "paired_oof_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(directory / "fold_seed_metrics.csv", index=False)
    pooled = [{"arm": arm, "seed": int(seed), "n": len(g), **metrics(g.y_true.to_numpy(), g.y_pred.to_numpy())}
              for (arm, seed), g in predictions.groupby(["arm", "seed"])]
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(directory / "pooled_seed_metrics.csv", index=False)
    pooled.groupby("arm")[["RMSE", "MAE", "R2", "Bias", "Pearson_r"]].agg(["mean", "std"]).to_csv(directory / "pooled_three_seed_summary.csv")
    dump(directory / "evaluation_complete.json", {"records_per_seed": 11818, "seeds": list(SEEDS),
         "predictions_sha256": sha(directory / "paired_oof_predictions.csv"), "selection_complete_before_evaluation": True})
    log("Outer evaluation complete for both arms on common 11818 records per seed")


def bootstrap():
    directory = RUN / "evaluation"
    frame = pd.read_csv(directory / "paired_oof_predictions.csv", dtype={"point_id": str})
    keys = ["point_id", "Year", "Month"]
    base = frame.loc[frame.arm.eq("baseline") & frame.seed.eq(42)].sort_values(keys)
    key_index = pd.MultiIndex.from_frame(base[keys])
    sites = sorted(base.point_id.unique())
    # Sufficient statistics permit grouped resampling without expanding duplicates.
    columns = ["n", "y", "p", "yy", "pp", "yp", "err", "abs_err", "sq_err"]
    statistics = np.zeros((2, 3, len(sites), len(columns)), np.float64)
    for ai, arm in enumerate(("baseline", "nested")):
        for si, seed in enumerate(SEEDS):
            part = frame.loc[frame.arm.eq(arm) & frame.seed.eq(seed)].set_index(keys).loc[key_index].reset_index()
            assert np.array_equal(part.y_true.to_numpy(), base.y_true.to_numpy())
            for pi, site in enumerate(sites):
                g = part.loc[part.point_id.eq(site)]
                y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
                e = p - y
                statistics[ai, si, pi] = [len(g), y.sum(), p.sum(), (y*y).sum(), (p*p).sum(),
                                         (y*p).sum(), e.sum(), np.abs(e).sum(), (e*e).sum()]

    point = metrics_from_sums(statistics.sum(axis=2)).mean(axis=1)
    rng = np.random.default_rng(20240724)
    samples = np.empty((5000, 2, 5))
    for b in range(5000):
        weights = np.bincount(rng.integers(0, len(sites), size=len(sites)), minlength=len(sites))
        samples[b] = metrics_from_sums(np.einsum("aspv,p->asv", statistics, weights)).mean(axis=1)
    differences = samples[:, 1] - samples[:, 0]
    output = []
    for j, metric in enumerate(["RMSE", "MAE", "R2", "Bias", "Pearson_r"]):
        lo, hi = np.percentile(differences[:, j], [2.5, 97.5])
        output.append({"metric": metric, "baseline_mean": point[0,j], "nested_mean": point[1,j],
                       "nested_minus_baseline": point[1,j]-point[0,j], "ci_low": lo, "ci_high": hi,
                       "relative_difference_percent": 100*(point[1,j]-point[0,j])/abs(point[0,j])})
    pd.DataFrame(output).to_csv(directory / "paired_site_bootstrap_comparison.csv", index=False)
    np.save(directory / "paired_site_bootstrap_differences.npy", differences)
    dump(directory / "bootstrap_complete.json", {"replicates": 5000, "seed": 20240724,
         "cluster": "point_id", "paired_across_arms_and_seeds": True,
         "interval_scope": "site_cluster_sampling"})
    log("Paired site-cluster bootstrap complete")


def check_inputs():
    x, y, index, folds = load_inputs()
    assert x.shape == (62244, 48, 19) and len(y) == len(index)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    for fold in range(1, 6):
        tr, va, te, held = masks(index, folds, fold)
        scaler = fit_scaler(x, y, tr)
        assert scaler == read(RUN / f"prepared/fold_{fold}/scaler.json")
        assert set(index.iloc[tr].point_id).isdisjoint(held)
        assert set(index.iloc[va].point_id).isdisjoint(held)
        assert index.iloc[tr].Year.between(2005, 2018).all()
        assert index.iloc[va].Year.between(2019, 2021).all()
        assert index.iloc[te].Year.between(2022, 2025).all()
        canonical().assert_pre_evaluation_isolation(bundle(x, y, tr[:8], va[:8], 6, scaler))
        selection = RUN / f"nested/fold_{fold}/window_validation_by_seed.csv"
        lock = RUN / f"nested/fold_{fold}/fold_lock.json"
        if selection.exists() and lock.exists():
            selected = int(rank_windows(pd.read_csv(selection).to_dict("records")).iloc[0].window)
            assert selected == int(read(lock)["selected_window"])
    bad = pd.read_csv(RUN / "prepared/invalid_inputs.csv", dtype={"point_id": str})
    assert np.array_equal(invalid_history(index, bad, 36), index.invalid_w36.to_numpy())
    assert np.array_equal(invalid_history(index, bad, 48), index.invalid_w48.to_numpy())
    keys = ["point_id", "Year", "Month"]
    current_w36 = index.loc[index.split.eq("test") & ~index.invalid_w36]
    common = index.loc[index.common_clean_evaluation]
    assert len(current_w36) == 11820 and len(common) == 11818
    additional = set(map(tuple, current_w36[keys].to_numpy())) - set(map(tuple, common[keys].to_numpy()))
    assert additional == {("233", 2025, 11), ("233", 2025, 12)}
    dump(RUN / "input_checks.json", {"five_site_scopes_disjoint": True,
         "training_only_scalers": True, "chronological_partitions": True,
         "W36_W48_exclusion_keys": True, "common_evaluation_records": len(common)})
    log("Input and partition checks passed")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare","check","fit-fold","run-all","evaluate","bootstrap"])
    p.add_argument("--fold", type=int, choices=range(1,6))
    p.add_argument("--source-root", type=Path, default=SOURCE,
                   help="Project root")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Fresh output directory; defaults to SOURCE/code/runtime/spatial_validation")
    p.add_argument("--data-dir", type=Path, default=None, help="Directory containing the three model-input CSV files")
    args = p.parse_args()
    global_source = args.source_root.resolve()
    globals()["SOURCE"] = global_source
    globals()["INPUTS"] = (args.data_dir.resolve() if args.data_dir else global_source / "public_data/model_input")
    globals()["RUN"] = (args.output_root.resolve() if args.output_root else global_source / "code/runtime/spatial_validation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.stage in ["fit-fold", "run-all", "evaluate"]:
        assert torch.cuda.is_available(), "CUDA unavailable for training/evaluation"
    if args.stage == "prepare": prepare()
    elif args.stage == "check": check_inputs()
    elif args.stage == "fit-fold":
        assert args.fold
        fit_fold(args.fold)
    elif args.stage == "run-all":
        prepare()
        check_inputs()
        for fold in range(1,6): fit_fold(fold)
        evaluate()
        bootstrap()
        dump(RUN / "complete.json", {"stage":"all_five_folds_and_paired_comparison_complete",
             "source_commit":COMMIT,"protocol_lock_sha256":sha(RUN/"protocol_lock.json"),
             "evaluation_records_per_seed":11818})
    elif args.stage == "evaluate": evaluate()
    elif args.stage == "bootstrap": bootstrap()


if __name__ == "__main__":
    main()
