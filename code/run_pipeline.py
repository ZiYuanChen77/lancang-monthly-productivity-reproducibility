"""Command dispatcher for the public W36 comparison-model workflow.

For the 13-model comparison branch, this dispatcher starts only after the
canonical pipeline has created ``pre_evaluation_lock_manifest.json`` and keeps
all comparison tuning/fitting on the Train/Validation side of the protocol
boundary. It also exposes the separate coordinate-clustered unseen-site
fit/evaluate branch, which requires the public 247-site coordinate table.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "code" / "model_comparison"
BASELINE_SCRIPT = ROOT / "code" / "baselines" / "ridge_and_climatology.py"
UNSEEN_TRAINING_SCRIPT = ROOT / "code" / "evaluation" / "unseen_site_training.py"
WATER_ENERGY_PREP_SCRIPT = ROOT / "code" / "water_energy_analysis" / "prepare_water_energy_inputs.py"
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def run(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    subprocess.run(command, check=True, cwd=str(ROOT))


def config_path(raw: str) -> Path:
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {path}. Copy configs/config.example.yaml "
            "to configs/config.yaml first."
        )
    return path


def configured_paths(path: Path) -> dict[str, Path]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = path.parent.parent
    result: dict[str, Path] = {}
    for key, value in (raw.get("paths") or {}).items():
        candidate = Path(value)
        result[key] = candidate if candidate.is_absolute() else (root / candidate).resolve()
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("help-workflow", help="Print the staged execution order")
    for name in (
        "validate-workflow",
        "tune-models",
        "train-models",
        "select-baselines",
        "evaluate-models",
        "evaluate-baselines",
        "metrics",
        "validate-structure",
        "workbook",
    ):
        q = sub.add_parser(name)
        q.add_argument("--config", default=str(DEFAULT_CONFIG))
    for name in ("fit-unseen-site", "evaluate-unseen-site"):
        q = sub.add_parser(name)
        q.add_argument("--config", default=str(DEFAULT_CONFIG))
        q.add_argument(
            "--site-table",
            required=True,
            help="Public 247-site CSV containing point_id, Lon and Lat.",
        )
    q = sub.add_parser(
        "prepare-water-energy",
        help="Build GAMM analysis inputs from public_data_v1.1.",
    )
    q.add_argument("--public-data-root", required=True)
    q.add_argument("--output-dir", default="data/water_energy")
    q.add_argument(
        "--ndvi-qa",
        help="Optional MOD13A1 monthly snow/ice QA CSV for the cold/snow sensitivity.",
    )
    return p


def _baseline_args(cfg: Path) -> list[str]:
    paths = configured_paths(cfg)
    return [
        "--data-dir",
        str(paths["processed_data"]),
        "--artifact-dir",
        str(ROOT / "runtime" / "models" / "ridge_monthly_baselines"),
        "--prediction-dir",
        str(paths["predictions_dir"]),
        "--window",
        "36",
    ]


def main() -> None:
    args = parser().parse_args()
    if args.command == "help-workflow":
        print(
            "PRE-EVALUATION: canonical select-window -> rime-search -> train-seeds -> "
            "lock-seeds; then validate-workflow -> tune-models -> "
            "train-models -> select-baselines.\n"
            "EVALUATION BOUNDARY: run canonical_pipeline.py evaluate once.\n"
            "POST-BOUNDARY: evaluate-models -> evaluate-baselines -> metrics -> "
            "validate-structure -> workbook.\n"
            "UNSEEN-SITE BRANCH: fit-unseen-site -> evaluate-unseen-site using the "
            "public 247-site coordinate table."
        )
        return

    if args.command == "prepare-water-energy":
        prep_args = [
            "--public-data-root",
            str(Path(args.public_data_root).resolve()),
            "--output-dir",
            str(Path(args.output_dir).resolve()),
        ]
        if args.ndvi_qa:
            prep_args.extend(["--ndvi-qa", str(Path(args.ndvi_qa).resolve())])
        run(WATER_ENERGY_PREP_SCRIPT, *prep_args)
        return

    cfg = config_path(args.config)
    config_args = ["--config", str(cfg)]
    if args.command == "validate-workflow":
        run(MODEL_DIR / "validate_workflow.py", *config_args)
    elif args.command == "tune-models":
        run(MODEL_DIR / "validation_grid_and_lock.py", *config_args)
    elif args.command == "train-models":
        run(MODEL_DIR / "train_and_predict.py", "fit", *config_args)
    elif args.command == "select-baselines":
        run(BASELINE_SCRIPT, "select", *_baseline_args(cfg))
    elif args.command == "evaluate-models":
        run(MODEL_DIR / "train_and_predict.py", "evaluate", *config_args)
    elif args.command == "evaluate-baselines":
        run(BASELINE_SCRIPT, "evaluate", *_baseline_args(cfg))
    elif args.command == "metrics":
        run(MODEL_DIR / "analyze_model_comparison.py", *config_args)
    elif args.command == "validate-structure":
        run(MODEL_DIR / "validate_ablation_structure.py", *config_args)
    elif args.command == "workbook":
        run(MODEL_DIR / "build_model_comparison_workbook.py", *config_args)
    elif args.command == "fit-unseen-site":
        run(
            UNSEEN_TRAINING_SCRIPT,
            "fit",
            "--config",
            str(cfg),
            "--site-table",
            str(Path(args.site_table).resolve()),
        )
    elif args.command == "evaluate-unseen-site":
        run(
            UNSEEN_TRAINING_SCRIPT,
            "evaluate",
            "--config",
            str(cfg),
            "--site-table",
            str(Path(args.site_table).resolve()),
        )


if __name__ == "__main__":
    main()
