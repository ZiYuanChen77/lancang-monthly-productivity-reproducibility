"""Run the R2 analysis modules using the repository's existing public inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
NDVI = CODE / "code/ndvi_quality"
WATER = CODE / "code/water_energy_analysis"
SPATIAL = CODE / "code/evaluation/spatial_validation/spatial_validation.py"
RUNTIME = CODE / "runtime/revision_r2"


def run(script, *args):
    if script.suffix == ".R":
        executable = os.environ.get("RSCRIPT") or shutil.which("Rscript")
        if not executable:
            raise RuntimeError("Rscript was not found; add R to PATH or set RSCRIPT.")
    else:
        executable = sys.executable
    subprocess.run([str(executable), str(script), *map(str, args)], cwd=CODE, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Rebuild NDVI keys and GAMM inputs from public_data")
    q.add_argument("--ndvi-qa", type=Path, help="Optional MOD13A1 monthly QA export")
    q = sub.add_parser("clean", help="Summarize canonical or paired three-seed predictions")
    q.add_argument("--input", type=Path, default=CODE / "runtime/runs/canonical/evaluation/evaluation_predictions_all_seeds.csv")
    q.add_argument("--paired", type=Path)
    q.add_argument("--arm")
    q.add_argument("--paired-arm")
    q.add_argument("--evaluation-keys", type=Path, help="Use common_evaluation_11818.csv for spatial pairing")
    q.add_argument("--output", type=Path, default=RUNTIME / "statistics")
    q.add_argument("--bootstrap", type=int, default=5000)
    sub.add_parser("shap", help="Filter and summarize the existing three-seed SHAP caches")
    sub.add_parser("shift", help="Summarize shifts in the prepared W36 inputs")
    q = sub.add_parser("spatial", help="Run fold-specific window selection and RIME")
    q.add_argument("--stage", choices=["prepare", "check", "fit-fold", "run-all", "evaluate", "bootstrap"], default="run-all")
    q.add_argument("--fold", type=int, choices=range(1, 6))
    q = sub.add_parser("water", help="Fit and jointly bootstrap six water-energy comparisons")
    q.add_argument("--replicates", type=int, default=2000)
    q.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    prepared = RUNTIME / "ndvi_clean"

    if args.command == "prepare":
        run(NDVI / "ndvi_clean_domain.py", "prepare", "--source-root", ROOT, "--output", prepared)
        options = ["--public-data-root", ROOT / "public_data", "--output-dir", CODE / "data/water_energy"]
        if args.ndvi_qa:
            options.extend(["--ndvi-qa", args.ndvi_qa.resolve()])
        run(WATER / "prepare_water_energy_inputs.py", *options)
    elif args.command == "clean":
        options = ["--input", args.input.resolve(), "--prepared", prepared,
                   "--output", args.output.resolve(), "--bootstrap", args.bootstrap,
                   "--residual-seed", 42]
        for name in ("paired", "evaluation_keys", "arm", "paired_arm"):
            value = getattr(args, name)
            if value is not None:
                options.extend(["--" + name.replace("_", "-"), value.resolve() if isinstance(value, Path) else value])
        run(NDVI / "clean_statistics.py", *options)
    elif args.command == "shap":
        manifest = CODE / "data_documentation/shap_sample_keys/sample_keys/shap_explained_E0.csv"
        caches = []
        for seed, name in [(42, "background_BG256_D1"), (2024, "seed_2024"), (3407, "seed_3407")]:
            source = CODE / "runtime/runs/shap/shap_runs" / f"shap_{name}.npz"
            target = RUNTIME / "shap_cache" / f"seed_{seed}.npz"
            run(NDVI / "ndvi_clean_domain.py", "filter-shap", "--input", source,
                "--manifest", manifest, "--prepared", prepared, "--output", target)
            caches.append(target)
        run(NDVI / "clean_shap_summary.py", "--caches", *caches,
            "--manifest", prepared / "shap_explained_E0_clean.csv",
            "--invalid-inputs", prepared / "invalid_ndvi_inputs.csv", "--output", RUNTIME / "shap_summary")
    elif args.command == "shift":
        data = CODE / "data/processed"
        run(NDVI / "clean_input_shift.py", "--tensor", data / "X_tensor_aligned_36.npy",
            "--index", data / "Sample_index_aligned.csv", "--metadata", data / "preprocessing_metadata.json",
            "--invalid-inputs", prepared / "invalid_ndvi_inputs.csv", "--output", RUNTIME / "predictor_shift.csv")
    elif args.command == "spatial":
        options = [args.stage, "--source-root", ROOT, "--data-dir", ROOT / "public_data/model_input",
                   "--output-root", CODE / "runtime/spatial_validation"]
        if args.fold is not None:
            options.extend(["--fold", args.fold])
        run(SPATIAL, *options)
    elif args.command == "water":
        run(WATER / "run_comparisons.R", "--input", CODE / "data/water_energy/water_energy_analysis_ready_2022_2025.csv",
            "--output", CODE / "outputs/water_energy/comparisons", "--replicates", args.replicates, "--workers", args.workers)


if __name__ == "__main__":
    main()
