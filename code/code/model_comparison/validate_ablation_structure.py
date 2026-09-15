from __future__ import annotations

import argparse
import csv

from model_workflow_common import (
    RUN_ROOT,
    architecture_signature,
    build_model,
    load_config,
    parameter_count,
    read_json,
    validate_public_config,
)


MODELS = ["Full canonical", "No Attention", "No CNN", "No BiLSTM", "No RIME"]
EXPECTED = {
    "Full canonical": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": True},
    "No Attention": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": False},
    "No CNN": {"uses_cnn": False, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": True},
    "No BiLSTM": {"uses_cnn": True, "uses_recurrent": False, "uses_bilstm": False, "uses_attention": True},
    "No RIME": {"uses_cnn": True, "uses_recurrent": True, "uses_bilstm": True, "uses_attention": True},
}
ROLE = {
    "Full canonical": "Reference model",
    "No Attention": "Controlled structural ablation",
    "No CNN": "Controlled structural ablation",
    "No BiLSTM": "Controlled structural ablation",
    "No RIME": "Optimization-strategy comparison",
}
EXPECTED_PARAMETER_COUNTS = {
    "Full canonical": 264514,
    "No Attention": 248001,
    "No CNN": 202114,
    "No BiLSTM": 28994,
    "No RIME": 99522,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the declared component structure of the workflow ablations.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_public_config(config)
    output_dir = config.path("results_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_width = int(config.training["hidden_size"])
    hp_lock_path = RUN_ROOT / "locks" / "workflow_selected_hyperparameters_lock.json"
    if not hp_lock_path.exists():
        raise FileNotFoundError(
            f"Hyperparameter lock not found: {hp_lock_path}. Run Validation-only tuning first."
        )
    hp_lock = read_json(hp_lock_path)
    no_rime_width = int(hp_lock["selected_hyperparameters"]["No RIME"]["width"])

    rows = []
    for model_name in MODELS:
        observed = architecture_signature(model_name)
        expected = EXPECTED[model_name]
        width = no_rime_width if model_name == "No RIME" else canonical_width
        model = build_model(model_name, config.input_features, width)
        count = parameter_count(model)
        expected_count = EXPECTED_PARAMETER_COUNTS[model_name]
        rows.append(
            {
                "model": model_name,
                "role": ROLE[model_name],
                **observed,
                "hidden_size": width,
                "parameter_count": count,
                "expected_parameter_count": expected_count,
                "model_class": type(model).__name__,
                "status": "PASS" if observed == expected and count == expected_count else "FAIL",
            }
        )

    output_path = output_dir / "workflow_ablation_structure_matrix.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Ablation structure verification failed.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()


