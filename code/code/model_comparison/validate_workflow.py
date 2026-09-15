from __future__ import annotations

import argparse
from pathlib import Path

from model_workflow_common import (
    SEEDS,
    load_config,
    validate_canonical_state,
    validate_public_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the W36 Train/Validation workflow before comparison-model "
            "tuning and fitting."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_public_config(config)

    state = validate_canonical_state(require_evaluation=False)
    if state.get("state") != "NOT_EVALUATED":
        raise RuntimeError(
            "Comparison-model selection must be run before canonical Evaluation. "
            "Use a fresh run directory to repeat model selection."
        )

    print(
        "Workflow validated: "
        f"W{state['final_selected_window']}; "
        f"seeds={','.join(map(str, SEEDS))}; "
        "selection data scope=Train/Validation."
    )


if __name__ == "__main__":
    main()
