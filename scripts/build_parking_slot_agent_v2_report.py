#!/usr/bin/env python3
"""Build a reusable Part2 baseline-versus-experiment visual report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting import write_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="Qwen 0.8B baseline")
    parser.add_argument("--experiment-name", default="OpenAI VLM workflow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_comparison_report(
        args.baseline_dir,
        args.experiment_dir,
        args.output_dir,
        baseline_name=args.baseline_name,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
