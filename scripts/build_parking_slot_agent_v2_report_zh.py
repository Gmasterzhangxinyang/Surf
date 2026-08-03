#!/usr/bin/env python3
"""生成 frame 9277 Part2 OpenAI Agent 的完整中文研究报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_zh import build_comprehensive_chinese_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part1",
        type=Path,
        default=PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277/part1_fresh30m/part1_output.json",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277/qwen3_5_0_8b",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277/openai_gpt_5_6_terra",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277/完整中文报告",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_comprehensive_chinese_report(
        part1_path=args.part1,
        baseline_dir=args.baseline_dir,
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
