#!/usr/bin/env python3
"""生成单个车位从 Part1 到 Part2 的完整决策链报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_case_zh import build_single_case_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-id", default="slot_1012")
    args = parser.parse_args()
    root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277"
    result = build_single_case_report(
        slot_id=args.slot_id,
        part1_map=root / "part1_fresh30m/raw/local_map.png",
        baseline_dir=root / "qwen3_5_0_8b",
        experiment_dir=root / "openai_gpt_5_6_terra",
        output_dir=root / "case报告" / args.slot_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
