#!/usr/bin/env python3
"""生成 frame 9277 Part2 75帧严格因果优化报告。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_extended75_zh import build_extended75_report


def main() -> None:
    root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277"
    result = build_extended75_report(
        part1_path=root / "part1_optimized_v2_extended75/part1_output.json",
        exhaustive_result=root / "openai_extended75_exhaustive_v2/part2_result.json",
        targeted_1258_result=root / "openai_extended75_slot_1258_regression_v1/part2_result.json",
        operational_result=root / "openai_extended75_operational_v1/part2_result.json",
        output_dir=root / "Part2_v4_75帧优化报告",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
