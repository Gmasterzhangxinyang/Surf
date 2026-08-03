#!/usr/bin/env python3
"""生成 frame 9277 Part2 v3 优化前后报告。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_optimization_zh import build_optimization_report


def main() -> None:
    root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277"
    result = build_optimization_report(
        part1_path=root / "part1_optimized_v1/part1_output.json",
        baseline_dir=root / "openai_gpt_5_6_terra",
        optimized_dir=root / "openai_optimized_v3",
        output_dir=root / "Part2_v3优化报告",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
