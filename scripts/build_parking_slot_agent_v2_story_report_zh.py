#!/usr/bin/env python3
"""生成以结果为先、面向人工阅读的 Part2 直观中文报告。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_story_zh import build_intuitive_chinese_report


def main() -> None:
    root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277"
    result = build_intuitive_chinese_report(
        baseline_dir=root / "qwen3_5_0_8b",
        experiment_dir=root / "openai_gpt_5_6_terra",
        technical_report=root / "完整中文报告/frame_9277_part2_openai完整中文报告.html",
        output_dir=root / "直观中文报告",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
