#!/usr/bin/env python3
"""生成 ParkingSlotAgent v2 Nature 风格中文研究稿与高清图集。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_nature_zh import build_nature_style_report


def main() -> None:
    frame_root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277"
    result = build_nature_style_report(
        window_ablation_path=PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper/window_ablation_frame_9277.json",
        cross_anchor_baseline_path=PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper/cross_anchor_baseline15_systematic6.json",
        cross_anchor_extended_path=PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper/cross_anchor_extended60_systematic6.json",
        openai_result_path=frame_root / "openai_extended60_exhaustive_v1/part2_result.json",
        operational_result_path=frame_root / "openai_extended60_operational_v1/part2_result.json",
        legacy_manifest_path=PROJECT_ROOT / "outputs/parking_slot_agent_v2_multi_anchor_30_codex/run_manifest.json",
        legacy_summary_path=PROJECT_ROOT / "outputs/parking_slot_agent_v2_multi_anchor_30_codex/codex_agent_summary.json",
        output_dir=PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper/Nature风格中文研究稿",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
