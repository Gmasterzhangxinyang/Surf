#!/usr/bin/env python3
"""Build the detailed bilingual parking-slot paper package."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_full_paper import build_full_paper_package


def main() -> None:
    paper_root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper"
    result = build_full_paper_package(
        paper_root=paper_root,
        source_report_dir=paper_root / "Nature风格中文研究稿",
        output_dir=paper_root / "CVPR完整论文包",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
