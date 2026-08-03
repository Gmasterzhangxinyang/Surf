#!/usr/bin/env python3
"""Build a visual-first report for one audited localization trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_slot_agent_v2.reporting_localization_trace import (
    build_localization_trace_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_localization_trace_report(
        run_dir=args.run_dir,
        part1_path=args.part1,
        output_dir=args.output_dir,
        slot_id=args.slot_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
