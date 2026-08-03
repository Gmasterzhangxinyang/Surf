#!/usr/bin/env python3
"""Build the complete multi-case Chinese Agent report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_slot_agent_v2.reporting_complete_agent_zh import build_complete_agent_report


def _spec(value: str) -> tuple[str, str, str]:
    fields = value.split("::")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("active case must be RUN_DIR::PART1_JSON::SLOT_ID")
    return fields[0], fields[1], fields[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-run", type=Path, required=True)
    parser.add_argument("--active-case", action="append", type=_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_complete_agent_report(
        batch_run_dir=args.batch_run,
        active_specs=args.active_case,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
