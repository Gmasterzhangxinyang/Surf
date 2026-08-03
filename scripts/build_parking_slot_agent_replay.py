#!/usr/bin/env python3
"""Build the visual-first Part2 agent replay demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_slot_agent_v2.reporting_agent_replay import build_agent_replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=ROOT / "outputs/parking_slot_agent_v2_frame_9277/openai_extended60_exhaustive_v1",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "outputs/parking_slot_agent_v2_frame_9277/agent_replay_demo",
        type=Path,
    )
    args = parser.parse_args()
    result = build_agent_replay(run_dir=args.run_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
