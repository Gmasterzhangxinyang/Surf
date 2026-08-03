#!/usr/bin/env python3
"""Build a Part1-compatible input with extended causal Part2 LiDAR resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_agent_v2.extended_lidar import build_extended_part2_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--dataset-id", default="pose-corrected-final")
    args = parser.parse_args()
    part1 = Part1Output.from_dict(
        json.loads(args.part1.read_text(encoding="utf-8"))
    )
    result = build_extended_part2_input(
        part1,
        frames_csv=args.frames_csv,
        slot_database=args.slot_db,
        map_points_dir=args.map_points_dir,
        output_dir=args.output_dir,
        window_frames=args.window,
        dataset_id=args.dataset_id,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
