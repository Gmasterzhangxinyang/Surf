#!/usr/bin/env python3
"""Render Part1's local map with the real Part2 Camera pre-call algorithm.

The optional Camera scene must contain the strict calibration, per-frame
Camera map pose, target ground height, and local static-obstacle map contracts.
When it is absent or incomplete the same algorithm is still executed and the
combined figure reports its fail-closed ``insufficient_information`` result;
no ego pose or placeholder calibration is substituted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.io import write_json_atomic
from parking_slot_hybrid_3d.local_map_visualization import render_local_map
from parking_slot_part2.camera_observability_map import (
    build_camera_observability_map_report,
)


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(value)


def render_combined_map(
    local_map_path: str | Path,
    *,
    output_path: str | Path,
    report_path: str | Path,
    camera_scene_path: str | Path | None = None,
    slot_database_path: str | Path | None = None,
    target_slot_ids: list[str] | None = None,
) -> dict[str, Any]:
    local_path = Path(local_map_path)
    local_map = _load_mapping(local_path, "local_map")
    camera_scene = (
        None
        if camera_scene_path is None
        else _load_mapping(Path(camera_scene_path), "camera_scene")
    )
    if slot_database_path is None:
        all_slots: Any = local_map.get("slots", ())
    else:
        database = _load_mapping(Path(slot_database_path), "slot_database")
        all_slots = database.get("slots", database)

    report = build_camera_observability_map_report(
        local_map,
        camera_scene=camera_scene,
        target_slot_ids=target_slot_ids,
    )
    image_output = Path(output_path)
    json_output = Path(report_path)
    protected_part1_image = local_path.with_name("local_map.png")
    if image_output.resolve() == protected_part1_image.resolve():
        raise ValueError(
            "combined Camera visualization must not overwrite Part1 local_map.png"
        )
    if json_output.resolve() == local_path.resolve():
        raise ValueError(
            "Camera observability report must not overwrite Part1 local_map.json"
        )
    render_local_map(
        local_map,
        all_slots,
        image_output,
        camera_overlay=report["overlay"],
    )
    report["local_map_input"] = str(local_path)
    report["camera_scene_input"] = (
        None if camera_scene_path is None else str(Path(camera_scene_path))
    )
    report["image_output"] = str(image_output)
    report["report_output"] = str(json_output)
    write_json_atomic(json_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-map", type=Path, required=True)
    parser.add_argument("--camera-scene", type=Path)
    parser.add_argument("--slot-database", type=Path)
    parser.add_argument(
        "--target-slot-id",
        action="append",
        dest="target_slot_ids",
        help="override Part1 A/B display targets; repeat at most twice",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.local_map.parent / "local_map_camera_observability.png"
    report_path = args.report or args.local_map.parent / "camera_observability_map.json"
    report = render_combined_map(
        args.local_map,
        output_path=output,
        report_path=report_path,
        camera_scene_path=args.camera_scene,
        slot_database_path=args.slot_database,
        target_slot_ids=args.target_slot_ids,
    )
    print(
        json.dumps(
            {
                "image_output": report["image_output"],
                "report_output": report["report_output"],
                "camera_geometry_drawn": report["camera_geometry_drawn"],
                "decisions": {
                    target["target_slot_id"]: target["assessment"]["decision"]
                    for target in report["targets"]
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
