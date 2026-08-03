#!/usr/bin/env python3
"""Create a derived frames CSV with a metric longitudinal pose offset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-m", type=float, required=True)
    parser.add_argument("--map-scale", type=float, required=True)
    parser.add_argument("--frame-min", type=int)
    parser.add_argument("--frame-max", type=int)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for name in (
        "longitudinal_pose_offset_m",
        "pre_longitudinal_map_x",
        "pre_longitudinal_map_y",
    ):
        if name not in fields:
            fields.append(name)

    changed = 0
    for row in rows:
        frame = int(row["frame"])
        in_scope = (
            (args.frame_min is None or frame >= args.frame_min)
            and (args.frame_max is None or frame <= args.frame_max)
        )
        if not in_scope:
            row["longitudinal_pose_offset_m"] = "0"
            continue
        x = float(row["map_x"])
        y = float(row["map_y"])
        yaw = float(row["map_yaw"])
        row["pre_longitudinal_map_x"] = repr(x)
        row["pre_longitudinal_map_y"] = repr(y)
        row["map_x"] = repr(
            x + args.offset_m * args.map_scale * math.cos(yaw)
        )
        row["map_y"] = repr(
            y + args.offset_m * args.map_scale * math.sin(yaw)
        )
        row["longitudinal_pose_offset_m"] = repr(args.offset_m)
        changed += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "offset_m": args.offset_m,
        "offset_convention": "negative=backward along heading",
        "map_scale_units_per_meter": args.map_scale,
        "frame_min": args.frame_min,
        "frame_max": args.frame_max,
        "changed_rows": changed,
        "note": "Derived visualization/annotation pose CSV; source dataset is untouched.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
