#!/usr/bin/env python3
"""Build a small derived map-point dataset after a longitudinal pose tune."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def resolve(path: str, csv_path: Path) -> Path:
    p = Path(path)
    for candidate in (p, csv_path.parent / p, Path.cwd() / p):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--offset-m", type=float, required=True)
    p.add_argument("--map-scale", type=float, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    points_dir = a.output_dir / "map_points"
    points_dir.mkdir()

    with a.frames.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        source_rows = list(reader)
        fields = list(reader.fieldnames or [])
    for key in (
        "longitudinal_pose_offset_m",
        "pre_longitudinal_map_x",
        "pre_longitudinal_map_y",
    ):
        if key not in fields:
            fields.append(key)

    rows = []
    for row in source_rows:
        frame = int(row["frame"])
        if not a.start_frame <= frame <= a.end_frame:
            continue
        yaw = float(row["map_yaw"])
        delta = np.asarray(
            [
                a.offset_m * a.map_scale * math.cos(yaw),
                a.offset_m * a.map_scale * math.sin(yaw),
            ],
            dtype=np.float64,
        )
        source_npz = resolve(row["map_points_path"], a.frames)
        with np.load(source_npz) as data:
            arrays = {key: data[key] for key in data.files}
        points = arrays["points_map_xyzi"].astype(np.float64)
        points[:, :2] += delta
        arrays["points_map_xyzi"] = points.astype(np.float32)
        if "ego_map_pose" in arrays:
            ego = arrays["ego_map_pose"].astype(np.float64)
            ego[:2] += delta
            arrays["ego_map_pose"] = ego.astype(np.float32)
        output_npz = points_dir / f"{frame:06d}.npz"
        np.savez_compressed(output_npz, **arrays)

        old_x, old_y = float(row["map_x"]), float(row["map_y"])
        row["pre_longitudinal_map_x"] = repr(old_x)
        row["pre_longitudinal_map_y"] = repr(old_y)
        row["map_x"] = repr(float(old_x + delta[0]))
        row["map_y"] = repr(float(old_y + delta[1]))
        row["map_points_path"] = str(output_npz.resolve())
        row["longitudinal_pose_offset_m"] = repr(a.offset_m)
        rows.append(row)

    output_csv = a.output_dir / "frames.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "pipeline": "longitudinally_tuned_map_window/1.0",
        "source_frames_csv": str(a.frames.resolve()),
        "frame_start": a.start_frame,
        "frame_end": a.end_frame,
        "frame_count": len(rows),
        "longitudinal_offset_m": a.offset_m,
        "offset_convention": "negative=backward along heading",
        "map_scale_units_per_meter": a.map_scale,
        "points_and_pose_shifted_together": True,
        "source_untouched": True,
    }
    (a.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
