#!/usr/bin/env python3
"""Rebuild projected point clouds from an already audited corrected trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_pose_correction.dataset import (  # noqa: E402
    lidar_to_map_xyzi,
    load_timestamp_file,
    synchronize_frame_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, default=Path("outputs/frame_map_dataset/frames.csv"))
    parser.add_argument("--pose-corrections", type=Path, default=Path("outputs/pose_drift_correction/pose_corrections.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ParkingAgent/dataset/dataset/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frame_map_dataset_pose_corrected"))
    parser.add_argument("--max-camera-delta-sec", type=float, default=0.04)
    parser.add_argument("--compress", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_delta_deg(corrected: float, original: float) -> float:
    return math.degrees((corrected - original + math.pi) % (2.0 * math.pi) - math.pi)


def main() -> None:
    args = parse_args()
    source_rows = read_csv(args.frames_csv)
    corrections = {int(row["frame"]): row for row in read_csv(args.pose_corrections)}
    missing = [int(row["frame"]) for row in source_rows if int(row["frame"]) not in corrections]
    if missing:
        raise RuntimeError(f"pose corrections missing {len(missing)} frames; examples={missing[:10]}")

    first_points = resolve(source_rows[0]["map_points_path"])
    with np.load(first_points) as data:
        map_scale = float(data["map_scale"][0])
    camera_frames, camera_timestamps = load_timestamp_file(args.dataset_root / "image" / "timestamps.txt")
    rows = synchronize_frame_rows(
        source_rows,
        camera_frames,
        camera_timestamps,
        image_dir=args.dataset_root / "image",
        max_delta_sec=args.max_camera_delta_sec,
    )

    output_dir = resolve(args.output_dir)
    points_dir = output_dir / "map_points"
    points_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    total_points = 0
    save = np.savez_compressed if args.compress else np.savez
    for index, source in enumerate(rows):
        row = dict(source)
        frame = int(row["frame"])
        correction = corrections[frame]
        original_pose = np.array(
            [float(row["map_x"]), float(row["map_y"]), float(row["map_yaw"])], dtype=np.float64
        )
        corrected_pose = np.array(
            [
                float(correction["corrected_map_x"]),
                float(correction["corrected_map_y"]),
                float(correction["corrected_map_yaw"]),
            ],
            dtype=np.float64,
        )
        raw = np.fromfile(resolve(row["lidar_path"]), dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"invalid LiDAR shape for frame {frame}")
        raw = raw.reshape(-1, 4)
        projected = lidar_to_map_xyzi(raw, corrected_pose, map_scale)
        points_path = points_dir / f"{frame:06d}.npz"
        save(
            points_path,
            points_map_xyzi=projected,
            ego_map_pose=corrected_pose.astype(np.float32),
            original_ego_map_pose=original_pose.astype(np.float32),
            frame=np.array([frame], dtype=np.int32),
            map_scale=np.array([map_scale], dtype=np.float32),
        )
        row.update(
            {
                "original_image_path": row.get("image_path", ""),
                "image_path": row["camera_image_path"],
                "map_points_path": str(points_path.relative_to(ROOT)),
                "original_map_x": float(original_pose[0]),
                "original_map_y": float(original_pose[1]),
                "original_map_yaw": float(original_pose[2]),
                "map_x": float(corrected_pose[0]),
                "map_y": float(corrected_pose[1]),
                "map_yaw": float(corrected_pose[2]),
                "pose_correction_dx_m": float(correction["correction_dx_m"]),
                "pose_correction_dy_m": float(correction["correction_dy_m"]),
                "pose_correction_dyaw_deg": normalize_delta_deg(corrected_pose[2], original_pose[2]),
                "num_points": int(len(projected)),
                "missing_image": int(not Path(row["camera_image_path"]).exists()),
                "missing_lidar": 0,
            }
        )
        output_rows.append(row)
        total_points += len(projected)
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(f"[dataset] {index + 1}/{len(rows)} points={total_points:,}", flush=True)

    fields = list(output_rows[0])
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    camera_delta = np.abs(np.asarray([float(row["camera_lidar_dt_sec"]) for row in output_rows]))
    metadata = {
        "pipeline": "frame_map_dataset_pose_corrected_v1",
        "description": "LiDAR points reprojected with audited smoothed map poses; camera images matched by timestamp.",
        "source_frames_csv": str(args.frames_csv),
        "pose_corrections": str(args.pose_corrections),
        "frame_count": len(output_rows),
        "total_projected_points": int(total_points),
        "map_scale_units_per_meter": map_scale,
        "camera_sync": {
            "valid_count": int(sum(int(row["camera_match_valid"]) for row in output_rows)),
            "abs_median_sec": float(np.median(camera_delta)),
            "abs_p95_sec": float(np.quantile(camera_delta, 0.95)),
            "abs_max_sec": float(camera_delta.max()),
        },
        "outputs": {
            "frames_csv": str(output_dir / "frames.csv"),
            "map_points_dir": str(points_dir),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
