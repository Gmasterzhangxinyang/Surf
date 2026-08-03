#!/usr/bin/env python3
"""Render LiDAR projections for explicit extrinsic-convention review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from audit_camera_extrinsic import candidates, project


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--lidar-frame", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        by_frame = {int(row["frame"]): row for row in csv.DictReader(handle)}
    rows: list[dict] = []
    for frame_id in args.lidar_frame:
        row = by_frame[frame_id]
        image_path = Path(row["camera_image_path"])
        lidar_path = Path(row["lidar_path"])
        image = np.asarray(Image.open(image_path).convert("RGB"))
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        for candidate in candidates():
            camera_points = candidate.transform(points)
            uv, valid = project(camera_points, image.shape[1], image.shape[0])
            selected = np.flatnonzero(valid)
            if len(selected) > 9000:
                selected = selected[
                    np.linspace(0, len(selected) - 1, 9000, dtype=np.int64)
                ]
            depth = camera_points[selected, 2]
            order = np.argsort(depth)[::-1]
            selected = selected[order]
            figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=140)
            axis.imshow(image)
            scatter = axis.scatter(
                uv[selected, 0],
                uv[selected, 1],
                c=np.clip(camera_points[selected, 2], 1.0, 40.0),
                s=2.0,
                cmap="turbo_r",
                alpha=0.72,
                linewidths=0,
                vmin=1.0,
                vmax=40.0,
            )
            figure.colorbar(scatter, ax=axis, fraction=0.025, label="LiDAR forward depth (m)")
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
            axis.axis("off")
            axis.set_title(
                f"LiDAR {frame_id:06d} → Camera {int(float(row['camera_frame'])):06d}"
                f" | {candidate.name} | dt={float(row['camera_lidar_dt_sec'])*1000:+.1f} ms",
                fontsize=12,
                fontweight="bold",
            )
            output = args.output_dir / f"frame_{frame_id:06d}__{candidate.name}.png"
            figure.savefig(output, bbox_inches="tight", pad_inches=0.08)
            plt.close(figure)
            rows.append(
                {
                    "lidar_frame": frame_id,
                    "camera_frame": int(float(row["camera_frame"])),
                    "camera_lidar_dt_sec": float(row["camera_lidar_dt_sec"]),
                    "candidate": candidate.name,
                    "projected_point_count": int(valid.sum()),
                    "image": str(output),
                }
            )
    manifest = {
        "schema_version": "lidar-camera-extrinsic-overlay/1.0",
        "terminal_calibration_claim": False,
        "review_required": True,
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "overlay_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
