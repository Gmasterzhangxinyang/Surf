#!/usr/bin/env python3
"""Render a compact visual grid for left-camera extrinsic candidates."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from audit_camera_extrinsic import project
from calibrate_left_camera_extrinsic import _camera_points


def _numbers(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--x-m", type=float, default=0.6)
    parser.add_argument("--z-m", type=float, default=-0.07)
    parser.add_argument("--y-values-m", default="0,0.2,0.6")
    parser.add_argument("--yaw-values-deg", default="-30,-15,0,15,30")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        row = next(
            row for row in csv.DictReader(handle) if int(row["frame"]) == args.frame
        )
    image = np.asarray(Image.open(row["camera_image_path"]).convert("RGB"))
    points = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 4)[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    y_values = _numbers(args.y_values_m)
    yaw_values = _numbers(args.yaw_values_deg)
    combinations = list(itertools.product(y_values, yaw_values))

    figure, axes = plt.subplots(
        len(y_values),
        len(yaw_values),
        figsize=(4.2 * len(yaw_values), 2.55 * len(y_values)),
        dpi=135,
        squeeze=False,
    )
    for axis, (y_m, yaw_deg) in zip(axes.flat, combinations):
        origin = np.asarray([args.x_m, y_m, args.z_m], dtype=np.float64)
        camera = _camera_points(points, origin_lidar_m=origin, yaw_deg=yaw_deg)
        uv, valid = project(camera, image.shape[1], image.shape[0])
        selected = np.flatnonzero(valid)
        if len(selected) > 6000:
            selected = selected[
                np.linspace(0, len(selected) - 1, 6000, dtype=np.int64)
            ]
        depth = camera[selected, 2]
        order = np.argsort(depth)[::-1]
        selected = selected[order]
        axis.imshow(image)
        axis.scatter(
            uv[selected, 0],
            uv[selected, 1],
            c=np.clip(camera[selected, 2], 1.0, 40.0),
            s=1.2,
            cmap="turbo_r",
            alpha=0.70,
            linewidths=0,
            vmin=1.0,
            vmax=40.0,
        )
        axis.set_title(f"left offset y={y_m:+.2f}m | left yaw={yaw_deg:+.1f}°")
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.axis("off")
    figure.suptitle(
        f"LiDAR {args.frame:06d} -> camera {int(float(row['camera_frame'])):06d}: "
        "choose the panel whose colored LiDAR surfaces align with visible objects",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
