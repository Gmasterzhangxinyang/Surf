#!/usr/bin/env python3
"""Estimate a left-mounted camera yaw/offset from LiDAR-to-RGB edge agreement."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from score_lidar_camera_extrinsic_edges import (
    _depth_edges,
    _gradient,
    _nearby_max,
    _sample,
)


OPTICAL_FROM_BODY = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def _camera_points(
    points_lidar: np.ndarray,
    *,
    origin_lidar_m: np.ndarray,
    yaw_deg: float,
) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    camera_body_in_lidar = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    body = (points_lidar - origin_lidar_m) @ camera_body_in_lidar
    return body @ OPTICAL_FROM_BODY.T


def _score(
    frames: list[dict],
    origin: np.ndarray,
    yaw_deg: float,
) -> dict[str, float]:
    observed_sum = 0.0
    control_sum = 0.0
    weight_sum = 0
    frame_lifts = []
    for frame in frames:
        camera_points = _camera_points(
            frame["points"],
            origin_lidar_m=origin,
            yaw_deg=yaw_deg,
        )
        pixels = _depth_edges(
            camera_points,
            frame["width"],
            frame["height"],
            cell_px=5,
            depth_jump_m=0.8,
        )
        observed = _sample(frame["edges"], pixels)
        controls = np.concatenate(
            [
                _sample(frame["edges"], pixels, dx=28),
                _sample(frame["edges"], pixels, dx=-28),
                _sample(frame["edges"], pixels, dy=20),
                _sample(frame["edges"], pixels, dy=-20),
            ]
        )
        weight = len(pixels)
        if not weight:
            continue
        observed_mean = float(np.mean(observed))
        control_mean = float(np.mean(controls))
        observed_sum += weight * observed_mean
        control_sum += weight * control_mean
        weight_sum += weight
        frame_lifts.append(observed_mean - control_mean)
    if not weight_sum:
        return {
            "edge_lift": float("-inf"),
            "observed": 0.0,
            "control": 0.0,
            "edge_points": 0,
            "positive_frame_rate": 0.0,
        }
    return {
        "edge_lift": (observed_sum - control_sum) / weight_sum,
        "observed": observed_sum / weight_sum,
        "control": control_sum / weight_sum,
        "edge_points": weight_sum,
        "positive_frame_rate": float(np.mean(np.asarray(frame_lifts) > 0.0)),
    }


def _axis(start: float, stop: float, step: float) -> list[float]:
    count = int(math.floor((stop - start) / step + 1e-9))
    return [start + index * step for index in range(count + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--frame-from", type=int, required=True)
    parser.add_argument("--frame-to", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--x-m", type=float, default=0.6)
    parser.add_argument("--z-m", type=float, default=-0.07)
    parser.add_argument("--y-min-m", type=float, default=0.0)
    parser.add_argument("--y-max-m", type=float, default=1.5)
    parser.add_argument("--y-step-m", type=float, default=0.1)
    parser.add_argument("--yaw-min-deg", type=float, default=0.0)
    parser.add_argument("--yaw-max-deg", type=float, default=60.0)
    parser.add_argument("--yaw-step-deg", type=float, default=2.5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if args.frame_from <= int(row["frame"]) <= args.frame_to
        ][:: max(1, args.frame_stride)]
    frames = []
    for row in selected:
        image = np.asarray(Image.open(row["camera_image_path"]).convert("RGB"))
        points = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 4)[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        frames.append(
            {
                "frame": int(row["frame"]),
                "points": points,
                "edges": _nearby_max(_gradient(image), radius=2),
                "height": image.shape[0],
                "width": image.shape[1],
            }
        )
    if not frames:
        raise ValueError("no frames selected")

    results = []
    for y_m in _axis(args.y_min_m, args.y_max_m, args.y_step_m):
        origin = np.asarray([args.x_m, y_m, args.z_m], dtype=np.float64)
        for yaw_deg in _axis(
            args.yaw_min_deg,
            args.yaw_max_deg,
            args.yaw_step_deg,
        ):
            metrics = _score(frames, origin, yaw_deg)
            results.append(
                {
                    "x_m": args.x_m,
                    "y_m": y_m,
                    "z_m": args.z_m,
                    "yaw_deg": yaw_deg,
                    **metrics,
                }
            )
    results.sort(
        key=lambda row: (
            row["edge_lift"],
            row["positive_frame_rate"],
            row["edge_points"],
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "left-camera-extrinsic-edge-grid/1.0",
        "terminal_calibration_claim": False,
        "frame_ids": [frame["frame"] for frame in frames],
        "search": {
            "x_m": args.x_m,
            "z_m": args.z_m,
            "y_m": [args.y_min_m, args.y_max_m, args.y_step_m],
            "yaw_deg": [args.yaw_min_deg, args.yaw_max_deg, args.yaw_step_deg],
        },
        "best": results[0],
        "top": results[: args.top_k],
        "all": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["best"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
