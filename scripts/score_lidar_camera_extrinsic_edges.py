#!/usr/bin/env python3
"""Rank explicit LiDAR-to-camera conventions with image-edge agreement.

This is a diagnostic, not a calibration certificate.  It scores projected
LiDAR depth discontinuities against nearby RGB image edges and compares them
with deterministic horizontal/vertical control shifts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_camera_extrinsic import candidates, project


def _gradient(image: np.ndarray) -> np.ndarray:
    gray = (
        image[..., 0].astype(np.float64) * 0.299
        + image[..., 1].astype(np.float64) * 0.587
        + image[..., 2].astype(np.float64) * 0.114
    )
    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dx, dy)
    scale = float(np.quantile(magnitude, 0.99))
    return np.clip(magnitude / max(scale, 1e-9), 0.0, 1.0)


def _nearby_max(values: np.ndarray, radius: int = 2) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    result = np.zeros_like(values)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result = np.maximum(
                result,
                padded[dy : dy + values.shape[0], dx : dx + values.shape[1]],
            )
    return result


def _depth_edges(
    points_camera: np.ndarray,
    width: int,
    height: int,
    *,
    cell_px: int = 4,
    depth_jump_m: float = 0.75,
) -> np.ndarray:
    uv, valid = project(points_camera, width, height)
    selected = points_camera[valid]
    pixels = np.rint(uv[valid]).astype(np.int64)
    depth = selected[:, 2]
    keep = (depth > 1.0) & (depth < 45.0)
    pixels = pixels[keep]
    depth = depth[keep]
    if not len(depth):
        return np.empty((0, 2), dtype=np.int64)

    grid_w = (width + cell_px - 1) // cell_px
    grid_h = (height + cell_px - 1) // cell_px
    gx = np.clip(pixels[:, 0] // cell_px, 0, grid_w - 1)
    gy = np.clip(pixels[:, 1] // cell_px, 0, grid_h - 1)
    flat = gy * grid_w + gx
    order = np.argsort(depth)
    flat = flat[order]
    depth = depth[order]
    _, first = np.unique(flat, return_index=True)
    sparse = np.full((grid_h, grid_w), np.nan, dtype=np.float64)
    sparse.flat[flat[first]] = depth[first]

    edge = np.zeros_like(sparse, dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        shifted = np.roll(sparse, shift=(dy, dx), axis=(0, 1))
        valid_pair = np.isfinite(sparse) & np.isfinite(shifted)
        edge |= valid_pair & (np.abs(sparse - shifted) >= depth_jump_m)
    gy, gx = np.nonzero(edge)
    return np.column_stack(
        [
            np.clip(gx * cell_px + cell_px // 2, 0, width - 1),
            np.clip(gy * cell_px + cell_px // 2, 0, height - 1),
        ]
    )


def _sample(values: np.ndarray, pixels: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    if not len(pixels):
        return np.empty(0, dtype=np.float64)
    x = np.clip(pixels[:, 0] + dx, 0, values.shape[1] - 1)
    y = np.clip(pixels[:, 1] + dy, 0, values.shape[0] - 1)
    return values[y, x]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--frame-from", type=int, required=True)
    parser.add_argument("--frame-to", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        source = [
            row
            for row in csv.DictReader(handle)
            if args.frame_from <= int(row["frame"]) <= args.frame_to
        ][:: max(1, args.stride)]

    totals: dict[str, list[dict[str, float]]] = {item.name: [] for item in candidates()}
    for row in source:
        image = np.asarray(Image.open(row["camera_image_path"]).convert("RGB"))
        edge_map = _nearby_max(_gradient(image), radius=2)
        points = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 4)[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        for item in candidates():
            pixels = _depth_edges(item.transform(points), image.shape[1], image.shape[0])
            observed = _sample(edge_map, pixels)
            controls = np.concatenate(
                [
                    _sample(edge_map, pixels, dx=24),
                    _sample(edge_map, pixels, dx=-24),
                    _sample(edge_map, pixels, dy=18),
                    _sample(edge_map, pixels, dy=-18),
                ]
            )
            totals[item.name].append(
                {
                    "frame": int(row["frame"]),
                    "edge_points": int(len(pixels)),
                    "observed_mean": float(np.mean(observed)) if len(observed) else 0.0,
                    "control_mean": float(np.mean(controls)) if len(controls) else 0.0,
                }
            )

    rankings = []
    for name, rows in totals.items():
        weights = np.asarray([row["edge_points"] for row in rows], dtype=np.float64)
        observed = np.asarray([row["observed_mean"] for row in rows], dtype=np.float64)
        controls = np.asarray([row["control_mean"] for row in rows], dtype=np.float64)
        denom = max(float(weights.sum()), 1.0)
        observed_mean = float(np.sum(weights * observed) / denom)
        control_mean = float(np.sum(weights * controls) / denom)
        rankings.append(
            {
                "candidate": name,
                "frame_count": len(rows),
                "edge_point_count": int(weights.sum()),
                "observed_edge_score": observed_mean,
                "shifted_control_score": control_mean,
                "edge_lift": observed_mean - control_mean,
                "per_frame": rows,
            }
        )
    rankings.sort(key=lambda row: row["edge_lift"], reverse=True)
    result = {
        "schema_version": "lidar-camera-edge-diagnostic/1.0",
        "terminal_calibration_claim": False,
        "method": "projected LiDAR depth discontinuities vs RGB gradient; ±24px x and ±18px y controls",
        "frame_range": [args.frame_from, args.frame_to],
        "stride": args.stride,
        "rankings": rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rankings"}))
    for rank, row in enumerate(rankings, start=1):
        print(
            rank,
            row["candidate"],
            f"lift={row['edge_lift']:.6f}",
            f"observed={row['observed_edge_score']:.6f}",
            f"control={row['shifted_control_score']:.6f}",
            f"points={row['edge_point_count']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
