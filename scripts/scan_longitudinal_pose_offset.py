#!/usr/bin/env python3
"""Scan a shared longitudinal pose offset against the static GLTF map.

Negative offsets move every evaluated vehicle pose backward along its heading.
The score is based on causal multi-frame LiDAR alignment to static structures,
not parking-slot occupancy returns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gltf_lidar_ndt import load_gltf_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=6241)
    parser.add_argument("--history-count", type=int, default=17)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--min-offset-m", type=float, default=-1.2)
    parser.add_argument("--max-offset-m", type=float, default=0.4)
    parser.add_argument("--step-m", type=float, default=0.02)
    parser.add_argument("--points-per-frame", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path, anchor: int, count: int, stride: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if int(row["frame_id"]) <= anchor]
    by_id = {int(row["frame_id"]): row for row in rows}
    wanted = [anchor - i * stride for i in range(count)]
    selected = [by_id[fid] for fid in reversed(wanted) if fid in by_id]
    if not selected:
        raise RuntimeError(f"No causal frames found up to {anchor} in {path}")
    return selected


def resolve_npz(row: dict, frames_csv: Path) -> Path:
    candidates = []
    for key in ("corrected_map_path", "map_npz", "npz_path", "pointcloud_path"):
        value = row.get(key)
        if value:
            p = Path(value)
            candidates.extend([p, frames_csv.parent / p, ROOT / p])
    frame_id = int(row["frame_id"])
    candidates.extend(
        [
            frames_csv.parent / "corrected_map_points" / f"{frame_id:06d}.npz",
            frames_csv.parent / "corrected_map_points" / f"frame_{frame_id}.npz",
            frames_csv.parent / "map_points" / f"{frame_id:06d}.npz",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Cannot resolve map point NPZ for frame {frame_id}")


def load_xyz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in ("points", "xyz", "map_points", "arr_0"):
            if key in data:
                points = np.asarray(data[key], dtype=np.float64)
                break
        else:
            raise KeyError(f"No point array in {path}; keys={list(data.keys())}")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Unexpected point shape {points.shape} in {path}")
    return points[:, :3]


def pose_value(row: dict, name: str) -> float:
    alternatives = {
        "x": ("map_x", "corrected_x", "x"),
        "y": ("map_y", "corrected_y", "y"),
        "yaw": ("map_yaw", "corrected_yaw", "yaw"),
    }
    for key in alternatives[name]:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing {name} pose field")


def prepare_scans(
    rows: list[dict], frames_csv: Path, points_per_frame: int
) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(6241)
    scans: list[dict] = []
    all_points = []
    for row in rows:
        x, y, yaw = pose_value(row, "x"), pose_value(row, "y"), pose_value(row, "yaw")
        points = load_xyz(resolve_npz(row, frames_csv))
        dx, dy = points[:, 0] - x, points[:, 1] - y
        distance = np.hypot(dx, dy)
        rel_z = points[:, 2]
        keep = (
            np.isfinite(points).all(axis=1)
            & (distance >= 2.0)
            & (distance <= 25.0)
            & (rel_z >= 0.3)
            & (rel_z <= 2.2)
        )
        points = points[keep]
        if len(points) > points_per_frame:
            points = points[rng.choice(len(points), points_per_frame, replace=False)]
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        scans.append(
            {
                "frame_id": int(row["frame_id"]),
                "pose": [x, y, yaw],
                "heading": heading,
                "points": points,
            }
        )
        all_points.append(points)
    return scans, np.concatenate(all_points, axis=0)


def score_offset(
    scans: list[dict], offset: float, tree: cKDTree
) -> dict[str, float | int]:
    distances = []
    frame_medians = []
    for scan in scans:
        shifted = scan["points"] + offset * scan["heading"][None, :]
        dists, _ = tree.query(shifted, k=1, workers=-1)
        distances.append(dists)
        frame_medians.append(float(np.median(dists)))
    dists = np.concatenate(distances)
    cutoff = np.quantile(dists, 0.35)
    trimmed = dists[dists <= cutoff]
    return {
        "offset_m": round(float(offset), 6),
        "point_count": int(len(dists)),
        "median_m": float(np.median(dists)),
        "trimmed35_rmse_m": float(np.sqrt(np.mean(trimmed**2))),
        "inlier_025": float(np.mean(dists <= 0.25)),
        "inlier_075": float(np.mean(dists <= 0.75)),
        "mean_frame_median_m": float(np.mean(frame_medians)),
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(
        args.frames_csv, args.anchor_frame, args.history_count, args.frame_stride
    )
    scans, scan_points = prepare_scans(rows, args.frames_csv, args.points_per_frame)
    bbox_min = np.nanmin(scan_points, axis=0) - np.array([3.0, 3.0, 2.0])
    bbox_max = np.nanmax(scan_points, axis=0) + np.array([3.0, 3.0, 2.0])
    static_points = load_gltf_map(
        args.gltf,
        include_names=("wall", "elevator", "arrester"),
        sample_spacing=0.025,
    )
    static_points = np.asarray(static_points, dtype=np.float64)[:, :3]
    keep = np.all((static_points >= bbox_min) & (static_points <= bbox_max), axis=1)
    static_points = static_points[keep]
    if len(static_points) == 0:
        raise RuntimeError("Static map crop is empty")
    tree = cKDTree(static_points)

    count = int(round((args.max_offset_m - args.min_offset_m) / args.step_m)) + 1
    offsets = np.linspace(args.min_offset_m, args.max_offset_m, count)
    scores = [score_offset(scans, float(offset), tree) for offset in offsets]
    ranked = sorted(
        scores,
        key=lambda row: (
            row["trimmed35_rmse_m"],
            row["median_m"],
            -row["inlier_025"],
        ),
    )
    current = min(scores, key=lambda row: abs(row["offset_m"]))
    best = ranked[0]
    result = {
        "method": "causal multi-frame longitudinal scan against GLTF static structures",
        "offset_convention": "negative=backward along each frame heading",
        "anchor_frame": args.anchor_frame,
        "frames": [scan["frame_id"] for scan in scans],
        "static_layers": ["wall", "elevator", "arrester"],
        "static_point_count": int(len(static_points)),
        "best": best,
        "current_zero": current,
        "trimmed_rmse_improvement_fraction": float(
            (current["trimmed35_rmse_m"] - best["trimmed35_rmse_m"])
            / current["trimmed35_rmse_m"]
        ),
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "ranked"}, indent=2))


if __name__ == "__main__":
    main()
