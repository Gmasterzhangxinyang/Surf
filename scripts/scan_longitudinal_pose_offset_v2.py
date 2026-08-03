#!/usr/bin/env python3
"""Measure a shared forward/backward pose error using causal LiDAR and GLTF."""

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


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-csv", type=Path, required=True)
    p.add_argument("--gltf", type=Path, required=True)
    p.add_argument("--anchor-frame", type=int, default=6241)
    p.add_argument("--history-count", type=int, default=17)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--min-offset-m", type=float, default=-1.2)
    p.add_argument("--max-offset-m", type=float, default=0.4)
    p.add_argument("--step-m", type=float, default=0.02)
    p.add_argument("--points-per-frame", type=int, default=900)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def fid(row: dict[str, str]) -> int:
    return int(row.get("frame", row.get("frame_id", "-1")))


def read_rows(path: Path, anchor: int, count: int, stride: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {fid(row): row for row in rows if fid(row) <= anchor}
    chosen = [by_id[x] for x in reversed([anchor - i * stride for i in range(count)]) if x in by_id]
    if not chosen:
        raise RuntimeError("No causal rows found")
    return chosen


def resolve(path: str, csv_path: Path) -> Path:
    candidate = Path(path)
    for p in (candidate, csv_path.parent / candidate, ROOT / candidate):
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(path)


def pose(row: dict[str, str]) -> tuple[float, float, float]:
    return float(row["map_x"]), float(row["map_y"]), float(row["map_yaw"])


def load_scans(
    rows: list[dict[str, str]], csv_path: Path, scale: float, limit: int
) -> list[dict]:
    rng = np.random.default_rng(6241)
    scans = []
    for row in rows:
        with np.load(resolve(row["map_points_path"], csv_path)) as d:
            pts = d["points_map_xyzi"].astype(np.float64)
        x, y, yaw = pose(row)
        radius_m = np.hypot(pts[:, 0] - x, pts[:, 1] - y) / scale
        keep = (
            np.isfinite(pts[:, :3]).all(axis=1)
            & (radius_m >= 2.0)
            & (radius_m <= 25.0)
            & (pts[:, 2] >= 0.3)
            & (pts[:, 2] <= 2.2)
        )
        xy = pts[keep, :2]
        if len(xy) > limit:
            xy = xy[rng.choice(len(xy), limit, replace=False)]
        scans.append(
            {
                "frame": fid(row),
                "xy": xy,
                "heading": np.array([math.cos(yaw), math.sin(yaw)]),
            }
        )
    return scans


def metric(scans: list[dict], tree: cKDTree, offset_m: float, scale: float) -> dict:
    chunks = []
    frame_medians = []
    for scan in scans:
        moved = scan["xy"] + offset_m * scale * scan["heading"][None, :]
        distance_m = tree.query(moved, k=1, workers=-1)[0] / scale
        chunks.append(distance_m)
        frame_medians.append(float(np.median(distance_m)))
    d = np.concatenate(chunks)
    trimmed = d[d <= np.quantile(d, 0.35)]
    return {
        "offset_m": round(float(offset_m), 6),
        "points": int(len(d)),
        "median_m": float(np.median(d)),
        "trimmed35_rmse_m": float(np.sqrt(np.mean(trimmed * trimmed))),
        "inlier_025": float(np.mean(d <= 0.25)),
        "inlier_075": float(np.mean(d <= 0.75)),
        "mean_frame_median_m": float(np.mean(frame_medians)),
    }


def main() -> None:
    a = args()
    metadata = json.loads((a.frames_csv.parent / "metadata.json").read_text(encoding="utf-8"))
    scale = float(metadata["map_scale_units_per_meter"])
    rows = read_rows(a.frames_csv, a.anchor_frame, a.history_count, a.frame_stride)
    scans = load_scans(rows, a.frames_csv, scale, a.points_per_frame)

    gltf = load_gltf_map(a.gltf, sample_step=0.025 * scale)
    chunks = [
        gltf.layers[name].points
        for name in ("wall", "elevator", "arrester")
        if name in gltf.layers and len(gltf.layers[name].points)
    ]
    static = np.vstack(chunks).astype(np.float64)
    all_scan = np.vstack([scan["xy"] for scan in scans])
    lo = all_scan.min(axis=0) - 3.0 * scale
    hi = all_scan.max(axis=0) + 3.0 * scale
    static = static[np.all((static >= lo) & (static <= hi), axis=1)]
    if not len(static):
        raise RuntimeError("Static map crop is empty")
    tree = cKDTree(static)

    n = int(round((a.max_offset_m - a.min_offset_m) / a.step_m)) + 1
    offsets = np.linspace(a.min_offset_m, a.max_offset_m, n)
    values = [metric(scans, tree, float(offset), scale) for offset in offsets]
    ranked = sorted(
        values,
        key=lambda x: (x["trimmed35_rmse_m"], x["median_m"], -x["inlier_025"]),
    )
    zero = min(values, key=lambda x: abs(x["offset_m"]))
    best = ranked[0]
    result = {
        "method": "causal multi-frame LiDAR/static-GLTF longitudinal scan",
        "offset_convention": "negative means backward along vehicle heading",
        "anchor_frame": a.anchor_frame,
        "frames": [scan["frame"] for scan in scans],
        "map_scale_units_per_meter": scale,
        "static_layers": ["wall", "elevator", "arrester"],
        "static_points": int(len(static)),
        "best": best,
        "current_zero": zero,
        "trimmed_rmse_improvement_fraction": (
            zero["trimmed35_rmse_m"] - best["trimmed35_rmse_m"]
        )
        / zero["trimmed35_rmse_m"],
        "ranked": ranked,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "ranked"}, indent=2))


if __name__ == "__main__":
    main()
