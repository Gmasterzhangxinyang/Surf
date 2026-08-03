#!/usr/bin/env python3
"""Fit a 2D similarity transform from pose CSV to map coordinates.

Example:
  python3 scripts/fit_pose_to_map_correspondences.py \
    --pose-csv outputs/azimuth_time_odometry_compatible.csv \
    --pairs "6,5.6,5.5;4000,8.2,10.1;11678,3.8,13.1" \
    --output-dir outputs/manual_pose_map_alignment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit pose trajectory to map using frame,map_x,map_y correspondences")
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--pairs", required=True, help='semicolon-separated "frame,map_x,map_y" entries')
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_pose(path: Path) -> tuple[list[dict[str, str]], dict[int, np.ndarray]]:
    rows = list(csv.DictReader(path.open(newline="")))
    by_frame = {int(r["frame"]): np.array([float(r["x"]), float(r["y"])], dtype=np.float64) for r in rows}
    return rows, by_frame


def parse_pairs(text: str, by_frame: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    src = []
    dst = []
    frames = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"bad pair: {item}")
        frame = int(parts[0])
        if frame not in by_frame:
            raise ValueError(f"frame {frame} not in pose csv")
        src.append(by_frame[frame])
        dst.append([float(parts[1]), float(parts[2])])
        frames.append(frame)
    if len(src) < 2:
        raise ValueError("need at least two correspondences")
    return np.asarray(src), np.asarray(dst), frames


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    a = src - src_c
    b = dst - dst_c
    cov = a.T @ b / len(src)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    var = float((a * a).sum() / len(src))
    scale = float(s.sum() / max(var, 1e-12))
    trans = dst_c - scale * (src_c @ r.T)
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    pred = scale * (src @ r.T) + trans
    rmse = float(np.sqrt(((pred - dst) ** 2).sum(axis=1).mean()))
    return scale, yaw, trans, rmse


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, by_frame = load_pose(args.pose_csv)
    src, dst, frames = parse_pairs(args.pairs, by_frame)
    scale, yaw, trans, rmse = fit_similarity(src, dst)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)

    csv_path = args.output_dir / "manual_aligned_trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "map_x", "map_y", "map_yaw"])
        writer.writeheader()
        for row in rows:
            xy = np.array([float(row["x"]), float(row["y"])], dtype=np.float64)
            mapped = scale * (xy @ rot.T) + trans
            yaw_pose = float(row.get("yaw_rad", 0.0)) + yaw
            yaw_pose = math.atan2(math.sin(yaw_pose), math.cos(yaw_pose))
            writer.writerow({"frame": int(row["frame"]), "map_x": mapped[0], "map_y": mapped[1], "map_yaw": yaw_pose})

    summary = {
        "pose_csv": str(args.pose_csv),
        "pairs_frames": frames,
        "scale": scale,
        "yaw": yaw,
        "translation": trans.tolist(),
        "fit_rmse_map_units": rmse,
        "output_csv": str(csv_path),
    }
    (args.output_dir / "alignment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[alignment] {args.output_dir / 'alignment.json'}")
    print(f"[csv] {csv_path}")
    print(f"[fit] scale={scale:.8f} yaw={yaw:.6f} trans=({trans[0]:.4f},{trans[1]:.4f}) rmse={rmse:.4f}")


if __name__ == "__main__":
    main()
