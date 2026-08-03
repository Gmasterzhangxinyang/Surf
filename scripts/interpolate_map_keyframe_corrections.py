#!/usr/bin/env python3
"""Interpolate NDT keyframe corrections to all LiDAR trajectory frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import draw_gltf_map, load_gltf_map, normalize_angle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply interpolated NDT keyframe corrections to all map poses")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    return parser.parse_args()


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def render(path: Path, gltf_map: object, init: np.ndarray, corrected: np.ndarray, key_xy: np.ndarray) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=220)
    draw_gltf_map(ax, gltf_map)
    ax.plot(init[:, 0], init[:, 1], color="#64748b", linewidth=1.0, alpha=0.45, label="initial Sim2 trajectory")
    ax.plot(corrected[:, 0], corrected[:, 1], color="#dc2626", linewidth=2.0, label="keyframe-corrected trajectory")
    if len(key_xy):
        ax.scatter(key_xy[:, 0], key_xy[:, 1], s=5, c="#2563eb", alpha=0.8, linewidths=0, label="NDT keyframes")
    ax.scatter([corrected[0, 0]], [corrected[0, 1]], s=34, c="#16a34a", zorder=8, label="start")
    ax.scatter([corrected[-1, 0]], [corrected[-1, 1]], s=34, c="#dc2626", zorder=8, label="end")
    ax.set_title("Full LiDAR trajectory matched to GLTF map")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.input_csv.open(newline="")))
    if not rows:
        raise SystemExit("empty input CSV")

    idx = np.arange(len(rows), dtype=np.float64)
    init = np.asarray([[float(r["init_x"]), float(r["init_y"]), float(r["init_yaw"])] for r in rows], dtype=np.float64)
    ndt = np.asarray([[float(r["ndt_x"]), float(r["ndt_y"]), float(r["ndt_yaw"])] for r in rows], dtype=np.float64)
    refined = np.asarray([r["refined"] == "True" for r in rows], dtype=bool)
    conf = np.asarray([float(r["confidence"]) for r in rows], dtype=np.float64)
    key_idx = idx[refined]
    if len(key_idx) < 2:
        raise SystemExit("not enough refined keyframes")

    dx = ndt[refined, 0] - init[refined, 0]
    dy = ndt[refined, 1] - init[refined, 1]
    dyaw = np.unwrap(angle_diff(ndt[refined, 2], init[refined, 2]))
    full_dx = np.interp(idx, key_idx, dx)
    full_dy = np.interp(idx, key_idx, dy)
    full_dyaw = np.interp(idx, key_idx, dyaw)

    corrected = init.copy()
    corrected[:, 0] += full_dx
    corrected[:, 1] += full_dy
    corrected[:, 2] = [normalize_angle(float(v)) for v in init[:, 2] + full_dyaw]
    corrected[refined] = ndt[refined]

    out_csv = args.output_dir / "global_lidar_map_poses_corrected.csv"
    fields = list(rows[0].keys()) + ["corr_x", "corr_y", "corr_yaw", "corr_dx", "corr_dy", "corr_dyaw"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out.update(
                {
                    "corr_x": corrected[i, 0],
                    "corr_y": corrected[i, 1],
                    "corr_yaw": corrected[i, 2],
                    "corr_dx": full_dx[i],
                    "corr_dy": full_dy[i],
                    "corr_dyaw": full_dyaw[i],
                }
            )
            writer.writerow(out)

    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    render(args.output_dir / "trajectory_on_map_corrected.png", gltf_map, init, corrected, ndt[refined, :2])

    path_len = float(np.linalg.norm(np.diff(corrected[:, :2], axis=0), axis=1).sum())
    inside = (
        (corrected[:, 0] >= gltf_map.bounds_min[0] - 0.5)
        & (corrected[:, 0] <= gltf_map.bounds_max[0] + 0.5)
        & (corrected[:, 1] >= gltf_map.bounds_min[1] - 0.5)
        & (corrected[:, 1] <= gltf_map.bounds_max[1] + 0.5)
    )
    summary = {
        "frames": len(rows),
        "keyframes": int(refined.sum()),
        "keyframe_confidence_mean": float(conf[refined].mean()),
        "keyframe_confidence_median": float(np.median(conf[refined])),
        "corrected_bounds_min": corrected[:, :2].min(axis=0).tolist(),
        "corrected_bounds_max": corrected[:, :2].max(axis=0).tolist(),
        "inside_map_plus_margin_ratio": float(inside.mean()),
        "corrected_path_length_map_units": path_len,
        "note": "Full LiDAR-only trajectory matched to GLTF map using NDT keyframes and interpolated corrections; no dataset pose file used.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[csv] {out_csv}")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[trajectory] {args.output_dir / 'trajectory_on_map_corrected.png'}")


if __name__ == "__main__":
    main()
