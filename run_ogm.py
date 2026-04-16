#!/usr/bin/env python3
"""Generate OGM (Occupancy Grid Map) from nuScenes data.

This script demonstrates OGM generation using:
1. Direct LiDAR projection (no ML, instant)
2. BEVFusion learned features (optional, needs trained model)

Usage:
    # Direct projection OGM (recommended for validation):
    python run_ogm.py --dataroot /path/to/nuscenes

    # With BEVFusion comparison:
    python run_ogm.py --dataroot /path/to/nuscenes --checkpoint train_output/best_model.pth

    # Multiple samples:
    python run_ogm.py --dataroot /path/to/nuscenes --num_samples 10
"""

import argparse
import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEVConfig
from data.nuscenes_loader import NuScenesLoader
from utils.ogm import (
    lidar_to_ogm,
    lidar_to_ogm_probabilistic,
    lidar_to_height_map,
    visualize_ogm,
    visualize_ogm_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="OGM Generation")
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="BEVFusion checkpoint for learned OGM comparison")
    parser.add_argument("--output_dir", type=str, default="ogm_output")
    parser.add_argument("--resolution", type=float, default=0.4,
                        help="BEV resolution in meters (smaller = finer grid)")
    parser.add_argument("--range", type=float, default=30.0,
                        help="BEV range in meters (for parking lot, 30m is enough)")
    args = parser.parse_args()

    # ---- Config for parking lot scenario ----
    r = args.range
    res = args.resolution
    grid_size = int(2 * r / res)
    cfg = BEVConfig(
        bev_x_range=(-r, r),
        bev_y_range=(-r, r),
        bev_resolution=res,
        bev_size=(grid_size, grid_size),
    )

    print("=" * 60)
    print("OGM (Occupancy Grid Map) Generation")
    print("=" * 60)
    print(f"BEV range:      {cfg.bev_x_range} x {cfg.bev_y_range} meters")
    print(f"Resolution:     {res}m per cell")
    print(f"Grid size:      {grid_size} x {grid_size}")
    print(f"Num samples:    {args.num_samples}")
    print()

    # ---- Load data ----
    print("Loading nuScenes...")
    loader = NuScenesLoader(args.dataroot, args.version, cfg)
    n_total = len(loader)
    print(f"Total samples available: {n_total}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Optional: load BEVFusion model ----
    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading BEVFusion model from {args.checkpoint}...")
        from models.bevfusion import BEVFusion
        model = BEVFusion(cfg)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model.eval()
        print("Model loaded.")
        print()

    # ---- Process samples ----
    for i in range(args.num_samples):
        idx = args.start_idx + i
        if idx >= n_total:
            break

        print(f"--- Sample {idx} ---")
        t0 = time.time()

        data = loader[idx]
        lidar_pts = data["lidar_points"]
        lidar_mask = data["lidar_mask"]

        n_valid = lidar_mask.sum().item()
        print(f"  LiDAR points: {n_valid}")

        # 1) Direct projection OGM
        ogm_soft, ogm_hard = lidar_to_ogm_probabilistic(
            lidar_pts, lidar_mask, cfg,
            height_range=(-1.5, 2.5),
            max_points=10,
        )

        # 2) Height map
        height_map = lidar_to_height_map(lidar_pts, lidar_mask, cfg)

        # 3) Point count
        _, point_count = lidar_to_ogm(lidar_pts, lidar_mask, cfg)

        ogm_np = ogm_soft[0].numpy()
        ogm_hard_np = ogm_hard[0].numpy()
        pc_np = point_count[0].numpy()
        hm_np = height_map[0].numpy()

        occupied = (ogm_hard_np > 0).sum()
        total_cells = ogm_hard_np.size
        occ_pct = 100 * occupied / total_cells
        print(f"  Occupied cells: {occupied}/{total_cells} ({occ_pct:.1f}%)")

        # Save OGM visualization
        save_path = os.path.join(args.output_dir, f"ogm_sample_{idx:03d}.png")
        visualize_ogm(ogm_np, point_count=pc_np, height_map=hm_np,
                       save_path=save_path, title=f"OGM Sample {idx}")

        # Save binary OGM
        save_path_binary = os.path.join(args.output_dir, f"ogm_binary_{idx:03d}.png")
        visualize_ogm(ogm_hard_np, save_path=save_path_binary,
                       title=f"Binary OGM Sample {idx}")

        # 4) Optional: BEVFusion learned OGM
        if model is not None:
            with torch.no_grad():
                logits, bev_seg = model(
                    data["images"], data["intrinsics"],
                    data["extrinsics"], lidar_pts, lidar_mask,
                )
                # Convert segmentation to occupancy: anything non-background = occupied
                learned_ogm = (bev_seg[0] > 0).float().numpy()

            comp_path = os.path.join(args.output_dir, f"ogm_comparison_{idx:03d}.png")
            visualize_ogm_comparison(ogm_hard_np, learned_ogm, save_path=comp_path)

        # Save full overview with camera images
        save_overview(data, ogm_np, ogm_hard_np, hm_np, idx, args.output_dir)

        dt = time.time() - t0
        print(f"  Time: {dt:.2f}s")
        print()

    # Save raw OGM data as numpy for downstream use
    print("Saving raw OGM data...")
    np.save(os.path.join(args.output_dir, "last_ogm_soft.npy"), ogm_np)
    np.save(os.path.join(args.output_dir, "last_ogm_binary.npy"), ogm_hard_np)
    np.save(os.path.join(args.output_dir, "last_height_map.npy"), hm_np)

    print("=" * 60)
    print(f"Done! All outputs saved to: {args.output_dir}/")
    print("=" * 60)


def save_overview(data, ogm_soft, ogm_binary, height_map, sample_idx, output_dir):
    """Save a full overview: cameras + LiDAR + OGM."""
    images = data["images"][0].numpy()
    lidar = data["lidar_points"][0].numpy()
    lidar_mask = data["lidar_mask"][0].numpy()

    fig = plt.figure(figsize=(22, 12))

    # Top row: 6 camera images
    for i in range(min(6, images.shape[0])):
        ax = fig.add_subplot(3, 4, i + 1)
        img = images[i].transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Cam {i}", fontsize=9)
        ax.axis("off")

    # Bottom-left: LiDAR top-down
    ax = fig.add_subplot(3, 4, 7)
    pts = lidar[lidar_mask]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c=pts[:, 2], cmap="viridis", alpha=0.5)
    ax.set_title("LiDAR Top-Down")
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_aspect("equal")

    # OGM soft
    ax = fig.add_subplot(3, 4, 8)
    ax.imshow(ogm_soft, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("OGM (Probabilistic)")

    # OGM binary
    ax = fig.add_subplot(3, 4, 11)
    ax.imshow(ogm_binary, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("OGM (Binary)")

    # Height map
    ax = fig.add_subplot(3, 4, 12)
    im = ax.imshow(height_map, origin="lower", cmap="viridis")
    ax.set_title("Height Map")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Sample {sample_idx} — OGM Overview", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f"overview_{sample_idx:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
