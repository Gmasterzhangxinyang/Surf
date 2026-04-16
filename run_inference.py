#!/usr/bin/env python3
"""BEVFusion inference script — runs the full pipeline on CPU/MPS.

Usage:
    # With dummy data (no nuScenes needed):
    python run_inference.py

    # With nuScenes mini:
    python run_inference.py --dataroot /path/to/nuscenes --version v1.0-mini --sample_idx 0
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEVConfig
from models.bevfusion import BEVFusion
from data.nuscenes_loader import generate_dummy_data, NuScenesLoader
from utils.visualize import visualize_bev_result, visualize_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="BEVFusion Inference")
    parser.add_argument("--dataroot", type=str, default=None,
                        help="Path to nuScenes dataset root (skip for dummy data)")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps"])
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    # Setup
    cfg = BEVConfig(device=args.device)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BEVFusion — Simplified Pure-PyTorch Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"BEV grid: {cfg.bev_size} at {cfg.bev_resolution}m resolution")
    print(f"BEV range: X{cfg.bev_x_range} Y{cfg.bev_y_range}")
    print(f"Depth bins: {cfg.depth_bins} ({cfg.depth_min}m - {cfg.depth_max}m)")
    print(f"Num classes: {cfg.num_classes}")
    print()

    # ---- Load data ----
    if args.dataroot is not None:
        print(f"Loading nuScenes from: {args.dataroot}")
        loader = NuScenesLoader(args.dataroot, args.version, cfg)
        data = loader[args.sample_idx]
        print(f"Loaded sample {args.sample_idx} / {len(loader)}")
    else:
        print("Using DUMMY data (no nuScenes path provided)")
        data = generate_dummy_data(cfg, batch_size=1)

    # Move to device
    for k, v in data.items():
        data[k] = v.to(device)

    print(f"  Images shape:       {data['images'].shape}")
    print(f"  LiDAR points shape: {data['lidar_points'].shape}")
    print(f"  LiDAR valid points: {data['lidar_mask'].sum().item()}")
    print()

    # ---- Build model ----
    print("Building BEVFusion model...")
    model = BEVFusion(cfg).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print()

    # ---- Inference ----
    print("Running inference...")
    t0 = time.time()

    with torch.no_grad():
        logits, bev_seg = model(
            data["images"],
            data["intrinsics"],
            data["extrinsics"],
            data["lidar_points"],
            data["lidar_mask"],
        )

    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.2f}s")
    print(f"  Output logits:  {logits.shape}")
    print(f"  Output BEV seg: {bev_seg.shape}")
    print()

    # ---- Results ----
    bev_np = bev_seg[0].cpu().numpy()           # (H, W)
    logits_np = logits[0].cpu().numpy()         # (C, H, W)

    # Class distribution
    unique, counts = np.unique(bev_np, return_counts=True)
    print("BEV class distribution:")
    class_names = ["background", "vehicle", "pedestrian", "road", "sidewalk", "other"]
    for cls_id, cnt in zip(unique, counts):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        pct = 100 * cnt / bev_np.size
        print(f"  {name}: {cnt} pixels ({pct:.1f}%)")
    print()

    # ---- Visualize ----
    # 1) BEV segmentation only
    bev_path = os.path.join(args.output_dir, "bev_segmentation.png")
    visualize_bev_result(bev_np, logits_np, save_path=bev_path)

    # 2) Full pipeline overview
    # Move data back to CPU for visualization
    data_cpu = {k: v.cpu() for k, v in data.items()}
    full_path = os.path.join(args.output_dir, "full_pipeline.png")
    visualize_full_pipeline(data_cpu, bev_np, logits_np, save_path=full_path)

    print("=" * 60)
    print("Done! Output saved to:")
    print(f"  {bev_path}")
    print(f"  {full_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
