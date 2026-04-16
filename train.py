#!/usr/bin/env python3
"""Train BEVFusion on nuScenes mini dataset.

Usage:
    python train.py --dataroot /path/to/nuscenes --epochs 10
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEVConfig
from models.bevfusion import BEVFusion
from data.nuscenes_loader import NuScenesLoader
from data.bev_gt import generate_bev_gt
from utils.visualize import visualize_bev_result


def train(args):
    # ---- Config ----
    device_name = "mps" if torch.backends.mps.is_available() and args.device == "auto" else "cpu"
    if args.device in ("cpu", "mps"):
        device_name = args.device
    device = torch.device(device_name)

    cfg = BEVConfig(device=device_name)

    print("=" * 60)
    print("BEVFusion Training")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Epochs:     {args.epochs}")
    print(f"LR:         {args.lr}")
    print(f"Dataroot:   {args.dataroot}")
    print()

    # ---- Data ----
    print("Loading nuScenes...")
    loader = NuScenesLoader(args.dataroot, args.version, cfg)
    nusc = loader.nusc
    n_samples = len(loader)
    print(f"Total samples: {n_samples}")

    # Pre-generate all BEV ground truths
    print("Generating BEV ground truth maps...")
    gt_maps = []
    valid_indices = []
    for i in range(n_samples):
        sample = nusc.sample[i]
        bev_gt = generate_bev_gt(nusc, sample, cfg)
        n_objects = (bev_gt > 0).sum().item()
        gt_maps.append(bev_gt)
        if n_objects > 0:
            valid_indices.append(i)
    print(f"Samples with objects: {len(valid_indices)} / {n_samples}")
    print()

    # ---- Model ----
    print("Building model...")
    model = BEVFusion(cfg).to(device)

    # Freeze ResNet backbone to speed up training
    for param in model.camera_encoder.layer1.parameters():
        param.requires_grad = False
    for param in model.camera_encoder.layer2.parameters():
        param.requires_grad = False
    for param in model.camera_encoder.layer3.parameters():
        param.requires_grad = False
    for param in model.camera_encoder.layer4.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable (backbone frozen)")

    # ---- Loss & Optimizer ----
    # Class weights: background is dominant, upweight rare classes
    class_weights = torch.tensor([0.1, 5.0, 8.0, 1.0, 1.0, 2.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Training Loop ----
    print()
    print("Starting training...")
    print("-" * 60)

    train_indices = valid_indices if args.objects_only else list(range(n_samples))
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t_epoch = time.time()

        # Shuffle
        indices = train_indices.copy()
        np.random.shuffle(indices)

        for step, idx in enumerate(indices):
            t0 = time.time()

            # Load data
            data = loader[idx]
            bev_gt = gt_maps[idx].unsqueeze(0).to(device)  # (1, H, W)

            # Move to device
            images = data["images"].to(device)
            intrinsics = data["intrinsics"].to(device)
            extrinsics = data["extrinsics"].to(device)
            lidar_pts = data["lidar_points"].to(device)
            lidar_mask = data["lidar_mask"].to(device)

            # Forward
            logits, bev_seg = model(images, intrinsics, extrinsics, lidar_pts, lidar_mask)

            # Loss
            loss = criterion(logits, bev_gt)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            epoch_loss += loss.item()
            correct = (bev_seg == bev_gt).sum().item()
            total_px = bev_gt.numel()
            epoch_correct += correct
            epoch_total += total_px

            dt = time.time() - t0
            if (step + 1) % 10 == 0 or step == 0:
                acc = 100.0 * correct / total_px
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(indices)} | "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.1f}% | {dt:.1f}s/step")

        epoch_time = time.time() - t_epoch
        avg_loss = epoch_loss / len(indices)
        avg_acc = 100.0 * epoch_correct / epoch_total

        print(f"  >> Epoch {epoch+1} done: Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.1f}%, Time={epoch_time:.0f}s")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> Saved best checkpoint: {ckpt_path}")

        # Visualize one sample at end of each epoch
        model.eval()
        with torch.no_grad():
            vis_idx = valid_indices[0] if valid_indices else 0
            vis_data = loader[vis_idx]
            vis_gt = gt_maps[vis_idx]
            for k, v in vis_data.items():
                vis_data[k] = v.to(device)
            vis_logits, vis_seg = model(
                vis_data["images"], vis_data["intrinsics"],
                vis_data["extrinsics"], vis_data["lidar_points"], vis_data["lidar_mask"]
            )

            seg_np = vis_seg[0].cpu().numpy()
            gt_np = vis_gt.numpy()
            logits_np = vis_logits[0].cpu().numpy()

            # Save prediction vs GT comparison
            save_comparison(seg_np, gt_np, logits_np, epoch + 1, args.output_dir)

        print()

    # Final checkpoint
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model: {final_path}")
    print(f"Best loss: {best_loss:.4f}")


def save_comparison(pred, gt, logits, epoch, output_dir):
    """Save side-by-side prediction vs ground truth."""
    import matplotlib.pyplot as plt
    from utils.visualize import colorize_bev, CLASS_COLORS, CLASS_NAMES
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    axes[0].imshow(colorize_bev(gt), origin="lower")
    axes[0].set_title("Ground Truth")

    # Prediction
    axes[1].imshow(colorize_bev(pred), origin="lower")
    axes[1].set_title(f"Prediction (Epoch {epoch})")

    # Confidence
    conf = logits.max(axis=0)
    im = axes[2].imshow(conf, origin="lower", cmap="hot")
    axes[2].set_title("Confidence")
    plt.colorbar(im, ax=axes[2])

    patches = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
               for k in sorted(CLASS_COLORS.keys())]
    axes[0].legend(handles=patches, loc="upper right", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"epoch_{epoch:02d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved visualization: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--output_dir", type=str, default="train_output")
    parser.add_argument("--objects_only", action="store_true",
                        help="Only train on samples that contain objects")
    args = parser.parse_args()
    train(args)
