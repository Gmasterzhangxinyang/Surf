"""OGM (Occupancy Grid Map) generation from sensor data.

Two modes:
1. Direct projection: LiDAR points → BEV grid counting → occupancy
2. Learned: BEVFusion network output → occupancy probability map
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def lidar_to_ogm(points, mask, cfg, height_range=(-1.5, 2.5), min_points=1):
    """Generate OGM by direct LiDAR point projection (no ML needed).

    Args:
        points: (B, N, 5) LiDAR points (x, y, z, intensity, ring)
        mask: (B, N) bool — valid points
        cfg: BEVConfig
        height_range: (min_z, max_z) filter points by height
        min_points: minimum points in a cell to mark as occupied

    Returns:
        ogm: (B, H, W) float tensor — occupancy probability [0, 1]
        point_count: (B, H, W) int tensor — point count per cell
    """
    B = points.shape[0]
    bev_h, bev_w = cfg.bev_size
    res_x = (cfg.bev_x_range[1] - cfg.bev_x_range[0]) / bev_w
    res_y = (cfg.bev_y_range[1] - cfg.bev_y_range[0]) / bev_h

    ogm = torch.zeros(B, bev_h, bev_w)
    point_count = torch.zeros(B, bev_h, bev_w, dtype=torch.long)

    for b in range(B):
        pts = points[b][mask[b]]  # (n_valid, 5)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Height filter
        z_mask = (z >= height_range[0]) & (z <= height_range[1])
        x, y, z = x[z_mask], y[z_mask], z[z_mask]

        # To grid indices
        ix = ((x - cfg.bev_x_range[0]) / res_x).long()
        iy = ((y - cfg.bev_y_range[0]) / res_y).long()

        valid = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)
        ix, iy = ix[valid], iy[valid]

        # Count points per cell
        for i in range(len(ix)):
            point_count[b, iy[i], ix[i]] += 1

        # Occupancy: normalize by min_points threshold
        ogm[b] = (point_count[b] >= min_points).float()

    return ogm, point_count


def lidar_to_ogm_probabilistic(points, mask, cfg, height_range=(-1.5, 2.5), max_points=10):
    """Generate soft OGM with occupancy probability based on point density.

    Returns:
        ogm: (B, H, W) float [0, 1] — probability of occupancy
    """
    ogm_hard, point_count = lidar_to_ogm(points, mask, cfg, height_range, min_points=1)

    # Soft probability: saturate at max_points
    ogm_soft = (point_count.float() / max_points).clamp(0, 1)
    return ogm_soft, ogm_hard


def lidar_to_height_map(points, mask, cfg, height_range=(-3.0, 3.0)):
    """Generate a height map from LiDAR — max height per cell.

    Returns:
        height_map: (B, H, W) float — max z value per cell (NaN for empty)
    """
    B = points.shape[0]
    bev_h, bev_w = cfg.bev_size
    res_x = (cfg.bev_x_range[1] - cfg.bev_x_range[0]) / bev_w
    res_y = (cfg.bev_y_range[1] - cfg.bev_y_range[0]) / bev_h

    height_map = torch.full((B, bev_h, bev_w), float("nan"))

    for b in range(B):
        pts = points[b][mask[b]]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        z_mask = (z >= height_range[0]) & (z <= height_range[1])
        x, y, z = x[z_mask], y[z_mask], z[z_mask]

        ix = ((x - cfg.bev_x_range[0]) / res_x).long()
        iy = ((y - cfg.bev_y_range[0]) / res_y).long()

        valid = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)
        ix, iy, z = ix[valid], iy[valid], z[valid]

        for i in range(len(ix)):
            cur = height_map[b, iy[i], ix[i]]
            if torch.isnan(cur) or z[i] > cur:
                height_map[b, iy[i], ix[i]] = z[i]

    return height_map


def visualize_ogm(ogm, point_count=None, height_map=None, save_path=None, title="OGM"):
    """Visualize OGM results.

    Args:
        ogm: (H, W) numpy — binary or probabilistic occupancy
        point_count: optional (H, W) — point counts
        height_map: optional (H, W) — height values
        save_path: file to save
    """
    n_plots = 1 + (point_count is not None) + (height_map is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # OGM
    cmap_ogm = mcolors.ListedColormap(["white", "black"])
    if ogm.max() > 1 or (ogm > 0).sum() > 0 and ogm.max() <= 1:
        # Probabilistic
        axes[0].imshow(ogm, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    else:
        axes[0].imshow(ogm, origin="lower", cmap=cmap_ogm, vmin=0, vmax=1)
    axes[0].set_title(title)
    axes[0].set_xlabel("X (cells)")
    axes[0].set_ylabel("Y (cells)")

    plot_idx = 1

    # Point count
    if point_count is not None:
        im = axes[plot_idx].imshow(point_count, origin="lower", cmap="hot")
        axes[plot_idx].set_title("Point Count per Cell")
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    # Height map
    if height_map is not None:
        hm = height_map.copy()
        im = axes[plot_idx].imshow(hm, origin="lower", cmap="viridis")
        axes[plot_idx].set_title("Height Map (max z)")
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def visualize_ogm_comparison(ogm_direct, ogm_learned, save_path=None):
    """Side-by-side comparison of direct projection vs learned OGM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(ogm_direct, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("Direct LiDAR Projection OGM")

    axes[1].imshow(ogm_learned, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("BEVFusion Learned OGM")

    for ax in axes:
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
