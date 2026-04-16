"""Visualization utilities for BEV segmentation output."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Class colors (RGB 0-1)
CLASS_COLORS = {
    0: (0.2, 0.2, 0.2),     # background — dark gray
    1: (1.0, 0.0, 0.0),     # vehicle — red
    2: (0.0, 0.0, 1.0),     # pedestrian — blue
    3: (0.5, 0.5, 0.5),     # road — gray
    4: (0.8, 0.8, 0.4),     # sidewalk — yellow-ish
    5: (0.0, 0.8, 0.0),     # other — green
}

CLASS_NAMES = {
    0: "background",
    1: "vehicle",
    2: "pedestrian",
    3: "road",
    4: "sidewalk",
    5: "other",
}


def colorize_bev(seg_map):
    """Convert integer segmentation map to RGB image.

    Args:
        seg_map: (H, W) numpy array of class indices

    Returns:
        rgb: (H, W, 3) numpy array
    """
    H, W = seg_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        mask = seg_map == cls_id
        rgb[mask] = color
    return rgb


def visualize_bev_result(bev_seg, logits=None, save_path=None, title="BEV Segmentation"):
    """Visualize BEV segmentation result.

    Args:
        bev_seg: (H, W) numpy array, integer class labels
        logits: optional (num_classes, H, W) for confidence overlay
        save_path: if given, save to file
        title: plot title
    """
    fig, axes = plt.subplots(1, 2 if logits is not None else 1, figsize=(14, 6))

    if logits is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes if not hasattr(axes, '__len__') else axes[0]

    # BEV segmentation map
    rgb = colorize_bev(bev_seg)
    ax1.imshow(rgb, origin="lower")
    ax1.set_title(title)
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in sorted(CLASS_COLORS.keys())]
    ax1.legend(handles=patches, loc="upper right", fontsize=8)

    # Confidence map (max logit)
    if logits is not None:
        confidence = logits.max(axis=0)
        im = ax2.imshow(confidence, origin="lower", cmap="hot")
        ax2.set_title("Confidence (max logit)")
        plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_full_pipeline(data_dict, bev_seg, logits, save_path=None):
    """Visualize the full pipeline: input cameras + LiDAR + BEV output.

    Args:
        data_dict: dict with 'images', 'lidar_points', 'lidar_mask'
        bev_seg: (H, W) numpy
        logits: (num_classes, H, W) numpy
        save_path: optional file path
    """
    images = data_dict["images"][0].numpy()     # (N, 3, H, W)
    lidar = data_dict["lidar_points"][0].numpy()  # (N_pts, 5)
    lidar_mask = data_dict["lidar_mask"][0].numpy()

    N_cams = images.shape[0]
    fig = plt.figure(figsize=(20, 10))

    # Top row: camera images
    for i in range(min(N_cams, 6)):
        ax = fig.add_subplot(2, 4, i + 1)
        img = images[i].transpose(1, 2, 0)  # (H, W, 3)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Camera {i}")
        ax.axis("off")

    # Bottom-left: LiDAR point cloud (top-down)
    ax_lidar = fig.add_subplot(2, 4, 7)
    pts = lidar[lidar_mask]
    ax_lidar.scatter(pts[:, 0], pts[:, 1], s=0.1, c=pts[:, 2], cmap="viridis", alpha=0.5)
    ax_lidar.set_title("LiDAR Top-Down")
    ax_lidar.set_xlim(-55, 55)
    ax_lidar.set_ylim(-55, 55)
    ax_lidar.set_aspect("equal")

    # Bottom-right: BEV segmentation
    ax_bev = fig.add_subplot(2, 4, 8)
    rgb = colorize_bev(bev_seg)
    ax_bev.imshow(rgb, origin="lower")
    ax_bev.set_title("BEV Output")
    patches = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in sorted(CLASS_COLORS.keys())]
    ax_bev.legend(handles=patches, loc="upper right", fontsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
