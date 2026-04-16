"""Geometry utilities for coordinate transforms."""

import numpy as np
import torch


def create_frustum(depth_bins, depth_min, depth_max, image_h, image_w, downsample=16):
    """Create a frustum grid of (depth, height, width) points in image space.

    Returns:
        frustum: (D, fH, fW, 3) tensor of (x_img, y_img, depth) for each point
    """
    fH, fW = image_h // downsample, image_w // downsample

    depth_values = torch.linspace(depth_min, depth_max, depth_bins)      # (D,)
    y_values = torch.linspace(0, image_h - 1, fH)                        # (fH,)
    x_values = torch.linspace(0, image_w - 1, fW)                        # (fW,)

    # meshgrid: depth, y, x
    depth_grid, y_grid, x_grid = torch.meshgrid(depth_values, y_values, x_values, indexing="ij")
    # stack -> (D, fH, fW, 3)  last dim = (x, y, depth)
    frustum = torch.stack([x_grid, y_grid, depth_grid], dim=-1)
    return frustum


def frustum_to_world(frustum, intrinsics, extrinsics):
    """Project frustum points from image coordinates to world/ego coordinates.

    Args:
        frustum: (D, fH, fW, 3) tensor with (u, v, d) in image space
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) camera-to-ego transform (cam2ego)

    Returns:
        points_ego: (D, fH, fW, 3) points in ego frame
    """
    D, fH, fW, _ = frustum.shape
    pts = frustum.reshape(-1, 3)  # (N, 3)  columns: u, v, d

    u, v, d = pts[:, 0], pts[:, 1], pts[:, 2]

    # un-project: pixel -> camera coords
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (u - cx) / fx * d
    y_cam = (v - cy) / fy * d
    z_cam = d

    pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)  # (N, 4)

    # camera -> ego
    pts_ego = (extrinsics @ pts_cam.T).T[:, :3]  # (N, 3)
    return pts_ego.reshape(D, fH, fW, 3)


def points_to_bev_indices(points, bev_x_range, bev_y_range, bev_size):
    """Convert 3D points to BEV grid indices.

    Args:
        points: (..., 3) x, y, z in ego coordinates
        bev_x_range: (min_x, max_x)
        bev_y_range: (min_y, max_y)
        bev_size: (H, W)

    Returns:
        bev_ix: integer x indices
        bev_iy: integer y indices
        valid_mask: boolean mask for points inside BEV range
    """
    x = points[..., 0]
    y = points[..., 1]

    bev_h, bev_w = bev_size
    res_x = (bev_x_range[1] - bev_x_range[0]) / bev_w
    res_y = (bev_y_range[1] - bev_y_range[0]) / bev_h

    bev_ix = ((x - bev_x_range[0]) / res_x).long()
    bev_iy = ((y - bev_y_range[0]) / res_y).long()

    valid = (bev_ix >= 0) & (bev_ix < bev_w) & (bev_iy >= 0) & (bev_iy < bev_h)
    return bev_ix, bev_iy, valid
