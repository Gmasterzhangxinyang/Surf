"""Generate BEV ground-truth segmentation maps from nuScenes 3D annotations."""

import numpy as np
import torch
from typing import Dict, List, Tuple
from pyquaternion import Quaternion


# nuScenes category -> our class id mapping
CATEGORY_MAP = {
    "vehicle.car": 1,
    "vehicle.truck": 1,
    "vehicle.bus.bendy": 1,
    "vehicle.bus.rigid": 1,
    "vehicle.construction": 1,
    "vehicle.trailer": 1,
    "human.pedestrian.adult": 2,
    "human.pedestrian.child": 2,
    "human.pedestrian.construction_worker": 2,
    "human.pedestrian.police_officer": 2,
    "vehicle.bicycle": 5,
    "vehicle.motorcycle": 5,
}


def box_to_bev_mask(center, size, rotation, bev_x_range, bev_y_range, bev_size):
    """Rasterize a single 3D bounding box onto the BEV grid.

    Args:
        center: (3,) box center in ego frame
        size: (3,) width, length, height
        rotation: Quaternion rotation
        bev_x_range: (min_x, max_x)
        bev_y_range: (min_y, max_y)
        bev_size: (H, W)

    Returns:
        mask: (H, W) boolean mask of occupied cells
    """
    bev_h, bev_w = bev_size
    res_x = (bev_x_range[1] - bev_x_range[0]) / bev_w
    res_y = (bev_y_range[1] - bev_y_range[0]) / bev_h

    w, l, h = size[0], size[1], size[2]

    # 4 bottom corners in box-local frame
    corners = np.array([
        [-w / 2, -l / 2, 0],
        [ w / 2, -l / 2, 0],
        [ w / 2,  l / 2, 0],
        [-w / 2,  l / 2, 0],
    ])

    # Rotate and translate to ego frame
    rot_mat = rotation.rotation_matrix
    corners_ego = (rot_mat @ corners.T).T + center

    # Project to BEV grid indices
    cx = ((corners_ego[:, 0] - bev_x_range[0]) / res_x)
    cy = ((corners_ego[:, 1] - bev_y_range[0]) / res_y)

    # Rasterize the polygon using scanline
    mask = np.zeros((bev_h, bev_w), dtype=bool)
    pts = np.stack([cx, cy], axis=-1)  # (4, 2)

    # Use cv2 fillPoly for simplicity
    import cv2
    pts_int = pts[:, ::-1].astype(np.int32)  # (y, x) for cv2
    pts_int = pts.astype(np.int32).reshape(1, -1, 2)
    # cv2 uses (x, y) ordering
    cv2.fillPoly(mask.astype(np.uint8), pts_int, 1)

    # Manual approach: use the polygon vertices
    mask = np.zeros((bev_h, bev_w), dtype=np.uint8)
    poly = np.array([[int(cx[i]), int(cy[i])] for i in range(4)], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)


def generate_bev_gt(nusc, sample, cfg):
    """Generate BEV ground-truth segmentation map for a sample.

    Args:
        nusc: NuScenes instance
        sample: sample dict from nusc.sample
        cfg: BEVConfig

    Returns:
        bev_gt: (H, W) int tensor with class IDs
    """
    import cv2

    bev_h, bev_w = cfg.bev_size
    bev_gt = np.zeros((bev_h, bev_w), dtype=np.int32)

    # Get ego pose for this sample's lidar
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_rot = Quaternion(ego_pose["rotation"])
    ego_trans = np.array(ego_pose["translation"])

    res_x = (cfg.bev_x_range[1] - cfg.bev_x_range[0]) / bev_w
    res_y = (cfg.bev_y_range[1] - cfg.bev_y_range[0]) / bev_h

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        category = ann["category_name"]

        # Map to our class
        cls_id = 0
        for cat_prefix, cid in CATEGORY_MAP.items():
            if category.startswith(cat_prefix):
                cls_id = cid
                break
        if cls_id == 0:
            continue

        # Box center/size/rotation in global frame
        center_global = np.array(ann["translation"])
        size = np.array(ann["size"])  # width, length, height
        rotation_global = Quaternion(ann["rotation"])

        # Transform to ego frame
        center_ego = ego_rot.inverse.rotation_matrix @ (center_global - ego_trans)
        rotation_ego = ego_rot.inverse * rotation_global

        # Check if box is within BEV range
        if (center_ego[0] < cfg.bev_x_range[0] - 5 or center_ego[0] > cfg.bev_x_range[1] + 5 or
            center_ego[1] < cfg.bev_y_range[0] - 5 or center_ego[1] > cfg.bev_y_range[1] + 5):
            continue

        # Get box corners in ego frame and rasterize
        w, l, h = size
        corners_local = np.array([
            [-w / 2, -l / 2],
            [ w / 2, -l / 2],
            [ w / 2,  l / 2],
            [-w / 2,  l / 2],
        ])
        rot_2d = rotation_ego.rotation_matrix[:2, :2]
        corners_ego_2d = (rot_2d @ corners_local.T).T + center_ego[:2]

        # To BEV grid coordinates
        bev_ix = ((corners_ego_2d[:, 0] - cfg.bev_x_range[0]) / res_x).astype(np.int32)
        bev_iy = ((corners_ego_2d[:, 1] - cfg.bev_y_range[0]) / res_y).astype(np.int32)

        poly = np.stack([bev_ix, bev_iy], axis=-1).reshape(1, -1, 2)
        cv2.fillPoly(bev_gt, poly, int(cls_id))

    return torch.tensor(bev_gt, dtype=torch.long)
