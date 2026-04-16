"""nuScenes data loader + dummy data generator for testing."""

import os
import numpy as np
import torch
from typing import Dict, Optional

from config import BEVConfig


def generate_dummy_data(cfg: BEVConfig, batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """Generate dummy data that mimics nuScenes format for pipeline testing.

    Returns dict with:
        - images: (B, N_cams, 3, H, W)
        - intrinsics: (B, N_cams, 3, 3)
        - extrinsics: (B, N_cams, 4, 4)  cam2ego
        - lidar_points: (B, max_points, 5)  x, y, z, intensity, ring
        - lidar_mask: (B, max_points) bool
    """
    N = cfg.num_cameras
    H, W = cfg.image_size

    images = torch.randn(batch_size, N, 3, H, W)

    # Simple intrinsics (focal length ~700, center at image center)
    intrinsics = torch.zeros(batch_size, N, 3, 3)
    for b in range(batch_size):
        for n in range(N):
            intrinsics[b, n] = torch.tensor([
                [700.0, 0, W / 2],
                [0, 700.0, H / 2],
                [0, 0, 1],
            ])

    # Extrinsics: place 6 cameras around the ego vehicle
    angles = [0, 60, -60, 180, -120, 120]  # front, front-right, front-left, back, back-left, back-right
    extrinsics = torch.zeros(batch_size, N, 4, 4)
    for b in range(batch_size):
        for n, angle in enumerate(angles):
            rad = np.radians(angle)
            # rotation around z-axis + translation 1.5m up, 1m forward
            extrinsics[b, n] = torch.tensor([
                [np.cos(rad), -np.sin(rad), 0, 1.5 * np.cos(rad)],
                [np.sin(rad),  np.cos(rad), 0, 1.5 * np.sin(rad)],
                [0, 0, 1, 1.8],
                [0, 0, 0, 1],
            ], dtype=torch.float32)

    # LiDAR point cloud: random points in the BEV range
    max_pts = 30000
    lidar_points = torch.zeros(batch_size, max_pts, 5)
    lidar_mask = torch.zeros(batch_size, max_pts, dtype=torch.bool)
    for b in range(batch_size):
        n_pts = np.random.randint(15000, max_pts)
        x = np.random.uniform(cfg.point_cloud_range[0], cfg.point_cloud_range[3], n_pts)
        y = np.random.uniform(cfg.point_cloud_range[1], cfg.point_cloud_range[4], n_pts)
        z = np.random.uniform(cfg.point_cloud_range[2], cfg.point_cloud_range[5], n_pts)
        intensity = np.random.uniform(0, 1, n_pts)
        ring = np.random.randint(0, 32, n_pts).astype(np.float32)
        lidar_points[b, :n_pts] = torch.tensor(
            np.stack([x, y, z, intensity, ring], axis=-1), dtype=torch.float32
        )
        lidar_mask[b, :n_pts] = True

    return {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "lidar_points": lidar_points,
        "lidar_mask": lidar_mask,
    }


class NuScenesLoader:
    """Load data from nuScenes dataset."""

    def __init__(self, dataroot: str, version: str = "v1.0-mini", cfg: Optional[BEVConfig] = None):
        from nuscenes.nuscenes import NuScenes
        import cv2

        self.cfg = cfg or BEVConfig()
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.cv2 = cv2
        self.samples = self.nusc.sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        cfg = self.cfg
        H, W = cfg.image_size

        images_list = []
        intrinsics_list = []
        extrinsics_list = []

        for cam_name in cfg.camera_names:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])

            # Load image
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            img = self.cv2.imread(img_path)
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            img = self.cv2.resize(img, (W, H))
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
            images_list.append(img)

            # Calibration
            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            intrinsic = np.array(calib["camera_intrinsic"])  # (3, 3)
            # Scale intrinsics for resized image
            intrinsic[0] *= W / orig_w
            intrinsic[1] *= H / orig_h
            intrinsics_list.append(torch.tensor(intrinsic, dtype=torch.float32))

            # cam2ego
            from pyquaternion import Quaternion
            rot = Quaternion(calib["rotation"]).rotation_matrix
            trans = np.array(calib["translation"])
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rot
            extrinsic[:3, 3] = trans
            extrinsics_list.append(torch.tensor(extrinsic, dtype=torch.float32))

        images = torch.stack(images_list)             # (N, 3, H, W)
        intrinsics = torch.stack(intrinsics_list)     # (N, 3, 3)
        extrinsics = torch.stack(extrinsics_list)     # (N, 4, 4)

        # Load LiDAR
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data["filename"])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)  # x,y,z,intensity,ring

        # LiDAR sensor to ego
        lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        from pyquaternion import Quaternion as Q
        lidar_rot = Q(lidar_calib["rotation"]).rotation_matrix
        lidar_trans = np.array(lidar_calib["translation"])
        # Transform to ego
        points_xyz = points[:, :3] @ lidar_rot.T + lidar_trans
        points = np.concatenate([points_xyz, points[:, 3:]], axis=-1)

        # Pad / truncate
        max_pts = 30000
        n_pts = min(points.shape[0], max_pts)
        lidar_padded = np.zeros((max_pts, 5), dtype=np.float32)
        lidar_padded[:n_pts] = points[:n_pts]
        lidar_mask = np.zeros(max_pts, dtype=bool)
        lidar_mask[:n_pts] = True

        return {
            "images": images.unsqueeze(0),                                    # (1, N, 3, H, W)
            "intrinsics": intrinsics.unsqueeze(0),                            # (1, N, 3, 3)
            "extrinsics": extrinsics.unsqueeze(0),                            # (1, N, 4, 4)
            "lidar_points": torch.tensor(lidar_padded).unsqueeze(0),          # (1, max_pts, 5)
            "lidar_mask": torch.tensor(lidar_mask).unsqueeze(0),              # (1, max_pts)
        }
