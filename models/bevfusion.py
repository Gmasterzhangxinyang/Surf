"""BEVFusion main model: assemble camera + LiDAR + fusion + head."""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEVConfig
from models.camera_encoder import CameraEncoder
from models.lidar_encoder import LiDAREncoder
from models.fusion import ConvFuser
from models.heads import BEVSegHead


class BEVFusion(nn.Module):
    """BEVFusion: multi-modal BEV perception model.

    Pipeline:
        Camera images → CameraEncoder → cam_bev  (B, C_cam, H, W)
        LiDAR points  → LiDAREncoder  → lid_bev  (B, C_lid, H, W)
        [cam_bev, lid_bev] → ConvFuser → fused    (B, C_fuse, H, W)
        fused → BEVSegHead → logits                (B, num_classes, H, W)
    """

    def __init__(self, cfg: BEVConfig):
        super().__init__()
        self.cfg = cfg
        self.camera_encoder = CameraEncoder(cfg)
        self.lidar_encoder = LiDAREncoder(cfg)
        self.fuser = ConvFuser(cfg.cam_channels, cfg.lidar_channels, cfg.fused_channels)
        self.seg_head = BEVSegHead(cfg.fused_channels, cfg.num_classes)

    def forward(self, images, intrinsics, extrinsics, lidar_points, lidar_mask):
        """
        Args:
            images:        (B, N_cams, 3, H, W)
            intrinsics:    (B, N_cams, 3, 3)
            extrinsics:    (B, N_cams, 4, 4)  cam2ego
            lidar_points:  (B, N_pts, 5)
            lidar_mask:    (B, N_pts) bool

        Returns:
            logits: (B, num_classes, bev_H, bev_W)
            bev_seg: (B, bev_H, bev_W)  argmax predictions
        """
        # Camera branch
        cam_bev = self.camera_encoder(images, intrinsics, extrinsics)

        # LiDAR branch
        lidar_bev = self.lidar_encoder(lidar_points, lidar_mask)

        # Fusion
        fused = self.fuser(cam_bev, lidar_bev)

        # Segmentation head
        logits = self.seg_head(fused)
        bev_seg = logits.argmax(dim=1)

        return logits, bev_seg
