"""BEV fusion module: merge camera and LiDAR BEV features."""

import torch
import torch.nn as nn


class ConvFuser(nn.Module):
    """Fuse camera BEV and LiDAR BEV via concatenation + convolution."""

    def __init__(self, cam_channels, lidar_channels, out_channels):
        super().__init__()
        in_ch = cam_channels + lidar_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, cam_bev, lidar_bev):
        """
        Args:
            cam_bev: (B, C_cam, H, W)
            lidar_bev: (B, C_lidar, H, W)

        Returns:
            fused: (B, C_out, H, W)
        """
        x = torch.cat([cam_bev, lidar_bev], dim=1)
        return self.fuse(x)
