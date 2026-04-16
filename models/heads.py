"""BEV segmentation head: predict per-cell class labels on the BEV grid."""

import torch
import torch.nn as nn


class BEVSegHead(nn.Module):
    """Simple convolution head for BEV semantic segmentation."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 4, num_classes, 1),
        )

    def forward(self, bev_features):
        """
        Args:
            bev_features: (B, C, H, W) fused BEV features

        Returns:
            logits: (B, num_classes, H, W) per-cell class logits
        """
        return self.head(bev_features)
