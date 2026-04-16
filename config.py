"""BEVFusion configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BEVConfig:
    # ---------- BEV grid ----------
    bev_x_range: Tuple[float, float] = (-51.2, 51.2)   # meters
    bev_y_range: Tuple[float, float] = (-51.2, 51.2)
    bev_resolution: float = 0.8                          # meters per pixel
    bev_size: Tuple[int, int] = (128, 128)               # H, W of BEV grid

    # ---------- Depth ----------
    depth_min: float = 1.0
    depth_max: float = 60.0
    depth_bins: int = 59                                  # number of discrete depth bins

    # ---------- Camera ----------
    image_size: Tuple[int, int] = (256, 704)              # H, W after resize
    cam_channels: int = 64                                # camera BEV feature dim
    num_cameras: int = 6
    camera_names: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ])

    # ---------- LiDAR ----------
    point_cloud_range: List[float] = field(default_factory=lambda: [
        -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
    ])
    pillar_size: List[float] = field(default_factory=lambda: [0.8, 0.8, 8.0])
    max_pillars: int = 20000
    max_points_per_pillar: int = 32
    lidar_in_channels: int = 5                            # x, y, z, intensity, ring
    lidar_channels: int = 64                              # lidar BEV feature dim

    # ---------- Fusion / Head ----------
    fused_channels: int = 128
    num_classes: int = 6   # background, vehicle, pedestrian, road, sidewalk, other

    # ---------- Device ----------
    device: str = "cpu"   # "cpu" or "mps"

    @property
    def pillar_grid_size(self) -> Tuple[int, int]:
        x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
        y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
        nx = int(x_range / self.pillar_size[0])
        ny = int(y_range / self.pillar_size[1])
        return (nx, ny)
