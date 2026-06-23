import numpy as np

from .datatypes import FREE, OCCUPIED, UNKNOWN


def grid_shape(grid_cfg: dict) -> tuple[int, int]:
    """Return BEV grid shape as (H, W)."""
    h = int(round((grid_cfg["x_max"] - grid_cfg["x_min"]) / grid_cfg["resolution"]))
    w = int(round((grid_cfg["y_max"] - grid_cfg["y_min"]) / grid_cfg["resolution"]))
    return h, w


def world_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    grid_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world x/y coordinates into grid row/col indices."""
    res = grid_cfg["resolution"]
    i = np.floor((x - grid_cfg["x_min"]) / res).astype(int)
    j = np.floor((y - grid_cfg["y_min"]) / res).astype(int)
    return i, j


def grid_to_world(
    i: np.ndarray,
    j: np.ndarray,
    grid_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert grid row/col indices to world x/y cell centers."""
    res = grid_cfg["resolution"]
    x = grid_cfg["x_min"] + (i + 0.5) * res
    y = grid_cfg["y_min"] + (j + 0.5) * res
    return x, y


def valid_grid_mask(i: np.ndarray, j: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return mask for valid grid indices."""
    h, w = shape
    return (i >= 0) & (i < h) & (j >= 0) & (j < w)


def build_occupancy_from_lidar(
    points: np.ndarray,
    grid_cfg: dict,
    obstacle_z_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build occupancy and height maps from LiDAR points.

    Occupancy values:
        0 = unknown
        1 = free
        2 = occupied

    Only observed cells are marked free/occupied. Unobserved cells remain unknown.
    """
    shape = grid_shape(grid_cfg)
    occupancy = np.full(shape, UNKNOWN, dtype=np.uint8)
    height = np.full(shape, np.nan, dtype=np.float32)

    if points.size == 0:
        return occupancy, height

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    i, j = world_to_grid(x, y, grid_cfg)
    mask = valid_grid_mask(i, j, shape)
    i, j, z = i[mask], j[mask], z[mask]

    for row, col, zz in zip(i, j, z):
        if np.isnan(height[row, col]) or zz > height[row, col]:
            height[row, col] = zz

    observed = ~np.isnan(height)
    occupancy[observed] = FREE
    occupancy[observed & (height > obstacle_z_threshold)] = OCCUPIED

    return occupancy, height
