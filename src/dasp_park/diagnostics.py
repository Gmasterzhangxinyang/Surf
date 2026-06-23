import numpy as np

from .datatypes import FREE, OCCLUDED_UNKNOWN, OCCUPIED, UNKNOWN, SlotHypothesis
from .occupancy import grid_shape, valid_grid_mask, world_to_grid


def compute_unknown_map(occupancy: np.ndarray) -> np.ndarray:
    """Return unknown mask as float map."""
    return (occupancy == UNKNOWN).astype(np.float32)


def dilate_binary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Simple binary dilation using numpy slicing."""
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di * di + dj * dj > radius * radius:
                continue
            src_i0 = max(0, -di)
            src_i1 = min(h, h - di)
            src_j0 = max(0, -dj)
            src_j1 = min(w, w - dj)
            dst_i0 = max(0, di)
            dst_i1 = min(h, h + di)
            dst_j0 = max(0, dj)
            dst_j1 = min(w, w + dj)
            out[dst_i0:dst_i1, dst_j0:dst_j1] |= mask[src_i0:src_i1, src_j0:src_j1]
    return out


def compute_obstacle_boundary_map(occupancy: np.ndarray, radius: int = 2) -> np.ndarray:
    """Compute obstacle boundary map around occupied cells."""
    occupied = occupancy == OCCUPIED
    return (dilate_binary(occupied, radius=radius) & ~occupied).astype(np.float32)


def compute_occlusion_map(
    occupancy: np.ndarray,
    grid_cfg: dict,
    num_rays: int = 720,
) -> np.ndarray:
    """
    Approximate occlusion by ray casting from ego origin.

    A ray is free until the first occupied cell. Cells behind the first occupied
    cell along the ray become occluded.
    """
    occlusion = np.zeros_like(occupancy, dtype=np.float32)
    angles = np.linspace(-np.pi / 2, np.pi / 2, num_rays)
    ranges = np.arange(0.0, grid_cfg["x_max"], grid_cfg["resolution"])
    shape = occupancy.shape
    for angle in angles:
        x = ranges * np.cos(angle)
        y = ranges * np.sin(angle)
        i, j = world_to_grid(x, y, grid_cfg)
        valid = valid_grid_mask(i, j, shape)
        i, j = i[valid], j[valid]
        seen_obstacle = False
        last = None
        for row, col in zip(i, j):
            cell = (int(row), int(col))
            if cell == last:
                continue
            last = cell
            if not seen_obstacle and occupancy[cell] == OCCUPIED:
                seen_obstacle = True
                continue
            if seen_obstacle and occupancy[cell] != OCCUPIED:
                occlusion[cell] = 1.0
    return occlusion


def compute_uncertainty_map(
    occupancy: np.ndarray,
    occlusion: np.ndarray,
    boundary: np.ndarray,
) -> np.ndarray:
    """Compute heuristic uncertainty map."""
    uncertainty = np.zeros_like(occlusion, dtype=np.float32)
    uncertainty[occupancy == UNKNOWN] = 0.75
    uncertainty[occupancy == FREE] = 0.10
    uncertainty[occupancy == OCCUPIED] = 0.20
    uncertainty[occupancy == OCCLUDED_UNKNOWN] = 1.00
    uncertainty = np.maximum(uncertainty, occlusion * 1.00)
    uncertainty = np.maximum(uncertainty, boundary * 0.55)
    return np.clip(uncertainty, 0.0, 1.0)


def mark_world_box(
    arr: np.ndarray,
    grid_cfg: dict,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    value: float,
    mode: str = "max",
) -> None:
    """Mark a world-space box into a grid map."""
    h, w = grid_shape(grid_cfg)
    xs = grid_cfg["x_min"] + (np.arange(h) + 0.5) * grid_cfg["resolution"]
    ys = grid_cfg["y_min"] + (np.arange(w) + 0.5) * grid_cfg["resolution"]
    mask = (xs[:, None] >= x_min) & (xs[:, None] <= x_max) & (ys[None, :] >= y_min) & (ys[None, :] <= y_max)
    if mode == "max":
        arr[mask] = np.maximum(arr[mask], value)
    elif mode == "set":
        arr[mask] = value
    else:
        raise ValueError(f"Unsupported mark mode: {mode}")


def compute_decision_impact_map(
    occupancy: np.ndarray,
    fake_slots: list[SlotHypothesis],
    grid_cfg: dict,
) -> np.ndarray:
    """Compute heuristic decision impact map."""
    decision_impact = np.zeros_like(occupancy, dtype=np.float32)
    mark_world_box(decision_impact, grid_cfg, 0.0, 28.0, -3.0, 3.0, 0.35)
    for slot in fake_slots:
        x0, y0 = slot.polygon_xy.min(axis=0)
        x1, y1 = slot.polygon_xy.max(axis=0)
        mark_world_box(decision_impact, grid_cfg, float(x0), float(x1), float(y0), float(y1), 0.45)
        if slot.is_target_candidate:
            mark_world_box(decision_impact, grid_cfg, float(x0), float(x1), float(y0), float(y1), 1.0)
            entrance_y = float(slot.entrance_xy[:, 1].mean())
            mark_world_box(
                decision_impact,
                grid_cfg,
                float(slot.entrance_xy[:, 0].min()),
                float(slot.entrance_xy[:, 0].max()),
                entrance_y - 1.5,
                entrance_y + 1.5,
                1.0,
            )
    return decision_impact


def compute_priority_map(
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    decision_impact: np.ndarray,
    boundary: np.ndarray,
    weights: dict,
) -> np.ndarray:
    """Compute decision-aware priority map."""
    priority = (
        weights["uncertainty"] * uncertainty
        + weights["occlusion"] * occlusion
        + weights["decision_impact"] * decision_impact
        + weights["obstacle_boundary"] * boundary
    ).astype(np.float32)
    priority = priority - float(priority.min())
    priority = priority / (float(priority.max()) + 1e-6)
    return priority
