from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parking_slot_box_scoring.geometry import ensure_ccw


@dataclass(frozen=True)
class SegmentClip:
    entry_xyz: np.ndarray
    exit_xyz: np.ndarray
    t_entry: float
    t_exit: float


def segment_aabb_mask(
    origin_xyz: np.ndarray,
    endpoints_xyz: np.ndarray,
    minimum_xyz: np.ndarray,
    maximum_xyz: np.ndarray,
) -> np.ndarray:
    origin = _finite_vector(origin_xyz, "origin_xyz")
    endpoints = np.asarray(endpoints_xyz, dtype=np.float64)
    minimum = _finite_vector(minimum_xyz, "minimum_xyz")
    maximum = _finite_vector(maximum_xyz, "maximum_xyz")
    if endpoints.ndim != 2 or endpoints.shape[1] != 3:
        raise ValueError("endpoints_xyz must have shape [N, 3]")
    if not np.isfinite(endpoints).all():
        raise ValueError("endpoints_xyz must contain finite values")
    if np.any(minimum >= maximum):
        raise ValueError("minimum_xyz must be lower than maximum_xyz")
    count = len(endpoints)
    t_entry = np.zeros(count, dtype=np.float64)
    t_exit = np.ones(count, dtype=np.float64)
    valid = np.ones(count, dtype=bool)
    direction = endpoints - origin
    epsilon = 1e-12
    for axis in range(3):
        axis_direction = direction[:, axis]
        parallel = np.abs(axis_direction) <= epsilon
        valid &= ~parallel | (
            (origin[axis] >= minimum[axis] - epsilon)
            & (origin[axis] <= maximum[axis] + epsilon)
        )
        moving = ~parallel
        if not np.any(moving):
            continue
        t0 = (minimum[axis] - origin[axis]) / axis_direction[moving]
        t1 = (maximum[axis] - origin[axis]) / axis_direction[moving]
        t_entry[moving] = np.maximum(t_entry[moving], np.minimum(t0, t1))
        t_exit[moving] = np.minimum(t_exit[moving], np.maximum(t0, t1))
    return valid & (t_entry <= t_exit + epsilon) & (t_exit >= -epsilon) & (t_entry <= 1.0 + epsilon)


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite values")
    return vector


def clip_segment_to_convex_prism(
    origin_xyz: np.ndarray,
    end_xyz: np.ndarray,
    polygon_xy: np.ndarray,
    z_range: tuple[float, float],
) -> SegmentClip | None:
    origin = _finite_vector(origin_xyz, "origin_xyz")
    end = _finite_vector(end_xyz, "end_xyz")
    polygon = ensure_ccw(np.asarray(polygon_xy, dtype=np.float64))
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon_xy must contain at least three 2D vertices")
    if not np.isfinite(polygon).all():
        raise ValueError("polygon_xy must contain finite values")
    z_min, z_max = (float(z_range[0]), float(z_range[1]))
    if not np.isfinite([z_min, z_max]).all() or z_min >= z_max:
        raise ValueError("z_range must contain finite ordered values")

    direction = end - origin
    t_entry = 0.0
    t_exit = 1.0
    epsilon = 1e-12

    for index, start in enumerate(polygon):
        edge = polygon[(index + 1) % len(polygon)] - start
        relative = origin[:2] - start
        numerator = float(edge[0] * relative[1] - edge[1] * relative[0])
        denominator = float(edge[0] * direction[1] - edge[1] * direction[0])
        if abs(denominator) <= epsilon:
            if numerator < -epsilon:
                return None
            continue
        boundary_t = -numerator / denominator
        if denominator > 0.0:
            t_entry = max(t_entry, boundary_t)
        else:
            t_exit = min(t_exit, boundary_t)
        if t_entry - t_exit > epsilon:
            return None

    if abs(direction[2]) <= epsilon:
        if origin[2] < z_min - epsilon or origin[2] > z_max + epsilon:
            return None
    else:
        z_t0 = (z_min - origin[2]) / direction[2]
        z_t1 = (z_max - origin[2]) / direction[2]
        t_entry = max(t_entry, min(z_t0, z_t1))
        t_exit = min(t_exit, max(z_t0, z_t1))
        if t_entry - t_exit > epsilon:
            return None

    if t_exit < -epsilon or t_entry > 1.0 + epsilon:
        return None
    t_entry = float(np.clip(t_entry, 0.0, 1.0))
    t_exit = float(np.clip(t_exit, 0.0, 1.0))
    if t_entry - t_exit > epsilon:
        return None
    entry = np.asarray(origin + t_entry * direction, dtype=np.float64)
    exit_point = np.asarray(origin + t_exit * direction, dtype=np.float64)
    entry.setflags(write=False)
    exit_point.setflags(write=False)
    return SegmentClip(entry_xyz=entry, exit_xyz=exit_point, t_entry=t_entry, t_exit=t_exit)


def traverse_voxels_3d(
    origin_xyz: np.ndarray,
    end_xyz: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[tuple[int, int, int], ...]:
    origin = _finite_vector(origin_xyz, "origin_xyz")
    end = _finite_vector(end_xyz, "end_xyz")
    size = np.asarray(voxel_size_xyz, dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("voxel_size_xyz must contain three positive finite values")

    current = np.floor(origin / size).astype(np.int64)
    target = np.floor(end / size).astype(np.int64)
    cells: list[tuple[int, int, int]] = [tuple(int(value) for value in current)]
    if np.array_equal(current, target):
        return tuple(cells)

    direction = end - origin
    step = np.sign(direction).astype(np.int64)
    t_max = np.full(3, np.inf, dtype=np.float64)
    t_delta = np.full(3, np.inf, dtype=np.float64)
    for axis in range(3):
        if direction[axis] > 0.0:
            next_boundary = (current[axis] + 1) * size[axis]
            t_max[axis] = (next_boundary - origin[axis]) / direction[axis]
            t_delta[axis] = size[axis] / direction[axis]
        elif direction[axis] < 0.0:
            next_boundary = current[axis] * size[axis]
            t_max[axis] = (next_boundary - origin[axis]) / direction[axis]
            t_delta[axis] = -size[axis] / direction[axis]

    maximum_steps = int(np.abs(target - current).sum()) + 4
    epsilon = 1e-12
    for _ in range(maximum_steps):
        if np.array_equal(current, target):
            break
        active_axes = current != target
        active_t_max = np.where(active_axes, t_max, np.inf)
        crossing_t = float(np.min(active_t_max))
        crossed_axes = np.flatnonzero(active_axes & (t_max <= crossing_t + epsilon))
        if len(crossed_axes) == 0:
            raise RuntimeError("voxel traversal made no progress")
        for axis in crossed_axes:
            current[axis] += step[axis]
            t_max[axis] += t_delta[axis]
        cells.append(tuple(int(value) for value in current))
    if not np.array_equal(current, target):
        raise RuntimeError("voxel traversal exceeded deterministic step bound")
    return tuple(cells)
