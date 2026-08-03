from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlotFrame:
    center: np.ndarray
    long_axis: np.ndarray
    short_axis: np.ndarray
    length: float
    width: float
    yaw: float


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def signed_polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    return polygon if signed_polygon_area(polygon) >= 0 else polygon[::-1].copy()


def compute_slot_frame(polygon: np.ndarray, center: np.ndarray | None = None) -> SlotFrame:
    polygon = np.asarray(polygon, dtype=np.float64)
    if center is None:
        center = polygon.mean(axis=0)
    else:
        center = np.asarray(center, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if len(lengths) == 0 or float(lengths.max()) <= 1e-12:
        long_axis = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        edge = edges[int(np.argmax(lengths))]
        long_axis = edge / max(np.linalg.norm(edge), 1e-12)
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float64)
    proj_long = (polygon - center) @ long_axis
    proj_short = (polygon - center) @ short_axis
    length = float(proj_long.max() - proj_long.min())
    width = float(proj_short.max() - proj_short.min())
    if width > length:
        long_axis, short_axis = short_axis, -long_axis
        length, width = width, length
    yaw = float(math.atan2(long_axis[1], long_axis[0]))
    return SlotFrame(center=center, long_axis=long_axis, short_axis=short_axis, length=length, width=width, yaw=yaw)


def global_to_local(points_xy: np.ndarray, frame: SlotFrame) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    delta = points_xy - frame.center
    return np.column_stack([delta @ frame.long_axis, delta @ frame.short_axis])


def local_to_global(points_local: np.ndarray, frame: SlotFrame) -> np.ndarray:
    points_local = np.asarray(points_local, dtype=np.float64)
    return frame.center + points_local[:, 0:1] * frame.long_axis + points_local[:, 1:2] * frame.short_axis


def box_polygon(center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    long_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    local = np.asarray(
        [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    return center + local[:, 0:1] * long_axis + local[:, 1:2] * short_axis


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(points) == 0 or len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= intersect
        j = i
    return inside


def points_in_rotated_box(points_xy: np.ndarray, center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    long_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    delta = points_xy - np.asarray(center, dtype=np.float64)
    x = delta @ long_axis
    y = delta @ short_axis
    return (np.abs(x) <= length / 2 + 1e-12) & (np.abs(y) <= width / 2 + 1e-12)


def _inside_half_plane(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    edge = end - start
    rel = point - start
    return float(edge[0] * rel[1] - edge[1] * rel[0]) >= -1e-12


def _line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-12:
        return p2
    qp = q1 - p1
    t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
    return p1 + t * r


def convex_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    subject = ensure_ccw(np.asarray(subject_polygon, dtype=np.float64))
    clip = ensure_ccw(np.asarray(clip_polygon, dtype=np.float64))
    output = subject.copy()
    for i in range(len(clip)):
        if len(output) == 0:
            break
        start = clip[i]
        end = clip[(i + 1) % len(clip)]
        input_list = output
        next_output: list[np.ndarray] = []
        prev = input_list[-1]
        prev_inside = _inside_half_plane(prev, start, end)
        for current in input_list:
            current_inside = _inside_half_plane(current, start, end)
            if current_inside:
                if not prev_inside:
                    next_output.append(_line_intersection(prev, current, start, end))
                next_output.append(current)
            elif prev_inside:
                next_output.append(_line_intersection(prev, current, start, end))
            prev = current
            prev_inside = current_inside
        output = np.asarray(next_output, dtype=np.float64)
    return output


def polygon_overlap_ratio(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> float:
    subject_area = polygon_area(subject_polygon)
    if subject_area <= 1e-12:
        return 0.0
    inter = convex_clip(subject_polygon, clip_polygon)
    return float(np.clip(polygon_area(inter) / subject_area, 0.0, 1.0))
