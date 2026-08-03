from __future__ import annotations

import math

import numpy as np

from .contracts import KnownSlot, MetricSlot


def map_xy_to_slot_m(points_map: np.ndarray, slot: MetricSlot) -> np.ndarray:
    points = np.asarray(points_map, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_map must have shape [N, 2]")
    delta_m = (points - slot.center_map) / slot.map_units_per_meter
    return np.column_stack([delta_m @ slot.long_axis_map, delta_m @ slot.short_axis_map])


def inset_local_rectangle(polygon_local_m: np.ndarray, inset_m: float) -> np.ndarray:
    """Inset an axis-aligned slot-local rectangle by a metric uncertainty band."""

    polygon = np.asarray(polygon_local_m, dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("polygon_local_m must be a finite [4,2] rectangle")
    if not math.isfinite(inset_m) or inset_m < 0.0:
        raise ValueError("inset_m must be finite and non-negative")
    if inset_m == 0.0:
        return polygon
    lower = polygon.min(axis=0) + float(inset_m)
    upper = polygon.max(axis=0) - float(inset_m)
    if np.any(lower >= upper):
        raise ValueError("inset_m collapses the slot core polygon")
    return np.asarray(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
        ],
        dtype=np.float64,
    )


def metric_slot(slot: KnownSlot, map_units_per_meter: float) -> MetricSlot:
    if not math.isfinite(map_units_per_meter) or map_units_per_meter <= 0.0:
        raise ValueError("map_units_per_meter must be positive and finite")
    yaw = math.radians(float(slot.heading_deg))
    long_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    long_axis.setflags(write=False)
    short_axis.setflags(write=False)
    base = MetricSlot(
        slot_id=slot.slot_id,
        center_map=slot.center_map,
        long_axis_map=long_axis,
        short_axis_map=short_axis,
        polygon_local_m=np.empty((0, 2), dtype=np.float64),
        core_polygon_local_m=np.empty((0, 2), dtype=np.float64),
        margin_polygon_local_m=np.empty((0, 2), dtype=np.float64),
        adjacent_slots=slot.adjacent_slots,
        map_units_per_meter=float(map_units_per_meter),
    )
    polygon = map_xy_to_slot_m(slot.polygon_map, base)
    core = map_xy_to_slot_m(slot.core_polygon_map, base)
    margin = map_xy_to_slot_m(slot.margin_polygon_map, base)
    for array in (polygon, core, margin):
        array.setflags(write=False)
    return MetricSlot(
        slot_id=slot.slot_id,
        center_map=slot.center_map,
        long_axis_map=long_axis,
        short_axis_map=short_axis,
        polygon_local_m=polygon,
        core_polygon_local_m=core,
        margin_polygon_local_m=margin,
        adjacent_slots=slot.adjacent_slots,
        map_units_per_meter=float(map_units_per_meter),
    )
