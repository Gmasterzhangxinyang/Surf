#!/usr/bin/env python3
"""Part 1 parking-slot evidence scoring from map-projected LiDAR.

This script is intentionally conservative: "free" requires observed free-space
evidence, while unseen slots remain unknown.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scipy is required for spatial indexing: {exc}") from exc

try:
    from sklearn.cluster import DBSCAN
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required for obstacle clustering: {exc}") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    HAS_MPL = True
except Exception:
    HAS_MPL = False

from gltf_lidar_ndt import accessor_array, decode_data_uri, node_matrix, transform_positions


DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_GLTF = Path("file.gltf")
DEFAULT_TRANSFORM = Path("outputs/alignment_transform.json")
DEFAULT_OUTPUT = Path("outputs/part1_slot_scoring")


@dataclass
class Slot:
    slot_id: str
    polygon: np.ndarray
    inner_polygon: np.ndarray
    core_polygon: np.ndarray
    margin_polygon: np.ndarray
    center: np.ndarray
    heading_rad: float
    length_map: float
    width_map: float
    length_m: float
    width_m: float
    area_m2: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    inner_bbox_min: np.ndarray
    inner_bbox_max: np.ndarray
    core_bbox_min: np.ndarray
    core_bbox_max: np.ndarray
    margin_bbox_min: np.ndarray
    margin_bbox_max: np.ndarray
    adjacent_slots: list[str] = field(default_factory=list)
    adjacent_overlap_zones: list[dict[str, object]] = field(default_factory=list)


@dataclass
class FrameEvidence:
    frame_id: int
    slot_id: str
    visibility_score: float
    observed_area_ratio: float
    ray_free_ratio: float
    occlusion_ratio: float
    obstacle_point_count_inner: int
    obstacle_point_count_margin: int
    object_hit_cell_ratio: float
    core_obstacle_point_count: int
    core_hit_cell_ratio: float
    edge_obstacle_point_count: int
    edge_hit_cell_ratio: float
    margin_obstacle_point_count: int
    margin_hit_cell_ratio: float
    adjacent_overlap_point_count: int
    adjacent_overlap_ratio: float
    cluster_top1_slot: str | None
    cluster_top1_overlap: int
    cluster_top2_slot: str | None
    cluster_top2_overlap: int
    cluster_ownership_margin: float
    cluster_ownership_status: str
    vehicle_like_cluster_count: int
    max_vehicle_like_score: float
    vehicle_cluster_core_overlap_ratio: float
    vehicle_cluster_owner_slot: str | None
    vehicle_cluster_risk: str
    vehicle_cluster_length_m: float
    vehicle_cluster_width_m: float
    vehicle_cluster_height_span_m: float
    height_span: float
    boundary_ratio: float
    distance_to_slot_m: float
    frame_state: str
    uncertainty_type: str | None
    reason: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 1 slot evidence scoring from projected LiDAR")
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--gltf", type=Path, default=DEFAULT_GLTF)
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-slot-distance-m", type=float, default=22.0)
    parser.add_argument("--grid-cell-m", type=float, default=0.20)
    parser.add_argument("--ray-step-m", type=float, default=0.20)
    parser.add_argument("--beam-width-m", type=float, default=0.22)
    parser.add_argument("--inner-shrink-m", type=float, default=0.25)
    parser.add_argument("--core-shrink-m", type=float, default=0.35)
    parser.add_argument("--edge-band-width-m", type=float, default=0.25)
    parser.add_argument("--margin-expand-m", type=float, default=0.25)
    parser.add_argument("--ground-quantile", type=float, default=0.08)
    parser.add_argument("--ground-low-m", type=float, default=0.20)
    parser.add_argument("--obstacle-z-min-m", type=float, default=0.30)
    parser.add_argument("--obstacle-z-max-m", type=float, default=2.50)
    parser.add_argument("--occupied-min-points", type=int, default=6)
    parser.add_argument("--occupied-min-hit-ratio", type=float, default=0.08)
    parser.add_argument("--occupied-min-height-span-m", type=float, default=0.45)
    parser.add_argument("--cluster-eps-m", type=float, default=0.45)
    parser.add_argument("--cluster-min-samples", type=int, default=5)
    parser.add_argument("--cluster-top-ratio", type=float, default=1.5)
    parser.add_argument("--vehicle-min-points", type=int, default=20)
    parser.add_argument("--vehicle-min-length-m", type=float, default=2.5)
    parser.add_argument("--vehicle-max-length-m", type=float, default=6.0)
    parser.add_argument("--vehicle-min-width-m", type=float, default=1.2)
    parser.add_argument("--vehicle-max-width-m", type=float, default=2.8)
    parser.add_argument("--vehicle-min-height-span-m", type=float, default=0.50)
    parser.add_argument("--vehicle-max-height-span-m", type=float, default=2.20)
    parser.add_argument("--vehicle-like-score-threshold", type=float, default=0.65)
    parser.add_argument("--vehicle-owner-ratio-min", type=float, default=0.65)
    parser.add_argument("--vehicle-conflict-score-max", type=float, default=0.35)
    parser.add_argument("--free-min-visibility", type=float, default=0.45)
    parser.add_argument("--free-min-ray-ratio", type=float, default=0.30)
    parser.add_argument("--free-max-hit-ratio", type=float, default=0.04)
    parser.add_argument("--free-support-frames", type=int, default=3)
    parser.add_argument("--occupied-support-frames", type=int, default=2)
    parser.add_argument("--report-frame-limit", type=int, default=6)
    parser.add_argument("--no-report-plots", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def map_units(value_m: float, scale: float) -> float:
    return float(value_m * scale)


def meters(value_map: float, scale: float) -> float:
    return float(value_map / scale)


def order_polygon(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
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


def rect_axes(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    center = poly.mean(axis=0)
    cov = np.cov((poly - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    local = (poly - center) @ axes
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    dims = maxs - mins
    length = float(max(dims[0], dims[1]))
    width = float(min(dims[0], dims[1]))
    if dims[1] > dims[0]:
        axes = axes[:, [1, 0]]
        local = (poly - center) @ axes
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
        dims = maxs - mins
    heading = math.atan2(float(axes[1, 0]), float(axes[0, 0]))
    return center, axes, float(dims[0]), float(dims[1]), heading


def rect_from_axes(center: np.ndarray, axes: np.ndarray, length: float, width: float) -> np.ndarray:
    hx = max(length * 0.5, 1e-6)
    hy = max(width * 0.5, 1e-6)
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    return center[None, :] + local @ axes.T


def extract_slots(gltf_path: Path, scale: float, inner_shrink_m: float, core_shrink_m: float, margin_expand_m: float) -> list[Slot]:
    data = json.loads(gltf_path.read_text(encoding="utf-8", errors="ignore"))
    buffer_bytes = decode_data_uri(data["buffers"][0]["uri"])
    raw_polys: list[np.ndarray] = []

    for node in data.get("nodes", []):
        if (node.get("extras") or {}).get("type") != "parkingSpace" or "mesh" not in node:
            continue
        mesh = data["meshes"][int(node["mesh"])]
        mat = node_matrix(node)
        for primitive in mesh.get("primitives", []):
            pos_accessor = primitive["attributes"].get("POSITION")
            if pos_accessor is None or "indices" not in primitive:
                continue
            positions = accessor_array(data, buffer_bytes, int(pos_accessor)).astype(np.float64)
            positions = transform_positions(positions, mat)[:, :2]
            indices = accessor_array(data, buffer_bytes, int(primitive["indices"])).astype(np.int64).reshape(-1)
            tris = indices.reshape(-1, 3)

            vert_to_tri: dict[int, list[int]] = defaultdict(list)
            for tri_idx, tri in enumerate(tris):
                for vertex in tri:
                    vert_to_tri[int(vertex)].append(tri_idx)

            seen = np.zeros(len(tris), dtype=bool)
            for tri_idx in range(len(tris)):
                if seen[tri_idx]:
                    continue
                stack = [tri_idx]
                seen[tri_idx] = True
                component: list[int] = []
                while stack:
                    current = stack.pop()
                    component.append(current)
                    for vertex in tris[current]:
                        for neighbor in vert_to_tri[int(vertex)]:
                            if not seen[neighbor]:
                                seen[neighbor] = True
                                stack.append(neighbor)

                unique_vertices = np.unique(tris[component].reshape(-1))
                poly_points = np.unique(positions[unique_vertices], axis=0)
                if len(poly_points) < 4:
                    continue
                polygon = order_polygon(poly_points)
                if polygon_area(polygon) <= 1e-8:
                    continue
                raw_polys.append(polygon)

    slots: list[Slot] = []
    inner_shrink = map_units(inner_shrink_m, scale)
    core_shrink = map_units(core_shrink_m, scale)
    margin_expand = map_units(margin_expand_m, scale)
    for idx, polygon in enumerate(raw_polys):
        center, axes, length_map, width_map, heading = rect_axes(polygon)
        inner_length = max(length_map - 2.0 * inner_shrink, length_map * 0.35)
        inner_width = max(width_map - 2.0 * inner_shrink, width_map * 0.35)
        core_length = max(length_map - 2.0 * core_shrink, length_map * 0.25)
        core_width = max(width_map - 2.0 * core_shrink, width_map * 0.25)
        margin_length = length_map + 2.0 * margin_expand
        margin_width = width_map + 2.0 * margin_expand
        inner_polygon = rect_from_axes(center, axes, inner_length, inner_width)
        core_polygon = rect_from_axes(center, axes, core_length, core_width)
        margin_polygon = rect_from_axes(center, axes, margin_length, margin_width)
        slots.append(
            Slot(
                slot_id=f"slot_{idx:04d}",
                polygon=polygon,
                inner_polygon=inner_polygon,
                core_polygon=core_polygon,
                margin_polygon=margin_polygon,
                center=center,
                heading_rad=heading,
                length_map=length_map,
                width_map=width_map,
                length_m=meters(length_map, scale),
                width_m=meters(width_map, scale),
                area_m2=polygon_area(polygon) / (scale * scale),
                bbox_min=polygon.min(axis=0),
                bbox_max=polygon.max(axis=0),
                inner_bbox_min=inner_polygon.min(axis=0),
                inner_bbox_max=inner_polygon.max(axis=0),
                core_bbox_min=core_polygon.min(axis=0),
                core_bbox_max=core_polygon.max(axis=0),
                margin_bbox_min=margin_polygon.min(axis=0),
                margin_bbox_max=margin_polygon.max(axis=0),
            )
        )

    centers = np.asarray([slot.center for slot in slots], dtype=np.float64)
    if len(centers) > 0:
        tree = cKDTree(centers)
        for slot in slots:
            threshold = max(slot.length_map, slot.width_map) * 1.35
            ids = tree.query_ball_point(slot.center, threshold)
            slot.adjacent_slots = [slots[i].slot_id for i in ids if slots[i].slot_id != slot.slot_id][:8]
    by_id = {slot.slot_id: slot for slot in slots}
    for slot in slots:
        zones: list[dict[str, object]] = []
        for adjacent_id in slot.adjacent_slots:
            adjacent = by_id[adjacent_id]
            if np.any(slot.margin_bbox_max < adjacent.margin_bbox_min) or np.any(adjacent.margin_bbox_max < slot.margin_bbox_min):
                continue
            zones.append(
                {
                    "adjacent_slot_id": adjacent_id,
                    "this_margin_polygon_map": slot.margin_polygon.tolist(),
                    "adjacent_margin_polygon_map": adjacent.margin_polygon.tolist(),
                }
            )
        slot.adjacent_overlap_zones = zones
    return slots


def load_frame_rows(path: Path, start_frame: int | None, end_frame: int | None, step: int, max_frames: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if start_frame is not None and frame < start_frame:
                continue
            if end_frame is not None and frame > end_frame:
                continue
            rows.append(row)
    rows = rows[:: max(step, 1)]
    if max_frames > 0:
        rows = rows[:max_frames]
    return rows


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def slot_to_json(slot: Slot) -> dict[str, object]:
    return {
        "slot_id": slot.slot_id,
        "polygon_map": slot.polygon.tolist(),
        "inner_polygon": slot.inner_polygon.tolist(),
        "core_polygon_map": slot.core_polygon.tolist(),
        "edge_band_polygon_map": {
            "outer_inner_polygon_map": slot.inner_polygon.tolist(),
            "inner_core_polygon_map": slot.core_polygon.tolist(),
        },
        "margin_polygon": slot.margin_polygon.tolist(),
        "margin_polygon_map": slot.margin_polygon.tolist(),
        "center_map": slot.center.tolist(),
        "heading_deg": math.degrees(slot.heading_rad),
        "length_m": slot.length_m,
        "width_m": slot.width_m,
        "area_m2": slot.area_m2,
        "adjacent_slots": slot.adjacent_slots,
        "adjacent_overlap_zones": slot.adjacent_overlap_zones,
    }


def write_slot_database(slots: list[Slot], output_dir: Path, scale: float) -> None:
    length_ok = [3.0 <= slot.length_m <= 7.0 for slot in slots]
    width_ok = [1.5 <= slot.width_m <= 3.5 for slot in slots]
    write_json(
        output_dir / "slot_database.json",
        {
            "map_units_per_meter": scale,
            "slot_count": len(slots),
            "slots": [slot_to_json(slot) for slot in slots],
        },
    )
    write_json(
        output_dir / "unit_check.json",
        {
            "map_units_per_meter": scale,
            "slot_count": len(slots),
            "length_m_min": min((slot.length_m for slot in slots), default=0.0),
            "length_m_max": max((slot.length_m for slot in slots), default=0.0),
            "width_m_min": min((slot.width_m for slot in slots), default=0.0),
            "width_m_max": max((slot.width_m for slot in slots), default=0.0),
            "area_m2_min": min((slot.area_m2 for slot in slots), default=0.0),
            "area_m2_max": max((slot.area_m2 for slot in slots), default=0.0),
            "length_ok_count": int(sum(length_ok)),
            "width_ok_count": int(sum(width_ok)),
            "warning": "Dimensions are derived from GLTF map units and transform scale; inspect if ok counts are low.",
        },
    )


def load_points(row: dict[str, str], base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    points_path = Path(row["map_points_path"])
    if not points_path.is_absolute():
        points_path = base_dir / points_path
    data = np.load(points_path)
    return data["points_map_xyzi"].astype(np.float64), data["ego_map_pose"].astype(np.float64)


def bbox_mask(points_xy: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return np.all((points_xy >= bmin) & (points_xy <= bmax), axis=1)


def candidate_slots(slots: list[Slot], center_tree: cKDTree, origin: np.ndarray, max_dist_map: float) -> list[int]:
    ids = center_tree.query_ball_point(origin, max_dist_map)
    return sorted(int(i) for i in ids)


def grid_points_for_slot(slot: Slot, cell_map: float) -> np.ndarray:
    bmin = slot.inner_bbox_min
    bmax = slot.inner_bbox_max
    xs = np.arange(bmin[0] + cell_map * 0.5, bmax[0], cell_map)
    ys = np.arange(bmin[1] + cell_map * 0.5, bmax[1], cell_map)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float64).reshape(2, -1).T
    return grid[points_in_polygon(grid, slot.inner_polygon)]


def grid_points_for_polygon(polygon: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, cell_map: float) -> np.ndarray:
    xs = np.arange(bmin[0] + cell_map * 0.5, bmax[0], cell_map)
    ys = np.arange(bmin[1] + cell_map * 0.5, bmax[1], cell_map)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float64).reshape(2, -1).T
    return grid[points_in_polygon(grid, polygon)]


def ray_free_points(origin: np.ndarray, hits_xy: np.ndarray, step_map: float, max_hits: int = 2400) -> np.ndarray:
    if len(hits_xy) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(hits_xy) > max_hits:
        stride = int(math.ceil(len(hits_xy) / max_hits))
        hits_xy = hits_xy[::stride]
    samples: list[np.ndarray] = []
    for hit in hits_xy:
        vec = hit - origin
        dist = float(np.linalg.norm(vec))
        if dist <= step_map:
            continue
        count = max(1, int(dist / step_map) - 1)
        t = np.linspace(0.0, max(0.0, 1.0 - step_map / dist), count, endpoint=True)
        samples.append(origin[None, :] + t[:, None] * vec[None, :])
    if not samples:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(samples)


def point_slot_region(points: np.ndarray, slot: Slot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    in_margin = points_in_polygon(points, slot.margin_polygon)
    in_inner = points_in_polygon(points, slot.inner_polygon)
    in_core = points_in_polygon(points, slot.core_polygon)
    in_edge = in_inner & ~in_core
    in_margin_only = in_margin & ~in_inner
    return in_core, in_edge, in_margin_only


def soft_range_score(value: float, min_value: float, max_value: float) -> float:
    if value <= 0:
        return 0.0
    if min_value <= value <= max_value:
        return 1.0
    if value < min_value:
        return float(np.clip(value / max(min_value, 1e-9), 0.0, 1.0))
    return float(np.clip(max_value / max(value, 1e-9), 0.0, 1.0))


def cluster_shape_metrics(cluster_xy: np.ndarray, cluster_z: np.ndarray, scale: float) -> dict[str, object]:
    if len(cluster_xy) == 0:
        return {
            "length_m": 0.0,
            "width_m": 0.0,
            "height_span_m": 0.0,
            "footprint_area_m2": 0.0,
            "compactness": 0.0,
            "heading_deg": 0.0,
            "obb_corners_map": [],
        }
    centered = cluster_xy - cluster_xy.mean(axis=0)
    if len(cluster_xy) >= 3 and np.linalg.norm(centered) > 1e-9:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes = vh[:2]
    else:
        axes = np.eye(2, dtype=np.float64)
    proj = centered @ axes.T
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    dims_map = np.maximum(maxs - mins, 0.0)
    length_map = float(max(dims_map))
    width_map = float(min(dims_map))
    length_m = meters(length_map, scale)
    width_m = meters(width_map, scale)
    height_span_m = float(cluster_z.max() - cluster_z.min()) if len(cluster_z) else 0.0
    footprint_area_m2 = float(max(length_m * width_m, 0.0))
    compactness = float(min(width_m / max(length_m, 1e-9), 1.0)) if length_m > 0 else 0.0
    long_axis = axes[0] if dims_map[0] >= dims_map[1] else axes[1]
    heading_deg = float(math.degrees(math.atan2(long_axis[1], long_axis[0])))
    corners_local = np.asarray(
        [
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ],
        dtype=np.float64,
    )
    corners_map = corners_local @ axes + cluster_xy.mean(axis=0)
    return {
        "length_m": length_m,
        "width_m": width_m,
        "height_span_m": height_span_m,
        "footprint_area_m2": footprint_area_m2,
        "compactness": compactness,
        "heading_deg": heading_deg,
        "obb_corners_map": corners_map.tolist(),
    }


def vehicle_like_assessment(
    point_count: int,
    shape: dict[str, object],
    top1_total: int,
    top1_core: int,
    edge_margin: int,
    adjacent_overlap_ratio: float,
    ownership_status: str,
    args: argparse.Namespace,
) -> tuple[float, str]:
    length_m = float(shape["length_m"])
    width_m = float(shape["width_m"])
    height_span_m = float(shape["height_span_m"])
    compactness = float(shape["compactness"])
    core_overlap_ratio = top1_core / max(top1_total, 1)
    point_score = min(1.0, point_count / max(args.vehicle_min_points, 1))
    length_score = soft_range_score(length_m, args.vehicle_min_length_m, args.vehicle_max_length_m)
    width_score = soft_range_score(width_m, args.vehicle_min_width_m, args.vehicle_max_width_m)
    height_score = soft_range_score(height_span_m, args.vehicle_min_height_span_m, args.vehicle_max_height_span_m)
    core_score = float(np.clip(core_overlap_ratio, 0.0, 1.0))
    ownership_score = 1.0 if ownership_status == "clear_core_owned" else 0.55 if ownership_status == "slot_vs_lane_possible" else 0.25
    boundary_penalty = 0.55 * min(1.0, adjacent_overlap_ratio) + 0.35 * min(1.0, edge_margin / max(top1_total, 1))
    score = (
        0.20 * length_score
        + 0.16 * width_score
        + 0.18 * height_score
        + 0.14 * point_score
        + 0.18 * core_score
        + 0.14 * ownership_score
        - boundary_penalty
    )
    score = float(np.clip(score, 0.0, 1.0))
    linearity = length_m / max(width_m, 1e-9) if width_m > 0 else float("inf")
    too_large = length_m > args.vehicle_max_length_m * 1.35 or width_m > args.vehicle_max_width_m * 1.35
    if too_large:
        score = min(score, 0.55)
        risk = "too_large_merged"
    elif height_span_m < args.vehicle_min_height_span_m * 0.55:
        risk = "low_height_structure"
    elif point_count < args.vehicle_min_points * 0.5 or (length_m < 1.2 and width_m < 1.0):
        risk = "too_small_static"
    elif linearity > 6.0 and width_m < 0.65:
        risk = "wall_like_linear"
    elif ownership_status != "clear_core_owned" or adjacent_overlap_ratio > 0.30 or edge_margin > top1_core:
        risk = "boundary_or_adjacent_contaminated"
    elif score >= args.vehicle_like_score_threshold:
        risk = "vehicle_like"
    else:
        risk = "not_vehicle_like"
    return score, risk


def cluster_obstacle_points(
    obstacle_xy: np.ndarray,
    obstacle_z: np.ndarray,
    candidate_ids: list[int],
    slots: list[Slot],
    scale: float,
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if len(obstacle_xy) == 0 or not candidate_ids:
        return {}, []
    near_mask = np.zeros(len(obstacle_xy), dtype=bool)
    for slot_idx in candidate_ids:
        slot = slots[slot_idx]
        mask = bbox_mask(obstacle_xy, slot.margin_bbox_min, slot.margin_bbox_max)
        if mask.any():
            candidates = obstacle_xy[mask]
            inside = points_in_polygon(candidates, slot.margin_polygon)
            idx = np.flatnonzero(mask)
            near_mask[idx[inside]] = True
    near_points = obstacle_xy[near_mask]
    near_z = obstacle_z[near_mask]
    if len(near_points) < args.cluster_min_samples:
        return {}, []

    labels = DBSCAN(eps=map_units(args.cluster_eps_m, scale), min_samples=args.cluster_min_samples).fit_predict(near_points)
    cluster_by_slot: dict[int, dict[str, object]] = {}
    debug_rows: list[dict[str, object]] = []
    candidate_set = set(candidate_ids)
    for cluster_id in sorted(int(v) for v in np.unique(labels) if v >= 0):
        cluster_points = near_points[labels == cluster_id]
        cluster_z = near_z[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        overlaps: list[dict[str, object]] = []
        for slot_idx in candidate_ids:
            slot = slots[slot_idx]
            if np.any(cluster_points.max(axis=0) < slot.margin_bbox_min) or np.any(cluster_points.min(axis=0) > slot.margin_bbox_max):
                continue
            in_core, in_edge, in_margin = point_slot_region(cluster_points, slot)
            core_count = int(in_core.sum())
            edge_count = int(in_edge.sum())
            margin_count = int(in_margin.sum())
            if core_count + edge_count + margin_count <= 0:
                continue
            overlaps.append(
                {
                    "slot_idx": slot_idx,
                    "slot_id": slot.slot_id,
                    "core_count": core_count,
                    "edge_count": edge_count,
                    "margin_count": margin_count,
                    "total_count": core_count + edge_count + margin_count,
                    "distance_to_center": float(np.linalg.norm(cluster_points.mean(axis=0) - slot.center)),
                }
            )
        if not overlaps:
            continue
        overlaps.sort(key=lambda row: (int(row["core_count"]), -float(row["distance_to_center"])), reverse=True)
        top1 = overlaps[0]
        top2 = overlaps[1] if len(overlaps) > 1 else None
        top1_core = int(top1["core_count"])
        top2_core = int(top2["core_count"]) if top2 else 0
        edge_margin = int(top1["edge_count"]) + int(top1["margin_count"])
        adjacent_overlap_count = sum(int(row["total_count"]) for row in overlaps[1:] if int(row["slot_idx"]) in candidate_set)
        adjacent_overlap_ratio = adjacent_overlap_count / max(adjacent_overlap_count + int(top1["total_count"]), 1)
        if top1_core >= args.occupied_min_points and top1_core >= args.cluster_top_ratio * max(top2_core, 1) and top1_core >= edge_margin and adjacent_overlap_ratio <= 0.30:
            status = "clear_core_owned"
        elif top1_core <= 0 and edge_margin > 0:
            status = "margin_only"
        elif top2_core > 0 and top1_core < args.cluster_top_ratio * top2_core:
            status = "adjacent_slot_conflict"
        elif edge_margin > top1_core or adjacent_overlap_ratio > 0.30:
            status = "boundary_conflict"
        else:
            status = "slot_vs_lane_possible"
        shape = cluster_shape_metrics(cluster_points, cluster_z, scale)
        vehicle_like_score, vehicle_cluster_risk = vehicle_like_assessment(
            point_count=int(len(cluster_points)),
            shape=shape,
            top1_total=int(top1["total_count"]),
            top1_core=top1_core,
            edge_margin=edge_margin,
            adjacent_overlap_ratio=adjacent_overlap_ratio,
            ownership_status=status,
            args=args,
        )
        core_overlap_ratio = top1_core / max(int(top1["total_count"]), 1)
        debug = {
            "cluster_id": cluster_id,
            "point_count": int(len(cluster_points)),
            "cluster_center_map": cluster_points.mean(axis=0).tolist(),
            "cluster_z_min": float(cluster_z.min()) if len(cluster_z) else 0.0,
            "cluster_z_max": float(cluster_z.max()) if len(cluster_z) else 0.0,
            "vehicle_like_score": vehicle_like_score,
            "vehicle_cluster_risk": vehicle_cluster_risk,
            "vehicle_cluster_length_m": float(shape["length_m"]),
            "vehicle_cluster_width_m": float(shape["width_m"]),
            "vehicle_cluster_height_span_m": float(shape["height_span_m"]),
            "vehicle_cluster_footprint_area_m2": float(shape["footprint_area_m2"]),
            "vehicle_cluster_compactness": float(shape["compactness"]),
            "vehicle_cluster_heading_deg": float(shape["heading_deg"]),
            "vehicle_cluster_obb_corners_map": shape["obb_corners_map"],
            "vehicle_cluster_core_overlap_ratio": core_overlap_ratio,
            "top1_slot": top1["slot_id"],
            "top1_core_overlap": top1_core,
            "top1_edge_overlap": int(top1["edge_count"]),
            "top1_margin_overlap": int(top1["margin_count"]),
            "top2_slot": top2["slot_id"] if top2 else None,
            "top2_core_overlap": top2_core,
            "adjacent_overlap_ratio": adjacent_overlap_ratio,
            "cluster_ownership_margin": edge_margin / max(int(top1["total_count"]), 1),
            "cluster_ownership_status": status,
        }
        debug_rows.append(debug)
        slot_idx = int(top1["slot_idx"])
        current = cluster_by_slot.get(slot_idx)
        if current is None or (vehicle_like_score, top1_core) > (float(current.get("vehicle_like_score", 0.0)), int(current.get("top1_core_overlap", 0))):
            cluster_by_slot[slot_idx] = debug
    return cluster_by_slot, debug_rows


def classify_frame_state(
    visibility_score: float,
    ray_free_ratio: float,
    object_hit_cell_ratio: float,
    core_hit_cell_ratio: float,
    core_obstacle: int,
    edge_obstacle: int,
    margin_obstacle: int,
    adjacent_overlap_ratio: float,
    height_span: float,
    boundary_ratio: float,
    cluster_ownership_status: str,
    args: argparse.Namespace,
) -> tuple[str, str | None, list[str]]:
    reason: list[str] = []
    uncertainty: str | None = None
    if visibility_score < 0.12:
        return "unknown", "low_visibility", ["slot interior has little or no ray/hit coverage"]
    if visibility_score < 0.30:
        return "unknown", "low_visibility", ["slot interior visibility is below conservative threshold"]

    occupied_like = (
        core_obstacle >= args.occupied_min_points
        and core_hit_cell_ratio >= args.occupied_min_hit_ratio
        and height_span >= args.occupied_min_height_span_m
        and boundary_ratio <= 0.45
        and adjacent_overlap_ratio <= 0.30
        and cluster_ownership_status == "clear_core_owned"
    )
    boundary_conflict = (
        boundary_ratio > 0.45
        or adjacent_overlap_ratio > 0.30
        or cluster_ownership_status in {"boundary_conflict", "adjacent_slot_conflict", "margin_only"}
        or (margin_obstacle + edge_obstacle > core_obstacle * 2 and margin_obstacle + edge_obstacle >= args.occupied_min_points)
    )

    if occupied_like:
        reason.extend(["non-ground obstacle evidence is inside slot core polygon", "cluster ownership clearly belongs to this slot core"])
        return "occupied_candidate", None, reason
    if core_obstacle >= args.occupied_min_points and cluster_ownership_status in {"slot_vs_lane_possible", "none"}:
        return "ambiguous", "slot_vs_lane", ["obstacle cluster is not clearly owned by this slot core"]
    if core_obstacle >= args.occupied_min_points and boundary_conflict:
        return "ambiguous", "boundary_conflict", ["obstacle evidence is mixed with edge, margin, or adjacent-slot overlap"]

    free_like = (
        visibility_score >= args.free_min_visibility
        and ray_free_ratio >= args.free_min_ray_ratio
        and object_hit_cell_ratio <= args.free_max_hit_ratio
        and core_obstacle < args.occupied_min_points
    )
    if free_like and not boundary_conflict:
        return "free_candidate", None, ["slot interior has ray-free evidence", "no strong non-ground obstacle evidence inside inner polygon"]
    if boundary_conflict:
        return "ambiguous", "boundary_conflict", ["points are concentrated near slot margin or adjacent boundary"]
    if object_hit_cell_ratio > args.free_max_hit_ratio:
        return "ambiguous", "static_vs_vehicle", ["weak non-ground evidence exists but is not strong enough for occupied"]
    return "ambiguous", "low_visibility", ["partial visibility is insufficient for a reliable free/occupied interpretation"]


def evidence_to_json(ev: FrameEvidence) -> dict[str, object]:
    return {
        "frame_id": ev.frame_id,
        "slot_id": ev.slot_id,
        "visibility_score": ev.visibility_score,
        "observed_area_ratio": ev.observed_area_ratio,
        "ray_free_ratio": ev.ray_free_ratio,
        "occlusion_ratio": ev.occlusion_ratio,
        "obstacle_point_count_inner": ev.obstacle_point_count_inner,
        "obstacle_point_count_margin": ev.obstacle_point_count_margin,
        "object_hit_cell_ratio": ev.object_hit_cell_ratio,
        "core_obstacle_point_count": ev.core_obstacle_point_count,
        "core_hit_cell_ratio": ev.core_hit_cell_ratio,
        "edge_obstacle_point_count": ev.edge_obstacle_point_count,
        "edge_hit_cell_ratio": ev.edge_hit_cell_ratio,
        "margin_obstacle_point_count": ev.margin_obstacle_point_count,
        "margin_hit_cell_ratio": ev.margin_hit_cell_ratio,
        "adjacent_overlap_point_count": ev.adjacent_overlap_point_count,
        "adjacent_overlap_ratio": ev.adjacent_overlap_ratio,
        "cluster_top1_slot": ev.cluster_top1_slot,
        "cluster_top1_overlap": ev.cluster_top1_overlap,
        "cluster_top2_slot": ev.cluster_top2_slot,
        "cluster_top2_overlap": ev.cluster_top2_overlap,
        "cluster_ownership_margin": ev.cluster_ownership_margin,
        "cluster_ownership_status": ev.cluster_ownership_status,
        "vehicle_like_cluster_count": ev.vehicle_like_cluster_count,
        "max_vehicle_like_score": ev.max_vehicle_like_score,
        "vehicle_cluster_core_overlap_ratio": ev.vehicle_cluster_core_overlap_ratio,
        "vehicle_cluster_owner_slot": ev.vehicle_cluster_owner_slot,
        "vehicle_cluster_risk": ev.vehicle_cluster_risk,
        "vehicle_cluster_length_m": ev.vehicle_cluster_length_m,
        "vehicle_cluster_width_m": ev.vehicle_cluster_width_m,
        "vehicle_cluster_height_span_m": ev.vehicle_cluster_height_span_m,
        "height_span": ev.height_span,
        "boundary_ratio": ev.boundary_ratio,
        "distance_to_slot_m": ev.distance_to_slot_m,
        "frame_state": ev.frame_state,
        "uncertainty_type": ev.uncertainty_type,
        "reason": ev.reason,
    }


def score_frame(
    row: dict[str, str],
    base_dir: Path,
    slots: list[Slot],
    center_tree: cKDTree,
    scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[FrameEvidence], list[dict[str, object]]]:
    frame_id = int(row["frame"])
    points, ego_pose = load_points(row, base_dir)
    origin = ego_pose[:2].astype(np.float64)
    xy = points[:, :2]
    z = points[:, 2]
    finite = np.isfinite(points).all(axis=1)
    xy = xy[finite]
    z = z[finite]
    ground_z = float(np.quantile(z, args.ground_quantile)) if len(z) else 0.0
    low_mask = z <= ground_z + args.ground_low_m
    obstacle_mask = (z >= ground_z + args.obstacle_z_min_m) & (z <= ground_z + args.obstacle_z_max_m)
    max_dist_map = map_units(args.max_slot_distance_m, scale)
    candidate_ids = candidate_slots(slots, center_tree, origin, max_dist_map)
    slot_by_id = {slot.slot_id: slot for slot in slots}
    obstacle_xy = xy[obstacle_mask]
    obstacle_z = z[obstacle_mask]
    cluster_by_slot, cluster_debug_rows = cluster_obstacle_points(obstacle_xy, obstacle_z, candidate_ids, slots, scale, args)
    for row_debug in cluster_debug_rows:
        row_debug["frame_id"] = frame_id
    hit_xy = xy[low_mask | obstacle_mask]
    free_xy = ray_free_points(origin, hit_xy, map_units(args.ray_step_m, scale))
    free_tree = cKDTree(free_xy) if len(free_xy) else None
    cell_map = map_units(args.grid_cell_m, scale)
    beam_map = map_units(args.beam_width_m, scale)

    evidence: list[FrameEvidence] = []
    for slot_idx in candidate_ids:
        slot = slots[slot_idx]
        grid = grid_points_for_slot(slot, cell_map)
        if len(grid) == 0:
            continue
        inner_mask = bbox_mask(xy, slot.inner_bbox_min, slot.inner_bbox_max)
        inner_points = xy[inner_mask]
        inner_z = z[inner_mask]
        inner_obstacle = obstacle_mask[inner_mask]
        inside_inner = points_in_polygon(inner_points, slot.inner_polygon)
        inside_core = points_in_polygon(inner_points, slot.core_polygon)
        inside_edge = inside_inner & ~inside_core
        obstacle_inner_mask = inside_inner & inner_obstacle
        obstacle_core_mask = inside_core & inner_obstacle
        obstacle_edge_mask = inside_edge & inner_obstacle
        obstacle_inner_points = inner_points[obstacle_inner_mask]
        obstacle_core_points = inner_points[obstacle_core_mask]
        obstacle_edge_points = inner_points[obstacle_edge_mask]
        obstacle_core_z = inner_z[obstacle_core_mask]

        margin_mask = bbox_mask(xy, slot.margin_bbox_min, slot.margin_bbox_max)
        margin_points = xy[margin_mask]
        margin_obstacle = obstacle_mask[margin_mask]
        inside_margin = points_in_polygon(margin_points, slot.margin_polygon)
        inside_margin_inner = points_in_polygon(margin_points, slot.inner_polygon)
        margin_only_mask = inside_margin & ~inside_margin_inner & margin_obstacle
        obstacle_margin = int(margin_only_mask.sum())
        margin_obstacle_points = margin_points[margin_only_mask]
        adjacent_overlap_mask = np.zeros(len(margin_points), dtype=bool)
        for adjacent_id in slot.adjacent_slots:
            adjacent = slot_by_id.get(adjacent_id)
            if adjacent is None:
                continue
            adjacent_overlap_mask |= inside_margin & points_in_polygon(margin_points, adjacent.margin_polygon)
        adjacent_overlap_point_count = int((adjacent_overlap_mask & margin_obstacle).sum())

        free_hits = np.zeros(len(grid), dtype=bool)
        if free_tree is not None:
            free_hits = np.asarray(free_tree.query_ball_point(grid, beam_map, return_length=True)) > 0
        object_hits = np.zeros(len(grid), dtype=bool)
        if len(obstacle_core_points):
            obj_tree = cKDTree(obstacle_core_points)
            object_hits = np.asarray(obj_tree.query_ball_point(grid, cell_map * 0.75, return_length=True)) > 0
        core_grid = grid_points_for_polygon(slot.core_polygon, slot.core_bbox_min, slot.core_bbox_max, cell_map)
        edge_grid = grid[~points_in_polygon(grid, slot.core_polygon)]
        margin_grid = grid_points_for_polygon(slot.margin_polygon, slot.margin_bbox_min, slot.margin_bbox_max, cell_map)
        if len(margin_grid):
            margin_grid = margin_grid[~points_in_polygon(margin_grid, slot.inner_polygon)]
        core_hits = np.zeros(len(core_grid), dtype=bool)
        if len(core_grid) and len(obstacle_core_points):
            core_hits = np.asarray(cKDTree(obstacle_core_points).query_ball_point(core_grid, cell_map * 0.75, return_length=True)) > 0
        edge_hits = np.zeros(len(edge_grid), dtype=bool)
        if len(edge_grid) and len(obstacle_edge_points):
            edge_hits = np.asarray(cKDTree(obstacle_edge_points).query_ball_point(edge_grid, cell_map * 0.75, return_length=True)) > 0
        margin_hits = np.zeros(len(margin_grid), dtype=bool)
        if len(margin_grid) and len(margin_obstacle_points):
            margin_hits = np.asarray(cKDTree(margin_obstacle_points).query_ball_point(margin_grid, cell_map * 0.75, return_length=True)) > 0
        low_hits = np.zeros(len(grid), dtype=bool)
        low_inner_points = inner_points[inside_inner & low_mask[inner_mask]]
        if len(low_inner_points):
            low_tree = cKDTree(low_inner_points)
            low_hits = np.asarray(low_tree.query_ball_point(grid, cell_map * 0.75, return_length=True)) > 0

        observed = free_hits | object_hits | low_hits
        visibility_score = float(observed.mean())
        ray_free_ratio = float(free_hits.mean())
        object_hit_cell_ratio = float(core_hits.mean()) if len(core_hits) else 0.0
        core_hit_cell_ratio = object_hit_cell_ratio
        edge_hit_cell_ratio = float(edge_hits.mean()) if len(edge_hits) else 0.0
        margin_hit_cell_ratio = float(margin_hits.mean()) if len(margin_hits) else 0.0
        occlusion_ratio = float(1.0 - visibility_score)
        height_span = float(obstacle_core_z.max() - obstacle_core_z.min()) if len(obstacle_core_z) else 0.0
        obstacle_inner = int(obstacle_inner_mask.sum())
        core_obstacle = int(obstacle_core_mask.sum())
        edge_obstacle = int(obstacle_edge_mask.sum())
        total_near_obstacle = core_obstacle + edge_obstacle + obstacle_margin + adjacent_overlap_point_count
        boundary_ratio = float((edge_obstacle + obstacle_margin + adjacent_overlap_point_count) / max(total_near_obstacle, 1))
        adjacent_overlap_ratio = float(adjacent_overlap_point_count / max(total_near_obstacle, 1))
        cluster_debug = cluster_by_slot.get(slot_idx, {})
        cluster_status = str(cluster_debug.get("cluster_ownership_status", "none"))
        cluster_top1_slot = cluster_debug.get("top1_slot")
        cluster_top1_overlap = int(cluster_debug.get("top1_core_overlap", 0))
        cluster_top2_slot = cluster_debug.get("top2_slot")
        cluster_top2_overlap = int(cluster_debug.get("top2_core_overlap", 0))
        cluster_ownership_margin = float(cluster_debug.get("cluster_ownership_margin", 1.0 if total_near_obstacle else 0.0))
        vehicle_like_score = float(cluster_debug.get("vehicle_like_score", 0.0))
        vehicle_cluster_risk = str(cluster_debug.get("vehicle_cluster_risk", "none"))
        vehicle_cluster_core_overlap_ratio = float(cluster_debug.get("vehicle_cluster_core_overlap_ratio", 0.0))
        vehicle_cluster_owner_slot = cluster_debug.get("top1_slot")
        vehicle_like_cluster_count = 1 if vehicle_like_score >= args.vehicle_like_score_threshold and vehicle_cluster_risk == "vehicle_like" else 0
        vehicle_cluster_length_m = float(cluster_debug.get("vehicle_cluster_length_m", 0.0))
        vehicle_cluster_width_m = float(cluster_debug.get("vehicle_cluster_width_m", 0.0))
        vehicle_cluster_height_span_m = float(cluster_debug.get("vehicle_cluster_height_span_m", 0.0))
        dist_m = meters(float(np.linalg.norm(slot.center - origin)), scale)
        state, uncertainty, reason = classify_frame_state(
            visibility_score,
            ray_free_ratio,
            object_hit_cell_ratio,
            core_hit_cell_ratio,
            core_obstacle,
            edge_obstacle,
            obstacle_margin,
            adjacent_overlap_ratio,
            height_span,
            boundary_ratio,
            cluster_status,
            args,
        )
        if dist_m > args.max_slot_distance_m * 0.9 and state == "free_candidate":
            state = "ambiguous"
            uncertainty = "low_visibility"
            reason.append("slot is near the configured range limit")
        evidence.append(
            FrameEvidence(
                frame_id=frame_id,
                slot_id=slot.slot_id,
                visibility_score=visibility_score,
                observed_area_ratio=visibility_score,
                ray_free_ratio=ray_free_ratio,
                occlusion_ratio=occlusion_ratio,
                obstacle_point_count_inner=obstacle_inner,
                obstacle_point_count_margin=obstacle_margin,
                object_hit_cell_ratio=object_hit_cell_ratio,
                core_obstacle_point_count=core_obstacle,
                core_hit_cell_ratio=core_hit_cell_ratio,
                edge_obstacle_point_count=edge_obstacle,
                edge_hit_cell_ratio=edge_hit_cell_ratio,
                margin_obstacle_point_count=obstacle_margin,
                margin_hit_cell_ratio=margin_hit_cell_ratio,
                adjacent_overlap_point_count=adjacent_overlap_point_count,
                adjacent_overlap_ratio=adjacent_overlap_ratio,
                cluster_top1_slot=str(cluster_top1_slot) if cluster_top1_slot is not None else None,
                cluster_top1_overlap=cluster_top1_overlap,
                cluster_top2_slot=str(cluster_top2_slot) if cluster_top2_slot is not None else None,
                cluster_top2_overlap=cluster_top2_overlap,
                cluster_ownership_margin=cluster_ownership_margin,
                cluster_ownership_status=cluster_status,
                vehicle_like_cluster_count=vehicle_like_cluster_count,
                max_vehicle_like_score=vehicle_like_score,
                vehicle_cluster_core_overlap_ratio=vehicle_cluster_core_overlap_ratio,
                vehicle_cluster_owner_slot=str(vehicle_cluster_owner_slot) if vehicle_cluster_owner_slot is not None else None,
                vehicle_cluster_risk=vehicle_cluster_risk,
                vehicle_cluster_length_m=vehicle_cluster_length_m,
                vehicle_cluster_width_m=vehicle_cluster_width_m,
                vehicle_cluster_height_span_m=vehicle_cluster_height_span_m,
                height_span=height_span,
                boundary_ratio=boundary_ratio,
                distance_to_slot_m=dist_m,
                frame_state=state,
                uncertainty_type=uncertainty,
                reason=reason,
            )
        )

    frame_stats = {
        "frame_id": frame_id,
        "point_count": int(len(z)),
        "ground_z": ground_z,
        "z_min": float(z.min()) if len(z) else 0.0,
        "z_max": float(z.max()) if len(z) else 0.0,
        "non_ground_count": int(obstacle_mask.sum()),
        "candidate_slot_count": int(len(candidate_ids)),
        "evidence_count": int(len(evidence)),
        "cluster_count": int(len(cluster_debug_rows)),
    }
    return frame_stats, evidence, cluster_debug_rows


def update_fusion(acc: dict[str, dict[str, object]], ev: FrameEvidence, args: argparse.Namespace) -> None:
    state = acc.setdefault(
        ev.slot_id,
        {
            "support_frame_count": 0,
            "free_support_frames": 0,
            "occupied_support_frames": 0,
            "ambiguous_support_frames": 0,
            "visibility_sum": 0.0,
            "max_visibility_score": 0.0,
            "max_ray_free_ratio": 0.0,
            "max_object_hit_cell_ratio": 0.0,
            "max_core_hit_cell_ratio": 0.0,
            "max_edge_hit_cell_ratio": 0.0,
            "max_margin_hit_cell_ratio": 0.0,
            "max_height_span": 0.0,
            "max_core_obstacle_point_count": 0,
            "max_edge_obstacle_point_count": 0,
            "max_margin_obstacle_point_count": 0,
            "max_adjacent_overlap_point_count": 0,
            "max_adjacent_overlap_ratio": 0.0,
            "max_boundary_ratio": 0.0,
            "clear_cluster_ownership_frames": 0,
            "boundary_conflict_frames": 0,
            "cluster_status_counts": defaultdict(int),
            "vehicle_like_support_frames": 0,
            "stable_vehicle_cluster_frames": 0,
            "max_vehicle_like_score": 0.0,
            "max_vehicle_cluster_core_overlap_ratio": 0.0,
            "vehicle_cluster_owner_counts": defaultdict(int),
            "vehicle_cluster_risk_counts": defaultdict(int),
            "max_vehicle_cluster_length_m": 0.0,
            "max_vehicle_cluster_width_m": 0.0,
            "max_vehicle_cluster_height_span_m": 0.0,
            "min_distance_to_slot": float("inf"),
            "last_observed_frame": 0,
            "uncertainty_counts": defaultdict(int),
            "reasons": [],
        },
    )
    state["support_frame_count"] = int(state["support_frame_count"]) + 1
    if ev.frame_state == "free_candidate":
        state["free_support_frames"] = int(state["free_support_frames"]) + 1
    elif ev.frame_state == "occupied_candidate":
        state["occupied_support_frames"] = int(state["occupied_support_frames"]) + 1
    elif ev.frame_state == "ambiguous":
        state["ambiguous_support_frames"] = int(state["ambiguous_support_frames"]) + 1
    state["visibility_sum"] = float(state["visibility_sum"]) + ev.visibility_score
    state["max_visibility_score"] = max(float(state["max_visibility_score"]), ev.visibility_score)
    state["max_ray_free_ratio"] = max(float(state["max_ray_free_ratio"]), ev.ray_free_ratio)
    state["max_object_hit_cell_ratio"] = max(float(state["max_object_hit_cell_ratio"]), ev.object_hit_cell_ratio)
    state["max_core_hit_cell_ratio"] = max(float(state["max_core_hit_cell_ratio"]), ev.core_hit_cell_ratio)
    state["max_edge_hit_cell_ratio"] = max(float(state["max_edge_hit_cell_ratio"]), ev.edge_hit_cell_ratio)
    state["max_margin_hit_cell_ratio"] = max(float(state["max_margin_hit_cell_ratio"]), ev.margin_hit_cell_ratio)
    state["max_height_span"] = max(float(state["max_height_span"]), ev.height_span)
    state["max_core_obstacle_point_count"] = max(int(state["max_core_obstacle_point_count"]), ev.core_obstacle_point_count)
    state["max_edge_obstacle_point_count"] = max(int(state["max_edge_obstacle_point_count"]), ev.edge_obstacle_point_count)
    state["max_margin_obstacle_point_count"] = max(int(state["max_margin_obstacle_point_count"]), ev.margin_obstacle_point_count)
    state["max_adjacent_overlap_point_count"] = max(int(state["max_adjacent_overlap_point_count"]), ev.adjacent_overlap_point_count)
    state["max_adjacent_overlap_ratio"] = max(float(state["max_adjacent_overlap_ratio"]), ev.adjacent_overlap_ratio)
    state["max_boundary_ratio"] = max(float(state["max_boundary_ratio"]), ev.boundary_ratio)
    if ev.cluster_ownership_status == "clear_core_owned":
        state["clear_cluster_ownership_frames"] = int(state["clear_cluster_ownership_frames"]) + 1
    if ev.cluster_ownership_status in {"boundary_conflict", "adjacent_slot_conflict", "margin_only"} or ev.boundary_ratio > 0.60:
        state["boundary_conflict_frames"] = int(state["boundary_conflict_frames"]) + 1
    state["cluster_status_counts"][ev.cluster_ownership_status] += 1
    if ev.max_vehicle_like_score > 0:
        if ev.max_vehicle_like_score > float(state["max_vehicle_like_score"]):
            state["max_vehicle_like_score"] = ev.max_vehicle_like_score
            state["max_vehicle_cluster_core_overlap_ratio"] = ev.vehicle_cluster_core_overlap_ratio
            state["max_vehicle_cluster_length_m"] = ev.vehicle_cluster_length_m
            state["max_vehicle_cluster_width_m"] = ev.vehicle_cluster_width_m
            state["max_vehicle_cluster_height_span_m"] = ev.vehicle_cluster_height_span_m
        state["vehicle_cluster_risk_counts"][ev.vehicle_cluster_risk] += 1
    if ev.vehicle_like_cluster_count > 0:
        state["vehicle_like_support_frames"] = int(state["vehicle_like_support_frames"]) + 1
        if ev.vehicle_cluster_owner_slot:
            state["vehicle_cluster_owner_counts"][ev.vehicle_cluster_owner_slot] += 1
        if ev.vehicle_cluster_owner_slot == ev.slot_id and ev.vehicle_cluster_core_overlap_ratio >= args.vehicle_owner_ratio_min:
            state["stable_vehicle_cluster_frames"] = int(state["stable_vehicle_cluster_frames"]) + 1
    state["min_distance_to_slot"] = min(float(state["min_distance_to_slot"]), ev.distance_to_slot_m)
    state["last_observed_frame"] = max(int(state["last_observed_frame"]), ev.frame_id)
    if ev.uncertainty_type:
        state["uncertainty_counts"][ev.uncertainty_type] += 1
    if len(state["reasons"]) < 8:
        state["reasons"].extend(ev.reason[:2])


def finalize_slot(slot: Slot, state: dict[str, object] | None, args: argparse.Namespace) -> dict[str, object]:
    if state is None:
        return {
            "slot_id": slot.slot_id,
            "state": "unknown",
            "p_free": 0.0,
            "p_occupied": 0.0,
            "p_unknown": 1.0,
            "availability_score": 0.0,
            "visibility_score": 0.0,
            "confidence": 0.0,
            "diagnostic_label": "",
            "evidence_level": "",
            "uncertainty_type": "low_visibility",
            "reason": ["slot was not observed in selected frames"],
            "evidence": {
                "support_frame_count": 0,
                "free_support_frames": 0,
                "occupied_support_frames": 0,
                "ambiguous_support_frames": 0,
                "conflict_score": 0.0,
                "visibility_accum": 0.0,
                "max_visibility_score": 0.0,
                "max_ray_free_ratio": 0.0,
                "max_object_hit_cell_ratio": 0.0,
                "max_core_hit_cell_ratio": 0.0,
                "max_edge_hit_cell_ratio": 0.0,
                "max_margin_hit_cell_ratio": 0.0,
                "max_height_span": 0.0,
                "max_core_obstacle_point_count": 0,
                "max_edge_obstacle_point_count": 0,
                "max_margin_obstacle_point_count": 0,
                "max_adjacent_overlap_point_count": 0,
                "max_adjacent_overlap_ratio": 0.0,
                "max_boundary_ratio": 0.0,
                "clear_cluster_ownership_frames": 0,
                "boundary_conflict_frames": 0,
                "vehicle_like_support_frames": 0,
                "stable_vehicle_cluster_frames": 0,
                "max_vehicle_like_score": 0.0,
                "max_vehicle_cluster_core_overlap_ratio": 0.0,
                "dominant_vehicle_owner_slot": "",
                "dominant_vehicle_owner_ratio": 0.0,
                "vehicle_cluster_conflict_score": 0.0,
                "vehicle_cluster_risk_counts": {},
                "max_vehicle_cluster_length_m": 0.0,
                "max_vehicle_cluster_width_m": 0.0,
                "max_vehicle_cluster_height_span_m": 0.0,
                "min_distance_to_slot": float("inf"),
            },
        }

    support = int(state["support_frame_count"])
    free_support = int(state["free_support_frames"])
    occ_support = int(state["occupied_support_frames"])
    amb_support = int(state["ambiguous_support_frames"])
    visibility_avg = float(state["visibility_sum"]) / max(support, 1)
    max_vis = float(state["max_visibility_score"])
    max_ray = float(state["max_ray_free_ratio"])
    max_obj = float(state["max_object_hit_cell_ratio"])
    max_core_hit = float(state["max_core_hit_cell_ratio"])
    max_height = float(state["max_height_span"])
    max_boundary = float(state["max_boundary_ratio"])
    max_adjacent = float(state["max_adjacent_overlap_ratio"])
    clear_cluster_frames = int(state["clear_cluster_ownership_frames"])
    vehicle_like_support = int(state["vehicle_like_support_frames"])
    stable_vehicle_frames = int(state["stable_vehicle_cluster_frames"])
    max_vehicle_score = float(state["max_vehicle_like_score"])
    vehicle_owner_counts = state.get("vehicle_cluster_owner_counts", {}) or {}
    if vehicle_owner_counts:
        dominant_vehicle_owner_slot, dominant_vehicle_owner_count = max(vehicle_owner_counts.items(), key=lambda kv: kv[1])
    else:
        dominant_vehicle_owner_slot, dominant_vehicle_owner_count = "", 0
    dominant_vehicle_owner_ratio = float(np.clip(dominant_vehicle_owner_count / max(vehicle_like_support, 1), 0.0, 1.0))
    vehicle_cluster_conflict_score = max(0.0, 1.0 - dominant_vehicle_owner_ratio) if vehicle_like_support else 0.0
    conflict_score = min(1.0, amb_support / max(support, 1) + (0.35 if free_support and occ_support else 0.0))
    uncertainty_counts = state["uncertainty_counts"]
    uncertainty_type = None
    if uncertainty_counts:
        uncertainty_type = max(uncertainty_counts.items(), key=lambda kv: kv[1])[0]

    if support == 0 or visibility_avg < 0.18:
        final = "unknown"
    elif (
        occ_support >= args.occupied_support_frames
        and max_core_hit >= args.occupied_min_hit_ratio
        and int(state["max_core_obstacle_point_count"]) >= args.occupied_min_points
        and max_height >= args.occupied_min_height_span_m
        and max_boundary <= 0.45
        and max_adjacent <= 0.30
        and clear_cluster_frames >= args.occupied_support_frames
        and conflict_score < 0.55
    ):
        final = "occupied"
    elif (
        free_support >= args.free_support_frames
        and visibility_avg >= args.free_min_visibility
        and max_ray >= args.free_min_ray_ratio
        and max_obj <= args.free_max_hit_ratio
        and conflict_score < 0.35
    ):
        final = "free"
    else:
        final = "ambiguous"

    occ_strength = min(1.0, 0.45 * (occ_support / max(args.occupied_support_frames, 1)) + 0.35 * (max_obj / max(args.occupied_min_hit_ratio, 1e-6)) + 0.20 * (max_height / 1.5))
    free_strength = min(1.0, 0.45 * (free_support / max(args.free_support_frames, 1)) + 0.35 * (max_ray / max(args.free_min_ray_ratio, 1e-6)) + 0.20 * visibility_avg)
    unknown_strength = max(0.0, 1.0 - min(1.0, visibility_avg + 0.15 * support))
    total = free_strength + occ_strength + unknown_strength + 1e-9
    p_free = free_strength / total
    p_occ = occ_strength / total
    p_unknown = unknown_strength / total
    if final == "free":
        p_free = max(p_free, 0.60)
        p_occ = min(p_occ, 0.25)
        p_unknown = min(p_unknown, 0.25)
    elif final == "occupied":
        p_occ = max(p_occ, 0.60)
        p_free = min(p_free, 0.25)
        p_unknown = min(p_unknown, 0.25)
    elif final == "unknown":
        p_unknown = max(p_unknown, 0.70)
        p_free = min(p_free, 0.15)
        p_occ = min(p_occ, 0.15)
    else:
        p_unknown = max(p_unknown, 0.25)
        p_free = min(p_free, 0.45)
        p_occ = min(p_occ, 0.45)
    norm = p_free + p_occ + p_unknown + 1e-9
    p_free /= norm
    p_occ /= norm
    p_unknown /= norm
    if final == "free":
        overflow = max(0.0, p_occ - 0.25) + max(0.0, p_unknown - 0.25)
        p_occ = min(p_occ, 0.25)
        p_unknown = min(p_unknown, 0.25)
        p_free += overflow
    elif final == "occupied":
        overflow = max(0.0, p_free - 0.25) + max(0.0, p_unknown - 0.25)
        p_free = min(p_free, 0.25)
        p_unknown = min(p_unknown, 0.25)
        p_occ += overflow
    elif final == "unknown":
        overflow = max(0.0, p_free - 0.15) + max(0.0, p_occ - 0.15)
        p_free = min(p_free, 0.15)
        p_occ = min(p_occ, 0.15)
        p_unknown += overflow
    else:
        overflow = max(0.0, p_free - 0.45) + max(0.0, p_occ - 0.45)
        p_free = min(p_free, 0.45)
        p_occ = min(p_occ, 0.45)
        p_unknown += overflow
    confidence = float(np.clip((1.0 - conflict_score) * min(1.0, visibility_avg / max(args.free_min_visibility, 1e-6)), 0.0, 1.0))
    availability = float(p_free * visibility_avg * confidence) if final == "free" else 0.0
    diagnostic_label = ""
    evidence_level = ""
    boundary_conflict_frames = int(state["boundary_conflict_frames"])
    boundary_frame_ratio = boundary_conflict_frames / max(support, 1)
    if (
        vehicle_like_support >= args.occupied_support_frames
        and max_vehicle_score >= args.vehicle_like_score_threshold
        and dominant_vehicle_owner_slot == slot.slot_id
        and dominant_vehicle_owner_ratio >= args.vehicle_owner_ratio_min
        and vehicle_cluster_conflict_score <= args.vehicle_conflict_score_max
    ):
        if boundary_frame_ratio <= 0.45:
            diagnostic_label = "likely_occupied_vehicle_cluster"
            evidence_level = "high" if stable_vehicle_frames >= args.occupied_support_frames and max_vehicle_score >= 0.75 else "medium"
        elif vehicle_like_support > 0:
            diagnostic_label = "vehicle_cluster_boundary_conflict"
            evidence_level = "medium"
    if not diagnostic_label and vehicle_like_support > 0:
        diagnostic_label = "possible_occupied_vehicle_cluster"
        evidence_level = "medium" if max_vehicle_score >= args.vehicle_like_score_threshold else "low"
    if not diagnostic_label and state.get("vehicle_cluster_risk_counts"):
        risk_counts = state.get("vehicle_cluster_risk_counts", {}) or {}
        static_like = sum(int(risk_counts.get(k, 0)) for k in ["too_small_static", "wall_like_linear", "low_height_structure"])
        if static_like > 0:
            diagnostic_label = "static_like_cluster"
            evidence_level = "low"
    return {
        "slot_id": slot.slot_id,
        "state": final,
        "p_free": float(p_free),
        "p_occupied": float(p_occ),
        "p_unknown": float(p_unknown),
        "availability_score": availability,
        "visibility_score": visibility_avg,
        "confidence": confidence,
        "diagnostic_label": diagnostic_label,
        "evidence_level": evidence_level,
        "uncertainty_type": uncertainty_type if final in {"unknown", "ambiguous"} else None,
        "reason": list(dict.fromkeys(state["reasons"]))[:6],
        "evidence": {
            "support_frame_count": support,
            "free_support_frames": free_support,
            "occupied_support_frames": occ_support,
            "ambiguous_support_frames": amb_support,
            "visibility_accum": float(state["visibility_sum"]),
            "max_visibility_score": max_vis,
            "max_ray_free_ratio": max_ray,
            "max_object_hit_cell_ratio": max_obj,
            "max_core_hit_cell_ratio": max_core_hit,
            "max_edge_hit_cell_ratio": float(state["max_edge_hit_cell_ratio"]),
            "max_margin_hit_cell_ratio": float(state["max_margin_hit_cell_ratio"]),
            "max_height_span": max_height,
            "max_core_obstacle_point_count": int(state["max_core_obstacle_point_count"]),
            "max_edge_obstacle_point_count": int(state["max_edge_obstacle_point_count"]),
            "max_margin_obstacle_point_count": int(state["max_margin_obstacle_point_count"]),
            "max_adjacent_overlap_point_count": int(state["max_adjacent_overlap_point_count"]),
            "max_adjacent_overlap_ratio": max_adjacent,
            "max_boundary_ratio": max_boundary,
            "clear_cluster_ownership_frames": clear_cluster_frames,
            "boundary_conflict_frames": int(state["boundary_conflict_frames"]),
            "vehicle_like_support_frames": vehicle_like_support,
            "stable_vehicle_cluster_frames": stable_vehicle_frames,
            "max_vehicle_like_score": max_vehicle_score,
            "max_vehicle_cluster_core_overlap_ratio": float(state["max_vehicle_cluster_core_overlap_ratio"]),
            "dominant_vehicle_owner_slot": str(dominant_vehicle_owner_slot),
            "dominant_vehicle_owner_ratio": dominant_vehicle_owner_ratio,
            "vehicle_cluster_conflict_score": vehicle_cluster_conflict_score,
            "vehicle_cluster_risk_counts": dict(state.get("vehicle_cluster_risk_counts", {})),
            "max_vehicle_cluster_length_m": float(state["max_vehicle_cluster_length_m"]),
            "max_vehicle_cluster_width_m": float(state["max_vehicle_cluster_width_m"]),
            "max_vehicle_cluster_height_span_m": float(state["max_vehicle_cluster_height_span_m"]),
            "cluster_status_counts": dict(state["cluster_status_counts"]),
            "min_distance_to_slot": float(state["min_distance_to_slot"]),
            "conflict_score": conflict_score,
            "last_observed_frame": int(state["last_observed_frame"]),
        },
    }


def validate_scores(scores: list[dict[str, object]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    report: list[dict[str, object]] = []
    for score in scores:
        original = score["state"]
        evidence = score["evidence"]
        reasons: list[str] = []
        if score["state"] == "free":
            if float(score["visibility_score"]) < args.free_min_visibility:
                reasons.append("visibility below free threshold")
            if int(evidence["free_support_frames"]) < args.free_support_frames:
                reasons.append("not enough free support frames")
            if float(evidence["max_object_hit_cell_ratio"]) > args.free_max_hit_ratio:
                reasons.append("object evidence too high for free")
            if float(evidence["conflict_score"]) >= 0.35:
                reasons.append("conflict score too high for free")
        if score["state"] == "occupied":
            if int(evidence["occupied_support_frames"]) < args.occupied_support_frames:
                reasons.append("not enough occupied support frames")
            if float(evidence.get("max_core_hit_cell_ratio", 0.0)) < args.occupied_min_hit_ratio:
                reasons.append("core hit ratio below occupied threshold")
            if int(evidence.get("max_core_obstacle_point_count", 0)) < args.occupied_min_points:
                reasons.append("not enough core obstacle points")
            if float(evidence.get("max_boundary_ratio", 1.0)) > 0.45:
                reasons.append("boundary ratio too high for occupied")
            if float(evidence.get("max_adjacent_overlap_ratio", 1.0)) > 0.30:
                reasons.append("adjacent overlap too high for occupied")
            if int(evidence.get("clear_cluster_ownership_frames", 0)) < args.occupied_support_frames:
                reasons.append("cluster ownership is not clearly core-supported")
            if float(evidence["conflict_score"]) >= 0.55:
                reasons.append("conflict score too high for occupied")
        if reasons:
            score["state"] = "ambiguous"
            score["availability_score"] = 0.0
            if any("boundary" in reason or "adjacent" in reason or "cluster ownership" in reason for reason in reasons):
                score["uncertainty_type"] = "boundary_conflict"
            else:
                score["uncertainty_type"] = score.get("uncertainty_type") or "temporal_conflict"
            score["reason"] = list(score.get("reason", [])) + [f"validator downgrade: {', '.join(reasons)}"]
            report.append({"slot_id": score["slot_id"], "from": original, "to": score["state"], "reasons": reasons})
    return scores, report


def make_case_cards(scores: list[dict[str, object]], limit: int = 250) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    candidates = [s for s in scores if s["state"] in {"ambiguous", "unknown"} and int(s["evidence"].get("support_frame_count", 0)) > 0]
    candidates.sort(key=lambda s: (s["state"] != "ambiguous", -float(s["visibility_score"]), -float(s["evidence"].get("max_object_hit_cell_ratio", 0.0))))
    for idx, score in enumerate(candidates[:limit]):
        uncertainty = score.get("uncertainty_type") or "low_visibility"
        question = {
            "slot_vs_lane": "Is this projected region visually a parking slot or a driving lane?",
            "boundary_conflict": "Do image evidence and neighboring slots resolve the boundary conflict?",
            "low_visibility": "Can camera or temporal frames confirm whether the slot is visible and empty?",
            "occlusion": "Is this slot occluded by a foreground object?",
            "static_vs_vehicle": "Are non-ground returns static map objects or a vehicle?",
            "temporal_conflict": "Which temporal frames explain the conflicting free and occupied evidence?",
            "pose_boundary_error": "Does pose error move evidence into an adjacent slot or lane?",
        }.get(str(uncertainty), "What evidence would reduce this slot uncertainty?")
        cards.append(
            {
                "case_id": f"case_{idx:05d}",
                "slot_id": score["slot_id"],
                "part1_state": score["state"],
                "uncertainty_type": uncertainty,
                "priority": "high" if score["state"] == "ambiguous" else "medium",
                "pointcloud_summary": {
                    "visibility_score": score["visibility_score"],
                    "ray_free_ratio": score["evidence"].get("max_ray_free_ratio", 0.0),
                    "object_hit_cell_ratio": score["evidence"].get("max_object_hit_cell_ratio", 0.0),
                    "conflict_score": score["evidence"].get("conflict_score", 0.0),
                },
                "question_for_part2": [question],
                "recommended_camera_tools": [
                    "project_slot_to_camera",
                    "crop_slot_region",
                    "detect_parking_lines",
                    "detect_vehicle",
                    "inspect_temporal_frames",
                ],
            }
        )
    return cards


def free_gate_failures(score: dict[str, object], args: argparse.Namespace) -> list[str]:
    evidence = score["evidence"]
    support = int(evidence.get("support_frame_count", 0))
    visibility_accum = float(evidence.get("visibility_accum", 0.0))
    max_ray = float(evidence.get("max_ray_free_ratio", 0.0))
    max_obj = float(evidence.get("max_object_hit_cell_ratio", 0.0))
    conflict = float(evidence.get("conflict_score", 0.0))
    min_distance = float(evidence.get("min_distance_to_slot", float("inf")))
    failures: list[str] = []
    if support < args.free_support_frames:
        failures.append("failed_support_frame_count")
    if visibility_accum < args.free_min_visibility * args.free_support_frames:
        failures.append("failed_visibility_accum")
    if max_ray < args.free_min_ray_ratio:
        failures.append("failed_ray_free_area")
    if max_obj > args.free_max_hit_ratio:
        failures.append("failed_object_hit")
    if not math.isfinite(min_distance) or min_distance > args.max_slot_distance_m * 0.9:
        failures.append("failed_distance")
    if conflict >= 0.35:
        failures.append("failed_conflict")
    return failures


def build_free_gate_analysis(scores: list[dict[str, object]], args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = {
        "slot_count": len(scores),
        "free_gate_thresholds": {
            "support_frame_count_min": args.free_support_frames,
            "visibility_accum_min": args.free_min_visibility * args.free_support_frames,
            "max_ray_free_area_ratio_min": args.free_min_ray_ratio,
            "max_object_hit_cell_ratio_max": args.free_max_hit_ratio,
            "min_distance_to_slot_m_max": args.max_slot_distance_m * 0.9,
            "conflict_score_max_exclusive": 0.35,
        },
        "failure_counts": defaultdict(int),
    }
    rows: list[dict[str, object]] = []
    for score in scores:
        evidence = score["evidence"]
        failures = free_gate_failures(score, args)
        for failure in failures:
            summary["failure_counts"][failure] += 1
        min_distance = float(evidence.get("min_distance_to_slot", float("inf")))
        rows.append(
            {
                "slot_id": score["slot_id"],
                "state": score["state"],
                "support_frame_count": int(evidence.get("support_frame_count", 0)),
                "visibility_accum": float(evidence.get("visibility_accum", 0.0)),
                "max_visibility_score": float(evidence.get("max_visibility_score", 0.0)),
                "max_ray_free_area_ratio": float(evidence.get("max_ray_free_ratio", 0.0)),
                "max_object_hit_cell_ratio": float(evidence.get("max_object_hit_cell_ratio", 0.0)),
                "min_distance_to_slot_m": min_distance if math.isfinite(min_distance) else None,
                "conflict_score": float(evidence.get("conflict_score", 0.0)),
                "failed_free_gates": failures,
            }
        )
    summary["failure_counts"] = dict(summary["failure_counts"])
    return summary, rows


def write_free_gate_analysis(output_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    write_json(output_dir / "free_gate_analysis.json", {"summary": summary, "slots": rows})
    fields = [
        "slot_id",
        "state",
        "support_frame_count",
        "visibility_accum",
        "max_visibility_score",
        "max_ray_free_area_ratio",
        "max_object_hit_cell_ratio",
        "min_distance_to_slot_m",
        "conflict_score",
        "failed_free_gates",
    ]
    with (output_dir / "free_gate_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["failed_free_gates"] = "|".join(row["failed_free_gates"])
            writer.writerow(out)


def build_evidence_by_slot(evidence_records: list[FrameEvidence]) -> dict[str, list[FrameEvidence]]:
    by_slot: dict[str, list[FrameEvidence]] = defaultdict(list)
    for ev in evidence_records:
        by_slot[ev.slot_id].append(ev)
    return by_slot


def strongest_occupied_frame(records: list[FrameEvidence]) -> FrameEvidence | None:
    if not records:
        return None
    return max(
        records,
        key=lambda ev: (
            ev.core_hit_cell_ratio,
            ev.core_obstacle_point_count,
            ev.height_span,
            ev.visibility_score,
            -ev.boundary_ratio,
        ),
    )


def build_occupied_audit(scores: list[dict[str, object]], evidence_records: list[FrameEvidence]) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_slot = build_evidence_by_slot(evidence_records)
    occupied_scores = [score for score in scores if score["state"] == "occupied"]
    rows: list[dict[str, object]] = []
    for score in occupied_scores:
        records = by_slot.get(str(score["slot_id"]), [])
        strongest = strongest_occupied_frame(records)
        evidence = score["evidence"]
        max_core_points = max((ev.core_obstacle_point_count for ev in records), default=0)
        max_edge_points = max((ev.edge_obstacle_point_count for ev in records), default=0)
        max_margin_points = max((ev.margin_obstacle_point_count for ev in records), default=0)
        max_boundary_ratio = max((ev.boundary_ratio for ev in records), default=0.0)
        max_adjacent = max((ev.adjacent_overlap_ratio for ev in records), default=0.0)
        if max_boundary_ratio > 0.60:
            occupied_risk = "high_boundary_risk"
        elif max_adjacent > 0.30:
            occupied_risk = "adjacent_slot_conflict"
        elif max_margin_points > max_core_points:
            occupied_risk = "margin_only_evidence"
        elif strongest and strongest.cluster_ownership_status == "slot_vs_lane_possible":
            occupied_risk = "slot_vs_lane_possible"
        elif strongest and strongest.cluster_ownership_status not in {"clear_core_owned", "none"}:
            occupied_risk = "static_structure_possible"
        else:
            occupied_risk = "low_core_supported"
        rows.append(
            {
                "slot_id": score["slot_id"],
                "occupied_support_frames": int(evidence.get("occupied_support_frames", 0)),
                "max_object_hit_cell_ratio": float(evidence.get("max_object_hit_cell_ratio", 0.0)),
                "core_obstacle_point_count": int(max_core_points),
                "core_hit_cell_ratio": float(evidence.get("max_core_hit_cell_ratio", 0.0)),
                "edge_obstacle_point_count": int(max_edge_points),
                "edge_hit_cell_ratio": float(evidence.get("max_edge_hit_cell_ratio", 0.0)),
                "margin_obstacle_point_count": int(max_margin_points),
                "margin_hit_cell_ratio": float(evidence.get("max_margin_hit_cell_ratio", 0.0)),
                "max_height_span_m": float(evidence.get("max_height_span", 0.0)),
                "max_obstacle_points_inner": int(max_core_points),
                "boundary_ratio": float(max_boundary_ratio),
                "adjacent_overlap_ratio": float(max_adjacent),
                "min_distance_to_slot_m": float(evidence.get("min_distance_to_slot", 0.0)),
                "strongest_frame_id": int(strongest.frame_id) if strongest else None,
                "strongest_frame_state": strongest.frame_state if strongest else "",
                "cluster_top1_slot": strongest.cluster_top1_slot if strongest else None,
                "cluster_top1_overlap": strongest.cluster_top1_overlap if strongest else 0,
                "cluster_top2_slot": strongest.cluster_top2_slot if strongest else None,
                "cluster_top2_overlap": strongest.cluster_top2_overlap if strongest else 0,
                "cluster_ownership_margin": strongest.cluster_ownership_margin if strongest else 0.0,
                "cluster_ownership_status": strongest.cluster_ownership_status if strongest else "",
                "static_or_boundary_risk": occupied_risk,
                "occupied_risk": occupied_risk,
                "reason": score.get("reason", []),
                "debug_bev_path": "",
            }
        )
    summary = {
        "occupied_count": len(rows),
        "high_boundary_risk_count": sum(1 for row in rows if row["occupied_risk"] == "high_boundary_risk"),
        "low_core_supported_count": sum(1 for row in rows if row["occupied_risk"] == "low_core_supported"),
        "adjacent_slot_conflict_count": sum(1 for row in rows if row["occupied_risk"] == "adjacent_slot_conflict"),
        "margin_only_evidence_count": sum(1 for row in rows if row["occupied_risk"] == "margin_only_evidence"),
        "slot_vs_lane_possible_count": sum(1 for row in rows if row["occupied_risk"] == "slot_vs_lane_possible"),
        "static_structure_possible_count": sum(1 for row in rows if row["occupied_risk"] == "static_structure_possible"),
        "debug_bev_available": False,
    }
    return summary, rows


def write_occupied_audit(output_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    write_json(output_dir / "occupied_audit.json", {"summary": summary, "occupied_slots": rows})
    fields = [
        "slot_id",
        "occupied_support_frames",
        "max_object_hit_cell_ratio",
        "core_obstacle_point_count",
        "core_hit_cell_ratio",
        "edge_obstacle_point_count",
        "edge_hit_cell_ratio",
        "margin_obstacle_point_count",
        "margin_hit_cell_ratio",
        "max_height_span_m",
        "max_obstacle_points_inner",
        "boundary_ratio",
        "adjacent_overlap_ratio",
        "min_distance_to_slot_m",
        "strongest_frame_id",
        "strongest_frame_state",
        "cluster_top1_slot",
        "cluster_top1_overlap",
        "cluster_top2_slot",
        "cluster_top2_overlap",
        "cluster_ownership_margin",
        "cluster_ownership_status",
        "static_or_boundary_risk",
        "occupied_risk",
        "reason",
        "debug_bev_path",
    ]
    with (output_dir / "occupied_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["reason"] = " | ".join(row.get("reason", []))
            writer.writerow(out)


def build_vehicle_cluster_audit(scores: list[dict[str, object]], cluster_rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    risk_counts: dict[str, int] = defaultdict(int)
    vehicle_like_clusters = 0
    for row in cluster_rows:
        risk = str(row.get("vehicle_cluster_risk", "none"))
        risk_counts[risk] += 1
        if risk == "vehicle_like":
            vehicle_like_clusters += 1
    audit_rows: list[dict[str, object]] = []
    for score in scores:
        ev = score["evidence"]
        diagnostic = str(score.get("diagnostic_label") or "")
        if not diagnostic and float(ev.get("max_vehicle_like_score", 0.0)) <= 0:
            continue
        audit_rows.append(
            {
                "slot_id": score["slot_id"],
                "state": score["state"],
                "diagnostic_label": diagnostic,
                "evidence_level": score.get("evidence_level", ""),
                "vehicle_like_support_frames": int(ev.get("vehicle_like_support_frames", 0)),
                "stable_vehicle_cluster_frames": int(ev.get("stable_vehicle_cluster_frames", 0)),
                "max_vehicle_like_score": float(ev.get("max_vehicle_like_score", 0.0)),
                "max_vehicle_cluster_core_overlap_ratio": float(ev.get("max_vehicle_cluster_core_overlap_ratio", 0.0)),
                "dominant_vehicle_owner_slot": str(ev.get("dominant_vehicle_owner_slot", "")),
                "dominant_vehicle_owner_ratio": float(ev.get("dominant_vehicle_owner_ratio", 0.0)),
                "vehicle_cluster_conflict_score": float(ev.get("vehicle_cluster_conflict_score", 0.0)),
                "max_vehicle_cluster_length_m": float(ev.get("max_vehicle_cluster_length_m", 0.0)),
                "max_vehicle_cluster_width_m": float(ev.get("max_vehicle_cluster_width_m", 0.0)),
                "max_vehicle_cluster_height_span_m": float(ev.get("max_vehicle_cluster_height_span_m", 0.0)),
                "vehicle_cluster_risk_counts": ev.get("vehicle_cluster_risk_counts", {}),
                "uncertainty_type": score.get("uncertainty_type") or "",
                "reason": score.get("reason", []),
            }
        )
    audit_rows.sort(key=lambda row: (-float(row["max_vehicle_like_score"]), -int(row["vehicle_like_support_frames"]), str(row["slot_id"])))
    label_counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        label_counts[str(row["diagnostic_label"] or "no_diagnostic_label")] += 1
    summary = {
        "cluster_count": len(cluster_rows),
        "vehicle_like_cluster_count": vehicle_like_clusters,
        "cluster_risk_counts": dict(risk_counts),
        "audited_slot_count": len(audit_rows),
        "diagnostic_label_counts": dict(label_counts),
    }
    return summary, audit_rows


def write_vehicle_cluster_audit(output_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    write_json(output_dir / "vehicle_cluster_summary.json", {"summary": summary, "slots": rows})
    fields = [
        "slot_id",
        "state",
        "diagnostic_label",
        "evidence_level",
        "vehicle_like_support_frames",
        "stable_vehicle_cluster_frames",
        "max_vehicle_like_score",
        "max_vehicle_cluster_core_overlap_ratio",
        "dominant_vehicle_owner_slot",
        "dominant_vehicle_owner_ratio",
        "vehicle_cluster_conflict_score",
        "max_vehicle_cluster_length_m",
        "max_vehicle_cluster_width_m",
        "max_vehicle_cluster_height_span_m",
        "vehicle_cluster_risk_counts",
        "uncertainty_type",
        "reason",
    ]
    with (output_dir / "vehicle_cluster_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["vehicle_cluster_risk_counts"] = json.dumps(row.get("vehicle_cluster_risk_counts", {}), ensure_ascii=False)
            out["reason"] = " | ".join(row.get("reason", []))
            writer.writerow(out)


def classify_part2_substate(score: dict[str, object], args: argparse.Namespace) -> str:
    diagnostic = str(score.get("diagnostic_label") or "")
    if diagnostic in {
        "likely_occupied_vehicle_cluster",
        "possible_occupied_vehicle_cluster",
        "vehicle_cluster_boundary_conflict",
        "static_like_cluster",
    }:
        return diagnostic
    evidence = score["evidence"]
    max_obj = float(evidence.get("max_object_hit_cell_ratio", 0.0))
    max_ray = float(evidence.get("max_ray_free_ratio", 0.0))
    visibility_accum = float(evidence.get("visibility_accum", 0.0))
    min_distance = float(evidence.get("min_distance_to_slot", float("inf")))
    conflict = float(evidence.get("conflict_score", 0.0))
    max_boundary = float(evidence.get("max_boundary_ratio", 0.0))
    max_adjacent = float(evidence.get("max_adjacent_overlap_ratio", 0.0))
    core_points = int(evidence.get("max_core_obstacle_point_count", 0))
    margin_points = int(evidence.get("max_margin_obstacle_point_count", 0))
    cluster_counts = evidence.get("cluster_status_counts", {}) or {}
    if score["state"] == "ambiguous" and (max_boundary > 0.60 or int(evidence.get("boundary_conflict_frames", 0)) > 0):
        if max_adjacent > 0.30 or int(cluster_counts.get("adjacent_slot_conflict", 0)) > 0:
            return "adjacent_slot_conflict"
        if margin_points > core_points:
            return "occupied_boundary_conflict"
        if int(cluster_counts.get("slot_vs_lane_possible", 0)) > 0:
            return "slot_vs_lane"
        return "occupied_boundary_conflict"
    possible_free = (
        score["state"] in {"unknown", "ambiguous"}
        and max_obj <= 0.08
        and (max_ray >= 0.20 or visibility_accum >= 0.60)
        and math.isfinite(min_distance)
        and min_distance <= args.max_slot_distance_m * 0.85
        and conflict < 0.60
    )
    if possible_free:
        return "possible_free_unconfirmed"
    return str(score.get("uncertainty_type") or "low_visibility")


def part2_priority(score: dict[str, object], substate: str, args: argparse.Namespace) -> str:
    evidence = score["evidence"]
    max_ray = float(evidence.get("max_ray_free_ratio", 0.0))
    visibility_accum = float(evidence.get("visibility_accum", 0.0))
    min_distance = float(evidence.get("min_distance_to_slot", float("inf")))
    conflict = float(evidence.get("conflict_score", 0.0))
    if substate == "likely_occupied_vehicle_cluster":
        return "high" if float(evidence.get("max_vehicle_like_score", 0.0)) >= 0.75 else "medium"
    if substate in {"possible_occupied_vehicle_cluster", "vehicle_cluster_boundary_conflict"}:
        return "medium"
    if substate == "static_like_cluster":
        return "low"
    if substate == "possible_free_unconfirmed" and max_ray >= 0.35 and visibility_accum >= 1.0 and min_distance <= args.max_slot_distance_m * 0.65 and conflict < 0.35:
        return "high"
    if substate in {"possible_free_unconfirmed", "boundary_conflict", "occupied_boundary_conflict", "adjacent_slot_conflict", "static_vs_vehicle", "slot_vs_lane"}:
        return "medium"
    return "low"


def why_part1_cannot_decide(score: dict[str, object], failed_gates: list[str], substate: str) -> list[str]:
    reasons: list[str] = []
    if substate == "possible_free_unconfirmed":
        reasons.append("slot has some ray-free or visibility evidence but failed strict free gates")
    if substate == "likely_occupied_vehicle_cluster":
        reasons.append("vehicle-like cluster evidence is strong, but Part 1 final occupied remains conservative")
    if substate == "possible_occupied_vehicle_cluster":
        reasons.append("vehicle-like cluster exists but temporal or ownership support is not yet strict")
    if substate == "vehicle_cluster_boundary_conflict":
        reasons.append("vehicle-like cluster crosses edge, margin, or adjacent-slot ownership boundary")
    if substate == "static_like_cluster":
        reasons.append("cluster geometry looks more like static structure than a full vehicle")
    gate_text = {
        "failed_support_frame_count": "not enough supporting free frames",
        "failed_visibility_accum": "accumulated visibility is below strict free threshold",
        "failed_ray_free_area": "ray-free area coverage is insufficient",
        "failed_object_hit": "object evidence is too high for a free decision",
        "failed_distance": "slot is too far or was never observed close enough",
        "failed_conflict": "evidence has boundary or temporal conflict",
    }
    reasons.extend(gate_text[g] for g in failed_gates if g in gate_text)
    if not reasons:
        reasons.extend(score.get("reason", [])[:3])
    return list(dict.fromkeys(reasons))[:5]


def build_part2_candidate_cases(scores: list[dict[str, object]], free_rows: list[dict[str, object]], args: argparse.Namespace, limit: int = 300) -> tuple[dict[str, object], list[dict[str, object]]]:
    failed_by_slot = {str(row["slot_id"]): list(row["failed_free_gates"]) for row in free_rows}
    candidates = [score for score in scores if score["state"] in {"unknown", "ambiguous"} and int(score["evidence"].get("support_frame_count", 0)) > 0]
    cases: list[dict[str, object]] = []
    for score in candidates:
        evidence = score["evidence"]
        substate = classify_part2_substate(score, args)
        failed = failed_by_slot.get(str(score["slot_id"]), [])
        question = {
            "possible_free_unconfirmed": "Camera check: is the slot actually empty, with visible parking lines or arrester and no vehicle?",
            "occupied_boundary_conflict": "Is the non-ground object inside the target slot core, an adjacent slot, or only on the boundary?",
            "adjacent_slot_conflict": "Does the object belong to the target slot or a neighboring slot?",
            "slot_vs_lane": "Is the projected region visually a parking slot or a driving lane?",
            "boundary_conflict": "Do image evidence and neighboring slots resolve whether points belong to this slot?",
            "low_visibility": "Can another view or temporal frame confirm this slot is visible?",
            "occlusion": "Is this slot blocked by a foreground vehicle, wall, or pillar?",
            "static_vs_vehicle": "Are the non-ground returns static structure or a parked vehicle?",
            "likely_occupied_vehicle_cluster": "Does camera or temporal evidence confirm this vehicle-like cluster occupies the target slot?",
            "possible_occupied_vehicle_cluster": "Is this vehicle-like cluster a parked vehicle in the target slot or an adjacent/static object?",
            "vehicle_cluster_boundary_conflict": "Which slot owns the vehicle-like cluster crossing the slot boundary?",
            "static_like_cluster": "Is this rejected cluster a wall, pillar, arrester, or other static structure?",
            "temporal_conflict": "Which frames explain conflicting free and occupied evidence?",
            "pose_boundary_error": "Could map-pose error shift evidence into an adjacent slot or lane?",
        }.get(substate, "What camera evidence would reduce this uncertainty?")
        cases.append(
            {
                "case_id": "",
                "slot_id": score["slot_id"],
                "part1_state": score["state"],
                "part1_substate": substate,
                "priority": part2_priority(score, substate, args),
                "pointcloud_summary": {
                    "support_frame_count": int(evidence.get("support_frame_count", 0)),
                    "visibility_accum": float(evidence.get("visibility_accum", 0.0)),
                    "max_ray_free_area_ratio": float(evidence.get("max_ray_free_ratio", 0.0)),
                    "max_object_hit_cell_ratio": float(evidence.get("max_object_hit_cell_ratio", 0.0)),
                    "min_distance_to_slot_m": float(evidence.get("min_distance_to_slot", 0.0)),
                    "conflict_score": float(evidence.get("conflict_score", 0.0)),
                    "vehicle_like_support_frames": int(evidence.get("vehicle_like_support_frames", 0)),
                    "stable_vehicle_cluster_frames": int(evidence.get("stable_vehicle_cluster_frames", 0)),
                    "max_vehicle_like_score": float(evidence.get("max_vehicle_like_score", 0.0)),
                    "dominant_vehicle_owner_slot": str(evidence.get("dominant_vehicle_owner_slot", "")),
                    "dominant_vehicle_owner_ratio": float(evidence.get("dominant_vehicle_owner_ratio", 0.0)),
                },
                "why_part1_cannot_decide": why_part1_cannot_decide(score, failed, substate),
                "question_for_part2": [question],
                "recommended_camera_tools": [
                    "project_slot_to_camera",
                    "crop_slot_region",
                    "detect_vehicle_in_crop",
                    "inspect_adjacent_slots",
                    "inspect_temporal_camera_frames",
                ],
            }
        )
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    substate_rank = {
        "likely_occupied_vehicle_cluster": 0,
        "possible_occupied_vehicle_cluster": 1,
        "vehicle_cluster_boundary_conflict": 2,
        "possible_free_unconfirmed": 0,
        "occupied_boundary_conflict": 3,
        "adjacent_slot_conflict": 4,
        "boundary_conflict": 5,
        "static_like_cluster": 6,
        "static_vs_vehicle": 7,
        "slot_vs_lane": 8,
        "low_visibility": 9,
    }
    cases.sort(
        key=lambda case: (
            priority_rank.get(str(case["priority"]), 9),
            substate_rank.get(str(case["part1_substate"]), 9),
            -float(case["pointcloud_summary"]["max_ray_free_area_ratio"]),
            -float(case["pointcloud_summary"]["visibility_accum"]),
            float(case["pointcloud_summary"]["min_distance_to_slot_m"]),
        )
    )
    cases = cases[:limit]
    for idx, case in enumerate(cases):
        case["case_id"] = f"case_{idx:05d}"
    substate_counts: dict[str, int] = defaultdict(int)
    priority_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        substate_counts[str(case["part1_substate"])] += 1
        priority_counts[str(case["priority"])] += 1
    summary = {
        "case_count": len(cases),
        "substate_counts": dict(substate_counts),
        "priority_counts": dict(priority_counts),
    }
    return summary, cases


def write_part2_candidate_cases(output_dir: Path, summary: dict[str, object], cases: list[dict[str, object]]) -> None:
    write_json(output_dir / "part2_candidate_cases.json", {"summary": summary, "cases": cases})


def write_scores_csv(path: Path, scores: list[dict[str, object]]) -> None:
    fields = [
        "slot_id",
        "state",
        "p_free",
        "p_occupied",
        "p_unknown",
        "availability_score",
        "visibility_score",
        "confidence",
        "diagnostic_label",
        "evidence_level",
        "support_frame_count",
        "free_support_frames",
        "occupied_support_frames",
        "ambiguous_support_frames",
        "max_ray_free_ratio",
        "max_object_hit_cell_ratio",
        "max_core_hit_cell_ratio",
        "max_edge_hit_cell_ratio",
        "max_margin_hit_cell_ratio",
        "max_height_span",
        "max_core_obstacle_point_count",
        "max_edge_obstacle_point_count",
        "max_margin_obstacle_point_count",
        "max_adjacent_overlap_ratio",
        "max_boundary_ratio",
        "clear_cluster_ownership_frames",
        "boundary_conflict_frames",
        "vehicle_like_support_frames",
        "stable_vehicle_cluster_frames",
        "max_vehicle_like_score",
        "max_vehicle_cluster_core_overlap_ratio",
        "dominant_vehicle_owner_slot",
        "dominant_vehicle_owner_ratio",
        "vehicle_cluster_conflict_score",
        "max_vehicle_cluster_length_m",
        "max_vehicle_cluster_width_m",
        "max_vehicle_cluster_height_span_m",
        "conflict_score",
        "uncertainty_type",
        "reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            ev = score["evidence"]
            writer.writerow(
                {
                    "slot_id": score["slot_id"],
                    "state": score["state"],
                    "p_free": score["p_free"],
                    "p_occupied": score["p_occupied"],
                    "p_unknown": score["p_unknown"],
                    "availability_score": score["availability_score"],
                    "visibility_score": score["visibility_score"],
                    "confidence": score["confidence"],
                    "diagnostic_label": score.get("diagnostic_label", ""),
                    "evidence_level": score.get("evidence_level", ""),
                    "support_frame_count": ev.get("support_frame_count", 0),
                    "free_support_frames": ev.get("free_support_frames", 0),
                    "occupied_support_frames": ev.get("occupied_support_frames", 0),
                    "ambiguous_support_frames": ev.get("ambiguous_support_frames", 0),
                    "max_ray_free_ratio": ev.get("max_ray_free_ratio", 0.0),
                    "max_object_hit_cell_ratio": ev.get("max_object_hit_cell_ratio", 0.0),
                    "max_core_hit_cell_ratio": ev.get("max_core_hit_cell_ratio", 0.0),
                    "max_edge_hit_cell_ratio": ev.get("max_edge_hit_cell_ratio", 0.0),
                    "max_margin_hit_cell_ratio": ev.get("max_margin_hit_cell_ratio", 0.0),
                    "max_height_span": ev.get("max_height_span", 0.0),
                    "max_core_obstacle_point_count": ev.get("max_core_obstacle_point_count", 0),
                    "max_edge_obstacle_point_count": ev.get("max_edge_obstacle_point_count", 0),
                    "max_margin_obstacle_point_count": ev.get("max_margin_obstacle_point_count", 0),
                    "max_adjacent_overlap_ratio": ev.get("max_adjacent_overlap_ratio", 0.0),
                    "max_boundary_ratio": ev.get("max_boundary_ratio", 0.0),
                    "clear_cluster_ownership_frames": ev.get("clear_cluster_ownership_frames", 0),
                    "boundary_conflict_frames": ev.get("boundary_conflict_frames", 0),
                    "vehicle_like_support_frames": ev.get("vehicle_like_support_frames", 0),
                    "stable_vehicle_cluster_frames": ev.get("stable_vehicle_cluster_frames", 0),
                    "max_vehicle_like_score": ev.get("max_vehicle_like_score", 0.0),
                    "max_vehicle_cluster_core_overlap_ratio": ev.get("max_vehicle_cluster_core_overlap_ratio", 0.0),
                    "dominant_vehicle_owner_slot": ev.get("dominant_vehicle_owner_slot", ""),
                    "dominant_vehicle_owner_ratio": ev.get("dominant_vehicle_owner_ratio", 0.0),
                    "vehicle_cluster_conflict_score": ev.get("vehicle_cluster_conflict_score", 0.0),
                    "max_vehicle_cluster_length_m": ev.get("max_vehicle_cluster_length_m", 0.0),
                    "max_vehicle_cluster_width_m": ev.get("max_vehicle_cluster_width_m", 0.0),
                    "max_vehicle_cluster_height_span_m": ev.get("max_vehicle_cluster_height_span_m", 0.0),
                    "conflict_score": ev.get("conflict_score", 0.0),
                    "uncertainty_type": score.get("uncertainty_type") or "",
                    "reason": " | ".join(score.get("reason", [])),
                }
            )


def draw_debug_plot(path: Path, slots: list[Slot], scores_by_id: dict[str, dict[str, object]], sample_ids: Iterable[str]) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=180)
    colors = {"free": "#16a34a", "occupied": "#dc2626", "ambiguous": "#f59e0b", "unknown": "#94a3b8"}
    for slot in slots:
        score = scores_by_id.get(slot.slot_id, {"state": "unknown"})
        alpha = 0.08 if slot.slot_id not in sample_ids else 0.55
        patch = MplPolygon(slot.polygon, closed=True, facecolor=colors.get(str(score["state"]), "#94a3b8"), edgecolor="#334155", alpha=alpha, linewidth=0.25)
        ax.add_patch(patch)
        if slot.slot_id in sample_ids:
            ax.add_patch(MplPolygon(slot.margin_polygon, closed=True, facecolor="none", edgecolor="#64748b", alpha=0.65, linewidth=0.35, linestyle="--"))
            ax.add_patch(MplPolygon(slot.core_polygon, closed=True, facecolor="none", edgecolor="#0f172a", alpha=0.85, linewidth=0.55))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Part 1 slot scoring overview")
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    output_dir: Path,
    slots: list[Slot],
    scores: list[dict[str, object]],
    validation: list[dict[str, object]],
    free_gate_summary: dict[str, object],
    occupied_audit_rows: list[dict[str, object]],
    vehicle_cluster_summary: dict[str, object],
    vehicle_cluster_rows: list[dict[str, object]],
    part2_cases: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    counts = defaultdict(int)
    for score in scores:
        counts[str(score["state"])] += 1
    top_free = sorted([s for s in scores if s["state"] == "free"], key=lambda s: -float(s["availability_score"]))[:20]
    top_occupied = sorted([s for s in scores if s["state"] == "occupied"], key=lambda s: -float(s["p_occupied"]))[:20]
    top_amb = sorted([s for s in scores if s["state"] == "ambiguous"], key=lambda s: -float(s["visibility_score"]))[:20]
    possible_free = [case for case in part2_cases if case["part1_substate"] == "possible_free_unconfirmed"][:20]
    boundary_cases = [case for case in part2_cases if case["part1_substate"] in {"occupied_boundary_conflict", "adjacent_slot_conflict", "boundary_conflict"}][:20]
    top_part2 = part2_cases[:20]
    sample_ids = {str(s["slot_id"]) for s in top_free[:8] + top_occupied[:8] + top_amb[:8]}
    plot_name = ""
    if not args.no_report_plots:
        plot_name = "slot_scoring_overview.png"
        draw_debug_plot(output_dir / plot_name, slots, {str(s["slot_id"]): s for s in scores}, sample_ids)

    def table(rows: list[dict[str, object]]) -> str:
        parts = ["<table><tr><th>slot</th><th>state</th><th>avail</th><th>p_free</th><th>p_occ</th><th>vis</th><th>reason</th></tr>"]
        for row in rows:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row['slot_id']))}</td>"
                f"<td>{html.escape(str(row['state']))}</td>"
                f"<td>{float(row['availability_score']):.3f}</td>"
                f"<td>{float(row['p_free']):.3f}</td>"
                f"<td>{float(row['p_occupied']):.3f}</td>"
                f"<td>{float(row['visibility_score']):.3f}</td>"
                f"<td>{html.escape(' | '.join(row.get('reason', [])[:3]))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    def free_gate_table(summary: dict[str, object]) -> str:
        failures = summary.get("failure_counts", {})
        parts = ["<table><tr><th>free gate</th><th>failed slot count</th></tr>"]
        for key in [
            "failed_support_frame_count",
            "failed_visibility_accum",
            "failed_ray_free_area",
            "failed_object_hit",
            "failed_distance",
            "failed_conflict",
        ]:
            parts.append(f"<tr><td>{html.escape(key)}</td><td>{int(failures.get(key, 0))}</td></tr>")
        parts.append("</table>")
        return "\n".join(parts)

    def occupied_audit_table(rows: list[dict[str, object]]) -> str:
        parts = ["<table><tr><th>slot</th><th>support</th><th>core hit</th><th>height span</th><th>core pts</th><th>edge pts</th><th>margin pts</th><th>boundary</th><th>adjacent</th><th>strongest frame</th><th>risk</th><th>reason</th></tr>"]
        for row in rows[:30]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row['slot_id']))}</td>"
                f"<td>{int(row['occupied_support_frames'])}</td>"
                f"<td>{float(row['core_hit_cell_ratio']):.3f}</td>"
                f"<td>{float(row['max_height_span_m']):.3f}</td>"
                f"<td>{int(row['core_obstacle_point_count'])}</td>"
                f"<td>{int(row['edge_obstacle_point_count'])}</td>"
                f"<td>{int(row['margin_obstacle_point_count'])}</td>"
                f"<td>{float(row['boundary_ratio']):.3f}</td>"
                f"<td>{float(row['adjacent_overlap_ratio']):.3f}</td>"
                f"<td>{html.escape(str(row['strongest_frame_id']))}</td>"
                f"<td>{html.escape(str(row['occupied_risk']))}</td>"
                f"<td>{html.escape(' | '.join(row.get('reason', [])[:3]))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    def vehicle_cluster_table(rows: list[dict[str, object]]) -> str:
        parts = ["<table><tr><th>slot</th><th>state</th><th>label</th><th>level</th><th>support</th><th>stable</th><th>score</th><th>L/W/H</th><th>core overlap</th><th>owner</th><th>owner ratio</th><th>risk counts</th></tr>"]
        for row in rows[:30]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row['slot_id']))}</td>"
                f"<td>{html.escape(str(row['state']))}</td>"
                f"<td>{html.escape(str(row['diagnostic_label']))}</td>"
                f"<td>{html.escape(str(row['evidence_level']))}</td>"
                f"<td>{int(row['vehicle_like_support_frames'])}</td>"
                f"<td>{int(row['stable_vehicle_cluster_frames'])}</td>"
                f"<td>{float(row['max_vehicle_like_score']):.3f}</td>"
                f"<td>{float(row['max_vehicle_cluster_length_m']):.2f}/{float(row['max_vehicle_cluster_width_m']):.2f}/{float(row['max_vehicle_cluster_height_span_m']):.2f}</td>"
                f"<td>{float(row['max_vehicle_cluster_core_overlap_ratio']):.3f}</td>"
                f"<td>{html.escape(str(row['dominant_vehicle_owner_slot']))}</td>"
                f"<td>{float(row['dominant_vehicle_owner_ratio']):.3f}</td>"
                f"<td>{html.escape(json.dumps(row.get('vehicle_cluster_risk_counts', {}), ensure_ascii=False))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    def part2_case_table(rows: list[dict[str, object]]) -> str:
        parts = ["<table><tr><th>case</th><th>slot</th><th>state</th><th>substate</th><th>priority</th><th>support</th><th>vis accum</th><th>ray free</th><th>object hit</th><th>distance</th><th>why</th></tr>"]
        for row in rows:
            summary = row["pointcloud_summary"]
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row['case_id']))}</td>"
                f"<td>{html.escape(str(row['slot_id']))}</td>"
                f"<td>{html.escape(str(row['part1_state']))}</td>"
                f"<td>{html.escape(str(row['part1_substate']))}</td>"
                f"<td>{html.escape(str(row['priority']))}</td>"
                f"<td>{int(summary['support_frame_count'])}</td>"
                f"<td>{float(summary['visibility_accum']):.3f}</td>"
                f"<td>{float(summary['max_ray_free_area_ratio']):.3f}</td>"
                f"<td>{float(summary['max_object_hit_cell_ratio']):.3f}</td>"
                f"<td>{float(summary['min_distance_to_slot_m']):.2f}</td>"
                f"<td>{html.escape(' | '.join(row.get('why_part1_cannot_decide', [])[:3]))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    body = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Part 1 Slot Scoring Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .note {{ padding: 12px; background: #fff7ed; border: 1px solid #fed7aa; margin-bottom: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; }}
  </style>
</head>
<body>
  <h1>Part 1 Slot Scoring Report</h1>
  <div class="note">free means observed free-space evidence, not absence of points.</div>
  <p>Slots: {len(slots)} | free: {counts['free']} | occupied: {counts['occupied']} | ambiguous: {counts['ambiguous']} | unknown: {counts['unknown']} | validator downgrades: {len(validation)}</p>
  <p>Boundary correction: occupied after correction = {counts['occupied']} | core-supported occupied = {sum(1 for row in occupied_audit_rows if row.get('occupied_risk') == 'low_core_supported')} | high-boundary occupied risk = {sum(1 for row in occupied_audit_rows if row.get('occupied_risk') == 'high_boundary_risk')} | boundary-conflict ambiguous cases = {sum(1 for case in part2_cases if case.get('part1_substate') in {'occupied_boundary_conflict', 'adjacent_slot_conflict', 'boundary_conflict'})}</p>
  <p>Vehicle-like cluster diagnostics: clusters = {vehicle_cluster_summary.get('cluster_count', 0)} | vehicle-like clusters = {vehicle_cluster_summary.get('vehicle_like_cluster_count', 0)} | audited slots = {vehicle_cluster_summary.get('audited_slot_count', 0)}</p>
  {f'<img src="{plot_name}" alt="slot scoring overview">' if plot_name else ''}
  <h2>Free Gate Failure Summary</h2>
  {free_gate_table(free_gate_summary)}
  <h2>Occupied Audit</h2>
  <p>debug_bev_path is currently unavailable in V1; use strongest_frame_id to inspect source evidence.</p>
  {occupied_audit_table(occupied_audit_rows)}
  <h2>Vehicle-Like Cluster Audit</h2>
  <pre>{html.escape(json.dumps(vehicle_cluster_summary, indent=2, ensure_ascii=False))}</pre>
  {vehicle_cluster_table(vehicle_cluster_rows)}
  <h2>Possible Free Unconfirmed Candidates</h2>
  {part2_case_table(possible_free)}
  <h2>Top Boundary Conflict Cases</h2>
  {part2_case_table(boundary_cases)}
  <h2>Top Part 2 Cases</h2>
  {part2_case_table(top_part2)}
  <h2>Top Free Candidates</h2>
  {table(top_free)}
  <h2>Top Occupied Candidates</h2>
  {table(top_occupied)}
  <h2>Top Ambiguous Cases</h2>
  {table(top_amb)}
</body>
</html>
"""
    (output_dir / "slot_scoring_report.html").write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    transform = json.loads(args.transform.read_text(encoding="utf-8"))
    scale = float(transform["scale"])
    slots = extract_slots(args.gltf, scale, args.inner_shrink_m, args.core_shrink_m, args.margin_expand_m)
    if not slots:
        raise SystemExit("No parking slots extracted from GLTF")
    write_slot_database(slots, args.output_dir, scale)

    rows = load_frame_rows(args.frames, args.start_frame, args.end_frame, args.frame_step, args.max_frames)
    if not rows:
        raise SystemExit("No frames selected")
    base_dir = args.frames.resolve().parents[2] if args.frames.is_absolute() else Path.cwd()
    centers = np.asarray([slot.center for slot in slots], dtype=np.float64)
    center_tree = cKDTree(centers)

    fusion: dict[str, dict[str, object]] = {}
    stats_path = args.output_dir / "frame_stats.jsonl"
    evidence_path = args.output_dir / "frame_slot_evidence.jsonl"
    cluster_debug_path = args.output_dir / "cluster_ownership_debug.jsonl"
    vehicle_cluster_debug_path = args.output_dir / "vehicle_cluster_debug.jsonl"
    total_evidence = 0
    evidence_records: list[FrameEvidence] = []
    cluster_debug_records: list[dict[str, object]] = []
    total_clusters = 0
    with (
        stats_path.open("w", encoding="utf-8") as stats_handle,
        evidence_path.open("w", encoding="utf-8") as evidence_handle,
        cluster_debug_path.open("w", encoding="utf-8") as cluster_handle,
        vehicle_cluster_debug_path.open("w", encoding="utf-8") as vehicle_cluster_handle,
    ):
        for idx, row in enumerate(rows):
            stats, evidence, cluster_debug_rows = score_frame(row, base_dir, slots, center_tree, scale, args)
            stats_handle.write(json.dumps(stats) + "\n")
            for ev in evidence:
                update_fusion(fusion, ev, args)
                evidence_records.append(ev)
                evidence_handle.write(json.dumps(evidence_to_json(ev)) + "\n")
            for cluster_row in cluster_debug_rows:
                cluster_debug_records.append(cluster_row)
                cluster_handle.write(json.dumps(cluster_row) + "\n")
                vehicle_cluster_handle.write(json.dumps(cluster_row) + "\n")
            total_evidence += len(evidence)
            total_clusters += len(cluster_debug_rows)
            if (idx + 1) % 10 == 0 or idx + 1 == len(rows):
                print(f"[progress] frames={idx + 1}/{len(rows)} evidence={total_evidence} clusters={total_clusters}", flush=True)

    scores = [finalize_slot(slot, fusion.get(slot.slot_id), args) for slot in slots]
    scores, validation = validate_scores(scores, args)
    state_order = {"free": 0, "occupied": 1, "ambiguous": 2, "unknown": 3}
    scores.sort(key=lambda s: (state_order.get(str(s["state"]), 9), -float(s["availability_score"]), str(s["slot_id"])))
    write_json(args.output_dir / "slot_belief_fused.json", {"slots": scores})
    write_scores_csv(args.output_dir / "slot_scores.csv", scores)
    write_json(args.output_dir / "validation_report.json", {"downgrades": validation})
    write_json(args.output_dir / "part2_case_cards.json", {"cases": make_case_cards(scores)})
    free_gate_summary, free_gate_rows = build_free_gate_analysis(scores, args)
    write_free_gate_analysis(args.output_dir, free_gate_summary, free_gate_rows)
    occupied_audit_summary, occupied_audit_rows = build_occupied_audit(scores, evidence_records)
    write_occupied_audit(args.output_dir, occupied_audit_summary, occupied_audit_rows)
    vehicle_cluster_summary, vehicle_cluster_rows = build_vehicle_cluster_audit(scores, cluster_debug_records)
    write_vehicle_cluster_audit(args.output_dir, vehicle_cluster_summary, vehicle_cluster_rows)
    part2_candidate_summary, part2_candidate_cases = build_part2_candidate_cases(scores, free_gate_rows, args)
    write_part2_candidate_cases(args.output_dir, part2_candidate_summary, part2_candidate_cases)
    write_report(args.output_dir, slots, scores, validation, free_gate_summary, occupied_audit_rows, vehicle_cluster_summary, vehicle_cluster_rows, part2_candidate_cases, args)

    counts = defaultdict(int)
    for score in scores:
        counts[str(score["state"])] += 1
    print(f"[done] frames={len(rows)} evidence={total_evidence} free={counts['free']} occupied={counts['occupied']} ambiguous={counts['ambiguous']} unknown={counts['unknown']}")
    print(
        "[diagnostics] "
        f"possible_free_unconfirmed={part2_candidate_summary['substate_counts'].get('possible_free_unconfirmed', 0)} "
        f"part2_cases={part2_candidate_summary['case_count']} "
        f"occupied_audit={occupied_audit_summary['occupied_count']}",
        flush=True,
    )
    print(f"[output] {args.output_dir}")


if __name__ == "__main__":
    main()
