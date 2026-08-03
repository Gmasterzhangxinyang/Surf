#!/usr/bin/env python3
"""Build a fail-closed static-obstacle candidate layer from map-frame LiDAR.

The caller must explicitly declare ``points_xy_unit=map_unit`` and
``points_z_unit=m``.  The tool refuses other or omitted unit contracts: it
does not infer coordinate units from values or filenames.  XYZ must already
be expressed in the explicit ``map_frame_id``; Z remains in that map frame's
vertical datum and is measured in metres.

Promotion is deliberately fail-closed:

* without a trusted dynamic-filter audit, persistent clusters are emitted only
  as ``candidate_obstacles`` and ``static_obstacles`` remains empty;
* mapped regions are marked ``static_obstacle_layer_complete=true`` only when
  both the dynamic audit and an independent coverage audit pass and bind to
  the exact inputs/configuration.

Dynamic-filter audit schema (``parkingagent.dynamic-filter-audit.v1``)::

    {
      "schema_version": "parkingagent.dynamic-filter-audit.v1",
      "status": "passed",
      "map_frame_id": "icpark/map/v1",
      "input_identity": { ... copied from a candidate-layer run ... },
      "method": "independent tracker review",
      "filter_scope_complete": true,
      "dynamic_regions": [
        {"region_id": "vehicle-1", "polygon_map": [[...], ...]},
        {"region_id": "vehicle-2", "polygon_map": [[...], ...],
         "frame_ids": [10, 11]}
      ]
    }

Coverage-audit schema (``parkingagent.coverage-audit.v1``)::

    {
      "schema_version": "parkingagent.coverage-audit.v1",
      "status": "passed",
      "map_frame_id": "icpark/map/v1",
      "input_identity": { ... copied from a candidate-layer run ... },
      "method": "independent trajectory/visibility audit",
      "complete": true,
      "regions": [{"region_id": "mapped-1", "polygon_map": [[...], ...]}]
    }

Run once without audits to obtain the exact ``input_identity``.  Audits must
copy that object byte-for-value; an audit for another frame selection, point
cloud set, slot database, or threshold configuration is not trusted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from parking_slot_box_scoring.geometry import points_in_polygon


OUTPUT_SCHEMA_VERSION = "static-obstacle-map/1.0"
GENERATOR_SCHEMA_VERSION = "parkingagent.lidar-static-layer-builder.v1"
DYNAMIC_AUDIT_SCHEMA_VERSION = "parkingagent.dynamic-filter-audit.v1"
COVERAGE_AUDIT_SCHEMA_VERSION = "parkingagent.coverage-audit.v1"


@dataclass(frozen=True)
class BuildConfig:
    ground_quantile: float = 0.08
    ground_clearance_m: float = 0.30
    max_height_above_ground_m: float = 2.50
    voxel_size_m: float = 0.25
    min_persistent_frames: int = 3
    min_persistence_ratio: float = 0.15
    cluster_neighbor_radius_m: float = 0.45
    min_cluster_voxels: int = 4

    def validate(self) -> None:
        if not 0.0 <= self.ground_quantile <= 0.5:
            raise ValueError("ground_quantile must be in [0, 0.5]")
        if not math.isfinite(self.ground_clearance_m) or self.ground_clearance_m < 0.0:
            raise ValueError("ground_clearance_m must be finite and non-negative")
        if not math.isfinite(self.max_height_above_ground_m) or self.max_height_above_ground_m <= self.ground_clearance_m:
            raise ValueError("max_height_above_ground_m must exceed ground_clearance_m")
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive and finite")
        if self.min_persistent_frames <= 0:
            raise ValueError("min_persistent_frames must be positive")
        if not 0.0 <= self.min_persistence_ratio <= 1.0:
            raise ValueError("min_persistence_ratio must be in [0, 1]")
        if not math.isfinite(self.cluster_neighbor_radius_m) or self.cluster_neighbor_radius_m <= 0.0:
            raise ValueError("cluster_neighbor_radius_m must be positive and finite")
        if self.min_cluster_voxels <= 0:
            raise ValueError("min_cluster_voxels must be positive")


@dataclass(frozen=True)
class FrameInput:
    frame_id: int
    points_path: Path


@dataclass
class VoxelEvidence:
    observed_frame_count: int = 0
    point_count: int = 0
    z_min_m: float = math.inf
    z_max_m: float = -math.inf


@dataclass(frozen=True)
class PolygonEntry:
    polygon: np.ndarray
    frame_ids: frozenset[int] | None = None


class PolygonGridIndex:
    """Small deterministic spatial index for repeated point/polygon queries."""

    def __init__(self, entries: Sequence[PolygonEntry], cell_size: float) -> None:
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("polygon index cell_size must be positive")
        self.entries = list(entries)
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for entry_index, entry in enumerate(self.entries):
            polygon = _validated_polygon(entry.polygon, "indexed polygon")
            lo = np.floor(polygon.min(axis=0) / self.cell_size).astype(np.int64)
            hi = np.floor(polygon.max(axis=0) / self.cell_size).astype(np.int64)
            for cell_x in range(int(lo[0]), int(hi[0]) + 1):
                for cell_y in range(int(lo[1]), int(hi[1]) + 1):
                    self.cells[(cell_x, cell_y)].append(entry_index)

    def mask(self, points_xy: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        points_xy = np.asarray(points_xy, dtype=np.float64)
        result = np.zeros(len(points_xy), dtype=bool)
        if not self.entries or len(points_xy) == 0:
            return result
        point_cells = np.floor(points_xy / self.cell_size).astype(np.int64)
        points_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        for point_index, cell in enumerate(point_cells):
            points_by_cell[(int(cell[0]), int(cell[1]))].append(point_index)
        boundary_tolerance = max(1e-10, self.cell_size * 1e-9)
        for cell, point_indices in points_by_cell.items():
            candidates = self.cells.get(cell)
            if not candidates:
                continue
            indices = np.asarray(point_indices, dtype=np.int64)
            cell_points = points_xy[indices]
            selected = np.zeros(len(indices), dtype=bool)
            for entry_index in candidates:
                entry = self.entries[entry_index]
                if frame_id is not None and entry.frame_ids is not None and frame_id not in entry.frame_ids:
                    continue
                selected |= _points_in_or_on_polygon(cell_points, entry.polygon, boundary_tolerance)
            result[indices] = selected
        return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_polygon(value: Any, label: str) -> np.ndarray:
    polygon = np.asarray(value, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError(f"{label} must contain at least three XY vertices")
    if not np.isfinite(polygon).all():
        raise ValueError(f"{label} contains non-finite coordinates")
    x = polygon[:, 0]
    y = polygon[:, 1]
    area = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5
    if area <= 1e-12:
        raise ValueError(f"{label} has zero area")
    return polygon


def _points_in_or_on_polygon(points_xy: np.ndarray, polygon: np.ndarray, tolerance: float) -> np.ndarray:
    inside = points_in_polygon(points_xy, polygon)
    if len(points_xy) == 0:
        return inside
    tolerance_sq = tolerance * tolerance
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-24:
            distance_sq = np.sum((points_xy - start) ** 2, axis=1)
        else:
            projection = np.clip(((points_xy - start) @ segment) / denominator, 0.0, 1.0)
            closest = start + projection[:, None] * segment
            distance_sq = np.sum((points_xy - closest) ** 2, axis=1)
        inside |= distance_sq <= tolerance_sq
    return inside


def _resolve_points_path(raw: str, manifest_path: Path) -> Path:
    candidate = Path(raw)
    candidates = [candidate] if candidate.is_absolute() else [
        manifest_path.parent / candidate,
        PROJECT_ROOT / candidate,
        Path.cwd() / candidate,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"map point cloud does not exist; tried: {rendered}")


def _load_frames(
    frames_path: Path,
    start_frame: int | None,
    end_frame: int | None,
    frame_step: int,
    max_frames: int,
) -> list[FrameInput]:
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    rows: list[tuple[int, str]] = []
    seen: set[int] = set()
    with frames_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "frame" not in reader.fieldnames or "map_points_path" not in reader.fieldnames:
            raise ValueError("frames CSV must contain frame and map_points_path columns")
        for row in reader:
            try:
                frame_id = int(row.get("frame", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError("frames CSV contains an invalid frame ID") from exc
            if frame_id in seen:
                raise ValueError(f"duplicate frame ID {frame_id}")
            seen.add(frame_id)
            raw_path = str(row.get("map_points_path", "")).strip()
            if not raw_path:
                raise ValueError(f"frame {frame_id} is missing map_points_path")
            if start_frame is not None and frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                continue
            rows.append((frame_id, raw_path))
    rows.sort(key=lambda item: item[0])
    rows = rows[::frame_step]
    if max_frames:
        rows = rows[:max_frames]
    if not rows:
        raise ValueError("frame selection is empty")
    return [FrameInput(frame_id, _resolve_points_path(raw_path, frames_path)) for frame_id, raw_path in rows]


def _load_slot_database(path: Path, map_frame_id: str) -> tuple[list[np.ndarray], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("slots"), list):
        raise ValueError("slot database must be an object containing a slots list")
    declared_map_frame = payload.get("map_frame_id")
    if declared_map_frame is not None and str(declared_map_frame) != map_frame_id:
        raise ValueError("slot database map_frame_id does not match the explicit map_frame_id")
    scale = float(payload.get("map_units_per_meter", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_units_per_meter must be positive and finite")
    raw_slots = payload["slots"]
    if payload.get("slot_count") is not None and int(payload["slot_count"]) != len(raw_slots):
        raise ValueError("slot_count does not match slots list")
    polygons: list[np.ndarray] = []
    for index, slot in enumerate(raw_slots):
        if not isinstance(slot, dict) or "polygon_map" not in slot:
            raise ValueError(f"slot {index} is missing polygon_map")
        polygons.append(_validated_polygon(slot["polygon_map"], f"slot {index} polygon_map"))
    return polygons, scale


def _read_audit(path: Path | None, expected_schema: str) -> tuple[dict[str, Any] | None, list[str], str | None]:
    if path is None:
        return None, ["audit_not_provided"], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"audit_unreadable:{type(exc).__name__}"], None
    if not isinstance(payload, dict):
        return None, ["audit_root_is_not_an_object"], sha256_file(path)
    reasons: list[str] = []
    if payload.get("schema_version") != expected_schema:
        reasons.append("schema_version_mismatch")
    return payload, reasons, sha256_file(path)


def _parse_dynamic_regions(payload: Mapping[str, Any] | None, selected_frame_ids: set[int]) -> tuple[list[PolygonEntry], list[str]]:
    if payload is None:
        return [], []
    raw_regions = payload.get("dynamic_regions", [])
    if not isinstance(raw_regions, list):
        return [], ["dynamic_regions_is_not_a_list"]
    entries: list[PolygonEntry] = []
    reasons: list[str] = []
    for index, region in enumerate(raw_regions):
        try:
            if not isinstance(region, dict):
                raise ValueError("region is not an object")
            polygon = _validated_polygon(region.get("polygon_map"), f"dynamic region {index}")
            raw_frame_ids = region.get("frame_ids")
            if raw_frame_ids is None:
                frame_ids = None
            elif not isinstance(raw_frame_ids, list) or not raw_frame_ids:
                raise ValueError("frame_ids must be a non-empty list when present")
            else:
                frame_ids = frozenset(int(value) for value in raw_frame_ids)
                if not frame_ids.issubset(selected_frame_ids):
                    raise ValueError("frame_ids are outside the selected frame set")
            entries.append(PolygonEntry(polygon=polygon, frame_ids=frame_ids))
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid_dynamic_region_{index}:{exc}")
    return entries, reasons


def _validate_common_audit(
    payload: Mapping[str, Any] | None,
    initial_reasons: Sequence[str],
    map_frame_id: str,
    input_identity: Mapping[str, str],
) -> list[str]:
    reasons = list(initial_reasons)
    if payload is None:
        return reasons
    if payload.get("status") != "passed":
        reasons.append("status_is_not_passed")
    if payload.get("map_frame_id") != map_frame_id:
        reasons.append("map_frame_id_mismatch")
    if payload.get("input_identity") != dict(input_identity):
        reasons.append("input_identity_mismatch")
    method = payload.get("method")
    if not isinstance(method, str) or not method.strip():
        reasons.append("audit_method_missing")
    return reasons


def _load_points(path: Path, frame_id: int) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "points_map_xyzi" not in payload.files:
                raise ValueError("missing points_map_xyzi")
            points = np.asarray(payload["points_map_xyzi"], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"cannot load frame {frame_id} point cloud {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"frame {frame_id} points_map_xyzi must have at least three columns")
    return points[:, :4] if points.shape[1] >= 4 else points[:, :3]


def _update_voxels(
    evidence: dict[tuple[int, int], VoxelEvidence],
    points: np.ndarray,
    voxel_size_map: float,
) -> None:
    if len(points) == 0:
        return
    keys = np.floor(points[:, :2] / voxel_size_map).astype(np.int64)
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    z_min = np.full(len(unique_keys), math.inf, dtype=np.float64)
    z_max = np.full(len(unique_keys), -math.inf, dtype=np.float64)
    np.minimum.at(z_min, inverse, points[:, 2])
    np.maximum.at(z_max, inverse, points[:, 2])
    for index, raw_key in enumerate(unique_keys):
        key = (int(raw_key[0]), int(raw_key[1]))
        item = evidence.setdefault(key, VoxelEvidence())
        item.observed_frame_count += 1
        item.point_count += int(counts[index])
        item.z_min_m = min(item.z_min_m, float(z_min[index]))
        item.z_max_m = max(item.z_max_m, float(z_max[index]))


def _persistent_voxels(
    evidence: Mapping[tuple[int, int], VoxelEvidence],
    frame_count: int,
    config: BuildConfig,
) -> dict[tuple[int, int], VoxelEvidence]:
    return {
        key: item
        for key, item in evidence.items()
        if item.observed_frame_count >= config.min_persistent_frames
        and item.observed_frame_count / frame_count >= config.min_persistence_ratio
    }


def _neighbor_offsets(voxel_size_m: float, radius_m: float) -> list[tuple[int, int]]:
    reach = max(1, int(math.ceil(radius_m / voxel_size_m)))
    offsets: list[tuple[int, int]] = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx == 0 and dy == 0:
                continue
            if math.hypot(dx * voxel_size_m, dy * voxel_size_m) <= radius_m + 1e-12:
                offsets.append((dx, dy))
    return sorted(offsets)


def _connected_components(
    voxels: Mapping[tuple[int, int], VoxelEvidence],
    offsets: Sequence[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    remaining = set(voxels)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue: deque[tuple[int, int]] = deque([start])
        component: list[tuple[int, int]] = []
        while queue:
            key = queue.popleft()
            component.append(key)
            for dx, dy in offsets:
                neighbor = (key[0] + dx, key[1] + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _convex_hull(points: Iterable[tuple[float, float]]) -> list[list[float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return [[float(x), float(y)] for x, y in unique]

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [[round(float(x), 9), round(float(y), 9)] for x, y in lower[:-1] + upper[:-1]]


def _obstacles_from_voxels(
    voxels: Mapping[tuple[int, int], VoxelEvidence],
    frame_count: int,
    map_units_per_meter: float,
    config: BuildConfig,
    map_frame_id: str,
    kind: str,
    source_id: str,
) -> list[dict[str, Any]]:
    offsets = _neighbor_offsets(config.voxel_size_m, config.cluster_neighbor_radius_m)
    voxel_size_map = config.voxel_size_m * map_units_per_meter
    obstacles: list[dict[str, Any]] = []
    for component in _connected_components(voxels, offsets):
        if len(component) < config.min_cluster_voxels:
            continue
        cell_corners: list[tuple[float, float]] = []
        for cell_x, cell_y in component:
            x0 = cell_x * voxel_size_map
            y0 = cell_y * voxel_size_map
            cell_corners.extend(
                [
                    (x0, y0),
                    (x0 + voxel_size_map, y0),
                    (x0 + voxel_size_map, y0 + voxel_size_map),
                    (x0, y0 + voxel_size_map),
                ]
            )
        items = [voxels[key] for key in component]
        observed_frame_count = max(item.observed_frame_count for item in items)
        persistence_ratio = observed_frame_count / frame_count
        weighted_x = sum((key[0] + 0.5) * voxel_size_map * voxels[key].point_count for key in component)
        weighted_y = sum((key[1] + 0.5) * voxel_size_map * voxels[key].point_count for key in component)
        point_count = sum(item.point_count for item in items)
        support_score = min(1.0, observed_frame_count / max(config.min_persistent_frames * 2, 1))
        size_score = min(1.0, len(component) / max(config.min_cluster_voxels * 2, 1))
        confidence = float(np.clip(0.55 * persistence_ratio + 0.25 * support_score + 0.20 * size_score, 0.0, 1.0))
        component_digest = _canonical_sha256(component)[:16]
        object_id = f"{kind}_{component_digest}"
        polygon_map = _convex_hull(cell_corners)
        observed_height_min_m = min(item.z_min_m for item in items)
        observed_height_max_m = max(item.z_max_m for item in items)
        obstacles.append(
            {
                "id": object_id,
                "object_id": object_id,
                "obstacle_id": object_id,
                # LiDAR geometry alone cannot safely distinguish a wall from
                # a column, so use the standard conservative class.
                "type": "other_static",
                "object_type": "other_static",
                "coordinate_frame": map_frame_id,
                "map_frame_id": map_frame_id,
                "polygon_xy": polygon_map,
                "polygon_map": polygon_map,
                "centroid_map": [round(weighted_x / point_count, 9), round(weighted_y / point_count, 9)],
                # Preserve the input map frame's vertical datum.  Ground
                # relative height is used only for filtering and must never
                # silently replace map Z in the published obstacle layer.
                "min_z": round(observed_height_min_m, 6),
                "max_z": round(observed_height_max_m, 6),
                "min_height_m": round(observed_height_min_m, 6),
                "max_height_m": round(observed_height_max_m, 6),
                "observed_height_min_m": round(observed_height_min_m, 6),
                "observed_height_max_m": round(observed_height_max_m, 6),
                "voxel_count": len(component),
                "point_observation_count": point_count,
                "observed_frame_count": observed_frame_count,
                "persistence_ratio": round(persistence_ratio, 6),
                "confidence": round(confidence, 6),
                "confidence_semantics": "heuristic_score_not_probability",
                "source": "lidar_mapping",
                "source_id": source_id,
                "review_required": kind == "candidate",
            }
        )
    return sorted(obstacles, key=lambda item: item["object_id"])


def _coverage_regions(payload: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    if payload is None:
        return [], []
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        return [], ["coverage_regions_missing_or_empty"]
    regions: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen_ids: set[str] = set()
    for index, region in enumerate(raw_regions):
        try:
            if not isinstance(region, dict):
                raise ValueError("region is not an object")
            region_id = str(region.get("region_id", "")).strip()
            if not region_id or region_id in seen_ids:
                raise ValueError("region_id is missing or duplicated")
            seen_ids.add(region_id)
            polygon = _validated_polygon(region.get("polygon_map"), f"coverage region {index}")
            regions.append({"region_id": region_id, "polygon_map": polygon.tolist()})
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid_coverage_region_{index}:{exc}")
    return regions, reasons


def build_static_obstacle_layer(
    frames_path: str | Path,
    slot_database_path: str | Path,
    map_frame_id: str,
    *,
    points_xy_unit: str,
    points_z_unit: str,
    output_path: str | Path | None = None,
    dynamic_filter_audit_path: str | Path | None = None,
    coverage_audit_path: str | Path | None = None,
    config: BuildConfig | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_step: int = 1,
    max_frames: int = 0,
) -> dict[str, Any]:
    config = config or BuildConfig()
    config.validate()
    map_frame_id = str(map_frame_id).strip()
    if not map_frame_id:
        raise ValueError("map_frame_id must be explicit and non-empty")
    if points_xy_unit != "map_unit":
        raise ValueError("points_xy_unit must be explicitly declared as 'map_unit'")
    if points_z_unit != "m":
        raise ValueError("points_z_unit must be explicitly declared as 'm'")
    frames_path = Path(frames_path).resolve()
    slot_database_path = Path(slot_database_path).resolve()
    dynamic_path = Path(dynamic_filter_audit_path).resolve() if dynamic_filter_audit_path else None
    coverage_path = Path(coverage_audit_path).resolve() if coverage_audit_path else None

    frames = _load_frames(frames_path, start_frame, end_frame, frame_step, max_frames)
    selected_frame_ids = [frame.frame_id for frame in frames]
    selected_frame_id_set = set(selected_frame_ids)
    slot_polygons, map_units_per_meter = _load_slot_database(slot_database_path, map_frame_id)
    slot_index = PolygonGridIndex(
        [PolygonEntry(polygon) for polygon in slot_polygons],
        cell_size=max(map_units_per_meter * 4.0, 1e-6),
    )

    dynamic_payload, dynamic_initial_reasons, dynamic_audit_sha256 = _read_audit(
        dynamic_path, DYNAMIC_AUDIT_SCHEMA_VERSION
    )
    dynamic_entries, dynamic_region_reasons = _parse_dynamic_regions(dynamic_payload, selected_frame_id_set)
    dynamic_index = PolygonGridIndex(
        dynamic_entries,
        cell_size=max(map_units_per_meter * 4.0, 1e-6),
    )

    raw_evidence: dict[tuple[int, int], VoxelEvidence] = {}
    dynamic_filtered_evidence: dict[tuple[int, int], VoxelEvidence] = {}
    voxel_size_map = config.voxel_size_m * map_units_per_meter
    point_cloud_digest = hashlib.sha256()
    processing_frames: list[dict[str, Any]] = []
    total_points = 0
    total_nonfinite = 0
    total_ground_removed = 0
    total_slot_removed = 0
    total_dynamic_removed = 0

    for frame in frames:
        cloud_sha256 = sha256_file(frame.points_path)
        point_cloud_digest.update(f"{frame.frame_id}:".encode("utf-8"))
        point_cloud_digest.update(cloud_sha256.encode("ascii"))
        points = _load_points(frame.points_path, frame.frame_id)
        total_points += len(points)
        finite = np.isfinite(points[:, :3]).all(axis=1)
        total_nonfinite += int((~finite).sum())
        points = points[finite]
        ground_z = float(np.quantile(points[:, 2], config.ground_quantile)) if len(points) else 0.0
        height = points[:, 2] - ground_z
        obstacle_mask = (height >= config.ground_clearance_m) & (height <= config.max_height_above_ground_m)
        ground_removed = int((~obstacle_mask).sum())
        total_ground_removed += ground_removed
        # Filtering uses height above the per-frame ground estimate, while
        # retained XYZ stays in the caller-declared map coordinate frame.
        obstacles = points[obstacle_mask].copy()
        slot_mask = slot_index.mask(obstacles[:, :2], frame.frame_id)
        slot_removed = int(slot_mask.sum())
        total_slot_removed += slot_removed
        obstacles = obstacles[~slot_mask]
        _update_voxels(raw_evidence, obstacles, voxel_size_map)

        dynamic_mask = dynamic_index.mask(obstacles[:, :2], frame.frame_id)
        dynamic_removed = int(dynamic_mask.sum())
        total_dynamic_removed += dynamic_removed
        _update_voxels(dynamic_filtered_evidence, obstacles[~dynamic_mask], voxel_size_map)
        processing_frames.append(
            {
                "frame_id": frame.frame_id,
                "point_count": int(len(points)),
                "ground_z_m": round(ground_z, 6),
                "ground_or_height_removed": ground_removed,
                "slot_polygon_removed": slot_removed,
                "dynamic_region_removed": dynamic_removed,
            }
        )

    input_identity = {
        "frames_sha256": sha256_file(frames_path),
        "slot_database_sha256": sha256_file(slot_database_path),
        "selected_frame_ids_sha256": _canonical_sha256(selected_frame_ids),
        "point_clouds_sha256": point_cloud_digest.hexdigest(),
        "build_config_sha256": _canonical_sha256(asdict(config)),
        "points_coordinate_frame": map_frame_id,
        "points_xy_unit": points_xy_unit,
        "points_z_unit": points_z_unit,
    }

    dynamic_reasons = _validate_common_audit(
        dynamic_payload,
        [*dynamic_initial_reasons, *dynamic_region_reasons],
        map_frame_id,
        input_identity,
    )
    if dynamic_payload is not None and dynamic_payload.get("filter_scope_complete") is not True:
        dynamic_reasons.append("filter_scope_complete_is_not_true")
    dynamic_reasons = sorted(set(dynamic_reasons))
    dynamic_trusted = dynamic_payload is not None and not dynamic_reasons

    coverage_payload, coverage_initial_reasons, coverage_audit_sha256 = _read_audit(
        coverage_path, COVERAGE_AUDIT_SCHEMA_VERSION
    )
    regions, region_reasons = _coverage_regions(coverage_payload)
    coverage_reasons = _validate_common_audit(
        coverage_payload,
        [*coverage_initial_reasons, *region_reasons],
        map_frame_id,
        input_identity,
    )
    if coverage_payload is not None and coverage_payload.get("complete") is not True:
        coverage_reasons.append("coverage_complete_is_not_true")
    coverage_reasons = sorted(set(coverage_reasons))
    coverage_trusted = coverage_payload is not None and not coverage_reasons

    raw_persistent = _persistent_voxels(raw_evidence, len(frames), config)
    filtered_persistent = _persistent_voxels(dynamic_filtered_evidence, len(frames), config)
    lidar_source_id = f"point-cloud-set:{input_identity['point_clouds_sha256']}"
    if dynamic_trusted:
        static_obstacles = _obstacles_from_voxels(
            filtered_persistent,
            len(frames),
            map_units_per_meter,
            config,
            map_frame_id,
            "static",
            lidar_source_id,
        )
        candidate_obstacles: list[dict[str, Any]] = []
    else:
        static_obstacles = []
        candidate_obstacles = _obstacles_from_voxels(
            raw_persistent,
            len(frames),
            map_units_per_meter,
            config,
            map_frame_id,
            "candidate",
            lidar_source_id,
        )

    mapped_complete = dynamic_trusted and coverage_trusted
    mapped_reasons: list[str] = []
    if not dynamic_trusted:
        mapped_reasons.append("dynamic_filter_audit_not_trusted")
    if not coverage_trusted:
        mapped_reasons.append("coverage_audit_not_trusted")
    mapped_regions = [
        {
            "region_id": region["region_id"],
            "coordinate_frame": map_frame_id,
            "polygon_xy": region["polygon_map"],
            "polygon_map": region["polygon_map"],
            "static_obstacle_layer_complete": mapped_complete,
            "coverage_status": "audited_complete" if mapped_complete else "partial",
        }
        for region in (regions if coverage_trusted else [])
    ]
    processing_payload = {
        "frame_count": len(frames),
        "input_point_count": total_points,
        "nonfinite_point_count": total_nonfinite,
        "ground_or_height_removed": total_ground_removed,
        "slot_polygon_removed": total_slot_removed,
        "dynamic_region_removed": total_dynamic_removed,
        "raw_voxel_count": len(raw_evidence),
        "raw_persistent_voxel_count": len(raw_persistent),
        "dynamic_filtered_persistent_voxel_count": len(filtered_persistent),
        "frames": processing_frames,
    }
    audits_payload = {
        "dynamic_filter": {
            "provided": dynamic_path is not None,
            "trusted": dynamic_trusted,
            "audit_sha256": dynamic_audit_sha256,
            "reasons": dynamic_reasons,
        },
        "coverage": {
            "provided": coverage_path is not None,
            "trusted": coverage_trusted,
            "audit_sha256": coverage_audit_sha256,
            "reasons": coverage_reasons,
        },
    }
    review_payload = {
        "required": not dynamic_trusted,
        "reasons": [] if dynamic_trusted else ["dynamic_filter_audit_not_trusted"],
        "candidate_obstacle_ids": [item["object_id"] for item in candidate_obstacles],
    }
    result: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "coordinate_frame": map_frame_id,
        "map_frame_id": map_frame_id,
        "map_units_per_meter": map_units_per_meter,
        "static_obstacle_layer_present": True,
        "input_identity": input_identity,
        "build_config": asdict(config),
        "source": {
            "frames_path": str(frames_path),
            "slot_database_path": str(slot_database_path),
            "selected_frame_ids": selected_frame_ids,
        },
        "processing": processing_payload,
        "audits": audits_payload,
        "static_obstacles": static_obstacles,
        "candidate_obstacles": candidate_obstacles,
        "review_required": review_payload,
        "mapped_regions": mapped_regions,
        "mapped_regions_complete": mapped_complete,
        "mapped_regions_summary": {
            "complete": mapped_complete,
            "reasons": mapped_reasons,
            "coverage_audit_sha256": coverage_audit_sha256 if coverage_trusted else None,
        },
        "metadata": {
            "generator": "scripts/build_static_obstacle_layer_from_lidar.py",
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "point_contract": {
                "array_key": "points_map_xyzi",
                "coordinate_frame": map_frame_id,
                "xy_unit": points_xy_unit,
                "z_unit": points_z_unit,
            },
            "input_identity": input_identity,
            "build_config": asdict(config),
            "audits": audits_payload,
            "review_required": review_payload,
            "mapped_regions_complete": mapped_complete,
            "mapped_regions_reasons": mapped_reasons,
            "processing": processing_payload,
        },
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="frames.csv containing map_points_path")
    parser.add_argument("--slot-database", type=Path, required=True)
    parser.add_argument("--map-frame-id", required=True, help="explicit, stable map-frame identity")
    parser.add_argument("--points-xy-unit", required=True, choices=("map_unit",))
    parser.add_argument("--points-z-unit", required=True, choices=("m",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamic-filter-audit", type=Path)
    parser.add_argument("--coverage-audit", type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ground-quantile", type=float, default=BuildConfig.ground_quantile)
    parser.add_argument("--ground-clearance-m", type=float, default=BuildConfig.ground_clearance_m)
    parser.add_argument("--max-height-above-ground-m", type=float, default=BuildConfig.max_height_above_ground_m)
    parser.add_argument("--voxel-size-m", type=float, default=BuildConfig.voxel_size_m)
    parser.add_argument("--min-persistent-frames", type=int, default=BuildConfig.min_persistent_frames)
    parser.add_argument("--min-persistence-ratio", type=float, default=BuildConfig.min_persistence_ratio)
    parser.add_argument("--cluster-neighbor-radius-m", type=float, default=BuildConfig.cluster_neighbor_radius_m)
    parser.add_argument("--min-cluster-voxels", type=int, default=BuildConfig.min_cluster_voxels)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        ground_quantile=args.ground_quantile,
        ground_clearance_m=args.ground_clearance_m,
        max_height_above_ground_m=args.max_height_above_ground_m,
        voxel_size_m=args.voxel_size_m,
        min_persistent_frames=args.min_persistent_frames,
        min_persistence_ratio=args.min_persistence_ratio,
        cluster_neighbor_radius_m=args.cluster_neighbor_radius_m,
        min_cluster_voxels=args.min_cluster_voxels,
    )
    result = build_static_obstacle_layer(
        args.frames,
        args.slot_database,
        args.map_frame_id,
        points_xy_unit=args.points_xy_unit,
        points_z_unit=args.points_z_unit,
        output_path=args.output,
        dynamic_filter_audit_path=args.dynamic_filter_audit,
        coverage_audit_path=args.coverage_audit,
        config=config,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "static_obstacle_count": len(result["static_obstacles"]),
                "candidate_obstacle_count": len(result["candidate_obstacles"]),
                "review_required": result["review_required"]["required"],
                "mapped_regions_complete": result["mapped_regions_complete"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
