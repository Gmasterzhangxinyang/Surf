"""Validated static-obstacle layers for geometry-only Camera preflight.

The current Part1 slot database contains slot polygons and a horizontal map
scale, but it does not contain a complete wall/pillar layer.  This module
therefore makes coverage an explicit data contract: an empty obstacle list is
usable only where a ``mapped_regions`` polygon explicitly declares complete
coverage.  Missing data and an audited empty layer are never conflated.

No point-cloud clustering or semantic inference is performed here.  In
particular, dynamic vehicles are rejected from the static layer.  A small
optional GLTF adapter accepts only explicitly named structural mesh layers;
it deliberately does not declare those layers spatially complete.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from parking_slot_box_scoring.geometry import points_in_polygon


STATIC_OBSTACLE_MAP_SCHEMA_VERSION = "static-obstacle-map/1.0"

Polygon = tuple[tuple[float, float], ...]

_DYNAMIC_OBJECT_TYPES = frozenset(
    {
        "vehicle",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "dynamic_vehicle",
        "moving_object",
    }
)
_SUPPORTED_STATIC_OBJECT_TYPES = frozenset(
    {
        "wall",
        "pillar",
        "column",
        "barrier",
        "bollard",
        "curb",
        "arrester",
        "wheel_stop",
        "elevator",
        "building",
        "fence",
        "gate",
        "structure",
        "static_obstacle",
        "other_static",
    }
)
_COMPLETE_COVERAGE_STATUSES = frozenset({"complete", "audited_complete"})
_EPSILON = 1e-10
_SUPPORTED_SOURCES = frozenset({"manual", "cad", "lidar_mapping"})


class StaticObstacleMapValidationError(ValueError):
    """A fail-closed schema or geometry validation error."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _fail(reason_code: str, message: str) -> None:
    raise StaticObstacleMapValidationError(reason_code, message)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name}_missing", f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name}_invalid", f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name}_invalid", f"{name} must be a finite number")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        _fail(f"{name}_invalid", f"{name} must be positive")
    return result


def _point(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        _fail(f"{name}_invalid", f"{name} must contain x and y")
    return (_finite(value[0], name), _finite(value[1], name))


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if abs(_cross(start, end, point)) > _EPSILON:
        return False
    return (
        min(start[0], end[0]) - _EPSILON
        <= point[0]
        <= max(start[0], end[0]) + _EPSILON
        and min(start[1], end[1]) - _EPSILON
        <= point[1]
        <= max(start[1], end[1]) + _EPSILON
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    c1 = _cross(first_start, first_end, second_start)
    c2 = _cross(first_start, first_end, second_end)
    c3 = _cross(second_start, second_end, first_start)
    c4 = _cross(second_start, second_end, first_end)
    if ((c1 > _EPSILON and c2 < -_EPSILON) or (c1 < -_EPSILON and c2 > _EPSILON)) and (
        (c3 > _EPSILON and c4 < -_EPSILON) or (c3 < -_EPSILON and c4 > _EPSILON)
    ):
        return True
    return (
        (abs(c1) <= _EPSILON and _point_on_segment(second_start, first_start, first_end))
        or (abs(c2) <= _EPSILON and _point_on_segment(second_end, first_start, first_end))
        or (abs(c3) <= _EPSILON and _point_on_segment(first_start, second_start, second_end))
        or (abs(c4) <= _EPSILON and _point_on_segment(first_end, second_start, second_end))
    )


def _segments_properly_cross(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    c1 = _cross(first_start, first_end, second_start)
    c2 = _cross(first_start, first_end, second_end)
    c3 = _cross(second_start, second_end, first_start)
    c4 = _cross(second_start, second_end, first_end)
    return (
        ((c1 > _EPSILON and c2 < -_EPSILON) or (c1 < -_EPSILON and c2 > _EPSILON))
        and ((c3 > _EPSILON and c4 < -_EPSILON) or (c3 < -_EPSILON and c4 > _EPSILON))
    )


def _polygon_area(polygon: Polygon) -> float:
    return 0.5 * sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon, polygon[1:] + polygon[:1])
    )


def _polygon_bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _grid_cells_for_bounds(
    bounds: tuple[float, float, float, float],
    cell_size_map: float,
) -> tuple[tuple[int, int], ...]:
    min_x, min_y, max_x, max_y = bounds
    first_x = math.floor(min_x / cell_size_map)
    last_x = math.floor(max_x / cell_size_map)
    first_y = math.floor(min_y / cell_size_map)
    last_y = math.floor(max_y / cell_size_map)
    return tuple(
        (x_index, y_index)
        for x_index in range(first_x, last_x + 1)
        for y_index in range(first_y, last_y + 1)
    )


def _validate_simple_polygon(value: Any, name: str) -> Polygon:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("polygon_invalid", f"{name} must be an array of [x, y] vertices")
    vertices = [_point(raw, name) for raw in value]
    if len(vertices) >= 2 and vertices[0] == vertices[-1]:
        vertices.pop()
    if len(vertices) < 3:
        _fail("polygon_invalid", f"{name} must have at least three vertices")
    if len(set(vertices)) != len(vertices):
        _fail("polygon_invalid", f"{name} contains duplicate vertices")
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        if math.dist(start, end) <= _EPSILON:
            _fail("polygon_invalid", f"{name} contains a zero-length edge")
    polygon = tuple(vertices)
    if abs(_polygon_area(polygon)) <= _EPSILON:
        _fail("polygon_invalid", f"{name} has zero area")
    edge_count = len(polygon)
    for first in range(edge_count):
        first_end = (first + 1) % edge_count
        for second in range(first + 1, edge_count):
            second_end = (second + 1) % edge_count
            if first == second or first_end == second or second_end == first:
                continue
            if _segments_intersect(
                polygon[first],
                polygon[first_end],
                polygon[second],
                polygon[second_end],
            ):
                _fail("polygon_self_intersection", f"{name} is self-intersecting")
    return polygon


def _point_in_or_on_polygon(point: tuple[float, float], polygon: Polygon) -> bool:
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        if _point_on_segment(point, start, end):
            return True
    inside = points_in_polygon(
        np.asarray([point], dtype=np.float64),
        np.asarray(polygon, dtype=np.float64),
    )
    return bool(inside[0])


def _polygons_intersect(first: Polygon, second: Polygon) -> bool:
    if any(_point_in_or_on_polygon(point, second) for point in first):
        return True
    if any(_point_in_or_on_polygon(point, first) for point in second):
        return True
    return any(
        _segments_intersect(first_start, first_end, second_start, second_end)
        for first_start, first_end in zip(first, first[1:] + first[:1])
        for second_start, second_end in zip(second, second[1:] + second[:1])
    )


def _polygon_contains_polygon(container: Polygon, subject: Polygon) -> bool:
    if not all(_point_in_or_on_polygon(point, container) for point in subject):
        return False
    # For a concave mapped region, corners alone are insufficient.  A proper
    # boundary crossing proves that part of the corridor leaves the region;
    # boundary touches are accepted as covered.
    return not any(
        _segments_properly_cross(subject_start, subject_end, region_start, region_end)
        for subject_start, subject_end in zip(subject, subject[1:] + subject[:1])
        for region_start, region_end in zip(container, container[1:] + container[:1])
    )


def _corridor_polygon(
    start_map: tuple[float, float],
    end_map: tuple[float, float],
    width_map: float,
) -> Polygon:
    dx = end_map[0] - start_map[0]
    dy = end_map[1] - start_map[1]
    length = math.hypot(dx, dy)
    if length <= _EPSILON:
        _fail("corridor_geometry_invalid", "corridor start and end must differ")
    half = width_map * 0.5
    normal = (-dy / length * half, dx / length * half)
    return (
        (start_map[0] + normal[0], start_map[1] + normal[1]),
        (end_map[0] + normal[0], end_map[1] + normal[1]),
        (end_map[0] - normal[0], end_map[1] - normal[1]),
        (start_map[0] - normal[0], start_map[1] - normal[1]),
    )


def _normalize_object_type(value: Any) -> str:
    object_type = _text(value, "object_type").lower().replace("-", "_").replace(" ", "_")
    if object_type in _DYNAMIC_OBJECT_TYPES:
        _fail(
            "dynamic_object_in_static_layer",
            f"dynamic object type {object_type!r} is forbidden in static_obstacles",
        )
    if object_type not in _SUPPORTED_STATIC_OBJECT_TYPES:
        _fail(
            "unsupported_static_object_type",
            f"unsupported static object type: {object_type}",
        )
    return object_type


def _validate_item_reference_frame(
    payload: Mapping[str, Any],
    coordinate_frame: str,
    map_units_per_meter: float,
    item_name: str,
) -> None:
    item_frame = payload.get("coordinate_frame")
    if item_frame is not None and _text(item_frame, "coordinate_frame") != coordinate_frame:
        _fail(
            "coordinate_frame_mismatch",
            f"{item_name} coordinate_frame differs from the layer",
        )
    item_scale = payload.get("map_units_per_meter")
    if item_scale is not None and not math.isclose(
        _positive(item_scale, "map_units_per_meter"),
        map_units_per_meter,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        _fail("map_units_mismatch", f"{item_name} map scale differs from the layer")


@dataclass(frozen=True, slots=True)
class StaticObstacle:
    object_id: str
    object_type: str
    polygon_map: Polygon
    min_height_m: float
    max_height_m: float
    confidence: float
    source: str
    source_id: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        coordinate_frame: str,
        map_units_per_meter: float,
    ) -> "StaticObstacle":
        if not isinstance(payload, Mapping):
            _fail("static_obstacle_invalid", "each static obstacle must be an object")
        object_id = _text(payload.get("object_id", payload.get("id")), "object_id")
        object_type = _normalize_object_type(
            payload.get("type", payload.get("object_type", payload.get("kind")))
        )
        _validate_item_reference_frame(payload, coordinate_frame, map_units_per_meter, object_id)
        polygon = _validate_simple_polygon(
            payload.get(
                "polygon_xy",
                payload.get("polygon_map", payload.get("polygon")),
            ),
            f"static_obstacles[{object_id}].polygon_map",
        )

        if "height_m" in payload:
            base = _finite(payload.get("base_height_m", 0.0), "base_height_m")
            height = _positive(payload["height_m"], "height_m")
            minimum = base
            maximum = base + height
        else:
            minimum = _finite(
                payload.get(
                    "min_z",
                    payload.get("min_height_m", payload.get("height_min_m", 0.0)),
                ),
                "min_height_m",
            )
            maximum_value = payload.get(
                "max_z",
                payload.get("max_height_m", payload.get("height_max_m")),
            )
            if maximum_value is None:
                _fail(
                    "static_obstacle_height_missing",
                    f"static obstacle {object_id} has no height_m or max_height_m",
                )
            maximum = _finite(maximum_value, "max_height_m")
        if maximum <= minimum:
            _fail(
                "static_obstacle_height_invalid",
                f"static obstacle {object_id} must satisfy min_z < max_z",
            )
        confidence = _finite(payload.get("confidence"), "confidence")
        if not 0.0 <= confidence <= 1.0:
            _fail("confidence_invalid", "confidence must be within [0, 1]")
        source_kind = _text(payload.get("source"), "source").lower()
        if source_kind not in _SUPPORTED_SOURCES:
            _fail(
                "static_obstacle_source_invalid",
                f"source must be one of {sorted(_SUPPORTED_SOURCES)}",
            )
        source = payload.get("source_id")
        if source is not None:
            source = _text(source, "source_id")
        return cls(
            object_id,
            object_type,
            polygon,
            minimum,
            maximum,
            confidence,
            source_kind,
            source,
        )

    def is_low_obstacle(self, minimum_camera_occluder_height_m: float) -> bool:
        return (
            self.max_height_m - self.min_height_m
            < minimum_camera_occluder_height_m
        )

    def to_mapping(self) -> dict[str, Any]:
        standard_type = (
            "wall"
            if self.object_type == "wall"
            else (
                "column"
                if self.object_type in {"column", "pillar"}
                else "other_static"
            )
        )
        return {
            "id": self.object_id,
            "object_id": self.object_id,
            "type": standard_type,
            "polygon_xy": [list(point) for point in self.polygon_map],
            "min_z": self.min_height_m,
            "max_z": self.max_height_m,
            "confidence": self.confidence,
            "source": self.source,
            # Backward-compatible aliases for existing geometry consumers.
            "object_type": self.object_type,
            "polygon_map": [list(point) for point in self.polygon_map],
            "min_height_m": self.min_height_m,
            "max_height_m": self.max_height_m,
            **({"source_id": self.source_id} if self.source_id is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class MappedRegion:
    region_id: str
    polygon_map: Polygon
    coverage_status: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        coordinate_frame: str,
        map_units_per_meter: float,
    ) -> "MappedRegion":
        if not isinstance(payload, Mapping):
            _fail("mapped_region_invalid", "each mapped region must be an object")
        region_id = _text(payload.get("region_id", payload.get("id")), "region_id")
        _validate_item_reference_frame(payload, coordinate_frame, map_units_per_meter, region_id)
        polygon = _validate_simple_polygon(
            payload.get(
                "polygon_xy",
                payload.get("polygon_map", payload.get("polygon")),
            ),
            f"mapped_regions[{region_id}].polygon_map",
        )
        complete_flag = payload.get("static_obstacle_layer_complete")
        if complete_flag is not None and not isinstance(complete_flag, bool):
            _fail(
                "mapped_region_completeness_invalid",
                "static_obstacle_layer_complete must be boolean",
            )
        raw_status = payload.get("coverage_status", payload.get("status"))
        if raw_status is None:
            status = "complete" if complete_flag is True else "partial"
        else:
            status = _text(raw_status, "coverage_status").lower()
        if status not in {"complete", "audited_complete", "partial", "unknown"}:
            _fail("mapped_region_status_invalid", f"unsupported coverage status: {status}")
        if complete_flag is not None and complete_flag != (
            status in _COMPLETE_COVERAGE_STATUSES
        ):
            _fail(
                "mapped_region_completeness_conflict",
                "coverage_status conflicts with static_obstacle_layer_complete",
            )
        return cls(region_id, polygon, status)

    @property
    def is_complete(self) -> bool:
        return self.coverage_status in _COMPLETE_COVERAGE_STATUSES

    def to_mapping(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "polygon_xy": [list(point) for point in self.polygon_map],
            "static_obstacle_layer_complete": self.is_complete,
            "coverage_status": self.coverage_status,
            "polygon_map": [list(point) for point in self.polygon_map],
        }


def _geojson_polygon(feature: Mapping[str, Any], feature_id: str) -> Any:
    geometry = feature.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
        _fail("geojson_geometry_invalid", f"feature {feature_id} must use Polygon geometry")
    coordinates = geometry.get("coordinates")
    if (
        not isinstance(coordinates, Sequence)
        or isinstance(coordinates, (str, bytes))
        or len(coordinates) != 1
    ):
        _fail(
            "geojson_polygon_holes_unsupported",
            f"feature {feature_id} must contain one exterior ring and no holes",
        )
    return coordinates[0]


def _normalize_geojson(payload: Mapping[str, Any]) -> dict[str, Any]:
    properties = payload.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        _fail("geojson_properties_invalid", "FeatureCollection properties must be an object")
    normalized: dict[str, Any] = dict(properties)
    normalized["schema_version"] = properties.get(
        "schema_version", STATIC_OBSTACLE_MAP_SCHEMA_VERSION
    )
    obstacles: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    features = payload.get("features", ())
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        _fail("geojson_features_invalid", "FeatureCollection features must be an array")
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            _fail("geojson_feature_invalid", f"feature {index} must be an object")
        feature_properties = feature.get("properties") or {}
        if not isinstance(feature_properties, Mapping):
            _fail("geojson_properties_invalid", f"feature {index} properties must be an object")
        row = dict(feature_properties)
        feature_id = str(feature.get("id", row.get("object_id", row.get("region_id", index))))
        role = str(row.get("feature_role", row.get("layer", ""))).lower()
        polygon = _geojson_polygon(feature, feature_id)
        if role in {"static_obstacle", "obstacle"}:
            row.setdefault("object_id", feature_id)
            row["polygon_map"] = polygon
            obstacles.append(row)
        elif role in {"mapped_region", "coverage"}:
            row.setdefault("region_id", feature_id)
            row["polygon_map"] = polygon
            regions.append(row)
        else:
            _fail(
                "geojson_feature_role_missing",
                f"feature {feature_id} must declare static_obstacle or mapped_region",
            )
    present = properties.get("static_obstacle_layer_present")
    if present is None:
        present = bool(obstacles)
    if not isinstance(present, bool):
        _fail("static_obstacle_layer_presence_invalid", "layer presence must be boolean")
    if present:
        normalized["static_obstacles"] = obstacles
    normalized["mapped_regions"] = regions
    return normalized


def _build_spatial_index(
    obstacles: Sequence[StaticObstacle],
    cell_size_map: float,
) -> tuple[
    Mapping[tuple[int, int], tuple[str, ...]],
    Mapping[str, StaticObstacle],
]:
    cells: dict[tuple[int, int], set[str]] = {}
    by_id = {obstacle.object_id: obstacle for obstacle in obstacles}
    for obstacle in obstacles:
        # Index every grid cell touched by the full AABB.  This intentionally
        # avoids a centroid-only index, which would miss a long wall whose
        # center is far from a narrow query corridor.
        for cell in _grid_cells_for_bounds(
            _polygon_bounds(obstacle.polygon_map),
            cell_size_map,
        ):
            cells.setdefault(cell, set()).add(obstacle.object_id)
    return (
        MappingProxyType(
            {cell: tuple(sorted(object_ids)) for cell, object_ids in cells.items()}
        ),
        MappingProxyType(by_id),
    )


@dataclass(frozen=True, slots=True)
class StaticObstacleQuery:
    information_sufficient: bool
    layer_present: bool
    local_coverage_complete: bool
    complete_empty: bool
    coordinate_frame: str | None
    map_units_per_meter: float | None
    source_id: str | None
    source_sha256: str | None
    corridor_polygon_map: Polygon
    mapped_region_ids: tuple[str, ...]
    obstacles: tuple[StaticObstacle, ...]
    camera_occluder_obstacles: tuple[StaticObstacle, ...]
    low_obstacle_ids: tuple[str, ...]
    spatial_candidate_count: int
    total_obstacle_count: int
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "information_sufficient": self.information_sufficient,
            "layer_present": self.layer_present,
            "local_coverage_complete": self.local_coverage_complete,
            "complete_empty": self.complete_empty,
            "coordinate_frame": self.coordinate_frame,
            "map_units_per_meter": self.map_units_per_meter,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "corridor_polygon_map": [list(point) for point in self.corridor_polygon_map],
            "mapped_region_ids": list(self.mapped_region_ids),
            "static_obstacles": [item.to_mapping() for item in self.obstacles],
            "camera_occluder_obstacles": [
                item.to_mapping() for item in self.camera_occluder_obstacles
            ],
            "low_obstacle_ids": list(self.low_obstacle_ids),
            "spatial_candidate_count": self.spatial_candidate_count,
            "total_obstacle_count": self.total_obstacle_count,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class StaticObstacleMap:
    coordinate_frame: str | None
    map_units_per_meter: float | None
    static_obstacles: tuple[StaticObstacle, ...]
    mapped_regions: tuple[MappedRegion, ...]
    layer_present: bool
    source_id: str | None = None
    source_sha256: str | None = None
    schema_version: str = STATIC_OBSTACLE_MAP_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    spatial_index_cell_size_m: float = 2.0
    _spatial_grid: Mapping[tuple[int, int], tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )
    _obstacle_by_id: Mapping[str, StaticObstacle] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source_id: str | None = None,
        source_sha256: str | None = None,
    ) -> "StaticObstacleMap":
        if not isinstance(payload, Mapping):
            _fail("static_obstacle_map_invalid", "static obstacle map must be an object")
        if payload.get("type") == "FeatureCollection":
            payload = _normalize_geojson(payload)

        schema = payload.get("schema_version", STATIC_OBSTACLE_MAP_SCHEMA_VERSION)
        if schema != STATIC_OBSTACLE_MAP_SCHEMA_VERSION:
            _fail("static_obstacle_schema_unsupported", f"unsupported schema_version: {schema}")

        obstacle_key_present = "static_obstacles" in payload
        explicit_presence = payload.get("static_obstacle_layer_present")
        if explicit_presence is not None:
            if not isinstance(explicit_presence, bool):
                _fail("static_obstacle_layer_presence_invalid", "layer presence must be boolean")
            if explicit_presence is False and obstacle_key_present:
                _fail(
                    "static_obstacle_layer_presence_conflict",
                    "explicit layer presence conflicts with static_obstacles key",
                )
            layer_present = explicit_presence
        else:
            layer_present = obstacle_key_present

        has_geometry_contract = layer_present or "mapped_regions" in payload
        raw_frame = payload.get("coordinate_frame")
        raw_scale = payload.get("map_units_per_meter")
        if not has_geometry_contract and raw_frame is None and raw_scale is None:
            return cls(None, None, (), (), False, source_id, source_sha256)
        coordinate_frame = _text(raw_frame, "coordinate_frame")
        map_units_per_meter = _positive(raw_scale, "map_units_per_meter")

        raw_obstacles = payload.get("static_obstacles", ())
        if not isinstance(raw_obstacles, Sequence) or isinstance(raw_obstacles, (str, bytes)):
            _fail("static_obstacles_invalid", "static_obstacles must be an array")
        obstacles = tuple(
            StaticObstacle.from_mapping(
                row,
                coordinate_frame=coordinate_frame,
                map_units_per_meter=map_units_per_meter,
            )
            for row in raw_obstacles
        )

        raw_regions = payload.get("mapped_regions", ())
        if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)):
            _fail("mapped_regions_invalid", "mapped_regions must be an array")
        parsed_regions: list[MappedRegion] = []
        for index, row in enumerate(raw_regions):
            if not isinstance(row, Mapping):
                _fail("mapped_region_invalid", "each mapped region must be an object")
            normalized_region = dict(row)
            normalized_region.setdefault("region_id", f"region:{index:04d}")
            parsed_regions.append(
                MappedRegion.from_mapping(
                    normalized_region,
                    coordinate_frame=coordinate_frame,
                    map_units_per_meter=map_units_per_meter,
                )
            )
        regions = tuple(parsed_regions)

        obstacle_ids = [item.object_id for item in obstacles]
        region_ids = [item.region_id for item in regions]
        if len(set(obstacle_ids)) != len(obstacle_ids):
            _fail("duplicate_static_obstacle_id", "static obstacle IDs must be unique")
        if len(set(region_ids)) != len(region_ids):
            _fail("duplicate_mapped_region_id", "mapped region IDs must be unique")

        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            _fail("static_obstacle_metadata_invalid", "metadata must be an object")
        cell_size_m = _positive(
            payload.get("spatial_index_cell_size_m", 2.0),
            "spatial_index_cell_size_m",
        )
        sorted_obstacles = tuple(sorted(obstacles, key=lambda item: item.object_id))
        spatial_grid, obstacle_by_id = _build_spatial_index(
            sorted_obstacles,
            cell_size_m * map_units_per_meter,
        )
        return cls(
            coordinate_frame=coordinate_frame,
            map_units_per_meter=map_units_per_meter,
            static_obstacles=sorted_obstacles,
            mapped_regions=tuple(sorted(regions, key=lambda item: item.region_id)),
            layer_present=layer_present,
            source_id=source_id,
            source_sha256=source_sha256,
            metadata=MappingProxyType(dict(raw_metadata)),
            spatial_index_cell_size_m=cell_size_m,
            _spatial_grid=spatial_grid,
            _obstacle_by_id=obstacle_by_id,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StaticObstacleMap":
        source = Path(path)
        content = source.read_bytes()
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail("static_obstacle_json_invalid", f"cannot parse {source}: {exc}")
        if not isinstance(payload, Mapping):
            _fail("static_obstacle_map_invalid", "top-level JSON must be an object")
        return cls.from_mapping(payload, source_id=str(source), source_sha256=digest)

    def query_corridor(
        self,
        start_map: Sequence[float],
        end_map: Sequence[float],
        corridor_width_m: float,
        *,
        coordinate_frame: str,
        map_units_per_meter: float,
        minimum_camera_occluder_height_m: float = 0.60,
    ) -> StaticObstacleQuery:
        start = _point(start_map, "corridor_start_map")
        end = _point(end_map, "corridor_end_map")
        width_m = _positive(corridor_width_m, "corridor_width_m")
        query_scale = _positive(map_units_per_meter, "map_units_per_meter")
        minimum_height = _positive(
            minimum_camera_occluder_height_m,
            "minimum_camera_occluder_height_m",
        )
        query_frame = _text(coordinate_frame, "coordinate_frame")
        corridor = _corridor_polygon(start, end, width_m * query_scale)

        reasons: list[str] = []
        if not self.layer_present:
            reasons.append("static_obstacle_layer_missing")
        frame_matches = self.coordinate_frame is not None and self.coordinate_frame == query_frame
        if self.coordinate_frame is None:
            reasons.append("static_obstacle_coordinate_frame_missing")
        elif not frame_matches:
            reasons.append("static_obstacle_coordinate_frame_mismatch")
        scale_matches = self.map_units_per_meter is not None and math.isclose(
            self.map_units_per_meter,
            query_scale,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        if self.map_units_per_meter is None:
            reasons.append("static_obstacle_map_units_missing")
        elif not scale_matches:
            reasons.append("static_obstacle_map_units_mismatch")

        covering_regions: tuple[MappedRegion, ...] = ()
        if frame_matches and scale_matches:
            covering_regions = tuple(
                region
                for region in self.mapped_regions
                if region.is_complete and _polygon_contains_polygon(region.polygon_map, corridor)
            )
        local_complete = bool(covering_regions)
        if not self.mapped_regions:
            reasons.append("static_obstacle_mapped_regions_missing")
        elif not local_complete:
            reasons.append("corridor_outside_complete_mapped_region")

        candidate_ids: set[str] = set()
        intersecting: tuple[StaticObstacle, ...] = ()
        if frame_matches and scale_matches:
            assert self.map_units_per_meter is not None
            index_cell_size_map = (
                self.spatial_index_cell_size_m * self.map_units_per_meter
            )
            for cell in _grid_cells_for_bounds(
                _polygon_bounds(corridor),
                index_cell_size_map,
            ):
                candidate_ids.update(self._spatial_grid.get(cell, ()))
            intersecting = tuple(
                obstacle
                for obstacle in (
                    self._obstacle_by_id[object_id]
                    for object_id in sorted(candidate_ids)
                )
                if _polygons_intersect(obstacle.polygon_map, corridor)
            )
        low = tuple(
            obstacle.object_id
            for obstacle in intersecting
            if obstacle.is_low_obstacle(minimum_height)
        )
        occluders = tuple(
            obstacle
            for obstacle in intersecting
            if not obstacle.is_low_obstacle(minimum_height)
        )
        sufficient = self.layer_present and frame_matches and scale_matches and local_complete
        return StaticObstacleQuery(
            information_sufficient=sufficient,
            layer_present=self.layer_present,
            local_coverage_complete=local_complete,
            complete_empty=sufficient and not intersecting,
            coordinate_frame=self.coordinate_frame,
            map_units_per_meter=self.map_units_per_meter,
            source_id=self.source_id,
            source_sha256=self.source_sha256,
            corridor_polygon_map=corridor,
            mapped_region_ids=tuple(region.region_id for region in covering_regions),
            obstacles=intersecting,
            camera_occluder_obstacles=occluders,
            low_obstacle_ids=low,
            spatial_candidate_count=len(candidate_ids),
            total_obstacle_count=len(self.static_obstacles),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


@dataclass(frozen=True, slots=True)
class StaticObstacleMapLoadResult:
    static_map: StaticObstacleMap | None
    reason_codes: tuple[str, ...]
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.static_map is not None


def try_load_static_obstacle_map(path: str | Path) -> StaticObstacleMapLoadResult:
    try:
        return StaticObstacleMapLoadResult(StaticObstacleMap.load(path), ())
    except FileNotFoundError as exc:
        return StaticObstacleMapLoadResult(None, ("static_obstacle_file_missing",), str(exc))
    except OSError as exc:
        return StaticObstacleMapLoadResult(None, ("static_obstacle_file_unreadable",), str(exc))
    except StaticObstacleMapValidationError as exc:
        return StaticObstacleMapLoadResult(None, (exc.reason_code,), str(exc))


@dataclass(frozen=True, slots=True)
class StaticObstacleMapCacheStats:
    hits: int
    misses: int
    load_errors: int
    entries: int


class StaticObstacleMapCache:
    """Small stat-keyed, thread-safe LRU cache for validated map files."""

    def __init__(self, max_entries: int = 8) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries: OrderedDict[
            str, tuple[tuple[int, int], StaticObstacleMap]
        ] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._load_errors = 0
        self._lock = RLock()

    def get(self, path: str | Path) -> StaticObstacleMap:
        source = Path(path).resolve()
        stat = source.stat()
        fingerprint = (int(stat.st_mtime_ns), int(stat.st_size))
        key = str(source)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached[0] == fingerprint:
                self._hits += 1
                self._entries.move_to_end(key)
                return cached[1]
            self._misses += 1
        try:
            loaded = StaticObstacleMap.load(source)
        except Exception:
            with self._lock:
                self._load_errors += 1
            raise
        with self._lock:
            self._entries[key] = (fingerprint, loaded)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return loaded

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> StaticObstacleMapCacheStats:
        with self._lock:
            return StaticObstacleMapCacheStats(
                self._hits,
                self._misses,
                self._load_errors,
                len(self._entries),
            )


def load_gltf_static_obstacle_map(
    path: str | Path,
    *,
    coordinate_frame: str,
    map_units_per_meter: float,
    mapped_regions: Sequence[Mapping[str, Any]] = (),
    semantic_types: Sequence[str] = ("wall",),
) -> StaticObstacleMap:
    """Load explicit structural GLTF meshes as triangle footprints.

    The adapter reuses the repository's GLTF accessor/matrix implementation.
    ``mapped_regions`` defaults to empty because semantic presence of a wall
    mesh is not evidence that pillars or all other obstacle classes were
    surveyed.  Consequently the default result remains fail-closed for local
    completeness checks.
    """

    frame = _text(coordinate_frame, "coordinate_frame")
    scale = _positive(map_units_per_meter, "map_units_per_meter")
    requested = tuple(_normalize_object_type(item) for item in semantic_types)
    if not requested:
        _fail("gltf_semantic_types_missing", "at least one static semantic type is required")

    source = Path(path)
    content = source.read_bytes()
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("gltf_json_invalid", f"cannot parse {source}: {exc}")
    try:
        from scripts.gltf_lidar_ndt import (  # pylint: disable=import-outside-toplevel
            accessor_array,
            decode_data_uri,
            node_matrix,
            transform_positions,
        )

        buffer_bytes = decode_data_uri(data["buffers"][0]["uri"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _fail("gltf_buffer_invalid", f"cannot decode GLTF buffer: {exc}")

    rows: list[dict[str, Any]] = []
    seen_footprints: set[tuple[Any, ...]] = set()
    found_semantics: set[str] = set()
    for node_index, node in enumerate(data.get("nodes", ())):
        semantic_raw = (node.get("extras") or {}).get("type")
        semantic_key = (
            str(semantic_raw).strip().lower().replace("-", "_").replace(" ", "_")
            if semantic_raw is not None
            else ""
        )
        if semantic_key not in requested or "mesh" not in node:
            continue
        semantic = _normalize_object_type(semantic_key)
        found_semantics.add(semantic)
        try:
            mesh = data["meshes"][int(node["mesh"])]
            transform = node_matrix(node)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            _fail("gltf_mesh_invalid", f"invalid GLTF node {node_index}: {exc}")
        primitives: list[tuple[int, np.ndarray, np.ndarray]] = []
        node_positions: list[np.ndarray] = []
        for primitive_index, primitive in enumerate(mesh.get("primitives", ())):
            if int(primitive.get("mode", 4)) != 4:
                continue
            try:
                accessor_index = int(primitive["attributes"]["POSITION"])
                positions = accessor_array(data, buffer_bytes, accessor_index).astype(np.float64)
                positions = transform_positions(positions, transform)
                if "indices" in primitive:
                    order = accessor_array(
                        data, buffer_bytes, int(primitive["indices"])
                    ).astype(np.int64).reshape(-1)
                else:
                    order = np.arange(len(positions), dtype=np.int64)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                _fail(
                    "gltf_primitive_invalid",
                    f"invalid GLTF node {node_index} primitive {primitive_index}: {exc}",
                )
            primitives.append((primitive_index, positions, order))
            node_positions.append(positions)
        if not node_positions:
            continue
        all_positions = np.vstack(node_positions)
        min_z_m = float(np.min(all_positions[:, 2]) / scale)
        max_z_m = float(np.max(all_positions[:, 2]) / scale)
        if (
            not math.isfinite(min_z_m)
            or not math.isfinite(max_z_m)
            or max_z_m <= min_z_m
        ):
            _fail(
                "gltf_static_height_invalid",
                f"GLTF semantic node {node_index} has no positive vertical extent",
            )
        for primitive_index, positions, order in primitives:
            for offset in range(0, len(order) - 2, 3):
                indices = order[offset : offset + 3]
                triangle = positions[indices, :2]
                polygon = tuple((float(point[0]), float(point[1])) for point in triangle)
                if abs(_polygon_area(polygon)) <= _EPSILON:
                    continue
                canonical = tuple(sorted((round(x, 10), round(y, 10)) for x, y in polygon))
                key = (semantic, node_index, canonical)
                if key in seen_footprints:
                    continue
                seen_footprints.add(key)
                rows.append(
                    {
                        "object_id": (
                            f"gltf:{semantic}:node-{node_index}:primitive-{primitive_index}:"
                            f"triangle-{offset // 3}"
                        ),
                        "object_type": semantic,
                        "polygon_map": [list(point) for point in polygon],
                        "min_z": min_z_m,
                        "max_z": max_z_m,
                        # Confidence describes exact CAD geometry ingestion;
                        # it does not claim the source surveyed every static
                        # obstacle class.  Completeness remains mapped-region
                        # controlled and defaults to false.
                        "confidence": 1.0,
                        "source": "cad",
                        "source_id": digest,
                    }
                )
    missing = sorted(set(requested) - found_semantics)
    if missing:
        _fail(
            "gltf_static_semantic_missing",
            f"GLTF does not contain requested semantic mesh layers: {missing}",
        )
    if not rows:
        _fail("gltf_static_footprint_missing", "GLTF static meshes have no area footprints")

    return StaticObstacleMap.from_mapping(
        {
            "schema_version": STATIC_OBSTACLE_MAP_SCHEMA_VERSION,
            "coordinate_frame": frame,
            "map_units_per_meter": scale,
            "static_obstacles": rows,
            "mapped_regions": list(mapped_regions),
            "metadata": {
                "source_format": "gltf_semantic_mesh",
                "semantic_types": list(requested),
                "coverage_warning": (
                    "GLTF semantic meshes do not imply complete wall/pillar survey coverage"
                ),
            },
        },
        source_id=str(source),
        source_sha256=digest,
    )


__all__ = [
    "STATIC_OBSTACLE_MAP_SCHEMA_VERSION",
    "MappedRegion",
    "StaticObstacle",
    "StaticObstacleMap",
    "StaticObstacleMapCache",
    "StaticObstacleMapCacheStats",
    "StaticObstacleMapLoadResult",
    "StaticObstacleMapValidationError",
    "StaticObstacleQuery",
    "load_gltf_static_obstacle_map",
    "try_load_static_obstacle_map",
]
