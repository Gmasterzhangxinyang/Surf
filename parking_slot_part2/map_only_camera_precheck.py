"""Conservative Camera-call precheck using only a Part1 local map.

The precheck is deliberately weaker than calibrated Camera observability.  It
uses the Part1 anchor pose as a Camera proxy and answers only whether a local
slot is a promising Camera-call candidate.  It never invokes a Camera model,
never consumes Camera detections, and never changes Part1 occupancy states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import numpy as np

from .camera_observability import _corridor_hit_t, _polygon_center, _samples


_EPSILON = 1e-9
_SUPPORTED_TARGET_STATES = frozenset({"free", "unknown"})
_KNOWN_SLOT_STATES = frozenset(
    {"free", "occupied", "unknown", "partial", "partial_route", "out_of_route"}
)


@dataclass(frozen=True)
class MapOnlyCameraPrecheckConfig:
    """Conservative defaults for the uncalibrated, map-only gate.

    These defaults are engineering policy values, not Camera calibration.
    Every distance is expressed in metres and is converted through Part1's
    ``map_units_per_meter`` before geometry is evaluated.
    """

    nominal_half_fov_deg: float = 90.0
    reliable_half_fov_deg: float = 80.0
    minimum_reliable_fov_coverage: float = 0.80
    minimum_clear_ray_ratio: float = 0.80
    maximum_blocked_ray_ratio: float = 0.20
    maximum_uncertain_ray_ratio: float = 0.20
    minimum_target_distance_m: float = 0.75
    maximum_target_distance_m: float = 12.0
    minimum_target_angular_width_deg: float = 6.0
    proxy_position_uncertainty_m: float = 0.25
    proxy_yaw_uncertainty_deg: float = 5.0
    visibility_corridor_width_m: float = 0.20
    obstacle_inflation_m: float = 0.05
    occupied_probability_blocker_threshold: float = 0.70
    occupied_probability_potential_threshold: float = 0.30
    low_obstacle_max_height_m: float = 0.40
    include_edge_midpoints: bool = True

    def __post_init__(self) -> None:
        finite = {
            item.name: float(getattr(self, item.name))
            for item in fields(self)
            if item.name != "include_edge_midpoints"
        }
        if not all(math.isfinite(value) for value in finite.values()):
            raise ValueError("map-only precheck thresholds must be finite")
        if not 0.0 < self.nominal_half_fov_deg <= 180.0:
            raise ValueError("nominal_half_fov_deg must be within (0,180]")
        if not 0.0 < self.reliable_half_fov_deg <= self.nominal_half_fov_deg:
            raise ValueError(
                "reliable_half_fov_deg must be positive and no wider than nominal FOV"
            )
        for name in (
            "minimum_reliable_fov_coverage",
            "minimum_clear_ray_ratio",
            "maximum_blocked_ray_ratio",
            "maximum_uncertain_ray_ratio",
            "occupied_probability_blocker_threshold",
            "occupied_probability_potential_threshold",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")
        if (
            self.occupied_probability_potential_threshold
            > self.occupied_probability_blocker_threshold
        ):
            raise ValueError("potential occupancy threshold cannot exceed blocker threshold")
        for name in (
            "minimum_target_distance_m",
            "proxy_position_uncertainty_m",
            "proxy_yaw_uncertainty_deg",
            "visibility_corridor_width_m",
            "obstacle_inflation_m",
            "low_obstacle_max_height_m",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.maximum_target_distance_m <= self.minimum_target_distance_m:
            raise ValueError("maximum_target_distance_m must exceed the minimum")
        if self.minimum_target_angular_width_deg <= 0.0:
            raise ValueError("minimum_target_angular_width_deg must be positive")
        if not isinstance(self.include_edge_midpoints, bool):
            raise TypeError("include_edge_midpoints must be bool")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MapOnlyCameraPrecheckConfig":
        if not isinstance(value, Mapping):
            raise TypeError("map-only precheck config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown map-only precheck config fields: {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True)
class _Slot:
    slot_id: str
    state: str
    polygon: np.ndarray
    center: np.ndarray
    occupancy_probability: float | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class _Occluder:
    object_id: str
    polygon: np.ndarray
    certainty: str
    source: str
    skip_when_proxy_inside: bool


def _finite_xy(value: Any, name: str, *, minimum_rows: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or len(array) < minimum_rows
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite [N,2] array")
    return array


def _finite_pair(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XY pair")
    return array


def _normalise_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _bearing_deg(origin: np.ndarray, yaw_rad: float, point: np.ndarray) -> float:
    delta = point - origin
    return _normalise_deg(math.degrees(math.atan2(delta[1], delta[0]) - yaw_rad))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Boundary-inclusive ray-casting test used to ignore the ego's own slot."""

    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-10
            and min(x1, x2) - _EPSILON <= x <= max(x1, x2) + _EPSILON
            and min(y1, y2) - _EPSILON <= y <= max(y1, y2) + _EPSILON
        ):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x >= x - _EPSILON:
                inside = not inside
        previous = current
    return inside


def _parse_probability(raw: Mapping[str, Any]) -> float | None:
    for key in ("occupied_probability", "occupancy_probability", "p_occupied"):
        value = raw.get(key)
        if value is None:
            continue
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{key} must be finite and within [0,1]")
        return result
    return None


def _parse_slots(local_map: Mapping[str, Any]) -> tuple[list[_Slot], list[str]]:
    raw_slots = local_map.get("slots")
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        raise ValueError("Part1 slots must be a sequence")
    slots: list[_Slot] = []
    limitations: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_slots):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Part1 slot {index} must be a mapping")
        slot_id = str(raw.get("slot_id", "")).strip()
        if not slot_id or slot_id in seen:
            raise ValueError("Part1 local slots require unique non-empty slot_id values")
        state = str(raw.get("state", "")).strip().lower()
        if state not in _KNOWN_SLOT_STATES:
            raise ValueError(f"unsupported Part1 state for {slot_id}: {state}")
        polygon = _finite_xy(
            raw.get("polygon_map"), f"slot {slot_id} polygon_map", minimum_rows=3
        )
        center_raw = raw.get("center_map")
        center = (
            _polygon_center(polygon)
            if center_raw is None
            else _finite_pair(center_raw, f"slot {slot_id} center_map")
        )
        slots.append(
            _Slot(
                slot_id=slot_id,
                state=state,
                polygon=polygon,
                center=center,
                occupancy_probability=_parse_probability(raw),
                raw=raw,
            )
        )
        seen.add(slot_id)
    if not slots:
        limitations.append("Part1 local map contains no slots.")
    return slots, limitations


def _slot_occlusion_class(
    slot: _Slot, config: MapOnlyCameraPrecheckConfig
) -> str | None:
    probability = slot.occupancy_probability
    if slot.state == "occupied":
        return "blocked"
    if slot.state == "free":
        return None
    if slot.state == "unknown":
        return "uncertain"
    if probability is not None:
        if probability >= config.occupied_probability_blocker_threshold:
            return "blocked"
        if probability >= config.occupied_probability_potential_threshold:
            return "uncertain"
        return None
    # A partial or out-of-route slot still represents unresolved physical space.
    return "uncertain"


def _preferred_physical_polygon(slot: _Slot) -> np.ndarray:
    for key in ("vehicle_polygon_map", "obstacle_polygon_map"):
        value = slot.raw.get(key)
        if value is not None:
            return _finite_xy(value, f"slot {slot.slot_id} {key}", minimum_rows=3)
    return slot.polygon


def _parse_local_obstacles(
    local_map: Mapping[str, Any], config: MapOnlyCameraPrecheckConfig
) -> tuple[list[_Occluder], list[str]]:
    raw_layer = local_map.get("local_obstacles")
    if raw_layer is None:
        raw_layer = local_map.get("static_obstacles")
    if raw_layer is None:
        return [], [
            "Part1 local_obstacles is absent; map-only LOS uses slot-state polygons only."
        ]
    if isinstance(raw_layer, Mapping):
        raw_layer = raw_layer.get("obstacles", raw_layer.get("static_obstacles"))
    if not isinstance(raw_layer, Sequence) or isinstance(raw_layer, (str, bytes)):
        raise ValueError("local_obstacles must be a sequence or obstacle-layer mapping")
    obstacles: list[_Occluder] = []
    for index, raw in enumerate(raw_layer):
        if not isinstance(raw, Mapping):
            raise ValueError(f"local obstacle {index} must be a mapping")
        object_id = str(raw.get("id", raw.get("object_id", f"obstacle_{index}"))).strip()
        polygon_value = raw.get("polygon_map", raw.get("polygon_xy"))
        polygon = _finite_xy(
            polygon_value, f"local obstacle {object_id} polygon", minimum_rows=3
        )
        blocks_view = raw.get("blocks_view")
        max_z = raw.get("max_z")
        if blocks_view is False:
            continue
        if max_z is not None:
            height = float(max_z)
            if not math.isfinite(height):
                raise ValueError(f"local obstacle {object_id} max_z must be finite")
            if height <= config.low_obstacle_max_height_m and blocks_view is not True:
                continue
        confidence = raw.get("confidence")
        certainty = "blocked"
        if confidence is not None:
            confidence_value = float(confidence)
            if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                raise ValueError(
                    f"local obstacle {object_id} confidence must be within [0,1]"
                )
            if confidence_value < config.occupied_probability_blocker_threshold:
                certainty = "uncertain"
        obstacles.append(
            _Occluder(
                object_id=object_id,
                polygon=polygon,
                certainty=certainty,
                source="local_obstacle",
                skip_when_proxy_inside=False,
            )
        )
    return obstacles, []


def _build_occluders(
    slots: Sequence[_Slot],
    target_id: str,
    local_obstacles: Sequence[_Occluder],
    config: MapOnlyCameraPrecheckConfig,
) -> list[_Occluder]:
    result = list(local_obstacles)
    for slot in slots:
        if slot.slot_id == target_id:
            continue
        certainty = _slot_occlusion_class(slot, config)
        if certainty is None:
            continue
        result.append(
            _Occluder(
                object_id=slot.slot_id,
                polygon=_preferred_physical_polygon(slot),
                certainty=certainty,
                source=f"slot_state:{slot.state}",
                skip_when_proxy_inside=True,
            )
        )
    return result


def _classify_ray(
    origin: np.ndarray,
    endpoint: np.ndarray,
    occluders: Sequence[_Occluder],
    corridor_half_width_map: float,
) -> tuple[str, str | None]:
    hits: list[tuple[float, int, str, str]] = []
    for occluder in occluders:
        if occluder.skip_when_proxy_inside and _point_in_polygon(origin, occluder.polygon):
            continue
        hit_t = _corridor_hit_t(
            origin, endpoint, occluder.polygon, corridor_half_width_map
        )
        if hit_t is None:
            continue
        # Explicit blockers win only at the same geometric depth; a nearer
        # unknown remains unresolved instead of being silently skipped.
        certainty_rank = 0 if occluder.certainty == "blocked" else 1
        hits.append((float(hit_t), certainty_rank, occluder.certainty, occluder.object_id))
    if not hits:
        return "clear", None
    _, _, certainty, object_id = min(hits, key=lambda item: (item[0], item[1], item[3]))
    return certainty, object_id


def _scenario_origins(anchor: np.ndarray, radius_map: float) -> tuple[np.ndarray, ...]:
    offsets = ((0.0, 0.0), (radius_map, 0.0), (-radius_map, 0.0), (0.0, radius_map), (0.0, -radius_map))
    unique: list[np.ndarray] = []
    for dx, dy in offsets:
        point = anchor + np.asarray([dx, dy], dtype=np.float64)
        if not any(np.linalg.norm(point - prior) <= _EPSILON for prior in unique):
            unique.append(point)
    return tuple(unique)


def _angular_width_deg(
    origin: np.ndarray, yaw_rad: float, center: np.ndarray, polygon: np.ndarray
) -> float:
    center_bearing = _bearing_deg(origin, yaw_rad, center)
    offsets = [
        _normalise_deg(_bearing_deg(origin, yaw_rad, point) - center_bearing)
        for point in polygon
    ]
    return float(max(offsets) - min(offsets))


def _display_labels(local_map: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = local_map.get("provisional_candidates", ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                slot_id = str(item.get("slot_id", "")).strip()
                label = str(item.get("display_label", "")).strip()
                if slot_id and label:
                    result[slot_id] = label
    return result


def _evaluate_target(
    *,
    slot: _Slot,
    display_label: str,
    slots: Sequence[_Slot],
    local_obstacles: Sequence[_Occluder],
    anchor: np.ndarray,
    yaw_rad: float,
    scale: float,
    config: MapOnlyCameraPrecheckConfig,
) -> dict[str, Any]:
    sample_points = _samples(slot.polygon, slot.center, config.include_edge_midpoints)
    origin_scenarios = _scenario_origins(
        anchor, config.proxy_position_uncertainty_m * scale
    )
    yaw_scenarios = tuple(
        yaw_rad + math.radians(offset)
        for offset in (-config.proxy_yaw_uncertainty_deg, 0.0, config.proxy_yaw_uncertainty_deg)
    )
    occluders = _build_occluders(slots, slot.slot_id, local_obstacles, config)
    corridor_half_width_map = (
        config.visibility_corridor_width_m * 0.5 + config.obstacle_inflation_m
    ) * scale

    center_bearing = _bearing_deg(anchor, yaw_rad, slot.center)
    distance_m = float(np.linalg.norm(slot.center - anchor) / scale)
    angular_width = _angular_width_deg(anchor, yaw_rad, slot.center, slot.polygon)
    nominal_bearings = [
        _bearing_deg(anchor, yaw_rad, point) for point in sample_points
    ]
    nominal_fov_coverage = sum(
        abs(value) <= config.nominal_half_fov_deg + _EPSILON
        for value in nominal_bearings
    ) / len(sample_points)
    reliable_fov_coverage = sum(
        abs(value) <= config.reliable_half_fov_deg + _EPSILON
        for value in nominal_bearings
    ) / len(sample_points)

    robust_center_checks = [
        abs(_bearing_deg(origin, scenario_yaw, slot.center))
        <= config.reliable_half_fov_deg + _EPSILON
        for origin in origin_scenarios
        for scenario_yaw in yaw_scenarios
    ]
    robust_center_fov_coverage = sum(robust_center_checks) / len(robust_center_checks)

    worst_clear = 1.0
    worst_blocked = 0.0
    worst_uncertain = 0.0
    center_potential = False
    blocking_ids: set[str] = set()
    potential_ids: set[str] = set()
    nominal_rays: list[dict[str, Any]] = []

    for origin in origin_scenarios:
        for scenario_yaw in yaw_scenarios:
            counts = {"clear": 0, "blocked": 0, "uncertain": 0}
            for sample_index, endpoint in enumerate(sample_points):
                bearing = _bearing_deg(origin, scenario_yaw, endpoint)
                if abs(bearing) > config.nominal_half_fov_deg + _EPSILON:
                    ray_status = "outside_fov"
                    object_id = None
                elif abs(bearing) > config.reliable_half_fov_deg + _EPSILON:
                    ray_status = "edge_unreliable"
                    object_id = None
                else:
                    ray_status, object_id = _classify_ray(
                        origin, endpoint, occluders, corridor_half_width_map
                    )
                    counts[ray_status] += 1
                    if ray_status == "blocked" and object_id is not None:
                        blocking_ids.add(object_id)
                    elif ray_status == "uncertain" and object_id is not None:
                        potential_ids.add(object_id)
                        if sample_index == 0:
                            center_potential = True
                if np.linalg.norm(origin - anchor) <= _EPSILON and abs(scenario_yaw - yaw_rad) <= _EPSILON:
                    nominal_rays.append(
                        {
                            "sample_index": sample_index,
                            "start_map_xy": [float(origin[0]), float(origin[1])],
                            "end_map_xy": [float(endpoint[0]), float(endpoint[1])],
                            "status": ray_status,
                            "object_id": object_id,
                        }
                    )
            denominator = float(len(sample_points))
            clear_ratio = counts["clear"] / denominator
            blocked_ratio = counts["blocked"] / denominator
            uncertain_ratio = counts["uncertain"] / denominator
            worst_clear = min(worst_clear, clear_ratio)
            worst_blocked = max(worst_blocked, blocked_ratio)
            worst_uncertain = max(worst_uncertain, uncertain_ratio)

    checks: list[tuple[bool, str]] = [
        (slot.state in _SUPPORTED_TARGET_STATES, "target_state_not_candidate_eligible"),
        (
            abs(center_bearing) <= config.reliable_half_fov_deg + _EPSILON,
            "target_center_outside_conservative_fov",
        ),
        (
            reliable_fov_coverage >= config.minimum_reliable_fov_coverage,
            "insufficient_reliable_fov_coverage",
        ),
        (
            robust_center_fov_coverage >= 1.0 - _EPSILON,
            "not_robust_to_proxy_pose_uncertainty",
        ),
        (
            distance_m >= config.minimum_target_distance_m,
            "target_too_close_for_map_only_gate",
        ),
        (
            distance_m <= config.maximum_target_distance_m,
            "target_too_far_for_map_only_gate",
        ),
        (
            angular_width >= config.minimum_target_angular_width_deg,
            "target_apparent_size_too_small",
        ),
        (
            worst_clear >= config.minimum_clear_ray_ratio,
            "insufficient_clear_ray_ratio",
        ),
        (
            worst_blocked <= config.maximum_blocked_ray_ratio,
            "explicit_line_of_sight_blockage",
        ),
        (
            worst_uncertain <= config.maximum_uncertain_ray_ratio and not center_potential,
            "unresolved_potential_occluder",
        ),
    ]
    failed = [reason for passed, reason in checks if not passed]
    marked = not failed
    if marked:
        reasons = [
            "inside_conservative_80deg_fov",
            "target_size_and_distance_sufficient",
            "map_line_of_sight_mostly_clear",
            "robust_to_proxy_pose_uncertainty",
            "marked_only_camera_not_called",
        ]
    else:
        reasons = failed + ["camera_not_called"]

    return {
        "target_slot_id": slot.slot_id,
        "display_label": display_label,
        "part1_state": slot.state,
        "status": "marked_candidate" if marked else "not_marked",
        "camera_candidate": marked,
        "camera_likely_observable": marked,
        "camera_call_requested": False,
        "part1_state_modified": False,
        "target_bearing_deg": round(center_bearing, 6),
        "target_distance_m": round(distance_m, 6),
        "target_angular_width_deg": round(angular_width, 6),
        "nominal_fov_coverage": round(nominal_fov_coverage, 6),
        "reliable_fov_coverage": round(reliable_fov_coverage, 6),
        "robust_center_fov_coverage": round(robust_center_fov_coverage, 6),
        "clear_ray_ratio": round(worst_clear, 6),
        "blocked_ray_ratio": round(worst_blocked, 6),
        "uncertain_ray_ratio": round(worst_uncertain, 6),
        "blocking_object_ids": sorted(blocking_ids),
        "potential_occluder_ids": sorted(potential_ids),
        "reason_codes": reasons,
        "rays": nominal_rays,
    }


def evaluate_map_only_camera_candidates(
    local_map: Mapping[str, Any],
    target_slot_ids: Sequence[str],
    *,
    config: MapOnlyCameraPrecheckConfig | Mapping[str, Any] | None = None,
    include_bev_masks: bool = True,
) -> dict[str, Any]:
    """Mark one or two local slots that are promising for a future Camera call.

    ``camera_call_requested`` and ``semantic_camera_model_called`` are always
    false.  A mark means only that the Part1-map proxy checks passed.
    """

    if not isinstance(local_map, Mapping):
        raise TypeError("local_map must be a mapping")
    if isinstance(target_slot_ids, (str, bytes)) or not isinstance(
        target_slot_ids, Sequence
    ):
        raise TypeError("target_slot_ids must be a sequence")
    target_ids = [str(item).strip() for item in target_slot_ids]
    if not 1 <= len(target_ids) <= 2 or any(not item for item in target_ids):
        raise ValueError("provide one or two non-empty target_slot_ids")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_slot_ids must be unique")
    if not isinstance(include_bev_masks, bool):
        raise TypeError("include_bev_masks must be bool")
    if config is None:
        policy = MapOnlyCameraPrecheckConfig()
    elif isinstance(config, MapOnlyCameraPrecheckConfig):
        policy = config
    else:
        policy = MapOnlyCameraPrecheckConfig.from_mapping(config)

    scale = float(local_map.get("map_units_per_meter"))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Part1 map_units_per_meter must be positive and finite")
    anchor_raw = local_map.get("anchor_pose")
    if not isinstance(anchor_raw, Mapping):
        raise ValueError("Part1 anchor_pose is required for the Camera proxy")
    anchor = _finite_pair(anchor_raw.get("map_xy"), "anchor_pose.map_xy")
    yaw_rad = float(anchor_raw.get("map_yaw_rad"))
    if not math.isfinite(yaw_rad):
        raise ValueError("anchor_pose.map_yaw_rad must be finite")

    slots, limitations = _parse_slots(local_map)
    by_id = {slot.slot_id: slot for slot in slots}
    missing = [slot_id for slot_id in target_ids if slot_id not in by_id]
    if missing:
        raise ValueError(f"targets are not in the Part1 local map: {missing}")
    local_obstacles, obstacle_limitations = _parse_local_obstacles(local_map, policy)
    limitations.extend(obstacle_limitations)
    limitations.insert(
        0, "Camera proxy is Part1 anchor pose, not a calibrated optical pose."
    )

    labels = _display_labels(local_map)
    targets = [
        _evaluate_target(
            slot=by_id[slot_id],
            display_label=labels.get(slot_id, chr(65 + index)),
            slots=slots,
            local_obstacles=local_obstacles,
            anchor=anchor,
            yaw_rad=yaw_rad,
            scale=scale,
            config=policy,
        )
        for index, slot_id in enumerate(target_ids)
    ]
    marked_ids = [
        target["target_slot_id"] for target in targets if target["camera_candidate"]
    ]
    result: dict[str, Any] = {
        "schema_version": "map-only-camera-precheck/1.0",
        "mode": "conservative_map_only_gate",
        "algorithm_executed": True,
        "semantic_camera_model_called": False,
        "camera_call_requested": False,
        "part1_slot_states_modified": False,
        "proxy_pose": {
            "source": "part1_anchor_pose_camera_proxy",
            "position_map_xy": [float(anchor[0]), float(anchor[1])],
            "yaw_rad": yaw_rad,
            "yaw_deg": math.degrees(yaw_rad),
            "position_uncertainty_m": policy.proxy_position_uncertainty_m,
            "yaw_uncertainty_deg": policy.proxy_yaw_uncertainty_deg,
            "frame_id": anchor_raw.get("frame_id"),
        },
        "policy": asdict(policy),
        "targets": targets,
        "marked_candidate_slot_ids": marked_ids,
        "not_marked_slot_ids": [
            target["target_slot_id"]
            for target in targets
            if not target["camera_candidate"]
        ],
        "input_limitations": list(dict.fromkeys(limitations)),
    }
    if include_bev_masks:
        result["machine_bev"] = {
            "input_only": True,
            "decision_status_encoded": False,
            "coordinate_frame": "part1_local_map",
            "channels": [
                "lidar_observed",
                "slot_free",
                "slot_occupied",
                "slot_unknown",
                "local_obstacle",
                "target",
                "camera_proxy",
            ],
            "note": "Use the deterministic camera_gate_bev raster renderer for pixels.",
        }
    return result


__all__ = [
    "MapOnlyCameraPrecheckConfig",
    "evaluate_map_only_camera_candidates",
]
