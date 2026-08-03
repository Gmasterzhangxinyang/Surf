"""JSON wire contracts for the independent Parking Slot Agent v2 pipeline.

The v2 contracts intentionally depend only on the Python standard library.
They do not inherit the immutable Part1/Part2 v1 queue records: a ``SlotCase``
is a mutable working object that accumulates FOV checks, tool evidence and
reasoning-round summaries while preserving its original Part1 result.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence, TypeVar


PART1_OUTPUT_SCHEMA_VERSION = "parking-slot-agent-v2-part1/1.0"


class SlotState(str, Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class CaseStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    EXHAUSTED = "exhausted"
    SELECTED = "selected"


class FovVisibility(str, Enum):
    NOT_CHECKED = "not_checked"
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"
    UNCERTAIN = "uncertain"


_EnumT = TypeVar("_EnumT", bound=Enum)
_MISSING = object()


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _fields(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] = set(),
) -> Mapping[str, Any]:
    payload = _mapping(value, path)
    missing = sorted(required - set(payload))
    unsupported = sorted(set(payload) - required - optional)
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}")
    if unsupported:
        raise ValueError(f"{path} has unsupported field(s): {', '.join(unsupported)}")
    return payload


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ValueError(f"{path} must be a non-empty string")
    return result


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value


def _optional_integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, path, minimum=minimum)


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return result


def _optional_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum, maximum=maximum)


def _enum(value: Any, enum_type: type[_EnumT], path: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{path} must be one of: {allowed}") from exc


def _optional_enum(
    value: Any,
    enum_type: type[_EnumT],
    path: str,
) -> _EnumT | None:
    if value is None:
        return None
    return _enum(value, enum_type, path)


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a JSON array")
    return value


def _strings(value: Any, path: str, *, unique: bool = True) -> list[str]:
    result = [
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    ]
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def _integers(value: Any, path: str, *, minimum: int = 0) -> list[int]:
    result = [
        _integer(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, path))
    ]
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def _point(value: Any, path: str, dimensions: int) -> tuple[float, ...]:
    raw = _sequence(value, path)
    if len(raw) != dimensions:
        raise ValueError(f"{path} must contain exactly {dimensions} numbers")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(raw))


def _polygon(value: Any, path: str, *, allow_empty: bool = False) -> list[list[float]]:
    raw = _sequence(value, path)
    if not raw and allow_empty:
        return []
    if len(raw) < 3:
        raise ValueError(f"{path} must contain at least three 2D points")
    result: list[list[float]] = []
    for index, item in enumerate(raw):
        point = _point(item, f"{path}[{index}]", 2)
        result.append([point[0], point[1]])
    return result


def _resources(value: Any, path: str) -> dict[str, str]:
    payload = _mapping(value, path)
    result: dict[str, str] = {}
    for key, item in payload.items():
        normalized_key = _string(key, f"{path}.<key>")
        normalized_value = _string(item, f"{path}.{normalized_key}")
        result[normalized_key] = normalized_value
    return dict(sorted(result.items()))


def _json_value(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} object keys must be strings")
        return {
            key: _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}")


@dataclass(slots=True)
class ConfidenceScores:
    free_confidence: float
    occupied_confidence: float
    unknown_confidence: float
    kind: str = "heuristic_gate_score"
    calibrated: bool = False

    def __post_init__(self) -> None:
        self.free_confidence = _number(
            self.free_confidence, "ConfidenceScores.free_confidence", minimum=0.0, maximum=1.0
        )
        self.occupied_confidence = _number(
            self.occupied_confidence,
            "ConfidenceScores.occupied_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        self.unknown_confidence = _number(
            self.unknown_confidence,
            "ConfidenceScores.unknown_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        self.kind = _string(self.kind, "ConfidenceScores.kind")
        self.calibrated = _boolean(self.calibrated, "ConfidenceScores.calibrated")

    def for_state(self, state: SlotState | str) -> float:
        normalized = _enum(state, SlotState, "state")
        return {
            SlotState.FREE: self.free_confidence,
            SlotState.OCCUPIED: self.occupied_confidence,
            SlotState.UNKNOWN: self.unknown_confidence,
        }[normalized]

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_confidence": self.free_confidence,
            "occupied_confidence": self.occupied_confidence,
            "unknown_confidence": self.unknown_confidence,
            "kind": self.kind,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ConfidenceScores":
        payload = _fields(
            value,
            "ConfidenceScores",
            required={
                "free_confidence",
                "occupied_confidence",
                "unknown_confidence",
                "kind",
                "calibrated",
            },
        )
        return cls(
            free_confidence=payload["free_confidence"],
            occupied_confidence=payload["occupied_confidence"],
            unknown_confidence=payload["unknown_confidence"],
            kind=payload["kind"],
            calibrated=payload["calibrated"],
        )


@dataclass(slots=True)
class MapSlot:
    slot_id: str
    polygon_map: list[list[float]]
    center_map: tuple[float, float] | list[float]
    heading_deg: float
    core_polygon_map: list[list[float]] = field(default_factory=list)
    margin_polygon_map: list[list[float]] = field(default_factory=list)
    adjacent_slot_ids: list[str] = field(default_factory=list)
    state: SlotState | str | None = None
    observed: bool | None = None
    distance_to_anchor_m: float | None = None

    def __post_init__(self) -> None:
        self.slot_id = _string(self.slot_id, "MapSlot.slot_id")
        self.polygon_map = _polygon(self.polygon_map, "MapSlot.polygon_map")
        self.center_map = _point(self.center_map, "MapSlot.center_map", 2)  # type: ignore[assignment]
        self.heading_deg = _number(self.heading_deg, "MapSlot.heading_deg")
        self.core_polygon_map = _polygon(
            self.core_polygon_map or self.polygon_map,
            "MapSlot.core_polygon_map",
        )
        self.margin_polygon_map = _polygon(
            self.margin_polygon_map or self.polygon_map,
            "MapSlot.margin_polygon_map",
        )
        self.adjacent_slot_ids = _strings(
            self.adjacent_slot_ids,
            "MapSlot.adjacent_slot_ids",
        )
        if self.slot_id in self.adjacent_slot_ids:
            raise ValueError("MapSlot.adjacent_slot_ids must not contain slot_id")
        self.state = _optional_enum(self.state, SlotState, "MapSlot.state")
        if self.observed is None:
            self.observed = self.state is not None
        else:
            self.observed = _boolean(self.observed, "MapSlot.observed")
        if bool(self.observed) != (self.state is not None):
            raise ValueError("MapSlot.observed must be true exactly when state is not null")
        self.distance_to_anchor_m = _optional_number(
            self.distance_to_anchor_m,
            "MapSlot.distance_to_anchor_m",
            minimum=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "polygon_map": copy.deepcopy(self.polygon_map),
            "core_polygon_map": copy.deepcopy(self.core_polygon_map),
            "margin_polygon_map": copy.deepcopy(self.margin_polygon_map),
            "center_map": list(self.center_map),
            "heading_deg": self.heading_deg,
            "adjacent_slot_ids": list(self.adjacent_slot_ids),
            "state": None if self.state is None else self.state.value,
            "observed": bool(self.observed),
            "distance_to_anchor_m": self.distance_to_anchor_m,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MapSlot":
        payload = _fields(
            value,
            "MapSlot",
            required={"slot_id", "polygon_map", "center_map", "heading_deg"},
            optional={
                "core_polygon_map",
                "margin_polygon_map",
                "adjacent_slot_ids",
                "state",
                "observed",
                "distance_to_anchor_m",
            },
        )
        return cls(
            slot_id=payload["slot_id"],
            polygon_map=payload["polygon_map"],
            center_map=payload["center_map"],
            heading_deg=payload["heading_deg"],
            core_polygon_map=payload.get("core_polygon_map", []),
            margin_polygon_map=payload.get("margin_polygon_map", []),
            adjacent_slot_ids=payload.get("adjacent_slot_ids", []),
            state=payload.get("state"),
            observed=payload.get("observed"),
            distance_to_anchor_m=payload.get("distance_to_anchor_m"),
        )


@dataclass(slots=True)
class SensorFrame:
    frame_id: int
    lidar_timestamp: float
    map_x: float
    map_y: float
    map_yaw_rad: float
    map_points_path: str | None = None
    lidar_path: str | None = None
    camera_frame_id: int | None = None
    camera_timestamp: float | None = None
    camera_image_path: str | None = None
    camera_lidar_dt_sec: float | None = None
    camera_match_valid: bool = False
    resources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_id = _integer(self.frame_id, "SensorFrame.frame_id", minimum=0)
        self.lidar_timestamp = _number(
            self.lidar_timestamp, "SensorFrame.lidar_timestamp"
        )
        self.map_x = _number(self.map_x, "SensorFrame.map_x")
        self.map_y = _number(self.map_y, "SensorFrame.map_y")
        self.map_yaw_rad = _number(self.map_yaw_rad, "SensorFrame.map_yaw_rad")
        self.map_points_path = _optional_string(
            self.map_points_path, "SensorFrame.map_points_path"
        )
        self.lidar_path = _optional_string(self.lidar_path, "SensorFrame.lidar_path")
        self.camera_frame_id = _optional_integer(
            self.camera_frame_id, "SensorFrame.camera_frame_id", minimum=0
        )
        self.camera_timestamp = _optional_number(
            self.camera_timestamp, "SensorFrame.camera_timestamp"
        )
        self.camera_image_path = _optional_string(
            self.camera_image_path, "SensorFrame.camera_image_path"
        )
        self.camera_lidar_dt_sec = _optional_number(
            self.camera_lidar_dt_sec, "SensorFrame.camera_lidar_dt_sec"
        )
        self.camera_match_valid = _boolean(
            self.camera_match_valid, "SensorFrame.camera_match_valid"
        )
        self.resources = _resources(self.resources, "SensorFrame.resources")
        if self.camera_match_valid and (
            self.camera_frame_id is None
            or self.camera_timestamp is None
            or self.camera_image_path is None
            or self.camera_lidar_dt_sec is None
        ):
            raise ValueError(
                "SensorFrame.camera_match_valid requires frame ID, timestamp, image path and dt"
            )

    @property
    def map_pose(self) -> tuple[float, float, float]:
        return self.map_x, self.map_y, self.map_yaw_rad

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "lidar_timestamp": self.lidar_timestamp,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "map_yaw_rad": self.map_yaw_rad,
            "map_points_path": self.map_points_path,
            "lidar_path": self.lidar_path,
            "camera_frame_id": self.camera_frame_id,
            "camera_timestamp": self.camera_timestamp,
            "camera_image_path": self.camera_image_path,
            "camera_lidar_dt_sec": self.camera_lidar_dt_sec,
            "camera_match_valid": self.camera_match_valid,
            "resources": dict(self.resources),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SensorFrame":
        payload = _fields(
            value,
            "SensorFrame",
            required={"frame_id", "lidar_timestamp", "map_x", "map_y", "map_yaw_rad"},
            optional={
                "map_points_path",
                "lidar_path",
                "camera_frame_id",
                "camera_timestamp",
                "camera_image_path",
                "camera_lidar_dt_sec",
                "camera_match_valid",
                "resources",
            },
        )
        return cls(
            frame_id=payload["frame_id"],
            lidar_timestamp=payload["lidar_timestamp"],
            map_x=payload["map_x"],
            map_y=payload["map_y"],
            map_yaw_rad=payload["map_yaw_rad"],
            map_points_path=payload.get("map_points_path"),
            lidar_path=payload.get("lidar_path"),
            camera_frame_id=payload.get("camera_frame_id"),
            camera_timestamp=payload.get("camera_timestamp"),
            camera_image_path=payload.get("camera_image_path"),
            camera_lidar_dt_sec=payload.get("camera_lidar_dt_sec"),
            camera_match_valid=payload.get("camera_match_valid", False),
            resources=payload.get("resources", {}),
        )


@dataclass(slots=True)
class SceneSnapshot:
    snapshot_id: str
    anchor_frame_id: int
    anchor_timestamp: float
    anchor_pose_map: tuple[float, float, float] | list[float]
    radius_m: float
    map_units_per_meter: float
    coordinate_frame: str | None
    slots: list[MapSlot]
    frames: list[SensorFrame]
    resources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.snapshot_id = _string(self.snapshot_id, "SceneSnapshot.snapshot_id")
        self.anchor_frame_id = _integer(
            self.anchor_frame_id, "SceneSnapshot.anchor_frame_id", minimum=0
        )
        self.anchor_timestamp = _number(
            self.anchor_timestamp, "SceneSnapshot.anchor_timestamp"
        )
        self.anchor_pose_map = _point(
            self.anchor_pose_map, "SceneSnapshot.anchor_pose_map", 3
        )  # type: ignore[assignment]
        self.radius_m = _number(
            self.radius_m, "SceneSnapshot.radius_m", minimum=1e-12
        )
        self.map_units_per_meter = _number(
            self.map_units_per_meter,
            "SceneSnapshot.map_units_per_meter",
            minimum=1e-12,
        )
        self.coordinate_frame = _optional_string(
            self.coordinate_frame, "SceneSnapshot.coordinate_frame"
        )
        self.resources = _resources(self.resources, "SceneSnapshot.resources")

        normalized_slots: list[MapSlot] = []
        for index, item in enumerate(_sequence(self.slots, "SceneSnapshot.slots")):
            slot = item if isinstance(item, MapSlot) else MapSlot.from_dict(item)
            delta_x = slot.center_map[0] - self.anchor_pose_map[0]
            delta_y = slot.center_map[1] - self.anchor_pose_map[1]
            calculated_distance = math.hypot(delta_x, delta_y) / self.map_units_per_meter
            if slot.distance_to_anchor_m is None:
                slot.distance_to_anchor_m = calculated_distance
            elif not math.isclose(
                slot.distance_to_anchor_m,
                calculated_distance,
                rel_tol=1e-7,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"SceneSnapshot.slots[{index}].distance_to_anchor_m does not match geometry"
                )
            if calculated_distance > self.radius_m + 1e-6:
                raise ValueError(
                    f"SceneSnapshot.slots[{index}] lies outside radius_m"
                )
            normalized_slots.append(slot)
        if len({slot.slot_id for slot in normalized_slots}) != len(normalized_slots):
            raise ValueError("SceneSnapshot.slots contains duplicate slot IDs")
        self.slots = sorted(normalized_slots, key=lambda item: item.slot_id)

        normalized_frames: list[SensorFrame] = []
        for item in _sequence(self.frames, "SceneSnapshot.frames"):
            normalized_frames.append(
                item if isinstance(item, SensorFrame) else SensorFrame.from_dict(item)
            )
        if not 1 <= len(normalized_frames) <= 100:
            raise ValueError("SceneSnapshot.frames must contain 1 to 100 causal frames")
        if len({frame.frame_id for frame in normalized_frames}) != len(normalized_frames):
            raise ValueError("SceneSnapshot.frames contains duplicate frame IDs")
        normalized_frames.sort(key=lambda item: item.frame_id)
        if normalized_frames[-1].frame_id != self.anchor_frame_id:
            raise ValueError("SceneSnapshot.anchor_frame_id must be the last causal frame")
        anchor = normalized_frames[-1]
        if not math.isclose(anchor.lidar_timestamp, self.anchor_timestamp, abs_tol=1e-6):
            raise ValueError("SceneSnapshot.anchor_timestamp does not match anchor frame")
        for actual, expected in zip(anchor.map_pose, self.anchor_pose_map):
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("SceneSnapshot.anchor_pose_map does not match anchor frame")
        if any(frame.lidar_timestamp > self.anchor_timestamp + 1e-9 for frame in normalized_frames):
            raise ValueError("SceneSnapshot.frames must be causal at or before t0")
        self.frames = normalized_frames

    @property
    def t0_frame(self) -> SensorFrame:
        return self.frames[-1]

    def slot_by_id(self, slot_id: str) -> MapSlot:
        normalized = _string(slot_id, "slot_id")
        for slot in self.slots:
            if slot.slot_id == normalized:
                return slot
        raise KeyError(normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "anchor_frame_id": self.anchor_frame_id,
            "anchor_timestamp": self.anchor_timestamp,
            "anchor_pose_map": list(self.anchor_pose_map),
            "radius_m": self.radius_m,
            "map_units_per_meter": self.map_units_per_meter,
            "coordinate_frame": self.coordinate_frame,
            "slots": [slot.to_dict() for slot in self.slots],
            "frames": [frame.to_dict() for frame in self.frames],
            "resources": dict(self.resources),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SceneSnapshot":
        payload = _fields(
            value,
            "SceneSnapshot",
            required={
                "snapshot_id",
                "anchor_frame_id",
                "anchor_timestamp",
                "anchor_pose_map",
                "radius_m",
                "map_units_per_meter",
                "coordinate_frame",
                "slots",
                "frames",
            },
            optional={"resources"},
        )
        return cls(
            snapshot_id=payload["snapshot_id"],
            anchor_frame_id=payload["anchor_frame_id"],
            anchor_timestamp=payload["anchor_timestamp"],
            anchor_pose_map=payload["anchor_pose_map"],
            radius_m=payload["radius_m"],
            map_units_per_meter=payload["map_units_per_meter"],
            coordinate_frame=payload["coordinate_frame"],
            slots=[MapSlot.from_dict(item) for item in _sequence(payload["slots"], "SceneSnapshot.slots")],
            frames=[SensorFrame.from_dict(item) for item in _sequence(payload["frames"], "SceneSnapshot.frames")],
            resources=payload.get("resources", {}),
        )


@dataclass(slots=True)
class FovResult:
    visibility: FovVisibility | str = FovVisibility.NOT_CHECKED
    confidence: float = 0.0
    reason: str = "not_checked"
    camera_frame_id: int | None = None
    candidate_regions: list[list[float]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.visibility = _enum(self.visibility, FovVisibility, "FovResult.visibility")
        self.confidence = _number(
            self.confidence, "FovResult.confidence", minimum=0.0, maximum=1.0
        )
        self.reason = _string(self.reason, "FovResult.reason")
        self.camera_frame_id = _optional_integer(
            self.camera_frame_id, "FovResult.camera_frame_id", minimum=0
        )
        normalized_regions: list[list[float]] = []
        for index, raw in enumerate(
            _sequence(self.candidate_regions, "FovResult.candidate_regions")
        ):
            region = _point(raw, f"FovResult.candidate_regions[{index}]", 4)
            if region[0] < 0.0 or region[1] < 0.0 or region[2] <= region[0] or region[3] <= region[1]:
                raise ValueError(
                    "FovResult candidate regions must be [x_min, y_min, x_max, y_max]"
                )
            normalized_regions.append(list(region))
        self.candidate_regions = normalized_regions
        self.details = _json_value(self.details, "FovResult.details")
        if not isinstance(self.details, dict):
            raise ValueError("FovResult.details must be a JSON object")
        if self.visibility is FovVisibility.NOT_CHECKED and (
            self.confidence != 0.0 or self.camera_frame_id is not None or self.candidate_regions
        ):
            raise ValueError("an unchecked FOV result cannot contain checked observations")

    @property
    def checked(self) -> bool:
        return self.visibility is not FovVisibility.NOT_CHECKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "visibility": self.visibility.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "camera_frame_id": self.camera_frame_id,
            "candidate_regions": copy.deepcopy(self.candidate_regions),
            "details": copy.deepcopy(self.details),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FovResult":
        payload = _fields(
            value,
            "FovResult",
            required={"visibility", "confidence", "reason"},
            optional={"camera_frame_id", "candidate_regions", "details"},
        )
        return cls(
            visibility=payload["visibility"],
            confidence=payload["confidence"],
            reason=payload["reason"],
            camera_frame_id=payload.get("camera_frame_id"),
            candidate_regions=payload.get("candidate_regions", []),
            details=payload.get("details", {}),
        )


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    tool_name: str
    round_index: int
    status: str
    artifact_paths: list[str]
    summary: str
    metadata: dict[str, Any]
    modality: str = "unknown"
    supports_state: SlotState | str | None = None
    scores: ConfidenceScores | None = None
    reason_codes: list[str] = field(default_factory=list)
    resource_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.evidence_id = _string(self.evidence_id, "EvidenceRecord.evidence_id")
        self.tool_name = _string(self.tool_name, "EvidenceRecord.tool_name")
        self.round_index = _integer(
            self.round_index, "EvidenceRecord.round_index", minimum=0, maximum=3
        )
        self.status = _string(self.status, "EvidenceRecord.status")
        if self.status not in {"ok", "unavailable", "failed"}:
            raise ValueError("EvidenceRecord.status must be ok, unavailable or failed")
        self.artifact_paths = _strings(
            self.artifact_paths, "EvidenceRecord.artifact_paths"
        )
        self.summary = _string(self.summary, "EvidenceRecord.summary")
        self.metadata = _json_value(self.metadata, "EvidenceRecord.metadata")
        if not isinstance(self.metadata, dict):
            raise ValueError("EvidenceRecord.metadata must be a JSON object")
        self.modality = _string(self.modality, "EvidenceRecord.modality")
        self.supports_state = _optional_enum(
            self.supports_state, SlotState, "EvidenceRecord.supports_state"
        )
        if self.scores is not None and not isinstance(self.scores, ConfidenceScores):
            self.scores = ConfidenceScores.from_dict(self.scores)
        self.reason_codes = _strings(self.reason_codes, "EvidenceRecord.reason_codes")
        self.resource_keys = _strings(self.resource_keys, "EvidenceRecord.resource_keys")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "tool_name": self.tool_name,
            "round_index": self.round_index,
            "status": self.status,
            "artifact_paths": list(self.artifact_paths),
            "summary": self.summary,
            "metadata": copy.deepcopy(self.metadata),
            "modality": self.modality,
            "supports_state": None if self.supports_state is None else self.supports_state.value,
            "scores": None if self.scores is None else self.scores.to_dict(),
            "reason_codes": list(self.reason_codes),
            "resource_keys": list(self.resource_keys),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceRecord":
        payload = _fields(
            value,
            "EvidenceRecord",
            required={
                "evidence_id",
                "tool_name",
                "round_index",
                "status",
                "artifact_paths",
                "summary",
                "metadata",
            },
            optional={
                "modality",
                "supports_state",
                "scores",
                "reason_codes",
                "resource_keys",
            },
        )
        raw_scores = payload.get("scores")
        return cls(
            evidence_id=payload["evidence_id"],
            tool_name=payload["tool_name"],
            round_index=payload["round_index"],
            status=payload["status"],
            artifact_paths=payload["artifact_paths"],
            summary=payload["summary"],
            metadata=payload["metadata"],
            modality=payload.get("modality", "unknown"),
            supports_state=payload.get("supports_state"),
            scores=None if raw_scores is None else ConfidenceScores.from_dict(raw_scores),
            reason_codes=payload.get("reason_codes", []),
            resource_keys=payload.get("resource_keys", []),
        )


@dataclass(slots=True)
class ReasoningRound:
    round_index: int
    reasoning_summary: str
    state_after: SlotState | str
    scores_after: ConfidenceScores
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    observation_summary: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    resolved_unknown_reasons: list[str] = field(default_factory=list)
    remaining_unknown_reasons: list[str] = field(default_factory=list)
    localization: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.round_index = _integer(
            self.round_index, "ReasoningRound.round_index", minimum=1, maximum=3
        )
        self.reasoning_summary = _string(
            self.reasoning_summary, "ReasoningRound.reasoning_summary"
        )
        self.state_after = _enum(self.state_after, SlotState, "ReasoningRound.state_after")
        if not isinstance(self.scores_after, ConfidenceScores):
            self.scores_after = ConfidenceScores.from_dict(self.scores_after)
        self.tool_name = _optional_string(self.tool_name, "ReasoningRound.tool_name")
        self.tool_arguments = _json_value(
            self.tool_arguments, "ReasoningRound.tool_arguments"
        )
        if not isinstance(self.tool_arguments, dict):
            raise ValueError("ReasoningRound.tool_arguments must be a JSON object")
        self.observation_summary = _string(
            self.observation_summary,
            "ReasoningRound.observation_summary",
            allow_empty=True,
        )
        self.evidence_ids = _strings(self.evidence_ids, "ReasoningRound.evidence_ids")
        self.resolved_unknown_reasons = _strings(
            self.resolved_unknown_reasons,
            "ReasoningRound.resolved_unknown_reasons",
        )
        self.remaining_unknown_reasons = _strings(
            self.remaining_unknown_reasons,
            "ReasoningRound.remaining_unknown_reasons",
        )
        self.localization = _json_value(
            self.localization, "ReasoningRound.localization"
        )
        if not isinstance(self.localization, dict):
            raise ValueError("ReasoningRound.localization must be a JSON object")
        overlap = set(self.resolved_unknown_reasons) & set(self.remaining_unknown_reasons)
        if overlap:
            raise ValueError(
                "ReasoningRound resolved and remaining unknown reasons must be disjoint"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "reasoning_summary": self.reasoning_summary,
            "tool_name": self.tool_name,
            "tool_arguments": copy.deepcopy(self.tool_arguments),
            "observation_summary": self.observation_summary,
            "evidence_ids": list(self.evidence_ids),
            "state_after": self.state_after.value,
            "scores_after": self.scores_after.to_dict(),
            "resolved_unknown_reasons": list(self.resolved_unknown_reasons),
            "remaining_unknown_reasons": list(self.remaining_unknown_reasons),
            "localization": copy.deepcopy(self.localization),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReasoningRound":
        payload = _fields(
            value,
            "ReasoningRound",
            required={"round_index", "reasoning_summary", "state_after", "scores_after"},
            optional={
                "tool_name",
                "tool_arguments",
                "observation_summary",
                "evidence_ids",
                "resolved_unknown_reasons",
                "remaining_unknown_reasons",
                "localization",
            },
        )
        return cls(
            round_index=payload["round_index"],
            reasoning_summary=payload["reasoning_summary"],
            state_after=payload["state_after"],
            scores_after=ConfidenceScores.from_dict(payload["scores_after"]),
            tool_name=payload.get("tool_name"),
            tool_arguments=payload.get("tool_arguments", {}),
            observation_summary=payload.get("observation_summary", ""),
            evidence_ids=payload.get("evidence_ids", []),
            resolved_unknown_reasons=payload.get("resolved_unknown_reasons", []),
            remaining_unknown_reasons=payload.get("remaining_unknown_reasons", []),
            localization=payload.get("localization", {}),
        )


_TERMINAL_CASE_STATUSES = {
    CaseStatus.RESOLVED,
    CaseStatus.EXHAUSTED,
    CaseStatus.SELECTED,
}


@dataclass(slots=True)
class SlotCase:
    case_id: str
    snapshot_id: str
    snapshot_frame_id: int
    snapshot_timestamp: float
    snapshot_pose_map: tuple[float, float, float] | list[float]
    slot: MapSlot
    evidence_anchor_frame_id: int
    evidence_frame_ids: list[int]
    part1_state: SlotState | str
    part1_scores: ConfidenceScores
    current_state: SlotState | str
    current_scores: ConfidenceScores
    decision_reason: str
    unknown_reasons: list[str]
    status: CaseStatus | str = CaseStatus.PENDING
    fov: FovResult = field(default_factory=FovResult)
    resources: dict[str, str] = field(default_factory=dict)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    rounds: list[ReasoningRound] = field(default_factory=list)
    max_rounds: int = 3
    final_state: SlotState | str | None = None
    final_scores: ConfidenceScores | None = None
    final_reason: str | None = None
    unresolved_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.case_id = _string(self.case_id, "SlotCase.case_id")
        self.snapshot_id = _string(self.snapshot_id, "SlotCase.snapshot_id")
        self.snapshot_frame_id = _integer(
            self.snapshot_frame_id, "SlotCase.snapshot_frame_id", minimum=0
        )
        self.snapshot_timestamp = _number(
            self.snapshot_timestamp, "SlotCase.snapshot_timestamp"
        )
        self.snapshot_pose_map = _point(
            self.snapshot_pose_map, "SlotCase.snapshot_pose_map", 3
        )  # type: ignore[assignment]
        if not isinstance(self.slot, MapSlot):
            self.slot = MapSlot.from_dict(self.slot)
        self.evidence_anchor_frame_id = _integer(
            self.evidence_anchor_frame_id,
            "SlotCase.evidence_anchor_frame_id",
            minimum=0,
        )
        self.evidence_frame_ids = sorted(
            _integers(self.evidence_frame_ids, "SlotCase.evidence_frame_ids")
        )
        if not self.evidence_frame_ids or len(self.evidence_frame_ids) > 15:
            raise ValueError("SlotCase.evidence_frame_ids must contain 1-15 frame IDs")
        if self.evidence_anchor_frame_id not in self.evidence_frame_ids:
            raise ValueError("SlotCase.evidence_anchor_frame_id must be an evidence frame")
        self.part1_state = _enum(self.part1_state, SlotState, "SlotCase.part1_state")
        if self.part1_state not in {SlotState.FREE, SlotState.UNKNOWN}:
            raise ValueError("SlotCase Part1 candidates must start as free or unknown")
        if not isinstance(self.part1_scores, ConfidenceScores):
            self.part1_scores = ConfidenceScores.from_dict(self.part1_scores)
        self.current_state = _enum(self.current_state, SlotState, "SlotCase.current_state")
        if not isinstance(self.current_scores, ConfidenceScores):
            self.current_scores = ConfidenceScores.from_dict(self.current_scores)
        self.decision_reason = _string(self.decision_reason, "SlotCase.decision_reason")
        self.unknown_reasons = _strings(
            self.unknown_reasons, "SlotCase.unknown_reasons"
        )
        if self.part1_state is SlotState.UNKNOWN and not self.unknown_reasons:
            raise ValueError("an initially unknown SlotCase requires unknown_reasons")
        if self.part1_state is SlotState.FREE and self.unknown_reasons:
            raise ValueError("an initially free SlotCase must not contain unknown_reasons")
        if self.slot.state is not self.part1_state or not self.slot.observed:
            raise ValueError("SlotCase.slot state must match the observed Part1 state")
        self.status = _enum(self.status, CaseStatus, "SlotCase.status")
        if not isinstance(self.fov, FovResult):
            self.fov = FovResult.from_dict(self.fov)
        self.resources = _resources(self.resources, "SlotCase.resources")
        self.max_rounds = _integer(
            self.max_rounds, "SlotCase.max_rounds", minimum=1, maximum=3
        )

        normalized_evidence: list[EvidenceRecord] = []
        for item in _sequence(self.evidence, "SlotCase.evidence"):
            normalized_evidence.append(
                item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_dict(item)
            )
        if len({item.evidence_id for item in normalized_evidence}) != len(normalized_evidence):
            raise ValueError("SlotCase.evidence contains duplicate evidence IDs")
        if any(item.round_index > self.max_rounds for item in normalized_evidence):
            raise ValueError("SlotCase evidence round exceeds max_rounds")
        self.evidence = normalized_evidence

        normalized_rounds: list[ReasoningRound] = []
        for item in _sequence(self.rounds, "SlotCase.rounds"):
            normalized_rounds.append(
                item if isinstance(item, ReasoningRound) else ReasoningRound.from_dict(item)
            )
        if len(normalized_rounds) > self.max_rounds:
            raise ValueError("SlotCase.rounds exceeds max_rounds")
        expected_indices = list(range(1, len(normalized_rounds) + 1))
        if [item.round_index for item in normalized_rounds] != expected_indices:
            raise ValueError("SlotCase.rounds must have consecutive 1-based indices")
        known_evidence_ids = {item.evidence_id for item in normalized_evidence}
        unknown_refs = sorted(
            {
                evidence_id
                for item in normalized_rounds
                for evidence_id in item.evidence_ids
                if evidence_id not in known_evidence_ids
            }
        )
        if unknown_refs:
            raise ValueError("SlotCase.rounds reference unknown evidence IDs")
        self.rounds = normalized_rounds

        self.final_state = _optional_enum(
            self.final_state, SlotState, "SlotCase.final_state"
        )
        if self.final_scores is not None and not isinstance(
            self.final_scores, ConfidenceScores
        ):
            self.final_scores = ConfidenceScores.from_dict(self.final_scores)
        self.final_reason = _optional_string(self.final_reason, "SlotCase.final_reason")
        self.unresolved_reasons = _strings(
            self.unresolved_reasons, "SlotCase.unresolved_reasons"
        )
        terminal = self.status in _TERMINAL_CASE_STATUSES
        final_fields_present = (
            self.final_state is not None
            and self.final_scores is not None
            and self.final_reason is not None
        )
        if terminal != final_fields_present:
            raise ValueError("terminal SlotCase status and final fields must agree")
        if terminal:
            if self.current_state is not self.final_state or self.current_scores != self.final_scores:
                raise ValueError("terminal SlotCase current and final state/scores must agree")
            if self.status is CaseStatus.EXHAUSTED and self.final_state is not SlotState.UNKNOWN:
                raise ValueError("an exhausted SlotCase must finish as unknown")
            if self.status is CaseStatus.SELECTED and self.final_state is not SlotState.FREE:
                raise ValueError("a selected SlotCase must finish as free")

    @property
    def slot_id(self) -> str:
        return self.slot.slot_id

    @property
    def rounds_remaining(self) -> int:
        return self.max_rounds - len(self.rounds)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_CASE_STATUSES

    def update_fov(self, result: FovResult | Mapping[str, Any]) -> None:
        if self.terminal:
            raise ValueError("cannot update FOV on a terminal SlotCase")
        normalized = result if isinstance(result, FovResult) else FovResult.from_dict(result)
        if not normalized.checked:
            raise ValueError("update_fov requires a checked FOV result")
        self.fov = normalized
        if self.status is CaseStatus.PENDING:
            self.status = CaseStatus.IN_PROGRESS

    def record_evidence(self, record: EvidenceRecord | Mapping[str, Any]) -> None:
        if self.terminal:
            raise ValueError("cannot record evidence on a terminal SlotCase")
        normalized = (
            record if isinstance(record, EvidenceRecord) else EvidenceRecord.from_dict(record)
        )
        if normalized.round_index > self.max_rounds:
            raise ValueError("evidence round exceeds SlotCase.max_rounds")
        if any(item.evidence_id == normalized.evidence_id for item in self.evidence):
            raise ValueError(f"duplicate evidence_id: {normalized.evidence_id}")
        missing_resources = sorted(set(normalized.resource_keys) - set(self.resources))
        if missing_resources:
            raise ValueError(
                "evidence references unknown SlotCase resource key(s): "
                + ", ".join(missing_resources)
            )
        self.evidence.append(normalized)
        if self.status is CaseStatus.PENDING:
            self.status = CaseStatus.IN_PROGRESS

    def record_round(self, reasoning_round: ReasoningRound | Mapping[str, Any]) -> None:
        if self.terminal:
            raise ValueError("cannot record a round on a terminal SlotCase")
        normalized = (
            reasoning_round
            if isinstance(reasoning_round, ReasoningRound)
            else ReasoningRound.from_dict(reasoning_round)
        )
        expected = len(self.rounds) + 1
        if normalized.round_index != expected:
            raise ValueError(f"next reasoning round must have round_index={expected}")
        if normalized.round_index > self.max_rounds:
            raise ValueError("SlotCase has reached max_rounds")
        known_evidence = {item.evidence_id for item in self.evidence}
        missing = sorted(set(normalized.evidence_ids) - known_evidence)
        if missing:
            raise ValueError("reasoning round references unknown evidence IDs")
        self.rounds.append(normalized)
        self.current_state = normalized.state_after
        self.current_scores = copy.deepcopy(normalized.scores_after)
        self.unresolved_reasons = list(normalized.remaining_unknown_reasons)
        self.status = CaseStatus.IN_PROGRESS

    def record(self, item: EvidenceRecord | ReasoningRound | Mapping[str, Any]) -> None:
        """Record one evidence or reasoning object.

        Mappings are dispatched by their strict field signature.  Calling the
        explicit methods is preferred when the caller already knows the type.
        """

        if isinstance(item, EvidenceRecord):
            self.record_evidence(item)
            return
        if isinstance(item, ReasoningRound):
            self.record_round(item)
            return
        payload = _mapping(item, "SlotCase.record")
        if "evidence_id" in payload:
            self.record_evidence(payload)
        elif "reasoning_summary" in payload:
            self.record_round(payload)
        else:
            raise ValueError("record mapping is neither EvidenceRecord nor ReasoningRound")

    def update_state(
        self,
        state: SlotState | str,
        scores: ConfidenceScores | Mapping[str, Any],
        *,
        unresolved_reasons: Sequence[str] | None = None,
    ) -> None:
        if self.terminal:
            raise ValueError("cannot update a terminal SlotCase")
        self.current_state = _enum(state, SlotState, "state")
        self.current_scores = (
            scores if isinstance(scores, ConfidenceScores) else ConfidenceScores.from_dict(scores)
        )
        if unresolved_reasons is not None:
            self.unresolved_reasons = _strings(
                unresolved_reasons, "unresolved_reasons"
            )
        if self.status is CaseStatus.PENDING:
            self.status = CaseStatus.IN_PROGRESS

    def update(
        self,
        *,
        state: SlotState | str | None = None,
        scores: ConfidenceScores | Mapping[str, Any] | None = None,
        fov: FovResult | Mapping[str, Any] | None = None,
        unresolved_reasons: Sequence[str] | None = None,
    ) -> None:
        if fov is not None:
            self.update_fov(fov)
        if (state is None) != (scores is None):
            raise ValueError("state and scores must be updated together")
        if state is not None and scores is not None:
            self.update_state(
                state,
                scores,
                unresolved_reasons=unresolved_reasons,
            )
        elif unresolved_reasons is not None:
            self.unresolved_reasons = _strings(
                unresolved_reasons, "unresolved_reasons"
            )

    def finalize(
        self,
        state: SlotState | str,
        scores: ConfidenceScores | Mapping[str, Any],
        reason: str,
        *,
        unresolved_reasons: Sequence[str] = (),
        exhausted: bool | None = None,
    ) -> None:
        if self.terminal:
            raise ValueError("SlotCase is already terminal")
        normalized_state = _enum(state, SlotState, "state")
        normalized_scores = (
            scores if isinstance(scores, ConfidenceScores) else ConfidenceScores.from_dict(scores)
        )
        normalized_reason = _string(reason, "reason")
        normalized_unresolved = _strings(unresolved_reasons, "unresolved_reasons")
        if exhausted is None:
            exhausted = normalized_state is SlotState.UNKNOWN
        exhausted = _boolean(exhausted, "exhausted")
        if exhausted and normalized_state is not SlotState.UNKNOWN:
            raise ValueError("only an unknown conclusion may be exhausted")
        if not exhausted and normalized_state is SlotState.UNKNOWN:
            raise ValueError("an unknown conclusion must be marked exhausted")
        self.current_state = normalized_state
        self.current_scores = copy.deepcopy(normalized_scores)
        self.final_state = normalized_state
        self.final_scores = copy.deepcopy(normalized_scores)
        self.final_reason = normalized_reason
        self.unresolved_reasons = normalized_unresolved
        self.status = CaseStatus.EXHAUSTED if exhausted else CaseStatus.RESOLVED

    def mark_selected(self) -> None:
        if self.status is not CaseStatus.RESOLVED or self.final_state is not SlotState.FREE:
            raise ValueError("only a resolved free SlotCase may be selected")
        self.status = CaseStatus.SELECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_frame_id": self.snapshot_frame_id,
            "snapshot_timestamp": self.snapshot_timestamp,
            "snapshot_pose_map": list(self.snapshot_pose_map),
            "slot": self.slot.to_dict(),
            "evidence_anchor_frame_id": self.evidence_anchor_frame_id,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "part1_state": self.part1_state.value,
            "part1_scores": self.part1_scores.to_dict(),
            "current_state": self.current_state.value,
            "current_scores": self.current_scores.to_dict(),
            "decision_reason": self.decision_reason,
            "unknown_reasons": list(self.unknown_reasons),
            "status": self.status.value,
            "fov": self.fov.to_dict(),
            "resources": dict(self.resources),
            "evidence": [item.to_dict() for item in self.evidence],
            "rounds": [item.to_dict() for item in self.rounds],
            "max_rounds": self.max_rounds,
            "final_state": None if self.final_state is None else self.final_state.value,
            "final_scores": None if self.final_scores is None else self.final_scores.to_dict(),
            "final_reason": self.final_reason,
            "unresolved_reasons": list(self.unresolved_reasons),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SlotCase":
        payload = _fields(
            value,
            "SlotCase",
            required={
                "case_id",
                "snapshot_id",
                "snapshot_frame_id",
                "snapshot_timestamp",
                "snapshot_pose_map",
                "slot",
                "evidence_anchor_frame_id",
                "evidence_frame_ids",
                "part1_state",
                "part1_scores",
                "current_state",
                "current_scores",
                "decision_reason",
                "unknown_reasons",
            },
            optional={
                "status",
                "fov",
                "resources",
                "evidence",
                "rounds",
                "max_rounds",
                "final_state",
                "final_scores",
                "final_reason",
                "unresolved_reasons",
            },
        )
        raw_final_scores = payload.get("final_scores")
        return cls(
            case_id=payload["case_id"],
            snapshot_id=payload["snapshot_id"],
            snapshot_frame_id=payload["snapshot_frame_id"],
            snapshot_timestamp=payload["snapshot_timestamp"],
            snapshot_pose_map=payload["snapshot_pose_map"],
            slot=MapSlot.from_dict(payload["slot"]),
            evidence_anchor_frame_id=payload["evidence_anchor_frame_id"],
            evidence_frame_ids=payload["evidence_frame_ids"],
            part1_state=payload["part1_state"],
            part1_scores=ConfidenceScores.from_dict(payload["part1_scores"]),
            current_state=payload["current_state"],
            current_scores=ConfidenceScores.from_dict(payload["current_scores"]),
            decision_reason=payload["decision_reason"],
            unknown_reasons=payload["unknown_reasons"],
            status=payload.get("status", CaseStatus.PENDING.value),
            fov=FovResult.from_dict(payload.get("fov", FovResult().to_dict())),
            resources=payload.get("resources", {}),
            evidence=[
                EvidenceRecord.from_dict(item)
                for item in _sequence(payload.get("evidence", []), "SlotCase.evidence")
            ],
            rounds=[
                ReasoningRound.from_dict(item)
                for item in _sequence(payload.get("rounds", []), "SlotCase.rounds")
            ],
            max_rounds=payload.get("max_rounds", 3),
            final_state=payload.get("final_state"),
            final_scores=(
                None
                if raw_final_scores is None
                else ConfidenceScores.from_dict(raw_final_scores)
            ),
            final_reason=payload.get("final_reason"),
            unresolved_reasons=payload.get("unresolved_reasons", []),
        )


@dataclass(slots=True)
class Part1Output:
    scene: SceneSnapshot
    slot_cases: list[SlotCase]
    schema_version: str = PART1_OUTPUT_SCHEMA_VERSION
    producer: str = "parking_slot_agent_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.scene, SceneSnapshot):
            self.scene = SceneSnapshot.from_dict(self.scene)
        self.schema_version = _string(
            self.schema_version, "Part1Output.schema_version"
        )
        if self.schema_version != PART1_OUTPUT_SCHEMA_VERSION:
            raise ValueError(
                f"Part1Output.schema_version must be {PART1_OUTPUT_SCHEMA_VERSION}"
            )
        self.producer = _string(self.producer, "Part1Output.producer")
        normalized: list[SlotCase] = []
        for item in _sequence(self.slot_cases, "Part1Output.slot_cases"):
            normalized.append(item if isinstance(item, SlotCase) else SlotCase.from_dict(item))
        if len({item.case_id for item in normalized}) != len(normalized):
            raise ValueError("Part1Output.slot_cases contains duplicate case IDs")
        if len({item.slot_id for item in normalized}) != len(normalized):
            raise ValueError("Part1Output.slot_cases contains duplicate slot IDs")
        scene_by_id = {slot.slot_id: slot for slot in self.scene.slots}
        scene_frame_ids = {frame.frame_id for frame in self.scene.frames}
        for case in normalized:
            if case.snapshot_id != self.scene.snapshot_id:
                raise ValueError("SlotCase snapshot_id does not match Part1Output scene")
            if case.snapshot_frame_id != self.scene.anchor_frame_id:
                raise ValueError("SlotCase snapshot_frame_id does not match scene t0")
            if not math.isclose(
                case.snapshot_timestamp, self.scene.anchor_timestamp, abs_tol=1e-6
            ):
                raise ValueError("SlotCase snapshot_timestamp does not match scene t0")
            if any(
                not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
                for actual, expected in zip(case.snapshot_pose_map, self.scene.anchor_pose_map)
            ):
                raise ValueError("SlotCase snapshot_pose_map does not match scene t0")
            scene_slot = scene_by_id.get(case.slot_id)
            if scene_slot is None:
                raise ValueError("SlotCase references a slot outside the SceneSnapshot")
            if case.slot.to_dict() != scene_slot.to_dict():
                raise ValueError("SlotCase slot geometry does not match SceneSnapshot slot")
            if scene_slot.state is not case.part1_state:
                raise ValueError("SlotCase Part1 state does not match SceneSnapshot slot")
            if case.evidence_anchor_frame_id not in scene_frame_ids:
                raise ValueError("SlotCase evidence anchor is outside the causal scene")
            if not set(case.evidence_frame_ids).issubset(scene_frame_ids):
                raise ValueError("SlotCase evidence frames are outside the causal scene")

        expected_candidate_ids = {
            slot.slot_id
            for slot in self.scene.slots
            if slot.state in {SlotState.FREE, SlotState.UNKNOWN}
        }
        actual_candidate_ids = {case.slot_id for case in normalized}
        if actual_candidate_ids != expected_candidate_ids:
            raise ValueError(
                "Part1Output.slot_cases must contain every and only free/unknown scene slot"
            )
        priority = {SlotState.FREE: 0, SlotState.UNKNOWN: 1}
        normalized.sort(
            key=lambda item: (
                priority[item.part1_state],
                item.slot.distance_to_anchor_m
                if item.slot.distance_to_anchor_m is not None
                else float("inf"),
                item.slot_id,
            )
        )
        self.slot_cases = normalized

    @property
    def candidate_cases(self) -> list[SlotCase]:
        return self.slot_cases

    @property
    def free_cases(self) -> list[SlotCase]:
        return [case for case in self.slot_cases if case.part1_state is SlotState.FREE]

    @property
    def unknown_cases(self) -> list[SlotCase]:
        return [case for case in self.slot_cases if case.part1_state is SlotState.UNKNOWN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "scene": self.scene.to_dict(),
            "slot_cases": [case.to_dict() for case in self.slot_cases],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "Part1Output":
        payload = _fields(
            value,
            "Part1Output",
            required={"schema_version", "producer", "scene", "slot_cases"},
        )
        return cls(
            scene=SceneSnapshot.from_dict(payload["scene"]),
            slot_cases=[
                SlotCase.from_dict(item)
                for item in _sequence(payload["slot_cases"], "Part1Output.slot_cases")
            ],
            schema_version=payload["schema_version"],
            producer=payload["producer"],
        )


__all__ = [
    "PART1_OUTPUT_SCHEMA_VERSION",
    "CaseStatus",
    "ConfidenceScores",
    "EvidenceRecord",
    "FovResult",
    "FovVisibility",
    "MapSlot",
    "Part1Output",
    "ReasoningRound",
    "SceneSnapshot",
    "SensorFrame",
    "SlotCase",
    "SlotState",
]
