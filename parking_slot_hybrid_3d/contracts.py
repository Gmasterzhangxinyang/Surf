from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class ScopeStatus(str, Enum):
    IN_ROUTE = "in_route_scope"
    PARTIAL_ROUTE = "partial_route_scope"
    OUT_OF_ROUTE = "out_of_route_scope"


class DecisionState(str, Enum):
    OCCUPIED = "occupied"
    FREE = "free"
    UNKNOWN = "unknown"


def _readonly_array(value: np.ndarray, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if columns is not None and (array.ndim != 2 or array.shape[1] != columns):
        raise ValueError(f"expected a two-dimensional array with {columns} columns")
    if not np.isfinite(array).all():
        raise ValueError("array contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    map_x: float
    map_y: float
    map_yaw: float
    map_points_path: Path | None
    lidar_path: Path | None = None
    lidar_timestamp: float | None = None
    camera_frame: int | None = None
    camera_image_path: Path | None = None
    camera_timestamp: float | None = None
    camera_lidar_dt_sec: float | None = None
    camera_match_valid: bool = False


@dataclass(frozen=True)
class KnownSlot:
    slot_id: str
    polygon_map: np.ndarray
    core_polygon_map: np.ndarray
    margin_polygon_map: np.ndarray
    center_map: np.ndarray
    heading_deg: float
    adjacent_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.slot_id:
            raise ValueError("slot_id is required")
        object.__setattr__(self, "polygon_map", _readonly_array(self.polygon_map, columns=2))
        object.__setattr__(self, "core_polygon_map", _readonly_array(self.core_polygon_map, columns=2))
        object.__setattr__(self, "margin_polygon_map", _readonly_array(self.margin_polygon_map, columns=2))
        center = _readonly_array(np.asarray(self.center_map, dtype=np.float64).reshape(1, 2), columns=2).reshape(2)
        center.setflags(write=False)
        object.__setattr__(self, "center_map", center)
        object.__setattr__(self, "adjacent_slots", tuple(sorted(str(value) for value in self.adjacent_slots)))


@dataclass(frozen=True)
class MetricSlot:
    slot_id: str
    center_map: np.ndarray
    long_axis_map: np.ndarray
    short_axis_map: np.ndarray
    polygon_local_m: np.ndarray
    core_polygon_local_m: np.ndarray
    margin_polygon_local_m: np.ndarray
    adjacent_slots: tuple[str, ...]
    map_units_per_meter: float


@dataclass(frozen=True)
class GroundModel:
    method: str
    valid: bool
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    candidate_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    residual_p95_m: float = float("inf")
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    origin_local_xyz: np.ndarray
    points_local_xyzi: np.ndarray
    ray_endpoints_local_xyz: np.ndarray
    ground_model: GroundModel
    quality_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotAccumulation:
    slot_id: str
    anchor_frame: int
    selected_frames: tuple[int, ...]
    points_local_xyzi: np.ndarray
    point_frame_ids: np.ndarray
    observations: tuple[FrameObservation, ...]
    excluded_frames: tuple[tuple[int, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class ScopeEvidence:
    slot_id: str
    scope_status: ScopeStatus
    near_frames: tuple[int, ...] = ()
    crossing_frames: tuple[int, ...] = ()
    hit_frames: tuple[int, ...] = ()
    missing_frames: tuple[int, ...] = ()
    core_ray_coverage: float = 0.0
    agent_observable: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | int | bool | str
    threshold: float | int | bool | str


@dataclass(frozen=True)
class HeightLayerEvidence:
    z_min_m: float
    z_max_m: float
    point_count: int
    supporting_frames: tuple[int, ...]
    per_frame_counts: tuple[tuple[int, int], ...]
    bev_coverage: float
    supported: bool


@dataclass(frozen=True)
class SplitEvidence:
    name: str
    left_frames: tuple[int, ...]
    right_frames: tuple[int, ...]
    left_point_count: int
    right_point_count: int
    left_voxel_count: int
    right_voxel_count: int
    support_jaccard: float
    voxel_overlap_ratio: float
    z95_difference_m: float | None
    left_clears_weak_gate: bool
    right_clears_weak_gate: bool
    agreement: bool
    consistency: float


@dataclass(frozen=True)
class ThreeDEvidence:
    point_count: int
    voxel_count: int
    z50_m: float
    z75_m: float
    z90_m: float
    z95_m: float
    height_span_m: float
    supported_frame_count: int
    temporal_support: float
    height_layers: tuple[HeightLayerEvidence, ...]
    supported_layer_count: int
    low_bev_coverage: float
    mid_bev_coverage: float
    high_bev_coverage: float
    core_overlap: float
    slot_overlap: float
    boundary_ratio: float
    adjacent_overlap: float
    outside_residual_ratio: float
    linearity: float
    planarity: float
    extent_x_m: float
    extent_y_m: float
    extent_z_m: float
    overall_clears_weak_gate: bool
    temporal_consistency: float
    splits: tuple[SplitEvidence, ...]
    core_point_count: int = 0
    core_voxel_count: int = 0
    core_supported_frame_count: int = 0
    pca_linearity: float = 0.0
    robust_pca_linearity: float = 0.0
    robust_extent_x_m: float = 0.0
    robust_extent_y_m: float = 0.0
    robust_supported_frame_count: int = 0
    robust_point_fraction: float = 0.0


@dataclass(frozen=True)
class OccupiedEvidence:
    strong: bool = False
    weak: bool = False
    strength: float = 0.0
    support_frame_count: int = 0
    temporal_consistency: float = 0.0
    core_ownership: float = 0.0
    static_structure_risk: float = 0.0
    best_box: tuple[tuple[str, float], ...] = ()
    gate_results: tuple[GateResult, ...] = ()
    failures: tuple[str, ...] = ()
    features: ThreeDEvidence | None = None


@dataclass(frozen=True)
class FreeSpaceDetails:
    free_voxels: tuple[tuple[int, int, int], ...] = ()
    hit_voxels: tuple[tuple[int, int, int], ...] = ()
    ground_voxels: tuple[tuple[int, int, int], ...] = ()
    unknown_voxels: tuple[tuple[int, int, int], ...] = ()
    occluded_voxels: tuple[tuple[int, int, int], ...] = ()
    unobserved_voxels: tuple[tuple[int, int, int], ...] = ()
    viewpoint_bearings_deg: tuple[float, ...] = ()
    total_core_voxels: int = 0
    weak_obstacle_point_count: int = 0
    weak_obstacle_frame_count: int = 0
    core_hit_point_count: int = 0
    core_hit_voxel_count: int = 0
    core_hit_frame_count: int = 0
    conflict_voxels: tuple[tuple[int, int, int], ...] = ()
    conflict_hit_frames: tuple[int, ...] = ()
    conflict_free_frames: tuple[int, ...] = ()
    conflict_hit_frame_count: int = 0
    conflict_free_frame_count: int = 0
    conflict_viewpoint_count: int = 0
    conflict_viewpoint_separation_deg: float = 0.0
    quality_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class FreeEvidence:
    strong: bool = False
    strength: float = 0.0
    ray_frame_count: int = 0
    viewpoint_count: int = 0
    observed_volume_ratio: float = 0.0
    core_ray_coverage: float = 0.0
    near_ground_bev_coverage: float = 0.0
    unobserved_component_ratio: float = 1.0
    viewpoint_separation_deg: float = 0.0
    occlusion_ratio: float = 1.0
    weak_obstacle: bool = False
    unresolved_core_hit: bool = False
    conflict: bool = False
    weak_ownership_resolved: bool = False
    positive_geometry: bool = False
    gate_results: tuple[GateResult, ...] = ()
    failures: tuple[str, ...] = ()
    details: FreeSpaceDetails | None = None


@dataclass(frozen=True)
class StabilityEvidence:
    pass_ratio: float = 0.0
    passing_variants: int = 0
    total_variants: int = 0
    stable: bool = False
    failures: tuple[str, ...] = ()
    variant_results: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class AgentContext:
    agent_observable: bool = False
    priority: str = ""
    suggested_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class WeakEvidenceAssessment:
    active: bool = False
    ownership: str = "none"
    morphology: tuple[str, ...] = ()
    temporal: str = "insufficient"
    free_context: str = "not_evaluated"
    disposition: str = "none"
    primary_reason: str = ""
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotDecision:
    slot_id: str
    scope_status: ScopeStatus | str
    state: DecisionState | str
    decision_reason: str
    unknown_reasons: tuple[str, ...] = ()
    occupied_evidence: OccupiedEvidence = field(default_factory=OccupiedEvidence)
    free_evidence: FreeEvidence = field(default_factory=FreeEvidence)
    stability: StabilityEvidence = field(default_factory=StabilityEvidence)
    reference_frames: tuple[int, ...] = ()
    agent_context: AgentContext = field(default_factory=AgentContext)
    weak_evidence: WeakEvidenceAssessment = field(default_factory=WeakEvidenceAssessment)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        scope_status = ScopeStatus(self.scope_status)
        state = DecisionState(self.state)
        if scope_status is ScopeStatus.OUT_OF_ROUTE:
            raise ValueError("out_of_route_scope must not produce a SlotDecision")
        object.__setattr__(self, "scope_status", scope_status)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "unknown_reasons", tuple(sorted(set(self.unknown_reasons))))
        object.__setattr__(self, "reference_frames", tuple(sorted(set(self.reference_frames))))
