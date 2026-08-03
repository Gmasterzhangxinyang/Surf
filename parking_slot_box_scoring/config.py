from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BoxScoringConfig:
    pipeline: str = "slot_constrained_box_scoring_v1"
    window_before: int = 28
    window_after: int = 28
    frame_stride: int = 4
    frame_selection: str = "anchor_window"
    top_k_visible: int = 15
    ground_quantile: float = 0.08
    low_z_min_m: float = 0.05
    low_z_max_m: float = 0.30
    vehicle_z_min_m: float = 0.30
    vehicle_z_max_m: float = 2.20
    strong_z95_m: float = 0.60
    strong_height_span_m: float = 0.35
    low_height_z95_m: float = 0.35
    low_height_span_m: float = 0.25
    min_vehicle_points_accumulated: int = 40
    min_frame_vehicle_points: int = 3
    min_frame_grid_cells: int = 2
    frame_z95_min_m: float = 0.35
    bev_grid_x: int = 8
    bev_grid_y: int = 4
    length_scales: list[float] = field(default_factory=lambda: [0.70, 0.85, 1.00, 1.10])
    width_scales: list[float] = field(default_factory=lambda: [0.55, 0.70, 0.85, 1.00])
    yaw_offsets_deg: list[float] = field(default_factory=lambda: [-8.0, -4.0, 0.0, 4.0, 8.0])
    longitudinal_offsets: list[float] = field(default_factory=lambda: [-0.15, 0.0, 0.15])
    lateral_offsets: list[float] = field(default_factory=lambda: [-0.12, 0.0, 0.12])
    state_vehicle_score_min: float = 0.55
    state_vehicle_height_min: float = 0.35
    state_vehicle_core_overlap_min: float = 0.45
    state_vehicle_adjacent_penalty_max: float = 0.35
    state_vehicle_boundary_penalty_max: float = 0.50
    state_adjacent_score_min: float = 0.45
    state_adjacent_penalty_min: float = 0.35
    state_boundary_score_min: float = 0.40
    state_boundary_penalty_min: float = 0.50
    state_static_linearity_min: float = 0.60
    low_height_penalty_state_min: float = 0.80
    roi_extra_m: float = 2.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
