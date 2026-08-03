from __future__ import annotations

from dataclasses import asdict, dataclass

from parking_slot_box_scoring.config import BoxScoringConfig


@dataclass(frozen=True)
class Hybrid3DConfig:
    schema_version: str = "1.0"
    pipeline: str = "slot_hybrid_3d_pose_corrected_v2"
    scope_max_distance_m: float = 25.0
    scope_min_near_frames: int = 3
    scope_min_ray_frames: int = 3
    scope_core_coverage_min: float = 0.10
    scope_grid_m: float = 0.50
    scope_prism_expand_m: float = 0.50
    scope_raw_z_min_m: float = -3.0
    scope_raw_z_max_m: float = 3.0
    window_before: int = 28
    window_after: int = 28
    frame_stride: int = 4
    ground_candidate_quantile: float = 0.35
    ground_fallback_quantile: float = 0.08
    ground_min_points: int = 30
    ground_inlier_threshold_m: float = 0.12
    ground_min_inlier_ratio: float = 0.45
    ground_max_residual_p95_m: float = 0.15
    voxel_xy_m: float = 0.25
    voxel_z_m: float = 0.20
    pca_max_points: int = 2000
    vehicle_z_min_m: float = 0.30
    vehicle_z_max_m: float = 2.20
    low_height_veto_z95_m: float = 0.35
    low_height_veto_span_m: float = 0.25
    height_layers_m: tuple[tuple[float, float], ...] = (
        (0.30, 0.80),
        (0.80, 1.40),
        (1.40, 2.20),
    )
    occupied_min_supported_layers: int = 2
    occupied_layer_min_frames: int = 2
    occupied_layer_min_points_per_frame: int = 3
    occupied_min_valid_frames: int = 3
    occupied_min_points: int = 40
    occupied_min_support_frames: int = 3
    occupied_min_temporal_support: float = 0.20
    occupied_min_z95_m: float = 0.60
    occupied_min_height_span_m: float = 0.35
    # Positive vehicle-shape evidence. Zero disables these gates for backward
    # compatibility; high-precision production profiles must set them.
    occupied_min_robust_short_extent_m: float = 0.0
    occupied_min_low_bev_coverage: float = 0.0
    # Safety veto for a thin horizontal cap whose upper quartile collapses
    # onto one height. Zero disables it and preserves the v1 model.
    occupied_min_upper_height_spread_ratio: float = 0.0
    occupied_min_core_overlap: float = 0.45
    occupied_max_adjacent_overlap: float = 0.35
    occupied_max_boundary_ratio: float = 0.50
    # Experimental inset applied to the slot core before assigning strong
    # Occupied ownership. Zero preserves the production geometry exactly.
    occupied_localization_uncertainty_m: float = 0.0
    occupied_min_voxels: int = 8
    occupied_max_linearity: float = 0.60
    occupied_pillar_min_pca_linearity: float = 0.95
    occupied_pillar_max_horizontal_extent_m: float = 0.75
    occupied_pillar_trim_quantile: float = 0.05
    occupied_pillar_min_inlier_fraction: float = 0.80
    # Secondary physical veto for a post/pillar diluted by nearby clutter.
    occupied_pillar_secondary_min_pca_linearity: float = 0.90
    occupied_pillar_secondary_min_inlier_fraction: float = 0.70
    occupied_pillar_max_footprint_area_m2: float = 0.30
    occupied_pillar_min_vertical_aspect_ratio: float = 1.50
    occupied_pillar_min_height_m: float = 0.80
    occupied_max_outside_residual: float = 0.60
    occupied_candidate_shortlist_size: int = 16
    # Closed front half-plane: +/-90 degrees = 180 degree total field.
    part2_candidate_half_fov_deg: float = 90.0
    free_min_ray_frames: int = 5
    free_min_viewpoints: int = 2
    free_min_viewpoint_separation_deg: float = 10.0
    free_min_volume_coverage: float = 0.70
    free_min_near_ground_bev_coverage: float = 0.70
    free_max_unobserved_component_ratio: float = 0.20
    free_max_occlusion_ratio: float = 0.20
    free_near_ground_z_min_m: float = 0.15
    free_near_ground_z_max_m: float = 0.80
    free_weak_obstacle_min_frames: int = 2
    free_weak_obstacle_min_points: int = 10
    free_conflict_min_voxels: int = 4
    free_conflict_min_hit_frames: int = 2
    free_conflict_min_free_frames: int = 2
    free_conflict_min_viewpoints: int = 2
    free_conflict_min_viewpoint_separation_deg: float = 10.0
    stability_translation_m: float = 0.20
    stability_yaw_deg: float = 0.50
    stability_min_pass_ratio: float = 0.80
    stability_refit_max_center_shift_m: float = 0.75
    stability_refit_max_yaw_shift_deg: float = 12.0
    stability_refit_min_box_overlap: float = 0.60

    def validate(self) -> None:
        positive_fields = (
            "scope_max_distance_m",
            "scope_min_near_frames",
            "scope_min_ray_frames",
            "scope_grid_m",
            "scope_prism_expand_m",
            "frame_stride",
            "ground_min_points",
            "ground_inlier_threshold_m",
            "ground_max_residual_p95_m",
            "voxel_xy_m",
            "voxel_z_m",
            "pca_max_points",
            "occupied_min_supported_layers",
            "occupied_layer_min_frames",
            "occupied_layer_min_points_per_frame",
            "occupied_min_valid_frames",
            "occupied_min_points",
            "occupied_min_support_frames",
            "occupied_min_voxels",
            "occupied_candidate_shortlist_size",
            "occupied_pillar_max_horizontal_extent_m",
            "occupied_pillar_max_footprint_area_m2",
            "occupied_pillar_min_vertical_aspect_ratio",
            "occupied_pillar_min_height_m",
            "part2_candidate_half_fov_deg",
            "free_min_ray_frames",
            "free_min_viewpoints",
            "free_min_viewpoint_separation_deg",
            "free_weak_obstacle_min_frames",
            "free_weak_obstacle_min_points",
            "free_conflict_min_voxels",
            "free_conflict_min_hit_frames",
            "free_conflict_min_free_frames",
            "free_conflict_min_viewpoints",
            "free_conflict_min_viewpoint_separation_deg",
            "stability_translation_m",
            "stability_yaw_deg",
            "stability_refit_max_center_shift_m",
            "stability_refit_max_yaw_shift_deg",
        )
        for name in positive_fields:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

        ratio_fields = (
            "scope_core_coverage_min",
            "ground_candidate_quantile",
            "ground_fallback_quantile",
            "ground_min_inlier_ratio",
            "occupied_min_temporal_support",
            "occupied_min_upper_height_spread_ratio",
            "occupied_min_low_bev_coverage",
            "occupied_min_core_overlap",
            "occupied_max_adjacent_overlap",
            "occupied_max_boundary_ratio",
            "occupied_max_linearity",
            "occupied_pillar_min_pca_linearity",
            "occupied_pillar_secondary_min_pca_linearity",
            "occupied_pillar_secondary_min_inlier_fraction",
            "occupied_pillar_trim_quantile",
            "occupied_pillar_min_inlier_fraction",
            "occupied_max_outside_residual",
            "free_min_volume_coverage",
            "free_min_near_ground_bev_coverage",
            "free_max_unobserved_component_ratio",
            "free_max_occlusion_ratio",
            "stability_min_pass_ratio",
            "stability_refit_min_box_overlap",
        )
        for name in ratio_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.occupied_pillar_trim_quantile >= 0.50:
            raise ValueError("occupied_pillar_trim_quantile must be lower than 0.50")
        if (
            not isinstance(self.occupied_min_robust_short_extent_m, (int, float))
            or not 0.0 <= float(self.occupied_min_robust_short_extent_m) <= 3.0
        ):
            raise ValueError(
                "occupied_min_robust_short_extent_m must be within [0, 3.0]"
            )
        if self.part2_candidate_half_fov_deg > 180.0:
            raise ValueError("part2_candidate_half_fov_deg must not exceed 180")
        if (
            not isinstance(self.occupied_localization_uncertainty_m, (int, float))
            or not 0.0 <= float(self.occupied_localization_uncertainty_m) <= 0.50
        ):
            raise ValueError(
                "occupied_localization_uncertainty_m must be within [0, 0.50]"
            )

        ordered_ranges = (
            ("scope_raw_z", self.scope_raw_z_min_m, self.scope_raw_z_max_m),
            ("vehicle_z", self.vehicle_z_min_m, self.vehicle_z_max_m),
            ("free_near_ground_z", self.free_near_ground_z_min_m, self.free_near_ground_z_max_m),
        )
        for name, lower, upper in ordered_ranges:
            if float(lower) >= float(upper):
                raise ValueError(f"{name}_min_m must be lower than {name}_max_m")

        previous_upper: float | None = None
        for lower, upper in self.height_layers_m:
            if lower >= upper:
                raise ValueError("height_layers_m entries must have lower < upper")
            if previous_upper is not None and lower < previous_upper:
                raise ValueError("height_layers_m entries must not overlap")
            if lower < self.vehicle_z_min_m or upper > self.vehicle_z_max_m:
                raise ValueError("height_layers_m entries must stay inside the vehicle height range")
            previous_upper = upper

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def box_hypothesis_config(self) -> BoxScoringConfig:
        return BoxScoringConfig()
