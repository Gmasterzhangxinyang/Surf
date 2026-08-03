from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .contracts import MetricSlot, SlotAccumulation
from .geometry import map_xy_to_slot_m


@dataclass(frozen=True)
class StaticSemanticAssessment:
    evaluated: bool
    veto: bool
    reason: str
    slot_above_ground_points: int
    static_explained_points: int
    residual_points: int
    static_explained_ratio: float
    residual_short_extent_m: float
    residual_long_extent_m: float


class SemanticStaticOccupiedVeto:
    """Fail-closed occupied veto using independent semantic static geometry."""

    def __init__(
        self,
        static_map_xy: np.ndarray,
        *,
        association_distance_m: float = 0.35,
        max_static_explained_ratio: float = 0.50,
        min_residual_short_extent_m: float = 0.75,
        min_points: int = 40,
        z_min_m: float = 0.30,
        z_max_m: float = 2.80,
    ) -> None:
        points = np.asarray(static_map_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("static_map_xy must be a finite [N,2] array")
        if len(points) == 0:
            raise ValueError("static_map_xy must not be empty")
        if association_distance_m <= 0.0:
            raise ValueError("association_distance_m must be positive")
        if not 0.0 <= max_static_explained_ratio <= 1.0:
            raise ValueError("max_static_explained_ratio must be in [0,1]")
        if min_residual_short_extent_m <= 0.0 or min_points <= 0:
            raise ValueError("residual extent and min_points must be positive")
        if z_min_m >= z_max_m:
            raise ValueError("z_min_m must be below z_max_m")
        self.static_map_xy = points
        self.association_distance_m = float(association_distance_m)
        self.max_static_explained_ratio = float(max_static_explained_ratio)
        self.min_residual_short_extent_m = float(min_residual_short_extent_m)
        self.min_points = int(min_points)
        self.z_min_m = float(z_min_m)
        self.z_max_m = float(z_max_m)

    @staticmethod
    def _robust_extents(points_xy: np.ndarray) -> tuple[float, float]:
        if len(points_xy) < 3:
            return 0.0, 0.0
        quantiles = np.quantile(points_xy, [0.05, 0.95], axis=0)
        extents = np.sort(quantiles[1] - quantiles[0])
        return float(extents[0]), float(extents[1])

    def assess(
        self,
        metric_slot: MetricSlot,
        accumulation: SlotAccumulation,
    ) -> StaticSemanticAssessment:
        points = np.asarray(accumulation.points_local_xyzi, dtype=np.float64)
        lower = np.asarray(metric_slot.polygon_local_m, dtype=np.float64).min(axis=0)
        upper = np.asarray(metric_slot.polygon_local_m, dtype=np.float64).max(axis=0)
        inside = (
            (points[:, 0] >= lower[0])
            & (points[:, 0] <= upper[0])
            & (points[:, 1] >= lower[1])
            & (points[:, 1] <= upper[1])
            & (points[:, 2] > self.z_min_m)
            & (points[:, 2] <= self.z_max_m)
        )
        obstacle = points[inside]
        if len(obstacle) < self.min_points:
            return StaticSemanticAssessment(
                evaluated=False,
                veto=False,
                reason="insufficient_slot_points",
                slot_above_ground_points=int(len(obstacle)),
                static_explained_points=0,
                residual_points=int(len(obstacle)),
                static_explained_ratio=0.0,
                residual_short_extent_m=0.0,
                residual_long_extent_m=0.0,
            )

        static_local = map_xy_to_slot_m(self.static_map_xy, metric_slot)
        padding = self.association_distance_m
        nearby = (
            (static_local[:, 0] >= lower[0] - padding)
            & (static_local[:, 0] <= upper[0] + padding)
            & (static_local[:, 1] >= lower[1] - padding)
            & (static_local[:, 1] <= upper[1] + padding)
        )
        static_local = static_local[nearby]
        if len(static_local):
            distances, _ = cKDTree(static_local).query(obstacle[:, :2], k=1)
            explained = distances <= self.association_distance_m
        else:
            explained = np.zeros(len(obstacle), dtype=bool)
        residual = obstacle[~explained]
        explained_ratio = float(np.mean(explained))
        short_extent, long_extent = self._robust_extents(residual[:, :2])
        static_veto = explained_ratio >= self.max_static_explained_ratio
        footprint_veto = (
            len(residual) >= self.min_points
            and short_extent < self.min_residual_short_extent_m
        )
        if static_veto:
            reason = "semantic_static_explained_ratio"
        elif footprint_veto:
            reason = "semantic_static_residual_too_narrow"
        else:
            reason = "semantic_static_veto_passed"
        return StaticSemanticAssessment(
            evaluated=True,
            veto=bool(static_veto or footprint_veto),
            reason=reason,
            slot_above_ground_points=int(len(obstacle)),
            static_explained_points=int(np.sum(explained)),
            residual_points=int(len(residual)),
            static_explained_ratio=explained_ratio,
            residual_short_extent_m=short_extent,
            residual_long_extent_m=long_extent,
        )
