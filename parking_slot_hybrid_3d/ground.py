from __future__ import annotations

import numpy as np

from .config import Hybrid3DConfig
from .contracts import GroundModel


MIN_FALLBACK_POINTS = 12
MAX_FIT_ITERATIONS = 4


def _invalid_model(failures: list[str], candidate_count: int = 0) -> GroundModel:
    return GroundModel(
        method="invalid",
        valid=False,
        candidate_count=int(candidate_count),
        failures=tuple(sorted(set(failures))),
    )


def _constant_fallback(points: np.ndarray, config: Hybrid3DConfig, failures: list[str]) -> GroundModel:
    if len(points) < MIN_FALLBACK_POINTS:
        failures.append("insufficient_ground_points")
        return _invalid_model(failures, candidate_count=len(points))
    c = float(np.quantile(points[:, 2], config.ground_fallback_quantile))
    residual = np.abs(points[:, 2] - c)
    return GroundModel(
        method="constant",
        valid=True,
        c=c,
        candidate_count=len(points),
        inlier_count=len(points),
        inlier_ratio=1.0,
        residual_p95_m=float(np.quantile(residual, 0.95)),
        failures=tuple(sorted(set(failures))),
    )


def fit_ground_model(points_local_xyz: np.ndarray, config: Hybrid3DConfig) -> GroundModel:
    points = np.asarray(points_local_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_local_xyz must have shape [N, >=3]")
    finite = points[np.isfinite(points[:, :3]).all(axis=1), :3]
    if len(finite) < MIN_FALLBACK_POINTS:
        return _invalid_model(["insufficient_ground_points"], candidate_count=len(finite))

    cutoff = float(np.quantile(finite[:, 2], config.ground_candidate_quantile))
    candidates = finite[finite[:, 2] <= cutoff + 1e-12]
    failures: list[str] = []
    if len(candidates) < config.ground_min_points:
        failures.append("insufficient_ground_candidates")
        return _constant_fallback(finite, config, failures)

    design = np.column_stack([candidates[:, 0], candidates[:, 1], np.ones(len(candidates))])
    if np.linalg.matrix_rank(design) < 3:
        failures.append("rank_deficient_ground_plane")
        return _constant_fallback(finite, config, failures)

    inliers = np.ones(len(candidates), dtype=bool)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(MAX_FIT_ITERATIONS):
        active_design = design[inliers]
        active_z = candidates[inliers, 2]
        if len(active_design) < config.ground_min_points or np.linalg.matrix_rank(active_design) < 3:
            failures.append("ground_plane_inliers_degenerate")
            return _constant_fallback(finite, config, failures)
        coefficients = np.linalg.lstsq(active_design, active_z, rcond=None)[0]
        signed_residual = candidates[:, 2] - design @ coefficients
        median = float(np.median(signed_residual))
        mad = float(np.median(np.abs(signed_residual - median)))
        threshold = max(config.ground_inlier_threshold_m, 2.5 * 1.4826 * mad)
        updated = np.abs(signed_residual - median) <= threshold
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    active_design = design[inliers]
    active_z = candidates[inliers, 2]
    if len(active_design) < config.ground_min_points or np.linalg.matrix_rank(active_design) < 3:
        failures.append("ground_plane_inliers_degenerate")
        return _constant_fallback(finite, config, failures)
    coefficients = np.linalg.lstsq(active_design, active_z, rcond=None)[0]
    residual = np.abs(active_z - active_design @ coefficients)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(len(candidates), 1)
    residual_p95 = float(np.quantile(residual, 0.95)) if len(residual) else float("inf")
    if inlier_ratio < config.ground_min_inlier_ratio:
        failures.append("low_ground_inlier_ratio")
    if residual_p95 > config.ground_max_residual_p95_m:
        failures.append("high_ground_residual")
    if failures:
        return _constant_fallback(finite, config, failures)
    return GroundModel(
        method="plane",
        valid=True,
        a=float(coefficients[0]),
        b=float(coefficients[1]),
        c=float(coefficients[2]),
        candidate_count=len(candidates),
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        residual_p95_m=residual_p95,
    )


def normalize_z(points_local_xyz: np.ndarray, model: GroundModel) -> np.ndarray:
    if not model.valid:
        raise ValueError("cannot normalize points with an invalid ground model")
    points = np.asarray(points_local_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_local_xyz must have shape [N, >=3]")
    predicted_ground = model.a * points[:, 0] + model.b * points[:, 1] + model.c
    return points[:, 2] - predicted_ground


def origin_height(origin_local_xyz: np.ndarray, model: GroundModel) -> float:
    if not model.valid:
        raise ValueError("cannot normalize an origin with an invalid ground model")
    origin = np.asarray(origin_local_xyz, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_local_xyz must contain three finite values")
    ground_z = model.a * origin[0] + model.b * origin[1] + model.c
    return float(origin[2] - ground_z)
