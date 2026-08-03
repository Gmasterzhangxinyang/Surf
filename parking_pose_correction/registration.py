from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RegistrationResult:
    dx_m: float
    dy_m: float
    dyaw_rad: float
    residual_before_m: float
    residual_after_m: float
    inlier_ratio: float
    matched_points: int
    improvement_ratio: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class AlignmentMetrics:
    residual_m: float
    inlier_ratio: float
    matched_points: int


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    src = source - source_center
    dst = target - target_center
    u, _, vt = np.linalg.svd(src.T @ dst)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _trimmed_metrics(
    points: np.ndarray,
    tree: cKDTree,
    scale: float,
    max_correspondence_m: float,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    distances_map, indices = tree.query(points, k=1)
    distances_m = distances_map / scale
    cap = min(max_correspondence_m, float(np.quantile(distances_m, trim_fraction)))
    mask = distances_m <= cap
    residual = float(np.median(distances_m[mask])) if np.any(mask) else float("inf")
    inlier_ratio = float(np.mean(distances_m <= max_correspondence_m))
    return indices, mask, residual, inlier_ratio


def evaluate_alignment(
    scan_points_map: np.ndarray,
    target_points_map: np.ndarray,
    *,
    map_units_per_meter: float,
    max_correspondence_m: float = 0.75,
    trim_fraction: float = 0.35,
) -> AlignmentMetrics:
    scan = np.asarray(scan_points_map, dtype=np.float64)
    target = np.asarray(target_points_map, dtype=np.float64)
    if len(scan) == 0 or len(target) == 0:
        return AlignmentMetrics(float("inf"), 0.0, 0)
    tree = cKDTree(target)
    _, mask, residual, inlier_ratio = _trimmed_metrics(
        scan,
        tree,
        float(map_units_per_meter),
        max_correspondence_m,
        trim_fraction,
    )
    return AlignmentMetrics(residual, inlier_ratio, int(mask.sum()))


def estimate_local_correction(
    scan_points_map: np.ndarray,
    target_points_map: np.ndarray,
    *,
    map_units_per_meter: float,
    max_translation_m: float = 1.5,
    max_yaw_deg: float = 5.0,
    max_correspondence_m: float = 0.75,
    trim_fraction: float = 0.35,
    max_iterations: int = 20,
) -> RegistrationResult:
    scan = np.asarray(scan_points_map, dtype=np.float64)
    target = np.asarray(target_points_map, dtype=np.float64)
    scale = float(map_units_per_meter)
    if scan.ndim != 2 or scan.shape[1] != 2 or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("scan and target points must have shape [N, 2]")
    if len(scan) < 10 or len(target) < 10 or scale <= 0:
        return RegistrationResult(0, 0, 0, float("inf"), float("inf"), 0, 0, 0, False, 0)

    tree = cKDTree(target)
    _, _, residual_before, _ = _trimmed_metrics(scan, tree, scale, max_correspondence_m, trim_fraction)
    max_yaw = math.radians(max_yaw_deg)
    candidates = []
    for initial_yaw in np.linspace(-max_yaw, max_yaw, 7):
        rotation_total = _rotation(float(initial_yaw))
        translation_total = np.zeros(2, dtype=np.float64)
        transformed = scan @ rotation_total.T
        converged = False
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            indices, mask, _, _ = _trimmed_metrics(
                transformed, tree, scale, max_correspondence_m, trim_fraction
            )
            if int(mask.sum()) < 10:
                break
            update_rotation, update_translation = _rigid_fit(transformed[mask], target[indices[mask]])
            previous_yaw = math.atan2(float(rotation_total[1, 0]), float(rotation_total[0, 0]))
            update_yaw = math.atan2(float(update_rotation[1, 0]), float(update_rotation[0, 0]))
            total_yaw = float(np.clip(previous_yaw + update_yaw, -max_yaw, max_yaw))
            candidate_rotation = _rotation(total_yaw)
            candidate_translation = translation_total @ update_rotation.T + update_translation
            translation_m = candidate_translation / scale
            norm_m = float(np.linalg.norm(translation_m))
            if norm_m > max_translation_m:
                candidate_translation *= max_translation_m / norm_m

            delta_translation_m = float(np.linalg.norm(candidate_translation - translation_total) / scale)
            delta_yaw = abs(total_yaw - previous_yaw)
            rotation_total = candidate_rotation
            translation_total = candidate_translation
            transformed = scan @ rotation_total.T + translation_total
            if delta_translation_m < 1e-3 and delta_yaw < math.radians(0.01):
                converged = True
                break
        _, final_mask, residual_after, inlier_ratio = _trimmed_metrics(
            transformed, tree, scale, max_correspondence_m, trim_fraction
        )
        candidates.append(
            (
                residual_after,
                -inlier_ratio,
                rotation_total,
                translation_total,
                final_mask,
                converged,
                iterations,
            )
        )

    residual_after, neg_inlier_ratio, rotation_total, translation_total, final_mask, converged, iterations = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    initial_yaw = math.atan2(float(rotation_total[1, 0]), float(rotation_total[0, 0]))

    def objective(params: np.ndarray) -> float:
        transformed_points = scan @ _rotation(float(params[2])).T + params[:2]
        distances_m = tree.query(transformed_points, k=1)[0] / scale
        keep = max(10, int(len(distances_m) * trim_fraction))
        trimmed = np.partition(distances_m, keep - 1)[:keep]
        clipped = np.minimum(trimmed, max_correspondence_m)
        return float(np.mean(clipped * clipped))

    optimized = minimize(
        objective,
        np.array([translation_total[0], translation_total[1], initial_yaw]),
        method="Powell",
        bounds=[
            (-max_translation_m * scale, max_translation_m * scale),
            (-max_translation_m * scale, max_translation_m * scale),
            (-max_yaw, max_yaw),
        ],
        options={"maxiter": 60, "xtol": 1e-5 * scale, "ftol": 1e-7},
    )
    translation_total = np.asarray(optimized.x[:2], dtype=np.float64)
    translation_norm_m = float(np.linalg.norm(translation_total) / scale)
    if translation_norm_m > max_translation_m:
        translation_total *= max_translation_m / translation_norm_m
    rotation_total = _rotation(float(optimized.x[2]))
    transformed = scan @ rotation_total.T + translation_total
    _, final_mask, residual_after, inlier_ratio = _trimmed_metrics(
        transformed, tree, scale, max_correspondence_m, trim_fraction
    )
    converged = bool(converged or optimized.success)
    improvement = 0.0
    if np.isfinite(residual_before) and residual_before > 1e-9 and np.isfinite(residual_after):
        improvement = max(0.0, 1.0 - residual_after / residual_before)
    yaw = math.atan2(float(rotation_total[1, 0]), float(rotation_total[0, 0]))
    translation_m = translation_total / scale
    return RegistrationResult(
        dx_m=float(translation_m[0]),
        dy_m=float(translation_m[1]),
        dyaw_rad=float(yaw),
        residual_before_m=residual_before,
        residual_after_m=residual_after,
        inlier_ratio=inlier_ratio,
        matched_points=int(final_mask.sum()),
        improvement_ratio=improvement,
        converged=converged or iterations == max_iterations,
        iterations=iterations,
    )
