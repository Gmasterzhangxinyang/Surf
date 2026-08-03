from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def angle_delta(a: float, b: float) -> float:
    return float(normalize_angle(a - b))


def select_keyframes(
    frames: np.ndarray,
    poses_map: np.ndarray,
    *,
    distance_m: float,
    yaw_deg: float,
    map_units_per_meter: float = 1.0,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.int64)
    poses = np.asarray(poses_map, dtype=np.float64)
    if poses.shape != (len(frames), 3):
        raise ValueError("poses_map must have shape [N, 3]")
    if len(frames) == 0:
        return np.empty(0, dtype=np.int64)

    selected = [0]
    distance_map = float(distance_m) * float(map_units_per_meter)
    yaw_rad = math.radians(float(yaw_deg))
    for index in range(1, len(frames) - 1):
        previous = selected[-1]
        displacement = float(np.linalg.norm(poses[index, :2] - poses[previous, :2]))
        turn = abs(angle_delta(float(poses[index, 2]), float(poses[previous, 2])))
        if displacement >= distance_map or turn >= yaw_rad:
            selected.append(index)
    if len(frames) > 1 and selected[-1] != len(frames) - 1:
        selected.append(len(frames) - 1)
    return np.asarray(selected, dtype=np.int64)


def interpolate_corrections(
    all_frames: np.ndarray,
    keyframes: np.ndarray,
    corrections: np.ndarray,
) -> np.ndarray:
    all_frames = np.asarray(all_frames, dtype=np.float64)
    keyframes = np.asarray(keyframes, dtype=np.float64)
    corrections = np.asarray(corrections, dtype=np.float64)
    if corrections.shape != (len(keyframes), 3):
        raise ValueError("corrections must have shape [K, 3]")
    if len(keyframes) == 0:
        return np.zeros((len(all_frames), 3), dtype=np.float64)
    yaw = np.unwrap(corrections[:, 2])
    dense = np.column_stack(
        [
            np.interp(all_frames, keyframes, corrections[:, 0]),
            np.interp(all_frames, keyframes, corrections[:, 1]),
            np.interp(all_frames, keyframes, yaw),
        ]
    )
    dense[:, 2] = normalize_angle(dense[:, 2])
    return dense


def stabilize_dense_corrections(
    corrections: np.ndarray,
    *,
    map_units_per_meter: float,
    window_frames: int = 101,
    max_translation_step_m: float = 0.02,
    max_yaw_step_deg: float = 0.05,
) -> np.ndarray:
    corrections = np.asarray(corrections, dtype=np.float64)
    if corrections.ndim != 2 or corrections.shape[1] != 3:
        raise ValueError("corrections must have shape [N, 3]")
    if len(corrections) < 3:
        return corrections.copy()
    window = min(int(window_frames), len(corrections) if len(corrections) % 2 else len(corrections) - 1)
    if window % 2 == 0:
        window -= 1
    window = max(3, window)
    polyorder = min(2, window - 1)

    filtered = np.zeros_like(corrections)
    for dimension in range(2):
        values = corrections[:, dimension]
        smooth = savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
        lower = min(0.0, float(values.min()))
        upper = max(0.0, float(values.max()))
        filtered[:, dimension] = np.clip(smooth, lower, upper)
    yaw = np.unwrap(corrections[:, 2])
    filtered[:, 2] = savgol_filter(yaw, window_length=window, polyorder=polyorder, mode="interp")
    filtered[0] = corrections[0]

    stable = np.zeros_like(filtered)
    stable[0] = filtered[0]
    max_translation_step_map = float(max_translation_step_m) * float(map_units_per_meter)
    max_yaw_step = math.radians(float(max_yaw_step_deg))
    for index in range(1, len(filtered)):
        delta_xy = filtered[index, :2] - stable[index - 1, :2]
        delta_norm = float(np.linalg.norm(delta_xy))
        if delta_norm > max_translation_step_map:
            delta_xy *= max_translation_step_map / delta_norm
        stable[index, :2] = stable[index - 1, :2] + delta_xy
        delta_yaw = float(np.clip(filtered[index, 2] - stable[index - 1, 2], -max_yaw_step, max_yaw_step))
        stable[index, 2] = stable[index - 1, 2] + delta_yaw
    stable[:, 2] = normalize_angle(stable[:, 2])
    return stable


def apply_pose_corrections(poses_map: np.ndarray, corrections: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_map, dtype=np.float64)
    corrections = np.asarray(corrections, dtype=np.float64)
    if poses.shape != corrections.shape or poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses_map and corrections must both have shape [N, 3]")
    result = poses + corrections
    result[:, 2] = normalize_angle(result[:, 2])
    return result


def smooth_keyframe_corrections(
    observed: np.ndarray,
    accepted: np.ndarray,
    confidence: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    observation_weight: float = 20.0,
    first_difference_weight: float = 0.05,
    second_difference_weight: float = 2.0,
    robust_window_m: float = 12.0,
    smooth_window_m: float = 30.0,
) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    accepted = np.asarray(accepted, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("observed must have shape [K, 3]")
    if accepted.shape != (len(observed),) or confidence.shape != (len(observed),):
        raise ValueError("accepted and confidence must have shape [K]")
    if len(observed) == 0 or not np.any(accepted):
        return np.zeros_like(observed)

    if positions is None:
        positions = np.arange(len(observed), dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (len(observed),) or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be a strictly increasing array with shape [K]")

    if len(observed) >= 7 and int(accepted.sum()) >= 3:
        spacing = float(np.median(np.diff(positions)))

        def odd_window(width: float, maximum: int, minimum: int) -> int:
            value = max(minimum, int(round(width / max(spacing, 1e-9))))
            if value % 2 == 0:
                value += 1
            largest = maximum if maximum % 2 == 1 else maximum - 1
            return max(minimum, min(value, largest))

        robust_window = odd_window(robust_window_m, len(observed), 3)
        smooth_window = odd_window(smooth_window_m, len(observed), 5)
        result = np.zeros_like(observed)
        valid_indices = np.flatnonzero(accepted & (confidence > 0))
        for dimension in range(3):
            values = observed[valid_indices, dimension]
            if dimension == 2:
                values = np.unwrap(values)
            interpolated = np.interp(
                positions,
                positions[valid_indices],
                values,
                left=float(values[0]),
                right=float(values[-1]),
            )
            robust = median_filter(interpolated, size=robust_window, mode="nearest")
            smoothed = savgol_filter(robust, window_length=smooth_window, polyorder=2, mode="interp")
            anchor_decay = np.exp(-(positions - positions[0]) / max(smooth_window_m * 0.5, 1e-9))
            smoothed -= smoothed[0] * anchor_decay
            lower = min(0.0, float(np.min(values)))
            upper = max(0.0, float(np.max(values)))
            result[:, dimension] = np.clip(smoothed, lower, upper)
        result[:, 2] = normalize_angle(result[:, 2])
        return result

    rows: list[np.ndarray] = []
    row_weights: list[float] = []
    observation_indices: list[int] = []
    for index in np.flatnonzero(accepted):
        row = np.zeros(len(observed), dtype=np.float64)
        row[index] = 1.0
        rows.append(row)
        row_weights.append(observation_weight * max(0.05, float(confidence[index])))
        observation_indices.append(int(index))
    for index in range(1, len(observed)):
        row = np.zeros(len(observed), dtype=np.float64)
        row[index - 1] = -1.0
        row[index] = 1.0
        rows.append(row)
        row_weights.append(first_difference_weight)
    for index in range(1, len(observed) - 1):
        row = np.zeros(len(observed), dtype=np.float64)
        row[index - 1] = 1.0
        row[index] = -2.0
        row[index + 1] = 1.0
        rows.append(row)
        row_weights.append(second_difference_weight)

    matrix = np.vstack(rows)
    weights = np.sqrt(np.asarray(row_weights, dtype=np.float64))
    weighted_matrix = matrix * weights[:, None]
    result = np.zeros_like(observed)
    for dimension in range(3):
        values = np.zeros(len(rows), dtype=np.float64)
        for row_index, observation_index in enumerate(observation_indices):
            values[row_index] = observed[observation_index, dimension]
        solution, *_ = np.linalg.lstsq(weighted_matrix, values * weights, rcond=None)
        observed_values = observed[accepted, dimension]
        lower = min(0.0, float(np.min(observed_values)))
        upper = max(0.0, float(np.max(observed_values)))
        result[:, dimension] = np.clip(solution, lower, upper)
    result[:, 2] = normalize_angle(result[:, 2])
    return result
