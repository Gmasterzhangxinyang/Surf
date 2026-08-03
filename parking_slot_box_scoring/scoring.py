from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .box_hypotheses import CandidateBox, enumerate_box_hypotheses
from .config import BoxScoringConfig
from .geometry import (
    box_polygon,
    compute_slot_frame,
    points_in_polygon,
    points_in_rotated_box,
    polygon_overlap_ratio,
)
from .point_filtering import estimate_local_ground_z


@dataclass
class BoxScore:
    slot_id: str
    score: float
    state: str
    reason: str
    center_x: float
    center_y: float
    yaw: float
    length: float
    width: float
    point_support: float
    inside_vehicle_point_count: int
    inside_low_point_count: int
    bev_coverage: float
    height_support: float
    z95_above_ground: float
    height_span: float
    slot_core_overlap: float
    slot_polygon_overlap: float
    boundary_ratio: float
    adjacent_overlap: float
    temporal_support: float
    supported_frame_count: int
    selected_frame_count: int
    low_height_penalty: float
    adjacent_penalty: float
    boundary_penalty: float
    linearity_penalty: float
    outside_box_residual_penalty: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ScoreContext:
    accumulated_points: np.ndarray
    frame_points: list[np.ndarray]
    z_rel_all: np.ndarray
    vehicle_all: np.ndarray
    low_all: np.ndarray
    vehicle_points_xy: np.ndarray
    vehicle_z_rel: np.ndarray
    low_points_xy: np.ndarray
    in_margin_vehicle_count: int


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def vehicle_height_mask(z_rel: np.ndarray, config: BoxScoringConfig) -> np.ndarray:
    return (z_rel >= config.vehicle_z_min_m) & (z_rel <= config.vehicle_z_max_m)


def low_height_mask(z_rel: np.ndarray, config: BoxScoringConfig) -> np.ndarray:
    return (z_rel >= config.low_z_min_m) & (z_rel < config.low_z_max_m)


def bev_coverage_for_points(points_xy: np.ndarray, box: CandidateBox, config: BoxScoringConfig) -> float:
    if len(points_xy) == 0:
        return 0.0
    long_axis = np.asarray([np.cos(box.yaw), np.sin(box.yaw)])
    short_axis = np.asarray([-np.sin(box.yaw), np.cos(box.yaw)])
    delta = points_xy - box.center
    x = delta @ long_axis
    y = delta @ short_axis
    xi = np.floor((x + box.length / 2) / max(box.length, 1e-9) * config.bev_grid_x).astype(int)
    yi = np.floor((y + box.width / 2) / max(box.width, 1e-9) * config.bev_grid_y).astype(int)
    valid = (xi >= 0) & (xi < config.bev_grid_x) & (yi >= 0) & (yi < config.bev_grid_y)
    touched = set(zip(xi[valid].tolist(), yi[valid].tolist()))
    return len(touched) / max(config.bev_grid_x * config.bev_grid_y, 1)


def height_support(z_rel: np.ndarray, config: BoxScoringConfig) -> tuple[float, float, float, float]:
    if len(z_rel) == 0:
        return 0.0, 0.0, 0.0, 1.0
    z95 = float(np.quantile(z_rel, 0.95))
    span = float(z_rel.max() - z_rel.min())
    z_score = clamp(z95 / max(config.strong_z95_m, 1e-9))
    span_score = clamp(span / max(config.strong_height_span_m, 1e-9))
    support = 0.55 * z_score + 0.45 * span_score
    low_penalty = 1.0 if z95 < config.low_height_z95_m or span < config.low_height_span_m else 0.0
    return clamp(support), z95, span, low_penalty


def linearity_penalty(points_xy: np.ndarray) -> float:
    if len(points_xy) < 5:
        return 0.0
    if len(points_xy) > 2000:
        step = max(len(points_xy) // 2000, 1)
        points_xy = points_xy[::step]
    centered = points_xy - points_xy.mean(axis=0)
    if np.linalg.norm(centered) <= 1e-12:
        return 0.0
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    ratio = float(np.sqrt(max(eigvals[-1], 0.0)) / max(np.sqrt(max(eigvals[0], 0.0)), 1e-9))
    return clamp((ratio - 5.0) / 5.0)


def adjacent_overlap_ratio(box_poly: np.ndarray, slot: dict, slots_by_id: dict[str, dict]) -> float:
    ratios: list[float] = []
    for adjacent_id in slot.get("adjacent_slots", []) or []:
        adjacent = slots_by_id.get(str(adjacent_id))
        if not adjacent:
            continue
        poly = adjacent.get("core_np", adjacent.get("polygon_np"))
        if poly is not None:
            ratios.append(polygon_overlap_ratio(box_poly, poly))
    return max(ratios, default=0.0)


def slot_polygon(slot: dict, key: str, fallback_key: str = "polygon_np") -> np.ndarray:
    if key in slot and slot[key] is not None:
        return np.asarray(slot[key], dtype=np.float64)
    if fallback_key in slot and slot[fallback_key] is not None:
        return np.asarray(slot[fallback_key], dtype=np.float64)
    if "polygon_map" in slot and slot["polygon_map"] is not None:
        return np.asarray(slot["polygon_map"], dtype=np.float64)
    raise KeyError(f"slot {slot.get('slot_id')} has no usable polygon data")


def evaluate_box(
    slot: dict,
    box: CandidateBox,
    context: ScoreContext,
    slots_by_id: dict[str, dict],
    config: BoxScoringConfig,
    include_temporal: bool = False,
) -> BoxScore:
    slot_id = str(slot["slot_id"])
    box_poly = box_polygon(box.center, box.yaw, box.length, box.width)
    accumulated_points = context.accumulated_points
    if len(accumulated_points):
        in_box_vehicle = points_in_rotated_box(context.vehicle_points_xy, box.center, box.yaw, box.length, box.width)
        in_box_low = points_in_rotated_box(context.low_points_xy, box.center, box.yaw, box.length, box.width)
        vehicle_points_xy = context.vehicle_points_xy[in_box_vehicle]
        z_rel_vehicle = context.vehicle_z_rel[in_box_vehicle]
        inside_vehicle_count = int(in_box_vehicle.sum())
        inside_low_count = int(in_box_low.sum())
    else:
        vehicle_points_xy = np.empty((0, 2), dtype=np.float64)
        z_rel_vehicle = np.empty(0, dtype=np.float64)
        inside_vehicle_count = 0
        inside_low_count = 0
    point_support = clamp(inside_vehicle_count / max(config.min_vehicle_points_accumulated, 1))
    bev_coverage = bev_coverage_for_points(vehicle_points_xy, box, config)
    h_support, z95, h_span, low_penalty = height_support(z_rel_vehicle, config)
    slot_core_overlap = polygon_overlap_ratio(box_poly, slot_polygon(slot, "core_np"))
    slot_polygon_overlap = polygon_overlap_ratio(box_poly, slot_polygon(slot, "polygon_np"))
    core_count = int((points_in_polygon(vehicle_points_xy, slot_polygon(slot, "core_np"))).sum()) if len(vehicle_points_xy) else 0
    boundary_count = max(inside_vehicle_count - core_count, 0)
    boundary_ratio = boundary_count / max(inside_vehicle_count, 1)
    boundary_penalty = clamp(boundary_ratio)
    adjacent_overlap = adjacent_overlap_ratio(box_poly, slot, slots_by_id)
    adjacent_penalty = clamp(adjacent_overlap)
    lin_penalty = linearity_penalty(vehicle_points_xy)
    outside_residual = max(context.in_margin_vehicle_count - inside_vehicle_count, 0)
    outside_box_residual_penalty = clamp(outside_residual / max(context.in_margin_vehicle_count, 1))
    supported_frame_count = 0
    if include_temporal:
        for points in context.frame_points:
            if len(points) == 0:
                continue
            frame_ground = estimate_local_ground_z(points[:, :3], config.ground_quantile)
            z_rel = points[:, 2] - frame_ground
            in_box_frame = points_in_rotated_box(points[:, :2], box.center, box.yaw, box.length, box.width)
            frame_vehicle = in_box_frame & vehicle_height_mask(z_rel, config)
            count = int(frame_vehicle.sum())
            if count < config.min_frame_vehicle_points:
                continue
            coverage = bev_coverage_for_points(points[frame_vehicle, :2], box, config)
            z95_frame = float(np.quantile(z_rel[frame_vehicle], 0.95)) if count else 0.0
            if coverage * config.bev_grid_x * config.bev_grid_y >= config.min_frame_grid_cells and z95_frame >= config.frame_z95_min_m:
                supported_frame_count += 1
    selected_frame_count = len(context.frame_points)
    temporal_support = supported_frame_count / max(selected_frame_count, 1)
    raw_score = (
        0.30 * point_support
        + 0.20 * bev_coverage
        + 0.20 * h_support
        + 0.15 * slot_core_overlap
        + 0.10 * temporal_support
        + 0.05 * slot_polygon_overlap
        - 0.25 * low_penalty
        - 0.20 * adjacent_penalty
        - 0.15 * boundary_penalty
        - 0.10 * lin_penalty
        - 0.10 * outside_box_residual_penalty
    )
    score = clamp(raw_score)
    state, reason = classify_state(
        score,
        selected_frame_count,
        inside_vehicle_count,
        h_support,
        slot_core_overlap,
        adjacent_penalty,
        boundary_penalty,
        low_penalty,
        lin_penalty,
        config,
    )
    return BoxScore(
        slot_id=slot_id,
        score=score,
        state=state,
        reason=reason,
        center_x=float(box.center[0]),
        center_y=float(box.center[1]),
        yaw=float(box.yaw),
        length=float(box.length),
        width=float(box.width),
        point_support=point_support,
        inside_vehicle_point_count=inside_vehicle_count,
        inside_low_point_count=inside_low_count,
        bev_coverage=float(bev_coverage),
        height_support=float(h_support),
        z95_above_ground=float(z95),
        height_span=float(h_span),
        slot_core_overlap=float(slot_core_overlap),
        slot_polygon_overlap=float(slot_polygon_overlap),
        boundary_ratio=float(boundary_ratio),
        adjacent_overlap=float(adjacent_overlap),
        temporal_support=float(temporal_support),
        supported_frame_count=int(supported_frame_count),
        selected_frame_count=int(selected_frame_count),
        low_height_penalty=float(low_penalty),
        adjacent_penalty=float(adjacent_penalty),
        boundary_penalty=float(boundary_penalty),
        linearity_penalty=float(lin_penalty),
        outside_box_residual_penalty=float(outside_box_residual_penalty),
    )


def classify_state(
    score: float,
    selected_frame_count: int,
    inside_vehicle_point_count: int,
    height_support_value: float,
    slot_core_overlap: float,
    adjacent_penalty: float,
    boundary_penalty: float,
    low_height_penalty: float,
    linearity_penalty_value: float,
    config: BoxScoringConfig,
) -> tuple[str, str]:
    if selected_frame_count < 3:
        return "box_unknown_insufficient_visibility", "selected frame count is below minimum"
    if low_height_penalty >= config.low_height_penalty_state_min and inside_vehicle_point_count < config.min_vehicle_points_accumulated:
        return "box_low_height_residual", "height support is too weak for vehicle evidence"
    if (
        score >= config.state_vehicle_score_min
        and height_support_value >= config.state_vehicle_height_min
        and slot_core_overlap >= config.state_vehicle_core_overlap_min
        and adjacent_penalty < config.state_vehicle_adjacent_penalty_max
        and boundary_penalty < config.state_vehicle_boundary_penalty_max
    ):
        return "box_vehicle_core_supported", "best slot-constrained box has vehicle-height core support"
    if score >= config.state_adjacent_score_min and adjacent_penalty >= config.state_adjacent_penalty_min:
        return "box_adjacent_conflict", "best box overlaps adjacent slot evidence"
    if score >= config.state_boundary_score_min and boundary_penalty >= config.state_boundary_penalty_min:
        return "box_boundary_conflict", "best box support is boundary dominated"
    if linearity_penalty_value >= config.state_static_linearity_min:
        return "box_wall_like_or_static_suspect", "box support is line-like or static-structure-like"
    if low_height_penalty >= config.low_height_penalty_state_min:
        return "box_low_height_residual", "height support is too weak for vehicle evidence"
    return "box_no_vehicle_evidence", "no slot-constrained box reached vehicle evidence gates"


def score_slot_points(
    slot: dict,
    accumulated_points: np.ndarray,
    frame_points: list[np.ndarray],
    slots_by_id: dict[str, dict],
    map_units_per_meter: float,
    config: BoxScoringConfig,
) -> BoxScore:
    polygon = slot_polygon(slot, "polygon_np")
    if "center_np" in slot and slot["center_np"] is not None:
        center = np.asarray(slot["center_np"], dtype=np.float64)
    elif "center_map" in slot and slot["center_map"] is not None:
        center = np.asarray(slot["center_map"], dtype=np.float64)
    else:
        center = np.asarray(polygon, dtype=np.float64).mean(axis=0)
    frame = compute_slot_frame(polygon, center)
    if len(accumulated_points):
        ground_z = estimate_local_ground_z(accumulated_points[:, :3], config.ground_quantile)
        z_rel_all = accumulated_points[:, 2] - ground_z
        vehicle_all = vehicle_height_mask(z_rel_all, config)
        low_all = low_height_mask(z_rel_all, config)
        vehicle_points_xy = accumulated_points[vehicle_all, :2]
        vehicle_z_rel = z_rel_all[vehicle_all]
        low_points_xy = accumulated_points[low_all, :2]
        slot_margin = slot_polygon(slot, "margin_np")
        in_margin_vehicle_count = int((points_in_polygon(accumulated_points[:, :2], slot_margin) & vehicle_all).sum())
    else:
        z_rel_all = np.empty(0, dtype=np.float64)
        vehicle_all = np.zeros(0, dtype=bool)
        low_all = np.zeros(0, dtype=bool)
        vehicle_points_xy = np.empty((0, 2), dtype=np.float64)
        vehicle_z_rel = np.empty(0, dtype=np.float64)
        low_points_xy = np.empty((0, 2), dtype=np.float64)
        in_margin_vehicle_count = 0
    context = ScoreContext(
        accumulated_points=accumulated_points,
        frame_points=frame_points,
        z_rel_all=z_rel_all,
        vehicle_all=vehicle_all,
        low_all=low_all,
        vehicle_points_xy=vehicle_points_xy,
        vehicle_z_rel=vehicle_z_rel,
        low_points_xy=low_points_xy,
        in_margin_vehicle_count=in_margin_vehicle_count,
    )
    best: BoxScore | None = None
    best_box: CandidateBox | None = None
    for box in enumerate_box_hypotheses(frame, config):
        score = evaluate_box(slot, box, context, slots_by_id, config, include_temporal=False)
        if best is None or score.score > best.score:
            best = score
            best_box = box
    if best is None:
        raise ValueError(f"no box hypotheses generated for {slot.get('slot_id')}")
    if best_box is None:
        return best
    return evaluate_box(slot, best_box, context, slots_by_id, config, include_temporal=True)
