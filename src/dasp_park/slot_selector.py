import numpy as np

from .datatypes import FREE, OCCLUDED_UNKNOWN, OCCUPIED, UNKNOWN, SlotHypothesis
from .diagnostics import mark_world_box
from .occupancy import grid_shape


def slot_mask(slot: SlotHypothesis, grid_cfg: dict) -> np.ndarray:
    """Return a binary mask for a known candidate parking slot."""
    mask = np.zeros(grid_shape(grid_cfg), dtype=np.float32)
    x_min, y_min = slot.polygon_xy.min(axis=0)
    x_max, y_max = slot.polygon_xy.max(axis=0)
    mark_world_box(mask, grid_cfg, float(x_min), float(x_max), float(y_min), float(y_max), 1.0, mode="set")
    return mask > 0


def slot_entrance_mask(slot: SlotHypothesis, grid_cfg: dict, depth_m: float = 2.0) -> np.ndarray:
    """Return a mask around the slot entrance."""
    mask = np.zeros(grid_shape(grid_cfg), dtype=np.float32)
    x_min = float(slot.entrance_xy[:, 0].min())
    x_max = float(slot.entrance_xy[:, 0].max())
    entrance_y = float(slot.entrance_xy[:, 1].mean())
    y_min = entrance_y - depth_m / 2
    y_max = entrance_y + depth_m / 2
    mark_world_box(mask, grid_cfg, x_min, x_max, y_min, y_max, 1.0, mode="set")
    return mask > 0


def score_candidate_slot(
    slot: SlotHypothesis,
    occupancy: np.ndarray,
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    grid_cfg: dict,
) -> dict:
    """Score one known slot using current belief, not ground truth."""
    mask = slot_mask(slot, grid_cfg)
    entrance = slot_entrance_mask(slot, grid_cfg)
    total = max(1, int(mask.sum()))
    entrance_total = max(1, int(entrance.sum()))

    free_ratio = float(((occupancy == FREE) & mask).sum() / total)
    occupied_ratio = float(((occupancy == OCCUPIED) & mask).sum() / total)
    unknown_ratio = float((((occupancy == UNKNOWN) | (occupancy == OCCLUDED_UNKNOWN)) & mask).sum() / total)
    uncertainty_mean = float(uncertainty[mask].mean()) if mask.any() else 1.0
    occlusion_mean = float(occlusion[mask].mean()) if mask.any() else 1.0
    entrance_unknown = float((((occupancy == UNKNOWN) | (occupancy == OCCLUDED_UNKNOWN)) & entrance).sum() / entrance_total)
    entrance_occupied = float(((occupancy == OCCUPIED) & entrance).sum() / entrance_total)
    entrance_free = float(((occupancy == FREE) & entrance).sum() / entrance_total)

    center = slot.metadata.get("center", [20.0, 0.0])
    distance_cost = float(np.hypot(center[0], center[1]) / 45.0)
    preference_bonus = float(slot.metadata.get("preference_bonus", 0.0))

    score = (
        preference_bonus
        + 1.20 * free_ratio
        - 6.00 * occupied_ratio
        - 0.55 * unknown_ratio
        - 0.55 * uncertainty_mean
        - 0.70 * occlusion_mean
        - 2.00 * entrance_occupied
        - 0.30 * entrance_unknown
        - 0.15 * distance_cost
    )
    return {
        "slot_id": slot.slot_id,
        "score": float(score),
        "free_ratio": free_ratio,
        "occupied_ratio": occupied_ratio,
        "unknown_ratio": unknown_ratio,
        "uncertainty_mean": uncertainty_mean,
        "occlusion_mean": occlusion_mean,
        "entrance_unknown": entrance_unknown,
        "entrance_occupied": entrance_occupied,
        "entrance_free": entrance_free,
        "distance_cost": distance_cost,
        "preference_bonus": preference_bonus,
    }


def select_target_slot_from_known_map(
    slots: list[SlotHypothesis],
    occupancy: np.ndarray,
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    grid_cfg: dict,
) -> tuple[SlotHypothesis, list[dict]]:
    """Select a target slot from a known candidate slot map using current belief."""
    scores = [score_candidate_slot(slot, occupancy, uncertainty, occlusion, grid_cfg) for slot in slots]
    scores.sort(key=lambda item: item["score"], reverse=True)
    best_id = scores[0]["slot_id"]
    for slot in slots:
        slot.is_target_candidate = slot.slot_id == best_id
    selected = next(slot for slot in slots if slot.slot_id == best_id)
    return selected, scores
