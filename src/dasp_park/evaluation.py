import numpy as np

from .datatypes import FREE, OCCLUDED_UNKNOWN, OCCUPIED, UNKNOWN, RegionProposal


def compute_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Return error map:
    0 = correct / ignored
    1 = false_free: pred free but gt occupied
    2 = false_occupied: pred occupied but gt free
    3 = unresolved_unknown: pred unknown or occluded where gt is known
    """
    error = np.zeros_like(pred, dtype=np.uint8)
    error[(pred == FREE) & (gt == OCCUPIED)] = 1
    error[(pred == OCCUPIED) & (gt == FREE)] = 2
    error[((pred == UNKNOWN) | (pred == OCCLUDED_UNKNOWN)) & ((gt == FREE) | (gt == OCCUPIED))] = 3
    return error


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute binary IoU with safe zero division."""
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def selected_region_error_overlap(
    error_before: np.ndarray,
    selected_regions: list[RegionProposal],
) -> float:
    """Compute fraction of error cells covered by selected regions."""
    total_error = int((error_before > 0).sum())
    if total_error == 0:
        return 0.0
    covered = np.zeros_like(error_before, dtype=bool)
    for region in selected_regions:
        i_min, j_min, i_max, j_max = region.bbox_cells
        covered[i_min:i_max, j_min:j_max] = True
    return float(((error_before > 0) & covered).sum() / total_error)


def target_slot_unknown_ratio(
    occupancy: np.ndarray,
    target_slot_mask: np.ndarray,
) -> float:
    """Compute unknown/occluded ratio inside target slot."""
    target = target_slot_mask > 0
    total = int(target.sum())
    if total == 0:
        return 0.0
    unknown = ((occupancy == UNKNOWN) | (occupancy == OCCLUDED_UNKNOWN)) & target
    return float(unknown.sum() / total)


def _reduction_percent(before: int, after: int) -> float:
    if before == 0:
        return 0.0
    return float(100.0 * (before - after) / before)


def compute_metrics_before_after(
    occupancy_before: np.ndarray,
    occupancy_after: np.ndarray,
    gt_occupancy: np.ndarray,
    selected_regions: list[RegionProposal],
    target_slot_mask: np.ndarray,
    num_tool_calls: int = 0,
) -> dict:
    """Compute before/after metrics for controlled synthetic demo."""
    error_before = compute_error_map(occupancy_before, gt_occupancy)
    error_after = compute_error_map(occupancy_after, gt_occupancy)

    false_free_before = int((error_before == 1).sum())
    false_free_after = int((error_after == 1).sum())
    false_occupied_before = int((error_before == 2).sum())
    false_occupied_after = int((error_after == 2).sum())
    unresolved_unknown_before = int((error_before == 3).sum())
    unresolved_unknown_after = int((error_after == 3).sum())

    return {
        "false_free_before": false_free_before,
        "false_free_after": false_free_after,
        "false_free_reduction_percent": _reduction_percent(false_free_before, false_free_after),
        "false_occupied_before": false_occupied_before,
        "false_occupied_after": false_occupied_after,
        "unresolved_unknown_before": unresolved_unknown_before,
        "unresolved_unknown_after": unresolved_unknown_after,
        "unresolved_unknown_reduction_percent": _reduction_percent(
            unresolved_unknown_before,
            unresolved_unknown_after,
        ),
        "occupied_iou_before": binary_iou(occupancy_before == OCCUPIED, gt_occupancy == OCCUPIED),
        "occupied_iou_after": binary_iou(occupancy_after == OCCUPIED, gt_occupancy == OCCUPIED),
        "unknown_cells_before": int(((occupancy_before == UNKNOWN) | (occupancy_before == OCCLUDED_UNKNOWN)).sum()),
        "unknown_cells_after": int(((occupancy_after == UNKNOWN) | (occupancy_after == OCCLUDED_UNKNOWN)).sum()),
        "target_slot_unknown_ratio_before": target_slot_unknown_ratio(occupancy_before, target_slot_mask),
        "target_slot_unknown_ratio_after": target_slot_unknown_ratio(occupancy_after, target_slot_mask),
        "selected_region_error_overlap": selected_region_error_overlap(error_before, selected_regions),
        "num_selected_regions": int(len(selected_regions)),
        "num_tool_calls": int(num_tool_calls),
    }
