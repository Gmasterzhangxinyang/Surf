import numpy as np

from .datatypes import FREE, OCCLUDED_UNKNOWN, OCCUPIED, UNKNOWN, RegionProposal, ToolResult
from .occupancy import grid_shape, valid_grid_mask, world_to_grid


def bbox_to_world_range(
    bbox: tuple[int, int, int, int],
    grid_cfg: dict,
) -> tuple[float, float, float, float]:
    """Convert bbox cells to world x/y min/max."""
    i_min, j_min, i_max, j_max = bbox
    res = grid_cfg["resolution"]
    x_min = grid_cfg["x_min"] + i_min * res
    x_max = grid_cfg["x_min"] + i_max * res
    y_min = grid_cfg["y_min"] + j_min * res
    y_max = grid_cfg["y_min"] + j_max * res
    return x_min, x_max, y_min, y_max


def lidar_geometry_checker(
    points_full: np.ndarray,
    region: RegionProposal,
    grid_cfg: dict,
) -> ToolResult:
    """
    Inspect local full LiDAR evidence inside selected region.

    This simulates active local re-inspection. It must not use gt_occupancy.
    """
    x_min, x_max, y_min, y_max = bbox_to_world_range(region.bbox_cells, grid_cfg)
    mask = (
        (points_full[:, 0] >= x_min)
        & (points_full[:, 0] < x_max)
        & (points_full[:, 1] >= y_min)
        & (points_full[:, 1] < y_max)
    )
    local_points = points_full[mask]
    point_count = int(local_points.shape[0])
    occupied_cells: list[list[int]] = []
    if point_count == 0:
        max_height = 0.0
        mean_height = 0.0
        occupied_confidence = 0.10
        free_confidence = 0.05
    else:
        max_height = float(local_points[:, 2].max())
        mean_height = float(local_points[:, 2].mean())
        if max_height > 0.25:
            occupied_confidence = min(1.0, 0.55 + max_height / 2.5)
            free_confidence = 0.05
        elif point_count >= 20:
            occupied_confidence = 0.10
            free_confidence = 0.85
        else:
            occupied_confidence = 0.20
            free_confidence = 0.20
        if max_height > 0.25:
            i, j = world_to_grid(local_points[:, 0], local_points[:, 1], grid_cfg)
            valid = valid_grid_mask(i, j, grid_shape(grid_cfg))
            i, j, z = i[valid], j[valid], local_points[:, 2][valid]
            high = z > 0.25
            cells = sorted({(int(row), int(col)) for row, col in zip(i[high], j[high])})
            occupied_cells = [[row, col] for row, col in cells]

    if occupied_confidence > free_confidence and occupied_confidence > 0.5:
        summary = "Local LiDAR evidence suggests this region is occupied."
    elif free_confidence > 0.5:
        summary = "Local LiDAR evidence suggests this region is likely free."
    else:
        summary = "Local LiDAR evidence is insufficient; keep unknown."

    updates = {
        "point_count": point_count,
        "max_height": float(max_height),
        "mean_height": float(mean_height),
        "occupied_confidence": float(occupied_confidence),
        "free_confidence": float(free_confidence),
        "occupied_cells": occupied_cells,
    }
    return ToolResult(
        tool_name="lidar_geometry_checker",
        region_id=region.region_id,
        confidence_delta=float(abs(occupied_confidence - free_confidence)),
        updates=updates,
        summary=summary,
    )


def occlusion_reasoning_tool(
    region: RegionProposal,
    occlusion: np.ndarray,
) -> ToolResult:
    """Inspect occlusion score inside selected region."""
    i_min, j_min, i_max, j_max = region.bbox_cells
    occlusion_mean = float(occlusion[i_min:i_max, j_min:j_max].mean())
    if occlusion_mean > 0.45:
        updates = {"occluded": True, "occlusion_mean": occlusion_mean}
        summary = "Region is behind an occupied boundary and should remain occluded_unknown."
        confidence_delta = 0.20
    else:
        updates = {"occluded": False, "occlusion_mean": occlusion_mean}
        summary = "Region is not strongly occluded."
        confidence_delta = 0.05
    return ToolResult(
        tool_name="occlusion_reasoning_tool",
        region_id=region.region_id,
        confidence_delta=confidence_delta,
        updates=updates,
        summary=summary,
    )


def image_crop_reinspect_placeholder(region: RegionProposal) -> ToolResult:
    """Placeholder visual re-inspection tool."""
    return ToolResult(
        tool_name="image_crop_reinspect_placeholder",
        region_id=region.region_id,
        confidence_delta=0.0,
        updates={"visual_confidence": 0.5},
        summary="Visual crop re-inspection is a placeholder in MVP.",
    )


def route_and_run_tools(
    region: RegionProposal,
    points_full: np.ndarray,
    occupancy: np.ndarray,
    occlusion: np.ndarray,
    grid_cfg: dict,
    selected_tool_names: list[str] | None = None,
) -> list[ToolResult]:
    """Route selected region to tools based on issue type."""
    del occupancy
    if region.issue_type == "occluded_unknown":
        allowed = ["occlusion_reasoning_tool"]
    elif region.issue_type == "decision_critical_uncertainty":
        allowed = ["lidar_geometry_checker", "image_crop_reinspect_placeholder"]
    else:
        allowed = ["lidar_geometry_checker"]

    requested = selected_tool_names or allowed
    tool_names = [name for name in requested if name in allowed]
    if not tool_names:
        tool_names = [allowed[0]]

    results = []
    for tool_name in tool_names:
        if tool_name == "occlusion_reasoning_tool":
            results.append(occlusion_reasoning_tool(region, occlusion))
        elif tool_name == "image_crop_reinspect_placeholder":
            results.append(image_crop_reinspect_placeholder(region))
        elif tool_name == "lidar_geometry_checker":
            results.append(lidar_geometry_checker(points_full, region, grid_cfg))
    return results


def update_occupancy_with_tool_results(
    occupancy_before: np.ndarray,
    regions: list[RegionProposal],
    all_tool_results: list[ToolResult],
    cfg: dict,
) -> np.ndarray:
    """Update occupancy locally based on tool results."""
    occupancy_after = occupancy_before.copy()
    by_region: dict[str, list[ToolResult]] = {}
    for result in all_tool_results:
        by_region.setdefault(result.region_id, []).append(result)

    tool_cfg = cfg["tools"]
    for region in regions:
        i_min, j_min, i_max, j_max = region.bbox_cells
        region_view = occupancy_after[i_min:i_max, j_min:j_max]
        for result in by_region.get(region.region_id, []):
            if result.tool_name == "occlusion_reasoning_tool" and result.updates.get("occluded"):
                region_view[region_view == UNKNOWN] = OCCLUDED_UNKNOWN
            if result.tool_name != "lidar_geometry_checker":
                continue
            occupied_conf = result.updates["occupied_confidence"]
            free_conf = result.updates["free_confidence"]
            point_count = result.updates["point_count"]
            if (
                occupied_conf > tool_cfg["occupied_confidence_threshold"]
                and point_count >= tool_cfg["min_points_for_occupied"]
            ):
                occupied_cells = result.updates.get("occupied_cells") or []
                if occupied_cells:
                    for row, col in occupied_cells:
                        if i_min <= row < i_max and j_min <= col < j_max:
                            occupancy_after[row, col] = OCCUPIED
                else:
                    region_view[:, :] = OCCUPIED
            elif (
                free_conf > tool_cfg["free_confidence_threshold"]
                and point_count >= tool_cfg["min_points_for_free"]
            ):
                region_view[region_view == UNKNOWN] = FREE
    return occupancy_after
