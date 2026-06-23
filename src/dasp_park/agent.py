import numpy as np

from .datatypes import RegionProposal, SlotHypothesis, ToolResult


def crop_region(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop array by bbox cells."""
    i_min, j_min, i_max, j_max = bbox
    return arr[i_min:i_max, j_min:j_max]


def bbox_from_center(
    center: tuple[int, int],
    region_size: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Create clipped bbox from center cell."""
    half = region_size // 2
    i, j = center
    i_min = max(0, i - half)
    j_min = max(0, j - half)
    i_max = min(shape[0], i_min + region_size)
    j_max = min(shape[1], j_min + region_size)
    i_min = max(0, i_max - region_size)
    j_min = max(0, j_max - region_size)
    return i_min, j_min, i_max, j_max


def diagnose_region_issue(
    bbox: tuple[int, int, int, int],
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    decision_impact: np.ndarray,
    boundary: np.ndarray,
) -> str:
    """Diagnose dominant issue type in a region."""
    u_mean = float(crop_region(uncertainty, bbox).mean())
    o_mean = float(crop_region(occlusion, bbox).mean())
    d_mean = float(crop_region(decision_impact, bbox).mean())
    b_mean = float(crop_region(boundary, bbox).mean())

    if o_mean > 0.45:
        return "occluded_unknown"
    if d_mean > 0.65 and u_mean > 0.35:
        return "decision_critical_uncertainty"
    if b_mean > 0.25:
        return "obstacle_boundary"
    if u_mean > 0.60:
        return "unknown_high_uncertainty"
    return "general_check"


def issue_to_active_question(issue_type: str) -> str:
    """Convert a region issue type into an explicit active-perception question."""
    return {
        "occluded_unknown": "Is this occluded area hiding an obstacle relevant to parking?",
        "decision_critical_uncertainty": "Is the target slot or entrance clear enough to trust?",
        "obstacle_boundary": "Is the obstacle boundary accurate enough for a parking maneuver?",
        "unknown_high_uncertainty": "Can local evidence reduce this high-uncertainty unknown area?",
        "general_check": "Does local evidence confirm or correct the current belief here?",
    }.get(issue_type, "What local evidence is missing in this region?")


def issue_to_tool_reason(issue_type: str) -> str:
    """Explain why a tool is selected for an issue type."""
    return {
        "occluded_unknown": "Use occlusion reasoning because raw visibility is blocked behind an occupied boundary.",
        "decision_critical_uncertainty": "Use local LiDAR evidence and a visual placeholder because this region affects the parking slot decision.",
        "obstacle_boundary": "Use LiDAR geometry because height evidence can refine occupied boundaries.",
        "unknown_high_uncertainty": "Use LiDAR geometry because local points can convert unknown cells into free or occupied evidence.",
        "general_check": "Use LiDAR geometry as a low-cost local consistency check.",
    }.get(issue_type, "Use the available local inspection tool.")


def issue_to_selection_reason(issue_type: str) -> str:
    """Explain why a region is worth active inspection."""
    return {
        "occluded_unknown": "Region has high occlusion and may hide parking-relevant risk.",
        "decision_critical_uncertainty": "Region has high decision impact and uncertainty near the selected target slot.",
        "obstacle_boundary": "Region lies near an obstacle boundary where occupancy errors matter for maneuver clearance.",
        "unknown_high_uncertainty": "Region has high uncertainty from missing observation.",
        "general_check": "Region has high combined priority from the diagnostic maps.",
    }.get(issue_type, "Region has high active-perception priority.")


def select_top_regions(
    priority: np.ndarray,
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    decision_impact: np.ndarray,
    boundary: np.ndarray,
    cfg: dict,
) -> list[RegionProposal]:
    """Select non-overlapping high-priority regions."""
    top_k = cfg["agent"]["top_k_regions"]
    region_size = cfg["agent"]["region_size_cells"]
    nms_radius = cfg["agent"]["nms_radius_cells"]
    shape = priority.shape
    work = priority.copy()
    selected: list[RegionProposal] = []

    for region_idx in range(top_k):
        flat_idx = int(np.argmax(work))
        score = float(work.flat[flat_idx])
        if score <= 0.0:
            break
        center = tuple(int(v) for v in np.unravel_index(flat_idx, shape))
        bbox = bbox_from_center(center, region_size, shape)
        issue_type = diagnose_region_issue(bbox, uncertainty, occlusion, decision_impact, boundary)
        selected.append(
            RegionProposal(
                region_id=f"r{region_idx}",
                center_cell=center,
                bbox_cells=bbox,
                priority=score,
                issue_type=issue_type,
                metadata={
                    "mean_uncertainty": float(crop_region(uncertainty, bbox).mean()),
                    "mean_occlusion": float(crop_region(occlusion, bbox).mean()),
                    "mean_decision_impact": float(crop_region(decision_impact, bbox).mean()),
                    "mean_boundary": float(crop_region(boundary, bbox).mean()),
                    "active_question": issue_to_active_question(issue_type),
                    "why_selected": issue_to_selection_reason(issue_type),
                    "chosen_tool_reason": issue_to_tool_reason(issue_type),
                },
            )
        )

        ci, cj = center
        i0, i1 = max(0, ci - nms_radius), min(shape[0], ci + nms_radius + 1)
        j0, j1 = max(0, cj - nms_radius), min(shape[1], cj + nms_radius + 1)
        yy, xx = np.ogrid[i0:i1, j0:j1]
        suppress = (yy - ci) ** 2 + (xx - cj) ** 2 <= nms_radius**2
        work[i0:i1, j0:j1][suppress] = -1.0

    if selected and not any(r.issue_type == "decision_critical_uncertainty" for r in selected):
        candidate_mask = (decision_impact >= 0.95) & (uncertainty > 0.35)
        candidate_scores = np.where(candidate_mask, priority, -1.0)
        for flat_idx in np.argsort(candidate_scores.ravel())[::-1]:
            score = float(candidate_scores.ravel()[flat_idx])
            if score <= 0.0:
                break
            center = tuple(int(v) for v in np.unravel_index(int(flat_idx), shape))
            bbox = bbox_from_center(center, region_size, shape)
            issue_type = diagnose_region_issue(bbox, uncertainty, occlusion, decision_impact, boundary)
            if issue_type != "decision_critical_uncertainty":
                continue
            replacement_idx = len(selected) - 1
            selected[replacement_idx] = RegionProposal(
                region_id=f"r{replacement_idx}",
                center_cell=center,
                bbox_cells=bbox,
                priority=score,
                issue_type=issue_type,
                metadata={
                    "mean_uncertainty": float(crop_region(uncertainty, bbox).mean()),
                    "mean_occlusion": float(crop_region(occlusion, bbox).mean()),
                    "mean_decision_impact": float(crop_region(decision_impact, bbox).mean()),
                    "mean_boundary": float(crop_region(boundary, bbox).mean()),
                    "selection_note": "highest-priority decision-critical candidate",
                    "active_question": issue_to_active_question(issue_type),
                    "why_selected": issue_to_selection_reason(issue_type),
                    "chosen_tool_reason": issue_to_tool_reason(issue_type),
                },
            )
            break

    return selected


def region_overlap_fraction(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Return intersection over the smaller bbox area."""
    ai0, aj0, ai1, aj1 = a
    bi0, bj0, bi1, bj1 = b
    ii0, ij0 = max(ai0, bi0), max(aj0, bj0)
    ii1, ij1 = min(ai1, bi1), min(aj1, bj1)
    inter = max(0, ii1 - ii0) * max(0, ij1 - ij0)
    area_a = max(1, (ai1 - ai0) * (aj1 - aj0))
    area_b = max(1, (bi1 - bi0) * (bj1 - bj0))
    return float(inter / min(area_a, area_b))


def filter_new_regions(
    candidates: list[RegionProposal],
    existing: list[RegionProposal],
    max_new: int,
    start_index: int,
    max_overlap: float = 0.25,
) -> list[RegionProposal]:
    """Keep deterministic non-duplicate regions and assign globally unique ids."""
    kept: list[RegionProposal] = []
    occupied = list(existing)
    for candidate in candidates:
        candidate_max_overlap = 0.85 if candidate.metadata.get("selection_note") == "target-slot verification action" else max_overlap
        if any(region_overlap_fraction(candidate.bbox_cells, region.bbox_cells) > candidate_max_overlap for region in occupied):
            continue
        new_region = RegionProposal(
            region_id=f"r{start_index + len(kept)}",
            center_cell=candidate.center_cell,
            bbox_cells=candidate.bbox_cells,
            priority=candidate.priority,
            issue_type=candidate.issue_type,
            metadata=dict(candidate.metadata),
        )
        kept.append(new_region)
        occupied.append(new_region)
        if len(kept) >= max_new:
            break
    return kept


def propose_target_verification_region(
    target_slot_mask: np.ndarray,
    priority: np.ndarray,
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    decision_impact: np.ndarray,
    boundary: np.ndarray,
    cfg: dict,
    existing_regions: list[RegionProposal] | None = None,
) -> RegionProposal | None:
    """Propose a target-slot verification region when the target is still uncertain."""
    mask = target_slot_mask > 0
    if existing_regions:
        mask = mask.copy()
        for region in existing_regions:
            i_min, j_min, i_max, j_max = region.bbox_cells
            mask[i_min:i_max, j_min:j_max] = False
    if not mask.any():
        return None
    target_score = np.where(mask, priority + 0.50 * uncertainty + 0.25 * decision_impact, -1.0)
    flat_idx = int(np.argmax(target_score))
    if float(target_score.flat[flat_idx]) <= 0.0:
        return None
    center = tuple(int(v) for v in np.unravel_index(flat_idx, priority.shape))
    bbox = bbox_from_center(center, cfg["agent"]["region_size_cells"], priority.shape)
    issue_type = "decision_critical_uncertainty"
    return RegionProposal(
        region_id="candidate_target",
        center_cell=center,
        bbox_cells=bbox,
        priority=float(priority[center]),
        issue_type=issue_type,
        metadata={
            "mean_uncertainty": float(crop_region(uncertainty, bbox).mean()),
            "mean_occlusion": float(crop_region(occlusion, bbox).mean()),
            "mean_decision_impact": float(crop_region(decision_impact, bbox).mean()),
            "mean_boundary": float(crop_region(boundary, bbox).mean()),
            "active_question": issue_to_active_question(issue_type),
            "why_selected": "Target slot is still uncertain, so the agent reserves an action to verify it directly.",
            "chosen_tool_reason": issue_to_tool_reason(issue_type),
            "selection_note": "target-slot verification action",
        },
    )


def propose_competitor_slot_region(
    slot_id: str,
    slot_rank: int,
    competitor_slot_mask: np.ndarray,
    priority: np.ndarray,
    uncertainty: np.ndarray,
    occlusion: np.ndarray,
    decision_impact: np.ndarray,
    boundary: np.ndarray,
    cfg: dict,
    existing_regions: list[RegionProposal] | None = None,
) -> RegionProposal | None:
    """Propose an exploration region inside a high-scoring competing slot."""
    mask = competitor_slot_mask > 0
    if existing_regions:
        mask = mask.copy()
        for region in existing_regions:
            i_min, j_min, i_max, j_max = region.bbox_cells
            mask[i_min:i_max, j_min:j_max] = False
    if not mask.any():
        return None
    explore_score = np.where(mask, priority + 0.45 * uncertainty + 0.30 * decision_impact, -1.0)
    flat_idx = int(np.argmax(explore_score))
    if float(explore_score.flat[flat_idx]) <= 0.0:
        return None
    center = tuple(int(v) for v in np.unravel_index(flat_idx, priority.shape))
    bbox = bbox_from_center(center, cfg["agent"]["region_size_cells"], priority.shape)
    issue_type = "decision_critical_uncertainty"
    return RegionProposal(
        region_id=f"candidate_competitor_{slot_rank}",
        center_cell=center,
        bbox_cells=bbox,
        priority=float(priority[center]),
        issue_type=issue_type,
        metadata={
            "slot_id": slot_id,
            "slot_rank": slot_rank,
            "mean_uncertainty": float(crop_region(uncertainty, bbox).mean()),
            "mean_occlusion": float(crop_region(occlusion, bbox).mean()),
            "mean_decision_impact": float(crop_region(decision_impact, bbox).mean()),
            "mean_boundary": float(crop_region(boundary, bbox).mean()),
            "active_question": f"Could competing slot {slot_id} be safer than the current target?",
            "why_selected": (
                f"{slot_id} is a high-scoring competing slot; exploring it reduces bias from only "
                "rechecking the current target."
            ),
            "chosen_tool_reason": issue_to_tool_reason(issue_type),
            "selection_note": "competitor-slot exploration action",
        },
    )


def summarize_belief_update(region: RegionProposal, results: list[ToolResult]) -> str:
    """Summarize what the tool results changed or preserved."""
    if not results:
        return "No local tool result was produced."
    summaries = []
    for result in results:
        if result.tool_name == "occlusion_reasoning_tool":
            if result.updates.get("occluded"):
                summaries.append("unknown cells remain marked as occluded_unknown")
            else:
                summaries.append("region is not strongly occluded")
        elif result.tool_name == "lidar_geometry_checker":
            if result.updates.get("occupied_confidence", 0.0) > 0.65:
                summaries.append("high local LiDAR points update evidence cells to occupied")
            elif result.updates.get("free_confidence", 0.0) > 0.80:
                summaries.append("low-height local LiDAR points update unknown cells to free")
            else:
                summaries.append("local LiDAR evidence is insufficient; keep belief conservative")
        elif result.tool_name == "image_crop_reinspect_placeholder":
            summaries.append("visual check is recorded as placeholder evidence")
    return "; ".join(summaries)


def build_agent_reasoning_trace(
    target_slot: SlotHypothesis,
    slot_scores: list[dict],
    selected_regions: list[RegionProposal],
    tool_results: list[ToolResult],
    metrics: dict,
    rounds: list[dict] | None = None,
) -> dict:
    """Build an explicit, human-readable reasoning trace for the active perception agent."""
    best_score = slot_scores[0] if slot_scores else {}
    results_by_region: dict[str, list[ToolResult]] = {}
    for result in tool_results:
        results_by_region.setdefault(result.region_id, []).append(result)

    actions = []
    for region in selected_regions:
        region_results = results_by_region.get(region.region_id, [])
        actions.append(
            {
                "region_id": region.region_id,
                "issue_type": region.issue_type,
                "active_question": region.metadata.get("active_question", issue_to_active_question(region.issue_type)),
                "why_selected": region.metadata.get("why_selected", issue_to_selection_reason(region.issue_type)),
                "parking_relevance": {
                    "mean_decision_impact": region.metadata.get("mean_decision_impact", 0.0),
                    "mean_uncertainty": region.metadata.get("mean_uncertainty", 0.0),
                    "mean_occlusion": region.metadata.get("mean_occlusion", 0.0),
                    "priority": region.priority,
                },
                "selected_tools": [r.tool_name for r in region_results],
                "chosen_tool_reason": region.metadata.get("chosen_tool_reason", issue_to_tool_reason(region.issue_type)),
                "belief_update_summary": summarize_belief_update(region, region_results),
            }
        )

    return {
        "goal": "Confirm a parking target using active perception before trusting the occupancy belief.",
        "agent_type": "decision-aware active perception policy, not an LLM and not a vehicle controller",
        "belief_inputs": [
            "known candidate slot map",
            "LiDAR-derived occupancy belief",
            "unknown map",
            "occlusion map",
            "uncertainty map",
            "decision impact map",
            "obstacle boundary map",
        ],
        "active_questions": [
            "Which known candidate slot is currently most plausible?",
            "What important parking area is still unknown or occluded?",
            "Which local region should be inspected next?",
            "Which tool can answer that local question?",
            "Did local evidence improve the belief without using ground truth?",
        ],
        "target_slot_selection": {
            "selected_slot_id": target_slot.slot_id,
            "reason": "highest score from current belief over the known candidate slot map",
            "best_score": best_score,
        },
        "actions": actions,
        "rounds": rounds or [],
        "observed_improvement": {
            "occupied_iou_before": metrics.get("occupied_iou_before"),
            "occupied_iou_after": metrics.get("occupied_iou_after"),
            "target_slot_unknown_ratio_before": metrics.get("target_slot_unknown_ratio_before"),
            "target_slot_unknown_ratio_after": metrics.get("target_slot_unknown_ratio_after"),
            "false_free_before": metrics.get("false_free_before"),
            "false_free_after": metrics.get("false_free_after"),
        },
    }
