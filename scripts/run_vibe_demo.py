#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dasp_park.agent import (
    build_agent_reasoning_trace,
    filter_new_regions,
    propose_competitor_slot_region,
    propose_target_verification_region,
    select_top_regions,
)
from dasp_park.config import ensure_output_dir, load_config
from dasp_park.decision import (
    COMMIT_SLOT,
    DRIVE_FORWARD_EXPLORE,
    evaluate_parking_decision,
)
from dasp_park.diagnostics import (
    compute_decision_impact_map,
    compute_obstacle_boundary_map,
    compute_occlusion_map,
    compute_priority_map,
    compute_uncertainty_map,
    compute_unknown_map,
)
from dasp_park.evaluation import compute_error_map, compute_metrics_before_after, target_slot_unknown_ratio
from dasp_park.llm_agent import (
    briefing_to_markdown,
    build_ai_policy_context,
    build_llm_agent_context,
    call_openai_policy_advice,
    call_openai_briefing,
    validate_policy_advice,
    validate_policy_plan,
)
from dasp_park.occupancy import build_occupancy_from_lidar
from dasp_park.slot_selector import select_target_slot_from_known_map, slot_mask
from dasp_park.synthetic_scene import create_synthetic_parking_scene
from dasp_park.tools import route_and_run_tools, update_occupancy_with_tool_results
from dasp_park.visualization import (
    save_effectiveness_panel,
    save_agent_reasoning_panel,
    save_ai_agent_briefing_panel,
    save_error_map,
    save_lidar_bev,
    save_map,
    save_occupancy_map,
    save_slot_selection_panel,
    save_scene_overview,
    save_selected_regions_map,
    save_tool_evidence_panel,
)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _region_to_trace(region, tool_results):
    return {
        "region_id": region.region_id,
        "center_cell": list(region.center_cell),
        "bbox_cells": list(region.bbox_cells),
        "priority": float(region.priority),
        "issue_type": region.issue_type,
        "selected_tools": [r.tool_name for r in tool_results if r.region_id == region.region_id],
        "metadata": _to_jsonable(region.metadata),
    }


def _tool_to_trace(result):
    return {
        "tool_name": result.tool_name,
        "region_id": result.region_id,
        "confidence_delta": float(result.confidence_delta),
        "summary": result.summary,
        "updates": _to_jsonable(result.updates),
        "metadata": _to_jsonable(result.metadata),
    }


def _region_candidate_actions(regions):
    actions = []
    for idx, region in enumerate(regions):
        actions.append(
            {
                "action_id": f"a{idx}",
                "region_id": region.region_id,
                "issue_type": region.issue_type,
                "priority": float(region.priority),
                "active_question": region.metadata.get("active_question", ""),
                "why_selected": region.metadata.get("why_selected", ""),
                "allowed_tools": (
                    ["occlusion_reasoning_tool"]
                    if region.issue_type == "occluded_unknown"
                    else ["lidar_geometry_checker", "image_crop_reinspect_placeholder"]
                    if region.issue_type == "decision_critical_uncertainty"
                    else ["lidar_geometry_checker"]
                ),
                "is_target_verification": region.metadata.get("selection_note") == "target-slot verification action",
                "is_competitor_slot_exploration": (
                    region.metadata.get("selection_note") == "competitor-slot exploration action"
                ),
                "slot_id": region.metadata.get("slot_id"),
                "slot_rank": region.metadata.get("slot_rank"),
            }
        )
    return actions


def _compute_belief_maps(occupancy, slots, grid_cfg, cfg):
    unknown = compute_unknown_map(occupancy)
    boundary = compute_obstacle_boundary_map(occupancy)
    occlusion = compute_occlusion_map(occupancy, grid_cfg)
    uncertainty = compute_uncertainty_map(occupancy, occlusion, boundary)
    decision_impact = compute_decision_impact_map(occupancy, slots, grid_cfg)
    priority = compute_priority_map(uncertainty, occlusion, decision_impact, boundary, cfg["agent"]["weights"])
    return {
        "unknown": unknown,
        "boundary": boundary,
        "occlusion": occlusion,
        "uncertainty": uncertainty,
        "decision_impact": decision_impact,
        "priority": priority,
    }


def main():
    config_path = ROOT / "configs" / "vibe_demo.yaml"
    cfg = load_config(config_path)
    print("[DASP-Park] Loaded config: configs/vibe_demo.yaml")

    out_dir = ROOT / cfg["output_dir"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_output_dir(out_dir)

    sample = create_synthetic_parking_scene(cfg)
    print("[DASP-Park] Created deterministic synthetic parking scene.")

    grid_cfg = cfg["grid"]
    occupancy_before, _height_before = build_occupancy_from_lidar(sample.lidar_points_observed, grid_cfg)
    print("[DASP-Park] Built initial occupancy from observed sparse LiDAR only.")

    initial_pre_slot_maps = _compute_belief_maps(occupancy_before, sample.fake_slots, grid_cfg, cfg)
    selected_slot, slot_scores = select_target_slot_from_known_map(
        sample.fake_slots,
        occupancy_before,
        initial_pre_slot_maps["uncertainty"],
        initial_pre_slot_maps["occlusion"],
        grid_cfg,
    )
    initial_slot_scores = slot_scores
    sample.target_slot_mask = slot_mask(selected_slot, grid_cfg).astype("float32")
    initial_maps = _compute_belief_maps(occupancy_before, sample.fake_slots, grid_cfg, cfg)
    unknown = initial_maps["unknown"]
    boundary = initial_maps["boundary"]
    occlusion = initial_maps["occlusion"]
    uncertainty = initial_maps["uncertainty"]
    decision_impact = initial_maps["decision_impact"]
    priority = initial_maps["priority"]
    print("[DASP-Park] Computed unknown, occlusion, uncertainty, decision impact, priority maps.")
    print(f"[DASP-Park] Selected target slot from known slot map: {selected_slot.slot_id}")

    selected_regions = []
    tool_results = []
    agent_rounds = []
    target_slot_history = [
        {
            "round": 0,
            "slot_id": selected_slot.slot_id,
            "reason": "initial highest score from known slot map and sparse LiDAR belief",
        }
    ]
    occupancy_current = occupancy_before.copy()
    max_rounds = cfg["agent"].get("max_rounds", 1)
    regions_per_round = cfg["agent"].get("regions_per_round", cfg["agent"]["top_k_regions"])
    candidate_pool_per_round = cfg["agent"].get("candidate_pool_per_round", regions_per_round)
    competitor_slots_per_round = cfg["agent"].get("competitor_slots_per_round", 0)
    top_k_total = cfg["agent"]["top_k_regions"]
    stop_ratio = cfg["agent"].get("stop_target_unknown_ratio", 0.30)
    max_competitor_explorations = cfg["agent"].get("max_competitor_explorations", 2)
    phase = "verify_top1"
    competitor_explorations = 0
    top1_history = [slot_scores[0]["slot_id"]]
    final_decision = None

    for round_idx in range(max_rounds):
        remaining = top_k_total - len(selected_regions)
        if remaining <= 0:
            break
        round_maps = _compute_belief_maps(occupancy_current, sample.fake_slots, grid_cfg, cfg)
        proposal_cfg = {
            **cfg,
            "agent": {
                **cfg["agent"],
                "top_k_regions": max(candidate_pool_per_round * 4, remaining),
            },
        }
        candidates = []
        target_unknown_before_round = target_slot_unknown_ratio(occupancy_current, sample.target_slot_mask)
        top3_before = [item["slot_id"] for item in slot_scores[:3]]
        current_phase = phase
        if current_phase == "verify_top1" and target_unknown_before_round <= stop_ratio:
            break
        if current_phase == "verify_top1":
            target_candidate = propose_target_verification_region(
                sample.target_slot_mask,
                round_maps["priority"],
                round_maps["uncertainty"],
                round_maps["occlusion"],
                round_maps["decision_impact"],
                round_maps["boundary"],
                cfg,
                existing_regions=selected_regions,
            )
            if target_candidate is not None:
                candidates.append(target_candidate)
        else:
            slot_lookup = {slot.slot_id: slot for slot in sample.fake_slots}
            competitor_candidates = [item for item in slot_scores if item["slot_id"] != selected_slot.slot_id]
            for rank, score_item in enumerate(competitor_candidates[:competitor_slots_per_round], start=1):
                competitor_slot = slot_lookup[score_item["slot_id"]]
                competitor_candidate = propose_competitor_slot_region(
                    competitor_slot.slot_id,
                    rank,
                    slot_mask(competitor_slot, grid_cfg).astype("float32"),
                    round_maps["priority"],
                    round_maps["uncertainty"],
                    round_maps["occlusion"],
                    round_maps["decision_impact"],
                    round_maps["boundary"],
                    cfg,
                    existing_regions=selected_regions + candidates,
                )
                if competitor_candidate is not None:
                    candidates.append(competitor_candidate)
            candidates.extend(select_top_regions(
                round_maps["priority"],
                round_maps["uncertainty"],
                round_maps["occlusion"],
                round_maps["decision_impact"],
                round_maps["boundary"],
                proposal_cfg,
            ))
        candidate_regions = filter_new_regions(
            candidates,
            selected_regions,
            candidate_pool_per_round,
            len(selected_regions),
        )
        if not candidate_regions:
            break

        candidate_actions = _region_candidate_actions(candidate_regions)
        max_actions_this_round = min(regions_per_round, remaining, len(candidate_regions))
        ai_policy_context = build_ai_policy_context(
            round_idx + 1,
            selected_slot.slot_id,
            candidate_actions,
            target_unknown_before_round,
            slot_scores,
            agent_rounds,
            phase=current_phase,
            ranking_context={
                "top3_before": top3_before,
                "stop_rule": "Stop when top3 ranking stays unchanged after a tool-backed belief update.",
                "competitor_explorations_used": competitor_explorations,
                "max_competitor_explorations": max_competitor_explorations,
            },
        )
        ai_policy_advice = call_openai_policy_advice(
            ai_policy_context,
            cfg.get("llm_agent", {}),
            max_actions=max_actions_this_round,
        )
        selected_plan = validate_policy_plan(ai_policy_advice, candidate_actions, max_actions=max_actions_this_round)
        selected_action_ids = [item["action_id"] for item in selected_plan]
        action_to_region = {action["action_id"]: region for action, region in zip(candidate_actions, candidate_regions)}
        action_to_action = {action["action_id"]: action for action in candidate_actions}
        action_to_plan = {item["action_id"]: item for item in selected_plan}
        new_regions = [action_to_region[action_id] for action_id in selected_action_ids]

        round_tool_results = []
        tool_call_reasons = []
        for region in new_regions:
            region_action_id = next(
                action_id for action_id, candidate_region in action_to_region.items() if candidate_region is region
            )
            action = action_to_action.get(region_action_id, {})
            plan_item = action_to_plan.get(
                region_action_id,
                {},
            )
            plan_reason = plan_item.get("reason") or action.get("why_selected") or action.get("active_question", "")
            region_results = route_and_run_tools(
                region,
                sample.lidar_points_full,
                occupancy_current,
                round_maps["occlusion"],
                grid_cfg,
                selected_tool_names=plan_item.get("tool_ids"),
            )
            for result in region_results:
                result.metadata["tool_call_reason"] = plan_reason
                result.metadata["active_question"] = action.get("active_question", "")
                result.metadata["validated_action_id"] = region_action_id
                tool_call_reasons.append(
                    {
                        "tool_name": result.tool_name,
                        "region_id": result.region_id,
                        "reason": plan_reason,
                        "active_question": action.get("active_question", ""),
                        "allowed_tools": action.get("allowed_tools", []),
                    }
                )
            round_tool_results.extend(region_results)

        occupancy_next = update_occupancy_with_tool_results(occupancy_current, new_regions, round_tool_results, cfg)
        target_unknown_after_round = target_slot_unknown_ratio(occupancy_next, sample.target_slot_mask)
        post_round_maps = _compute_belief_maps(occupancy_next, sample.fake_slots, grid_cfg, cfg)
        previous_slot_id = selected_slot.slot_id
        updated_slot, updated_slot_scores = select_target_slot_from_known_map(
            sample.fake_slots,
            occupancy_next,
            post_round_maps["uncertainty"],
            post_round_maps["occlusion"],
            grid_cfg,
        )
        top3_after = [item["slot_id"] for item in updated_slot_scores[:3]]
        ranking_changed = top3_after != top3_before
        top1_history.append(updated_slot_scores[0]["slot_id"])
        if ranking_changed and current_phase == "verify_top1":
            phase = "explore_competitor"
            stop_reason = "top3_changed_after_top1_verification"
        elif ranking_changed:
            competitor_explorations += 1
            phase = "explore_competitor"
            stop_reason = "top3_changed_after_competitor_exploration"
        else:
            stop_reason = "top3_ranking_stable_but_final_checks_required"
        switched_target = updated_slot.slot_id != previous_slot_id
        if switched_target:
            selected_slot = updated_slot
            slot_scores = updated_slot_scores
            sample.target_slot_mask = slot_mask(selected_slot, grid_cfg).astype("float32")
            target_slot_history.append(
                {
                    "round": round_idx + 1,
                    "slot_id": selected_slot.slot_id,
                    "reason": "target re-scored after local tool evidence changed the belief",
                }
            )
        else:
            slot_scores = updated_slot_scores
        selected_regions.extend(new_regions)
        tool_results.extend(round_tool_results)
        final_decision = evaluate_parking_decision(
            slot_scores,
            top1_history,
            rounds_completed=round_idx + 1,
            max_rounds=max_rounds,
            competitor_explorations=competitor_explorations,
            max_competitor_explorations=max_competitor_explorations,
            cfg=cfg,
        )
        stop_after_round = final_decision.action in {COMMIT_SLOT, DRIVE_FORWARD_EXPLORE}
        if stop_after_round:
            stop_reason = final_decision.action
        agent_rounds.append(
            {
                "round": round_idx + 1,
                "observation": {
                    "target_slot_unknown_ratio_before": target_unknown_before_round,
                    "target_slot_before": previous_slot_id,
                    "num_regions_already_selected": len(selected_regions) - len(new_regions),
                },
                "selected_regions": [region.region_id for region in new_regions],
                "candidate_actions": candidate_actions,
                "ai_policy_advice": ai_policy_advice,
                "validated_plan": selected_plan,
                "validated_action_ids": selected_action_ids,
                "active_questions": [
                    region.metadata.get("active_question", "What local evidence is missing?") for region in new_regions
                ],
                "tool_calls": [result.tool_name for result in round_tool_results],
                "tool_call_reasons": tool_call_reasons,
                "phase": current_phase,
                "top3_before": top3_before,
                "top3_after": top3_after,
                "ranking_changed": ranking_changed,
                "parking_decision_after_round": final_decision.to_dict(),
                "target_slot_unknown_ratio_after": target_unknown_after_round,
                "target_slot_after": selected_slot.slot_id,
                "switched_target": switched_target,
                "stop_reason": stop_reason,
                "stop_after_round": stop_after_round or len(selected_regions) >= top_k_total,
            }
        )
        occupancy_current = occupancy_next
        if stop_after_round or len(selected_regions) >= top_k_total:
            break

    if final_decision is None:
        final_decision = evaluate_parking_decision(
            slot_scores,
            top1_history,
            rounds_completed=len(agent_rounds),
            max_rounds=max_rounds,
            competitor_explorations=competitor_explorations,
            max_competitor_explorations=max_competitor_explorations,
            cfg=cfg,
        )

    print(f"[DASP-Park] Completed {len(agent_rounds)} active perception rounds.")
    print(f"[DASP-Park] Selected {len(selected_regions)} high-priority regions.")
    print(f"[DASP-Park] Ran {len(tool_results)} tool calls.")

    occupancy_after = occupancy_current
    error_before = compute_error_map(occupancy_before, sample.gt_occupancy)
    error_after = compute_error_map(occupancy_after, sample.gt_occupancy)
    metrics = compute_metrics_before_after(
        occupancy_before,
        occupancy_after,
        sample.gt_occupancy,
        selected_regions,
        sample.target_slot_mask,
        num_tool_calls=len(tool_results),
    )
    agent_reasoning = build_agent_reasoning_trace(
        selected_slot,
        slot_scores,
        selected_regions,
        tool_results,
        metrics,
        rounds=agent_rounds,
        final_decision=final_decision.to_dict(),
    )
    print("[DASP-Park] Computed before/after metrics.")

    slots = sample.fake_slots
    save_scene_overview(sample, out_dir / "00_scene_overview.png", grid_cfg)
    save_slot_selection_panel(initial_slot_scores, out_dir / "00_slot_selection_scores.png")
    save_slot_selection_panel(slot_scores, out_dir / "00_final_slot_selection_scores.png")
    save_occupancy_map(sample.gt_occupancy, "Ground Truth Occupancy", out_dir / "01_gt_occupancy.png", grid_cfg, slots)
    save_lidar_bev(
        sample.lidar_points_observed,
        "Observed LiDAR Returns: Ground Faint, Obstacles Colored",
        out_dir / "02_initial_observed_lidar.png",
        grid_cfg,
        slots,
    )
    save_occupancy_map(occupancy_before, "Occupancy Before Active Refinement", out_dir / "03_occupancy_before.png", grid_cfg, slots)
    save_error_map(error_before, "Error Before Active Refinement", out_dir / "04_error_before.png", grid_cfg, slots=slots)
    save_map(unknown, "Unknown Map", out_dir / "05_unknown_map.png", grid_cfg, cmap="gray", vmin=0, vmax=1, colorbar_label="unknown", slots=slots)
    save_map(occlusion, "Occlusion Map", out_dir / "06_occlusion_map.png", grid_cfg, cmap="cividis", vmin=0, vmax=1, colorbar_label="occlusion", slots=slots)
    save_map(uncertainty, "Uncertainty Map", out_dir / "07_uncertainty_map.png", grid_cfg, cmap="viridis", vmin=0, vmax=1, colorbar_label="uncertainty", slots=slots)
    save_map(decision_impact, "Decision Impact Map", out_dir / "08_decision_impact_map.png", grid_cfg, cmap="plasma", vmin=0, vmax=1, colorbar_label="decision impact", slots=slots)
    save_map(priority, "Priority Map", out_dir / "09_priority_map.png", grid_cfg, cmap="inferno", vmin=0, vmax=1, colorbar_label="priority", slots=slots)
    save_selected_regions_map(priority, "Selected Regions on Priority", out_dir / "10_selected_regions_on_priority.png", grid_cfg, selected_regions, slots)
    save_error_map(error_before, "Selected Regions on Error Before", out_dir / "11_selected_regions_on_error.png", grid_cfg, selected_regions, slots)
    save_tool_evidence_panel(selected_regions, tool_results, out_dir / "12_tool_evidence.png")
    save_occupancy_map(occupancy_after, "Occupancy After Active Refinement", out_dir / "13_occupancy_after.png", grid_cfg, slots, selected_regions)
    save_error_map(error_after, "Error After Active Refinement", out_dir / "14_error_after.png", grid_cfg, selected_regions, slots)
    save_effectiveness_panel(
        sample.gt_occupancy,
        occupancy_before,
        error_before,
        priority,
        occupancy_after,
        error_after,
        metrics,
        out_dir / "15_effectiveness_panel.png",
        grid_cfg,
        selected_regions,
        slots,
    )
    save_agent_reasoning_panel(agent_reasoning, out_dir / "16_agent_reasoning_panel.png")

    llm_cfg = cfg.get("llm_agent", {})
    ai_briefing = None
    if llm_cfg.get("enabled", False):
        llm_context = build_llm_agent_context(agent_reasoning, target_slot_history, metrics, final_decision.to_dict())
        ai_briefing = call_openai_briefing(llm_context, llm_cfg)
        with open(out_dir / llm_cfg.get("output_json", "ai_agent_briefing.json"), "w", encoding="utf-8") as f:
            json.dump(ai_briefing, f, indent=2, ensure_ascii=False)
        with open(out_dir / llm_cfg.get("output_markdown", "ai_agent_briefing.md"), "w", encoding="utf-8") as f:
            f.write(briefing_to_markdown(ai_briefing))
        save_ai_agent_briefing_panel(ai_briefing, out_dir / "17_ai_agent_briefing.png")

    agent_trace = {
        "sample_id": sample.sample_id,
        "note": "Controlled synthetic demo. Ground truth is used only for evaluation, not for agent selection or belief update.",
        "grid": cfg["grid"],
        "agent_reasoning": agent_reasoning,
        "ai_agent_briefing": ai_briefing,
        "target_slot_history": target_slot_history,
        "selected_regions": [_region_to_trace(r, tool_results) for r in selected_regions],
        "target_slot": {
            "slot_id": selected_slot.slot_id,
            "note": "Selected from known candidate slot map using current belief, not ground truth.",
        },
        "final_parking_decision": final_decision.to_dict(),
        "slot_scores": slot_scores,
        "tool_results": [_tool_to_trace(r) for r in tool_results],
        "metrics_summary": {
            "false_free_before": metrics["false_free_before"],
            "false_free_after": metrics["false_free_after"],
            "occupied_iou_before": metrics["occupied_iou_before"],
            "occupied_iou_after": metrics["occupied_iou_after"],
        },
    }
    with open(out_dir / "agent_trace.json", "w", encoding="utf-8") as f:
        json.dump(agent_trace, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_before_after.json", "w", encoding="utf-8") as f:
        json.dump({"sample_id": sample.sample_id, "metrics": metrics}, f, indent=2, ensure_ascii=False)

    print("[DASP-Park] Saved visualizations to outputs/vibe_demo")
    print("[DASP-Park] Done.")


if __name__ == "__main__":
    main()
