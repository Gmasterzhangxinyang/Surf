#!/usr/bin/env python3
"""Blind rerun using an audited target-projected Camera sheet as an Agent tool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from parking_slot_agent_v2.agent import SingleSlotAgent
from parking_slot_agent_v2.calibrated_tools import CalibratedCameraToolSuite
from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_agent_v2.model import ReplayModelAdapter
from parking_slot_agent_v2.openai_adapter import OpenAIResponsesAdapter
from parking_slot_agent_v2.pipeline import run_candidate_queue
from parking_slot_agent_v2.prompts import SYSTEM_PROMPT


CALIBRATED_AGENT_POLICY = r"""You are ParkingAgent v2, an autonomous visual Agent deciding one target parking slot.

Goal
- Decide Free, Occupied, or Unknown for exactly one SlotCase.
- Use Plan -> Execute -> Observe -> Re-plan/Final.
- Each response is one structured tool action or one structured final action.
- The tool action rationale is the concise, auditable Plan. Do not reveal hidden chain-of-thought.

Evidence priority
- mandatory_fov provides structured geometry and tool availability. The large
  ego-centric location map is retained for human audit but is not an occupancy
  image. camera_context visual reasoning uses only the cyan target Camera
  contact sheet, avoiding map-to-pixel confusion.
1. available_tools is authoritative.
2. If camera_context is available, it is always the first evidence tool. Camera
   is the primary occupancy sensor; do not call lidar_detail first.
3. Execute camera_context, then observe every causal image tile.
4. If the best tile is localized but its cyan target or vehicle boundary is too
   small, blurred, or edge-compressed, autonomously call camera_crop before
   LiDAR. Select one useful tile from contact_sheet_layout and give a tighter
   bbox in the full contact-sheet coordinate space. Crop is for zoom, not a
   new occupancy label.
5. If Camera is visually decisive, return Final immediately. Do not call LiDAR
   to confirm or override a clear Camera result.
6. Re-plan to lidar_detail only when Camera is absent/failed, or the original
   and cropped pixels genuinely hide target ownership.
7. If Camera is not offered, use lidar_detail when available; otherwise return
   Unknown citing the FOV evidence.
8. Never request a tool that is absent or collect redundant evidence. One crop
   is normally enough; a retry must change bbox or enhancement.

Direct Camera task
- The cyan quadrilateral is the mapped ground footprint of the TARGET parking
  slot. It is already localized; do not reconstruct parking rows or numerical
  depth.
- Judge all causal tiles together.
- Free: visible target pavement/interior is consistently clear in at least two
  usable tiles and no parked vehicle belongs to the target. A narrow or skewed
  polygon is normal perspective and does not invalidate otherwise clear ground.
- Camera occupied candidate: the same parked vehicle visibly fills the target
  footprint consistently across at least two usable tiles.
- Important: the cyan overlay is transparent and can cross a closer foreground
  or adjacent vehicle. Therefore Camera overlap creates an occupied CANDIDATE,
  not an immediate terminal Occupied when lidar_detail is available. Call
  lidar_detail to test whether the apparent vehicle is foreground/adjacent.
- Unknown/Camera fallback: most target pixels are genuinely hidden by an object,
  clipped by the image edge, severely blurred, or contradictory across tiles.
- Do not call something occluded merely because a pillar/wall is near the cyan
  outline. It must visibly cover most target interior. Conversely, if an object
  actually covers most target pixels, do not infer occupancy from that overlap.
- A single bumper/wheel/grille intersection is insufficient. Use the whole
  vehicle footprint and temporal consistency.
- Camera-visible clear pavement may return Free directly at threshold 0.60.
- Do not calculate wheel-contact coordinates, painted-line visibility, row
  indices, or numerical 3D distance. Ordinary visual perspective and occlusion
  understanding are allowed.

Projected-target localization fields
- Before Camera is observed, localization may be not_attempted.
- After successful camera_context, the cyan polygon itself is the localization
  proof. For a Camera final use stage=supported,
  hypothesis_id="cyan_target_polygon", matched_landmarks=["cyan_target_polygon"],
  confidence_after equal to localization_confidence, and leave depth/row/bbox
  unknown or null. Do not invent other landmarks.
- For camera_crop, set localization.bbox_norm exactly equal to the requested
  arguments.bbox_norm. Use hypothesis_id="cyan_target_crop", name the chosen
  tile in target_row, and cite cyan_target_polygon as a matched landmark. The
  resulting Observation reports selected_tile_indices and selected_frame_ids.
- For Camera-only Free/Occupied, localization_confidence and
  occupancy_confidence must both be at least 0.60.
- For genuine Camera occlusion followed by LiDAR, report localization as
  ambiguous and name the actual visual occluder in ambiguity_reasons.

LiDAR fallback
- LiDAR is secondary and target-local. Use it for a Camera occupied candidate,
  genuine Camera occlusion, or failed Camera.
- Fusion is asymmetric because Part1 can confuse pillars/walls with vehicles:
  * LiDAR free_eligible with adequate clear-space coverage and no unresolved
    core hit may resolve a visual overlap as Free.
  * LiDAR occupied_eligible may support Occupied only when Camera independently
    supports a target-owned vehicle.
  * If Camera supports a target vehicle but LiDAR is weak/non-terminal and does
    not support Free, Camera may still decide Occupied; cite Camera, not the
    ineligible LiDAR card.
  * If Camera ownership is ambiguous, LiDAR obstacle points alone cannot produce
    Occupied; return Unknown.
- A LiDAR terminal result must obey its terminal_geometry_gate when cited as
  supporting evidence.

Decision rules
- The runtime may request one final Camera self-check when all tools are
  exhausted, the tentative state is Unknown, and residual Free or Occupied
  evidence lies in [0.30, 0.60). Treat this as a normal re-observation step:
  inspect every Camera tile again and decide from actual pixels; do not assume
  a requested self-check implies any particular label.
- Return Free when free_confidence >= 0.60 and occupied_confidence < 0.60.
- Return Occupied when occupied_confidence >= 0.60 and free_confidence < 0.60.
- Return Unknown only after useful evidence is exhausted and neither terminal
  state qualifies, or the modalities remain in a real conflict.
- Cite only evidence IDs actually observed and supporting the result.
"""



def calibrated_prompt() -> str:
    schema_start = SYSTEM_PROMPT.index("Return exactly one JSON object")
    return CALIBRATED_AGENT_POLICY + "\n" + SYSTEM_PROMPT[schema_start:]


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def semantic(result) -> dict:
    return {
        "run_stop_reason": result.stop_reason,
        "slots": [
            {
                "slot_id": row.case.slot_id,
                "final_state": (
                    None if row.case.final_state is None else row.case.final_state.value
                ),
                "final_scores": (
                    None if row.case.final_scores is None else row.case.final_scores.to_dict()
                ),
                "tools": [
                    e.tool_name for e in row.case.evidence if e.round_index >= 1
                ],
                "stop_reason": row.stop_reason,
            }
            for row in result.slot_results
        ],
    }


def plan_execute_observe_trace(result, calls: list[dict]) -> dict:
    """Build an explicit audit view from model plans and executed observations."""

    calls_by_case: dict[str, list[dict]] = {}
    for call in calls:
        calls_by_case.setdefault(str(call["case_id"]), []).append(call)
    slots = []
    for row in result.slot_results:
        case = row.case
        model_calls = sorted(
            calls_by_case.get(case.case_id, []), key=lambda item: item["turn"]
        )
        rounds_by_index = {item.round_index: item for item in case.rounds}
        tool_round_index = 0
        steps = []
        for call in model_calls:
            action = call["action"]
            if action["type"] == "tool":
                tool_round_index += 1
                reasoning_round = rounds_by_index.get(tool_round_index)
                observed_evidence = (
                    None
                    if reasoning_round is None
                    else next(
                        (
                            evidence
                            for evidence in case.evidence
                            if evidence.evidence_id in reasoning_round.evidence_ids
                        ),
                        None,
                    )
                )
                steps.append(
                    {
                        "turn": call["turn"],
                        "phase": "plan_execute_observe",
                        "plan": action["rationale"],
                        "execute": {
                            "tool": action["tool"],
                            "arguments": action["arguments"],
                        },
                        "observe": None
                        if reasoning_round is None
                        else {
                            "status": (
                                "unknown"
                                if observed_evidence is None
                                else observed_evidence.status
                            ),
                            "summary": reasoning_round.observation_summary,
                            "evidence_ids": list(reasoning_round.evidence_ids),
                            "selected_tile_indices": (
                                []
                                if observed_evidence is None
                                else observed_evidence.metadata.get(
                                    "selected_tile_indices", []
                                )
                            ),
                            "selected_frame_ids": (
                                []
                                if observed_evidence is None
                                else observed_evidence.metadata.get(
                                    "selected_frame_ids", []
                                )
                            ),
                            "model_image_paths": (
                                []
                                if observed_evidence is None
                                else observed_evidence.metadata.get(
                                    "model_image_paths", []
                                )
                            ),
                        },
                    }
                )
            else:
                accepted = call is model_calls[-1]
                steps.append(
                    {
                        "turn": call["turn"],
                        "phase": "final" if accepted else "preliminary_final_recheck",
                        "accepted": accepted,
                        "plan": (
                            "Stop because observed evidence is sufficient for a structured decision."
                            if accepted
                            else "Proposed final was not accepted; runtime requested one bounded correction or Camera self-check."
                        ),
                        "execute": {
                            "action": "agent_final"
                            if accepted
                            else "proposed_final_not_accepted"
                        },
                        "observe": {
                            "state": action["state"],
                            "free_confidence": action["free_confidence"],
                            "occupied_confidence": action["occupied_confidence"],
                            "reason": action["reason"],
                            "reason_codes": action["reason_codes"],
                            "evidence_ids": action["evidence_ids"],
                        },
                    }
                )
        slots.append(
            {
                "slot_id": case.slot_id,
                "case_id": case.case_id,
                "initial_camera_gate": case.fov.to_dict(),
                "steps": steps,
                "final_state": None
                if case.final_state is None
                else case.final_state.value,
                "stop_reason": row.stop_reason,
            }
        )
    return {
        "schema_version": "parking-slot-agent-plan-execute-observe/1.0",
        "policy": "camera_first_then_lidar_only_for_absent_failed_or_occluded_camera",
        "slots": slots,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--camera-sheets", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.6-terra")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--api-key-file", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)

    part1 = Part1Output.from_dict(json.loads(a.input.read_text(encoding="utf-8")))
    attached = []
    camera_manifest = (a.camera_sheets / "manifest.json").resolve(strict=False)
    for case in part1.slot_cases:
        sheet = (a.camera_sheets / f"{case.slot_id}.jpg").resolve(strict=False)
        if sheet.is_file():
            case.resources["calibrated_camera_sheet_path"] = str(sheet)
            if camera_manifest.is_file():
                case.resources["calibrated_camera_manifest_path"] = str(camera_manifest)
            depth = (a.camera_sheets / f"{case.slot_id}.depth.json").resolve(
                strict=False
            )
            if depth.is_file():
                case.resources["calibrated_camera_depth_metadata_path"] = str(depth)
            attached.append(case.slot_id)
    write(a.output_dir / "part1_input_with_calibrated_sheets.json", part1.to_dict())
    prompt = calibrated_prompt()
    (a.output_dir / "system_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    import parking_slot_agent_v2.openai_adapter as adapter_module

    adapter_module.SYSTEM_PROMPT = prompt
    live_adapter = OpenAIResponsesAdapter(
        model=a.model,
        reasoning_effort=a.reasoning_effort,
        image_detail="original",
        audit_dir=a.output_dir / "openai_audit",
        api_key_file=a.api_key_file,
    )
    live = run_candidate_queue(
        part1,
        SingleSlotAgent(live_adapter, CalibratedCameraToolSuite(map_radius_m=part1.scene.radius_m)),
        a.output_dir / "live",
        stop_at_first_free=False,
        resume=False,
    )
    write(
        a.output_dir / "agent_plan_execute_observe.json",
        plan_execute_observe_trace(live, live_adapter.calls),
    )
    replay_payload = live_adapter.replay_payload()
    for result in live.slot_results:
        case_id = result.case.case_id
        sequence = replay_payload["actions"].setdefault(case_id, [])
        bindings = replay_payload["evidence_bindings"].setdefault(case_id, [])
        missing = max(0, result.model_turns - len(sequence))
        for _ in range(missing):
            sequence.append({"type": "replay_error", "error_type": "ValueError", "message": "recorded invalid structured action"})
            bindings.append([])
    write(a.output_dir / "replay_actions.json", replay_payload)

    replay_input = Part1Output.from_dict(
        json.loads(
            (a.output_dir / "part1_input_with_calibrated_sheets.json").read_text(
                encoding="utf-8"
            )
        )
    )
    replay = run_candidate_queue(
        replay_input,
        SingleSlotAgent(
            ReplayModelAdapter.from_path(a.output_dir / "replay_actions.json"),
            CalibratedCameraToolSuite(map_radius_m=replay_input.scene.radius_m),
        ),
        a.output_dir / "replay",
        stop_at_first_free=False,
        resume=False,
    )
    live_semantic, replay_semantic = semantic(live), semantic(replay)
    verified = live_semantic == replay_semantic
    write(
        a.output_dir / "replay_verification.json",
        {"verified": verified, "live": live_semantic, "replay": replay_semantic},
    )
    if not verified:
        raise RuntimeError("Replay mismatch")
    counts = Counter(row["final_state"] for row in live_semantic["slots"])
    resolved = counts["free"] + counts["occupied"]
    summary = {
        "architecture": "bounded Plan-Execute-Observe loop; model autonomously chooses tools",
        "evidence_priority": "camera_first; lidar_only_for_absent_failed_or_occluded_camera",
        "prediction_blind": True,
        "gt_used": False,
        "camera_sheet_slots": attached,
        "processed": len(live_semantic["slots"]),
        "state_counts": dict(counts),
        "resolved": resolved,
        "resolution_rate": resolved / max(1, len(live_semantic["slots"])),
        "replay_verified": True,
        "model": a.model,
    }
    write(a.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
