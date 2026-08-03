"""Bounded ReAct execution for one mutable :class:`SlotCase`."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    CaseStatus,
    ConfidenceScores,
    EvidenceRecord,
    FovResult,
    FovVisibility,
    ReasoningRound,
    SceneSnapshot,
    SlotCase,
    SlotState,
)
from .model import (
    ActionError,
    BeliefEstimate,
    FinalAction,
    LocalizationEstimate,
    ModelAdapter,
    ModelProviderError,
    ToolAction,
    parse_action,
)
from .tools import V2ToolSuite, describe_lidar_capability
from .lidar_geometry import assess_terminal_geometry


TERMINAL_CONFIDENCE = 0.60
CAMERA_OCCUPIED_CONFIDENCE = 0.60
MAX_MODEL_TURNS = 6
CAMERA_USABLE_FOV = frozenset({
    FovVisibility.VISIBLE,
    FovVisibility.PARTIALLY_VISIBLE,
})


def _camera_usable(case: SlotCase) -> bool:
    return case.fov.visibility in CAMERA_USABLE_FOV


@dataclass(frozen=True, slots=True)
class SlotRunResult:
    case: SlotCase
    stop_reason: str
    model_turns: int
    tool_rounds: int
    validation_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "stop_reason": self.stop_reason,
            "model_turns": self.model_turns,
            "tool_rounds": self.tool_rounds,
            "validation_errors": list(self.validation_errors),
        }


def _status(record: EvidenceRecord) -> str:
    return record.status


def _successful_detail_evidence(case: SlotCase) -> tuple[EvidenceRecord, ...]:
    return tuple(
        record
        for record in case.evidence
        if record.round_index >= 1 and _status(record) == "ok"
        and record.metadata.get("is_incremental_over_part1") is not False
    )


def _image_paths(case: SlotCase) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    allowed_tools = {
        "camera_context",
        "camera_sequence",
        "camera_crop",
        "lidar_detail",
    }
    if not _camera_usable(case):
        allowed_tools = {"lidar_detail"}
    for evidence in case.evidence:
        if _status(evidence) != "ok" or evidence.tool_name not in allowed_tools:
            continue
        model_paths = evidence.metadata.get("model_image_paths")
        has_explicit_model_paths = (
            isinstance(model_paths, list)
            and bool(model_paths)
            and all(isinstance(item, str) for item in model_paths)
        )
        raw_paths = model_paths if has_explicit_model_paths else evidence.artifact_paths
        for raw in raw_paths:
            path = Path(raw)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if path not in seen:
                seen.add(path)
                result.append(path)
        # A tool may retain raw/card/composite artifacts for audit while
        # explicitly selecting only the composite for the model. Do not append
        # every audit resource again; duplicate images waste tokens and can
        # overweight the same observation.
        if has_explicit_model_paths:
            continue
        for key in evidence.resource_keys:
            raw = case.resources.get(key)
            if not raw:
                continue
            path = Path(raw)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if path not in seen:
                seen.add(path)
                result.append(path)
    return tuple(result)


def _small_part1_details(case: SlotCase) -> dict[str, Any]:
    for evidence in case.evidence:
        if evidence.tool_name != "part1_15frame":
            continue
        details = evidence.metadata
        occupied = details.get("occupied_evidence", {})
        free = details.get("free_evidence", {})
        stability = details.get("stability", {})
        occupied_features = (
            occupied.get("features", {}) if isinstance(occupied, Mapping) else {}
        )
        if not isinstance(occupied_features, Mapping):
            occupied_features = {}
        return {
            "occupied": {
                "strength_uncalibrated": occupied.get("strength"),
                "strong_gate": bool(occupied.get("strong", False)),
                "weak_gate": bool(occupied.get("weak", False)),
                "failures": list(occupied.get("failures", [])),
                "vehicle_point_count": occupied_features.get("point_count"),
                "core_point_count": occupied_features.get("core_point_count"),
                "support_frame_count": occupied_features.get("supported_frame_count"),
                "temporal_consistency": occupied_features.get("temporal_consistency"),
                "supported_height_layers": occupied_features.get("supported_layer_count"),
                "z95_m": occupied_features.get("z95_m"),
                "height_span_m": occupied_features.get("height_span_m"),
                "extent_long_m": occupied_features.get("extent_x_m"),
                "extent_short_m": occupied_features.get("extent_y_m"),
                "core_overlap": occupied_features.get("core_overlap"),
                "adjacent_overlap": occupied_features.get("adjacent_overlap"),
                "boundary_ratio": occupied_features.get("boundary_ratio"),
                "linearity_risk": occupied_features.get("linearity"),
            }
            if isinstance(occupied, Mapping)
            else {},
            "free": {
                "strength_uncalibrated": free.get("strength"),
                "strong_gate": bool(free.get("strong", False)),
                "failures": list(free.get("failures", [])),
                "core_ray_coverage": free.get("core_ray_coverage"),
                "observed_volume_ratio": free.get("observed_volume_ratio"),
                "near_ground_bev_coverage": free.get("near_ground_bev_coverage"),
                "unobserved_component_ratio": free.get("unobserved_component_ratio"),
                "occlusion_ratio": free.get("occlusion_ratio"),
                "unresolved_core_hit": bool(free.get("unresolved_core_hit", False)),
            }
            if isinstance(free, Mapping)
            else {},
            "stability": {
                key: stability.get(key)
                for key in ("stable", "pass_ratio", "failures")
                if isinstance(stability, Mapping) and key in stability
            },
        }
    return {}


def _evidence_rows(case: SlotCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence in case.evidence:
        if evidence.tool_name == "part1_15frame":
            continue
        rows.append(
            {
                "evidence_id": evidence.evidence_id,
                "modality": evidence.modality,
                "tool_name": evidence.tool_name,
                "round_index": evidence.round_index,
                "status": _status(evidence),
                "summary": evidence.summary,
                "reason_codes": list(evidence.reason_codes),
                "metadata": evidence.metadata,
                "has_visual_artifact": bool(evidence.artifact_paths)
                or any(key in case.resources for key in evidence.resource_keys),
            }
        )
    return rows


def _available_tools(case: SlotCase) -> tuple[str, ...]:
    """Return legal tools that can still add evidence to this SlotCase."""
    if case.rounds_remaining <= 0:
        return ()
    used = {evidence.tool_name for evidence in case.evidence}
    lidar_available = (
        describe_lidar_capability(case)["available"]
        and "lidar_detail" not in used
    )
    if not _camera_usable(case):
        return ("lidar_detail",) if lidar_available else ()
    result: list[str] = []
    if "camera_context" not in used:
        result.append("camera_context")
    context_ok = any(
        evidence.tool_name == "camera_context" and _status(evidence) == "ok"
        for evidence in case.evidence
    )
    calibrated_context = bool(case.resources.get("calibrated_camera_sheet_path"))
    # The calibrated context is already a target-projected causal sequence. Raw
    # t0 crop/sequence coordinates are a different image space, so advertising
    # them after that sheet creates invalid ownership follow-ups.
    if context_ok and not calibrated_context and "camera_sequence" not in used:
        result.append("camera_sequence")
    # Crop is available after either raw context or an audited calibrated
    # contact sheet. The concrete tool publishes the coordinate space.
    if context_ok:
        result.append("camera_crop")
    # Keep LiDAR available as a genuine fallback after the projected Camera
    # sheet. The Camera-first prompt requires a direct final for a clear target,
    # while allowing the Agent to re-plan when the sheet failed or the target is
    # truly occluded. This preserves autonomy without letting LiDAR override a
    # clear Camera observation.
    if lidar_available:
        result.append("lidar_detail")
    return tuple(result)


def _tool_capabilities(case: SlotCase) -> dict[str, Any]:
    available = set(_available_tools(case))
    camera_legal = _camera_usable(case)
    context_ok = any(
        evidence.tool_name == "camera_context" and _status(evidence) == "ok"
        for evidence in case.evidence
    )
    return {
        "camera_context": {
            "available": "camera_context" in available,
            "requires_horizontal_projection_gate": True,
            "camera_legal": camera_legal,
        },
        "camera_sequence": {
            "available": "camera_sequence" in available,
            "requires_successful_camera_context": True,
            "camera_context_succeeded": context_ok,
        },
        "camera_crop": {
            "available": "camera_crop" in available,
            "requires_successful_camera_context": True,
            "camera_context_succeeded": context_ok,
            "repeatable_with_changed_bbox_or_enhancement": True,
            "bbox_norm_coordinate_space": (
                "calibrated_contact_sheet"
                if case.resources.get("calibrated_camera_sheet_path")
                else "raw_full_camera"
            ),
            "attempt_count": sum(
                item.tool_name == "camera_crop" for item in case.rounds
            ),
        },
        "lidar_detail": describe_lidar_capability(case),
    }


def _crop_signature(arguments: Mapping[str, Any]) -> tuple[tuple[float, ...], str] | None:
    raw_bbox = arguments.get("bbox_norm")
    if (
        isinstance(raw_bbox, (str, bytes))
        or not isinstance(raw_bbox, Sequence)
        or len(raw_bbox) != 4
    ):
        return None
    try:
        bbox = tuple(round(float(value), 8) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    enhancement = str(arguments.get("enhancement", "none")).strip().lower()
    return bbox, enhancement


def _previous_crop_signatures(
    case: SlotCase,
) -> set[tuple[tuple[float, ...], str]]:
    """Return every audited Crop parameter pair already attempted."""

    result: set[tuple[tuple[float, ...], str]] = set()
    for item in case.rounds:
        if item.tool_name != "camera_crop":
            continue
        signature = _crop_signature(item.tool_arguments)
        if signature is not None:
            result.add(signature)
    # Evidence-only cases occur in resumed/audit fixtures. Keep retry
    # validation correct even when a historical ReasoningRound is unavailable.
    for item in case.evidence:
        if item.tool_name != "camera_crop":
            continue
        signature = _crop_signature(item.metadata)
        if signature is not None:
            result.add(signature)
    return result


def _ensure_tool_available(tool: str, case: SlotCase) -> None:
    """Enforce the same legal-tool set that is sent to the model."""

    available = _available_tools(case)
    if tool in available:
        return
    capability = describe_lidar_capability(case)
    if tool == "lidar_detail" and not capability["available"]:
        raise ActionError(
            "lidar_detail is unavailable as incremental Part2 evidence: "
            f"{capability['blocking_reason']}"
        )
    raise ActionError(
        f"{tool} is not currently available; legal tools are {list(available)}"
    )


def _request(
    scene: SceneSnapshot,
    case: SlotCase,
    *,
    turn: int,
    feedback: Sequence[str],
) -> dict[str, Any]:
    available_tools = _available_tools(case)
    return {
        "schema_version": "parking-slot-agent-v2-turn/1.0",
        "case_id": case.case_id,
        "turn": turn,
        "target": {
            "slot_id": case.slot_id,
            "distance_to_t0_anchor_m": case.slot.distance_to_anchor_m,
            "part1_state": case.part1_state.value,
            "part1_scores": case.part1_scores.to_dict(),
            "current_state": case.current_state.value,
            "current_scores": case.current_scores.to_dict(),
            "decision_reason": case.decision_reason,
            "unknown_reasons": list(case.unknown_reasons),
            "part1_evidence_summary": _small_part1_details(case),
            "bearing_sign_convention": "positive_is_camera_left",
            "semantic_map_convention": {
                "forward": "up",
                "map_left": "camera_left",
                "map_right": "camera_right",
            },
        },
        "scene": {
            "snapshot_id": scene.snapshot_id,
            "t0_lidar_frame": scene.anchor_frame_id,
            "t0_timestamp": scene.anchor_timestamp,
            "radius_m": scene.radius_m,
            "nearby_slot_count": len(scene.slots),
            "evaluated_slot_count": sum(slot.observed for slot in scene.slots),
            "camera_frame_id": scene.t0_frame.camera_frame_id,
            "camera_match_valid": scene.t0_frame.camera_match_valid,
        },
        "mandatory_fov": case.fov.to_dict(),
        "observations": _evidence_rows(case),
        "localization_history": [
            dict(round_item.localization)
            for round_item in case.rounds
            if round_item.localization
        ],
        "feedback": list(feedback),
        "budget": {
            "max_tool_rounds": case.max_rounds,
            "tool_rounds_used": len(case.rounds),
            "tool_rounds_remaining": case.rounds_remaining,
            "terminal_confidence": TERMINAL_CONFIDENCE,
            "part2_detail_attempt_required": bool(available_tools),
            "successful_detail_required_for_free_or_occupied": True,
            "no_useful_tools_remaining": not available_tools,
        },
        "available_tools": list(available_tools),
        "tool_capabilities": _tool_capabilities(case),
    }


def _localization_payload(value: LocalizationEstimate) -> dict[str, Any]:
    return {
        "stage": value.stage,
        "hypothesis_id": value.hypothesis_id,
        "target_side": value.target_side,
        "depth_band": value.depth_band,
        "target_row": value.target_row,
        "target_order_in_row": value.target_order_in_row,
        "bbox_norm": None if value.bbox_norm is None else list(value.bbox_norm),
        "matched_landmarks": list(value.matched_landmarks),
        "missing_landmarks": list(value.missing_landmarks),
        "confidence_before": value.confidence_before,
        "confidence_after": value.confidence_after,
        "ambiguity_reasons": list(value.ambiguity_reasons),
    }


def _validate_localization_action(action: ToolAction, case: SlotCase) -> None:
    """Require a falsifiable map-to-Camera hypothesis for follow-up tools."""

    _ensure_tool_available(action.tool, case)
    # LiDAR is target-local metric geometry and must never depend on a Camera
    # row/bbox hypothesis, even when Camera context was inspected first.
    if action.tool == "lidar_detail":
        return
    localization = action.localization
    context_ok = any(
        item.tool_name == "camera_context" and _status(item) == "ok"
        for item in case.evidence
    )
    if action.tool == "camera_context":
        return
    if action.tool not in {"camera_crop", "camera_sequence", "lidar_detail"}:
        return
    if not context_ok:
        if action.tool == "lidar_detail":
            return
        raise ActionError(f"{action.tool} requires successful camera_context evidence first")
    if localization.stage in {"not_attempted", "not_visible"}:
        raise ActionError(f"{action.tool} requires an explicit Camera localization hypothesis")
    if not localization.hypothesis_id:
        raise ActionError("Camera localization hypothesis requires hypothesis_id")
    calibrated_crop = (
        action.tool == "camera_crop"
        and bool(case.resources.get("calibrated_camera_sheet_path"))
    )
    if (
        not calibrated_crop
        and (
            localization.target_side == "unknown"
            or localization.depth_band == "unknown"
        )
    ):
        raise ActionError("Camera localization hypothesis requires side and depth")
    if not localization.target_row:
        raise ActionError("Camera localization hypothesis requires target_row")
    if not localization.matched_landmarks and not localization.ambiguity_reasons:
        raise ActionError("Camera localization requires matched landmarks or concrete ambiguity")
    if calibrated_crop and "cyan_target_polygon" not in localization.matched_landmarks:
        raise ActionError("calibrated camera_crop must be anchored to cyan_target_polygon")
    if action.tool == "lidar_detail":
        return
    if action.tool == "camera_crop":
        if localization.bbox_norm is None:
            raise ActionError("camera_crop requires localization.bbox_norm")
        action_bbox = tuple(float(value) for value in action.arguments["bbox_norm"])
        if any(abs(left - right) > 1e-6 for left, right in zip(action_bbox, localization.bbox_norm)):
            raise ActionError("camera_crop bbox must match the Agent localization hypothesis")
        signature = _crop_signature(action.arguments)
        if signature in _previous_crop_signatures(case):
            raise ActionError(
                "camera_crop retry must change bbox_norm or enhancement"
            )


def _unknown_scores(case: SlotCase) -> ConfidenceScores:
    return ConfidenceScores(
        free_confidence=case.current_scores.free_confidence,
        occupied_confidence=case.current_scores.occupied_confidence,
        unknown_confidence=max(
            0.0,
            1.0
            - max(
                case.current_scores.free_confidence,
                case.current_scores.occupied_confidence,
            ),
        ),
        kind="agent_fused_confidence",
        calibrated=False,
    )


def _final_scores(action: FinalAction) -> ConfidenceScores:
    unknown = (
        max(0.0, 1.0 - max(action.free_confidence, action.occupied_confidence))
        if action.state == "unknown"
        else 0.0
    )
    return ConfidenceScores(
        free_confidence=action.free_confidence,
        occupied_confidence=action.occupied_confidence,
        unknown_confidence=unknown,
        kind="agent_fused_confidence",
        calibrated=False,
    )


def _belief_scores(belief: BeliefEstimate) -> ConfidenceScores:
    return ConfidenceScores(
        free_confidence=belief.free_confidence,
        occupied_confidence=belief.occupied_confidence,
        unknown_confidence=belief.unknown_confidence,
        kind="agent_fused_confidence",
        calibrated=False,
    )


def _apply_belief_to_previous_round(
    belief: BeliefEstimate,
    case: SlotCase,
) -> None:
    """Persist the model's fusion of evidence available before its next tool."""

    scores = _belief_scores(belief)
    case.update_state(
        belief.state,
        scores,
        unresolved_reasons=belief.remaining_unknown_reasons,
    )
    if not case.rounds:
        return
    previous = case.rounds[-1]
    previous.state_after = SlotState(belief.state)
    previous.scores_after = scores
    previous.resolved_unknown_reasons = list(belief.resolved_unknown_reasons)
    previous.remaining_unknown_reasons = list(belief.remaining_unknown_reasons)


def _apply_final_to_previous_round(action: FinalAction, case: SlotCase) -> None:
    if not case.rounds:
        return
    scores = _final_scores(action)
    previous = case.rounds[-1]
    unresolved_before = list(case.unresolved_reasons or case.unknown_reasons)
    if action.state == "unknown":
        remaining = list(action.reason_codes or unresolved_before or ("part2_unresolved",))
        resolved: list[str] = []
    else:
        remaining = []
        resolved = unresolved_before
    previous.state_after = SlotState(action.state)
    previous.scores_after = scores
    previous.resolved_unknown_reasons = resolved
    previous.remaining_unknown_reasons = remaining
    previous.localization = _localization_payload(action.localization)
    case.update_state(action.state, scores, unresolved_reasons=remaining)


def _validate_final(action: FinalAction, case: SlotCase) -> None:
    evidence_by_id = {item.evidence_id: item for item in case.evidence}
    if any(evidence_id not in evidence_by_id for evidence_id in action.evidence_ids):
        raise ActionError("final action references unknown evidence_ids")
    cited = tuple(evidence_by_id[value] for value in action.evidence_ids)
    detail_attempts = tuple(
        record for record in case.evidence if record.round_index >= 1
    )
    successful = _successful_detail_evidence(case)

    if action.state == "free":
        if not successful:
            raise ActionError("free requires at least one successful Part2 detail tool")
        if not any(item in successful for item in cited):
            raise ActionError("free must cite successful Part2 detail evidence")
        if action.free_confidence < TERMINAL_CONFIDENCE:
            raise ActionError("free requires free_confidence >= 0.60")
        if action.occupied_confidence >= TERMINAL_CONFIDENCE:
            raise ActionError("free is contradictory when occupied_confidence >= 0.50")
    elif action.state == "occupied":
        if not successful:
            raise ActionError("occupied requires at least one successful Part2 detail tool")
        if not any(item in successful for item in cited):
            raise ActionError("occupied must cite successful Part2 detail evidence")
        if action.occupied_confidence < TERMINAL_CONFIDENCE:
            raise ActionError("occupied requires occupied_confidence >= 0.60")
        if action.free_confidence >= TERMINAL_CONFIDENCE:
            raise ActionError("occupied is contradictory when free_confidence >= 0.50")
    else:
        no_useful_tools = not _available_tools(case)
        if not detail_attempts:
            if not no_useful_tools:
                raise ActionError(
                    "unknown requires a Part2 detail attempt while useful tools remain"
                )
            if not cited:
                raise ActionError(
                    "unknown with no useful tools must cite the mandatory FOV evidence"
                )
            if not any(item.tool_name == "check_fov" for item in cited):
                raise ActionError(
                    "unknown with no useful tools must cite the mandatory FOV evidence"
                )
        elif not any(item in detail_attempts for item in cited):
            raise ActionError("unknown must cite attempted Part2 detail evidence")
        if len(case.rounds) < case.max_rounds and _available_tools(case):
            raise ActionError(
                "unknown requires an exhausted tool budget or no useful tools remaining"
            )
        exactly_one_terminal = (
            action.free_confidence >= TERMINAL_CONFIDENCE
        ) != (action.occupied_confidence >= TERMINAL_CONFIDENCE)
        if exactly_one_terminal:
            raise ActionError(
                "unknown is invalid when exactly one terminal state reaches 0.60"
            )

    cited_camera = tuple(
        item
        for item in cited
        if item.tool_name in {"camera_context", "camera_sequence", "camera_crop"}
    )
    cited_lidar = tuple(item for item in cited if item.tool_name == "lidar_detail")
    cited_camera_verification = tuple(
        item
        for item in cited_camera
        if item.tool_name in {"camera_crop", "camera_sequence"}
    )
    if cited_camera_verification:
        if action.localization.stage not in {"supported", "refuted", "ambiguous"}:
            raise ActionError(
                "final action after Camera Crop/Sequence must report the hypothesis "
                "as supported, refuted, or ambiguous"
            )
        if not action.localization.hypothesis_id:
            raise ActionError(
                "final action after Camera Crop/Sequence requires hypothesis_id"
            )
        if action.localization.bbox_norm is None:
            raise ActionError(
                "final action after Camera Crop/Sequence requires the verified bbox"
            )
    if action.state in {"free", "occupied"} and cited_lidar:
        _validate_lidar_terminal_geometry(action, cited_lidar)
    camera_was_observed = any(
        item.tool_name == "camera_context" and item.status == "ok"
        for item in case.evidence
    )
    if (
        action.state == "occupied"
        and cited_camera
        and not cited_lidar
        and "lidar_detail" in _available_tools(case)
    ):
        raise ActionError(
            "Camera occupied candidate requires lidar_detail verification when "
            "that incremental tool is available"
        )
    if (
        action.state == "occupied"
        and cited_lidar
        and camera_was_observed
        and action.localization.stage != "supported"
    ):
        raise ActionError(
            "LiDAR Occupied cannot override ambiguous Camera target ownership; "
            "return Unknown unless Camera independently supports the target vehicle"
        )
    if action.state in {"free", "occupied"} and cited_camera and not cited_lidar:
        camera_threshold = (
            CAMERA_OCCUPIED_CONFIDENCE
            if action.state == "occupied"
            else TERMINAL_CONFIDENCE
        )
        if (
            action.localization_confidence is None
            or action.occupancy_confidence is None
            or min(action.localization_confidence, action.occupancy_confidence)
            < camera_threshold
        ):
            raise ActionError(
                "camera-only terminal decision requires localization and occupancy confidence "
                f">= {camera_threshold:.2f}"
            )
        if (
            action.localization.stage != "supported"
            or not action.localization.hypothesis_id
            or action.localization.confidence_after < TERMINAL_CONFIDENCE
            or not action.localization.matched_landmarks
        ):
            raise ActionError(
                "camera-only terminal decision requires a supported, landmark-grounded localization hypothesis"
            )


def _validate_lidar_terminal_geometry(
    action: FinalAction,
    cited_lidar: Sequence[EvidenceRecord],
) -> None:
    """Enforce geometry vetoes in code instead of trusting prompt compliance."""

    cards = [
        item.metadata.get("geometry_card")
        for item in cited_lidar
        if item.status == "ok" and isinstance(item.metadata.get("geometry_card"), Mapping)
    ]
    if not cards:
        raise ActionError("LiDAR terminal decision requires an auditable geometry_card")

    for raw_card in cards:
        card = raw_card if isinstance(raw_card, Mapping) else {}
        assessment = assess_terminal_geometry(card)
        if assessment[f"{action.state}_eligible"]:
            return
    raise ActionError(
        f"{action.state} is vetoed by LiDAR geometry/stability hard gates"
    )


def _fusion_record(action: FinalAction, case: SlotCase) -> EvidenceRecord:
    payload = {
        "case_id": case.case_id,
        "state": action.state,
        "free_confidence": action.free_confidence,
        "occupied_confidence": action.occupied_confidence,
        "localization_confidence": action.localization_confidence,
        "occupancy_confidence": action.occupancy_confidence,
        "evidence_ids": list(action.evidence_ids),
        "reason_codes": list(action.reason_codes),
        "localization": _localization_payload(action.localization),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvidenceRecord(
        evidence_id="ev_" + digest,
        tool_name="agent_final",
        round_index=len(case.rounds),
        status="ok",
        artifact_paths=[],
        summary=action.reason,
        metadata=payload,
        modality="fusion",
        supports_state=action.state,
        scores=_final_scores(action),
        reason_codes=list(action.reason_codes),
    )


class SingleSlotAgent:
    """Execute mandatory FOV then at most three evidence-tool rounds."""

    def __init__(
        self,
        model: ModelAdapter,
        tools: V2ToolSuite,
        *,
        max_model_turns: int = MAX_MODEL_TURNS,
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_model_turns = int(max_model_turns)
        if self.max_model_turns < 1:
            raise ValueError("max_model_turns must be positive")

    def _invoke_tool(
        self,
        action: ToolAction,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: Path,
        round_index: int,
    ) -> EvidenceRecord:
        _ensure_tool_available(action.tool, case)
        if action.tool in {"camera_context", "camera_sequence", "lidar_detail"} and any(
            item.tool_name == action.tool for item in case.evidence
        ):
            raise ActionError(f"{action.tool} cannot be repeated for the same SlotCase")
        if action.tool in {"camera_context", "camera_sequence", "camera_crop"} and (
            not _camera_usable(case)
        ):
            raise ActionError(
                "Camera tools are forbidden unless the horizontal projection gate "
                "is visible or partially_visible"
            )
        if action.tool == "camera_crop" and not any(
            item.tool_name == "camera_context" and _status(item) == "ok"
            for item in case.evidence
        ):
            raise ActionError("camera_crop requires successful camera_context evidence first")
        if action.tool == "camera_context":
            return self.tools.inspect_camera_context(scene, case, output_dir, round_index)
        if action.tool == "camera_sequence":
            return self.tools.inspect_camera_sequence(
                scene, case, output_dir, round_index
            )
        if action.tool == "camera_crop":
            return self.tools.crop_camera(
                scene,
                case,
                output_dir,
                round_index,
                action.arguments["bbox_norm"],
                action.arguments["enhancement"],
            )
        return self.tools.inspect_lidar_detail(scene, case, output_dir, round_index)

    def run(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
    ) -> SlotRunResult:
        if case.terminal:
            raise ValueError("SingleSlotAgent requires a non-terminal SlotCase")
        if case.snapshot_id != scene.snapshot_id:
            raise ValueError("SlotCase and SceneSnapshot identities do not match")
        media_dir = Path(output_dir).resolve(strict=False)
        validation_errors: list[str] = []

        # The concrete v2 suite also writes an audit-only FOV overlay on the
        # original Part1 occupancy map.  Keep duck-typed test/tool substitutes
        # compatible with the original two-argument contract.
        if isinstance(self.tools, V2ToolSuite):
            fov_evidence = self.tools.check_fov(scene, case, media_dir)
        else:
            fov_evidence = self.tools.check_fov(scene, case)
        # A tool implementation should update the mutable SlotCase. Fail closed
        # if it could not, while retaining the evidence record.
        if not case.fov.checked:
            case.update_fov(
                FovResult(
                    visibility=FovVisibility.UNCERTAIN,
                    confidence=0.0,
                    reason="fov_tool_did_not_publish_checked_result",
                    camera_frame_id=scene.t0_frame.camera_frame_id,
                    details={"method": "map_bearing_only"},
                )
            )
        case.record_evidence(fov_evidence)

        # The runtime only performs the mandatory FOV check.  Precomputed LiDAR
        # remains an available resource, but the Agent must actively choose
        # lidar_detail (or Camera context when legal).  Consequently all three
        # evidence rounds are genuine Agent actions and remain visible in the
        # ReAct trace.

        model_turns = 0
        stop_reason = "model_turn_limit"
        feedback: list[str] = []
        while model_turns < self.max_model_turns:
            model_turns += 1
            request = _request(
                scene,
                case,
                turn=model_turns,
                feedback=feedback,
            )
            feedback = []
            try:
                raw_action = self.model.next_action(
                    request,
                    image_paths=_image_paths(case),
                )
                action = parse_action(raw_action)
                if isinstance(action, ToolAction):
                    if case.rounds_remaining <= 0:
                        raise ActionError("three-round evidence-tool budget is exhausted")
                    _validate_localization_action(action, case)
                    round_index = len(case.rounds) + 1
                    evidence = self._invoke_tool(
                        action,
                        scene,
                        case,
                        media_dir,
                        round_index,
                    )
                    evidence.metadata["agent_localization"] = _localization_payload(
                        action.localization
                    )
                    _apply_belief_to_previous_round(action.belief, case)
                    case.record_evidence(evidence)
                    case.record_round(
                        ReasoningRound(
                            round_index=round_index,
                            reasoning_summary=action.rationale,
                            tool_name=action.tool,
                            tool_arguments=dict(action.arguments),
                            observation_summary=evidence.summary,
                            evidence_ids=[evidence.evidence_id],
                            state_after=case.current_state,
                            scores_after=case.current_scores,
                            resolved_unknown_reasons=list(
                                action.belief.resolved_unknown_reasons
                            ),
                            remaining_unknown_reasons=list(
                                action.belief.remaining_unknown_reasons
                                or case.unresolved_reasons
                                or case.unknown_reasons
                            ),
                            localization=_localization_payload(action.localization),
                        )
                    )
                    continue

                needs_camera_final_recheck = (
                    action.state == "unknown"
                    and any(
                        item.tool_name == "camera_context" and item.status == "ok"
                        for item in case.evidence
                    )
                    and not _available_tools(case)
                    and 0.30
                    <= max(action.free_confidence, action.occupied_confidence)
                    < TERMINAL_CONFIDENCE
                    and case.resources.get("camera_final_recheck_done") != "true"
                )
                if needs_camera_final_recheck:
                    case.resources["camera_final_recheck_done"] = "true"
                    feedback = [
                        "Final Camera self-check required before abstaining. "
                        "Re-observe every cyan-target tile directly. If clear "
                        "pavement is consistent, return Free; if the same parked "
                        "vehicle clearly belongs to the target across tiles, "
                        "return Occupied citing Camera; if ownership remains "
                        "genuinely hidden or ambiguous, return Unknown. Do not "
                        "let weak/non-terminal LiDAR erase clear Camera pixels, "
                        "and do not turn mere 2D overlap into occupancy."
                    ]
                    continue
                _validate_final(action, case)
                scores = _final_scores(action)
                _apply_final_to_previous_round(action, case)
                fusion = _fusion_record(action, case)
                case.record_evidence(fusion)
                if action.state == "unknown":
                    unresolved = list(case.unresolved_reasons or case.unknown_reasons)
                    if not unresolved:
                        unresolved = list(action.reason_codes or ("part2_unresolved",))
                    case.finalize(
                        SlotState.UNKNOWN,
                        scores,
                        action.reason,
                        unresolved_reasons=unresolved,
                        exhausted=True,
                    )
                    stop_reason = (
                        "three_rounds_unresolved"
                        if len(case.rounds) >= case.max_rounds
                        else "no_useful_tools_remaining"
                    )
                else:
                    case.finalize(
                        action.state,
                        scores,
                        action.reason,
                        unresolved_reasons=(),
                        exhausted=False,
                    )
                    stop_reason = "terminal_confidence"
                break
            except ActionError as exc:
                message = str(exc)
                validation_errors.append(message)
                feedback = [message]
                continue
            except ValueError as exc:
                # Invalid structured-action semantics are not parking-scene
                # evidence. Let the central Agent self-correct on its next turn
                # instead of silently converting the case to Unknown.
                message = f"invalid_model_or_tool_action:{type(exc).__name__}:{exc}"
                validation_errors.append(message)
                feedback = [
                    "Your previous structured action was invalid. Correct its "
                    "semantics using the same evidence and choose a valid tool "
                    "or final action. " + str(exc)
                ]
                continue
            except ModelProviderError:
                # Provider availability is a run-level failure, never semantic
                # evidence that the parking slot is Unknown.
                raise
            except Exception as exc:
                validation_errors.append(f"model_or_tool_error:{type(exc).__name__}")
                stop_reason = "model_or_tool_error"
                break

        if not case.terminal:
            unresolved = list(case.unresolved_reasons or case.unknown_reasons)
            if not unresolved:
                unresolved = [stop_reason]
            case.finalize(
                SlotState.UNKNOWN,
                _unknown_scores(case),
                f"Part2 stopped without terminal confidence: {stop_reason}",
                unresolved_reasons=unresolved,
                exhausted=True,
            )
        return SlotRunResult(
            case=case,
            stop_reason=stop_reason,
            model_turns=model_turns,
            tool_rounds=len(case.rounds),
            validation_errors=tuple(validation_errors),
        )


__all__ = [
    "MAX_MODEL_TURNS",
    "TERMINAL_CONFIDENCE",
    "SingleSlotAgent",
    "SlotRunResult",
]
