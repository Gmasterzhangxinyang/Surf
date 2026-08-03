from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    AgentContext,
    DecisionState,
    FreeEvidence,
    OccupiedEvidence,
    ScopeEvidence,
    ScopeStatus,
    SlotDecision,
    StabilityEvidence,
    WeakEvidenceAssessment,
)


_OWNERSHIP_FAILURE_CODES = frozenset(
    {
        "adjacent_overlap_conflict",
        "boundary_dominated",
        "low_core_overlap",
        "outside_residual_conflict",
    }
)


@dataclass(frozen=True)
class QualityEvidence:
    passed: bool = True
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", tuple(sorted(set(self.failures))))


def assess_weak_evidence(
    occupied: OccupiedEvidence,
    free: FreeEvidence,
) -> WeakEvidenceAssessment:
    """Describe weak evidence without using it to manufacture a terminal state."""

    active = bool(
        occupied.weak
        or (
            not occupied.strong
            and (free.weak_obstacle or free.unresolved_core_hit)
        )
    )
    if not active:
        return WeakEvidenceAssessment()

    occupied_failures = set(occupied.failures)
    ownership_failures = occupied_failures & _OWNERSHIP_FAILURE_CODES
    if ownership_failures:
        ownership = (
            "shared_or_boundary"
            if "ownership_conflict" in free.failures
            else "external"
        )
    elif free.weak_obstacle or free.unresolved_core_hit:
        ownership = "target_core"
    elif occupied.weak:
        ownership = "target_core"
    else:
        ownership = "unknown"

    morphology: list[str] = []
    if free.weak_obstacle:
        morphology.append("repeatable_core_obstacle")
    elif free.unresolved_core_hit:
        morphology.append("sparse_core_hit")
    if "linear_static_structure" in occupied_failures:
        morphology.append("linear_static")
    if "low_height_structure" in occupied_failures:
        morphology.append("low_height")
    if occupied.weak and not ({"linear_static_structure", "low_height_structure"} & occupied_failures):
        morphology.append("vehicle_incomplete")
    if not morphology:
        morphology.append("vehicle_incomplete")

    features = occupied.features
    if features is None or not features.splits:
        temporal = "insufficient"
    elif any(not split.agreement for split in features.splits):
        temporal = "inconsistent"
    elif all(split.agreement for split in features.splits):
        temporal = "persistent"
    else:
        temporal = "partial"

    if free.conflict:
        free_context = "independent_hit_free_conflict"
    elif free.positive_geometry:
        free_context = "positive_geometry"
    else:
        free_context = "visibility_limited"

    may_enter_free_stability = bool(
        ownership == "external"
        and free.positive_geometry
        and free.strong
        and free.weak_ownership_resolved
        and not free.unresolved_core_hit
        and not free.weak_obstacle
        and not free.conflict
    )
    disposition = "free_stability" if may_enter_free_stability else "agent_only"

    if free.conflict:
        primary_reason = "weak_hit_free_conflict"
    elif free.positive_geometry and free.unresolved_core_hit:
        primary_reason = "weak_core_clearance_conflict"
    elif ownership == "shared_or_boundary":
        primary_reason = "weak_shared_ownership"
    elif "linear_static" in morphology or "low_height" in morphology:
        primary_reason = "weak_static_structure"
    elif temporal == "inconsistent":
        primary_reason = "weak_temporal_inconsistent"
    elif may_enter_free_stability:
        primary_reason = "weak_external_ownership"
    elif free_context == "visibility_limited":
        primary_reason = "weak_visibility_limited"
    else:
        primary_reason = "weak_vehicle_incomplete"

    reason_codes = (
        f"weak_ownership_{ownership}",
        *(f"weak_morphology_{value}" for value in morphology),
        f"weak_temporal_{temporal}",
        f"weak_free_{free_context}",
        f"weak_disposition_{disposition}",
    )
    return WeakEvidenceAssessment(
        active=True,
        ownership=ownership,
        morphology=tuple(morphology),
        temporal=temporal,
        free_context=free_context,
        disposition=disposition,
        primary_reason=primary_reason,
        reason_codes=tuple(reason_codes),
    )


def _validate_evidence_contracts(
    occupied: OccupiedEvidence,
    free: FreeEvidence,
) -> None:
    if occupied.strong and occupied.failures:
        raise ValueError("strong occupied evidence contains failed gates")
    if free.strong and free.failures:
        raise ValueError("strong free evidence contains failed gates")
    if occupied.strong and any(not gate.passed for gate in occupied.gate_results):
        raise ValueError("strong occupied evidence contains failed gates")
    if free.strong and any(not gate.passed for gate in free.gate_results):
        raise ValueError("strong free evidence contains failed gates")
    if free.strong and free.unresolved_core_hit:
        raise ValueError("strong free evidence contains an unresolved core hit")
    if free.strong and not free.positive_geometry:
        raise ValueError("strong free evidence lacks positive free geometry")
    if free.weak_ownership_resolved:
        if not occupied.weak:
            raise ValueError("resolved weak ownership requires weak occupied evidence")
        if free.weak_obstacle:
            raise ValueError("resolved weak ownership contains a core obstacle")
        if free.unresolved_core_hit:
            raise ValueError("resolved weak ownership contains an unresolved core hit")
        if not (_OWNERSHIP_FAILURE_CODES & set(occupied.failures)):
            raise ValueError("resolved weak ownership lacks an ownership failure")


def route_decision(
    scope: ScopeEvidence,
    occupied: OccupiedEvidence,
    free: FreeEvidence,
    quality: QualityEvidence,
    stability: StabilityEvidence,
    agent_context: AgentContext,
) -> SlotDecision | None:
    scope_status = ScopeStatus(scope.scope_status)
    if scope_status is ScopeStatus.OUT_OF_ROUTE:
        return None
    _validate_evidence_contracts(occupied, free)
    weak_assessment = assess_weak_evidence(occupied, free)
    reference_frames = tuple(sorted(set(scope.crossing_frames) | set(scope.hit_frames)))

    def decision(
        state: DecisionState,
        reason: str,
        unknown_reasons: tuple[str, ...] = (),
    ) -> SlotDecision:
        return SlotDecision(
            slot_id=scope.slot_id,
            scope_status=scope_status,
            state=state,
            decision_reason=reason,
            unknown_reasons=unknown_reasons,
            occupied_evidence=occupied,
            free_evidence=free,
            stability=stability,
            reference_frames=reference_frames,
            agent_context=agent_context,
            weak_evidence=weak_assessment,
        )

    if scope_status is ScopeStatus.PARTIAL_ROUTE:
        reasons = tuple(scope.reasons) + ("partial_route_scope",)
        return decision(DecisionState.UNKNOWN, "partial_route_scope", reasons)

    if not quality.passed:
        reasons = quality.failures or ("quality_gate_failed",)
        return decision(DecisionState.UNKNOWN, "quality_gate_failed", reasons)

    evidence_conflict = bool(free.conflict or (occupied.strong and free.strong))
    if evidence_conflict:
        return decision(
            DecisionState.UNKNOWN,
            "occupied_free_conflict",
            ("occupied_free_conflict",),
        )

    if stability.failures:
        return decision(
            DecisionState.UNKNOWN,
            "stability_evaluation_error",
            stability.failures,
        )

    if occupied.strong:
        if not stability.stable:
            return decision(
                DecisionState.UNKNOWN,
                "pose_unstable",
                ("pose_sensitive_terminal",),
            )
        return decision(DecisionState.OCCUPIED, "strong_occupied_evidence")

    ownership_resolved_free = bool(
        free.strong
        and occupied.weak
        and not free.weak_obstacle
        and free.weak_ownership_resolved
    )
    if (occupied.weak or free.weak_obstacle) and not ownership_resolved_free:
        reasons: list[str] = list(occupied.failures)
        if occupied.weak:
            reasons.append("weak_vehicle_evidence")
        if free.weak_obstacle:
            reasons.append("weak_obstacle_evidence")
        if weak_assessment.primary_reason:
            reasons.append(weak_assessment.primary_reason)
        return decision(
            DecisionState.UNKNOWN,
            "weak_obstacle_evidence",
            tuple(reasons),
        )

    if free.strong:
        if not stability.stable:
            return decision(
                DecisionState.UNKNOWN,
                "pose_unstable",
                ("pose_sensitive_terminal",),
            )
        return decision(DecisionState.FREE, "strong_free_space_evidence")

    remaining_reasons = tuple(occupied.failures) + tuple(free.failures)
    if not remaining_reasons:
        remaining_reasons = ("insufficient_terminal_evidence",)
    return decision(
        DecisionState.UNKNOWN,
        "insufficient_terminal_evidence",
        remaining_reasons,
    )
