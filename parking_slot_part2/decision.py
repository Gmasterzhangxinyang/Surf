"""Deterministic semantic gate, overlap consolidation, and immutable base merge."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .contracts import QueueEnvelope, QueueItem
from .grouping import build_groups
from .model import SlotAssessment
from .orchestrator import GroupRunResult, ToolAttemptRecord
from .preflight import EvidenceCatalog, EvidenceRecord
from .queueing import canonical_json_bytes


DECISION_POLICY_VERSION = "part2-decision-gate/1.0"
RESOLUTIONS_SCHEMA_VERSION = "part2-resolutions/1.0"
FINAL_ROUTE_STATES_SCHEMA_VERSION = "part2-final-route-states/1.0"

_TERMINAL_STATES = frozenset({"occupied", "free"})
_STATES = _TERMINAL_STATES | {"unknown"}
_ROUTE_SCOPES = frozenset(
    {"in_route_scope", "partial_route_scope", "out_of_route_scope"}
)


def _stable(values: Sequence[str] | set[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _plain_json(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


@dataclass(frozen=True, slots=True)
class GatedAssessment:
    """One group occurrence after deterministic terminal-state gating."""

    task_id: str
    slot_id: str
    scope_status: str
    group_id: str
    trace_span_id: str
    proposed_state: str
    state: str
    attempt_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "slot_id": self.slot_id,
            "scope_status": self.scope_status,
            "group_id": self.group_id,
            "trace_span_id": self.trace_span_id,
            "proposed_state": self.proposed_state,
            "state": self.state,
            "attempt_ids": list(self.attempt_ids),
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class SlotResolution:
    """Exactly one consolidated Part2 resolution for one queue task."""

    task_id: str
    slot_id: str
    scope_status: str
    input_state: str
    state: str
    decision_source: str
    model_turns: int
    tool_calls_attempted: int
    stop_reasons: tuple[str, ...]
    completion_status: str
    group_ids: tuple[str, ...]
    trace_span_ids: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.slot_id:
            raise ValueError("slot resolution requires task_id and slot_id")
        if self.scope_status not in {"in_route_scope", "partial_route_scope"}:
            raise ValueError("slot resolution scope must be route-observable")
        if self.input_state != "unknown":
            raise ValueError("slot resolution input_state must be unknown")
        if self.state not in _STATES:
            raise ValueError("slot resolution state is invalid")
        expected_source = (
            "part2_agent_validated" if self.state in _TERMINAL_STATES else "part2_agent_unknown"
        )
        if self.decision_source != expected_source:
            raise ValueError("slot resolution decision_source does not match state")
        expected_completion = "resolved" if self.state in _TERMINAL_STATES else "remained_unknown"
        if self.completion_status != expected_completion:
            raise ValueError("slot resolution completion_status does not match state")
        if (
            isinstance(self.model_turns, bool)
            or not isinstance(self.model_turns, int)
            or self.model_turns < 0
            or isinstance(self.tool_calls_attempted, bool)
            or not isinstance(self.tool_calls_attempted, int)
            or self.tool_calls_attempted < 0
        ):
            raise ValueError("slot resolution counters must be non-negative integers")
        for field_name in (
            "stop_reasons",
            "group_ids",
            "trace_span_ids",
            "attempt_ids",
            "evidence_refs",
            "reason_codes",
        ):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"slot resolution {field_name} must be unique and sorted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "slot_id": self.slot_id,
            "scope_status": self.scope_status,
            "input_state": self.input_state,
            "state": self.state,
            "decision_source": self.decision_source,
            "model_turns": self.model_turns,
            "tool_calls_attempted": self.tool_calls_attempted,
            "stop_reasons": list(self.stop_reasons),
            "completion_status": self.completion_status,
            "group_ids": list(self.group_ids),
            "trace_span_ids": list(self.trace_span_ids),
            "attempt_ids": list(self.attempt_ids),
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
        }


def _attempts_for_ref(
    result: GroupRunResult,
    evidence_id: str,
) -> tuple[ToolAttemptRecord, ...]:
    return tuple(
        attempt
        for attempt in result.tool_attempt_records
        if attempt.evidence_id == evidence_id
    )


def _successful_attempts(
    result: GroupRunResult,
    evidence_id: str,
    record: EvidenceRecord,
) -> tuple[ToolAttemptRecord, ...]:
    return tuple(
        attempt
        for attempt in _attempts_for_ref(result, evidence_id)
        if attempt.executed
        and attempt.disposition == "executed"
        and attempt.result_status == "ok"
        and attempt.tool_name == record.tool_name
        and dict(attempt.arguments) == {"evidence_id": evidence_id}
    )


def _successful_target_lidar_attempt_exists(
    item: QueueItem,
    result: GroupRunResult,
    catalog: EvidenceCatalog,
) -> bool:
    for attempt in result.tool_attempt_records:
        if (
            not attempt.executed
            or attempt.disposition != "executed"
            or attempt.result_status != "ok"
            or attempt.tool_name != "inspect_lidar_map"
            or attempt.evidence_id is None
        ):
            continue
        try:
            record = catalog.get(attempt.evidence_id)
        except (KeyError, TypeError, ValueError):
            continue
        if record.task_id == item.task_id and record.tool_name == "inspect_lidar_map":
            return True
    return False


def gate_assessment(
    item: QueueItem,
    assessment: SlotAssessment,
    group_result: GroupRunResult,
    catalog: EvidenceCatalog,
) -> GatedAssessment:
    """Conservatively accept or veto one model assessment.

    Only successful, executed evidence attempts owned by ``item`` may support a
    terminal result. The catalog supplies capability metadata but no semantic
    inference; semantics remain explicit structured claims in ``assessment``.
    """

    if assessment.task_id != item.task_id or assessment.slot_id != item.slot_id:
        raise ValueError("assessment task/slot does not match the queue item")
    if group_result.group_id != group_result.proposal.group_id:
        raise ValueError("group result and proposal group_id differ")
    matching = [row for row in group_result.assessments if row.task_id == item.task_id]
    if len(matching) != 1 or matching[0] != assessment:
        raise ValueError("assessment is not the unique matching group occurrence")

    declared_refs = _stable(assessment.evidence_refs)
    audit_attempt_ids: set[str] = set()
    reason_codes: set[str] = set(assessment.reason_codes)

    if assessment.proposed_state == "unknown":
        reason_codes.add("gate_unknown")
        for evidence_id in declared_refs:
            audit_attempt_ids.update(
                attempt.attempt_id for attempt in _attempts_for_ref(group_result, evidence_id)
            )
        return GatedAssessment(
            task_id=item.task_id,
            slot_id=item.slot_id,
            scope_status=item.scope_status,
            group_id=group_result.group_id,
            trace_span_id=group_result.trace_span_id,
            proposed_state="unknown",
            state="unknown",
            attempt_ids=_stable(audit_attempt_ids),
            evidence_refs=declared_refs,
            reason_codes=_stable(reason_codes),
        )

    vetoes: set[str] = set()
    usable_records: list[EvidenceRecord] = []
    if assessment.proposed_state not in _TERMINAL_STATES:
        vetoes.add("invalid_terminal_state")
    if assessment.proposed_state not in item.allowed_final_states:
        vetoes.add("state_not_allowed")
    if not declared_refs:
        vetoes.add("terminal_evidence_unusable")

    for evidence_id in declared_refs:
        all_attempts = _attempts_for_ref(group_result, evidence_id)
        audit_attempt_ids.update(attempt.attempt_id for attempt in all_attempts)
        try:
            record = catalog.get(evidence_id)
        except (KeyError, TypeError, ValueError):
            vetoes.add("terminal_evidence_unusable")
            continue
        if record.task_id != item.task_id:
            vetoes.add("terminal_evidence_cross_task")
            continue
        if not record.available:
            vetoes.add("terminal_evidence_unusable")
            continue
        successful = _successful_attempts(group_result, evidence_id, record)
        if not successful:
            vetoes.add("terminal_evidence_unusable")
            continue
        audit_attempt_ids.update(attempt.attempt_id for attempt in successful)
        usable_records.append(record)

    if assessment.proposed_state == "occupied":
        if assessment.semantic_finding != "vehicle_or_occupying_object":
            vetoes.add("occupied_semantic_mismatch")
        if assessment.target_ownership != "target":
            vetoes.add("occupied_ownership_not_target")
        has_lidar = any(record.tool_name == "inspect_lidar_map" for record in usable_records)
        has_rgb = any(
            record.tool_name == "inspect_rgb_frame"
            and "can_assess_occupied" in record.capabilities
            and assessment.target_visibility == "clear_full"
            for record in usable_records
        )
        if not (has_lidar or has_rgb):
            vetoes.add("occupied_terminal_support_missing")
    elif assessment.proposed_state == "free":
        if assessment.target_visibility != "clear_full":
            vetoes.add("free_visibility_not_clear_full")
        if assessment.semantic_finding != "empty":
            vetoes.add("free_semantic_mismatch")
        if assessment.target_ownership not in {"target", "not_applicable"}:
            vetoes.add("free_ownership_invalid")
        if set(assessment.resolved_unknown_reasons) != set(item.unknown_reasons):
            vetoes.add("free_unknown_reasons_not_exactly_resolved")
        if assessment.unresolved_blockers:
            vetoes.add("free_unresolved_blockers")
        has_free_frame = any(
            record.tool_name == "inspect_rgb_frame"
            and "can_assess_free" in record.capabilities
            for record in usable_records
        )
        if not has_free_frame:
            vetoes.add("free_terminal_support_missing")
        if (
            _successful_target_lidar_attempt_exists(item, group_result, catalog)
            and "lidar_no_contradiction" not in assessment.reason_codes
        ):
            vetoes.add("free_lidar_no_contradiction_missing")

    if vetoes:
        reason_codes.update(vetoes)
        reason_codes.add("terminal_proposal_vetoed")
        state = "unknown"
    else:
        state = assessment.proposed_state
        reason_codes.add(f"gate_{state}")

    return GatedAssessment(
        task_id=item.task_id,
        slot_id=item.slot_id,
        scope_status=item.scope_status,
        group_id=group_result.group_id,
        trace_span_id=group_result.trace_span_id,
        proposed_state=assessment.proposed_state,
        state=state,
        attempt_ids=_stable(audit_attempt_ids),
        evidence_refs=declared_refs,
        reason_codes=_stable(reason_codes),
    )


def consolidate_group_assessments(
    queue: QueueEnvelope,
    group_results: Sequence[GroupRunResult],
    catalog: EvidenceCatalog,
) -> tuple[SlotResolution, ...]:
    """Gate all expected group occurrences and emit one stable row per task."""

    groups = build_groups(queue)
    expected_by_id = {group.group_id: group for group in groups}
    supplied_by_id: dict[str, GroupRunResult] = {}
    for result in group_results:
        if result.group_id in supplied_by_id:
            raise ValueError(f"duplicate group result: {result.group_id}")
        supplied_by_id[result.group_id] = result
    extra_group_ids = sorted(set(supplied_by_id) - set(expected_by_id))
    if extra_group_ids:
        raise ValueError(f"extra group result(s): {', '.join(extra_group_ids)}")

    item_by_task = {item.task_id: item for item in queue.items}
    expected_group_ids: dict[str, list[str]] = {task_id: [] for task_id in item_by_task}
    gated_by_task: dict[str, list[GatedAssessment]] = {
        task_id: [] for task_id in item_by_task
    }
    missing_by_task: dict[str, bool] = {task_id: False for task_id in item_by_task}
    results_by_task: dict[str, list[GroupRunResult]] = {
        task_id: [] for task_id in item_by_task
    }

    for group in groups:
        for task_id in group.task_ids:
            expected_group_ids[task_id].append(group.group_id)
        result = supplied_by_id.get(group.group_id)
        if result is None:
            for task_id in group.task_ids:
                missing_by_task[task_id] = True
            continue
        for task_id in group.task_ids:
            results_by_task[task_id].append(result)
        if result.proposal.group_id != group.group_id:
            raise ValueError("group result proposal group_id does not match expected group")
        assessments: dict[str, SlotAssessment] = {}
        for assessment in result.assessments:
            if assessment.task_id in assessments:
                raise ValueError(
                    f"duplicate assessment occurrence for group {group.group_id}"
                )
            if assessment.task_id not in group.task_ids:
                raise ValueError(f"extra assessment occurrence for group {group.group_id}")
            assessments[assessment.task_id] = assessment
        for task_id in group.task_ids:
            assessment = assessments.get(task_id)
            if assessment is None:
                missing_by_task[task_id] = True
                continue
            gated_by_task[task_id].append(
                gate_assessment(item_by_task[task_id], assessment, result, catalog)
            )

    resolutions: list[SlotResolution] = []
    for task_id in sorted(item_by_task):
        item = item_by_task[task_id]
        occurrences = gated_by_task[task_id]
        reasons = {code for row in occurrences for code in row.reason_codes}
        if missing_by_task[task_id]:
            state = "unknown"
            reasons.add("overlap_result_missing")
        else:
            occurrence_states = {row.state for row in occurrences}
            if len(occurrence_states) == 1:
                state = next(iter(occurrence_states))
            else:
                state = "unknown"
                reasons.add("overlap_disagreement")
        resolutions.append(
            SlotResolution(
                task_id=item.task_id,
                slot_id=item.slot_id,
                scope_status=item.scope_status,
                input_state="unknown",
                state=state,
                decision_source=(
                    "part2_agent_validated"
                    if state in _TERMINAL_STATES
                    else "part2_agent_unknown"
                ),
                model_turns=sum(result.model_turns for result in results_by_task[task_id]),
                tool_calls_attempted=sum(
                    result.tool_calls_attempted for result in results_by_task[task_id]
                ),
                stop_reasons=_stable(
                    {result.stop_reason for result in results_by_task[task_id]}
                ),
                completion_status=(
                    "resolved" if state in _TERMINAL_STATES else "remained_unknown"
                ),
                group_ids=_stable(expected_group_ids[task_id]),
                trace_span_ids=_stable({row.trace_span_id for row in occurrences}),
                attempt_ids=_stable(
                    {attempt_id for row in occurrences for attempt_id in row.attempt_ids}
                ),
                evidence_refs=_stable(
                    {evidence_id for row in occurrences for evidence_id in row.evidence_refs}
                ),
                reason_codes=_stable(reasons),
            )
        )
    return tuple(resolutions)


def _validate_base_decisions(payload: Any) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not isinstance(payload, Mapping):
        raise ValueError("base decisions envelope must be a JSON object")
    expected_fields = {"schema_version", "pipeline", "phase", "decisions"}
    if set(payload) != expected_fields:
        raise ValueError("base decisions envelope fields do not match the strict schema")
    for field_name in ("schema_version", "pipeline", "phase"):
        value = payload[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"base decisions envelope {field_name} must be non-empty")
    if payload["schema_version"] != "1.0":
        raise ValueError("base decisions envelope schema_version must be '1.0'")
    raw_rows = payload["decisions"]
    if not isinstance(raw_rows, (list, tuple)):
        raise ValueError("base decisions envelope decisions must be an array")
    rows: list[dict[str, Any]] = []
    seen_slots: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"base decisions row {index} must be an object")
        if not {"slot_id", "scope_status", "state"} <= set(raw):
            raise ValueError(f"base decisions row {index} is missing required fields")
        slot_id = raw["slot_id"]
        scope_status = raw["scope_status"]
        state = raw["state"]
        if not isinstance(slot_id, str) or not slot_id:
            raise ValueError(f"base decisions row {index} slot_id must be non-empty")
        if slot_id in seen_slots:
            raise ValueError(f"duplicate base slot: {slot_id}")
        seen_slots.add(slot_id)
        if scope_status not in _ROUTE_SCOPES:
            raise ValueError(f"base decisions row {index} has invalid scope_status")
        if state not in _STATES:
            raise ValueError(f"base decisions row {index} has invalid state")
        rows.append(_plain_json(raw))
    metadata = {
        "schema_version": payload["schema_version"],
        "pipeline": payload["pipeline"],
        "phase": payload["phase"],
    }
    return metadata, tuple(rows)


def merge_final_route_states(
    base_payload: Any,
    queue: QueueEnvelope,
    resolutions: Sequence[SlotResolution],
    *,
    part2_run_id: str,
) -> dict[str, Any]:
    """Overlay exact queue resolutions without mutating Part1 terminal rows."""

    if not isinstance(part2_run_id, str) or not part2_run_id:
        raise ValueError("part2_run_id must be a non-empty string")
    base_metadata, base_rows = _validate_base_decisions(base_payload)
    base_by_slot = {row["slot_id"]: row for row in base_rows}

    resolution_by_task: dict[str, SlotResolution] = {}
    resolution_by_slot: dict[str, SlotResolution] = {}
    for resolution in resolutions:
        if not isinstance(resolution, SlotResolution):
            raise ValueError("resolutions must contain SlotResolution records")
        if resolution.task_id in resolution_by_task:
            raise ValueError(f"duplicate resolution task: {resolution.task_id}")
        if resolution.slot_id in resolution_by_slot:
            raise ValueError(f"duplicate resolution slot: {resolution.slot_id}")
        resolution_by_task[resolution.task_id] = resolution
        resolution_by_slot[resolution.slot_id] = resolution

    queue_by_task = {item.task_id: item for item in queue.items}
    expected_tasks = set(queue_by_task)
    supplied_tasks = set(resolution_by_task)
    missing = sorted(expected_tasks - supplied_tasks)
    extra = sorted(supplied_tasks - expected_tasks)
    if missing:
        raise ValueError(f"missing resolution task(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"extra resolution task(s): {', '.join(extra)}")

    for task_id, item in queue_by_task.items():
        resolution = resolution_by_task[task_id]
        if resolution.slot_id != item.slot_id or resolution.scope_status != item.scope_status:
            raise ValueError(f"resolution task/slot/scope mismatch for {task_id}")
        base = base_by_slot.get(item.slot_id)
        if base is None:
            raise ValueError(f"queue slot is missing from base decisions: {item.slot_id}")
        if base["scope_status"] != item.scope_status:
            raise ValueError(f"queue slot scope does not match base: {item.slot_id}")
        if base["state"] != "unknown":
            raise ValueError(f"queue slot must be a base unknown: {item.slot_id}")

    merged_rows: list[dict[str, Any]] = []
    for base in base_rows:
        if base["scope_status"] == "out_of_route_scope":
            continue
        resolution = resolution_by_slot.get(base["slot_id"])
        if resolution is None:
            merged_rows.append(base)
            continue
        merged = dict(base)
        merged["state"] = resolution.state
        merged["part2_resolution"] = resolution.to_dict()
        merged_rows.append(merged)

    return {
        "schema_version": FINAL_ROUTE_STATES_SCHEMA_VERSION,
        "part2_run_id": part2_run_id,
        "queue_id": queue.queue_id,
        "base_decisions": base_metadata,
        "decisions": sorted(merged_rows, key=lambda row: row["slot_id"]),
    }


__all__ = [
    "DECISION_POLICY_VERSION",
    "FINAL_ROUTE_STATES_SCHEMA_VERSION",
    "RESOLUTIONS_SCHEMA_VERSION",
    "GatedAssessment",
    "SlotResolution",
    "consolidate_group_assessments",
    "gate_assessment",
    "merge_final_route_states",
]
