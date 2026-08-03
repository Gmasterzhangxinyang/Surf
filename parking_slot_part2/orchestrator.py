"""Bounded provider-neutral orchestration for one Part2 task group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import QueueEnvelope, QueueItem, freeze_json
from .grouping import MAX_GROUP_SIZE, TaskGroup
from .model import (
    MODEL_TURN_SCHEMA_VERSION,
    ActionValidationError,
    GroupFinalProposal,
    MissingReplayError,
    ModelAdapter,
    ReplayExhaustedError,
    SlotAssessment,
    ToolRequest,
    parse_model_action,
)
from .preflight import OPAQUE_EVIDENCE_ID
from .queueing import canonical_json_bytes, canonical_sha256
from .tools import TOOL_NAMES, ToolRegistry, ToolResult
from .trace import EvidenceLedger


MAX_MODEL_TURNS = 4
MAX_TOOL_ATTEMPTS = 3


def _mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ToolAttemptRecord:
    """Immutable audit record for one counted tool attempt."""

    attempt_id: str
    attempt_index: int
    model_turn: int
    tool_name: str | None
    arguments: Mapping[str, Any]
    evidence_id: str | None
    disposition: str
    executed: bool
    result_status: str
    error_code: str | None
    error_message: str | None
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.disposition not in {"executed", "duplicate", "prevalidation_rejected"}:
            raise ValueError("invalid tool attempt disposition")
        if self.executed != (self.disposition == "executed"):
            raise ValueError("executed must match the tool attempt disposition")
        object.__setattr__(self, "arguments", freeze_json(dict(self.arguments)))
        object.__setattr__(self, "data", freeze_json(dict(self.data)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "model_turn": self.model_turn,
            "tool_name": self.tool_name,
            "arguments": _mutable_json(self.arguments),
            "evidence_id": self.evidence_id,
            "disposition": self.disposition,
            "executed": self.executed,
            "result_status": self.result_status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "data": _mutable_json(self.data),
        }


@dataclass(frozen=True, slots=True)
class GroupRunResult:
    """Complete bounded-run result, including proposal and immutable audit data."""

    group_id: str
    proposal: GroupFinalProposal
    model_turns: int
    tool_attempts: int
    stop_reason: str
    trace_span_id: str
    ledger: EvidenceLedger
    tool_attempt_records: tuple[ToolAttemptRecord, ...]
    tool_results: tuple[ToolResult, ...]

    @property
    def assessments(self) -> tuple[SlotAssessment, ...]:
        return self.proposal.assessments

    @property
    def trace_events(self) -> tuple[Mapping[str, Any], ...]:
        return self.ledger.events

    @property
    def trace_jsonl(self) -> str:
        return self.ledger.to_jsonl()

    @property
    def tool_attempt_count(self) -> int:
        return self.tool_attempts

    @property
    def tool_calls_attempted(self) -> int:
        return self.tool_attempts

    @property
    def attempted_tool_calls(self) -> int:
        return self.tool_attempts


def _group_items(group: TaskGroup, queue: QueueEnvelope) -> dict[str, QueueItem]:
    if not group.task_ids:
        raise ValueError("run_group requires at least one decision task")
    if len(group.task_ids) > MAX_GROUP_SIZE:
        raise ValueError(f"run_group accepts at most {MAX_GROUP_SIZE} decision tasks")
    if len(set(group.task_ids)) != len(group.task_ids):
        raise ValueError("run_group task_ids must be unique")
    queue_items = {item.task_id: item for item in queue.items}
    selected: dict[str, QueueItem] = {}
    for task_id in group.task_ids:
        try:
            item = queue_items[task_id]
        except KeyError as exc:
            raise ValueError(f"group references queue-missing task_id: {task_id}") from exc
        if item.encounter.encounter_id != group.encounter_id:
            raise ValueError("group task encounter does not match group.encounter_id")
        selected[task_id] = item
    return selected


def make_attempt_id(group_id: str, attempt_index: int) -> str:
    """Derive the only valid public identity for one counted tool attempt."""

    return "attempt_" + canonical_sha256(
        {
            "group_id": group_id,
            "attempt_index": attempt_index,
        }
    ).split(":", 1)[1]


def _safe_raw_request(action: Any) -> tuple[str | None, dict[str, Any], str | None]:
    if not isinstance(action, Mapping):
        return None, {}, None
    raw_name = action.get("tool_name")
    tool_name = raw_name if isinstance(raw_name, str) and raw_name in TOOL_NAMES else None
    raw_arguments = action.get("arguments")
    if not isinstance(raw_arguments, Mapping) or set(raw_arguments) != {"evidence_id"}:
        return tool_name, {}, None
    evidence_id = raw_arguments.get("evidence_id")
    if not isinstance(evidence_id, str) or OPAQUE_EVIDENCE_ID.fullmatch(evidence_id) is None:
        return tool_name, {}, None
    return tool_name, {"evidence_id": evidence_id}, evidence_id


def _prevalidation_attempt(
    *,
    group_id: str,
    attempt_index: int,
    model_turn: int,
    action: Any,
    error: ActionValidationError,
) -> ToolAttemptRecord:
    tool_name, arguments, evidence_id = _safe_raw_request(action)
    return ToolAttemptRecord(
        attempt_id=make_attempt_id(group_id, attempt_index),
        attempt_index=attempt_index,
        model_turn=model_turn,
        tool_name=tool_name,
        arguments=arguments,
        evidence_id=evidence_id,
        disposition="prevalidation_rejected",
        executed=False,
        result_status="invalid",
        error_code=error.code,
        error_message=error.message,
        data=error.details,
    )


def _duplicate_raw_attempt(
    *,
    group_id: str,
    attempt_index: int,
    model_turn: int,
    action: Any,
) -> ToolAttemptRecord:
    tool_name, arguments, evidence_id = _safe_raw_request(action)
    return ToolAttemptRecord(
        attempt_id=make_attempt_id(group_id, attempt_index),
        attempt_index=attempt_index,
        model_turn=model_turn,
        tool_name=tool_name,
        arguments=arguments,
        evidence_id=evidence_id,
        disposition="duplicate",
        executed=False,
        result_status="duplicate",
        error_code="duplicate_tool_request",
        error_message="an identical tool request was already attempted",
        data={},
    )


def _executed_attempt(
    *,
    group_id: str,
    attempt_index: int,
    model_turn: int,
    request: ToolRequest,
    result: ToolResult,
) -> ToolAttemptRecord:
    return ToolAttemptRecord(
        attempt_id=make_attempt_id(group_id, attempt_index),
        attempt_index=attempt_index,
        model_turn=model_turn,
        tool_name=request.tool_name,
        arguments=request.arguments,
        evidence_id=request.evidence_id,
        disposition="executed",
        executed=True,
        result_status=result.status,
        error_code=result.error_code,
        error_message=result.error_message,
        data=result.data,
    )


def _feedback(error: ActionValidationError, *, model_turn: int) -> dict[str, Any]:
    return {
        "kind": "validation_feedback",
        "model_turn": model_turn,
        "code": error.code,
        "message": error.message,
        "details": _mutable_json(error.details),
    }


def _record_validation(
    ledger: EvidenceLedger,
    observation: Mapping[str, Any],
) -> None:
    ledger.record(
        "validation",
        "validation_feedback",
        {
            "model_turn": observation["model_turn"],
            "code": observation["code"],
            "details": observation["details"],
        },
    )


def _model_request(
    *,
    group: TaskGroup,
    items: Mapping[str, QueueItem],
    turn: int,
    tool_attempt_count: int,
    basic_context: Mapping[str, Any],
    observations: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_TURN_SCHEMA_VERSION,
        "group_id": group.group_id,
        "turn": turn,
        "decision_tasks": [
            {
                "task_id": item.task_id,
                "slot_id": item.slot_id,
                "unknown_reasons": list(item.unknown_reasons),
                "allowed_final_states": list(item.allowed_final_states),
                "occupied_evidence": _mutable_json(item.occupied_evidence),
                "free_evidence": _mutable_json(item.free_evidence),
            }
            for item in items.values()
        ],
        "basic_context": _mutable_json(basic_context),
        "observations": [_mutable_json(observation) for observation in observations],
        "budget": {
            "max_model_turns": MAX_MODEL_TURNS,
            "model_turns_used": turn - 1,
            "model_turns_remaining_after_this": MAX_MODEL_TURNS - turn,
            "max_tool_attempts": MAX_TOOL_ATTEMPTS,
            "tool_attempts_used": tool_attempt_count,
            "tool_attempts_remaining": MAX_TOOL_ATTEMPTS - tool_attempt_count,
        },
    }


def _safe_tool_result(request: ToolRequest, registry: ToolRegistry) -> ToolResult:
    try:
        result = registry.invoke(request.tool_name, request.arguments)
    except Exception as exc:
        return ToolResult(
            tool_name=request.tool_name,
            status="failed",
            evidence_id=request.evidence_id,
            data=freeze_json({}),
            error_code="tool_execution_error",
            error_message=type(exc).__name__,
        )
    if not isinstance(result, ToolResult):
        return ToolResult(
            tool_name=request.tool_name,
            status="failed",
            evidence_id=request.evidence_id,
            data=freeze_json({}),
            error_code="invalid_tool_result",
            error_message="tool registry returned an invalid result type",
        )
    return result


def _validate_request_scope(
    request: ToolRequest,
    *,
    task_ids: frozenset[str],
    registry: ToolRegistry,
) -> None:
    catalog = getattr(registry, "catalog", None)
    getter = getattr(catalog, "get", None)
    if getter is None:
        return
    try:
        record = getter(request.evidence_id)
    except KeyError:
        return
    except Exception as exc:
        raise ActionValidationError(
            "evidence_catalog_error",
            "evidence scope could not be validated",
            details={"error_type": type(exc).__name__},
        ) from exc
    if getattr(record, "task_id", None) not in task_ids:
        raise ActionValidationError(
            "evidence_outside_group",
            "tool evidence belongs to a task outside this decision group",
        )


def _validate_proposal_evidence(
    proposal: GroupFinalProposal,
    *,
    task_ids: frozenset[str],
    registry: ToolRegistry,
) -> None:
    catalog = getattr(registry, "catalog", None)
    getter = getattr(catalog, "get", None)
    if getter is None:
        return
    for assessment in proposal.assessments:
        for evidence_id in assessment.evidence_refs:
            try:
                record = getter(evidence_id)
            except KeyError as exc:
                raise ActionValidationError(
                    "unknown_evidence_ref",
                    "assessment references evidence outside the preflight catalog",
                ) from exc
            except Exception as exc:
                raise ActionValidationError(
                    "evidence_catalog_error",
                    "assessment evidence scope could not be validated",
                    details={"error_type": type(exc).__name__},
                ) from exc
            record_task_id = getattr(record, "task_id", None)
            if record_task_id not in task_ids:
                raise ActionValidationError(
                    "evidence_outside_group",
                    "assessment references evidence outside this decision group",
                )
            if record_task_id != assessment.task_id:
                raise ActionValidationError(
                    "evidence_task_mismatch",
                    "assessment evidence belongs to a different decision task",
                )


def make_fallback_proposal(
    group: TaskGroup,
    items: Mapping[str, QueueItem],
    stop_reason: str,
) -> GroupFinalProposal:
    """Build the deterministic safe proposal used for every non-final stop."""

    reason_code = f"orchestrator_{stop_reason}"
    assessments = tuple(
        SlotAssessment(
            task_id=item.task_id,
            slot_id=item.slot_id,
            proposed_state="unknown",
            target_visibility="unknown",
            target_ownership="uncertain",
            semantic_finding="unclear",
            resolved_unknown_reasons=(),
            unresolved_blockers=item.unknown_reasons or (stop_reason,),
            evidence_refs=(),
            reason_codes=(reason_code,),
        )
        for item in items.values()
    )
    return GroupFinalProposal(group.group_id, assessments)


def run_group(
    group: TaskGroup,
    queue: QueueEnvelope,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    basic_context: Mapping[str, Any],
    ledger: EvidenceLedger | None = None,
) -> GroupRunResult:
    """Run one group with hard four-turn/three-tool bounds and safe fallback."""

    items = _group_items(group, queue)
    if not isinstance(basic_context, Mapping):
        raise ValueError("basic_context must be a JSON object")
    canonical_json_bytes(basic_context)
    immutable_context = freeze_json(dict(basic_context))
    evidence_ledger = ledger or EvidenceLedger(group.group_id)
    if evidence_ledger.group_id != group.group_id:
        raise ValueError("ledger group_id must match the task group")
    if evidence_ledger.events or evidence_ledger.sealed:
        raise ValueError("run_group requires a fresh, unsealed ledger")

    task_ids = frozenset(items)
    observations: list[Mapping[str, Any]] = []
    attempts: list[ToolAttemptRecord] = []
    tool_results: list[ToolResult] = []
    seen_requests: set[bytes] = set()
    model_turns = 0
    tool_attempt_count = 0
    proposal: GroupFinalProposal | None = None
    stop_reason: str | None = None

    evidence_ledger.record(
        "lifecycle",
        "group_started",
        {
            "group_id": group.group_id,
            "task_ids": list(group.task_ids),
            "max_model_turns": MAX_MODEL_TURNS,
            "max_tool_attempts": MAX_TOOL_ATTEMPTS,
        },
    )

    for turn in range(1, MAX_MODEL_TURNS + 1):
        model_turns = turn
        evidence_ledger.record(
            "model",
            "model_turn_started",
            {
                "model_turn": turn,
                "tool_attempts_used": tool_attempt_count,
                "observation_count": len(observations),
            },
        )
        request_payload = _model_request(
            group=group,
            items=items,
            turn=turn,
            tool_attempt_count=tool_attempt_count,
            basic_context=immutable_context,
            observations=observations,
        )
        try:
            raw_action = model.next_action(request_payload)
        except MissingReplayError:
            evidence_ledger.record("model", "replay_missing", {"model_turn": turn})
            stop_reason = "missing_replay"
            break
        except ReplayExhaustedError:
            evidence_ledger.record("model", "replay_exhausted", {"model_turn": turn})
            stop_reason = "replay_exhausted"
            break
        except Exception as exc:
            evidence_ledger.record(
                "model",
                "model_error",
                {"model_turn": turn, "error_type": type(exc).__name__},
            )
            stop_reason = "model_error"
            break

        action_type = (
            raw_action.get("type")
            if isinstance(raw_action, Mapping)
            and raw_action.get("type") in {"tool_request", "final_proposal"}
            else "unrecognized"
        )
        evidence_ledger.record(
            "model",
            "model_action_received",
            {"model_turn": turn, "action_type": action_type},
        )

        declared_tool_request = (
            isinstance(raw_action, Mapping) and raw_action.get("type") == "tool_request"
        )
        if declared_tool_request and tool_attempt_count >= MAX_TOOL_ATTEMPTS:
            error = ActionValidationError(
                "tool_budget_exhausted",
                "no additional tool request may be attempted",
            )
            observation = _feedback(error, model_turn=turn)
            observations.append(freeze_json(observation))
            _record_validation(evidence_ledger, observation)
            stop_reason = "tool_budget_exhausted"
            break

        if declared_tool_request:
            tool_attempt_count += 1

        raw_request_key: bytes | None = None
        if declared_tool_request:
            try:
                raw_request_key = canonical_json_bytes(raw_action)
            except (TypeError, ValueError):
                raw_request_key = None
            if raw_request_key is not None and raw_request_key in seen_requests:
                duplicate = _duplicate_raw_attempt(
                    group_id=group.group_id,
                    attempt_index=tool_attempt_count,
                    model_turn=turn,
                    action=raw_action,
                )
                attempts.append(duplicate)
                evidence_ledger.record(
                    "tool",
                    "tool_attempted",
                    {
                        "attempt_id": duplicate.attempt_id,
                        "attempt_index": duplicate.attempt_index,
                        "model_turn": turn,
                        "tool_name": duplicate.tool_name,
                        "evidence_id": duplicate.evidence_id,
                        "disposition": duplicate.disposition,
                    },
                )
                error = ActionValidationError(
                    "duplicate_tool_request",
                    "an identical tool request was already attempted",
                )
                observation = _feedback(error, model_turn=turn)
                observations.append(freeze_json(observation))
                _record_validation(evidence_ledger, observation)
                continue
            if raw_request_key is not None:
                seen_requests.add(raw_request_key)

        try:
            action = parse_model_action(
                raw_action,
                group_id=group.group_id,
                expected_items=items,
            )
            if isinstance(action, GroupFinalProposal):
                _validate_proposal_evidence(
                    action,
                    task_ids=task_ids,
                    registry=registry,
                )
        except ActionValidationError as error:
            if declared_tool_request:
                attempt = _prevalidation_attempt(
                    group_id=group.group_id,
                    attempt_index=tool_attempt_count,
                    model_turn=turn,
                    action=raw_action,
                    error=error,
                )
                attempts.append(attempt)
                evidence_ledger.record(
                    "tool",
                    "tool_attempted",
                    {
                        "attempt_id": attempt.attempt_id,
                        "attempt_index": attempt.attempt_index,
                        "model_turn": turn,
                        "tool_name": attempt.tool_name,
                        "evidence_id": attempt.evidence_id,
                        "disposition": attempt.disposition,
                    },
                )
            observation = _feedback(error, model_turn=turn)
            observations.append(freeze_json(observation))
            _record_validation(evidence_ledger, observation)
            continue

        if isinstance(action, GroupFinalProposal):
            proposal = action
            evidence_ledger.record(
                "validation",
                "final_proposal_accepted",
                {
                    "model_turn": turn,
                    "assessment_task_ids": [row.task_id for row in action.assessments],
                    "proposal": action.to_dict(),
                },
            )
            stop_reason = "final_proposal"
            break

        try:
            _validate_request_scope(action, task_ids=task_ids, registry=registry)
        except ActionValidationError as error:
            rejected = _prevalidation_attempt(
                group_id=group.group_id,
                attempt_index=tool_attempt_count,
                model_turn=turn,
                action=action.to_action(),
                error=error,
            )
            attempts.append(rejected)
            evidence_ledger.record(
                "tool",
                "tool_attempted",
                {
                    "attempt_id": rejected.attempt_id,
                    "attempt_index": rejected.attempt_index,
                    "model_turn": turn,
                    "tool_name": rejected.tool_name,
                    "evidence_id": rejected.evidence_id,
                    "disposition": rejected.disposition,
                },
            )
            observation = _feedback(error, model_turn=turn)
            observations.append(freeze_json(observation))
            _record_validation(evidence_ledger, observation)
            continue

        result = _safe_tool_result(action, registry)
        tool_results.append(result)
        executed = _executed_attempt(
            group_id=group.group_id,
            attempt_index=tool_attempt_count,
            model_turn=turn,
            request=action,
            result=result,
        )
        attempts.append(executed)
        evidence_ledger.record(
            "tool",
            "tool_attempted",
            {
                "attempt_id": executed.attempt_id,
                "attempt_index": executed.attempt_index,
                "model_turn": turn,
                "tool_name": executed.tool_name,
                "evidence_id": executed.evidence_id,
                "disposition": executed.disposition,
            },
        )
        tool_observation = {
            "kind": "tool_result",
            "model_turn": turn,
            "attempt_id": executed.attempt_id,
            "attempt_index": executed.attempt_index,
            "tool_name": result.tool_name,
            "evidence_id": result.evidence_id,
            "status": result.status,
            "error_code": result.error_code,
            "data": _mutable_json(result.data),
        }
        observations.append(freeze_json(tool_observation))
        evidence_ledger.record(
            "tool",
            "tool_result",
            {
                "attempt_id": executed.attempt_id,
                "attempt_index": executed.attempt_index,
                "model_turn": turn,
                "status": result.status,
                "error_code": result.error_code,
                "data": result.data,
            },
        )

    if stop_reason is None:
        stop_reason = "model_turn_limit"
    if proposal is None:
        proposal = make_fallback_proposal(group, items, stop_reason)

    evidence_ledger.record(
        "lifecycle",
        "group_finished",
        {
            "group_id": group.group_id,
            "model_turns": model_turns,
            "tool_attempt_count": tool_attempt_count,
            "assessment_count": len(proposal.assessments),
        },
    )
    evidence_ledger.record(
        "stop",
        "group_stopped",
        {
            "stop_reason": stop_reason,
            "model_turns": model_turns,
            "tool_attempt_count": tool_attempt_count,
        },
    )
    evidence_ledger.seal()
    return GroupRunResult(
        group_id=group.group_id,
        proposal=proposal,
        model_turns=model_turns,
        tool_attempts=tool_attempt_count,
        stop_reason=stop_reason,
        trace_span_id=evidence_ledger.trace_span_id,
        ledger=evidence_ledger,
        tool_attempt_records=tuple(attempts),
        tool_results=tuple(tool_results),
    )


__all__ = [
    "MAX_MODEL_TURNS",
    "MAX_TOOL_ATTEMPTS",
    "GroupRunResult",
    "ToolAttemptRecord",
    "make_attempt_id",
    "make_fallback_proposal",
    "run_group",
]
