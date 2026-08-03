"""Deterministic Part2 run identities, base loading, and atomic output writers."""

from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from .contracts import QueueEnvelope
from .decision import (
    DECISION_POLICY_VERSION,
    FINAL_ROUTE_STATES_SCHEMA_VERSION,
    RESOLUTIONS_SCHEMA_VERSION,
    SlotResolution,
    consolidate_group_assessments,
    merge_final_route_states,
)
from .grouping import GROUPING_POLICY_VERSION, build_groups
from .model import (
    MODEL_TURN_SCHEMA_VERSION,
    ActionValidationError,
    GroupFinalProposal,
    SlotAssessment,
    parse_model_action,
)
from .orchestrator import (
    MAX_MODEL_TURNS,
    MAX_TOOL_ATTEMPTS,
    GroupRunResult,
    ToolAttemptRecord,
    make_attempt_id,
    make_fallback_proposal,
)
from .preflight import EvidenceCatalog, PREFLIGHT_POLICY_VERSION
from .queueing import (
    SHA256_PATTERN,
    TOOL_NAMES,
    TOOL_REGISTRY_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from .tools import TOOLS_POLICY_VERSION, ToolRegistry, ToolResult
from .trace import EVENT_TYPES, TRACE_SCHEMA_VERSION, EvidenceLedger


RUN_IDENTITY_SCHEMA_VERSION = "part2-run-identity/1.0"
RUN_MANIFEST_SCHEMA_VERSION = "part2-run-manifest/1.0"
SUMMARY_SCHEMA_VERSION = "part2-run-summary/1.0"
TOOL_ATTEMPT_SCHEMA_VERSION = "part2-tool-attempt/1.0"
REPORTING_POLICY_VERSION = "part2-deterministic-reporting/1.0"
ORCHESTRATOR_POLICY_VERSION = "part2-bounded-group-orchestrator/1.0"

_OUTPUT_NAMES = frozenset(
    {
        "part2_resolutions.json",
        "final_route_states.json",
        "decision_trace.jsonl",
        "summary.json",
        "report.html",
        "run_manifest.json",
        "tool_artifacts",
    }
)


def _policy_versions(queue: QueueEnvelope) -> dict[str, Any]:
    if queue.tool_registry_version != TOOL_REGISTRY_VERSION:
        raise ValueError(
            f"queue tool_registry_version must be {TOOL_REGISTRY_VERSION!r}"
        )
    return {
        "decision": DECISION_POLICY_VERSION,
        "grouping": GROUPING_POLICY_VERSION,
        "model_turn_schema": MODEL_TURN_SCHEMA_VERSION,
        "orchestrator": ORCHESTRATOR_POLICY_VERSION,
        "orchestrator_limits": {
            "max_model_turns": MAX_MODEL_TURNS,
            "max_tool_attempts": MAX_TOOL_ATTEMPTS,
        },
        "preflight": PREFLIGHT_POLICY_VERSION,
        "reporting": REPORTING_POLICY_VERSION,
        "tool_registry": TOOL_REGISTRY_VERSION,
        "tools": TOOLS_POLICY_VERSION,
        "trace": TRACE_SCHEMA_VERSION,
    }


def _derive_part2_run_id_from_replay_identity(
    queue: QueueEnvelope,
    replay_identity: str,
) -> str:
    if not isinstance(replay_identity, str) or SHA256_PATTERN.fullmatch(replay_identity) is None:
        raise ValueError("replay_identity must be a canonical lowercase sha256 identity")
    return canonical_sha256(
        {
            "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
            "queue_identity": queue.queue_id,
            "replay_identity": replay_identity,
            "policies": _policy_versions(queue),
        }
    )


def derive_part2_run_id(queue: QueueEnvelope, replay_payload: Mapping[str, Any]) -> str:
    """Derive a stable run identity from both inputs and every policy boundary."""

    if not isinstance(replay_payload, Mapping):
        raise ValueError("replay payload must be a JSON object")
    replay_identity = canonical_sha256(replay_payload)
    return _derive_part2_run_id_from_replay_identity(queue, replay_identity)


def _local_path(uri: str, base_dir: Path) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file"):
        return None
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        return None
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_base_slot_decisions(
    queue: QueueEnvelope,
    *,
    base_dir: str | Path,
) -> Mapping[str, Any]:
    """Rehash and parse the declared Part1 base file at the merge boundary."""

    resources = [
        resource
        for resource in queue.resources.values()
        if resource.kind == "base_slot_decisions"
    ]
    if len(resources) != 1:
        raise ValueError("queue must declare exactly one base_slot_decisions resource")
    resource = resources[0]
    if resource.sha256 is None:
        raise ValueError("base slot decisions resource must declare sha256")
    path = _local_path(resource.uri, Path(base_dir))
    if path is None:
        raise ValueError("base slot decisions resource must be a local file")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"base slot decisions file is unavailable: {type(exc).__name__}") from exc
    actual_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    if actual_hash != resource.sha256:
        raise ValueError("base slot decisions sha256 no longer matches the queue resource")
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"base slot decisions JSON could not be parsed: {type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("base slot decisions JSON must be an object")
    return payload


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, canonical_json_bytes(payload) + b"\n")


def _resolution_payload(
    part2_run_id: str,
    queue: QueueEnvelope,
    resolutions: Sequence[SlotResolution],
) -> dict[str, Any]:
    return {
        "schema_version": RESOLUTIONS_SCHEMA_VERSION,
        "part2_run_id": part2_run_id,
        "queue_id": queue.queue_id,
        "resolutions": [row.to_dict() for row in sorted(resolutions, key=lambda row: row.task_id)],
    }


def _summary(
    part2_run_id: str,
    queue: QueueEnvelope,
    group_results: Sequence[GroupRunResult],
    resolutions: Sequence[SlotResolution],
) -> dict[str, Any]:
    states = Counter(row.state for row in resolutions)
    completion = Counter(row.completion_status for row in resolutions)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "part2_run_id": part2_run_id,
        "queue_id": queue.queue_id,
        "queue_task_count": len(queue.items),
        "group_count": len(group_results),
        "resolution_count": len(resolutions),
        "state_counts": {state: states.get(state, 0) for state in ("occupied", "free", "unknown")},
        "completion_counts": {
            state: completion.get(state, 0)
            for state in ("resolved", "remained_unknown")
        },
        "model_turns": sum(result.model_turns for result in group_results),
        "tool_calls_attempted": sum(
            result.tool_calls_attempted for result in group_results
        ),
        "trace_event_count": sum(len(result.trace_events) for result in group_results),
    }


def _report_html(summary: Mapping[str, Any], resolutions: Sequence[SlotResolution]) -> str:
    rows = []
    for resolution in sorted(resolutions, key=lambda row: row.task_id):
        rows.append(
            "<tr>"
            f"<td>{html.escape(resolution.task_id)}</td>"
            f"<td>{html.escape(resolution.slot_id)}</td>"
            f"<td>{html.escape(resolution.scope_status)}</td>"
            f"<td>{html.escape(resolution.state)}</td>"
            f"<td>{html.escape(resolution.completion_status)}</td>"
            f"<td>{html.escape(', '.join(resolution.reason_codes))}</td>"
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="6">No queued tasks</td></tr>'
    counts = summary["state_counts"]
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8"><title>Part2 Unknown Agent v1</title>'
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:.4rem;text-align:left}</style></head><body>"
        "<h1>Part2 Unknown Agent v1</h1>"
        f"<p>Run: {html.escape(str(summary['part2_run_id']))}</p>"
        f"<p>Queued: {summary['queue_task_count']}; occupied: {counts['occupied']}; "
        f"free: {counts['free']}; unknown: {counts['unknown']}.</p>"
        "<table><thead><tr><th>Task</th><th>Slot</th><th>Scope</th><th>State</th>"
        f"<th>Completion</th><th>Reason codes</th></tr></thead><tbody>{body}</tbody></table>"
        "</body></html>\n"
    )


def _prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output directory path is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    unsupported = sorted(path.name for path in output_dir.iterdir() if path.name not in _OUTPUT_NAMES)
    if unsupported:
        raise ValueError("output directory contains unsupported existing entries")
    tool_dir = output_dir / "tool_artifacts"
    if tool_dir.exists():
        if not tool_dir.is_dir():
            raise ValueError("tool_artifacts path is not a directory")
        shutil.rmtree(tool_dir)
    tool_dir.mkdir()
    return tool_dir


def _attempt_payloads(
    part2_run_id: str,
    group_results: Sequence[GroupRunResult],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    rows: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for result in sorted(group_results, key=lambda row: row.group_id):
        for attempt in sorted(
            result.tool_attempt_records,
            key=lambda row: (row.attempt_index, row.attempt_id),
        ):
            if (
                not attempt.attempt_id
                or Path(attempt.attempt_id).name != attempt.attempt_id
                or "\\" in attempt.attempt_id
            ):
                raise ValueError("tool attempt ID is not a safe filename component")
            if attempt.attempt_id in seen_ids:
                raise ValueError(f"duplicate tool attempt ID: {attempt.attempt_id}")
            seen_ids.add(attempt.attempt_id)
            relative_path = f"tool_artifacts/{attempt.attempt_id}.json"
            rows.append(
                (
                    relative_path,
                    {
                        "schema_version": TOOL_ATTEMPT_SCHEMA_VERSION,
                        "part2_run_id": part2_run_id,
                        "group_id": result.group_id,
                        "trace_span_id": result.trace_span_id,
                        "attempt": attempt.to_dict(),
                    },
                )
            )
    return tuple(rows)


def _validated_resolutions(
    queue: QueueEnvelope,
    resolutions: Sequence[SlotResolution],
) -> tuple[SlotResolution, ...]:
    queue_by_task: dict[str, Any] = {}
    queue_by_slot: dict[str, Any] = {}
    for item in queue.items:
        if item.task_id in queue_by_task:
            raise ValueError(f"duplicate queue task for resolution output: {item.task_id}")
        if item.slot_id in queue_by_slot:
            raise ValueError(f"duplicate queue slot for resolution output: {item.slot_id}")
        queue_by_task[item.task_id] = item
        queue_by_slot[item.slot_id] = item

    resolution_by_task: dict[str, SlotResolution] = {}
    resolution_by_slot: dict[str, SlotResolution] = {}
    for resolution in resolutions:
        if not isinstance(resolution, SlotResolution):
            raise ValueError("resolution output contains a non-SlotResolution record")
        if resolution.task_id in resolution_by_task:
            raise ValueError(f"duplicate resolution task: {resolution.task_id}")
        if resolution.slot_id in resolution_by_slot:
            raise ValueError(f"duplicate resolution slot: {resolution.slot_id}")
        resolution_by_task[resolution.task_id] = resolution
        resolution_by_slot[resolution.slot_id] = resolution

    missing = sorted(set(queue_by_task) - set(resolution_by_task))
    extra = sorted(set(resolution_by_task) - set(queue_by_task))
    if missing:
        raise ValueError(f"missing resolution task(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"extra resolution task(s): {', '.join(extra)}")
    for task_id, item in queue_by_task.items():
        resolution = resolution_by_task[task_id]
        if resolution.slot_id != item.slot_id:
            raise ValueError(f"resolution slot mismatch for queue task {task_id}")
        if resolution.scope_status != item.scope_status:
            raise ValueError(f"resolution scope mismatch for queue task {task_id}")
    return tuple(resolution_by_task[task_id] for task_id in sorted(resolution_by_task))


def _same_json(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _exact_trace_event(
    event: Mapping[str, Any],
    *,
    category: str,
    event_type: str,
    data: Mapping[str, Any],
) -> bool:
    return (
        event.get("category") == category
        and event.get("event_type") == event_type
        and _same_json(event.get("data"), data)
    )


def _validate_trace_integrity(
    result: GroupRunResult,
    *,
    group: Any,
) -> tuple[Mapping[str, Any], ...]:
    ledger = result.ledger
    if not isinstance(ledger, EvidenceLedger):
        raise ValueError(f"group {group.group_id} ledger has an invalid type")
    if ledger.group_id != group.group_id:
        raise ValueError(f"group {group.group_id} ledger belongs to a different group")
    if not ledger.sealed:
        raise ValueError(f"group {group.group_id} ledger must be sealed")
    events = ledger.events
    if not events:
        raise ValueError(f"group {group.group_id} ledger must not be empty")
    expected_span = EvidenceLedger(group.group_id).trace_span_id
    if ledger.trace_span_id != expected_span or result.trace_span_id != expected_span:
        raise ValueError(f"group {group.group_id} trace span identity is inconsistent")
    early_attempt_ids: list[str] = []
    for attempt in result.tool_attempt_records:
        if not isinstance(attempt, ToolAttemptRecord):
            continue
        if not isinstance(attempt.attempt_id, str):
            raise ValueError(f"group {group.group_id} tool attempt ID is invalid")
        early_attempt_ids.append(attempt.attempt_id)
    if len(early_attempt_ids) != len(set(early_attempt_ids)):
        raise ValueError(f"duplicate tool attempt ID in group {group.group_id}")

    expected_fields = {
        "schema_version",
        "trace_span_id",
        "sequence",
        "category",
        "event_type",
        "data",
        "event_id",
    }
    replayed_ledger = EvidenceLedger(group.group_id)
    for sequence, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or set(event) != expected_fields:
            raise ValueError(f"group {group.group_id} trace event schema is invalid")
        category = event["category"]
        event_type = event["event_type"]
        data = event["data"]
        if (
            event["schema_version"] != TRACE_SCHEMA_VERSION
            or event["trace_span_id"] != expected_span
            or isinstance(event["sequence"], bool)
            or not isinstance(event["sequence"], int)
            or event["sequence"] != sequence
            or not isinstance(category, str)
            or category not in EVENT_TYPES
            or not isinstance(event_type, str)
            or event_type not in EVENT_TYPES[category]
            or not isinstance(data, Mapping)
        ):
            raise ValueError(f"group {group.group_id} trace event metadata is inconsistent")
        identity = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_span_id": expected_span,
            "sequence": sequence,
            "category": category,
            "event_type": event_type,
            "data": data,
        }
        expected_event_id = "event_" + canonical_sha256(identity).split(":", 1)[1]
        if event["event_id"] != expected_event_id:
            raise ValueError(f"group {group.group_id} trace event identity is inconsistent")
        try:
            replayed_event = replayed_ledger.record(category, event_type, data)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"group {group.group_id} trace payload contains a forbidden or invalid field"
            ) from exc
        if not _same_json(replayed_event, event):
            raise ValueError(f"group {group.group_id} trace event is not reproducible")

    started = {
        "group_id": group.group_id,
        "task_ids": list(group.task_ids),
        "max_model_turns": MAX_MODEL_TURNS,
        "max_tool_attempts": MAX_TOOL_ATTEMPTS,
    }
    if not _exact_trace_event(
        events[0],
        category="lifecycle",
        event_type="group_started",
        data=started,
    ):
        raise ValueError(f"group {group.group_id} trace start event is inconsistent")
    if len(events) < 4:
        raise ValueError(f"group {group.group_id} ledger is missing terminal events")
    finished = {
        "group_id": group.group_id,
        "model_turns": result.model_turns,
        "tool_attempt_count": result.tool_attempts,
        "assessment_count": len(result.assessments),
    }
    if not _exact_trace_event(
        events[-2],
        category="lifecycle",
        event_type="group_finished",
        data=finished,
    ):
        raise ValueError(
            f"group {group.group_id} trace finish event has inconsistent model turn "
            "or tool attempt counters"
        )
    stopped = {
        "stop_reason": result.stop_reason,
        "model_turns": result.model_turns,
        "tool_attempt_count": result.tool_attempts,
    }
    if not _exact_trace_event(
        events[-1],
        category="stop",
        event_type="group_stopped",
        data=stopped,
    ):
        raise ValueError(f"group {group.group_id} trace stop event is inconsistent")

    current_turn = 0
    attempts_used = 0
    observation_count = 0
    action_turns: set[int] = set()
    for event in events[1:-2]:
        if event["event_type"] == "model_turn_started":
            expected_turn = current_turn + 1
            expected_data = {
                "model_turn": expected_turn,
                "tool_attempts_used": attempts_used,
                "observation_count": observation_count,
            }
            if not _exact_trace_event(
                event,
                category="model",
                event_type="model_turn_started",
                data=expected_data,
            ):
                raise ValueError(f"group {group.group_id} model turn order is inconsistent")
            current_turn = expected_turn
            continue
        if event["category"] == "lifecycle" or event["category"] == "stop":
            raise ValueError(f"group {group.group_id} trace lifecycle order is inconsistent")
        model_turn = event["data"].get("model_turn")
        if isinstance(model_turn, bool) or model_turn != current_turn or current_turn == 0:
            raise ValueError(f"group {group.group_id} trace model turn is inconsistent")
        if event["event_type"] == "model_action_received":
            action_type = event["data"].get("action_type")
            if (
                event["category"] != "model"
                or set(event["data"]) != {"model_turn", "action_type"}
                or action_type not in {"tool_request", "final_proposal", "unrecognized"}
                or current_turn in action_turns
            ):
                raise ValueError(f"group {group.group_id} model action trace is inconsistent")
            action_turns.add(current_turn)
        elif event["event_type"] == "tool_attempted":
            attempts_used += 1
        elif event["event_type"] in {"tool_result", "validation_feedback"}:
            observation_count += 1
    if current_turn != result.model_turns:
        raise ValueError(f"group {group.group_id} model turn counter is inconsistent")
    no_action_on_stop_turn = result.stop_reason in {
        "missing_replay",
        "replay_exhausted",
        "model_error",
    }
    expected_action_turns = set(
        range(1, result.model_turns if no_action_on_stop_turn else result.model_turns + 1)
    )
    if action_turns != expected_action_turns:
        raise ValueError(f"group {group.group_id} model action turn coverage is inconsistent")
    return events


def _validate_attempts_and_results(
    result: GroupRunResult,
    *,
    group: Any,
    catalog: EvidenceCatalog,
    events: tuple[Mapping[str, Any], ...],
) -> None:
    attempts = result.tool_attempt_records
    if not isinstance(attempts, tuple):
        raise ValueError(f"group {group.group_id} tool attempt records must be a tuple")
    if result.tool_attempts != len(attempts):
        raise ValueError(f"group {group.group_id} tool attempt counter is inconsistent")
    attempt_ids = [attempt.attempt_id for attempt in attempts if isinstance(attempt, ToolAttemptRecord)]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError(f"duplicate tool attempt ID in group {group.group_id}")

    attempted_events = tuple(
        event for event in events if event["event_type"] == "tool_attempted"
    )
    if len(attempted_events) != len(attempts):
        raise ValueError(
            f"group {group.group_id} ledger tool attempt events do not match tool attempt records"
        )
    prior_signatures: set[bytes] = set()
    executed: list[ToolAttemptRecord] = []
    for expected_index, (attempt, trace_event) in enumerate(
        zip(attempts, attempted_events),
        start=1,
    ):
        if not isinstance(attempt, ToolAttemptRecord):
            raise ValueError(f"group {group.group_id} contains an invalid tool attempt record")
        if (
            isinstance(attempt.attempt_index, bool)
            or not isinstance(attempt.attempt_index, int)
            or attempt.attempt_index != expected_index
        ):
            raise ValueError(f"group {group.group_id} tool attempt index is inconsistent")
        if attempt.attempt_id != make_attempt_id(group.group_id, expected_index):
            raise ValueError(f"group {group.group_id} tool attempt ID is not derived")
        if (
            isinstance(attempt.model_turn, bool)
            or not isinstance(attempt.model_turn, int)
            or not 1 <= attempt.model_turn <= result.model_turns
            or (expected_index > 1 and attempt.model_turn <= attempts[expected_index - 2].model_turn)
        ):
            raise ValueError(f"group {group.group_id} tool attempt model turn is inconsistent")
        expected_trace_data = {
            "attempt_id": attempt.attempt_id,
            "attempt_index": attempt.attempt_index,
            "model_turn": attempt.model_turn,
            "tool_name": attempt.tool_name,
            "evidence_id": attempt.evidence_id,
            "disposition": attempt.disposition,
        }
        if not _exact_trace_event(
            trace_event,
            category="tool",
            event_type="tool_attempted",
            data=expected_trace_data,
        ):
            raise ValueError(f"group {group.group_id} tool attempt trace is inconsistent")
        signature = canonical_json_bytes(
            {
                "tool_name": attempt.tool_name,
                "arguments": attempt.arguments,
                "evidence_id": attempt.evidence_id,
            }
        )
        if attempt.disposition == "executed":
            if (
                not attempt.executed
                or attempt.tool_name not in TOOL_NAMES
                or set(attempt.arguments) != {"evidence_id"}
                or attempt.arguments.get("evidence_id") != attempt.evidence_id
            ):
                raise ValueError(f"group {group.group_id} executed tool attempt is inconsistent")
            executed.append(attempt)
        elif attempt.disposition == "duplicate":
            if (
                attempt.executed
                or attempt.result_status != "duplicate"
                or attempt.error_code != "duplicate_tool_request"
                or attempt.data
                or signature not in prior_signatures
            ):
                raise ValueError(f"group {group.group_id} duplicate tool attempt is inconsistent")
        elif attempt.disposition == "prevalidation_rejected":
            if (
                attempt.executed
                or attempt.result_status != "invalid"
                or not isinstance(attempt.error_code, str)
                or not attempt.error_code
                or not isinstance(attempt.error_message, str)
                or not attempt.error_message
            ):
                raise ValueError(
                    f"group {group.group_id} prevalidation tool attempt is inconsistent"
                )
        else:
            raise ValueError(f"group {group.group_id} tool attempt disposition is invalid")
        prior_signatures.add(signature)

        next_index = trace_event["sequence"]
        if next_index >= len(events):
            raise ValueError(f"group {group.group_id} tool attempt lacks a trace outcome")
        outcome = events[next_index]
        if attempt.executed:
            expected_type = "tool_result"
            expected_category = "tool"
            expected_data = {
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
                "model_turn": attempt.model_turn,
                "status": attempt.result_status,
                "error_code": attempt.error_code,
                "data": attempt.data,
            }
        else:
            expected_type = "validation_feedback"
            expected_category = "validation"
            expected_data = {
                "model_turn": attempt.model_turn,
                "code": attempt.error_code,
                "details": attempt.data,
            }
        if not _exact_trace_event(
            outcome,
            category=expected_category,
            event_type=expected_type,
            data=expected_data,
        ):
            raise ValueError(f"group {group.group_id} tool attempt outcome trace is inconsistent")

    stored_results = result.tool_results
    if not isinstance(stored_results, tuple) or len(stored_results) != len(executed):
        raise ValueError(
            f"group {group.group_id} executed attempts and tool results must be one-to-one"
        )
    result_events = tuple(event for event in events if event["event_type"] == "tool_result")
    if len(result_events) != len(executed):
        raise ValueError(
            f"group {group.group_id} executed attempts and trace tool results must be one-to-one"
        )
    registry = ToolRegistry(catalog)
    for attempt, stored, trace_event in zip(executed, stored_results, result_events):
        if not isinstance(stored, ToolResult):
            raise ValueError(f"group {group.group_id} contains an invalid tool result")
        if (
            stored.tool_name != attempt.tool_name
            or stored.evidence_id != attempt.evidence_id
            or stored.status != attempt.result_status
            or stored.error_code != attempt.error_code
            or stored.error_message != attempt.error_message
            or not _same_json(stored.data, attempt.data)
        ):
            raise ValueError(f"group {group.group_id} executed attempt and tool result differ")
        expected_trace_data = {
            "attempt_id": attempt.attempt_id,
            "attempt_index": attempt.attempt_index,
            "model_turn": attempt.model_turn,
            "status": stored.status,
            "error_code": stored.error_code,
            "data": stored.data,
        }
        if not _exact_trace_event(
            trace_event,
            category="tool",
            event_type="tool_result",
            data=expected_trace_data,
        ):
            raise ValueError(f"group {group.group_id} stored tool result trace is inconsistent")
        recomputed = registry.invoke(attempt.tool_name, attempt.arguments)
        if not isinstance(recomputed, ToolResult) or not _same_json(
            recomputed.to_dict(),
            stored.to_dict(),
        ):
            raise ValueError(
                f"group {group.group_id} tool result differs from the re-executed registry result"
            )


def _validate_proposal_and_stop(
    result: GroupRunResult,
    *,
    group: Any,
    item_by_task: Mapping[str, Any],
    catalog: EvidenceCatalog,
    events: tuple[Mapping[str, Any], ...],
) -> None:
    stop_reasons = {
        "final_proposal",
        "missing_replay",
        "replay_exhausted",
        "model_error",
        "tool_budget_exhausted",
        "model_turn_limit",
    }
    if result.stop_reason not in stop_reasons:
        raise ValueError(f"group {group.group_id} stop reason is invalid")
    if not isinstance(result.proposal, GroupFinalProposal):
        raise ValueError(f"group {group.group_id} proposal has an invalid type")
    expected_items = {task_id: item_by_task[task_id] for task_id in group.task_ids}
    if result.stop_reason == "final_proposal":
        try:
            parsed = parse_model_action(
                result.proposal.to_action(),
                group_id=group.group_id,
                expected_items=expected_items,
            )
        except (ActionValidationError, TypeError, ValueError) as exc:
            raise ValueError(f"group {group.group_id} proposal schema is invalid") from exc
        if not isinstance(parsed, GroupFinalProposal) or not _same_json(
            parsed.to_dict(),
            result.proposal.to_dict(),
        ):
            raise ValueError(f"group {group.group_id} proposal is not canonical")
    else:
        expected_fallback = make_fallback_proposal(
            group,
            expected_items,
            result.stop_reason,
        )
        if not _same_json(expected_fallback.to_dict(), result.proposal.to_dict()):
            raise ValueError(f"group {group.group_id} fallback proposal is inconsistent")

    for assessment in result.assessments:
        for evidence_id in assessment.evidence_refs:
            try:
                record = catalog.get(evidence_id)
            except KeyError as exc:
                raise ValueError(
                    f"group {group.group_id} proposal references unknown evidence"
                ) from exc
            if record.task_id != assessment.task_id or record.task_id not in expected_items:
                raise ValueError(
                    f"group {group.group_id} proposal evidence belongs to another task"
                )

    accepted = tuple(
        event for event in events if event["event_type"] == "final_proposal_accepted"
    )
    if result.stop_reason == "final_proposal":
        expected_data = {
            "model_turn": result.model_turns,
            "assessment_task_ids": [row.task_id for row in result.assessments],
            "proposal": result.proposal.to_dict(),
        }
        if len(accepted) != 1 or not _exact_trace_event(
            accepted[0],
            category="validation",
            event_type="final_proposal_accepted",
            data=expected_data,
        ):
            raise ValueError(f"group {group.group_id} final accepted trace is inconsistent")
        if accepted[0] is not events[-3]:
            raise ValueError(f"group {group.group_id} final accepted trace is out of order")
        prior = events[-4]
        if (
            prior["category"] != "model"
            or prior["event_type"] != "model_action_received"
            or prior["data"].get("model_turn") != result.model_turns
            or prior["data"].get("action_type") != "final_proposal"
        ):
            raise ValueError(f"group {group.group_id} final action trace is inconsistent")
    elif accepted:
        raise ValueError(f"group {group.group_id} fallback run has a final accepted trace")

    terminal_markers = {
        "missing_replay": ("model", "replay_missing"),
        "replay_exhausted": ("model", "replay_exhausted"),
        "model_error": ("model", "model_error"),
    }
    if result.stop_reason in terminal_markers:
        category, event_type = terminal_markers[result.stop_reason]
        marker = events[-3]
        if (
            marker["category"] != category
            or marker["event_type"] != event_type
            or marker["data"].get("model_turn") != result.model_turns
        ):
            raise ValueError(f"group {group.group_id} stop marker trace is inconsistent")
        prior = events[-4]
        if (
            prior["category"] != "model"
            or prior["event_type"] != "model_turn_started"
            or prior["data"].get("model_turn") != result.model_turns
        ):
            raise ValueError(f"group {group.group_id} stop marker order is inconsistent")
    elif result.stop_reason == "tool_budget_exhausted":
        marker = events[-3]
        if (
            marker["category"] != "validation"
            or marker["event_type"] != "validation_feedback"
            or marker["data"].get("model_turn") != result.model_turns
            or marker["data"].get("code") != "tool_budget_exhausted"
            or result.tool_attempts != MAX_TOOL_ATTEMPTS
        ):
            raise ValueError(f"group {group.group_id} tool budget stop trace is inconsistent")
        prior = events[-4]
        if (
            prior["category"] != "model"
            or prior["event_type"] != "model_action_received"
            or prior["data"].get("model_turn") != result.model_turns
            or prior["data"].get("action_type") != "tool_request"
        ):
            raise ValueError(f"group {group.group_id} tool budget action trace is inconsistent")
    elif result.stop_reason == "model_turn_limit" and result.model_turns != MAX_MODEL_TURNS:
        raise ValueError(f"group {group.group_id} model turn limit is inconsistent")


def _validated_group_results(
    queue: QueueEnvelope,
    group_results: Sequence[GroupRunResult],
    catalog: EvidenceCatalog,
) -> tuple[GroupRunResult, ...]:
    expected_groups = build_groups(queue)
    expected_by_id = {group.group_id: group for group in expected_groups}
    if len(expected_by_id) != len(expected_groups):
        raise ValueError("duplicate expected group ID")
    supplied_by_id: dict[str, GroupRunResult] = {}
    for result in group_results:
        if not isinstance(result, GroupRunResult):
            raise ValueError("group results contain an invalid record")
        if not isinstance(result.group_id, str) or not result.group_id:
            raise ValueError("group result ID must be a non-empty string")
        if result.group_id in supplied_by_id:
            raise ValueError(f"duplicate group result: {result.group_id}")
        supplied_by_id[result.group_id] = result
    missing = sorted(set(expected_by_id) - set(supplied_by_id))
    extra = sorted(set(supplied_by_id) - set(expected_by_id))
    if missing:
        raise ValueError(f"missing group result(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"extra group result(s): {', '.join(extra)}")

    item_by_task = {item.task_id: item for item in queue.items}
    for group_id, group in expected_by_id.items():
        result = supplied_by_id[group_id]
        if (
            isinstance(result.model_turns, bool)
            or not isinstance(result.model_turns, int)
            or not 1 <= result.model_turns <= MAX_MODEL_TURNS
        ):
            raise ValueError(f"group {group_id} model turn counter is outside the budget")
        if (
            isinstance(result.tool_attempts, bool)
            or not isinstance(result.tool_attempts, int)
            or not 0 <= result.tool_attempts <= MAX_TOOL_ATTEMPTS
        ):
            raise ValueError(f"group {group_id} tool attempt counter is outside the budget")
        if not isinstance(result.proposal, GroupFinalProposal):
            raise ValueError(f"group {group_id} proposal has an invalid type")
        if not isinstance(result.stop_reason, str) or not result.stop_reason:
            raise ValueError(f"group {group_id} stop reason is invalid")
        if result.proposal.group_id != group_id:
            raise ValueError(f"group proposal ID mismatch for {group_id}")
        if any(not isinstance(assessment, SlotAssessment) for assessment in result.assessments):
            raise ValueError(f"group {group_id} proposal contains an invalid assessment")
        assessment_ids = [assessment.task_id for assessment in result.assessments]
        if any(not isinstance(task_id, str) for task_id in assessment_ids):
            raise ValueError(f"group {group_id} assessment task ID is invalid")
        if len(set(assessment_ids)) != len(assessment_ids):
            raise ValueError(f"duplicate assessment in group result {group_id}")
        expected_tasks = set(group.task_ids)
        supplied_tasks = set(assessment_ids)
        missing_tasks = sorted(expected_tasks - supplied_tasks)
        extra_tasks = sorted(supplied_tasks - expected_tasks)
        if missing_tasks:
            raise ValueError(
                f"missing assessment task(s) in group {group_id}: {', '.join(missing_tasks)}"
            )
        if extra_tasks:
            raise ValueError(
                f"extra assessment task(s) in group {group_id}: {', '.join(extra_tasks)}"
            )
        for assessment in result.assessments:
            if assessment.slot_id != item_by_task[assessment.task_id].slot_id:
                raise ValueError(f"assessment slot mismatch in group {group_id}")
        events = _validate_trace_integrity(result, group=group)
        _validate_attempts_and_results(
            result,
            group=group,
            catalog=catalog,
            events=events,
        )
        _validate_proposal_and_stop(
            result,
            group=group,
            item_by_task=item_by_task,
            catalog=catalog,
            events=events,
        )
    return tuple(supplied_by_id[group.group_id] for group in expected_groups)


def _validate_final_route_states(
    final_route_states: Mapping[str, Any],
    *,
    part2_run_id: str,
    queue: QueueEnvelope,
    resolutions: Sequence[SlotResolution],
) -> None:
    if not isinstance(final_route_states, Mapping):
        raise ValueError("final route states must be a JSON object")
    expected_fields = {
        "schema_version",
        "part2_run_id",
        "queue_id",
        "base_decisions",
        "decisions",
    }
    if set(final_route_states) != expected_fields:
        raise ValueError("final route states fields do not match the strict schema")
    if final_route_states["schema_version"] != FINAL_ROUTE_STATES_SCHEMA_VERSION:
        raise ValueError("final route states schema_version is invalid")
    if final_route_states["part2_run_id"] != part2_run_id:
        raise ValueError("final route states part2_run_id does not match the run")
    if final_route_states["queue_id"] != queue.queue_id:
        raise ValueError("final route states queue identity does not match the run")
    if not isinstance(final_route_states["base_decisions"], Mapping):
        raise ValueError("final route states base_decisions must be an object")
    rows = final_route_states["decisions"]
    if not isinstance(rows, (list, tuple)):
        raise ValueError("final route states decisions must be an array")

    row_by_slot: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"final decision row {index} must be an object")
        slot_id = row.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            raise ValueError(f"final decision row {index} requires slot_id")
        if slot_id in row_by_slot:
            raise ValueError(f"duplicate final decision slot: {slot_id}")
        row_by_slot[slot_id] = row

    resolution_by_slot = {resolution.slot_id: resolution for resolution in resolutions}
    for slot_id, row in row_by_slot.items():
        if "part2_resolution" in row and slot_id not in resolution_by_slot:
            raise ValueError(f"extra final Part2 decision row: {slot_id}")
    for resolution in resolutions:
        row = row_by_slot.get(resolution.slot_id)
        if row is None:
            raise ValueError(f"missing queued final decision row: {resolution.slot_id}")
        if row.get("scope_status") != resolution.scope_status:
            raise ValueError(f"queued final decision scope mismatch: {resolution.slot_id}")
        if row.get("state") != resolution.state:
            raise ValueError(f"queued final decision state mismatch: {resolution.slot_id}")
        audit = row.get("part2_resolution")
        if not isinstance(audit, Mapping):
            raise ValueError(f"queued final decision lacks part2_resolution: {resolution.slot_id}")
        if canonical_json_bytes(audit) != canonical_json_bytes(resolution.to_dict()):
            raise ValueError(f"queued final part2_resolution mismatch: {resolution.slot_id}")


def _canonical_attempt_contents(
    part2_run_id: str,
    group_results: Sequence[GroupRunResult],
) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (relative_path, canonical_json_bytes(payload) + b"\n")
        for relative_path, payload in _attempt_payloads(part2_run_id, group_results)
    )


def write_run_outputs(
    output_dir: str | Path,
    *,
    part2_run_id: str,
    queue: QueueEnvelope,
    catalog: EvidenceCatalog,
    base_dir: str | Path,
    replay_identity: str,
    group_results: Sequence[GroupRunResult],
    resolutions: Sequence[SlotResolution],
    final_route_states: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write the canonical run artifact set and a non-self-hashing manifest."""

    if not isinstance(part2_run_id, str) or SHA256_PATTERN.fullmatch(part2_run_id) is None:
        raise ValueError("part2_run_id must be a canonical lowercase sha256 identity")
    expected_run_id = _derive_part2_run_id_from_replay_identity(queue, replay_identity)
    if part2_run_id != expected_run_id:
        raise ValueError(
            "part2_run_id run identity does not match the queue, replay identity, "
            "and current policies"
        )
    if not isinstance(catalog, EvidenceCatalog) or catalog.queue is not queue:
        raise ValueError("evidence catalog does not belong to this queue")
    stable_resolutions = _validated_resolutions(queue, resolutions)
    stable_results = _validated_group_results(queue, group_results, catalog)
    attempt_contents = _canonical_attempt_contents(part2_run_id, stable_results)
    recomputed_resolutions = consolidate_group_assessments(
        queue,
        stable_results,
        catalog,
    )
    if canonical_json_bytes(
        [resolution.to_dict() for resolution in stable_resolutions]
    ) != canonical_json_bytes(
        [resolution.to_dict() for resolution in recomputed_resolutions]
    ):
        raise ValueError("supplied resolutions do not match the recomputed decision gate")
    _validate_final_route_states(
        final_route_states,
        part2_run_id=part2_run_id,
        queue=queue,
        resolutions=stable_resolutions,
    )
    fresh_base = load_base_slot_decisions(queue, base_dir=base_dir)
    recomputed_final = merge_final_route_states(
        fresh_base,
        queue,
        recomputed_resolutions,
        part2_run_id=part2_run_id,
    )
    if canonical_json_bytes(final_route_states) != canonical_json_bytes(recomputed_final):
        raise ValueError("full final route states do not match freshly merged base")

    resolution_payload = _resolution_payload(
        part2_run_id,
        queue,
        stable_resolutions,
    )
    summary = _summary(part2_run_id, queue, stable_results, stable_resolutions)
    trace = "".join(result.trace_jsonl for result in stable_results).encode("utf-8")
    main_contents = {
        "part2_resolutions.json": canonical_json_bytes(resolution_payload) + b"\n",
        "final_route_states.json": canonical_json_bytes(final_route_states) + b"\n",
        "decision_trace.jsonl": trace,
        "summary.json": canonical_json_bytes(summary) + b"\n",
        "report.html": _report_html(summary, stable_resolutions).encode("utf-8"),
    }
    # No output path is touched until every cross-input and canonicalization
    # check above has succeeded.
    destination = Path(output_dir)
    tool_dir = _prepare_output_dir(destination)
    for name, content in main_contents.items():
        _atomic_write(destination / name, content)

    for relative_path, content in attempt_contents:
        _atomic_write(tool_dir / Path(relative_path).name, content)

    artifact_paths = [destination / name for name in main_contents]
    artifact_paths.extend(sorted(tool_dir.glob("*.json")))
    artifacts = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(
            artifact_paths,
            key=lambda path: path.relative_to(destination).as_posix(),
        )
    ]
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "part2_run_id": part2_run_id,
        "inputs": {
            "queue_identity": queue.queue_id,
            "replay_identity": replay_identity,
        },
        "policies": _policy_versions(queue),
        "artifacts": artifacts,
    }
    _write_json(destination / "run_manifest.json", manifest)
    return manifest


__all__ = [
    "ORCHESTRATOR_POLICY_VERSION",
    "REPORTING_POLICY_VERSION",
    "RUN_IDENTITY_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "TOOL_ATTEMPT_SCHEMA_VERSION",
    "derive_part2_run_id",
    "load_base_slot_decisions",
    "write_run_outputs",
]
