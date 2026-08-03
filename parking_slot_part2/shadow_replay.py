"""Compile blind shadow judgements into bounded, deterministic Replay actions.

The compiler is deliberately smaller than a live-model adapter.  It accepts
only one strict, path-free semantic record per queue slot, binds terminal
claims to catalog-issued target evidence, and emits the existing
``part2-replay-actions/1.0`` wire format.  It never reads labels or provider
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import QueueEnvelope, QueueItem
from .grouping import TaskGroup, build_groups
from .model import (
    FINAL_STATES,
    REPLAY_SCHEMA_VERSION,
    SEMANTIC_FINDINGS,
    TARGET_OWNERSHIPS,
    TARGET_VISIBILITIES,
    GroupFinalProposal,
    ReplayModelAdapter,
    SlotAssessment,
    ToolRequest,
    parse_model_action,
)
from .orchestrator import MAX_MODEL_TURNS, MAX_TOOL_ATTEMPTS
from .preflight import EvidenceCatalog, EvidenceRecord
from .queueing import canonical_json_bytes
from .tools import ToolRegistry


SHADOW_REPLAY_POLICY_VERSION = "part2-shadow-replay-compiler/2.0"

_TERMINAL_STATES = frozenset({"occupied", "free"})
_REQUIRED_BLIND_FIELDS = frozenset(
    {"state", "visibility", "ownership", "finding", "reason_codes"}
)
_OPTIONAL_BLIND_FIELDS = frozenset({"confidence"})
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")

# These fields must not cross the blind-inference boundary, even if they are
# hidden under an otherwise unsupported metadata object.  Exact record-field
# validation runs after this recursive check so the failure mode remains
# explicit and auditable.
_PROVIDER_FIELD_COMPACT_NAMES = frozenset(
    {
        "provider",
        "providerid",
        "providername",
        "model",
        "modelid",
        "modelname",
        "apikey",
        "baseurl",
        "endpoint",
        "messages",
        "prompt",
        "completion",
        "responseid",
    }
)
_GT_FIELD_COMPACT_NAMES = frozenset(
    {
        "gt",
        "gtlabel",
        "gtstate",
        "groundtruth",
        "groundtruthlabel",
        "humanlabel",
        "annotation",
        "annotationsource",
    }
)


@dataclass(frozen=True, slots=True)
class BlindSlotJudgement:
    """Validated semantic input for one slot; confidence is scheduling-only."""

    state: str
    visibility: str
    ownership: str
    finding: str
    reason_codes: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class _PreparedDecision:
    item: QueueItem
    judgement: BlindSlotJudgement
    state: str
    evidence: EvidenceRecord | None
    compiler_reason_codes: tuple[str, ...] = ()


def _compact_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _forbidden_field_category(key: str) -> str | None:
    compact = _compact_field_name(key)
    if "path" in compact or compact.endswith(("uri", "url", "file", "filename")):
        return "path"
    if (
        compact in _GT_FIELD_COMPACT_NAMES
        or compact.startswith("gt")
        or compact.endswith("gt")
        or "groundtruth" in compact
        or compact in {"label", "truth", "truthlabel"}
    ):
        return "ground_truth"
    if (
        compact in _PROVIDER_FIELD_COMPACT_NAMES
        or "provider" in compact
        or compact.startswith(("model", "prompt", "message", "completion"))
    ):
        return "provider"
    return None


def _reject_forbidden_fields(value: Any, location: str = "blind_records") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} keys must be strings")
            category = _forbidden_field_category(key)
            if category is not None:
                raise ValueError(
                    f"{location} contains forbidden {category} field {key!r}"
                )
            _reject_forbidden_fields(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{location}[{index}]")


def _reason_codes(value: Any, *, slot_id: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"blind record for {slot_id} reason_codes must be an array")
    result: list[str] = []
    for code in value:
        if not isinstance(code, str) or _CODE.fullmatch(code) is None:
            raise ValueError(
                f"blind record for {slot_id} reason_codes must contain lowercase codes"
            )
        result.append(code)
    if len(set(result)) != len(result):
        raise ValueError(f"blind record for {slot_id} reason_codes must be unique")
    return tuple(result)


def _parse_blind_record(slot_id: str, value: Any) -> BlindSlotJudgement:
    if not isinstance(value, Mapping):
        raise ValueError(f"blind record for {slot_id} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"blind record for {slot_id} keys must be strings")
    fields = set(value)
    allowed = _REQUIRED_BLIND_FIELDS | _OPTIONAL_BLIND_FIELDS
    if not _REQUIRED_BLIND_FIELDS <= fields or not fields <= allowed:
        missing = sorted(_REQUIRED_BLIND_FIELDS - fields)
        unsupported = sorted(fields - allowed)
        raise ValueError(
            f"blind record for {slot_id} has invalid fields; "
            f"missing={missing}, unsupported={unsupported}"
        )

    state = value["state"]
    visibility = value["visibility"]
    ownership = value["ownership"]
    finding = value["finding"]
    if not isinstance(state, str) or state not in FINAL_STATES:
        raise ValueError(f"blind record for {slot_id} has invalid state")
    if not isinstance(visibility, str) or visibility not in TARGET_VISIBILITIES:
        raise ValueError(f"blind record for {slot_id} has invalid visibility")
    if not isinstance(ownership, str) or ownership not in TARGET_OWNERSHIPS:
        raise ValueError(f"blind record for {slot_id} has invalid ownership")
    if not isinstance(finding, str) or finding not in SEMANTIC_FINDINGS:
        raise ValueError(f"blind record for {slot_id} has invalid finding")

    confidence = value.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"blind record for {slot_id} confidence must be numeric")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError(f"blind record for {slot_id} confidence must be within [0, 1]")

    return BlindSlotJudgement(
        state=state,
        visibility=visibility,
        ownership=ownership,
        finding=finding,
        reason_codes=_reason_codes(value["reason_codes"], slot_id=slot_id),
        confidence=confidence_value,
    )


def _parse_blind_records(
    queue: QueueEnvelope,
    blind_records: Any,
) -> dict[str, BlindSlotJudgement]:
    if not isinstance(blind_records, Mapping):
        raise ValueError("blind_records must be an object keyed by slot_id")
    if any(not isinstance(key, str) or not key for key in blind_records):
        raise ValueError("blind_records keys must be non-empty slot_id strings")

    expected = {item.slot_id for item in queue.items}
    supplied = set(blind_records)
    # Top-level keys normally are slot identities, not fields.  Known slot IDs
    # are therefore exempt from field-name inspection, while any injected
    # extra key is still checked before exact-coverage validation.
    for key, value in blind_records.items():
        if key not in expected:
            category = _forbidden_field_category(key)
            if category is not None:
                raise ValueError(
                    f"blind_records contains forbidden {category} field {key!r}"
                )
        _reject_forbidden_fields(value, f"blind_records.{key}")

    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise ValueError(
            "blind_records must contain exactly one record per queue slot; "
            f"missing={missing}, extra={extra}"
        )
    parsed: dict[str, BlindSlotJudgement] = {}
    for slot_id in sorted(supplied):
        raw_record = blind_records[slot_id]
        parsed[slot_id] = _parse_blind_record(slot_id, raw_record)
    return parsed


def _task_evidence(catalog: EvidenceCatalog, task_id: str) -> tuple[EvidenceRecord, ...]:
    return tuple(
        sorted(
            (
                record
                for record in catalog.entries.values()
                if record.task_id == task_id and record.available
            ),
            key=lambda record: (record.tool_name, record.evidence_id),
        )
    )


def _select_terminal_evidence(
    item: QueueItem,
    judgement: BlindSlotJudgement,
    catalog: EvidenceCatalog,
    usable: Callable[[EvidenceRecord], bool],
) -> EvidenceRecord | None:
    records = _task_evidence(catalog, item.task_id)
    if judgement.state == "free":
        candidates = (
            record
            for record in records
            if record.tool_name == "inspect_rgb_frame"
            and "can_assess_free" in record.capabilities
            and usable(record)
        )
        return next(candidates, None)

    if judgement.state != "occupied":
        return None

    # LiDAR is the preferred occupied support because it does not add the
    # clear_full camera requirement.  An eligible single RGB frame is the only
    # camera fallback accepted by the deterministic terminal gate.
    lidar = next(
        (
            record
            for record in records
            if record.tool_name == "inspect_lidar_map" and usable(record)
        ),
        None,
    )
    if lidar is not None:
        return lidar
    if judgement.visibility != "clear_full":
        return None
    return next(
        (
            record
            for record in records
            if record.tool_name == "inspect_rgb_frame"
            and "can_assess_occupied" in record.capabilities
            and usable(record)
        ),
        None,
    )


def _prepare_decisions(
    queue: QueueEnvelope,
    judgements: Mapping[str, BlindSlotJudgement],
    catalog: EvidenceCatalog,
    usable: Callable[[EvidenceRecord], bool],
) -> dict[str, _PreparedDecision]:
    prepared: dict[str, _PreparedDecision] = {}
    for item in sorted(queue.items, key=lambda row: row.task_id):
        if "unknown" not in item.allowed_final_states:
            raise ValueError(
                f"queue task {item.task_id} must allow unknown for fail-closed shadow replay"
            )
        judgement = judgements[item.slot_id]
        state = judgement.state
        evidence: EvidenceRecord | None = None
        compiler_reasons: list[str] = []
        if state in _TERMINAL_STATES and state not in item.allowed_final_states:
            state = "unknown"
            compiler_reasons.append("shadow_state_not_allowed")
        elif state in _TERMINAL_STATES:
            evidence = _select_terminal_evidence(item, judgement, catalog, usable)
            if evidence is None:
                state = "unknown"
                compiler_reasons.append(
                    "shadow_free_rgb_evidence_unavailable"
                    if judgement.state == "free"
                    else "shadow_terminal_evidence_unavailable"
                )
        prepared[item.task_id] = _PreparedDecision(
            item=item,
            judgement=judgement,
            state=state,
            evidence=evidence,
            compiler_reason_codes=tuple(compiler_reasons),
        )
    return prepared


def _apply_global_tool_budget(
    groups: tuple[TaskGroup, ...],
    prepared: dict[str, _PreparedDecision],
) -> None:
    """Globally freeze low-confidence terminals until every group fits."""

    while True:
        violating = tuple(
            group
            for group in groups
            if sum(
                prepared[task_id].state in _TERMINAL_STATES
                for task_id in group.task_ids
            )
            > MAX_TOOL_ATTEMPTS
        )
        if not violating:
            return
        candidates = {
            task_id
            for group in violating
            for task_id in group.task_ids
            if prepared[task_id].state in _TERMINAL_STATES
        }
        loser_id = min(
            candidates,
            key=lambda task_id: (
                prepared[task_id].judgement.confidence,
                prepared[task_id].item.slot_id,
                task_id,
            ),
        )
        loser = prepared[loser_id]
        prepared[loser_id] = replace(
            loser,
            state="unknown",
            evidence=None,
            compiler_reason_codes=tuple(
                sorted(
                    set(loser.compiler_reason_codes)
                    | {"shadow_tool_budget_downgrade"}
                )
            ),
        )


def _assessment(decision: _PreparedDecision) -> SlotAssessment:
    item = decision.item
    judgement = decision.judgement
    reason_codes = tuple(
        sorted(set(judgement.reason_codes) | set(decision.compiler_reason_codes))
    )
    if decision.state == "unknown":
        if not reason_codes:
            reason_codes = ("insufficient_evidence",)
        return SlotAssessment(
            task_id=item.task_id,
            slot_id=item.slot_id,
            proposed_state="unknown",
            target_visibility="unknown",
            target_ownership="uncertain",
            semantic_finding="unclear",
            resolved_unknown_reasons=(),
            unresolved_blockers=item.unknown_reasons,
            evidence_refs=(),
            reason_codes=reason_codes,
        )

    if decision.evidence is None:  # pragma: no cover - guarded by preparation
        raise AssertionError("terminal shadow decision has no target evidence")
    return SlotAssessment(
        task_id=item.task_id,
        slot_id=item.slot_id,
        proposed_state=decision.state,
        target_visibility=judgement.visibility,
        target_ownership=judgement.ownership,
        semantic_finding=judgement.finding,
        resolved_unknown_reasons=(
            item.unknown_reasons if decision.state == "free" else ()
        ),
        unresolved_blockers=(),
        evidence_refs=(decision.evidence.evidence_id,),
        reason_codes=reason_codes,
    )


def _compile_group_actions(
    group: TaskGroup,
    prepared: Mapping[str, _PreparedDecision],
) -> list[dict[str, Any]]:
    terminal = tuple(
        prepared[task_id]
        for task_id in group.task_ids
        if prepared[task_id].state in _TERMINAL_STATES
    )
    if len(terminal) > MAX_TOOL_ATTEMPTS:  # pragma: no cover - budget invariant
        raise AssertionError("shadow group exceeds the Part2 tool budget")

    actions = [
        ToolRequest(
            decision.evidence.tool_name,
            {"evidence_id": decision.evidence.evidence_id},
        ).to_action()
        for decision in terminal
        if decision.evidence is not None
    ]
    proposal = GroupFinalProposal(
        group_id=group.group_id,
        assessments=tuple(_assessment(prepared[task_id]) for task_id in group.task_ids),
    )
    actions.append(proposal.to_action())
    if len(actions) > MAX_MODEL_TURNS:  # pragma: no cover - budget invariant
        raise AssertionError("shadow group exceeds the Part2 model-turn budget")
    return actions


def _validate_compiled_replay(
    payload: Mapping[str, Any],
    queue: QueueEnvelope,
    catalog: EvidenceCatalog,
    groups: tuple[TaskGroup, ...],
    usable: Callable[[EvidenceRecord], bool],
) -> None:
    # ReplayModelAdapter applies the strict envelope/live-provider checks and
    # canonical finite-JSON conversion used by production replay.
    ReplayModelAdapter(payload)
    items = {item.task_id: item for item in queue.items}
    actions_by_group = payload["actions"]
    if set(actions_by_group) != {group.group_id for group in groups}:
        raise AssertionError("compiled replay does not cover every decision group")

    for group in groups:
        expected_items = {task_id: items[task_id] for task_id in group.task_ids}
        actions = actions_by_group[group.group_id]
        tool_count = 0
        for index, raw_action in enumerate(actions):
            parsed = parse_model_action(
                raw_action,
                group_id=group.group_id,
                expected_items=expected_items,
            )
            if isinstance(parsed, ToolRequest):
                tool_count += 1
                record = catalog.get(parsed.evidence_id)
                if (
                    record.task_id not in expected_items
                    or record.tool_name != parsed.tool_name
                    or not record.available
                    or not usable(record)
                ):
                    raise AssertionError("compiled tool request is not usable target evidence")
            elif index != len(actions) - 1:
                raise AssertionError("final proposal must be the last replay action")
        if tool_count > MAX_TOOL_ATTEMPTS or len(actions) > MAX_MODEL_TURNS:
            raise AssertionError("compiled replay exceeds an orchestrator budget")


def compile_shadow_replay(
    queue: QueueEnvelope,
    blind_records: Mapping[str, Mapping[str, Any]],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compile one blind judgement per queue slot into strict Replay actions.

    Missing confidence defaults to ``0.0`` and is used only when a group has
    more terminal requests than the three-tool budget.  A task downgraded for
    budget remains unknown in every overlapping group.
    """

    if not isinstance(queue, QueueEnvelope):
        raise TypeError("queue must be a validated QueueEnvelope")
    judgements = _parse_blind_records(queue, blind_records)
    catalog = EvidenceCatalog(queue, base_dir=base_dir)
    registry = ToolRegistry(catalog)
    usability_cache: dict[str, bool] = {}

    def usable(record: EvidenceRecord) -> bool:
        cached = usability_cache.get(record.evidence_id)
        if cached is None:
            cached = registry.execute(
                record.tool_name,
                {"evidence_id": record.evidence_id},
            ).ok
            usability_cache[record.evidence_id] = cached
        return cached

    groups = build_groups(queue)
    prepared = _prepare_decisions(queue, judgements, catalog, usable)
    _apply_global_tool_budget(groups, prepared)

    payload: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "actions": {
            group.group_id: _compile_group_actions(group, prepared)
            for group in groups
        },
    }
    # Assert JSON compatibility before returning a mutable, provider-neutral
    # payload to the caller.
    canonical_json_bytes(payload)
    _validate_compiled_replay(payload, queue, catalog, groups, usable)
    return payload


__all__ = [
    "SHADOW_REPLAY_POLICY_VERSION",
    "BlindSlotJudgement",
    "compile_shadow_replay",
]
