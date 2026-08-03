"""Provider-neutral structured model actions and deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import QueueItem, freeze_json
from .preflight import OPAQUE_EVIDENCE_ID
from .queueing import canonical_json_bytes
from .tools import TOOL_NAMES


REPLAY_SCHEMA_VERSION = "part2-replay-actions/1.0"
MODEL_TURN_SCHEMA_VERSION = "part2-model-turn/1.0"
ACTION_TYPES = ("tool_request", "final_proposal")
FINAL_STATES = ("occupied", "free", "unknown")
TARGET_VISIBILITIES = (
    "clear_full",
    "clear_partial",
    "occluded",
    "unavailable",
    "unknown",
)
TARGET_OWNERSHIPS = (
    "target",
    "adjacent",
    "shared",
    "uncertain",
    "not_applicable",
)
SEMANTIC_FINDINGS = (
    "vehicle_or_occupying_object",
    "empty",
    "static_structure",
    "lane_object",
    "occluded",
    "unclear",
)

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_LIVE_PROVIDER_FIELDS = frozenset(
    {
        "provider",
        "model",
        "api_key",
        "base_url",
        "endpoint",
        "messages",
        "prompt",
        "completion",
        "response_id",
    }
)
_ASSESSMENT_FIELDS = frozenset(
    {
        "task_id",
        "slot_id",
        "proposed_state",
        "target_visibility",
        "target_ownership",
        "semantic_finding",
        "resolved_unknown_reasons",
        "unresolved_blockers",
        "evidence_refs",
        "reason_codes",
    }
)


@runtime_checkable
class ModelAdapter(Protocol):
    """Provider-neutral boundary returning one raw JSON action per turn."""

    def next_action(self, request: Mapping[str, Any]) -> Any:
        """Return the next action without exposing provider-specific messages."""


class ReplayError(RuntimeError):
    """Base class for deterministic replay lookup failures."""


class MissingReplayError(ReplayError):
    """The replay file has no action sequence for the requested group."""


class ReplayExhaustedError(ReplayError):
    """The requested group's replay sequence has no remaining action."""


class ActionValidationError(ValueError):
    """A stable validation code suitable for a repair observation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = freeze_json(dict(details or {}))


def _plain_json(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("replay payload must contain only finite JSON values") from exc


def _reject_live_provider_fields(value: Any, path: str = "actions") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.strip().lower() in _LIVE_PROVIDER_FIELDS:
                raise ValueError(f"{path} contains forbidden live-provider field {key}")
            _reject_live_provider_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_live_provider_fields(item, f"{path}[{index}]")


def _exact_fields(
    value: Any,
    expected: frozenset[str],
    *,
    code: str,
    subject: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionValidationError(code, f"{subject} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ActionValidationError(code, f"{subject} keys must be strings")
    fields = set(value)
    if fields != expected:
        raise ActionValidationError(
            code,
            f"{subject} fields do not match the strict v1 schema",
            details={
                "missing_fields": sorted(expected - fields),
                "unsupported_field_count": len(fields - expected),
            },
        )
    return value


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ActionValidationError(
            "invalid_assessment_schema",
            f"{field_name} must be a JSON array",
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or _CODE.fullmatch(item) is None:
            raise ActionValidationError(
                "invalid_assessment_schema",
                f"{field_name} must contain only lowercase identifier codes",
            )
        result.append(item)
    if len(set(result)) != len(result):
        raise ActionValidationError(
            "invalid_assessment_schema",
            f"{field_name} must not contain duplicates",
        )
    return tuple(result)


def _evidence_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ActionValidationError(
            "invalid_assessment_schema",
            "evidence_refs must be a JSON array",
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or OPAQUE_EVIDENCE_ID.fullmatch(item) is None:
            raise ActionValidationError(
                "invalid_evidence_ref",
                "evidence_refs must contain only opaque preflight evidence IDs",
            )
        result.append(item)
    if len(set(result)) != len(result):
        raise ActionValidationError(
            "invalid_evidence_ref",
            "evidence_refs must not contain duplicates",
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """One strictly allowlisted request using an opaque evidence ID."""

    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tool_name not in TOOL_NAMES:
            raise ValueError("tool_name is not allowlisted")
        arguments = dict(self.arguments)
        if set(arguments) != {"evidence_id"}:
            raise ValueError("tool arguments must contain exactly evidence_id")
        evidence_id = arguments["evidence_id"]
        if not isinstance(evidence_id, str) or OPAQUE_EVIDENCE_ID.fullmatch(evidence_id) is None:
            raise ValueError("evidence_id must be an opaque preflight ID")
        object.__setattr__(self, "arguments", freeze_json(arguments))

    @property
    def evidence_id(self) -> str:
        return self.arguments["evidence_id"]

    def to_action(self) -> dict[str, Any]:
        return {
            "type": "tool_request",
            "tool_name": self.tool_name,
            "arguments": {"evidence_id": self.evidence_id},
        }


@dataclass(frozen=True, slots=True)
class SlotAssessment:
    """A model proposal record; Task 4, not the model, judges its semantics."""

    task_id: str
    slot_id: str
    proposed_state: str
    target_visibility: str
    target_ownership: str
    semantic_finding: str
    resolved_unknown_reasons: tuple[str, ...]
    unresolved_blockers: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "slot_id": self.slot_id,
            "proposed_state": self.proposed_state,
            "target_visibility": self.target_visibility,
            "target_ownership": self.target_ownership,
            "semantic_finding": self.semantic_finding,
            "resolved_unknown_reasons": list(self.resolved_unknown_reasons),
            "unresolved_blockers": list(self.unresolved_blockers),
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class GroupFinalProposal:
    """A complete, task-ordered proposal for one decision group."""

    group_id: str
    assessments: tuple[SlotAssessment, ...]

    def to_action(self) -> dict[str, Any]:
        return {
            "type": "final_proposal",
            "assessments": [assessment.to_dict() for assessment in self.assessments],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "assessments": [assessment.to_dict() for assessment in self.assessments],
        }


def _parse_tool_request(payload: Mapping[str, Any]) -> ToolRequest:
    action = _exact_fields(
        payload,
        frozenset({"type", "tool_name", "arguments"}),
        code="invalid_tool_request_schema",
        subject="tool request",
    )
    tool_name = action["tool_name"]
    if not isinstance(tool_name, str) or tool_name not in TOOL_NAMES:
        raise ActionValidationError(
            "tool_not_allowlisted",
            "tool_name is not registered in Part2 v1",
        )
    arguments = _exact_fields(
        action["arguments"],
        frozenset({"evidence_id"}),
        code="invalid_tool_arguments",
        subject="tool arguments",
    )
    evidence_id = arguments["evidence_id"]
    if not isinstance(evidence_id, str) or OPAQUE_EVIDENCE_ID.fullmatch(evidence_id) is None:
        raise ActionValidationError(
            "invalid_evidence_id",
            "evidence_id must be an opaque preflight ID",
        )
    return ToolRequest(tool_name, {"evidence_id": evidence_id})


def _parse_assessment(
    payload: Any,
    *,
    expected_items: Mapping[str, QueueItem],
) -> SlotAssessment:
    row = _exact_fields(
        payload,
        _ASSESSMENT_FIELDS,
        code="invalid_assessment_schema",
        subject="slot assessment",
    )
    task_id = row["task_id"]
    slot_id = row["slot_id"]
    if not isinstance(task_id, str) or not task_id:
        raise ActionValidationError("invalid_assessment_schema", "task_id must be non-empty")
    if task_id not in expected_items:
        raise ActionValidationError(
            "unexpected_assessment",
            "assessment task_id is not a decision task in this group",
        )
    item = expected_items[task_id]
    if not isinstance(slot_id, str) or slot_id != item.slot_id:
        raise ActionValidationError(
            "assessment_slot_mismatch",
            "assessment slot_id does not match its queue task",
            details={"task_id": task_id},
        )
    proposed_state = row["proposed_state"]
    if not isinstance(proposed_state, str) or proposed_state not in FINAL_STATES:
        raise ActionValidationError(
            "invalid_proposed_state",
            "proposed_state is not a Part2 final state",
        )
    if proposed_state not in item.allowed_final_states:
        raise ActionValidationError(
            "state_not_allowed",
            "proposed_state is not allowed for this queue task",
            details={"task_id": task_id, "proposed_state": proposed_state},
        )
    target_visibility = row["target_visibility"]
    if not isinstance(target_visibility, str) or target_visibility not in TARGET_VISIBILITIES:
        raise ActionValidationError(
            "invalid_target_visibility",
            "target_visibility is not in the v1 allowlist",
        )
    target_ownership = row["target_ownership"]
    if not isinstance(target_ownership, str) or target_ownership not in TARGET_OWNERSHIPS:
        raise ActionValidationError(
            "invalid_target_ownership",
            "target_ownership is not in the v1 allowlist",
        )
    semantic_finding = row["semantic_finding"]
    if not isinstance(semantic_finding, str) or semantic_finding not in SEMANTIC_FINDINGS:
        raise ActionValidationError(
            "invalid_semantic_finding",
            "semantic_finding is not in the v1 allowlist",
        )
    return SlotAssessment(
        task_id=task_id,
        slot_id=slot_id,
        proposed_state=proposed_state,
        target_visibility=target_visibility,
        target_ownership=target_ownership,
        semantic_finding=semantic_finding,
        resolved_unknown_reasons=_string_list(
            row["resolved_unknown_reasons"],
            "resolved_unknown_reasons",
        ),
        unresolved_blockers=_string_list(row["unresolved_blockers"], "unresolved_blockers"),
        evidence_refs=_evidence_list(row["evidence_refs"]),
        reason_codes=_string_list(row["reason_codes"], "reason_codes"),
    )


def _parse_final_proposal(
    payload: Mapping[str, Any],
    *,
    group_id: str,
    expected_items: Mapping[str, QueueItem],
) -> GroupFinalProposal:
    action = _exact_fields(
        payload,
        frozenset({"type", "assessments"}),
        code="invalid_final_proposal_schema",
        subject="final proposal",
    )
    raw_assessments = action["assessments"]
    if not isinstance(raw_assessments, list):
        raise ActionValidationError(
            "invalid_final_proposal_schema",
            "assessments must be a JSON array",
        )
    parsed: dict[str, SlotAssessment] = {}
    for raw in raw_assessments:
        assessment = _parse_assessment(raw, expected_items=expected_items)
        if assessment.task_id in parsed:
            raise ActionValidationError(
                "duplicate_assessment",
                "a decision task may appear exactly once",
                details={"task_id": assessment.task_id},
            )
        parsed[assessment.task_id] = assessment
    missing = [task_id for task_id in expected_items if task_id not in parsed]
    if missing:
        raise ActionValidationError(
            "missing_assessment",
            "the final proposal must assess every decision task",
            details={"missing_task_ids": missing},
        )
    return GroupFinalProposal(
        group_id=group_id,
        assessments=tuple(parsed[task_id] for task_id in expected_items),
    )


def parse_model_action(
    payload: Any,
    *,
    group_id: str,
    expected_items: Mapping[str, QueueItem],
) -> ToolRequest | GroupFinalProposal:
    """Parse one strict v1 action and bind final assessments to queue tasks."""

    if not isinstance(payload, Mapping):
        raise ActionValidationError("invalid_action", "model action must be a JSON object")
    action_type = payload.get("type")
    if not isinstance(action_type, str):
        raise ActionValidationError("invalid_action", "model action requires a string type")
    if action_type == "tool_request":
        return _parse_tool_request(payload)
    if action_type == "final_proposal":
        return _parse_final_proposal(
            payload,
            group_id=group_id,
            expected_items=expected_items,
        )
    raise ActionValidationError(
        "unsupported_action_type",
        "model action type is not allowlisted",
    )


class ReplayModelAdapter:
    """Return raw JSON actions from a versioned, group-keyed replay envelope."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        document = _plain_json(payload)
        if not isinstance(document, dict) or set(document) != {"schema_version", "actions"}:
            raise ValueError("replay envelope fields must be exactly schema_version and actions")
        if document["schema_version"] != REPLAY_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPLAY_SCHEMA_VERSION}")
        actions = document["actions"]
        if not isinstance(actions, dict):
            raise ValueError("replay actions must be an object keyed by group_id")
        _reject_live_provider_fields(actions)
        normalized: dict[str, tuple[Any, ...]] = {}
        for group_id, group_actions in actions.items():
            if not isinstance(group_id, str) or not group_id:
                raise ValueError("replay group_id keys must be non-empty strings")
            if not isinstance(group_actions, list):
                raise ValueError("each replay group must contain an action array")
            normalized[group_id] = tuple(freeze_json(action) for action in group_actions)
        self._actions = normalized
        self._offsets = {group_id: 0 for group_id in normalized}

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayModelAdapter":
        """Load strict UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON number is not allowed: {value}")

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        return cls(payload)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._actions))

    def remaining(self, group_id: str) -> int:
        if group_id not in self._actions:
            return 0
        return len(self._actions[group_id]) - self._offsets[group_id]

    def reset(self) -> None:
        for group_id in self._offsets:
            self._offsets[group_id] = 0

    def next_action(self, request: Mapping[str, Any]) -> Any:
        if not isinstance(request, Mapping):
            raise TypeError("model turn request must be a mapping")
        group_id = request.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("model turn request requires a non-empty group_id")
        if group_id not in self._actions:
            raise MissingReplayError("no replay actions exist for this group_id")
        offset = self._offsets[group_id]
        actions = self._actions[group_id]
        if offset >= len(actions):
            raise ReplayExhaustedError("replay actions are exhausted for this group_id")
        self._offsets[group_id] = offset + 1
        return _plain_json(actions[offset])


__all__ = [
    "ACTION_TYPES",
    "FINAL_STATES",
    "MODEL_TURN_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "SEMANTIC_FINDINGS",
    "TARGET_OWNERSHIPS",
    "TARGET_VISIBILITIES",
    "ActionValidationError",
    "GroupFinalProposal",
    "MissingReplayError",
    "ModelAdapter",
    "ReplayError",
    "ReplayExhaustedError",
    "ReplayModelAdapter",
    "SlotAssessment",
    "ToolRequest",
    "parse_model_action",
]
