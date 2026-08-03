"""Deterministic append-only JSONL evidence ledger for Part2 orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import freeze_json
from .queueing import canonical_json_bytes, canonical_sha256


TRACE_SCHEMA_VERSION = "part2-evidence-ledger/1.0"

EVENT_TYPES = {
    "lifecycle": frozenset({"group_started", "group_finished"}),
    "model": frozenset(
        {
            "model_turn_started",
            "model_action_received",
            "model_error",
            "replay_missing",
            "replay_exhausted",
        }
    ),
    "tool": frozenset({"tool_attempted", "tool_result"}),
    "validation": frozenset({"validation_feedback", "final_proposal_accepted"}),
    "stop": frozenset({"group_stopped"}),
}

_FORBIDDEN_KEYS = frozenset(
    {
        "chain_of_thought",
        "cot",
        "reasoning",
        "thought",
        "thoughts",
        "internal_reasoning",
        "timestamp",
        "wall_clock",
        "created_at",
        "updated_at",
        "provider_message",
        "prompt",
        "completion",
    }
)


def _validate_payload(value: Any, path: str = "data") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            if key.strip().lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"{path} contains forbidden trace field {key}")
            _validate_payload(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_payload(item, f"{path}[{index}]")
        return
    canonical_json_bytes(value)


def _mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(item) for item in value]
    return value


class EvidenceLedger:
    """Accumulate canonical events with deterministic span, sequence, and event IDs."""

    def __init__(self, group_id: str) -> None:
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        self.group_id = group_id
        self.trace_span_id = "span_" + canonical_sha256(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "group_id": group_id,
            }
        ).split(":", 1)[1]
        self._events: list[Mapping[str, Any]] = []
        self._sealed = False

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    @property
    def sealed(self) -> bool:
        return self._sealed

    def event_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(_mutable_json(event) for event in self._events)

    def record(
        self,
        category: str,
        event_type: str,
        data: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Append one allowlisted event and return its immutable record."""

        if self._sealed:
            raise RuntimeError("evidence ledger is sealed")
        if category not in EVENT_TYPES:
            raise ValueError(f"unsupported trace event category: {category}")
        if event_type not in EVENT_TYPES[category]:
            raise ValueError(f"unsupported {category} trace event type: {event_type}")
        payload = {} if data is None else data
        if not isinstance(payload, Mapping):
            raise ValueError("trace event data must be a JSON object")
        _validate_payload(payload)
        sequence = len(self._events) + 1
        identity = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_span_id": self.trace_span_id,
            "sequence": sequence,
            "category": category,
            "event_type": event_type,
            "data": payload,
        }
        event = {
            **identity,
            "event_id": "event_" + canonical_sha256(identity).split(":", 1)[1],
        }
        immutable = freeze_json(event)
        self._events.append(immutable)
        return immutable

    def seal(self) -> None:
        self._sealed = True

    def to_jsonl(self) -> str:
        if not self._events:
            return ""
        return "".join(
            canonical_json_bytes(event).decode("utf-8") + "\n"
            for event in self._events
        )

    def write_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.to_jsonl(), encoding="utf-8")
        return destination


__all__ = ["EVENT_TYPES", "TRACE_SCHEMA_VERSION", "EvidenceLedger"]
