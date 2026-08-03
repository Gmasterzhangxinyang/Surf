"""No-GT shadow-transition reporting for the legacy 202-slot baseline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_SCOPE_STATES = {"in_route_scope", "partial_route_scope", "out_of_route_scope"}
_DECISION_STATES = {"occupied", "free", "unknown"}


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        slot_id = str(row.get("slot_id", "")).strip()
        if not slot_id:
            raise ValueError(f"{label} row has no slot_id")
        if slot_id in indexed:
            raise ValueError(f"duplicate {label} slot_id: {slot_id}")
        indexed[slot_id] = row
    return indexed


def _text_list(value: Any, field: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must contain a JSON array") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _migration_reasons(
    scope: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> list[str]:
    if decision is None:
        return _text_list(scope.get("reasons"), "scope.reasons")
    reasons: list[str] = []
    decision_reason = str(decision.get("decision_reason", "")).strip()
    if decision_reason:
        reasons.append(decision_reason)
    for reason in _text_list(decision.get("unknown_reasons"), "decision.unknown_reasons"):
        if reason not in reasons:
            reasons.append(reason)
    return reasons


@dataclass(frozen=True)
class ShadowComparison:
    common_rows: tuple[dict[str, Any], ...]
    new_route_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def build_shadow_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    scope_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> ShadowComparison:
    """Join legacy evidence and hybrid decisions without implying GT accuracy."""

    baseline = _index_rows(baseline_rows, "baseline")
    scopes = _index_rows(scope_rows, "scope")
    decisions = _index_rows(decision_rows, "decision")
    missing_scope = sorted(set(baseline) - set(scopes))
    if missing_scope:
        raise ValueError(f"baseline slot_id missing from scope: {missing_scope[0]}")
    unknown_decisions = sorted(set(decisions) - set(scopes))
    if unknown_decisions:
        raise ValueError(f"decision slot_id missing from scope: {unknown_decisions[0]}")

    for slot_id, scope in scopes.items():
        status = str(scope.get("scope_status", ""))
        if status not in _SCOPE_STATES:
            raise ValueError(f"invalid scope_status for {slot_id}: {status}")
        decision = decisions.get(slot_id)
        if status == "out_of_route_scope":
            if decision is not None:
                raise ValueError(f"out-of-route slot has decision: {slot_id}")
            continue
        if decision is None:
            raise ValueError(f"missing decision for path-relevant slot: {slot_id}")
        state = str(decision.get("state", ""))
        if state not in _DECISION_STATES:
            raise ValueError(f"invalid decision state for {slot_id}: {state}")

    common_rows: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    for slot_id in sorted(baseline):
        old = baseline[slot_id]
        scope = scopes[slot_id]
        decision = decisions.get(slot_id)
        new_state = str(decision.get("state", "")) if decision is not None else ""
        transition_key = (
            f"{old.get('state', '')} -> {scope.get('scope_status', '')}"
            f" -> {new_state or 'no_decision'}"
        )
        transition_counts[transition_key] += 1
        common_rows.append(
            {
                "slot_id": slot_id,
                "old_state": str(old.get("state", "")),
                "old_score": str(old.get("score", "")),
                "old_reason": str(old.get("reason", "")),
                "new_scope_status": str(scope.get("scope_status", "")),
                "new_state": new_state,
                "new_decision_reason": (
                    str(decision.get("decision_reason", "")) if decision is not None else ""
                ),
                "new_unknown_reasons": (
                    _text_list(decision.get("unknown_reasons"), "decision.unknown_reasons")
                    if decision is not None
                    else []
                ),
                "migration_reasons": _migration_reasons(scope, decision),
            }
        )

    new_route_rows: list[dict[str, Any]] = []
    for slot_id in sorted(set(scopes) - set(baseline)):
        scope = scopes[slot_id]
        if scope.get("scope_status") == "out_of_route_scope":
            continue
        decision = decisions[slot_id]
        new_route_rows.append(
            {
                "slot_id": slot_id,
                "new_scope_status": str(scope.get("scope_status", "")),
                "new_state": str(decision.get("state", "")),
                "new_decision_reason": str(decision.get("decision_reason", "")),
                "new_unknown_reasons": _text_list(
                    decision.get("unknown_reasons"), "decision.unknown_reasons"
                ),
                "agent_observable": _bool_value(decision.get("agent_observable", False)),
            }
        )

    common_route = sum(row["new_scope_status"] != "out_of_route_scope" for row in common_rows)
    summary = {
        "schema_version": "shadow-comparison/1.0",
        "comparison_semantics": "state_transition_and_coverage_without_gt_not_accuracy",
        "gt_status": "unavailable",
        "baseline_common_count": len(common_rows),
        "baseline_common_route_count": common_route,
        "baseline_common_out_of_route_count": len(common_rows) - common_route,
        "newly_evaluated_route_count": len(new_route_rows),
        "new_scope_counts": dict(
            sorted(Counter(str(row.get("scope_status", "")) for row in scopes.values()).items())
        ),
        "new_decision_counts": dict(
            sorted(Counter(str(row.get("state", "")) for row in decisions.values()).items())
        ),
        "common_transition_counts": dict(sorted(transition_counts.items())),
    }
    return ShadowComparison(
        common_rows=tuple(common_rows),
        new_route_rows=tuple(new_route_rows),
        summary=summary,
    )


__all__ = ["ShadowComparison", "build_shadow_comparison"]
