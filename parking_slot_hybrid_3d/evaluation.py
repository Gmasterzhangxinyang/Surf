"""Offline-only ground-truth providers and honest abstention-aware metrics.

This module intentionally has no import path from inference, decision routing, or
Agent queue generation.  Ground truth can therefore evaluate finalized Part 1
artifacts, but can never influence a Part 1 decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .contracts import DecisionState, SlotDecision


_GT_LABELS = frozenset({"occupied", "free", "ignore"})


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} is required")
    return value.strip()


@dataclass(frozen=True)
class GTMetadata:
    dataset_id: str
    gt_version: str
    annotation_source: str

    def validate(self) -> None:
        _required_text(self.dataset_id, "metadata.dataset_id")
        _required_text(self.gt_version, "metadata.gt_version")
        _required_text(self.annotation_source, "metadata.annotation_source")


@dataclass(frozen=True)
class SlotGroundTruth:
    dataset_id: str
    gt_version: str
    slot_id: str
    label: str
    annotation_source: str


class GroundTruthProvider(Protocol):
    """Boundary implemented by optional offline GT sources."""

    def available(self) -> bool:
        ...

    def load_metadata(self) -> GTMetadata:
        ...

    def load_slot_labels(self) -> Sequence[SlotGroundTruth]:
        ...


@dataclass(frozen=True)
class UnavailableGroundTruthProvider:
    reason: str = "ground_truth_not_available"

    def available(self) -> bool:
        return False

    def load_metadata(self) -> GTMetadata:
        raise RuntimeError(self.reason)

    def load_slot_labels(self) -> Sequence[SlotGroundTruth]:
        raise RuntimeError(self.reason)


@dataclass(frozen=True)
class SyntheticGroundTruthProvider:
    metadata: GTMetadata
    labels: Sequence[SlotGroundTruth]

    def available(self) -> bool:
        return True

    def load_metadata(self) -> GTMetadata:
        return self.metadata

    def load_slot_labels(self) -> Sequence[SlotGroundTruth]:
        return tuple(self.labels)


@dataclass(frozen=True)
class JsonGroundTruthProvider:
    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))

    @property
    def reason(self) -> str:
        return "gt_file_missing"

    def available(self) -> bool:
        return self.path.is_file()

    def _load_payload(self) -> Mapping[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load GT JSON {self.path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("GT JSON root must be an object")
        return payload

    def load_metadata(self) -> GTMetadata:
        payload = self._load_payload()
        raw = payload.get("metadata")
        if not isinstance(raw, Mapping):
            raise ValueError("metadata must be an object")
        metadata = GTMetadata(
            dataset_id=_required_text(raw.get("dataset_id"), "metadata.dataset_id"),
            gt_version=_required_text(raw.get("gt_version"), "metadata.gt_version"),
            annotation_source=_required_text(
                raw.get("annotation_source"), "metadata.annotation_source"
            ),
        )
        metadata.validate()
        return metadata

    def load_slot_labels(self) -> Sequence[SlotGroundTruth]:
        payload = self._load_payload()
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list):
            raise ValueError("labels must be an array")
        labels: list[SlotGroundTruth] = []
        for index, raw in enumerate(raw_labels):
            path = f"labels[{index}]"
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path} must be an object")
            labels.append(
                SlotGroundTruth(
                    dataset_id=_required_text(raw.get("dataset_id"), f"{path}.dataset_id"),
                    gt_version=_required_text(raw.get("gt_version"), f"{path}.gt_version"),
                    slot_id=_required_text(raw.get("slot_id"), f"{path}.slot_id"),
                    label=_required_text(raw.get("label"), f"{path}.label"),
                    annotation_source=_required_text(
                        raw.get("annotation_source"), f"{path}.annotation_source"
                    ),
                )
            )
        return tuple(labels)


@dataclass(frozen=True)
class FractionMetric:
    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True)
class RiskCoveragePoint:
    minimum_strength: float
    coverage: FractionMetric
    risk: FractionMetric

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_strength": self.minimum_strength,
            "coverage": self.coverage.to_dict(),
            "risk": self.risk.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationResult:
    gt_status: str
    reason: str = ""
    metadata: GTMetadata | None = None
    scope_slot_ids: tuple[str, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()
    metrics: tuple[tuple[str, FractionMetric], ...] = ()
    risk_coverage: tuple[RiskCoveragePoint, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.gt_status == "unavailable":
            return {
                "gt_status": "unavailable",
                "reason": self.reason or "ground_truth_not_available",
            }
        if self.metadata is None:
            raise ValueError("available evaluation result requires GT metadata")
        return {
            "gt_status": "available",
            "dataset_id": self.metadata.dataset_id,
            "gt_version": self.metadata.gt_version,
            "annotation_source": self.metadata.annotation_source,
            "scope_slot_ids": list(self.scope_slot_ids),
            "counts": dict(self.counts),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics},
            "risk_coverage": [point.to_dict() for point in self.risk_coverage],
            "score_semantics": "rule_evidence_strength_ranking_not_probability",
        }


def _provider_reason(provider: GroundTruthProvider) -> str:
    reason = getattr(provider, "reason", "ground_truth_not_available")
    return reason if isinstance(reason, str) and reason else "ground_truth_not_available"


def _validated_scope_ids(scope_slot_ids: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(slot_id) for slot_id in scope_slot_ids)
    if any(not value for value in values):
        raise ValueError("scope slot_id must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError("duplicate scope slot_id")
    return tuple(sorted(values))


def _validated_decisions(
    decisions: Sequence[SlotDecision], scope_ids: frozenset[str]
) -> dict[str, SlotDecision]:
    result: dict[str, SlotDecision] = {}
    for decision in decisions:
        if decision.slot_id in result:
            raise ValueError(f"duplicate decision slot_id: {decision.slot_id}")
        if decision.slot_id not in scope_ids:
            raise ValueError(f"unknown decision slot_id: {decision.slot_id}")
        result[decision.slot_id] = decision
    return result


def _validated_labels(
    metadata: GTMetadata,
    labels: Sequence[SlotGroundTruth],
    scope_ids: frozenset[str],
) -> tuple[SlotGroundTruth, ...]:
    result: list[SlotGroundTruth] = []
    seen: set[str] = set()
    for label in labels:
        slot_id = _required_text(label.slot_id, "GT label slot_id")
        if slot_id in seen:
            raise ValueError(f"duplicate GT slot_id: {slot_id}")
        seen.add(slot_id)
        if slot_id not in scope_ids:
            raise ValueError(f"unknown GT slot_id: {slot_id}")
        if label.label not in _GT_LABELS:
            raise ValueError(f"invalid GT label for {slot_id}: {label.label}")
        if label.dataset_id != metadata.dataset_id:
            raise ValueError(f"GT label {slot_id} dataset_id does not match metadata")
        if label.gt_version != metadata.gt_version:
            raise ValueError(f"GT label {slot_id} gt_version does not match metadata")
        if label.annotation_source != metadata.annotation_source:
            raise ValueError(f"GT label {slot_id} annotation_source does not match metadata")
        result.append(label)
    return tuple(sorted(result, key=lambda row: row.slot_id))


def _terminal_strength(decision: SlotDecision) -> float:
    if decision.state is DecisionState.OCCUPIED:
        strength = float(decision.occupied_evidence.strength)
    elif decision.state is DecisionState.FREE:
        strength = float(decision.free_evidence.strength)
    else:
        raise ValueError("unknown decisions do not have terminal ranking strength")
    if not math.isfinite(strength):
        raise ValueError(f"non-finite decision strength for {decision.slot_id}")
    return strength


def _risk_coverage_points(
    terminal: Sequence[tuple[float, str, bool]], eligible_count: int
) -> tuple[RiskCoveragePoint, ...]:
    ordered = sorted(terminal, key=lambda row: (-row[0], row[1]))
    points: list[RiskCoveragePoint] = []
    retained = 0
    errors = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            retained += 1
            errors += int(not ordered[index][2])
            index += 1
        points.append(
            RiskCoveragePoint(
                minimum_strength=threshold,
                coverage=FractionMetric(retained, eligible_count),
                risk=FractionMetric(errors, retained),
            )
        )
    return tuple(points)


def evaluate_against_gt(
    decisions: Sequence[SlotDecision],
    provider: GroundTruthProvider,
    scope_slot_ids: Iterable[str],
) -> EvaluationResult:
    """Evaluate finalized decisions; this function is never called by inference."""

    if not provider.available():
        return EvaluationResult(gt_status="unavailable", reason=_provider_reason(provider))

    metadata = provider.load_metadata()
    metadata.validate()
    scope = _validated_scope_ids(scope_slot_ids)
    scope_set = frozenset(scope)
    decision_by_slot = _validated_decisions(decisions, scope_set)
    labels = _validated_labels(metadata, provider.load_slot_labels(), scope_set)

    ignored = 0
    true_occupied = 0
    false_occupied = 0
    false_free = 0
    true_free = 0
    unknown = 0
    actual_occupied = 0
    actual_free = 0
    terminal: list[tuple[float, str, bool]] = []

    for truth in labels:
        if truth.label == "ignore":
            ignored += 1
            continue
        actual_occupied += int(truth.label == "occupied")
        actual_free += int(truth.label == "free")
        decision = decision_by_slot.get(truth.slot_id)
        if decision is None or decision.state is DecisionState.UNKNOWN:
            unknown += 1
            continue

        correct = decision.state.value == truth.label
        terminal.append((_terminal_strength(decision), truth.slot_id, correct))
        if truth.label == "occupied" and decision.state is DecisionState.OCCUPIED:
            true_occupied += 1
        elif truth.label == "free" and decision.state is DecisionState.OCCUPIED:
            false_occupied += 1
        elif truth.label == "occupied" and decision.state is DecisionState.FREE:
            false_free += 1
        else:
            true_free += 1

    eligible = actual_occupied + actual_free
    terminal_count = true_occupied + false_occupied + false_free + true_free
    counts = (
        ("eligible", eligible),
        ("ignored", ignored),
        ("true_occupied", true_occupied),
        ("false_occupied", false_occupied),
        ("false_free", false_free),
        ("true_free", true_free),
        ("unknown", unknown),
    )
    metrics = (
        ("occupied_precision", FractionMetric(true_occupied, true_occupied + false_occupied)),
        ("occupied_recall", FractionMetric(true_occupied, actual_occupied)),
        ("free_precision", FractionMetric(true_free, true_free + false_free)),
        ("free_recall", FractionMetric(true_free, actual_free)),
        ("false_free_rate", FractionMetric(false_free, actual_occupied)),
        ("unknown_coverage", FractionMetric(unknown, eligible)),
        ("terminal_coverage", FractionMetric(terminal_count, eligible)),
    )
    return EvaluationResult(
        gt_status="available",
        metadata=metadata,
        scope_slot_ids=scope,
        counts=counts,
        metrics=metrics,
        risk_coverage=_risk_coverage_points(terminal, eligible),
    )


__all__ = [
    "EvaluationResult",
    "FractionMetric",
    "GTMetadata",
    "GroundTruthProvider",
    "JsonGroundTruthProvider",
    "RiskCoveragePoint",
    "SlotGroundTruth",
    "SyntheticGroundTruthProvider",
    "UnavailableGroundTruthProvider",
    "evaluate_against_gt",
]
