"""Honest metrics for human-resolved parking-slot labels."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _label_value(value: Any) -> tuple[str, str]:
    if isinstance(value, Mapping):
        return str(value.get("human_label", "")), str(value.get("reason", ""))
    return str(value), ""


def evaluate(
    predictions: Mapping[str, str],
    labels: Mapping[str, Any],
    total_cases: int,
) -> dict[str, Any]:
    total = max(0, int(total_cases))
    human_resolved = 0
    human_unobservable = 0
    algorithm_decided = 0
    true_occupied = false_occupied = false_free = true_free = 0
    reasons: Counter[str] = Counter()
    for slot_id, raw_label in labels.items():
        human_label, reason = _label_value(raw_label)
        if human_label == "unobservable":
            human_unobservable += 1
            if reason:
                reasons[reason] += 1
            continue
        if human_label not in {"occupied", "free"}:
            continue
        human_resolved += 1
        prediction = str(predictions.get(slot_id, "unknown"))
        if prediction not in {"occupied", "free"}:
            continue
        algorithm_decided += 1
        if human_label == "occupied" and prediction == "occupied":
            true_occupied += 1
        elif human_label == "free" and prediction == "occupied":
            false_occupied += 1
        elif human_label == "occupied" and prediction == "free":
            false_free += 1
        else:
            true_free += 1
    correct = true_occupied + true_free
    explicit_unknown = sum(str(value) == "unknown" for value in predictions.values())
    missing_predictions = max(0, total - len(predictions))
    occupied_precision = _ratio(true_occupied, true_occupied + false_occupied)
    occupied_recall = _ratio(true_occupied, true_occupied + false_free)
    occupied_f1 = (
        2 * occupied_precision * occupied_recall / (occupied_precision + occupied_recall)
        if occupied_precision is not None and occupied_recall is not None and occupied_precision + occupied_recall
        else None
    )
    return {
        "total_case_count": total,
        "labeled_count": len(labels),
        "unlabeled_count": max(0, total - len(labels)),
        "human_resolved_count": human_resolved,
        "human_unobservable_count": human_unobservable,
        "human_verifiable_coverage": _ratio(human_resolved, total),
        "algorithm_decided_count": algorithm_decided,
        "algorithm_decision_coverage_on_resolved": _ratio(algorithm_decided, human_resolved),
        "algorithm_unknown_count": explicit_unknown,
        "algorithm_missing_count": missing_predictions,
        "algorithm_unknown_rate": _ratio(explicit_unknown, total),
        "true_occupied": true_occupied,
        "false_occupied": false_occupied,
        "false_free": false_free,
        "true_free": true_free,
        "correct_count": correct,
        "resolved_decided_accuracy": _ratio(correct, algorithm_decided),
        "occupied_precision": occupied_precision,
        "occupied_recall": occupied_recall,
        "occupied_f1": occupied_f1,
        "free_precision": _ratio(true_free, true_free + false_free),
        "free_recall": _ratio(true_free, true_free + false_occupied),
        "unobservable_reason_counts": dict(sorted(reasons.items())),
    }
