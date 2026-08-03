#!/usr/bin/env python3
"""Evaluate Part1 and Agent outputs only on the locked human-GT slot IDs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

STATES = ("free", "occupied", "unknown")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def part1_predictions(path: Path) -> dict[str, str]:
    payload = read_json(path)
    rows = payload.get("decisions", payload)
    if isinstance(rows, dict):
        rows = list(rows.values())
    return {
        str(row["slot_id"]): str(row["state"]).lower()
        for row in rows
        if str(row.get("state", "")).lower() in STATES
    }


def agent_predictions(path: Path) -> dict[str, str]:
    payload = read_json(path)
    rows = payload["slot_results"]
    result = {}
    for row in rows:
        case = row["case"]
        sid = str(case["slot"]["slot_id"])
        state = case.get("final_state", case.get("current_state", "unknown"))
        result[sid] = str(state or "unknown").lower()
    return result


def merged_system(
    gt_ids: list[str],
    part1: dict[str, str],
    agent: dict[str, str] | None = None,
) -> dict[str, str]:
    result = {}
    for sid in gt_ids:
        base = part1.get(sid, "unknown")
        result[sid] = (agent or {}).get(sid, base)
    return result


def evaluate(gt: dict[str, str], prediction: dict[str, str]) -> dict:
    rows = []
    confusion = {truth: {pred: 0 for pred in STATES} for truth in STATES}
    for sid, truth in gt.items():
        pred = prediction.get(sid, "unknown")
        if pred not in STATES:
            pred = "unknown"
        confusion[truth][pred] += 1
        rows.append(
            {
                "slot_id": sid,
                "gt_state": truth,
                "prediction": pred,
                "correct": pred == truth,
            }
        )
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    terminal = [row for row in rows if row["prediction"] != "unknown"]
    terminal_correct = sum(row["correct"] for row in terminal)
    determinate_gt = [row for row in rows if row["gt_state"] != "unknown"]
    determinate_correct = sum(row["correct"] for row in determinate_gt)
    recalls = {}
    for state in STATES:
        subset = [row for row in rows if row["gt_state"] == state]
        recalls[state] = (
            sum(row["correct"] for row in subset) / len(subset) if subset else None
        )
    unknown_false_resolution = sum(
        row["gt_state"] == "unknown" and row["prediction"] != "unknown"
        for row in rows
    )
    return {
        "evaluated_gt_slots": total,
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "three_class_accuracy": correct / total if total else None,
        "terminal_coverage": len(terminal) / total if total else None,
        "terminal_selective_accuracy": (
            terminal_correct / len(terminal) if terminal else None
        ),
        "determinate_gt_accuracy_unknown_counted_wrong": (
            determinate_correct / len(determinate_gt) if determinate_gt else None
        ),
        "unknown_false_resolution_count": unknown_false_resolution,
        "unknown_false_resolution_rate": (
            unknown_false_resolution
            / max(1, sum(row["gt_state"] == "unknown" for row in rows))
        ),
        "per_class_recall": recalls,
        "confusion_matrix_gt_rows_pred_columns": confusion,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--original-part1", type=Path, required=True)
    p.add_argument("--tuned-part1", type=Path, required=True)
    p.add_argument("--baseline-agent", type=Path, required=True)
    p.add_argument("--new-agent", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.gt.open(newline="", encoding="utf-8-sig") as f:
        gt_rows = list(csv.DictReader(f))
    gt = {row["slot_id"]: row["gt_state"].lower() for row in gt_rows}
    ids = list(gt)
    original = part1_predictions(a.original_part1)
    tuned = part1_predictions(a.tuned_part1)
    baseline_agent = agent_predictions(a.baseline_agent)
    methods = {
        "original_part1": merged_system(ids, original),
        "pose_tuned_part1": merged_system(ids, tuned),
        "pose_tuned_part1_plus_baseline_agent": merged_system(
            ids, tuned, baseline_agent
        ),
    }
    if a.new_agent and a.new_agent.exists():
        methods["pose_tuned_part1_plus_occlusion_agent"] = merged_system(
            ids, tuned, agent_predictions(a.new_agent)
        )
    payload = {
        "schema_version": "frame6241-locked-gt-evaluation/1.0",
        "gt_path": str(a.gt),
        "gt_slot_count": len(gt),
        "gt_counts": dict(Counter(gt.values())),
        "policy": {
            "scope": "only locked GT slot IDs",
            "missing_system_prediction": "unknown",
            "agent_merge": "Agent final overrides Part1 for processed candidates; otherwise Part1 retained",
            "unknown_is_a_semantic_class": True,
        },
        "methods": {name: evaluate(gt, pred) for name, pred in methods.items()},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        name: {
            key: value
            for key, value in result.items()
            if key
            in {
                "three_class_accuracy",
                "terminal_coverage",
                "terminal_selective_accuracy",
                "determinate_gt_accuracy_unknown_counted_wrong",
                "unknown_false_resolution_rate",
                "prediction_counts",
            }
        }
        for name, result in payload["methods"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
