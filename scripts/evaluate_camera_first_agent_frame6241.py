#!/usr/bin/env python3
"""Evaluate the Camera-first frame6241 Agent against the frozen human GT."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

STATES = ("free", "occupied", "unknown")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(rows: list[dict[str, str]], predictions: dict[str, str]) -> dict[str, Any]:
    detailed = []
    confusion = {truth: {pred: 0 for pred in STATES} for truth in STATES}
    for row in rows:
        sid = row["slot_id"]
        truth = row["gt_state"]
        pred = predictions.get(sid, "unknown")
        if pred not in STATES:
            pred = "unknown"
        confusion[truth][pred] += 1
        detailed.append(
            {
                **row,
                "prediction": pred,
                "correct": pred == truth,
            }
        )
    terminal = [row for row in detailed if row["prediction"] != "unknown"]
    determinate = [row for row in detailed if row["gt_state"] != "unknown"]
    correct = sum(row["correct"] for row in detailed)
    terminal_correct = sum(row["correct"] for row in terminal)
    determinate_correct = sum(row["correct"] for row in determinate)
    unknown_rows = [row for row in detailed if row["gt_state"] == "unknown"]
    unknown_false = [
        row for row in unknown_rows if row["prediction"] != "unknown"
    ]
    false_occupied = [
        row
        for row in detailed
        if row["prediction"] == "occupied" and row["gt_state"] != "occupied"
    ]
    false_free = [
        row
        for row in detailed
        if row["prediction"] == "free" and row["gt_state"] != "free"
    ]
    recalls = {}
    for state in STATES:
        subset = [row for row in detailed if row["gt_state"] == state]
        recalls[state] = (
            sum(row["correct"] for row in subset) / len(subset) if subset else None
        )
    total = len(detailed)
    return {
        "evaluated_gt_slots": total,
        "prediction_counts": dict(Counter(row["prediction"] for row in detailed)),
        "three_class_accuracy": correct / total if total else None,
        "terminal_coverage": len(terminal) / total if total else None,
        "terminal_selective_accuracy": (
            terminal_correct / len(terminal) if terminal else None
        ),
        "determinate_gt_accuracy_unknown_counted_wrong": (
            determinate_correct / len(determinate) if determinate else None
        ),
        "unknown_false_resolution_count": len(unknown_false),
        "unknown_false_resolution_rate": (
            len(unknown_false) / len(unknown_rows) if unknown_rows else None
        ),
        "false_occupied_count": len(false_occupied),
        "false_free_count": len(false_free),
        "per_class_recall": recalls,
        "confusion_matrix_gt_rows_pred_columns": confusion,
        "rows": detailed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--agent-result", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    args = parser.parse_args()

    with args.gt.open(newline="", encoding="utf-8-sig") as handle:
        gt_rows = [
            {
                key: str(value).strip().lower()
                if key in {"gt_state", "gt_observability", "identity_verified"}
                else str(value).strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]

    old_metrics = load_json(args.baseline_metrics)
    baseline_method = old_metrics["methods"]["best_part1_w45k3"]
    baseline_predictions = {
        row["slot_id"]: row["prediction"] for row in baseline_method["rows"]
    }

    agent_payload = load_json(args.agent_result)
    agent_rows = agent_payload["slot_results"]
    agent_predictions: dict[str, str] = {}
    agent_details: dict[str, dict[str, Any]] = {}
    for row in agent_rows:
        case = row["case"]
        sid = case["slot"]["slot_id"]
        state = str(case.get("final_state") or case.get("current_state") or "unknown")
        scores = case.get("final_scores") or case.get("current_scores") or {}
        agent_predictions[sid] = state
        agent_details[sid] = {
            "state": state,
            "scores": scores,
            "stop_reason": row.get("stop_reason"),
            "model_turns": row.get("model_turns"),
            "tool_rounds": row.get("tool_rounds"),
            "validation_errors": row.get("validation_errors", []),
            "decision_reason": case.get("decision_reason"),
        }

    trace_payload = load_json(args.trace)
    trace_by_slot = {row["slot_id"]: row for row in trace_payload["slots"]}
    for sid, detail in agent_details.items():
        trace_row = trace_by_slot.get(sid, {})
        steps = trace_row.get("steps", [])
        detail["tools"] = [
            step["execute"]["tool"]
            for step in steps
            if step.get("phase") == "plan_execute_observe"
            and "tool" in step.get("execute", {})
        ]
        detail["plans"] = [
            step.get("plan")
            for step in steps
            if step.get("phase") == "plan_execute_observe"
        ]
        final_steps = [step for step in steps if step.get("phase") == "final"]
        detail["final_observation"] = (
            final_steps[-1].get("observe", {}) if final_steps else {}
        )

    final_predictions = dict(baseline_predictions)
    final_predictions.update(agent_predictions)
    processed_ids = set(agent_predictions)

    subsets = {
        "all_locked_gt_14": gt_rows,
        "agent_processed_11": [
            row for row in gt_rows if row["slot_id"] in processed_ids
        ],
        "camera_visible_processed": [
            row
            for row in gt_rows
            if row["slot_id"] in processed_ids
            and row["gt_observability"] == "visible"
        ],
        "ambiguous_processed": [
            row
            for row in gt_rows
            if row["slot_id"] in processed_ids
            and row["gt_observability"] == "ambiguous"
        ],
        "occluded_processed": [
            row
            for row in gt_rows
            if row["slot_id"] in processed_ids
            and row["gt_observability"] == "occluded"
        ],
        "identity_verified_processed": [
            row
            for row in gt_rows
            if row["slot_id"] in processed_ids
            and row["identity_verified"] == "yes"
        ],
        "all_determinate_gt": [
            row for row in gt_rows if row["gt_state"] in {"free", "occupied"}
        ],
        "high_confidence_gt_0_90": [
            row
            for row in gt_rows
            if float(row.get("state_confidence") or 0.0) >= 0.90
        ],
        "high_confidence_determinate_gt_0_90": [
            row
            for row in gt_rows
            if row["gt_state"] in {"free", "occupied"}
            and float(row.get("state_confidence") or 0.0) >= 0.90
        ],
    }

    baseline_eval = evaluate(gt_rows, baseline_predictions)
    final_eval = evaluate(gt_rows, final_predictions)
    payload = {
        "schema_version": "frame6241-camera-first-agent-evaluation/1.0",
        "prediction_blind": True,
        "gt_used_by_agent": False,
        "gt_path": str(args.gt),
        "agent_result_path": str(args.agent_result),
        "agent_trace_path": str(args.trace),
        "merge_policy": (
            "Agent final overrides best W45/K3 Part1 only for the 11 processed "
            "front-180 locked-GT IDs; all other locked-GT IDs retain Part1."
        ),
        "threshold": 0.60,
        "gt_counts": dict(Counter(row["gt_state"] for row in gt_rows)),
        "methods": {
            "best_part1_w45k3": baseline_eval,
            "best_part1_w45k3_plus_camera_first_agent": final_eval,
        },
        "subsets": {
            name: evaluate(rows, final_predictions) for name, rows in subsets.items()
        },
        "delta_vs_best_part1": {
            "three_class_accuracy": (
                final_eval["three_class_accuracy"] - baseline_eval["three_class_accuracy"]
            ),
            "terminal_coverage": (
                final_eval["terminal_coverage"] - baseline_eval["terminal_coverage"]
            ),
            "correct_slot_count": (
                sum(row["correct"] for row in final_eval["rows"])
                - sum(row["correct"] for row in baseline_eval["rows"])
            ),
        },
        "agent_details": agent_details,
        "first_tool_camera_count": sum(
            bool(detail["tools"]) and detail["tools"][0] == "camera_context"
            for detail in agent_details.values()
        ),
        "camera_first_policy_compliance_rate": (
            sum(
                bool(detail["tools"]) and detail["tools"][0] == "camera_context"
                for detail in agent_details.values()
            )
            / max(1, len(agent_details))
        ),
        "replay_verified": load_json(
            args.agent_result.parents[1] / "replay_verification.json"
        ).get("verified", False),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    args.rows_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "slot_id",
            "gt_state",
            "gt_observability",
            "identity_verified",
            "state_confidence",
            "part1_w45k3",
            "agent_processed",
            "agent_state",
            "final_system_state",
            "correct",
            "tools",
            "agent_confidence",
            "agent_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in gt_rows:
            sid = row["slot_id"]
            detail = agent_details.get(sid, {})
            scores = detail.get("scores", {})
            state = detail.get("state")
            confidence_key = (
                f"{state}_confidence" if state in STATES else "unknown_confidence"
            )
            writer.writerow(
                {
                    "slot_id": sid,
                    "gt_state": row["gt_state"],
                    "gt_observability": row["gt_observability"],
                    "identity_verified": row["identity_verified"],
                    "state_confidence": row.get("state_confidence", ""),
                    "part1_w45k3": baseline_predictions.get(sid, "unknown"),
                    "agent_processed": sid in processed_ids,
                    "agent_state": state or "",
                    "final_system_state": final_predictions.get(sid, "unknown"),
                    "correct": final_predictions.get(sid, "unknown") == row["gt_state"],
                    "tools": " -> ".join(detail.get("tools", [])),
                    "agent_confidence": scores.get(confidence_key, ""),
                    "agent_reason": detail.get("final_observation", {}).get(
                        "reason", detail.get("decision_reason", "")
                    ),
                }
            )

    compact = {
        "all14": {
            key: final_eval[key]
            for key in (
                "three_class_accuracy",
                "terminal_coverage",
                "terminal_selective_accuracy",
                "false_occupied_count",
                "false_free_count",
                "prediction_counts",
            )
        },
        "camera_visible": payload["subsets"]["camera_visible_processed"],
        "camera_first_policy_compliance_rate": payload[
            "camera_first_policy_compliance_rate"
        ],
        "replay_verified": payload["replay_verified"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
