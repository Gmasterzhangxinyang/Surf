#!/usr/bin/env python3
"""Score v5 W30/K5 on the six observable frozen human-review cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    terminal = [row for row in rows if row["prediction"] in {"free", "occupied"}]
    correct = [row for row in terminal if row["prediction"] == row["gt_state"]]
    occupied_gt = [row for row in rows if row["gt_state"] == "occupied"]
    free_gt = [row for row in rows if row["gt_state"] == "free"]
    predicted_occupied = [row for row in rows if row["prediction"] == "occupied"]
    predicted_free = [row for row in rows if row["prediction"] == "free"]
    return {
        "observable_gt_count": len(rows),
        "terminal_prediction_count": len(terminal),
        "correct_terminal_count": len(correct),
        "selective_terminal_accuracy": len(correct) / len(terminal) if terminal else None,
        "coverage": len(terminal) / len(rows),
        "effective_exact_rate_unknown_as_incorrect": len(correct) / len(rows),
        "occupied_recall": (
            sum(row["prediction"] == "occupied" for row in occupied_gt) / len(occupied_gt)
            if occupied_gt
            else None
        ),
        "free_recall": (
            sum(row["prediction"] == "free" for row in free_gt) / len(free_gt)
            if free_gt
            else None
        ),
        "occupied_precision": (
            sum(row["gt_state"] == "occupied" for row in predicted_occupied)
            / len(predicted_occupied)
            if predicted_occupied
            else None
        ),
        "free_precision": (
            sum(row["gt_state"] == "free" for row in predicted_free)
            / len(predicted_free)
            if predicted_free
            else None
        ),
        "false_occupied_count": sum(
            row["prediction"] == "occupied" and row["gt_state"] == "free"
            for row in rows
        ),
        "false_free_count": sum(
            row["prediction"] == "free" and row["gt_state"] == "occupied"
            for row in rows
        ),
        "unknown_count": sum(row["prediction"] == "unknown" for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    labels_payload = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = labels_payload["labels"]
    rows: list[dict[str, object]] = []
    for sample in selection["samples"]:
        label = labels[sample["sample_id"]]
        gt_state = str(label["human_label"]).lower()
        if gt_state not in {"free", "occupied"}:
            continue
        anchor = int(sample["evidence_anchor_frame"])
        decision_path = (
            args.artifact_root / f"frame_{anchor:06d}" / "slot_decisions.json"
        )
        decisions = json.loads(decision_path.read_text(encoding="utf-8"))["decisions"]
        decision = next(
            row for row in decisions if row["slot_id"] == sample["slot_id"]
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "anchor_frame": anchor,
                "slot_id": sample["slot_id"],
                "gt_state": gt_state,
                "prediction": decision["state"],
                "decision_reason": decision["decision_reason"],
                "unknown_reasons": ";".join(decision.get("unknown_reasons", [])),
                "semantic_static_gate": next(
                    (
                        failure
                        for failure in decision["occupied_evidence"].get("failures", [])
                        if str(failure).startswith("semantic_static_")
                    ),
                    "",
                ),
            }
        )

    payload = {
        "schema_version": "parkingagent-v5-frozen-challenge-score/1.0",
        "evaluation_status": "retrospective_frozen_challenge_not_independent_holdout",
        "model": "v5_static_semantic_w30k5",
        "sampling": {"history_span_W": 30, "sample_count_K": 5, "causal": True},
        "labels_path": str(args.labels),
        "labels_sha256": sha256(args.labels),
        "selection_path": str(args.selection),
        "selection_sha256": sha256(args.selection),
        "metrics": metrics(rows),
        "rows": rows,
        "limitations": [
            "The labels were frozen before this replay, but the challenge cases came from a historical evidence-selection process.",
            "Only six occupancy-observable cases are available (four occupied, two free).",
            "This is a retrospective safety/coverage challenge, not a new independent holdout.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
