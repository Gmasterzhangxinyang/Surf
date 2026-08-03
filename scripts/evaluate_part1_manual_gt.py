#!/usr/bin/env python3
"""Evaluate abstaining Part1 predictions against manual slot labels.

The evaluator reports terminal accuracy and coverage separately.  Unknown is
an abstention: it is never counted as a correct terminal decision, but it also
does not become a false Occupied prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VALID_STATES = {"free", "occupied", "unknown"}


def _prediction_items(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        items: list[dict[str, Any]] = []
        for item_path in sorted(path.glob("*.json")):
            payload = json.loads(item_path.read_text(encoding="utf-8"))
            item = payload.get("decision", payload)
            if "slot_id" in item and "state" in item:
                items.append(item)
        return items
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
        return payload["decisions"]
    if isinstance(payload, dict) and isinstance(payload.get("slot_results"), list):
        items = []
        for row in payload["slot_results"]:
            case = row.get("case", {})
            slot = case.get("slot", {})
            if "slot_id" in slot and "final_state" in case:
                items.append(
                    {"slot_id": slot["slot_id"], "state": case["final_state"]}
                )
        return items
    if isinstance(payload, dict) and "slot_id" in payload and "state" in payload:
        return [payload]
    raise ValueError(f"unsupported prediction schema: {path}")


def _load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for item in _prediction_items(path):
        slot_id = str(item["slot_id"])
        state = str(item["state"]).lower()
        if state not in VALID_STATES:
            raise ValueError(f"invalid state {state!r} for {slot_id} in {path}")
        predictions[slot_id] = state
    return predictions


def _load_gt(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        slot_id = str(row["slot_id"])
        state = str(row["gt_state"]).lower()
        if state not in {"free", "occupied"}:
            raise ValueError(f"manual GT must be terminal: {slot_id}={state!r}")
        labels[slot_id] = {
            "state": state,
            "source": row.get("annotator") or "manual_gt_csv",
            "notes": row.get("notes") or "",
        }
    return labels


def _parse_named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _parse_supplement(value: str) -> tuple[str, str]:
    slot_id, separator, state = value.partition("=")
    state = state.lower()
    if not separator or not slot_id or state not in {"free", "occupied"}:
        raise argparse.ArgumentTypeError("expected SLOT_ID=free|occupied")
    return slot_id, state


def _evaluate(
    name: str,
    prediction_path: Path,
    predictions: dict[str, str],
    labels: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for slot_id, gt in labels.items():
        predicted = predictions.get(slot_id)
        in_scope = predicted is not None
        terminal = predicted in {"free", "occupied"}
        correct = bool(terminal and predicted == gt["state"])
        rows.append(
            {
                "model": name,
                "slot_id": slot_id,
                "gt_state": gt["state"],
                "gt_source": gt["source"],
                "prediction": predicted or "out_of_scope",
                "in_prediction_scope": in_scope,
                "terminal_prediction": terminal,
                "correct_terminal": correct,
                "false_occupied": predicted == "occupied" and gt["state"] == "free",
                "false_free": predicted == "free" and gt["state"] == "occupied",
            }
        )

    overlap = [row for row in rows if row["in_prediction_scope"]]
    terminal = [row for row in overlap if row["terminal_prediction"]]
    correct = [row for row in terminal if row["correct_terminal"]]
    false_occupied = [row["slot_id"] for row in terminal if row["false_occupied"]]
    false_free = [row["slot_id"] for row in terminal if row["false_free"]]
    predicted_occupied = [
        row for row in terminal if row["prediction"] == "occupied"
    ]
    true_occupied = [
        row
        for row in terminal
        if row["prediction"] == "occupied" and row["gt_state"] == "occupied"
    ]
    counts = {
        state: sum(row["prediction"] == state for row in overlap)
        for state in ("free", "occupied", "unknown")
    }
    result = {
        "model": name,
        "prediction_path": str(prediction_path),
        "prediction_total": len(predictions),
        "gt_total": len(labels),
        "gt_in_prediction_scope": len(overlap),
        "gt_out_of_prediction_scope": len(labels) - len(overlap),
        "state_counts_on_gt_overlap": counts,
        "terminal_count": len(terminal),
        "correct_terminal_count": len(correct),
        "selective_accuracy": (
            len(correct) / len(terminal) if terminal else None
        ),
        "terminal_coverage_on_overlap": (
            len(terminal) / len(overlap) if overlap else None
        ),
        "effective_correct_coverage_on_overlap": (
            len(correct) / len(overlap) if overlap else None
        ),
        "occupied_precision": (
            len(true_occupied) / len(predicted_occupied)
            if predicted_occupied
            else None
        ),
        "false_occupied_count": len(false_occupied),
        "false_occupied_slot_ids": false_occupied,
        "false_free_count": len(false_free),
        "false_free_slot_ids": false_free,
        "unknown_is_abstention": True,
    }
    return result, rows


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        type=_parse_named_path,
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--supplemental-label",
        type=_parse_supplement,
        action="append",
        default=[],
        metavar="SLOT_ID=STATE",
    )
    parser.add_argument(
        "--supplemental-source",
        default="user_manual_observation_2026-07-29",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    labels = _load_gt(args.gt_csv)
    original_gt_count = len(labels)
    for slot_id, state in args.supplemental_label:
        labels[slot_id] = {
            "state": state,
            "source": args.supplemental_source,
            "notes": "supplement supplied separately from the frozen 24-slot GT CSV",
        }

    results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for name, path in args.prediction:
        result, rows = _evaluate(name, path, _load_predictions(path), labels)
        results.append(result)
        all_rows.extend(rows)

    payload = {
        "schema_version": "part1-manual-gt-evaluation/1.0",
        "metric_policy": {
            "unknown": "abstention",
            "out_of_scope": "not_evaluated_as_prediction",
            "selective_accuracy_denominator": "terminal predictions on GT overlap",
            "coverage_denominator": "GT labels present in prediction scope",
        },
        "gt_csv": str(args.gt_csv),
        "frozen_gt_count": original_gt_count,
        "supplemental_labels": [
            {
                "slot_id": slot_id,
                "state": state,
                "source": args.supplemental_source,
            }
            for slot_id, state in args.supplemental_label
        ],
        "combined_gt_count": len(labels),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = list(all_rows[0]) if all_rows else []
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [
        "# frame 9277 人工 GT 对照",
        "",
        f"- 冻结 GT：{original_gt_count} 个车位",
        f"- 补充 GT：{len(args.supplemental_label)} 个车位（不修改原 GT 文件）",
        "- Unknown 按拒判处理；准确率与覆盖率必须分开报告。",
        "",
        "| 模型 | GT交集 | Free | Occupied | Unknown | 选择性准确率 | 终态覆盖率 | 错误Occupied |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        counts = result["state_counts_on_gt_overlap"]
        lines.append(
            "| {model} | {overlap} | {free} | {occupied} | {unknown} | "
            "{accuracy} | {coverage} | {false_occupied} |".format(
                model=result["model"],
                overlap=result["gt_in_prediction_scope"],
                free=counts["free"],
                occupied=counts["occupied"],
                unknown=counts["unknown"],
                accuracy=_percentage(result["selective_accuracy"]),
                coverage=_percentage(result["terminal_coverage_on_overlap"]),
                false_occupied=result["false_occupied_count"],
            )
        )
    lines.extend(
        [
            "",
            "说明：没有任何 Occupied 输出时，Occupied precision 记为 N/A，"
            "不能写成 100%；它表示系统选择了 fail-closed，而不是已证明占用识别率完美。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
