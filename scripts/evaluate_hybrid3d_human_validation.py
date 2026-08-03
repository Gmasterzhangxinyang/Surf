#!/usr/bin/env python3
"""Leakage-resistant frozen Part1 evaluation on protected human labels.

The workflow is deliberately split:

  select  -> writes only sample identifiers, never label values
  predict -> loads the frozen config and selection, never opens human_labels.json
  score   -> joins immutable predictions to human labels and reports metrics
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.io import (
    load_frame_records,
    load_known_slots,
    write_json_atomic,
    write_text_atomic,
)
from scripts.run_systematic_candidate_experiments import _causal_pool, _localize, _run


FROZEN_CONFIG_SHA256 = "56e4350dc031854d5fd5a8ebe76e0d66ae06d5aa55bdaf0e4d5d29bf1084b083"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def select_cases(manifest_path: Path, labels_path: Path, output: Path) -> None:
    manifest = load_json(manifest_path)
    labels = load_json(labels_path)
    cases = {str(item["sample_id"]): item for item in manifest["cases"]}
    label_records = labels["labels"]
    missing = sorted(set(label_records) - set(cases))
    if missing:
        raise ValueError(f"{len(missing)} labeled samples are absent from the manifest")
    # No label value, reason, reviewer or timestamp leaves this step.
    rows = [
        {
            "sample_id": sample_id,
            "slot_id": str(cases[sample_id]["slot_id"]),
            "source_anchor_frame": int(cases[sample_id]["source_anchor_frame"]),
            "evidence_anchor_frame": max(
                int(frame["lidar_frame"])
                for frame in cases[sample_id]["evidence_frames"]
            ),
            "evidence_lidar_frames": [
                int(frame["lidar_frame"])
                for frame in cases[sample_id]["evidence_frames"]
            ],
        }
        for sample_id in sorted(label_records)
    ]
    write_json_atomic(
        output,
        {
            "schema_version": "parkingagent-human-eval-selection/1.0",
            "dataset_id": manifest["dataset_id"],
            "manifest_id": manifest["manifest_id"],
            "selection_rule": "all samples that have a protected human annotation; label values excluded",
            "sample_count": len(rows),
            "samples": rows,
        },
    )


def _decision_record(decision: Any | None) -> dict[str, Any]:
    if decision is None:
        return {
            "state": "unknown",
            "decision_reason": "missing_from_causal_local_map",
            "unknown_reasons": ["missing_from_causal_local_map"],
            "agent_observable": False,
        }
    record = asdict(decision)
    record["scope_status"] = decision.scope_status.value
    record["state"] = decision.state.value
    return record


def predict(
    frozen_path: Path,
    selection_path: Path,
    output: Path,
    expected_sha256: str,
) -> None:
    actual_hash = file_sha256(frozen_path)
    if actual_hash != expected_sha256:
        raise ValueError(
            "frozen config hash mismatch: "
            f"expected {expected_sha256}, got {actual_hash}"
        )
    frozen = load_json(frozen_path)
    selection = load_json(selection_path)
    data = frozen["data"]
    frames_path = PROJECT_ROOT / data["frames_csv"]
    slots_path = PROJECT_ROOT / data["slot_database"]
    points_path = PROJECT_ROOT / data["map_points_dir"]
    all_frames = load_frame_records(frames_path)
    slots, scale = load_known_slots(slots_path)
    valid_slot_ids = {slot.slot_id for slot in slots}
    frame_count = int(frozen["temporal_policy"]["frame_count"])
    shared = dict(frozen["shared_overrides"])
    predictions: list[dict[str, Any]] = []
    for sample in selection["samples"]:
        slot_id = str(sample["slot_id"])
        if slot_id not in valid_slot_ids:
            raise ValueError(f"unknown slot in evaluation selection: {slot_id}")
        protocols = (
            ("legacy_source_anchor", int(sample["source_anchor_frame"])),
            ("evidence_aligned", int(sample["evidence_anchor_frame"])),
        )
        for protocol_id, anchor in protocols:
            selected = _causal_pool(all_frames, anchor, frame_count)
            for variant in frozen["variants"]:
                config = replace(
                    Hybrid3DConfig(),
                    **shared,
                    occupied_localization_uncertainty_m=float(
                        variant["occupied_localization_uncertainty_m"]
                    ),
                    window_before=max(Hybrid3DConfig().window_before, frame_count - 1),
                    window_after=max(Hybrid3DConfig().window_after, frame_count - 1),
                )
                config.validate()
                started = time.perf_counter()
                result, pipeline_seconds, provider = _run(
                    slots,
                    scale,
                    selected,
                    points_path,
                    config,
                    cache_size=max(32, frame_count),
                    slot_ids=[slot_id],
                )
                local = _localize(result, provider)
                by_id = {decision.slot_id: decision for decision in local.decisions}
                predictions.append(
                    {
                        "sample_id": str(sample["sample_id"]),
                        "slot_id": slot_id,
                        "protocol_id": protocol_id,
                        "anchor_frame": anchor,
                        "legacy_source_anchor_frame": int(sample["source_anchor_frame"]),
                        "evidence_anchor_frame": int(sample["evidence_anchor_frame"]),
                        "selected_frame_ids": [frame.frame_id for frame in selected],
                        "causal": True,
                        "variant_id": str(variant["id"]),
                        "uncertainty_width_m": float(
                            variant["occupied_localization_uncertainty_m"]
                        ),
                        "decision": _decision_record(by_id.get(slot_id)),
                        "pipeline_seconds": pipeline_seconds,
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    json.dumps(
                        {
                            "sample_id": str(sample["sample_id"])[:12],
                            "slot_id": slot_id,
                            "protocol": protocol_id,
                            "anchor": anchor,
                            "variant": variant["id"],
                            "state": predictions[-1]["decision"]["state"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    write_json_atomic(
        output,
        {
            "schema_version": "parkingagent-frozen-human-eval-predictions/1.0",
            "frozen_config_sha256": actual_hash,
            "human_labels_opened_by_prediction_phase": False,
            "selection_manifest": str(selection_path),
            "prediction_count": len(predictions),
            "predictions": predictions,
        },
    )


def _variant_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observable = [row for row in rows if row["human_label"] in {"free", "occupied"}]
    terminal = [row for row in observable if row["prediction"] in {"free", "occupied"}]
    correct = [row for row in terminal if row["prediction"] == row["human_label"]]
    contradictions = [row for row in terminal if row["prediction"] != row["human_label"]]
    metrics: dict[str, Any] = {
        "labeled_count": len(rows),
        "observable_ground_truth_count": len(observable),
        "unobservable_ground_truth_count": len(rows) - len(observable),
        "terminal_prediction_count": len(terminal),
        "unknown_prediction_count_on_observable": len(observable) - len(terminal),
        "correct_terminal_count": len(correct),
        "terminal_contradiction_count": len(contradictions),
        "coverage": len(terminal) / len(observable) if observable else None,
        "coverage_wilson_95": wilson(len(terminal), len(observable)),
        "selective_terminal_accuracy": len(correct) / len(terminal) if terminal else None,
        "selective_terminal_accuracy_wilson_95": wilson(len(correct), len(terminal)),
        "effective_exact_rate_unknown_as_incorrect": len(correct) / len(observable) if observable else None,
        "effective_exact_wilson_95": wilson(len(correct), len(observable)),
    }
    for truth in ("occupied", "free"):
        truth_rows = [row for row in observable if row["human_label"] == truth]
        truth_correct = [row for row in truth_rows if row["prediction"] == truth]
        predicted = [row for row in observable if row["prediction"] == truth]
        predicted_correct = [row for row in predicted if row["human_label"] == truth]
        metrics[f"{truth}_recall"] = (
            len(truth_correct) / len(truth_rows) if truth_rows else None
        )
        metrics[f"{truth}_recall_counts"] = [len(truth_correct), len(truth_rows)]
        metrics[f"{truth}_precision"] = (
            len(predicted_correct) / len(predicted) if predicted else None
        )
        metrics[f"{truth}_precision_counts"] = [len(predicted_correct), len(predicted)]
    return metrics


def pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def score(
    predictions_path: Path,
    labels_path: Path,
    output_json: Path,
    output_md: Path,
    expected_sha256: str,
    expected_labels_sha256: str | None = None,
) -> None:
    payload = load_json(predictions_path)
    if payload.get("frozen_config_sha256") != expected_sha256:
        raise ValueError("predictions were not produced by the registered frozen config")
    labels_sha256 = file_sha256(labels_path)
    if (
        expected_labels_sha256 is not None
        and labels_sha256 != expected_labels_sha256
    ):
        raise ValueError(
            "human-label hash mismatch: "
            f"expected {expected_labels_sha256}, got {labels_sha256}"
        )
    labels = load_json(labels_path)["labels"]
    joined: list[dict[str, Any]] = []
    for prediction in payload["predictions"]:
        sample_id = str(prediction["sample_id"])
        if sample_id not in labels:
            raise ValueError(f"prediction has no human label: {sample_id}")
        label = labels[sample_id]
        if str(label["slot_id"]) != str(prediction["slot_id"]):
            raise ValueError(f"slot mismatch for {sample_id}")
        joined.append(
            {
                "sample_id": sample_id,
                "slot_id": str(prediction["slot_id"]),
                "protocol_id": str(prediction["protocol_id"]),
                "anchor_frame": int(prediction["anchor_frame"]),
                "variant_id": str(prediction["variant_id"]),
                "human_label": str(label["human_label"]),
                "prediction": str(prediction["decision"]["state"]),
                "decision_reason": str(prediction["decision"]["decision_reason"]),
                "unknown_reasons": list(prediction["decision"].get("unknown_reasons", [])),
            }
        )
    protocols = sorted({row["protocol_id"] for row in joined})
    variants = sorted({row["variant_id"] for row in joined})
    metrics = {
        protocol: {
            variant: _variant_metrics(
                [
                    row
                    for row in joined
                    if row["protocol_id"] == protocol
                    and row["variant_id"] == variant
                ]
            )
            for variant in variants
        }
        for protocol in protocols
    }
    result = {
        "schema_version": "parkingagent-frozen-human-eval-score/1.0",
        "frozen_config_sha256": expected_sha256,
        "human_labels_sha256": labels_sha256,
        "protocol": {
            "primary": "evidence_aligned",
            "audit_only": "legacy_source_anchor",
            "observable_labels": ["free", "occupied"],
            "excluded_from_occupancy_accuracy": ["unobservable"],
            "unknown_is_abstention": True,
            "effective_exact_rate_treats_unknown_as_incorrect": True,
        },
        "metrics": metrics,
        "joined_rows": joined,
        "limitations": [
            "Only eight protected samples currently have human labels.",
            "Only six labels are occupancy-observable; confidence intervals are therefore wide.",
            "The human evidence frames can precede source_anchor_frame; this score assumes slot occupancy did not change during that short encounter.",
        ],
    }
    write_json_atomic(output_json, result)

    lines = [
        "# 冻结 Part1 人工真值盲评",
        "",
        f"- 冻结配置 SHA-256：`{expected_sha256}`",
        f"- 人工标签 SHA-256：`{labels_sha256}`",
        "- 预测阶段没有打开 `human_labels.json`；预测完成后才由 score 阶段连接标签。",
        "- 当前只有 8 个已标注样本，其中 Occupied 4、Free 2、Unobservable 2；占用准确率只统计前 6 个。",
        "- `evidence_aligned` 是主结果；`legacy_source_anchor` 与人工证据相差 55–320 帧，只作为时序错位审计。",
        "",
        "## 结果",
        "",
        "| 协议 | 冻结版本 | 终态正确率（选择性） | 覆盖率 | Unknown按错计的有效正确率 | Occupied recall | Free recall | 终态冲突 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("evidence_aligned", "legacy_source_anchor"):
        if protocol not in metrics:
            continue
        for variant in variants:
            row = metrics[protocol][variant]
            lines.append(
                "| {protocol} | {variant} | {selective} ({correct}/{terminal}) | {coverage} | "
                "{effective} | {occ} | {free} | {contradictions} |".format(
                    protocol=protocol,
                    variant=variant,
                    selective=pct(row["selective_terminal_accuracy"]),
                    correct=row["correct_terminal_count"],
                    terminal=row["terminal_prediction_count"],
                    coverage=pct(row["coverage"]),
                    effective=pct(row["effective_exact_rate_unknown_as_incorrect"]),
                    occ=pct(row["occupied_recall"]),
                    free=pct(row["free_recall"]),
                    contradictions=row["terminal_contradiction_count"],
                )
            )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- 选择性正确率只回答“系统敢下 Free/Occupied 终态时有多准”。",
            "- 覆盖率回答“在人工可判定样本中，系统有多少没有弃权为 Unknown”。",
            "- 有效正确率把 Unknown 也按未完成任务计入，防止用大量弃权制造虚假的高准确率。",
            "- 样本量过小，不能据此声称已经达到 95% 或 Nature 级泛化；必须继续增加独立标注。",
            "",
            "## 逐样本审计",
            "",
            "| 协议 | 版本 | slot | anchor | 人工 | Part1 | 原因 |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for row in joined:
        reason = row["decision_reason"].replace("|", "/")
        lines.append(
            f"| {row['protocol_id']} | {row['variant_id']} | {row['slot_id']} | {row['anchor_frame']} | "
            f"{row['human_label']} | {row['prediction']} | {reason} |"
        )
    write_text_atomic(output_md, "\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--labels", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--frozen-config", type=Path, required=True)
    predict_parser.add_argument("--selection", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument(
        "--expected-sha256",
        default=FROZEN_CONFIG_SHA256,
    )
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--labels", type=Path, required=True)
    score_parser.add_argument("--output-json", type=Path, required=True)
    score_parser.add_argument("--output-md", type=Path, required=True)
    score_parser.add_argument(
        "--expected-sha256",
        default=FROZEN_CONFIG_SHA256,
    )
    score_parser.add_argument("--expected-labels-sha256")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "select":
        select_cases(args.manifest, args.labels, args.output)
    elif args.command == "predict":
        predict(
            args.frozen_config,
            args.selection,
            args.output,
            args.expected_sha256,
        )
    else:
        score(
            args.predictions,
            args.labels,
            args.output_json,
            args.output_md,
            args.expected_sha256,
            args.expected_labels_sha256,
        )


if __name__ == "__main__":
    main()
