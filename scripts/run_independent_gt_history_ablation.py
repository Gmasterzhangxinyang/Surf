#!/usr/bin/env python3
"""Run causal W/K history ablation against independent terminal GT labels."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_json_atomic,
)
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline
from parking_slot_hybrid_3d.static_semantics import SemanticStaticOccupiedVeto
from gltf_lidar_ndt import load_gltf_map


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> Hybrid3DConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = Hybrid3DConfig(**payload)
    config.validate()
    return config


def _load_gt(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["slot_id"]: row["gt_state"].lower() for row in rows}
    if not labels or any(state not in {"free", "occupied"} for state in labels.values()):
        raise ValueError("GT input must contain non-empty terminal free/occupied labels")
    return labels


def _pool(all_frames, anchor: int, count: int):
    selected = tuple(frame for frame in all_frames if frame.frame_id <= anchor)[-count:]
    if len(selected) != count or selected[-1].frame_id != anchor:
        raise ValueError(f"anchor {anchor} lacks {count} causal frames")
    return selected


def _sample(pool, count: int):
    if not 3 <= count <= len(pool):
        raise ValueError("sample count must be between 3 and history span")
    if count == len(pool):
        return pool
    indices = np.rint(np.linspace(0, len(pool) - 1, num=count)).astype(np.int64)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("uniform sample contains duplicate frames")
    selected = tuple(pool[int(index)] for index in indices)
    if selected[-1].frame_id != pool[-1].frame_id:
        raise RuntimeError("uniform sample omitted anchor")
    return selected


def _evaluate(
    labels: dict[str, str],
    predictions: dict[str, str],
    diagnostics: dict[str, dict],
) -> dict:
    rows = []
    for slot_id, target in labels.items():
        predicted = predictions.get(slot_id, "out_of_scope")
        terminal = predicted in {"free", "occupied"}
        correct = terminal and predicted == target
        rows.append(
            {
                "slot_id": slot_id,
                "gt_state": target,
                "prediction": predicted,
                "terminal": terminal,
                "correct": correct,
                "false_occupied": predicted == "occupied" and target == "free",
                "false_free": predicted == "free" and target == "occupied",
                **diagnostics.get(slot_id, {}),
            }
        )
    terminal = [row for row in rows if row["terminal"]]
    correct = [row for row in terminal if row["correct"]]
    false_occupied = [row["slot_id"] for row in rows if row["false_occupied"]]
    false_free = [row["slot_id"] for row in rows if row["false_free"]]
    return {
        "gt_count": len(rows),
        "prediction_scope_count": sum(row["prediction"] != "out_of_scope" for row in rows),
        "state_counts": {
            state: sum(row["prediction"] == state for row in rows)
            for state in ("free", "occupied", "unknown", "out_of_scope")
        },
        "terminal_count": len(terminal),
        "correct_terminal_count": len(correct),
        "selective_accuracy": len(correct) / len(terminal) if terminal else None,
        "terminal_coverage_all_gt": len(terminal) / len(rows),
        "effective_correct_coverage_all_gt": len(correct) / len(rows),
        "false_occupied_count": len(false_occupied),
        "false_occupied_slot_ids": false_occupied,
        "false_free_count": len(false_free),
        "false_free_slot_ids": false_free,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gltf-static-map", type=Path)
    parser.add_argument("--static-map-sample-step", type=float, default=0.012)
    parser.add_argument("--static-map-distance-m", type=float, default=0.35)
    parser.add_argument("--static-map-max-explained-ratio", type=float, default=0.50)
    parser.add_argument("--static-map-min-residual-short-extent-m", type=float, default=0.75)
    parser.add_argument("--anchor", type=int, default=9277)
    parser.add_argument("--history-spans", type=int, nargs="+", default=[15, 30, 60, 100])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--cache-size", type=int, default=128)
    args = parser.parse_args()

    labels = _load_gt(args.gt_csv)
    all_frames = load_frame_records(args.frames_csv)
    slots, scale = load_known_slots(args.slot_db)
    base = _load_config(args.config)
    static_occupied_veto = None
    if args.gltf_static_map is not None:
        gltf = load_gltf_map(args.gltf_static_map, args.static_map_sample_step)
        static_layers = [
            gltf.layers[name].points
            for name in ("wall", "elevator", "arrester")
            if name in gltf.layers and len(gltf.layers[name].points)
        ]
        if not static_layers:
            raise RuntimeError("glTF contains no wall/elevator/arrester points")
        static_occupied_veto = SemanticStaticOccupiedVeto(
            np.vstack(static_layers),
            association_distance_m=args.static_map_distance_m,
            max_static_explained_ratio=args.static_map_max_explained_ratio,
            min_residual_short_extent_m=args.static_map_min_residual_short_extent_m,
        )
    combinations = sorted(
        (span, count)
        for span in set(args.history_spans)
        for count in set(args.sample_counts)
        if 3 <= count <= span
    )
    runs = []
    for span, count in combinations:
        selected = _sample(_pool(all_frames, args.anchor, span), count)
        frame_span = selected[-1].frame_id - selected[0].frame_id
        config = replace(
            base,
            frame_stride=1,
            window_before=max(base.window_before, frame_span),
            window_after=max(base.window_after, frame_span),
        )
        provider = FramePointProvider(
            {frame.frame_id: frame for frame in selected},
            args.map_points_dir,
            cache_size=max(args.cache_size, len(selected)),
            project_root=PROJECT_ROOT,
        )
        started = time.perf_counter()
        result = Hybrid3DPipeline(
            slots,
            selected,
            provider,
            scale,
            config,
            phase="full",
            slot_ids=tuple(labels),
            static_occupied_veto=static_occupied_veto,
        ).run()
        elapsed = time.perf_counter() - started
        predictions = {decision.slot_id: decision.state.value for decision in result.decisions}
        diagnostics = {}
        for decision in result.decisions:
            features = decision.occupied_evidence.features
            diagnostics[decision.slot_id] = {
                "decision_reason": decision.decision_reason,
                "unknown_reasons": list(decision.unknown_reasons),
                "occupied_failures": list(decision.occupied_evidence.failures),
                "occupied_strength": float(decision.occupied_evidence.strength),
                "stability_pass_ratio": float(decision.stability.pass_ratio),
                "occupied_gates": [
                    {
                        "name": gate.name,
                        "passed": gate.passed,
                        "value": gate.value,
                        "threshold": gate.threshold,
                    }
                    for gate in decision.occupied_evidence.gate_results
                ],
                "occupied_features": (
                    None
                    if features is None
                    else {
                        "robust_short_extent_m": float(
                            min(features.robust_extent_x_m, features.robust_extent_y_m)
                        ),
                        "robust_long_extent_m": float(
                            max(features.robust_extent_x_m, features.robust_extent_y_m)
                        ),
                        "low_bev_coverage": float(features.low_bev_coverage),
                        "outside_residual_ratio": float(features.outside_residual_ratio),
                        "boundary_ratio": float(features.boundary_ratio),
                        "pca_linearity": float(features.pca_linearity),
                        "robust_pca_linearity": float(features.robust_pca_linearity),
                    }
                ),
            }
        evaluation = _evaluate(labels, predictions, diagnostics)
        runs.append(
            {
                "history_span_W": span,
                "sample_count_K": count,
                "sampling": "uniform_inclusive_anchor",
                "selected_frame_ids": [frame.frame_id for frame in selected],
                "runtime_seconds": elapsed,
                **evaluation,
            }
        )
        print(
            json.dumps(
                {
                    "W": span,
                    "K": count,
                    "correct": evaluation["correct_terminal_count"],
                    "false_occupied": evaluation["false_occupied_count"],
                    "coverage": evaluation["terminal_coverage_all_gt"],
                    "seconds": round(elapsed, 3),
                }
            ),
            flush=True,
        )

    best = max(
        runs,
        key=lambda row: (
            -int(row["false_occupied_count"] + row["false_free_count"]),
            int(row["correct_terminal_count"]),
            float(row["selective_accuracy"] or 0.0),
            -int(row["sample_count_K"]),
            -int(row["history_span_W"]),
            -float(row["runtime_seconds"]),
        ),
    )
    payload = {
        "schema_version": "independent-gt-history-ablation/1.0",
        "anchor": args.anchor,
        "causal_only": True,
        "sampling": "uniform inclusive anchor",
        "gt_csv": str(args.gt_csv),
        "gt_sha256": _sha256(args.gt_csv),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "semantic_static_veto": {
            "enabled": static_occupied_veto is not None,
            "gltf": None if args.gltf_static_map is None else str(args.gltf_static_map),
            "association_distance_m": args.static_map_distance_m,
            "max_static_explained_ratio": args.static_map_max_explained_ratio,
            "min_residual_short_extent_m": args.static_map_min_residual_short_extent_m,
        },
        "gt_state_counts": {
            state: sum(value == state for value in labels.values())
            for state in ("free", "occupied")
        },
        "occupied_recall_evaluable": any(value == "occupied" for value in labels.values()),
        "best_objective": (
            "zero terminal errors, then maximum correct terminal coverage, "
            "then lower K, lower W, lower runtime"
        ),
        "best": {
            key: value
            for key, value in best.items()
            if key not in {"rows", "selected_frame_ids"}
        },
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_json, payload)

    csv_rows = []
    for row in runs:
        csv_rows.append(
            {
                key: row[key]
                for key in (
                    "history_span_W",
                    "sample_count_K",
                    "runtime_seconds",
                    "prediction_scope_count",
                    "terminal_count",
                    "correct_terminal_count",
                    "selective_accuracy",
                    "terminal_coverage_all_gt",
                    "effective_correct_coverage_all_gt",
                    "false_occupied_count",
                    "false_free_count",
                )
            }
        )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# 独立 GT 历史帧 W/K 消融",
        "",
        f"- anchor：{args.anchor}",
        f"- GT：{len(labels)} 个高置信终态；Free={sum(v == 'free' for v in labels.values())}，"
        f"Occupied={sum(v == 'occupied' for v in labels.values())}",
        "- 目标按“零终态错误 → 最大正确覆盖 → 更低 K/W/耗时”预注册排序。",
        "",
        "| W | K | 正确终态 | 错误Occupied | 选择性准确率 | 全GT终态覆盖率 | 秒 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        accuracy = (
            "N/A"
            if row["selective_accuracy"] is None
            else f"{100 * row['selective_accuracy']:.1f}%"
        )
        lines.append(
            f"| {row['history_span_W']} | {row['sample_count_K']} | "
            f"{row['correct_terminal_count']} | {row['false_occupied_count']} | "
            f"{accuracy} | {100 * row['terminal_coverage_all_gt']:.1f}% | "
            f"{row['runtime_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"最佳组合：W={best['history_span_W']}，K={best['sample_count_K']}。",
            "",
            "限制：当前独立GT没有可确认Occupied，因此本消融只能选择Free安全性与覆盖，"
            "不能证明Occupied召回；该指标必须在含独立Occupied标签的其他anchor上补齐。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
