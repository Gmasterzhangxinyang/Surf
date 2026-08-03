#!/usr/bin/env python3
"""Run causal W/K ablation without turning predictions into ground truth.

This experiment is deliberately label-free.  It measures coverage, runtime and
cross-configuration stability on a prediction-independent geometry universe.
It must not be reported as an accuracy experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gltf_lidar_ndt import load_gltf_map
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_json_atomic,
)
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline
from parking_slot_hybrid_3d.static_semantics import SemanticStaticOccupiedVeto


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_config(path: Path) -> Hybrid3DConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = Hybrid3DConfig(**payload)
    config.validate()
    return config


def _load_universe(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("slots")
    if not isinstance(rows, list):
        raise ValueError("geometry universe must contain a slots list")
    slot_ids = tuple(str(row["slot_id"]) for row in rows)
    if not slot_ids or len(slot_ids) != len(set(slot_ids)):
        raise ValueError("geometry universe slot IDs must be non-empty and unique")
    return slot_ids


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gltf-static-map", type=Path, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--history-spans", type=int, nargs="+", default=[15, 30, 60, 100])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--static-map-sample-step", type=float, default=0.012)
    parser.add_argument("--static-map-distance-m", type=float, default=0.35)
    parser.add_argument("--static-map-max-explained-ratio", type=float, default=0.50)
    parser.add_argument("--static-map-min-residual-short-extent-m", type=float, default=0.75)
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    universe_ids = _load_universe(args.universe)
    all_frames = load_frame_records(args.frames_csv)
    slots, scale = load_known_slots(args.slot_db)
    known_ids = {slot.slot_id for slot in slots}
    missing = sorted(set(universe_ids) - known_ids)
    if missing:
        raise ValueError(f"universe contains unknown slot IDs: {missing[:5]}")
    base = _load_config(args.config)

    gltf = load_gltf_map(args.gltf_static_map, args.static_map_sample_step)
    static_layers = [
        gltf.layers[name].points
        for name in ("wall", "elevator", "arrester")
        if name in gltf.layers and len(gltf.layers[name].points)
    ]
    if not static_layers:
        raise RuntimeError("glTF contains no wall/elevator/arrester points")
    static_veto = SemanticStaticOccupiedVeto(
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
    runs: list[dict] = []
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
            slot_ids=universe_ids,
            static_occupied_veto=static_veto,
        ).run()
        elapsed = time.perf_counter() - started
        decision_by_slot = {decision.slot_id: decision for decision in result.decisions}
        states = {
            slot_id: (
                decision_by_slot[slot_id].state.value
                if slot_id in decision_by_slot
                else "out_of_scope"
            )
            for slot_id in universe_ids
        }
        state_counts = dict(Counter(states.values()))
        rows = []
        for slot_id in universe_ids:
            decision = decision_by_slot.get(slot_id)
            rows.append(
                {
                    "slot_id": slot_id,
                    "state": states[slot_id],
                    "decision_reason": None if decision is None else decision.decision_reason,
                    "unknown_reasons": (
                        [] if decision is None else list(decision.unknown_reasons)
                    ),
                }
            )
        run = {
            "history_span_W": span,
            "sample_count_K": count,
            "sampling": "uniform_inclusive_anchor",
            "selected_frame_ids": [frame.frame_id for frame in selected],
            "runtime_seconds": elapsed,
            "universe_count": len(universe_ids),
            "prediction_scope_count": sum(value != "out_of_scope" for value in states.values()),
            "terminal_count": sum(value in {"free", "occupied"} for value in states.values()),
            "terminal_coverage": (
                sum(value in {"free", "occupied"} for value in states.values())
                / len(universe_ids)
            ),
            "state_counts": state_counts,
            "rows": rows,
        }
        runs.append(run)
        print(
            json.dumps(
                {
                    "W": span,
                    "K": count,
                    "states": state_counts,
                    "terminal": run["terminal_count"],
                    "seconds": round(elapsed, 3),
                }
            ),
            flush=True,
        )

    state_history: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            state_history[row["slot_id"]].append(row["state"])
    terminal_history = {
        slot_id: [state for state in states if state in {"free", "occupied"}]
        for slot_id, states in state_history.items()
    }
    contradictory_slots = sorted(
        slot_id
        for slot_id, states in terminal_history.items()
        if len(set(states)) > 1
    )

    for run in runs:
        state_map = {row["slot_id"]: row["state"] for row in run["rows"]}
        conflicts = []
        terminal_matches = 0
        terminal_comparisons = 0
        all_state_matches = 0
        for slot_id, predicted in state_map.items():
            modal_state = Counter(state_history[slot_id]).most_common(1)[0][0]
            all_state_matches += int(predicted == modal_state)
            other_terminal = set(terminal_history[slot_id])
            if predicted in {"free", "occupied"} and other_terminal:
                terminal_comparisons += 1
                majority_terminal = Counter(terminal_history[slot_id]).most_common(1)[0][0]
                terminal_matches += int(predicted == majority_terminal)
                if any(value != predicted for value in other_terminal):
                    conflicts.append(slot_id)
        run["cross_config_terminal_conflict_count"] = len(conflicts)
        run["cross_config_terminal_conflict_slot_ids"] = conflicts
        run["terminal_majority_agreement"] = (
            terminal_matches / terminal_comparisons if terminal_comparisons else None
        )
        run["all_state_modal_agreement"] = all_state_matches / len(universe_ids)

    recommended = max(
        runs,
        key=lambda row: (
            -int(row["cross_config_terminal_conflict_count"]),
            float(row["all_state_modal_agreement"]),
            int(row["terminal_count"]),
            -int(row["sample_count_K"]),
            -int(row["history_span_W"]),
            -float(row["runtime_seconds"]),
        ),
    )
    payload = {
        "schema_version": "label-free-history-ablation/1.0",
        "anchor": args.anchor,
        "causal_only": True,
        "prediction_independent_geometry_universe": str(args.universe),
        "universe_sha256": _sha256(args.universe),
        "universe_count": len(universe_ids),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "ground_truth_used": False,
        "accuracy_evaluable": False,
        "scientific_limit": (
            "No same-anchor independent human Free/Occupied labels exist for this "
            "random sample. Coverage and cross-configuration stability are measured; "
            "accuracy and false-occupied rate are not."
        ),
        "ranking_objective": (
            "zero cross-configuration terminal conflicts, then all-state modal "
            "agreement, terminal coverage, lower K/W/runtime"
        ),
        "recommended_for_gt_followup_not_accuracy_best": {
            key: value
            for key, value in recommended.items()
            if key not in {"rows", "selected_frame_ids"}
        },
        "cross_configuration_contradictory_slot_ids": contradictory_slots,
        "semantic_static_veto": {
            "enabled": True,
            "layers": ["wall", "elevator", "arrester"],
            "association_distance_m": args.static_map_distance_m,
        },
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_json, payload)

    fields = [
        "history_span_W",
        "sample_count_K",
        "runtime_seconds",
        "prediction_scope_count",
        "terminal_count",
        "terminal_coverage",
        "cross_config_terminal_conflict_count",
        "terminal_majority_agreement",
        "all_state_modal_agreement",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in runs)

    lines = [
        "# 随机锚点 W/K 无标签消融",
        "",
        f"- anchor：{args.anchor}",
        f"- 独立几何全集：{len(universe_ids)} 个车位。",
        "- 未读取 Part1 状态，也没有使用旧候选构造 GT。",
        "- 当前没有该随机时刻的独立人工终态标签，因此准确率、误报率不可计算。",
        "",
        "| W | K | 终态数 | 终态覆盖 | 跨配置终态冲突 | 状态众数一致率 | 秒 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {row['history_span_W']} | {row['sample_count_K']} | "
            f"{row['terminal_count']} | {100 * row['terminal_coverage']:.1f}% | "
            f"{row['cross_config_terminal_conflict_count']} | "
            f"{100 * row['all_state_modal_agreement']:.1f}% | "
            f"{row['runtime_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "仅按稳定性选出的后续人工GT优先组合："
            f"W={recommended['history_span_W']}，K={recommended['sample_count_K']}。",
            "它不是“准确率最佳”，需在同锚点独立人工GT完成后才可下该结论。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
