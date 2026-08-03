#!/usr/bin/env python3
"""Run the current-mainline pillar/FOV study and causal W/K ablation.

The experiment deliberately reports evidence resolution and agreement against
a dense causal reference.  It does not call that reference ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import DecisionState, FrameRecord, SlotDecision
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_json_atomic,
)
from parking_slot_hybrid_3d.local_map import (
    LocalMapConfig,
    actual_lidar_coverage_polygon,
    build_local_map_snapshot,
    restrict_result_to_local_map,
)
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline, PipelineResult
from parking_slot_hybrid_3d.reporting import part2_candidate_in_forward_fov


SCHEMA_VERSION = "parkingagent-systematic-candidate-study/1.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-anchors", type=int, nargs="+", default=[5283, 7605, 9277])
    parser.add_argument("--heldout-anchors", type=int, nargs="+", default=[234, 3160, 9443])
    parser.add_argument("--history-spans", type=int, nargs="+", default=[15, 30, 60, 100])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--dense-reference-frames", type=int, default=100)
    parser.add_argument("--cache-size", type=int, default=128)
    return parser


def _causal_pool(all_frames: list[FrameRecord], anchor: int, count: int) -> tuple[FrameRecord, ...]:
    selected = tuple(frame for frame in all_frames if frame.frame_id <= anchor)[-count:]
    if len(selected) != count or selected[-1].frame_id != anchor:
        raise ValueError(f"anchor {anchor} lacks {count} causal input records")
    return selected


def _uniform_sample(pool: tuple[FrameRecord, ...], count: int) -> tuple[FrameRecord, ...]:
    if not 3 <= count <= len(pool):
        raise ValueError("sample count must be between 3 and the history span")
    if count == len(pool):
        return pool
    raw = np.linspace(0, len(pool) - 1, num=count)
    indices = np.rint(raw).astype(np.int64)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("uniform sampler produced duplicate indices")
    selected = tuple(pool[int(index)] for index in indices)
    if selected[-1].frame_id != pool[-1].frame_id:
        raise RuntimeError("uniform sampler omitted the causal anchor")
    return selected


def _configured(base: Hybrid3DConfig, frames: tuple[FrameRecord, ...]) -> Hybrid3DConfig:
    span = frames[-1].frame_id - frames[0].frame_id
    return replace(
        base,
        frame_stride=1,
        window_before=max(base.window_before, span),
        window_after=max(base.window_after, span),
    )


def _run(
    slots: list[Any],
    scale: float,
    frames: tuple[FrameRecord, ...],
    points_dir: Path,
    base_config: Hybrid3DConfig,
    cache_size: int,
    slot_ids: Iterable[str] | None = None,
) -> tuple[PipelineResult, float, FramePointProvider]:
    config = _configured(base_config, frames)
    provider = FramePointProvider(
        {frame.frame_id: frame for frame in frames},
        points_dir,
        cache_size=max(cache_size, len(frames)),
        project_root=PROJECT_ROOT,
    )
    started = time.perf_counter()
    result = Hybrid3DPipeline(
        slots,
        frames,
        provider,
        scale,
        config,
        phase="full",
        slot_ids=tuple(slot_ids) if slot_ids is not None else None,
    ).run()
    return result, time.perf_counter() - started, provider


def _localize(result: PipelineResult, provider: FramePointProvider) -> PipelineResult:
    settings = LocalMapConfig()
    coverage = actual_lidar_coverage_polygon(
        result.frames,
        provider,
        result.map_units_per_meter,
        settings.local_radius_m,
    )
    snapshot = build_local_map_snapshot(
        result,
        settings,
        coverage_polygon_map=coverage,
    )
    return restrict_result_to_local_map(result, snapshot)


def _counts(decisions: Iterable[SlotDecision]) -> dict[str, int]:
    counts = Counter(decision.state.value for decision in decisions)
    return {name: int(counts.get(name, 0)) for name in ("free", "occupied", "unknown")}


def _record(decision: SlotDecision | None) -> dict[str, Any]:
    if decision is None:
        return {"state": "unknown", "decision_reason": "missing_from_sampled_scope", "unknown_reasons": ["missing_from_sampled_scope"]}
    features = decision.occupied_evidence.features
    return {
        "state": decision.state.value,
        "decision_reason": decision.decision_reason,
        "unknown_reasons": list(decision.unknown_reasons),
        "stability_pass_ratio": float(decision.stability.pass_ratio),
        "occupied_failures": list(decision.occupied_evidence.failures),
        "free_failures": list(decision.free_evidence.failures),
        "occupied_features": None if features is None else {
            "robust_pca_linearity": float(features.robust_pca_linearity),
            "robust_extent_x_m": float(features.robust_extent_x_m),
            "robust_extent_y_m": float(features.robust_extent_y_m),
            "extent_z_m": float(features.extent_z_m),
            "robust_point_fraction": float(features.robust_point_fraction),
            "boundary_ratio": float(features.boundary_ratio),
        },
    }


def _candidate_ids(result: PipelineResult, config: Hybrid3DConfig) -> tuple[str, ...]:
    current = max(result.frames, key=lambda frame: frame.frame_id)
    slots = {slot.slot_id: slot for slot in result.all_slots}
    selected: list[str] = []
    for decision in result.decisions:
        if decision.state is not DecisionState.UNKNOWN or not decision.agent_context.agent_observable:
            continue
        accepted, _ = part2_candidate_in_forward_fov(
            current, slots[decision.slot_id].center_map, config.part2_candidate_half_fov_deg
        )
        if accepted:
            selected.append(decision.slot_id)
    return tuple(sorted(selected))


def _paired_row(
    anchor: int,
    baseline: PipelineResult,
    improved: PipelineResult,
    baseline_seconds: float,
    improved_seconds: float,
    baseline_config: Hybrid3DConfig,
    improved_config: Hybrid3DConfig,
) -> dict[str, Any]:
    base = {item.slot_id: item for item in baseline.decisions}
    agent = {item.slot_id: item for item in improved.decisions}
    slot_by_id = {slot.slot_id: slot for slot in improved.all_slots}
    current = max(improved.frames, key=lambda frame: frame.frame_id)
    transitions: list[dict[str, Any]] = []
    for slot_id in sorted(set(base) | set(agent)):
        left = base.get(slot_id)
        right = agent.get(slot_id)
        before = "missing" if left is None else left.state.value
        after = "missing" if right is None else right.state.value
        if before != after:
            transitions.append({"slot_id": slot_id, "baseline_state": before, "agent_state": after, "agent_record": _record(right)})
    baseline_unknown = [item for item in baseline.decisions if item.state is DecisionState.UNKNOWN and item.agent_context.agent_observable]
    front_count = 0
    for item in baseline_unknown:
        accepted, _ = part2_candidate_in_forward_fov(
            current, slot_by_id[item.slot_id].center_map, improved_config.part2_candidate_half_fov_deg
        )
        front_count += int(accepted)
    return {
        "anchor_frame": anchor,
        "baseline_counts": _counts(baseline.decisions),
        "agent_counts": _counts(improved.decisions),
        "baseline_seconds": baseline_seconds,
        "agent_seconds": improved_seconds,
        "state_transitions": transitions,
        "baseline_observable_unknown": len(baseline_unknown),
        "baseline_part2_candidates_360deg": len(_candidate_ids(baseline, baseline_config)),
        "same_unknowns_in_forward_180deg": front_count,
        "agent_part2_candidates_forward_180deg": len(_candidate_ids(improved, improved_config)),
    }


def _sample_run_row(
    anchor: int,
    history_span: int,
    sample_count: int,
    selected: tuple[FrameRecord, ...],
    candidate_ids: tuple[str, ...],
    result: PipelineResult,
    seconds: float,
) -> dict[str, Any]:
    by_id = {item.slot_id: item for item in result.decisions}
    rows = {slot_id: _record(by_id.get(slot_id)) for slot_id in candidate_ids}
    counts = Counter(row["state"] for row in rows.values())
    return {
        "anchor_frame": anchor,
        "history_span_W": history_span,
        "sample_count_K": sample_count,
        "sampling": "uniform_inclusive_anchor",
        "selected_frame_ids": [frame.frame_id for frame in selected],
        "candidate_count": len(candidate_ids),
        "state_counts": {name: int(counts.get(name, 0)) for name in ("free", "occupied", "unknown")},
        "resolved_count": int(counts.get("free", 0) + counts.get("occupied", 0)),
        "processing_seconds": seconds,
        "rows": rows,
    }


def _aggregate(experiments: list[dict[str, Any]], dense: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    keys = sorted({(row["history_span_W"], row["sample_count_K"]) for row in experiments})
    aggregates: list[dict[str, Any]] = []
    for history_span, sample_count in keys:
        selected = [row for row in experiments if (row["history_span_W"], row["sample_count_K"]) == (history_span, sample_count)]
        total = resolved = exact = terminal_agree = contradictions = 0
        runtime = 0.0
        for run in selected:
            runtime += float(run["processing_seconds"])
            reference = dense[int(run["anchor_frame"])]
            for slot_id, record in run["rows"].items():
                state = str(record["state"])
                target = reference.get(slot_id, "unknown")
                total += 1
                resolved += int(state in {"free", "occupied"})
                exact += int(state == target)
                terminal_agree += int(state == target and state in {"free", "occupied"})
                contradictions += int(state in {"free", "occupied"} and target in {"free", "occupied"} and state != target)
        aggregates.append({
            "history_span_W": history_span,
            "sample_count_K": sample_count,
            "case_count": total,
            "resolved_count": resolved,
            "resolved_rate": resolved / total if total else 0.0,
            "exact_dense_agreement_count": exact,
            "exact_dense_agreement_rate": exact / total if total else 0.0,
            "terminal_dense_agreement_count": terminal_agree,
            "terminal_contradiction_count": contradictions,
            "processing_seconds_total": runtime,
        })
    return aggregates


def _best(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    # Preregistered lexicographic objective: safety first, useful agreements
    # second, exact dense-reference agreement third, then lower K/W/runtime.
    return max(
        aggregates,
        key=lambda row: (
            -int(row["terminal_contradiction_count"]),
            int(row["terminal_dense_agreement_count"]),
            int(row["exact_dense_agreement_count"]),
            -int(row["sample_count_K"]),
            -int(row["history_span_W"]),
            -float(row["processing_seconds_total"]),
        ),
    )


def main() -> int:
    args = _parser().parse_args()
    if set(args.development_anchors) & set(args.heldout_anchors):
        raise ValueError("development and held-out anchors must be disjoint")
    if args.dense_reference_frames < max(args.history_spans):
        raise ValueError("dense reference must cover the largest history span")
    combos = sorted({(w, k) for w in args.history_spans for k in args.sample_counts if 3 <= k <= w})
    all_frames = load_frame_records(args.frames_csv)
    slots, scale = load_known_slots(args.slot_db)
    improved_config = Hybrid3DConfig()
    baseline_config = replace(
        improved_config,
        occupied_pillar_min_height_m=1000.0,
        part2_candidate_half_fov_deg=180.0,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "ground_truth_available": False,
        "dense_reference_is_ground_truth": False,
        "development_anchors": list(args.development_anchors),
        "heldout_anchors": list(args.heldout_anchors),
        "history_sample_combinations": [{"W": w, "K": k} for w, k in combos],
        "baseline_config": baseline_config.to_dict(),
        "agent_config": improved_config.to_dict(),
        "paired_part1": [],
        "development_experiments": [],
        "heldout_experiments": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, payload)

    development_dense: dict[int, dict[str, str]] = {}
    heldout_dense: dict[int, dict[str, str]] = {}
    def part1_pair(anchor: int) -> tuple[PipelineResult, tuple[str, ...]]:
        frames15 = _causal_pool(all_frames, anchor, 15)
        base_raw, base_seconds, base_provider = _run(slots, scale, frames15, args.map_points_dir, baseline_config, args.cache_size)
        base_local = _localize(base_raw, base_provider)
        improved_raw, improved_seconds, improved_provider = _run(slots, scale, frames15, args.map_points_dir, improved_config, args.cache_size)
        improved_local = _localize(improved_raw, improved_provider)
        payload["paired_part1"].append(_paired_row(anchor, base_local, improved_local, base_seconds, improved_seconds, baseline_config, improved_config))
        candidate_ids = _candidate_ids(improved_local, improved_config)
        write_json_atomic(args.output, payload)
        print(json.dumps({"stage": "part1_pair", "anchor": anchor, "candidates_front180": len(candidate_ids), "baseline": _counts(base_local.decisions), "agent": _counts(improved_local.decisions)}), flush=True)
        return improved_local, candidate_ids

    for anchor in args.development_anchors:
        _, candidate_ids = part1_pair(anchor)
        dense_frames = _causal_pool(all_frames, anchor, args.dense_reference_frames)
        dense_result, dense_seconds, _ = _run(slots, scale, dense_frames, args.map_points_dir, improved_config, args.cache_size, candidate_ids)
        dense_by_id = {item.slot_id: item.state.value for item in dense_result.decisions}
        development_dense[anchor] = {slot_id: dense_by_id.get(slot_id, "unknown") for slot_id in candidate_ids}
        for history_span, sample_count in combos:
            pool = _causal_pool(all_frames, anchor, history_span)
            selected = _uniform_sample(pool, sample_count)
            result, seconds, _ = _run(slots, scale, selected, args.map_points_dir, improved_config, args.cache_size, candidate_ids)
            row = _sample_run_row(anchor, history_span, sample_count, selected, candidate_ids, result, seconds)
            payload["development_experiments"].append(row)
            write_json_atomic(args.output, payload)
            print(json.dumps({"stage": "development", "anchor": anchor, "W": history_span, "K": sample_count, "resolved": row["resolved_count"], "candidates": len(candidate_ids), "seconds": round(seconds, 3)}), flush=True)

    development_aggregate = _aggregate(payload["development_experiments"], development_dense)
    selected_best = _best(development_aggregate)
    payload["development_dense_reference"] = development_dense
    payload["development_aggregate"] = development_aggregate
    payload["selected_best"] = selected_best
    write_json_atomic(args.output, payload)
    best_pair = (int(selected_best["history_span_W"]), int(selected_best["sample_count_K"]))
    validation_pairs = tuple(dict.fromkeys(((15, 15), best_pair)))

    for anchor in args.heldout_anchors:
        _, candidate_ids = part1_pair(anchor)
        dense_frames = _causal_pool(all_frames, anchor, args.dense_reference_frames)
        dense_result, _, _ = _run(slots, scale, dense_frames, args.map_points_dir, improved_config, args.cache_size, candidate_ids)
        dense_by_id = {item.slot_id: item.state.value for item in dense_result.decisions}
        heldout_dense[anchor] = {slot_id: dense_by_id.get(slot_id, "unknown") for slot_id in candidate_ids}
        for history_span, sample_count in validation_pairs:
            selected = _uniform_sample(_causal_pool(all_frames, anchor, history_span), sample_count)
            result, seconds, _ = _run(slots, scale, selected, args.map_points_dir, improved_config, args.cache_size, candidate_ids)
            row = _sample_run_row(anchor, history_span, sample_count, selected, candidate_ids, result, seconds)
            payload["heldout_experiments"].append(row)
            write_json_atomic(args.output, payload)
            print(json.dumps({"stage": "heldout", "anchor": anchor, "W": history_span, "K": sample_count, "resolved": row["resolved_count"], "candidates": len(candidate_ids), "seconds": round(seconds, 3)}), flush=True)

    payload["heldout_dense_reference"] = heldout_dense
    payload["heldout_aggregate"] = _aggregate(payload["heldout_experiments"], heldout_dense)
    payload["status"] = "complete"
    payload["completed_unix_time"] = time.time()
    write_json_atomic(args.output, payload)
    print(json.dumps({"stage": "complete", "best_W": best_pair[0], "best_K": best_pair[1], "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
