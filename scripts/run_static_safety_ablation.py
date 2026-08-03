#!/usr/bin/env python3
"""Run a causal multi-anchor ablation for non-bypassable static-structure gates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.io import load_frame_records, load_known_slots, write_json_atomic
from scripts.run_localization_uncertainty_ablation import record
from scripts.run_systematic_candidate_experiments import (
    _candidate_ids,
    _causal_pool,
    _counts,
    _localize,
    _run,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frames-csv", type=Path, required=True)
    result.add_argument("--slot-db", type=Path, required=True)
    result.add_argument("--map-points-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=[234, 3160, 5283, 7605, 9277, 9443],
    )
    result.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15],
    )
    result.add_argument("--localization-width-m", type=float, default=0.10)
    result.add_argument("--frame-count", type=int, default=15)
    return result


def main() -> None:
    args = parser().parse_args()
    ratios = sorted(set(float(value) for value in args.ratios))
    if not ratios or ratios[0] != 0.0:
        raise ValueError("ratios must include 0.0 as the reference")
    frames = load_frame_records(args.frames_csv)
    slots, scale = load_known_slots(args.slot_db)
    runs: list[dict[str, Any]] = []
    for anchor in args.anchors:
        selected = _causal_pool(frames, int(anchor), int(args.frame_count))
        reference: dict[str, dict[str, Any]] | None = None
        for ratio in ratios:
            config = replace(
                Hybrid3DConfig(),
                frame_stride=1,
                window_before=max(Hybrid3DConfig().window_before, args.frame_count - 1),
                window_after=max(Hybrid3DConfig().window_after, args.frame_count - 1),
                occupied_localization_uncertainty_m=float(args.localization_width_m),
                occupied_min_upper_height_spread_ratio=ratio,
            )
            started = time.perf_counter()
            result, pipeline_seconds, provider = _run(
                slots,
                scale,
                selected,
                args.map_points_dir,
                config,
                cache_size=max(64, args.frame_count),
            )
            local = _localize(result, provider)
            current = {decision.slot_id: record(decision) for decision in local.decisions}
            if reference is None:
                reference = current
            transitions = []
            for slot_id in sorted(set(reference) | set(current)):
                before = reference.get(slot_id, {"state": "missing"})
                after = current.get(slot_id, {"state": "missing"})
                if before["state"] != after["state"]:
                    transitions.append(
                        {
                            "slot_id": slot_id,
                            "reference_state": before["state"],
                            "safety_state": after["state"],
                            "reference": before,
                            "safety": after,
                        }
                    )
            unsafe_promotions = sum(
                row["reference_state"] in {"free", "unknown"}
                and row["safety_state"] == "occupied"
                for row in transitions
            )
            occupied_to_unknown = sum(
                row["reference_state"] == "occupied"
                and row["safety_state"] == "unknown"
                for row in transitions
            )
            row = {
                "anchor_frame": int(anchor),
                "upper_height_spread_ratio": ratio,
                "localization_width_m": float(args.localization_width_m),
                "selected_frames": [frame.frame_id for frame in selected],
                "counts": _counts(local.decisions),
                "candidate_count": len(_candidate_ids(local, config)),
                "transition_count": len(transitions),
                "occupied_to_unknown": occupied_to_unknown,
                "unsafe_promotions_to_occupied": unsafe_promotions,
                "transitions_from_ratio0": transitions,
                "pipeline_seconds": pipeline_seconds,
                "total_seconds": time.perf_counter() - started,
            }
            runs.append(row)
            print(
                json.dumps(
                    {
                        "anchor": anchor,
                        "ratio": ratio,
                        "counts": row["counts"],
                        "transitions": len(transitions),
                        "occupied_to_unknown": occupied_to_unknown,
                        "unsafe_promotions": unsafe_promotions,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_json_atomic(
        args.output,
        {
            "schema_version": "parkingagent-static-safety-ablation/1.0",
            "status": "complete",
            "ground_truth_available": False,
            "interpretation": "Safety/workload sensitivity only; not accuracy without independent labels.",
            "anchors": [int(value) for value in args.anchors],
            "ratios": ratios,
            "localization_width_m": float(args.localization_width_m),
            "frame_count": int(args.frame_count),
            "non_bypassable_gates": [
                "compact_vertical_structure",
                "compact_vertical_footprint",
                "upper_height_spread",
            ],
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
