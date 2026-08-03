#!/usr/bin/env python3
"""Run a causal Part1 ablation over slot-core localization uncertainty widths."""

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
from parking_slot_hybrid_3d.contracts import DecisionState
from parking_slot_hybrid_3d.io import load_frame_records, load_known_slots, write_json_atomic
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
    result.add_argument("--anchors", type=int, nargs="+", default=[9277])
    result.add_argument("--widths-m", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.30])
    result.add_argument("--frame-count", type=int, default=15)
    return result


def record(decision: Any) -> dict[str, Any]:
    features = decision.occupied_evidence.features
    return {
        "state": decision.state.value,
        "decision_reason": decision.decision_reason,
        "unknown_reasons": list(decision.unknown_reasons),
        "occupied_strong": bool(decision.occupied_evidence.strong),
        "occupied_strength": float(decision.occupied_evidence.strength),
        "occupied_failures": list(decision.occupied_evidence.failures),
        "core_overlap": None if features is None else float(features.core_overlap),
        "boundary_ratio": None if features is None else float(features.boundary_ratio),
        "stability_pass_ratio": float(decision.stability.pass_ratio),
        "stability_stable": bool(decision.stability.stable),
    }


def main() -> None:
    args = parser().parse_args()
    widths = sorted(set(float(value) for value in args.widths_m))
    if not widths or widths[0] != 0.0:
        raise ValueError("widths-m must include 0.0 as the reference")
    frames = load_frame_records(args.frames_csv)
    slots, scale = load_known_slots(args.slot_db)
    runs: list[dict[str, Any]] = []
    for anchor in args.anchors:
        selected = _causal_pool(frames, int(anchor), int(args.frame_count))
        reference: dict[str, dict[str, Any]] | None = None
        for width in widths:
            config = replace(
                Hybrid3DConfig(),
                frame_stride=1,
                window_before=max(Hybrid3DConfig().window_before, args.frame_count - 1),
                window_after=max(Hybrid3DConfig().window_after, args.frame_count - 1),
                occupied_localization_uncertainty_m=width,
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
                            "uncertainty_state": after["state"],
                            "reference": before,
                            "uncertainty": after,
                        }
                    )
            occupied_to_unknown = sum(
                row["reference_state"] == "occupied" and row["uncertainty_state"] == "unknown"
                for row in transitions
            )
            unsafe_promotions = sum(
                row["reference_state"] in ("free", "unknown")
                and row["uncertainty_state"] == "occupied"
                for row in transitions
            )
            runs.append(
                {
                    "anchor_frame": int(anchor),
                    "uncertainty_width_m": width,
                    "selected_frames": [frame.frame_id for frame in selected],
                    "counts": _counts(local.decisions),
                    "candidate_count": len(_candidate_ids(local, config)),
                    "observable_unknown": sum(
                        decision.state is DecisionState.UNKNOWN
                        and decision.agent_context.agent_observable
                        for decision in local.decisions
                    ),
                    "occupied_to_unknown": occupied_to_unknown,
                    "unsafe_promotions_to_occupied": unsafe_promotions,
                    "transitions_from_width0": transitions,
                    "decisions": current,
                    "pipeline_seconds": pipeline_seconds,
                    "total_seconds": time.perf_counter() - started,
                }
            )
            print(
                json.dumps(
                    {
                        "anchor": anchor,
                        "width_m": width,
                        "counts": runs[-1]["counts"],
                        "transitions": len(transitions),
                        "unsafe_promotions": unsafe_promotions,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    payload = {
        "schema_version": "parkingagent-localization-uncertainty-ablation/1.0",
        "status": "complete",
        "ground_truth_available": False,
        "interpretation": "Stricter ownership safety ablation; state changes are not accuracy without GT.",
        "anchors": [int(value) for value in args.anchors],
        "widths_m": widths,
        "frame_count": int(args.frame_count),
        "runs": runs,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "run_count": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
