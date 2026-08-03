#!/usr/bin/env python3
"""Run reproducible causal-window and cross-anchor hard-gate ablations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_agent_v2.lidar_geometry import (
    assess_terminal_geometry,
    build_geometry_card,
)
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_json_atomic,
)
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline


def _state(gate: dict[str, object]) -> str:
    if gate["free_eligible"]:
        return "free"
    if gate["occupied_eligible"]:
        return "occupied"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-description", required=True)
    args = parser.parse_args()
    if any(window < 15 for window in args.windows):
        raise ValueError("every window must contain at least 15 frames")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_by_anchor = {
        int(row["anchor_frame"]): row for row in manifest["anchors"]
    }
    missing = sorted(set(args.anchors) - set(manifest_by_anchor))
    if missing:
        raise ValueError(f"anchors absent from manifest: {missing}")
    all_frames = load_frame_records(args.frames_csv)
    slots, scale = load_known_slots(args.slot_db)
    experiments: list[dict[str, object]] = []

    for anchor in args.anchors:
        part1_path = Path(manifest_by_anchor[anchor]["part1_output"])
        part1 = Part1Output.from_dict(
            json.loads(part1_path.read_text(encoding="utf-8"))
        )
        original_unknown = {case.slot_id: case for case in part1.unknown_cases}
        for window in sorted(set(args.windows)):
            selected = tuple(
                frame for frame in all_frames if int(frame.frame_id) <= anchor
            )[-window:]
            if len(selected) != window or int(selected[-1].frame_id) != anchor:
                raise ValueError(f"anchor {anchor} lacks a {window}-frame causal window")
            span = int(selected[-1].frame_id) - int(selected[0].frame_id)
            base = Hybrid3DConfig()
            config = replace(
                base,
                scope_max_distance_m=float(part1.scene.radius_m),
                frame_stride=1,
                window_before=max(base.window_before, span),
                window_after=max(base.window_after, span),
            )
            provider = FramePointProvider(
                {int(frame.frame_id): frame for frame in selected},
                args.map_points_dir,
                cache_size=max(96, window),
                project_root=PROJECT_ROOT,
            )
            started = time.monotonic()
            result = Hybrid3DPipeline(
                slots, selected, provider, scale, config, phase="full"
            ).run()
            elapsed = time.monotonic() - started
            decision_by_id = {row.slot_id: row for row in result.decisions}
            rows: list[dict[str, object]] = []
            for slot_id, case in original_unknown.items():
                decision = decision_by_id.get(slot_id)
                if decision is None:
                    rows.append(
                        {
                            "slot_id": slot_id,
                            "gate_state": "unknown",
                            "raw_extended_state": "missing",
                            "gate": {
                                "free_eligible": False,
                                "occupied_eligible": False,
                                "free_blockers": ["extended_pipeline_missing"],
                                "occupied_blockers": ["extended_pipeline_missing"],
                            },
                        }
                    )
                    continue
                decision_payload = asdict(decision)
                decision_payload["selected_frames"] = [
                    int(frame.frame_id) for frame in selected
                ]
                pack = SimpleNamespace(
                    valid_frames=np.asarray(
                        [int(frame.frame_id) for frame in selected], dtype=np.int64
                    ),
                    points_local_xyzi=np.empty((0, 4), dtype=np.float32),
                )
                card = build_geometry_card(
                    case,
                    pack,
                    decision_override=decision_payload,
                    decision_source="part2_extended_causal_lidar_ablation",
                )
                gate = assess_terminal_geometry(card)
                rows.append(
                    {
                        "slot_id": slot_id,
                        "gate_state": _state(gate),
                        "raw_extended_state": decision.state.value,
                        "decision_reason": decision.decision_reason,
                        "unknown_reasons": list(decision.unknown_reasons),
                        "geometry_card": card,
                        "gate": gate,
                    }
                )
            gate_counts = Counter(str(row["gate_state"]) for row in rows)
            raw_counts = Counter(str(row["raw_extended_state"]) for row in rows)
            resolved = gate_counts["free"] + gate_counts["occupied"]
            experiments.append(
                {
                    "anchor_frame": anchor,
                    "window_frames": window,
                    "first_frame": int(selected[0].frame_id),
                    "part1_path": str(part1_path.resolve()),
                    "part1_unknown_count": len(original_unknown),
                    "gate_state_counts": dict(gate_counts),
                    "raw_extended_state_counts": dict(raw_counts),
                    "resolved_count": resolved,
                    "resolved_rate": resolved / len(original_unknown) if original_unknown else 0.0,
                    "processing_seconds": elapsed,
                    "rows": rows,
                }
            )
            print(
                json.dumps(
                    {
                        "anchor": anchor,
                        "window": window,
                        "unknown": len(original_unknown),
                        "resolved": resolved,
                        "states": dict(gate_counts),
                        "seconds": round(elapsed, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    payload = {
        "schema_version": "parking-slot-agent-v2-extended-lidar-ablation/1.0",
        "selection_description": args.selection_description,
        "anchors": list(args.anchors),
        "windows": sorted(set(args.windows)),
        "strictly_causal": True,
        "terminal_threshold": 0.90,
        "ground_truth_available": False,
        "experiments": experiments,
    }
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
