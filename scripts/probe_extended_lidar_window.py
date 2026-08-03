#!/usr/bin/env python3
"""Audit a longer causal LiDAR window without changing the 15-frame Part1 contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

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


TARGETS = {
    "slot_1010", "slot_1251", "slot_1250", "slot_1011", "slot_1257", "slot_1012",
    "slot_1258", "slot_1255", "slot_1254", "slot_1014", "slot_1015", "slot_1016",
    "slot_1248", "slot_1017", "slot_1247", "slot_1267", "slot_1018", "slot_0968",
    "slot_1268", "slot_1246", "slot_1035", "slot_1269",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=75)
    parser.add_argument("--anchor", type=int, default=9277)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.window < 15:
        raise ValueError("window must be at least 15")
    frames = load_frame_records(
        PROJECT_ROOT / "outputs/frame_map_dataset_pose_corrected_final/frames.csv"
    )
    selected = tuple(row for row in frames if row.frame_id <= args.anchor)[-args.window :]
    if len(selected) != args.window:
        raise ValueError("not enough causal frames")
    slots, scale = load_known_slots(
        PROJECT_ROOT / "outputs/full_icpark_allframes_vehicle_cluster/slot_database.json"
    )
    span = selected[-1].frame_id - selected[0].frame_id
    base = Hybrid3DConfig()
    config = replace(
        base,
        scope_max_distance_m=30.0,
        frame_stride=1,
        window_before=max(base.window_before, span),
        window_after=max(base.window_after, span),
    )
    provider = FramePointProvider(
        {row.frame_id: row for row in selected},
        PROJECT_ROOT / "outputs/frame_map_dataset_pose_corrected_final/map_points",
        cache_size=max(96, args.window),
        project_root=PROJECT_ROOT,
    )
    started = time.monotonic()
    result = Hybrid3DPipeline(slots, selected, provider, scale, config, phase="full").run()
    rows = []
    for decision in result.decisions:
        if decision.slot_id not in TARGETS:
            continue
        occupied = decision.occupied_evidence
        free = decision.free_evidence
        features = occupied.features
        rows.append(
            {
                "slot_id": decision.slot_id,
                "state": decision.state.value,
                "decision_reason": decision.decision_reason,
                "unknown_reasons": list(decision.unknown_reasons),
                "occupied": {
                    "strength_uncalibrated": occupied.strength,
                    "strong": occupied.strong,
                    "weak": occupied.weak,
                    "failures": list(occupied.failures),
                    "support_frame_count": occupied.support_frame_count,
                    "boundary_ratio": None if features is None else features.boundary_ratio,
                    "core_point_count": None if features is None else features.core_point_count,
                    "linearity": None if features is None else features.linearity,
                },
                "free": {
                    "strength_uncalibrated": free.strength,
                    "strong": free.strong,
                    "failures": list(free.failures),
                    "viewpoint_count": free.viewpoint_count,
                    "viewpoint_separation_deg": free.viewpoint_separation_deg,
                    "observed_volume_ratio": free.observed_volume_ratio,
                    "near_ground_bev_coverage": free.near_ground_bev_coverage,
                    "occlusion_ratio": free.occlusion_ratio,
                    "unresolved_core_hit": free.unresolved_core_hit,
                },
                "stability": {
                    "stable": decision.stability.stable,
                    "failures": list(decision.stability.failures),
                    "passing_variants": decision.stability.passing_variants,
                    "total_variants": decision.stability.total_variants,
                },
            }
        )
    payload = {
        "schema_version": "parking-slot-agent-v2-extended-window-probe/1.0",
        "window": args.window,
        "first_frame": selected[0].frame_id,
        "anchor_frame": selected[-1].frame_id,
        "elapsed_seconds": time.monotonic() - started,
        "target_count": len(rows),
        "rows": sorted(rows, key=lambda item: item["slot_id"]),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
