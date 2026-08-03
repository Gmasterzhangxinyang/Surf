#!/usr/bin/env python3
"""Build prediction-blind LiDAR review views for an independent GT universe."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_gt_neutral_lidar_views import _contact_sheets, _render
from parking_slot_hybrid_3d.accumulation import build_slot_accumulation
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import ScopeEvidence, ScopeStatus
from parking_slot_hybrid_3d.geometry import metric_slot
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    frame_ids = tuple(int(value) for value in universe["history_frames"])
    if not frame_ids or any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("universe history_frames must be strictly increasing")
    anchor = int(universe["anchor_frame"])
    if frame_ids[-1] != anchor or any(value > anchor for value in frame_ids):
        raise ValueError("universe frame window is not causal at the anchor")
    target_ids = {str(row["slot_id"]) for row in universe["slots"]}

    all_frames = load_frame_records(args.frames_csv)
    frames = [frame for frame in all_frames if frame.frame_id in set(frame_ids)]
    if tuple(frame.frame_id for frame in frames) != frame_ids:
        raise ValueError("frames CSV does not contain the complete universe window")
    slots, scale = load_known_slots(args.slot_db)
    selected_slots = [slot for slot in slots if slot.slot_id in target_ids]
    if {slot.slot_id for slot in selected_slots} != target_ids:
        raise ValueError("slot database does not contain the complete universe")

    provider = FramePointProvider(
        {frame.frame_id: frame for frame in frames},
        args.map_points_dir,
        cache_size=max(64, len(frames)),
    )
    config = replace(
        Hybrid3DConfig(),
        frame_stride=1,
        window_before=len(frames) - 1,
        window_after=0,
    )
    packs = args.output_dir / "evidence_packs"
    images = args.output_dir / "slot_views"
    packs.mkdir(parents=True)
    images.mkdir(parents=True)
    rows: list[dict] = []
    for slot in selected_slots:
        # This synthetic scope fixes the accumulation anchor only.  It is not
        # a Part1 observability or state decision and is never exported as GT.
        scope = ScopeEvidence(
            slot_id=slot.slot_id,
            scope_status=ScopeStatus.IN_ROUTE,
            near_frames=frame_ids,
            crossing_frames=(anchor,),
            agent_observable=False,
            reasons=("gt_neutral_forced_causal_accumulation",),
        )
        accumulation = build_slot_accumulation(
            slot,
            scope,
            frames,
            provider,
            scale,
            config,
        )
        converted = metric_slot(slot, scale)
        pack_path = packs / f"{slot.slot_id}.npz"
        np.savez_compressed(
            pack_path,
            points_local_xyzi=accumulation.points_local_xyzi,
            point_frame_ids=accumulation.point_frame_ids,
            polygon_local_m=converted.polygon_local_m,
            core_polygon_local_m=converted.core_polygon_local_m,
            margin_polygon_local_m=converted.margin_polygon_local_m,
            selected_frames=np.asarray(accumulation.selected_frames, dtype=np.int64),
            valid_frames=np.asarray(
                [row.frame_id for row in accumulation.observations],
                dtype=np.int64,
            ),
        )
        row = _render(
            pack_path,
            slot.slot_id,
            images / f"{slot.slot_id}.png",
        )
        row["excluded_frames"] = [
            [int(frame_id), list(reasons)]
            for frame_id, reasons in accumulation.excluded_frames
        ]
        rows.append(row)

    sheets = _contact_sheets(rows, args.output_dir / "contact_sheets")
    manifest = {
        "schema_version": "independent-gt-neutral-lidar/1.0",
        "prediction_blind": True,
        "part1_scope_used": False,
        "part1_state_used": False,
        "anchor_frame": anchor,
        "causal_frames": list(frame_ids),
        "slot_count": len(rows),
        "provider_stats": provider.stats,
        "cases": rows,
        "contact_sheets": sheets,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "slot_count": len(rows),
                "contact_sheet_count": len(sheets),
                "provider_stats": provider.stats,
                "prediction_blind": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
