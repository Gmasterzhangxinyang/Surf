#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.io import FramePointProvider, load_frame_records, load_known_slots
from parking_slot_hybrid_3d.local_map import LocalMapConfig, select_local_frame_window
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline
from parking_slot_hybrid_3d.reporting import OutputContext, write_pipeline_outputs
from parking_slot_hybrid_3d.static_semantics import SemanticStaticOccupiedVeto
from gltf_lidar_ndt import load_gltf_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run conservative Part1 hybrid-3D parking-slot evidence on a causal "
            "local window of 3-15 consecutive LiDAR frames."
        ),
    )
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--map-points-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("scope", "occupied", "full"), default="full")
    parser.add_argument("--slot-ids", default="", help="Comma-separated known slot IDs.")
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--dataset-id", default="pose-corrected-dataset")
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument(
        "--gltf-static-map",
        type=Path,
        help="Optional independent wall/elevator/arrester semantic map for occupied veto.",
    )
    parser.add_argument("--static-map-sample-step", type=float, default=0.012)
    parser.add_argument("--static-map-distance-m", type=float, default=0.35)
    parser.add_argument("--static-map-max-explained-ratio", type=float, default=0.50)
    parser.add_argument("--static-map-min-residual-short-extent-m", type=float, default=0.75)
    parser.add_argument(
        "--camera-projection-audit",
        type=Path,
        help="Optional hash-bound projection audit; absent or non-passing keeps RGB fail-closed.",
    )
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument(
        "--anchor-frame",
        type=int,
        help=(
            "Frame ID at which the causal local window ends; defaults to the "
            "last frame in --frames-csv."
        ),
    )
    parser.add_argument(
        "--local-frame-count",
        type=int,
        choices=range(3, 16),
        default=LocalMapConfig().frame_count,
        metavar="N",
        help=(
            "Maximum number of consecutive LiDAR frames in the local Part1 "
            "window (3-15; default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--history-span",
        type=int,
        help="Causal pool size W ending at --anchor-frame; requires --history-sample-count.",
    )
    parser.add_argument(
        "--history-sample-count",
        type=int,
        help="Uniform inclusive-anchor sample count K from --history-span.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_config(path: Path | None) -> Hybrid3DConfig:
    if path is None:
        config = Hybrid3DConfig()
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("config JSON must contain an object")
        config = Hybrid3DConfig(**payload)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config_json)
    all_frames = load_frame_records(args.frames_csv)
    if (args.history_span is None) != (args.history_sample_count is None):
        raise ValueError("--history-span and --history-sample-count must be supplied together")
    if args.history_span is None:
        frames = select_local_frame_window(
            all_frames,
            anchor_frame_id=args.anchor_frame,
            frame_count=args.local_frame_count,
        )
    else:
        if args.history_span < 3:
            raise ValueError("--history-span must be at least 3")
        if not 3 <= args.history_sample_count <= args.history_span:
            raise ValueError("--history-sample-count must be in [3, history-span]")
        anchor = args.anchor_frame if args.anchor_frame is not None else all_frames[-1].frame_id
        pool = tuple(frame for frame in all_frames if frame.frame_id <= anchor)[-args.history_span :]
        if len(pool) != args.history_span or pool[-1].frame_id != anchor:
            raise ValueError(f"anchor {anchor} lacks a complete causal history pool")
        if args.history_sample_count == args.history_span:
            frames = pool
        else:
            indices = np.rint(
                np.linspace(0, len(pool) - 1, num=args.history_sample_count)
            ).astype(np.int64)
            if len(set(indices.tolist())) != args.history_sample_count:
                raise RuntimeError("uniform history sampler produced duplicate indices")
            frames = tuple(pool[int(index)] for index in indices)
        if frames[-1].frame_id != anchor:
            raise RuntimeError("uniform history sampler omitted the causal anchor")
    # The local evidence window is already hard-bounded. Consume every
    # selected record (also when frame IDs have gaps) instead of inheriting the
    # historical full-route replay stride.
    frame_id_span = int(frames[-1].frame_id - frames[0].frame_id)
    config = replace(
        config,
        frame_stride=1,
        window_before=max(int(config.window_before), frame_id_span),
        window_after=max(int(config.window_after), frame_id_span),
    )
    config.validate()
    slots, map_units_per_meter = load_known_slots(args.slot_db)
    provider = FramePointProvider(
        {frame.frame_id: frame for frame in frames},
        args.map_points_dir,
        cache_size=args.cache_size,
        project_root=PROJECT_ROOT,
    )
    slot_ids = tuple(
        value.strip() for value in args.slot_ids.split(",") if value.strip()
    ) or None
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
    result = Hybrid3DPipeline(
        slots,
        frames,
        provider,
        map_units_per_meter,
        config,
        phase=args.phase,
        slot_ids=slot_ids,
        max_slots=args.max_slots,
        static_occupied_veto=static_occupied_veto,
    ).run()
    summary = write_pipeline_outputs(
        result,
        args.output_dir,
        config,
        OutputContext(
            dataset_id=args.dataset_id,
            slot_database_path=args.slot_db,
            frames_csv_path=args.frames_csv,
            map_points_dir=args.map_points_dir,
            camera_calibration_path=args.camera_calibration,
            camera_projection_audit_path=args.camera_projection_audit,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
