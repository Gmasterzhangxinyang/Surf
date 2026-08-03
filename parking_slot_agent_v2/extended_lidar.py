"""Build identity-bound, strictly causal extended LiDAR evidence for Part2.

Part1 deliberately remains a 15-frame contract.  This module adds a separate
Part2 resource which may inspect a longer history ending at the same t0.  The
resource is bound to the slot id, selected source files, configuration hash,
map hash, and anchor frame by the existing evidence-pack format.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from parking_slot_hybrid_3d.accumulation import build_slot_accumulation
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.geometry import map_xy_to_slot_m, metric_slot
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_json_atomic,
)
from parking_slot_hybrid_3d.part2_evidence import build_lidar_evidence_pack
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline
from parking_slot_part2 import canonical_sha256

from .contracts import Part1Output


SCHEMA_VERSION = "parking-slot-agent-v2-extended-causal-lidar/1.0"


def _map_points_source(frame: Any, map_points_dir: Path, project_root: Path) -> Path:
    candidates: list[Path] = []
    raw = getattr(frame, "map_points_path", None)
    if raw is not None:
        candidate = Path(raw)
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.extend((project_root / candidate, map_points_dir / candidate))
    candidates.append(map_points_dir / f"{int(frame.frame_id):06d}.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"map points unavailable for frame {frame.frame_id}")


def build_extended_part2_input(
    part1: Part1Output,
    *,
    frames_csv: str | Path,
    slot_database: str | Path,
    map_points_dir: str | Path,
    output_dir: str | Path,
    window_frames: int = 60,
    dataset_id: str = "pose-corrected-final",
) -> dict[str, Any]:
    """Attach extended causal packs/decisions and write a fresh Part1 input.

    The returned summary is suitable for a run manifest.  The input contract
    still contains exactly the original Part1 evidence frame ids; only the two
    explicit ``extended_lidar_*`` resource paths are added to each SlotCase.
    """

    if window_frames < 15:
        raise ValueError("extended LiDAR window must contain at least 15 frames")
    destination = Path(output_dir).resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    frame_path = Path(frames_csv).resolve()
    slot_path = Path(slot_database).resolve()
    points_path = Path(map_points_dir).resolve()
    project_root = Path.cwd().resolve()

    all_frames = load_frame_records(frame_path)
    causal = tuple(
        frame for frame in all_frames if int(frame.frame_id) <= part1.scene.anchor_frame_id
    )[-window_frames:]
    if len(causal) != window_frames:
        raise ValueError("not enough strictly causal frames for requested window")
    if int(causal[-1].frame_id) != int(part1.scene.anchor_frame_id):
        raise ValueError("extended LiDAR window does not end at the Part1 anchor")

    slots, scale = load_known_slots(slot_path)
    span = int(causal[-1].frame_id) - int(causal[0].frame_id)
    base = Hybrid3DConfig()
    config = replace(
        base,
        scope_max_distance_m=float(part1.scene.radius_m),
        frame_stride=1,
        window_before=max(base.window_before, span),
        window_after=max(base.window_after, span),
    )
    provider = FramePointProvider(
        {int(frame.frame_id): frame for frame in causal},
        points_path,
        cache_size=max(96, window_frames),
        project_root=project_root,
    )
    result = Hybrid3DPipeline(
        slots,
        causal,
        provider,
        scale,
        config,
        phase="full",
    ).run()

    decision_by_id = {decision.slot_id: decision for decision in result.decisions}
    slot_by_id = {slot.slot_id: slot for slot in result.all_slots}
    scope_by_id = {scope.slot_id: scope for scope in result.scopes}
    frame_by_id = {int(frame.frame_id): frame for frame in result.frames}
    candidate_ids = {case.slot_id for case in part1.slot_cases}
    missing = sorted(candidate_ids - set(decision_by_id))
    if missing:
        raise ValueError("extended pipeline omitted Part1 candidates: " + ", ".join(missing))

    config_hash = canonical_sha256(config.to_dict())
    slot_map_hash = canonical_sha256(
        [
            {"slot_id": slot.slot_id, "polygon_map": slot.polygon_map.tolist()}
            for slot in result.all_slots
        ]
    )
    pack_dir = destination / "packs"
    decision_dir = destination / "decisions"
    pack_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    states: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for case in part1.slot_cases:
        slot = slot_by_id[case.slot_id]
        scope = scope_by_id[case.slot_id]
        decision = decision_by_id[case.slot_id]
        accumulation = build_slot_accumulation(
            slot,
            scope,
            result.frames,
            provider,
            result.map_units_per_meter,
            config,
        )
        if not accumulation.observations:
            raise ValueError(f"no extended LiDAR observations for {case.slot_id}")
        source_paths = {
            int(frame_id): _map_points_source(
                frame_by_id[int(frame_id)], points_path, project_root
            )
            for frame_id in accumulation.selected_frames
        }
        metric = metric_slot(slot, result.map_units_per_meter)
        adjacent = {
            adjacent_id: map_xy_to_slot_m(
                slot_by_id[adjacent_id].polygon_map,
                metric,
            )
            for adjacent_id in slot.adjacent_slots
            if adjacent_id in slot_by_id
        }
        encounter_id = (
            f"extended-{min(accumulation.selected_frames):06d}-"
            f"{max(accumulation.selected_frames):06d}"
        )
        pack = build_lidar_evidence_pack(
            pack_dir / f"extended_{case.slot_id}.npz",
            accumulation=accumulation,
            slot=metric,
            source_paths=source_paths,
            task_id=f"parking-slot-agent-v2-extended:{case.slot_id}:{encounter_id}",
            encounter_id=encounter_id,
            dataset_id=str(dataset_id),
            config_hash=config_hash,
            slot_map_hash=slot_map_hash,
            adjacent_polygons_local_m=adjacent,
        )
        decision_path = decision_dir / f"extended_{case.slot_id}.json"
        decision_payload = {
            "schema_version": SCHEMA_VERSION,
            "slot_id": case.slot_id,
            "anchor_frame": int(part1.scene.anchor_frame_id),
            "alignment_anchor_frame": int(accumulation.anchor_frame),
            "first_frame": int(causal[0].frame_id),
            "window_frames": int(window_frames),
            "selected_frames": [int(value) for value in accumulation.selected_frames],
            "config_hash": config_hash,
            "slot_map_hash": slot_map_hash,
            "pack_path": str(pack.path.resolve()),
            "decision": asdict(decision),
        }
        write_json_atomic(decision_path, decision_payload)
        case.resources["extended_lidar_evidence_path"] = str(pack.path.resolve())
        case.resources["extended_lidar_decision_path"] = str(decision_path.resolve())
        state = decision.state.value
        states[state] = states.get(state, 0) + 1
        rows.append(
            {
                "slot_id": case.slot_id,
                "part1_state": case.part1_state.value,
                "extended_state": state,
                "decision_reason": decision.decision_reason,
                "unknown_reasons": list(decision.unknown_reasons),
                "pack_path": str(pack.path.resolve()),
                "decision_path": str(decision_path.resolve()),
            }
        )

    part1_output_path = destination / "part1_output.json"
    write_json_atomic(part1_output_path, part1.to_dict())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "part1_output_path": str(part1_output_path.resolve()),
        "anchor_frame": int(part1.scene.anchor_frame_id),
        "first_frame": int(causal[0].frame_id),
        "window_frames": int(window_frames),
        "candidate_count": len(rows),
        "state_counts": states,
        "rows": rows,
    }
    write_json_atomic(destination / "manifest.json", summary)
    return summary


__all__ = ["SCHEMA_VERSION", "build_extended_part2_input"]
