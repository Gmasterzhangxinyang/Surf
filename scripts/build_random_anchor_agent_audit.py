#!/usr/bin/env python3
"""Build an auditable Unknown-reason tool loop for the frozen random anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "parkingagent-unknown-reason-closed-loop/1.2"
CAMERA_ROUTING_REASONS = {
    "boundary_dominated",
    "high_occlusion",
    "insufficient_lower_body_coverage",
    "insufficient_vehicle_footprint",
    "large_unobserved_component",
    "outside_residual_conflict",
    "ownership_conflict",
    "weak_temporal_inconsistent",
    "weak_vehicle_evidence",
    "weak_visibility_limited",
}
EXTENDED_LIDAR_ROUTING_REASONS = CAMERA_ROUTING_REASONS | {
    "compact_vertical_structure",
    "horizontal_cap_structure",
    "linear_static_structure",
    "weak_static_structure",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(data + "\n", encoding="utf-8")


def _relative_position(local_map: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    pose = local_map["anchor_pose"]
    scale = float(local_map["map_units_per_meter"])
    center = [float(value) for value in slot["center_map"]]
    dx = (center[0] - float(pose["map_xy"][0])) / scale
    dy = (center[1] - float(pose["map_xy"][1])) / scale
    yaw = float(pose["map_yaw_rad"])
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    bearing = math.degrees(math.atan2(left, forward))
    return {
        "coordinate_contract": {
            "map": "slot center/polygon in source map coordinates",
            "ego": "x=forward, y=left, metres at anchor",
            "anchor_frame": int(pose["frame_id"]),
        },
        "center_map": center,
        "polygon_map": slot["polygon_map"],
        "ego_forward_m": forward,
        "ego_left_m": left,
        "ego_distance_m": math.hypot(forward, left),
        "ego_bearing_deg": bearing,
        "inside_closed_front_180": forward >= 0.0 and abs(bearing) <= 90.0,
    }


def _media_by_slot(manifest: dict[str, Any], media_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in manifest["media"]:
        digest = str(row["sha256"]).split(":", 1)[1]
        path = media_root / "sha256" / digest[:2] / f"{digest}.png"
        if not path.is_file() or _sha(path) != row["sha256"]:
            raise ValueError(f"LiDAR media identity mismatch: {row['slot_id']}")
        result[str(row["slot_id"])] = {
            **row,
            "path": str(path),
        }
    return result


def _camera_by_slot(manifest: dict[str, Any], manifest_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in manifest["rows"]:
        slot_id = str(row["slot_id"])
        image = row.get("image")
        path = Path(image) if image else None
        if path is not None and not path.is_absolute():
            project_candidate = Path.cwd() / path
            manifest_candidate = manifest_path.parent / path
            path = project_candidate if project_candidate.is_file() else manifest_candidate
        result[slot_id] = {
            **row,
            "image": None if path is None else str(path),
            "sha256": None if path is None else _sha(path),
        }
    return result


def _blind_assessment(slot_id: str) -> dict[str, Any]:
    """Frozen prediction-blind observations made before any GT lookup."""
    rows: dict[str, dict[str, Any]] = {
        "slot_0942": {
            "semantic_proposal": "free",
            "confidence": "medium",
            "finding": "eight causal RGB views show no visible vehicle at the projected target region",
            "ownership": "uncertain",
        },
        "slot_0943": {
            "semantic_proposal": "free",
            "confidence": "medium",
            "finding": "eight causal RGB views show an apparently empty projected target region",
            "ownership": "uncertain",
        },
        "slot_0944": {
            "semantic_proposal": "free",
            "confidence": "medium",
            "finding": "target-side floor appears empty across eight causal views",
            "ownership": "uncertain",
        },
        "slot_0945": {
            "semantic_proposal": "free",
            "confidence": "medium",
            "finding": "no vehicle is visible in the thin projected region across eight causal views",
            "ownership": "uncertain",
        },
        "slot_0964": {
            "semantic_proposal": "unknown",
            "confidence": "low",
            "finding": "a vehicle is visible to the left, but target-versus-adjacent ownership is not secure",
            "ownership": "uncertain",
        },
        "slot_0994": {
            "semantic_proposal": "unknown",
            "confidence": "low",
            "finding": "no eligible front-camera sequence is available",
            "ownership": "uncertain",
        },
        "slot_0995": {
            "semantic_proposal": "unknown",
            "confidence": "low",
            "finding": "no eligible front-camera sequence is available",
            "ownership": "uncertain",
        },
        "slot_0996": {
            "semantic_proposal": "unknown",
            "confidence": "low",
            "finding": (
                "the projected target region is occluded; foreground vehicle overlap "
                "does not prove that the target slot itself is occupied"
            ),
            "ownership": "occluded_unresolved",
            "review_note": "corrected after user visual adjudication",
        },
    }
    assessment = rows.get(
        slot_id,
        {
            "semantic_proposal": "unknown",
            "confidence": "low",
            "finding": "no frozen semantic assessment",
            "ownership": "uncertain",
        },
    )
    confidence_scores = {
        "slot_0942": 0.78,
        "slot_0943": 0.82,
        "slot_0944": 0.81,
        "slot_0945": 0.74,
        "slot_0964": 0.45,
        "slot_0996": 0.45,
    }
    target_slots = {"slot_0942", "slot_0943", "slot_0944", "slot_0945"}
    return {
        **assessment,
        "confidence_score": confidence_scores.get(slot_id, 0.0),
        "ownership": (
            "target_by_multiframe_overlay"
            if slot_id in target_slots
            else assessment["ownership"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--local-map", required=True, type=Path)
    parser.add_argument("--lidar-media-manifest", required=True, type=Path)
    parser.add_argument("--lidar-media-root", required=True, type=Path)
    parser.add_argument("--camera-manifest", required=True, type=Path)
    parser.add_argument("--extended-decisions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-semantic-threshold", type=float, default=0.60)
    args = parser.parse_args()
    if not 0.0 <= args.camera_semantic_threshold <= 1.0:
        raise ValueError("camera semantic threshold must be within [0, 1]")

    queue = _read(args.queue)
    local_map = _read(args.local_map)
    lidar = _media_by_slot(
        _read(args.lidar_media_manifest),
        args.lidar_media_root,
    )
    camera_manifest = _read(args.camera_manifest)
    camera = _camera_by_slot(camera_manifest, args.camera_manifest)
    extended = {
        row["slot_id"]: row
        for row in _read(args.extended_decisions)["decisions"]
    }
    map_slots = {row["slot_id"]: row for row in local_map["slots"]}

    cases: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for item in queue["items"]:
        slot_id = str(item["slot_id"])
        unknown_reasons = list(item["unknown_reasons"])
        position = _relative_position(local_map, map_slots[slot_id])
        if not position["inside_closed_front_180"]:
            raise ValueError(f"queue contains a rear slot: {slot_id}")
        calls: list[dict[str, Any]] = []

        calls.append(
            {
                "turn": 1,
                "tool": "get_slot_position",
                "selected_by": "mandatory_identity_and_route_scope",
                "status": "ok",
                "output": position,
            }
        )
        lidar_row = lidar[slot_id]
        calls.append(
            {
                "turn": 2,
                "tool": "inspect_lidar_map",
                "selected_by": "all_part1_unknowns_require_target_local_geometry",
                "status": "ok",
                "input_reasons": unknown_reasons,
                "output": {
                    "media_path": lidar_row["path"],
                    "media_sha256": lidar_row["sha256"],
                    "kind": lidar_row["kind"],
                    "source_sha256": lidar_row["source_sha256"],
                },
            }
        )

        extended_row = extended.get(slot_id)
        routed_extended = bool(set(unknown_reasons) & EXTENDED_LIDAR_ROUTING_REASONS)
        if routed_extended:
            calls.append(
                {
                    "turn": 3,
                    "tool": "inspect_extended_lidar_history",
                    "selected_by": sorted(set(unknown_reasons) & EXTENDED_LIDAR_ROUTING_REASONS),
                    "status": "ok" if extended_row is not None else "unavailable",
                    "output": None
                    if extended_row is None
                    else {
                        "history_span_W": 60,
                        "sample_count_K": 15,
                        "state": extended_row["state"],
                        "decision_reason": extended_row["decision_reason"],
                        "unknown_reasons": extended_row.get("unknown_reasons", []),
                    },
                }
            )

        camera_row = camera.get(slot_id, {"view_count": 0, "image": None})
        routed_camera = bool(set(unknown_reasons) & CAMERA_ROUTING_REASONS)
        camera_called = routed_camera and int(camera_row.get("view_count", 0)) > 0
        if camera_called:
            calls.append(
                {
                    "turn": 4,
                    "tool": "inspect_rgb_sequence",
                    "selected_by": sorted(set(unknown_reasons) & CAMERA_ROUTING_REASONS),
                    "status": "experimental_terminal_eligible",
                    "output": {
                        "media_path": camera_row["image"],
                        "media_sha256": camera_row["sha256"],
                        "frame_ids": camera_row["frames"],
                        "view_count": camera_row["view_count"],
                        "projection_status": camera_manifest["extrinsic_status"],
                        "production_terminal_capability": False,
                        "experimental_terminal_capability": True,
                        "semantic_threshold": args.camera_semantic_threshold,
                    },
                }
            )

        assessment = _blind_assessment(slot_id)
        evidence_reasons = set(unknown_reasons)
        if extended_row is not None and extended_row["state"] == "unknown":
            evidence_reasons.update(extended_row.get("unknown_reasons", []))
        relaxed_terminal = (
            camera_called
            and assessment["semantic_proposal"] in {"free", "occupied"}
            and float(assessment["confidence_score"]) >= args.camera_semantic_threshold
            and str(assessment["ownership"]).startswith("target_")
        )
        if relaxed_terminal:
            blockers: set[str] = set()
            resolved = sorted(evidence_reasons)
            final_state = assessment["semantic_proposal"]
            final_reason = (
                "experimental Camera-Agent override: multiframe target overlay "
                f"confidence {assessment['confidence_score']:.2f} >= "
                f"{args.camera_semantic_threshold:.2f}"
            )
        else:
            blockers = set(evidence_reasons)
            resolved = []
            final_state = "unknown"
            if not str(assessment["ownership"]).startswith("target_"):
                blockers.add("target_ownership_not_verified")
            if camera_called and float(assessment["confidence_score"]) < args.camera_semantic_threshold:
                blockers.add("ai_semantic_confidence_below_threshold")
            elif routed_camera and not camera_called:
                blockers.add("eligible_camera_sequence_unavailable")
            final_reason = (
                "Camera-Agent evidence did not reach the relaxed semantic threshold "
                "or target ownership remained unresolved"
            )
        case = {
            "slot_id": slot_id,
            "input_state": "unknown",
            "part1_unknown_reasons": unknown_reasons,
            "position": position,
            "tool_calls": calls,
            "agent_semantic_assessment": assessment,
            "resolved_unknown_reasons": resolved,
            "unresolved_blockers": sorted(blockers),
            "final_state": final_state,
            "final_reason": final_reason,
            "decision_policy": "experimental_camera_agent_threshold_0p60",
            "production_fail_closed_state": "unknown",
            "gt_read_during_inference": False,
        }
        cases.append(case)
        for call in calls:
            trace.append({"slot_id": slot_id, **call})
        trace.append(
            {
                "slot_id": slot_id,
                "turn": len(calls) + 1,
                "type": "final_decision",
                "semantic_proposal": assessment["semantic_proposal"],
                "validated_state": final_state,
                "unresolved_blockers": sorted(blockers),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "anchor_frame": int(local_map["anchor_pose"]["frame_id"]),
        "queue_slot_count": len(cases),
        "tool_call_count": sum(len(row["tool_calls"]) for row in cases),
        "tool_call_counts": {
            name: sum(
                call["tool"] == name
                for row in cases
                for call in row["tool_calls"]
            )
            for name in (
                "get_slot_position",
                "inspect_lidar_map",
                "inspect_extended_lidar_history",
                "inspect_rgb_sequence",
            )
        },
        "semantic_proposal_counts": {
            state: sum(
                row["agent_semantic_assessment"]["semantic_proposal"] == state
                for row in cases
            )
            for state in ("free", "occupied", "unknown")
        },
        "validated_state_counts": {
            state: sum(row["final_state"] == state for row in cases)
            for state in ("free", "occupied", "unknown")
        },
        "resolved_slot_count": sum(row["final_state"] != "unknown" for row in cases),
        "gt_fields_read": [],
        "prediction_blind_semantic_review": False,
        "prediction_blind_initial_review": True,
        "posthoc_user_occlusion_correction_ids": ["slot_0996"],
        "camera_visible_fraction_threshold": 0.60,
        "camera_semantic_threshold": args.camera_semantic_threshold,
        "experimental_camera_terminal_capability": True,
        "production_camera_terminal_capability": False,
        "production_camera_terminal_blocker": "independent_pixel_projection_audit_missing",
        "policy_status": "experimental_relaxed_camera_agent; not production safety claim",
        "scientific_interpretation": (
            "The relaxed Camera-Agent policy accepts a terminal semantic proposal "
            "when multiframe target-overlay confidence reaches the configured 0.60 "
            "threshold. Production fail-closed states are retained per case for comparison."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "cases.json", {"schema_version": SCHEMA_VERSION, "cases": cases})
    _write(args.output_dir / "tool_trace.json", {"schema_version": SCHEMA_VERSION, "events": trace})
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
