#!/usr/bin/env python3
"""Build a geometry-only front-180 input for the autonomous Part2 Agent.

No slot IDs, expected labels, or per-case tool routes are encoded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_agent_v2.prompts import SYSTEM_PROMPT


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _ego_position(center_map: list[float], anchor_pose: list[float], scale: float) -> dict[str, Any]:
    dx = (float(center_map[0]) - float(anchor_pose[0])) / scale
    dy = (float(center_map[1]) - float(anchor_pose[1])) / scale
    yaw = float(anchor_pose[2])
    forward_m = math.cos(yaw) * dx + math.sin(yaw) * dy
    left_m = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return {
        "forward_m": forward_m,
        "left_m": left_m,
        "distance_m": math.hypot(forward_m, left_m),
        "bearing_deg": math.degrees(math.atan2(left_m, forward_m)),
        "inside_closed_front_180": forward_m >= -1e-9,
    }


def build(input_path: Path, output_path: Path, manifest_path: Path, prompt_path: Path) -> Part1Output:
    payload = _read_json(input_path)
    Part1Output.from_dict(payload)
    scene = payload["scene"]
    anchor_pose = scene["anchor_pose_map"]
    scale = float(scene["map_units_per_meter"])
    cases_by_id = {str(case["slot"]["slot_id"]): case for case in payload["slot_cases"]}
    selected_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for slot in scene["slots"]:
        slot_id = str(slot["slot_id"])
        geometry = _ego_position(slot["center_map"], anchor_pose, scale)
        case = cases_by_id.get(slot_id)
        state_before = case.get("part1_state") if case else slot.get("state")
        candidate = case is not None and state_before in {"free", "unknown"}
        selected = candidate and bool(geometry["inside_closed_front_180"])
        if selected:
            selected_ids.add(slot_id)
        elif candidate:
            slot["state"] = None
            slot["observed"] = False
        rows.append({
            "slot_id": slot_id,
            "part1_state_before_scope": state_before,
            **geometry,
            "selected_for_part2": selected,
            "scope_reason": "front_half_plane" if selected else "not_part1_candidate_or_rear",
        })
    payload["slot_cases"] = [
        case for case in payload["slot_cases"] if case["slot"]["slot_id"] in selected_ids
    ]
    source_producer = payload.get("producer")
    payload["producer"] = f"{source_producer}+front180_geometry_scope"
    validated = Part1Output.from_dict(payload)
    _write_json(output_path, validated.to_dict())
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    prompt_sha = hashlib.sha256((SYSTEM_PROMPT + "\n").encode("utf-8")).hexdigest()
    _write_json(manifest_path, {
        "schema_version": "autonomous-agent-input-manifest/1.0",
        "source_input": str(input_path),
        "output_input": str(output_path),
        "anchor_frame": validated.scene.anchor_frame_id,
        "history_frame_ids": [frame.frame_id for frame in validated.scene.frames],
        "history_frame_count": len(validated.scene.frames),
        "scope_formula": "forward=cos(yaw)*dx+sin(yaw)*dy; selected iff Part1 in {Free,Unknown} and forward>=0",
        "slot_specific_policy": False,
        "selected_case_count": len(validated.slot_cases),
        "selected_case_ids": [case.slot_id for case in validated.slot_cases],
        "geometry_audit": rows,
        "system_prompt_path": str(prompt_path),
        "system_prompt_sha256": f"sha256:{prompt_sha}",
    })
    return validated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.output, args.manifest, args.system_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
