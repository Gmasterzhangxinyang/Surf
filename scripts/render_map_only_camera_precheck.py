#!/usr/bin/env python3
"""Run the Part1-map-only Camera precheck and render its audit artifacts.

The command marks slots that are likely observable from a conservative Camera
proxy.  It never requests or calls a Camera semantic model and never overwrites
Part1's ``local_map.json`` or ``local_map.png``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.io import write_json_atomic
from parking_slot_part2.map_only_camera_precheck import (
    MapOnlyCameraPrecheckConfig,
    evaluate_map_only_camera_candidates,
)
from parking_slot_part2.map_only_camera_precheck_visualization import (
    render_annotated_map_only_precheck,
    render_camera_gate_bev,
)


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(value)


def _default_target_ids(local_map: Mapping[str, Any]) -> list[str]:
    raw = local_map.get("provisional_candidates", ())
    if not isinstance(raw, list):
        raise ValueError(
            "target IDs are required when Part1 provisional_candidates is absent"
        )
    result: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Part1 provisional_candidates must contain objects")
        slot_id = str(item.get("slot_id", "")).strip()
        if slot_id and slot_id not in result:
            result.append(slot_id)
        if len(result) == 2:
            break
    if not result:
        raise ValueError(
            "no map-only targets: pass --target-slot-id once or twice"
        )
    return result


def _load_config(path: str | Path | None) -> MapOnlyCameraPrecheckConfig:
    if path is None:
        return MapOnlyCameraPrecheckConfig()
    mapping = _load_mapping(Path(path), "map-only precheck config")
    factory = getattr(MapOnlyCameraPrecheckConfig, "from_mapping", None)
    if callable(factory):
        return factory(mapping)
    return MapOnlyCameraPrecheckConfig(**mapping)


def render_map_only_precheck(
    local_map_path: str | Path,
    *,
    machine_bev_path: str | Path,
    annotated_path: str | Path,
    report_path: str | Path,
    target_slot_ids: list[str] | None = None,
    config_path: str | Path | None = None,
    resolution_m_per_pixel: float = 0.05,
) -> dict[str, Any]:
    """Execute the gate once and write all three independent artifacts."""

    local_path = Path(local_map_path)
    machine_output = Path(machine_bev_path)
    annotated_output = Path(annotated_path)
    json_output = Path(report_path)
    protected = {local_path.resolve(), local_path.with_name("local_map.png").resolve()}
    outputs = {
        machine_output.resolve(),
        annotated_output.resolve(),
        json_output.resolve(),
    }
    if len(outputs) != 3:
        raise ValueError("machine BEV, annotated image, and report paths must differ")
    if protected & outputs:
        raise ValueError("map-only artifacts must not overwrite Part1 outputs")

    local_map = _load_mapping(local_path, "Part1 local map")
    targets = list(target_slot_ids or _default_target_ids(local_map))
    if not 1 <= len(targets) <= 2 or len(set(targets)) != len(targets):
        raise ValueError("pass one or two unique --target-slot-id values")
    config = _load_config(config_path)
    raw_report = evaluate_map_only_camera_candidates(
        local_map,
        targets,
        config=config,
        include_bev_masks=False,
    )
    if not isinstance(raw_report, Mapping):
        raise TypeError("map-only precheck must return a mapping")
    report = deepcopy(dict(raw_report))

    machine_metadata = render_camera_gate_bev(
        local_map,
        report,
        machine_output,
        resolution_m_per_pixel=resolution_m_per_pixel,
    )
    annotated_metadata = render_annotated_map_only_precheck(
        local_map,
        report,
        annotated_output,
    )
    report["source_part1_local_map"] = str(local_path)
    report["artifacts"] = {
        "machine_bev": machine_metadata,
        "annotated_precheck": annotated_metadata,
        "report_path": str(json_output),
    }
    # Reassert the architectural boundary in the persisted artifact even when
    # a caller supplies a custom configuration.
    if report.get("semantic_camera_model_called") is not False:
        raise ValueError("map-only precheck unexpectedly called a semantic model")
    if report.get("camera_call_requested") is not False:
        raise ValueError("map-only precheck unexpectedly requested a Camera call")
    write_json_atomic(json_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-map", type=Path, required=True)
    parser.add_argument(
        "--target-slot-id",
        action="append",
        dest="target_slot_ids",
        help="slot to precheck; repeat at most twice (defaults to Part1 A/B)",
    )
    parser.add_argument("--config", type=Path, help="optional map-only policy JSON")
    parser.add_argument("--machine-bev", type=Path)
    parser.add_argument("--annotated", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resolution-m-per-pixel", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.local_map.parent
    report = render_map_only_precheck(
        args.local_map,
        machine_bev_path=args.machine_bev or directory / "camera_gate_bev.png",
        annotated_path=args.annotated or directory / "map_only_camera_precheck.png",
        report_path=args.report or directory / "map_only_camera_precheck.json",
        target_slot_ids=args.target_slot_ids,
        config_path=args.config,
        resolution_m_per_pixel=args.resolution_m_per_pixel,
    )
    print(
        json.dumps(
            {
                "machine_bev": report["artifacts"]["machine_bev"]["path"],
                "annotated": report["artifacts"]["annotated_precheck"]["path"],
                "report": report["artifacts"]["report_path"],
                "camera_called": False,
                "targets": {
                    item["target_slot_id"]: item["status"]
                    for item in report.get("targets", ())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
