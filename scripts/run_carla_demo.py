#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dasp_park.carla_bridge import (  # noqa: E402
    connect_carla,
    configure_synchronous_world,
    destroy_carla_actors,
    lidar_measurement_to_ego_points,
    load_slot_map_from_config,
    restore_world_settings,
    spawn_carla_scene,
    tick_and_get_lidar,
)
from dasp_park.config import ensure_output_dir, load_config  # noqa: E402
from dasp_park.decision import evaluate_parking_decision  # noqa: E402
from dasp_park.diagnostics import (  # noqa: E402
    compute_decision_impact_map,
    compute_obstacle_boundary_map,
    compute_occlusion_map,
    compute_priority_map,
    compute_uncertainty_map,
    compute_unknown_map,
)
from dasp_park.occupancy import build_occupancy_from_lidar  # noqa: E402
from dasp_park.slot_selector import select_target_slot_from_known_map  # noqa: E402
from dasp_park.visualization import (  # noqa: E402
    save_lidar_bev,
    save_map,
    save_occupancy_map,
    save_slot_selection_panel,
)


def _compute_belief_maps(occupancy, slots, grid_cfg, cfg):
    unknown = compute_unknown_map(occupancy)
    boundary = compute_obstacle_boundary_map(occupancy)
    occlusion = compute_occlusion_map(occupancy, grid_cfg)
    uncertainty = compute_uncertainty_map(occupancy, occlusion, boundary)
    decision_impact = compute_decision_impact_map(occupancy, slots, grid_cfg)
    priority = compute_priority_map(uncertainty, occlusion, decision_impact, boundary, cfg["agent"]["weights"])
    return {
        "unknown": unknown,
        "boundary": boundary,
        "occlusion": occlusion,
        "uncertainty": uncertainty,
        "decision_impact": decision_impact,
        "priority": priority,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DASP-Park on a live CARLA LiDAR frame.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "carla_demo.yaml"))
    parser.add_argument("--keep-world", action="store_true", help="Do not destroy spawned actors after capture.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg["output_dir"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_output_dir(out_dir)

    carla_cfg = cfg["carla"]
    print(f"[DASP-Park CARLA] Connecting to {carla_cfg['host']}:{carla_cfg['port']}")
    client = connect_carla(carla_cfg["host"], int(carla_cfg["port"]), float(carla_cfg["timeout_s"]))

    if carla_cfg.get("map"):
        print(f"[DASP-Park CARLA] Loading map {carla_cfg['map']}")
        world = client.load_world(carla_cfg["map"])
    else:
        world = client.get_world()

    previous_settings = configure_synchronous_world(world, float(carla_cfg.get("fixed_delta_seconds", 0.05)))
    actors = None
    try:
        actors = spawn_carla_scene(world, cfg)
        measurement = tick_and_get_lidar(
            world,
            actors.lidar_queue,
            int(carla_cfg.get("warmup_frames", 10)),
            float(carla_cfg.get("timeout_s", 10.0)),
        )
        points = lidar_measurement_to_ego_points(measurement, carla_cfg["lidar"]["transform"])
        print(f"[DASP-Park CARLA] Captured {points.shape[0]} LiDAR points.")

        grid_cfg = cfg["grid"]
        slots = load_slot_map_from_config(cfg)
        occupancy, _height = build_occupancy_from_lidar(points, grid_cfg)
        maps = _compute_belief_maps(occupancy, slots, grid_cfg, cfg)
        selected_slot, slot_scores = select_target_slot_from_known_map(
            slots,
            occupancy,
            maps["uncertainty"],
            maps["occlusion"],
            grid_cfg,
        )
        top1_history = [slot_scores[0]["slot_id"]]
        decision = evaluate_parking_decision(
            slot_scores,
            top1_history,
            rounds_completed=1,
            max_rounds=int(cfg["agent"].get("max_rounds", 1)),
            competitor_explorations=0,
            max_competitor_explorations=int(cfg["agent"].get("max_competitor_explorations", 2)),
            cfg=cfg,
        )

        save_lidar_bev(
            points,
            "CARLA LiDAR Returns: Ground Faint, Obstacles Colored",
            out_dir / "01_carla_lidar_bev.png",
            grid_cfg,
            slots,
        )
        save_occupancy_map(occupancy, "CARLA Occupancy Belief", out_dir / "02_carla_occupancy.png", grid_cfg, slots)
        save_map(
            maps["priority"],
            "CARLA Active Perception Priority",
            out_dir / "03_carla_priority.png",
            grid_cfg,
            cmap="inferno",
            vmin=0,
            vmax=1,
            colorbar_label="priority",
            slots=slots,
        )
        save_slot_selection_panel(slot_scores, out_dir / "04_carla_slot_scores.png")

        trace = {
            "source": "carla",
            "note": (
                "CARLA is used for sensor capture. This script converts one LiDAR observation into "
                "DASP-Park occupancy belief and applies deterministic slot decision gates."
            ),
            "scene_complexity": {
                "num_known_slots": len(slots),
                "num_configured_parked_vehicles": len(carla_cfg.get("parked_vehicles", [])),
                "num_configured_static_obstacles": len(carla_cfg.get("static_obstacles", [])),
            },
            "lidar_points": int(points.shape[0]),
            "selected_slot": selected_slot.slot_id,
            "slot_scores": slot_scores,
            "final_parking_decision": decision.to_dict(),
        }
        with open(out_dir / "carla_agent_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        print(f"[DASP-Park CARLA] Selected slot candidate: {selected_slot.slot_id}")
        print(f"[DASP-Park CARLA] Final decision: {decision.action}")
        print(f"[DASP-Park CARLA] Saved outputs to {out_dir}")
    finally:
        if actors is not None and not args.keep_world:
            destroy_carla_actors(actors)
        restore_world_settings(world, previous_settings)


if __name__ == "__main__":
    main()
