# DASP-Park

Decision-Aware Active Structured Perception for Parking.

This repository contains a parking-oriented AI-assisted active perception MVP. The current version uses a controlled synthetic parking scene with known ground truth for evaluation only. It demonstrates how a LiDAR parking perception system can build an occupancy belief, use stable rules to generate legal perception queries, let an AI advisor choose query actions and tools under a strict validator, update the local perception belief, and use downstream slot ranking only as a stability/evaluation signal.

## Run

```bash
pip install -r requirements.txt
python scripts/run_vibe_demo.py
```

## CARLA Demo

CARLA is optional and version-sensitive. Keep the synthetic demo as the fast
unit test, then run CARLA when a CARLA server is already available.

Start CARLA first on Ubuntu/Windows or a remote GPU machine:

```bash
./CarlaUE4.sh -quality-level=Low
```

Install the matching CARLA Python API for that server version. For example:

```bash
python3 -m pip install carla==0.9.16
```

Then run:

```bash
python scripts/run_carla_demo.py --config configs/carla_demo.yaml
```

The CARLA script currently captures a live LiDAR frame, converts it into the
same DASP-Park occupancy belief, scores the known slot map, and applies the
deterministic final decision gate:

```text
COMMIT_SLOT
CONTINUE_PERCEPTION
DRIVE_FORWARD_EXPLORE
```

Outputs are written to:

```text
outputs/carla_demo/
```

Important CARLA config fields:

```text
carla.host / carla.port       CARLA server address
carla.map                     CARLA map to load
carla.ego_spawn               ego vehicle pose
carla.lidar                   LiDAR mount and attributes
slot_map.slots                known parking slot polygons in ego coordinates
```

## Outputs

The demo writes results to:

```text
outputs/vibe_demo/
```

Important files:

```text
00_scene_overview.png
01_gt_occupancy.png
03_occupancy_before.png
04_error_before.png
09_priority_map.png
10_selected_regions_on_priority.png
11_selected_regions_on_error.png
13_occupancy_after.png
14_error_after.png
15_effectiveness_panel.png
metrics_before_after.json
agent_trace.json
```

## Current Agent Flow

The demo is rule-dominant and AI-assisted:

```text
known slot map + sparse LiDAR
→ initial occupancy belief
→ downstream ranking exposes which perception evidence matters
→ AI chooses a perception query and an allowed tool
→ local tool updates occupancy belief
→ ranking is recomputed only to test belief stability
→ final deterministic gate checks stability, margin, occupancy, unknown, occlusion, and entrance risk
→ if checks pass, commit the slot
→ if checks fail but budget remains, continue perception
→ if checks still fail after bounded queries, drive forward for a new viewpoint
```

AI does not create coordinates, does not modify occupancy, and does not read ground truth. It only selects perception-query `action_id` and `tool_ids` from rule-generated candidates. The validator rejects anything outside the legal action/tool set.

## Why this demo is not fake

This MVP uses a controlled synthetic parking scene with known ground truth. The initial perception is built only from sparse observed LiDAR points. Ground truth is used only for evaluation and visualization. The active agent selects local perception queries based on current belief, ranking stability, uncertainty, occlusion, and decision impact. It then refines only selected regions using local tool evidence. The before/after metrics are computed against the synthetic ground truth and are not hard-coded.

This demo does not claim real-world parking performance. It only demonstrates the proposed active perception loop in a controlled setting.

## What this MVP demonstrates

- Synthetic parking scene with LiDAR-based occupancy.
- Ground-truth occupancy for evaluation.
- Degraded observed LiDAR for initial perception.
- Occupancy / unknown / occlusion / uncertainty maps.
- Decision-aware active perception.
- AI-selected legal perception query and tool plans.
- CARLA LiDAR adapter for live simulation input.
- Top-3 ranking stability as a downstream evaluation signal.
- Bounded competitor-region querying when belief is unstable.
- Final deterministic output: `COMMIT_SLOT`, `CONTINUE_PERCEPTION`, or `DRIVE_FORWARD_EXPLORE`.
- Local tool-based belief update.
- Before/after error maps and metrics.
- Clear visual explanation through `15_effectiveness_panel.png`.

## What is not included yet

- nuScenes loader.
- SUPS loader.
- BEVFusion.
- Neural perception model.
- Real parking slot detector.
- Closed-loop active viewpoint.
- Real-world parking success evaluation.
