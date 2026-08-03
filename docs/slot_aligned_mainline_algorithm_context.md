# Parking Slot Occupancy Evidence Workflow Context

Date: 2026-07-09

This document summarizes the current parking-slot point-cloud workflow, the algorithmic assumptions, the decision rules, the current test results, and the known limitations. It is written as a standalone factual context document and does not include recommendations.

## 1. Objective

The current goal is not to directly output a final, safety-critical `free` or `occupied` decision for every parking slot.

The current goal is to generate **per-slot evidence** from LiDAR point cloud, global map geometry, and vehicle pose:

```text
global map parking slots
+ LiDAR points projected into global map coordinates
+ vehicle pose / trajectory
→ per-slot point-cloud evidence
→ candidate states:
   core-supported occupied evidence
   adjacent-slot conflict
   boundary conflict
   static-like evidence
   no reliable vehicle evidence
→ camera review reports for human validation
```

The workflow is deliberately conservative. If point-cloud evidence is ambiguous, the system does not force an `occupied` or `free` label.

## 2. Available Data

The project uses the following data sources:

```text
1. Global parking map / glTF-derived slot polygons
2. LiDAR point clouds projected into the same global map coordinate system
3. Vehicle pose / aligned trajectory
4. Camera images
5. Camera timestamps
```

Important current paths:

```text
outputs/frame_map_dataset/frames.csv
outputs/frame_map_dataset/map_points/*.npz
outputs/full_icpark_allframes_vehicle_cluster/slot_database.json
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/
```

The main slot database is:

```text
outputs/full_icpark_allframes_vehicle_cluster/slot_database.json
```

It contains parsed parking slot geometry from the global map.

The main frame index is:

```text
outputs/frame_map_dataset/frames.csv
```

It contains, per LiDAR frame:

```text
frame
image_path
lidar_path
map_points_path
map_x
map_y
map_yaw
lidar_timestamp
odom_timestamp
time_delta_sec
```

The projected point cloud for each frame is stored in:

```text
outputs/frame_map_dataset/map_points/<frame>.npz
```

Each `.npz` contains at least:

```text
points_map_xyzi
ego_map_pose
frame
map_scale
```

## 3. Coordinate Assumption

The current workflow assumes:

```text
Parking slot polygons and LiDAR points are both in the same global map coordinate system.
```

Therefore, in principle, it is possible to test:

```text
Does this point/cluster fall inside this slot polygon?
Does it fall in the slot core?
Does it fall near a boundary?
Does it overlap adjacent slots?
```

This is the key reason the workflow is map-constrained rather than raw point-cloud-only.

## 4. Why Single-Frame Detection Was Not Enough

Early experiments used per-frame evidence scoring. That was too weak for reliable slot-level decisions because:

```text
1. The LiDAR point cloud is sparse.
2. A single frame often observes only a small portion of a parking slot.
3. Non-ground points near slot boundaries can belong to adjacent slots.
4. Walls, pillars, arresters, lane structures, or passing vehicles can create misleading non-ground points.
5. Absence of points does not mean the slot is free.
```

Therefore, the current mainline is **multi-frame accumulation**, not single-frame scoring.

## 5. Current Mainline

The current mainline is:

```text
slot-aligned multi-frame accumulation
```

For each candidate parking slot:

```text
1. Use pose to find the LiDAR frame where the ego vehicle is closest to that slot.
   This is called the anchor frame.

2. Around the anchor frame, take:
   anchor - 28 LiDAR frames
   through
   anchor + 28 LiDAR frames

3. Do not use every frame.
   Sample every 4 LiDAR frames.

4. This normally gives:
   15 sampled LiDAR frames
   covering roughly 5.6 seconds.

5. Accumulate points from those sampled frames.

6. Filter and cluster the accumulated non-ground points.

7. Attribute clusters to slot core / edge / margin / adjacent overlap.

8. Output an evidence state for each slot.
```

Current output directory:

```text
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/
```

Current main reports:

```text
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/slot_aligned_global_map_report_embedded.html
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/camera_comparison_review/camera_comparison_review_report.html
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/static_like_camera_previous28_review/camera_comparison_review_report.html
```

## 6. Slot Geometry Model

The system does not use only the raw parking slot polygon.

Each slot is modeled with multiple regions:

```text
slot polygon
  Original parking slot polygon from the map.

core polygon
  Conservative inner region.
  Strong occupied evidence is expected to come mainly from here.

edge band
  Boundary region inside/near the slot.
  Points here are risky because they may belong to adjacent slots, lane structures, or pose error.

margin polygon
  Expanded buffer region around the slot.
  Used for pose error and vehicle line-crossing tolerance, but not strong ownership.

adjacent overlap
  Region where this slot's margin overlaps neighboring slot margins.
  Evidence here is not assigned strongly to a single slot.
```

The important principle is:

```text
Points in slot core can support strong slot ownership.
Points in edge/margin/adjacent-overlap produce uncertainty rather than strong ownership.
```

## 7. Point-Cloud Processing

The current point-cloud algorithm is a traditional geometry pipeline, not a neural 3D detector.

Current method:

```text
1. Load projected LiDAR points for sampled frames.
2. Accumulate points around the slot-aligned anchor frame.
3. Estimate local ground height by low quantile.
4. Keep non-ground candidate points using height thresholds.
5. Run DBSCAN clustering in BEV / map XY.
6. Compute cluster geometry:
   length
   width
   height span
   point count
   footprint
   orientation
7. Compute overlap between each cluster and slot regions:
   core overlap
   edge overlap
   margin overlap
   adjacent slot overlap
8. Assign cluster ownership or conflict.
9. Classify evidence state.
```

Current DBSCAN parameters:

```text
eps = 0.25
min_samples = 8
```

Current accumulation parameters:

```text
window_before = 28 LiDAR frames
window_after = 28 LiDAR frames
window_frame_stride = 4
```

## 8. Cluster Ownership Rules

A cluster can support a slot only if it is clearly owned by that slot.

The intended ownership logic is:

```text
If cluster overlaps one slot core strongly,
and that overlap is clearly greater than the second-best slot core overlap,
and the cluster is not mostly edge/margin,
and adjacent overlap is not high,
then cluster ownership is clear_core_owned.
```

Otherwise, it may be:

```text
adjacent_slot_conflict
  Top slot and second slot are too close in evidence.

boundary_conflict
  Evidence lies mainly on edge/margin.

margin_only
  Evidence lies mostly outside core.

slot_vs_lane_possible
  Evidence may belong to lane or outside-slot structure.
```

The workflow intentionally avoids:

```text
Any non-ground point inside the raw polygon → occupied
```

because that creates many false positives in dense parking lots.

## 9. Evidence States

The current output states are evidence labels, not final parking availability labels.

### 9.1 accumulated_vehicle_core_supported

Meaning:

```text
The accumulated point cloud contains a vehicle-like cluster,
and the cluster is mainly inside the target slot core.
```

This is a **high occupied-evidence candidate**, but it is not yet a final `occupied` decision.

Typical evidence:

```text
vehicle-like score high
core overlap high
boundary ratio not too high
adjacent overlap not too high
cluster dimensions roughly vehicle-like
```

### 9.2 accumulated_adjacent_conflict

Meaning:

```text
The cluster may represent a vehicle,
but it overlaps adjacent slots too strongly.
The system cannot decide which slot owns it.
```

This often happens when:

```text
vehicles are close together
cluster is merged across slot boundaries
pose or map alignment is slightly off
point cloud falls near a row boundary
```

### 9.3 accumulated_boundary_conflict

Meaning:

```text
The cluster is concentrated near slot boundary, edge band, or margin.
It is unsafe to assign it as strong evidence for this slot.
```

### 9.4 accumulated_static_like

Meaning:

```text
There is a non-ground cluster,
but its geometry does not look like a clean single vehicle.
```

Static-like does not mean the slot definitely contains a static object. It means:

```text
The evidence is not safe to interpret as a vehicle occupying this exact slot.
```

Static-like sub-reasons:

```text
low_height_structure
  Height span too low. Could be lane line, arrester, curb, floor reflection, or low obstacle.

too_large_merged
  Cluster is too large for a single vehicle. Could be multiple vehicles merged, wall, lane structure, pillar region, or accumulated background.

wall_like_linear
  Cluster is too long/thin. More wall-like or line-like than car-like.
```

### 9.5 accumulated_no_vehicle_evidence

Meaning:

```text
The accumulated point cloud does not contain reliable vehicle-like evidence for this slot.
```

This is not the same as `free`.

Because:

```text
No vehicle evidence does not prove the slot was fully observed.
No points does not mean free.
```

## 10. Current Results

Latest mainline result:

```text
input/output:
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/

candidate slots = 202
```

State counts:

```text
accumulated_vehicle_core_supported = 36
accumulated_adjacent_conflict = 22
accumulated_boundary_conflict = 13
accumulated_static_like = 25
accumulated_no_vehicle_evidence = 106
```

Static-like breakdown:

```text
low_height_structure = 12
too_large_merged = 11
wall_like_linear = 2
```

Examples:

```text
slot_0944:
  reason = low_height_structure
  height_span ≈ 0.124 m
  Not high enough to be vehicle-like.

slot_1069:
  reason = low_height_structure
  core_overlap = 0
  boundary_ratio = 1.0
  height_span ≈ 0.09 m

slot_0988:
  reason = too_large_merged
  cluster_length ≈ 21.4 m
  cluster_width ≈ 12.0 m
  Too large for a single vehicle.

slot_1075:
  reason = too_large_merged
  cluster_length ≈ 32.1 m
  cluster_width ≈ 17.1 m
  adjacent_overlap ≈ 0.794
```

## 11. Camera Review Workflow

The current pipeline includes visual review reports because point-cloud evidence alone is not reliable enough yet.

Camera images are linked using timestamps, not by assuming the same frame number.

There are two camera selection modes:

### 11.1 Nearest timestamp camera

Used for some earlier reports:

```text
LiDAR anchor timestamp
→ camera timestamp closest to it
```

This typically gives camera-LiDAR time difference:

```text
around 0.0 to 0.016 seconds
```

### 11.2 Previous-28 camera

Latest user-requested static-like review uses:

```text
LiDAR anchor timestamp
→ find camera frame not later than LiDAR timestamp
→ go back 28 camera frames
→ use the earliest image in that 28-frame previous window
```

Since camera is around 30Hz, this image is often:

```text
about 0.9 seconds before the LiDAR anchor
```

Current static-like previous-28 report:

```text
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/static_like_camera_previous28_review/camera_comparison_review_report.html
```

It contains, for each case:

```text
1. Global target slot map
2. Accumulated point-cloud local evidence
3. Selected camera image
4. Camera image with LiDAR overlay
5. Evidence metrics and reason
```

Core-supported camera comparison report:

```text
outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8/camera_comparison_review/camera_comparison_review_report.html
```

## 12. Why Not Final Occupied / Free Yet

The current evidence states are not final `occupied/free` outputs because:

```text
1. Point cloud is sparse.
2. Multi-frame accumulation can merge multiple objects.
3. DBSCAN can connect adjacent vehicles or background structures.
4. Parking slots are dense and adjacent margins overlap.
5. Walls, pillars, lane boundaries, arresters, and curbs can produce non-ground clusters.
6. Pose/map alignment error can shift points across slot boundaries.
7. No vehicle evidence does not prove visibility or free space.
```

Therefore:

```text
core-supported = strong occupied evidence candidate
conflict/static-like = needs further analysis or camera validation
no evidence = not enough point-cloud vehicle evidence
```

The current system is an evidence generator, not a complete autonomous parking availability classifier.

## 13. Current Strong Points

The current workflow already does these things well:

```text
1. Uses global map slot geometry instead of raw point cloud alone.
2. Distinguishes slot core from boundary/margin.
3. Avoids assigning boundary points as strong occupancy.
4. Uses multi-frame accumulation to reduce single-frame sparsity.
5. Produces explainable uncertainty categories.
6. Generates visual reports for human inspection.
7. Keeps static-like and conflict cases separate instead of forcing occupied.
```

## 14. Known Weaknesses

The main weaknesses are:

```text
1. too_large_merged clusters are common.
   DBSCAN can merge multiple cars, walls, lane-side objects, or background points.

2. static-like may include real vehicles that were merged or poorly segmented.
   A large merged cluster might contain a car, but cannot be assigned to one slot.

3. No reliable free detection yet.
   The current mainline focuses on occupied evidence, not ray-traced free-space proof.

4. Anchor selection is based on closest ego pose to slot center.
   This is simple but may not always correspond to best visibility or best viewpoint.

5. Camera review currently supports human validation, not automatic fusion.
```

## 15. Current Verification

Latest checks before this document:

```text
python3 -m unittest discover -s tests -v
```

Result:

```text
7 tests passed
```

Current active output index:

```text
outputs/current_mainline/README.md
```

Archive manifest for older experiments:

```text
outputs/archive/2026-07-09_cleanup_before_mainline/archive_manifest.csv
outputs/archive/2026-07-09_cleanup_before_mainline/archive_manifest.json
```
