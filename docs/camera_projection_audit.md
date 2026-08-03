# Camera projection audit v2

This audit is the trust gate between usable camera geometry and RGB terminal
decision capability. A calibration file alone is not trusted.

The only accepted evidence is an independently observed image pixel paired
with a known 3D point. Do not fill `observed_pixel_uv` by projecting the same
3D point with the calibration being audited; that would be circular evidence.
The older `camera-projection-audit/1.0` status-only format is readable for
migration, but is always downgraded to `untrusted`.

## Reference input

The builder accepts only `camera-projection-reference-set/2.0`. The incompatible
v1 reference format is rejected and cannot produce a passing audit. A minimal
v2 document is:

```json
{
  "schema_version": "camera-projection-reference-set/2.0",
  "dataset_id": "route-a",
  "calibration_sha256": "sha256:<hash of exact calibration file bytes>",
  "observation_source": {
    "kind": "manual_pixel_labels",
    "artifact_uri": "annotations/manual-corners.json",
    "artifact_sha256": "sha256:<hash of exact annotation export bytes>"
  },
  "images": [
    {
      "image_id": "camera-frame-123",
      "image_uri": "images/000123.png",
      "image_sha256": "sha256:<hash of exact image bytes>",
      "decoded_width_px": 1280,
      "decoded_height_px": 720,
      "camera_frame": 123,
      "camera_timestamp_sec": 173.456
    }
  ],
  "point_sources": [
    {
      "point_source_id": "lidar-frame-123",
      "lidar_uri": "lidar/000123.bin",
      "lidar_sha256": "sha256:<hash of exact LiDAR frame bytes>",
      "lidar_frame": 123,
      "lidar_timestamp_sec": 173.438,
      "pose_uri": "poses/000123.json",
      "pose_sha256": "sha256:<hash of exact pose JSON bytes>"
    }
  ],
  "references": [
    {
      "id": "frame-000123-corner-a",
      "image_id": "camera-frame-123",
      "point_source_id": "lidar-frame-123",
      "observed_pixel_uv": [105.25, 348.5],
      "point_lidar_xyz_m": [12.3, 4.1, -0.72]
    },
    {
      "id": "frame-000123-corner-b",
      "image_id": "camera-frame-123",
      "point_source_id": "lidar-frame-123",
      "observed_pixel_uv": [640.0, 355.75],
      "point_map_xyz_m": [32.1, 18.4, 0.0]
    }
  ]
}
```

`kind` must be one of `manual_pixel_labels`,
`ground_truth_correspondences`, or `calibration_target`. The annotation artifact,
every image, every LiDAR source, and every pose source must use a relative local
path or `file://` URI. Network URIs are rejected. The builder reads the actual
bytes and checks every declared SHA-256.

Images are safely decoded with Pillow as single-frame PNG or JPEG files. Their
actual decoded size must match both the registry and the exact calibration.
`observed_pixel_uv` is checked against those decoded coordinates. Duplicate
image IDs, paths, content hashes, or `(camera_frame, camera_timestamp_sec)`
identities are rejected. A reference contains `image_id`; the old detached
per-reference `image_sha256` field is forbidden.

Every reference also contains `point_source_id`, which resolves to a deduplicated
LiDAR frame registry entry with an actual local artifact, SHA-256, frame number,
and timestamp. The referenced camera and LiDAR timestamps must differ by no
more than 40 ms by default; `--maximum-camera-lidar-delta-sec` may tighten but
never loosen that production limit. Every reference then uses exactly one 3D
form:

- `point_lidar_xyz_m`; or
- `point_map_xyz_m`, whose point source must additionally bind a pose JSON.

The pose JSON uses `camera-projection-pose-source/1.0`:

```json
{
  "schema_version": "camera-projection-pose-source/1.0",
  "lidar_frame": 123,
  "lidar_timestamp_sec": 173.438,
  "lidar_from_map": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

Map and LiDAR coordinates are metric XYZ. A 2D pose approximation is not
accepted. The transform convention is
`p_lidar = lidar_from_map @ [p_map_x, p_map_y, p_map_z, 1]`. Pixels must lie
inside the decoded image. Invalid projections, phantom/missing/tampered source
files, resize or dimension mismatches, duplicate registries, unresolved IDs,
dataset/hash mismatches, and automatic-projection sources fail closed.

This provenance proves which image, annotation, LiDAR, and pose bytes were
used. It cannot by itself prove that a human annotation was independent, or
that an arbitrary LiDAR file contains the declared manually selected XYZ.
Those remain dataset-governance and review requirements; do not describe a
hash-only audit as independent ground truth without that external process.

## Acceptance policy

The default policy requires all of the following:

- at least 15 valid references total;
- at least 5 references in each of the left, center, and right image thirds,
  assigned from the independently observed pixel `u` coordinate;
- overall and per-zone RMSE no greater than 3 pixels;
- overall and per-zone maximum error no greater than 8 pixels; and
- camera/LiDAR timestamp delta no greater than 0.04 seconds for every
  reference (or the tighter recorded audit threshold); and
- zero invalid 3D projections.

Missing or empty `references` therefore produces an audit artifact with
`status=untrusted`; it can never produce `passed`.

## Build command

```bash
python3 scripts/build_camera_projection_audit.py \
  --calibration path/to/calib.txt \
  --references path/to/real_pixel_references.json \
  --dataset-id route-a \
  --output outputs/camera_projection_audit.json
```

The command writes a deterministic, hash-bound `camera-projection-audit/2.0`
artifact. It exits `0` only for `passed`, `3` for a valid but untrusted audit,
and `2` for malformed input or I/O failure. Threshold flags may only tighten
the production 15/5/3 px/8 px/40 ms policy. The selected values are recorded
in the audit and are revalidated when the audit is loaded.

The artifact records `reference_set_uri` and `reference_set_sha256`, plus
`image_count/image_registry_sha256`,
`point_source_count/point_source_registry_sha256`, and the annotation artifact
SHA-256. Loading v2 requires the exact calibration file. The loader reopens and
rehashes the reference set and every registered source, safely decodes the
images, replays pose transforms, then rebuilds all metrics, failures, status,
and provenance digests with the recorded thresholds. A detached or hand-edited
aggregate cannot grant trust. Missing, changed, oversized, or undecodable input
fails closed. `load_camera_model(...)` always performs this recomputation before
exposing `calibration_audit_status=passed`.
