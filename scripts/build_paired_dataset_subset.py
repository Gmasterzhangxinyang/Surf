#!/usr/bin/env python3
"""Build a small self-contained paired camera/point-cloud subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {
    "frame",
    "camera_match_valid",
    "missing_image",
    "missing_lidar",
    "camera_lidar_dt_sec",
    "camera_image_path",
    "map_points_path",
    "lidar_timestamp",
    "camera_frame",
    "camera_timestamp",
    "map_x",
    "map_y",
    "map_yaw",
    "num_points",
}

OUTPUT_FIELDS = [
    "pair_index",
    "frame",
    "image_path",
    "pointcloud_path",
    "camera_frame",
    "lidar_timestamp",
    "camera_timestamp",
    "camera_lidar_dt_sec",
    "map_x",
    "map_y",
    "map_yaw",
    "num_points",
]


def resolve_source_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_eligible_rows(
    manifest_path: Path,
    repo_root: Path,
    max_delta_sec: float,
) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"manifest missing fields: {sorted(missing_fields)}")
        rows = list(reader)

    eligible: list[dict[str, str]] = []
    previous_timestamp: float | None = None
    for row in rows:
        if row["camera_match_valid"] != "1":
            continue
        if row["missing_image"] != "0" or row["missing_lidar"] != "0":
            continue
        if abs(float(row["camera_lidar_dt_sec"])) > max_delta_sec:
            continue
        image_path = resolve_source_path(row["camera_image_path"], repo_root)
        points_path = resolve_source_path(row["map_points_path"], repo_root)
        if not image_path.is_file() or not points_path.is_file():
            continue
        timestamp = float(row["lidar_timestamp"])
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("eligible rows are not monotonic by lidar_timestamp")
        previous_timestamp = timestamp
        eligible.append(row)
    return eligible


def uniform_sample(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("count must be positive")
    if len(rows) < count:
        raise ValueError(f"need {count} eligible rows, found {len(rows)}")
    if count == 1:
        return [rows[0]]
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise ValueError("uniform sampling produced duplicate indices")
    return [rows[index] for index in indices]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inside(root: Path, candidate: Path) -> None:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes subset: {candidate}") from exc


def validate_subset(
    output_dir: Path,
    expected_count: int,
    max_delta_sec: float,
) -> dict[str, float | int]:
    with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} manifest rows, found {len(rows)}")
    frames = [int(row["frame"]) for row in rows]
    if len(set(frames)) != expected_count:
        raise ValueError("duplicate frames in output manifest")

    deltas: list[float] = []
    for row in rows:
        image_rel = Path(row["image_path"])
        points_rel = Path(row["pointcloud_path"])
        if image_rel.is_absolute() or points_rel.is_absolute():
            raise ValueError("output manifest contains an absolute path")
        if image_rel.stem != points_rel.stem:
            raise ValueError(f"pair stem mismatch: {image_rel} vs {points_rel}")
        image_path = output_dir / image_rel
        points_path = output_dir / points_rel
        _require_inside(output_dir, image_path)
        _require_inside(output_dir, points_path)
        with Image.open(image_path) as image:
            image.verify()
        with np.load(points_path, allow_pickle=False) as data:
            if "points_map_xyzi" not in data:
                raise ValueError(f"missing points_map_xyzi in {points_path}")
            point_count = int(data["points_map_xyzi"].shape[0])
        if point_count != int(row["num_points"]):
            raise ValueError(f"point count mismatch for frame {row['frame']}")
        delta = abs(float(row["camera_lidar_dt_sec"]))
        if delta > max_delta_sec:
            raise ValueError(f"sync delta exceeds limit for frame {row['frame']}")
        deltas.append(delta)

    image_count = len(list((output_dir / "images").glob("*.png")))
    points_count = len(list((output_dir / "pointclouds").glob("*.npz")))
    if image_count != expected_count or points_count != expected_count:
        raise ValueError("output file counts do not match expected pair count")
    return {
        "pair_count": expected_count,
        "sync_abs_min_sec": min(deltas),
        "sync_abs_max_sec": max(deltas),
    }


def readme_text(pair_count: int, dataset_name: str) -> str:
    return f"""# Paired Camera/Point-Cloud Subset

This self-contained subset contains {pair_count} timestamp-matched pairs.
Files with the same six-digit stem belong together. Point clouds remain in the
corrected global map frame and are not directly in camera pixel coordinates.

```python
from pathlib import Path
import csv
import numpy as np
from PIL import Image

root = Path("{dataset_name}")
with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
    row = next(csv.DictReader(handle))
image = Image.open(root / row["image_path"])
with np.load(root / row["pointcloud_path"], allow_pickle=False) as data:
    points_map_xyzi = data["points_map_xyzi"]
```
"""


def _source_label(source_manifest: Path, repo_root: Path) -> str:
    try:
        return source_manifest.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return source_manifest.name


def build_subset(
    source_manifest: Path,
    repo_root: Path,
    output_dir: Path,
    archive_path: Path,
    count: int = 100,
    max_delta_sec: float = 0.04,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    try:
        images_dir = temporary / "images"
        points_dir = temporary / "pointclouds"
        images_dir.mkdir()
        points_dir.mkdir()
        eligible = load_eligible_rows(source_manifest, repo_root, max_delta_sec)
        selected = uniform_sample(eligible, count)
        output_rows: list[dict[str, object]] = []
        for pair_index, row in enumerate(selected):
            stem = f"{int(row['frame']):06d}"
            image_rel = Path("images") / f"{stem}.png"
            points_rel = Path("pointclouds") / f"{stem}.npz"
            shutil.copy2(
                resolve_source_path(row["camera_image_path"], repo_root),
                temporary / image_rel,
            )
            shutil.copy2(
                resolve_source_path(row["map_points_path"], repo_root),
                temporary / points_rel,
            )
            output_rows.append(
                {
                    "pair_index": pair_index,
                    "frame": int(row["frame"]),
                    "image_path": image_rel.as_posix(),
                    "pointcloud_path": points_rel.as_posix(),
                    "camera_frame": int(row["camera_frame"]),
                    "lidar_timestamp": float(row["lidar_timestamp"]),
                    "camera_timestamp": float(row["camera_timestamp"]),
                    "camera_lidar_dt_sec": float(row["camera_lidar_dt_sec"]),
                    "map_x": float(row["map_x"]),
                    "map_y": float(row["map_y"]),
                    "map_yaw": float(row["map_yaw"]),
                    "num_points": int(row["num_points"]),
                }
            )

        with (temporary / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        validation = validate_subset(temporary, count, max_delta_sec)
        metadata = {
            "source_manifest": _source_label(source_manifest, repo_root),
            "selection_method": "uniform_manifest_order_including_endpoints",
            "pair_count": count,
            "max_camera_lidar_delta_sec": max_delta_sec,
            "sync_abs_min_sec": validation["sync_abs_min_sec"],
            "sync_abs_max_sec": validation["sync_abs_max_sec"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pointcloud_key": "points_map_xyzi",
            "pointcloud_coordinate_frame": "corrected_global_map",
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            readme_text(count, output_dir.name), encoding="utf-8"
        )

        with tarfile.open(temporary_archive, "w") as archive:
            archive.add(temporary, arcname=output_dir.name)
        temporary.rename(output_dir)
        os.replace(temporary_archive, archive_path)
        return {
            **validation,
            "output_dir": str(output_dir),
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if temporary_archive.exists():
            temporary_archive.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small paired camera/point-cloud subset"
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs/frame_map_dataset_pose_corrected_final/frames.csv",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/parking_dataset_100"
    )
    parser.add_argument(
        "--archive", type=Path, default=ROOT.parent / "parking_dataset_100.tar"
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-camera-delta-sec", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_subset(
        args.source_manifest,
        args.repo_root,
        args.output_dir,
        args.archive,
        args.count,
        args.max_camera_delta_sec,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
