#!/usr/bin/env python3
"""Audit LiDAR-to-camera axis/translation conventions against CARLA depth images."""

from __future__ import annotations

import argparse
import base64
import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


FX = 527.525085
FY = 527.525085
CX = 636.297913
CY = 357.787354
MAX_DEPTH_M = 1000.0


@dataclass(frozen=True)
class Candidate:
    name: str
    optical_from_lidar: np.ndarray
    translation_lidar: np.ndarray
    translation_mode: str

    def transform(self, points_lidar: np.ndarray) -> np.ndarray:
        points = np.asarray(points_lidar, dtype=np.float64)
        if self.translation_mode == "camera_origin_in_lidar":
            relative = points - self.translation_lidar
        elif self.translation_mode == "lidar_origin_in_camera":
            relative = points + self.translation_lidar
        elif self.translation_mode == "legacy_optical_add":
            optical = points @ self.optical_from_lidar.T
            return optical + self.translation_lidar
        else:
            raise ValueError(self.translation_mode)
        return relative @ self.optical_from_lidar.T


STANDARD_AXES = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
MIRRORED_AXES = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def candidates() -> list[Candidate]:
    translation_lidar = np.asarray([0.6, 0.0, -0.07], dtype=np.float64)
    return [
        Candidate("standard_camera_origin", STANDARD_AXES, translation_lidar, "camera_origin_in_lidar"),
        Candidate("standard_lidar_origin", STANDARD_AXES, translation_lidar, "lidar_origin_in_camera"),
        Candidate("mirrored_camera_origin", MIRRORED_AXES, translation_lidar, "camera_origin_in_lidar"),
        Candidate("mirrored_lidar_origin", MIRRORED_AXES, translation_lidar, "lidar_origin_in_camera"),
        Candidate("legacy", STANDARD_AXES, translation_lidar, "legacy_optical_add"),
    ]


def decode_carla_depth(path: Path, *, red_is_low_byte: bool) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    if red_is_low_byte:
        packed = rgb[..., 0] + 256.0 * rgb[..., 1] + 65536.0 * rgb[..., 2]
    else:
        packed = rgb[..., 2] + 256.0 * rgb[..., 1] + 65536.0 * rgb[..., 0]
    return packed / float(256**3 - 1) * MAX_DEPTH_M


def project(points_camera: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    u = FX * points_camera[:, 0] / np.maximum(z, 1e-9) + CX
    v = FY * points_camera[:, 1] / np.maximum(z, 1e-9) + CY
    valid = (z > 1.0) & (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
    return np.column_stack([u, v]), valid


def depth_residuals(points_camera: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    height, width = depth_m.shape
    uv, valid = project(points_camera, width, height)
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    selected = points_camera[valid]
    pixels = np.rint(uv[valid]).astype(np.int64)
    camera_depth = np.linalg.norm(selected, axis=1)
    sampled = []
    for u, v in pixels:
        patch = depth_m[v - 1 : v + 2, u - 1 : u + 2]
        sampled.append(float(np.median(patch)))
    sampled_depth = np.asarray(sampled, dtype=np.float64)
    keep = (sampled_depth > 0.5) & (sampled_depth < 200.0) & (camera_depth < 200.0)
    return np.abs(sampled_depth[keep] - camera_depth[keep])


def grayscale_range_pairs(points_camera: np.ndarray, grayscale: np.ndarray) -> np.ndarray:
    """Return log-range/depth-intensity pairs after a sparse nearest-point z-buffer."""
    height, width = grayscale.shape
    uv, valid = project(points_camera, width, height)
    selected = points_camera[valid]
    pixels = np.rint(uv[valid]).astype(np.int64)
    ranges = np.linalg.norm(selected, axis=1)
    keep = (ranges > 1.0) & (ranges < 120.0)
    pixels = pixels[keep]
    ranges = ranges[keep]
    if not len(ranges):
        return np.empty((0, 2), dtype=np.float64)
    flat = pixels[:, 1] * width + pixels[:, 0]
    order = np.argsort(ranges)
    flat = flat[order]
    pixels = pixels[order]
    ranges = ranges[order]
    _, first = np.unique(flat, return_index=True)
    pixels = pixels[first]
    ranges = ranges[first]
    intensity = grayscale[pixels[:, 1], pixels[:, 0]]
    return np.column_stack([np.log(ranges), intensity])


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(frames_csv: Path, dataset_root: Path, max_frames: int, stride: int) -> dict[str, dict[str, float]]:
    depth_orders = {"red_low": True, "blue_low": False}
    residuals: dict[str, list[np.ndarray]] = {
        f"{candidate.name}/{order}": []
        for candidate in candidates()
        for order in depth_orders
    }
    correlation_pairs: dict[str, list[np.ndarray]] = {candidate.name: [] for candidate in candidates()}
    used = 0
    for row in load_rows(frames_csv)[:: max(1, stride)]:
        camera_frame = int(float(row["camera_frame"]))
        depth_path = dataset_root / "depth" / f"depth{camera_frame:06d}.png"
        lidar_path = Path(row["lidar_path"])
        if not depth_path.exists() or not lidar_path.exists():
            continue
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) > 10000:
            points = points[:: max(1, len(points) // 10000)]
        grayscale = np.asarray(Image.open(depth_path).convert("L"), dtype=np.float64)
        for candidate in candidates():
            pairs = grayscale_range_pairs(candidate.transform(points), grayscale)
            if len(pairs):
                correlation_pairs[candidate.name].append(pairs)
        for order, red_is_low_byte in depth_orders.items():
            depth = decode_carla_depth(depth_path, red_is_low_byte=red_is_low_byte)
            for candidate in candidates():
                values = depth_residuals(candidate.transform(points), depth)
                if len(values):
                    residuals[f"{candidate.name}/{order}"].append(values)
        used += 1
        if used >= max_frames:
            break

    result: dict[str, dict[str, float]] = {}
    for name, chunks in residuals.items():
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        candidate_name = name.split("/", 1)[0]
        pair_chunks = correlation_pairs[candidate_name]
        pairs = np.concatenate(pair_chunks) if pair_chunks else np.empty((0, 2), dtype=np.float64)
        correlation = 0.0
        if len(pairs) >= 2 and np.std(pairs[:, 0]) > 0 and np.std(pairs[:, 1]) > 0:
            correlation = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
        result[name] = {
            "frames": float(used),
            "points": float(len(values)),
            "median_abs_depth_error_m": float(np.median(values)) if len(values) else float("inf"),
            "p75_abs_depth_error_m": float(np.quantile(values, 0.75)) if len(values) else float("inf"),
            "within_1m_rate": float(np.mean(values <= 1.0)) if len(values) else 0.0,
            "grayscale_log_range_correlation": correlation,
            "correlation_points": float(len(pairs)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("outputs/frame_map_dataset_pose_corrected_final/frames.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ParkingAgent/dataset/dataset/dataset"))
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--preview-image", type=Path)
    args = parser.parse_args()
    if args.preview_image:
        image = Image.open(args.preview_image).convert("RGB")
        image.thumbnail((640, 360))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=65, optimize=True)
        print(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return 0
    results = audit(args.frames, args.dataset_root, args.max_frames, args.stride)
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["median_abs_depth_error_m"]):
        print(name, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
