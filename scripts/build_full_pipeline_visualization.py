#!/usr/bin/env python3
"""Build a data-driven, step-by-step Part1-to-Part2 visualization for frame 9277."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Rectangle, Wedge
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
STATE_COLORS = {
    "free": "#14b86e",
    "occupied": "#f05252",
    "unknown": "#8c98a8",
}
INK = "#162033"
MUTED = "#657086"
BLUE = "#2e6ee6"
ORANGE = "#f59e0b"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="#f7f8fb")
    plt.close(fig)


def setup_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#e7eaf0", linewidth=0.7, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#dfe3ea")


def ego_xy(points: np.ndarray, anchor_xy: np.ndarray, yaw: float, scale: float) -> np.ndarray:
    delta = (np.asarray(points, dtype=float) - anchor_xy[None, :]) / scale
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]))


def decode_metadata(pack: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = bytes(np.asarray(pack["metadata_json"], dtype=np.uint8).tolist())
    return json.loads(raw.decode("utf-8"))


def decision_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["slot_id"]: row for row in payload["decisions"]}


def gate_ratio(gate: dict[str, Any]) -> float:
    value = gate.get("value")
    threshold = str(gate.get("threshold", ""))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 1.0 if gate.get("passed") else 0.65
    try:
        target = float(
            threshold.replace(">=", "").replace("<=", "").replace(">", "").replace("<", "")
        )
    except ValueError:
        return 1.0 if gate.get("passed") else 0.65
    if target == 0:
        return 1.0
    if threshold.startswith("<"):
        return min(target / max(float(value), 1e-9), 1.35)
    return min(float(value) / target, 1.35)


def slot_pack_map(queue: dict[str, Any], artifact: Path) -> dict[str, Path]:
    raw_resources = queue.get("resources", {})
    if isinstance(raw_resources, dict):
        resources = raw_resources
    else:
        resources = {row["resource_id"]: row for row in raw_resources}
    result: dict[str, Path] = {}
    for item in queue["items"]:
        ids = item["encounter"].get("pointcloud_resource_ids", [])
        if not ids:
            continue
        resource = resources.get(ids[0], {})
        candidate = resource.get("path") or resource.get("uri") or resource.get("local_path")
        if candidate:
            path = Path(candidate)
            if not path.is_absolute():
                path = artifact / path
            if path.is_file():
                result[item["slot_id"]] = path
                continue
        task_hash = item["task_id"].split(":", 1)[-1]
        fallback = artifact / "part2_lidar_evidence" / f"{task_hash}.npz"
        if fallback.is_file():
            result[item["slot_id"]] = fallback
    if len(result) != len(queue["items"]):
        remaining = list((artifact / "part2_lidar_evidence").glob("*.npz"))
        metadata_lookup: dict[str, Path] = {}
        for path in remaining:
            with np.load(path, allow_pickle=False) as pack:
                metadata = decode_metadata(pack)
            slot_id = metadata.get("slot_id") or metadata.get("target_slot_id")
            if slot_id:
                metadata_lookup[str(slot_id)] = path
        for item in queue["items"]:
            if item["slot_id"] in metadata_lookup:
                result[item["slot_id"]] = metadata_lookup[item["slot_id"]]
    return result


def copy_camera_frames(frames: list[dict[str, Any]], assets: Path) -> list[dict[str, Any]]:
    selected = [frames[0], frames[len(frames) // 2], frames[-1]]
    rows: list[dict[str, Any]] = []
    for frame in selected:
        source = Path(frame["camera_image_path"])
        target = assets / f"camera_lidar_{frame['frame_id']:06d}.jpg"
        if source.is_file():
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((1280, 720))
                image.save(target, quality=88)
        rows.append(
            {
                "frame_id": frame["frame_id"],
                "camera_frame": frame["camera_frame"],
                "dt_ms": 1000.0 * frame["camera_lidar_dt_sec"],
                "asset": f"full_pipeline_assets/{target.name}" if target.is_file() else None,
            }
        )
    return rows


def load_frame_clouds(frames: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[int]]:
    clouds: list[np.ndarray] = []
    counts: list[int] = []
    for row in frames:
        path = ROOT / row["map_points_path"]
        with np.load(path, allow_pickle=False) as pack:
            cloud = np.asarray(pack["points_map_xyzi"], dtype=np.float32)
        clouds.append(cloud)
        counts.append(int(cloud.shape[0]))
    return clouds, counts


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def reconstruct_original_xy(
    corrected_xy: np.ndarray,
    corrected_pose: np.ndarray,
    original_pose: np.ndarray,
    scale: float,
) -> np.ndarray:
    delta = (np.asarray(corrected_xy, dtype=float) - corrected_pose[None, :2]) / scale
    c, s = math.cos(float(corrected_pose[2])), math.sin(float(corrected_pose[2]))
    local = np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]))
    co, so = math.cos(float(original_pose[2])), math.sin(float(original_pose[2]))
    world_delta = np.column_stack((co * local[:, 0] - so * local[:, 1], so * local[:, 0] + co * local[:, 1]))
    return original_pose[None, :2] + scale * world_delta


def load_pose_audit(
    frames: list[dict[str, Any]],
    clouds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    corrected: list[np.ndarray] = []
    original: list[np.ndarray] = []
    original_clouds: list[np.ndarray] = []
    for frame, cloud in zip(frames, clouds):
        path = ROOT / frame["map_points_path"]
        with np.load(path, allow_pickle=False) as pack:
            corr = np.asarray(pack["ego_map_pose"], dtype=float)
            orig = np.asarray(pack["original_ego_map_pose"], dtype=float)
            scale = float(np.asarray(pack["map_scale"]).reshape(-1)[0])
        corrected.append(corr)
        original.append(orig)
        reconstructed = cloud.copy()
        reconstructed[:, :2] = reconstruct_original_xy(cloud[:, :2], corr, orig, scale)
        original_clouds.append(reconstructed)
    return np.asarray(corrected), np.asarray(original), original_clouds


def voxel_downsample(points: np.ndarray, voxel_m: float = 0.18, max_points: int = 7000) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    result = points[np.sort(indices)]
    if len(result) > max_points:
        step = int(math.ceil(len(result) / max_points))
        result = result[::step]
    return result


def rigid_fit_2d(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left = source - source_center
    right = target - target_center
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def cloud_residual(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if len(source) == 0 or len(target) == 0:
        return float("nan"), float("nan")
    distances, _ = cKDTree(target).query(source, k=1, workers=-1)
    distances = distances[np.isfinite(distances)]
    return float(np.median(distances)), float(np.quantile(distances, 0.90))


def local_icp(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    current = source.copy()
    total_rotation = np.eye(2)
    total_translation = np.zeros(2)
    tree = cKDTree(target)
    for _ in range(10):
        distances, indices = tree.query(current, k=1, workers=-1)
        mask = np.isfinite(distances) & (distances <= 1.0)
        if int(mask.sum()) < 40:
            break
        cutoff = float(np.quantile(distances[mask], 0.75))
        mask &= distances <= cutoff
        rotation, translation = rigid_fit_2d(current[mask], target[indices[mask]])
        current = current @ rotation.T + translation
        total_translation = total_translation @ rotation.T + translation
        total_rotation = rotation @ total_rotation
        if float(np.linalg.norm(translation)) < 1e-4 and abs(math.atan2(rotation[1, 0], rotation[0, 0])) < 1e-5:
            break
    median, p90 = cloud_residual(current, target)
    yaw = math.degrees(math.atan2(total_rotation[1, 0], total_rotation[0, 0]))
    return current, median, p90, float(np.linalg.norm(total_translation)), float(yaw)


def registration_audit(
    clouds: list[np.ndarray],
    original_clouds: list[np.ndarray],
    anchor_xy: np.ndarray,
    yaw: float,
    scale: float,
) -> dict[str, Any]:
    def prepare(cloud: np.ndarray) -> np.ndarray:
        xy = ego_xy(cloud[:, :2], anchor_xy, yaw, scale)
        z = cloud[:, 2]
        mask = (np.linalg.norm(xy, axis=1) <= 20.0) & (z >= 0.15) & (z <= 2.5)
        return voxel_downsample(xy[mask])

    corrected_local = [prepare(cloud) for cloud in clouds]
    original_local = [prepare(cloud) for cloud in original_clouds]
    pairs: list[dict[str, Any]] = []
    for index in range(1, len(clouds)):
        corr_p50, corr_p90 = cloud_residual(corrected_local[index], corrected_local[index - 1])
        orig_p50, orig_p90 = cloud_residual(original_local[index], original_local[index - 1])
        _, icp_p50, icp_p90, icp_translation, icp_yaw = local_icp(
            corrected_local[index], corrected_local[index - 1]
        )
        pairs.append(
            {
                "left_index": index - 1,
                "right_index": index,
                "corrected_p50_m": corr_p50,
                "corrected_p90_m": corr_p90,
                "original_p50_m": orig_p50,
                "original_p90_m": orig_p90,
                "icp_p50_m": icp_p50,
                "icp_p90_m": icp_p90,
                "icp_translation_m": icp_translation,
                "icp_yaw_deg": icp_yaw,
            }
        )
    return {
        "pairs": pairs,
        "corrected_median_p50_m": float(np.nanmedian([row["corrected_p50_m"] for row in pairs])),
        "original_median_p50_m": float(np.nanmedian([row["original_p50_m"] for row in pairs])),
        "icp_median_p50_m": float(np.nanmedian([row["icp_p50_m"] for row in pairs])),
        "corrected_median_p90_m": float(np.nanmedian([row["corrected_p90_m"] for row in pairs])),
        "original_median_p90_m": float(np.nanmedian([row["original_p90_m"] for row in pairs])),
        "icp_median_p90_m": float(np.nanmedian([row["icp_p90_m"] for row in pairs])),
        "median_icp_translation_m": float(np.nanmedian([row["icp_translation_m"] for row in pairs])),
        "median_abs_icp_yaw_deg": float(np.nanmedian(np.abs([row["icp_yaw_deg"] for row in pairs]))),
    }


def figure_localization_audit(
    frames: list[dict[str, Any]],
    corrected: np.ndarray,
    original: np.ndarray,
    registration: dict[str, Any],
    scale: float,
    assets: Path,
) -> dict[str, Any]:
    position_shift = np.linalg.norm((corrected[:, :2] - original[:, :2]) / scale, axis=1)
    yaw_shift = np.degrees([wrap_angle(a - b) for a, b in zip(corrected[:, 2], original[:, 2])])
    anchor_xy = corrected[-1, :2]
    anchor_yaw = corrected[-1, 2]
    corr_local = ego_xy(corrected[:, :2], anchor_xy, anchor_yaw, scale)
    orig_local = ego_xy(original[:, :2], anchor_xy, anchor_yaw, scale)
    ids = [row["frame_id"] for row in frames]
    pairs = registration["pairs"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    setup_axes(axes[0, 0], "Original pose vs corrected pose trajectory")
    axes[0, 0].plot(orig_local[:, 0], orig_local[:, 1], "--o", color="#94a3b8", label="original pose", markersize=3)
    axes[0, 0].plot(corr_local[:, 0], corr_local[:, 1], "-o", color=BLUE, label="corrected pose", markersize=3)
    for old, new in zip(orig_local, corr_local):
        axes[0, 0].plot([old[0], new[0]], [old[1], new[1]], color=ORANGE, linewidth=0.8)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("anchor-forward (m)")
    axes[0, 0].set_ylabel("anchor-left (m)")
    axes[0, 0].legend(frameon=False)
    setup_axes(axes[0, 1], "Applied pose correction magnitude")
    axes[0, 1].bar(ids, position_shift, color=BLUE, label="position shift")
    axes[0, 1].set_ylabel("position correction (m)")
    twin = axes[0, 1].twinx()
    twin.plot(ids, yaw_shift, "-o", color=ORANGE, markersize=3, label="yaw shift")
    twin.set_ylabel("yaw correction (deg)", color=ORANGE)
    axes[0, 1].tick_params(axis="x", rotation=50)
    setup_axes(axes[1, 0], "Consecutive-cloud nearest-neighbor residual (p50)")
    pair_labels = [f"{frames[row['left_index']]['frame_id']}→{frames[row['right_index']]['frame_id']}" for row in pairs]
    x = np.arange(len(pairs))
    axes[1, 0].plot(x, [row["original_p50_m"] for row in pairs], "-o", color="#94a3b8", label="original pose", markersize=3)
    axes[1, 0].plot(x, [row["corrected_p50_m"] for row in pairs], "-o", color=BLUE, label="corrected pose", markersize=3)
    axes[1, 0].plot(x, [row["icp_p50_m"] for row in pairs], "-o", color="#14b86e", label="local ICP diagnostic", markersize=3)
    axes[1, 0].set_xticks(x, pair_labels, rotation=55, ha="right", fontsize=6)
    axes[1, 0].set_ylabel("nearest-neighbor residual (m)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    setup_axes(axes[1, 1], "Residual correction still suggested by local ICP")
    axes[1, 1].bar(x - 0.18, [row["icp_translation_m"] for row in pairs], 0.36, color="#14b86e", label="translation")
    axes[1, 1].set_ylabel("extra translation (m)")
    twin2 = axes[1, 1].twinx()
    twin2.bar(x + 0.18, [abs(row["icp_yaw_deg"]) for row in pairs], 0.36, color=ORANGE, alpha=0.75, label="|yaw|")
    twin2.set_ylabel("extra |yaw| (deg)", color=ORANGE)
    axes[1, 1].set_xticks(x, [str(frames[row["right_index"]]["frame_id"]) for row in pairs], rotation=50)
    fig.suptitle("LOCALIZATION AUDIT A · Pose correction and multi-frame consistency", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "02A_localization_pose_and_residual.png")
    return {
        "position_shift_m": position_shift.tolist(),
        "yaw_shift_deg": yaw_shift.tolist(),
        "max_position_shift_m": float(position_shift.max()),
        "median_position_shift_m": float(np.median(position_shift)),
        "max_abs_yaw_shift_deg": float(np.max(np.abs(yaw_shift))),
        **{key: value for key, value in registration.items() if key != "pairs"},
    }


def point_membership(cloud_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(cloud_xy) == 0:
        return np.zeros(0, dtype=bool)
    low = polygon.min(axis=0)
    high = polygon.max(axis=0)
    box = np.all((cloud_xy >= low) & (cloud_xy <= high), axis=1)
    result = np.zeros(len(cloud_xy), dtype=bool)
    if box.any():
        result[box] = MplPath(polygon).contains_points(cloud_xy[box], radius=1e-9)
    return result


def boundary_audit(
    clouds: list[np.ndarray],
    original_clouds: list[np.ndarray],
    local_db: dict[str, Any],
) -> dict[str, Any]:
    slot_ids = [row["slot_id"] for row in local_db["slots"]]
    result: dict[str, Any] = {"slot_ids": slot_ids}
    for label, source_clouds in (("corrected", clouds), ("original", original_clouds)):
        slot_counts = np.zeros((len(source_clouds), len(slot_ids)), dtype=int)
        core_counts = np.zeros_like(slot_counts)
        for frame_index, cloud in enumerate(source_clouds):
            height = (cloud[:, 2] >= 0.3) & (cloud[:, 2] <= 2.2)
            xy = cloud[height, :2]
            for slot_index, slot in enumerate(local_db["slots"]):
                polygon = np.asarray(slot["polygon_map"], dtype=float)
                core = np.asarray(slot["core_polygon_map"], dtype=float)
                slot_counts[frame_index, slot_index] = int(point_membership(xy, polygon).sum())
                core_counts[frame_index, slot_index] = int(point_membership(xy, core).sum())
        boundary = np.maximum(slot_counts - core_counts, 0)
        share = boundary / np.maximum(slot_counts, 1)
        result[label] = {
            "slot_counts": slot_counts.tolist(),
            "core_counts": core_counts.tolist(),
            "boundary_share": share.tolist(),
            "median_boundary_share": float(np.median(share[slot_counts > 0])) if np.any(slot_counts > 0) else float("nan"),
        }
    return result


def slot_pose_sensitivity(pack_path: Path) -> dict[str, Any]:
    with np.load(pack_path, allow_pickle=False) as pack:
        points = np.asarray(pack["points_local_xyzi"])
        polygon = np.asarray(pack["polygon_local_m"])
        core = np.asarray(pack["core_polygon_local_m"])
    points = points[(points[:, 2] >= 0.3) & (points[:, 2] <= 2.2), :2]

    def metrics(transformed: np.ndarray) -> tuple[int, float, float]:
        in_slot = point_membership(transformed, polygon)
        in_core = point_membership(transformed, core)
        total = int(in_slot.sum())
        core_count = int((in_slot & in_core).sum())
        boundary_count = max(total - core_count, 0)
        return total, core_count / max(total, 1), boundary_count / max(total, 1)

    base_count, base_core, base_boundary = metrics(points)
    translations: list[dict[str, Any]] = []
    for magnitude in (0.0, 0.10, 0.25, 0.50):
        samples = []
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            shifted = points + magnitude * np.asarray([math.cos(angle), math.sin(angle)])
            samples.append(metrics(shifted))
        translations.append(
            {
                "magnitude_m": magnitude,
                "count_min": min(row[0] for row in samples),
                "count_max": max(row[0] for row in samples),
                "core_share_min": min(row[1] for row in samples),
                "core_share_max": max(row[1] for row in samples),
                "boundary_share_min": min(row[2] for row in samples),
                "boundary_share_max": max(row[2] for row in samples),
            }
        )
    rotations: list[dict[str, Any]] = []
    for degrees in (0.0, 0.5, 1.0, 2.0):
        samples = []
        for signed in ({0.0} if degrees == 0 else {-degrees, degrees}):
            angle = math.radians(signed)
            c, s = math.cos(angle), math.sin(angle)
            rotated = points @ np.asarray([[c, s], [-s, c]])
            samples.append(metrics(rotated))
        rotations.append(
            {
                "yaw_deg": degrees,
                "count_min": min(row[0] for row in samples),
                "count_max": max(row[0] for row in samples),
                "core_share_min": min(row[1] for row in samples),
                "core_share_max": max(row[1] for row in samples),
                "boundary_share_min": min(row[2] for row in samples),
                "boundary_share_max": max(row[2] for row in samples),
            }
        )
    return {
        "base_count": base_count,
        "base_core_share": base_core,
        "base_boundary_share": base_boundary,
        "translations": translations,
        "rotations": rotations,
    }


def figure_boundary_and_sensitivity(
    frames: list[dict[str, Any]],
    boundary: dict[str, Any],
    sensitivity: dict[str, Any],
    assets: Path,
) -> dict[str, Any]:
    slot_ids = boundary["slot_ids"]
    corrected_share = np.asarray(boundary["corrected"]["boundary_share"], dtype=float)
    corrected_counts = np.asarray(boundary["corrected"]["slot_counts"], dtype=int)
    original_share = np.asarray(boundary["original"]["boundary_share"], dtype=float)
    index = slot_ids.index("slot_1253")
    ids = [row["frame_id"] for row in frames]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    setup_axes(axes[0, 0], "Per-frame boundary-ring share after corrected pose")
    image = axes[0, 0].imshow(corrected_share.T, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[0, 0].set_xticks(range(len(ids)), ids, rotation=50, fontsize=6)
    axes[0, 0].set_yticks(range(len(slot_ids)), [sid.split("_")[-1] for sid in slot_ids], fontsize=6)
    axes[0, 0].set_xlabel("LiDAR frame")
    axes[0, 0].set_ylabel("slot id suffix")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.72, label="boundary share")
    setup_axes(axes[0, 1], "slot_1253 boundary evidence changes frame by frame")
    axes[0, 1].plot(ids, corrected_share[:, index], "-o", color=BLUE, label="corrected pose", markersize=4)
    axes[0, 1].plot(ids, original_share[:, index], "--o", color="#94a3b8", label="original pose", markersize=4)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[0, 1].set_ylabel("boundary-ring points / in-slot points")
    axes[0, 1].tick_params(axis="x", rotation=50)
    axes[0, 1].legend(frameon=False)
    translations = sensitivity["translations"]
    x = np.asarray([row["magnitude_m"] for row in translations])
    low = np.asarray([row["core_share_min"] for row in translations])
    high = np.asarray([row["core_share_max"] for row in translations])
    setup_axes(axes[1, 0], "slot_1253 sensitivity to XY localization error")
    axes[1, 0].fill_between(x, low, high, color=ORANGE, alpha=0.25, label="16 error directions")
    axes[1, 0].plot(x, (low + high) / 2, "-o", color=ORANGE)
    axes[1, 0].axhline(sensitivity["base_core_share"], color=BLUE, linestyle="--", label="zero perturbation")
    axes[1, 0].set_xlabel("translation error magnitude (m)")
    axes[1, 0].set_ylabel("core ownership share")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(frameon=False)
    rotations = sensitivity["rotations"]
    rx = np.asarray([row["yaw_deg"] for row in rotations])
    rlow = np.asarray([row["core_share_min"] for row in rotations])
    rhigh = np.asarray([row["core_share_max"] for row in rotations])
    setup_axes(axes[1, 1], "slot_1253 sensitivity to yaw error")
    axes[1, 1].fill_between(rx, rlow, rhigh, color="#ef4444", alpha=0.22, label="±yaw")
    axes[1, 1].plot(rx, (rlow + rhigh) / 2, "-o", color="#ef4444")
    axes[1, 1].axhline(sensitivity["base_core_share"], color=BLUE, linestyle="--", label="zero perturbation")
    axes[1, 1].set_xlabel("|yaw error| (deg)")
    axes[1, 1].set_ylabel("core ownership share")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(frameon=False)
    fig.suptitle("LOCALIZATION AUDIT B · Slot-boundary leakage and decision sensitivity", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "02B_slot_boundary_and_pose_sensitivity.png")
    valid = corrected_counts > 0
    return {
        "corrected_median_boundary_share": float(np.median(corrected_share[valid])),
        "original_median_boundary_share": float(np.median(original_share[np.asarray(boundary["original"]["slot_counts"]) > 0])),
        "slot1253_corrected_boundary_share_min": float(corrected_share[:, index].min()),
        "slot1253_corrected_boundary_share_max": float(corrected_share[:, index].max()),
        **sensitivity,
    }


def figure_inputs(
    frames: list[dict[str, Any]],
    counts: list[int],
    camera_rows: list[dict[str, Any]],
    assets: Path,
) -> None:
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    for index, row in enumerate(camera_rows):
        ax = fig.add_subplot(grid[0, index])
        if row["asset"]:
            image = np.asarray(Image.open(ROOT / "Nature_ParkingAgent_实验报告_20260728" / row["asset"]))
            ax.imshow(image)
        ax.set_title(
            f"LiDAR {row['frame_id']} / RGB {row['camera_frame']}\nΔt={row['dt_ms']:.2f} ms",
            fontsize=10,
            color=INK,
        )
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :2])
    setup_axes(ax, "15 causal LiDAR records used by Part1")
    ids = [row["frame_id"] for row in frames]
    ax.plot(ids, counts, color=BLUE, marker="o", linewidth=2)
    ax.fill_between(ids, counts, color=BLUE, alpha=0.12)
    ax.set_xlabel("LiDAR frame id")
    ax.set_ylabel("map points / frame")
    ax.ticklabel_format(style="plain", axis="y")
    ax = fig.add_subplot(grid[1, 2])
    setup_axes(ax, "Sensor synchronization audit")
    dts = [1000 * row["camera_lidar_dt_sec"] for row in frames]
    ax.barh(range(len(dts)), dts, color=ORANGE)
    ax.axvline(10, color="#b91c1c", linestyle="--", linewidth=1, label="10 ms")
    ax.set_xlabel("|camera − LiDAR| (ms)")
    ax.set_yticks([0, 7, 14], [str(ids[0]), str(ids[7]), str(ids[-1])])
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("STEP 1 · What data enters the system", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "01_inputs_and_sync.png")


def figure_alignment(
    frames: list[dict[str, Any]],
    clouds: list[np.ndarray],
    anchor_xy: np.ndarray,
    yaw: float,
    scale: float,
    assets: Path,
) -> dict[str, Any]:
    local_clouds: list[np.ndarray] = []
    for cloud in clouds:
        local = ego_xy(cloud[:, :2], anchor_xy, yaw, scale)
        mask = np.linalg.norm(local, axis=1) <= 22
        local_clouds.append(np.column_stack((local[mask], cloud[mask, 2:4])))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    chosen = [0, len(local_clouds) // 2, len(local_clouds) - 1]
    for idx, color in zip(chosen, ("#9db7ee", "#5b8def", "#153f9e")):
        points = local_clouds[idx][:: max(1, len(local_clouds[idx]) // 12000)]
        axes[0].scatter(points[:, 0], points[:, 1], s=0.35, c=color, alpha=0.45, label=str(frames[idx]["frame_id"]))
    setup_axes(axes[0], "Three aligned frames in anchor ego coordinates")
    axes[0].scatter([0], [0], marker="^", s=80, color="#111827", label="ego @ 9277")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-22, 22)
    axes[0].set_ylim(-22, 22)
    axes[0].set_xlabel("forward (m)")
    axes[0].set_ylabel("left (m)")
    axes[0].legend(frameon=False, fontsize=8)
    origins = ego_xy(
        np.asarray([[row["map_x"], row["map_y"]] for row in frames]),
        anchor_xy,
        yaw,
        scale,
    )
    setup_axes(axes[1], "Pose correction turns 15 scans into one local map")
    axes[1].plot(origins[:, 0], origins[:, 1], "-o", color=ORANGE, linewidth=2, markersize=4)
    for idx in (0, 7, 14):
        axes[1].annotate(str(frames[idx]["frame_id"]), origins[idx], xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].quiver(
        origins[:, 0],
        origins[:, 1],
        np.cos([row["map_yaw"] - yaw for row in frames]),
        np.sin([row["map_yaw"] - yaw for row in frames]),
        color=BLUE,
        width=0.006,
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("forward (m)")
    axes[1].set_ylabel("left (m)")
    merged = np.concatenate(local_clouds, axis=0)
    plot_points = merged[:: max(1, len(merged) // 90000)]
    setup_axes(axes[2], "Accumulated 15-frame point cloud")
    scatter = axes[2].scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        s=0.3,
        c=np.clip(plot_points[:, 2], -1, 3),
        cmap="turbo",
        alpha=0.55,
    )
    axes[2].set_xlim(-22, 22)
    axes[2].set_ylim(-22, 22)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("forward (m)")
    axes[2].set_ylabel("left (m)")
    fig.colorbar(scatter, ax=axes[2], shrink=0.72, label="height z (m)")
    fig.suptitle("STEP 2 · Pose-aligned multi-frame accumulation", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "02_pose_alignment_and_accumulation.png")
    motion = float(np.linalg.norm(origins[-1] - origins[0]))
    yaw_change = math.degrees(float(frames[-1]["map_yaw"] - frames[0]["map_yaw"]))
    return {"merged_points": int(sum(len(row) for row in local_clouds)), "motion_m": motion, "yaw_change_deg": yaw_change}


def draw_slot(ax: plt.Axes, polygon: np.ndarray, color: str, alpha: float = 0.2, lw: float = 1.0) -> None:
    ax.add_patch(Polygon(polygon, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=lw))


def figure_scope(
    full_db: dict[str, Any],
    local_db: dict[str, Any],
    local_map: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    anchor_xy: np.ndarray,
    yaw: float,
    scale: float,
    assets: Path,
) -> None:
    all_slots = full_db.get("slots", full_db.get("slot_database", []))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), layout="constrained")
    setup_axes(axes[0], "Known geometry database: 1,397 slots")
    for slot in all_slots:
        polygon = np.asarray(slot.get("polygon_map") or slot.get("polygon"), dtype=float)
        if polygon.shape == (4, 2):
            axes[0].plot(polygon[:, 0], polygon[:, 1], color="#aeb7c5", linewidth=0.25, alpha=0.65)
    axes[0].scatter(anchor_xy[0], anchor_xy[1], marker="^", color="#111827", s=40)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("map x")
    axes[0].set_ylabel("map y")
    setup_axes(axes[1], "Actual LiDAR envelope + 21 local slots")
    coverage = ego_xy(np.asarray(local_map["lidar_coverage"]["polygon_map"]), anchor_xy, yaw, scale)
    axes[1].add_patch(Polygon(coverage, closed=True, facecolor="#dbeafe", edgecolor=BLUE, alpha=0.35, linewidth=1.5))
    for slot in local_db["slots"]:
        poly = ego_xy(np.asarray(slot["polygon_map"]), anchor_xy, yaw, scale)
        state = decisions[slot["slot_id"]]["state"]
        draw_slot(axes[1], poly, STATE_COLORS[state], 0.4, 1.0)
        center = poly.mean(axis=0)
        axes[1].text(center[0], center[1], slot["slot_id"].split("_")[-1], fontsize=6, ha="center", va="center")
    axes[1].scatter([0], [0], marker="^", color="#111827", s=70, zorder=8)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-20, 20)
    axes[1].set_ylim(-20, 20)
    axes[1].set_xlabel("forward (m)")
    axes[1].set_ylabel("left (m)")
    setup_axes(axes[2], "Scope reduction is omission, not Occupied")
    labels = ["known geometry", "local evidence scope", "outside scope"]
    values = [len(all_slots), len(local_db["slots"]), len(all_slots) - len(local_db["slots"])]
    colors = [BLUE, "#14b86e", "#cbd2dc"]
    axes[2].barh(labels, values, color=colors)
    for index, value in enumerate(values):
        axes[2].text(value + 18, index, f"{value:,}", va="center", fontsize=11, color=INK, fontweight="bold")
    axes[2].set_xlim(0, max(values) * 1.18)
    axes[2].set_xlabel("slot count")
    fig.suptitle("STEP 3 · Restrict global geometry to actual local LiDAR scope", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "03_scope_and_coverage.png")


def plot_polygon(ax: plt.Axes, polygon: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
    closed = np.vstack((polygon, polygon[0]))
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2, linestyle=linestyle, label=label)


def figure_slot_extraction(
    pack_path: Path,
    frame_ids: np.ndarray,
    assets: Path,
) -> dict[str, int]:
    with np.load(pack_path, allow_pickle=False) as pack:
        points = np.asarray(pack["points_local_xyzi"])
        point_frames = np.asarray(pack["point_frame_ids"])
        polygon = np.asarray(pack["polygon_local_m"])
        core = np.asarray(pack["core_polygon_local_m"])
        margin = np.asarray(pack["margin_polygon_local_m"])
        origins = np.asarray(pack["observation_origins_local_xyz"])
    sample = np.arange(0, len(points), max(1, len(points) // 70000))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    setup_axes(axes[0], "Evidence ROI around slot_1253")
    scat = axes[0].scatter(points[sample, 0], points[sample, 1], s=0.55, c=points[sample, 2], cmap="turbo", alpha=0.55)
    plot_polygon(axes[0], margin, ORANGE, "margin ROI", "--")
    plot_polygon(axes[0], polygon, BLUE, "slot polygon")
    plot_polygon(axes[0], core, "#14b86e", "core polygon")
    axes[0].scatter(origins[:, 0], origins[:, 1], s=16, c="#111827", marker="x", label="15 origins")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("slot-local x (m)")
    axes[0].set_ylabel("slot-local y (m)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.colorbar(scat, ax=axes[0], shrink=0.68, label="z (m)")
    setup_axes(axes[1], "The same ROI, side view")
    axes[1].scatter(points[sample, 0], points[sample, 2], s=0.55, c=point_frames[sample], cmap="viridis", alpha=0.5)
    axes[1].axhspan(0.3, 0.8, color="#60a5fa", alpha=0.08, label="low layer")
    axes[1].axhspan(0.8, 1.4, color="#34d399", alpha=0.08, label="mid layer")
    axes[1].axhspan(1.4, 2.2, color="#fbbf24", alpha=0.08, label="high layer")
    axes[1].set_xlabel("slot-local x (m)")
    axes[1].set_ylabel("height z (m)")
    axes[1].set_ylim(-1, 3.2)
    axes[1].legend(frameon=False, fontsize=8)
    setup_axes(axes[2], "Temporal evidence: points retained per frame")
    counts = Counter(point_frames.tolist())
    values = [counts.get(int(frame), 0) for frame in frame_ids]
    axes[2].bar(frame_ids, values, color=BLUE)
    axes[2].set_xlabel("LiDAR frame")
    axes[2].set_ylabel("ROI points")
    axes[2].tick_params(axis="x", rotation=50)
    fig.suptitle("STEP 4 · Crop one slot into core / boundary / adjacent evidence", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "04_slot_evidence_extraction.png")
    return {"pack_points": int(len(points)), "valid_frames": int(len(set(point_frames.tolist())))}


def figure_occupied(decision: dict[str, Any], pack_path: Path, assets: Path) -> None:
    evidence = decision["occupied_evidence"]
    gates = evidence["gate_results"]
    with np.load(pack_path, allow_pickle=False) as pack:
        points = np.asarray(pack["points_local_xyzi"])
        frame_ids = np.asarray(pack["point_frame_ids"])
        polygon = np.asarray(pack["polygon_local_m"])
    best = dict(evidence["best_box"])
    center = np.asarray([best["center_x_m"], best["center_y_m"]])
    yaw = float(best["yaw_rad"])
    delta = points[:, :2] - center
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]))
    inside = (np.abs(local[:, 0]) <= best["length_m"] / 2) & (np.abs(local[:, 1]) <= best["width_m"] / 2)
    vehicle = points[inside]
    vehicle_frames = frame_ids[inside]
    sample = np.arange(0, len(points), max(1, len(points) // 50000))
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    setup_axes(ax, "Best vehicle-box hypothesis in slot-local BEV")
    ax.scatter(points[sample, 0], points[sample, 1], s=0.35, color="#cbd5e1", alpha=0.35)
    ax.scatter(vehicle[:, 0], vehicle[:, 1], s=4, c=vehicle[:, 2], cmap="turbo", alpha=0.8)
    plot_polygon(ax, polygon, BLUE, "slot")
    corners = np.array(
        [
            [-best["length_m"] / 2, -best["width_m"] / 2],
            [best["length_m"] / 2, -best["width_m"] / 2],
            [best["length_m"] / 2, best["width_m"] / 2],
            [-best["length_m"] / 2, best["width_m"] / 2],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    box = corners @ rot.T + center
    plot_polygon(ax, box, "#ef4444", "best box")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[0, 1])
    setup_axes(ax, "3D morphology of retained obstacle points")
    ax.scatter(vehicle[:, 0], vehicle[:, 2], s=5, c=vehicle_frames, cmap="viridis", alpha=0.75)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_ylim(-0.5, 2.6)
    ax = fig.add_subplot(grid[0, 2])
    setup_axes(ax, "Temporal support across 15 frames")
    counts = Counter(vehicle_frames.tolist())
    refs = decision["reference_frames"]
    ax.bar(refs, [counts.get(frame, 0) for frame in refs], color=ORANGE)
    ax.axhline(1, color="#111827", linewidth=0.7)
    ax.set_xlabel("frame")
    ax.set_ylabel("best-box points")
    ax.tick_params(axis="x", rotation=50)
    ax = fig.add_subplot(grid[1, :])
    setup_axes(ax, "Occupied gate audit — every gate must pass for strong Occupied")
    names = [row["name"] for row in gates]
    ratios = [gate_ratio(row) for row in gates]
    colors = ["#14b86e" if row["passed"] else "#ef4444" for row in gates]
    bars = ax.bar(range(len(names)), ratios, color=colors)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.48)
    ax.set_xticks(range(len(names)), names, rotation=42, ha="right")
    ax.set_ylabel("value / threshold (direction-normalized)")
    for bar, row in zip(bars, gates):
        marker = "PASS" if row["passed"] else "FAIL"
        ax.text(bar.get_x() + bar.get_width() / 2, min(bar.get_height() + 0.035, 1.39), marker, ha="center", fontsize=6, color=INK)
    fig.suptitle("STEP 5A · Occupied branch: box search → 3D features → hard gates", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "05_occupied_branch.png")


def voxel_array(details: dict[str, Any], name: str) -> np.ndarray:
    rows = details.get(name, [])
    return np.asarray(rows, dtype=int).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=int)


def figure_free(decision: dict[str, Any], pack_path: Path, assets: Path) -> None:
    evidence = decision["free_evidence"]
    details = evidence["details"]
    gates = evidence["gate_results"]
    with np.load(pack_path, allow_pickle=False) as pack:
        origins = np.asarray(pack["observation_origins_local_xyz"])
        core = np.asarray(pack["core_polygon_local_m"])
    categories = [
        ("unobserved_voxels", "#d7dce5", "unobserved"),
        ("free_voxels", "#76d7ae", "free traversal"),
        ("occluded_voxels", "#f5b041", "occluded"),
        ("hit_voxels", "#e74c3c", "hit"),
    ]
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    setup_axes(ax, "15 viewpoints and slot core")
    center = core.mean(axis=0)
    for origin in origins:
        ax.plot([origin[0], center[0]], [origin[1], center[1]], color="#93a4bf", linewidth=0.55, alpha=0.55)
    ax.scatter(origins[:, 0], origins[:, 1], c=np.arange(len(origins)), cmap="viridis", s=30)
    plot_polygon(ax, core, BLUE, "core")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[0, 1])
    setup_axes(ax, "Ray-classified voxel grid, BEV")
    for key, color, label in categories:
        vox = voxel_array(details, key)
        if len(vox):
            ax.scatter(vox[:, 0], vox[:, 1], s=9, color=color, alpha=0.75, label=f"{label} ({len(vox)})")
    ax.set_aspect("equal")
    ax.set_xlabel("voxel x")
    ax.set_ylabel("voxel y")
    ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(grid[0, 2])
    setup_axes(ax, "Ray-classified voxel grid, side view")
    for key, color, label in categories:
        vox = voxel_array(details, key)
        if len(vox):
            ax.scatter(vox[:, 0], vox[:, 2], s=9, color=color, alpha=0.7, label=label)
    ax.set_xlabel("voxel x")
    ax.set_ylabel("voxel z")
    ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(grid[1, :])
    setup_axes(ax, "Free-space gate audit — positive visibility is required; missing rays never mean Free")
    names = [row["name"] for row in gates]
    ratios = [gate_ratio(row) for row in gates]
    colors = ["#14b86e" if row["passed"] else "#ef4444" for row in gates]
    bars = ax.bar(range(len(names)), ratios, color=colors)
    ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.48)
    ax.set_xticks(range(len(names)), names, rotation=38, ha="right")
    ax.set_ylabel("direction-normalized gate score")
    for bar, row in zip(bars, gates):
        ax.text(bar.get_x() + bar.get_width() / 2, min(bar.get_height() + 0.035, 1.4), "PASS" if row["passed"] else "FAIL", ha="center", fontsize=6)
    fig.suptitle("STEP 5B · Free branch: ray traversal → visibility voxels → hard gates", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "06_free_ray_branch.png")


def figure_fusion(decisions: dict[str, dict[str, Any]], assets: Path) -> None:
    rows = list(decisions.values())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    setup_axes(axes[0], "Occupied strength vs Free strength")
    for row in rows:
        state = row["state"]
        x = row["occupied_evidence"]["strength"]
        y = row["free_evidence"]["strength"]
        axes[0].scatter(x, y, s=70, color=STATE_COLORS[state], edgecolor="white", linewidth=0.8)
        axes[0].annotate(row["slot_id"].split("_")[-1], (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
    axes[0].set_xlim(0, 1.04)
    axes[0].set_ylim(0, 1.04)
    axes[0].set_xlabel("Occupied evidence strength")
    axes[0].set_ylabel("Free evidence strength")
    axes[0].text(0.03, 0.96, "Strength is diagnostic;\nstrong state still needs all hard gates.", transform=axes[0].transAxes, va="top", fontsize=8, color=MUTED)
    setup_axes(axes[1], "Part1 three-state output")
    counts = Counter(row["state"] for row in rows)
    labels = ["free", "occupied", "unknown"]
    bars = axes[1].bar(labels, [counts[x] for x in labels], color=[STATE_COLORS[x] for x in labels])
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(int(bar.get_height())), ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(counts.values()) + 3)
    axes[1].set_ylabel("local slots")
    setup_axes(axes[2], "Why slots remain Unknown")
    reasons = Counter()
    for row in rows:
        if row["state"] == "unknown":
            reasons.update(row["unknown_reasons"])
    common = reasons.most_common(8)
    axes[2].barh([x[0].replace("_", " ") for x in common][::-1], [x[1] for x in common][::-1], color="#8c98a8")
    axes[2].set_xlabel("occurrences")
    fig.suptitle("STEP 6 · Deterministic three-state fusion", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "07_three_state_fusion.png")


def figure_pillar(
    baseline: dict[str, Any],
    improved: dict[str, Any],
    pack_path: Path,
    assets: Path,
) -> None:
    old = baseline["occupied_evidence"]["features"]
    new = improved["occupied_evidence"]["features"]
    with np.load(pack_path, allow_pickle=False) as pack:
        points = np.asarray(pack["points_local_xyzi"])
        polygon = np.asarray(pack["polygon_local_m"])
    mask = (points[:, 2] >= 0.3) & (points[:, 2] <= 2.3)
    sample = np.flatnonzero(mask)[:: max(1, int(mask.sum()) // 45000)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    setup_axes(axes[0], "Actual slot_1253 evidence geometry")
    axes[0].scatter(points[sample, 0], points[sample, 1], s=0.55, c=points[sample, 2], cmap="turbo", alpha=0.6)
    plot_polygon(axes[0], polygon, BLUE, "slot")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(frameon=False, fontsize=8)
    setup_axes(axes[1], "Legacy gate: line-like only")
    legacy_values = [
        old["robust_pca_linearity"] / 0.95,
        0.75 / max(old["robust_extent_x_m"], old["robust_extent_y_m"]),
        old["robust_point_fraction"] / 0.80,
    ]
    labels = ["PCA / 0.95", "0.75 / maxXY", "inlier / 0.80"]
    colors = ["#14b86e" if value >= 1 else "#ef4444" for value in legacy_values]
    axes[1].bar(labels, legacy_values, color=colors)
    axes[1].axhline(1, color="#111827", linestyle="--")
    axes[1].set_ylim(0, 1.55)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].text(0.04, 0.94, "PCA=0.915 < 0.95\n→ old pillar veto did not trigger\n→ Occupied", transform=axes[1].transAxes, va="top", fontsize=10, color="#b91c1c", fontweight="bold")
    setup_axes(axes[2], "New physical footprint gate")
    area = old["robust_extent_x_m"] * old["robust_extent_y_m"]
    aspect = old["extent_z_m"] / max(old["robust_extent_x_m"], old["robust_extent_y_m"])
    values = [
        old["robust_pca_linearity"] / 0.90,
        0.30 / area,
        old["extent_z_m"] / 0.80,
        aspect / 1.50,
        old["robust_point_fraction"] / 0.70,
    ]
    labels = ["PCA", "area", "height", "aspect", "inlier"]
    axes[2].bar(labels, np.minimum(values, 2.0), color=["#14b86e" if value >= 1 else "#ef4444" for value in values])
    axes[2].axhline(1, color="#111827", linestyle="--")
    axes[2].set_ylim(0, 2.15)
    axes[2].text(0.04, 0.94, f"All physical ratios pass\narea={area:.3f} m², aspect={aspect:.2f}\n→ block strong Occupied → Unknown", transform=axes[2].transAxes, va="top", fontsize=10, color="#047857", fontweight="bold")
    fig.suptitle("STEP 7 · Pillar-risk protection: legacy Occupied → safer Unknown", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "08_pillar_before_after.png")


def figure_part1_map(
    local_db: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    anchor_xy: np.ndarray,
    yaw: float,
    scale: float,
    assets: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.0), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    setup_axes(axes[0], "Part1 output map: every local slot has a state")
    for slot in local_db["slots"]:
        sid = slot["slot_id"]
        poly = ego_xy(np.asarray(slot["polygon_map"]), anchor_xy, yaw, scale)
        draw_slot(axes[0], poly, STATE_COLORS[decisions[sid]["state"]], 0.58, 1.3)
        center = poly.mean(axis=0)
        axes[0].text(center[0], center[1], sid.split("_")[-1], ha="center", va="center", fontsize=7, color=INK)
    axes[0].arrow(0, 0, 3.4, 0, head_width=0.45, head_length=0.6, color="#111827", length_includes_head=True)
    axes[0].text(1.3, 0.55, "forward", fontsize=8)
    axes[0].set_xlim(-19, 19)
    axes[0].set_ylim(-19, 19)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("forward (m)")
    axes[0].set_ylabel("left (m)")
    axes[0].legend(
        handles=[Rectangle((0, 0), 1, 1, color=STATE_COLORS[x], label=x) for x in ("free", "occupied", "unknown")],
        frameon=False,
        loc="upper right",
    )
    axes[1].axis("off")
    y = 0.97
    axes[1].text(0, y, "21-slot decision ledger", fontsize=14, fontweight="bold", color=INK, transform=axes[1].transAxes)
    y -= 0.06
    for sid in sorted(decisions, key=lambda x: int(x.split("_")[-1])):
        row = decisions[sid]
        axes[1].add_patch(Rectangle((0, y - 0.018), 0.035, 0.032, transform=axes[1].transAxes, color=STATE_COLORS[row["state"]]))
        axes[1].text(0.05, y, sid, fontsize=8, color=INK, va="center", transform=axes[1].transAxes)
        axes[1].text(0.25, y, row["state"], fontsize=8, color=STATE_COLORS[row["state"]], va="center", fontweight="bold", transform=axes[1].transAxes)
        axes[1].text(0.43, y, row["decision_reason"].replace("_", " "), fontsize=7.2, color=MUTED, va="center", transform=axes[1].transAxes)
        y -= 0.041
    fig.suptitle("STEP 8 · Part1 local-map output", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "09_part1_output_map.png")


def candidate_rows(
    local_db: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    queue: dict[str, Any],
    anchor_xy: np.ndarray,
    yaw: float,
    scale: float,
) -> list[dict[str, Any]]:
    queued = {row["slot_id"]: row for row in queue["items"]}
    rows: list[dict[str, Any]] = []
    for slot in local_db["slots"]:
        sid = slot["slot_id"]
        center = ego_xy(np.asarray([slot["center_map"]]), anchor_xy, yaw, scale)[0]
        bearing = math.degrees(math.atan2(center[1], center[0]))
        distance = float(np.linalg.norm(center))
        rows.append(
            {
                "slot_id": sid,
                "state": decisions[sid]["state"],
                "agent_observable": bool(decisions[sid]["agent_context"]["agent_observable"]),
                "forward_m": float(center[0]),
                "left_m": float(center[1]),
                "distance_m": distance,
                "bearing_deg": bearing,
                "in_front_180": abs(bearing) <= 90.0 + 1e-9,
                "queued": sid in queued,
                "unknown_reasons": decisions[sid]["unknown_reasons"],
            }
        )
    return rows


def figure_candidates(rows: list[dict[str, Any]], assets: Path) -> None:
    unknown = [row for row in rows if row["state"] == "unknown"]
    queued = [row for row in rows if row["queued"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    ax = axes[0]
    setup_axes(ax, "Unknown slots in current ego frame")
    ax.add_patch(Wedge((0, 0), 18, -90, 90, facecolor="#dbeafe", edgecolor=BLUE, alpha=0.35))
    ax.add_patch(Wedge((0, 0), 18, 90, 270, facecolor="#f1f3f6", edgecolor="#cbd2dc", alpha=0.45))
    for row in unknown:
        color = "#14b86e" if row["queued"] else "#9aa5b3"
        marker = "o" if row["queued"] else "x"
        ax.scatter(row["forward_m"], row["left_m"], color=color, marker=marker, s=80, zorder=5)
        ax.annotate(row["slot_id"].split("_")[-1], (row["forward_m"], row["left_m"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.arrow(0, 0, 3.5, 0, color="#111827", head_width=0.45, head_length=0.6, length_includes_head=True)
    ax.set_xlim(-19, 19)
    ax.set_ylim(-19, 19)
    ax.set_aspect("equal")
    ax.set_xlabel("forward (m)")
    ax.set_ylabel("left (m)")
    setup_axes(axes[1], "Candidate gate waterfall")
    labels = ["local states", "Unknown", "observable", "distance ≤18m", "front |β|≤90°", "evidence pack"]
    values = [
        len(rows),
        len(unknown),
        sum(row["agent_observable"] for row in unknown),
        sum(row["agent_observable"] and row["distance_m"] <= 18 for row in unknown),
        len(queued),
        len(queued),
    ]
    axes[1].plot(values, range(len(values)), "-o", linewidth=3, markersize=9, color=BLUE)
    axes[1].fill_betweenx(range(len(values)), values, color=BLUE, alpha=0.12)
    axes[1].invert_yaxis()
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_xlim(0, max(values) + 3)
    axes[1].set_xlabel("remaining slots")
    for index, value in enumerate(values):
        axes[1].text(value + 0.35, index, str(value), va="center", fontweight="bold")
    setup_axes(axes[2], "Final Part2 queue, sorted for audit")
    ordered = sorted(queued, key=lambda row: abs(row["bearing_deg"]))
    axes[2].barh(
        [row["slot_id"] for row in ordered],
        [row["bearing_deg"] for row in ordered],
        color=["#60a5fa" if row["bearing_deg"] >= 0 else "#f59e0b" for row in ordered],
    )
    axes[2].axvline(0, color="#111827", linewidth=1)
    axes[2].axvline(90, color="#b91c1c", linestyle="--")
    axes[2].axvline(-90, color="#b91c1c", linestyle="--")
    axes[2].set_xlim(-100, 100)
    axes[2].set_xlabel("relative bearing β (deg)")
    for index, row in enumerate(ordered):
        axes[2].text(row["bearing_deg"] + (2 if row["bearing_deg"] >= 0 else -2), index, f"{row['bearing_deg']:+.1f}°", va="center", ha="left" if row["bearing_deg"] >= 0 else "right", fontsize=8)
    fig.suptitle("STEP 9 · Candidate selection: Unknown → forward 180° → evidence-ready queue", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "10_candidate_filter.png")


def figure_packs(
    queue: dict[str, Any],
    pack_paths: dict[str, Path],
    assets: Path,
) -> list[dict[str, Any]]:
    items = {row["slot_id"]: row for row in queue["items"]}
    column_count = 3 if len(pack_paths) > 4 else 2
    row_count = int(math.ceil(len(pack_paths) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 5.5 * row_count),
        layout="constrained",
        squeeze=False,
    )
    summaries: list[dict[str, Any]] = []
    for ax, sid in zip(axes.ravel(), sorted(pack_paths)):
        with np.load(pack_paths[sid], allow_pickle=False) as pack:
            points = np.asarray(pack["points_local_xyzi"])
            frames = np.asarray(pack["selected_frames"])
            polygon = np.asarray(pack["polygon_local_m"])
            origins = np.asarray(pack["observation_origins_local_xyz"])
        sample = np.arange(0, len(points), max(1, len(points) // 45000))
        setup_axes(ax, f"{sid} · {len(points):,} points · {len(frames)} frames")
        scat = ax.scatter(points[sample, 0], points[sample, 1], s=0.45, c=points[sample, 2], cmap="turbo", alpha=0.55)
        plot_polygon(ax, polygon, BLUE, "slot")
        ax.scatter(origins[:, 0], origins[:, 1], marker="x", s=18, color="#111827", label="viewpoints")
        ax.set_aspect("equal")
        ax.set_xlabel("slot-local x (m)")
        ax.set_ylabel("slot-local y (m)")
        ax.legend(frameon=False, fontsize=7)
        item = items[sid]
        summaries.append(
            {
                "slot_id": sid,
                "point_count": int(len(points)),
                "selected_frames": [int(x) for x in frames],
                "bearing_deg": item["audit"]["candidate_relative_bearing_deg"],
                "occupied_strength": item["occupied_evidence"]["strength"],
                "free_strength": item["free_evidence"]["strength"],
                "unknown_reasons": item["unknown_reasons"],
                "modalities": item["available_modalities"],
            }
        )
    for ax in axes.ravel()[len(pack_paths) :]:
        ax.axis("off")
    fig.suptitle(
        f"STEP 10 · What Part2 actually receives: {len(pack_paths)} self-contained LiDAR evidence packs",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    savefig(fig, assets / "11_part2_evidence_packs.png")
    return summaries


def figure_effects(
    baseline: dict[str, Any],
    improved: dict[str, Any],
    study: dict[str, Any],
    current_queue_count: int,
    assets: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    setup_axes(axes[0], "Part1 state counts @ frame 9277")
    labels = ["free", "occupied", "unknown"]
    before = Counter(row["state"] for row in baseline.values())
    after = Counter(row["state"] for row in improved.values())
    x = np.arange(3)
    axes[0].bar(x - 0.18, [before[k] for k in labels], 0.36, label="legacy", color="#94a3b8")
    axes[0].bar(x + 0.18, [after[k] for k in labels], 0.36, label="pillar-safe", color=[STATE_COLORS[k] for k in labels])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("slots")
    axes[0].legend(frameon=False)
    paired_9277 = next(row for row in study["paired_part1"] if int(row["anchor_frame"]) == 9277)
    load_values = [
        int(paired_9277["baseline_part2_candidates_360deg"]),
        int(paired_9277["same_unknowns_in_forward_180deg"]),
        int(current_queue_count),
    ]
    setup_axes(axes[1], "Part2 load @ frame 9277")
    axes[1].bar(["legacy 360°\nUnknown", "same Unknown\nfront 180°", "new Part1 +\nfront 180°"], load_values, color=["#94a3b8", "#60a5fa", BLUE])
    axes[1].set_ylabel("candidate tasks")
    for index, value in enumerate(load_values):
        axes[1].text(index, value + 0.2, str(value), ha="center", fontweight="bold")
    aggregates = study["development_aggregate"]
    ws = sorted({int(row["history_span_W"]) for row in aggregates})
    ks = sorted({int(row["sample_count_K"]) for row in aggregates})
    matrix = np.full((len(ks), len(ws)), np.nan)
    for row in aggregates:
        matrix[ks.index(int(row["sample_count_K"])), ws.index(int(row["history_span_W"]))] = 100 * float(row["exact_dense_agreement_rate"])
    setup_axes(axes[2], "W/K development agreement to dense proxy (%)")
    image = axes[2].imshow(matrix, cmap="Blues", vmin=np.nanmin(matrix), vmax=100, aspect="auto")
    axes[2].set_xticks(range(len(ws)), ws)
    axes[2].set_yticks(range(len(ks)), ks)
    axes[2].set_xlabel("history span W")
    axes[2].set_ylabel("sample count K")
    for i in range(len(ks)):
        for j in range(len(ws)):
            if np.isfinite(matrix[i, j]):
                axes[2].text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(image, ax=axes[2], shrink=0.7)
    fig.suptitle("STEP 11 · Measured effect and its limits", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "12_effects_and_ablation.png")


def figure_high_precision_gt(
    comparison: dict[str, Any],
    gate_audit: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    assets: Path,
) -> None:
    results = {row["model"]: row for row in comparison["results"]}
    selected = [results["v3"], results["v4_high_precision"]]
    names = ["v3", "v4 high precision"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")

    setup_axes(axes[0, 0], "Manual-GT overlap: terminal decisions vs abstentions")
    x = np.arange(2)
    free = np.asarray([row["state_counts_on_gt_overlap"]["free"] for row in selected])
    occupied = np.asarray([row["state_counts_on_gt_overlap"]["occupied"] for row in selected])
    unknown = np.asarray([row["state_counts_on_gt_overlap"]["unknown"] for row in selected])
    axes[0, 0].bar(x, free, color=STATE_COLORS["free"], label="Free")
    axes[0, 0].bar(x, occupied, bottom=free, color=STATE_COLORS["occupied"], label="Occupied")
    axes[0, 0].bar(x, unknown, bottom=free + occupied, color=STATE_COLORS["unknown"], label="Unknown / abstain")
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].set_ylabel("slots on GT overlap")
    axes[0, 0].legend(frameon=False)
    for index, row in enumerate(selected):
        axes[0, 0].text(index, free[index] + occupied[index] + unknown[index] + 0.2, f"false Occupied = {row['false_occupied_count']}", ha="center", fontsize=9, fontweight="bold")

    setup_axes(axes[0, 1], "Accuracy must be reported together with coverage")
    accuracy = [100 * float(row["selective_accuracy"] or 0.0) for row in selected]
    coverage = [100 * float(row["terminal_coverage_on_overlap"] or 0.0) for row in selected]
    effective = [100 * float(row["effective_correct_coverage_on_overlap"] or 0.0) for row in selected]
    axes[0, 1].bar(x - 0.24, accuracy, 0.24, color=BLUE, label="selective accuracy")
    axes[0, 1].bar(x, coverage, 0.24, color=ORANGE, label="terminal coverage")
    axes[0, 1].bar(x + 0.24, effective, 0.24, color="#14b86e", label="effective correct coverage")
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylim(0, 112)
    axes[0, 1].set_ylabel("percent")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].text(0.02, 0.96, "v4 Occupied precision = N/A\n(no Occupied output)", transform=axes[0, 1].transAxes, va="top", fontsize=9, color="#b91c1c", fontweight="bold")

    setup_axes(axes[1, 0], "slot_1252: why v4 refuses Occupied")
    slot = decisions["slot_1252"]
    gates = {
        row["name"]: row
        for row in slot["occupied_evidence"]["gate_results"]
    }
    short_actual = float(gates["robust_short_extent"]["value"])
    short_min = 0.75
    outside_actual = float(gates["outside_residual"]["value"])
    outside_max = 0.35
    low_actual = float(gates["low_layer_coverage"]["value"])
    low_min = 0.05
    ratios = [short_actual / short_min, outside_max / max(outside_actual, 1e-9), low_actual / low_min]
    gate_names = ["footprint\nactual / minimum", "purity\nmaximum / actual", "low body\nactual / minimum"]
    colors = ["#14b86e" if value >= 1 else "#ef4444" for value in ratios]
    axes[1, 0].bar(gate_names, ratios, color=colors)
    axes[1, 0].axhline(1.0, color="#111827", linestyle="--", linewidth=1.2)
    axes[1, 0].set_ylim(0, max(2.2, max(ratios) + 0.2))
    axes[1, 0].set_ylabel("pass ratio (≥1 passes)")
    axes[1, 0].text(0.02, 0.96, f"short extent {short_actual:.3f} < {short_min:.2f} m\noutside residual {outside_actual:.3f} > {outside_max:.2f}\n→ Unknown, not Occupied", transform=axes[1, 0].transAxes, va="top", fontsize=10, color="#b91c1c", fontweight="bold")

    setup_axes(axes[1, 1], "Historical 60-frame Occupied decisions under v4 added gates")
    audit_rows = gate_audit["rows"]
    slots = [row["slot_id"] for row in audit_rows]
    failure_counts = [len(str(row["strict_rejection_reasons"]).split(";")) for row in audit_rows]
    colors = [ORANGE if slot == "slot_1017" else "#ef4444" for slot in slots]
    axes[1, 1].barh(slots[::-1], failure_counts[::-1], color=colors[::-1])
    axes[1, 1].set_xlabel("number of failed v4 added gates")
    axes[1, 1].set_xticks(range(0, max(failure_counts) + 1))
    axes[1, 1].text(0.02, 0.04, "8/8 rejected; orange 1017 is GT Occupied\nand shows the deliberate recall cost", transform=axes[1, 1].transAxes, fontsize=9, color=MUTED)

    fig.suptitle("STEP 13 · High-precision Occupied policy: measured safety gain and coverage cost", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "14_high_precision_gt_audit.png")


def figure_uncertainty_ablation(ablation: dict[str, Any], assets: Path) -> dict[str, Any]:
    runs = ablation["runs"]
    widths = sorted({float(row["uncertainty_width_m"]) for row in runs})
    anchors = [int(value) for value in ablation["anchors"]]
    aggregates: list[dict[str, Any]] = []
    for width in widths:
        selected = [row for row in runs if float(row["uncertainty_width_m"]) == width]
        aggregates.append(
            {
                "width_m": width,
                "free": sum(int(row["counts"]["free"]) for row in selected),
                "occupied": sum(int(row["counts"]["occupied"]) for row in selected),
                "unknown": sum(int(row["counts"]["unknown"]) for row in selected),
                "candidates": sum(int(row["candidate_count"]) for row in selected),
                "occupied_to_unknown": sum(int(row["occupied_to_unknown"]) for row in selected),
                "unsafe_promotions": sum(int(row["unsafe_promotions_to_occupied"]) for row in selected),
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    x = np.arange(len(widths))
    free = np.asarray([row["free"] for row in aggregates])
    occupied = np.asarray([row["occupied"] for row in aggregates])
    unknown = np.asarray([row["unknown"] for row in aggregates])
    setup_axes(axes[0], "Six-anchor Part1 state totals")
    axes[0].bar(x, free, color=STATE_COLORS["free"], label="Free")
    axes[0].bar(x, occupied, bottom=free, color=STATE_COLORS["occupied"], label="Occupied")
    axes[0].bar(x, unknown, bottom=free + occupied, color=STATE_COLORS["unknown"], label="Unknown")
    axes[0].set_xticks(x, [f"{value:.2f}" for value in widths])
    axes[0].set_xlabel("uncertainty inset (m)")
    axes[0].set_ylabel("slot states across 6 anchors")
    axes[0].legend(frameon=False, fontsize=8)
    setup_axes(axes[1], "Transitions relative to width=0")
    axes[1].bar(x - 0.18, [row["occupied_to_unknown"] for row in aggregates], 0.36, color=ORANGE, label="Occupied→Unknown")
    axes[1].bar(x + 0.18, [row["unsafe_promotions"] for row in aggregates], 0.36, color="#b91c1c", label="Free/Unknown→Occupied")
    axes[1].set_xticks(x, [f"{value:.2f}" for value in widths])
    axes[1].set_xlabel("uncertainty inset (m)")
    axes[1].set_ylabel("transition count")
    axes[1].legend(frameon=False, fontsize=8)
    setup_axes(axes[2], "Part2 candidate workload")
    axes[2].plot(x, [row["candidates"] for row in aggregates], "-o", color=BLUE, linewidth=2)
    axes[2].set_xticks(x, [f"{value:.2f}" for value in widths])
    axes[2].set_xlabel("uncertainty inset (m)")
    axes[2].set_ylabel("forward candidates across 6 anchors")
    for index, row in enumerate(aggregates):
        axes[2].text(index, row["candidates"] + 0.25, str(row["candidates"]), ha="center", fontsize=8)
    fig.suptitle("LOCALIZATION AUDIT C · Core uncertainty-width ablation", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "02C_localization_uncertainty_ablation.png")
    safe = [row for row in aggregates if row["unsafe_promotions"] == 0]
    max_safe_width = max(row["width_m"] for row in safe)
    return {
        "aggregates": aggregates,
        "anchors": anchors,
        "max_width_without_unsafe_promotion_m": max_safe_width,
        "selected_experimental_width_m": min(max_safe_width, 0.10),
        "ground_truth_available": bool(ablation.get("ground_truth_available", False)),
    }


def figure_final_validation(
    blinded: dict[str, Any],
    development: dict[str, Any],
    cap_005: dict[str, Any],
    cap_010: dict[str, Any],
    part2: dict[str, Any],
    assets: Path,
) -> dict[str, Any]:
    blind = blinded["metrics"]["evidence_aligned"]["frozen_v1_width_0p10"]
    dev_key = next(iter(development["metrics"]["evidence_aligned"]))
    dev = development["metrics"]["evidence_aligned"][dev_key]

    def total_transitions(payload: dict[str, Any], ratio: float) -> tuple[int, int]:
        rows = [
            row
            for row in payload["runs"]
            if abs(float(row["upper_height_spread_ratio"]) - ratio) < 1e-9
        ]
        return (
            sum(int(row["occupied_to_unknown"]) for row in rows),
            sum(int(row["unsafe_promotions_to_occupied"]) for row in rows),
        )

    transitions_005, unsafe_005 = total_transitions(cap_005, 0.05)
    transitions_010, unsafe_010 = total_transitions(cap_010, 0.10)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    setup_axes(axes[0], "Human labels: safety vs coverage")
    names = ["v1 blind", "v3 dev"]
    x = np.arange(2)
    width = 0.24
    for offset, key, label, color in (
        (-width, "selective_terminal_accuracy", "terminal accuracy", BLUE),
        (0.0, "coverage", "coverage", ORANGE),
        (width, "effective_exact_rate_unknown_as_incorrect", "effective exact", STATE_COLORS["free"]),
    ):
        values = [100 * float(blind[key]), 100 * float(dev[key])]
        axes[0].bar(x + offset, values, width, color=color, label=label)
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + 2, f"{value:.0f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("percent")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(0.02, 0.02, "8 labels / 6 observable\nv3 is development, not holdout", transform=axes[0].transAxes, fontsize=8, color=MUTED)

    setup_axes(axes[1], "Static-cap safety/workload")
    ratios = ["0.05", "0.10"]
    transitions = [transitions_005, transitions_010]
    unsafe = [unsafe_005, unsafe_010]
    axes[1].bar(np.arange(2) - 0.18, transitions, 0.36, color=ORANGE, label="Occupied→Unknown")
    axes[1].bar(np.arange(2) + 0.18, unsafe, 0.36, color="#b91c1c", label="unsafe promotion")
    axes[1].set_xticks(np.arange(2), ratios)
    axes[1].set_xlabel("upper-height-spread threshold")
    axes[1].set_ylabel("six-anchor transitions")
    axes[1].legend(frameon=False, fontsize=8)
    for index, value in enumerate(transitions):
        axes[1].text(index - 0.18, value + 0.4, str(value), ha="center", fontsize=9)

    setup_axes(axes[2], "Part2: current vs historical audit")
    current = part2["current_v3"]["agent_result"]
    historical = part2["overlapping_historical_openai_audit"]["overlapping_slots"]
    current_values = [current["free"], current["occupied"], current["unknown"]]
    historical_values = [
        sum(row["historical_agent_state"] == "free" for row in historical.values()),
        sum(row["historical_agent_state"] == "occupied" for row in historical.values()),
        sum(row["historical_agent_state"] == "unknown" for row in historical.values()),
    ]
    labels = ["Free", "Occupied", "Unknown"]
    x = np.arange(3)
    axes[2].bar(x - 0.18, current_values, 0.36, color=BLUE, label="current v3 strict")
    axes[2].bar(x + 0.18, historical_values, 0.36, color=ORANGE, label="historical OpenAI overlap")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("slot count")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].text(0.02, 0.96, "Historical result is not merged:\nqueue/config hashes differ", transform=axes[2].transAxes, va="top", fontsize=8, color=MUTED)
    fig.suptitle("STEP 12 · Ground-truth audit, safety ablation and Part2 boundary", fontsize=18, fontweight="bold", color=INK)
    savefig(fig, assets / "13_validation_and_part2.png")
    return {
        "blind_terminal_accuracy": blind["selective_terminal_accuracy"],
        "blind_coverage": blind["coverage"],
        "development_terminal_accuracy": dev["selective_terminal_accuracy"],
        "development_coverage": dev["coverage"],
        "cap_005_occupied_to_unknown": transitions_005,
        "cap_010_occupied_to_unknown": transitions_010,
        "unsafe_promotions_005": unsafe_005,
        "current_part2": current,
        "historical_overlap_count": len(historical),
    }


def slim_decision(row: dict[str, Any]) -> dict[str, Any]:
    def slim_gates(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": gate["name"],
                "passed": gate["passed"],
                "value": gate.get("value"),
                "threshold": gate.get("threshold"),
            }
            for gate in gates
        ]

    return {
        "slot_id": row["slot_id"],
        "state": row["state"],
        "decision_reason": row["decision_reason"],
        "unknown_reasons": row["unknown_reasons"],
        "agent_observable": row["agent_context"]["agent_observable"],
        "occupied": {
            "strong": row["occupied_evidence"]["strong"],
            "weak": row["occupied_evidence"]["weak"],
            "strength": row["occupied_evidence"]["strength"],
            "failures": row["occupied_evidence"]["failures"],
            "gates": slim_gates(row["occupied_evidence"]["gate_results"]),
        },
        "free": {
            "strong": row["free_evidence"]["strong"],
            "strength": row["free_evidence"]["strength"],
            "failures": row["free_evidence"]["failures"],
            "gates": slim_gates(row["free_evidence"]["gate_results"]),
        },
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ParkingAgent · Part1→Part2 全流程可视化</title>
<style>
:root{--ink:#162033;--muted:#657086;--blue:#2e6ee6;--line:#e1e5ec;--bg:#f4f6f9;--card:#fff;--free:#14b86e;--occ:#f05252;--unk:#8c98a8;--orange:#f59e0b}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.55}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr)}aside{position:sticky;top:0;height:100vh;padding:26px 18px;background:#101827;color:#fff;overflow:auto}aside h1{font-size:19px;margin:0 0 6px}aside p{font-size:12px;color:#aeb9ca;margin:0 0 22px}.nav a{display:flex;gap:10px;align-items:center;color:#bdc7d8;text-decoration:none;padding:9px 10px;border-radius:8px;font-size:13px}.nav a:hover,.nav a.active{background:#26344d;color:#fff}.num{display:grid;place-items:center;width:23px;height:23px;border-radius:50%;background:#33425c;font-size:11px;flex:none}
main{min-width:0}.hero{padding:64px clamp(28px,6vw,90px) 54px;background:linear-gradient(125deg,#fff 30%,#eaf1ff)}.eyebrow{color:var(--blue);font-size:13px;font-weight:800;letter-spacing:.12em}.hero h2{font-size:clamp(34px,5vw,66px);line-height:1.05;max-width:1000px;margin:14px 0 18px;letter-spacing:-.04em}.hero p{font-size:18px;color:var(--muted);max-width:900px}.hero-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:30px}.metric{background:rgba(255,255,255,.84);border:1px solid #dfe6f2;border-radius:14px;padding:16px}.metric b{display:block;font-size:27px}.metric span{font-size:12px;color:var(--muted)}
.truth{margin-top:24px;padding:15px 18px;border-left:5px solid var(--orange);background:#fff7e8;border-radius:8px;font-size:14px}.content{max-width:1450px;margin:auto;padding:26px clamp(20px,4vw,58px) 90px}.stage{scroll-margin-top:20px;background:var(--card);border:1px solid var(--line);border-radius:18px;margin:24px 0;padding:30px;box-shadow:0 6px 24px rgba(29,45,75,.05)}.stage-head{display:flex;align-items:flex-start;gap:17px}.badge{display:grid;place-items:center;width:43px;height:43px;background:var(--blue);color:#fff;border-radius:12px;font-weight:800;flex:none}.stage h3{font-size:27px;line-height:1.2;margin:2px 0 7px}.stage-head p{margin:0;color:var(--muted)}.io{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:22px 0}.io>div{border:1px solid var(--line);border-radius:11px;padding:13px;background:#fafbfc}.io small{display:block;color:var(--muted);font-size:10px;font-weight:800;letter-spacing:.09em;margin-bottom:5px}.io b{font-size:13px}.figure{position:relative;margin:22px 0 10px;background:#f7f8fb;border:1px solid var(--line);border-radius:14px;overflow:hidden}.figure img{width:100%;display:block;cursor:zoom-in}.caption{padding:12px 16px;background:#fff;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}.formula{font-family:"SFMono-Regular",Consolas,monospace;background:#111827;color:#e5edff;border-radius:12px;padding:17px 20px;overflow:auto;margin:16px 0;font-size:13px}.callout{padding:14px 16px;border-radius:10px;background:#edf4ff;border-left:4px solid var(--blue);margin:16px 0}.callout.red{background:#fff0f0;border-color:var(--occ)}.callout.green{background:#eafaf3;border-color:var(--free)}.chips{display:flex;gap:7px;flex-wrap:wrap}.chip{padding:4px 9px;border-radius:20px;background:#edf0f5;font-size:11px}.state-free{color:var(--free)}.state-occupied{color:var(--occ)}.state-unknown{color:var(--unk)}
.explorer{display:grid;grid-template-columns:260px 1fr;gap:18px}.slot-list{max-height:620px;overflow:auto;border:1px solid var(--line);border-radius:12px;padding:8px}.slot-button{display:flex;width:100%;justify-content:space-between;border:0;background:transparent;padding:9px 10px;border-radius:8px;color:var(--ink);cursor:pointer}.slot-button:hover,.slot-button.active{background:#edf4ff}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.detail{border:1px solid var(--line);border-radius:12px;padding:18px;min-width:0}.detail h4{font-size:21px;margin:0 0 6px}.score-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.score{padding:12px;border-radius:9px;background:#f7f8fb}.score b{font-size:22px}.gate-table{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}.gate-table th,.gate-table td{text-align:left;padding:7px;border-bottom:1px solid var(--line)}.pass{color:var(--free);font-weight:800}.fail{color:var(--occ);font-weight:800}.pack-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.pack{border:1px solid var(--line);border-radius:11px;padding:13px}.pack h4{margin:0 0 8px}.provenance{font-size:12px;color:var(--muted);border-top:1px solid var(--line);margin-top:25px;padding-top:15px}.modal{display:none;position:fixed;z-index:99;inset:0;background:rgba(3,9,20,.92);align-items:center;justify-content:center;padding:25px}.modal.open{display:flex}.modal img{max-width:96vw;max-height:92vh;background:#fff}.close{position:absolute;right:22px;top:15px;color:#fff;font-size:35px;cursor:pointer}
@media(max-width:1050px){.layout{grid-template-columns:1fr}aside{display:none}.hero-grid{grid-template-columns:repeat(2,1fr)}.io{grid-template-columns:repeat(2,1fr)}.explorer{grid-template-columns:1fr}.pack-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:620px){.hero{padding:38px 20px}.content{padding:14px}.stage{padding:18px}.hero-grid,.io,.score-grid,.pack-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="layout">
<aside><h1>Part1 → Part2</h1><p>frame 9277 · 真实数据逐步回放</p><nav class="nav">
<a href="#s1"><span class="num">1</span>传感器输入</a><a href="#s2"><span class="num">2</span>位姿对齐与融合</a><a href="#loc"><span class="num">L</span>定位质量审计</a><a href="#s3"><span class="num">3</span>局部范围</a><a href="#s4"><span class="num">4</span>车位证据裁剪</a><a href="#s5"><span class="num">5</span>Occupied 分支</a><a href="#s6"><span class="num">6</span>Free-ray 分支</a><a href="#s7"><span class="num">7</span>三态融合</a><a href="#s8"><span class="num">8</span>高精度安全门</a><a href="#s9"><span class="num">9</span>Part1 地图</a><a href="#s10"><span class="num">10</span>候选筛选</a><a href="#s11"><span class="num">11</span>Part2 证据包</a><a href="#s12"><span class="num">12</span>效果与边界</a><a href="#s13"><span class="num">13</span>旧验证审计</a><a href="#s14"><span class="num">14</span>新人工GT审计</a>
</nav></aside>
<main>
<header class="hero"><div class="eyebrow">DATA-DRIVEN PIPELINE REPLAY · V4 HIGH PRECISION</div><h2>从15帧 LiDAR 到5个 Part2 候选，完整看见每一步</h2><p>本页只使用 frame 9277 的真实输入和当前代码输出。每个阶段都明确展示：用了什么数据、做了什么处理、中间量是什么、输出如何影响下一步。</p>
<div class="hero-grid"><div class="metric"><b>15</b><span>因果 LiDAR 帧 9263–9277</span></div><div class="metric"><b id="mergedMetric">—</b><span>局部融合点（半径22m内）</span></div><div class="metric"><b>1397→21</b><span>全库几何→实际局部范围</span></div><div class="metric"><b>2 / 0 / 19</b><span>v4 Free / Occupied / Unknown</span></div><div class="metric"><b>19→5</b><span>Unknown→前向 Part2 完整队列</span></div></div>
<div class="truth"><b>先说明真实系统边界：</b>v4 贯彻“只有非常确认才 Occupied”：1252 从错误 Occupied 降为 Unknown，当前21车位不再输出任何 Occupied。5个前方180° evidence pack 已完整生成；由于没有可信 Camera target correspondence，本页不伪造 Part2 终态。</div></header>
<div class="content">

<section class="stage" id="s1"><div class="stage-head"><span class="badge">1</span><div><h3>传感器输入与15帧因果窗口</h3><p>只看当前帧及之前的数据，不使用未来帧。</p></div></div>
<div class="io"><div><small>输入</small><b>LiDAR 9263–9277 + 对应位姿</b></div><div><small>算法</small><b>连续因果窗口；Camera 仅审计同步</b></div><div><small>中间量</small><b>15个点云、15个位姿、15个时间差</b></div><div><small>输出/效果</small><b>Part1 可复现输入清单</b></div></div>
<div class="figure"><img src="full_pipeline_assets/01_inputs_and_sync.png"><div class="caption">上：三个真实 RGB 同步帧；下：15个真实 LiDAR 点数与相机时间差。Camera 当前没有 metric depth，因此不参加 Part1 三态判断。</div></div>
<div class="formula">frames = [9263, 9264, …, 9277]  ·  causal = true  ·  future_frames = ∅</div></section>

<section class="stage" id="s2"><div class="stage-head"><span class="badge">2</span><div><h3>位姿校正、坐标变换与多帧融合</h3><p>把每帧点从自己的 LiDAR/地图坐标统一到 anchor 9277 的车辆坐标。</p></div></div>
<div class="io"><div><small>输入</small><b>每帧 points_map_xyzi + map pose</b></div><div><small>算法</small><b>平移到 anchor，再旋转 −yaw</b></div><div><small>中间量</small><b>15个对齐后的 ego-frame 点云</b></div><div><small>输出/效果</small><b>一个时序累积局部点云</b></div></div>
<div class="formula">Δp = (p_map − p_anchor) / map_scale<br>p_ego = R(−yaw_anchor) · Δp</div>
<div class="figure"><img src="full_pipeline_assets/02_pose_alignment_and_accumulation.png"><div class="caption">左：三个时刻对齐结果；中：真实15帧车辆运动和朝向；右：融合点云按高度着色。融合增加支持帧和观察角度，但不会把未来信息带入。</div></div></section>

<section class="stage" id="loc"><div class="stage-head"><span class="badge">L</span><div><h3>定位质量审计：先判断15帧是否真的对齐</h3><p>这里不使用最终车位状态倒推定位好坏，而是直接比较 original pose、corrected pose、相邻点云残差和车位边界证据。</p></div></div>
<div class="io"><div><small>真实数据</small><b>15个 map_points NPZ 中的 ego_map_pose / original_ego_map_pose</b></div><div><small>诊断算法</small><b>重建原始点云；相邻帧最近邻残差；局部2D ICP仅作残差探针</b></div><div><small>定位中间量</small><b>平移/航向修正、p50/p90残差、额外ICP修正</b></div><div><small>对下游影响</small><b>core/boundary归属漂移会改变 Occupied ownership</b></div></div>
<div class="figure"><img src="full_pipeline_assets/02A_localization_pose_and_residual.png"><div class="caption">左上比较原始与修正轨迹；右上是每帧实际施加的pose correction；下方直接比较相邻点云一致性。ICP结果只是“仍有多少局部刚体修正空间”的诊断代理，不当作真值位姿。</div></div>
<div class="score-grid"><div class="score">最大位置修正<br><b id="poseShift">—</b></div><div class="score">最大航向修正<br><b id="yawShift">—</b></div><div class="score">original / corrected p50<br><b id="poseResidual">—</b></div><div class="score">local ICP 后 p50<br><b id="icpResidual">—</b></div></div>
<div class="callout" id="registrationConclusion">正在从真实数据计算定位结论。</div>
<div class="figure"><img src="full_pipeline_assets/02B_slot_boundary_and_pose_sensitivity.png"><div class="caption">上：21车位逐帧边界环点占比和 slot_1253 原始/修正位姿对比；下：把 slot_1253 点云人为平移0.10/0.25/0.50m或旋转0.5/1/2°，观察core ownership的变化范围。</div></div>
<div class="score-grid"><div class="score">slot_1253 逐帧 boundary 范围<br><b id="boundaryRange">—</b></div><div class="score">0.25m误差下 core ownership 范围<br><b id="translationSensitivity">—</b></div></div>
<div class="callout" id="boundaryConclusion">正在计算车位地图对齐敏感性。</div>
<div class="figure"><img src="full_pipeline_assets/02C_localization_uncertainty_ablation.png"><div class="caption">六个开发/留出锚点的0/0.05/0.10/0.15m消融。0.15m在frame 3160触发1个Unknown→Occupied的候选重排序反例，因此被淘汰；0.10m是当前“无危险提升”的最大实验宽度，不代表已由GT证明最准确。</div></div>
<div class="score-grid"><div class="score">无危险提升的最大宽度<br><b id="maxSafeWidth">—</b></div><div class="score">当前实验 operating point<br><b id="selectedWidth">—</b></div></div>
<div class="callout red"><b>解释原则：</b>最近邻残差会混入动态车辆和遮挡，因此它是配准一致性指标，不是绝对定位GT；边界环占比也不是 pipeline 内部的 boundary_ratio。只有两类指标同时异常，才能支持“定位/地图对齐正在污染车位证据”的判断。</div>
<div class="provenance"><b>本节数据：</b> outputs/frame_map_dataset_pose_corrected_final/map_points/009263.npz … 009277.npz；每个文件同时保存 corrected 与 original ego pose。<br><b>处理：</b>0.18m voxel downsample，20m半径，z∈[0.15,2.5]m，相邻帧 robust 2D ICP诊断。</div></section>

<section class="stage" id="s3"><div class="stage-head"><span class="badge">3</span><div><h3>从1397个已知几何车位缩到21个“有局部证据”的车位</h3><p>灰色/省略不是 Occupied，而是当前局部 LiDAR 没有为其建立 Part1 状态。</p></div></div>
<div class="io"><div><small>输入</small><b>1397个车位几何 + 融合点云</b></div><div><small>算法</small><b>实际 LiDAR 包络 + 18m局部半径 + 路线交会</b></div><div><small>中间量</small><b>coverage polygon、near/crossing/hit frames</b></div><div><small>输出/效果</small><b>21个 in_route_scope；1376个省略</b></div></div>
<div class="figure"><img src="full_pipeline_assets/03_scope_and_coverage.png"><div class="caption">中图蓝色区域是实际点云覆盖包络，不是理想160°扇形；局部21个车位才进入 Part1 判定。</div></div>
<div class="callout">这一步解决“11个灰色是不是 Occupied”的误解：灰色只代表没有当前 Part1 状态；遮挡、超范围和未进入局部包络必须与 Occupied 分开。</div></section>

<section class="stage" id="s4"><div class="stage-head"><span class="badge">4</span><div><h3>针对单个车位裁剪 core / boundary / adjacent 证据</h3><p>以 slot_1253 为主案例，直接展示交给判定器的真实点。</p></div></div>
<div class="io"><div><small>输入</small><b>15帧融合点 + slot_1253 几何</b></div><div><small>算法</small><b>投到车位局部坐标；划分 core、边缘和相邻车位</b></div><div><small>中间量</small><b>点高度、所属帧、ROI/核心多边形</b></div><div><small>输出/效果</small><b>Occupied 与 Free 两条分支共用证据</b></div></div>
<div class="figure"><img src="full_pipeline_assets/04_slot_evidence_extraction.png"><div class="caption">同一批数据的俯视、侧视和逐帧支持。这里开始所有指标都相对具体车位计算。</div></div></section>

<section class="stage" id="s5"><div class="stage-head"><span class="badge">5A</span><div><h3>Occupied 分支：候选车框搜索、3D形态和硬门限</h3><p>不是“车位里有一个点就 Occupied”，而是搜索最合理车体框并要求全部安全门通过。</p></div></div>
<div class="io"><div><small>输入</small><b>车位ROI点、所属帧、相邻车位</b></div><div><small>算法</small><b>枚举 box hypotheses → 特征 → gates</b></div><div><small>中间量</small><b>点数、帧支持、高度层、ownership、PCA、体素</b></div><div><small>输出/效果</small><b>strong / weak Occupied evidence</b></div></div>
<div class="formula">Occupied_strong = ALL(valid_frames, points, support, temporal, height, actual_short_extent≥0.75m, low_body_coverage≥0.05, ownership, voxels, layers, upper_height_spread≥0.05, pillar_veto, outside_residual≤0.35, stability=7/7)<br>candidate box 尺寸不能替代真实点云足迹；任一门失败 → Unknown<br>generic linearity = candidate-local（窄框截断不能一票否决后续车辆框）</div>
<div class="figure"><img src="full_pipeline_assets/05_occupied_branch.png"><div class="caption">红色 FAIL 会关闭 strong Occupied。柱体保护也是硬门之一；命中后只降级 Unknown，不会直接说 Free。</div></div></section>

<section class="stage" id="s6"><div class="stage-head"><span class="badge">5B</span><div><h3>Free-ray 分支：必须看见射线穿过车位，不能用“没打到点”证明空闲</h3><p>遮挡会产生 occluded/unobserved voxel；这些区域绝不能算作 Free。</p></div></div>
<div class="io"><div><small>输入</small><b>15个观察原点 + LiDAR hits + core 体素</b></div><div><small>算法</small><b>ray traversal 将体素分成 free/hit/occluded/unobserved</b></div><div><small>中间量</small><b>观察体积、近地覆盖、视角数、最大未观测连通域</b></div><div><small>输出/效果</small><b>只有正向可见性充分才 strong Free</b></div></div>
<div class="formula">Free_strong = ray_frames≥5 ∧ viewpoints≥2 ∧ separation≥10° ∧ volume≥0.70 ∧ ground≥0.70 ∧ unobserved_component≤0.20 ∧ occlusion≤0.20 ∧ no_hit/conflict</div>
<div class="figure"><img src="full_pipeline_assets/06_free_ray_branch.png"><div class="caption">slot_1253 的15个视点实际上只构成1个独立视角，体积覆盖不足且 core 中仍有 hit，因此 Free 分支失败。这正是“被挡住为什么是灰色/Unknown”的算法原因。</div></div></section>

<section class="stage" id="s7"><div class="stage-head"><span class="badge">6</span><div><h3>确定性三态融合：Free / Occupied / Unknown</h3><p>两个分支都要满足强证据才出终态；冲突、弱证据、姿态敏感或可见性不足统一保守为 Unknown。</p></div></div>
<div class="formula">if occupied.strong and stable → Occupied<br>else if free.strong and no occupied veto → Free<br>else → Unknown(reason_codes)</div>
<div class="figure"><img src="full_pipeline_assets/07_three_state_fusion.png"><div class="caption">frame 9277 的 v4 结果为 2 Free / 0 Occupied / 19 Unknown。散点“strength”只用于诊断；终态仍由全部硬门决定。</div></div>
<div class="explorer"><div class="slot-list" id="slotList"></div><div class="detail" id="slotDetail"></div></div></section>

<section class="stage" id="s8"><div class="stage-head"><span class="badge">7</span><div><h3>高精度安全门：柱体、薄盖、假大框与定位污染</h3><p>v4 不再相信候选框名义尺寸，必须检查框内真实回波足迹；柱体、薄水平盖、短边不足、低层缺失、框外污染或任一位姿扰动失败都降为 Unknown。</p></div></div>
<div class="formula">pillar_risk = PCA≥0.90 ∧ footprint_area≤0.30m² ∧ height≥0.80m ∧ height/maxXY≥1.50 ∧ robust_frames≥2 ∧ inlier≥0.70<br>vehicle_confirmation = robust_short_extent≥0.75m ∧ low_layer_coverage≥0.05 ∧ outside_residual≤0.35 ∧ perturbation_pass=7/7</div>
<div class="figure"><img src="full_pipeline_assets/08_pillar_before_after.png"><div class="caption">原有柱体保护仍保留；v4 再加入真实足迹、低层覆盖、框外纯度和全扰动稳定门。slot_1252 的真实短边仅约0.54m、框外残差约0.60，因此明确拒绝 Occupied。</div></div></section>

<section class="stage" id="s9"><div class="stage-head"><span class="badge">8</span><div><h3>Part1 输出：21个车位的局部状态地图</h3><p>这张图才是 Part1 的正式输出；A/B临时展示标记不参与 Part2。</p></div></div>
<div class="figure"><img src="full_pipeline_assets/09_part1_output_map.png"><div class="caption">左侧是车辆坐标下的21个状态；右侧列出每个车位的决定原因。车位编号可与上方证据浏览器逐一对应。</div></div></section>

<section class="stage" id="s10"><div class="stage-head"><span class="badge">9</span><div><h3>Part2 candidate 筛选：只把前方180°的 Unknown 送入队列</h3><p>这一步不修改 Part1 状态，只决定 Agent 是否值得为该 Unknown 分配任务。</p></div></div>
<div class="formula">candidate = (state = Unknown) ∧ agent_observable ∧ distance≤18m ∧ |atan2(left,forward)|≤90° ∧ evidence_pack_ready</div>
<div class="figure"><img src="full_pipeline_assets/10_candidate_filter.png"><div class="caption">19个 Unknown 经过 agent-observable、18m、前向 |β|≤90° 与 evidence-ready 门后，完整队列为5个：slot_1250、slot_1253、slot_1254、slot_1255、slot_1256。不是原先的160°理想扇形方法；摘要里的“最多2个”只属于临时 shortlist 显示上限，不是完整候选数量。</div></div></section>

<section class="stage" id="s11"><div class="stage-head"><span class="badge">10</span><div><h3>Part2 交接：每个候选得到一个独立 LiDAR evidence pack</h3><p>Agent 不应只收到“Unknown”三个字，而要收到 Part1 原因、两类证据、15帧点云、几何和相邻关系。</p></div></div>
<div class="figure"><img src="full_pipeline_assets/11_part2_evidence_packs.png"><div class="caption">五个图直接来自 v4 正式 queue 对应的 NPZ。每包包含 selected_frames、points_local_xyzi、point_frame_ids、15个观察原点、slot/core/margin polygon 和 adjacent geometry。</div></div>
<div class="pack-grid" id="packGrid"></div>
<div class="callout red"><b>当前真实终点：</b>5个 pack 已生成并通过交接合同。Part1 原因与原始 LiDAR 证据已一起交给 Part2；Camera target correspondence 仍不可用，所以当前不能声称 Agent 已把这5个变成 Free 或 Occupied。</div></section>

<section class="stage" id="s12"><div class="stage-head"><span class="badge">11</span><div><h3>当前改动效果、消融结果与不能越界的结论</h3><p>把“确实发生的变化”和“尚未验证的准确率”分开。</p></div></div>
<div class="figure"><img src="full_pipeline_assets/12_effects_and_ablation.png"><div class="caption">左：v4 将可疑 Occupied 全部安全降级；中：更保守的 Part1 会把5个前方 Unknown 交给 Part2；右：开发集 W/K 对 dense causal proxy 的一致率，不是GT准确率。</div></div>
<div class="callout green"><b>已经验证：</b>frame 9277 v4完整重跑、1252 Free补充GT对照、前方180°完整5车位队列、evidence pack 完整性，以及历史60帧8个Occupied的新增硬门反事实审计。</div>
<div class="callout red"><b>仍不能越界：</b>v4 在当前人工GT交集没有输出 Occupied，因此 Occupied precision 是 N/A，不是100%。选择性准确率100%来自2/2 Free终态，终态覆盖率仅14.3%；这是安全性结果，不是高召回结果。</div>
<div class="provenance"><b>数据来源：</b> frames.csv、15个 map_points NPZ、1397-slot database、frame_009277_v4_high_precision 的 local map / decisions / queue / evidence packs、24车位冻结GT、1252补充GT、systematic study及历史门控审计。<br><b>生成脚本：</b> scripts/build_full_pipeline_visualization.py。</div></section>

<section class="stage" id="s13"><div class="stage-head"><span class="badge">12</span><div><h3>人工真值、静态门消融与 Part2 Agent 对照</h3><p>同时展示准确性、安全性、覆盖率和协议不兼容，避免只挑一个好看的数字。</p></div></div>
<div class="figure"><img src="full_pipeline_assets/13_validation_and_part2.png"><div class="caption">左：冻结v1盲评与v3开发集结果；中：0.05/0.10薄盖阈值在六锚点的保守降级；右：当前v3严格Agent与历史OpenAI重叠审计。历史结果因哈希不一致没有合并。</div></div>
<div class="callout"><b>旧验证的用途：</b>这一节保留用于追踪 v1/v3 迭代历史，不再代表当前 operating point。v4 当前策略由下一节新人工GT审计约束。</div>
<div class="callout red"><b>论文级缺口：</b>现有受保护集只有8条标签（6条可判占用）。要声称“非常高准确率”，必须新增未参与任何调参的独立holdout，并同时报告终态precision、覆盖率、Unknown率与95%置信区间。</div></section>

<section class="stage" id="s14"><div class="stage-head"><span class="badge">13</span><div><h3>新人工GT审计：1252修复有效，但必须诚实报告覆盖率代价</h3><p>冻结24车位GT保持不变；用户新增确认 slot_1252=Free 作为独立补充标签，所有结果均按三态选择性分类统计。</p></div></div>
<div class="figure"><img src="full_pipeline_assets/14_high_precision_gt_audit.png"><div class="caption">左上：v3/v4在GT交集的三态分布；右上：选择性准确率、终态覆盖率、有效正确覆盖率必须并列；左下：1252失败的真实足迹和框外纯度门；右下：历史8个Occupied在v4新增门下8/8被明确拒绝，其中1017是真Occupied，直接展示召回率代价。</div></div>
<div class="callout green"><b>已证明的改进：</b>同一14车位GT交集上，错误Occupied从1降至0；slot_1252 从 Occupied 正确降为 Unknown。历史60帧7个GT假阳性和1个GT真阳性均不能绕过v4新增硬门。</div>
<div class="callout red"><b>严谨结论：</b>v4 是 fail-closed operating point。它符合“只有非常确认才Occupied”，但当前没有一个Occupied终态可用于估计precision，也没有识别出GT中的Occupied。下一步应依赖新的可观测帧/Camera对应提高召回，而不是放松Part1硬门。</div>
<div class="formula">reported metrics = selective_accuracy + terminal_coverage + effective_correct_coverage + Unknown_rate + false_Occupied_count<br>禁止：把 Unknown 算正确；把 0 个 Occupied 输出写成 100% Occupied precision</div></section>
</div></main></div>
<div class="modal" id="modal"><span class="close" id="close">×</span><img id="modalImg"></div>
<script>
const DATA=__DATA__;
document.getElementById("mergedMetric").textContent=DATA.alignment.merged_points.toLocaleString();
const LP=DATA.localization.pose_and_registration,LB=DATA.localization.boundary_and_sensitivity;
const LA=DATA.localization.uncertainty_ablation;
document.getElementById("poseShift").textContent=LP.max_position_shift_m.toFixed(3)+" m";
document.getElementById("yawShift").textContent=LP.max_abs_yaw_shift_deg.toFixed(3)+"°";
document.getElementById("poseResidual").textContent=LP.original_median_p50_m.toFixed(3)+" / "+LP.corrected_median_p50_m.toFixed(3)+" m";
document.getElementById("icpResidual").textContent=LP.icp_median_p50_m.toFixed(3)+" m";
document.getElementById("boundaryRange").textContent=(100*LB.slot1253_corrected_boundary_share_min).toFixed(1)+"% – "+(100*LB.slot1253_corrected_boundary_share_max).toFixed(1)+"%";
const T25=LB.translations.find(x=>Math.abs(x.magnitude_m-0.25)<1e-6);
document.getElementById("translationSensitivity").textContent=(100*T25.core_share_min).toFixed(1)+"% – "+(100*T25.core_share_max).toFixed(1)+"%";
document.getElementById("maxSafeWidth").textContent=LA.max_width_without_unsafe_promotion_m.toFixed(2)+" m";
document.getElementById("selectedWidth").textContent=LA.selected_experimental_width_m.toFixed(2)+" m (experimental)";
const change=100*(LP.original_median_p50_m-LP.corrected_median_p50_m)/LP.original_median_p50_m;
document.getElementById("registrationConclusion").innerHTML="<b>当前样例结论：</b> corrected pose 相对 original pose 的相邻帧p50残差变化 "+(change>=0?"下降 ":"上升 ")+Math.abs(change).toFixed(1)+"%；但局部ICP仍建议中位 "+LP.median_icp_translation_m.toFixed(3)+"m 平移和 "+LP.median_abs_icp_yaw_deg.toFixed(3)+"° 航向修正。说明 corrected pose 是否改善要看残差方向，同时仍存在可量化的局部不一致。";
document.getElementById("boundaryConclusion").innerHTML="<b>问题定位：</b> 相邻帧局部ICP只再建议约 "+(100*LP.median_icp_translation_m).toFixed(1)+"cm 平移，说明15帧相对配准不是主要崩溃源；但 slot_1253 的逐帧边界占比跨越 "+(100*LB.slot1253_corrected_boundary_share_min).toFixed(1)+"%–"+(100*LB.slot1253_corrected_boundary_share_max).toFixed(1)+"%，而0.25m偏移会让core ownership跨越 "+(100*T25.core_share_min).toFixed(1)+"%–"+(100*T25.core_share_max).toFixed(1)+"%。当前更强的风险是<b>绝对pose/车位地图对齐 + 硬边界归属</b>，不是简单的LiDAR测距噪声。";
const stateColor={free:"#14b86e",occupied:"#f05252",unknown:"#8c98a8"};
const list=document.getElementById("slotList"),detail=document.getElementById("slotDetail");
function showSlot(id){
  const row=DATA.decisions.find(x=>x.slot_id===id);
  document.querySelectorAll(".slot-button").forEach(x=>x.classList.toggle("active",x.dataset.id===id));
  const gateTable=(title,gates)=>`<h5>${title}</h5><table class="gate-table"><thead><tr><th>Gate</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead><tbody>${gates.map(g=>`<tr><td>${g.name}</td><td>${typeof g.value==="number"?g.value.toFixed(3):g.value}</td><td>${g.threshold}</td><td class="${g.passed?"pass":"fail"}">${g.passed?"PASS":"FAIL"}</td></tr>`).join("")}</tbody></table>`;
  detail.innerHTML=`<h4>${row.slot_id} · <span class="state-${row.state}">${row.state.toUpperCase()}</span></h4><p>${row.decision_reason}</p><div class="score-grid"><div class="score">Occupied strength<br><b>${row.occupied.strength.toFixed(3)}</b><br>${row.occupied.strong?"strong":"not strong"}</div><div class="score">Free strength<br><b>${row.free.strength.toFixed(3)}</b><br>${row.free.strong?"strong":"not strong"}</div></div><p><b>Unknown reasons</b><br>${row.unknown_reasons.length?row.unknown_reasons.map(x=>`<span class="chip">${x}</span>`).join(" "):"—"}</p>${gateTable("Occupied gates",row.occupied.gates)}${gateTable("Free gates",row.free.gates)}`;
}
DATA.decisions.forEach(row=>{const b=document.createElement("button");b.className="slot-button";b.dataset.id=row.slot_id;b.innerHTML=`<span><i class="dot" style="background:${stateColor[row.state]}"></i>${row.slot_id}</span><b class="state-${row.state}">${row.state}</b>`;b.onclick=()=>showSlot(row.slot_id);list.appendChild(b)});showSlot("slot_1253");
document.getElementById("packGrid").innerHTML=DATA.packs.map(p=>`<div class="pack"><h4>${p.slot_id}</h4><div><b>${p.point_count.toLocaleString()}</b> points · ${p.selected_frames.length} frames</div><div>bearing ${p.bearing_deg.toFixed(1)}°</div><div>Occ ${p.occupied_strength.toFixed(3)} / Free ${p.free_strength.toFixed(3)}</div><div class="chips">${p.unknown_reasons.map(x=>`<span class="chip">${x}</span>`).join("")}</div></div>`).join("");
const modal=document.getElementById("modal"),modalImg=document.getElementById("modalImg");document.querySelectorAll(".figure img").forEach(img=>img.onclick=()=>{modalImg.src=img.src;modal.classList.add("open")});document.getElementById("close").onclick=()=>modal.classList.remove("open");modal.onclick=e=>{if(e.target===modal)modal.classList.remove("open")};
const nav=[...document.querySelectorAll(".nav a")],sections=[...document.querySelectorAll(".stage")];new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){nav.forEach(a=>a.classList.toggle("active",a.getAttribute("href")==="#"+e.target.id))}}),{rootMargin:"-20% 0px -70% 0px"}).observe && sections.forEach(s=>new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting)nav.forEach(a=>a.classList.toggle("active",a.getAttribute("href")==="#"+e.target.id))}),{rootMargin:"-20% 0px -70% 0px"}).observe(s));
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = args.report_root.resolve()
    assets = report / "full_pipeline_assets"
    assets.mkdir(parents=True, exist_ok=True)
    artifact = report / "artifacts" / "frame_009277_v4_high_precision"
    baseline_artifact = report / "artifacts" / "frame_009277_baseline"
    frames_payload = load_json(artifact / "local_frame_manifest.json")
    frames = frames_payload["frames"]
    local_db = load_json(artifact / "local_slot_database.json")
    local_map = load_json(artifact / "local_map.json")
    queue = load_json(artifact / "unknown_agent_queue.json")
    improved_payload = load_json(artifact / "slot_decisions.json")
    baseline_payload = load_json(baseline_artifact / "slot_decisions.json")
    improved = decision_map(improved_payload)
    baseline = decision_map(baseline_payload)
    study = load_json(report / "results" / "systematic_study.json")
    uncertainty_ablation = load_json(
        report / "results" / "localization_uncertainty_ablation_six_anchors.json"
    )
    blinded_score = load_json(report / "results" / "frozen_human_eval_score_v2.json")
    development_score = load_json(report / "results" / "development_v3_score.json")
    static_cap_005 = load_json(report / "results" / "static_safety_ablation_ratio_0p05.json")
    static_cap_010 = load_json(report / "results" / "static_safety_ablation_six_anchors.json")
    part2_comparison = load_json(report / "results" / "part2_agent_comparison.json")
    manual_gt_comparison = load_json(
        report / "results" / "frame9277_part1_manual_gt_comparison_v4.json"
    )
    high_precision_gate_audit = load_json(
        report / "results" / "historical_extended60_high_precision_gate_audit_v4.json"
    )
    full_db = load_json(ROOT / "outputs" / "full_icpark_allframes_vehicle_cluster" / "slot_database.json")
    anchor = local_map["anchor_pose"]
    anchor_xy = np.asarray(anchor["map_xy"], dtype=float)
    yaw = float(anchor["map_yaw_rad"])
    scale = float(local_map["map_units_per_meter"])
    camera_rows = copy_camera_frames(frames, assets)
    clouds, point_counts = load_frame_clouds(frames)
    corrected_poses, original_poses, original_clouds = load_pose_audit(frames, clouds)
    figure_inputs(frames, point_counts, camera_rows, assets)
    alignment = figure_alignment(frames, clouds, anchor_xy, yaw, scale, assets)
    registration = registration_audit(clouds, original_clouds, anchor_xy, yaw, scale)
    localization_pose = figure_localization_audit(
        frames, corrected_poses, original_poses, registration, scale, assets
    )
    figure_scope(full_db, local_db, local_map, improved, anchor_xy, yaw, scale, assets)
    packs = slot_pack_map(queue, artifact)
    if "slot_1253" not in packs:
        raise RuntimeError(f"slot_1253 evidence pack not found; resolved={sorted(packs)}")
    boundary = boundary_audit(clouds, original_clouds, local_db)
    sensitivity = slot_pose_sensitivity(packs["slot_1253"])
    localization_boundary = figure_boundary_and_sensitivity(
        frames, boundary, sensitivity, assets
    )
    localization_ablation = figure_uncertainty_ablation(uncertainty_ablation, assets)
    extraction = figure_slot_extraction(packs["slot_1253"], np.asarray([row["frame_id"] for row in frames]), assets)
    figure_occupied(improved["slot_1253"], packs["slot_1253"], assets)
    figure_free(improved["slot_1253"], packs["slot_1253"], assets)
    figure_fusion(improved, assets)
    figure_pillar(baseline["slot_1253"], improved["slot_1253"], packs["slot_1253"], assets)
    figure_part1_map(local_db, improved, anchor_xy, yaw, scale, assets)
    candidates = candidate_rows(local_db, improved, queue, anchor_xy, yaw, scale)
    figure_candidates(candidates, assets)
    pack_summaries = figure_packs(queue, packs, assets)
    figure_effects(baseline, improved, study, len(queue["items"]), assets)
    final_validation = figure_final_validation(
        blinded_score,
        development_score,
        static_cap_005,
        static_cap_010,
        part2_comparison,
        assets,
    )
    figure_high_precision_gt(
        manual_gt_comparison,
        high_precision_gate_audit,
        improved,
        assets,
    )
    shutil.copy2(artifact / "local_map.png", assets / "original_part1_local_map.png")
    web_data = {
        "decisions": [slim_decision(improved[sid]) for sid in sorted(improved, key=lambda x: int(x.split("_")[-1]))],
        "candidates": candidates,
        "packs": pack_summaries,
        "camera_frames": camera_rows,
        "alignment": alignment,
        "extraction": extraction,
        "localization": {
            "pose_and_registration": localization_pose,
            "boundary_and_sensitivity": localization_boundary,
            "uncertainty_ablation": localization_ablation,
        },
        "final_validation": final_validation,
    }
    html = HTML.replace("__DATA__", json.dumps(web_data, ensure_ascii=False, separators=(",", ":")))
    index_path = report / "index.html"
    index_path.write_text(html, encoding="utf-8")
    (assets / "pipeline_data.json").write_text(json.dumps(web_data, ensure_ascii=False, indent=2), encoding="utf-8")
    image_refs = re.findall(r'<img src="([^"]+)"', html)
    missing_refs = [reference for reference in image_refs if not (report / reference).is_file()]
    if missing_refs:
        raise RuntimeError(f"HTML has missing image references: {missing_refs}")
    if len(improved) != 21 or len(queue["items"]) != 5:
        raise RuntimeError("frame 9277 visualization invariants changed unexpectedly")
    manifest_targets = sorted(
        path
        for path in report.rglob("*")
        if path.is_file() and path.name != "FULL_PIPELINE_MANIFEST.sha256"
    )
    manifest_lines = []
    for path in manifest_targets:
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(report)}")
    (report / "FULL_PIPELINE_MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "index": str(report / "index.html"),
                "assets": len(list(assets.iterdir())),
                "slots": len(improved),
                "queue_items": len(queue["items"]),
                "pack_paths": {key: str(value) for key, value in packs.items()},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
