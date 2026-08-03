#!/usr/bin/env python3
"""Planar NDT localization for the parking dataset.

The dataset stores LiDAR scans as KITTI-style float32 .bin files
(`x y z intensity`) and the parking map as a flat OBJ point/mesh file.
This script projects both to BEV, refines each frame with a 2D NDT
scan-to-map alignment, and writes poses plus PNG visualizations.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image

    HAS_PIL = True
except Exception:
    HAS_PIL = False


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


@dataclass
class NDTCell:
    mean: np.ndarray
    sqrt_info: np.ndarray
    count: int


@dataclass
class AlignStats:
    score: float
    inliers: int
    iterations: int
    step_norm: float


@dataclass
class ObjMapGeometry:
    vertices: np.ndarray
    faces: List[np.ndarray]
    line_segments: List[np.ndarray]
    face_edge_segments: List[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NDT localize parking LiDAR scans against an OBJ map")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--map-obj", type=Path, default=None)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ndt_localization"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--map-voxel", type=float, default=1.0)
    parser.add_argument("--scan-voxel", type=float, default=0.35)
    parser.add_argument("--min-cell-points", type=int, default=3)
    parser.add_argument("--max-map-points", type=int, default=250_000)
    parser.add_argument("--max-scan-points", type=int, default=12_000)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--z-min", type=float, default=-2.5)
    parser.add_argument("--z-max", type=float, default=2.5)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--seed-mode",
        choices=("relative", "pose", "pca"),
        default="relative",
        help="pose uses dataset pose directly; relative uses pose deltas inside the OBJ map frame",
    )
    parser.add_argument("--diagnose", action="store_true", help="Print coordinate ranges for the first selected frame")
    return parser.parse_args()


def resolve_map_path(dataset_root: Path, requested: Optional[Path]) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend([dataset_root / "file.obj", Path("file.obj")])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No OBJ map found. Pass --map-obj explicitly.")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotation_to_yaw(rot: np.ndarray) -> float:
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_matrix_to_xyyaw(mat: np.ndarray) -> np.ndarray:
    return np.array([mat[0, 3], mat[1, 3], rotation_to_yaw(mat[:2, :2])], dtype=np.float64)


def xyyaw_to_matrix(pose: np.ndarray) -> np.ndarray:
    rot = yaw_to_rot(float(pose[2]))
    mat = np.eye(4, dtype=np.float64)
    mat[:2, :2] = rot
    mat[0, 3] = float(pose[0])
    mat[1, 3] = float(pose[1])
    return mat


def transform_xy(points_xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rot = yaw_to_rot(float(pose[2]))
    return points_xy @ rot.T + pose[:2]


def compose_xyyaw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    out[:2] = a[:2] + yaw_to_rot(float(a[2])) @ b[:2]
    out[2] = normalize_angle(float(a[2] + b[2]))
    return out


def inverse_xyyaw(pose: np.ndarray) -> np.ndarray:
    rot_t = yaw_to_rot(float(pose[2])).T
    out = np.zeros(3, dtype=np.float64)
    out[:2] = -(rot_t @ pose[:2])
    out[2] = normalize_angle(float(-pose[2]))
    return out


def between_xyyaw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return compose_xyyaw(inverse_xyyaw(a), b)


def parse_obj_index(token: str, vertex_count: int) -> Optional[int]:
    head = token.split("/", 1)[0]
    if not head:
        return None
    try:
        raw = int(head)
    except ValueError:
        return None
    idx = raw - 1 if raw > 0 else vertex_count + raw
    if idx < 0:
        return None
    return idx


def load_obj_geometry(path: Path) -> ObjMapGeometry:
    vertices: List[Tuple[float, float]] = []
    face_indices: List[List[int]] = []
    line_indices: List[List[int]] = []

    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vals = np.fromstring(line[2:], sep=" ")
                if vals.size >= 2 and np.isfinite(vals[:2]).all():
                    vertices.append((float(vals[0]), float(vals[1])))
            elif line.startswith("f "):
                idxs = [parse_obj_index(token, len(vertices)) for token in line.split()[1:]]
                idxs = [idx for idx in idxs if idx is not None]
                if len(idxs) >= 3:
                    face_indices.append(idxs)
            elif line.startswith("l "):
                idxs = [parse_obj_index(token, len(vertices)) for token in line.split()[1:]]
                idxs = [idx for idx in idxs if idx is not None]
                if len(idxs) >= 2:
                    line_indices.append(idxs)

    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")

    verts = np.asarray(vertices, dtype=np.float64)
    faces: List[np.ndarray] = []
    face_edges: List[np.ndarray] = []
    for idxs in face_indices:
        if max(idxs) >= len(verts):
            continue
        poly = verts[idxs, :2]
        faces.append(poly)
        for i in range(len(poly)):
            face_edges.append(np.vstack([poly[i], poly[(i + 1) % len(poly)]]))

    line_segments: List[np.ndarray] = []
    for idxs in line_indices:
        if max(idxs) >= len(verts):
            continue
        pts = verts[idxs, :2]
        for i in range(len(pts) - 1):
            line_segments.append(np.vstack([pts[i], pts[i + 1]]))

    return ObjMapGeometry(vertices=verts[:, :2], faces=faces, line_segments=line_segments, face_edge_segments=face_edges)


def load_obj_xy(path: Path) -> np.ndarray:
    return load_obj_geometry(path).vertices


def load_lidar_xy(path: Path, z_min: float, z_max: float, range_max: float) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} is not a KITTI-style x y z intensity scan")
    pts = raw.reshape((-1, 4))[:, :3].astype(np.float64)
    finite = np.isfinite(pts).all(axis=1)
    z_ok = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    r_ok = np.linalg.norm(pts[:, :2], axis=1) <= range_max
    pts = pts[finite & z_ok & r_ok]
    return pts[:, :2]


def voxel_downsample_xy(points: np.ndarray, voxel: float, max_points: int = 0) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    out = points[unique_idx]
    if max_points > 0 and len(out) > max_points:
        stride = int(math.ceil(len(out) / max_points))
        out = out[::stride]
    return out


def load_pose_rows(path: Path) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    if not path.exists():
        return poses
    with path.open("r") as handle:
        for line in handle:
            vals = np.fromstring(line.strip(), sep=" ")
            if vals.size != 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = vals.reshape(3, 4)
            poses.append(mat)
    return poses


def load_match_pose_index(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not path.exists():
        return mapping
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) < 19:
                continue
            lidar_idx = fields[0]
            for pose_idx in fields[14:19]:
                if pose_idx >= 0:
                    mapping[lidar_idx] = pose_idx
                    break
    return mapping


def load_match_image_index(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not path.exists():
        return mapping
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) < 4:
                continue
            lidar_idx = fields[0]
            image_idx = fields[1]
            if image_idx >= 0:
                mapping[lidar_idx] = image_idx
    return mapping


def pose_seed_for_frame(frame_idx: int, pose_rows: Sequence[np.ndarray], pose_index: Dict[int, int]) -> Optional[np.ndarray]:
    idx = pose_index.get(frame_idx)
    if idx is None:
        idx = frame_idx if frame_idx < len(pose_rows) else None
    if idx is None or idx < 0 or idx >= len(pose_rows):
        return None
    return pose_matrix_to_xyyaw(pose_rows[idx])


def build_ndt_grid(points_xy: np.ndarray, voxel: float, min_points: int) -> Tuple[Dict[Tuple[int, int], NDTCell], np.ndarray]:
    origin = points_xy.min(axis=0) - 1e-6
    bins: Dict[Tuple[int, int], List[np.ndarray]] = {}
    keys = np.floor((points_xy - origin) / voxel).astype(np.int64)
    for key, point in zip(keys, points_xy):
        bins.setdefault((int(key[0]), int(key[1])), []).append(point)

    grid: Dict[Tuple[int, int], NDTCell] = {}
    for key, bucket in bins.items():
        arr = np.asarray(bucket, dtype=np.float64)
        if len(arr) < min_points:
            continue
        mean = arr.mean(axis=0)
        cov = np.cov(arr.T) if len(arr) > 1 else np.eye(2) * voxel * voxel
        if cov.shape != (2, 2):
            cov = np.eye(2) * voxel * voxel
        cov = cov + np.eye(2) * max(1e-4, 0.03 * voxel * voxel)
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.maximum(eigval, 1e-4)
        sqrt_info = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
        grid[key] = NDTCell(mean=mean, sqrt_info=sqrt_info, count=len(arr))
    if not grid:
        raise ValueError("NDT map grid is empty. Increase --map-voxel or lower --min-cell-points.")
    return grid, origin


def principal_yaw(points_xy: np.ndarray) -> float:
    centered = points_xy - points_xy.mean(axis=0)
    cov = np.cov(centered.T)
    eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(eigval))]
    return math.atan2(float(axis[1]), float(axis[0]))


def count_grid_hits(points_xy: np.ndarray, grid: Dict[Tuple[int, int], NDTCell], origin: np.ndarray, voxel: float) -> int:
    keys = np.floor((points_xy - origin) / voxel).astype(np.int64)
    hits = 0
    for key in keys:
        if (int(key[0]), int(key[1])) in grid:
            hits += 1
    return hits


def pca_seed_pose(scan_xy: np.ndarray, map_xy: np.ndarray, grid: Dict[Tuple[int, int], NDTCell], origin: np.ndarray, voxel: float) -> np.ndarray:
    scan_center = scan_xy.mean(axis=0)
    map_center = map_xy.mean(axis=0)
    scan_yaw = principal_yaw(scan_xy)
    map_yaw = principal_yaw(map_xy)
    sample = scan_xy[:: max(1, len(scan_xy) // 6000)]
    best_pose = None
    best_hits = -1
    for yaw_extra in (0.0, math.pi, math.pi / 2.0, -math.pi / 2.0):
        yaw = normalize_angle(map_yaw - scan_yaw + yaw_extra)
        translation = map_center - yaw_to_rot(yaw) @ scan_center
        pose = np.array([translation[0], translation[1], yaw], dtype=np.float64)
        hits = count_grid_hits(transform_xy(sample, pose), grid, origin, voxel)
        if hits > best_hits:
            best_pose = pose
            best_hits = hits
    assert best_pose is not None
    return best_pose


def bounds_text(points_xy: np.ndarray) -> str:
    return f"min={points_xy.min(axis=0)} max={points_xy.max(axis=0)} mean={points_xy.mean(axis=0)}"


def ndt_align_2d(
    source_xy: np.ndarray,
    grid: Dict[Tuple[int, int], NDTCell],
    origin: np.ndarray,
    voxel: float,
    initial_pose: np.ndarray,
    iterations: int,
) -> Tuple[np.ndarray, AlignStats]:
    pose = initial_pose.astype(np.float64).copy()
    best_pose = pose.copy()
    best_stats = AlignStats(score=float("inf"), inliers=0, iterations=0, step_norm=0.0)

    for itr in range(iterations):
        world = transform_xy(source_xy, pose)
        hessian = np.zeros((3, 3), dtype=np.float64)
        gradient = np.zeros(3, dtype=np.float64)
        score = 0.0
        inliers = 0

        for point in world:
            key_arr = np.floor((point - origin) / voxel).astype(np.int64)
            cell = grid.get((int(key_arr[0]), int(key_arr[1])))
            if cell is None:
                continue

            diff = point - cell.mean
            residual = cell.sqrt_info @ diff
            mahal = float(residual @ residual)
            if mahal > 25.0:
                continue

            jac = np.array([[1.0, 0.0, -point[1]], [0.0, 1.0, point[0]]], dtype=np.float64)
            weighted_jac = cell.sqrt_info @ jac
            hessian += weighted_jac.T @ weighted_jac
            gradient += weighted_jac.T @ residual
            score += mahal
            inliers += 1

        if inliers > 0 and score / inliers < best_stats.score:
            best_pose = pose.copy()
            best_stats = AlignStats(score=score / inliers, inliers=inliers, iterations=itr + 1, step_norm=best_stats.step_norm)

        if inliers < 20:
            return best_pose, AlignStats(score=best_stats.score, inliers=inliers, iterations=itr + 1, step_norm=0.0)

        hessian += np.eye(3) * (1e-3 + 1e-8 * np.trace(hessian))
        try:
            delta = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        max_translation_step = 1.0
        max_yaw_step = 0.15
        trans_norm = float(np.linalg.norm(delta[:2]))
        if trans_norm > max_translation_step:
            delta[:2] *= max_translation_step / trans_norm
        delta[2] = float(np.clip(delta[2], -max_yaw_step, max_yaw_step))

        pose[:2] += delta[:2]
        pose[2] = normalize_angle(float(pose[2] + delta[2]))
        step_norm = float(np.linalg.norm(delta))
        best_stats.step_norm = step_norm
        if step_norm < 1e-4:
            break

    return best_pose, best_stats


def smooth_pose(poses: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    out[:2] = arr[:, :2].mean(axis=0)
    out[2] = math.atan2(np.sin(arr[:, 2]).mean(), np.cos(arr[:, 2]).mean())
    return out


def car_footprint_xy(pose: np.ndarray, length: float = 4.6, width: float = 1.9) -> np.ndarray:
    half_l = length * 0.5
    half_w = width * 0.5
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    return corners @ yaw_to_rot(float(pose[2])).T + pose[:2]


def draw_map_layers(ax: plt.Axes, geometry: ObjMapGeometry, show_legend: bool = False) -> None:
    if geometry.faces:
        ax.add_collection(
            PolyCollection(
                geometry.faces,
                facecolors="#eef0e8",
                edgecolors="none",
                alpha=0.92,
                zorder=1,
                label="road / paved surface" if show_legend else None,
            )
        )
    if geometry.face_edge_segments:
        ax.add_collection(
            LineCollection(
                geometry.face_edge_segments,
                colors="#b6afa3",
                linewidths=0.18,
                alpha=0.55,
                zorder=2,
                label="surface mesh edges" if show_legend else None,
            )
        )
    if geometry.line_segments:
        ax.add_collection(
            LineCollection(
                geometry.line_segments,
                colors="#111827",
                linewidths=0.58,
                alpha=0.96,
                zorder=3,
                label="parking / road markings" if show_legend else None,
            )
        )

    bounds_min = geometry.vertices.min(axis=0)
    bounds_max = geometry.vertices.max(axis=0)
    span = bounds_max - bounds_min
    pad = max(0.35, float(max(span)) * 0.04)
    ax.set_xlim(bounds_min[0] - pad, bounds_max[0] + pad)
    ax.set_ylim(bounds_min[1] - pad, bounds_max[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(math.floor(bounds_min[0]), math.ceil(bounds_max[0]) + 1, 1.0))
    ax.set_yticks(np.arange(math.floor(bounds_min[1]), math.ceil(bounds_max[1]) + 1, 1.0))
    ax.grid(True, color="#cbd5e1", linewidth=0.45, alpha=0.55, zorder=0)


def draw_car_pose(ax: plt.Axes, pose: np.ndarray, frame_idx: int, label: str = "CURRENT CAR") -> None:
    body = car_footprint_xy(pose)
    body_closed = np.vstack([body, body[0]])
    ax.fill(body[:, 0], body[:, 1], color="#ef4444", alpha=0.34, zorder=8)
    ax.plot(body_closed[:, 0], body_closed[:, 1], color="#991b1b", linewidth=2.1, zorder=9)

    axis_len = 2.8
    heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
    left = np.array([-math.sin(pose[2]), math.cos(pose[2])])
    ax.arrow(pose[0], pose[1], heading[0] * axis_len, heading[1] * axis_len, color="#dc2626", width=0.08, zorder=10)
    ax.arrow(pose[0], pose[1], left[0] * axis_len * 0.6, left[1] * axis_len * 0.6, color="#16a34a", width=0.06, zorder=10)
    ax.scatter([pose[0]], [pose[1]], s=34, c="#111827", zorder=11)
    ax.annotate(
        f"{label}\nframe {frame_idx:06d}\nx={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f} rad",
        xy=(pose[0], pose[1]),
        xytext=(14, 16),
        textcoords="offset points",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#991b1b", "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "color": "#991b1b", "lw": 1.2},
        zorder=12,
    )


def render_frame(
    output_path: Path,
    frame_idx: int,
    map_geometry: ObjMapGeometry,
    scan_world: np.ndarray,
    trajectory: Sequence[np.ndarray],
    pose: np.ndarray,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 12), dpi=180)
    draw_map_layers(ax, map_geometry, show_legend=False)
    if len(scan_world):
        scan_sample = scan_world[:: max(1, len(scan_world) // 10_000)]
        ax.scatter(scan_sample[:, 0], scan_sample[:, 1], s=0.45, c="#2563eb", alpha=0.33, linewidths=0, zorder=4)
    if trajectory:
        traj = np.asarray(trajectory, dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 1], color="#d97706", linewidth=2.0, zorder=7)
    draw_car_pose(ax, pose, frame_idx)
    ax.set_title(f"Clear OBJ map with localized vehicle - frame {frame_idx:06d}")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_current_location_map(
    output_path: Path,
    frame_idx: int,
    map_geometry: ObjMapGeometry,
    trajectory: Sequence[np.ndarray],
    pose: np.ndarray,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(11, 13), dpi=220)
    draw_map_layers(ax, map_geometry, show_legend=True)
    if trajectory:
        traj = np.asarray(trajectory, dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 1], color="#d97706", linewidth=2.4, zorder=7, label="estimated trajectory")
        ax.scatter([traj[0, 0]], [traj[0, 1]], s=38, c="#16a34a", zorder=9, label="start")
    draw_car_pose(ax, pose, frame_idx)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8)
    ax.set_title("Parking Map: Current Vehicle Pose")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_trajectory_map(
    output_path: Path,
    map_geometry: ObjMapGeometry,
    frame_indices: Sequence[int],
    trajectory: Sequence[np.ndarray],
) -> None:
    if not HAS_MATPLOTLIB or not trajectory:
        return
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(11, 13), dpi=220)
    draw_map_layers(ax, map_geometry, show_legend=True)
    traj = np.asarray(trajectory, dtype=np.float64)
    ax.plot(traj[:, 0], traj[:, 1], color="#d97706", linewidth=2.4, zorder=7, label="estimated trajectory")
    ax.scatter([traj[0, 0]], [traj[0, 1]], s=44, c="#16a34a", zorder=9, label="start")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=52, c="#dc2626", zorder=10, label="end")
    draw_car_pose(ax, traj[0], int(frame_indices[0]), label="START")
    draw_car_pose(ax, traj[-1], int(frame_indices[-1]), label="END")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8)
    ax.set_title("Parking Map: Full Vehicle Trajectory")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def export_frame_photo(source_path: Path, output_path: Path) -> bool:
    if not HAS_PIL or not source_path.exists():
        return False
    ensure_dir(output_path.parent)
    image = Image.open(source_path)
    image.save(output_path)
    return True


def render_start_end_photos(
    output_path: Path,
    start_path: Optional[Path],
    end_path: Optional[Path],
    start_frame_idx: int,
    end_frame_idx: int,
) -> None:
    if not HAS_MATPLOTLIB or not HAS_PIL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    panels = [
        (axes[0], start_path, f"Start frame {start_frame_idx:06d}"),
        (axes[1], end_path, f"End frame {end_frame_idx:06d}"),
    ]
    for ax, image_path, title in panels:
        ax.set_title(title)
        ax.axis("off")
        if image_path is not None and image_path.exists():
            ax.imshow(Image.open(image_path))
        else:
            ax.text(0.5, 0.5, "image not found", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    fields = [
        "frame_idx",
        "x",
        "y",
        "yaw",
        "smoothed_x",
        "smoothed_y",
        "smoothed_yaw",
        "score",
        "inliers",
        "iterations",
        "step_norm",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    map_path = resolve_map_path(args.dataset_root, args.map_obj)
    scan_dir = args.dataset_root / "velodyne"
    pose_path = args.dataset_root / "pose" / "poses.txt"

    if not scan_dir.exists():
        raise FileNotFoundError(f"Missing scan directory: {scan_dir}")

    ensure_dir(args.output_dir)
    print(f"[load] dataset: {args.dataset_root}")
    print(f"[load] map: {map_path}")

    map_geometry = load_obj_geometry(map_path)
    map_xy_full = map_geometry.vertices
    map_xy = voxel_downsample_xy(map_xy_full, max(args.map_voxel * 0.35, 0.05), args.max_map_points)
    grid, origin = build_ndt_grid(map_xy, args.map_voxel, args.min_cell_points)
    print(
        f"[map] vertices={len(map_xy_full):,} faces={len(map_geometry.faces):,} "
        f"lines={len(map_geometry.line_segments):,} ndt_cells={len(grid):,} voxel={args.map_voxel:.2f}"
    )

    pose_rows = load_pose_rows(pose_path)
    pose_index = load_match_pose_index(args.match_file)
    image_index = load_match_image_index(args.match_file)
    print(
        f"[pose] rows={len(pose_rows):,} matched_lidar={len(pose_index):,} "
        f"matched_images={len(image_index):,}"
    )

    frame_indices = list(range(args.start_index, args.start_index + args.num_frames * args.step, args.step))
    history: deque[np.ndarray] = deque(maxlen=max(1, args.smooth_window))
    trajectory: List[np.ndarray] = []
    rows: List[dict] = []
    last_pose: Optional[np.ndarray] = None
    first_global_seed: Optional[np.ndarray] = None
    first_map_seed: Optional[np.ndarray] = None

    for ordinal, frame_idx in enumerate(frame_indices):
        scan_path = scan_dir / f"{frame_idx:06d}.bin"
        if not scan_path.exists():
            print(f"[skip] missing scan {scan_path}")
            continue

        scan_xy = load_lidar_xy(scan_path, args.z_min, args.z_max, args.range_max)
        scan_xy = voxel_downsample_xy(scan_xy, args.scan_voxel, args.max_scan_points)
        global_seed = pose_seed_for_frame(frame_idx, pose_rows, pose_index)

        if args.seed_mode == "pose":
            seed = global_seed if global_seed is not None else last_pose
        elif args.seed_mode == "pca":
            seed = pca_seed_pose(scan_xy, map_xy, grid, origin, args.map_voxel) if last_pose is None else last_pose.copy()
        else:
            if first_map_seed is None:
                first_map_seed = pca_seed_pose(scan_xy, map_xy, grid, origin, args.map_voxel)
                first_global_seed = global_seed.copy() if global_seed is not None else None
            if global_seed is not None and first_global_seed is not None:
                seed = compose_xyyaw(first_map_seed, between_xyyaw(first_global_seed, global_seed))
            else:
                seed = first_map_seed.copy() if last_pose is None else last_pose.copy()

        if seed is None:
            raise RuntimeError(f"No pose seed available for frame {frame_idx:06d}")
        if args.diagnose and ordinal == 0:
            print(f"[diagnose] map {bounds_text(map_xy)}")
            print(f"[diagnose] scan_local {bounds_text(scan_xy)}")
            print(f"[diagnose] seed={seed}")
            print(f"[diagnose] scan_seeded {bounds_text(transform_xy(scan_xy, seed))}")

        estimated, stats = ndt_align_2d(scan_xy, grid, origin, args.map_voxel, seed, args.iterations)
        if stats.inliers < 20 and last_pose is not None:
            estimated = last_pose.copy()

        history.append(estimated)
        smoothed = smooth_pose(list(history))
        trajectory.append(smoothed.copy())
        last_pose = smoothed.copy()

        scan_world = transform_xy(scan_xy, smoothed)
        if not args.no_render and ordinal % max(1, args.render_every) == 0:
            render_frame(args.output_dir / "frames" / f"frame_{frame_idx:06d}.png", frame_idx, map_geometry, scan_world, trajectory, smoothed)

        rows.append(
            {
                "frame_idx": frame_idx,
                "x": estimated[0],
                "y": estimated[1],
                "yaw": estimated[2],
                "smoothed_x": smoothed[0],
                "smoothed_y": smoothed[1],
                "smoothed_yaw": smoothed[2],
                "score": stats.score,
                "inliers": stats.inliers,
                "iterations": stats.iterations,
                "step_norm": stats.step_norm,
            }
        )
        print(
            f"[frame {frame_idx:06d}] x={smoothed[0]:.2f} y={smoothed[1]:.2f} "
            f"yaw={smoothed[2]:.3f} score={stats.score:.3f} inliers={stats.inliers}"
        )

    write_csv(args.output_dir / "estimated_poses.csv", rows)
    if trajectory:
        start_frame_idx = int(rows[0]["frame_idx"])
        final_frame_idx = int(rows[-1]["frame_idx"])
    else:
        start_frame_idx = frame_indices[0]
        final_frame_idx = frame_indices[-1]

    if HAS_MATPLOTLIB and trajectory and not args.no_render:
        render_current_location_map(
            args.output_dir / "current_location_map.png",
            final_frame_idx,
            map_geometry,
            trajectory,
            trajectory[-1],
        )
        render_trajectory_map(
            args.output_dir / "trajectory.png",
            map_geometry,
            [int(row["frame_idx"]) for row in rows],
            trajectory,
        )

    image_dir = args.dataset_root / "image"
    start_image_idx = image_index.get(start_frame_idx)
    end_image_idx = image_index.get(final_frame_idx)
    start_image_src = image_dir / f"left{start_image_idx:06d}.png" if start_image_idx is not None else None
    end_image_src = image_dir / f"left{end_image_idx:06d}.png" if end_image_idx is not None else None
    start_image_out = args.output_dir / "start_photo.png"
    end_image_out = args.output_dir / "end_photo.png"
    if start_image_src is not None:
        export_frame_photo(start_image_src, start_image_out)
    if end_image_src is not None:
        export_frame_photo(end_image_src, end_image_out)
    if not args.no_render:
        render_start_end_photos(
            args.output_dir / "start_end_photos.png",
            start_image_out if start_image_out.exists() else start_image_src,
            end_image_out if end_image_out.exists() else end_image_src,
            start_frame_idx,
            final_frame_idx,
        )

    print(f"[done] wrote {len(rows)} poses to {args.output_dir / 'estimated_poses.csv'}")
    if start_image_src is not None:
        print(f"[photo] start frame {start_frame_idx:06d} -> {start_image_src.name}")
    if end_image_src is not None:
        print(f"[photo] end frame {final_frame_idx:06d} -> {end_image_src.name}")
    if not HAS_MATPLOTLIB and not args.no_render:
        print("[warn] matplotlib is unavailable, so PNG visualizations were not written")


if __name__ == "__main__":
    main()
