#!/usr/bin/env python3
"""LiDAR-to-GLTF 2D NDT localization prototype.

This script uses the semantic GLTF parking map as the reference map and
localizes LiDAR scans by scan-to-map matching. Dataset poses are optional and
used only for evaluation or explicit seeding.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    HAS_MPL = True
except Exception:
    HAS_MPL = False


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


SEMANTIC_COLORS = {
    "parkingSpace": "#d6ead8",
    "parkingSpaceBorder": "#111827",
    "laneLines": "#f5c542",
    "wall": "#5b6472",
    "arrester": "#d97706",
    "elevator": "#38bdf8",
    "ground": "#eef0e8",
}

REFERENCE_SEMANTICS = {
    "parkingSpaceBorder": 1.0,
    "laneLines": 0.75,
    "wall": 2.0,
    "arrester": 1.8,
}


@dataclass
class SemanticLayer:
    name: str
    points: np.ndarray
    lines: list[np.ndarray]


@dataclass
class GltfMap:
    layers: dict[str, SemanticLayer]
    label_points: np.ndarray
    reference_points: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass
class NDTCell:
    mean: np.ndarray
    sqrt_info: np.ndarray
    count: int


@dataclass
class NDTGrid:
    cells: dict[tuple[int, int], NDTCell]
    origin: np.ndarray
    voxel: float


@dataclass
class AlignResult:
    pose: np.ndarray
    score: float
    inliers: int
    iterations: int
    confidence: float
    init_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize LiDAR scans on a semantic GLTF parking map with 2D NDT")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gltf_lidar_ndt"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    parser.add_argument("--map-voxel", type=float, default=0.25)
    parser.add_argument("--scan-voxel", type=float, default=0.18)
    parser.add_argument("--range-max", type=float, default=25.0)
    parser.add_argument("--z-min", type=float, default=-0.45)
    parser.add_argument("--z-max", type=float, default=1.35)
    parser.add_argument("--max-scan-points", type=int, default=3500)
    parser.add_argument("--coarse-xy-step", type=float, default=0.50)
    parser.add_argument("--coarse-yaw-step-deg", type=float, default=10.0)
    parser.add_argument("--coarse-margin", type=float, default=1.5)
    parser.add_argument("--relocalize-below", type=float, default=0.30, help="Run coarse search again when previous confidence is below this value")
    parser.add_argument("--coarse-every", type=int, default=0, help="Force coarse search every N processed frames; 0 disables")
    parser.add_argument("--ndt-iterations", type=int, default=30)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Do not print every processed frame")
    parser.add_argument(
        "--init-mode",
        choices=("auto", "pose"),
        default="auto",
        help="auto uses only LiDAR-to-map coarse search; pose uses dataset pose as an explicit seed for debugging",
    )
    parser.add_argument("--eval-pose", action="store_true", help="Report pose-truth deltas after fitting truth to GLTF by similarity")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_xy(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ yaw_to_rot(float(pose[2])).T + pose[:2]


def voxel_downsample(points: np.ndarray, voxel: float, max_points: int = 0) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    out = points[idx]
    if max_points > 0 and len(out) > max_points:
        stride = int(math.ceil(len(out) / max_points))
        out = out[::stride]
    return out


def sample_segment(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    dist = float(np.linalg.norm(b - a))
    count = max(2, int(math.ceil(dist / step)) + 1)
    t = np.linspace(0.0, 1.0, count)
    return a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]


def decode_data_uri(uri: str) -> bytes:
    prefix = "base64,"
    pos = uri.find(prefix)
    if pos < 0:
        raise ValueError("Only embedded base64 GLTF buffers are supported")
    return base64.b64decode(uri[pos + len(prefix) :])


def accessor_array(data: dict, buffer_bytes: bytes, accessor_idx: int) -> np.ndarray:
    accessor = data["accessors"][accessor_idx]
    view = data["bufferViews"][accessor["bufferView"]]
    component_type = accessor["componentType"]
    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    type_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    dtype = dtype_map[component_type]
    item_count = type_count[accessor["type"]]
    count = int(accessor["count"])
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_stride = view.get("byteStride")
    if byte_stride is None:
        arr = np.frombuffer(buffer_bytes, dtype=dtype, count=count * item_count, offset=byte_offset)
        return arr.reshape(count, item_count).copy()

    out = np.empty((count, item_count), dtype=dtype)
    item_bytes = np.dtype(dtype).itemsize * item_count
    for i in range(count):
        start = byte_offset + i * int(byte_stride)
        out[i] = np.frombuffer(buffer_bytes[start : start + item_bytes], dtype=dtype, count=item_count)
    return out


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        mat = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    else:
        mat = np.eye(4, dtype=np.float64)
        if "translation" in node:
            mat[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    return mat


def transform_positions(positions: np.ndarray, mat: np.ndarray) -> np.ndarray:
    hom = np.column_stack([positions[:, :3], np.ones(len(positions), dtype=np.float64)])
    return (hom @ mat.T)[:, :3]


def primitive_lines(positions_xy: np.ndarray, indices: Optional[np.ndarray], mode: int, step: float) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    if mode == 1:
        order = indices.reshape(-1) if indices is not None else np.arange(len(positions_xy), dtype=np.int64)
        for i in range(0, len(order) - 1, 2):
            a = positions_xy[int(order[i])]
            b = positions_xy[int(order[i + 1])]
            if np.linalg.norm(b - a) > 1e-6:
                lines.append(np.vstack([a, b]))
    elif mode == 4:
        order = indices.reshape(-1) if indices is not None else np.arange(len(positions_xy), dtype=np.int64)
        for i in range(0, len(order) - 2, 3):
            tri = positions_xy[order[i : i + 3].astype(np.int64)]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if np.linalg.norm(b - a) > step * 0.25:
                    lines.append(np.vstack([a, b]))
    return lines


def load_gltf_map(path: Path, sample_step: float) -> GltfMap:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    buffer_bytes = decode_data_uri(data["buffers"][0]["uri"])

    mesh_nodes: list[tuple[str, dict]] = []
    label_points: list[np.ndarray] = []
    for node in data.get("nodes", []):
        semantic = (node.get("extras") or {}).get("type")
        mat = node_matrix(node)
        if semantic == "parkingSpaceNumberByCSS2DRender":
            label_points.append(mat[:2, 3].copy())
        if semantic and "mesh" in node:
            mesh_nodes.append((semantic, node))

    layers: dict[str, SemanticLayer] = {}
    for semantic, node in mesh_nodes:
        mesh = data["meshes"][int(node["mesh"])]
        mat = node_matrix(node)
        layer_points: list[np.ndarray] = []
        layer_lines: list[np.ndarray] = []
        for primitive in mesh.get("primitives", []):
            pos_accessor = primitive["attributes"].get("POSITION")
            if pos_accessor is None:
                continue
            positions = accessor_array(data, buffer_bytes, int(pos_accessor)).astype(np.float64)
            positions = transform_positions(positions, mat)
            positions_xy = positions[:, :2]
            indices = None
            if "indices" in primitive:
                indices = accessor_array(data, buffer_bytes, int(primitive["indices"])).astype(np.int64).reshape(-1)
            mode = int(primitive.get("mode", 4))
            lines = primitive_lines(positions_xy, indices, mode, sample_step)
            layer_lines.extend(lines)
            if lines:
                sampled = [sample_segment(seg[0], seg[1], sample_step) for seg in lines]
                layer_points.append(np.vstack(sampled))
            else:
                layer_points.append(positions_xy)

        if layer_points:
            points = np.vstack(layer_points)
        else:
            points = np.empty((0, 2), dtype=np.float64)
        if semantic in layers:
            old = layers[semantic]
            points = np.vstack([old.points, points])
            layer_lines = old.lines + layer_lines
        layers[semantic] = SemanticLayer(semantic, points, layer_lines)

    all_points = [layer.points for layer in layers.values() if len(layer.points)]
    if label_points:
        all_points.append(np.asarray(label_points, dtype=np.float64))
    if not all_points:
        raise ValueError(f"No usable map points found in {path}")

    ref_chunks: list[np.ndarray] = []
    for semantic, weight in REFERENCE_SEMANTICS.items():
        layer = layers.get(semantic)
        if layer is None or len(layer.points) == 0:
            continue
        repeat = max(1, int(round(weight)))
        ref_chunks.extend([layer.points] * repeat)
    if not ref_chunks:
        ref_chunks = [np.vstack(all_points)]
    reference_points = voxel_downsample(np.vstack(ref_chunks), sample_step * 0.75)
    extent_points = np.vstack(all_points)
    return GltfMap(
        layers=layers,
        label_points=np.asarray(label_points, dtype=np.float64) if label_points else np.empty((0, 2)),
        reference_points=reference_points,
        bounds_min=extent_points.min(axis=0),
        bounds_max=extent_points.max(axis=0),
    )


def build_ndt_grid(points: np.ndarray, voxel: float, min_points: int = 3) -> NDTGrid:
    origin = points.min(axis=0) - 1e-6
    keys = np.floor((points - origin) / voxel).astype(np.int64)
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for key, point in zip(keys, points):
        buckets.setdefault((int(key[0]), int(key[1])), []).append(point)

    cells: dict[tuple[int, int], NDTCell] = {}
    for key, values in buckets.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < min_points:
            continue
        mean = arr.mean(axis=0)
        cov = np.cov(arr.T) if len(arr) > 1 else np.eye(2) * voxel * voxel
        if cov.shape != (2, 2):
            cov = np.eye(2) * voxel * voxel
        cov += np.eye(2) * max(1e-4, 0.025 * voxel * voxel)
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.maximum(eigval, 1e-4)
        sqrt_info = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
        cells[key] = NDTCell(mean=mean, sqrt_info=sqrt_info, count=len(arr))
    if not cells:
        raise ValueError("NDT grid is empty; lower --map-voxel or --map-sample-step")
    return NDTGrid(cells=cells, origin=origin, voxel=voxel)


def load_lidar_scan(path: Path, z_min: float, z_max: float, range_max: float, voxel: float, max_points: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Invalid KITTI-style LiDAR file: {path}")
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    mask = np.isfinite(pts).all(axis=1)
    mask &= (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    mask &= np.linalg.norm(pts[:, :2], axis=1) <= range_max
    xy = pts[mask, :2]
    return voxel_downsample(xy, voxel, max_points=max_points)


def coarse_search(
    scan_xy: np.ndarray,
    map_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    xy_step: float,
    yaw_step_deg: float,
    margin: float,
) -> tuple[np.ndarray, float, int]:
    tree = cKDTree(map_points)
    sample = scan_xy
    if len(sample) > 900:
        sample = sample[:: int(math.ceil(len(sample) / 900))]

    xs = np.arange(bounds_min[0], bounds_max[0] + 1e-9, xy_step)
    ys = np.arange(bounds_min[1], bounds_max[1] + 1e-9, xy_step)
    yaws = np.arange(-math.pi, math.pi, math.radians(yaw_step_deg))

    best_pose = np.array([float(xs[len(xs) // 2]), float(ys[len(ys) // 2]), 0.0], dtype=np.float64)
    best_cost = float("inf")
    best_inliers = 0
    low = bounds_min - margin
    high = bounds_max + margin

    for yaw in yaws:
        rot_scan = sample @ yaw_to_rot(float(yaw)).T
        for x in xs:
            shifted_x = rot_scan[:, 0] + x
            x_mask = (shifted_x >= low[0]) & (shifted_x <= high[0])
            if int(x_mask.sum()) < 20:
                continue
            for y in ys:
                shifted = np.column_stack([shifted_x[x_mask], rot_scan[x_mask, 1] + y])
                inside = (shifted[:, 1] >= low[1]) & (shifted[:, 1] <= high[1])
                candidate = shifted[inside]
                if len(candidate) < 20:
                    continue
                dist, _ = tree.query(candidate, k=1, workers=-1)
                trimmed = np.partition(dist, min(len(dist) - 1, max(5, int(len(dist) * 0.70))))[: max(5, int(len(dist) * 0.70))]
                inliers = int((dist < 0.35).sum())
                cost = float(trimmed.mean()) + 0.12 / math.sqrt(max(1, inliers))
                if cost < best_cost:
                    best_cost = cost
                    best_inliers = inliers
                    best_pose = np.array([float(x), float(y), float(yaw)], dtype=np.float64)
    return best_pose, best_cost, best_inliers


def ndt_align(scan_xy: np.ndarray, grid: NDTGrid, initial_pose: np.ndarray, iterations: int) -> tuple[np.ndarray, float, int, int]:
    pose = initial_pose.astype(np.float64).copy()
    best_pose = pose.copy()
    best_score = float("inf")
    best_inliers = 0
    best_iter = 0

    for itr in range(iterations):
        yaw = float(pose[2])
        rot = yaw_to_rot(yaw)
        world = scan_xy @ rot.T + pose[:2]
        keys = np.floor((world - grid.origin) / grid.voxel).astype(np.int64)
        hessian = np.zeros((3, 3), dtype=np.float64)
        gradient = np.zeros(3, dtype=np.float64)
        score = 0.0
        inliers = 0

        d_rot = np.array([[-math.sin(yaw), -math.cos(yaw)], [math.cos(yaw), -math.sin(yaw)]], dtype=np.float64)
        d_yaw = scan_xy @ d_rot.T

        for i, key in enumerate(keys):
            cell = grid.cells.get((int(key[0]), int(key[1])))
            if cell is None:
                continue
            diff = world[i] - cell.mean
            residual = cell.sqrt_info @ diff
            mahal = float(residual @ residual)
            if mahal > 16.0:
                continue
            jac = np.array([[1.0, 0.0, d_yaw[i, 0]], [0.0, 1.0, d_yaw[i, 1]]], dtype=np.float64)
            weighted_jac = cell.sqrt_info @ jac
            hessian += weighted_jac.T @ weighted_jac
            gradient += weighted_jac.T @ residual
            score += mahal
            inliers += 1

        if inliers > 0:
            mean_score = score / inliers
            if mean_score < best_score:
                best_score = mean_score
                best_inliers = inliers
                best_pose = pose.copy()
                best_iter = itr + 1

        if inliers < 15:
            break

        hessian += np.eye(3) * (1e-3 + 1e-8 * np.trace(hessian))
        try:
            delta = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        trans_norm = float(np.linalg.norm(delta[:2]))
        if trans_norm > 0.45:
            delta[:2] *= 0.45 / trans_norm
        delta[2] = float(np.clip(delta[2], -0.10, 0.10))
        pose[:2] += delta[:2]
        pose[2] = normalize_angle(float(pose[2] + delta[2]))
        if float(np.linalg.norm(delta)) < 1e-4:
            break

    return best_pose, best_score, best_inliers, best_iter


def confidence_from(score: float, inliers: int, scan_count: int) -> float:
    if not math.isfinite(score) or scan_count <= 0:
        return 0.0
    inlier_ratio = min(1.0, inliers / max(1, min(scan_count, 900)))
    score_term = math.exp(-0.5 * max(0.0, score))
    return float(np.clip(0.65 * score_term + 0.35 * inlier_ratio, 0.0, 1.0))


def load_pose_rows(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    if not path.exists():
        return poses
    for line in path.read_text().splitlines():
        vals = np.fromstring(line, sep=" ")
        if vals.size != 12:
            continue
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = vals.reshape(3, 4)
        poses.append(mat)
    return poses


def pose_to_xyyaw(mat: np.ndarray) -> np.ndarray:
    return np.array([mat[0, 3], mat[1, 3], math.atan2(float(mat[1, 0]), float(mat[0, 0]))], dtype=np.float64)


def load_match_pose_index(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not path.exists():
        return mapping
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = [int(x) for x in line.split()]
        if len(fields) >= 19:
            lidar_idx = fields[0]
            valid = [idx for idx in fields[14:19] if idx >= 0]
            if valid:
                mapping[lidar_idx] = valid[0]
    return mapping


def get_truth_pose(frame_idx: int, pose_rows: Sequence[np.ndarray], pose_index: dict[int, int]) -> Optional[np.ndarray]:
    idx = pose_index.get(frame_idx, frame_idx if frame_idx < len(pose_rows) else -1)
    if idx < 0 or idx >= len(pose_rows):
        return None
    return pose_to_xyyaw(pose_rows[idx])


def draw_gltf_map(ax: plt.Axes, gltf_map: GltfMap) -> None:
    for semantic, layer in gltf_map.layers.items():
        color = SEMANTIC_COLORS.get(semantic, "#64748b")
        if layer.lines:
            lw = 1.8 if semantic in {"parkingSpaceBorder", "laneLines"} else 1.0
            ax.add_collection(LineCollection(layer.lines, colors=color, linewidths=lw, alpha=0.9, zorder=4))
        elif len(layer.points):
            ax.scatter(layer.points[:, 0], layer.points[:, 1], s=0.2, c=color, alpha=0.35, linewidths=0)
    if len(gltf_map.label_points):
        labels = gltf_map.label_points[:: max(1, len(gltf_map.label_points) // 180)]
        ax.scatter(labels[:, 0], labels[:, 1], s=4, c="#dc2626", alpha=0.5, linewidths=0, zorder=5)
    span = gltf_map.bounds_max - gltf_map.bounds_min
    pad = max(0.35, float(span.max()) * 0.04)
    ax.set_xlim(gltf_map.bounds_min[0] - pad, gltf_map.bounds_max[0] + pad)
    ax.set_ylim(gltf_map.bounds_min[1] - pad, gltf_map.bounds_max[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.4, alpha=0.45)


def car_outline(pose: np.ndarray, length: float = 4.3, width: float = 1.85) -> np.ndarray:
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]], dtype=np.float64)
    return pts @ yaw_to_rot(float(pose[2])).T + pose[:2]


def draw_pose(ax: plt.Axes, pose: np.ndarray, color: str, label: str) -> None:
    body = car_outline(pose)
    closed = np.vstack([body, body[0]])
    ax.fill(body[:, 0], body[:, 1], color=color, alpha=0.22, zorder=7)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.6, zorder=8)
    heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
    ax.arrow(pose[0], pose[1], heading[0] * 1.6, heading[1] * 1.6, color=color, width=0.035, zorder=9)
    ax.scatter([pose[0]], [pose[1]], s=20, c=color, zorder=10, label=label)


def render_map_debug(path: Path, gltf_map: GltfMap) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=180)
    draw_gltf_map(ax, gltf_map)
    ax.scatter(gltf_map.reference_points[:, 0], gltf_map.reference_points[:, 1], s=0.25, c="#2563eb", alpha=0.45, linewidths=0, label="NDT reference points")
    ax.set_title("Semantic GLTF map used by LiDAR NDT")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_frame(
    path: Path,
    gltf_map: GltfMap,
    frame_idx: int,
    scan_xy: np.ndarray,
    result: AlignResult,
    trajectory: Sequence[np.ndarray],
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=180)
    draw_gltf_map(ax, gltf_map)
    transformed = transform_xy(scan_xy, result.pose)
    inside = np.all((transformed >= gltf_map.bounds_min - 1.5) & (transformed <= gltf_map.bounds_max + 1.5), axis=1)
    pts = transformed[inside]
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c="#2563eb", alpha=0.38, linewidths=0, zorder=6, label="LiDAR scan in map frame")
    if trajectory:
        traj = np.asarray(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color="#ef4444", linewidth=1.8, zorder=8, label="estimated trajectory")
    draw_pose(ax, result.pose, "#dc2626", "estimated car")
    ax.set_title(
        f"LiDAR-to-GLTF NDT | frame {frame_idx:06d} | score={result.score:.3f} "
        f"inliers={result.inliers} conf={result.confidence:.2f}"
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)


def render_trajectory(path: Path, gltf_map: GltfMap, frame_indices: Sequence[int], trajectory: Sequence[np.ndarray]) -> None:
    if not HAS_MPL or not trajectory:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=220)
    draw_gltf_map(ax, gltf_map)
    traj = np.asarray(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], color="#ef4444", linewidth=2.2, zorder=8, label="LiDAR NDT trajectory")
    draw_pose(ax, traj[0], "#16a34a", f"start {frame_indices[0]:06d}")
    draw_pose(ax, traj[-1], "#dc2626", f"end {frame_indices[-1]:06d}")
    ax.set_title("LiDAR NDT trajectory on semantic GLTF map")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    ndt_grid = build_ndt_grid(gltf_map.reference_points, args.map_voxel)
    render_map_debug(args.output_dir / "gltf_semantic_map.png", gltf_map)

    pose_rows = load_pose_rows(args.dataset_root / "pose" / "poses.txt") if args.init_mode == "pose" or args.eval_pose else []
    pose_index = load_match_pose_index(args.match_file) if pose_rows else {}

    frame_indices = list(range(args.start_index, args.start_index + args.num_frames * args.step, args.step))
    rows: list[dict[str, object]] = []
    trajectory: list[np.ndarray] = []
    previous: Optional[np.ndarray] = None
    previous_confidence = 0.0

    for seq_idx, frame_idx in enumerate(frame_indices):
        scan_path = args.dataset_root / "velodyne" / f"{frame_idx:06d}.bin"
        if not scan_path.exists():
            continue
        scan_xy = load_lidar_scan(scan_path, args.z_min, args.z_max, args.range_max, args.scan_voxel, args.max_scan_points)
        if len(scan_xy) < 20:
            continue

        if args.init_mode == "pose":
            truth = get_truth_pose(frame_idx, pose_rows, pose_index)
            if truth is None:
                init, coarse_score, coarse_inliers = coarse_search(
                    scan_xy,
                    gltf_map.reference_points,
                    gltf_map.bounds_min,
                    gltf_map.bounds_max,
                    args.coarse_xy_step,
                    args.coarse_yaw_step_deg,
                    args.coarse_margin,
                )
                init_source = "auto"
            else:
                init = np.array([float(np.clip(truth[0], gltf_map.bounds_min[0], gltf_map.bounds_max[0])), float(np.clip(truth[1], gltf_map.bounds_min[1], gltf_map.bounds_max[1])), truth[2]], dtype=np.float64)
                coarse_score = float("nan")
                coarse_inliers = 0
                init_source = "pose_debug"
        elif (
            previous is not None
            and previous_confidence >= args.relocalize_below
            and not (args.coarse_every > 0 and seq_idx % args.coarse_every == 0)
        ):
            init = previous.copy()
            coarse_score = float("nan")
            coarse_inliers = 0
            init_source = "previous"
        else:
            init, coarse_score, coarse_inliers = coarse_search(
                scan_xy,
                gltf_map.reference_points,
                gltf_map.bounds_min,
                gltf_map.bounds_max,
                args.coarse_xy_step,
                args.coarse_yaw_step_deg,
                args.coarse_margin,
            )
            init_source = "auto"

        pose, score, inliers, iterations = ndt_align(scan_xy, ndt_grid, init, args.ndt_iterations)
        confidence = confidence_from(score, inliers, len(scan_xy))
        result = AlignResult(pose=pose, score=score, inliers=inliers, iterations=iterations, confidence=confidence, init_source=init_source)
        previous = pose.copy()
        previous_confidence = confidence
        trajectory.append(pose.copy())

        rows.append(
            {
                "frame": frame_idx,
                "x_map": pose[0],
                "y_map": pose[1],
                "yaw_rad": pose[2],
                "ndt_score": score,
                "ndt_inliers": inliers,
                "confidence": confidence,
                "iterations": iterations,
                "init_source": init_source,
                "coarse_score": coarse_score,
                "coarse_inliers": coarse_inliers,
                "scan_points": len(scan_xy),
            }
        )

        if not args.no_render and args.render_every > 0 and seq_idx % args.render_every == 0:
            render_frame(args.output_dir / "frames" / f"frame_{frame_idx:06d}.png", gltf_map, frame_idx, scan_xy, result, trajectory)
        if not args.quiet:
            print(
                f"frame={frame_idx:06d} x={pose[0]:.3f} y={pose[1]:.3f} yaw={pose[2]:.3f} "
                f"score={score:.3f} inliers={inliers} conf={confidence:.2f} init={init_source}"
            )

    csv_path = args.output_dir / "lidar_ndt_poses.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        render_trajectory(args.output_dir / "trajectory.png", gltf_map, [int(r["frame"]) for r in rows], trajectory)
        confidences = np.asarray([float(r["confidence"]) for r in rows], dtype=np.float64)
        summary = {
            "frames_processed": len(rows),
            "mean_confidence": float(confidences.mean()),
            "min_confidence": float(confidences.min()),
            "max_confidence": float(confidences.max()),
            "relocalization_threshold": float(args.relocalize_below),
            "frames_below_relocalization_threshold": int((confidences < args.relocalize_below).sum()),
            "frames_below_0_5": int((confidences < 0.5).sum()),
            "map_reference_points": int(len(gltf_map.reference_points)),
            "map_bounds_min": gltf_map.bounds_min.tolist(),
            "map_bounds_max": gltf_map.bounds_max.tolist(),
            "note": (
                "This is LiDAR-to-GLTF scan matching. Low confidence means the scan/map match is ambiguous; "
                "do not treat those frames as accurate localization without additional initialization or validation."
            ),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[map] {args.output_dir / 'gltf_semantic_map.png'}")
    print(f"[csv] {csv_path}")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[trajectory] {args.output_dir / 'trajectory.png'}")
    print(f"[frames] {args.output_dir / 'frames'}")


if __name__ == "__main__":
    run()
