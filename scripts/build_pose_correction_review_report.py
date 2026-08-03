#!/usr/bin/env python3
"""Build a synchronized camera/LiDAR/global-map before-after correction report."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gltf_lidar_ndt import load_gltf_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-frames", type=Path, default=Path("outputs/frame_map_dataset/frames.csv"))
    parser.add_argument("--corrected-frames", type=Path, default=Path("outputs/frame_map_dataset_pose_corrected/frames.csv"))
    parser.add_argument("--keyframe-audit", type=Path, default=Path("outputs/pose_drift_correction/keyframe_registration.csv"))
    parser.add_argument("--pose-corrections", type=Path, default=Path("outputs/pose_drift_correction/pose_corrections.csv"))
    parser.add_argument("--slot-db", type=Path, default=Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json"))
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_drift_correction/review"))
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--extra-large-correction-count", type=int, default=6)
    parser.add_argument("--map-radius-m", type=float, default=16.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_slots(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(resolve(path).read_text(encoding="utf-8"))
    slots = payload.get("slots", payload)
    return list(slots.values()) if isinstance(slots, dict) else list(slots)


def choose_frames(
    audit: list[dict[str, str]],
    corrections: list[dict[str, str]],
    sample_count: int,
    extra_count: int,
) -> list[int]:
    audit_distance = np.asarray([float(row["travel_distance_m"]) for row in audit])
    targets = np.linspace(float(audit_distance.min()), float(audit_distance.max()), sample_count)
    selected = {int(audit[int(np.argmin(np.abs(audit_distance - target)))]["frame"]) for target in targets}
    ranked = sorted(
        corrections,
        key=lambda row: math.hypot(float(row["correction_dx_m"]), float(row["correction_dy_m"])),
        reverse=True,
    )
    extras: list[int] = []
    for row in ranked:
        frame = int(row["frame"])
        if all(abs(frame - existing) >= 100 for existing in extras):
            extras.append(frame)
        if len(extras) >= extra_count:
            break
    selected.update(extras)
    return sorted(selected)


def point_mask(raw: np.ndarray) -> np.ndarray:
    ranges = np.linalg.norm(raw[:, :2], axis=1)
    return (ranges >= 1.5) & (ranges <= 25.0) & (raw[:, 2] >= 0.30) & (raw[:, 2] <= 2.20)


def downsample(points: np.ndarray, maximum: int = 7000) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=int)]


def nearby_slots(slots: list[dict[str, Any]], center: np.ndarray, radius_map: float) -> list[dict[str, Any]]:
    result = []
    for slot in slots:
        slot_center = np.asarray(slot["center_map"], dtype=np.float64)
        if np.linalg.norm(slot_center - center) <= radius_map:
            result.append(slot)
    return result


def draw_map_panel(
    ax: plt.Axes,
    title: str,
    points: np.ndarray,
    pose: np.ndarray,
    slots: list[dict[str, Any]],
    structural: np.ndarray,
    radius_map: float,
) -> None:
    local_structural = structural[
        (np.abs(structural[:, 0] - pose[0]) <= radius_map)
        & (np.abs(structural[:, 1] - pose[1]) <= radius_map)
    ]
    if len(local_structural):
        local_structural = downsample(local_structural, 10000)
        ax.scatter(local_structural[:, 0], local_structural[:, 1], s=0.4, color="#111827", alpha=0.2)
    for slot in slots:
        polygon = np.asarray(slot["polygon_map"], dtype=np.float64)
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#64748b", linewidth=0.65, alpha=0.8)
    sampled = downsample(points, 7000)
    if len(sampled):
        ax.scatter(sampled[:, 0], sampled[:, 1], c=sampled[:, 2], cmap="turbo", s=1.5, alpha=0.8)
    ax.scatter([pose[0]], [pose[1]], s=35, color="#dc2626", zorder=7)
    arrow = radius_map * 0.16
    ax.arrow(
        pose[0], pose[1], arrow * math.cos(pose[2]), arrow * math.sin(pose[2]),
        color="#dc2626", width=radius_map * 0.008, head_width=radius_map * 0.06, zorder=8,
    )
    ax.set(xlim=(pose[0] - radius_map, pose[0] + radius_map), ylim=(pose[1] - radius_map, pose[1] + radius_map))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.15)


def render_case(
    destination: Path,
    frame: int,
    original_row: dict[str, str],
    corrected_row: dict[str, str],
    slots: list[dict[str, Any]],
    structural: np.ndarray,
    map_scale: float,
    radius_m: float,
) -> None:
    raw = np.fromfile(resolve(original_row["lidar_path"]), dtype=np.float32).reshape(-1, 4)
    raw_vehicle = raw[point_mask(raw)]
    with np.load(resolve(original_row["map_points_path"])) as data:
        original_points = data["points_map_xyzi"].astype(np.float64)
        original_pose = data["ego_map_pose"].astype(np.float64)
    with np.load(resolve(corrected_row["map_points_path"])) as data:
        corrected_points = data["points_map_xyzi"].astype(np.float64)
        corrected_pose = data["ego_map_pose"].astype(np.float64)
    mask = point_mask(raw)
    original_points = original_points[mask]
    corrected_points = corrected_points[mask]
    radius_map = radius_m * map_scale
    center = 0.5 * (original_pose[:2] + corrected_pose[:2])
    local_slots = nearby_slots(slots, center, radius_map * 1.4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    sampled_raw = downsample(raw_vehicle, 7000)
    axes[0].scatter(sampled_raw[:, 0], sampled_raw[:, 1], c=sampled_raw[:, 2], cmap="turbo", s=1.5)
    axes[0].scatter([0], [0], s=35, color="#dc2626")
    axes[0].arrow(0, 0, 2.0, 0, color="#dc2626", width=0.08, head_width=0.5)
    axes[0].set(xlim=(-radius_m, radius_m), ylim=(-radius_m, radius_m), title="Raw LiDAR BEV [m]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.15)
    draw_map_panel(axes[1], "Original map projection", original_points, original_pose, local_slots, structural, radius_map)
    draw_map_panel(axes[2], "Pose-corrected map projection", corrected_points, corrected_pose, local_slots, structural, radius_map)
    fig.suptitle(f"LiDAR frame {frame:06d} | vehicle-height returns 0.30-2.20 m", fontsize=13)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def mime_for(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def embed_images(source_html: str, report_dir: Path) -> str:
    result = source_html
    for asset in sorted((report_dir / "assets").glob("*")):
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        data_url = f"data:{mime_for(asset)};base64,{encoded}"
        result = result.replace(f'src="assets/{asset.name}"', f'src="{data_url}"')
        result = result.replace(f"src='assets/{asset.name}'", f"src='{data_url}'")
    return result


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    original_rows = {int(row["frame"]): row for row in read_csv(args.original_frames)}
    corrected_rows = {int(row["frame"]): row for row in read_csv(args.corrected_frames)}
    audit = read_csv(args.keyframe_audit)
    corrections = read_csv(args.pose_corrections)
    correction_by_frame = {int(row["frame"]): row for row in corrections}
    audit_frames = np.asarray([int(row["frame"]) for row in audit])
    slots = load_slots(args.slot_db)
    with np.load(resolve(original_rows[min(original_rows)]["map_points_path"])) as data:
        map_scale = float(data["map_scale"][0])
    gltf = load_gltf_map(resolve(args.gltf), 0.04)
    structural = np.vstack(
        [gltf.layers[name].points for name in ("wall", "elevator", "arrester") if name in gltf.layers]
    )
    selected = choose_frames(audit, corrections, args.sample_count, args.extra_large_correction_count)

    cases = []
    for ordinal, frame in enumerate(selected, start=1):
        original = original_rows[frame]
        corrected = corrected_rows[frame]
        camera_source = Path(corrected["camera_image_path"])
        camera_name = f"{ordinal:02d}_lidar_{frame:06d}_camera_{int(corrected['camera_frame']):06d}{camera_source.suffix.lower()}"
        camera_destination = assets / camera_name
        shutil.copy2(camera_source, camera_destination)
        compare_name = f"{ordinal:02d}_lidar_{frame:06d}_pointcloud_map_compare.png"
        render_case(
            assets / compare_name,
            frame,
            original,
            corrected,
            slots,
            structural,
            map_scale,
            args.map_radius_m,
        )
        nearest_audit = audit[int(np.argmin(np.abs(audit_frames - frame)))]
        correction = correction_by_frame[frame]
        cases.append(
            {
                "frame": frame,
                "camera_frame": int(corrected["camera_frame"]),
                "camera_dt": float(corrected["camera_lidar_dt_sec"]),
                "camera_asset": f"assets/{camera_name}",
                "compare_asset": f"assets/{compare_name}",
                "travel_distance_m": float(correction["travel_distance_m"]),
                "dx_m": float(correction["correction_dx_m"]),
                "dy_m": float(correction["correction_dy_m"]),
                "dyaw_deg": float(correction["correction_dyaw_deg"]),
                "nearest_keyframe": int(nearest_audit["frame"]),
                "constraint_accepted": bool(int(nearest_audit["accepted"])),
                "residual_before_m": float(nearest_audit["residual_before_m"]),
                "smoothed_residual_m": float(nearest_audit["smoothed_residual_m"]),
            }
        )
        print(f"[report] {ordinal}/{len(selected)} lidar={frame} camera={cases[-1]['camera_frame']}", flush=True)

    sections = []
    for index, case in enumerate(cases, start=1):
        norm = math.hypot(case["dx_m"], case["dy_m"])
        warning = " <span class='warning'>correction near 1.5 m limit</span>" if norm >= 1.49 else ""
        sections.append(
            f"<section class='case'><h2>{index:02d}. LiDAR {case['frame']:06d} / Camera {case['camera_frame']:06d}{warning}</h2>"
            f"<p>distance={case['travel_distance_m']:.1f} m | camera dt={case['camera_dt']:+.4f} s | "
            f"correction=({case['dx_m']:+.2f}, {case['dy_m']:+.2f}) m, {case['dyaw_deg']:+.2f} deg | "
            f"nearest keyframe={case['nearest_keyframe']} ({'accepted' if case['constraint_accepted'] else 'rejected'}) | "
            f"structural residual {case['residual_before_m']:.3f} -> {case['smoothed_residual_m']:.3f} m</p>"
            f"<div class='media'><figure><img src='{case['camera_asset']}'><figcaption>Timestamp-synchronized camera image</figcaption></figure>"
            f"<figure class='wide'><img src='{case['compare_asset']}'><figcaption>Raw LiDAR, original global projection, corrected global projection</figcaption></figure></div></section>"
        )
    summary = {
        "case_count": len(cases),
        "selected_lidar_frames": selected,
        "camera_binding": "nearest timestamp",
        "map_scale_units_per_meter": map_scale,
        "cases": cases,
    }
    (output_dir / "review_index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Pose correction correspondence review</title>
<style>body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:22px;background:#f5f7fa;color:#172033}}h1{{margin-bottom:4px}}.note{{background:#e8f0ff;border-left:4px solid #2563eb;padding:12px;margin:16px 0}}.case{{background:white;border:1px solid #d8dee9;border-radius:6px;padding:16px;margin:18px 0}}.media{{display:grid;grid-template-columns:minmax(320px,0.75fr) minmax(640px,2fr);gap:14px;align-items:start}}figure{{margin:0}}img{{display:block;width:100%;height:auto;border:1px solid #cbd5e1}}figcaption{{font-size:12px;color:#475569;margin-top:5px}}.warning{{color:#b91c1c;font-size:13px}}@media(max-width:900px){{.media{{grid-template-columns:1fr}}}}</style></head><body>
<h1>点云、照片与全局地图校正对照</h1><p>共 {len(cases)} 个代表帧，按累计里程排序。</p>
<div class='note'>照片按时间戳匹配，不再使用同编号图片。点云颜色表示 LiDAR z；灰线是车位 polygon，黑色背景点是 glTF 静态结构。该报告用于人工校验 pose correction，不直接修改车位 occupied/free 状态。</div>
{''.join(sections)}</body></html>"""
    report_path = output_dir / "pose_correction_correspondence_report.html"
    report_path.write_text(page, encoding="utf-8")
    embedded_path = output_dir / "pose_correction_correspondence_report_embedded.html"
    embedded_path.write_text(embed_images(page, output_dir), encoding="utf-8")
    print(f"[report] {report_path}")
    print(f"[embedded] {embedded_path}")


if __name__ == "__main__":
    main()
