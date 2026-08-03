#!/usr/bin/env python3
"""Build a front-camera-visible multiframe review for LiDAR slot candidates."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_top5_camera_review as camera_projection
from scripts import build_slot_aligned_camera_correspondence_report as camera_correspondence


def relative_bearing_deg(ego_pose: np.ndarray, slot_center: np.ndarray) -> float:
    """Return target bearing in the ego/LiDAR frame; positive is left."""
    delta = np.asarray(slot_center, dtype=np.float64) - np.asarray(ego_pose[:2], dtype=np.float64)
    yaw = float(ego_pose[2])
    c = np.cos(yaw)
    s = np.sin(yaw)
    forward = c * delta[0] + s * delta[1]
    left = -s * delta[0] + c * delta[1]
    return float(np.degrees(np.arctan2(left, forward)))


def projection_metrics(
    polygon_map: np.ndarray,
    ego_pose: np.ndarray,
    scale: float,
    ground_z: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Project one map slot polygon and summarize its image visibility."""
    uv, _ = camera_projection.polygon_to_uv(
        np.asarray(polygon_map, dtype=np.float64),
        np.asarray(ego_pose, dtype=np.float64),
        float(scale),
        float(ground_z),
    )
    quality = camera_projection.projection_quality([uv], int(image_width), int(image_height))
    return {
        "projection_score": float(quality["score"]),
        "projected_area_px": float(quality["area"]),
        "in_image_vertices": int(quality["in_image_points"]),
        "finite_vertices": int(quality["finite_points"]),
        "polygon_uv": uv.tolist(),
    }


def select_visible_assessments(
    assessments: list[dict[str, Any]],
    anchor_frame: int,
    half_fov_deg: float = 40.0,
    min_projection_score: float = 0.5,
    min_projected_area_px: float = 200.0,
    max_frames: int = 5,
) -> list[dict[str, Any]]:
    """Select the best valid pre-anchor frames, returned in time order."""
    valid = [
        row
        for row in assessments
        if int(row["lidar_frame"]) <= int(anchor_frame)
        and abs(float(row["bearing_deg"])) <= float(half_fov_deg)
        and float(row["projection_score"]) >= float(min_projection_score)
        and float(row["projected_area_px"]) >= float(min_projected_area_px)
    ]
    valid.sort(
        key=lambda row: (
            float(row["projection_score"]),
            float(row["projected_area_px"]),
            -abs(float(row["bearing_deg"])),
        ),
        reverse=True,
    )
    selected = valid[: max(0, int(max_frames))]
    return sorted(selected, key=lambda row: int(row["lidar_frame"]))


def nearest_camera(timestamps: dict[int, float], target: float) -> tuple[int, float, float]:
    """Match camera and LiDAR by timestamp, never by frame number."""
    return camera_correspondence.nearest_timestamp(timestamps, target)


def case_from_assessments(
    slot_id: str,
    anchor_frame: int,
    assessments: list[dict[str, Any]],
    max_frames: int = 5,
) -> dict[str, Any]:
    """Build the review-state portion of a slot case."""
    selected = assessments[: max(0, int(max_frames))]
    if selected:
        return {
            "slot_id": slot_id,
            "anchor_frame": int(anchor_frame),
            "review_status": "reviewable",
            "selected_frames": selected,
            "automatic_label": "",
        }
    return {
        "slot_id": slot_id,
        "anchor_frame": int(anchor_frame),
        "review_status": "camera_unobservable",
        "selected_frames": [],
        "automatic_label": "camera_unobservable",
    }

def rejection_reasons(
    assessment: dict[str, Any],
    anchor_frame: int,
    half_fov_deg: float,
    min_projection_score: float,
    min_projected_area_px: float,
) -> list[str]:
    """Explain every gate that prevents a camera frame from being reviewed."""
    reasons: list[str] = []
    if int(assessment["lidar_frame"]) > int(anchor_frame):
        reasons.append("after_anchor")
    if abs(float(assessment["bearing_deg"])) > float(half_fov_deg):
        reasons.append("outside_safe_fov")
    if float(assessment["projection_score"]) < float(min_projection_score):
        reasons.append("projection_score_too_low")
    if float(assessment["projected_area_px"]) < float(min_projected_area_px):
        reasons.append("projected_area_too_small")
    if not bool(assessment.get("camera_image_exists", False)):
        reasons.append("camera_image_missing")
    return reasons


def assess_sampled_frame(
    lidar_frame: int,
    frame_row: dict[str, str],
    slot: dict[str, Any],
    map_units_per_meter: float,
    image_timestamps: dict[int, float],
    dataset_root: Path,
    base_dir: Path,
    ground_quantile: float,
) -> dict[str, Any]:
    """Measure whether one sampled LiDAR frame can visually validate a slot."""
    points_path = Path(frame_row["map_points_path"])
    if not points_path.is_absolute():
        points_path = base_dir / points_path
    with np.load(points_path) as data:
        points = data["points_map_xyzi"].astype(np.float64)
        ego_pose = data["ego_map_pose"].astype(np.float64)
    finite_z = points[np.isfinite(points).all(axis=1), 2]
    ground_z = float(np.quantile(finite_z, ground_quantile)) if len(finite_z) else 0.0
    center = np.asarray(slot["center_map"], dtype=np.float64)
    polygon = np.asarray(slot["polygon_map"], dtype=np.float64)
    metrics = projection_metrics(polygon, ego_pose, map_units_per_meter, ground_z, 1280, 720)
    lidar_timestamp = float(frame_row["lidar_timestamp"])
    camera_frame, camera_timestamp, camera_delta = nearest_camera(image_timestamps, lidar_timestamp)
    camera_source = dataset_root / "image" / f"left{camera_frame:06d}.png"
    result: dict[str, Any] = {
        "lidar_frame": int(lidar_frame),
        "lidar_timestamp": lidar_timestamp,
        "camera_frame": int(camera_frame),
        "camera_timestamp": float(camera_timestamp),
        "camera_lidar_dt_sec": float(camera_delta),
        "camera_image_source": str(camera_source),
        "camera_image_exists": camera_source.exists(),
        "bearing_deg": relative_bearing_deg(ego_pose, center),
        "distance_m": float(np.linalg.norm(center - ego_pose[:2]) / max(map_units_per_meter, 1e-9)),
        "ground_z": ground_z,
    }
    result.update(metrics)
    return result


def draw_camera_slot_overlay(
    source_path: Path,
    output_path: Path,
    polygon_uv: list[list[float]],
    slot_id: str,
) -> None:
    """Copy a camera image and draw the projected target slot in green."""
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    finite_points = [
        (float(point[0]), float(point[1]))
        for point in polygon_uv
        if len(point) >= 2 and np.isfinite(point[:2]).all()
    ]
    if len(finite_points) >= 3:
        draw.line(finite_points + [finite_points[0]], fill=(0, 255, 80), width=6)
        cx = sum(point[0] for point in finite_points) / len(finite_points)
        cy = sum(point[1] for point in finite_points) / len(finite_points)
        label = f"{slot_id} target"
        draw.rectangle((cx - 4, cy - 18, cx + 8 * len(label), cy + 4), fill=(0, 0, 0))
        draw.text((cx, cy - 16), label, fill=(0, 255, 80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


REVIEW_LABELS = [
    ("target_occupied", "目标车位有车（判断准确）"),
    ("adjacent_occupied", "相邻车位有车（串位）"),
    ("target_empty", "目标车位为空（误报）"),
    ("occluded", "目标区域被遮挡"),
    ("bad_projection", "投影/对齐错误"),
    ("unclear", "仍然看不清"),
]


def render_report(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Render the self-contained labeling UI; image assets remain relative files."""
    case_sections: list[str] = []
    buttons = "".join(
        f"<button type='button' data-label='{html.escape(value)}'>{html.escape(text)}</button>"
        for value, text in REVIEW_LABELS
    )
    for index, case in enumerate(cases, start=1):
        slot_id = str(case["slot_id"])
        map_path = str(case.get("slot_global_map_path", ""))
        lidar_path = str(case.get("slot_accumulated_zoom_path", ""))
        context_cards = []
        if map_path:
            context_cards.append(
                f"<figure><figcaption>地图目标车位</figcaption><img src='{html.escape(map_path)}' loading='lazy'></figure>"
            )
        if lidar_path:
            context_cards.append(
                f"<figure><figcaption>LiDAR 多帧累积证据</figcaption><img src='{html.escape(lidar_path)}' loading='lazy'></figure>"
            )
        frame_cards = []
        for frame in case.get("selected_frames", []):
            overlay_path = html.escape(str(frame.get("overlay_path", "")))
            frame_cards.append(
                "<figure class='frame-card'>"
                f"<figcaption>LiDAR {int(frame['lidar_frame'])} → Camera {int(frame['camera_frame'])}</figcaption>"
                f"<img src='{overlay_path}' loading='lazy'>"
                "<div class='frame-metrics'>"
                f"bearing={float(frame['bearing_deg']):+.1f}° · "
                f"distance={float(frame['distance_m']):.1f}m · "
                f"projection={float(frame['projection_score']):.2f} · "
                f"area={float(frame['projected_area_px']):.0f}px² · "
                f"dt={float(frame['camera_lidar_dt_sec']):+.3f}s"
                "</div></figure>"
            )
        review_status = str(case.get("review_status", "camera_unobservable"))
        if review_status == "camera_unobservable":
            frame_html = (
                "<div class='unobservable'>camera_unobservable：锚点之前没有同时满足 "
                "±40° FOV、真实投影和最小面积门槛的相机帧。本项不计入准确率分母。</div>"
            )
            controls = "<span class='automatic'>自动弃权：camera_unobservable</span>"
        else:
            frame_html = "<div class='frame-strip'>" + "".join(frame_cards) + "</div>"
            controls = (
                "<div class='review-controls'>"
                + buttons
                + f"<input type='text' placeholder='{html.escape(slot_id)} 备注'>"
                + "<span class='chosen'>未标注</span></div>"
            )
        case_sections.append(
            f"<section class='case' id='case-{index}' data-index='{index - 1}' "
            f"data-status='{html.escape(review_status)}'>"
            f"<h2>{index}. {html.escape(slot_id)} · anchor {int(case['anchor_frame'])}</h2>"
            f"<div class='context-grid'>{''.join(context_cards)}</div>"
            "<h3>锚点前、前置相机可见的多帧</h3>"
            f"{frame_html}{controls}</section>"
        )
    case_json = json.dumps(cases, ensure_ascii=False).replace("</", "<\\/")
    summary_json = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
    document = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>前置相机可见的多帧车位人工复核</title>
<style>
:root { color-scheme: light; --green:#047857; --border:#d1d5db; --muted:#4b5563; }
body { font-family:system-ui,-apple-system,sans-serif; margin:24px; color:#111827; background:#f8fafc; }
header,.summary-panel,.case { max-width:1500px; margin:0 auto 20px; }
header,.summary-panel { background:white; border:1px solid var(--border); border-radius:12px; padding:16px; }
.note { background:#ecfdf5; border:1px solid #a7f3d0; padding:12px; border-radius:8px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin-top:12px; }
.stat { background:#f3f4f6; border:1px solid var(--border); border-radius:8px; padding:10px; }
.stat strong { display:block; font-size:20px; }
.case { background:white; border:1px solid var(--border); border-radius:12px; padding:18px; }
.context-grid { display:grid; grid-template-columns:repeat(2,minmax(280px,1fr)); gap:12px; }
.frame-strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }
figure { margin:0; }
figcaption { font-weight:700; margin-bottom:6px; }
img { width:100%; max-height:540px; object-fit:contain; border:1px solid var(--border); background:#111827; }
.frame-metrics { color:var(--muted); font-size:12px; padding:6px 0 12px; }
.review-controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:14px; }
button { padding:8px 10px; border:1px solid #9ca3af; border-radius:7px; background:white; cursor:pointer; }
button.selected { color:white; background:var(--green); border-color:var(--green); }
input { min-width:260px; padding:8px; border:1px solid #9ca3af; border-radius:7px; }
.unobservable { padding:16px; background:#fff7ed; border:1px solid #fdba74; border-radius:8px; }
.automatic { color:#9a3412; font-weight:700; }
textarea { width:100%; min-height:260px; font-family:ui-monospace,monospace; }
pre { overflow:auto; }
@media(max-width:800px) { .context-grid { grid-template-columns:1fr; } body { margin:10px; } }
</style>
</head>
<body>
<header>
<h1>前置相机可见的多帧车位人工复核</h1>
<div class="note">只展示锚点之前、位于前置相机安全 FOV 内且车位投影真实进入画面的帧。整组只标一次。</div>
<pre>__SUMMARY__</pre>
</header>
<div class="summary-panel">
<h2>实时统计</h2>
<div class="stats">
<div class="stat">已标注<strong id="reviewed-count">0</strong></div>
<div class="stat">verifiable_count<strong id="verifiable-count">0</strong></div>
<div class="stat">abstention_count<strong id="abstention-count">0</strong></div>
<div class="stat">precision<strong id="precision">—</strong></div>
<div class="stat">相邻串位率<strong id="adjacent-rate">—</strong></div>
<div class="stat">空位误报率<strong id="empty-rate">—</strong></div>
<div class="stat">可验证覆盖率<strong id="coverage">—</strong></div>
</div>
</div>
__SECTIONS__
<div class="summary-panel">
<h2>导出人工标签 JSON</h2>
<button type="button" id="refresh-export">刷新 JSON</button>
<textarea id="export-json" spellcheck="false"></textarea>
</div>
<script id="case-data" type="application/json">__CASE_JSON__</script>
<script>
const cases = JSON.parse(document.getElementById("case-data").textContent);
const storageKey = "front-camera-multiframe-slot-review-v1";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { saved = {}; }
const reviews = cases.map((item, index) => ({
  slot_id: item.slot_id,
  anchor_frame: item.anchor_frame,
  review_status: item.review_status,
  selected_lidar_frames: item.selected_frames.map(frame => frame.lidar_frame),
  label: item.review_status === "camera_unobservable" ? "camera_unobservable" : (saved[index]?.label || ""),
  note: saved[index]?.note || ""
}));
function ratio(n, d) {
  return d ? (n + "/" + d + " (" + (100*n/d).toFixed(1) + "%)") : "0/0 (—)";
}
function refreshStats() {
  const labels = reviews.map(row => row.label).filter(Boolean);
  const verifiable = labels.filter(label => ["target_occupied","adjacent_occupied","target_empty","bad_projection"].includes(label));
  const abstentions = labels.filter(label => ["occluded","unclear","camera_unobservable"].includes(label));
  const correct = verifiable.filter(label => label === "target_occupied").length;
  const adjacent = verifiable.filter(label => label === "adjacent_occupied").length;
  const empty = verifiable.filter(label => label === "target_empty").length;
  document.getElementById("reviewed-count").textContent = labels.length + "/" + cases.length;
  document.getElementById("verifiable-count").textContent = verifiable.length;
  document.getElementById("abstention-count").textContent = abstentions.length;
  document.getElementById("precision").textContent = ratio(correct, verifiable.length);
  document.getElementById("adjacent-rate").textContent = ratio(adjacent, verifiable.length);
  document.getElementById("empty-rate").textContent = ratio(empty, verifiable.length);
  document.getElementById("coverage").textContent = ratio(verifiable.length, cases.length);
}
function persist() {
  const userRows = Object.fromEntries(reviews.map((row, index) => [index, {label:row.label,note:row.note}]));
  localStorage.setItem(storageKey, JSON.stringify(userRows));
  document.getElementById("export-json").value = JSON.stringify(reviews, null, 2);
  refreshStats();
}
document.querySelectorAll(".case").forEach(section => {
  const index = Number(section.dataset.index);
  const chosen = section.querySelector(".chosen");
  const input = section.querySelector("input");
  if (!input) return;
  input.value = reviews[index].note;
  input.addEventListener("input", () => { reviews[index].note = input.value; persist(); });
  section.querySelectorAll("button[data-label]").forEach(button => {
    if (button.dataset.label === reviews[index].label) button.classList.add("selected");
    button.addEventListener("click", () => {
      section.querySelectorAll("button[data-label]").forEach(other => other.classList.remove("selected"));
      button.classList.add("selected");
      reviews[index].label = button.dataset.label;
      chosen.textContent = button.textContent;
      persist();
    });
  });
  if (reviews[index].label) chosen.textContent = reviews[index].label;
});
document.getElementById("refresh-export").addEventListener("click", persist);
persist();
</script>
</body>
</html>
"""
    return (
        document.replace("__SUMMARY__", summary_json)
        .replace("__SECTIONS__", "".join(case_sections))
        .replace("__CASE_JSON__", case_json)
    )


def preselect_pose_visible_frame_ids(
    frame_rows: dict[int, dict[str, str]],
    slot_center: np.ndarray,
    anchor_frame: int,
    lookback_frames: int,
    frame_stride: int,
    half_fov_deg: float,
    limit: int,
) -> list[int]:
    """Find close approach frames that are geometrically inside the front FOV."""
    stride = max(1, int(frame_stride))
    start = int(anchor_frame) - max(0, int(lookback_frames))
    candidates: list[tuple[float, float, int]] = []
    center = np.asarray(slot_center, dtype=np.float64)
    for frame_id, row in frame_rows.items():
        if frame_id < start or frame_id > int(anchor_frame):
            continue
        if (int(anchor_frame) - frame_id) % stride != 0:
            continue
        pose = np.asarray(
            [float(row["map_x"]), float(row["map_y"]), float(row["map_yaw"])],
            dtype=np.float64,
        )
        bearing = relative_bearing_deg(pose, center)
        if abs(bearing) > float(half_fov_deg):
            continue
        distance = float(np.linalg.norm(center - pose[:2]))
        candidates.append((distance, abs(bearing), int(frame_id)))
    candidates.sort()
    selected = [frame_id for _, _, frame_id in candidates[: max(0, int(limit))]]
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a pre-anchor, front-camera-visible multiframe review for core-supported slots"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8"),
    )
    parser.add_argument("--frames", type=Path, default=Path("outputs/frame_map_dataset/frames.csv"))
    parser.add_argument(
        "--slot-database",
        type=Path,
        default=Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ParkingAgent/dataset/dataset/dataset"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--half-fov-deg", type=float, default=40.0)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--min-projection-score", type=float, default=0.5)
    parser.add_argument("--min-projected-area-px", type=float, default=200.0)
    parser.add_argument("--ground-quantile", type=float, default=0.08)
    parser.add_argument("--camera-lookback-frames", type=int, default=400)
    parser.add_argument("--camera-frame-stride", type=int, default=5)
    parser.add_argument("--camera-pose-prefilter-limit", type=int, default=30)
    args = parser.parse_args()

    base_dir = Path.cwd()
    output_dir = args.output_dir or (args.input_dir / "front_camera_multiframe_review")
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows = camera_correspondence.read_csv(args.input_dir / "slot_aligned_evidence.csv")
    evidence_rows = [
        row
        for row in evidence_rows
        if row.get("state_by_slot_aligned_accumulation") == "accumulated_vehicle_core_supported"
    ]
    evidence_rows.sort(
        key=lambda row: (
            -camera_correspondence.f(row, "max_vehicle_like_score"),
            -camera_correspondence.i(row, "core_overlap_count"),
        )
    )
    frame_rows = {
        int(row["frame"]): row
        for row in camera_correspondence.read_csv(args.frames)
    }
    image_timestamps = camera_correspondence.load_timestamps(args.dataset_root / "image" / "timestamps.txt")
    slot_database = camera_correspondence.load_json(args.slot_database)
    map_units_per_meter = float(slot_database["map_units_per_meter"])
    slots = {str(slot["slot_id"]): slot for slot in slot_database["slots"]}

    cases: list[dict[str, Any]] = []
    for index, evidence in enumerate(evidence_rows, start=1):
        slot_id = str(evidence["slot_id"])
        slot = slots.get(slot_id)
        anchor_frame = camera_correspondence.i(evidence, "anchor_frame")
        sampled_frames = [
            int(value)
            for value in camera_correspondence.parse_list(evidence.get("sampled_frame_ids", ""))
            if str(value).strip()
        ]
        assessments: list[dict[str, Any]] = []
        camera_search_frames: list[int] = []
        if slot is not None:
            camera_search_frames = preselect_pose_visible_frame_ids(
                frame_rows,
                np.asarray(slot["center_map"], dtype=np.float64),
                anchor_frame,
                args.camera_lookback_frames,
                args.camera_frame_stride,
                args.half_fov_deg,
                args.camera_pose_prefilter_limit,
            )
            for lidar_frame in camera_search_frames:
                frame_row = frame_rows.get(lidar_frame)
                if frame_row is None:
                    continue
                try:
                    assessment = assess_sampled_frame(
                        lidar_frame,
                        frame_row,
                        slot,
                        map_units_per_meter,
                        image_timestamps,
                        args.dataset_root,
                        base_dir,
                        args.ground_quantile,
                    )
                except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                    assessment = {
                        "lidar_frame": int(lidar_frame),
                        "bearing_deg": 999.0,
                        "projection_score": 0.0,
                        "projected_area_px": 0.0,
                        "camera_image_exists": False,
                        "assessment_error": str(exc),
                    }
                assessment["rejection_reasons"] = rejection_reasons(
                    assessment,
                    anchor_frame,
                    args.half_fov_deg,
                    args.min_projection_score,
                    args.min_projected_area_px,
                )
                assessments.append(assessment)

        selected = select_visible_assessments(
            assessments,
            anchor_frame,
            args.half_fov_deg,
            args.min_projection_score,
            args.min_projected_area_px,
            args.max_frames,
        )
        case = case_from_assessments(slot_id, anchor_frame, selected, args.max_frames)
        prefix = f"{index:02d}_{slot_id}"
        slot_map_path = assets_dir / f"{prefix}_global_slot.png"
        lidar_zoom_path = assets_dir / f"{prefix}_slot_accumulated_zoom.png"
        if slot is not None:
            camera_correspondence.draw_slot_global_context(slot_map_path, slots, slot_id)
            camera_correspondence.draw_slot_accumulated_zoom(
                lidar_zoom_path,
                slots,
                slot_id,
                sampled_frames,
                frame_rows,
                base_dir,
            )
        for selected_index, assessment in enumerate(case["selected_frames"], start=1):
            source = Path(str(assessment["camera_image_source"]))
            overlay_path = assets_dir / (
                f"{prefix}_{selected_index:02d}_lidar_{int(assessment['lidar_frame']):06d}"
                f"_camera_{int(assessment['camera_frame']):06d}_slot_overlay.png"
            )
            draw_camera_slot_overlay(
                source,
                overlay_path,
                assessment["polygon_uv"],
                slot_id,
            )
            assessment["overlay_path"] = f"assets/{overlay_path.name}"

        case.update(
            {
                "index": index,
                "state": evidence.get("state_by_slot_aligned_accumulation", ""),
                "max_vehicle_like_score": camera_correspondence.f(evidence, "max_vehicle_like_score"),
                "core_overlap_count": camera_correspondence.i(evidence, "core_overlap_count"),
                "boundary_ratio": camera_correspondence.f(evidence, "boundary_ratio"),
                "adjacent_overlap_ratio": camera_correspondence.f(evidence, "adjacent_overlap_ratio"),
                "sampled_frame_ids": sampled_frames,
                "assessed_frames": assessments,
                "slot_global_map_path": f"assets/{slot_map_path.name}" if slot_map_path.exists() else "",
                "slot_accumulated_zoom_path": f"assets/{lidar_zoom_path.name}" if lidar_zoom_path.exists() else "",
            }
        )
        cases.append(case)
        print(
            f"[{index:02d}/{len(evidence_rows):02d}] {slot_id} "
            f"review={case['review_status']} frames={len(case['selected_frames'])}",
            flush=True,
        )

    status_counts = Counter(str(case["review_status"]) for case in cases)
    selected_counts = Counter(len(case["selected_frames"]) for case in cases)
    summary = {
        "candidate_count": len(cases),
        "review_status_counts": dict(status_counts),
        "selected_frame_count_distribution": {
            str(key): value for key, value in sorted(selected_counts.items())
        },
        "selection_policy": {
            "pre_anchor_only": True,
            "physical_horizontal_fov_deg": 90.0,
            "safe_half_fov_deg": args.half_fov_deg,
            "max_frames_per_slot": args.max_frames,
            "camera_lookback_frames": args.camera_lookback_frames,
            "camera_frame_stride": args.camera_frame_stride,
            "camera_pose_prefilter_limit": args.camera_pose_prefilter_limit,
            "min_projection_score": args.min_projection_score,
            "min_projected_area_px": args.min_projected_area_px,
            "camera_match": "nearest_timestamp",
        },
    }
    json_path = output_dir / "front_camera_multiframe_cases.json"
    csv_path = output_dir / "front_camera_multiframe_cases.csv"
    html_path = output_dir / "front_camera_multiframe_review.html"
    json_path.write_text(
        json.dumps({"summary": summary, "cases": cases}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    csv_fields = [
        "index",
        "slot_id",
        "anchor_frame",
        "review_status",
        "automatic_label",
        "max_vehicle_like_score",
        "core_overlap_count",
        "boundary_ratio",
        "adjacent_overlap_ratio",
        "selected_frame_count",
        "selected_lidar_frames",
        "selected_camera_frames",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "index": case["index"],
                    "slot_id": case["slot_id"],
                    "anchor_frame": case["anchor_frame"],
                    "review_status": case["review_status"],
                    "automatic_label": case["automatic_label"],
                    "max_vehicle_like_score": case["max_vehicle_like_score"],
                    "core_overlap_count": case["core_overlap_count"],
                    "boundary_ratio": case["boundary_ratio"],
                    "adjacent_overlap_ratio": case["adjacent_overlap_ratio"],
                    "selected_frame_count": len(case["selected_frames"]),
                    "selected_lidar_frames": json.dumps(
                        [frame["lidar_frame"] for frame in case["selected_frames"]]
                    ),
                    "selected_camera_frames": json.dumps(
                        [frame["camera_frame"] for frame in case["selected_frames"]]
                    ),
                }
            )
    html_path.write_text(render_report(cases, summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": summary,
                "html": str(html_path),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
