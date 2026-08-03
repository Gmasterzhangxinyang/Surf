#!/usr/bin/env python3
"""Generate top-5 camera overlay/crop review assets for possible-free cases."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_PART2 = Path("outputs/part2_case_review_pack")
DEFAULT_PART1 = Path("outputs/part1_slot_scoring_1000_boundary_fixed")
DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_OUTPUT = DEFAULT_PART2 / "top5_camera_review"
IMAGE_W = 1280
IMAGE_H = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-5 possible-free camera review overlays")
    parser.add_argument("--part2-pack", type=Path, default=DEFAULT_PART2)
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--crop-pad-px", type=int, default=80)
    parser.add_argument("--ground-quantile", type=float, default=0.08)
    parser.add_argument("--min-visible-score", type=float, default=0.01)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_frames(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="") as handle:
        return {int(row["frame"]): row for row in csv.DictReader(handle)}


def load_slot_evidence(path: Path, slot_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    by_slot = {slot_id: [] for slot_id in slot_ids}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            slot_id = str(row.get("slot_id"))
            if slot_id in by_slot:
                by_slot[slot_id].append(row)
    for rows in by_slot.values():
        rows.sort(
            key=lambda r: (
                -float(r.get("ray_free_ratio", 0.0)),
                -float(r.get("visibility_score", 0.0)),
                float(r.get("distance_to_slot_m", 1e9)),
            )
        )
    return by_slot


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def map_xy_to_lidar_xy(points_map: np.ndarray, ego_pose: np.ndarray, scale: float) -> np.ndarray:
    delta = (points_map - ego_pose[:2]) / max(scale, 1e-9)
    return delta @ yaw_to_rot(-float(ego_pose[2])).T


def lidar_to_camera(points_lidar_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_lidar_xyz, dtype=np.float64)
    # The calibration translation is the camera origin in LiDAR/vehicle
    # axes. Subtract it before converting x-forward/y-left/z-up into the
    # camera optical axes x-right/y-down/z-forward.
    camera_origin_lidar = np.array([0.6, 0.0, -0.07], dtype=np.float64)
    relative = points - camera_origin_lidar
    return np.column_stack([-relative[:, 1], -relative[:, 2], relative[:, 0]])


def project_camera(points_cam_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx = 527.525085
    fy = 527.525085
    cx = 636.297913
    cy = 357.787354
    z = points_cam_xyz[:, 2]
    valid = z > 1e-4
    uv = np.full((len(points_cam_xyz), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = fx * points_cam_xyz[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points_cam_xyz[valid, 1] / z[valid] + cy
    return uv, valid


def load_points_for_frame(frame_id: int, frame_rows: dict[int, dict[str, str]], repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    row = frame_rows[frame_id]
    path = Path(row["map_points_path"])
    if not path.is_absolute():
        path = repo_root / path
    data = np.load(path)
    return data["points_map_xyzi"].astype(np.float64), data["ego_map_pose"].astype(np.float64)


def polygon_to_uv(polygon_map: np.ndarray, ego_pose: np.ndarray, scale: float, z_lidar: float) -> tuple[np.ndarray, np.ndarray]:
    xy_lidar = map_xy_to_lidar_xy(polygon_map, ego_pose, scale)
    xyz_lidar = np.column_stack([xy_lidar, np.full(len(xy_lidar), z_lidar, dtype=np.float64)])
    return project_camera(lidar_to_camera(xyz_lidar))


def polygon_area(uv: np.ndarray) -> float:
    valid = np.isfinite(uv).all(axis=1)
    pts = uv[valid]
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def projection_quality(uv_arrays: list[np.ndarray], w: int = IMAGE_W, h: int = IMAGE_H) -> dict[str, object]:
    finite = []
    in_image_points = 0
    finite_points = 0
    for uv in uv_arrays:
        valid = np.isfinite(uv).all(axis=1)
        if valid.any():
            pts = uv[valid]
            finite.append(pts)
            finite_points += int(len(pts))
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            in_image_points += int(inside.sum())
    if not finite:
        return {"score": 0.0, "bbox": None, "in_image_points": 0, "finite_points": 0, "area": 0.0}
    pts = np.vstack(finite)
    x0 = max(0.0, float(pts[:, 0].min()))
    y0 = max(0.0, float(pts[:, 1].min()))
    x1 = min(float(w), float(pts[:, 0].max()))
    y1 = min(float(h), float(pts[:, 1].max()))
    overlap_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    raw_area = sum(polygon_area(uv) for uv in uv_arrays)
    point_score = in_image_points / max(finite_points, 1)
    area_score = min(1.0, overlap_area / max(raw_area, 1.0))
    score = 0.65 * area_score + 0.35 * point_score
    bbox = None
    if overlap_area > 0.0:
        bbox = [int(x0), int(y0), int(x1), int(y1)]
    return {
        "score": float(score),
        "bbox": bbox,
        "in_image_points": int(in_image_points),
        "finite_points": int(finite_points),
        "area": float(overlap_area),
    }


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= intersect
        j = i
    return inside


def draw_poly(draw: ImageDraw.ImageDraw, uv: np.ndarray, color: tuple[int, int, int], width: int, label: str | None = None) -> None:
    valid = np.isfinite(uv).all(axis=1)
    if valid.sum() < 3:
        return
    pts = [tuple(map(float, p)) for p in uv[valid]]
    draw.line(pts + [pts[0]], fill=color, width=width)
    if label:
        c = uv[valid].mean(axis=0)
        draw.text((float(c[0]) + 4.0, float(c[1]) + 4.0), label, fill=color)


def bbox_from_uv(arrays: list[np.ndarray], w: int, h: int, pad: int) -> tuple[int, int, int, int] | None:
    pts = []
    for uv in arrays:
        valid = np.isfinite(uv).all(axis=1)
        if valid.any():
            pts.append(uv[valid])
    if not pts:
        return None
    all_pts = np.vstack(pts)
    x0 = max(0, int(math.floor(float(all_pts[:, 0].min()))) - pad)
    y0 = max(0, int(math.floor(float(all_pts[:, 1].min()))) - pad)
    x1 = min(w, int(math.ceil(float(all_pts[:, 0].max()))) + pad)
    y1 = min(h, int(math.ceil(float(all_pts[:, 1].max()))) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def report_rel(path: Path, report_dir: Path) -> str:
    try:
        return path.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def select_top_cases(part2_pack: Path, top_k: int) -> list[dict[str, object]]:
    cases = load_json(part2_pack / "case_cards_expanded.json")["cases"]  # type: ignore[index]
    possible = [c for c in cases if c.get("uncertainty_type") == "possible_free_unconfirmed"]
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    possible.sort(
        key=lambda c: (
            priority_rank.get(str(c.get("priority")), 9),
            -float((c.get("pointcloud_summary") or {}).get("max_ray_free_area_ratio", 0.0)),
            -float((c.get("pointcloud_summary") or {}).get("visibility_accum", 0.0)),
        )
    )
    return possible[:top_k]


def select_cases_by_id(part2_pack: Path, case_ids: list[str]) -> list[dict[str, object]]:
    cases = load_json(part2_pack / "case_cards_expanded.json")["cases"]  # type: ignore[index]
    by_id = {str(case["case_id"]): case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"unknown case ids: {', '.join(missing)}")
    return [by_id[case_id] for case_id in case_ids]


def score_frame_for_case(slot: dict[str, object], frame_id: int, frame_rows: dict[int, dict[str, str]], scale: float, args: argparse.Namespace, repo_root: Path) -> dict[str, object] | None:
    if frame_id not in frame_rows:
        return None
    try:
        points, ego_pose = load_points_for_frame(frame_id, frame_rows, repo_root)
    except FileNotFoundError:
        return None
    ground_z = float(np.quantile(points[:, 2], args.ground_quantile))
    core_uv, _ = polygon_to_uv(np.asarray(slot["core_polygon_map"], dtype=np.float64), ego_pose, scale, ground_z)
    inner_uv, _ = polygon_to_uv(np.asarray(slot["inner_polygon"], dtype=np.float64), ego_pose, scale, ground_z)
    margin_uv, _ = polygon_to_uv(np.asarray(slot["margin_polygon_map"], dtype=np.float64), ego_pose, scale, ground_z)
    quality = projection_quality([core_uv, inner_uv, margin_uv])
    quality["frame_id"] = int(frame_id)
    return quality


def choose_camera_frame(
    case: dict[str, object],
    slot: dict[str, object],
    slot_evidence: dict[str, list[dict[str, object]]],
    frame_rows: dict[int, dict[str, str]],
    scale: float,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, object]:
    slot_id = str(case["slot_id"])
    candidate_ids = []
    for row in slot_evidence.get(slot_id, []):
        candidate_ids.append(int(row["frame_id"]))
    strongest = int(case["strongest_frame_id"])
    if strongest not in candidate_ids:
        candidate_ids.insert(0, strongest)

    best: dict[str, object] | None = None
    for frame_id in candidate_ids:
        quality = score_frame_for_case(slot, frame_id, frame_rows, scale, args, repo_root)
        if quality is None:
            continue
        if best is None or float(quality["score"]) > float(best["score"]):
            best = quality
    if best is None:
        return {"frame_id": strongest, "score": 0.0, "bbox": None, "in_image_points": 0, "finite_points": 0, "area": 0.0}
    return best


def build_case(
    case: dict[str, object],
    slots: dict[str, dict[str, object]],
    slot_evidence: dict[str, list[dict[str, object]]],
    frame_rows: dict[int, dict[str, str]],
    scale: float,
    output_dir: Path,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, object]:
    slot_id = str(case["slot_id"])
    slot = slots[slot_id]
    selected = choose_camera_frame(case, slot, slot_evidence, frame_rows, scale, args, repo_root)
    frame_id = int(selected["frame_id"])
    image_path = Path(str(case["strongest_frame_image_path"]))
    if frame_id in frame_rows:
        image_path = Path(frame_rows[frame_id]["image_path"])
    points, ego_pose = load_points_for_frame(frame_id, frame_rows, repo_root)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    ground_z = float(np.quantile(points[:, 2], args.ground_quantile))

    core_uv, core_valid = polygon_to_uv(np.asarray(slot["core_polygon_map"], dtype=np.float64), ego_pose, scale, ground_z)
    inner_uv, inner_valid = polygon_to_uv(np.asarray(slot["inner_polygon"], dtype=np.float64), ego_pose, scale, ground_z)
    margin_uv, margin_valid = polygon_to_uv(np.asarray(slot["margin_polygon_map"], dtype=np.float64), ego_pose, scale, ground_z)

    obstacle_uv = np.empty((0, 2), dtype=np.float64)
    poly_margin = np.asarray(slot["margin_polygon_map"], dtype=np.float64)
    near = points_in_polygon(points[:, :2], poly_margin)
    obstacle = near & (points[:, 2] >= ground_z + 0.30) & (points[:, 2] <= ground_z + 2.50)
    if obstacle.any():
        xy_lidar = map_xy_to_lidar_xy(points[obstacle, :2], ego_pose, scale)
        xyz_lidar = np.column_stack([xy_lidar, points[obstacle, 2]])
        obstacle_uv, obstacle_valid = project_camera(lidar_to_camera(xyz_lidar))
        inside_img = obstacle_valid & (obstacle_uv[:, 0] >= 0) & (obstacle_uv[:, 0] < w) & (obstacle_uv[:, 1] >= 0) & (obstacle_uv[:, 1] < h)
        obstacle_uv = obstacle_uv[inside_img]

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    draw_poly(draw, margin_uv, (245, 158, 11), 3, "margin")
    draw_poly(draw, inner_uv, (37, 99, 235), 3, "inner")
    draw_poly(draw, core_uv, (22, 163, 74), 4, f"{slot_id} core")
    for p in obstacle_uv[:: max(1, len(obstacle_uv) // 400)]:
        x, y = float(p[0]), float(p[1])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 38, 38))

    bbox = bbox_from_uv([core_uv, inner_uv, margin_uv], w, h, args.crop_pad_px)
    projection_status = "visible_projected"
    if bbox is None:
        projection_status = "not_visible_or_projection_failed"
        bbox = (0, 0, w, h)
    elif not ((np.isfinite(core_uv).all(axis=1).sum() >= 3) or (np.isfinite(inner_uv).all(axis=1).sum() >= 3)):
        projection_status = "partial_projection"

    overlay_path = output_dir / f"{case['case_id']}_overlay.png"
    crop_path = output_dir / f"{case['case_id']}_crop.png"
    overlay.save(overlay_path)
    overlay.crop(bbox).save(crop_path)

    result = {
        "case_id": case["case_id"],
        "slot_id": slot_id,
        "frame_id": frame_id,
        "part1_strongest_frame_id": int(case["strongest_frame_id"]),
        "camera_frame_selection": selected,
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "crop_path": str(crop_path),
        "projection_status": projection_status,
        "visible_polygon": {
            "core_valid_points": int(np.isfinite(core_uv).all(axis=1).sum()),
            "inner_valid_points": int(np.isfinite(inner_uv).all(axis=1).sum()),
            "margin_valid_points": int(np.isfinite(margin_uv).all(axis=1).sum()),
        },
        "crop_bbox": list(map(int, bbox)),
        "projected_obstacle_points": int(len(obstacle_uv)),
        "part1_summary": case.get("pointcloud_summary", {}),
        "manual_visual_notes_placeholder": "",
        "question_for_part2": case.get("question_for_part2", []),
        "expected_agent_outputs": case.get("expected_agent_outputs", []),
    }
    return result


def write_report(output_dir: Path, rows: list[dict[str, object]]) -> None:
    parts = []
    for row in rows:
        overlay = report_rel(Path(str(row["overlay_path"])), output_dir)
        crop = report_rel(Path(str(row["crop_path"])), output_dir)
        summary = row.get("part1_summary", {})
        selection = row.get("camera_frame_selection", {})
        parts.append(
            "<section>"
            f"<h2>{html.escape(str(row['case_id']))} {html.escape(str(row['slot_id']))} frame {row['frame_id']}</h2>"
            f"<p>Status: {html.escape(str(row['projection_status']))} | selected frame: {row['frame_id']} | point-cloud strongest frame: {row.get('part1_strongest_frame_id')} | projection score: {float(selection.get('score', 0.0)):.3f} | obstacle pts projected: {row['projected_obstacle_points']}</p>"
            f"<p>Part1: support={summary.get('support_frame_count')} vis={float(summary.get('visibility_accum',0.0)):.3f} ray={float(summary.get('max_ray_free_area_ratio',0.0)):.3f} obj={float(summary.get('max_object_hit_cell_ratio',0.0)):.3f}</p>"
            f"<div class='grid'><div><h3>Overlay</h3><img src='{html.escape(overlay)}'></div><div><h3>Crop</h3><img src='{html.escape(crop)}'></div></div>"
            f"<p>Questions: {html.escape(' | '.join(row.get('question_for_part2', [])))}</p>"
            "</section>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Top 5 Camera Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ padding: 12px; background: #fff7ed; border: 1px solid #fed7aa; margin-bottom: 16px; }}
    section {{ border-top: 1px solid #d1d5db; padding: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; }}
  </style>
</head>
<body>
  <h1>Top 5 Possible-Free Camera Review</h1>
  <div class="note">This visual review does not change Part 1 free/occupied states. Green=core, blue=inner, orange=margin, red=projected obstacle candidates.</div>
  {''.join(parts)}
</body>
</html>
"""
    (output_dir / "top5_case_review.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    repo_root = Path.cwd()
    slot_db = load_json(args.part1_dir / "slot_database.json")
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}  # type: ignore[index]
    scale = float(slot_db["map_units_per_meter"])  # type: ignore[index]
    frame_rows = load_frames(args.frames)
    top_cases = select_cases_by_id(args.part2_pack, args.case_ids) if args.case_ids else select_top_cases(args.part2_pack, args.top_k)
    slot_ids = {str(case["slot_id"]) for case in top_cases}
    slot_evidence = load_slot_evidence(args.part1_dir / "frame_slot_evidence.jsonl", slot_ids)
    rows = [build_case(case, slots, slot_evidence, frame_rows, scale, args.output_dir, args, repo_root) for case in top_cases]
    write_json(args.output_dir / "top5_case_review.json", {"cases": rows})
    write_report(args.output_dir, rows)
    print(f"[done] top_cases={len(rows)} output={args.output_dir}")
    for row in rows:
        print(row["case_id"], row["slot_id"], row["frame_id"], row["projection_status"], row["crop_path"])


if __name__ == "__main__":
    main()
