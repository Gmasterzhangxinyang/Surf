#!/usr/bin/env python3
"""Build a Part 2 review pack from conservative Part 1 slot-scoring output."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    HAS_MPL = True
except Exception:
    HAS_MPL = False


DEFAULT_PART1 = Path("outputs/part1_slot_scoring_1000_boundary_fixed")
DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_OUTPUT = Path("outputs/part2_case_review_pack")

EXPECTED_AGENT_OUTPUTS = [
    "confirmed_free",
    "confirmed_occupied_target",
    "occupied_adjacent_slot",
    "lane_object_not_slot",
    "static_structure",
    "occluded_unknown",
    "still_ambiguous",
    "camera_not_available",
]

CAMERA_TOOLS = [
    "project_slot_to_camera",
    "crop_slot_region",
    "inspect_adjacent_slots",
    "detect_vehicle_in_crop",
    "inspect_temporal_camera_frames",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Part 2 case review pack")
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-frame-window", type=int, default=25)
    parser.add_argument("--group-distance-m", type=float, default=2.0)
    parser.add_argument("--max-debug-assets", type=int, default=140)
    parser.add_argument("--no-debug-plots", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_frames(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="") as handle:
        return {int(row["frame"]): row for row in csv.DictReader(handle)}


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


def evidence_strength(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row.get("core_hit_cell_ratio", row.get("object_hit_cell_ratio", 0.0))),
        float(row.get("core_obstacle_point_count", row.get("obstacle_point_count_inner", 0))),
        float(row.get("visibility_score", 0.0)),
        -float(row.get("boundary_ratio", 0.0)),
    )


def build_evidence_index(rows: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_slot: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_slot[str(row["slot_id"])].append(row)
    return by_slot


def strongest_evidence(slot_id: str, by_slot: dict[str, list[dict[str, object]]]) -> dict[str, object] | None:
    rows = by_slot.get(slot_id, [])
    if not rows:
        return None
    return max(rows, key=evidence_strength)


def slot_map(slot_db: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(slot["slot_id"]): slot for slot in slot_db["slots"]}  # type: ignore[index]


def slot_center(slot: dict[str, object]) -> np.ndarray:
    return np.asarray(slot["center_map"], dtype=np.float64)


def nearest_image_for_frame(frame_id: int | None, frame_rows: dict[int, dict[str, str]]) -> str:
    if frame_id is None:
        return ""
    row = frame_rows.get(frame_id)
    if not row:
        return ""
    path = row.get("image_path", "")
    return path if path and Path(path).exists() else ""


def frame_points_path(frame_id: int | None, frame_rows: dict[int, dict[str, str]], repo_root: Path) -> Path | None:
    if frame_id is None:
        return None
    row = frame_rows.get(frame_id)
    if not row:
        return None
    path = Path(row.get("map_points_path", ""))
    if not path:
        return None
    if not path.is_absolute():
        path = repo_root / path
    return path if path.exists() else None


def expand_case(
    case: dict[str, object],
    slots: dict[str, dict[str, object]],
    by_slot: dict[str, list[dict[str, object]]],
    frame_rows: dict[int, dict[str, str]],
    repo_root: Path,
) -> dict[str, object]:
    slot_id = str(case["slot_id"])
    slot = slots[slot_id]
    ev = strongest_evidence(slot_id, by_slot)
    frame_id = int(ev["frame_id"]) if ev else None
    substate = str(case.get("part1_substate", case.get("uncertainty_type", "")))
    questions = list(case.get("question_for_part2", []))
    if substate in {"occupied_boundary_conflict", "adjacent_slot_conflict", "boundary_conflict"}:
        questions = [
            "Which slot does the obstacle belong to?",
            "Is the object a parked vehicle, passing vehicle, or static structure?",
            "Is the target slot actually free, occupied, or not visible?",
        ]
    elif substate == "possible_free_unconfirmed":
        questions = [
            "Is the target slot visibly empty?",
            "Are parking lines or arresters visible enough to confirm the slot extent?",
            "Is there any vehicle, static structure, or occlusion inside the target slot?",
        ]
    expanded = {
        **case,
        "case_id": str(case["case_id"]),
        "slot_id": slot_id,
        "adjacent_slots": slot.get("adjacent_slots", []),
        "uncertainty_type": substate,
        "pointcloud_evidence_summary": case.get("pointcloud_summary", {}),
        "strongest_frame_id": frame_id,
        "strongest_frame_image_path": nearest_image_for_frame(frame_id, frame_rows),
        "debug_bev_path": "",
        "why_part1_cannot_decide": case.get("why_part1_cannot_decide", []),
        "question_for_part2": questions,
        "recommended_camera_tools": CAMERA_TOOLS,
        "expected_agent_outputs": EXPECTED_AGENT_OUTPUTS,
    }
    if ev:
        expanded["strongest_frame_evidence"] = {
            "visibility_score": ev.get("visibility_score", 0.0),
            "ray_free_ratio": ev.get("ray_free_ratio", 0.0),
            "core_obstacle_point_count": ev.get("core_obstacle_point_count", 0),
            "core_hit_cell_ratio": ev.get("core_hit_cell_ratio", 0.0),
            "edge_obstacle_point_count": ev.get("edge_obstacle_point_count", 0),
            "margin_obstacle_point_count": ev.get("margin_obstacle_point_count", 0),
            "boundary_ratio": ev.get("boundary_ratio", 0.0),
            "adjacent_overlap_ratio": ev.get("adjacent_overlap_ratio", 0.0),
            "cluster_top1_slot": ev.get("cluster_top1_slot"),
            "cluster_top1_overlap": ev.get("cluster_top1_overlap", 0),
            "cluster_top2_slot": ev.get("cluster_top2_slot"),
            "cluster_top2_overlap": ev.get("cluster_top2_overlap", 0),
            "cluster_ownership_status": ev.get("cluster_ownership_status", "none"),
        }
    return expanded


def copy_camera_thumbnail(case: dict[str, object], assets_dir: Path) -> str:
    image_path = str(case.get("strongest_frame_image_path", ""))
    if not image_path:
        return ""
    src = Path(image_path)
    if not src.exists():
        return ""
    dst = assets_dir / f"{case['case_id']}_camera{src.suffix.lower() or '.png'}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst)


def draw_case_bev(
    case: dict[str, object],
    slots: dict[str, dict[str, object]],
    frame_rows: dict[int, dict[str, str]],
    repo_root: Path,
    assets_dir: Path,
) -> str:
    if not HAS_MPL:
        return ""
    slot_id = str(case["slot_id"])
    slot = slots.get(slot_id)
    if not slot:
        return ""
    frame_id = case.get("strongest_frame_id")
    points_xy = np.empty((0, 2), dtype=np.float64)
    points_z = np.empty((0,), dtype=np.float64)
    points_path = frame_points_path(int(frame_id) if frame_id is not None else None, frame_rows, repo_root)
    if points_path:
        data = np.load(points_path)
        pts = data["points_map_xyzi"]
        points_xy = pts[:, :2].astype(np.float64)
        points_z = pts[:, 2].astype(np.float64)

    target_poly = np.asarray(slot["polygon_map"], dtype=np.float64)
    all_polys = [target_poly]
    for adj in slot.get("adjacent_slots", [])[:8]:
        if adj in slots:
            all_polys.append(np.asarray(slots[adj]["polygon_map"], dtype=np.float64))
    stacked = np.vstack(all_polys)
    bmin = stacked.min(axis=0) - 0.35
    bmax = stacked.max(axis=0) + 0.35
    mask = np.zeros(len(points_xy), dtype=bool)
    if len(points_xy):
        mask = np.all((points_xy >= bmin) & (points_xy <= bmax), axis=1)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=170)
    if mask.any():
        z = points_z[mask]
        obstacle = z > np.quantile(points_z, 0.08) + 0.30 if len(points_z) else np.zeros(mask.sum(), dtype=bool)
        local = points_xy[mask]
        ax.scatter(local[~obstacle, 0], local[~obstacle, 1], s=1, c="#94a3b8", alpha=0.35, label="low/free hit")
        ax.scatter(local[obstacle, 0], local[obstacle, 1], s=2, c="#dc2626", alpha=0.65, label="obstacle candidate")
    for adj in slot.get("adjacent_slots", [])[:8]:
        if adj not in slots:
            continue
        poly = np.asarray(slots[adj]["polygon_map"], dtype=np.float64)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="#e5e7eb", edgecolor="#64748b", alpha=0.18, linewidth=0.8))
        c = slot_center(slots[adj])
        ax.text(c[0], c[1], adj.replace("slot_", ""), fontsize=5, color="#475569", ha="center", va="center")
    ax.add_patch(MplPolygon(np.asarray(slot["margin_polygon_map"], dtype=np.float64), closed=True, facecolor="none", edgecolor="#f59e0b", linewidth=1.0, linestyle="--", label="margin"))
    ax.add_patch(MplPolygon(np.asarray(slot["inner_polygon"], dtype=np.float64), closed=True, facecolor="none", edgecolor="#2563eb", linewidth=1.0, label="inner"))
    ax.add_patch(MplPolygon(np.asarray(slot["core_polygon_map"], dtype=np.float64), closed=True, facecolor="#22c55e", edgecolor="#166534", alpha=0.20, linewidth=1.2, label="core"))
    c = slot_center(slot)
    ax.text(c[0], c[1], slot_id.replace("slot_", ""), fontsize=7, color="#111827", ha="center", va="center", weight="bold")
    ax.set_xlim(bmin[0], bmax[0])
    ax.set_ylim(bmin[1], bmax[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{case['case_id']} {slot_id} frame={frame_id} {case.get('part1_substate')}")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    dst = assets_dir / f"{case['case_id']}_bev.png"
    fig.savefig(dst)
    plt.close(fig)
    return str(dst)


def case_connects(a: dict[str, object], b: dict[str, object], slots: dict[str, dict[str, object]], frame_window: int, dist_thresh: float, scale: float) -> bool:
    slot_a = str(a["slot_id"])
    slot_b = str(b["slot_id"])
    if slot_a == slot_b:
        return True
    sub_a = str(a["part1_substate"])
    sub_b = str(b["part1_substate"])
    boundary_like = {"occupied_boundary_conflict", "adjacent_slot_conflict", "boundary_conflict", "static_vs_vehicle", "slot_vs_lane"}
    adj = slot_b in slots[slot_a].get("adjacent_slots", []) or slot_a in slots[slot_b].get("adjacent_slots", [])
    frame_a = a.get("strongest_frame_id")
    frame_b = b.get("strongest_frame_id")
    frame_close = frame_a is None or frame_b is None or abs(int(frame_a) - int(frame_b)) <= frame_window
    dist = float(np.linalg.norm(slot_center(slots[slot_a]) - slot_center(slots[slot_b]))) / max(scale, 1e-9)
    if sub_a in boundary_like and sub_b in boundary_like:
        return adj and frame_close
    if sub_a == "possible_free_unconfirmed" and sub_b == "possible_free_unconfirmed":
        return dist <= dist_thresh and frame_close
    return False


def build_case_groups(cases: list[dict[str, object]], slots: dict[str, dict[str, object]], scale: float, frame_window: int, dist_thresh: float) -> list[dict[str, object]]:
    n = len(cases)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if case_connects(cases[i], cases[j], slots, frame_window, dist_thresh, scale):
                graph[i].append(j)
                graph[j].append(i)
    seen = [False] * n
    groups: list[dict[str, object]] = []
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        ids = []
        while q:
            cur = q.popleft()
            ids.append(cur)
            for nxt in graph[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        members = [cases[k] for k in ids]
        sub_counts = Counter(str(c["part1_substate"]) for c in members)
        dominant = sub_counts.most_common(1)[0][0]
        priority = min((str(c["priority"]) for c in members), key=lambda p: priority_rank.get(p, 9))
        strongest = sorted({int(c["strongest_frame_id"]) for c in members if c.get("strongest_frame_id") is not None})
        slots_in_group = sorted({str(c["slot_id"]) for c in members})
        questions = [
            "Which slot does the obstacle belong to?",
            "Is the object a parked vehicle, passing vehicle, or static structure?",
            "Is the target slot actually free, occupied, or not visible?",
        ]
        if dominant == "possible_free_unconfirmed":
            questions = [
                "Is the target slot actually free?",
                "Is the apparent free-space evidence sufficient in camera view?",
                "Is the slot occluded or not visible?",
            ]
        groups.append(
            {
                "case_group_id": "",
                "slots": slots_in_group,
                "case_ids": [str(c["case_id"]) for c in members],
                "dominant_uncertainty": dominant,
                "priority": priority,
                "strongest_frame_ids": strongest[:8],
                "pointcloud_summary": summarize_group(members),
                "question_for_part2": questions,
                "recommended_tools": CAMERA_TOOLS,
                "debug_bev_path": next((str(c.get("debug_bev_path")) for c in members if c.get("debug_bev_path")), ""),
                "camera_image_path": next((str(c.get("strongest_frame_image_path")) for c in members if c.get("strongest_frame_image_path")), ""),
            }
        )
    groups.sort(key=lambda g: (priority_rank.get(str(g["priority"]), 9), -len(g["slots"]), str(g["dominant_uncertainty"]), g["slots"][0]))
    for idx, group in enumerate(groups, 1):
        group["case_group_id"] = f"group_{idx:03d}"
    return groups


def summarize_group(members: list[dict[str, object]]) -> dict[str, object]:
    summaries = [m.get("pointcloud_summary") or m.get("pointcloud_evidence_summary") or {} for m in members]
    return {
        "case_count": len(members),
        "max_support_frame_count": max((int(s.get("support_frame_count", 0)) for s in summaries), default=0),
        "max_visibility_accum": max((float(s.get("visibility_accum", 0.0)) for s in summaries), default=0.0),
        "max_ray_free_area_ratio": max((float(s.get("max_ray_free_area_ratio", 0.0)) for s in summaries), default=0.0),
        "max_object_hit_cell_ratio": max((float(s.get("max_object_hit_cell_ratio", 0.0)) for s in summaries), default=0.0),
        "min_distance_to_slot_m": min((float(s.get("min_distance_to_slot_m", float("inf"))) for s in summaries), default=0.0),
        "max_conflict_score": max((float(s.get("conflict_score", 0.0)) for s in summaries), default=0.0),
    }


def core_owned_analysis(part1_dir: Path, slots: dict[str, dict[str, object]], scores: dict[str, dict[str, object]]) -> dict[str, object]:
    clusters = load_jsonl(part1_dir / "cluster_ownership_debug.jsonl")
    clear = [row for row in clusters if row.get("cluster_ownership_status") == "clear_core_owned"]
    slot_ids = {str(row["top1_slot"]) for row in clear}
    failure_counts: Counter[str] = Counter()
    possible: list[dict[str, object]] = []
    for slot_id in sorted(slot_ids):
        score = scores.get(slot_id)
        if not score or score.get("state") == "occupied":
            continue
        ev = score.get("evidence", {})
        reasons = []
        if int(ev.get("clear_cluster_ownership_frames", 0)) < 2 or int(ev.get("occupied_support_frames", 0)) < 2:
            reasons.append("insufficient_temporal_support")
        if float(ev.get("max_core_hit_cell_ratio", 0.0)) < 0.08:
            reasons.append("core_hit_ratio_too_low")
        if float(ev.get("max_boundary_ratio", 0.0)) > 0.45:
            reasons.append("boundary_ratio_too_high")
        if float(ev.get("max_adjacent_overlap_ratio", 0.0)) > 0.30:
            reasons.append("adjacent_competition")
        if float(ev.get("min_distance_to_slot", 999.0)) > 19.8:
            reasons.append("distance_too_high")
        cluster_counts = ev.get("cluster_status_counts", {}) or {}
        if int(cluster_counts.get("slot_vs_lane_possible", 0)) > 0:
            reasons.append("static_lane_possibility")
        if not reasons:
            reasons.append("ambiguous_after_boundary_validation")
        failure_counts.update(reasons)
        possible.append(
            {
                "slot_id": slot_id,
                "state": score.get("state"),
                "failure_reasons": reasons,
                "support_frame_count": ev.get("support_frame_count", 0),
                "clear_cluster_ownership_frames": ev.get("clear_cluster_ownership_frames", 0),
                "occupied_support_frames": ev.get("occupied_support_frames", 0),
                "max_core_hit_cell_ratio": ev.get("max_core_hit_cell_ratio", 0.0),
                "max_core_obstacle_point_count": ev.get("max_core_obstacle_point_count", 0),
                "max_boundary_ratio": ev.get("max_boundary_ratio", 0.0),
                "max_adjacent_overlap_ratio": ev.get("max_adjacent_overlap_ratio", 0.0),
                "min_distance_to_slot_m": ev.get("min_distance_to_slot", 0.0),
            }
        )
    possible.sort(key=lambda r: (-int(r["clear_cluster_ownership_frames"]), -float(r["max_core_hit_cell_ratio"]), -int(r["max_core_obstacle_point_count"])))
    return {
        "clear_core_owned_cluster_count": len(clear),
        "clear_core_owned_slot_count": len(slot_ids),
        "failure_gate_counts": dict(failure_counts),
        "possible_occupied_unconfirmed_count": len(possible),
        "top_20_possible_occupied_unconfirmed": possible[:20],
    }


def write_index_csv(path: Path, cases: list[dict[str, object]], group_by_case: dict[str, str]) -> None:
    fields = [
        "case_id",
        "case_group_id",
        "slot_id",
        "adjacent_slots",
        "part1_state",
        "uncertainty_type",
        "priority",
        "strongest_frame_id",
        "strongest_frame_image_path",
        "debug_bev_path",
        "expected_agent_outputs",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "case_group_id": group_by_case.get(str(case["case_id"]), ""),
                    "slot_id": case["slot_id"],
                    "adjacent_slots": "|".join(case.get("adjacent_slots", [])),
                    "part1_state": case["part1_state"],
                    "uncertainty_type": case["uncertainty_type"],
                    "priority": case["priority"],
                    "strongest_frame_id": case.get("strongest_frame_id", ""),
                    "strongest_frame_image_path": case.get("strongest_frame_image_path", ""),
                    "debug_bev_path": case.get("debug_bev_path", ""),
                    "expected_agent_outputs": "|".join(case.get("expected_agent_outputs", [])),
                }
            )


def write_report(path: Path, groups: list[dict[str, object]], cases: list[dict[str, object]], core_analysis: dict[str, object]) -> None:
    sub_counts = Counter(str(c["uncertainty_type"]) for c in cases)
    high_groups = [g for g in groups if g["priority"] == "high"]
    top_groups = groups[:20]

    def esc(v: object) -> str:
        return html.escape(str(v))

    def report_rel(v: object) -> str:
        if not v:
            return ""
        raw = Path(str(v))
        try:
            if raw.is_absolute():
                return raw.relative_to(path.parent).as_posix()
            if raw.exists():
                return raw.relative_to(path.parent).as_posix()
            return raw.as_posix()
        except ValueError:
            try:
                return raw.resolve().relative_to(path.parent.resolve()).as_posix()
            except ValueError:
                return raw.as_posix()

    rows = []
    for group in top_groups:
        summary = group["pointcloud_summary"]
        img_src = report_rel(group.get("debug_bev_path"))
        img = f'<img src="{esc(img_src)}" alt="bev" loading="lazy">' if img_src else ""
        camera = f'<div class="path">{esc(group["camera_image_path"])}</div>' if group.get("camera_image_path") else "not available"
        rows.append(
            "<tr>"
            f"<td>{esc(group['case_group_id'])}</td>"
            f"<td>{esc(', '.join(group['slots']))}</td>"
            f"<td>{esc(group['dominant_uncertainty'])}</td>"
            f"<td>{esc(group['priority'])}</td>"
            f"<td>{esc(group['strongest_frame_ids'])}</td>"
            f"<td>cases={summary['case_count']} vis={float(summary['max_visibility_accum']):.2f} ray={float(summary['max_ray_free_area_ratio']):.2f} obj={float(summary['max_object_hit_cell_ratio']):.2f}</td>"
            f"<td>{img}</td>"
            f"<td>{camera}</td>"
            f"<td>{esc(' | '.join(group['recommended_tools']))}</td>"
            f"<td>{esc(' | '.join(group['question_for_part2']))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Part 2 Case Review Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 260px; border: 1px solid #d1d5db; }}
    .path {{ max-width: 300px; overflow-wrap: anywhere; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 10px; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>Part 2 Case Review Report</h1>
  <div class="note">This pack does not change Part 1 final states. possible_free_unconfirmed and possible_occupied_unconfirmed are review tasks only.</div>
  <p>Total cases: {len(cases)} | case groups: {len(groups)} | high priority groups: {len(high_groups)}</p>
  <p>possible_free_unconfirmed: {sub_counts['possible_free_unconfirmed']} | boundary conflict: {sub_counts['occupied_boundary_conflict'] + sub_counts['boundary_conflict']} | adjacent conflict: {sub_counts['adjacent_slot_conflict']}</p>
  <p>possible_occupied_unconfirmed: {core_analysis['possible_occupied_unconfirmed_count']} | clear_core_owned clusters: {core_analysis['clear_core_owned_cluster_count']} | clear_core_owned slots: {core_analysis['clear_core_owned_slot_count']}</p>
  <h2>Top 20 High Priority / Review Groups</h2>
  <table>
    <tr><th>group</th><th>slots</th><th>uncertainty</th><th>priority</th><th>frames</th><th>point cloud summary</th><th>debug BEV</th><th>camera image</th><th>tools</th><th>questions</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    assets_dir = args.output_dir / "debug_assets"
    ensure_dir(assets_dir)
    repo_root = Path.cwd()

    part1_dir = args.part1_dir
    slot_db = load_json(part1_dir / "slot_database.json")
    slots = slot_map(slot_db)  # type: ignore[arg-type]
    scale = float(slot_db["map_units_per_meter"])  # type: ignore[index]
    raw_cases = load_json(part1_dir / "part2_candidate_cases.json")["cases"]  # type: ignore[index]
    evidence_rows = load_jsonl(part1_dir / "frame_slot_evidence.jsonl")
    by_slot = build_evidence_index(evidence_rows)
    frame_rows = load_frames(args.frames)
    scores_data = load_json(part1_dir / "slot_belief_fused.json")["slots"]  # type: ignore[index]
    scores = {str(row["slot_id"]): row for row in scores_data}

    expanded = [expand_case(case, slots, by_slot, frame_rows, repo_root) for case in raw_cases]
    for case in expanded[: args.max_debug_assets]:
        if not args.no_debug_plots:
            case["debug_bev_path"] = draw_case_bev(case, slots, frame_rows, repo_root, assets_dir)
        case["camera_debug_copy_path"] = copy_camera_thumbnail(case, assets_dir)

    groups = build_case_groups(expanded, slots, scale, args.group_frame_window, args.group_distance_m)
    group_by_case = {}
    for group in groups:
        for case_id in group["case_ids"]:
            group_by_case[str(case_id)] = str(group["case_group_id"])
    core_analysis = core_owned_analysis(part1_dir, slots, scores)

    write_json(args.output_dir / "case_groups.json", {"summary": summarize_groups(groups), "case_groups": groups})
    write_json(args.output_dir / "case_cards_expanded.json", {"cases": expanded})
    write_index_csv(args.output_dir / "case_review_index.csv", expanded, group_by_case)
    write_json(args.output_dir / "core_owned_but_not_occupied_analysis.json", core_analysis)
    write_report(args.output_dir / "case_review_report.html", groups, expanded, core_analysis)

    sub_counts = Counter(str(case["uncertainty_type"]) for case in expanded)
    print(f"[done] cases={len(expanded)} groups={len(groups)} high_priority_groups={sum(1 for g in groups if g['priority'] == 'high')}")
    print(
        "[summary] "
        f"possible_free_unconfirmed={sub_counts['possible_free_unconfirmed']} "
        f"boundary_conflict={sub_counts['occupied_boundary_conflict'] + sub_counts['boundary_conflict']} "
        f"adjacent_conflict={sub_counts['adjacent_slot_conflict']} "
        f"possible_occupied_unconfirmed={core_analysis['possible_occupied_unconfirmed_count']}"
    )
    print(f"[output] {args.output_dir}")


def summarize_groups(groups: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_group_count": len(groups),
        "high_priority_group_count": sum(1 for g in groups if g["priority"] == "high"),
        "dominant_uncertainty_counts": dict(Counter(str(g["dominant_uncertainty"]) for g in groups)),
        "priority_counts": dict(Counter(str(g["priority"]) for g in groups)),
    }


if __name__ == "__main__":
    main()
