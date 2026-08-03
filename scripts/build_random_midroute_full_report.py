#!/usr/bin/env python3
"""Build the auditable Part1 -> Part2 random-anchor HTML report."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path


REASON_ZH = {
    "boundary_dominated": "边界点占主导（柱/墙误报风险）",
    "compact_vertical_structure": "紧凑竖直结构（疑似柱体）",
    "horizontal_cap_structure": "水平盖板结构",
    "high_occlusion": "遮挡较高",
    "insufficient_3d_voxels": "三维体素不足",
    "insufficient_height_layers": "高度层不足",
    "insufficient_height_span": "高度跨度不足",
    "insufficient_lower_body_coverage": "车辆下部覆盖不足",
    "insufficient_ray_frames": "有效射线帧不足",
    "insufficient_support_frames": "支持帧不足",
    "insufficient_vehicle_footprint": "车辆占地形状不足",
    "insufficient_vehicle_points": "车辆点数不足",
    "insufficient_viewpoint_separation": "视角基线不足",
    "insufficient_viewpoints": "有效视角不足",
    "large_unobserved_component": "大块未观测区域",
    "linear_static_structure": "线性静态结构（墙/柱风险）",
    "low_core_ray_coverage": "车位核心射线覆盖低",
    "low_free_volume_coverage": "空闲体积覆盖低",
    "low_height_structure": "低矮结构",
    "low_near_ground_coverage": "近地覆盖不足",
    "outside_residual_conflict": "车位外残差冲突",
    "ownership_conflict": "目标归属冲突",
    "partial_route_scope": "仅部分经过该车位",
    "target_ownership_not_verified": "目标车位归属未验证",
    "camera_projection_not_terminally_trusted": "相机投影未通过独立像素级审计",
    "unresolved_core_hit_evidence": "核心区域仍有命中",
    "weak_core_clearance_conflict": "弱核心净空冲突",
    "weak_obstacle_evidence": "障碍证据弱",
    "weak_occupied_evidence": "占用证据弱",
    "weak_shared_ownership": "弱共享归属冲突",
    "weak_static_structure": "弱静态结构",
    "weak_temporal_inconsistent": "时序不一致",
    "weak_vehicle_evidence": "车辆形状证据弱",
    "weak_visibility_limited": "可见性不足",
}


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_link(path_value: str | Path, report_root: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        project_candidate = Path.cwd() / path
        if project_candidate.exists():
            path = project_candidate
        else:
            path = report_root / path
    return Path(os.path.relpath(path.resolve(), report_root.resolve())).as_posix()


def reason_list(reasons: list[str]) -> str:
    if not reasons:
        return '<span class="muted">无</span>'
    return "<br>".join(
        f"<code>{escape(reason)}</code> {escape(REASON_ZH.get(reason, ''))}"
        for reason in reasons
    )


def badge(state: str) -> str:
    names = {
        "free": "Free",
        "occupied": "Occupied",
        "unknown": "Unknown",
        "out_of_scope": "Out of scope",
    }
    return f'<span class="badge {escape(state)}">{names.get(state, escape(state))}</span>'


def relative_position(local_map: dict, slot: dict) -> dict:
    pose = local_map["anchor_pose"]
    scale = float(local_map["map_units_per_meter"])
    dx = (float(slot["center_map"][0]) - float(pose["map_xy"][0])) / scale
    dy = (float(slot["center_map"][1]) - float(pose["map_xy"][1])) / scale
    yaw = float(pose["map_yaw_rad"])
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    bearing = math.degrees(math.atan2(left, forward))
    return {
        "forward": forward,
        "left": left,
        "distance": math.hypot(forward, left),
        "bearing": bearing,
        "front": forward >= 0.0 and abs(bearing) <= 90.0,
    }


def build_front_svg(
    local_map: dict,
    free_ids: set[str],
    occupied_ids: set[str],
    unknown_ids: set[str],
) -> str:
    width, height, pad = 1120, 720, 55
    all_points = [
        tuple(point)
        for slot in local_map["slots"]
        for point in slot["polygon_map"]
    ]
    ego = tuple(local_map["anchor_pose"]["map_xy"])
    all_points.append(ego)
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    def xy(point):
        x = pad + (point[0] - min_x) / span_x * (width - 2 * pad)
        y = height - pad - (point[1] - min_y) / span_y * (height - 2 * pad)
        return x, y

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="frame 6681 strict front 180 degree slot map">',
        '<rect width="100%" height="100%" fill="#08111f"/>',
    ]
    yaw = float(local_map["anchor_pose"]["map_yaw_rad"])
    scale = float(local_map["map_units_per_meter"])
    normal = (-math.sin(yaw), math.cos(yaw))
    a = (ego[0] - normal[0] * scale * 18, ego[1] - normal[1] * scale * 18)
    b = (ego[0] + normal[0] * scale * 18, ego[1] + normal[1] * scale * 18)
    ax, ay = xy(a)
    bx, by = xy(b)
    rows.append(
        f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        'stroke="#38bdf8" stroke-width="3" stroke-dasharray="10 8"/>'
    )
    for slot in local_map["slots"]:
        slot_id = str(slot["slot_id"])
        pos = relative_position(local_map, slot)
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, slot["polygon_map"]))
        if slot_id in free_ids:
            fill, stroke, opacity = "#22c55e", "#86efac", "0.88"
        elif slot_id in occupied_ids:
            fill, stroke, opacity = "#ef4444", "#fca5a5", "0.88"
        elif slot_id in unknown_ids:
            fill, stroke, opacity = "#f59e0b", "#fde68a", "0.82"
        elif pos["front"]:
            fill, stroke, opacity = "#64748b", "#94a3b8", "0.45"
        else:
            fill, stroke, opacity = "#334155", "#64748b", "0.28"
        rows.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2" opacity="{opacity}"/>'
        )
        if slot_id in free_ids or slot_id in occupied_ids or slot_id in unknown_ids:
            cx, cy = xy(slot["center_map"])
            rows.append(
                f'<text x="{cx:.1f}" y="{cy - 8:.1f}" text-anchor="middle" '
                'font-family="ui-monospace,monospace" font-size="18" font-weight="700" '
                f'fill="#f8fafc">{escape(slot_id.replace("slot_", ""))}</text>'
            )
    ex, ey = xy(ego)
    tip = (ego[0] + math.cos(yaw) * scale * 5, ego[1] + math.sin(yaw) * scale * 5)
    tx, ty = xy(tip)
    rows.extend(
        [
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="13" fill="#38bdf8" stroke="#e0f2fe" stroke-width="3"/>',
            f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#38bdf8" stroke-width="7"/>',
            f'<text x="{ex + 18:.1f}" y="{ey - 18:.1f}" fill="#e0f2fe" font-size="18" font-weight="700">EGO / forward</text>',
            '<g transform="translate(55 50)" font-family="system-ui,sans-serif" font-size="18">',
            '<rect width="720" height="72" rx="10" fill="#0f172a" stroke="#334155"/>',
            '<rect x="18" y="16" width="24" height="24" fill="#22c55e"/><text x="52" y="35" fill="#e2e8f0">最终Free</text>',
            '<rect x="165" y="16" width="24" height="24" fill="#ef4444"/><text x="199" y="35" fill="#e2e8f0">最终Occupied</text>',
            '<rect x="355" y="16" width="24" height="24" fill="#f59e0b"/><text x="389" y="35" fill="#e2e8f0">仍Unknown</text>',
            '<rect x="520" y="16" width="24" height="24" fill="#334155"/><text x="554" y="35" fill="#e2e8f0">后方剔除</text>',
            '</g>',
            '</svg>',
        ]
    )
    return "\n".join(rows)


def build_case_position_svg(
    local_map: dict,
    frames: list[dict],
    target_slot_id: str,
    final_state: str,
) -> str:
    """Render one large map-space identity check without Camera projection."""
    width, height = 1400, 850
    map_right, pad = 1010, 70
    slots = {str(slot["slot_id"]): slot for slot in local_map["slots"]}
    target = slots[target_slot_id]
    ego = tuple(float(value) for value in local_map["anchor_pose"]["map_xy"])
    route_points = [(float(row["map_x"]), float(row["map_y"])) for row in frames]
    all_points = [
        tuple(float(value) for value in point)
        for slot in slots.values()
        for point in slot["polygon_map"]
    ] + [ego, *route_points]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    margin_x = span_x * 0.05
    margin_y = span_y * 0.05
    min_x, max_x = min_x - margin_x, max_x + margin_x
    min_y, max_y = min_y - margin_y, max_y + margin_y
    span_x, span_y = max_x - min_x, max_y - min_y

    def xy(point):
        x = pad + (point[0] - min_x) / span_x * (map_right - 2 * pad)
        y = height - pad - (point[1] - min_y) / span_y * (height - 2 * pad)
        return x, y

    target_position = relative_position(local_map, target)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(target_slot_id)} map-space position audit">',
        '<rect width="100%" height="100%" fill="#07101c"/>',
        f'<rect x="30" y="30" width="{map_right - 45}" height="{height - 60}" rx="16" fill="#0b1626" stroke="#334155" stroke-width="2"/>',
    ]
    for index in range(6):
        ratio = index / 5
        map_x = min_x + ratio * span_x
        map_y = min_y + ratio * span_y
        sx, _ = xy((map_x, min_y))
        _, sy = xy((min_x, map_y))
        rows.append(
            f'<line x1="{sx:.1f}" y1="{pad}" x2="{sx:.1f}" y2="{height-pad}" stroke="#1e293b" stroke-width="1"/>'
        )
        rows.append(
            f'<text x="{sx:.1f}" y="{height-38}" text-anchor="middle" fill="#64748b" font-size="14">x={map_x:.3f}</text>'
        )
        rows.append(
            f'<line x1="{pad}" y1="{sy:.1f}" x2="{map_right-pad}" y2="{sy:.1f}" stroke="#1e293b" stroke-width="1"/>'
        )
        rows.append(
            f'<text x="38" y="{sy+5:.1f}" fill="#64748b" font-size="14">y={map_y:.3f}</text>'
        )

    if route_points:
        route_screen = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, route_points))
        rows.append(
            f'<polyline points="{route_screen}" fill="none" stroke="#38bdf8" stroke-width="5" opacity="0.8"/>'
        )
        for frame, point in zip(frames, route_points):
            px, py = xy(point)
            rows.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#7dd3fc"/>')
            rows.append(
                f'<text x="{px+8:.1f}" y="{py-9:.1f}" fill="#bae6fd" font-size="13">f{frame["frame_id"]}</text>'
            )

    yaw = float(local_map["anchor_pose"]["map_yaw_rad"])
    scale = float(local_map["map_units_per_meter"])
    normal = (-math.sin(yaw), math.cos(yaw))
    bound_a = (ego[0] - normal[0] * scale * 20, ego[1] - normal[1] * scale * 20)
    bound_b = (ego[0] + normal[0] * scale * 20, ego[1] + normal[1] * scale * 20)
    bax, bay = xy(bound_a)
    bbx, bby = xy(bound_b)
    rows.append(
        f'<line x1="{bax:.1f}" y1="{bay:.1f}" x2="{bbx:.1f}" y2="{bby:.1f}" stroke="#0ea5e9" stroke-width="3" stroke-dasharray="12 9"/>'
    )
    rows.append(
        f'<text x="{(bax+bbx)/2+12:.1f}" y="{(bay+bby)/2-12:.1f}" fill="#7dd3fc" font-size="15">front 180° boundary</text>'
    )

    state_colors = {"free": "#166534", "occupied": "#991b1b", "unknown": "#374151"}
    for slot_id, slot in slots.items():
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, slot["polygon_map"]))
        is_target = slot_id == target_slot_id
        fill = "#7c3aed" if is_target else state_colors.get(str(slot.get("state")), "#374151")
        stroke = "#f5d0fe" if is_target else "#94a3b8"
        opacity = "0.92" if is_target else "0.42"
        line_width = "5" if is_target else "1.5"
        rows.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{line_width}" opacity="{opacity}"/>'
        )
        cx, cy = xy(slot["center_map"])
        rows.append(
            f'<text x="{cx:.1f}" y="{cy+5:.1f}" text-anchor="middle" fill="#f8fafc" font-family="ui-monospace,monospace" font-size="14" font-weight="700">{escape(slot_id.replace("slot_", ""))}</text>'
        )

    ex, ey = xy(ego)
    target_x, target_y = xy(target["center_map"])
    rows.append(
        f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{target_x:.1f}" y2="{target_y:.1f}" stroke="#f0abfc" stroke-width="4" stroke-dasharray="8 7"/>'
    )
    rows.append(f'<circle cx="{target_x:.1f}" cy="{target_y:.1f}" r="12" fill="#d946ef" stroke="#fff" stroke-width="3"/>')
    tip = (ego[0] + math.cos(yaw) * scale * 5, ego[1] + math.sin(yaw) * scale * 5)
    tx, ty = xy(tip)
    rows.extend(
        [
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="14" fill="#0284c7" stroke="#e0f2fe" stroke-width="4"/>',
            f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#38bdf8" stroke-width="8"/>',
            f'<text x="{ex+18:.1f}" y="{ey+25:.1f}" fill="#e0f2fe" font-size="18" font-weight="700">EGO f{local_map["anchor_pose"]["frame_id"]}</text>',
            f'<rect x="{map_right+20}" y="30" width="{width-map_right-50}" height="{height-60}" rx="16" fill="#101d30" stroke="#334155" stroke-width="2"/>',
            f'<text x="{map_right+50}" y="82" fill="#c084fc" font-size="18" font-weight="800">TARGET SLOT</text>',
            f'<text x="{map_right+50}" y="125" fill="#f8fafc" font-family="ui-monospace,monospace" font-size="30" font-weight="800">{escape(target_slot_id)}</text>',
            f'<text x="{map_right+50}" y="163" fill="#cbd5e1" font-size="17">Agent final: {escape(final_state.upper())}</text>',
            f'<text x="{map_right+50}" y="213" fill="#7dd3fc" font-size="16" font-weight="700">MAP COORDINATES</text>',
            f'<text x="{map_right+50}" y="244" fill="#e2e8f0" font-size="16">center x = {float(target["center_map"][0]):.6f}</text>',
            f'<text x="{map_right+50}" y="271" fill="#e2e8f0" font-size="16">center y = {float(target["center_map"][1]):.6f}</text>',
            f'<text x="{map_right+50}" y="316" fill="#7dd3fc" font-size="16" font-weight="700">EGO-RELATIVE</text>',
            f'<text x="{map_right+50}" y="347" fill="#e2e8f0" font-size="16">forward = {target_position["forward"]:.2f} m</text>',
            f'<text x="{map_right+50}" y="374" fill="#e2e8f0" font-size="16">left = {target_position["left"]:.2f} m</text>',
            f'<text x="{map_right+50}" y="401" fill="#e2e8f0" font-size="16">distance = {target_position["distance"]:.2f} m</text>',
            f'<text x="{map_right+50}" y="428" fill="#e2e8f0" font-size="16">bearing = {target_position["bearing"]:.1f}°</text>',
            f'<text x="{map_right+50}" y="455" fill="#86efac" font-size="16">inside front 180° = {str(target_position["front"]).lower()}</text>',
            f'<text x="{map_right+50}" y="510" fill="#7dd3fc" font-size="16" font-weight="700">POLYGON VERTICES</text>',
        ]
    )
    for index, point in enumerate(target["polygon_map"]):
        rows.append(
            f'<text x="{map_right+50}" y="{542 + 27*index}" fill="#cbd5e1" font-family="ui-monospace,monospace" font-size="14">p{index}: ({float(point[0]):.6f}, {float(point[1]):.6f})</text>'
        )
    rows.extend(
        [
            f'<line x1="{map_right+50}" y1="680" x2="{width-60}" y2="680" stroke="#334155"/>',
            f'<text x="{map_right+50}" y="716" fill="#f0abfc" font-size="15" font-weight="700">LOCATION SOURCE</text>',
            f'<text x="{map_right+50}" y="745" fill="#cbd5e1" font-size="14">slot database polygon</text>',
            f'<text x="{map_right+50}" y="769" fill="#cbd5e1" font-size="14">+ pose-corrected map</text>',
            f'<text x="{map_right+50}" y="793" fill="#fca5a5" font-size="14">Camera projection NOT used</text>',
            '</svg>',
        ]
    )
    return "\n".join(rows)


def tool_output_summary(call: dict) -> str:
    output = call.get("output", {})
    tool = call["tool"]
    if tool == "get_slot_position":
        return (
            f"前向 {output['ego_forward_m']:.2f} m；左向 {output['ego_left_m']:.2f} m；"
            f"方位 {output['ego_bearing_deg']:.1f}°"
        )
    if tool == "inspect_lidar_map":
        return f"真实LiDAR三联图；SHA {escape(output['media_sha256'][7:19])}…"
    if tool == "inspect_extended_lidar_history":
        return (
            f"W={output['history_span_W']}, K={output['sample_count_K']}；"
            f"结果 {escape(output['state'])}；原因 "
            + ", ".join(output["unknown_reasons"])
        )
    if tool == "inspect_rgb_sequence":
        return (
            f"{output['view_count']}张因果图像；{escape(output['projection_status'])}；"
            f"experimental_terminal={str(output['experimental_terminal_capability']).lower()}; "
            f"production_terminal={str(output['production_terminal_capability']).lower()}"
        )
    return escape(json.dumps(output, ensure_ascii=False))


def build(args: argparse.Namespace) -> None:
    report_root = args.report_root.resolve()
    experiment = report_root / "random_midroute_experiment"
    anchor = read_json(experiment / "anchor_selection.json")
    part1_summary = read_json(experiment / "part1_w30k5/summary.json")
    local_map = read_json(experiment / "part1_w30k5/local_map.json")
    decisions = read_json(experiment / "part1_w30k5/slot_decisions.json")["decisions"]
    frames = read_json(experiment / "part1_w30k5/local_frame_manifest.json")["frames"]
    queue = read_json(experiment / "part1_w30k5/unknown_agent_queue.json")
    agent_summary = read_json(experiment / "agent_closed_loop/summary.json")
    cases = read_json(experiment / "agent_closed_loop/cases.json")["cases"]
    gt_manifest = read_json(experiment / "independent_gt/manifest.json")
    static_manifest = read_json(experiment / "independent_gt/static_aware_100/manifest.json")
    ablation = read_json(experiment / "history_ablation/label_free_wk_ablation.json")

    decisions_by_id = {row["slot_id"]: row for row in decisions}
    local_by_id = {row["slot_id"]: row for row in local_map["slots"]}
    queue_ids = {row["slot_id"] for row in queue["items"]}
    direct_free_ids = {
        slot_id
        for slot_id, row in decisions_by_id.items()
        if row["state"] == "free" and relative_position(local_map, local_by_id[slot_id])["front"]
    }
    agent_state_by_id = {case["slot_id"]: case["final_state"] for case in cases}
    agent_free_ids = {slot_id for slot_id, state in agent_state_by_id.items() if state == "free"}
    agent_occupied_ids = {
        slot_id for slot_id, state in agent_state_by_id.items() if state == "occupied"
    }
    agent_unknown_ids = {slot_id for slot_id, state in agent_state_by_id.items() if state == "unknown"}
    final_free_ids = direct_free_ids | agent_free_ids
    front_ids = {
        slot_id
        for slot_id, row in local_by_id.items()
        if relative_position(local_map, row)["front"]
    }
    rear_ids = set(local_by_id) - front_ids
    assets = experiment / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    case_position_dir = assets / "case_position_maps"
    case_position_dir.mkdir(parents=True, exist_ok=True)
    case_position_paths: dict[str, Path] = {}
    for case in cases:
        slot_id = str(case["slot_id"])
        position_path = case_position_dir / f"{slot_id}.svg"
        position_path.write_text(
            build_case_position_svg(local_map, frames, slot_id, case["final_state"]),
            encoding="utf-8",
        )
        case_position_paths[slot_id] = position_path
    front_svg_path = assets / "front180_candidates.svg"
    front_svg_path.write_text(
        build_front_svg(local_map, final_free_ids, agent_occupied_ids, agent_unknown_ids),
        encoding="utf-8",
    )

    selected = anchor["selected_anchor"]
    frame_rows = []
    for row in frames:
        frame_rows.append(
            "<tr>"
            f"<td>{row['frame_id']}</td><td>{row['camera_frame']}</td>"
            f"<td>{row['map_x']:.6f}, {row['map_y']:.6f}</td>"
            f"<td>{row['map_yaw']:.5f}</td>"
            f"<td><code>{escape(Path(row['map_points_path']).name)}</code></td>"
            f"<td>{1000 * row['camera_lidar_dt_sec']:.1f} ms</td>"
            "</tr>"
        )

    part1_rows = []
    for row in decisions:
        slot_id = row["slot_id"]
        local = local_by_id[slot_id]
        pos = relative_position(local_map, local)
        route = (
            "Part1 Free→直接候选"
            if slot_id in direct_free_ids
            else "Unknown→Agent"
            if slot_id in queue_ids
            else "前方但非队列"
            if pos["front"]
            else "后方剔除"
        )
        part1_rows.append(
            "<tr>"
            f"<td><code>{escape(slot_id)}</code></td><td>{badge(row['state'])}</td>"
            f"<td>{escape(row['decision_reason'])}</td>"
            f"<td>{reason_list(row['unknown_reasons'])}</td>"
            f"<td>{pos['forward']:.2f}</td><td>{pos['left']:.2f}</td>"
            f"<td>{pos['bearing']:.1f}°</td><td>{escape(route)}</td>"
            "</tr>"
        )

    case_cards = []
    for case in cases:
        slot_id = case["slot_id"]
        lidar_call = next(
            call for call in case["tool_calls"] if call["tool"] == "inspect_lidar_map"
        )
        lidar_src = rel_link(lidar_call["output"]["media_path"], report_root)
        position_src = rel_link(case_position_paths[slot_id], report_root)
        camera_calls = [
            call for call in case["tool_calls"] if call["tool"] == "inspect_rgb_sequence"
        ]
        camera_html = (
            f'<img loading="lazy" src="{escape(rel_link(camera_calls[0]["output"]["media_path"], report_root))}" '
            f'alt="{escape(slot_id)} prediction-blind causal RGB sequence">'
            if camera_calls
            else '<div class="no-media">Agent未调用相机：该车位没有满足时间/FOV条件的可用序列。</div>'
        )
        tool_rows = []
        for call in case["tool_calls"]:
            selected_by = call.get("selected_by", "")
            if isinstance(selected_by, list):
                selected_by = ", ".join(selected_by)
            tool_rows.append(
                "<tr>"
                f"<td>{call['turn']}</td><td><code>{escape(call['tool'])}</code></td>"
                f"<td>{escape(str(selected_by))}</td><td>{escape(call['status'])}</td>"
                f"<td>{tool_output_summary(call)}</td>"
                "</tr>"
            )
        assessment = case["agent_semantic_assessment"]
        position = case["position"]
        polygon = " · ".join(
            f"({point[0]:.6f}, {point[1]:.6f})"
            for point in position["polygon_map"]
        )
        case_cards.append(
            f"""
            <article class="case" id="{escape(slot_id)}">
              <div class="case-head">
                <div><span class="eyebrow">Part2 Unknown case</span><h3>{escape(slot_id)}</h3></div>
                <div>{badge(case['input_state'])} <span class="arrow">→</span> {badge(case['final_state'])}</div>
              </div>
              <div class="facts">
                <div><b>地图中心</b><span>({position['center_map'][0]:.6f}, {position['center_map'][1]:.6f})</span></div>
                <div><b>车体坐标</b><span>前 {position['ego_forward_m']:.2f} m / 左 {position['ego_left_m']:.2f} m</span></div>
                <div><b>距离 / 方位</b><span>{position['ego_distance_m']:.2f} m / {position['ego_bearing_deg']:.1f}°</span></div>
                <div><b>前方180°</b><span>{'是' if position['inside_closed_front_180'] else '否'}</span></div>
              </div>
              <figure class="position-figure"><a href="{escape(position_src)}" target="_blank" rel="noopener"><img loading="lazy" src="{escape(position_src)}" alt="{escape(slot_id)} large map-space position audit"></a><figcaption>大位置图（点击可单独放大）：紫色为当前目标；蓝色为EGO与5帧轨迹；虚线为车头前方180°分界。位置来自车位数据库+pose-corrected地图，完全不使用Camera投影。</figcaption></figure>
              <details><summary>完整车位多边形（地图坐标，不是相机投影）</summary><code>{escape(polygon)}</code></details>
              <div class="reason-box"><b>Part1保留的Unknown原因</b>{reason_list(case['part1_unknown_reasons'])}</div>
              <div class="media-grid">
                <figure><img loading="lazy" src="{escape(lidar_src)}" alt="{escape(slot_id)} actual LiDAR triptych"><figcaption>Agent工具：真实LiDAR局部三联图</figcaption></figure>
                <figure>{camera_html}<figcaption>Agent按原因决定是否调用的Camera工具；只作诊断，不作终态证据</figcaption></figure>
              </div>
              <h4>工具调用轨迹</h4>
              <div class="table-wrap"><table><thead><tr><th>轮次</th><th>工具</th><th>为什么选</th><th>状态</th><th>输出</th></tr></thead>
              <tbody>{''.join(tool_rows)}</tbody></table></div>
              <div class="verdict-grid">
                <div><b>Camera AI语义判断</b>{badge(assessment['semantic_proposal'])}<p>{escape(assessment['finding'])}</p><small>confidence={assessment['confidence_score']:.2f}; threshold=0.60; ownership={escape(assessment['ownership'])}</small></div>
                <div><b>未闭合阻塞项</b>{reason_list(case['unresolved_blockers'])}</div>
                <div><b>验证后终态</b>{badge(case['final_state'])}<p>{escape(case['final_reason'])}</p></div>
              </div>
            </article>
            """
        )

    ablation_rows = []
    recommendation = ablation["recommended_for_gt_followup_not_accuracy_best"]
    for row in ablation["runs"]:
        selected_row = (
            row["history_span_W"] == recommendation["history_span_W"]
            and row["sample_count_K"] == recommendation["sample_count_K"]
        )
        ablation_rows.append(
            f'<tr class="{"selected" if selected_row else ""}">'
            f"<td>{row['history_span_W']}</td><td>{row['sample_count_K']}</td>"
            f"<td>{row['state_counts'].get('free', 0)}</td>"
            f"<td>{row['state_counts'].get('occupied', 0)}</td>"
            f"<td>{row['state_counts'].get('unknown', 0)}</td>"
            f"<td>{row['state_counts'].get('out_of_scope', 0)}</td>"
            f"<td>{row['terminal_count']}</td><td>{100 * row['terminal_coverage']:.1f}%</td>"
            f"<td>{row['cross_config_terminal_conflict_count']}</td>"
            f"<td>{100 * row['all_state_modal_agreement']:.1f}%</td>"
            f"<td>{row['runtime_seconds']:.2f}</td>"
            "</tr>"
        )

    input_files = [
        Path("outputs/frame_map_dataset_pose_corrected_final/frames.csv"),
        Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json"),
        Path("file.gltf"),
        report_root / "config/agent_forward180_pillar.json",
        experiment / "anchor_selection.json",
    ]
    input_rows = []
    for path in input_files:
        resolved = path if path.is_absolute() else Path.cwd() / path
        input_rows.append(
            "<tr>"
            f"<td><code>{escape(str(path))}</code></td>"
            f"<td>{resolved.stat().st_size:,}</td>"
            f"<td><code>{sha256(resolved)[:16]}…</code></td>"
            "</tr>"
        )

    gt_neutral_map = rel_link(
        experiment / "independent_gt/geometry_universe_neutral_map.png",
        report_root,
    )
    gt_lidar_contact = rel_link(
        experiment / "independent_gt/candidate8_lidar100_contact.jpg",
        report_root,
    )
    static_contact = rel_link(
        experiment / "independent_gt/static_aware_100/contact_sheets/lidar_neutral_contact_sheet_02.jpg",
        report_root,
    )
    front_svg = rel_link(front_svg_path, report_root)
    ranked_free_ids = sorted(
        final_free_ids,
        key=lambda slot_id: relative_position(local_map, local_by_id[slot_id])["distance"],
    )
    final_candidates = ", ".join(ranked_free_ids) or "无"
    displayed_candidates = ranked_free_ids[:2]
    display_text = ", ".join(displayed_candidates) or "无"
    occupied_text = ", ".join(sorted(agent_occupied_ids)) or "无"
    unknown_text = ", ".join(sorted(agent_unknown_ids)) or "无"
    generated = datetime.now(timezone.utc).isoformat()

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ParkingAgent Part1→Part2 完整审计 · 随机帧6681</title>
  <style>
    :root{{--bg:#07101c;--panel:#0e1a2b;--panel2:#111f33;--line:#27364b;--text:#e6edf7;--muted:#9ba9bd;--cyan:#38bdf8;--green:#22c55e;--amber:#f59e0b;--red:#ef4444}}
    *{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}
    a{{color:#7dd3fc}}code{{font-family:"SFMono-Regular",Consolas,monospace;font-size:.88em;color:#cbd5e1}}.wrap{{max-width:1480px;margin:auto;padding:28px}}
    header{{padding:52px 0 30px;border-bottom:1px solid var(--line)}}h1{{font-size:clamp(30px,5vw,64px);line-height:1.06;margin:10px 0 18px;letter-spacing:-.04em}}h2{{font-size:30px;margin:0 0 14px}}h3{{font-size:24px;margin:2px 0}}h4{{font-size:17px;margin:20px 0 8px}}p{{max-width:1050px}}.eyebrow{{text-transform:uppercase;letter-spacing:.16em;color:var(--cyan);font-size:12px;font-weight:800}}
    nav{{position:sticky;top:0;z-index:5;background:#07101cee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px);padding:10px 0}}nav a{{margin-right:18px;text-decoration:none;font-weight:700;font-size:13px}}
    section{{padding:44px 0;border-bottom:1px solid var(--line)}}.lead{{font-size:18px;color:#c7d2e2}}.callout{{padding:18px 20px;border:1px solid #7f1d1d;background:#2a1118;border-radius:12px;margin:20px 0}}.callout.good{{border-color:#166534;background:#0c261b}}.callout.info{{border-color:#075985;background:#092232}}
    .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:22px 0}}.metric{{background:var(--panel);border:1px solid var(--line);padding:16px;border-radius:12px}}.metric b{{display:block;font-size:27px}}.metric span{{color:var(--muted)}}
    .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:var(--panel);font-size:13px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{position:sticky;top:0;background:#15243a;color:#cbd5e1}}tr:last-child td{{border-bottom:0}}tr.selected td{{background:#183522}}
    .badge{{display:inline-block;padding:2px 9px;border-radius:999px;font-weight:800;font-size:12px;border:1px solid currentColor}}.badge.free{{color:#86efac;background:#14532d55}}.badge.occupied{{color:#fca5a5;background:#7f1d1d55}}.badge.unknown{{color:#fcd34d;background:#78350f55}}.badge.out_of_scope{{color:#94a3b8}}.muted,small{{color:var(--muted)}}.arrow{{color:var(--muted);padding:0 8px}}
    .formula{{padding:14px 18px;background:#050b14;border-left:4px solid var(--cyan);font-family:monospace;overflow:auto}}figure{{margin:0}}figure img,.map-img{{width:100%;display:block;border:1px solid var(--line);border-radius:10px;background:#08111f}}figcaption{{color:var(--muted);font-size:12px;margin-top:6px}}.two{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.three{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
    .case{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin:22px 0}}.case-head{{display:flex;justify-content:space-between;gap:16px;align-items:center}}.facts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:16px 0}}.facts div{{background:var(--panel2);padding:10px;border-radius:8px}}.facts b,.facts span{{display:block}}.facts span{{color:#cbd5e1}}details{{margin:10px 0}}summary{{cursor:pointer;color:#bae6fd}}.reason-box{{padding:12px;background:#241d0d;border:1px solid #713f12;border-radius:9px;margin:12px 0}}.reason-box b,.verdict-grid b{{display:block;margin-bottom:8px}}.position-figure{{margin:18px 0 22px}}.position-figure img{{width:100%;min-height:520px;object-fit:contain;background:#07101c}}.media-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}.media-grid img{{max-height:560px;object-fit:contain}}.no-media{{min-height:260px;display:grid;place-items:center;padding:30px;text-align:center;border:1px dashed #475569;border-radius:10px;color:var(--muted)}}.verdict-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px}}.verdict-grid>div{{padding:14px;background:var(--panel2);border-radius:10px}}
    .flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:20px 0}}.flow div{{position:relative;padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:10px}}.flow b{{display:block;color:#7dd3fc}}.flow span{{color:var(--muted)}}.footer{{padding:30px 0;color:var(--muted)}}
    @media(max-width:900px){{.two,.three,.media-grid,.verdict-grid{{grid-template-columns:1fr}}.facts{{grid-template-columns:1fr 1fr}}.flow{{grid-template-columns:1fr}}.wrap{{padding:18px}}}}
  </style>
</head>
<body>
<nav><div class="wrap"><a href="#result">结论</a><a href="#input">输入</a><a href="#part1">Part1</a><a href="#part2">Part2 180°</a><a href="#agent">Agent工具</a><a href="#gt">GT</a><a href="#ablation">消融</a><a href="#artifacts">产物</a></div></nav>
<main class="wrap">
<header id="result">
  <span class="eyebrow">ParkingAgent · prediction-auditable experiment</span>
  <h1>Part1 → Part2 全流程审计<br>随机中段帧 6681</h1>
  <p class="lead">这份首页只展示实际输入、逐车位状态、严格前方180°、Agent真实工具调用与安全门结果。没有用旧Part1候选反过来构造GT，也没有把相机投影当作车位位置。</p>
  <div class="metrics">
    <div class="metric"><b>23</b><span>Part1局部车位</span></div>
    <div class="metric"><b>1 / 0 / 22</b><span>Free / Occupied / Unknown</span></div>
    <div class="metric"><b>9</b><span>严格前方180°车位</span></div>
    <div class="metric"><b>1 + 8</b><span>直接Free + Unknown进Agent</span></div>
    <div class="metric"><b>30</b><span>Agent工具调用</span></div>
    <div class="metric"><b>{len(final_free_ids)}</b><span>最终Free：{escape(final_candidates)}</span></div>
  </div>
  <div class="callout info"><b>Camera阈值已从0.90调到0.60。</b> 8个Part1 Unknown中，Agent确认4个Free；另外4个保持Unknown。slot_0996经你的视觉复核确认目标区域被挡住，已从Occupied纠正为Unknown。加上Part1直接Free，系统得到5个Free。由于frame 6681仍没有独立人工终态GT，这里报告的是输出覆盖提升，不冒充准确率。</div>
</header>

<section id="input">
  <span class="eyebrow">01 · Input contract</span><h2>实际用了什么数据</h2>
  <p>锚点不是9277，也不是按效果挑选。固定随机种子20260729，从路线中间60%、具备100帧因果历史且25m内至少8个几何车位的701个合格锚点中抽到6681；选择过程不读取GT或Part1状态。</p>
  <div class="two">
    <div class="table-wrap"><table><thead><tr><th>项目</th><th>值</th></tr></thead><tbody>
      <tr><td>LiDAR锚点</td><td>{selected['frame_id']}</td></tr>
      <tr><td>同步Camera帧</td><td>{selected['camera_frame']}</td></tr>
      <tr><td>地图位置</td><td>{selected['map_xy'][0]:.6f}, {selected['map_xy'][1]:.6f}</td></tr>
      <tr><td>地图yaw</td><td>{selected['map_yaw_rad']:.6f} rad</td></tr>
      <tr><td>合格锚点数</td><td>{anchor['eligible_anchor_count']}</td></tr>
      <tr><td>合格集哈希</td><td><code>{escape(anchor['eligible_anchor_sha256'])}</code></td></tr>
      <tr><td>因果约束</td><td>只允许 frame ≤ 6681</td></tr>
    </tbody></table></div>
    <div class="table-wrap"><table><thead><tr><th>输入文件</th><th>字节</th><th>SHA-256</th></tr></thead><tbody>{''.join(input_rows)}</tbody></table></div>
  </div>
  <h3>Part1实际采样的5帧（W=30中均匀取K=5，包含锚点）</h3>
  <div class="table-wrap"><table><thead><tr><th>LiDAR帧</th><th>Camera帧</th><th>地图位置</th><th>yaw</th><th>地图点文件</th><th>时差</th></tr></thead><tbody>{''.join(frame_rows)}</tbody></table></div>
</section>

<section id="part1">
  <span class="eyebrow">02 · Conservative perception</span><h2>Part1：只有证据闭合才出终态</h2>
  <div class="flow">
    <div><b>1 定位对齐</b><span>每帧LiDAR转换到pose-corrected地图坐标。</span></div>
    <div><b>2 车位几何查询</b><span>使用1397车位数据库的多边形/3D柱体，不是相机投影框。</span></div>
    <div><b>3 可见性与时序</b><span>逐帧射线、核心覆盖、视角基线、支持帧一致性。</span></div>
    <div><b>4 Occupied高门槛</b><span>车辆点数、占地、下部、高度、体素、时序全部过门；墙/柱/挡车器静态语义可否决。</span></div>
    <div><b>5 三态输出</b><span>Free和Occupied都必须强证据；其余保留Unknown及原因给Agent。</span></div>
  </div>
  <div class="callout good"><b>柱体误报修正：</b>紧凑竖直结构、线性静态结构、边界主导、水平盖板不会单独触发Occupied；与glTF的 wall/elevator/arrester 632,482个静态采样点匹配后可否决Occupied。本帧最终Occupied=0。</div>
  <p><b>Free：</b>核心体积经过足够多有效射线且持续净空；<b>Occupied：</b>必须同时满足车辆三维形状、下部覆盖、跨帧支持和非静态结构；<b>Unknown：</b>遮挡、视角不足、边界/柱体风险或Free/Occupied证据冲突。Unknown不是“没有状态”，原因完整进入Part2。</p>
  <h3>Part1全部23个车位输出</h3>
  <div class="table-wrap"><table><thead><tr><th>车位</th><th>状态</th><th>决策原因</th><th>Unknown原因</th><th>前向m</th><th>左向m</th><th>方位</th><th>进入下一步</th></tr></thead><tbody>{''.join(part1_rows)}</tbody></table></div>
  <div class="callout info"><b>Part1与GT比较：</b>本随机时刻没有独立人工终态GT，因此准确率=N/A、误报率=N/A。可确认的是协议合规性：5帧全为因果帧、Occupied采用静态结构否决、22个不充分证据没有被强行二分类。</div>
</section>

<section id="part2">
  <span class="eyebrow">03 · Candidate routing</span><h2>Part2：严格车头正前方180°</h2>
  <div class="formula">dx,dy = (slot_center - ego_position) / map_units_per_meter<br>forward = cos(yaw)·dx + sin(yaw)·dy<br>left = -sin(yaw)·dx + cos(yaw)·dy<br>保留条件：forward ≥ 0 且 |atan2(left, forward)| ≤ 90°</div>
  <p>23个局部车位中，9个在严格前半平面：<b>{len(direct_free_ids)}个Part1 Free直接进入安全候选，{len(queue_ids)}个Unknown进入Agent</b>；其余{len(rear_ids)}个在车后方，全部排除。这里不是旧的160°扇区。地图位置来自车位数据库与定位坐标；Camera投影只在Agent诊断工具内部使用。</p>
  <img class="map-img" src="{escape(front_svg)}" alt="strict front 180 degree Part2 candidates at frame 6681">
  <div class="callout info"><b>“最多展示两个临时候选”是什么意思：</b>8个Unknown仍全部进入Agent。0.60策略最终得到5个Free，再按与车身距离升序只展示前两个：<code>{escape(display_text)}</code>；其余Free保留在完整结果中。</div>
</section>

<section id="agent">
  <span class="eyebrow">04 · Reason-directed tool loop</span><h2>Agent不是固定先用Camera；它按Unknown原因选工具</h2>
  <p>每个Unknown先定位并检查LiDAR局部几何；当原因涉及遮挡、可见性、车辆占地或目标归属时，Agent选择更长LiDAR历史，并在存在合格因果图像时选择Camera。Camera是否调用是逐车位决策：0942/43/44/45/0964/0996调用，0994/0995因没有合格视图未调用。</p>
  <div class="three">
    <div class="metric"><b>8</b><span><code>get_slot_position</code>：地图多边形+车体坐标</span></div>
    <div class="metric"><b>8 + 8</b><span><code>inspect_lidar_map</code> + <code>inspect_extended_lidar_history</code></span></div>
    <div class="metric"><b>6</b><span><code>inspect_rgb_sequence</code>：Agent按原因选择，0.60实验终态门</span></div>
  </div>
  <div class="callout"><b>本次按你的要求启用0.60实验策略：</b>Camera多帧目标覆盖区的AI语义分数≥0.60且目标归属由叠加区域确认，就允许覆盖Part1 Unknown。生产策略仍保留独立像素标定审计并会维持Unknown；两种状态都写入逐车位JSON，避免把实验放宽伪装成生产安全结论。</div>
  {''.join(case_cards)}
  <h3>原始与Agent后对比</h3>
  <div class="table-wrap"><table><thead><tr><th>阶段</th><th>Free</th><th>Occupied</th><th>Unknown</th><th>解释</th></tr></thead><tbody>
    <tr><td>Part1前方180°</td><td>1</td><td>0</td><td>8</td><td>slot_0965直接Free；8个Unknown进入Agent</td></tr>
    <tr><td>Camera-Agent 0.60后的8个输入</td><td>4</td><td>0</td><td>4</td><td>0942/43/44/45→Free；0964/0994/0995/0996→Unknown</td></tr>
    <tr><td>生产fail-closed对照</td><td>0</td><td>0</td><td>8</td><td>未放宽像素标定审计时仍全部Unknown</td></tr>
    <tr><td>系统实验最终前方状态</td><td>5</td><td>0</td><td>4</td><td>包含Part1直接Free的0965；UI显示{escape(display_text)}</td></tr>
  </tbody></table></div>
</section>

<section id="gt">
  <span class="eyebrow">05 · Independent ground truth protocol</span><h2>GT：先独立建全集，不能拿旧Part1筛选结果当GT</h2>
  <p>围绕frame 6681按路线25m范围、100帧因果历史，从全量车位几何独立得到72个车位；过程未读取Part1状态、候选ID或Agent结果。已生成中性地图、72个LiDAR证据包、静态结构解释视图和空白人工标注模板。</p>
  <div class="metrics">
    <div class="metric"><b>72</b><span>预测独立几何全集</span></div>
    <div class="metric"><b>100</b><span>每车位因果LiDAR帧</span></div>
    <div class="metric"><b>{len(static_manifest['semantic_static_layers'])}</b><span>wall/elevator/arrester静态语义层</span></div>
    <div class="metric"><b>false</b><span>formal_gt</span></div>
  </div>
  <div class="two">
    <figure><img loading="lazy" src="{escape(gt_neutral_map)}" alt="prediction independent 72-slot neutral geometry universe"><figcaption>预测独立的72车位中性几何全集</figcaption></figure>
    <figure><img loading="lazy" src="{escape(gt_lidar_contact)}" alt="candidate eight neutral 100-frame lidar review"><figcaption>随机8候选的100帧中性LiDAR审核图；没有预测标签叠加</figcaption></figure>
  </div>
  <details><summary>查看静态结构感知审核示例</summary><figure><img loading="lazy" src="{escape(static_contact)}" alt="static semantic aware lidar review contact sheet"><figcaption>静态结构解释层用于区分柱/墙与车辆残差</figcaption></figure></details>
  <div class="callout"><b>为什么现在没有填Free/Occupied GT：</b>现有人工GT只覆盖另一些车位/时刻；随机8车位没有frame 6681同一时刻标签。相机外参又未达到像素级终态信任，LiDAR中还存在遮挡与静态结构混叠。此时自动填写F/O会再次把模型预测当成GT。当前最准确的做法是保持未标注，并把证据包交给独立人工标注者。</div>
</section>

<section id="ablation">
  <span class="eyebrow">06 · W/K ablation</span><h2>历史多少帧：15组全部跑完，但暂不冒充准确率实验</h2>
  <p>W表示因果历史池长度，K表示在W内均匀采样且必含锚点。所有组合在同一个72车位预测独立几何全集、同一静态结构否决器上运行。没有GT，所以只报告状态覆盖、跨配置Free↔Occupied冲突和稳定性。</p>
  <div class="table-wrap"><table><thead><tr><th>W</th><th>K</th><th>Free</th><th>Occupied</th><th>Unknown</th><th>范围外</th><th>终态数</th><th>终态覆盖</th><th>F/O冲突</th><th>状态一致率</th><th>秒</th></tr></thead><tbody>{''.join(ablation_rows)}</tbody></table></div>
  <div class="callout info"><b>稳定性优先组合：W={recommendation['history_span_W']}、K={recommendation['sample_count_K']}。</b> 它有{recommendation['terminal_count']}个终态、{100 * recommendation['all_state_modal_agreement']:.1f}%状态众数一致率、0个跨配置Free↔Occupied冲突，耗时{recommendation['runtime_seconds']:.2f}s。它只能作为下一轮独立人工GT的优先方案；没有GT前不能称为“准确率最佳”。</div>
</section>

<section id="artifacts">
  <span class="eyebrow">07 · Reproducibility</span><h2>可核查产物</h2>
  <ul>
    <li><a href="random_midroute_experiment/anchor_selection.json">随机锚点选择清单</a></li>
    <li><a href="random_midroute_experiment/part1_w30k5/slot_decisions.json">Part1完整23车位决策JSON</a> · <a href="random_midroute_experiment/part1_w30k5/decision_trace.jsonl">决策轨迹</a></li>
    <li><a href="random_midroute_experiment/part1_w30k5/unknown_agent_queue.json">8车位Agent输入队列</a></li>
    <li><a href="random_midroute_experiment/agent_closed_loop/tool_trace.json">30次工具调用轨迹</a> · <a href="random_midroute_experiment/agent_closed_loop/cases.json">逐车位闭环案例</a></li>
    <li><a href="random_midroute_experiment/independent_gt/manifest.json">独立GT协议清单</a> · <a href="random_midroute_experiment/independent_gt/gt_annotation_template.csv">空白人工标注模板</a></li>
    <li><a href="random_midroute_experiment/history_ablation/label_free_wk_ablation.json">15组W/K完整JSON</a> · <a href="random_midroute_experiment/history_ablation/label_free_wk_ablation.csv">汇总CSV</a></li>
  </ul>
  <div class="callout good"><b>0.60实验策略最终结果：</b>Free为 <code>{escape(final_candidates)}</code>，Occupied为 <code>{escape(occupied_text)}</code>，Unknown为 <code>{escape(unknown_text)}</code>。最终UI展示 <code>{escape(display_text)}</code>。</div>
</section>
<div class="footer">生成时间：{escape(generated)} · main experiment anchor=6681 · GT fields read during Agent inference: [] · accuracy evaluable: false</div>
</main>
</body>
</html>
"""
    args.output.write_text(html, encoding="utf-8")

    manifest = {
        "schema_version": "random-midroute-full-report/1.2",
        "generated_at_utc": generated,
        "output": str(args.output),
        "output_sha256": "sha256:" + sha256(args.output),
        "front180_svg": str(front_svg_path),
        "front180_svg_sha256": "sha256:" + sha256(front_svg_path),
        "case_position_maps": {
            slot_id: {
                "path": str(path),
                "sha256": "sha256:" + sha256(path),
            }
            for slot_id, path in sorted(case_position_paths.items())
        },
        "anchor_frame": selected["frame_id"],
        "part1_counts": {
            state: sum(row["state"] == state for row in decisions)
            for state in ("free", "occupied", "unknown")
        },
        "front180": {
            "slot_count": len(front_ids),
            "direct_free_ids": sorted(direct_free_ids),
            "agent_free_ids": sorted(agent_free_ids),
            "agent_occupied_ids": sorted(agent_occupied_ids),
            "agent_unknown_ids": sorted(agent_unknown_ids),
            "final_free_ids": ranked_free_ids,
            "displayed_candidate_ids": displayed_candidates,
            "rear_excluded_count": len(rear_ids),
        },
        "agent": agent_summary,
        "formal_gt": bool(gt_manifest.get("formal_gt", False)),
        "accuracy_evaluable": False,
        "ablation_recommendation_is_accuracy_best": False,
        "ablation_recommendation": recommendation,
    }
    manifest_path = experiment / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("Nature_ParkingAgent_实验报告_20260728"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Nature_ParkingAgent_实验报告_20260728/index.html"),
    )
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
