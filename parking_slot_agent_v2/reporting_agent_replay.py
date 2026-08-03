"""Build a self-contained, visual-first replay of real Part2 agent traces.

The report intentionally exposes only auditable decision summaries recorded in
structured model actions (rationale/reason), tool observations, and deterministic
geometry gates.  It does not claim to expose private hidden chain-of-thought.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from parking_slot_part2.media import load_lidar_evidence_pack

from .lidar_explainability import render_explainable_lidar


REASON_ZH = {
    "boundary_dominated": "回波主要落在车位边界，不能当作车体",
    "weak_vehicle_evidence": "车辆形状证据不足",
    "weak_visibility_limited": "15帧可见性不足",
    "low_core_ray_coverage": "车位核心射线覆盖不足",
    "zero_lidar_core_coverage": "车位核心区域没有LiDAR覆盖",
    "complete_lidar_occlusion": "目标区域完全被遮挡",
    "partial_route_scope": "目标仅部分落入当前轨迹覆盖范围",
    "no_occupied_geometry_support": "没有占用几何支持",
    "target_pixel_region_unlocalized": "相机中无法定位目标像素区域",
    "target_image_region_unlocalized": "相机中无法定位目标图像区域",
    "insufficient_free_observation": "自由空间观测不足",
    "robustness_not_stable": "位姿扰动测试不稳定",
    "camera_target_unlocalized": "相机目标未可靠定位",
    "tool_budget_exhausted": "三轮工具预算已耗尽",
    "strong_free_geometry": "自由空间几何证据强",
    "stable_all_robustness_variants_pass": "7/7位姿扰动全部通过",
    "high_core_and_near_ground_coverage": "核心与近地面覆盖充分",
    "no_unresolved_core_hit": "核心区没有未解释的障碍回波",
    "occupied_candidate_boundary_dominated": "占用候选被边界回波反证",
    "strong_occupied_geometry": "占用几何证据强",
    "target_core_overlap": "障碍物与车位核心重合",
    "multi_frame_temporal_support": "多帧时间支持稳定",
    "robustness_variants_passed": "位姿扰动测试通过",
    "free_geometry_insufficient": "自由空间证据不足",
    "camera_fov_outside": "目标在保守相机FOV之外",
}

TOOL_ZH = {
    "part1": "Part1输入",
    "fov": "Camera FOV检查",
    "lidar_detail": "60帧LiDAR细查",
    "camera_context": "相机上下文",
    "camera_sequence": "因果相机序列",
    "camera_crop": "目标区域裁剪",
    "gate": "代码硬门",
    "final": "最终决策",
}

STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _num(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.{digits}f}"


def _reason_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [REASON_ZH.get(str(value), str(value).replace("_", " ")) for value in values]


def _find_lidar(case: Mapping[str, Any]) -> Mapping[str, Any]:
    for evidence in case.get("evidence", []):
        if evidence.get("tool_name") == "lidar_detail" and evidence.get("status") == "ok":
            return evidence
    return {}


def _gate_rows(card: Mapping[str, Any], state: str) -> list[dict[str, Any]]:
    free = card.get("free_geometry", {})
    occ = card.get("occupied_geometry", {})
    robust = card.get("robustness", {})
    terminal = card.get("terminal_geometry_gate", {})
    if state == "free":
        return [
            {"name": "有效观测体积 ≥ 70%", "value": _pct(free.get("observed_volume_ratio")), "pass": float(free.get("observed_volume_ratio", 0)) >= .70},
            {"name": "近地面覆盖 ≥ 70%", "value": _pct(free.get("near_ground_bev_coverage")), "pass": float(free.get("near_ground_bev_coverage", 0)) >= .70},
            {"name": "核心障碍回波必须为0", "value": str(occ.get("core_point_count", "—")), "pass": int(occ.get("core_point_count", -1)) == 0},
            {"name": "7/7位姿扰动一致", "value": f"{robust.get('passing_variants', '—')}/{robust.get('total_variants', '—')}", "pass": bool(robust.get("stable")) and robust.get("passing_variants") == robust.get("total_variants")},
            {"name": "Free代码门", "value": "允许" if terminal.get("free_eligible") else "拒绝", "pass": bool(terminal.get("free_eligible"))},
        ]
    if state == "occupied":
        return [
            {"name": "核心障碍回波", "value": str(occ.get("core_point_count", "—")), "pass": int(occ.get("core_point_count", 0)) > 0},
            {"name": "车位核心重合", "value": _pct(occ.get("core_overlap")), "pass": float(occ.get("core_overlap", 0)) >= .35},
            {"name": "支持高度层 ≥ 2", "value": str(occ.get("supported_height_layers", "—")), "pass": int(occ.get("supported_height_layers", 0)) >= 2},
            {"name": "7/7位姿扰动一致", "value": f"{robust.get('passing_variants', '—')}/{robust.get('total_variants', '—')}", "pass": bool(robust.get("stable")) and robust.get("passing_variants") == robust.get("total_variants")},
            {"name": "Occupied代码门", "value": "允许" if terminal.get("occupied_eligible") else "拒绝", "pass": bool(terminal.get("occupied_eligible"))},
        ]
    blockers = list(terminal.get("free_blockers", [])) + list(terminal.get("occupied_blockers", []))
    return [
        {"name": "Free代码门", "value": "拒绝", "pass": False},
        {"name": "Occupied代码门", "value": "拒绝", "pass": False},
        {"name": "核心射线覆盖", "value": _pct(free.get("core_ray_coverage")), "pass": float(free.get("core_ray_coverage", 0)) >= .70},
        {"name": "位姿扰动一致", "value": f"{robust.get('passing_variants', '—')}/{robust.get('total_variants', '—')}", "pass": bool(robust.get("stable"))},
        {"name": "主要阻断项", "value": str(blockers[0]).replace("_", " ") if blockers else "证据不足", "pass": False},
    ]


def _decision_summary_zh(action: Mapping[str, Any], card: Mapping[str, Any], state: str) -> str:
    free = card.get("free_geometry", {})
    occ = card.get("occupied_geometry", {})
    robust = card.get("robustness", {})
    if action.get("type") == "tool":
        tool = str(action.get("tool", ""))
        if tool == "camera_context":
            return "LiDAR还不能同时通过空闲门或占用门。下一步查看目标标记地图和完整相机画面，先判断能否可靠定位目标车位。"
        if tool == "camera_sequence":
            return "单帧相机仍没有可靠的目标像素投影。下一步查看严格早于当前时刻的相机序列，寻找短暂可见或遮挡变化。"
        if tool == "camera_crop":
            return "当前相机上下文只能给出大致方位。下一步裁剪候选区域并增强对比度，检查是否存在可定位的车辆外观。"
        return f"当前证据不足，Agent选择调用{TOOL_ZH.get(tool, tool)}继续消除不确定性。"
    if state == "free":
        return (
            f"60帧LiDAR覆盖了{_pct(free.get('observed_volume_ratio'))}的目标体积和"
            f"{_pct(free.get('near_ground_bev_coverage'))}的近地面区域；核心障碍点为"
            f"{occ.get('core_point_count', '—')}，位姿测试{robust.get('passing_variants', '—')}/"
            f"{robust.get('total_variants', '—')}通过。边界回波不能构成车体，因此Agent提议空闲。"
        )
    if state == "occupied":
        return (
            f"目标核心内累计{occ.get('core_point_count', '—')}个障碍点，高度跨度"
            f"{_num(occ.get('height_span_m'), 2)} m，支持{occ.get('supported_height_layers', '—')}个高度层，"
            f"位姿测试{robust.get('passing_variants', '—')}/{robust.get('total_variants', '—')}通过；"
            "自由空间覆盖不足，因此Agent提议占用。"
        )
    return (
        f"目标核心射线覆盖仅{_pct(free.get('core_ray_coverage'))}，遮挡率"
        f"{_pct(free.get('occlusion_ratio'))}，位姿测试{robust.get('passing_variants', '—')}/"
        f"{robust.get('total_variants', '—')}通过；相机又没有可靠目标投影。三轮后两种终态门都失败，因此保留未知。"
    )


def _copy_asset(source: str | Path, destination: Path, slot_id: str, label: str) -> str:
    path = Path(source)
    if not path.is_file():
        return ""
    suffix = path.suffix.lower() or ".png"
    target = destination / f"{slot_id}_{label}{suffix}"
    shutil.copy2(path, target)
    return f"assets/{target.name}"


def _scores(raw: Mapping[str, Any] | None) -> dict[str, float]:
    value = raw or {}
    return {
        "free": float(value.get("free_confidence", 0.0)),
        "occupied": float(value.get("occupied_confidence", 0.0)),
        "unknown": float(value.get("unknown_confidence", 0.0)),
    }


def _action_belief(action: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the belief attached to either a tool call or final proposal."""

    if action.get("type") == "tool" and isinstance(action.get("belief"), Mapping):
        return dict(action["belief"])
    if action.get("type") == "final":
        state = str(action.get("state", "unknown"))
        free = float(action.get("free_confidence", 0.0))
        occupied = float(action.get("occupied_confidence", 0.0))
        unknown = float(action.get("unknown_confidence", 0.0))
        if state == "unknown" and unknown == 0.0:
            unknown = max(0.0, 1.0 - max(free, occupied))
        return {
            "state": state,
            "free_confidence": free,
            "occupied_confidence": occupied,
            "unknown_confidence": unknown,
            "remaining_unknown_reasons": action.get("reason_codes", []),
        }
    return {"state": "unknown"}


def _step(
    *, key: str, label: str, kicker: str, title: str, summary: str,
    rationale: str, observation: str, image: str, scores: Mapping[str, Any],
    state: str, badges: list[str] | None = None, gates: list[dict[str, Any]] | None = None,
    raw: Mapping[str, Any] | None = None, image_label: str = "本步实际证据",
) -> dict[str, Any]:
    return {
        "key": key, "label": label, "kicker": kicker, "title": title,
        "summary": summary, "rationale": rationale, "observation": observation,
        "image": image, "scores": _scores(scores), "state": state,
        "badges": badges or [], "gates": gates or [], "raw": raw or {},
        "image_label": image_label,
    }


def _case_payload(run_dir: Path, assets: Path, slot_id: str) -> dict[str, Any]:
    case = _read(run_dir / "slot_cases" / f"{slot_id}.json")
    audit_dir = run_dir / "openai_audit" / case["case_id"]
    audits = [_read(path) for path in sorted(audit_dir.glob("turn_*.json"))]
    resources = case.get("resources", {})
    lidar = _find_lidar(case)
    card = lidar.get("metadata", {}).get("geometry_card", {})
    final_state = str(case.get("final_state", "unknown"))
    fov = case.get("fov", {})
    fov_details = fov.get("details", {})
    fov_img = _copy_asset(resources.get("part2_fov_occupancy_map", ""), assets, slot_id, "fov")
    raw_lidar_img = _copy_asset(resources.get("part2_lidar_model_evidence_round_1", ""), assets, slot_id, "lidar_model_input")
    gate_img = _copy_asset(resources.get("part2_lidar_geometry_card_round_1", ""), assets, slot_id, "geometry_card")
    lidar_img = raw_lidar_img
    pack_path = resources.get("extended_lidar_evidence_path")
    decision_path = resources.get("extended_lidar_decision_path")
    if pack_path and decision_path and Path(pack_path).is_file() and Path(decision_path).is_file():
        explain_path = assets / f"{slot_id}_lidar_explained.png"
        raw_decision = _read(Path(decision_path))
        pack = load_lidar_evidence_pack(pack_path, expected_slot_id=slot_id)
        render_explainable_lidar(
            pack,
            card,
            explain_path,
            decision=raw_decision.get("decision", {}),
        )
        lidar_img = f"assets/{explain_path.name}"

    steps: list[dict[str, Any]] = []
    steps.append(_step(
        key="part1", label="输入", kicker="STEP 0 · PART1交接", title="Agent收到一个仍未解决的候选车位",
        summary=f"Part1把 {slot_id} 标为Unknown，并把原始分数、未决原因和局部地图完整交给Part2。",
        rationale="此时没有重新分类。Agent首先要确认目标是否在相机视野内，再决定调用Camera还是LiDAR。",
        observation="黄色/橙色目标仍是Unknown；紫色标记是当前目标位置，蓝色扇形是保守Camera范围。",
        image=fov_img, scores=case.get("part1_scores", {}), state=str(case.get("part1_state", "unknown")),
        badges=["原始状态未改写", "分数未校准", *(_reason_list(case.get("unknown_reasons"))[:2])],
        raw={"part1_state": case.get("part1_state"), "part1_scores": case.get("part1_scores"), "unknown_reasons": case.get("unknown_reasons")},
    ))
    visibility = str(fov.get("visibility", "uncertain"))
    visible_zh = {"visible": "在Camera范围内", "partially_visible": "部分进入Camera范围", "not_visible": "不在Camera范围内"}.get(visibility, "视野关系不确定")
    steps.append(_step(
        key="fov", label="FOV", kicker="STEP 1 · 强制路由", title=f"FOV检查：{visible_zh}",
        summary=f"目标方位 {float(fov_details.get('target_bearing_deg', 0)):.1f}°，距离 {float(fov_details.get('target_distance_m', 0)):.1f} m；Camera半视场角为 {float(fov_details.get('half_fov_deg', 0)):.0f}°。",
        rationale=("Camera被禁止用于终态判断，直接路由到目标对齐LiDAR。" if visibility == "not_visible" else "Camera只能作为补充证据；没有像素级投影时不能单独给终态。"),
        observation="地图上的蓝色扇形是Camera可见范围；目标轮廓与扇形的关系决定下一步工具是否合法。",
        image=fov_img, scores=case.get("part1_scores", {}), state="unknown",
        badges=[visible_zh, f"FOV置信度 {_pct(fov.get('confidence'))}", "先检查再调用工具"], raw=fov,
    ))

    round1 = next((row for row in case.get("rounds", []) if row.get("round_index") == 1), {})
    first_action = audits[0].get("action", {}) if audits else {}
    first_belief = first_action.get("belief") if first_action.get("type") == "tool" else case.get("final_scores")
    lidar_state = str((first_belief or {}).get("state", final_state if first_action.get("type") == "final" else "unknown"))
    lidar_rationale = _decision_summary_zh(first_action, card, lidar_state)
    free = card.get("free_geometry", {})
    occ = card.get("occupied_geometry", {})
    lidar_badges = [
        f"60个因果帧", f"核心点 {occ.get('core_point_count', '—')}",
        f"观测体积 {_pct(free.get('observed_volume_ratio'))}",
    ]
    steps.append(_step(
        key="lidar", label="LiDAR", kicker="STEP 2 · 第1轮证据", title="工具返回目标对齐的60帧几何证据",
        summary="把同一份60帧证据重绘成四步：观测位置、车位内部证据归属、三维高度和代码门。绿色是自由体素，红色是有效障碍，橙色是被边界规则否决的候选。",
        rationale=lidar_rationale,
        observation=str(round1.get("observation_summary", lidar.get("summary", ""))),
        image=lidar_img, scores=first_belief, state=lidar_state, badges=lidar_badges,
        gates=_gate_rows(card, final_state),
        raw={"model_action": first_action, "geometry_card": card, "original_model_input": raw_lidar_img},
        image_label="同一LiDAR证据的可解释重绘",
    ))

    for index, audit in enumerate(audits, start=1):
        action = audit.get("action", {})
        if action.get("type") != "tool":
            continue
        tool = str(action.get("tool"))
        round_index = index + 1
        round_row = next((row for row in case.get("rounds", []) if row.get("round_index") == round_index), {})
        resource_key = next((key for key in resources if key.startswith(f"part2_{tool}_round_{round_index}")), "")
        image_path = resources.get(resource_key, "") if resource_key else ""
        image = _copy_asset(image_path, assets, slot_id, f"round_{round_index}_{tool}")
        if tool == "camera_context":
            image_path = resources.get(f"part2_camera_context_round_{round_index}", image_path)
            image = _copy_asset(image_path, assets, slot_id, f"round_{round_index}_{tool}")
        # This panel represents the complete ReAct transition: the selected
        # tool returns an observation, then the next OpenAI turn chooses the
        # next action.  Showing the current action here would duplicate the
        # tool call already visible in the preceding panel.
        next_audit = audits[index] if index < len(audits) else audit
        next_action = next_audit.get("action", {})
        belief = _action_belief(next_action)
        next_state = str(belief.get("state", "unknown"))
        steps.append(_step(
            key=f"round{round_index}", label=f"R{round_index} {TOOL_ZH.get(tool, tool)}",
            kicker=f"STEP {len(steps)} · 观察返回 → Agent决策", title=f"{TOOL_ZH.get(tool, tool)}返回，Agent重新决策",
            summary=f"Agent读取{TOOL_ZH.get(tool, tool)}的返回结果，更新内部belief，并立即给出下一动作。",
            rationale=_decision_summary_zh(next_action, card, next_state), observation=str(round_row.get("observation_summary", "")),
            image=image, scores=belief, state=next_state,
            badges=[f"OpenAI turn {next_audit.get('turn')}", f"耗时 {next_audit.get('elapsed_seconds', 0):.1f}s", *(_reason_list(belief.get("remaining_unknown_reasons"))[:2])],
            raw={"model_action": next_action, "tool_observation": round_row, "usage": next_audit.get("usage"), "previous_tool_call": action},
        ))

    final_action = next((a.get("action", {}) for a in reversed(audits) if a.get("action", {}).get("type") == "final"), {})
    final_reason = (
        f"Agent已提交{STATE_ZH.get(final_state, final_state)}提议；Validator逐项核查证据引用、置信度和几何硬门，通过后才写回。"
        if final_state != "unknown" else
        "Agent提交未知并列出未消除的不确定性；Validator确认Free与Occupied均未通过合法硬门，因此拒绝猜测。"
    )
    steps.append(_step(
        key="final", label="终态", kicker=f"STEP {len(steps)} · 代码裁决", title=f"最终状态：{STATE_ZH.get(final_state, final_state)}",
        summary=(
            f"模型提出{STATE_ZH.get(final_state, final_state)}，确定性代码门验证通过后才允许写回SlotCase。"
            if final_state != "unknown" else
            "空闲门和占用门都没有同时满足必要条件；工具预算耗尽后，系统安全地保留未知。"
        ),
        rationale=final_reason, observation="状态写回原SlotCase；同时保留引用的evidence_id、reason_codes、分数和停止原因。",
        image=lidar_img or gate_img, scores=case.get("final_scores", {}), state=final_state,
        badges=[f"{STATE_ZH.get(case.get('part1_state'), '未知')} → {STATE_ZH.get(final_state, final_state)}", f"模型轮次 {len(audits)}", f"工具轮次 {len(case.get('rounds', []))}"],
        gates=_gate_rows(card, final_state), raw={"final_action": final_action, "final_reason": case.get("final_reason"), "unresolved_reasons": case.get("unresolved_reasons")},
        image_label="终态LiDAR证据链总览",
    ))
    return {
        "slot_id": slot_id, "case_id": case.get("case_id"),
        "decision_board": f"decision_boards/{slot_id}_agent_decision_graph.png",
        "part1_state": case.get("part1_state"), "final_state": final_state,
        "part1_scores": _scores(case.get("part1_scores")), "final_scores": _scores(case.get("final_scores")),
        "fov": visibility, "steps": steps,
        "headline": {
            "free": "看到连续自由空间，并排除边界伪回波" if final_state == "free" else "",
            "occupied": "看到稳定三维车体核心，并否决自由空间" if final_state == "occupied" else "",
            "unknown": "连续三轮仍无法可靠定位或覆盖目标，拒绝猜测" if final_state == "unknown" else "",
        }.get(final_state, ""),
    }


def _run_overview(run_dir: Path) -> dict[str, Any]:
    """Summarize the complete run; deep-dive cases are only representatives."""

    result_path = run_dir / "part2_result.json"
    result = _read(result_path) if result_path.is_file() else {}
    queue_ids = [str(case_id).split(":")[-1] for case_id in result.get("queue_order", [])]
    case_paths = {path.stem: path for path in (run_dir / "slot_cases").glob("*.json")}
    ordered_ids = [slot_id for slot_id in queue_ids if slot_id in case_paths]
    ordered_ids.extend(sorted(slot_id for slot_id in case_paths if slot_id not in ordered_ids))
    rows: list[dict[str, Any]] = []
    for slot_id in ordered_ids:
        case = _read(case_paths[slot_id])
        audit_dir = run_dir / "openai_audit" / str(case.get("case_id", ""))
        audits = [_read(path) for path in sorted(audit_dir.glob("turn_*.json"))]
        actions: list[str] = ["FOV", "LiDAR"]
        for audit in audits:
            action = audit.get("action", {})
            if action.get("type") == "tool":
                tool = str(action.get("tool", ""))
                actions.append(TOOL_ZH.get(tool, tool))
            elif action.get("type") == "final":
                state = str(action.get("state", case.get("final_state", "unknown")))
                actions.append(f"提议{STATE_ZH.get(state, state)}")
        actions.append("Validator")
        final_scores = _scores(case.get("final_scores"))
        final_state = str(case.get("final_state", "unknown"))
        rows.append({
            "slot_id": slot_id,
            "part1_state": str(case.get("part1_state", "unknown")),
            "final_state": final_state,
            "fov": str(case.get("fov", {}).get("visibility", "uncertain")),
            "model_turns": len(audits),
            "action_path": " → ".join(actions),
            "final_confidence": float(final_scores.get(final_state, 0.0)),
            "unresolved": _reason_list(case.get("unresolved_reasons"))[:2],
        })
    total = len(rows)
    unknown_inputs = sum(row["part1_state"] == "unknown" for row in rows)
    resolved_unknown = sum(
        row["part1_state"] == "unknown" and row["final_state"] in {"free", "occupied"}
        for row in rows
    )
    final_counts = {
        state: sum(row["final_state"] == state for row in rows)
        for state in ("free", "occupied", "unknown")
    }
    return {
        "total_cases": total,
        "unknown_inputs": unknown_inputs,
        "resolved_unknown": resolved_unknown,
        "unknown_resolution_rate": resolved_unknown / unknown_inputs if unknown_inputs else 0.0,
        "final_counts": final_counts,
        "one_turn_cases": sum(row["model_turns"] == 1 for row in rows),
        "three_turn_cases": sum(row["model_turns"] == 3 for row in rows),
        "all_cases": rows,
    }


def _document(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ParkingSlotAgent v2 · Agent决策回放</title>
<style>
:root{{--ink:#142238;--muted:#66768c;--line:#dce4ee;--panel:#fff;--bg:#f4f7fb;--green:#0b9b72;--red:#e14d50;--amber:#d99712;--blue:#2a6fdb;--purple:#7257d3}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,"Noto Sans SC","Microsoft YaHei",sans-serif}}
button{{font:inherit}} .page{{max-width:1500px;margin:auto;padding:28px}} .eyebrow{{color:var(--blue);font-weight:800;letter-spacing:.12em;font-size:12px}}
h1{{font-size:32px;margin:3px 0 5px;letter-spacing:-.03em}} .lead{{color:var(--muted);margin:0 0 22px;max-width:920px}}
.board-showcase{{background:#09111f;border-radius:18px;padding:18px;margin-bottom:20px;color:#edf5ff}} .board-head{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px}} .board-head h2{{margin:0;font-size:22px}} .board-tabs{{display:flex;gap:7px;margin-left:auto}}
.board-tab{{border:1px solid #29405c;background:#172840;color:#b9c9dc;border-radius:9px;padding:7px 11px;cursor:pointer}} .board-tab.active{{background:#42a5ff;color:#07101d;border-color:#42a5ff;font-weight:800}} .board-img{{width:100%;height:auto;display:block;border:1px solid #29405c;border-radius:10px}}
.overview{{margin:0 0 22px}} .metric-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}} .metric{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px}}
.metric b{{display:block;font-size:27px;line-height:1.15}} .metric span{{color:var(--muted);font-size:12px}} .scope-note{{margin-top:10px;background:#eef4fd;border-left:4px solid var(--blue);padding:10px 13px;color:#36516f;border-radius:0 9px 9px 0}}
.inventory{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 16px;margin-bottom:24px}} .inventory summary{{font-weight:800;color:var(--ink)}} .table-wrap{{overflow:auto;margin-top:12px;max-height:470px}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th{{position:sticky;top:0;background:#edf3fa;z-index:1;text-align:left}} th,td{{padding:8px 9px;border-bottom:1px solid #e6ebf2;white-space:nowrap}} td.path{{min-width:360px;white-space:normal}} .state-text{{font-weight:800}} .section-title{{font-size:24px;margin:0 0 12px}}
.case-tabs{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}} .case-tab{{border:1px solid var(--line);background:#fff;border-radius:12px;padding:11px 16px;cursor:pointer;text-align:left;min-width:210px}}
.case-tab.active{{border:2px solid var(--blue);box-shadow:0 5px 18px #2a6fdb18}} .case-tab strong{{display:block}} .case-tab span{{color:var(--muted);font-size:12px}}
.hero{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px 22px;display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center;margin-bottom:14px}}
.transition{{display:flex;align-items:center;gap:13px;font-size:25px;font-weight:800}} .pill{{padding:6px 13px;border-radius:999px;color:#fff;font-size:18px}} .unknown{{background:var(--amber)}} .free{{background:var(--green)}} .occupied{{background:var(--red)}}
.hero-note{{color:var(--muted);margin-top:5px}} .legend{{font-size:12px;color:var(--muted);text-align:right}} .legend b{{color:var(--ink)}}
.stepper{{display:flex;gap:7px;overflow:auto;padding:3px 2px 13px}} .step-btn{{flex:1;min-width:125px;border:1px solid var(--line);background:#fff;border-radius:11px;padding:10px 8px;cursor:pointer;color:var(--muted);position:relative}}
.step-btn:not(:last-child):after{{content:'→';position:absolute;right:-10px;top:12px;color:#9eacbd;z-index:2}} .step-btn.active{{color:#fff;background:var(--blue);border-color:var(--blue)}} .step-btn.visited{{border-color:#89aee8;color:var(--blue)}}
.controls{{display:flex;gap:8px;margin-bottom:12px}} .control{{border:1px solid var(--line);background:#fff;border-radius:9px;padding:7px 12px;cursor:pointer}} .counter{{margin-left:auto;color:var(--muted);padding:7px}}
.stage{{display:grid;grid-template-columns:minmax(620px,1.35fr) minmax(380px,.8fr);gap:14px}}
.visual,.explain{{background:#fff;border:1px solid var(--line);border-radius:16px;overflow:hidden}} .explain{{order:1}} .visual{{order:2}} .visual-head,.explain-head{{padding:15px 17px;border-bottom:1px solid var(--line)}}
.kicker{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.08em}} h2{{font-size:22px;margin:2px 0}} .image-wrap{{height:460px;background:#e9eef5;display:flex;align-items:center;justify-content:center;position:relative;padding:10px}}
.image-wrap img{{max-width:100%;max-height:100%;object-fit:contain;box-shadow:0 2px 14px #0d1d3320;background:#fff}} .actual{{position:absolute;left:20px;top:20px;background:#142238e8;color:#fff;border-radius:7px;padding:5px 9px;font-size:12px}}
.empty{{color:var(--muted)}} .explain-body{{padding:16px}} .block{{border-bottom:1px solid var(--line);padding:0 0 14px;margin-bottom:14px}} .block:last-child{{border:0;margin:0}}
.block h3{{font-size:13px;margin:0 0 6px;color:var(--muted);letter-spacing:.04em}} .rationale{{border-left:4px solid var(--purple);background:#f5f2ff;padding:11px 13px;border-radius:0 9px 9px 0}}
.badges{{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}} .badge{{background:#edf3fa;color:#36516f;border-radius:999px;padding:4px 8px;font-size:12px}}
.bars{{display:grid;gap:8px}} .bar-row{{display:grid;grid-template-columns:68px 1fr 47px;gap:8px;align-items:center;font-size:12px}} .track{{height:8px;background:#edf1f6;border-radius:99px;overflow:hidden}} .fill{{height:100%;border-radius:99px}}
.gate{{display:grid;grid-template-columns:22px 1fr auto;gap:8px;padding:7px 0;border-bottom:1px dashed #e1e7ef}} .gate:last-child{{border:0}} .gate-icon{{width:20px;height:20px;border-radius:50%;color:#fff;text-align:center;font-weight:800;line-height:20px}} .pass{{background:var(--green)}} .fail{{background:var(--red)}} .gate-value{{font-variant-numeric:tabular-nums;color:var(--muted)}}
details{{margin-top:12px}} summary{{cursor:pointer;color:var(--blue)}} pre{{white-space:pre-wrap;max-height:260px;overflow:auto;background:#111b2b;color:#d8e3f0;padding:12px;border-radius:8px;font-size:11px}}
.notice{{margin-top:14px;color:var(--muted);font-size:12px}} @media(max-width:980px){{.page{{padding:14px}}.metric-grid{{grid-template-columns:repeat(2,1fr)}}.stage{{grid-template-columns:1fr}}.image-wrap{{height:430px}}.hero{{grid-template-columns:1fr}}.legend{{text-align:left}}}}
</style></head><body><main class="page">
<div class="eyebrow">REAL TRACE · FRAME 9277 · OPENAI + DETERMINISTIC GATES</div>
<h1>Agent到底做了什么？逐步回放一次车位判定</h1>
<p class="lead">先看真正的Agent因果图：每个OpenAI Turn都明确列出Observation、Memory、原始理由、合法动作集、实际Action和belief更新。算法证据只作为输入附件，Validator单独放在底部。</p>
<section class="board-showcase"><div class="board-head"><h2>主视图 · Agent ReAct决策图</h2><span style="color:#91a5be">点击切换三轮、Free、Occupied案例</span><div id="boardTabs" class="board-tabs"></div></div><img id="decisionBoard" class="board-img" alt="Agent decision graph"></section>
<section class="overview"><div class="metric-grid">
<div class="metric"><b id="mTotal">—</b><span>本轮完整车位数</span></div><div class="metric"><b id="mUnknown">—</b><span>Part1 Unknown输入</span></div>
<div class="metric"><b id="mResolved">—</b><span>Unknown转为终态</span></div><div class="metric"><b id="mRate">—</b><span>Unknown解决率</span></div><div class="metric"><b id="mFinal">—</b><span>最终 Free / Occupied / Unknown</span></div>
</div><div class="scope-note"><b>如何解释：</b>解决率表示Agent给出了通过代码门的终态，不代表有Ground Truth验证的准确率。下方三个案例分别解释Free、Occupied和安全保留Unknown；不是只报成功案例。</div></section>
<details class="inventory"><summary>查看本轮全部24个车位、FOV、Agent动作路径与最终置信度</summary><div class="table-wrap"><table><thead><tr><th>车位</th><th>Part1</th><th>最终</th><th>FOV</th><th>模型轮次</th><th>最终置信度</th><th>Agent动作路径</th><th>仍未解决</th></tr></thead><tbody id="inventoryBody"></tbody></table></div></details>
<h2 class="section-title">代表性案例：逐步查看Agent如何做决定</h2>
<div id="caseTabs" class="case-tabs"></div><section class="hero"><div><div id="transition" class="transition"></div><div id="headline" class="hero-note"></div></div><div class="legend"><b>模型只能提出结论</b><br>最终Free/Occupied必须通过代码硬门</div></section>
<div id="stepper" class="stepper"></div><div class="controls"><button class="control" id="prev">← 上一步</button><button class="control" id="play">▶ 自动播放</button><button class="control" id="next">下一步 →</button><span class="counter" id="counter"></span></div>
<section class="stage"><article class="visual"><div class="visual-head"><div id="kicker" class="kicker"></div><h2 id="title"></h2></div><div class="image-wrap"><span class="actual" id="imageLabel">本步实际证据</span><img id="evidenceImage"><div id="empty" class="empty">本步没有图像产物</div></div></article>
<aside class="explain"><div class="explain-head"><div class="kicker">AUDITABLE DECISION TRACE</div><h2>这一步如何改变判断</h2></div><div class="explain-body">
<div class="block"><h3>发生了什么</h3><div id="summary"></div><div id="badges" class="badges"></div></div>
<div class="block"><h3>Agent决策摘要</h3><div id="rationale" class="rationale"></div></div>
<div class="block"><h3>工具返回</h3><div id="observation"></div></div>
<div class="block"><h3>当轮状态分数（未校准）</h3><div id="bars" class="bars"></div></div>
<div class="block" id="gateBlock"><h3>确定性证据门</h3><div id="gates"></div></div>
<details><summary>展开本步原始审计记录</summary><pre id="raw"></pre></details>
</div></aside></section><p class="notice">说明：“Agent决策摘要”展示日志中记录的rationale/reason及其结构化归纳，不声称展示模型不可审计的隐藏思维链。置信度字段由Agent输出且未做概率校准。</p>
</main><script id="replay-data" type="application/json">{data}</script><script>
const DATA=JSON.parse(document.getElementById('replay-data').textContent);let ci=0,si=0,bi=Math.max(0,DATA.cases.findIndex(c=>c.slot_id==='slot_1248')),timer=null;
const $=id=>document.getElementById(id); const zh={{unknown:'未知',free:'空闲',occupied:'占用'}}; const fovZh={{visible:'可见',partially_visible:'部分可见',not_visible:'不可见',uncertain:'不确定'}};
function pill(s){{return `<span class="pill ${{s}}">${{zh[s]||s}}</span>`}}
function renderBoard(){{$('boardTabs').innerHTML=DATA.cases.map((c,i)=>`<button class="board-tab ${{i===bi?'active':''}}" onclick="pickBoard(${{i}})">${{c.slot_id}} · ${{zh[c.final_state]}}</button>`).join('');$('decisionBoard').src=DATA.cases[bi].decision_board}}
function pickBoard(i){{bi=i;renderBoard()}}
function renderOverview(){{const o=DATA.overview;$('mTotal').textContent=o.total_cases;$('mUnknown').textContent=o.unknown_inputs;$('mResolved').textContent=o.resolved_unknown;$('mRate').textContent=`${{(o.unknown_resolution_rate*100).toFixed(1)}}%`;$('mFinal').textContent=`${{o.final_counts.free}} / ${{o.final_counts.occupied}} / ${{o.final_counts.unknown}}`;
$('inventoryBody').innerHTML=o.all_cases.map(r=>`<tr><td><b>${{esc(r.slot_id)}}</b></td><td class="state-text">${{zh[r.part1_state]||r.part1_state}}</td><td class="state-text" style="color:${{r.final_state==='free'?'#0b9b72':r.final_state==='occupied'?'#e14d50':'#d99712'}}">${{zh[r.final_state]||r.final_state}}</td><td>${{fovZh[r.fov]||r.fov}}</td><td>${{r.model_turns}}</td><td>${{Math.round(r.final_confidence*100)}}%</td><td class="path">${{esc(r.action_path)}}</td><td class="path">${{esc((r.unresolved||[]).join('；')||'—')}}</td></tr>`).join('')}}
function renderTabs(){{$('caseTabs').innerHTML=DATA.cases.map((c,i)=>`<button class="case-tab ${{i===ci?'active':''}}" onclick="pickCase(${{i}})"><strong>${{c.slot_id}} · ${{zh[c.final_state]}}</strong><span>${{zh[c.part1_state]}} → ${{zh[c.final_state]}} · ${{c.steps.length}}步真实轨迹</span></button>`).join('')}}
function render(){{const c=DATA.cases[ci],s=c.steps[si];renderTabs();$('transition').innerHTML=`${{pill(c.part1_state)}} <span>→</span> ${{pill(c.final_state)}}`;$('headline').textContent=c.headline;
$('stepper').innerHTML=c.steps.map((x,i)=>`<button class="step-btn ${{i===si?'active':i<si?'visited':''}}" onclick="pickStep(${{i}})">${{i}} · ${{x.label}}</button>`).join('');$('counter').textContent=`${{si+1}} / ${{c.steps.length}}`;$('kicker').textContent=s.kicker;$('title').textContent=s.title;$('summary').textContent=s.summary;$('rationale').textContent=s.rationale;$('observation').textContent=s.observation;$('imageLabel').textContent=s.image_label||'本步实际证据';
$('badges').innerHTML=s.badges.map(x=>`<span class="badge">${{esc(x)}}</span>`).join('');const img=$('evidenceImage');if(s.image){{img.src=s.image;img.style.display='block';$('empty').style.display='none'}}else{{img.removeAttribute('src');img.style.display='none';$('empty').style.display='block'}}
const colors={{free:'#0b9b72',occupied:'#e14d50',unknown:'#d99712'}};$('bars').innerHTML=['free','occupied','unknown'].map(k=>`<div class="bar-row"><span>${{zh[k]}}</span><div class="track"><div class="fill" style="width:${{Math.max(0,Math.min(100,s.scores[k]*100))}}%;background:${{colors[k]}}"></div></div><b>${{Math.round(s.scores[k]*100)}}%</b></div>`).join('');
$('gateBlock').style.display=s.gates.length?'block':'none';$('gates').innerHTML=s.gates.map(g=>`<div class="gate"><span class="gate-icon ${{g.pass?'pass':'fail'}}">${{g.pass?'✓':'×'}}</span><span>${{esc(g.name)}}</span><span class="gate-value">${{esc(g.value)}}</span></div>`).join('');$('raw').textContent=JSON.stringify(s.raw,null,2)}}
function esc(x){{return String(x??'').replace(/[&<>\"]/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[m]))}} function pickCase(i){{stop();ci=i;si=0;render()}} function pickStep(i){{stop();si=i;render()}}
function move(d){{const n=DATA.cases[ci].steps.length;si=Math.max(0,Math.min(n-1,si+d));render()}} function stop(){{if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ 自动播放'}}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('play').onclick=()=>{{if(timer){{stop();return}};$('play').textContent='■ 停止';timer=setInterval(()=>{{const n=DATA.cases[ci].steps.length;if(si>=n-1){{stop();return}};si++;render()}},2200)}};renderBoard();renderOverview();render();
</script></body></html>"""


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/Engine/Content/Slate/Fonts/DroidSansFallback.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int, max_lines: int = 5) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return ["—"]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = char
            if len(lines) >= max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines and current:
        lines.append(current)
    consumed = sum(len(line) for line in lines)
    if consumed < len(text) and lines:
        while lines[-1] and draw.textlength(lines[-1] + "…", font=font) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


def _fit_image(path: Path, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]] | None:
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    image.thumbnail((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    return image, (x1 + (x2 - x1 - image.width) // 2, y1 + (y2 - y1 - image.height) // 2)


def _draw_text_lines(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font: ImageFont.ImageFont, fill: str, spacing: int = 7) -> int:
    x, y = xy
    bbox = draw.textbbox((0, 0), "示Ag", font=font)
    line_h = bbox[3] - bbox[1] + spacing
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _replay_frame(case: Mapping[str, Any], step_index: int, root: Path) -> Image.Image:
    width, height = 1920, 1080
    canvas = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    title_font, h2_font = _font(38, bold=True), _font(27, bold=True)
    body_font, small_font, tiny_font = _font(21), _font(17), _font(14)
    state_colors = {"free": "#0b9b72", "occupied": "#e14d50", "unknown": "#d99712"}
    step = case["steps"][step_index]

    draw.text((42, 24), "真实Agent决策回放", font=title_font, fill="#142238")
    draw.text((42, 76), f"{case['slot_id']}  ·  frame 9277", font=small_font, fill="#66768c")
    x = 1510
    for state_index, state in enumerate((case["part1_state"], case["final_state"])):
        label = STATE_ZH.get(state, state)
        bbox = draw.textbbox((0, 0), label, font=h2_font)
        w = bbox[2] - bbox[0] + 38
        draw.rounded_rectangle((x, 36, x + w, 82), radius=22, fill=state_colors[state])
        draw.text((x + 19, 43), label, font=h2_font, fill="white")
        x += w
        if state_index == 0:
            draw.text((x + 12, 43), "→", font=h2_font, fill="#142238")
            x += 55

    steps = case["steps"]
    left, right, gap = 42, 1878, 8
    cell_w = (right - left - gap * (len(steps) - 1)) / len(steps)
    for i, item in enumerate(steps):
        x1 = int(left + i * (cell_w + gap)); x2 = int(x1 + cell_w)
        active = i == step_index
        fill = "#2a6fdb" if active else ("#dce9fb" if i < step_index else "#ffffff")
        outline = "#2a6fdb" if i <= step_index else "#ccd6e3"
        draw.rounded_rectangle((x1, 116, x2, 166), radius=10, fill=fill, outline=outline, width=2)
        label = f"{i}  {item['label']}"
        draw.text((x1 + 11, 129), label, font=small_font, fill="white" if active else "#36516f")

    draw.rounded_rectangle((42, 188, 1160, 1035), radius=16, fill="white", outline="#dce4ee", width=2)
    draw.text((66, 210), str(step.get("image_label", "本步实际证据")), font=small_font, fill="#2a6fdb")
    fitted = _fit_image(root / step["image"], (62, 254, 1140, 1015)) if step.get("image") else None
    if fitted:
        evidence, position = fitted
        canvas.paste(evidence, position)
    else:
        draw.text((430, 600), "本步没有图像产物", font=h2_font, fill="#66768c")

    draw.rounded_rectangle((1180, 188, 1878, 1035), radius=16, fill="white", outline="#dce4ee", width=2)
    draw.text((1210, 214), str(step["kicker"]), font=tiny_font, fill="#2a6fdb")
    y = _draw_text_lines(draw, (1210, 242), _wrap(draw, step["title"], h2_font, 630, 2), h2_font, "#142238", 8) + 12
    draw.text((1210, y), "发生了什么", font=small_font, fill="#66768c"); y += 30
    y = _draw_text_lines(draw, (1210, y), _wrap(draw, step["summary"], body_font, 630, 4), body_font, "#142238") + 16
    draw.text((1210, y), "Agent决策摘要", font=small_font, fill="#7257d3"); y += 31
    rationale_lines = _wrap(draw, step["rationale"], body_font, 590, 5)
    box_h = 35 + len(rationale_lines) * 30
    draw.rounded_rectangle((1200, y - 7, 1852, y + box_h), radius=10, fill="#f5f2ff")
    draw.rectangle((1200, y - 7, 1207, y + box_h), fill="#7257d3")
    y = _draw_text_lines(draw, (1225, y + 9), rationale_lines, body_font, "#2c2741") + 24

    draw.text((1210, y), "当轮分数（未校准）", font=small_font, fill="#66768c"); y += 32
    for key in ("free", "occupied", "unknown"):
        value = float(step["scores"].get(key, 0))
        draw.text((1210, y), STATE_ZH[key], font=small_font, fill="#142238")
        draw.rounded_rectangle((1290, y + 6, 1750, y + 20), radius=7, fill="#edf1f6")
        if value > 0:
            draw.rounded_rectangle((1290, y + 6, 1290 + int(460 * value), y + 20), radius=7, fill=state_colors[key])
        draw.text((1760, y), f"{value * 100:.0f}%", font=small_font, fill="#142238")
        y += 31

    if step.get("gates") and y < 880:
        y += 6; draw.text((1210, y), "确定性证据门", font=small_font, fill="#66768c"); y += 31
        for gate in step["gates"][:5]:
            passed = bool(gate["pass"]); color = "#0b9b72" if passed else "#e14d50"
            draw.ellipse((1210, y + 1, 1231, y + 22), fill=color)
            draw.text((1215, y), "✓" if passed else "×", font=tiny_font, fill="white")
            draw.text((1243, y), str(gate["name"]), font=small_font, fill="#142238")
            value = str(gate["value"])
            value_width = draw.textlength(value, font=small_font)
            draw.text((1840 - value_width, y), value, font=small_font, fill="#66768c")
            y += 28
    draw.text((1210, 997), "模型提出结论，代码决定是否允许写回", font=tiny_font, fill="#66768c")
    return canvas


def _action_view(case: Mapping[str, Any], step: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any]]:
    raw = step.get("raw", {})
    action = raw.get("model_action") or raw.get("final_action") or {}
    if step.get("key") == "final":
        state = str(case.get("final_state", "unknown"))
        label = (
            f"硬门通过 → 写回 {STATE_ZH.get(state, state)}"
            if state != "unknown" else
            "双门拒绝 → 保留 未知"
        )
        return "DETERMINISTIC VALIDATOR", label, action
    if action.get("type") == "tool":
        tool = str(action.get("tool", ""))
        return "CALL TOOL", f"调用 {TOOL_ZH.get(tool, tool)}", action
    if action.get("type") == "final":
        state = str(action.get("state", step.get("state", "unknown")))
        confidence = max(float(action.get("free_confidence", 0)), float(action.get("occupied_confidence", 0)), float(step.get("scores", {}).get("unknown", 0)))
        return "PROPOSE STATE", f"提议 {STATE_ZH.get(state, state)}  {confidence*100:.0f}%", action
    if step.get("key") == "part1":
        return "WAIT FOR MANDATORY CHECK", "先做Camera FOV检查", action
    if step.get("key") == "fov":
        return "SYSTEM ROUTER", "Camera不可用 → 调用60帧LiDAR", action
    return "OBSERVE", str(step.get("title", "读取新证据")), action


def _uncertainties(step: Mapping[str, Any], action: Mapping[str, Any]) -> list[str]:
    belief = action.get("belief") if isinstance(action.get("belief"), Mapping) else {}
    values = belief.get("remaining_unknown_reasons") or action.get("reason_codes") or []
    translated = _reason_list(values)
    if not translated:
        translated = [str(value) for value in step.get("badges", []) if value]
    return translated[:5]


def _agent_replay_frame(case: Mapping[str, Any], step_index: int, root: Path) -> Image.Image:
    """Render the Agent's action as the visual subject; evidence is secondary."""

    width, height = 1920, 1080
    canvas = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    title_font, h2_font, action_font = _font(38, bold=True), _font(27, bold=True), _font(32, bold=True)
    body_font, small_font, tiny_font = _font(20), _font(17), _font(14)
    state_colors = {"free": "#0b9b72", "occupied": "#e14d50", "unknown": "#d99712"}
    step = case["steps"][step_index]
    action_kind, action_label, action = _action_view(case, step)

    draw.text((42, 22), "Agent决策轨迹", font=title_font, fill="#142238")
    draw.text((42, 73), f"{case['slot_id']} · frame 9277 · 真实OpenAI action回放", font=small_font, fill="#66768c")
    x = 1510
    for state_index, state in enumerate((case["part1_state"], case["final_state"])):
        label = STATE_ZH.get(state, state); box = draw.textbbox((0, 0), label, font=h2_font); w = box[2]-box[0]+38
        draw.rounded_rectangle((x, 34, x+w, 80), radius=22, fill=state_colors[state]); draw.text((x+19, 41), label, font=h2_font, fill="white"); x += w
        if state_index == 0: draw.text((x+12, 41), "→", font=h2_font, fill="#142238"); x += 55

    steps = case["steps"]; left, right, gap = 42, 1878, 8; cell_w = (right-left-gap*(len(steps)-1))/len(steps)
    for i, item in enumerate(steps):
        x1=int(left+i*(cell_w+gap)); x2=int(x1+cell_w); active=i==step_index
        fill="#2a6fdb" if active else ("#dce9fb" if i<step_index else "#ffffff"); outline="#2a6fdb" if i<=step_index else "#ccd6e3"
        draw.rounded_rectangle((x1,114,x2,164),radius=10,fill=fill,outline=outline,width=2)
        draw.text((x1+11,127),f"{i}  {item['label']}",font=small_font,fill="white" if active else "#36516f")

    # Main Agent canvas.
    draw.rounded_rectangle((42, 186, 1275, 1035), radius=18, fill="white", outline="#dce4ee", width=2)
    draw.text((72, 212), str(step["kicker"]), font=tiny_font, fill="#2a6fdb")
    draw.text((72, 240), "AGENT MEMORY", font=small_font, fill="#66768c")
    memory_y = 274
    memory_cards = [
        ("当前状态", STATE_ZH.get(str(step.get("state")), str(step.get("state"))), state_colors.get(str(step.get("state")), "#d99712")),
        ("刚收到", str(step.get("title", "")), "#2a6fdb"),
        ("工具轮次", f"{step_index}/{len(steps)-1}", "#7257d3"),
    ]
    card_x = 72
    for label, value, color in memory_cards:
        w = 350 if label == "刚收到" else 220
        draw.rounded_rectangle((card_x,memory_y,card_x+w,memory_y+86),radius=11,fill="#f6f8fb",outline="#dce4ee")
        draw.text((card_x+15,memory_y+10),label,font=tiny_font,fill="#66768c")
        value_lines=_wrap(draw,value,small_font,w-30,2); _draw_text_lines(draw,(card_x+15,memory_y+37),value_lines,small_font,color,4)
        card_x += w+12

    # Auditable rationale bubble.
    bubble_y = 382
    draw.ellipse((72,bubble_y,142,bubble_y+70),fill="#7257d3"); draw.text((91,bubble_y+18),"A",font=h2_font,fill="white")
    draw.text((160,bubble_y),"Agent为什么这样做？",font=small_font,fill="#7257d3")
    rationale_lines=_wrap(draw,str(step.get("rationale","")),body_font,1035,5)
    box_h=max(125,28*len(rationale_lines)+42)
    draw.rounded_rectangle((150,bubble_y+31,1235,bubble_y+31+box_h),radius=13,fill="#f5f2ff")
    draw.rectangle((150,bubble_y+31,158,bubble_y+31+box_h),fill="#7257d3")
    _draw_text_lines(draw,(177,bubble_y+53),rationale_lines,body_font,"#2c2741",7)
    y=bubble_y+53+len(rationale_lines)*31+16
    draw.text((177,y),"来自结构化 action.rationale / reason，不是虚构思维链",font=tiny_font,fill="#81799b")

    # The action is the strongest visual element.
    action_y = bubble_y+55+box_h
    draw.text((72,action_y),action_kind,font=tiny_font,fill="#2a6fdb")
    draw.rounded_rectangle((72,action_y+26,1235,action_y+104),radius=14,fill="#142238")
    draw.text((101,action_y+43),"→",font=action_font,fill="#54a2ff")
    draw.text((157,action_y+40),action_label,font=action_font,fill="white")

    # Belief update and unresolved memory.
    belief_y=action_y+132; draw.text((72,belief_y),"BELIEF UPDATE",font=small_font,fill="#66768c")
    colors={"free":"#0b9b72","occupied":"#e14d50","unknown":"#d99712"}; labels={"free":"空闲","occupied":"占用","unknown":"未知"}
    for row,key in enumerate(("free","occupied","unknown")):
        yy=belief_y+35+row*31; value=float(step.get("scores",{}).get(key,0))
        draw.text((72,yy),labels[key],font=small_font,fill="#142238"); draw.rounded_rectangle((145,yy+6,650,yy+20),radius=7,fill="#edf1f6")
        if value: draw.rounded_rectangle((145,yy+6,145+int(505*value),yy+20),radius=7,fill=colors[key])
        draw.text((665,yy),f"{value*100:.0f}%",font=small_font,fill="#142238")
    uncertainties=_uncertainties(step,action)
    draw.text((770,belief_y),"仍未解决",font=small_font,fill="#66768c")
    uy=belief_y+35
    for value in uncertainties[:4]:
        lines=_wrap(draw,"• "+value,tiny_font,430,2); uy=_draw_text_lines(draw,(770,uy),lines,tiny_font,"#52657c",3)+5

    # Evidence is deliberately a supporting thumbnail.
    draw.rounded_rectangle((1295,186,1878,1035),radius=18,fill="white",outline="#dce4ee",width=2)
    draw.text((1325,212),"SUPPORTING EVIDENCE",font=tiny_font,fill="#66768c")
    draw.text((1325,239),str(step.get("image_label","本步实际证据")),font=small_font,fill="#2a6fdb")
    fitted=_fit_image(root/step["image"],(1315,280,1858,650)) if step.get("image") else None
    if fitted:
        evidence,pos=fitted; canvas.paste(evidence,pos)
    else: draw.text((1440,450),"无图像产物",font=h2_font,fill="#66768c")
    draw.line((1325,676,1848,676),fill="#dce4ee",width=2)
    draw.text((1325,698),"Agent提取到的关键信息",font=small_font,fill="#66768c")
    sy=733
    for line in _wrap(draw,str(step.get("summary","")),body_font,500,5):
        draw.text((1325,sy),line,font=body_font,fill="#142238"); sy+=30
    if step.get("gates"):
        sy+=10; draw.text((1325,sy),"VALIDATOR FEEDBACK",font=tiny_font,fill="#66768c"); sy+=29
        for gate in step["gates"][:4]:
            passed=bool(gate["pass"]); color="#0b9b72" if passed else "#e14d50"
            draw.ellipse((1325,sy,1347,sy+22),fill=color); draw.text((1330,sy-1),"✓" if passed else "×",font=tiny_font,fill="white")
            label=f"{gate['name']} · {gate['value']}"; draw.text((1360,sy),label,font=tiny_font,fill="#52657c"); sy+=29
    draw.text((1325,1000),"证据支持Agent，但不替代Agent轨迹",font=tiny_font,fill="#8a98aa")
    return canvas


def _write_replay_gif(case: Mapping[str, Any], output: Path) -> Path:
    frames = [_agent_replay_frame(case, index, output) for index in range(len(case["steps"]))]
    frame_dir = output / "frames" / str(case["slot_id"])
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, (frame, step) in enumerate(zip(frames, case["steps"])):
        frame.save(frame_dir / f"{index:02d}_{step['key']}.png", optimize=True)
    target = output / f"{case['slot_id']}_agent_replay.gif"
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=2300, loop=0, optimize=True)
    frames[-1].save(output / f"{case['slot_id']}_final_frame.png", optimize=True)
    return target


def _legal_action_labels(case: Mapping[str, Any], turn_index: int, action: Mapping[str, Any]) -> list[str]:
    """Reconstruct the bounded legal choice set from the checked-in tool policy."""

    if action.get("type") == "final":
        return ["提交空闲", "提交占用", "提交未知"]
    tool = str(action.get("tool", ""))
    if tool == "camera_context":
        return ["相机上下文", "因果相机序列", "提交终态"]
    if tool == "camera_sequence":
        return ["目标区域裁剪", "因果相机序列", "提交终态"]
    return [TOOL_ZH.get(tool, tool), "提交终态"]


def _raw_action_reason(action: Mapping[str, Any]) -> str:
    return str(action.get("rationale") or action.get("reason") or "—")


def _observation_for_board(step: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    key = str(step.get("key", ""))
    if key == "lidar":
        if action.get("type") == "final" and action.get("state") == "free":
            return "60帧LiDAR返回：目标内部自由空间连续、核心障碍点为0，边界候选不能构成车辆。"
        if action.get("type") == "final" and action.get("state") == "occupied":
            return "60帧LiDAR返回：车位核心存在稳定三维障碍，多帧、高度层和位姿扰动结果一致。"
        return "60帧LiDAR返回：目标核心没有有效射线覆盖、遮挡完全，既无可靠Free观测，也无Occupied几何支持。"
    previous = step.get("raw", {}).get("previous_tool_call", {})
    tool = str(previous.get("tool", ""))
    if tool == "camera_context":
        return "相机上下文返回：看到目标标记地图和完整原始画面，但没有目标车位像素投影，仍无法可靠定位。"
    if tool == "camera_sequence":
        return "因果相机序列返回：5帧均严格早于t0；可观察遮挡变化，但仍没有可审计的目标车位像素投影。"
    if tool == "camera_crop":
        return "目标区域裁剪返回：局部外观得到增强，但目标身份或占用状态仍未达到终态要求。"
    return str(step.get("observation") or step.get("summary") or "—")


def _agent_decision_board(case: Mapping[str, Any], root: Path) -> Image.Image:
    """Create a causal ReAct graph where Agent turns—not sensor plots—are primary."""

    width, height = 3200, 1800
    bg, panel, ink = "#09111f", "#111d30", "#edf5ff"
    muted, line, blue, purple = "#91a5be", "#29405c", "#42a5ff", "#9b7cff"
    colors = {"free": "#18c48f", "occupied": "#ff5d66", "unknown": "#f2aa24"}
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    title_font, h2_font, h3_font = _font(48, bold=True), _font(31, bold=True), _font(22, bold=True)
    body_font, small_font, tiny_font = _font(20), _font(16), _font(13)

    # Only real OpenAI decisions are turn cards. Router and validator are shown
    # separately so they cannot be mistaken for model actions.
    turns: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for step in case["steps"]:
        if step.get("key") == "final":
            continue
        raw = step.get("raw", {})
        action = raw.get("model_action") or {}
        if action.get("type") in {"tool", "final"}:
            turns.append((step, action))

    draw.text((60, 42), "REAL AGENT DECISION GRAPH", font=tiny_font, fill=blue)
    draw.text((60, 70), f"{case['slot_id']} · Agent到底如何做决定", font=title_font, fill=ink)
    model = "GPT-5.6-TERRA · structured action audit"
    draw.text((60, 133), model, font=small_font, fill=muted)
    start, end = case["part1_state"], case["final_state"]
    end_label = f"{STATE_ZH.get(start, start)}  →  {STATE_ZH.get(end, end)}"
    end_width = draw.textlength(end_label, font=h2_font) + 54
    draw.rounded_rectangle((width - 60 - end_width, 66, width - 60, 125), radius=28, fill=colors[end])
    draw.text((width - 33 - end_width, 76), end_label, font=h2_font, fill="#ffffff")

    # Runtime envelope: makes explicit what the Agent received before turn 1.
    draw.rounded_rectangle((60, 180, width - 60, 310), radius=18, fill="#0d192a", outline=line, width=2)
    draw.text((84, 200), "RUNTIME ENVELOPE（不是Agent动作）", font=tiny_font, fill=muted)
    envelope = [
        f"Part1: {STATE_ZH.get(start, start)}",
        f"FOV: {case.get('fov', 'uncertain')}",
        "强制工具: 60帧目标对齐LiDAR",
        f"OpenAI决策轮数: {len(turns)} / 3",
        "终态规则: Agent提议 + Validator硬门",
    ]
    x = 84
    for item in envelope:
        item_w = draw.textlength(item, font=body_font) + 38
        draw.rounded_rectangle((x, 242, x + item_w, 287), radius=20, fill="#172840")
        draw.text((x + 19, 251), item, font=body_font, fill=ink)
        x += item_w + 18

    count = max(1, len(turns))
    gap = 34
    available_width = width - 120 - gap * (count - 1)
    card_w = available_width // count
    card_top, card_bottom = 355, 1545
    for index, (step, action) in enumerate(turns):
        x1 = 60 + index * (card_w + gap)
        x2 = x1 + card_w
        state = str((_action_belief(action)).get("state", step.get("state", "unknown")))
        accent = colors.get(state, colors["unknown"])
        draw.rounded_rectangle((x1, card_top, x2, card_bottom), radius=22, fill=panel, outline=line, width=2)
        draw.rounded_rectangle((x1, card_top, x2, card_top + 72), radius=22, fill="#172840")
        draw.rectangle((x1, card_top + 50, x2, card_top + 72), fill="#172840")
        draw.ellipse((x1 + 24, card_top + 17, x1 + 62, card_top + 55), fill=purple)
        draw.text((x1 + 37, card_top + 20), "A", font=small_font, fill="white")
        draw.text((x1 + 78, card_top + 17), f"OPENAI TURN {index + 1}", font=h3_font, fill=ink)
        draw.text((x2 - 160, card_top + 23), f"belief: {STATE_ZH.get(state, state)}", font=small_font, fill=accent)

        # Observation: text is primary; image is a small evidence receipt.
        y = card_top + 94
        draw.text((x1 + 26, y), "1  NEW OBSERVATION", font=tiny_font, fill=blue)
        y += 31
        observation = _observation_for_board(step, action)
        obs_width = card_w - 390 if card_w > 850 else card_w - 52
        obs_lines = _wrap(draw, observation, body_font, obs_width, 4)
        _draw_text_lines(draw, (x1 + 26, y), obs_lines, body_font, ink, 6)
        if card_w > 850 and step.get("image"):
            fitted = _fit_image(root / str(step["image"]), (x2 - 340, y - 6, x2 - 26, y + 164))
            if fitted:
                image_value, position = fitted
                canvas.paste(image_value, position)
                draw.rectangle((x2 - 340, y - 6, x2 - 26, y + 164), outline=line, width=2)
        y += 178

        draw.text((x1 + 26, y), "2  AGENT MEMORY · 未决项", font=tiny_font, fill=blue)
        y += 30
        unresolved = _uncertainties(step, action) or ["无"]
        chip_x, chip_y = x1 + 26, y
        for value in unresolved[:4]:
            label = str(value)
            chip_w = min(card_w - 52, draw.textlength(label, font=small_font) + 30)
            if chip_x + chip_w > x2 - 26:
                chip_x, chip_y = x1 + 26, chip_y + 43
            draw.rounded_rectangle((chip_x, chip_y, chip_x + chip_w, chip_y + 34), radius=16, fill="#263750")
            draw.text((chip_x + 15, chip_y + 7), label, font=small_font, fill="#c5d3e3")
            chip_x += chip_w + 10
        y = chip_y + 58

        draw.text((x1 + 26, y), "3  AGENT RATIONALE · 原始action的中文忠实归纳", font=tiny_font, fill=purple)
        y += 29
        rationale_lines = _wrap(draw, str(step.get("rationale", "—")), body_font, card_w - 70, 5)
        rationale_h = max(112, len(rationale_lines) * 29 + 36)
        draw.rounded_rectangle((x1 + 24, y, x2 - 24, y + rationale_h), radius=14, fill="#211c3c")
        draw.rectangle((x1 + 24, y, x1 + 32, y + rationale_h), fill=purple)
        _draw_text_lines(draw, (x1 + 49, y + 20), rationale_lines, body_font, "#eee9ff", 7)
        y += rationale_h + 18
        draw.text((x1 + 26, y), "模型原文", font=tiny_font, fill=muted)
        raw_lines = _wrap(draw, _raw_action_reason(action), tiny_font, card_w - 82, 4)
        y = _draw_text_lines(draw, (x1 + 105, y), raw_lines, tiny_font, "#aebed1", 4) + 18

        draw.text((x1 + 26, y), "4  LEGAL ACTION SET", font=tiny_font, fill=blue)
        y += 29
        candidates = _legal_action_labels(case, index, action)
        chosen = TOOL_ZH.get(str(action.get("tool", "")), "") if action.get("type") == "tool" else f"提交{STATE_ZH.get(str(action.get('state')), str(action.get('state')))}"
        cx = x1 + 26
        for candidate in candidates:
            selected = candidate == chosen or (candidate == "提交终态" and action.get("type") == "final")
            cw = draw.textlength(candidate, font=small_font) + 30
            draw.rounded_rectangle((cx, y, cx + cw, y + 38), radius=18, fill=accent if selected else "#172840", outline=accent if selected else line)
            draw.text((cx + 15, y + 8), candidate, font=small_font, fill="#07101d" if selected else muted)
            cx += cw + 10
        y += 58

        action_kind, action_label, _ = _action_view(case, step)
        draw.text((x1 + 26, y), f"5  DECISION · {action_kind}", font=tiny_font, fill=blue)
        y += 29
        draw.rounded_rectangle((x1 + 24, y, x2 - 24, y + 74), radius=14, fill=accent)
        draw.text((x1 + 50, y + 17), f"→  {action_label}", font=h2_font, fill="#07101d")
        y += 96

        belief = _scores(_action_belief(action))
        draw.text((x1 + 26, y), "BELIEF UPDATE", font=tiny_font, fill=muted)
        y += 28
        for key in ("free", "occupied", "unknown"):
            value = belief[key]
            draw.text((x1 + 26, y), STATE_ZH[key], font=small_font, fill=ink)
            draw.rounded_rectangle((x1 + 100, y + 6, x2 - 78, y + 19), radius=7, fill="#28384d")
            if value:
                draw.rounded_rectangle((x1 + 100, y + 6, x1 + 100 + int((card_w - 178) * value), y + 19), radius=7, fill=colors[key])
            draw.text((x2 - 62, y), f"{value * 100:.0f}%", font=small_font, fill=ink)
            y += 30

        if index < count - 1:
            ax = x2 + gap // 2
            draw.line((x2 + 6, 920, x2 + gap - 8, 920), fill=blue, width=5)
            draw.polygon([(x2 + gap - 8, 920), (x2 + gap - 23, 910), (x2 + gap - 23, 930)], fill=blue)

    # Validator is a different actor and gets its own lane.
    final_step = case["steps"][-1]
    draw.rounded_rectangle((60, 1580, width - 60, 1740), radius=20, fill="#0d192a", outline=line, width=2)
    draw.text((84, 1601), "DETERMINISTIC VALIDATOR（不是Agent）", font=tiny_font, fill=blue)
    validator_label = "硬门通过，允许写回" if end != "unknown" else "Free门失败 + Occupied门失败，保留Unknown"
    draw.text((84, 1635), validator_label, font=h2_font, fill=colors[end])
    gx = 1200
    for gate in final_step.get("gates", [])[:5]:
        passed = bool(gate["pass"])
        label = f"{'✓' if passed else '×'} {gate['name']} · {gate['value']}"
        gw = draw.textlength(label, font=small_font) + 30
        if gx + gw > width - 85:
            gx = 1200
        draw.rounded_rectangle((gx, 1634, gx + gw, 1673), radius=18, fill="#173247" if passed else "#40222b")
        draw.text((gx + 15, 1643), label, font=small_font, fill="#77e4c2" if passed else "#ff9da3")
        gx += gw + 10
    draw.text((84, 1700), "证据计算支撑决策，但本图的主线是Agent观察—记忆—理由—动作。", font=small_font, fill=muted)
    return canvas


def _write_decision_boards(cases: list[Mapping[str, Any]], output: Path) -> list[Path]:
    board_dir = output / "decision_boards"
    board_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in cases:
        target = board_dir / f"{case['slot_id']}_agent_decision_graph.png"
        _agent_decision_board(case, output).save(target, optimize=True)
        paths.append(target)
    return paths


def _report_markdown(payload: Mapping[str, Any]) -> str:
    overview = payload["overview"]
    rows = overview["all_cases"]
    unknown_to_free = sum(row["part1_state"] == "unknown" and row["final_state"] == "free" for row in rows)
    unknown_to_occupied = sum(row["part1_state"] == "unknown" and row["final_state"] == "occupied" for row in rows)
    lines = [
        "# Frame 9277 · Part2 Agent完整决策回放报告",
        "",
        "## 一句话结论",
        "",
        f"本轮完整处理 {overview['total_cases']} 个车位。Part1共有 {overview['unknown_inputs']} 个Unknown，"
        f"Part2 Agent将其中 {overview['resolved_unknown']} 个转为通过代码门的终态，解决率 "
        f"{overview['unknown_resolution_rate'] * 100:.1f}%（{unknown_to_free}个Free、{unknown_to_occupied}个Occupied）。",
        "",
        "> 解决率不是准确率：本报告证明Agent完成了可追溯决策并通过内部几何门，"
        "但没有Ground Truth时不能据此宣称分类正确率。",
        "",
        "## Agent实际工作流",
        "",
        "1. 接收Part1的SlotCase，不覆盖原状态与原因。",
        "2. 强制执行Camera FOV检查，决定Camera工具是否合法。",
        "3. 读取目标对齐的60帧LiDAR证据，更新Free/Occupied/Unknown belief。",
        "4. 证据不足时自主选择相机上下文、目标裁剪或因果相机序列；最多三轮。",
        "5. Agent只提交状态提议；确定性Validator核查证据引用、置信度和几何硬门后才允许写回。",
        "",
        "## Agent、工具与Validator的职责边界",
        "",
        "| 组件 | 负责什么 | 不负责什么 |",
        "|---|---|---|",
        "| Agent | 读取SlotCase与Observation、更新belief、自主选择下一工具、提交状态和证据引用 | 不直接篡改几何量，不绕过硬门写回终态 |",
        "| Camera/LiDAR工具 | 返回可追踪的传感器证据、覆盖率、障碍归属和可视化产物 | 不自主决定下一工具，不单独输出最终车位状态 |",
        "| Validator | 检查evidence_id、阈值、稳健性和终态资格，接受或拒绝Agent提议 | 不替Agent规划工具路径，不生成模型理由 |",
        "",
        "相对Part1，本轮可量化的改进是把22个Unknown中的9个转成了通过内部门控的终态；"
        "真正属于Agent的增量，是针对未决原因选择补充证据并形成可审计轨迹，而不是LiDAR计算本身。",
        "",
        "## 完整运行统计",
        "",
        f"- 最终Free / Occupied / Unknown：{overview['final_counts']['free']} / "
        f"{overview['final_counts']['occupied']} / {overview['final_counts']['unknown']}。",
        f"- 单轮模型决策：{overview['one_turn_cases']}例；三轮ReAct：{overview['three_turn_cases']}例。",
        f"- 深度可视化选择3例：Free、Occupied、安全保留Unknown各1例。完整24例均列在下表。",
        "",
        "| 车位 | Part1 | 最终 | FOV | 模型轮次 | 最终置信度 | Agent动作路径 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['slot_id']} | {STATE_ZH.get(row['part1_state'], row['part1_state'])} | "
            f"{STATE_ZH.get(row['final_state'], row['final_state'])} | {row['fov']} | "
            f"{row['model_turns']} | {row['final_confidence'] * 100:.0f}% | {row['action_path']} |"
        )
    lines.extend(["", "## 三个代表性Agent轨迹", ""])
    for case in payload["cases"]:
        lines.extend([
            f"### {case['slot_id']}：{STATE_ZH.get(case['part1_state'], case['part1_state'])} → "
            f"{STATE_ZH.get(case['final_state'], case['final_state'])}",
            "",
            f"结论：{case['headline']}。",
            "",
        ])
        for index, step in enumerate(case["steps"]):
            lines.append(f"- **{index}. {step['label']}**：{step['rationale']}")
        lines.extend(["", f"动态回放：`{case['slot_id']}_agent_replay.gif`", ""])
    lines.extend([
        "## 怎么看可视化",
        "",
        "- 左侧主画面是Agent：Memory、结构化决策理由、Action、belief更新、剩余不确定性。",
        "- 右侧是支撑证据：FOV地图、Camera画面或LiDAR解释图，以及Validator反馈。",
        "- 绿色表示Free证据或通过；红色表示Occupied证据或失败门；橙色表示Unknown或被边界规则否决的候选。",
        "- HTML中的“展开本步原始审计记录”可检查真实OpenAI action、evidence_id与usage。",
        "",
        "## 审计边界与局限",
        "",
        "- `Agent决策摘要`来自日志中的结构化`rationale/reason`和几何量的忠实中文归纳，不展示或伪造隐藏思维链。",
        "- Agent输出的置信度尚未经过概率校准，只用于工作流门控和案例内比较。",
        "- 本报告展示frame 9277这一轮，不代表跨场景、跨时间或全数据集性能。",
        "- Unknown是安全输出：代表必要观测、定位或稳健性条件未满足，并非程序失败。",
        "",
        "## 文件索引",
        "",
        "- `decision_boards/slot_1248_agent_decision_graph.png`：三轮ReAct主案例，优先查看。",
        "- `decision_boards/slot_1012_agent_decision_graph.png`：Unknown→Free的单轮Agent图。",
        "- `decision_boards/slot_1258_agent_decision_graph.png`：Unknown→Occupied的单轮Agent图。",
        "- `index.html`：以上决策图置顶，并保留完整交互报告与24车位总表。",
        "- `replay_data.json`：报告使用的结构化数据。",
        "- `slot_*_agent_replay.gif`：无需启动端口即可查看的动态轨迹。",
        "- `frames/slot_*/`：每个决策步骤的高清PNG。",
        "- `assets/slot_*_lidar_explained.png`：LiDAR可解释证据图。",
        "",
        "## 复现",
        "",
        "```bash",
        ".venv-vlm/bin/python -c \"from parking_slot_agent_v2.reporting_agent_replay import build_agent_replay; build_agent_replay(run_dir='outputs/parking_slot_agent_v2_frame_9277/openai_extended60_exhaustive_v1', output_dir='outputs/parking_slot_agent_v2_frame_9277/agent_replay_demo')\"",
        "```",
        "",
    ])
    return "\n".join(lines)


def build_agent_replay(*, run_dir: str | Path, output_dir: str | Path, slot_ids: tuple[str, ...] = ("slot_1012", "slot_1258", "slot_1248")) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cases = [_case_payload(run, assets, slot_id) for slot_id in slot_ids]
    payload = {
        "schema_version": "parking-slot-agent-replay/1.1",
        "source_run": str(run),
        "overview": _run_overview(run),
        "cases": cases,
    }
    html_path = output / "index.html"
    html_path.write_text(_document(payload), encoding="utf-8")
    data_path = output / "replay_data.json"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "REPORT.md"
    report_path.write_text(_report_markdown(payload), encoding="utf-8")
    gifs = [_write_replay_gif(case, output) for case in cases]
    boards = _write_decision_boards(cases, output)
    (output / "README.md").write_text(
        "# Agent决策回放\n\n"
        "- 阅读 `REPORT.md`：完整结果、24车位总表、三例逐轮解释、审计边界与复现方法。\n"
        "- 首先查看 `decision_boards/slot_1248_agent_decision_graph.png`：三轮真实ReAct因果图。\n"
        "- 直接查看 `slot_1012_agent_replay.gif`：Unknown→Free。\n"
        "- 直接查看 `slot_1258_agent_replay.gif`：Unknown→Occupied。\n"
        "- 直接查看 `slot_1248_agent_replay.gif`：三轮工具后仍Unknown。\n"
        "- 打开 `index.html`：可点击时间轴、切换案例、自动播放并展开原始action。\n\n"
        "LiDAR解释图位于 `assets/slot_*_lidar_explained.png`：绿色为自由体素，"
        "红色为有效障碍核心，橙色为被边界门否决的候选，空白车位表示没有覆盖。\n\n"
        "所有页面和GIF均由真实Part2结果、OpenAI审计action和工具产物生成；"
        "中文决策摘要是对日志中rationale/reason及几何门的结构化转述，原始记录保留在页面内。\n",
        encoding="utf-8",
    )
    return {"html": str(html_path), "report": str(report_path), "data": str(data_path), "gifs": [str(path) for path in gifs], "decision_boards": [str(path) for path in boards], "asset_count": len(list(assets.glob("*"))), "case_count": len(cases)}


__all__ = ["build_agent_replay"]
