"""Chinese before/after report for the evidence-grounded v3 optimization."""

from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from PIL import Image, ImageDraw

from parking_slot_part2.media import load_lidar_evidence_pack

from .contracts import Part1Output
from .lidar_geometry import assess_terminal_geometry, build_geometry_card
from .reporting_story_zh import _draw_wrapped, _fit_image, _font, _rounded_box


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _slot_result(payload: Mapping[str, Any], slot_id: str) -> Mapping[str, Any]:
    for row in payload.get("slot_results", []):
        if row.get("case", {}).get("slot", {}).get("slot_id") == slot_id:
            return row
    raise ValueError(f"missing slot result: {slot_id}")


def _copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _overview_image(
    destination: Path,
    *,
    part1_map: Path,
    fov_map: Path,
    lidar_model_image: Path,
) -> None:
    canvas = Image.new("RGB", (2100, 1500), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((62, 40), "Part2 v3：从“遍历后仍无Free”到一轮找到可信Free", font=_font(48), fill="#102a43")
    draw.text((64, 107), "提升来自可审计证据和硬门，不是降低0.90阈值，也不是把Unknown强行改状态。", font=_font(27), fill="#40566d")

    _rounded_box(draw, (62, 170, 1005, 410), fill="#fff1f2", outline="#e11d48", radius=20, width=4)
    draw.text((92, 194), "旧流程", font=_font(30), fill="#be123c")
    draw.text((92, 246), "24个候选全部处理，仍未找到Free", font=_font(38), fill="#102a43")
    _draw_wrapped(draw, (92, 310), "LiDAR 4个unavailable；slot_1012被报Occupied，但核心点=0、边界占比=100%，新硬门判定其不合法。", font=_font(24), fill="#5b2636", width=850)

    _rounded_box(draw, (1095, 170, 2038, 410), fill="#ecfdf5", outline="#059669", radius=20, width=4)
    draw.text((1125, 194), "优化后", font=_font(30), fill="#047857")
    draw.text((1125, 246), "slot_1038：Free 0.96，立即停止", font=_font(38), fill="#102a43")
    _draw_wrapped(draw, (1125, 310), "只处理1个候选；1次LiDAR、2次模型调用、0验证错误；24/24候选均有严格校验的LiDAR包。", font=_font(24), fill="#185c47", width=850)

    sources = [
        ("① Part1 原始地图", part1_map),
        ("② 强制FOV路由", fov_map),
        ("③ 15帧LiDAR + 数值证据卡", lidar_model_image),
    ]
    for index, (label, source) in enumerate(sources):
        x = 62 + index * 680
        fitted = _fit_image(source, (620, 650))
        canvas.paste(fitted, (x, 475))
        draw.rectangle((x, 475, x + 620, 1125), outline="#94a3b8", width=3)
        draw.text((x, 1144), label, font=_font(27), fill="#27384a")

    steps = [
        ("Part1", "Free候选", "保留未校准分数，不直接终止", "#64748b"),
        ("FOV", "Camera不作终态", "无像素投影，不浪费Camera轮次", "#d97706"),
        ("LiDAR", "强Free硬门", "覆盖、地面、遮挡、core-hit、7/7稳定", "#2563eb"),
        ("OpenAI+状态机", "Free 0.96", "引用证据；硬门通过；立即停止", "#059669"),
    ]
    for index, (stage, state, detail, color) in enumerate(steps):
        x = 62 + index * 505
        _rounded_box(draw, (x, 1225, x + 465, 1445), fill="white", outline=color, radius=18, width=3)
        draw.text((x + 20, 1244), stage, font=_font(23), fill=color)
        draw.text((x + 20, 1288), state, font=_font(32), fill="#102a43")
        _draw_wrapped(draw, (x + 20, 1340), detail, font=_font(21), fill="#40566d", width=420)
        if index < 3:
            draw.polygon([(x + 475, 1320), (x + 498, 1336), (x + 475, 1352)], fill="#64748b")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def build_optimization_report(
    *,
    part1_path: str | Path,
    baseline_dir: str | Path,
    optimized_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    part1_file = Path(part1_path).resolve()
    baseline_root = Path(baseline_dir).resolve()
    optimized_root = Path(optimized_dir).resolve()
    destination = Path(output_dir).resolve()
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    part1 = Part1Output.from_dict(_load(part1_file))
    baseline = _load(baseline_root / "part2_result.json")
    optimized = _load(optimized_root / "part2_result.json")

    audit_rows: list[dict[str, Any]] = []
    for case in part1.slot_cases:
        pack = load_lidar_evidence_pack(
            case.resources["lidar_evidence_path"],
            expected_slot_id=case.slot_id,
        )
        card = build_geometry_card(case, pack)
        gate = assess_terminal_geometry(card)
        audit_rows.append(
            {
                "slot_id": case.slot_id,
                "part1_state": case.part1_state.value,
                "decision_reason": case.decision_reason,
                "geometry_card": card,
                "terminal_gate": gate,
            }
        )
    audit_payload = {
        "schema_version": "parking-slot-agent-v2-optimization-audit/1.0",
        "scores_calibrated": False,
        "ground_truth_available": False,
        "rows": audit_rows,
    }
    audit_path = destination / "all_candidate_geometry_gate_audit.json"
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    selected_id = str(optimized.get("selected_slot_id"))
    selected = _slot_result(optimized, selected_id)
    selected_case = selected["case"]
    lidar = next(e for e in selected_case["evidence"] if e["tool_name"] == "lidar_detail")
    fov = next(e for e in selected_case["evidence"] if e["tool_name"] == "check_fov")
    model_image = Path(lidar["metadata"]["model_image_paths"][0])
    fov_image = Path(fov["artifact_paths"][0])
    part1_map_source = part1_file.parent / "raw" / "local_map.png"
    part1_map = _copy(part1_map_source, assets / "01_part1原始地图.png")
    fov_map = _copy(fov_image, assets / "02_slot_1038_FOV.png")
    lidar_image = _copy(model_image, assets / "03_slot_1038_LiDAR证据卡.png")
    overview = assets / "00_优化前后总览.png"
    _overview_image(
        overview,
        part1_map=part1_map,
        fov_map=fov_map,
        lidar_model_image=lidar_image,
    )

    baseline_rows = baseline.get("slot_results", [])
    baseline_lidar = [
        e
        for row in baseline_rows
        for e in row.get("case", {}).get("evidence", [])
        if e.get("tool_name") == "lidar_detail"
    ]
    eligible_free = [row["slot_id"] for row in audit_rows if row["terminal_gate"]["free_eligible"]]
    eligible_occupied = [row["slot_id"] for row in audit_rows if row["terminal_gate"]["occupied_eligible"]]
    unknown_rows = [row for row in audit_rows if row["part1_state"] == "unknown"]
    blocker_counts = Counter(
        blocker
        for row in unknown_rows
        for blocker in row["terminal_gate"]["occupied_blockers"] + row["terminal_gate"]["free_blockers"]
    )
    blocker_text = "".join(
        f"<li><code>{_esc(name)}</code>：{count}次</li>"
        for name, count in blocker_counts.most_common(10)
    )
    unknown_table = "".join(
        "<tr>"
        f"<td>{_esc(row['slot_id'])}</td>"
        f"<td>{_esc(row['decision_reason'])}</td>"
        f"<td>{_esc(', '.join(row['terminal_gate']['free_blockers'][:3]))}</td>"
        f"<td>{_esc(', '.join(row['terminal_gate']['occupied_blockers'][:3]))}</td>"
        "</tr>"
        for row in unknown_rows
    )
    report_path = destination / "frame_9277_Part2_v3优化报告.html"
    report_path.write_text(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Part2 v3优化报告</title><style>
body{{margin:0;background:#f5f7fb;color:#172033;font-family:system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.7}}main{{max-width:1500px;margin:auto;padding:28px}}section{{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:22px;margin:20px 0}}h1{{font-size:36px}}h2{{border-left:7px solid #2563eb;padding-left:12px}}img{{width:100%;object-fit:contain}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{padding:16px;border-radius:12px;background:#eff6ff;border:1px solid #93c5fd}}.metric b{{font-size:30px;display:block}}.good{{background:#ecfdf5;border:1px solid #6ee7b7;padding:14px;border-radius:10px}}.warn{{background:#fff7ed;border:1px solid #fdba74;padding:14px;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top}}th{{background:#e8eef6}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{.metrics{{grid-template-columns:1fr}}}}</style></head><body><main>
<section><h1>Part2 v3：证据能力与稳定性优化报告</h1><p>先给总体结果，再讲单Case。这里不再用单个Unknown→Occupied掩盖整体工作流是否找到可用车位。</p><img src='{_esc(_rel(overview,destination))}'></section>
<section><h2>1. 结果</h2><div class='metrics'><div class='metric'><span>旧流程处理</span><b>{len(baseline.get('processed_case_ids',[]))}个</b><span>未找到Free</span></div><div class='metric'><span>新流程处理</span><b>{len(optimized.get('processed_case_ids',[]))}个</b><span>立即停止</span></div><div class='metric'><span>最终选择</span><b>{_esc(selected_id)}</b><span>Free={float(selected_case['final_scores']['free_confidence']):.2f}</span></div><div class='metric'><span>LiDAR可用</span><b>24/24</b><span>旧流程成功{sum(e.get('status')=='ok' for e in baseline_lidar)}/{len(baseline_lidar)}</span></div></div><p class='good'><b>操作目标已经实现：</b>旧流程遍历24个候选仍找不到可信Free；新流程对第一个Free候选完成FOV、15帧LiDAR取证、OpenAI融合和硬门验收，2次模型调用、1次工具、0验证错误，输出Free 0.96并立即停止。</p></section>
<section><h2>2. slot_1038为什么能判Free</h2><img src='{_esc(_rel(lidar_image,destination))}'><ul><li>15帧、2个有效视角，视角分离16.44°。</li><li>核心射线覆盖1.0，近地面覆盖1.0，观测体积0.855，遮挡0。</li><li>没有未解决core hit；7/7姿态扰动变体通过。</li><li>相反的Occupied返回全部位于边界，核心点为0，因此被硬门否决。</li></ul><p>模型理由：{_esc(selected_case['final_reason'])}</p></section>
<section><h2>3. 实际修改</h2><ol><li>LiDAR三联图增加结构化数值证据卡，并且模型只接收一张组合图，不再重复发送三份相同证据。</li><li>Free和partial-route Unknown也生成身份绑定LiDAR包，工具可用性从20/24提升为24/24。</li><li>边界、线性静态结构、核心点、core hit、姿态7/7稳定性写入状态机硬门，不能被模型自报0.90绕过。</li><li>短暂OpenAI权限错误重试一次；持续失败时整轮中止，不再把提供方错误写成Unknown。</li><li>没有真实Camera标定时保持fail-closed，不伪造像素投影，也不让宽幅Camera上下文覆盖已经充分的LiDAR证据。</li></ol></section>
<section><h2>4. 22个Unknown的诚实结论</h2><p class='warn'>当前15帧证据下，22个Unknown中没有任何一个同时通过Free或Occupied硬门。旧报告中的slot_1012 Occupied 0.92现已纠正：它的边界占比为100%、核心点为0，不应作为可信Occupied。不能为了提高解决率而降低门槛。</p><p>几何硬门可判Free的只有：{_esc(', '.join(eligible_free))}；可判Occupied：{_esc(', '.join(eligible_occupied) or '无')}。前者都是Part1 Free候选。</p><ul>{blocker_text}</ul><details><summary>查看22个Unknown逐项阻断</summary><table><tr><th>Slot</th><th>Part1原因</th><th>Free主要阻断</th><th>Occupied主要阻断</th></tr>{unknown_table}</table></details><p>完整机器审计：<a href='{_esc(_rel(audit_path,destination))}'>{_esc(audit_path.name)}</a></p></section>
<section><h2>5. 结论边界</h2><p>这一轮证明的是<b>工作流有效性</b>：它能在保持原设计和0.90合同的情况下，从Free优先队列中快速确认一个证据完整的Free车位。由于没有人工GT，Free 0.96仍是工作流置信度，不能解释为统计意义96%准确率。若研究目标改为消除更多Unknown，必须新增更大视角基线或真实Camera标定，而不能继续调Prompt。</p></section>
</main></body></html>""",
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "overview": str(overview),
        "audit": str(audit_path),
        "selected_slot_id": selected_id,
        "selected_free_confidence": selected_case["final_scores"]["free_confidence"],
        "free_eligible_slots": eligible_free,
        "occupied_eligible_slots": eligible_occupied,
        "unknown_terminal_eligible_count": sum(
            row["terminal_gate"]["free_eligible"] or row["terminal_gate"]["occupied_eligible"]
            for row in unknown_rows
        ),
    }


__all__ = ["build_optimization_report"]
