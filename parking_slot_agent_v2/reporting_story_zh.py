"""Outcome-first, plain-Chinese visual report for the frame-9277 Part2 run."""

from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .reporting import DETAIL_TOOLS, collect_run_metrics


FONT_CJK = Path(
    "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
)
FONT_FALLBACK = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}
FOV_ZH = {
    "visible": "可见",
    "partially_visible": "部分可见",
    "uncertain": "不确定",
    "not_visible": "不可见",
}
TOOL_ZH = {
    "check_fov": "FOV",
    "lidar_detail": "LiDAR",
    "camera_context": "Camera全图",
    "camera_sequence": "Camera时序",
    "camera_crop": "Camera裁剪",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _rel(path: str | Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()


def _font(size: int) -> ImageFont.FreeTypeFont:
    source = FONT_CJK if FONT_CJK.is_file() else FONT_FALLBACK
    return ImageFont.truetype(str(source), size=size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return draw.textbbox((0, 0), text, font=font)[2]


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and _text_width(draw, candidate, font) > width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
    width: int,
    spacing: int = 8,
) -> int:
    x, y = xy
    line_height = font.size + spacing
    for line in _wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    radius: int = 22,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _build_one_page_summary(path: Path, baseline: Any, experiment: Any) -> None:
    image = Image.new("RGB", (1800, 1160), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    title = _font(54)
    subtitle = _font(27)
    large = _font(50)
    medium = _font(30)
    small = _font(23)
    tiny = _font(19)
    draw.text((72, 52), "Frame 9277 · Part2到底改善了什么？", font=title, fill="#102a43")
    draw.text((74, 126), "答案：不是已经证明“识别更准”，而是 Agent 从几乎不会用工具，变成能按流程取证并给出合法结论。", font=subtitle, fill="#40566d")

    panels = [
        (72, 200, 850, 665, "旧：Qwen 0.8B", "#fff1f2", "#be123c"),
        (950, 200, 1728, 665, "新：OpenAI工作流", "#eff6ff", "#1d4ed8"),
    ]
    for x1, y1, x2, y2, panel_title, fill, outline in panels:
        _rounded_box(draw, (x1, y1, x2, y2), fill=fill, outline=outline)
        draw.text((x1 + 32, y1 + 24), panel_title, font=medium, fill=outline)

    old_steps = [
        ("24", "候选车位"),
        ("139", "次模型输出，全部直接Final"),
        ("138", "次被状态机拒绝"),
        ("1/24", "只有一个车位形成合法终态"),
    ]
    new_steps = [
        ("24", "候选全部尝试细节工具"),
        ("42", "次真实工具调用"),
        ("22/24", "获得成功Camera或LiDAR证据"),
        ("23/24", "形成合法终态"),
    ]
    for start_x, steps, color in ((112, old_steps, "#be123c"), (990, new_steps, "#1d4ed8")):
        y = 286
        for index, (number, label) in enumerate(steps):
            draw.ellipse((start_x, y + 8, start_x + 22, y + 30), fill=color)
            draw.text((start_x + 48, y - 8), number, font=large, fill=color)
            draw.text((start_x + 225, y + 8), label, font=small, fill="#27384a")
            if index < len(steps) - 1:
                draw.line((start_x + 11, y + 42, start_x + 11, y + 88), fill=color, width=5)
            y += 91

    draw.text((72, 718), "四个最直观的变化", font=medium, fill="#102a43")
    kpis = [
        ("4.2% → 91.7%", "成功细节证据覆盖", "#047857"),
        ("4.2% → 95.8%", "合法终态覆盖", "#047857"),
        ("138 → 1", "错误次数", "#be123c"),
        ("139 → 51", "模型轮次", "#1d4ed8"),
    ]
    card_width = 393
    for index, (number, label, color) in enumerate(kpis):
        x1 = 72 + index * (card_width + 26)
        _rounded_box(draw, (x1, 772, x1 + card_width, 938), fill="#ffffff", outline="#cbd5e1")
        draw.text((x1 + 24, 805), number, font=medium, fill=color)
        draw.text((x1 + 24, 864), label, font=small, fill="#40566d")

    _rounded_box(draw, (72, 980, 1728, 1100), fill="#fff7ed", outline="#f59e0b", radius=18)
    draw.text((98, 1004), "最终输出：0个可信Free，1个Occupied，23个Unknown。", font=medium, fill="#9a3412")
    draw.text((98, 1054), "没有人工GT，所以当前能证明“流程变可靠”，不能证明“车位识别准确率提升”。", font=small, fill="#7c2d12")
    image.save(path, quality=95)


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        converted = source.convert("RGB")
        fitted = ImageOps.contain(converted, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#e8edf3")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _tool_images(case: Mapping[str, Any]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for evidence_raw in _list(case.get("evidence")):
        evidence = _dict(evidence_raw)
        tool = str(evidence.get("tool_name"))
        paths = [
            Path(str(raw))
            for raw in _list(evidence.get("artifact_paths"))
            if Path(str(raw)).suffix.lower() in IMAGE_SUFFIXES and Path(str(raw)).is_file()
        ]
        if paths:
            result.setdefault(tool, []).extend(paths)
    return result


def _build_case_collage(
    path: Path,
    *,
    title: str,
    takeaway: str,
    panels: list[tuple[str, Path]],
    footer: str,
    accent: str,
) -> None:
    width = 1800
    columns = 2 if len(panels) <= 4 else 3
    rows = (len(panels) + columns - 1) // columns
    panel_w = 800 if columns == 2 else 520
    panel_h = 410
    gap = 34
    left = 72
    top = 225
    height = top + rows * (panel_h + 72) + 190
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((72, 45), title, font=_font(47), fill="#102a43")
    _rounded_box(draw, (72, 118, 1728, 193), fill="#ffffff", outline=accent, radius=16, width=3)
    draw.text((96, 138), takeaway, font=_font(27), fill=accent)
    for index, (label, source) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = left + col * (panel_w + gap)
        if columns == 2:
            x = 72 + col * 850
        if len(panels) == 3 and index == 2:
            x = (width - panel_w) // 2
        y = top + row * (panel_h + 72)
        fitted = _fit_image(source, (panel_w, panel_h))
        image.paste(fitted, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline="#94a3b8", width=2)
        draw.text((x, y + panel_h + 12), label, font=_font(23), fill="#40566d")
    footer_y = top + rows * (panel_h + 72) + 20
    _rounded_box(draw, (72, footer_y, 1728, footer_y + 112), fill="#ffffff", outline="#cbd5e1", radius=16)
    _draw_wrapped(draw, (96, footer_y + 20), footer, font=_font(24), fill="#27384a", width=1580)
    image.save(path, quality=95)


def _plain_tool_summary(case: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for evidence_raw in _list(case.get("evidence")):
        evidence = _dict(evidence_raw)
        tool = str(evidence.get("tool_name"))
        if tool not in DETAIL_TOOLS:
            continue
        status = "成功" if evidence.get("status") == "ok" else "不可用"
        parts.append(f"{TOOL_ZH.get(tool, tool)}：{status}")
    return "；".join(parts) if parts else "没有执行细节工具"


def _plain_case_conclusion(case: Mapping[str, Any]) -> str:
    state = str(case.get("final_state"))
    fov = str(_dict(case.get("fov")).get("visibility"))
    scores = _dict(case.get("final_scores"))
    if state == "occupied":
        return f"Camera为{FOV_ZH.get(fov, fov)}，LiDAR提供了足够的车位内持续障碍证据，因此判为占用（{float(scores.get('occupied_confidence',0)):.2f}）。"
    if state == "free":
        return f"多模态证据满足空闲终态合同，因此判为空闲（{float(scores.get('free_confidence',0)):.2f}）。"
    if fov == "not_visible":
        return "Camera不在可靠视野内；LiDAR证据不足或不可用，因此保持未知，不强行判断。"
    return "已经调用可用工具，但目标定位或占用证据仍不够一致，因此保持未知。"


def _all_evidence_figures(case: Mapping[str, Any], report_dir: Path) -> str:
    figures: list[str] = []
    for evidence_raw in _list(case.get("evidence")):
        evidence = _dict(evidence_raw)
        tool = str(evidence.get("tool_name"))
        if tool not in DETAIL_TOOLS | {"check_fov"}:
            continue
        for raw in _list(evidence.get("artifact_paths")):
            source = Path(str(raw))
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = _rel(source, report_dir)
            figures.append(
                "<figure>"
                f"<a href='{_esc(relative)}' target='_blank'><img src='{_esc(relative)}' loading='lazy'></a>"
                f"<figcaption>{_esc(TOOL_ZH.get(tool, tool))} · {_esc(source.name)}</figcaption></figure>"
            )
    return "".join(figures)


def _round_steps(case: Mapping[str, Any]) -> str:
    items: list[str] = []
    for row_raw in _list(case.get("rounds")):
        row = _dict(row_raw)
        tool = str(row.get("tool_name"))
        if tool == "lidar_detail":
            text = "检查15帧局部点云，看车位内部是否存在持续障碍。"
        elif tool == "camera_context":
            text = "把FOV地图和原始Camera放在一起，寻找目标车位的大致区域。"
        elif tool == "camera_crop":
            text = "裁剪并增强疑似目标区域，检查是否能看清车辆或空位。"
        elif tool == "camera_sequence":
            text = "查看t0之前5张Camera，检查目标区域是否在多帧中一致。"
        else:
            text = "执行证据工具。"
        items.append(
            f"<li><b>第{int(row.get('round_index',0))}轮 · {_esc(TOOL_ZH.get(tool,tool))}</b><span>{_esc(text)}</span></li>"
        )
    return "".join(items)


def build_intuitive_chinese_report(
    *,
    baseline_dir: str | Path,
    experiment_dir: str | Path,
    technical_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_root = Path(baseline_dir).resolve()
    experiment_root = Path(experiment_dir).resolve()
    technical_path = Path(technical_report).resolve()
    destination = Path(output_dir).resolve()
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    baseline_result = _load(baseline_root / "part2_result.json")
    experiment_result = _load(experiment_root / "part2_result.json")
    baseline_metrics = collect_run_metrics(baseline_root, name="Qwen 0.8B")
    experiment_metrics = collect_run_metrics(experiment_root, name="OpenAI")
    experiment_rows = [_dict(row) for row in _list(experiment_result.get("slot_results"))]
    baseline_by_slot = {
        str(_dict(_dict(row).get("case")).get("slot", {}).get("slot_id")): _dict(row)
        for row in _list(baseline_result.get("slot_results"))
    }
    by_slot = {
        str(_dict(row.get("case")).get("slot", {}).get("slot_id")): row
        for row in experiment_rows
    }

    summary_png = assets / "00_一页结论总览.png"
    _build_one_page_summary(summary_png, baseline_metrics, experiment_metrics)
    case_specs = [
        (
            "slot_1012",
            "01_案例_slot1012_有证据的Occupied.png",
            "案例A：slot_1012 · 从反复非法结论到一次LiDAR取证后合法Occupied",
            "直观变化：旧模型对该车位连续6轮仍无法形成合法结论；新流程先按FOV路由到LiDAR，再以0.92判为Occupied。",
            "这里能证明的是“判定过程合法且有证据”；由于没有人工GT，不能证明Occupied一定正确。",
            "#be123c",
            [("FOV：目标不在可靠Camera视野", "check_fov", 0), ("LiDAR：15帧目标局部点云", "lidar_detail", 0)],
        ),
        (
            "slot_1038",
            "02_案例_slot1038_保守Unknown.png",
            "案例B：slot_1038 · Part1认为Free，但Part2没有为了出结果而硬判",
            "直观变化：先后检查LiDAR、Camera全图和增强裁剪；三轮后Free与Occupied证据仍冲突，因此保留Unknown。",
            "这不是“没工作”，而是严格执行阈值：目标定位不可靠、证据冲突时，不输出伪高置信Free。",
            "#1d4ed8",
            [("FOV：处于不确定带", "check_fov", 0), ("LiDAR局部细查", "lidar_detail", 0), ("Camera地图+全图", "camera_context", 2), ("Camera增强裁剪", "camera_crop", 0)],
        ),
        (
            "slot_1248",
            "03_案例_slot1248_LiDAR缺失降级.png",
            "案例C：slot_1248 · LiDAR缺失时按计划降级到Camera时序",
            "直观变化：LiDAR证据包不可用，Agent没有报错退出，而是继续使用Camera全图和t0前5帧时序；仍无法定位目标后输出Unknown。",
            "这个案例展示的是故障可控：工具缺失不会被包装成确定结论，系统会继续尝试合法替代工具并安全停止。",
            "#047857",
            [("FOV：部分可见", "check_fov", 0), ("Camera地图+全图", "camera_context", 2), ("t0前5帧Camera时序", "camera_sequence", 0)],
        ),
    ]
    collage_paths: list[Path] = []
    for slot_id, filename, title, takeaway, footer, accent, requested in case_specs:
        case = _dict(by_slot[slot_id].get("case"))
        images = _tool_images(case)
        panels: list[tuple[str, Path]] = []
        for label, tool, index in requested:
            available = images.get(tool, [])
            if available:
                panels.append((label, available[min(index, len(available) - 1)]))
        target = assets / filename
        _build_case_collage(target, title=title, takeaway=takeaway, panels=panels, footer=footer, accent=accent)
        collage_paths.append(target)

    tiles: list[str] = []
    details: list[str] = []
    visual_references = 0
    for index, row in enumerate(experiment_rows, start=1):
        case = _dict(row.get("case"))
        slot = _dict(case.get("slot"))
        slot_id = str(slot.get("slot_id"))
        state = str(case.get("final_state"))
        fov = str(_dict(case.get("fov")).get("visibility"))
        state_class = "occupied" if state == "occupied" else "unknown"
        tools = int(row.get("tool_rounds", 0))
        tiles.append(
            f"<a class='slot-tile {state_class}' href='#{_esc(slot_id)}'><b>{_esc(slot_id)}</b><span>{STATE_ZH.get(state,state)}</span><small>FOV {FOV_ZH.get(fov,fov)} · {tools}个工具</small></a>"
        )
        baseline_case = _dict(baseline_by_slot.get(slot_id, {}).get("case"))
        baseline_state = str(baseline_case.get("final_state"))
        images_html = _all_evidence_figures(case, destination)
        visual_references += images_html.count("<figure>")
        errors = "；".join(str(value) for value in _list(row.get("validation_errors"))) or "无"
        details.append(
            f"<details class='case-detail' id='{_esc(slot_id)}' {'open' if slot_id in {'slot_1012'} else ''}>"
            f"<summary><b>{index:02d}. {_esc(slot_id)}</b><span>Part1 {STATE_ZH.get(str(case.get('part1_state')),case.get('part1_state'))} → OpenAI {STATE_ZH.get(state,state)}</span><span>FOV {FOV_ZH.get(fov,fov)}</span><span>{tools}个工具</span></summary>"
            "<div class='case-body'>"
            f"<p class='plain-conclusion'><b>一句话结论：</b>{_esc(_plain_case_conclusion(case))}</p>"
            f"<p><b>工具实际执行：</b>{_esc(_plain_tool_summary(case))}</p>"
            f"<p><b>与基线相比：</b>Qwen最终为{STATE_ZH.get(baseline_state,baseline_state)}，模型轮次{int(baseline_by_slot.get(slot_id,{}).get('model_turns',0))}；OpenAI最终为{STATE_ZH.get(state,state)}，模型轮次{int(row.get('model_turns',0))}。</p>"
            f"<ol class='steps'>{_round_steps(case)}</ol>"
            f"<p class='error-line'><b>错误：</b>{_esc(errors)}</p>"
            f"<div class='evidence-gallery'>{images_html}</div>"
            f"<details class='tech'><summary>查看原始英文最终说明</summary><pre>{_esc(case.get('final_reason',''))}</pre></details>"
            "</div></details>"
        )

    old_technical_assets = technical_path.parent / "assets"
    appendix_charts = [
        old_technical_assets / "01_overview_dashboard.png",
        old_technical_assets / "02_per_slot_confidence_matrix.png",
        old_technical_assets / "03_per_slot_tool_matrix.png",
        old_technical_assets / "04_openai_api_audit.png",
        old_technical_assets / "05_sensor_timeline.png",
    ]
    appendix_html = "".join(
        f"<figure><a href='{_esc(_rel(path,destination))}' target='_blank'><img src='{_esc(_rel(path,destination))}' loading='lazy'></a><figcaption>{_esc(path.name)}</figcaption></figure>"
        for path in appendix_charts if path.is_file()
    )
    report_path = destination / "frame_9277_part2_直观中文报告.html"
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Frame 9277 Part2直观中文报告</title><style>
:root{{--ink:#172033;--muted:#526273;--blue:#2563eb;--red:#be123c;--green:#047857;--line:#cbd5e1;--bg:#f4f7fb}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.65}}nav{{position:sticky;top:0;z-index:5;background:#102a43;padding:13px 24px}}nav a{{color:#dbeafe;text-decoration:none;margin-right:22px}}main{{max-width:1500px;margin:auto;padding:26px}}h1{{font-size:34px}}h2{{margin-top:44px;border-left:7px solid var(--blue);padding-left:13px}}.hero,.panel{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 3px 14px #0f172a0c}}.hero img,.wide{{width:100%;object-fit:contain}}.answer{{font-size:27px;font-weight:700;color:#102a43}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.yes,.no{{padding:18px;border-radius:12px}}.yes{{background:#ecfdf5;border:1px solid #6ee7b7}}.no{{background:#fff7ed;border:1px solid #fdba74}}.case-story{{background:white;border:1px solid var(--line);border-radius:14px;padding:18px;margin:22px 0}}.case-story img{{width:100%;object-fit:contain}}.case-story p{{font-size:19px}}.slot-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.slot-tile{{display:flex;flex-direction:column;padding:12px;border-radius:10px;text-decoration:none;color:var(--ink);border:1px solid var(--line);background:#eef2f7}}.slot-tile.occupied{{background:#fee2e2;border-color:#f87171}}.slot-tile b{{font-size:18px}}.slot-tile span{{font-weight:700}}.slot-tile small{{color:var(--muted)}}.case-detail{{background:white;border:1px solid var(--line);border-radius:11px;margin:11px 0;scroll-margin-top:65px}}.case-detail>summary{{display:grid;grid-template-columns:140px 1fr 140px 100px;gap:10px;cursor:pointer;padding:14px;background:#edf3f9}}.case-body{{padding:17px}}.plain-conclusion{{font-size:19px;background:#eff6ff;border-left:5px solid var(--blue);padding:13px}}.steps{{list-style:none;padding:0}}.steps li{{display:flex;gap:16px;border-left:4px solid #93c5fd;margin:9px 0;padding:8px 12px;background:#f8fafc}}.steps li b{{min-width:180px}}.evidence-gallery,.appendix-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}}figure{{margin:0;border:1px solid var(--line);padding:8px;background:#f8fafc;border-radius:8px}}figure img{{width:100%;height:350px;object-fit:contain;background:#0f172a}}figcaption{{font-size:12px;color:var(--muted);overflow-wrap:anywhere}}.error-line{{color:#9f1239}}pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px}}.tech>summary{{cursor:pointer;color:var(--muted)}}.more-link{{display:inline-block;padding:10px 16px;background:#1d4ed8;color:white;text-decoration:none;border-radius:8px}}@media(max-width:800px){{.split{{grid-template-columns:1fr}}.case-detail>summary{{grid-template-columns:1fr}}}}@media print{{nav{{display:none}}details{{display:block}}.case-detail:not([open])>.case-body{{display:block}}}}
</style></head><body><nav><a href='#top'>一句话结论</a><a href='#cases'>三个案例</a><a href='#all'>24个车位</a><a href='#evidence'>全部证据</a><a href='#appendix'>技术附录</a></nav><main id='top'>
<section class='hero'><h1>Frame 9277 · Part2结果，直观版</h1><p class='answer'>一句话：旧模型基本只会直接猜答案；新流程会先看FOV，再调用Camera或LiDAR取证，最后按合同决定。</p><img src='{_esc(_rel(summary_png,destination))}' alt='一页结论总览'></section>
<section><h2>先说清楚：到底提升了什么？</h2><div class='split'><div class='yes'><h3>已经看得见的提升</h3><ul><li>24个候选全部真正调用过细节工具。</li><li>22个候选拿到了成功的Camera或LiDAR证据。</li><li>23个候选形成了合法终态，不再反复提交非法Final。</li><li>错误从138次降到1次，模型轮次从139降到51。</li></ul></div><div class='no'><h3>目前不能说的结论</h3><ul><li>不能说“识别准确率提高了”。</li><li>原因是这24个车位还没有人工GT。</li><li>最终仍是0 Free、1 Occupied、23 Unknown。</li><li>新旧运行的FOV参数也不同，不是纯模型单变量对比。</li></ul></div></div></section>
<section id='cases'><h2>三个案例，比大表格更容易看懂</h2>
<article class='case-story'><img src='{_esc(_rel(collage_paths[0],destination))}'><p><b>看点：</b>slot_1012 展示“有证据的终态”。旧模型对它反复给非法结论，新流程只用一次自动LiDAR路由便形成合法Occupied。</p></article>
<article class='case-story'><img src='{_esc(_rel(collage_paths[1],destination))}'><p><b>看点：</b>slot_1038 展示“保守而不是乱判”。Part1把它排在Free候选前列，但三种细节工具仍无法消除冲突，所以没有把它包装成可信Free。</p></article>
<article class='case-story'><img src='{_esc(_rel(collage_paths[2],destination))}'><p><b>看点：</b>slot_1248 展示“工具故障后的安全降级”。LiDAR缺失后继续看Camera时序，仍无法确定就输出Unknown。</p></article></section>
<section id='all'><h2>24个车位，一眼看全</h2><p>红色是Occupied，灰色是Unknown。点击任意卡片跳到该车位的证据。</p><div class='slot-grid'>{''.join(tiles)}</div></section>
<section id='evidence'><h2>24个车位的全部证据（默认折叠）</h2><p>主报告只讲人能看懂的过程；需要核对时再展开。共保留{visual_references}个Part2图片引用。</p>{''.join(details)}</section>
<section id='appendix'><h2>技术附录</h2><div class='panel'><p>下面保留总体指标、置信度矩阵、工具矩阵、API审计和15帧同步图。它们不再放在开头干扰主结论。</p><div class='appendix-grid'>{appendix_html}</div><p><a class='more-link' href='{_esc(_rel(technical_path,destination))}'>打开完整技术审计版报告</a></p></div></section>
</main></body></html>"""
    report_path.write_text(page, encoding="utf-8")
    manifest = {
        "schema_version": "parking-slot-agent-v2-intuitive-report/1.0",
        "html_report": str(report_path),
        "summary_png": str(summary_png),
        "representative_collages": [str(path) for path in collage_paths],
        "slot_count": len(experiment_rows),
        "evidence_visual_references": visual_references,
        "message": "workflow reliability improved; accuracy is not established without GT",
    }
    manifest_path = destination / "直观报告_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"html_report": str(report_path), "manifest": str(manifest_path), **manifest}


__all__ = ["build_intuitive_chinese_report"]
