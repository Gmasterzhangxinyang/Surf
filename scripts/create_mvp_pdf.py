#!/usr/bin/env python3
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vibe_demo"
PDF_DIR = ROOT / "outputs" / "pdf"
PDF_PATH = PDF_DIR / "DASP-Park_MVP_Report.pdf"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def register_fonts():
    pdfmetrics.registerFont(TTFont("CN", FONT_PATH))


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="CN",
            fontSize=28,
            leading=34,
            textColor=colors.HexColor("#111111"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="CN",
            fontSize=13,
            leading=20,
            textColor=colors.HexColor("#6e6e73"),
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="CN",
            fontSize=20,
            leading=26,
            textColor=colors.HexColor("#111111"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="CN",
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#111111"),
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="CN",
            fontSize=10.5,
            leading=17,
            textColor=colors.HexColor("#1d1d1f"),
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="CN",
            fontSize=8.8,
            leading=13,
            textColor=colors.HexColor("#6e6e73"),
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="CN",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#6e6e73"),
            alignment=TA_CENTER,
            spaceBefore=3,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="CN",
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#111111"),
            alignment=TA_LEFT,
        ),
    }


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 8)
    canvas.setFillColor(colors.HexColor("#8e8e93"))
    canvas.drawString(18 * mm, 10 * mm, "DASP-Park MVP Report")
    canvas.drawRightString(192 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def img(path, width_mm, height_mm=None):
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    draw_w = width_mm * mm
    if height_mm is None:
        draw_h = draw_w * ih / iw
    else:
        draw_h = height_mm * mm
        ratio = min(draw_w / iw, draw_h / ih)
        draw_w = iw * ratio
        draw_h = ih * ratio
    item = Image(str(path), width=draw_w, height=draw_h)
    item.hAlign = "CENTER"
    return item


def p(text, styles, name="body"):
    return Paragraph(text, styles[name])


def section_title(text, styles):
    return p(text, styles, "h1")


def metric_table(metrics, styles):
    data = [
        ["指标", "Before", "After", "说明"],
        ["False-free errors", str(metrics["false_free_before"]), str(metrics["false_free_after"]), "误判为空的格子减少"],
        ["Occupied IoU", f'{metrics["occupied_iou_before"]:.3f}', f'{metrics["occupied_iou_after"]:.3f}', "占据区域重合度提高"],
        ["Unknown cells", str(metrics["unknown_cells_before"]), str(metrics["unknown_cells_after"]), "未知格子减少"],
        [
            "Target unknown ratio",
            f'{metrics["target_slot_unknown_ratio_before"]:.3f}',
            f'{metrics["target_slot_unknown_ratio_after"]:.3f}',
            "目标相关未知比例下降",
        ],
        ["AI-selected tool calls", "-", str(metrics["num_tool_calls"]), "AI 选择的感知工具调用"],
    ]
    table = Table(data, colWidths=[42 * mm, 28 * mm, 28 * mm, 62 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "CN"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d2d2d7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (2, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def round_table(rounds):
    data = [["轮次", "阶段", "AI 选择的 query/tool", "Top3 ranking 变化", "停止信号"]]
    for item in rounds:
        plan = item.get("validated_plan", [{}])[0]
        tools = ", ".join(plan.get("tool_ids", []))
        before = " -> ".join(item.get("top3_before", []))
        after = " -> ".join(item.get("top3_after", []))
        data.append(
            [
                f'Round {item.get("round")}',
                item.get("phase", ""),
                f'{plan.get("action_id", "")} / {tools}',
                f"{before}<br/>变为<br/>{after}",
                item.get("stop_reason", ""),
            ]
        )
    table = Table(
        [[Paragraph(str(cell), ParagraphStyle("cell", fontName="CN", fontSize=7.8, leading=11)) for cell in row] for row in data],
        colWidths=[18 * mm, 31 * mm, 42 * mm, 52 * mm, 34 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d2d2d7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_pdf():
    register_fonts()
    styles = make_styles()
    metrics = load_json(OUT / "metrics_before_after.json")["metrics"]
    trace = load_json(OUT / "agent_trace.json")
    rounds = trace["agent_reasoning"]["rounds"]
    history = trace["target_slot_history"]
    final_slot = trace["target_slot"]["slot_id"]

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
    )
    story = []

    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph("DASP-Park MVP 报告", styles["title"]))
    story.append(
        Paragraph(
            "面向停车场景的 AI 辅助主动感知：AI 选择 perception query 和工具，局部证据更新 occupancy belief，ranking 只作为下游稳定性验证信号。",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(img(OUT / "18_quick_storyboard.png", 158))
    story.append(Spacer(1, 7 * mm))
    story.append(
        p(
            "一句话：本项目不是让 AI 直接选择停车位，而是让 AI 在规则约束下选择下一步感知查询，让 perception 工具补充局部证据，最终得到更可靠的 occupancy belief。",
            styles,
            "callout",
        )
    )
    story.append(PageBreak())

    story.append(section_title("1. 项目目的", styles))
    story.append(
        p(
            "停车场景里，初始 LiDAR perception 往往很稀疏。系统可能知道有哪些候选停车位，也能得到一个初始 occupancy belief，但很多关键区域仍然是 unknown 或 occluded_unknown。直接把初始 ranking 当作停车决策，会把 perception 的不确定性误当成确定结论。",
            styles,
        )
    )
    story.append(
        p(
            "DASP-Park 的目标是做 decision-aware active perception：利用停车任务上下文判断哪些感知证据最关键，再由 AI Advisor 在规则生成的合法 query/tool 中做选择，最后由局部工具更新 occupancy belief。",
            styles,
        )
    )
    story.append(
        p(
            "因此，本 MVP 的核心输出不是“AI 选出的停车位”，而是“经过主动查询后被更新的 perception belief”。slot ranking 只是用来验证 belief 更新是否影响下游任务。",
            styles,
        )
    )

    story.append(section_title("2. MVP 系统设计", styles))
    design_rows = [
        ["模块", "作用"],
        ["Sparse LiDAR perception", "由稀疏观测点云生成初始 occupancy belief。"],
        ["Known slot map", "提供停车任务上下文，指出哪些区域可能影响下游 ranking。"],
        ["Rule generator", "稳定地产生合法 perception query 和 allowed_tools。"],
        ["AI Advisor", "只选择 action_id 和 tool_ids，不编造坐标，不修改地图。"],
        ["Validator", "校验 action/tool 是否在规则允许范围内。"],
        ["Local tools", "读取局部证据并更新 occupancy belief。"],
    ]
    design_table = Table(design_rows, colWidths=[48 * mm, 120 * mm], repeatRows=1)
    design_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "CN"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d2d2d7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(design_table)
    story.append(PageBreak())

    story.append(section_title("3. 实验输入", styles))
    story.append(
        p(
            "实验使用可控合成停车场景。ground truth 只用于评价和可视化，不允许 Agent 在选择 query 或更新 belief 时读取。",
            styles,
        )
    )
    story.append(img(OUT / "02_initial_observed_lidar.png", 145))
    story.append(Paragraph("图 1. 初始稀疏 LiDAR。很多区域没有被观测到，因此不能把 unknown 当作 free。", styles["caption"]))
    story.append(Spacer(1, 5 * mm))
    story.append(img(OUT / "00_slot_selection_scores.png", 145))
    story.append(Paragraph("图 2. 初始 ranking 只是任务探针，提示哪些区域的 perception 证据最值得补。", styles["caption"]))
    story.append(PageBreak())

    story.append(section_title("4. Agent 流程", styles))
    story.append(
        p(
            "当前 live AI 版本使用 gpt-5.4-mini 作为 policy advisor。每轮预算为一个 query。AI 必须在规则生成的 candidate_actions 和 allowed_tools 中选择，Validator 会过滤非法输出。",
            styles,
        )
    )
    story.append(round_table(rounds))
    story.append(Spacer(1, 6 * mm))
    story.append(
        p(
            f"运行轨迹：初始任务探针指向 {history[0]['slot_id']}；三轮局部 query 后，downstream current best candidate 为 {final_slot}。这不是 AI 直接拍板，而是工具更新 occupancy belief 后 ranking 的变化结果。",
            styles,
        )
    )
    story.append(img(OUT / "ai_thinking_frames" / "frame_03.png", 150))
    story.append(Paragraph("图 3. AI 输出 action_id 和 tool_ids，Validator 校验后才执行工具。", styles["caption"]))
    story.append(PageBreak())

    story.append(section_title("5. Perception 更新发生在哪里", styles))
    story.append(
        p(
            "本 MVP 中真正更新 perception 的不是 LLM，而是局部工具。AI 只负责在有限预算下选择 query 和 tool。",
            styles,
        )
    )
    story.append(
        p(
            "lidar_geometry_checker 会读取选中区域内的局部 LiDAR 证据，估计点数、高度、occupied_confidence 和 free_confidence，并把局部格子更新为 FREE 或 OCCUPIED。occlusion_reasoning_tool 用于判断被遮挡区域是否应保持 OCCLUDED_UNKNOWN。visual_crop_checker 当前是占位接口，后续可接真实图像模型。",
            styles,
        )
    )
    story.append(img(OUT / "12_tool_evidence.png", 150))
    story.append(Paragraph("图 4. 工具证据只来自选中的局部区域，不读取 ground truth。", styles["caption"]))
    story.append(Spacer(1, 4 * mm))
    story.append(img(OUT / "13_occupancy_after.png", 84))
    story.append(Paragraph("图 5. 工具执行后得到更新后的 occupancy belief。", styles["caption"]))
    story.append(PageBreak())

    story.append(section_title("6. 实验结果", styles))
    story.append(
        p(
            "结果表明，局部主动感知让 occupancy belief 有可度量改善。ranking 的变化说明初始 belief 对下游任务并不稳定，因此主动查询是有意义的。",
            styles,
        )
    )
    story.append(metric_table(metrics, styles))
    story.append(Spacer(1, 6 * mm))
    story.append(img(OUT / "15_effectiveness_panel.png", 150))
    story.append(Paragraph("图 6. Before/After 效果面板。主动工具减少 false-free 和 unknown，并提升 occupied IoU。", styles["caption"]))
    story.append(PageBreak())

    story.append(section_title("7. 为什么这不是直接决策", styles))
    story.append(
        p(
            "容易误解的一点是：系统里出现了 slot ranking，所以看起来像在直接做停车位决策。实际上 ranking 在这里主要是任务探针和验证信号。",
            styles,
        )
    )
    story.append(
        p(
            "正确理解是：任务上下文告诉 perception 哪些未知区域重要；AI 选择下一步 perception query；工具更新 occupancy belief；ranking 只用来观察这个 belief 更新是否影响下游。",
            styles,
        )
    )
    story.append(
        p(
            "所以，本项目的核心贡献是 perception refinement，而不是 planner 或 controller。真实停车控制、轨迹规划、最终安全停车策略都不在当前 MVP 范围内。",
            styles,
        )
    )

    story.append(section_title("8. MVP 结论和下一步", styles))
    story.append(
        p(
            "MVP 已验证一个闭环：sparse perception -> AI-assisted perception query -> local tool evidence -> occupancy belief update -> downstream ranking/stability evaluation。",
            styles,
        )
    )
    story.append(
        p(
            "下一步应补齐真实视觉 crop 工具、Replay 模式、消融实验和真实数据验证。尤其是 visual_crop_checker 应从占位接口升级为真实相机局部检查器，用于识别锥桶、车位线、行人和低矮障碍。",
            styles,
        )
    )

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(PDF_PATH)


if __name__ == "__main__":
    build_pdf()
