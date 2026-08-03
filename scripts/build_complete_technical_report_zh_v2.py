#!/usr/bin/env python3
"""Build the comprehensive Chinese ParkingAgent Part1/Part2 technical report.

The report deliberately separates current evaluated operating points, the
formal 3--15 frame local contract, and historical route-shadow artifacts.
All numerical claims are tied to repository artifacts available on 2026-07-30.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "ParkingAgent_完整技术报告_20260730"
OUT_PDF = OUT_DIR / "ParkingAgent_完整技术报告_20260730_v2_精排版.pdf"
FONT_PATH = Path(
    "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/"
    "Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
)
FONT = "DroidSansFallback"
DOC_DATE = "2026-07-30"


def register_font() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"CJK font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
        )
        self.addPageTemplates(
            [
                PageTemplate(
                    id="normal",
                    frames=[frame],
                    onPage=self._header_footer,
                )
            ]
        )

    def _header_footer(self, canvas, doc) -> None:
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
            canvas.setLineWidth(0.45)
            canvas.line(17 * mm, 283 * mm, 193 * mm, 283 * mm)
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(colors.HexColor("#667085"))
            canvas.drawString(17 * mm, 286 * mm, "ParkingAgent 完整技术报告｜Part 1 + Part 2")
            canvas.drawRightString(193 * mm, 286 * mm, f"版本日期 {DOC_DATE}")
            canvas.line(17 * mm, 13 * mm, 193 * mm, 13 * mm)
            canvas.drawString(17 * mm, 8.7 * mm, "内部技术基线｜所有准确率主张均受报告中的 GT 边界约束")
            canvas.drawRightString(193 * mm, 8.7 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"Heading1", "Heading2"}:
            return
        level = 0 if style_name == "Heading1" else 1
        text = flowable.getPlainText()
        key = f"h{level}-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    result = {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=FONT,
            fontSize=30,
            leading=42,
            textColor=colors.HexColor("#102A43"),
            alignment=TA_LEFT,
            spaceAfter=8 * mm,
            wordWrap="CJK",
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=13.5,
            leading=21,
            textColor=colors.HexColor("#486581"),
            wordWrap="CJK",
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=FONT,
            fontSize=20,
            leading=28,
            textColor=colors.HexColor("#173F5F"),
            spaceBefore=6 * mm,
            spaceAfter=3.5 * mm,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=FONT,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#20639B"),
            spaceBefore=4.5 * mm,
            spaceAfter=2.2 * mm,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.5,
            leading=18,
            textColor=colors.HexColor("#243B53"),
            alignment=TA_JUSTIFY,
            firstLineIndent=21,
            spaceAfter=2.4 * mm,
            wordWrap="CJK",
        ),
        "BodyNoIndent": ParagraphStyle(
            "BodyNoIndent",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.5,
            leading=18,
            textColor=colors.HexColor("#243B53"),
            alignment=TA_LEFT,
            firstLineIndent=0,
            spaceAfter=2.4 * mm,
            wordWrap="CJK",
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.2,
            leading=17,
            textColor=colors.HexColor("#243B53"),
            leftIndent=6 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=1.2 * mm,
            spaceAfter=1.3 * mm,
            wordWrap="CJK",
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#627D98"),
            alignment=TA_CENTER,
            spaceBefore=1.2 * mm,
            spaceAfter=3.5 * mm,
            wordWrap="CJK",
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.2,
            leading=17,
            textColor=colors.HexColor("#12344D"),
            backColor=colors.HexColor("#EAF4FF"),
            borderColor=colors.HexColor("#8FC5F4"),
            borderWidth=0.7,
            borderPadding=8,
            borderRadius=4,
            spaceBefore=2 * mm,
            spaceAfter=3.2 * mm,
            wordWrap="CJK",
        ),
        "Warning": ParagraphStyle(
            "Warning",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.2,
            leading=17,
            textColor=colors.HexColor("#5C3700"),
            backColor=colors.HexColor("#FFF7E6"),
            borderColor=colors.HexColor("#F1B85B"),
            borderWidth=0.7,
            borderPadding=8,
            borderRadius=4,
            spaceBefore=2 * mm,
            spaceAfter=3.2 * mm,
            wordWrap="CJK",
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#486581"),
            wordWrap="CJK",
        ),
        "TOCHeading": ParagraphStyle(
            "TOCHeading",
            parent=base["Heading1"],
            fontName=FONT,
            fontSize=19,
            leading=28,
            textColor=colors.HexColor("#173F5F"),
            spaceAfter=5 * mm,
        ),
    }
    return result


S = {}


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: Iterable[str]) -> list[Flowable]:
    return [Paragraph(f"• {item}", S["Bullet"]) for item in items]


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    widths: Sequence[float] | None = None,
    *,
    font_size: float = 8.5,
) -> Table:
    cell_style = ParagraphStyle(
        "TableCell",
        parent=S["Small"],
        fontSize=font_size,
        leading=font_size * 1.48,
        textColor=colors.HexColor("#243B53"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    data = [[Paragraph(str(value), head_style) for value in headers]]
    data.extend(
        [[Paragraph(str(value), cell_style) for value in row] for row in rows]
    )
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20639B")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D6E2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    return result


def figure(path: str | Path, caption: str, max_width: float = 174 * mm, max_height: float = 110 * mm) -> list[Flowable]:
    source = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not source.is_file():
        return [p(f"图像缺失：{source}", "Warning")]
    with PILImage.open(source) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    rendered = Image(str(source), width=width * scale, height=height * scale)
    rendered.hAlign = "CENTER"
    return [rendered, p(caption, "Caption")]


def metric_cards() -> Table:
    cards = [
        ("1397", "已知车位几何"),
        ("W30/K5", "最新评估运行点"),
        ("3 / 0 / 19", "9277：Free/Occupied/Unknown"),
        ("5 → 5 Unknown", "9277 Part2 前向队列"),
    ]
    data = []
    for value, label in cards:
        data.append(
            [
                Paragraph(
                    f"<font size='18' color='#173F5F'><b>{value}</b></font><br/>"
                    f"<font size='8.2' color='#627D98'>{label}</font>",
                    ParagraphStyle(
                        f"metric-{label}",
                        parent=S["BodyNoIndent"],
                        alignment=TA_CENTER,
                        leading=18,
                        backColor=colors.HexColor("#F0F7FF"),
                        borderColor=colors.HexColor("#B8D8F4"),
                        borderWidth=0.6,
                        borderPadding=7,
                    ),
                )
            ]
        )
    result = Table([[row[0] for row in data]], colWidths=[43.5 * mm] * 4)
    result.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return result


def build_story() -> list[Flowable]:
    story: list[Flowable] = []

    # Cover
    story.extend(
        [
            Spacer(1, 19 * mm),
            p("ParkingAgent", "Title"),
            p("完整技术报告：Part 1 局部三维车位证据 + Part 2 硬门控多模态 Agent", "Subtitle"),
            Spacer(1, 8 * mm),
            metric_cards(),
            Spacer(1, 11 * mm),
            p(
                "<b>报告目的</b><br/>完整说明当前仓库已经实现的功能、数据合同、算法流程、"
                "工程边界、实验结果、验证证据与已知限制。本文以 2026-07-30 工作区为准，"
                "并明确区分最新版 v5 评估运行点、正式 3–15 帧局部合同和历史 route-shadow，"
                "防止不同实验协议的数字被混用。",
                "Callout",
            ),
            Spacer(1, 8 * mm),
            table(
                ["文档项", "内容"],
                [
                    ["项目根目录", "/home/ParkingAgent/ParkingAgent"],
                    ["报告范围", "Pose/同步/地图/Part 1/Part 2/Agent/相机审计/人工 GT/测试/复现"],
                    ["报告日期", DOC_DATE],
                    ["状态", "工程实现盘点与当前证据汇总；不是生产安全认证"],
                    ["生成器", "scripts/build_complete_technical_report_zh.py"],
                ],
                widths=[35 * mm, 139 * mm],
                font_size=8.7,
            ),
            Spacer(1, 17 * mm),
            p(
                "重要声明：本系统输出的是<b>占用证据状态</b>，不是停车场占用真值；"
                "Unknown 是安全拒判，不应默认解释为 Free。未通过独立相机投影审计的 RGB "
                "不能单独产生终态。任何“选择性准确率”都只适用于报告明确给出的人工 GT 交集与分母。",
                "Warning",
            ),
            PageBreak(),
        ]
    )

    # TOC
    story.append(p("目录", "TOCHeading"))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName=FONT,
            fontSize=9.5,
            leading=18,
            leftIndent=0,
            firstLineIndent=0,
            textColor=colors.HexColor("#173F5F"),
        ),
        ParagraphStyle(
            "TOC2",
            fontName=FONT,
            fontSize=8.4,
            leading=13,
            leftIndent=12 * mm,
            firstLineIndent=0,
            textColor=colors.HexColor("#486581"),
        ),
    ]
    story.extend([toc, PageBreak()])

    # 1
    story.append(p("1. 执行摘要与当前结论", "Heading1"))
    story.append(
        p(
            "ParkingAgent 已形成一条从地下停车场多传感器原始数据到可审计车位三态结果的完整工程链。"
            "上游执行 LiDAR/Camera 时间同步、LiDAR 静态结构到 glTF 地图的局部配准、SE(2) "
            "关键帧修正与时序平滑；Part 1 只在当前因果窗口真实射线覆盖的已知车位上构建三维占用/"
            "自由空间证据；Part 2 只接收可执行 Unknown，使用受限工具、严格身份绑定和确定性终态门进行消歧。"
        )
    )
    story.extend(
        bullets(
            [
                "<b>最新版评估运行点：</b>v5 静态语义模型，历史跨度 W=30、均匀采样 K=5、Part 2 前向总视场 180°；选中帧为 9248/9255/9262/9270/9277。",
                "<b>frame 9277 v5：</b>1397 个几何车位中，22 个进入局部输出，Part 1 为 3 Free、0 Occupied、19 Unknown，1375 个无当前射线覆盖车位省略；处理约 17.02 s。",
                "<b>frame 9277 Part 2：</b>前向门形成 5 个任务，11 个模型轮次、8 次工具尝试、50 个 trace 事件；最终 5 个全部保持 Unknown，没有强行升级。",
                "<b>正式 15 帧局部快照：</b>旧但仍保留的 part1-local-lidar-map/1.1 工程合同在 frame 9277 输出 21 个局部车位：2 Free、8 Occupied、11 Unknown，处理约 41.58 s；它与 v5 W30/K5 不是同一次配置。",
                "<b>人工 GT 证据：</b>frame 9277 的 v4 高精度版本在 14 个 GT 交集上只输出 2 个终态，均正确；选择性准确率 100%，终态覆盖率 14.3%，Occupied precision 为 N/A。",
                "<b>冻结历史挑战：</b>6 个可观测人工样本中 v5 仅作答 1 个且正确，覆盖率 16.7%、Occupied recall 25%、无 false free/false occupied；这是回顾性挑战，不是独立 holdout。",
                "<b>测试证据：</b>当前环境中，排除唯一缺失 scikit-learn 的实验性 slot-aligned accumulation 测试文件后，546 tests + 239 subtests 全部通过。",
            ]
        )
    )
    story.append(
        p(
            "<b>当前最重要的工程判断：</b>系统已经实现“可执行、可审计、失败关闭”的两阶段证据流程；"
            "但生产停车目标选择策略、独立大规模 GT、通过审计的 RGB 像素投影、跨路线泛化和车辆运动控制均未完成，"
            "因此不能宣称已具备全自动泊车或生产安全能力。",
            "Callout",
        )
    )

    story.append(p("1.1 功能状态矩阵", "Heading2"))
    story.append(
        table(
            ["能力", "当前状态", "说明"],
            [
                ["原始数据同步与身份链接", "已实现", "时间戳最近邻、配对数据集、哈希/manifest 与数据完整性检查。"],
                ["Pose 漂移校正", "已实现并有产物", "局部 ICP/SE(2) 约束、关键帧校正、插值平滑、原始数据不覆盖。"],
                ["已知车位几何数据库", "已实现", "1397 个 slot；polygon/core/margin/center/heading/adjacency 保持固定。"],
                ["Part 1 局部三态判断", "已实现", "Occupied / Free / Unknown；远处无当前覆盖车位省略。"],
                ["Part 1 最终停车位选择", "未实现", "candidate 与 candidate_slot_id 固定 null；A/B 仅调试 shortlist。"],
                ["Part 2 Unknown 队列", "已实现", "只接收 state=unknown 且 agent_observable=true 的可执行任务。"],
                ["Part 2 LiDAR 工具", "已实现并验证", "身份绑定 evidence pack、三联图、几何卡、扩展因果历史。"],
                ["Part 2 RGB 工具", "媒体桥接已实现", "真实投影审计未 passing，终态能力 fail closed。"],
                ["OpenAI / 本地 VLM 适配", "已实现", "严格 JSON、有限重试；本地端点仅允许 loopback。"],
                ["确定性回放与审计", "已实现", "canonical JSON、SHA-256、原子 checkpoint、replay 等价验证。"],
                ["人工 GT 标注与评价服务", "已实现", "本地 Web、持久化标签、导出 CSV/JSON、选择性指标。"],
                ["生产车辆控制/轨迹规划", "不在范围", "系统不输出转向、制动、轨迹或安全停车动作。"],
            ],
            widths=[43 * mm, 31 * mm, 100 * mm],
            font_size=8.4,
        )
    )

    # 2
    story.append(PageBreak())
    story.append(p("2. 系统范围、语义与关键合同", "Heading1"))
    story.append(
        p(
            "ParkingAgent 的核心不是“给每个地图车位强制二分类”，而是选择性证据系统。"
            "已知地图只提供车位位置和几何先验；点云与相机只判断这些已知车位在当前时刻是否被观测、"
            "是否形成足够强的车辆证据或自由空间证据。无法证明时保留 Unknown。"
        )
    )
    story.append(p("2.1 三态定义", "Heading2"))
    story.append(
        table(
            ["状态", "成立条件", "禁止的错误解释"],
            [
                ["Occupied", "目标 core 内存在多帧、跨高度层、车辆形态与归属合格的 3D 证据，并通过姿态/地面扰动稳定性。", "不等于人工确认；不能只因有点云或高置信文字而成立。"],
                ["Free", "多帧、多视点射线正向穿越目标核心体积，覆盖/遮挡/未观测连通块/弱障碍冲突均通过。", "不能把“没有回波”或“没有检测到车”当作 Free。"],
                ["Unknown", "部分观测、遮挡、弱证据、归属冲突、静态结构、姿态敏感、数据异常或终态门不足。", "不是错误，也不是默认 Free；它是系统的安全拒判状态。"],
            ],
            widths=[25 * mm, 94 * mm, 55 * mm],
            font_size=8.5,
        )
    )
    story.append(p("2.2 范围与候选边界", "Heading2"))
    story.extend(
        bullets(
            [
                "当前正式局部输出只包含“距离在局部半径内且本窗口至少一条真实射线穿过车位 prism 或回波落入车位”的车位。",
                "Out-of-route 不属于正式三态域；没有当前覆盖的远处车位被省略，不能用历史全路线状态补全当前地图。",
                "Part 1 不执行生产停车候选选择；最多两个 A/B provisional display 项只用于局部图解释，Occupied 永远排除。",
                "Part 2 不能推翻 Part 1 已终态的 Free/Occupied；只处理可执行 Unknown，并受 allowed_final_states 和几何硬门约束。",
                "最新版 Part 2 前向候选使用闭合 ±90° 半平面，总视场 180°；正后方排除，边界保留。",
            ]
        )
    )
    story.append(
        p(
            "<b>协议并存说明：</b>README 维护的正式局部 writer 仍强调 3–15 个因果输入记录；"
            "2026-07-29 的 v5 研究运行点则从 W=30 的历史跨度均匀抽取 K=5 帧。两者共享三态和失败关闭语义，"
            "但输入采样合同和结果数字不同。本报告分别呈现，不把 v5 结果反写成旧 15 帧快照。",
            "Warning",
        )
    )

    # 3
    story.append(PageBreak())
    story.append(p("3. 总体架构与端到端数据流", "Heading1"))
    story.extend(
        figure(
            "Nature_ParkingAgent_实验报告_20260728/figures/Figure_0_complete_algorithm.png",
            "图 3-1　仓库内生成的完整算法流程图：Part 1 几何证据、Part 2 前向候选与历史采样研究。",
            max_height=132 * mm,
        )
    )
    story.append(
        table(
            ["阶段", "输入", "处理", "输出"],
            [
                ["0 数据接入", "LiDAR、Camera、pose、timestamps、glTF", "文件身份、时间戳和坐标元数据解析", "帧级数据索引"],
                ["1 同步", "LiDAR/Camera 时间序列", "最近邻时间匹配与 Δt 审计", "paired frame records"],
                ["2 定位校正", "静态 LiDAR 结构 + glTF", "局部配准、SE(2) 约束、时序平滑", "corrected trajectory"],
                ["3 点云重投影", "原始 LiDAR + corrected pose", "逐帧重投影与 map_points NPZ", "校正后点云数据集"],
                ["4 Part 1", "已知车位 + 因果 LiDAR", "scope、地面、3D 占用、ray free、稳定性", "局部三态 + trace + queue"],
                ["5 Part 2", "可执行 Unknown + evidence catalog", "FOV、工具调用、模型提议、硬门", "resolution / final states / trace"],
                ["6 验证报告", "JSON/CSV/PNG/labels", "人工 GT、消融、shadow、回放一致性", "PDF/HTML/manifest"],
            ],
            widths=[27 * mm, 42 * mm, 62 * mm, 43 * mm],
            font_size=8.3,
        )
    )
    story.append(p("3.1 核心软件包职责", "Heading2"))
    story.append(
        table(
            ["包", "主要职责"],
            [
                ["parking_pose_correction", "数据集读取、时间匹配、SE(2) 运算、局部配准和校正报告。"],
                ["parking_slot_box_scoring", "车位内车辆候选框、几何、帧选择、点过滤与旧迁移基线评分。"],
                ["parking_slot_hybrid_3d", "当前 Part 1：累计、地面、occupied/free、稳定性、静态语义、局部图、shadow。"],
                ["parking_slot_part2", "不可变队列、preflight、媒体、工具、grouping、决策、orchestrator、local VLM。"],
                ["parking_slot_agent_v2", "独立单车位 Agent 合同、FOV、工具交互、OpenAI 适配、pipeline 与报告。"],
                ["parking_slot_validation", "人工标签模型、持久化、指标和本地 Web 服务。"],
                ["scripts", "数据构建、主流程、实验消融、审计、报告、复现与诊断入口。"],
            ],
            widths=[48 * mm, 126 * mm],
            font_size=8.4,
        )
    )

    # 4
    story.append(PageBreak())
    story.append(p("4. 数据、坐标系、同步与身份治理", "Heading1"))
    story.append(p("4.1 主要数据输入", "Heading2"))
    story.extend(
        bullets(
            [
                "原始 LiDAR：逐帧点云，保留 lidar_path、frame_id 与时间戳；校正后另写 map_points NPZ。",
                "相机：原始图片、P0/Tr 标定、camera frame 与时间戳；不会覆盖源图。",
                "Pose：原始轨迹与校正轨迹并存；校正输出包括关键帧 registration、插值 correction 与 review index。",
                "glTF/OBJ 地图：静态结构与车位几何来源；车位数据库含 polygon/core/margin/center/heading/adjacency。",
                "时间索引：LiDAR 与 Camera 使用最近邻匹配，Camera 投影正式终态默认要求配对绝对误差不超过 40 ms，并可在审计中收紧。",
            ]
        )
    )
    story.append(p("4.2 坐标与变换原则", "Heading2"))
    story.append(
        p(
            "实现中区分原始传感器坐标、车辆/自车坐标、地图坐标和车位局部坐标。"
            "车位证据首先用 corrected pose 把点云变换到地图，再以目标车位长短轴构造 metric local frame；"
            "map_units_per_meter 显式进入转换。相机投影要求每个 image/point-source 与对应 pose source、"
            "calibration bytes 和 registry digest 一致，禁止使用未绑定的手写汇总。"
        )
    )
    story.append(p("4.3 可复现身份", "Heading2"))
    story.extend(
        bullets(
            [
                "队列、配置、数据集、slot map、evidence pack、图片、pose 和投影 reference 均可绑定 SHA-256。",
                "Part 2 对外只暴露 opaque evidence_id；公开 ToolResult/trace 不携带绝对路径、base64 或原始图像字节。",
                "NPZ evidence pack 固定成员顺序、时间戳和 dtype；writer 使用 canonical JSON 与原子 replace。",
                "resume/replay 核验 snapshot、queue order、模式与资源身份，防止跨实验复用历史动作。",
            ]
        )
    )

    # 5 pose
    story.append(PageBreak())
    story.append(p("5. Pose 漂移校正与点云重建", "Heading1"))
    story.append(
        p(
            "Pose 子系统从 LiDAR 静态结构和 glTF 目标结构估计局部 SE(2) 修正。"
            "核心 registration 使用 cKDTree 最近邻、trimmed correspondence、二维刚体拟合、多初始 yaw ICP，"
            "再用 Powell 有界优化细化 dx/dy/dyaw；修正受到最大平移、最大偏航、匹配残差和 inlier ratio 约束。"
        )
    )
    story.extend(
        bullets(
            [
                "关键帧校正写入 pose_corrections.csv 与 keyframe_registration.csv。",
                "关键帧之间通过时序插值和平滑传播，并限制平移/偏航变化速度，避免单点跳变。",
                "全部原始 LiDAR 使用 corrected trajectory 重新投影，写到 frame_map_dataset_pose_corrected_final。",
                "原始 pose、原始 LiDAR、原始图片从不覆盖；所有校正产物进入独立目录。",
            ]
        )
    )
    story.extend(
        figure(
            "outputs/pose_drift_correction_v3/trajectory_before_after_on_map.png",
            "图 5-1　校正前后轨迹叠加到地图的审计图。",
            max_height=96 * mm,
        )
    )
    story.extend(
        figure(
            "outputs/pose_drift_correction_v3/structural_residual_before_after.png",
            "图 5-2　静态结构残差校正前后对比；用于定位链路审计，不是占用 GT。",
            max_height=82 * mm,
        )
    )

    # 6 Part1
    story.append(PageBreak())
    story.append(p("6. Part 1：局部混合三维车位证据", "Heading1"))
    story.append(p("6.1 因果窗口与 scope", "Heading2"))
    story.append(
        p(
            "Part 1 以 anchor 帧为当前时刻，只选择 t≤t0 的帧。正式局部合同要求 3–15 帧连续因果尾窗；"
            "研究扩展支持从历史跨度 W 中均匀抽取 K 帧。Scope 先用中心距、near frame、prism 射线穿越、"
            "回波命中和 core ray coverage 判断车位是否真实被当前窗口观察。仅靠“离车近”不能进入三态判断。"
        )
    )
    story.append(
        table(
            ["Scope", "含义", "正式局部输出"],
            [
                ["in_route_scope", "当前因果窗口真实覆盖足够", "进入 Occupied/Free/Unknown 决策"],
                ["partial_route_scope", "获得部分扫描或数据质量不足", "输出 Unknown；满足资源合同才可进入 Agent"],
                ["out_of_route_scope", "无足够实际射线/回波覆盖", "局部正式结果中省略，不伪造 Unknown"],
            ],
            widths=[41 * mm, 76 * mm, 57 * mm],
        )
    )
    story.append(p("6.2 地面估计与多帧累计", "Heading2"))
    story.extend(
        bullets(
            [
                "逐帧从局部点云低分位候选中拟合倾斜地面；默认至少 30 点、inlier 阈值 0.12 m、inlier ratio≥0.45、P95 残差≤0.15 m。",
                "退化平面使用常数分位回退；点不足、非有限或质量失败的帧记录为 invalid，不当作“空帧”。",
                "点云按目标车位长轴/短轴变换到 metric slot-local frame，并保留每个点、射线终点和观测原点的 frame provenance。",
                "体素默认为 0.25 m × 0.25 m × 0.20 m；车辆高度区间 0.30–2.20 m，分为 0.30–0.80、0.80–1.40、1.40–2.20 m 三层。",
            ]
        )
    )
    story.append(p("6.3 Occupied 证据链", "Heading2"))
    story.append(
        p(
            "Occupied 不是通用聚类结果，而是在已知目标车位内部枚举受约束车辆候选框，再对有限 shortlist "
            "计算点数、体素、垂直层、时间支持、core 归属、相邻重叠、边界比例、PCA/线性风险、"
            "稳健裁剪 footprint、外部残余和静态语义。任一不可绕过 gate 失败都会降为弱证据或 Unknown。"
        )
    )
    story.extend(
        bullets(
            [
                "默认最低 40 个车辆点、3 个有效/支持帧、时间支持≥0.20、至少 2 个高度层、z95≥0.60 m、height span≥0.35 m、3D voxel≥8。",
                "目标 core overlap≥0.45；adjacent overlap≤0.35；boundary ratio<0.50；XY linearity penalty<0.60；outside residual≤0.60（高精度 profile 为 0.35）。",
                "柱体/杆体抑制同时检查高 PCA 线性、水平 footprint、稳健 inlier fraction、垂直纵横比与最小高度。",
                "v4/v5 高精度 profile 增加 robust short extent≥0.75 m、low BEV coverage≥0.05、upper-height-spread ratio≥0.05、定位不确定性带 0.10 m。",
                "v5 静态语义 gate 使用 glTF/审计静态层解释回波；被墙、柱、轮挡等充分解释或残余过窄时，Occupied 硬否决为 Unknown。",
            ]
        )
    )
    story.extend(
        figure(
            "Nature_ParkingAgent_实验报告_20260728/figures/Figure_4_pillar_case_slot1253.png",
            "图 6-1　典型柱体/静态结构案例。该类回波必须通过稳健 footprint、形态与静态语义否决，不能仅凭高度判为车辆。",
            max_height=92 * mm,
        )
    )
    story.append(p("6.4 Free 证据链", "Heading2"))
    story.append(
        p(
            "Free 必须由正向射线几何证明。每条射线从真实观测原点遍历到测量终点；只把终点之前实际穿过的体素标为 free，"
            "终点障碍、其后的遮挡区域和从未穿越区域分别保留。单视角、稀疏射线、大片未观测连通块、"
            "核心车辆高度回波或多帧 hit/free 冲突都会阻止 Free。"
        )
    )
    story.extend(
        bullets(
            [
                "默认至少 5 个 ray frame、2 个视点、视点分离≥10°。",
                "核心体积 observed ratio≥0.70、近地 BEV coverage≥0.70。",
                "最大未观测连通块比例≤0.20、遮挡比例≤0.20。",
                "弱障碍默认至少跨 2 帧且 10 点即触发 veto；强车辆 core hit 永远不能被较高 free coverage 覆盖。",
                "只有同一连通体素中具备独立多帧 hit/free 和视角条件，才判定真正时序冲突。",
            ]
        )
    )
    story.append(p("6.5 稳定性、弱证据分类与终态决策", "Heading2"))
    story.append(
        p(
            "终态需要在原始姿态和 ±0.20 m x/y、±0.50° yaw 等共 7 组变体上复核。"
            "旧默认通过率为 ≥0.80；高精度 profile 要求 1.0。固定候选失败时，只允许同目标受约束重拟合："
            "中心移动≤0.75 m、轴向 yaw 变化≤12°、双向最小覆盖≥0.60。稳定性执行的是完整终态重判，"
            "不是只比较候选框是否原样存在。"
        )
    )
    story.extend(
        bullets(
            [
                "weak evidence 被拆成 ownership、morphology、temporal、free_context、disposition，便于 Part 2 精确知道 blocker。",
                "决策优先处理数据质量和 partial scope；然后检查强 Occupied、强 Free、两者冲突、弱障碍与稳定性。",
                "任何矛盾或不完整门均保留 Unknown；Agent observability 只决定是否可交接，不改变 Part 1 状态。",
            ]
        )
    )

    story.append(p("6.6 Part 1 输出合同与文件", "Heading2"))
    story.append(
        table(
            ["文件/字段", "用途"],
            [
                ["summary.json", "局部数量、原因、gate failures、性能、cache、scope 与候选策略状态。"],
                ["slot_decisions.json / .csv", "逐车位状态、原因、证据、稳定性、弱分类和 trace refs。"],
                ["local_map.json / .png", "局部、不完整快照；A/B 仅 provisional display。"],
                ["local_slot_database.json", "只包含当前局部车位几何，避免把完整库伪装为感知结果。"],
                ["known_slot_scope.csv", "当前窗口的 scope 证据和遗漏原因。"],
                ["unknown_agent_queue.json", "只包含可执行 Unknown 与绑定资源。"],
                ["local_frame_manifest.json", "anchor、因果帧、源文件与哈希身份。"],
                ["camera_capability.json", "相机意图/有效能力；未通过 audit 时有效终态能力为空。"],
            ],
            widths=[58 * mm, 116 * mm],
        )
    )
    story.extend(
        figure(
            "Nature_ParkingAgent_实验报告_20260728/final_v5_assets/part1_v5_local_map.png",
            "图 6-2　frame 9277 最新 v5 W30/K5 局部 Part 1 地图。它不是全场占用图。",
            max_height=108 * mm,
        )
    )

    # 7 Part2
    story.append(PageBreak())
    story.append(p("7. Part 2：受限工具、多模态 Agent 与确定性硬门", "Heading1"))
    story.append(p("7.1 队列与分组", "Heading2"))
    story.append(
        p(
            "Part 2 的输入是不可变 QueueEnvelope。每个 QueueItem 绑定 task_id、slot_id、scope/state、"
            "unknown reasons、priority、available modalities、suggested tools、allowed final states、"
            "occupied/free 证据摘要、encounter、相邻/冲突关系和资源身份。重叠车位按 group 闭包处理，"
            "防止同一空间的多个车位被独立输出互相矛盾终态。"
        )
    )
    story.extend(
        figure(
            "Nature_ParkingAgent_实验报告_20260728/final_v5_assets/front180_v5_queue.png",
            "图 7-1　v5 前向 180° Part 2 队列。前向门只改变 Agent 候选资格，不修改 Part 1 状态。",
            max_height=100 * mm,
        )
    )
    story.append(p("7.2 Preflight 与证据目录", "Heading2"))
    story.extend(
        bullets(
            [
                "执行前核验 queue_id、dataset_id、config hash、slot map hash、encounter、帧身份和源文件 SHA-256。",
                "工具只能接收本次 preflight catalog 发放的 opaque evidence_id，不能自由访问文件系统路径。",
                "无有效 observation、资源缺失、哈希错误、工具类型不匹配或 schema 不兼容均返回显式 unavailable/invalid，不伪装成 Unknown 证据。",
                "Part 1 原 evidence pack 可审计但不算 Part 2 新动作；扩展 LiDAR 只有包含额外因果帧并绑定 decision sidecar 才可调用。",
            ]
        )
    )
    story.append(p("7.3 已实现工具", "Heading2"))
    story.append(
        table(
            ["工具/能力", "实现内容", "终态边界"],
            [
                ["LiDAR detail / inspect_lidar", "加载身份绑定 NPZ，渲染 1536×512 三联图与几何卡，返回数值 gate。", "可支持 Free/Occupied，但仍需 deterministic terminal gate。"],
                ["FOV / map context", "基于地图方位、相机视场和当前 pose 判断 visible/partial/not visible。", "只做工具路由，不等于像素投影。"],
                ["Camera context", "目标标记自车地图 + 未修改全帧图，帮助 Agent 建立目标方位。", "投影 audit 未过时不能单独终态。"],
                ["Camera crop", "Agent 提交规范化 bbox；要求已有 grounded localization hypothesis，记录支持/反证/歧义。", "bbox 不是 GT；非法、越界或缺前置上下文被拒绝。"],
                ["Camera sequence", "按时间排序的 1–5 帧因果 contact sheet，排除 t>t0。", "提供时序语义上下文，不能绕过几何门。"],
                ["RGB projected frame/sequence", "使用审计通过的 target/adjacent polygon 渲染内容寻址 PNG。", "当前真实数据 audit 未 passing，生产能力 fail closed。"],
            ],
            widths=[39 * mm, 80 * mm, 55 * mm],
            font_size=8.3,
        )
    )
    story.append(p("7.4 Agent 状态机与硬门", "Heading2"))
    story.extend(
        bullets(
            [
                "运行时先执行强制 FOV；Agent 每轮只能返回严格 JSON tool_request 或 final proposal。",
                "工具最多有限轮次；非法工具、重复无增益动作、字段缺失、非有限置信度和 provider 错误都有独立异常路径。",
                "模型置信度是未校准 heuristic score，不能直接解释为真实概率。",
                "模型提议 Free/Occupied 后，代码重新检查 allowed state、证据身份、互斥分数、终态几何、稳定性、group 一致性和相机能力。",
                "可信 Free 在 operational 模式触发全局 early stop；exhaustive 只用于离线评估并显式标记，不能与线上语义混淆。",
                "若合法工具耗尽或门不足，最终保持 Unknown；provider/schema/媒体错误属于 Error，而不是被吞成 Unknown。",
            ]
        )
    )
    story.append(p("7.5 模型适配器", "Heading2"))
    story.append(
        table(
            ["适配器", "能力与安全限制"],
            [
                ["ReplayModelAdapter", "按 case 重放冻结动作；可重绑定当前 evidence_id；用于确定性回归和无网络复现。"],
                ["OpenAI adapter", "严格根对象 schema、有限 retry budget、密钥文件权限检查、结构化 tool normalization、失败前置检查。"],
                ["LocalVLMAdapter", "只连接 localhost/loopback OpenAI-compatible /chat/completions；禁用环境代理，不启动服务、不搜索 provider、不接收 API key。"],
                ["Qwen 本地脚本", "仓库提供 Qwen3-VL-4B-Instruct-FP8 与相关启动/调用入口；真实可复现本地端点结果仍需单独冻结。"],
            ],
            widths=[43 * mm, 131 * mm],
        )
    )
    story.append(p("7.6 相机投影审计 v2", "Heading2"))
    story.append(
        p(
            "camera-projection-audit/2.0 是 RGB 终态能力的不可绕过门。它只接受 v2 reference set，"
            "要求 annotation、图片、LiDAR point source、pose、calibration、image/point registry 全部真实可读、"
            "hash 一致、身份去重、尺寸一致、变换来源明确且默认同步误差≤40 ms。加载时重新读取源字节并重算 metrics/status；"
            "v1 reference 不能伪造 passing。当前数据没有满足 v2 的独立真实像素 reference bundle，因此真实 RGB 仍 fail closed。"
        )
    )
    story.append(
        p(
            "现有 selected shadow 曾声明 22 张卡片、87 个 RGB 帧引用，但 audit gate 全部拒绝；"
            "媒体实现通过条件性内存测试完成 87/87 frame 与 22/22 sequence 的解码/裁剪/渲染，"
            "这只证明媒体桥接代码可用，不代表真实投影已经校准。",
            "Warning",
        )
    )

    # 8 Supporting
    story.append(PageBreak())
    story.append(p("8. 支撑功能、审计工具与报告体系", "Heading1"))
    story.append(
        table(
            ["功能组", "当前实现"],
            [
                ["旧 Box Scoring 迁移基线", "车位内车辆框搜索、帧选择、点过滤和无 GT 对照；已降级，不是执行主线。"],
                ["静态障碍层", "从 LiDAR 构建候选；只有受信动态审计可排除车辆，否则只出 review candidates，不直接写正式静态层。"],
                ["相机可观测地图", "地图方向预筛、camera observability、map-only precheck、FOV BEV 与多帧 review。"],
                ["可解释 LiDAR", "目标/邻接/高度层/体素/帧来源三联图、几何卡、弱证据文字与修复回归报告。"],
                ["独立 GT 工具", "中性几何全集、LiDAR/camera review、标注模板、盲评选择/预测/评分分离。"],
                ["人工标注 Web", "本地 ThreadingHTTPServer、case/summary/export API、sample-slot 身份校验、持久化 CSV/JSON。"],
                ["报告生成", "局部一页 PDF、中文故事报告、Nature/CVPR 草稿、Agent replay、global shadow、定位 trace、完整技术报告。"],
                ["数据完整性", "paired subset、自包含 archive、manifest、SHA-256、路径泄漏扫描和 source linkage。"],
            ],
            widths=[47 * mm, 127 * mm],
            font_size=8.4,
        )
    )

    # 9 Results
    story.append(PageBreak())
    story.append(p("9. 当前实验结果与证据解释", "Heading1"))
    story.append(p("9.1 三个必须分开的运行协议", "Heading2"))
    story.append(
        table(
            ["协议/产物", "输入", "Part 1", "Part 2", "解释"],
            [
                ["v5 最新评估，frame 9277", "W=30, K=5；5 帧", "3 F / 0 O / 19 U（22）", "前向 5 → 全部 U", "当前最新版研究运行点；静态语义硬否决。"],
                ["正式 15 帧局部快照", "连续 15 帧", "2 F / 8 O / 11 U（21）", "旧交接队列 11", "part1-local-lidar-map/1.1；与 v5 配置不同。"],
                ["历史全路线 v2", "11,673 帧路线级", "192 O / 57 F / 307 U；841 out", "44 task shadow：7 O / 37 U", "仅历史审计；不能解释为当前全场状态。"],
            ],
            widths=[39 * mm, 31 * mm, 43 * mm, 36 * mm, 45 * mm],
            font_size=8.0,
        )
    )
    story.append(p("9.2 v5 frame 9277", "Heading2"))
    story.extend(
        bullets(
            [
                "选中因果帧：9248、9255、9262、9270、9277。",
                "Part 1：22 个局部车位；3 Free、0 Occupied、19 Unknown；1375 个未覆盖车位省略。",
                "主要原因：3 strong_free_space_evidence、16 weak_obstacle_evidence、3 pose_unstable。",
                "Part 2 前向队列：slot_1250 / 1253 / 1254 / 1255 / 1256；8 次工具尝试，最终全 Unknown。",
                "这一结果显示 v5 以降低覆盖换取高精度/失败关闭，不能简单与旧 15 帧的 8 Occupied 作逐车位准确率比较。",
            ]
        )
    )
    story.extend(
        figure(
            "Nature_ParkingAgent_实验报告_20260728/final_v5_assets/final_model_comparison.png",
            "图 9-1　最终模型对比。请同时阅读各版本的采样、静态语义与 GT 分母，不能只比较终态数量。",
            max_height=92 * mm,
        )
    )
    story.append(p("9.3 W/K 开发消融与留出结果", "Heading2"))
    story.append(
        table(
            ["W/K", "样本数", "Resolved", "Resolved rate", "处理时间", "Dense exact agreement*"],
            [
                ["15/5", 25, 1, "4%", "9.53 s", "68%"],
                ["30/5", 25, 7, "28%", "9.86 s", "68%"],
                ["60/5", 25, 9, "36%", "12.12 s", "92%"],
                ["60/10", 25, 10, "40%", "16.85 s", "88%"],
                ["100/5", 25, 11, "44%", "12.19 s", "92%"],
                ["100/20", 25, 10, "40%", "23.12 s", "96%"],
                ["留出 15/15", 17, 0, "0%", "9.47 s", "94.1%"],
                ["留出 100/20", 17, 0, "0%", "8.74 s", "94.1%"],
            ],
            widths=[24 * mm, 25 * mm, 28 * mm, 32 * mm, 31 * mm, 34 * mm],
            font_size=8.3,
        )
    )
    story.append(
        p(
            "* Dense reference 不是 Ground Truth。开发集上更长历史能提高内部终态数，但留出锚点两端均为 0 resolved；"
            "因此不能宣称 W=100/K=20 普适最优。最新版选择 W30/K5 是综合计算、静态安全和保守覆盖后的工程运行点。",
            "Warning",
        )
    )
    story.extend(
        figure(
            "Nature_ParkingAgent_实验报告_20260728/final_v5_assets/wk_ablation_v5.png",
            "图 9-2　W/K 消融。Resolved 是内部硬门消歧数，不是分类准确率。",
            max_height=90 * mm,
        )
    )
    story.append(p("9.4 人工 GT 对照", "Heading2"))
    story.append(
        table(
            ["模型", "GT 交集", "终态", "正确终态", "选择性准确率", "终态覆盖", "False O / False F"],
            [
                ["v3", 14, 3, 2, "66.7%", "21.4%", "1 / 0"],
                ["v4 high precision", 14, 2, 2, "100%", "14.3%", "0 / 0"],
                ["historical extended60", 24, 12, 5, "41.7%", "50.0%", "7 / 0"],
            ],
            widths=[38 * mm, 22 * mm, 21 * mm, 26 * mm, 34 * mm, 28 * mm, 30 * mm],
            font_size=8.7,
        )
    )
    story.append(
        p(
            "frame 9277 冻结人工 GT 有 24 个车位，另有 slot_1252 的 1 个补充人工标签，合计 25；"
            "但不同预测只与其局部 scope 交集评价。v4 没有输出任何 Occupied，因此 Occupied precision 是 N/A，"
            "不能写成 100%。100% 选择性准确率只表示它选择作答的 2 个 Free 正确。",
            "Callout",
        )
    )
    story.append(p("9.5 冻结历史挑战", "Heading2"))
    story.append(
        table(
            ["版本", "可观测 GT", "终态作答", "选择性准确率", "覆盖率", "Occupied recall", "终态矛盾"],
            [
                ["旧版本", 6, 4, "75%", "66.7%", "75%", 1],
                ["v5 W30/K5", 6, 1, "100%", "16.7%", "25%", 0],
            ],
            widths=[35 * mm, 26 * mm, 28 * mm, 34 * mm, 27 * mm, 31 * mm, 28 * mm],
            font_size=8.3,
        )
    )
    story.append(
        p(
            "标签虽在回放前冻结，但案例来自历史 evidence-selection，只有 4 Occupied + 2 Free 可观测样本；"
            "因此状态是 retrospective frozen challenge，不是新的独立 holdout。v5 的结论是“已减少矛盾但显著降低覆盖”，"
            "而不是已经证明总体准确率。",
            "Warning",
        )
    )
    story.append(p("9.6 Part 2 clean-agent shadow", "Heading2"))
    story.extend(
        bullets(
            [
                "5 个固定回归 seed + 5 strata×2 强 O×2 强 F，经 group closure 从 25 seeds 展开为 47 个车位。",
                "47 个 Part 1 全部 Unknown；44 个有可执行 LiDAR 卡片，3 个 no_valid_observations 不伪造任务。",
                "44/44 三联图生成成功；相机 87 帧引用因 projection audit 缺失全部无终态能力。",
                "盲代理提出并经 hard gate 保留 7 Occupied / 37 Unknown / 0 Free；两次回放逐字节一致。",
                "7 个 Unknown→Occupied 为 0784、0991、1050、1054、1071、1267、1311，但没有对应独立 GT，不能称为正确。",
            ]
        )
    )
    story.extend(
        figure(
            "outputs/slot_part2_clean_agent_shadow_v1/global_map_review/global_map_part2_evaluated_before_after.png",
            "图 9-3　历史 Part 2 shadow 的 44 个真正 evaluated 车位前后对比；背景车位不属于当前局部感知。",
            max_height=105 * mm,
        )
    )

    # 10 Testing
    story.append(PageBreak())
    story.append(p("10. 测试、质量与失败关闭验证", "Heading1"))
    story.append(
        p(
            "仓库当前有 70 个 test_*.py 文件、573 个以 test_ 命名的测试函数。"
            "在现有 .venv-vlm 环境中，完整收集被 tests/test_slot_aligned_accumulation.py 的可选依赖 "
            "scikit-learn 缺失阻断；排除该单个实验测试文件后，执行结果为 <b>546 passed, 239 subtests passed</b>，"
            "耗时约 30.05 s。该结果不把未收集测试算作通过。"
        )
    )
    story.append(
        table(
            ["测试域", "覆盖重点"],
            [
                ["Pose/同步", "最近时间匹配、漂移校正、报告和异常输入。"],
                ["Part 1 contracts/IO", "schema 冻结、有限数、重复 ID、canonical writer、cache 与错误。"],
                ["Ground/accumulation", "斜地面、退化回退、frame provenance、missing/invalid frame。"],
                ["Occupied/Free", "每个 gate failure code、柱体、车辆形态、ray traversal、遮挡、冲突。"],
                ["Scope/local map", "3–15 帧、因果尾窗、隐藏远处、A/B shortlist、不可提升 candidate。"],
                ["Stability/semantics", "7 变体、受约束重拟合、静态语义、终态不变量。"],
                ["Part 2 queue/tools", "opaque ID、哈希/身份、group closure、tool allowlist、媒体隐私。"],
                ["Camera", "标定、FOV、observability、projection audit v2、RGB frame/sequence。"],
                ["Agent adapters", "严格 JSON、密钥权限、loopback、本地 VLM、retry、replay/resume。"],
                ["GT/报告", "评价分母、空指标为 null、标注存储、可视化与报告数字一致性。"],
            ],
            widths=[45 * mm, 129 * mm],
            font_size=8.4,
        )
    )

    # 11 Repro
    story.append(PageBreak())
    story.append(p("11. 运行入口、复现与产物位置", "Heading1"))
    story.append(p("11.1 主流程入口", "Heading2"))
    story.append(
        table(
            ["入口", "用途"],
            [
                ["scripts/run_pose_drift_correction.py", "生成关键帧校正和 corrected trajectory。"],
                ["scripts/build_frame_map_dataset.py", "使用 corrected pose 重投影逐帧 LiDAR 并同步相机。"],
                ["scripts/run_hybrid_3d_slot_evidence.py", "运行 Part 1 scope/full/local；支持 anchor、窗口与 config。"],
                ["scripts/run_part2_agent.py", "prepare-shadow、replay 或 loopback local VLM 的 Part 2 执行。"],
                ["scripts/run_autonomous_parking_workflow.py", "串联 Part 1/queue/Part 2/semantic result 的工作流 manifest。"],
                ["scripts/evaluate_part1_manual_gt.py", "按 GT 交集计算选择性准确率、覆盖和 false terminal。"],
                ["scripts/score_v5_frozen_challenge.py", "回顾性冻结挑战评分。"],
                ["scripts/build_final_v5_report.py", "生成 v5 汇总首页、图和 final_v5_study.json。"],
            ],
            widths=[65 * mm, 109 * mm],
            font_size=8.3,
        )
    )
    story.append(p("11.2 关键保留产物", "Heading2"))
    story.extend(
        bullets(
            [
                "outputs/pose_drift_correction_v3/：pose correction、残差和 review。",
                "outputs/frame_map_dataset_pose_corrected_final/：校正后逐帧 map_points 与 frames.csv。",
                "outputs/full_icpark_allframes_vehicle_cluster/slot_database.json：1397 车位几何数据库。",
                "outputs/part1_local_lidar_frame_9277/：正式 15 帧局部 Part 1 快照。",
                "Nature_ParkingAgent_实验报告_20260728/artifacts/frame_009277_v5_static_semantic_w30k5/：最新版 v5 frame 9277 产物。",
                "Nature_ParkingAgent_实验报告_20260728/results/final_v5_study.json：最新版机器可读总汇。",
                "outputs/slot_part2_clean_agent_shadow_v1/：历史 Part 2 无 GT shadow 与逐字节回放。",
                "protected_artifacts/human_validation/：受保护人工标签；不得被实验脚本覆盖。",
            ]
        )
    )
    story.append(
        p(
            "复现命令与 selection/predict/score 隔离流程见 Nature_ParkingAgent_实验报告_20260728/REPRODUCE.md。"
            "历史 route-shadow 使用 11,673 帧，而当前正式 CLI writer 是局部合同；如需重建 route-shadow，"
            "必须提供专用、显式标记 ROUTE SHADOW / NO GT 的 runner，不能用局部 CLI 冒充。",
            "Warning",
        )
    )

    # 12 Limitations
    story.append(PageBreak())
    story.append(p("12. 已知限制、风险与下一步", "Heading1"))
    story.append(p("12.1 当前不能宣称的能力", "Heading2"))
    story.extend(
        bullets(
            [
                "不能宣称全停车场实时占用真值；正式输出是当前局部、不完整证据快照。",
                "不能宣称总体分类准确率、Nature/CVPR 级统计显著性或生产安全性。",
                "不能宣称 RGB 已完成精确车位像素投影；camera audit v2 尚无 passing reference bundle。",
                "不能宣称 W30/K5、W60 或 W100/K20 跨场景普适最优；开发/留出表现存在明显差异。",
                "不能把 Agent 未校准 confidence 当作概率，也不能把语言理由当作安全证明。",
                "不能把历史 shadow 的 7 个 Occupied 当作正确标签；它只证明流程可执行和可确定回放。",
                "不能执行车辆运动控制、路径规划或最终停车目标选择；生产 selection policy 仍为 TBD。",
            ]
        )
    )
    story.append(p("12.2 主要工程风险", "Heading2"))
    story.append(
        table(
            ["风险", "影响", "现有缓解", "建议"],
            [
                ["定位/地图误差", "车位归属与边界比例错误", "0.10 m uncertainty、7 变体稳定性、pose audit", "用独立定位误差分布校准扰动宽度"],
                ["静态结构污染", "柱体/墙/轮挡误判 Occupied", "PCA/footprint/upper spread/glTF 静态 gate", "扩大冻结 GT 静态结构样本"],
                ["长历史动态过时", "车辆移动造成时序冲突", "因果约束、frame provenance、冲突门", "按视点/覆盖增益动态停止"],
                ["相机投影不可信", "错误 crop 影响 Agent", "audit v2 fail closed", "建立独立像素 reference 和双人标注"],
                ["GT 数量小", "指标区间宽、易过拟合", "selection/predict/score 隔离、Unknown abstention", "多路线、多时段、盲法双标注与仲裁"],
                ["模型/provider 失败", "非结构化输出或服务中断", "strict schema、retry、Error 分支、replay", "冻结本地模型版本和端点产物"],
            ],
            widths=[35 * mm, 42 * mm, 55 * mm, 42 * mm],
            font_size=8.05,
        )
    )
    story.append(p("12.3 建议优先级", "Heading2"))
    story.extend(
        bullets(
            [
                "<b>P0：</b>冻结多路线独立 GT，至少双标注者、盲法、仲裁与 inter-annotator agreement；先建立风险—覆盖曲线。",
                "<b>P0：</b>完成 camera-projection-reference-set/2.0 的真实像素基准并通过 audit，之后才开放 RGB 终态能力。",
                "<b>P1：</b>实现独立 route-shadow runner，明确局部与路线级 writer，避免协议漂移。",
                "<b>P1：</b>补齐 scikit-learn 环境依赖并把全部 573 测试纳入统一锁定环境。",
                "<b>P1：</b>在同一 GT/候选/early-stop 下比较 15 帧、v5、无静态 gate、无稳定性、deterministic-only、本地 VLM 与 OpenAI。",
                "<b>P2：</b>定义生产停车选择效用、可达性、规划接口和安全停止策略；在此之前保持 candidate=null。",
            ]
        )
    )

    # 13 Summary
    story.append(PageBreak())
    story.append(p("13. 总结", "Heading1"))
    story.append(
        p(
            "ParkingAgent 当前已经实现了一套完整的、可追溯的两阶段停车位占用证据系统。"
            "Part 1 通过 corrected pose、已知车位几何、局部地面、多帧 3D 形态、射线自由空间、"
            "柱体/静态语义否决与稳定性复核，输出 Free/Occupied/Unknown；Part 2 只接收可执行 Unknown，"
            "用身份绑定的 LiDAR/RGB 工具、结构化 Agent 和不可绕过的 deterministic gate 消歧。"
            "工程上最成熟的部分是数据合同、失败关闭、可解释证据、回放和审计；最主要的研究缺口是独立 GT、"
            "真实相机投影验收、跨路线泛化和生产候选/控制策略。"
        )
    )
    story.append(
        p(
            "<b>一句话结论：</b>目前系统已能可靠地“只在证据充分时作答、证据不足时明确拒判”，"
            "但还不能把这种工程可靠性等同为最终占用准确率或自动泊车能力。",
            "Callout",
        )
    )

    # Appendices
    story.append(PageBreak())
    story.append(p("附录 A：关键阈值总表", "Heading1"))
    story.append(
        table(
            ["类别", "参数", "默认/高精度值", "语义"],
            [
                ["Scope", "max distance / near frames / ray frames", "25 m / 3 / 3", "进入局部观测范围的基础门"],
                ["Ground", "min points / inlier / residual P95", "30 / 0.45 / 0.15 m", "逐帧地面质量"],
                ["Voxel", "xy / z", "0.25 / 0.20 m", "3D 证据离散化"],
                ["Occupied", "points / support frames / layers", "40 / 3 / 2", "最小空间、时间、垂直支持"],
                ["Occupied", "z95 / height span / voxels", "0.60 / 0.35 m / 8", "排除低矮与稀疏结构"],
                ["Ownership", "core / adjacent / boundary", "≥0.45 / ≤0.35 / <0.50", "绑定目标并抑制邻接/边界污染"],
                ["Shape", "XY linearity / outside residual", "<0.60 / ≤0.60（HP 0.35）", "排除线状或外部结构"],
                ["High precision", "robust short extent / low BEV / upper spread", "≥0.75 m / ≥0.05 / ≥0.05", "柱体、水平盖板与窄残余否决"],
                ["Free", "ray frames / viewpoints / separation", "≥5 / ≥2 / ≥10°", "多帧多视角正向自由空间"],
                ["Free", "volume / near-ground coverage", "≥0.70 / ≥0.70", "核心体积与停车 footprint 覆盖"],
                ["Free", "unobserved / occlusion", "≤0.20 / ≤0.20", "限制不可见区域"],
                ["Stability", "translation / yaw variants", "±0.20 m / ±0.50°", "定位与地面扰动复核"],
                ["Stability", "pass ratio", "默认≥0.80；HP=1.0", "终态稳定门"],
                ["Part 2", "candidate half FOV", "90°", "总前向视场 180°"],
            ],
            widths=[30 * mm, 64 * mm, 42 * mm, 38 * mm],
            font_size=8.0,
        )
    )

    story.append(p("附录 B：主要证据来源", "Heading1"))
    story.append(
        table(
            ["来源", "本报告使用内容"],
            [
                ["README.md", "当前主线、状态合同、局部/历史边界、运行与目录说明。"],
                ["parking_slot_hybrid_3d/config.py", "Part 1 默认阈值和可验证配置合同。"],
                ["Nature…/config/production_hybrid3d_v4_high_precision.json", "高精度/v5 安全 gate profile。"],
                ["Nature…/results/final_v5_study.json", "最新版 W30/K5、frame 9277、Part 2 与 frozen challenge 总汇。"],
                ["Nature…/results/frame9277_part1_manual_gt_comparison_v4.json", "v3/v4/extended60 人工 GT 交集指标。"],
                ["Nature…/results/Table_3_development_WK.csv", "开发 W/K 消融。"],
                ["Nature…/results/Table_4_heldout.csv", "留出端点对照。"],
                ["outputs/part1_local_lidar_frame_9277/summary.json", "正式 15 帧局部快照统计。"],
                ["outputs/slot_part2_clean_agent_shadow_v1/", "历史无 GT shadow、回放与地图对比。"],
                ["parking_slot_part2 / parking_slot_agent_v2", "队列、工具、媒体、模型适配、决策与报告实现。"],
                ["tests/ 与 parking_slot_agent_v2/tests/", "功能合同与失败关闭测试覆盖。"],
            ],
            widths=[74 * mm, 100 * mm],
            font_size=8.7,
        )
    )
    story.append(
        p(
            "本报告没有从互联网引入新的性能主张；所有结论均来自当前工作区代码、配置、机器可读产物和测试执行。"
            "历史报告若与 2026-07-29 的 final_v5_study 冲突，以后者作为最新版评估证据，同时保留正式 15 帧 writer 的独立工程合同。",
            "Small",
        )
    )
    return story


def build() -> Path:
    register_font()
    global S
    S = styles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
        title="ParkingAgent 完整技术报告：Part 1 + Part 2",
        author="ParkingAgent / Codex",
        subject="ParkingAgent current implementation, algorithms, evidence and limitations",
        creator="scripts/build_complete_technical_report_zh.py",
    )
    doc.multiBuild(build_story())
    return OUT_PDF


if __name__ == "__main__":
    path = build()
    print(path)
