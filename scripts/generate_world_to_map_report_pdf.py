#!/usr/bin/env python3
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "reports"
OUT_PDF = OUT_DIR / "world_to_map_missing_extrinsic_report.pdf"


def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("STSong-Light", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 12 * mm, "ParkingAgent 定位与地图匹配问题说明")
    canvas.drawRightString(192 * mm, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def p(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullet(text, style):
    return Paragraph(f"• {text}", style)


def code_block(text, style):
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Table(
        [[Paragraph(escaped, style)]],
        colWidths=[170 * mm],
        style=[
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D8DEE9")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ],
    )


def build_pdf():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=base["Title"],
        fontName="STSong-Light",
        fontSize=22,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
        spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=base["Normal"],
        fontName="STSong-Light",
        fontSize=11,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=20,
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=base["Heading1"],
        fontName="STSong-Light",
        fontSize=15,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=base["Heading2"],
        fontName="STSong-Light",
        fontSize=12.5,
        leading=19,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=base["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=17,
        alignment=TA_JUSTIFY,
        firstLineIndent=0,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "SmallCN",
        parent=body,
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#374151"),
    )
    code = ParagraphStyle(
        "CodeCN",
        parent=base["Code"],
        fontName="STSong-Light",
        fontSize=9.2,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="当前定位与地图匹配问题说明报告",
        author="ParkingAgent",
    )

    story = []
    story.append(p("当前定位与地图匹配问题说明报告", title))
    story.append(p("关于 world_to_map 外参缺失、此前视频生成方式及后续解决路径", subtitle))

    story.append(p("1. 项目目标", h1))
    story.append(
        p(
            "本项目希望利用数据集中的 LiDAR 点云、车辆 pose 和 ICpark 停车场地图，实现车辆在停车场地图中的实时定位，并进一步判断当前车辆附近哪些停车位被占据、哪些停车位可以作为候选空位。",
            body,
        )
    )
    story.append(p("最终需要在 2D 地图上展示车辆位置、车辆朝向、完整轨迹、停车位编号、被占据车位以及候选空车位。", body))

    story.append(p("2. 当前已有数据", h1))
    data_table = Table(
        [
            ["数据", "作用", "当前状态"],
            ["poses.txt", "提供车辆在 dataset/world 坐标系下的连续位姿", "可用"],
            ["LiDAR 点云", "用于观察障碍物、判断车位占据、辅助地图匹配", "可用"],
            ["相机图像", "用于查看当前视野，可与 LiDAR 投影结果对齐", "可用"],
            ["相机-LiDAR 标定", "用于将 LiDAR 点投影到同帧相机图片", "可用"],
            ["ICpark 地图", "提供停车位、墙体、车道线、阻车器、地面边界", "可用"],
            ["world_to_map", "将 dataset/world 坐标转换到 ICpark/map 坐标", "缺失"],
        ],
        colWidths=[32 * mm, 94 * mm, 35 * mm],
    )
    data_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "STSong-Light", 9.2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(data_table)
    story.append(Spacer(1, 4 * mm))

    story.append(p("3. 已完成的验证工作", h1))
    for item in [
        "已成功读取 poses.txt，并提取车辆连续轨迹。",
        "已成功解析 ICpark 地图数据，确认其中包含 1397 个停车位，以及墙体、车道线、阻车器和地面边界。",
        "已生成过轨迹图、HTML 动态播放和部分视频可视化结果。",
        "已验证 LiDAR 点云可以根据相机内参和 LiDAR-相机外参投影到同一帧相机图像。",
        "已做过单帧车位占据评分 demo。",
        "已做过 LiDAR/NDT 地图匹配原型测试，但结果不能作为准确定位结果交付。",
    ]:
        story.append(bullet(item, body))

    story.append(p("4. 之前视频和动态图是如何生成的", h1))
    story.append(
        p(
            "之前的视频和动态图主要是为了展示车辆轨迹和地图渲染效果，不是严格意义上的 LiDAR + pose 地图定位结果。它主要使用了 poses.txt 中的车辆轨迹，以及 ICpark 地图结构。",
            body,
        )
    )
    story.append(
        code_block(
            "读取 poses.txt\n"
            "提取每帧车辆 x、y、yaw\n"
            "读取 ICpark 地图边界\n"
            "将 pose 轨迹通过缩放和平移放入地图显示范围\n"
            "绘制车辆图标、历史轨迹、起点、终点和地图背景\n"
            "导出图片、HTML 或视频",
            code,
        )
    )
    story.append(
        p(
            "这个流程中的关键问题是：轨迹叠加到地图上使用的是可视化拟合，而不是数据集提供的真实坐标变换。因此它可以说明 pose 轨迹能读、地图能画、动画能播放，但不能证明车辆真实位于图中显示的那个停车位旁边。",
            body,
        )
    )

    story.append(p("5. LiDAR 在之前结果中的作用", h1))
    story.append(
        p(
            "之前确实做过 LiDAR 相关实验，包括点云显示、点云投影到相机图像、NDT 原型测试和单帧车位占据评分。但是之前的动态轨迹视频并不是逐帧用 LiDAR 匹配地图算出来的。",
            body,
        )
    )
    story.append(p("更准确地说，之前的视频属于 pose 驱动的轨迹可视化，LiDAR 没有参与逐帧地图定位闭环。", body))

    story.append(p("6. 当前真正缺少的外参", h1))
    story.append(
        p(
            "当前缺少的不是 LiDAR 到相机的外参。LiDAR 到相机的外参在数据集中已经存在，可以用于将点云投影到同一帧相机图像。",
            body,
        )
    )
    story.append(p("当前真正缺少的是 dataset/world 坐标系到 ICpark/map 坐标系的外参，也就是 world_to_map。", body))
    story.append(code_block("p_map = scale * R(yaw) * p_world + translation", code))
    story.append(p("这个外参需要确定 scale、rotation、translation_x 和 translation_y。它的作用是把 poses.txt 中的车辆位置转换到 ICpark 地图坐标中。", body))

    story.append(PageBreak())

    story.append(p("7. 为什么这个外参非常关键", h1))
    story.append(
        p(
            "如果没有 world_to_map，系统只能知道车辆在 dataset/world 坐标系里的位置，无法知道这个位置对应 ICpark 地图中的哪个道路、哪个车位旁边或哪个区域。",
            body,
        )
    )
    story.append(p("同样，LiDAR 点云也无法可靠投到地图坐标中。这样就无法判断某个障碍物点是否落在某个停车位 polygon 内。", body))
    story.append(
        code_block(
            "有 world_to_map：\n"
            "  pose/world -> map 中车辆位置\n"
            "  LiDAR/world -> map 中障碍物点\n"
            "  parkingSpace polygon -> 可直接判断占据\n\n"
            "没有 world_to_map：\n"
            "  只能做 pose 轨迹展示\n"
            "  不能做准确地图定位\n"
            "  不能可靠判断具体哪个车位被占",
            code,
        )
    )

    story.append(p("8. 为什么不能只靠简单缩放解决", h1))
    story.append(
        p(
            "简单缩放可以把轨迹放进地图范围内，但不能保证它是真实对齐。两个坐标系之间不仅可能存在尺度差异，还可能存在平移差异、旋转差异、原点差异、坐标轴方向差异和比例差异。",
            body,
        )
    )
    story.append(
        p(
            "如果只是按照 bounding box 把 world 轨迹缩放到地图范围内，视觉上可能看起来合理，但它没有证明转弯点、道路、墙体、车位和 LiDAR 静态结构真实对齐。因此它只能作为展示或粗初值，不能作为最终定位结果。",
            body,
        )
    )

    story.append(p("9. 为什么 NDT 不能从零解决全部问题", h1))
    story.append(
        p(
            "NDT 或 ICP 可以用于 LiDAR 和地图的匹配，但它们更适合在已有较好初值的情况下做局部精修。如果没有 world_to_map 初值，直接在整张停车场地图中全局搜索，会面临较大的不确定性。",
            body,
        )
    )
    for item in [
        "停车场结构重复，大量车位和通道形状相似。",
        "LiDAR 点云相对稀疏，局部特征不一定唯一。",
        "当前环境中存在动态车辆、行人或临时障碍物，会干扰静态地图匹配。",
        "全局搜索空间大，容易陷入错误局部最优。",
        "匹配结果可能局部看起来合理，但全局位置错误。",
    ]:
        story.append(bullet(item, body))
    story.append(p("因此，NDT 更适合在 pose 和 world_to_map 给出的初值附近做小范围修正，而不是直接替代 world_to_map。", body))

    story.append(p("10. 后续解决方案", h1))
    story.append(p("后续工作的核心是先获得 world_to_map，然后再做 LiDAR 精修和车位占据判断。", body))
    story.append(p("推荐流程如下：", h2))
    story.append(
        code_block(
            "读取 ICpark 语义地图\n"
            "读取 poses.txt 轨迹\n"
            "求解 world_to_map\n"
            "将每帧车辆 pose 投到地图坐标\n"
            "将 LiDAR 点云投到地图坐标\n"
            "在 parkingSpace polygon 中统计障碍物点\n"
            "输出 occupied / candidate / unknown",
            code,
        )
    )
    story.append(p("world_to_map 可以通过两种方式获得。", body))
    story.append(p("第一种是人工少量点标定。选择 3 到 5 对对应点，求解 scale、rotation 和 translation。这种方式最稳，成本也最低。", body))
    story.append(p("第二种是自动地图配准。使用多帧 LiDAR 和 pose 累积 BEV 点云地图，再和 ICpark 地图中的墙体、车道线、阻车器等结构匹配。这种方式不需要人工参与，但由于停车场结构重复，必须对结果做严格验证。", body))

    story.append(p("11. 当前结论", h1))
    story.append(
        p(
            "当前项目已经具备 pose 轨迹读取、ICpark 地图解析、LiDAR 点云读取、LiDAR 投影相机图像、初步车位评分和轨迹可视化能力。但准确的停车场地图定位还没有完成。",
            body,
        )
    )
    story.append(
        p(
            "核心原因是缺少 dataset/world 到 ICpark/map 的外参 world_to_map。之前生成的视频只是将 pose 轨迹缩放和平移后叠加到地图上，属于可视化结果，不是准确定位结果。",
            body,
        )
    )
    story.append(
        p(
            "要实现真正可靠的车辆地图定位、完整轨迹绘制、LiDAR 点云落图、停车位占据判断和候选车位筛选，必须先解决 world_to_map 的获取问题。",
            body,
        )
    )

    story.append(Spacer(1, 6 * mm))
    story.append(
        p(
            "简要结论：现在系统不是缺 LiDAR-相机外参，而是缺 dataset/world 到 ICpark/map 的外参。没有这个外参，之前的地图视频只能算轨迹展示，不能算准确定位。",
            small,
        )
    )

    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)


if __name__ == "__main__":
    build_pdf()
    print(OUT_PDF)
