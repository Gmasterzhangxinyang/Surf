#!/usr/bin/env python3
"""Render the ParkingAgent IEEE-style manuscript to a polished two-column PDF."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


from build_parkingagent_scientific_diagrams import agent_drawing, system_drawing

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Nature_ParkingAgent_实验报告_20260728/paper_parkingagent_20260801"
TEX = PAPER / "parkingagent.tex"
BIB = PAPER / "references.bib"
PDF = PAPER / "ParkingAgent_Evidence_Gated_Multimodal_Parking.pdf"

PAGE_W, PAGE_H = LETTER
MARGIN_X = 0.53 * inch
MARGIN_Y = 0.48 * inch

pdfmetrics.registerFont(TTFont("DejaVuSerif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSerifItalic", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
GUTTER = 0.22 * inch
COL_W = (PAGE_W - 2 * MARGIN_X - GUTTER) / 2


REF_KEYS = [
    "suhr2013parking", "zinelli2019deep", "avpslam2020", "pointpillars2019",
    "centerpoint2021", "bevfusion2022", "icp1992", "ndt2003", "react2023",
    "toolformer2023", "qwen2vl2024", "geifman2017selective",
    "guo2017calibration", "kendall2017uncertainty",
]
REF_NUM = {key: idx + 1 for idx, key in enumerate(REF_KEYS)}

REF_TEXT = [
    "J. K. Suhr and H. G. Jung, “Full-automatic recognition of various parking slot markings using a hierarchical tree structure,” Optical Engineering, 2013.",
    "A. Zinelli, L. Musto, and F. Pizzati, “A deep-learning approach for parking slot detection on surround-view images,” IEEE Intelligent Vehicles Symposium, 2019.",
    "T. Qin, T. Chen, Y. Chen, and Q. Su, “AVP-SLAM: Semantic visual mapping and localization for autonomous vehicles in the parking lot,” IEEE/RSJ IROS, 2020.",
    "A. H. Lang et al., “PointPillars: Fast encoders for object detection from point clouds,” IEEE/CVF CVPR, 2019.",
    "T. Yin, X. Zhou, and P. Krähenbühl, “Center-based 3D object detection and tracking,” IEEE/CVF CVPR, 2021.",
    "T. Liang et al., “BEVFusion: A simple and robust LiDAR-camera fusion framework,” NeurIPS, 2022.",
    "P. J. Besl and N. D. McKay, “A method for registration of 3-D shapes,” IEEE TPAMI, 1992.",
    "P. Biber and W. Straßer, “The normal distributions transform: A new approach to laser scan matching,” IEEE/RSJ IROS, 2003.",
    "S. Yao et al., “ReAct: Synergizing reasoning and acting in language models,” ICLR, 2023.",
    "T. Schick et al., “Toolformer: Language models can teach themselves to use tools,” NeurIPS, 2023.",
    "Qwen Team, “Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution,” arXiv:2409.12191, 2024.",
    "Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” NeurIPS, 2017.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” ICML, 2017.",
    "A. Kendall and Y. Gal, “What uncertainties do we need in Bayesian deep learning for computer vision?” NeurIPS, 2017.",
]

EQUATIONS = {
    1: r"\hat y_i\in\{\mathrm{Free},\mathrm{Occupied},\mathrm{Unknown}\}",
    2: r"j_m=\operatorname{round}\!\left(\frac{m(W-1)}{K-1}\right),\quad \mathcal{H}_{t_0}(W,K)=\{\mathcal{P}_{t_0}^{W}[j_m]\}_{m=0}^{K-1},\quad K\geq2",
    3: r"\hat y_i^{(1)}=\mathrm{F}\ \mathrm{if}\ G_i^F\wedge\neg G_i^O;\quad \mathrm{O}\ \mathrm{if}\ G_i^O\wedge\neg G_i^F;\quad \mathrm{U}\ \mathrm{otherwise}",
    4: r"q_i=\mathbb{1}[\hat y_i^{(1)}=\mathrm{U}]\cdot\mathbb{1}[d_i\leq r]\cdot\mathbb{1}[|\mathrm{wrap}(\psi_i-\psi_{ego})|\leq\pi/2]",
    5: r"a_k\sim\pi_\theta(\cdot\mid b_k,\mathcal{T}_k),\quad e_{k+1}=\mathrm{Tool}(a_k),\quad b_{k+1}=\mathcal{U}(b_k,e_{k+1})",
    6: r"V_i=\mathbb{1}[\mathcal{E}^{cite}_i\subseteq\mathcal{E}^{obs}_i]\,\mathbb{1}[g_{mod}(\hat y_i,\mathcal{E}^{obs}_i)=1]\,\mathbb{1}[p(\hat y_i)\geq\tau]",
}


EQUATION_MARKUP = {
    1: "<i>ŷ</i><sub>i</sub> ∈ {Free, Occupied, Unknown}",
    2: "<i>j</i><sub>m</sub> = round( <i>m</i>(<i>W</i>−1)/(<i>K</i>−1) )<br/><i>H</i><sub>t0</sub>(<i>W</i>,<i>K</i>) = { <i>P</i><super>W</super><sub>t0</sub>[<i>j</i><sub>m</sub>] : <i>m</i>=0,…,<i>K</i>−1 }",
    3: "<i>ŷ</i><super>(1)</super><sub>i</sub> = F if <i>G</i><super>F</super><sub>i</sub> ∧ ¬<i>G</i><super>O</super><sub>i</sub>; O if <i>G</i><super>O</super><sub>i</sub> ∧ ¬<i>G</i><super>F</super><sub>i</sub>; U otherwise",
    4: "<i>q</i><sub>i</sub> = 1[<i>ŷ</i><super>(1)</super><sub>i</sub>=U] · 1[<i>d</i><sub>i</sub>≤<i>r</i>] · 1[|wrap(<i>ψ</i><sub>i</sub>−<i>ψ</i><sub>ego</sub>)|≤<i>π</i>/2]",
    5: "<i>a</i><sub>k</sub> ∼ <i>π</i><sub>θ</sub>(· | <i>b</i><sub>k</sub>, <i>T</i><sub>k</sub>),   <i>e</i><sub>k+1</sub> = Tool(<i>a</i><sub>k</sub>)<br/><i>b</i><sub>k+1</sub> = U(<i>b</i><sub>k</sub>, <i>e</i><sub>k+1</sub>)",
    6: "<i>V</i><sub>i</sub> = 1[<i>E</i><super>cite</super><sub>i</sub> ⊆ <i>E</i><super>obs</super><sub>i</sub>] · 1[<i>g</i><sub>mod</sub>(<i>ŷ</i><sub>i</sub>,<i>E</i><super>obs</super><sub>i</sub>)=1] · 1[<i>p</i>(<i>ŷ</i><sub>i</sub>)≥<i>τ</i>]",
}


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Times-Bold", fontSize=18.5, leading=20.5, alignment=TA_CENTER, spaceAfter=5),
        "authors": ParagraphStyle("Authors", parent=base["Normal"], fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=5),
        "abstract": ParagraphStyle("Abstract", parent=base["Normal"], fontName="Times-Roman", fontSize=7.8, leading=9.2, alignment=TA_JUSTIFY, leftIndent=12, rightIndent=12),
        "section": ParagraphStyle("Section", parent=base["Heading1"], fontName="Times-Bold", fontSize=9.5, leading=11, alignment=TA_CENTER, spaceBefore=6, spaceAfter=3, keepWithNext=True),
        "subsection": ParagraphStyle("Subsection", parent=base["Heading2"], fontName="Times-BoldItalic", fontSize=8.8, leading=10, alignment=TA_LEFT, spaceBefore=4, spaceAfter=2, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.25, leading=9.65, alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=3.1, splitLongWords=False),
        "caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Times-Roman", fontSize=7.2, leading=8.4, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=5),
        "tablecap": ParagraphStyle("TableCaption", parent=base["Normal"], fontName="Times-Roman", fontSize=7.1, leading=8.2, alignment=TA_CENTER, spaceBefore=3, spaceAfter=3, keepWithNext=True),
        "equation": ParagraphStyle("Equation", parent=base["Normal"], fontName="Times-Italic", fontSize=8.4, leading=10, alignment=TA_CENTER, leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=4),
        "refs": ParagraphStyle("Refs", parent=base["Normal"], fontName="Times-Roman", fontSize=6.8, leading=7.8, alignment=TA_LEFT, leftIndent=12, firstLineIndent=-12, spaceAfter=2),
        "small": ParagraphStyle("Small", parent=base["Normal"], fontName="Times-Roman", fontSize=7, leading=8.2, alignment=TA_JUSTIFY),
    }


S = styles()


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.35)
    canvas.line(MARGIN_X, PAGE_H - 0.34 * inch, PAGE_W - MARGIN_X, PAGE_H - 0.34 * inch)
    canvas.setFont("Times-Roman", 6.7)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(MARGIN_X, 0.27 * inch, "ParkingAgent — manuscript draft")
    canvas.drawRightString(PAGE_W - MARGIN_X, 0.27 * inch, str(doc.page))
    canvas.restoreState()


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=LETTER, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=MARGIN_Y, bottomMargin=MARGIN_Y,
                         title="ParkingAgent: Evidence-Gated Multimodal Reasoning", author="Anonymous")
        col_h = PAGE_H - 2 * MARGIN_Y
        # Keep title, author line, and the full abstract in the spanning frame.
        # If the abstract spills into the first column, the following explicit
        # FrameBreak skips that column and leaves a visually empty half-page.
        title_h = 2.78 * inch
        first_top = Frame(MARGIN_X, PAGE_H - MARGIN_Y - title_h, PAGE_W - 2 * MARGIN_X, title_h, id="first_title", leftPadding=0, rightPadding=0, topPadding=6, bottomPadding=4)
        first_bottom_h = col_h - title_h - 0.05 * inch
        first_left = Frame(MARGIN_X, MARGIN_Y, COL_W, first_bottom_h, id="first_left", leftPadding=0, rightPadding=5, topPadding=3, bottomPadding=0)
        first_right = Frame(MARGIN_X + COL_W + GUTTER, MARGIN_Y, COL_W, first_bottom_h, id="first_right", leftPadding=5, rightPadding=0, topPadding=3, bottomPadding=0)

        left = Frame(MARGIN_X, MARGIN_Y, COL_W, col_h, id="left", leftPadding=0, rightPadding=5, topPadding=2, bottomPadding=0)
        right = Frame(MARGIN_X + COL_W + GUTTER, MARGIN_Y, COL_W, col_h, id="right", leftPadding=5, rightPadding=0, topPadding=2, bottomPadding=0)
        diagram = Frame(MARGIN_X, MARGIN_Y, PAGE_W - 2 * MARGIN_X, col_h, id="diagram", leftPadding=0, rightPadding=0, topPadding=2, bottomPadding=0)

        fig_h = 3.55 * inch
        fig_top = Frame(MARGIN_X, PAGE_H - MARGIN_Y - fig_h, PAGE_W - 2 * MARGIN_X, fig_h, id="fig_top", leftPadding=0, rightPadding=0, topPadding=2, bottomPadding=2)
        fig_bottom_h = col_h - fig_h - 0.05 * inch
        fig_left = Frame(MARGIN_X, MARGIN_Y, COL_W, fig_bottom_h, id="fig_left", leftPadding=0, rightPadding=5, topPadding=3, bottomPadding=0)
        fig_right = Frame(MARGIN_X + COL_W + GUTTER, MARGIN_Y, COL_W, fig_bottom_h, id="fig_right", leftPadding=5, rightPadding=0, topPadding=3, bottomPadding=0)

        agent_h = 4.65 * inch
        agent_top = Frame(MARGIN_X, PAGE_H - MARGIN_Y - agent_h, PAGE_W - 2 * MARGIN_X, agent_h, id="agent_top", leftPadding=0, rightPadding=0, topPadding=2, bottomPadding=2)
        agent_bottom_h = col_h - agent_h - 0.05 * inch
        agent_left = Frame(MARGIN_X, MARGIN_Y, COL_W, agent_bottom_h, id="agent_left", leftPadding=0, rightPadding=5, topPadding=3, bottomPadding=0)
        agent_right = Frame(MARGIN_X + COL_W + GUTTER, MARGIN_Y, COL_W, agent_bottom_h, id="agent_right", leftPadding=5, rightPadding=0, topPadding=3, bottomPadding=0)

        arch_h = 4.25 * inch
        arch_top = Frame(MARGIN_X, PAGE_H - MARGIN_Y - arch_h, PAGE_W - 2 * MARGIN_X, arch_h, id="arch_top", leftPadding=0, rightPadding=0, topPadding=2, bottomPadding=2)
        arch_bottom_h = col_h - arch_h - 0.05 * inch
        arch_left = Frame(MARGIN_X, MARGIN_Y, COL_W, arch_bottom_h, id="arch_left", leftPadding=0, rightPadding=5, topPadding=3, bottomPadding=0)
        arch_right = Frame(MARGIN_X + COL_W + GUTTER, MARGIN_Y, COL_W, arch_bottom_h, id="arch_right", leftPadding=5, rightPadding=0, topPadding=3, bottomPadding=0)

        self.addPageTemplates([
            PageTemplate(id="First", frames=[first_top, first_left, first_right], onPage=on_page, autoNextPageTemplate="TwoCol"),
            PageTemplate(id="TwoCol", frames=[left, right], onPage=on_page),
            PageTemplate(id="ArchPage", frames=[arch_top, arch_left, arch_right], onPage=on_page, autoNextPageTemplate="TwoCol"),
            PageTemplate(id="AgentPage", frames=[agent_top, agent_left, agent_right], onPage=on_page, autoNextPageTemplate="TwoCol"),
            PageTemplate(id="DiagramPage", frames=[diagram], onPage=on_page, autoNextPageTemplate="TwoCol"),
            PageTemplate(id="FigurePage", frames=[fig_top, fig_left, fig_right], onPage=on_page, autoNextPageTemplate="TwoCol"),
        ])


def extract_abstract(tex: str) -> str:
    return re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).strip()


def section_text(tex: str, name: str) -> str:
    match = re.search(rf"\\section\{{{re.escape(name)}\}}(.*?)(?=\\section\{{|\\balance|\\bibliographystyle)", tex, re.S)
    if not match:
        raise ValueError(f"missing section {name}")
    return match.group(1).strip()


def remove_environments(text: str) -> str:
    for env in ("figure*", "figure", "table"):
        text = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}", "", text, flags=re.S)
    return text


def plain_latex(value: str) -> str:
    value = value.replace("\n", " ")
    # Normalize compact inline math before the generic command stripping below.
    # This keeps the generated PDF readable even when a TeX installation is not
    # available and the ReportLab fallback renderer is used.
    value = value.replace(r"^{\circ}", " deg").replace(r"{=}", "=")
    value = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(str(REF_NUM[k]) for k in m.group(1).split(",")) + "]", value)
    ref_map = {"fig:architecture": "1", "fig:agent": "2", "fig:qualitative": "3", "fig:history": "4", "fig:vlm": "5", "tab:related": "I", "tab:main": "II", "tab:history": "III", "tab:vlm": "IV"}
    value = re.sub(r"\\ref\{([^}]+)\}", lambda m: ref_map.get(m.group(1), "?"), value)
    value = re.sub(r"\\(?:textsc|emph|texttt|textbf)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\hat\s*", "", value)
    value = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathcal\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", value)
    value = value.replace("$", "").replace("~", " ")
    replacements = {
        r"\%": "%", r"\_": "_", r"\circ": " deg", r"\leq": "<=", r"\geq": ">=", r"\land": "and",
        r"\tau": "tau", r"\theta": "theta", r"\pi": "pi", r"\beta": "beta", r"\cup": "union",
        r"\in": " in ", r"\{": "{", r"\}": "}", "--": "–",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = value.replace("{", "").replace("}", "").replace("^", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_blocks(text: str):
    text = remove_environments(text)
    eq_index = 0
    def eq_repl(match):
        nonlocal eq_index
        eq_index += 1
        return f"\n\n[[EQ{eq_index}]]\n\n"
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", eq_repl, text, flags=re.S)
    parts = re.split(r"(\\subsection\{[^}]+\})", text)
    blocks = []
    for part in parts:
        if not part.strip():
            continue
        sm = re.fullmatch(r"\\subsection\{([^}]+)\}", part.strip())
        if sm:
            blocks.append(("subsection", sm.group(1)))
            continue
        for para in re.split(r"\n\s*\n", part):
            para = para.strip()
            if not para:
                continue
            em = re.fullmatch(r"\[\[EQ(\d+)\]\]", para)
            if em:
                blocks.append(("equation", int(em.group(1))))
            else:
                clean = plain_latex(para)
                if clean:
                    blocks.append(("paragraph", clean))
    return blocks


def paragraph(text: str):
    return Paragraph(html.escape(text), S["body"])


def add_blocks(story, blocks, *, skip_subsections: set[str] | None = None):
    skip = False
    skip_subsections = skip_subsections or set()
    for kind, value in blocks:
        if kind == "subsection":
            skip = value in skip_subsections
            if not skip:
                story.append(Paragraph(value, S["subsection"]))
        elif skip:
            continue
        elif kind == "paragraph":
            story.append(paragraph(value))
        elif kind == "equation":
            story.append(rendered_equation(value))


def pub_image(path: Path, width: float, max_height: float):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def vector_figure(factory, width: float, max_height: float):
    """Scale a ReportLab Drawing while retaining vector paths and live text."""
    drawing = factory()
    scale = min(width / drawing.width, max_height / drawing.height)
    drawing.scale(scale, scale)
    drawing.width *= scale
    drawing.height *= scale
    return drawing


def rendered_equation(index: int):
    """Return selectable vector text with mathematical sub/superscript layout."""
    formula_style = ParagraphStyle(
        f"VectorEquation{index}",
        fontName="DejaVuSerif",
        fontSize=8.1 if index in {2, 4, 5, 6} else 8.5,
        leading=10.4,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
    )
    number_style = ParagraphStyle(
        f"VectorEquationNumber{index}",
        fontName="DejaVuSerif",
        fontSize=7.7,
        leading=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
    )
    table = Table(
        [[Paragraph(EQUATION_MARKUP[index], formula_style), Paragraph(f"({index})", number_style)]],
        colWidths=[COL_W - 27, 27],
        hAlign="CENTER",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return KeepTogether([Spacer(1, 1), table, Spacer(1, 2)])


def styled_table(data, widths, header_bg="#E2E8F0", font_size=6.5):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.3),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#94A3B8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
    ]))
    return table


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    story = []
    story.append(Paragraph("ParkingAgent: Evidence-Gated Multimodal Reasoning<br/>for Reliable Parking-Slot Occupancy under Occlusion", S["title"]))
    story.append(Paragraph("Anonymous Author(s)<br/><font size='7'>Author information withheld for manuscript preparation</font>", S["authors"]))
    story.append(Paragraph("<b>Abstract—</b> " + html.escape(plain_latex(extract_abstract(tex))), S["abstract"]))
    story.append(FrameBreak())

    intro = parse_blocks(section_text(tex, "Introduction"))
    story.append(Paragraph("I. INTRODUCTION", S["section"]))
    add_blocks(story, intro)

    story.append(Paragraph("II. RELATED WORK", S["section"]))
    add_blocks(story, parse_blocks(section_text(tex, "Related Work")))

    # Put both related architecture figures on one dedicated scientific-figure page.
    story.append(NextPageTemplate("DiagramPage"))
    story.append(PageBreak())
    story.append(vector_figure(system_drawing, PAGE_W - 2 * MARGIN_X, 3.16 * inch))
    story.append(Paragraph("<b>Fig. 1.</b> ParkingAgent system overview. Part 1 assigns a causal three-state estimate; only forward unresolved slots enter Part 2, where visibility gates the legal evidence tools.", S["caption"]))
    story.append(Spacer(1, 5))
    story.append(vector_figure(agent_drawing, PAGE_W - 2 * MARGIN_X, 3.16 * inch))
    story.append(Paragraph("<b>Fig. 2.</b> Active-perception Agent. The center VLM selects a legal observation, receives a structured EvidenceRecord, and re-plans. Final proposals are accepted only by the deterministic validator; rejected proposals return to the loop or fail closed to Unknown.", S["caption"]))

    # Resume two-column text, retaining the related-work table before Methodology.
    story.append(NextPageTemplate("TwoCol"))
    story.append(PageBreak())
    story.append(Paragraph("TABLE I: Positioning relative to representative system families.", S["tablecap"]))
    story.append(styled_table([
        ["Family", "Map", "Temporal", "Tools", "Abstain"],
        ["Marking detector", "–", "–", "–", "–"],
        ["BEV fusion", "–", "yes", "–", "optional"],
        ["General VLM agent", "optional", "optional", "yes", "optional"],
        ["ParkingAgent", "yes", "yes", "yes", "yes"],
    ], [0.88 * inch, 0.40 * inch, 0.53 * inch, 0.42 * inch, 0.54 * inch], font_size=6.1))
    story.append(Spacer(1, 4))
    story.append(Paragraph("III. METHODOLOGY", S["section"]))
    add_blocks(story, parse_blocks(section_text(tex, "Methodology")))

    story.append(Paragraph("IV. EXPERIMENTS", S["section"]))
    exp_blocks = parse_blocks(section_text(tex, "Experiments"))
    protocol_blocks = []
    result_blocks = []
    target = protocol_blocks
    for block in exp_blocks:
        if block == ("subsection", "Main Results"):
            target = result_blocks
        target.append(block)
    add_blocks(story, protocol_blocks)

    # Multiple real cases: visible Free, visible Occupied, abstention, and failure.
    story.append(NextPageTemplate("FigurePage"))
    story.append(PageBreak())
    story.append(pub_image(PAPER / "figures/fig3_multicase_evidence.png", PAGE_W - 2 * MARGIN_X, 2.88 * inch))
    story.append(Paragraph("<b>Fig. 3.</b> Four representative locked-GT cases pair the target map and causal Camera sheet: Camera-direct Free, Camera-plus-LiDAR Occupied, correct Unknown abstention, and the sole false resolution under occlusion.", S["caption"]))
    story.append(FrameBreak())
    add_blocks(story, result_blocks)
    story.append(Paragraph("TABLE II: Locked-GT results at frame 6241 (percent).", S["tablecap"]))
    story.append(styled_table([
        ["Method", "Acc.", "Cov.", "Sel. acc.", "Agent-11", "O/F"],
        ["Part 1 W45/K3", "50.00", "7.14", "100.00", "–", "0/0"],
        ["ParkingAgent full", "92.86", "64.29", "88.89", "90.91", "1/0"],
    ], [0.93 * inch, 0.38 * inch, 0.38 * inch, 0.50 * inch, 0.49 * inch, 0.31 * inch], font_size=6.1))
    story.append(Spacer(1, 4))

    story.append(Paragraph("V. ABLATION STUDY", S["section"]))
    ablation = parse_blocks(section_text(tex, "Ablation Study"))
    # Insert the history plot after its subsection prose and before the VLM subsection.
    history_blocks = []
    vlm_blocks = []
    target = history_blocks
    for block in ablation:
        if block == ("subsection", "Center-VLM Capacity"):
            target = vlm_blocks
        target.append(block)
    add_blocks(story, history_blocks)
    story.append(KeepTogether([
        pub_image(PAPER / "figures/fig3_history_ablation.png", COL_W - 4, 1.65 * inch),
        Paragraph("<b>Fig. 4.</b> History ablation recomputed against the revised locked GT. The star marks W45/K3.", S["caption"]),
    ]))
    story.append(Paragraph("TABLE III: Representative revised-GT history settings.", S["tablecap"]))
    story.append(styled_table([
        ["W/K", "Accuracy", "Coverage", "False-U"],
        ["15/3", "42.86", "14.29", "16.67"],
        ["30/3", "42.86", "0.00", "0.00"],
        ["45/3", "50.00", "7.14", "0.00"],
        ["45/5", "42.86", "14.29", "16.67"],
        ["60/5", "50.00", "7.14", "0.00"],
        ["60/15", "28.57", "14.29", "33.33"],
    ], [0.55 * inch, 0.72 * inch, 0.72 * inch, 0.63 * inch], font_size=6.1))
    story.append(Spacer(1, 4))
    add_blocks(story, vlm_blocks)
    story.append(Paragraph("TABLE IV: Center-VLM ablation (percent).", S["tablecap"]))
    story.append(styled_table([
        ["Center model", "14-GT", "Agent-11", "Visible", "First Cam."],
        ["OpenAI reference", "92.86", "90.91", "100.0", "11/11"],
        ["Qwen3-VL-2B", "50.00", "36.36", "0.0", "11/11"],
        ["Qwen3.5-0.8B", "50.00", "36.36", "0.0", "0/11"],
    ], [0.95 * inch, 0.48 * inch, 0.55 * inch, 0.47 * inch, 0.60 * inch], font_size=6.0))
    story.append(KeepTogether([
        pub_image(PAPER / "figures/fig4_vlm_ablation.png", COL_W - 4, 1.58 * inch),
        Paragraph("<b>Fig. 5.</b> Model capacity affects protocol adherence before final accuracy. OpenAI is contextual; the strict model-only comparison is between the two Qwen rows.", S["caption"]),
    ]))

    story.append(Paragraph("VI. DISCUSSION AND CONCLUSION", S["section"]))
    add_blocks(story, parse_blocks(section_text(tex, "Discussion and Conclusion")))

    story.append(Paragraph("REFERENCES", S["section"]))
    for idx, ref in enumerate(REF_TEXT, 1):
        story.append(Paragraph(f"[{idx}] {html.escape(ref)}", S["refs"]))

    doc = PaperDoc(str(PDF))
    doc.build(story)
    reader = PdfReader(PDF)
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    (PAPER / "ParkingAgent_Evidence_Gated_Multimodal_Parking.txt").write_text(extracted, encoding="utf-8")
    print(json.dumps({"pdf": str(PDF), "pages": len(reader.pages), "bytes": PDF.stat().st_size, "extracted_characters": len(extracted)}, indent=2))


if __name__ == "__main__":
    main()
