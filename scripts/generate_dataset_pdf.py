#!/usr/bin/env python3
"""Render DATASET_CARD.md as a publication-ready PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)


INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#586277")
TEAL = colors.HexColor("#087E8B")
TEAL_DARK = colors.HexColor("#075D66")
TEAL_PALE = colors.HexColor("#E9F6F7")
BLUE_PALE = colors.HexColor("#EDF3FA")
AMBER_PALE = colors.HexColor("#FFF4DA")
RED_PALE = colors.HexColor("#FCEBEC")
LINE = colors.HexColor("#CFD7E3")
PAPER = colors.HexColor("#FFFFFF")
CODE_BG = colors.HexColor("#F4F6F8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "DATASET_CARD.md"
DEFAULT_OUTPUT = ROOT / "output/pdf/deadlock-rag-dataset-v1-documentation.pdf"


def register_fonts() -> None:
    fonts = {
        "DatasetSans": "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "DatasetSans-Bold": "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        "DatasetMono": "/System/Library/Fonts/SFNSMono.ttf",
    }
    for name, path in fonts.items():
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))


def normalize_ascii(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u00a0": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def inline_markup(text: str) -> str:
    """Convert the small inline Markdown subset used by the dataset card."""
    text = normalize_ascii(text)
    tokens: list[str] = []

    def stash(value: str) -> str:
        tokens.append(value)
        return f"@@TOKEN{len(tokens) - 1}@@"

    text = re.sub(
        r"\[(`[^`]+`|[^\]]+)\]\(([^)]+)\)",
        lambda match: stash(
            '<link href="{}" color="#075D66"><u>{}</u></link>'.format(
                html.escape(match.group(2), quote=True),
                inline_markup(match.group(1)),
            )
        ),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda match: stash(
            f'<font name="DatasetMono" color="#7A284B">{html.escape(match.group(1))}</font>'
        ),
        text,
    )
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
    for index, token in enumerate(tokens):
        escaped = escaped.replace(f"@@TOKEN{index}@@", token)
    return escaped


def build_styles():
    styles = getSampleStyleSheet()
    common = dict(fontName="DatasetSans", textColor=INK, splitLongWords=True)
    return {
        "body": ParagraphStyle(
            "Body", parent=styles["BodyText"], fontSize=8.7, leading=13.1,
            spaceAfter=5.5, **common
        ),
        "small": ParagraphStyle(
            "Small", parent=styles["BodyText"], fontSize=7.2, leading=10.2,
            textColor=MUTED, fontName="DatasetSans"
        ),
        "h1": ParagraphStyle(
            "H1", parent=styles["Heading1"], fontName="DatasetSans-Bold",
            fontSize=19, leading=23, textColor=INK, spaceBefore=4, spaceAfter=10,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=styles["Heading2"], fontName="DatasetSans-Bold",
            fontSize=13, leading=16, textColor=TEAL_DARK, spaceBefore=12,
            spaceAfter=6, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=styles["Heading3"], fontName="DatasetSans-Bold",
            fontSize=10, leading=13, textColor=INK, spaceBefore=9,
            spaceAfter=4, keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=styles["BodyText"], fontName="DatasetSans",
            fontSize=8.6, leading=12.5, leftIndent=12, firstLineIndent=-7,
            bulletIndent=2, spaceAfter=3, textColor=INK,
        ),
        "table": ParagraphStyle(
            "TableCell", parent=styles["BodyText"], fontName="DatasetSans",
            fontSize=6.9, leading=9.3, textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=styles["BodyText"], fontName="DatasetSans-Bold",
            fontSize=6.8, leading=9, textColor=PAPER,
        ),
        "code": ParagraphStyle(
            "Code", parent=styles["Code"], fontName="DatasetMono", fontSize=6.15,
            leading=8.35, textColor=colors.HexColor("#253044"), backColor=CODE_BG,
            borderColor=LINE, borderWidth=0.5, borderPadding=7, spaceBefore=3,
            spaceAfter=8, leftIndent=0, rightIndent=0,
        ),
    }


def page_chrome(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 5 * mm, width, 5 * mm, stroke=0, fill=1)
    if doc.page > 1:
        canvas.setFont("DatasetSans", 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, height - 11 * mm, "DEADLOCK RAG DATASET  |  VERSION 1")
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
    canvas.setFont("DatasetSans", 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Generated from DATASET_CARD.md")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"PAGE {doc.page}")
    canvas.restoreState()


def cover_story(styles):
    metric_style = ParagraphStyle(
        "Metric", parent=styles["body"], alignment=TA_CENTER, fontSize=18,
        leading=21, fontName="DatasetSans-Bold", textColor=TEAL_DARK,
    )
    metric_label = ParagraphStyle(
        "MetricLabel", parent=styles["small"], alignment=TA_CENTER,
        fontSize=6.8, leading=9, textColor=MUTED,
    )
    title = ParagraphStyle(
        "CoverTitle", parent=styles["h1"], fontSize=30, leading=35,
        textColor=INK, alignment=TA_LEFT, spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "CoverSubtitle", parent=styles["body"], fontSize=12, leading=18,
        textColor=MUTED, spaceAfter=18,
    )
    metrics = Table(
        [[
            [Paragraph("6,418", metric_style), Paragraph("GRAPH SNAPSHOTS", metric_label)],
            [Paragraph("360", metric_style), Paragraph("CURATED RUNS", metric_label)],
            [Paragraph("3", metric_style), Paragraph("STATE LABELS", metric_label)],
        ]],
        colWidths=[55 * mm] * 3,
        rowHeights=[31 * mm],
    )
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL_PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    facts = Table(
        [
            [Paragraph("Platform", styles["table_header"]), Paragraph("ARM64 Linux in QEMU", styles["table"])],
            [Paragraph("Sensors", styles["table_header"]), Paragraph("pthread uprobes, futex and scheduler tracepoints, perf", styles["table"])],
            [Paragraph("Task", styles["table_header"]), Paragraph("Classify active resource-allocation graphs as safe, unsafe, or deadlocked", styles["table"])],
            [Paragraph("Release", styles["table_header"]), Paragraph("v1 - collected 25 July 2026", styles["table"])],
        ],
        colWidths=[35 * mm, 130 * mm],
    )
    facts.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), TEAL_DARK),
        ("BACKGROUND", (1, 0), (1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [
        Spacer(1, 24 * mm),
        Paragraph("DATASET DOCUMENTATION", ParagraphStyle(
            "Kicker", parent=styles["small"], fontName="DatasetSans-Bold",
            fontSize=8, leading=10, textColor=TEAL, spaceAfter=8,
        )),
        Paragraph("Deadlock Resource-Allocation Graph Dataset", title),
        Paragraph(
            "A labeled temporal graph corpus built from controlled pthread workloads, "
            "eBPF observations, and run-level performance counters.",
            subtitle,
        ),
        HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=15),
        metrics,
        Spacer(1, 15 * mm),
        facts,
        Spacer(1, 18 * mm),
        Paragraph(
            "This document defines collection, graph semantics, labels, splits, "
            "quality checks, limitations, loading, and reproduction for the curated release.",
            styles["body"],
        ),
        PageBreak(),
    ]


def pipeline_table(styles):
    node_style = ParagraphStyle(
        "PipelineNode", parent=styles["table"], alignment=TA_CENTER,
        fontName="DatasetSans-Bold", fontSize=7, leading=9,
    )
    arrow = Paragraph("<font color='#087E8B'><b>-&gt;</b></font>", node_style)
    cells = [
        ("pthread<br/>workload", BLUE_PALE),
        ("uPROBES<br/>FUTEX<br/>SCHEDULER<br/>PERF", TEAL_PALE),
        ("timestamped<br/>events", BLUE_PALE),
        ("RAG<br/>builder", AMBER_PALE),
        ("Tarjan cycle<br/>detection", RED_PALE),
        ("labeled<br/>JSONL", TEAL_PALE),
    ]
    row = []
    widths = []
    for index, (label, _) in enumerate(cells):
        row.append(Paragraph(label, node_style))
        widths.append(24 * mm if index != 1 else 29 * mm)
        if index != len(cells) - 1:
            row.append(arrow)
            widths.append(5 * mm)
    table = Table([row], colWidths=widths, rowHeights=[26 * mm])
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for index, (_, background) in enumerate(cells):
        column = index * 2
        commands.extend([
            ("BACKGROUND", (column, 0), (column, 0), background),
            ("BOX", (column, 0), (column, 0), 0.7, TEAL),
        ])
    table.setStyle(TableStyle(commands))
    return table


def table_from_markdown(rows, styles, available_width):
    columns = len(rows[0])
    numeric_columns = set()
    for index, cell in enumerate(rows[1]):
        if re.fullmatch(r":?-+:?", cell.strip()) and cell.strip().endswith(":"):
            numeric_columns.add(index)
    content = rows[:1] + rows[2:]
    formatted = []
    for row_index, row in enumerate(content):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        formatted.append([Paragraph(inline_markup(cell.strip()), style) for cell in row])

    if columns == 2:
        col_widths = [available_width * 0.31, available_width * 0.69]
    elif columns == 3:
        col_widths = [available_width * 0.27, available_width * 0.53, available_width * 0.20]
    elif columns == 4:
        col_widths = [available_width * 0.19, available_width * 0.24,
                      available_width * 0.16, available_width * 0.41]
    else:
        col_widths = [available_width / columns] * columns

    table = Table(formatted, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL_DARK),
        ("BOX", (0, 0), (-1, -1), 0.55, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, colors.HexColor("#F7F9FB")]),
    ]
    for column in numeric_columns:
        commands.append(("ALIGN", (column, 1), (column, -1), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return [table, Spacer(1, 7)]


def markdown_story(source: Path, styles, available_width):
    lines = normalize_ascii(source.read_text(encoding="utf-8")).splitlines()
    story = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    code_language = ""

    def flush_paragraph():
        if paragraph_lines:
            story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"]))
            paragraph_lines.clear()

    def flush_table():
        if table_lines:
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                    for line in table_lines]
            story.extend(table_from_markdown(rows, styles, available_width))
            table_lines.clear()

    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            if not in_code:
                in_code = True
                code_language = stripped[3:].strip()
                code_lines = []
            else:
                in_code = False
                if code_language == "mermaid":
                    story.extend([pipeline_table(styles), Spacer(1, 8)])
                else:
                    story.append(XPreformatted(
                        html.escape("\n".join(code_lines)), styles["code"]
                    ))
                code_lines = []
                code_language = ""
            continue
        if in_code:
            code_lines.append(line)
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines.append(stripped)
            continue
        flush_table()
        heading = re.match(r"^(#{2,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            story.append(Paragraph(inline_markup(heading.group(2)), styles[f"h{level}"]))
            continue
        if re.match(r"^-\s+", stripped):
            flush_paragraph()
            story.append(Paragraph(
                inline_markup(stripped[2:]), styles["bullet"], bulletText="-"
            ))
            continue
        if not stripped:
            flush_paragraph()
        else:
            paragraph_lines.append(stripped)
    flush_paragraph()
    flush_table()
    return story


def build_pdf(source: Path, output: Path) -> None:
    register_fonts()
    styles = build_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    frame = Frame(
        18 * mm, 16 * mm, width - 36 * mm, height - 32 * mm,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    template = PageTemplate(id="DatasetCard", frames=[frame], onPage=page_chrome)
    document = BaseDocTemplate(
        str(output), pagesize=A4, title="Deadlock Resource-Allocation Graph Dataset",
        author="Dataset project", subject="Dataset documentation, version 1",
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    document.addPageTemplates([template])
    story = cover_story(styles)
    story.extend(markdown_story(source, styles, width - 36 * mm))
    document.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_pdf(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
