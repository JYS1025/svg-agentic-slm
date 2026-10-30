#!/usr/bin/env python3
"""Build one cumulative execution report from all completed run6 cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEPS = Path("/tmp/svg-report-deps")
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/simple_bench_run6_distill_stage2_ckpt150"
OUTPUT = RUN_DIR / "run6_all_cases_execution_report.pdf"

NAVY = colors.HexColor("#13253D")
BLUE = colors.HexColor("#2C6BED")
CYAN = colors.HexColor("#DDEBFF")
INK = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#D8E0EA")
PALE = colors.HexColor("#F5F8FC")
GREEN_BG = colors.HexColor("#E6F6EF")
AMBER_BG = colors.HexColor("#FFF3D6")
RED_BG = colors.HexColor("#FDEBE9")


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


pdfmetrics.registerFont(
    TTFont("Nanum", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
)
pdfmetrics.registerFont(
    TTFont("Nanum-Bold", "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf")
)
pdfmetrics.registerFontFamily("Nanum", normal="Nanum", bold="Nanum-Bold")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="KTitle", fontName="Nanum-Bold", fontSize=22, leading=29, textColor=NAVY, spaceAfter=6 * mm))
styles.add(ParagraphStyle(name="KSub", fontName="Nanum", fontSize=9.5, leading=14, textColor=MUTED, spaceAfter=3 * mm))
styles.add(ParagraphStyle(name="H1K", fontName="Nanum-Bold", fontSize=16, leading=21, textColor=NAVY, spaceAfter=4 * mm))
styles.add(ParagraphStyle(name="H2K", fontName="Nanum-Bold", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="BodyK", fontName="Nanum", fontSize=8.2, leading=12, textColor=INK))
styles.add(ParagraphStyle(name="SmallK", fontName="Nanum", fontSize=7.0, leading=9.5, textColor=INK))
styles.add(ParagraphStyle(name="TinyK", fontName="Nanum", fontSize=6.1, leading=8.0, textColor=INK))
styles.add(ParagraphStyle(name="WhiteK", fontName="Nanum-Bold", fontSize=8.0, leading=10, textColor=colors.white))
styles.add(ParagraphStyle(name="Metric", fontName="Nanum-Bold", fontSize=15, leading=18, textColor=NAVY, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", fontName="Nanum", fontSize=6.5, leading=8, textColor=MUTED, alignment=TA_CENTER))


def P(text: object, style: str = "BodyK") -> Paragraph:
    return Paragraph(esc(text), styles[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Nanum", 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.7 * mm, "SVG Agentic SLM · run6 checkpoint-150 execution report")
    canvas.drawRightString(192 * mm, 8.7 * mm, str(doc.page))
    canvas.restoreState()


def section_header(text: str) -> Table:
    table = Table([[P(text, "WhiteK")]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    return table


def metric(value: object, label: str, bg=PALE) -> Table:
    table = Table([[P(value, "Metric")], [P(label, "MetricLabel")]], colWidths=[41 * mm], rowHeights=[10 * mm, 8 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(RUN_DIR.glob("text_to_svg_*.json")):
        run = json.loads(path.read_text())
        artifacts = sorted(RUN_DIR.glob(f"{path.stem}.artifacts/*/attempts"))
        if not artifacts:
            continue
        attempts_dir = artifacts[-1]
        feedback = {item["target_attempt_id"]: item for item in run.get("critic_feedback", [])}
        records = []
        for index, meta in enumerate(run["metadata"]["generator"]["attempts"]):
            fb = feedback.get(meta["attempt_id"], {})
            validation_path = attempts_dir / f"attempt-{index:03d}.critic-call-000.validation.json"
            validation = json.loads(validation_path.read_text()) if validation_path.exists() else {"success": False, "status": "missing", "warnings": []}
            image = attempts_dir / f"attempt-{index:03d}.critic.png"
            similarity = meta.get("critic_evidence", {}).get("similarity_evidence", {})
            records.append({"index": index, "meta": meta, "feedback": fb, "validation": validation, "image": image, "similarity": similarity})
        cases.append({"name": path.stem, "run": run, "records": records})
    return cases


def score_bg(score):
    if score is None:
        return PALE
    if score <= 1:
        return RED_BG
    if score == 2:
        return AMBER_BG
    return GREEN_BG


cases = load_cases()
if not cases:
    raise SystemExit(f"No completed cases found in {RUN_DIR}")

attempt_count = sum(len(case["records"]) for case in cases)
warning_count = sum(len(rec["validation"].get("warnings", [])) for case in cases for rec in case["records"])
accepted_count = sum(case["run"].get("outcome") == "accepted" for case in cases)

doc = BaseDocTemplate(
    str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
    topMargin=17 * mm, bottomMargin=18 * mm,
    title="SVG Agentic SLM run6 전체 실행 보고서", author="OpenAI Codex",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=footer)])
story = [
    Spacer(1, 7 * mm),
    P("SVG Agentic SLM · Run 6 실행 보고서", "KTitle"),
    P("Gemma-4 generator → Render → checkpoint-150 critic → Revision", "H1K"),
    P(f"완료된 케이스를 누적 반영한 보고서 · 현재 {len(cases)}/12 케이스", "KSub"),
    Spacer(1, 4 * mm),
]
cards = Table([[
    metric(f"{len(cases)}/12", "완료 케이스", CYAN),
    metric(attempt_count, "generator attempts"),
    metric(accepted_count, "threshold accepted", GREEN_BG),
    metric(warning_count, "contract warnings", AMBER_BG if warning_count else GREEN_BG),
]], colWidths=[43.5 * mm] * 4)
cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
story += [cards, Spacer(1, 7 * mm), section_header("케이스 요약"), Spacer(1, 3 * mm)]

overview = [[P("케이스", "SmallK"), P("요청", "SmallK"), P("시도", "SmallK"), P("결과", "SmallK"), P("최종/최고 critic", "SmallK"), P("선택 SigLIP2", "SmallK")]]
for case in cases:
    run = case["run"]
    selection = run["metadata"].get("selection", {})
    selected_id = selection.get("selected_attempt_id")
    selected = next((r for r in case["records"] if r["meta"]["attempt_id"] == selected_id), case["records"][-1])
    overview.append([
        P(case["name"].replace("text_to_svg_", ""), "TinyK"),
        P(run["instruction"], "TinyK"),
        P(len(case["records"]), "TinyK"),
        P(run.get("outcome"), "TinyK"),
        P(selection.get("best_critic_score", selected["feedback"].get("score", "-")), "TinyK"),
        P(f'{selected["similarity"].get("score", 0):.6f}', "TinyK"),
    ])
ot = Table(overview, colWidths=[22 * mm, 59 * mm, 12 * mm, 27 * mm, 27 * mm, 27 * mm], repeatRows=1)
ot.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), CYAN), ("GRID", (0, 0), (-1, -1), 0.35, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
]))
story += [ot, Spacer(1, 5 * mm), P("각 attempt의 render, 구조화된 critic 평가·issue·수정 지시, critic 점수, SigLIP2 점수 및 contract validation 결과를 아래에 수록했다.", "SmallK")]

for case_no, case in enumerate(cases, 1):
    run = case["run"]
    records = case["records"]
    selection = run["metadata"].get("selection", {})
    story += [PageBreak(), P(f"{case_no}. {case['name'].replace('text_to_svg_', '')} · {run['instruction']}", "H1K")]
    story += [Table([[
        metric(len(records), "attempts"),
        metric(selection.get("best_critic_score", "-"), "best critic", score_bg(selection.get("best_critic_score"))),
        metric(run.get("revision_count", 0), "revisions", CYAN),
        metric("YES" if selection.get("rolled_back") else "NO", "rolled back", AMBER_BG if selection.get("rolled_back") else GREEN_BG),
    ]], colWidths=[43.5 * mm] * 4), Spacer(1, 5 * mm)]
    story += [P(f"Outcome: {run.get('outcome')} · Stop reason: {run.get('stop_reason')}", "BodyK"), Spacer(1, 3 * mm), section_header("Attempt 흐름"), Spacer(1, 2 * mm)]
    flow = [[P("Attempt", "SmallK"), P("mode", "SmallK"), P("critic", "SmallK"), P("match", "SmallK"), P("SigLIP2", "SmallK"), P("contract", "SmallK")]]
    for rec in records:
        fb = rec["feedback"]
        flow.append([
            P(f'{rec["index"]:03d}', "SmallK"), P(rec["meta"].get("mode"), "SmallK"),
            P(fb.get("score", "-"), "SmallK"), P("YES" if fb.get("matches_instruction") else "NO", "SmallK"),
            P(f'{rec["similarity"].get("score", 0):.6f}', "SmallK"), P(rec["validation"].get("status"), "SmallK"),
        ])
    ft = Table(flow, colWidths=[20 * mm, 31 * mm, 24 * mm, 22 * mm, 36 * mm, 41 * mm], repeatRows=1)
    ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), CYAN), ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm)]))
    story += [ft, Spacer(1, 4 * mm), P(f"Selected attempt: {selection.get('selected_attempt_id')} · Last attempt: {selection.get('last_attempt_id')}", "SmallK")]

    for rec in records:
        fb = rec["feedback"]
        validation = rec["validation"]
        story += [PageBreak(), P(f"{case['name'].replace('text_to_svg_', '')} · Attempt {rec['index']:03d}", "H1K")]
        image_cell = Image(str(rec["image"]), width=72 * mm, height=72 * mm) if rec["image"].exists() else P("render missing", "BodyK")
        warning_text = ", ".join(w.get("code", "warning") for w in validation.get("warnings", [])) or "none"
        facts = Table([
            [P("Critic score", "MetricLabel"), P(fb.get("score", "-"), "Metric")],
            [P("SigLIP2", "MetricLabel"), P(f'{rec["similarity"].get("score", 0):.6f}', "Metric")],
            [P("Instruction match", "MetricLabel"), P("YES" if fb.get("matches_instruction") else "NO", "H2K")],
            [P("Contract", "MetricLabel"), P(f'{validation.get("status")} · {warning_text}', "SmallK")],
        ], colWidths=[34 * mm, 60 * mm], rowHeights=[14 * mm, 14 * mm, 12 * mm, 23 * mm])
        head = Table([[image_cell, facts]], colWidths=[78 * mm, 96 * mm])
        head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (1, 0), (1, 0), 0.5, LINE), ("BACKGROUND", (1, 0), (1, 0), PALE)]))
        story += [head, Spacer(1, 4 * mm), section_header("Critic scorecard · 1–4점"), Spacer(1, 2 * mm)]
        eval_rows = [[P("범주", "SmallK"), P("항목", "SmallK"), P("적용", "SmallK"), P("점수", "SmallK"), P("판단 근거", "SmallK")]]
        for ev in fb.get("evaluations", []):
            eval_rows.append([P(ev.get("category"), "TinyK"), P(ev.get("type"), "TinyK"), P("Y" if ev.get("applicable") else "N", "TinyK"), P(ev.get("score", "N/A"), "TinyK"), P(ev.get("reason", ""), "TinyK")])
        et = Table(eval_rows, colWidths=[19 * mm, 22 * mm, 12 * mm, 13 * mm, 108 * mm], repeatRows=1)
        et_style = [("BACKGROUND", (0, 0), (-1, 0), CYAN), ("GRID", (0, 0), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2 * mm)]
        for row_no, ev in enumerate(fb.get("evaluations", []), 1):
            et_style.append(("BACKGROUND", (3, row_no), (3, row_no), score_bg(ev.get("score") if ev.get("applicable") else None)))
        et.setStyle(TableStyle(et_style))
        story += [et, Spacer(1, 4 * mm), section_header("구조화된 issue 및 다음 generator 지시"), Spacer(1, 2 * mm)]
        issues = fb.get("issues", [])
        if not issues:
            story += [P("통과 기준 미만으로 연결된 issue 없음.", "BodyK")]
        for number, issue in enumerate(issues, 1):
            targets = ", ".join(issue.get("target_ids", [])) or "-"
            rows = [
                [P(f"{number}. {issue.get('category')}/{issue.get('type')} · scope={issue.get('scope')} · targets={targets}", "WhiteK")],
                [P(f"Observed: {issue.get('observed', '')}", "SmallK")],
                [P(f"Expected: {issue.get('expected', '')}", "SmallK")],
                [P(f"Fix → next generator: {issue.get('fix', '')}", "SmallK")],
            ]
            it = Table(rows, colWidths=[174 * mm])
            it.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), NAVY), ("BOX", (0, 0), (-1, -1), 0.45, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm), ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm), ("TOPPADDING", (0, 0), (-1, -1), 1.6 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * mm)]))
            story += [it, Spacer(1, 2 * mm)]
        suggestions = fb.get("suggestions", [])
        if suggestions:
            story += [P("Critic suggestions", "H2K"), P(" · ".join(map(str, suggestions)), "SmallK")]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
print(f"{OUTPUT} ({len(cases)}/12 cases, {attempt_count} attempts)")
