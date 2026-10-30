#!/usr/bin/env python3
"""Build a compact PDF walkthrough for the icon_1 critic/revision run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEPS = Path("/tmp/svg-report-deps")
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

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
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_JSON = ROOT / "outputs/simple_bench_run5_distill_stage2_ckpt100/text_to_svg_icon_1.json"
ATTEMPTS = ROOT / (
    "outputs/simple_bench_run5_distill_stage2_ckpt100/"
    "text_to_svg_icon_1.artifacts/"
    "run_d488a7734a934e228a1f0b8f178c7539.a8cf9b8a618d/attempts"
)
OUTPUT = ROOT / "outputs/simple_bench_run5_distill_stage2_ckpt100/icon_1_execution_report.pdf"

NAVY = colors.HexColor("#13253D")
BLUE = colors.HexColor("#2C6BED")
CYAN = colors.HexColor("#DDEBFF")
INK = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#D8E0EA")
PALE = colors.HexColor("#F5F8FC")
GREEN = colors.HexColor("#138A5B")
GREEN_BG = colors.HexColor("#E6F6EF")
AMBER = colors.HexColor("#B36B00")
AMBER_BG = colors.HexColor("#FFF3D6")
RED = colors.HexColor("#B42318")
RED_BG = colors.HexColor("#FDEBE9")


def esc(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


pdfmetrics.registerFont(TTFont("Nanum", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"))
pdfmetrics.registerFont(TTFont("Nanum-Bold", "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"))
pdfmetrics.registerFontFamily("Nanum", normal="Nanum", bold="Nanum-Bold")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="KTitle", fontName="Nanum-Bold", fontSize=24, leading=31, textColor=NAVY, spaceAfter=7 * mm))
styles.add(ParagraphStyle(name="KSub", fontName="Nanum", fontSize=10.3, leading=16, textColor=MUTED, spaceAfter=3 * mm))
styles.add(ParagraphStyle(name="H1K", fontName="Nanum-Bold", fontSize=17, leading=22, textColor=NAVY, spaceAfter=4 * mm))
styles.add(ParagraphStyle(name="H2K", fontName="Nanum-Bold", fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="BodyK", fontName="Nanum", fontSize=8.7, leading=13, textColor=INK))
styles.add(ParagraphStyle(name="SmallK", fontName="Nanum", fontSize=7.2, leading=10.3, textColor=INK))
styles.add(ParagraphStyle(name="TinyK", fontName="Nanum", fontSize=6.4, leading=8.5, textColor=INK))
styles.add(ParagraphStyle(name="WhiteK", fontName="Nanum-Bold", fontSize=8.2, leading=11, textColor=colors.white))
styles.add(ParagraphStyle(name="Metric", fontName="Nanum-Bold", fontSize=16, leading=19, textColor=NAVY, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", fontName="Nanum", fontSize=6.8, leading=9, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Footer", fontName="Nanum", fontSize=6.5, leading=8, textColor=MUTED, alignment=TA_CENTER))


def P(text: object, style: str = "BodyK") -> Paragraph:
    return Paragraph(esc(text), styles[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Nanum", 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.7 * mm, "SVG Agentic SLM · icon_1 execution trace")
    canvas.drawRightString(192 * mm, 8.7 * mm, f"{doc.page}")
    canvas.restoreState()


def metric_card(value: str, label: str, bg=PALE):
    table = Table([[P(value, "Metric")], [P(label, "MetricLabel")]], colWidths=[39 * mm], rowHeights=[10 * mm, 9 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    return table


def section_header(text: str):
    t = Table([[P(text, "WhiteK")]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
    ]))
    return t


def score_color(score):
    if score is None:
        return colors.HexColor("#EEF2F6")
    if score <= 1:
        return RED_BG
    if score == 2:
        return AMBER_BG
    return GREEN_BG


run = json.loads(RUN_JSON.read_text())
attempt_meta = run["metadata"]["generator"]["attempts"]
feedback_by_attempt = {f["target_attempt_id"]: f for f in run["critic_feedback"]}
selection = run["metadata"]["selection"]

records = []
for idx, meta in enumerate(attempt_meta):
    raw = json.loads((ATTEMPTS / f"attempt-{idx:03d}.critic-call-000.raw.txt").read_text())
    valid = json.loads((ATTEMPTS / f"attempt-{idx:03d}.critic-call-000.validation.json").read_text())
    feedback = feedback_by_attempt[meta["attempt_id"]]
    similarity = meta["critic_evidence"]["similarity_evidence"]
    records.append({
        "idx": idx,
        "meta": meta,
        "raw": raw,
        "validation": valid,
        "feedback": feedback,
        "similarity": similarity,
        "image": ATTEMPTS / f"attempt-{idx:03d}.critic.png",
    })


doc = BaseDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=17 * mm,
    bottomMargin=18 * mm,
    title="SVG Agentic SLM 실행 예시 — Two hands hold together",
    author="OpenAI Codex",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=footer)])
story = []

# Cover / executive summary
story += [
    Spacer(1, 8 * mm),
    P("SVG Agentic SLM 실행 예시", "KTitle"),
    P("Generator → Render → Critic → Revision 흐름", "H1K"),
    P("요청: “Two hands hold together” · Distill stage2 checkpoint-100", "KSub"),
    Spacer(1, 5 * mm),
]

metrics = Table([[
    metric_card("3", "generator attempts"),
    metric_card("3/3", "critic contract valid", GREEN_BG),
    metric_card("1.0", "모든 critic 점수", RED_BG),
    metric_card("2", "revision 횟수", CYAN),
]], colWidths=[43.5 * mm] * 4)
metrics.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
story += [metrics, Spacer(1, 8 * mm), section_header("실행 흐름"), Spacer(1, 4 * mm)]

flow_data = [[P("단계", "SmallK"), P("결과", "SmallK"), P("Critic", "SmallK"), P("SigLIP2", "SmallK"), P("오케스트레이터 처리", "SmallK")]]
labels = ["초기 생성", "1차 수정", "2차 수정"]
actions = ["Rejected → revision", "Selected best", "Rolled back"]
for label, action, rec in zip(labels, actions, records):
    flow_data.append([
        P(label, "SmallK"),
        P(rec["meta"]["mode"], "SmallK"),
        P(f'{rec["feedback"]["score"]:.1f} / revise', "SmallK"),
        P(f'{rec["similarity"]["score"]:.6f}', "SmallK"),
        P(action, "SmallK"),
    ])
flow = Table(flow_data, colWidths=[26 * mm, 27 * mm, 35 * mm, 32 * mm, 54 * mm], repeatRows=1)
flow.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), CYAN),
    ("GRID", (0, 0), (-1, -1), 0.45, LINE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
    ("BACKGROUND", (0, 2), (-1, 2), GREEN_BG),
    ("BACKGROUND", (0, 3), (-1, 3), RED_BG),
]))
story += [flow, Spacer(1, 7 * mm)]

story += [section_header("한눈에 보는 결론"), Spacer(1, 4 * mm)]
summary = [
    [P("✓", "H2K"), P("세 critic 응답 모두 구조화 contract validation에 성공했다.", "BodyK")],
    [P("→", "H2K"), P("SigLIP2는 0.000710 → 0.001927 → 0.002051로 소폭 증가했지만 절대값은 매우 낮다.", "BodyK")],
    [P("!", "H2K"), P("Critic 점수는 세 시도 모두 1.0이며, 핵심 실패 원인은 holding/grasping 동작과 손가락 구조의 부재다.", "BodyK")],
    [P("↩", "H2K"), P("2차 수정은 critic 개선이 없어 rollback되었고, 1차 수정 결과가 최종 선택되었다.", "BodyK")],
]
sumtab = Table(summary, colWidths=[12 * mm, 162 * mm])
sumtab.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LINEBELOW", (0, 0), (-1, -2), 0.35, LINE),
    ("TOPPADDING", (0, 0), (-1, -1), 2.3 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3 * mm),
]))
story += [sumtab, Spacer(1, 7 * mm), P("본 보고서는 저장된 SVG render, raw critic JSON, validation JSON 및 run metadata를 직접 구조화한 것이다.", "SmallK")]


def attempt_page(rec):
    idx = rec["idx"]
    fb = rec["feedback"]
    raw = rec["raw"]
    meta = rec["meta"]
    sim = rec["similarity"]
    if idx == 0:
        stage = "초기 Generator 출력"
        outcome = "Rejected → revision 시작"
    elif idx == 1:
        stage = "Critic 피드백 반영: 1차 수정"
        outcome = "Selected best · 최종 SVG"
    else:
        stage = "Critic 피드백 반영: 2차 수정"
        outcome = "Rolled back · 개선 없음"

    out = [PageBreak(), P(f"Attempt {idx:03d} · {stage}", "H1K")]
    header = Table([
        [Image(str(rec["image"]), width=74 * mm, height=74 * mm),
         Table([
             [P("Critic 점수", "MetricLabel"), P(f'{fb["score"]:.1f}', "Metric")],
             [P("SigLIP2", "MetricLabel"), P(f'{sim["score"]:.6f}', "Metric")],
             [P("Contract", "MetricLabel"), P("VALID" if rec["validation"]["success"] else "FAILED", "H2K")],
             [P("Instruction match", "MetricLabel"), P("NO" if not fb["matches_instruction"] else "YES", "H2K")],
             [P("처리", "MetricLabel"), P(outcome, "SmallK")],
         ], colWidths=[31 * mm, 57 * mm], rowHeights=[12 * mm, 12 * mm, 10 * mm, 10 * mm, 16 * mm])]
    ], colWidths=[80 * mm, 94 * mm])
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (1, 0), (1, 0), 0.6, LINE),
        ("BACKGROUND", (1, 0), (1, 0), PALE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    out += [header, Spacer(1, 4 * mm), section_header("Critic scorecard · 1–4점"), Spacer(1, 2.5 * mm)]

    score_rows = [[P("범주", "SmallK"), P("항목", "SmallK"), P("점수", "SmallK"), P("판단 근거", "SmallK")]]
    for ev in raw["evaluations"]:
        score = "N/A" if not ev["applicable"] else str(ev["score"])
        score_rows.append([P(ev["category"], "TinyK"), P(ev["type"], "TinyK"), P(score, "TinyK"), P(ev["reason"], "TinyK")])
    score_table = Table(score_rows, colWidths=[20 * mm, 25 * mm, 13 * mm, 116 * mm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), CYAN),
        ("GRID", (0, 0), (-1, -1), 0.3, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.25 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.25 * mm),
    ]
    for r, ev in enumerate(raw["evaluations"], 1):
        style.append(("BACKGROUND", (2, r), (2, r), score_color(ev["score"] if ev["applicable"] else None)))
    score_table.setStyle(TableStyle(style))
    out += [score_table, PageBreak(), P(f"Attempt {idx:03d} · 구조화된 개선 지시", "H1K")]

    for no, issue in enumerate(raw["issues"], 1):
        label = f'{no}. {issue["category"]}/{issue["type"]} · scope={issue["scope"]} · targets={", ".join(issue["target_ids"])}'
        issue_table = Table([
            [P(label, "WhiteK")],
            [Table([
                [P("Observed", "SmallK"), P(issue["observed"], "SmallK")],
                [P("Expected", "SmallK"), P(issue["expected"], "SmallK")],
                [P("Fix → 다음 generator", "SmallK"), P(issue["fix"], "SmallK")],
            ], colWidths=[34 * mm, 132 * mm])],
        ], colWidths=[174 * mm])
        issue_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), NAVY),
            ("BOX", (0, 0), (-1, -1), 0.55, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
            ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ]))
        out += [KeepTogether(issue_table), Spacer(1, 3 * mm)]

    out += [Spacer(1, 2 * mm), section_header("이 단계의 해석"), Spacer(1, 3 * mm)]
    interpretations = [
        "중앙의 삼각형 형태가 손가락이나 맞잡은 손으로 읽히지 않는다. Critic은 state=1을 병목 점수로 판단했다.",
        "중앙 형태를 크게 만들었지만 두 손이 접촉한다기보다 하나의 추상 도형처럼 보인다. scale과 occlusion은 개선됐으나 state=1은 유지됐다.",
        "양쪽 형태가 더 멀어지고 손가락 구조가 사라져 contour·proportion·topology·spacing이 1점으로 악화됐다. 따라서 이전 best로 rollback됐다.",
    ]
    out += [P(interpretations[idx], "BodyK")]
    return out


for rec in records:
    story += attempt_page(rec)

# Final comparison page
story += [PageBreak(), P("최종 비교 및 선택", "H1K")]
comparison = [[P("지표", "SmallK")], [P("Attempt 000", "SmallK")], [P("Attempt 001", "SmallK")], [P("Attempt 002", "SmallK")]]
comparison = [[P("지표", "SmallK"), P("Attempt 000", "SmallK"), P("Attempt 001", "SmallK"), P("Attempt 002", "SmallK")]]
rows = [
    ("Critic score", *[f'{r["feedback"]["score"]:.1f}' for r in records]),
    ("SigLIP2", *[f'{r["similarity"]["score"]:.6f}' for r in records]),
    ("semantic/state", *[str(next(e["score"] for e in r["raw"]["evaluations"] if e["category"] == "semantic" and e["type"] == "state")) for r in records]),
    ("geometry/contour", *[str(next(e["score"] for e in r["raw"]["evaluations"] if e["category"] == "geometry" and e["type"] == "contour")) for r in records]),
    ("layout/spacing", *[str(next(e["score"] for e in r["raw"]["evaluations"] if e["category"] == "layout" and e["type"] == "spacing")) for r in records]),
    ("결과", "rejected", "selected best", "rolled back"),
]
for row in rows:
    comparison.append([P(x, "SmallK") for x in row])
ct = Table(comparison, colWidths=[48 * mm, 42 * mm, 42 * mm, 42 * mm])
ct.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 1), (0, -1), CYAN),
    ("BACKGROUND", (2, 1), (2, -1), GREEN_BG),
    ("BACKGROUND", (3, 1), (3, -1), RED_BG),
    ("GRID", (0, 0), (-1, -1), 0.45, LINE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
]))
story += [ct, Spacer(1, 6 * mm)]

imgs = Table([[Image(str(r["image"]), width=54 * mm, height=54 * mm) for r in records]], colWidths=[58 * mm] * 3)
imgs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, LINE)]))
story += [imgs, Spacer(1, 5 * mm), section_header("선택 로직"), Spacer(1, 3 * mm)]
story += [
    P(f'최종 선택 attempt ID: {selection["selected_attempt_id"]}', "SmallK"),
    P(f'마지막 attempt ID: {selection["last_attempt_id"]}', "SmallK"),
    P(f'best critic score: {selection["best_critic_score"]} · rolled_back={selection["rolled_back"]}', "SmallK"),
    Spacer(1, 3 * mm),
    P("Attempt 001은 초기 결과보다 SigLIP2가 증가했지만 critic 최저점은 1.0으로 동일했다. Attempt 002 역시 critic 점수 개선이 없었고 여러 기하 항목이 1점으로 하락하여, 오케스트레이터가 Attempt 001을 최종 결과로 보존했다.", "BodyK"),
    Spacer(1, 5 * mm),
    section_header("재현 정보"),
    Spacer(1, 3 * mm),
    P(f'Run ID: {run["run_id"]}', "SmallK"),
    P(f'Generator: {run["runtime"]["model_config"]["model_id"]}', "SmallK"),
    P(f'Critic: {run["runtime"]["critic_model_config"]["model_id"]} + stage2 checkpoint-100 adapter', "SmallK"),
    P(f'Similarity: {run["runtime"]["similarity_model_config"]["model_id"]}', "SmallK"),
    P(f'Stop reason: {run["stop_reason"]}', "SmallK"),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
print(OUTPUT)
