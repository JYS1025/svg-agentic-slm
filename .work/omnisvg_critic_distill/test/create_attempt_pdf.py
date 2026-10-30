"""Create a readable, self-contained PDF for the saved first v2 teacher call."""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / ".work/omnisvg_critic_distill/teacher_runs/test_first_v2"
OUTPUT = RUN / "attempt-000-prompt-and-output.pdf"
SOURCE_RUN = ROOT / ".work/omnisvg_critic_distill/runs/full_10k"

manifest = json.loads((RUN / "run_manifest.json").read_text())
results = [json.loads(line) for line in (RUN / "results.jsonl").read_text().splitlines() if line]
assert len(results) == 1, "Expected exactly one saved result"
result = results[0]
request_dir = RUN / result["request_dir"]
request_bytes = (request_dir / "input.json").read_bytes()
response_bytes = (request_dir / "response-000.json").read_bytes()
request = json.loads(request_bytes)
response = json.loads(response_bytes)
assert result["api_attempts"] == 1
assert result["sample_id"] == request["sample_id"]
assert response["output_text"]
assert json.loads(response["output_text"]) == result["feedback"]
assert hashlib.sha256(manifest["system_prompt"].encode()).hexdigest() == request["system_prompt_sha256"]

candidate_image = SOURCE_RUN / request["candidate_png_path"]
reference_image = SOURCE_RUN / request["reference_png_path"]
assert candidate_image.is_file() and reference_image.is_file()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleSmall", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=20, spaceAfter=8))
styles.add(ParagraphStyle(name="HeadingSmall", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.5, leading=12, spaceAfter=4))
styles.add(ParagraphStyle(name="FootSmall", parent=styles["BodyText"], fontSize=7.4, leading=10, textColor=colors.HexColor("#4b5563")))
styles.add(ParagraphStyle(name="MonoSmall", fontName="Courier", fontSize=7.4, leading=10.5, spaceAfter=5))
styles.add(ParagraphStyle(name="CellSmall", fontName="Helvetica", fontSize=7.2, leading=9.5))
styles.add(ParagraphStyle(name="ImageLabel", fontName="Helvetica-Bold", fontSize=8, leading=11, alignment=TA_CENTER))


def para(text: str, style: str = "BodySmall") -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])


def pre(text: str) -> Preformatted:
    # PDF line wraps are presentation only. The exact UTF-8 files are attached.
    return Preformatted(text, styles["MonoSmall"], maxLineLength=106, splitChars=" ,;:/\\<>-=_")


def section(title: str, text: str) -> list:
    return [Paragraph(title, styles["HeadingSmall"]), pre(text), Spacer(1, 7)]


def framed_image(path: Path, label: str, width: int = 205) -> Table:
    items = [Paragraph(label, styles["ImageLabel"]), Image(str(path), width=width, height=width)]
    table = Table([[items[0]], [items[1]]], colWidths=[width + 14], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BOX", (0, 1), (0, 1), 0.7, colors.HexColor("#9ca3af")),
        ("BACKGROUND", (0, 1), (0, 1), colors.white),
        ("LEFTPADDING", (0, 1), (0, 1), 7),
        ("RIGHTPADDING", (0, 1), (0, 1), 7),
        ("TOPPADDING", (0, 1), (0, 1), 7),
        ("BOTTOMPADDING", (0, 1), (0, 1), 7),
    ]))
    return table


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(43, 26, "Saved teacher call | test_first_v2 | one API attempt")
    canvas.drawRightString(568, 26, f"Page {doc.page}")
    canvas.restoreState()


parts = request["user_text_parts"]
assert len(parts) == 3
original_instruction = json.loads(parts[0].split("<original_instruction_json>\n", 1)[1].split("\n</original_instruction_json>", 1)[0])
feedback = result["feedback"]
issue = feedback["issues"][0]
evaluations = feedback["evaluations"]
applicable = [e for e in evaluations if e["applicable"]]

story = [
    Paragraph("OmniSVG critic distillation: one teacher call", styles["TitleSmall"]),
    para(f"Sample: {result['sample_id']}  |  OmniSVG attempt: {result['source_attempt']}  |  API attempt: 0  |  Status: {result['status']}", "FootSmall"),
    para(f"Model: {manifest['identity']['model']}  |  Response ID: {result['response_id']}  |  Tokens: {result['usage']['input_tokens']} input + {result['usage']['output_tokens']} output", "FootSmall"),
    Paragraph("Original design prompt", styles["HeadingSmall"]),
    para(original_instruction),
    Spacer(1, 7),
]

image_table = Table([[
    framed_image(candidate_image, "1. Candidate render (evaluated)"),
    framed_image(reference_image, "2. Reference render (supporting)"),
]], colWidths=[250, 250], hAlign="CENTER")
image_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.extend([
    image_table,
    Spacer(1, 7),
    para("Both saved renders are blank white. The request explicitly marked the reference render as blank and fallible.", "FootSmall"),
    Paragraph("Teacher result at a glance", styles["HeadingSmall"]),
    para(f"18 score pairs returned; {len(applicable)} applicable; {len(evaluations)-len(applicable)} not applicable; {len(feedback['issues'])} issue. The only issue targets {', '.join(issue['target_ids'])}."),
    para("Observed: " + issue["observed"]),
    para("Suggested fix: " + issue["fix"]),
    Spacer(1, 8),
    para("Following pages show the complete system instructions, ordered user content (including both images), all 18 scores, and the returned JSON. Exact saved input and response files are embedded as PDF attachments.", "FootSmall"),
    PageBreak(),
    Paragraph("All 18 evaluation pairs", styles["TitleSmall"]),
])

rows = [["Category", "Type", "Applicable", "Score (0-4)", "Reason"]]
for e in evaluations:
    rows.append([
        e["category"], e["type"], "yes" if e["applicable"] else "no",
        str(e["score"]) if e["score"] is not None else "N/A",
        Paragraph(escape(e["reason"]), styles["CellSmall"]),
    ])
score_table = Table(rows, colWidths=[62, 75, 60, 65, 238], repeatRows=1, hAlign="LEFT")
score_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.2),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.extend([score_table, PageBreak(), Paragraph("Exact API input: system instructions", styles["TitleSmall"]),
              para("These instructions are the saved system prompt sent to the Responses API. Visual line wrapping does not change the source text.", "FootSmall"),
              pre(manifest["system_prompt"]), Spacer(1, 16),
              Paragraph("Exact API input: ordered user content", styles["TitleSmall"]),
              para("The request sent each text/image item in this order. Candidate image detail=high; reference image detail=low.", "FootSmall")])

story.extend(section("1. input_text", parts[0]))
story.extend([Paragraph("2. input_image: Candidate Render (detail=high)", styles["HeadingSmall"]),
              framed_image(candidate_image, "Candidate Render", width=200), Spacer(1, 8)])
story.extend(section("3. input_text", parts[1]))
story.extend([Paragraph("4. input_image: Reference Render (detail=low)", styles["HeadingSmall"]),
              framed_image(reference_image, "Reference Render", width=200), Spacer(1, 8)])
story.extend(section("5. input_text", parts[2]))
story.extend([PageBreak(), Paragraph("Output JSON returned by the teacher", styles["TitleSmall"]),
              para("The JSON below is the parsed API output with indentation added for reading. The exact raw output string is attached as output-raw.json.", "FootSmall"),
              pre(json.dumps(json.loads(response["output_text"]), ensure_ascii=False, indent=2))])

temp_pdf = OUTPUT.with_suffix(".tmp.pdf")
doc = SimpleDocTemplate(str(temp_pdf), pagesize=(612, 792), leftMargin=43, rightMargin=43,
                        topMargin=38, bottomMargin=46,
                        title="OmniSVG teacher critic: first v2 API attempt")
doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)

reader = PdfReader(str(temp_pdf))
writer = PdfWriter()
writer.append_pages_from_reader(reader)
writer.add_metadata({"/Title": "OmniSVG teacher critic: first v2 API attempt",
                     "/Subject": result["sample_id"]})
writer.add_attachment(filename="system-prompt.txt", data=manifest["system_prompt"].encode())
writer.add_attachment(filename="input.json", data=request_bytes)
writer.add_attachment(filename="response-000.json", data=response_bytes)
writer.add_attachment(filename="output-raw.json", data=response["output_text"].encode())
with OUTPUT.open("wb") as handle:
    writer.write(handle)
temp_pdf.unlink()

check = PdfReader(str(OUTPUT))
assert len(check.pages) >= 4
all_text = "\n".join(page.extract_text() for page in check.pages)
for needle in ("Original design prompt", "Candidate render", "Reference render", "All 18 evaluation pairs", "Output JSON returned by the teacher"):
    assert needle in all_text, needle
assert set(check.attachments) == {"system-prompt.txt", "input.json", "response-000.json", "output-raw.json"}
print(f"Created {OUTPUT} ({len(check.pages)} pages, {OUTPUT.stat().st_size} bytes)")
