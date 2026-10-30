"""Create a self-contained PDF for the validated nonblank v3 teacher test."""

from __future__ import annotations

import base64
import json
import runpy
from html import escape
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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
RUN = ROOT / ".work/omnisvg_critic_distill/teacher_runs/test_nonblank_v3"
OUTPUT = RUN / "attempt-000-prompt-and-output.pdf"

manifest_bytes = (RUN / "run_manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
result_bytes = (RUN / "results.jsonl").read_bytes()
result = json.loads(result_bytes.splitlines()[0])
request_dir = RUN / result["request_dir"]
input_bytes = (request_dir / "input.json").read_bytes()
response_bytes = (request_dir / "response-000.json").read_bytes()
request = json.loads(input_bytes)
response = json.loads(response_bytes)
candidate_path = RUN / request["candidate_rendered_png_path"]
reference_path = RUN / request["reference_rendered_png_path"]
candidate_bytes = candidate_path.read_bytes()
reference_bytes = reference_path.read_bytes()
content_meta = request["user_content"]
assert [item["type"] for item in content_meta] == [
    "input_image", "input_text", "input_image", "input_text"
]
assert result["status"] == "validated"
assert not result["candidate_blank_white"] and not result["reference_blank_white"]
assert "Required evaluations order:" not in manifest["system_prompt"]

teacher = runpy.run_path(str(ROOT / "scripts/run_omnisvg_critic_teacher.py"))
api_schema = teacher["api_schema"]()
exact_payload = {
    "model": manifest["identity"]["model"],
    "instructions": manifest["system_prompt"],
    "input": [{
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64," + base64.b64encode(candidate_bytes).decode(),
                "detail": "high",
            },
            {"type": "input_text", "text": content_meta[1]["text"]},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64," + base64.b64encode(reference_bytes).decode(),
                "detail": "low",
            },
            {"type": "input_text", "text": content_meta[3]["text"]},
        ],
    }],
    "text": {"format": {
        "type": "json_schema", "name": "critic_output", "strict": True,
        "schema": api_schema,
    }},
    "reasoning": {"effort": manifest["identity"]["reasoning_effort"]},
    "max_output_tokens": 6000,
    "store": False,
}
payload_bytes = (json.dumps(exact_payload, ensure_ascii=False, indent=2) + "\n").encode()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="T", parent=styles["Title"], fontSize=17, leading=21, spaceAfter=8))
styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name="B", parent=styles["BodyText"], fontSize=8.5, leading=12, spaceAfter=5))
styles.add(ParagraphStyle(name="S", parent=styles["BodyText"], fontSize=7.2, leading=10, textColor=colors.HexColor("#475569")))
styles.add(ParagraphStyle(name="M", fontName="Courier", fontSize=7.0, leading=9.8))
styles.add(ParagraphStyle(name="I", fontName="Helvetica-Bold", fontSize=8, leading=10, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="C", fontName="Helvetica", fontSize=7.1, leading=9.2))


def paragraph(text: str, style: str = "B") -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])


def pre(text: str) -> Preformatted:
    return Preformatted(text, styles["M"], maxLineLength=107, splitChars=" ,;:/\\<>-=_")


def image_box(path: Path, label: str, width: int = 205) -> Table:
    table = Table(
        [[Paragraph(label, styles["I"])], [Image(str(path), width=width, height=width)]],
        colWidths=[width + 14], hAlign="CENTER",
    )
    table.setStyle(TableStyle([
        ("BOX", (0, 1), (0, 1), 0.8, colors.HexColor("#94a3b8")),
        ("PADDING", (0, 1), (0, 1), 7),
        ("BACKGROUND", (0, 1), (0, 1), colors.white),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(42, 25, "OmniSVG nonblank teacher test | v3")
    canvas.drawRightString(570, 25, f"Page {doc.page}")
    canvas.restoreState()


issues = result["feedback"]["issues"]
evaluations = result["feedback"]["evaluations"]
story = [
    Paragraph("Validated teacher test with corrected rendering", styles["T"]),
    paragraph(
        f"Sample {result['sample_id']} | index {result['sample_index']} | "
        f"Response {result['response_id']} | {manifest['identity']['model']}", "S"
    ),
    paragraph(
        f"Status: {result['status']} | API attempts: {result['api_attempts']} | "
        f"Tokens: {result['usage']['input_tokens']} input + {result['usage']['output_tokens']} output",
        "S",
    ),
    Paragraph("Images actually sent", styles["H"]),
    Table(
        [[image_box(candidate_path, "content[0]: Candidate Render, detail=high"),
          image_box(reference_path, "content[2]: Reference Render, detail=low")]],
        colWidths=[250, 250], hAlign="CENTER",
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
    ),
    Spacer(1, 8),
    paragraph(
        "Both images were freshly rendered from their SVG code at 512 x 512 on white. "
        "Empty fill and stroke attributes were removed before rendering. The candidate "
        "was confirmed nonblank before the API call.", "S"
    ),
    Paragraph("Prompt review", styles["H"]),
    paragraph(
        "The system instructions reuse the current critic_v1 prompt builder. The teacher "
        "adds only reference-evidence rules. The old explicit 'Required evaluations order' "
        "numbered block is absent. The response is still locally validated for all 18 pairs."
    ),
    Paragraph("Teacher result at a glance", styles["H"]),
    paragraph(f"Returned {len(evaluations)} evaluations and {len(issues)} actionable issues."),
]
for index, issue in enumerate(issues, 1):
    story.append(paragraph(
        f"{index}. {issue['category']}/{issue['type']} [{', '.join(issue['target_ids'])}] — "
        f"{issue['observed']} Fix: {issue['fix']}"
    ))

story.extend([
    PageBreak(),
    Paragraph("Exact Responses API request structure", styles["T"]),
    paragraph(
        "Image data URLs are abbreviated on this page. The attached api-request-exact.json "
        "contains the complete base64 payload.", "S"
    ),
    pre(json.dumps({
        "model": exact_payload["model"],
        "instructions": "<full text on following pages>",
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": f"<candidate sha256={request['candidate_rendered_png_sha256']}>", "detail": "high"},
            {"type": "input_text", "text": "<exact content[1] below>"},
            {"type": "input_image", "image_url": f"<reference sha256={request['reference_rendered_png_sha256']}>", "detail": "low"},
            {"type": "input_text", "text": "<exact content[3] below>"},
        ]}],
        "text.format": {"type": "json_schema", "name": "critic_output", "strict": True},
        "reasoning": exact_payload["reasoning"],
        "max_output_tokens": 6000,
        "store": False,
    }, indent=2)),
    Spacer(1, 10),
    Paragraph("Exact system instructions", styles["T"]),
    paragraph("Passed through the Responses API instructions parameter without text changes.", "S"),
    pre(manifest["system_prompt"]),
    PageBreak(),
    Paragraph("Exact ordered user content", styles["T"]),
    paragraph("One user message contained these four items in this exact order.", "S"),
    KeepTogether([
        Paragraph("content[0] — input_image, detail=high", styles["H"]),
        image_box(candidate_path, "Candidate Render actually sent", 205),
    ]),
    Paragraph("content[1] — input_text", styles["H"]),
    pre(content_meta[1]["text"]),
    KeepTogether([
        Paragraph("content[2] — input_image, detail=low", styles["H"]),
        image_box(reference_path, "Reference Render actually sent", 205),
    ]),
    Paragraph("content[3] — input_text", styles["H"]),
    pre(content_meta[3]["text"]),
    PageBreak(),
    Paragraph("All 18 validated evaluations", styles["T"]),
])

rows = [["Category", "Type", "Applicable", "Score", "Reason"]]
for item in evaluations:
    rows.append([
        item["category"], item["type"], "yes" if item["applicable"] else "no",
        str(item["score"]) if item["score"] is not None else "N/A",
        Paragraph(escape(item["reason"]), styles["C"]),
    ])
table = Table(rows, colWidths=[62, 75, 55, 45, 263], repeatRows=1)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.1),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.extend([
    table,
    PageBreak(),
    Paragraph("Raw teacher output JSON", styles["T"]),
    paragraph("Indentation is added here for reading. output-raw.json contains the exact returned string.", "S"),
    pre(json.dumps(json.loads(response["output_text"]), ensure_ascii=False, indent=2)),
])

temporary = OUTPUT.with_suffix(".tmp.pdf")
SimpleDocTemplate(
    str(temporary), pagesize=(612, 792), leftMargin=42, rightMargin=42,
    topMargin=38, bottomMargin=45, title="OmniSVG teacher v3 nonblank test",
).build(story, onFirstPage=footer, onLaterPages=footer)

reader = PdfReader(str(temporary))
writer = PdfWriter()
writer.append_pages_from_reader(reader)
writer.add_metadata({"/Title": "OmniSVG teacher v3 nonblank test", "/Subject": result["sample_id"]})
attachments = {
    "api-request-exact.json": payload_bytes,
    "input.json": input_bytes,
    "response-000.json": response_bytes,
    "results.jsonl": result_bytes,
    "run-manifest.json": manifest_bytes,
    "system-prompt.txt": manifest["system_prompt"].encode(),
    "output-raw.json": response["output_text"].encode(),
    "candidate-actually-sent.png": candidate_bytes,
    "reference-actually-sent.png": reference_bytes,
}
for name, data in attachments.items():
    writer.add_attachment(filename=name, data=data)
with OUTPUT.open("wb") as handle:
    writer.write(handle)
temporary.unlink()

check = PdfReader(str(OUTPUT))
all_text = "\n".join(page.extract_text() or "" for page in check.pages)
for required in (
    "Images actually sent", "Exact Responses API request structure",
    "Exact system instructions", "Exact ordered user content",
    "All 18 validated evaluations", "Raw teacher output JSON",
):
    assert required in all_text, required
assert "Required evaluations order:" not in manifest["system_prompt"]
assert set(check.attachments) == set(attachments)
print(json.dumps({
    "output": str(OUTPUT), "pages": len(check.pages), "bytes": OUTPUT.stat().st_size,
    "attachments": sorted(check.attachments),
}, indent=2))
