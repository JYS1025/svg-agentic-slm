"""Document the exact saved v2 teacher request and the reference-render defect."""

from __future__ import annotations

import base64
import hashlib
import json
import runpy
from html import escape
from pathlib import Path
from xml.etree import ElementTree

import cairosvg
from PIL import Image as PILImage
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
TEACHER_RUN = ROOT / ".work/omnisvg_critic_distill/teacher_runs/test_first_v2"
SOURCE_RUN = ROOT / ".work/omnisvg_critic_distill/runs/full_10k"
INPUT_ROOT = ROOT / ".work/omnisvg_critic_distill/inputs"
OUTPUT = TEACHER_RUN / "attempt-000-prompt-and-output.pdf"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_empty_paint(svg: str) -> str:
    """Remove empty fill/stroke presentation attributes so SVG defaults apply."""
    root = ElementTree.fromstring(svg)
    for element in root.iter():
        for name in ("fill", "stroke"):
            if element.get(name) == "":
                del element.attrib[name]
    return ElementTree.tostring(root, encoding="unicode")


manifest = json.loads((TEACHER_RUN / "run_manifest.json").read_text())
result = json.loads((TEACHER_RUN / "results.jsonl").read_text().splitlines()[0])
request_dir = TEACHER_RUN / result["request_dir"]
input_bytes = (request_dir / "input.json").read_bytes()
response_bytes = (request_dir / "response-000.json").read_bytes()
request_meta = json.loads(input_bytes)
response = json.loads(response_bytes)

candidate_path = SOURCE_RUN / request_meta["candidate_png_path"]
actual_reference_path = SOURCE_RUN / request_meta["reference_png_path"]
candidate_bytes = candidate_path.read_bytes()
actual_reference_bytes = actual_reference_path.read_bytes()
reference_svg_path = INPUT_ROOT / request_meta["reference_svg_path"]
reference_svg_bytes = reference_svg_path.read_bytes()
reference_svg = reference_svg_bytes.decode("utf-8")

assert candidate_bytes == actual_reference_bytes
assert request_meta["reference_blank_white"] is True
assert PILImage.open(candidate_path).convert("RGB").getextrema() == ((255, 255),) * 3
assert PILImage.open(actual_reference_path).convert("RGB").getextrema() == ((255, 255),) * 3

corrected_reference_path = request_dir / "reference.corrected-preview.not-sent.png"
corrected_svg = normalize_empty_paint(reference_svg)
corrected_reference_path.write_bytes(
    cairosvg.svg2png(
        bytestring=corrected_svg.encode(),
        output_width=512,
        output_height=512,
        background_color="#ffffff",
        unsafe=False,
    )
)
assert PILImage.open(corrected_reference_path).convert("RGB").getextrema() != ((255, 255),) * 3

teacher_module = runpy.run_path(str(ROOT / "scripts/run_omnisvg_critic_teacher.py"))
api_schema = teacher_module["api_schema"]()
parts = request_meta["user_text_parts"]
exact_payload = {
    "model": manifest["identity"]["model"],
    "instructions": manifest["system_prompt"],
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": parts[0]},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(candidate_bytes).decode(),
                    "detail": "high",
                },
                {"type": "input_text", "text": parts[1]},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(actual_reference_bytes).decode(),
                    "detail": "low",
                },
                {"type": "input_text", "text": parts[2]},
            ],
        }
    ],
    "text": {
        "format": {
            "type": "json_schema",
            "name": "critic_output",
            "strict": True,
            "schema": api_schema,
        }
    },
    "reasoning": {"effort": manifest["identity"]["reasoning_effort"]},
    "max_output_tokens": 6000,
    "store": False,
}
payload_bytes = (json.dumps(exact_payload, ensure_ascii=False, indent=2) + "\n").encode()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], fontSize=17, leading=21, spaceAfter=8))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.6, leading=12, spaceAfter=5))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.3, leading=10, textColor=colors.HexColor("#4b5563")))
styles.add(ParagraphStyle(name="Monox", fontName="Courier", fontSize=7.1, leading=10))
styles.add(ParagraphStyle(name="ImageLabelx", fontName="Helvetica-Bold", fontSize=8, leading=10, alignment=TA_CENTER))


def p(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])


def code(text: str, width: int = 106) -> Preformatted:
    return Preformatted(text, styles["Monox"], maxLineLength=width, splitChars=" ,;:/\\<>-=_")


def image_box(path: Path, label: str, width: int = 190) -> Table:
    table = Table(
        [[Paragraph(label, styles["ImageLabelx"])], [Image(str(path), width=width, height=width)]],
        colWidths=[width + 14],
        hAlign="CENTER",
    )
    table.setStyle(TableStyle([
        ("BOX", (0, 1), (0, 1), 0.8, colors.HexColor("#94a3b8")),
        ("BACKGROUND", (0, 1), (0, 1), colors.white),
        ("PADDING", (0, 1), (0, 1), 7),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(42, 25, "Exact saved Responses API request | test_first_v2")
    canvas.drawRightString(570, 25, f"Page {doc.page}")
    canvas.restoreState()


candidate_hash = sha(candidate_bytes)
reference_hash = sha(actual_reference_bytes)
original_instruction = json.loads(
    parts[0].split("<original_instruction_json>\n", 1)[1]
    .split("\n</original_instruction_json>", 1)[0]
)
story = [
    Paragraph("Exact teacher request: saved API attempt 0", styles["DocTitle"]),
    p(
        f"Sample {result['sample_id']} | Response {response['response_id']} | "
        f"Model {manifest['identity']['model']}",
        "Smallx",
    ),
    Paragraph("Finding", styles["H2x"]),
    p(
        "The previous PDF used the same PNG bytes that were actually sent to the teacher. "
        "The candidate and reference PNGs were byte-for-byte identical blank-white images. "
        "This was caused by reference preprocessing: the source SVG contains fill=\"\", "
        "which CairoSVG rendered as transparent on the white background."
    ),
    p(f"Candidate PNG SHA-256: {candidate_hash}\nReference PNG SHA-256: {reference_hash}", "Smallx"),
    Paragraph("Images actually sent to the teacher", styles["H2x"]),
    Table(
        [[image_box(candidate_path, "content[1]: Candidate, detail=high"),
          image_box(actual_reference_path, "content[3]: Reference, detail=low")]],
        colWidths=[250, 250], hAlign="CENTER",
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
    ),
    Spacer(1, 8),
    p(
        "The teacher also received the explicit text reference_render_is_blank_white=true. "
        "Therefore the blank reference in the historical call and its output is real, not a PDF-layout error.",
        "Smallx",
    ),
    Paragraph("Source reference after correcting empty fill (NOT sent)", styles["H2x"]),
    Table(
        [[image_box(actual_reference_path, "Actual API image: blank"),
          image_box(corrected_reference_path, "Corrected local preview: not sent")]],
        colWidths=[250, 250], hAlign="CENTER",
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
    ),
    Spacer(1, 5),
    p(
        "The corrected preview removes the empty fill attribute, allowing the SVG default black fill. "
        "It documents the rendering defect only and must not be interpreted as part of the saved API call.",
        "Smallx",
    ),
    PageBreak(),
    Paragraph("Responses API call structure", styles["DocTitle"]),
    code(json.dumps({
        "model": exact_payload["model"],
        "instructions": "<full system prompt on following pages>",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "<content[0] below>"},
                {"type": "input_image", "image_url": f"<embedded PNG sha256={candidate_hash}>", "detail": "high"},
                {"type": "input_text", "text": "<content[2] below>"},
                {"type": "input_image", "image_url": f"<embedded PNG sha256={reference_hash}>", "detail": "low"},
                {"type": "input_text", "text": "<content[4] below>"},
            ],
        }],
        "text.format": {"type": "json_schema", "name": "critic_output", "strict": True},
        "reasoning": exact_payload["reasoning"],
        "max_output_tokens": 6000,
        "store": False,
    }, indent=2)),
    Spacer(1, 10),
    Paragraph("Original design instruction", styles["H2x"]),
    p(original_instruction),
    Paragraph("Exact system instructions", styles["DocTitle"]),
    p("Passed through the Responses API instructions parameter. Text is unchanged; only visual line wrapping is added.", "Smallx"),
    code(manifest["system_prompt"]),
    PageBreak(),
    Paragraph("Exact ordered user content", styles["DocTitle"]),
    p("One user message contained the following five content items in this exact order.", "Smallx"),
    Paragraph("content[0] — input_text", styles["H2x"]), code(parts[0]),
    KeepTogether([
        Paragraph("content[1] — input_image, detail=high", styles["H2x"]),
        image_box(candidate_path, "Candidate Render actually sent", 205),
    ]),
    Paragraph("content[2] — input_text", styles["H2x"]), code(parts[1]),
    KeepTogether([
        Paragraph("content[3] — input_image, detail=low", styles["H2x"]),
        image_box(actual_reference_path, "Reference Render actually sent", 205),
    ]),
    Paragraph("content[4] — input_text", styles["H2x"]), code(parts[2]),
    PageBreak(),
    Paragraph("Teacher output JSON", styles["DocTitle"]),
    p("Indentation is added for readability. output-raw.json is attached with the exact returned string.", "Smallx"),
    code(json.dumps(json.loads(response["output_text"]), ensure_ascii=False, indent=2)),
]

temporary = OUTPUT.with_suffix(".new.pdf")
SimpleDocTemplate(
    str(temporary), pagesize=(612, 792), leftMargin=42, rightMargin=42,
    topMargin=38, bottomMargin=45,
    title="Exact OmniSVG teacher request and reference rendering diagnosis",
).build(story, onFirstPage=footer, onLaterPages=footer)

reader = PdfReader(str(temporary))
writer = PdfWriter()
writer.append_pages_from_reader(reader)
writer.add_metadata({
    "/Title": "Exact OmniSVG teacher request and reference rendering diagnosis",
    "/Subject": result["sample_id"],
})
attachments = {
    "api-request-exact.json": payload_bytes,
    "input-metadata.json": input_bytes,
    "response-000.json": response_bytes,
    "output-raw.json": response["output_text"].encode(),
    "system-prompt.txt": manifest["system_prompt"].encode(),
    "candidate-actually-sent.png": candidate_bytes,
    "reference-actually-sent.png": actual_reference_bytes,
    "reference-source.svg": reference_svg_bytes,
    "reference-corrected-preview-not-sent.png": corrected_reference_path.read_bytes(),
}
for name, data in attachments.items():
    writer.add_attachment(filename=name, data=data)
with OUTPUT.open("wb") as handle:
    writer.write(handle)
temporary.unlink()

check = PdfReader(str(OUTPUT))
text = "\n".join(page.extract_text() or "" for page in check.pages)
for required in (
    "Images actually sent to the teacher",
    "Corrected local preview: not sent",
    "Exact system instructions",
    "Exact ordered user content",
    "content[3] — input_image, detail=low",
    "Teacher output JSON",
):
    assert required in text, required
assert set(check.attachments) == set(attachments)
print(json.dumps({
    "output": str(OUTPUT),
    "pages": len(check.pages),
    "bytes": OUTPUT.stat().st_size,
    "actual_images_identical": candidate_hash == reference_hash,
    "actual_image_sha256": candidate_hash,
    "attachments": sorted(check.attachments),
}, indent=2))
