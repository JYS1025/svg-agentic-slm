"""Resumable, rate-limited teacher critique of live OmniSVG rollout results."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from svg_agentic_slm.prompts.vlm_critic import (  # noqa: E402
    VLM_CRITIC_PROMPT_VERSION,
    build_vlm_critic_prompt,
    build_vlm_critic_system_prompt,
)
from svg_agentic_slm.svg.labeler import CriticLabeler  # noqa: E402

DEFAULT_SOURCE = ROOT / ".work/omnisvg_critic_distill/runs/full_10k"
DEFAULT_OUTPUT = ROOT / ".work/omnisvg_critic_distill/teacher_runs/luna_critic_v3_nonblank"
DEFAULT_ENV_FILE = ROOT / ".env"
MODEL = "openai/gpt-5.6-luna"
PROMPT_VERSION = "omnisvg-teacher-critic-v3-nonblank-rerender-2026-09-23"
PAIRS = [
    ("semantic", "presence"),
    ("semantic", "count"),
    ("semantic", "identity"),
    ("semantic", "state"),
    ("semantic", "text_content"),
    ("geometry", "contour"),
    ("geometry", "proportion"),
    ("geometry", "topology"),
    ("layout", "placement"),
    ("layout", "scale"),
    ("layout", "orientation"),
    ("layout", "spacing"),
    ("layout", "occlusion"),
    ("layout", "framing"),
    ("appearance", "color"),
    ("appearance", "surface"),
    ("appearance", "stroke"),
    ("appearance", "typography"),
]
SYSTEM_ADDENDUM = """DISTILLATION REFERENCE EVIDENCE

The first attached image is the Candidate Render and is the only image to
evaluate. The Original Design Prompt is authoritative.

The second attached image and the Reference SVG are fallible supporting
examples, not an exact answer key. They may clarify properties consistent with
the Original Design Prompt. Do not penalize a different valid realization of
unspecified details. If the reference is malformed, visually uninformative, or
contradicts the Original Design Prompt, disregard it. Never transfer reference
defects to the candidate.

Use only the Candidate Labeled SVG to locate visible findings. All target_ids
must come from the candidate's allowed_target_ids, never the Reference SVG.
"""


def system_prompt() -> str:
    return (
        build_vlm_critic_system_prompt(score_threshold=3.0)
        + "\n\n"
        + SYSTEM_ADDENDUM
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def get_api_key() -> str:
    """Use an exported key first, then the repository's ignored .env file."""
    key = os.environ.get("BIZROUTER_API_KEY", "").strip()
    if key:
        return key
    if DEFAULT_ENV_FILE.is_file():
        from dotenv import dotenv_values

        key = (dotenv_values(DEFAULT_ENV_FILE).get("BIZROUTER_API_KEY") or "").strip()
        if key:
            return key
    raise ValueError(
        f"Set BIZROUTER_API_KEY in the environment or {DEFAULT_ENV_FILE}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def repair_jsonl_tail(path: Path) -> None:
    if not path.exists():
        return
    with path.open("rb+") as handle:
        data = handle.read()
        if not data or data.endswith(b"\n"):
            return
        boundary = data.rfind(b"\n") + 1
        atomic_json(
            path.with_name(f"{path.name}.partial.{int(time.time())}.json"),
            {"truncated_tail_utf8": data[boundary:].decode("utf-8", errors="replace")},
        )
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                break  # A live writer may still be appending the final line.
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def source_rows(source_dir: Path, gpus: list[int]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for gpu in gpus:
        for row in jsonl_rows(source_dir / f"results.gpu{gpu}.jsonl"):
            latest[row["sample_id"]] = row
    return sorted(
        (row for row in latest.values() if row.get("status") == "generated"),
        key=lambda row: row["sample_index"],
    )


def input_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for row in jsonl_rows(path):
        sample_id = row["sample_id"]
        if sample_id in records:
            raise ValueError(f"Duplicate input sample ID: {sample_id}")
        records[sample_id] = row
    return records


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["sample_id"], row["candidate_svg_sha256"]


def latest_teacher_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in jsonl_rows(path):
        latest[result_key(row)] = row
    return latest


def blank_white_png(data: bytes) -> bool:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        extrema = image.convert("RGB").getextrema()
    return extrema == ((255, 255), (255, 255), (255, 255))


def render_svg(svg: str) -> tuple[bytes, str]:
    """Render safely after treating empty paint attributes as absent SVG attributes."""
    import cairosvg

    root = ElementTree.fromstring(svg)
    if not root.tag.endswith("svg"):
        raise ValueError("Document root is not SVG")
    for element in root.iter():
        local_name = element.tag.split("}")[-1]
        if local_name in {"script", "foreignObject", "image"}:
            raise ValueError(f"SVG contains unsupported element: {local_name}")
        for name in ("fill", "stroke"):
            if element.get(name) == "":
                del element.attrib[name]
        for attribute, value in element.attrib.items():
            attr_name = attribute.split("}")[-1]
            if attr_name.startswith("on"):
                raise ValueError("SVG contains an event handler")
            if attr_name in {"href", "src"} and not value.startswith("#"):
                raise ValueError("SVG contains an external reference")
            if "url(" in value and "url(#" not in value:
                raise ValueError("SVG contains an external URL")
    normalized = ElementTree.tostring(root, encoding="unicode")
    png = cairosvg.svg2png(
        bytestring=normalized.encode("utf-8"),
        output_width=512,
        output_height=512,
        background_color="#ffffff",
        unsafe=False,
    )
    return png, normalized


class BlankCandidateError(Exception):
    def __init__(self, metadata: dict[str, Any]) -> None:
        super().__init__("Rendered OmniSVG candidate is blank white")
        self.metadata = metadata


def checked_bytes(path: Path, expected_hash: str) -> bytes:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return data


def artifact_stem(sample_id: str, candidate_hash: str) -> str:
    return hashlib.sha256(f"{sample_id}:{candidate_hash}".encode()).hexdigest()[:24]


def prepare_request(
    source_dir: Path,
    input_dir: Path,
    output_dir: Path,
    row: dict[str, Any],
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    sample_id = row["sample_id"]
    if (
        sample_id != record["sample_id"]
        or row["selected_text_field"] != record["selected_text_field"]
    ):
        raise ValueError(f"Input/result provenance mismatch: {sample_id}")
    candidate_svg = checked_bytes(
        source_dir / row["candidate_svg_path"], row["candidate_svg_sha256"]
    ).decode("utf-8")
    reference_svg = checked_bytes(
        input_dir / row["reference_svg_path"], row["reference_svg_sha256"]
    ).decode("utf-8")
    stem = artifact_stem(sample_id, row["candidate_svg_sha256"])
    relative_dir = Path("requests") / stem[:2] / stem
    request_dir = output_dir / relative_dir
    candidate_png, candidate_render_svg = render_svg(candidate_svg)
    candidate_render_path = relative_dir / "candidate.rendered.png"
    atomic_bytes(output_dir / candidate_render_path, candidate_png)
    candidate_render_hash = hashlib.sha256(candidate_png).hexdigest()
    candidate_blank = blank_white_png(candidate_png)
    common_metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_index": row["sample_index"],
        "candidate_svg_sha256": row["candidate_svg_sha256"],
        "selected_text_field": record["selected_text_field"],
        "candidate_blank_white": candidate_blank,
        "source_candidate_svg_path": row["candidate_svg_path"],
        "source_reference_svg_path": row["reference_svg_path"],
        "candidate_rendered_png_path": str(candidate_render_path),
        "candidate_rendered_png_sha256": candidate_render_hash,
        "render_policy": "CairoSVG 512x512 white; empty fill/stroke attributes removed",
        "system_prompt_sha256": hashlib.sha256(system_prompt().encode()).hexdigest(),
    }
    if candidate_blank:
        atomic_json(
            request_dir / "input.json",
            {
                **common_metadata,
                "status": "skipped_blank_candidate",
                "skip_reason": (
                    "teacher API not called because the freshly rendered candidate "
                    "is blank white"
                ),
                "user_content": [],
            },
        )
        raise BlankCandidateError(
            {
                "request_dir": str(relative_dir),
                "candidate_blank_white": True,
                "candidate_rendered_png_sha256": candidate_render_hash,
            }
        )

    reference_png, reference_render_svg = render_svg(reference_svg)
    reference_render_path = relative_dir / "reference.rendered.png"
    atomic_bytes(output_dir / reference_render_path, reference_png)
    reference_render_hash = hashlib.sha256(reference_png).hexdigest()
    labeling = CriticLabeler().label(candidate_svg, attempt_id=sample_id)
    allowed_ids = sorted(labeling.elements)
    if not allowed_ids:
        raise ValueError(f"No candidate target IDs: {sample_id}")
    reference_blank = blank_white_png(reference_png)
    instruction = record["instruction"]
    critic_prompt = build_vlm_critic_prompt(
        instruction,
        labeled_svg=labeling.labeled_svg,
        allowed_target_ids=allowed_ids,
        score_threshold=3.0,
    )
    reference_label = (
        "\n\nReference evidence follows. The next image is Reference Render, a fallible "
        "supporting example rather than the image being evaluated. "
        f"reference_render_is_blank_white={str(reference_blank).lower()}. "
        "If blank or inconsistent with the prompt, disregard it."
    )
    reference_footer = (
        "Reference SVG (fallible supporting evidence only):\n"
        "<reference_svg_json>\n"
        + json.dumps(reference_svg, ensure_ascii=False)
        + "\n</reference_svg_json>\n\n"
        "Use the reference only as permitted by the system instructions. Evaluate and "
        "target the candidate only."
    )
    content = [
        {
            "type": "input_image",
            "image_url": "data:image/png;base64," + base64.b64encode(candidate_png).decode("ascii"),
            "detail": "high",
        },
        {"type": "input_text", "text": critic_prompt + reference_label},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64," + base64.b64encode(reference_png).decode("ascii"),
            "detail": "low",
        },
        {"type": "input_text", "text": reference_footer},
    ]
    atomic_json(
        request_dir / "input.json",
        {
            **common_metadata,
            "status": "ready_for_teacher",
            "reference_blank_white": reference_blank,
            "reference_rendered_png_path": str(reference_render_path),
            "reference_rendered_png_sha256": reference_render_hash,
            "candidate_render_svg_sha256": hashlib.sha256(
                candidate_render_svg.encode("utf-8")
            ).hexdigest(),
            "reference_render_svg_sha256": hashlib.sha256(
                reference_render_svg.encode("utf-8")
            ).hexdigest(),
            "allowed_target_ids": allowed_ids,
            "user_content": [
                {"index": 0, "type": "input_image", "label": "Candidate Render", "detail": "high"},
                {"index": 1, "type": "input_text", "text": critic_prompt + reference_label},
                {"index": 2, "type": "input_image", "label": "Reference Render", "detail": "low"},
                {"index": 3, "type": "input_text", "text": reference_footer},
            ],
        },
    )
    return (
        content,
        allowed_ids,
        {
            "request_dir": str(relative_dir),
            "candidate_blank_white": False,
            "reference_blank_white": reference_blank,
            "candidate_rendered_png_sha256": candidate_render_hash,
            "reference_rendered_png_sha256": reference_render_hash,
        },
    )


def api_schema() -> dict[str, Any]:
    """API-supported subset; the full repository schema is checked locally."""
    categories = ["semantic", "geometry", "layout", "appearance"]
    kinds = [kind for _, kind in PAIRS]
    evaluation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "type", "applicable", "score", "reason"],
        "properties": {
            "category": {"type": "string", "enum": categories},
            "type": {"type": "string", "enum": kinds},
            "applicable": {"type": "boolean"},
            "score": {"type": ["integer", "null"]},
            "reason": {"type": "string"},
        },
    }
    issue = {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "type", "scope", "target_ids", "observed", "expected", "fix"],
        "properties": {
            "category": {"type": "string", "enum": categories},
            "type": {"type": "string", "enum": kinds},
            "scope": {"type": "string", "enum": ["global", "object", "part"]},
            "target_ids": {"type": "array", "items": {"type": "string"}},
            "observed": {"type": "string"},
            "expected": {"type": "string"},
            "fix": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["evaluations", "issues"],
        "properties": {
            "evaluations": {"type": "array", "items": evaluation},
            "issues": {"type": "array", "items": issue},
        },
    }


def validate_feedback(text: str, allowed_ids: list[str]) -> dict[str, Any]:
    import jsonschema

    value = json.loads(text)
    schema = json.loads((ROOT / "src/svg_agentic_slm/agents/critic_output.schema.json").read_text())
    jsonschema.validate(value, schema)
    evaluations = value["evaluations"]
    evaluation_by_pair = {
        (item["category"], item["type"]): item for item in evaluations
    }
    if len(evaluation_by_pair) != len(evaluations):
        raise ValueError("Evaluations contain a duplicate category-type pair")
    if set(evaluation_by_pair) != set(PAIRS):
        raise ValueError("Evaluations must contain every required category-type pair")
    # The production critic contract requires complete, unique pair coverage but
    # does not reject an otherwise valid answer solely because array order differs.
    # Store distillation rows in one stable order without spending another API call.
    evaluations = [evaluation_by_pair[pair] for pair in PAIRS]
    value["evaluations"] = evaluations
    if not any(item["applicable"] for item in evaluations):
        raise ValueError("At least one evaluation must be applicable")
    below = set()
    for item in evaluations:
        if not item["reason"].strip():
            raise ValueError("An evaluation reason is empty")
        if item["applicable"] and item["score"] < 3:
            below.add((item["category"], item["type"]))
    issues = value["issues"]
    if bool(below) != bool(issues):
        raise ValueError("Issues must be present exactly when scores fall below 3")
    for item in issues:
        if (item["category"], item["type"]) not in below:
            raise ValueError("Issue does not correspond to a score below 3")
        if not set(item["target_ids"]) <= set(allowed_ids):
            raise ValueError("Issue cites an ID outside the candidate SVG")
        if (
            not item["target_ids"]
            and item["scope"] != "global"
            and not (
                item["category"] == "semantic" and item["type"] in {"presence", "text_content"}
            )
        ):
            raise ValueError("Empty target_ids is not permitted for this issue")
        if any(not item[key].strip() for key in ("observed", "expected", "fix")):
            raise ValueError("An issue explanation is empty")
    return value


class CallBudgetExhaustedError(Exception):
    pass


class CallGate:
    def __init__(self, max_calls: int, min_interval: float) -> None:
        self.max_calls = max_calls
        self.min_interval = min_interval
        self.calls = 0
        self.next_allowed = 0.0
        self.halted_reason: str | None = None
        self.lock = threading.Lock()

    def reserve(self) -> int:
        with self.lock:
            if self.halted_reason:
                raise CallBudgetExhaustedError(self.halted_reason)
            if self.calls >= self.max_calls:
                raise CallBudgetExhaustedError("API call budget reached")
            delay = self.next_allowed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.calls += 1
            self.next_allowed = time.monotonic() + self.min_interval
            return self.calls

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            self.next_allowed = max(self.next_allowed, time.monotonic() + seconds)

    def halt(self, reason: str) -> None:
        with self.lock:
            self.halted_reason = reason


def retry_delay(exc: Exception, attempt: int) -> float:
    headers = getattr(exc, "response", None)
    raw = None if headers is None else headers.headers.get("retry-after")
    try:
        if raw is not None:
            return min(120.0, max(1.0, float(raw)))
    except (TypeError, ValueError):
        pass
    return min(120.0, 5.0 * 2**attempt + random.uniform(0.0, 2.0))


def total_usage(attempts: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum((item.get("usage") or {}).get(field, 0) for item in attempts)
        for field in ("input_tokens", "output_tokens", "total_tokens")
    }


def validation_retry_text(validation_error: str) -> str:
    """Give a contract retry a changed, actionable input independent of sampling."""
    return (
        "The previous response failed local output validation. Return the complete "
        "critic JSON object again and correct the validation problem below. Preserve "
        "the same image-grounded judgment except where correction is required. Include "
        "each of the 18 category and type pairs exactly once.\n\n"
        "<previous_validation_error_json>\n"
        f"{json.dumps(validation_error, ensure_ascii=False)}\n"
        "</previous_validation_error_json>"
    )


def latest_validation_error(result: dict[str, Any] | None) -> str | None:
    """Recover the last contract error when resuming a failed sample."""
    if result is None:
        return None
    for attempt in reversed(result.get("attempts") or []):
        error = attempt.get("validation_error")
        if isinstance(error, str) and error:
            return error
    return None


def call_one(
    client: Any,
    gate: CallGate,
    source_dir: Path,
    input_dir: Path,
    output_dir: Path,
    row: dict[str, Any],
    record: dict[str, Any],
    max_attempts: int,
    reasoning_effort: str,
    initial_validation_error: str | None = None,
) -> dict[str, Any]:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    started = time.monotonic()
    basis: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "sample_index": row["sample_index"],
        "candidate_svg_sha256": row["candidate_svg_sha256"],
        "source_attempt": row["attempt"],
        "created_at": utc_now(),
    }
    try:
        content, allowed_ids, metadata = prepare_request(
            source_dir, input_dir, output_dir, row, record
        )
        basis.update(metadata)
    except BlankCandidateError as exc:
        return {
            **basis,
            **exc.metadata,
            "status": "skipped_blank_candidate",
            "api_attempts": 0,
            "usage_total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "latency_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {**basis, "status": "input_failed", "error": f"{type(exc).__name__}: {exc}"}

    attempts = []
    retry_validation_error = initial_validation_error
    for attempt in range(max_attempts):
        try:
            call_number = gate.reserve()
        except CallBudgetExhaustedError as exc:
            if attempts:
                return {
                    **basis,
                    "status": "retry_deferred",
                    "attempts": attempts,
                    "usage_total": total_usage(attempts),
                    "error": str(exc),
                }
            return {**basis, "status": "budget_deferred", "attempts": [], "error": str(exc)}
        called_at = utc_now()
        try:
            request_content = content
            if retry_validation_error is not None:
                request_content = [
                    *content,
                    {"type": "input_text", "text": validation_retry_text(retry_validation_error)},
                ]
            response = client.responses.create(
                model=MODEL,
                instructions=system_prompt(),
                input=[{"role": "user", "content": request_content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "critic_output",
                        "strict": True,
                        "schema": api_schema(),
                    }
                },
                reasoning={"effort": reasoning_effort},
                max_output_tokens=6000,
                store=False,
            )
            output_text = response.output_text or ""
            usage = (
                response.usage.model_dump(exclude_none=True) if response.usage is not None else None
            )
            attempt_data = {
                "attempt": attempt,
                "call_number_this_run": call_number,
                "called_at": called_at,
                "response_id": response.id,
                "response_status": response.status,
                "usage": usage,
                "output_text": output_text,
            }
            if retry_validation_error is not None:
                attempt_data["retry_validation_error"] = retry_validation_error
            attempts.append(attempt_data)
            atomic_json(
                output_dir / basis["request_dir"] / f"response-{attempt:03d}.json", attempt_data
            )
            try:
                feedback = validate_feedback(output_text, allowed_ids)
            except Exception as exc:
                retry_validation_error = f"{type(exc).__name__}: {exc}"[:600]
                attempt_data["validation_error"] = retry_validation_error
                atomic_json(
                    output_dir / basis["request_dir"] / f"response-{attempt:03d}.json",
                    attempt_data,
                )
                if attempt + 1 < max_attempts:
                    gate.cooldown(retry_delay(exc, attempt))
                continue
            return {
                **basis,
                "status": "validated",
                "feedback": feedback,
                "response_id": response.id,
                "usage": usage,
                "usage_total": total_usage(attempts),
                "api_attempts": len(attempts),
                "latency_seconds": round(time.monotonic() - started, 3),
            }
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            delay = retry_delay(exc, attempt)
            gate.cooldown(delay)
            attempts.append(
                {
                    "attempt": attempt,
                    "called_at": called_at,
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "retry_after_seconds": round(delay, 2),
                }
            )
        except APIStatusError as exc:
            retryable = exc.status_code >= 500
            delay = retry_delay(exc, attempt) if retryable else 0.0
            if retryable:
                gate.cooldown(delay)
            elif exc.status_code in {400, 401, 402, 403, 404}:
                gate.halt(f"Non-retryable HTTP {exc.status_code}; inspect response")
            attempts.append(
                {
                    "attempt": attempt,
                    "called_at": called_at,
                    "status_code": exc.status_code,
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "retry_after_seconds": round(delay, 2),
                }
            )
            if not retryable:
                break
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "called_at": called_at,
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                }
            )
            if attempt + 1 < max_attempts:
                gate.cooldown(retry_delay(exc, attempt))
    return {
        **basis,
        "status": "api_or_validation_failed",
        "api_attempts": len(attempts),
        "attempts": attempts,
        "usage_total": total_usage(attempts),
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def summary(output_dir: Path, gate: CallGate | None = None) -> dict[str, Any]:
    rows = jsonl_rows(output_dir / "results.jsonl")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[result_key(row)] = row
    counts = Counter(row["status"] for row in latest.values())
    input_tokens = sum((row.get("usage_total") or {}).get("input_tokens", 0) for row in rows)
    output_tokens = sum((row.get("usage_total") or {}).get("output_tokens", 0) for row in rows)
    result: dict[str, Any] = {
        "total_result_versions": len(latest),
        "validated": counts["validated"],
        "skipped_blank_candidate": counts["skipped_blank_candidate"],
        "processed": counts["validated"] + counts["skipped_blank_candidate"],
        "by_status": dict(counts),
        "total_api_calls_recorded": sum(row.get("api_attempts", 0) for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_calls_this_run": None if gate is None else gate.calls,
        "halted_reason": None if gate is None else gate.halted_reason,
        "updated_at": utc_now(),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        teacher_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_dir = Path(teacher_manifest["identity"]["source_run"])
        source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
        source = source_rows(source_dir, source_manifest["settings"]["gpus"])
        terminal = {
            key
            for key, row in latest.items()
            if row["status"] in {"validated", "skipped_blank_candidate"}
        }
        result["source_generated"] = len(source)
        result["awaiting_teacher"] = sum(result_key(row) not in terminal for row in source)
        result["source_run_finished"] = (source_dir / "summary.json").is_file()
    return result


def run(args: argparse.Namespace) -> None:
    from openai import OpenAI

    if args.max_api_calls < 1 or args.max_concurrency < 1 or args.max_attempts < 1:
        raise ValueError("Call budget, concurrency, and attempts must be positive")
    if args.min_interval < 0 or args.poll_interval <= 0:
        raise ValueError("Intervals must be nonnegative, and polling must be positive")
    key = get_api_key()
    source_dir = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text())
    settings = source_manifest["settings"]
    input_path = Path(settings["input_manifest"])
    if sha256_file(input_path) != settings["input_sha256"]:
        raise ValueError("Source input manifest changed since OmniSVG inference began")
    records = input_records(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another teacher runner owns {output_dir}") from exc
        manifest_path = output_dir / "run_manifest.json"
        identity = {
            "source_run": str(source_dir),
            "source_input_sha256": settings["input_sha256"],
            "model": MODEL,
            "api_base_url": args.base_url,
            "prompt_version": PROMPT_VERSION,
            "critic_prompt_version": VLM_CRITIC_PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(system_prompt().encode()).hexdigest(),
            "api_schema_sha256": hashlib.sha256(
                json.dumps(api_schema(), sort_keys=True).encode()
            ).hexdigest(),
            "reasoning_effort": args.reasoning_effort,
            "render_policy": "CairoSVG 512x512 white; empty fill/stroke attributes removed",
            "blank_candidate_policy": "skip exact blank-white candidate before API dispatch",
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing["identity"] != identity:
                raise ValueError("Teacher run identity changed; choose a new output directory")
        else:
            atomic_json(
                manifest_path,
                {"created_at": utc_now(), "identity": identity, "system_prompt": system_prompt()},
            )
        repair_jsonl_tail(output_dir / "results.jsonl")
        gate = CallGate(args.max_api_calls, args.min_interval)
        client = OpenAI(api_key=key, base_url=args.base_url, max_retries=0, timeout=args.timeout)
        scheduled = 0
        last_report = 0.0
        pending: dict[Future[dict[str, Any]], tuple[str, str]] = {}
        seen_in_flight: set[tuple[str, str]] = set()
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
            while True:
                latest = latest_teacher_results(output_dir / "results.jsonl")
                available = source_rows(source_dir, settings["gpus"])
                submitted_any = False
                for row in available:
                    if len(pending) >= args.max_concurrency or gate.halted_reason:
                        break
                    if args.max_samples is not None and scheduled >= args.max_samples:
                        break
                    if gate.calls + len(pending) >= args.max_api_calls:
                        break
                    if args.sample_id is not None and row["sample_id"] != args.sample_id:
                        continue
                    key_tuple = result_key(row)
                    if key_tuple in seen_in_flight:
                        continue
                    previous = latest.get(key_tuple)
                    if previous:
                        if previous["status"] in {"validated", "skipped_blank_candidate"}:
                            continue
                        if not args.retry_failures:
                            continue
                    previous_validation_error = latest_validation_error(previous)
                    record = records.get(row["sample_id"])
                    if record is None:
                        raise ValueError(
                            f"Source sample absent from input manifest: {row['sample_id']}"
                        )
                    future = pool.submit(
                        call_one,
                        client,
                        gate,
                        source_dir,
                        input_path.parent,
                        output_dir,
                        row,
                        record,
                        args.max_attempts,
                        args.reasoning_effort,
                        previous_validation_error,
                    )
                    pending[future] = key_tuple
                    seen_in_flight.add(key_tuple)
                    scheduled += 1
                    submitted_any = True
                if pending:
                    done, _ = wait(pending, timeout=args.poll_interval, return_when=FIRST_COMPLETED)
                    for future in done:
                        key_tuple = pending.pop(future)
                        seen_in_flight.remove(key_tuple)
                        result = future.result()
                        if result["status"] != "budget_deferred":
                            append_jsonl(output_dir / "results.jsonl", result)
                        print(
                            f"{utc_now()} sample={result['sample_id']} status={result['status']} "
                            f"api_calls={gate.calls}/{gate.max_calls}",
                            flush=True,
                        )
                else:
                    time.sleep(args.poll_interval if args.follow else 0)
                now = time.monotonic()
                if now - last_report >= args.progress_interval:
                    current = summary(output_dir, gate)
                    atomic_json(output_dir / "progress.json", current)
                    print(json.dumps(current, ensure_ascii=False), flush=True)
                    last_report = now
                if pending:
                    continue
                if gate.halted_reason or gate.calls >= args.max_api_calls:
                    break
                if args.max_samples is not None and scheduled >= args.max_samples:
                    break
                if submitted_any:
                    continue
                if not args.follow or (source_dir / "summary.json").exists():
                    break
        final = summary(output_dir, gate)
        atomic_json(output_dir / "progress.json", final)
        print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="critique completed OmniSVG results")
    run_parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--base-url", default="https://api.bizrouter.ai/v1")
    run_parser.add_argument("--sample-id")
    run_parser.add_argument("--max-samples", type=int)
    run_parser.add_argument("--max-api-calls", type=int, default=1)
    run_parser.add_argument("--max-concurrency", type=int, default=1)
    run_parser.add_argument("--min-interval", type=float, default=5.0)
    run_parser.add_argument("--max-attempts", type=int, default=2)
    run_parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="medium"
    )
    run_parser.add_argument("--timeout", type=float, default=180.0)
    run_parser.add_argument("--poll-interval", type=float, default=5.0)
    run_parser.add_argument("--progress-interval", type=float, default=15.0)
    run_parser.add_argument("--follow", action="store_true", help="wait for new OmniSVG results")
    run_parser.add_argument("--retry-failures", action="store_true")
    status_parser = sub.add_parser("status", help="inspect durable teacher progress")
    status_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    status_parser.add_argument("--watch", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        while True:
            print(
                json.dumps(summary(args.output_dir.resolve()), ensure_ascii=False, indent=2),
                flush=True,
            )
            if args.watch <= 0:
                break
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
