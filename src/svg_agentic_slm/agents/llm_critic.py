"""Schema-constrained multimodal SVG critic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import (
    CriticCallTrace, CriticFeedback, CriticInput, CriticIssue, CriticTraceError,
    validate_critic_feedback,
)
from svg_agentic_slm.models.base import BaseMultimodalModelBackend
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.critic_prompts import (
    MULTIMODAL_CRITIC_PROMPT_VERSION, MULTIMODAL_CRITIC_SYSTEM_PROMPT,
    build_multimodal_critic_prompt,
    build_critic_prompt,
)
from svg_agentic_slm.prompts.system_prompts import get_svg_critic_system_prompt

if TYPE_CHECKING:
    from svg_agentic_slm.models.base import BaseModelBackend

LLM_CRITIC_VERSION = "llm-multimodal-critic-v2-score-v1"
ALLOWED_TYPES = {
    "content": {"element_presence_or_count", "object_identity_or_state", "reference_or_instance", "text_or_label_content"},
    "layout": {"viewport_or_clipping", "placement_or_transform", "relative_scale_alignment_or_spacing", "stacking_or_occlusion"},
    "shape": {"contour_or_curve_geometry", "closure_or_part_connectivity", "topology_or_fill_region"},
    "style": {"fill_or_paint_server", "stroke_or_marker", "visibility_opacity_or_compositing", "typography_or_glyph_appearance"},
}


class LLMCritic(BaseCritic):
    def __init__(self, model_backend: BaseModelBackend, max_retries: int = 1) -> None:
        self._model = model_backend
        self._max_retries = max_retries
        self._schema = json.loads((Path(__file__).with_name("critic_output.schema.json")).read_text())

    @property
    def name(self) -> str: return "LLMCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        """Backward-compatible v1 entry point; production factory never uses it for vision."""
        response = self._model.generate(
            build_critic_prompt(instruction, svg_content),
            system_prompt=get_svg_critic_system_prompt(),
        )
        if isinstance(response, str): response = ModelResponse(text=response, model_id="legacy-backend")
        text = response.text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start: raise ValueError("LLM critic response must contain a JSON object.")
        try: payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc: raise ValueError("LLM critic response contains invalid JSON.") from exc
        required = {"score", "is_valid", "matches_instruction", "issues", "suggestions"}
        missing = sorted(required - set(payload))
        if missing: raise ValueError("LLM critic response is missing required field(s): " + ", ".join(missing))
        return validate_critic_feedback(CriticFeedback(
            score=payload["score"], is_valid=payload["is_valid"],
            matches_instruction=payload["matches_instruction"], issues=payload["issues"],
            suggestions=payload["suggestions"], critic_type="llm", raw_response=response.text,
            critic_version="llm-critic-v1", model_id=response.model_id,
            model_revision=response.model_revision, prompt_version="critic-json-v1",
        ))

    def critique_attempt(self, value: CriticInput) -> CriticFeedback:
        if not isinstance(self._model, BaseMultimodalModelBackend):
            raise TypeError("Multimodal LLMCritic requires a multimodal backend.")
        if value.attempt_id != value.labeling.attempt_id:
            raise ValueError("Critic evidence attempt IDs do not match.")
        allowed_ids = sorted(value.labeling.elements)
        prompt = build_multimodal_critic_prompt(
            value.instruction, value.labeling.labeled_svg, allowed_ids, value.attempt_id,
            value.render_width, value.render_height,
        )
        last_error: Exception | None = None
        traces: list[CriticCallTrace] = []
        for retry in range(self._max_retries + 1):
            retry_prompt = prompt if retry == 0 else f"{prompt}\n\nPrevious response error: {last_error}. Return corrected JSON."
            response = self._model.generate_multimodal(
                retry_prompt, [value.render_png], system_prompt=MULTIMODAL_CRITIC_SYSTEM_PROMPT,
                response_format={"type": "json_object", "schema": self._schema},
            )
            trace = CriticCallTrace(
                critic_call_id=f"critic_call_{uuid4().hex}", retry_index=retry,
                response=response, prompt=retry_prompt,
                system_prompt=MULTIMODAL_CRITIC_SYSTEM_PROMPT,
                response_format={"type": "json_object", "schema": self._schema},
                generation_parameters=dict(response.metadata.get("generation_parameters", {})),
            )
            traces.append(trace)
            try:
                payload = json.loads(response.text)
                status, issues, preserve = _validate_payload(payload, set(allowed_ids))
                score = compatibility_score(status, issues)
                feedback = CriticFeedback(
                    score=score, is_valid=True, matches_instruction=status == "pass",
                    issues=[item.observed for item in issues], suggestions=[item.fix for item in issues],
                    critic_type="llm_multimodal", raw_response=response.text,
                    critic_version=LLM_CRITIC_VERSION, model_id=response.model_id,
                    model_revision=response.model_revision, prompt_version=MULTIMODAL_CRITIC_PROMPT_VERSION,
                    status=status, structured_issues=issues, preserve=preserve, schema_version=2,
                    metadata=dict(response.metadata), model_calls=traces,
                )
                trace.validation_success = True
                return validate_critic_feedback(feedback)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                trace.validation_error = f"{type(exc).__name__}: {exc}"
        raise CriticTraceError(
            f"Critic output validation failed after retry: {last_error}", traces,
        )


def compatibility_score(status: str, issues: list[CriticIssue]) -> float:
    if status == "pass": return 10.0
    if status == "invalid": return 0.0
    raw = 10.0 - sum({"critical": 5.0, "major": 2.0, "minor": 0.5}[i.severity] for i in issues)
    return min(10.0, max(0.0, raw))


def _validate_payload(payload: Any, allowed_ids: set[str]) -> tuple[str, list[CriticIssue], list[str]]:
    if not isinstance(payload, dict) or set(payload) != {"status", "issues", "preserve"}:
        raise ValueError("Critic output must contain only status, issues, and preserve.")
    status = payload["status"]
    raw_issues, preserve = payload["issues"], payload["preserve"]
    if status not in {"pass", "revise"} or not isinstance(raw_issues, list) or not isinstance(preserve, list):
        raise ValueError("Invalid status/issues/preserve types.")
    if len(raw_issues) > 3 or len(preserve) > 3 or any(not isinstance(x, str) or not x.strip() for x in preserve):
        raise ValueError("Issue/preserve limits were violated.")
    if status == "pass" and (raw_issues or preserve): raise ValueError("pass must be empty.")
    if status == "revise" and not raw_issues: raise ValueError("revise needs issues.")
    issues: list[CriticIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in raw_issues:
        required = {"category", "type", "severity", "scope", "target_ids", "observed", "expected", "fix"}
        if not isinstance(raw, dict) or set(raw) != required: raise ValueError("Invalid issue fields.")
        category, issue_type = raw["category"], raw["type"]
        if category not in ALLOWED_TYPES or issue_type not in ALLOWED_TYPES[category]: raise ValueError("Invalid category/type.")
        targets = raw["target_ids"]
        if not isinstance(targets, list) or len(targets) > 4 or len(set(targets)) != len(targets): raise ValueError("Invalid targets.")
        if any(t not in allowed_ids for t in targets): raise ValueError("Unknown target ID.")
        if not targets and not (category == "content" and issue_type == "element_presence_or_count") and raw["scope"] != "global":
            raise ValueError("Empty targets require a missing object or global issue.")
        if raw["severity"] not in {"critical", "major", "minor"} or raw["scope"] not in {"global", "object", "part"}:
            raise ValueError("Invalid severity/scope.")
        if any(not isinstance(raw[k], str) or not raw[k].strip() for k in ("observed", "expected", "fix")):
            raise ValueError("Issue text must be non-empty.")
        key = (category, issue_type, tuple(targets))
        if key in seen: raise ValueError("Duplicate issue.")
        seen.add(key); issues.append(CriticIssue(**raw))
    return status, issues, list(dict.fromkeys(preserve))
