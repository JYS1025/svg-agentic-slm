"""Image-grounded critic for rendered SVG output."""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Protocol, cast
from uuid import uuid4

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import (
    CriticCallTrace,
    CriticCategory,
    CriticFeedback,
    CriticInput,
    CriticIssue,
    CriticScope,
    CriticSeverity,
    CriticStatus,
    validate_critic_feedback,
)
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.vlm_critic import (
    VLM_CRITIC_PROMPT_VERSION,
    build_vlm_critic_prompt,
)
from svg_agentic_slm.svg.labeler import CriticLabeler
from svg_agentic_slm.svg.validator import SVGValidator

logger = logging.getLogger(__name__)

VLM_CRITIC_VERSION = "vlm-critic-v2-grounded"
_SYSTEM_PROMPT = ""
_TARGET_ID_RE = re.compile(r"^[sged][0-9]{4}$")
_ISSUE_KEYS = {
    "category",
    "type",
    "severity",
    "scope",
    "target_ids",
    "observed",
    "expected",
    "fix",
}
_ISSUE_TAXONOMY: dict[str, frozenset[str]] = {
    "content": frozenset(
        {
            "element_presence_or_count",
            "object_identity_or_state",
            "reference_or_instance",
            "text_or_label_content",
        }
    ),
    "layout": frozenset(
        {
            "viewport_or_clipping",
            "placement_or_transform",
            "relative_scale_alignment_or_spacing",
            "stacking_or_occlusion",
        }
    ),
    "shape": frozenset(
        {
            "contour_or_curve_geometry",
            "closure_or_part_connectivity",
            "topology_or_fill_region",
        }
    ),
    "style": frozenset(
        {
            "fill_or_paint_server",
            "stroke_or_marker",
            "visibility_opacity_or_compositing",
            "typography_or_glyph_appearance",
        }
    ),
}
_CRITIC_RESPONSE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "name": "critic_output",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "issues", "preserve"],
        "properties": {
            "status": {"enum": ["pass", "revise"]},
            "issues": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_ISSUE_KEYS),
                    "properties": {
                        "category": {"enum": list(_ISSUE_TAXONOMY)},
                        "type": {"type": "string"},
                        "severity": {"enum": ["critical", "major", "minor"]},
                        "scope": {"enum": ["global", "object", "part"]},
                        "target_ids": {
                            "type": "array",
                            "maxItems": 4,
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "pattern": "^[sged][0-9]{4}$",
                            },
                        },
                        "observed": {"type": "string"},
                        "expected": {"type": "string"},
                        "fix": {"type": "string"},
                    },
                },
            },
            "preserve": {
                "type": "array",
                "maxItems": 3,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
        },
    },
}


class VisionModel(Protocol):
    """Model operation required by the VLM critic."""

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        mime_type: str,
        max_new_tokens: int,
        do_sample: bool,
    ) -> ModelResponse: ...


class BytesRenderer(Protocol):
    """In-memory renderer operation required by the VLM critic."""

    def render_bytes(
        self,
        svg_content: str,
        *,
        output_width: int,
        output_height: int,
        background_color: str,
    ) -> bytes: ...


class VLMCritic(BaseCritic):
    """Evaluate a safely rendered SVG through a vision-language model."""

    def __init__(
        self,
        model: VisionModel,
        renderer: BytesRenderer,
        *,
        render_width: int = 512,
        render_height: int = 512,
        background_color: str = "#ffffff",
        max_new_tokens: int = 384,
    ) -> None:
        if render_width <= 0 or render_height <= 0:
            raise ValueError("VLM critic render dimensions must be positive.")
        if not background_color.strip():
            raise ValueError("VLM critic background_color must not be empty.")
        if max_new_tokens <= 0:
            raise ValueError("VLM critic max_new_tokens must be positive.")
        self._model = model
        self._renderer = renderer
        self._render_width = render_width
        self._render_height = render_height
        self._background_color = background_color
        self._max_new_tokens = max_new_tokens
        self._validator = SVGValidator()
        self._labeler = CriticLabeler()

    @property
    def name(self) -> str:
        return "VLMCritic"

    def critique_attempt(self, value: CriticInput) -> CriticFeedback:
        """Critique pre-rendered evidence correlated to one Generator attempt."""
        input_error = _validate_critic_input(value)
        if input_error is not None:
            return _invalid_feedback(
                input_error,
                attempt_id=getattr(value, "attempt_id", None),
                stage="critic_input",
            )

        allowed_target_ids = sorted(value.labeling.elements)
        calls: list[CriticCallTrace] = []
        retry_error: str | None = None

        for retry_index in range(2):
            prompt = build_vlm_critic_prompt(
                value.instruction,
                labeled_svg=value.labeling.labeled_svg,
                allowed_target_ids=allowed_target_ids,
                retry_error=retry_error,
            )
            response = self._model.generate_with_image(
                prompt,
                bytes(value.render_png),
                mime_type="image/png",
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
            if not isinstance(response, ModelResponse):
                raise TypeError("Vision model must return ModelResponse.")

            trace = CriticCallTrace(
                critic_call_id=f"critic_call_{uuid4().hex}",
                retry_index=retry_index,
                response=response,
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                response_format=copy.deepcopy(_CRITIC_RESPONSE_FORMAT),
                generation_parameters={
                    "max_new_tokens": self._max_new_tokens,
                    "do_sample": False,
                    "mime_type": "image/png",
                    "render_width": value.render_width,
                    "render_height": value.render_height,
                },
            )
            try:
                payload = _parse_json_object(response.text)
                status, issues, preserve = _parse_critic_contract(
                    payload,
                    allowed_target_ids=set(allowed_target_ids),
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                retry_error = f"{type(exc).__name__}: {exc}"
                trace.validation_error = retry_error
                calls.append(trace)
                logger.warning(
                    "Rejected grounded VLM response on try %d: %s",
                    retry_index + 1,
                    retry_error,
                )
                if retry_index == 0:
                    continue
                return _invalid_feedback(
                    f"VLM critic response violated the critic_v1 contract after retry: "
                    f"{retry_error}",
                    attempt_id=value.attempt_id,
                    stage="response_contract",
                    response=response,
                    model_calls=calls,
                )

            trace.validation_success = True
            calls.append(trace)
            return _contract_feedback(
                status=status,
                issues=issues,
                preserve=preserve,
                attempt_id=value.attempt_id,
                response=response,
                model_calls=calls,
                allowed_target_ids=allowed_target_ids,
            )

        raise AssertionError("Grounded VLM retry loop terminated unexpectedly.")

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        """Legacy adapter that validates, renders, labels, then critiques one SVG."""
        attempt_id = f"legacy_attempt_{uuid4().hex}"
        try:
            validation = self._validator.validate(svg_content)
        except Exception as exc:
            logger.warning("SVG validation failed before VLM critique: %s", exc)
            return _invalid_feedback(
                f"SVG validation failed: {type(exc).__name__}: {exc}",
                attempt_id=attempt_id,
                stage="svg_validation",
            )

        if not validation.is_valid:
            messages = validation.errors or ["SVG validation failed."]
            return _invalid_feedback(
                *messages,
                attempt_id=attempt_id,
                stage="svg_validation",
            )

        try:
            rendered = self._renderer.render_bytes(
                svg_content,
                output_width=self._render_width,
                output_height=self._render_height,
                background_color=self._background_color,
            )
        except Exception as exc:
            logger.warning("SVG rendering failed before VLM critique: %s", exc)
            return _invalid_feedback(
                f"SVG rendering failed: {type(exc).__name__}: {exc}",
                attempt_id=attempt_id,
                stage="svg_rendering",
            )

        if not isinstance(rendered, (bytes, bytearray)) or not rendered:
            return _invalid_feedback(
                "SVG rendering produced no PNG bytes.",
                attempt_id=attempt_id,
                stage="svg_rendering",
            )

        try:
            labeling = self._labeler.label(svg_content, attempt_id)
        except Exception as exc:
            logger.warning("SVG labeling failed before VLM critique: %s", exc)
            return _invalid_feedback(
                f"SVG labeling failed: {type(exc).__name__}: {exc}",
                attempt_id=attempt_id,
                stage="svg_labeling",
            )

        return self.critique_attempt(
            CriticInput(
                attempt_id=attempt_id,
                instruction=instruction,
                canonical_svg=svg_content,
                render_png=bytes(rendered),
                labeling=labeling,
                render_width=self._render_width,
                render_height=self._render_height,
            )
        )


def _validate_critic_input(value: object) -> str | None:
    if not isinstance(value, CriticInput):
        return "VLM critic input must be CriticInput."
    if not value.attempt_id.strip():
        return "CriticInput.attempt_id must not be empty."
    if not value.instruction.strip():
        return "CriticInput.instruction must not be empty."
    if not value.canonical_svg.strip():
        return "CriticInput.canonical_svg must not be empty."
    if not isinstance(value.render_png, (bytes, bytearray)) or not value.render_png:
        return "CriticInput.render_png must contain PNG bytes."
    if value.render_width <= 0 or value.render_height <= 0:
        return "CriticInput render dimensions must be positive."
    if value.labeling.attempt_id != value.attempt_id:
        return "CriticInput labeling does not match its attempt_id."
    if not value.labeling.labeled_svg.strip():
        return "CriticInput labeling must contain a labeled SVG."
    malformed_ids = sorted(
        target_id
        for target_id in value.labeling.elements
        if _TARGET_ID_RE.fullmatch(target_id) is None
    )
    if malformed_ids:
        return "CriticInput labeling contains malformed target ID(s): " + ", ".join(
            malformed_ids
        )
    return None


def _parse_json_object(raw_response: str) -> dict[str, object]:
    if not isinstance(raw_response, str):
        raise TypeError("VLM critic response text must be a string.")
    text = raw_response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3:
            raise ValueError("VLM critic response contains an incomplete code fence.")
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("VLM critic response must be exactly one valid JSON object.") from exc
    if not isinstance(payload, dict):
        raise ValueError("VLM critic response JSON must be an object.")
    return payload


def _parse_critic_contract(
    payload: dict[str, object],
    *,
    allowed_target_ids: set[str],
) -> tuple[CriticStatus, list[CriticIssue], list[str]]:
    expected_keys = {"status", "issues", "preserve"}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError(
            "VLM critic response must contain exactly status, issues, preserve"
            + (": " + "; ".join(details) if details else "")
        )

    status_value = payload["status"]
    if status_value not in ("pass", "revise"):
        raise ValueError("VLM critic status must be pass or revise.")
    status = cast(CriticStatus, status_value)

    raw_issues = payload["issues"]
    if not isinstance(raw_issues, list):
        raise ValueError("VLM critic issues must be an array.")
    if len(raw_issues) > 3:
        raise ValueError("VLM critic issues may contain at most three entries.")
    issues = [
        _parse_issue(item, allowed_target_ids=allowed_target_ids)
        for item in raw_issues
    ]
    preserve = _require_unique_nonempty_strings(
        payload["preserve"],
        field_name="preserve",
        maximum=3,
    )

    if status == "pass" and issues:
        raise ValueError("status=pass requires issues=[].")
    if status == "pass":
        preserve = []
    if status == "revise" and not issues:
        raise ValueError("status=revise requires at least one issue.")
    return status, issues, preserve


def _parse_issue(
    value: object,
    *,
    allowed_target_ids: set[str],
) -> CriticIssue:
    if not isinstance(value, dict) or set(value) != _ISSUE_KEYS:
        raise ValueError(
            "Each VLM critic issue must contain exactly: "
            + ", ".join(sorted(_ISSUE_KEYS))
        )

    category_value = value["category"]
    if not isinstance(category_value, str) or category_value not in _ISSUE_TAXONOMY:
        raise ValueError("VLM critic issue category is outside the critic_v1 taxonomy.")
    category = cast(CriticCategory, category_value)

    issue_type_value = value["type"]
    if (
        not isinstance(issue_type_value, str)
        or issue_type_value not in _ISSUE_TAXONOMY[category]
    ):
        raise ValueError(
            f"VLM critic issue type is invalid for category '{category}'."
        )

    severity_value = value["severity"]
    if severity_value not in ("critical", "major", "minor"):
        raise ValueError("VLM critic issue severity must be critical, major, or minor.")
    severity = cast(CriticSeverity, severity_value)

    scope_value = value["scope"]
    if scope_value not in ("global", "object", "part"):
        raise ValueError("VLM critic issue scope must be global, object, or part.")
    scope = cast(CriticScope, scope_value)

    target_ids = _require_unique_nonempty_strings(
        value["target_ids"],
        field_name="target_ids",
        maximum=4,
    )
    malformed_ids = sorted(
        target_id for target_id in target_ids if _TARGET_ID_RE.fullmatch(target_id) is None
    )
    if malformed_ids:
        raise ValueError("VLM critic issue contains malformed target ID(s): " + ", ".join(malformed_ids))
    unknown_ids = sorted(set(target_ids) - allowed_target_ids)
    if unknown_ids:
        raise ValueError("VLM critic issue contains unknown target ID(s): " + ", ".join(unknown_ids))
    if (
        not target_ids
        and scope != "global"
        and not (
            category == "content"
            and issue_type_value == "element_presence_or_count"
        )
    ):
        raise ValueError(
            "Empty target_ids is allowed only for global issues or missing objects/parts."
        )

    observed = _require_nonempty_string(value["observed"], "observed")
    expected = _require_nonempty_string(value["expected"], "expected")
    fix = _require_nonempty_string(value["fix"], "fix")
    return CriticIssue(
        category=category,
        type=issue_type_value,
        severity=severity,
        scope=scope,
        target_ids=target_ids,
        observed=observed,
        expected=expected,
        fix=fix,
    )


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"VLM critic field '{field_name}' must be a non-empty string.")
    return value.strip()


def _require_unique_nonempty_strings(
    value: object,
    *,
    field_name: str,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"VLM critic field '{field_name}' must be an array.")
    if len(value) > maximum:
        raise ValueError(
            f"VLM critic field '{field_name}' may contain at most {maximum} entries."
        )
    result = [_require_nonempty_string(item, field_name) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"VLM critic field '{field_name}' must contain unique entries.")
    return result


def _contract_feedback(
    *,
    status: CriticStatus,
    issues: list[CriticIssue],
    preserve: list[str],
    attempt_id: str,
    response: ModelResponse,
    model_calls: list[CriticCallTrace],
    allowed_target_ids: list[str],
) -> CriticFeedback:
    legacy_issues = [_legacy_issue_text(issue) for issue in issues]
    legacy_suggestions = [_legacy_suggestion_text(issue) for issue in issues]
    if status == "pass":
        score = 10.0
        is_valid = True
        matches_instruction = True
    else:
        severity_scores = {"critical": 2.0, "major": 5.0, "minor": 7.0}
        score = min(severity_scores[issue.severity] for issue in issues)
        is_valid = not any(issue.severity == "critical" for issue in issues)
        matches_instruction = False

    return validate_critic_feedback(
        CriticFeedback(
            score=score,
            is_valid=is_valid,
            matches_instruction=matches_instruction,
            issues=legacy_issues,
            suggestions=legacy_suggestions,
            critic_type="vlm",
            raw_response=response.text,
            critic_version=VLM_CRITIC_VERSION,
            model_id=response.model_id,
            model_revision=response.model_revision,
            prompt_version=VLM_CRITIC_PROMPT_VERSION,
            status=status,
            structured_issues=issues,
            preserve=preserve,
            schema_version=2,
            metadata={
                "attempt_id": attempt_id,
                "allowed_target_ids": allowed_target_ids,
                "contract": "critic_v1",
            },
            model_calls=model_calls,
        )
    )


def _invalid_feedback(
    *messages: str,
    attempt_id: str | None,
    stage: str,
    response: ModelResponse | None = None,
    model_calls: list[CriticCallTrace] | None = None,
) -> CriticFeedback:
    details = [message.strip() for message in messages if message and message.strip()]
    if not details:
        details = ["VLM critic could not produce valid feedback."]
    structured = [
        CriticIssue(
            category="validity",
            type=stage,
            severity="critical",
            scope="global",
            target_ids=[],
            observed=message,
            expected="The critic input and response must satisfy the critic_v1 contract.",
            fix="Correct the reported contract or evidence failure before accepting this attempt.",
        )
        for message in details[:3]
    ]
    return validate_critic_feedback(
        CriticFeedback(
            score=0.0,
            is_valid=False,
            matches_instruction=False,
            issues=[f"[invalid/{stage}] {message}" for message in details],
            suggestions=[
                "Correct the critic evidence or response contract and re-run evaluation."
            ],
            critic_type="vlm",
            raw_response=response.text if response is not None else None,
            critic_version=VLM_CRITIC_VERSION,
            model_id=response.model_id if response is not None else None,
            model_revision=response.model_revision if response is not None else None,
            prompt_version=VLM_CRITIC_PROMPT_VERSION,
            status="invalid",
            structured_issues=structured,
            preserve=[],
            schema_version=2,
            metadata={
                "attempt_id": attempt_id,
                "stage": stage,
                "contract": "critic_v1",
            },
            model_calls=list(model_calls or []),
        )
    )


def _legacy_issue_text(issue: CriticIssue) -> str:
    targets = ",".join(issue.target_ids) if issue.target_ids else "global_or_missing"
    return (
        f"[{issue.severity}/{issue.category}/{issue.type}] targets={targets}; "
        f"observed={issue.observed}; expected={issue.expected}"
    )


def _legacy_suggestion_text(issue: CriticIssue) -> str:
    targets = ",".join(issue.target_ids) if issue.target_ids else "global_or_missing"
    return f"[{targets}] {issue.fix}"
