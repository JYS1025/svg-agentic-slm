"""Image-grounded critic for rendered SVG output."""

from __future__ import annotations

import json
import logging
from typing import Protocol

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import CriticFeedback, validate_critic_feedback
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.vlm_critic import (
    VLM_CRITIC_PROMPT_VERSION,
    build_vlm_critic_prompt,
)
from svg_agentic_slm.svg.validator import SVGValidator

logger = logging.getLogger(__name__)

VLM_CRITIC_VERSION = "vlm-critic-v1"


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

    @property
    def name(self) -> str:
        return "VLMCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        """Render and visually evaluate one SVG without exposing unsafe input."""
        try:
            validation = self._validator.validate(svg_content)
        except Exception as exc:
            logger.warning("SVG validation failed before VLM critique: %s", exc)
            return _failure_feedback(f"SVG validation failed: {exc}")

        if not validation.is_valid:
            blockers = validation.errors or ["SVG validation failed."]
            return _failure_feedback(
                *blockers,
                suggestions=["Correct the SVG validation errors before visual review."],
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
            return _failure_feedback(
                f"SVG rendering failed: {exc}",
                suggestions=["Correct the SVG so it can be rendered for visual review."],
            )

        if not isinstance(rendered, (bytes, bytearray)) or not rendered:
            return _failure_feedback(
                "SVG rendering produced no PNG bytes.",
                suggestions=["Correct the SVG so it produces a non-empty PNG render."],
            )

        prompt = build_vlm_critic_prompt(instruction)
        response = self._model.generate_with_image(
            prompt,
            bytes(rendered),
            mime_type="image/png",
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        if not isinstance(response, ModelResponse):
            raise TypeError("Vision model must return ModelResponse.")

        try:
            payload = _parse_json_object(response.text)
            required_fields = {
                "score",
                "is_valid",
                "matches_instruction",
                "blocking_issues",
                "issues",
                "suggestions",
            }
            missing_fields = sorted(required_fields - payload.keys())
            if missing_fields:
                raise ValueError(
                    "VLM critic response is missing required field(s): "
                    + ", ".join(missing_fields)
                )

            blockers = _require_string_list(
                payload["blocking_issues"],
                "blocking_issues",
            )
            issues = _require_string_list(payload["issues"], "issues")
            suggestions = _require_string_list(payload["suggestions"], "suggestions")
            _validate_semantic_consistency(
                payload,
                blockers=blockers,
                issues=issues,
                suggestions=suggestions,
            )
            merged_issues = _unique_strings(
                [*(f"[blocking] {issue}" for issue in blockers), *issues]
            )
            feedback = CriticFeedback(
                score=min(float(payload["score"]), 4.0) if blockers else payload["score"],
                is_valid=payload["is_valid"] if not blockers else False,
                matches_instruction=payload["matches_instruction"],
                issues=merged_issues,
                suggestions=suggestions,
                critic_type="vlm",
                raw_response=response.text,
                critic_version=VLM_CRITIC_VERSION,
                model_id=response.model_id,
                model_revision=response.model_revision,
                prompt_version=VLM_CRITIC_PROMPT_VERSION,
            )
            return validate_critic_feedback(feedback)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Rejected inconsistent VLM critic response: %s", exc)
            return _failure_feedback(
                f"VLM critic response violated its required contract: {exc}",
                suggestions=["Re-evaluate the rendered SVG using the required rubric."],
                response=response,
            )


def _failure_feedback(
    *blocking_issues: str,
    suggestions: list[str] | None = None,
    response: ModelResponse | None = None,
) -> CriticFeedback:
    blockers = [issue for issue in blocking_issues if issue]
    return validate_critic_feedback(
        CriticFeedback(
            score=0.0,
            is_valid=False,
            matches_instruction=False,
            issues=[f"[blocking] {issue}" for issue in blockers],
            suggestions=suggestions or [],
            critic_type="vlm",
            raw_response=response.text if response is not None else None,
            critic_version=VLM_CRITIC_VERSION,
            model_id=response.model_id if response is not None else None,
            model_revision=response.model_revision if response is not None else None,
            prompt_version=VLM_CRITIC_PROMPT_VERSION,
        )
    )


def _parse_json_object(raw_response: str) -> dict[str, object]:
    text = raw_response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start < 0 or object_end < object_start:
        raise ValueError("VLM critic response must contain a JSON object.")
    try:
        payload = json.loads(text[object_start : object_end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("VLM critic response contains invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("VLM critic response JSON must be an object.")
    return payload


def _require_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"VLM critic field '{field_name}' must be an array of strings.")
    return value


def _validate_semantic_consistency(
    payload: dict[str, object],
    *,
    blockers: list[str],
    issues: list[str],
    suggestions: list[str],
) -> None:
    score_value = payload["score"]
    if not isinstance(score_value, (int, float)) or isinstance(score_value, bool):
        raise ValueError("VLM critic score must be a number from 0 to 10.")
    score = float(score_value)
    if not 0.0 <= score <= 10.0:
        raise ValueError("VLM critic score must be a number from 0 to 10.")

    is_valid = payload["is_valid"]
    matches_instruction = payload["matches_instruction"]
    if not isinstance(is_valid, bool) or not isinstance(matches_instruction, bool):
        raise ValueError("VLM critic validity and alignment fields must be booleans.")
    if score < 8.0 and not (blockers or issues):
        raise ValueError("A VLM critic score below 8 requires at least one issue.")
    if score < 8.0 and not suggestions:
        raise ValueError("A VLM critic score below 8 requires an actionable suggestion.")
    if score <= 4.0 and is_valid and matches_instruction:
        raise ValueError("A score of 4 or below cannot be valid and aligned.")
    if score >= 8.0 and (not is_valid or not matches_instruction):
        raise ValueError("A score of 8 or above must be valid and aligned.")


def _unique_strings(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
