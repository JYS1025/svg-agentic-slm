"""Schemas for agent communication.

Defines the data structures passed between agents and the orchestrator.
Using explicit schemas instead of raw dictionaries improves type safety
and makes the data flow self-documenting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.svg.schemas import SVGDiagnostic, SVGLabelingResult

CriticStatus = Literal["pass", "revise", "invalid"]
CriticCategory = Literal["validity", "content", "layout", "shape", "style"]
CriticSeverity = Literal["critical", "major", "minor"]
CriticScope = Literal["global", "object", "part"]

_CRITIC_ISSUE_TYPES: dict[str, set[str]] = {
    "content": {
        "element_presence_or_count",
        "object_identity_or_state",
        "reference_or_instance",
        "text_or_label_content",
    },
    "layout": {
        "viewport_or_clipping",
        "placement_or_transform",
        "relative_scale_alignment_or_spacing",
        "stacking_or_occlusion",
    },
    "shape": {
        "contour_or_curve_geometry",
        "closure_or_part_connectivity",
        "topology_or_fill_region",
    },
    "style": {
        "fill_or_paint_server",
        "stroke_or_marker",
        "visibility_opacity_or_compositing",
        "typography_or_glyph_appearance",
    },
}


@dataclass(frozen=True)
class CriticIssue:
    category: CriticCategory
    type: str
    severity: CriticSeverity
    scope: CriticScope
    target_ids: list[str]
    observed: str
    expected: str
    fix: str


@dataclass(frozen=True)
class CriticInput:
    attempt_id: str
    instruction: str
    canonical_svg: str
    render_png: bytes
    labeling: SVGLabelingResult
    render_width: int = 256
    render_height: int = 256


@dataclass
class CriticEvidence:
    attempt_id: str
    png: bytes
    labeling: SVGLabelingResult
    diagnostics: list[SVGDiagnostic] = field(default_factory=list)
    renderer: str = "cairosvg"
    renderer_version: str | None = None
    width: int = 256
    height: int = 256


@dataclass
class GenerationRequest:
    """A request to generate an SVG from a text instruction.

    Attributes:
        instruction: Natural language description of the desired SVG.
        task: Task identifier (default: 'text_to_svg').
        config_overrides: Optional overrides for generation config.
    """

    instruction: str
    task: str = "text_to_svg"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None


@dataclass
class ModelCallTrace:
    """Trace for one model invocation within a Generator operation."""

    model_call_id: str
    response: ModelResponse
    prompt: str = ""
    system_prompt: str | None = None
    generation_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticCallTrace:
    """Complete trace of one Critic model invocation, including failed retries."""

    critic_call_id: str
    retry_index: int
    response: ModelResponse
    prompt: str
    system_prompt: str | None
    response_format: dict[str, Any]
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    validation_success: bool = False
    validation_error: str | None = None


class CriticTraceError(RuntimeError):
    """Critic failure that retains model-call traces for durable diagnostics."""

    def __init__(self, message: str, model_calls: list[CriticCallTrace]) -> None:
        super().__init__(message)
        self.model_calls = model_calls


@dataclass
class GeneratorOutput:
    """Typed output of one initial generation or revision operation."""

    attempt_id: str
    mode: Literal["initial", "revision"]
    svg: str
    raw_output: str
    status: Literal["succeeded", "failed"]
    prompt_version: str
    model_calls: list[ModelCallTrace] = field(default_factory=list)
    parent_attempt_id: str | None = None
    trigger_feedback_id: str | None = None
    error: str | None = None
    context_item_ids: list[str] = field(default_factory=list)
    truncated_context_item_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    critic_evidence: CriticEvidence | None = None
    critic_error_calls: list[CriticCallTrace] = field(default_factory=list)


@dataclass
class GenerationResult:
    """The result of the SVG generation pipeline.

    Attributes:
        instruction: The original instruction.
        generated_svg: The generated SVG string.
        is_valid: Whether the SVG passed validation.
        render_path: Path to the rendered image, if rendering was performed.
        critic_feedback: Feedback from the critic, if critic was used.
        revision_count: Number of revision rounds performed.
        metadata: Additional metadata about the generation process.
    """

    instruction: str
    generated_svg: str = ""
    is_valid: bool = False
    render_path: str | None = None
    critic_feedback: list[CriticFeedback] = field(default_factory=list)
    revision_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    attempts: list[GeneratorOutput] = field(default_factory=list)
    feedback_events: list[CriticFeedbackEvent] = field(default_factory=list)


@dataclass
class CriticFeedback:
    """Structured feedback from a critic agent.

    Attributes:
        score: Overall quality score (1-10).
        is_valid: Whether the critic considers the SVG valid.
        matches_instruction: Whether the SVG matches the instruction.
        issues: List of identified issues.
        suggestions: List of improvement suggestions.
        critic_type: Type of critic that produced this feedback.
        raw_response: Raw critic response, if from an LLM critic.
    """

    score: float = 0.0
    is_valid: bool = False
    matches_instruction: bool = False
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    critic_type: str = "unknown"
    raw_response: str | None = None
    critic_version: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    status: CriticStatus | None = None
    structured_issues: list[CriticIssue] = field(default_factory=list)
    preserve: list[str] = field(default_factory=list)
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    model_calls: list[CriticCallTrace] = field(default_factory=list)


@dataclass
class CriticFeedbackEvent:
    """A Critic payload correlated to the exact SVG attempt it reviewed."""

    feedback_id: str
    target_attempt_id: str
    feedback: CriticFeedback


def validate_critic_feedback(value: object) -> CriticFeedback:
    """Validate a Critic response at every runtime composition boundary."""
    if not isinstance(value, CriticFeedback):
        raise TypeError("Critic must return CriticFeedback.")
    if (
        not isinstance(value.score, (int, float))
        or isinstance(value.score, bool)
        or not math.isfinite(float(value.score))
        or not 0.0 <= float(value.score) <= 10.0
    ):
        raise ValueError("CriticFeedback.score must be a finite number between 0 and 10.")
    if not isinstance(value.is_valid, bool):
        raise TypeError("CriticFeedback.is_valid must be a boolean.")
    if not isinstance(value.matches_instruction, bool):
        raise TypeError("CriticFeedback.matches_instruction must be a boolean.")
    _validate_string_list(value.issues, "CriticFeedback.issues")
    _validate_string_list(value.suggestions, "CriticFeedback.suggestions")
    _validate_string_list(value.preserve, "CriticFeedback.preserve")
    if value.status not in (None, "pass", "revise", "invalid"):
        raise ValueError("CriticFeedback.status must be pass, revise, invalid, or None.")
    if not isinstance(value.schema_version, int) or value.schema_version < 1:
        raise ValueError("CriticFeedback.schema_version must be a positive integer.")
    if value.schema_version >= 2 and value.status is None:
        raise ValueError("CriticFeedback.status is required for schema version 2 or newer.")
    if value.status is not None and value.schema_version < 2:
        raise ValueError("Structured CriticFeedback.status requires schema version 2 or newer.")
    if len(value.structured_issues) > 3:
        raise ValueError("CriticFeedback.structured_issues may contain at most 3 issues.")
    issue_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in value.structured_issues:
        _validate_critic_issue(item, allow_validity=value.status == "invalid")
        issue_key = (item.category, item.type, tuple(item.target_ids))
        if issue_key in issue_keys:
            raise ValueError("CriticFeedback.structured_issues contains a duplicate issue.")
        issue_keys.add(issue_key)
    if len(value.preserve) > 3 or len(set(value.preserve)) != len(value.preserve):
        raise ValueError("CriticFeedback.preserve must contain at most 3 unique entries.")
    if any(not item.strip() for item in value.preserve):
        raise ValueError("CriticFeedback.preserve entries must be non-empty.")
    if value.status == "pass":
        if value.structured_issues or value.preserve:
            raise ValueError("pass feedback cannot contain issues or preserve entries.")
        if not value.is_valid or not value.matches_instruction:
            raise ValueError("Only pass feedback may represent an accepted Critic result.")
    elif value.status == "revise":
        if not value.structured_issues:
            raise ValueError("revise feedback must contain at least one structured issue.")
        if value.matches_instruction:
            raise ValueError("revise feedback cannot be marked as instruction-matching.")
    elif value.status == "invalid":
        if not value.structured_issues:
            raise ValueError("invalid feedback must contain at least one structured issue.")
        if value.is_valid or value.matches_instruction:
            raise ValueError("invalid feedback cannot be valid or instruction-matching.")
    elif value.structured_issues or value.preserve or value.model_calls:
        raise ValueError("Structured Critic fields require schema version 2 feedback.")
    if not isinstance(value.metadata, dict):
        raise TypeError("CriticFeedback.metadata must be a dictionary.")
    call_ids: set[str] = set()
    for call in value.model_calls:
        if not isinstance(call, CriticCallTrace):
            raise TypeError("CriticFeedback.model_calls must contain CriticCallTrace values.")
        if not isinstance(call.critic_call_id, str) or not call.critic_call_id.strip():
            raise ValueError("CriticCallTrace.critic_call_id must be non-empty.")
        if call.critic_call_id in call_ids:
            raise ValueError("CriticFeedback.model_calls contains a duplicate call ID.")
        call_ids.add(call.critic_call_id)
        if not isinstance(call.retry_index, int) or isinstance(call.retry_index, bool) or call.retry_index < 0:
            raise ValueError("CriticCallTrace.retry_index must be a non-negative integer.")
        if not isinstance(call.response, ModelResponse):
            raise TypeError("CriticCallTrace.response must be ModelResponse.")
        if not isinstance(call.prompt, str) or not call.prompt.strip():
            raise ValueError("CriticCallTrace.prompt must be non-empty.")
        if call.system_prompt is not None and not isinstance(call.system_prompt, str):
            raise TypeError("CriticCallTrace.system_prompt must be a string or None.")
        if not isinstance(call.response_format, dict):
            raise TypeError("CriticCallTrace.response_format must be a dictionary.")
        if not isinstance(call.generation_parameters, dict):
            raise TypeError("CriticCallTrace.generation_parameters must be a dictionary.")
        if not isinstance(call.validation_success, bool):
            raise TypeError("CriticCallTrace.validation_success must be a boolean.")
        if call.validation_error is not None and not isinstance(call.validation_error, str):
            raise TypeError("CriticCallTrace.validation_error must be a string or None.")
    if not isinstance(value.critic_type, str) or not value.critic_type.strip():
        raise TypeError("CriticFeedback.critic_type must be a non-empty string.")
    for field_name in (
        "raw_response",
        "critic_version",
        "model_id",
        "model_revision",
        "prompt_version",
    ):
        field_value = getattr(value, field_name)
        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(f"CriticFeedback.{field_name} must be a string or None.")
    return value


def _validate_string_list(value: object, field_name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings.")


def _validate_critic_issue(value: object, *, allow_validity: bool) -> None:
    if not isinstance(value, CriticIssue):
        raise TypeError("CriticFeedback.structured_issues must contain CriticIssue values.")
    if value.category == "validity":
        if not allow_validity:
            raise ValueError("validity issues are reserved for invalid Critic feedback.")
        if value.severity != "critical" or value.scope != "global" or value.target_ids:
            raise ValueError("validity issues must be critical, global, and have no targets.")
    else:
        allowed_types = _CRITIC_ISSUE_TYPES.get(value.category)
        if allowed_types is None or value.type not in allowed_types:
            raise ValueError("CriticIssue.type is not valid for its category.")
    if value.severity not in {"critical", "major", "minor"}:
        raise ValueError("CriticIssue.severity is invalid.")
    if value.scope not in {"global", "object", "part"}:
        raise ValueError("CriticIssue.scope is invalid.")
    if (
        not isinstance(value.target_ids, list)
        or len(value.target_ids) > 4
        or len(set(value.target_ids)) != len(value.target_ids)
        or any(
            not isinstance(target_id, str)
            or len(target_id) != 5
            or target_id[0] not in "sged"
            or not target_id[1:].isdigit()
            for target_id in value.target_ids
        )
    ):
        raise ValueError("CriticIssue.target_ids contains invalid or duplicate IDs.")
    missing_object = (
        value.category == "content" and value.type == "element_presence_or_count"
    )
    if not value.target_ids and not missing_object and value.scope != "global" and value.category != "validity":
        raise ValueError("Empty target_ids require a missing object or global issue.")
    for field_name in ("type", "observed", "expected", "fix"):
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"CriticIssue.{field_name} must be non-empty.")
