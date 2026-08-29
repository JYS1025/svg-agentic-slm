"""Utilities for loading generated SVG artifact bundles."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCORECARD_TYPES = {
    "semantic": {"presence", "count", "identity", "state", "text_content"},
    "geometry": {"contour", "proportion", "topology"},
    "layout": {"placement", "scale", "orientation", "spacing", "occlusion", "framing"},
    "appearance": {"color", "surface", "stroke", "typography"},
}
_SCORECARD_PAIRS = {
    (category, issue_type)
    for category, issue_types in _SCORECARD_TYPES.items()
    for issue_type in issue_types
}


@dataclass
class ModelCallArtifactRecord:
    """Typed model-call trace stored under a generation attempt."""

    model_call_id: str
    raw_output_path: Path | None
    prompt_path: Path | None
    system_prompt_path: Path | None
    generation_parameters: dict[str, Any]
    model_id: str | None
    model_revision: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float | None
    metadata: dict[str, Any]


@dataclass
class GenerationAttemptRecord:
    """Typed attempt lineage and persisted trace references."""

    attempt_id: str
    mode: str
    parent_attempt_id: str | None
    trigger_feedback_id: str | None
    svg_path: Path | None
    raw_output_path: Path | None
    status: str
    error: str | None
    outcome: str | None
    stop_reason: str | None
    prompt_version: str | None
    context_item_ids: list[str]
    truncated_context_item_ids: list[str]
    model_calls: list[ModelCallArtifactRecord]
    metadata: dict[str, Any]


@dataclass
class GenerationArtifactRecord:
    """Typed view over a generated SVG artifact bundle."""

    instruction: str
    svg_path: Path
    metadata_path: Path
    render_path: Path | None
    is_valid: bool
    revision_count: int
    critic_feedback: list[dict[str, Any]]
    runtime: dict[str, Any]
    metadata: dict[str, Any]
    generated_at_utc: str | None
    schema_version: int = 0
    run_id: str | None = None
    outcome: str | None = None
    stop_reason: str | None = None
    attempts: list[GenerationAttemptRecord] = field(default_factory=list)


def load_generation_artifact(path: str | Path) -> GenerationArtifactRecord:
    """Load a generated artifact record from an SVG or metadata path."""
    path = Path(path)
    metadata_path = _resolve_metadata_path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return parse_generation_artifact_payload(payload, metadata_path)


def parse_generation_artifact_payload(
    payload: object,
    metadata_path: str | Path,
) -> GenerationArtifactRecord:
    """Validate and parse an artifact payload relative to its intended sidecar."""
    metadata_path = Path(metadata_path)
    if not isinstance(payload, dict):
        raise ValueError("Generation artifact metadata root must be an object.")
    schema_version = _schema_version(payload.get("schema_version", 0))
    strict_references = schema_version >= 1
    if strict_references:
        _validate_v1_payload(
            payload,
            allow_v2_feedback=schema_version >= 2,
            allow_v2_outcomes=schema_version >= 2,
        )
    if schema_version >= 2:
        _validate_v2_payload(payload)

    svg_path = _resolve_artifact_reference(
        payload.get("svg_path"),
        metadata_path=metadata_path,
        fallback=metadata_path.with_suffix(".svg"),
        strict=strict_references,
    )
    render_path_value = payload.get("render_path")
    render_path = (
        _resolve_artifact_reference(
            render_path_value,
            metadata_path=metadata_path,
            strict=strict_references,
        )
        if render_path_value
        else None
    )
    metadata = _as_dict(payload.get("metadata"))
    attempts = _load_attempts(
        metadata,
        metadata_path,
        strict_references=strict_references,
        allow_v2_outcomes=schema_version >= 2,
    )
    if strict_references:
        _validate_v1_consistency(payload, attempts, svg_path)
    outcome = _optional_string(payload.get("outcome"))
    stop_reason = _optional_string(payload.get("stop_reason"))
    if attempts:
        outcome = outcome or attempts[-1].outcome
        stop_reason = stop_reason or attempts[-1].stop_reason

    return GenerationArtifactRecord(
        instruction=payload.get("instruction", ""),
        svg_path=svg_path,
        metadata_path=metadata_path,
        render_path=render_path,
        is_valid=payload.get("is_valid", False),
        revision_count=payload.get("revision_count", 0),
        critic_feedback=payload.get("critic_feedback", []),
        runtime=_as_dict(payload.get("runtime")),
        metadata=metadata,
        generated_at_utc=payload.get("generated_at_utc"),
        schema_version=schema_version,
        run_id=payload.get("run_id"),
        outcome=outcome,
        stop_reason=stop_reason,
        attempts=attempts,
    )


def list_generation_artifacts(directory: str | Path) -> list[GenerationArtifactRecord]:
    """Load all generation artifacts described by JSON sidecars in a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Artifact directory not found: {directory}")

    records = [
        load_generation_artifact(metadata_path)
        for metadata_path in sorted(directory.glob("*.json"))
    ]
    return records


def _resolve_metadata_path(path: Path) -> Path:
    if path.suffix == ".json":
        metadata_path = path
    else:
        metadata_path = path.with_suffix(".json")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Artifact metadata file not found: {metadata_path}")

    return metadata_path


def _resolve_artifact_reference(
    value: object,
    *,
    metadata_path: Path,
    fallback: Path | None = None,
    strict: bool = False,
) -> Path:
    """Resolve new sidecar-relative paths while retaining legacy cwd paths."""
    if value is None:
        if strict:
            raise ValueError("Schema v1 artifact path is missing.")
        if fallback is None:
            raise ValueError("Artifact path is missing from generation metadata.")
        return fallback
    if not isinstance(value, str):
        raise ValueError("Artifact paths in generation metadata must be strings.")

    path = Path(value)
    if path.is_absolute():
        if strict:
            raise ValueError("Schema v1 artifact paths must be sidecar-relative.")
        return path

    sidecar_relative = metadata_path.parent / path
    if strict:
        resolved_parent = metadata_path.parent.resolve()
        resolved_path = sidecar_relative.resolve()
        if not resolved_path.is_relative_to(resolved_parent):
            raise ValueError("Schema v1 artifact path escapes the sidecar directory.")
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Artifact bundle file not found: {resolved_path}")
        return resolved_path
    if sidecar_relative.exists():
        return sidecar_relative
    if path.exists():
        return path
    return sidecar_relative


def _load_attempts(
    metadata: dict[str, Any],
    metadata_path: Path,
    *,
    strict_references: bool,
    allow_v2_outcomes: bool = False,
) -> list[GenerationAttemptRecord]:
    generator_metadata = _as_dict(metadata.get("generator"))
    raw_attempts = generator_metadata.get("attempts", [])
    if not isinstance(raw_attempts, list):
        if strict_references:
            raise ValueError("metadata.generator.attempts must be a list.")
        return []

    attempts: list[GenerationAttemptRecord] = []
    for attempt_index, raw_attempt in enumerate(raw_attempts):
        if not isinstance(raw_attempt, dict):
            if strict_references:
                raise ValueError(
                    f"metadata.generator.attempts[{attempt_index}] must be an object."
                )
            continue
        field_prefix = f"metadata.generator.attempts[{attempt_index}]"
        if strict_references:
            _validate_v1_attempt(
                raw_attempt,
                field_prefix,
                allow_v2_outcomes=allow_v2_outcomes,
            )
        model_calls = _load_model_calls(
            raw_attempt.get("model_calls"),
            metadata_path,
            strict_references=strict_references,
            field_prefix=f"{field_prefix}.model_calls",
        )
        attempts.append(
            GenerationAttemptRecord(
                attempt_id=str(raw_attempt.get("attempt_id", "")),
                mode=str(raw_attempt.get("mode", "")),
                parent_attempt_id=_optional_string(raw_attempt.get("parent_attempt_id")),
                trigger_feedback_id=_optional_string(
                    raw_attempt.get("trigger_feedback_id")
                ),
                svg_path=_resolve_optional_reference(
                    raw_attempt.get("svg_ref"),
                    metadata_path,
                    strict=strict_references,
                ),
                raw_output_path=_resolve_optional_reference(
                    raw_attempt.get("raw_output_ref"),
                    metadata_path,
                    strict=strict_references,
                ),
                status=str(raw_attempt.get("status", "")),
                error=_optional_string(raw_attempt.get("error")),
                outcome=_optional_string(raw_attempt.get("outcome")),
                stop_reason=_optional_string(raw_attempt.get("stop_reason")),
                prompt_version=_optional_string(raw_attempt.get("prompt_version")),
                context_item_ids=_string_list(raw_attempt.get("context_item_ids")),
                truncated_context_item_ids=_string_list(
                    raw_attempt.get("truncated_context_item_ids")
                ),
                model_calls=model_calls,
                metadata=_as_dict(raw_attempt.get("metadata")),
            )
        )
    return attempts


def _load_model_calls(
    raw_model_calls: object,
    metadata_path: Path,
    *,
    strict_references: bool,
    field_prefix: str,
) -> list[ModelCallArtifactRecord]:
    if not isinstance(raw_model_calls, list):
        if strict_references:
            raise ValueError(f"{field_prefix} must be a list.")
        return []

    records: list[ModelCallArtifactRecord] = []
    for call_index, raw_call in enumerate(raw_model_calls):
        if not isinstance(raw_call, dict):
            if strict_references:
                raise ValueError(f"{field_prefix}[{call_index}] must be an object.")
            continue
        if strict_references:
            _validate_v1_model_call(raw_call, f"{field_prefix}[{call_index}]")
        records.append(
            ModelCallArtifactRecord(
                model_call_id=str(raw_call.get("model_call_id", "")),
                raw_output_path=_resolve_optional_reference(
                    raw_call.get("raw_output_ref"),
                    metadata_path,
                    strict=strict_references,
                ),
                prompt_path=_resolve_optional_reference(
                    raw_call.get("prompt_ref"),
                    metadata_path,
                    strict=strict_references,
                ),
                system_prompt_path=_resolve_optional_reference(
                    raw_call.get("system_prompt_ref"),
                    metadata_path,
                    strict=strict_references,
                ),
                generation_parameters=_as_dict(
                    raw_call.get("generation_parameters")
                ),
                model_id=_optional_string(raw_call.get("model_id")),
                model_revision=_optional_string(raw_call.get("model_revision")),
                finish_reason=_optional_string(raw_call.get("finish_reason")),
                prompt_tokens=_optional_int(raw_call.get("prompt_tokens")),
                completion_tokens=_optional_int(raw_call.get("completion_tokens")),
                latency_seconds=_optional_float(raw_call.get("latency_seconds")),
                metadata=_as_dict(raw_call.get("metadata")),
            )
        )
    return records


def _resolve_optional_reference(
    value: object,
    metadata_path: Path,
    *,
    strict: bool,
) -> Path | None:
    if value is None:
        return None
    return _resolve_artifact_reference(
        value,
        metadata_path=metadata_path,
        strict=strict,
    )


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _schema_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("schema_version must be a non-negative integer.")
    if value > 3:
        raise ValueError(f"Unsupported generation artifact schema_version: {value}")
    return value


def _validate_v1_payload(
    payload: dict[str, Any],
    *,
    allow_v2_feedback: bool = False,
    allow_v2_outcomes: bool = False,
) -> None:
    _require_string(payload.get("run_id"), "run_id", non_empty=True)
    _require_string(payload.get("instruction"), "instruction", non_empty=True)
    _require_string(payload.get("svg_path"), "svg_path", non_empty=True)
    _require_bool(payload.get("is_valid"), "is_valid")
    _require_nonnegative_int(payload.get("revision_count"), "revision_count")
    _require_optional_string(payload.get("render_path"), "render_path")
    outcomes = {"accepted", "rejected", "failed"}
    if allow_v2_outcomes:
        outcomes.update({"selected_best", "rolled_back", "critic_contract_failure"})
    _require_choice(
        payload.get("outcome"),
        "outcome",
        outcomes,
    )
    _require_string(payload.get("stop_reason"), "stop_reason", non_empty=True)
    _require_string(
        payload.get("generated_at_utc"),
        "generated_at_utc",
        non_empty=True,
    )
    _require_mapping(payload.get("runtime"), "runtime")
    metadata = _require_mapping(payload.get("metadata"), "metadata")
    generator = _require_mapping(metadata.get("generator"), "metadata.generator")
    attempts = generator.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("metadata.generator.attempts must be a list.")
    if not attempts:
        raise ValueError("metadata.generator.attempts must not be empty.")

    feedback_items = payload.get("critic_feedback")
    if not isinstance(feedback_items, list):
        raise ValueError("critic_feedback must be a list.")
    for index, feedback in enumerate(feedback_items):
        prefix = f"critic_feedback[{index}]"
        if not isinstance(feedback, dict):
            raise ValueError(f"{prefix} must be an object.")
        _require_score(feedback.get("score"), f"{prefix}.score")
        _require_bool(feedback.get("is_valid"), f"{prefix}.is_valid")
        _require_bool(
            feedback.get("matches_instruction"),
            f"{prefix}.matches_instruction",
        )
        _require_string(feedback.get("critic_type"), f"{prefix}.critic_type")
        critic_schema_version = feedback.get("critic_schema_version", 1)
        if allow_v2_feedback and critic_schema_version >= 3:
            _validate_v3_feedback(feedback, prefix)
        elif allow_v2_feedback and critic_schema_version >= 2:
            _validate_v2_feedback(feedback, prefix)
        else:
            _require_string_list(feedback.get("issues"), f"{prefix}.issues")
        _require_string_list(feedback.get("suggestions"), f"{prefix}.suggestions")
        _require_string(
            feedback.get("feedback_id"),
            f"{prefix}.feedback_id",
            non_empty=True,
        )
        _require_string(
            feedback.get("target_attempt_id"),
            f"{prefix}.target_attempt_id",
            non_empty=True,
        )


def _validate_v2_payload(payload: dict[str, Any]) -> None:
    """Validate structured Critic evidence and cross-record correlation."""
    metadata = _require_mapping(payload.get("metadata"), "metadata")
    generator = _require_mapping(metadata.get("generator"), "metadata.generator")
    attempts_value = generator.get("attempts")
    if not isinstance(attempts_value, list) or not attempts_value:
        raise ValueError("metadata.generator.attempts must be a non-empty list.")

    attempts: dict[str, dict[str, Any]] = {}
    call_ids_by_attempt: dict[str, set[str]] = {}
    for index, attempt_value in enumerate(attempts_value):
        prefix = f"metadata.generator.attempts[{index}]"
        attempt = _require_mapping(attempt_value, prefix)
        attempt_id = attempt.get("attempt_id")
        _require_string(attempt_id, f"{prefix}.attempt_id", non_empty=True)
        if attempt_id in attempts:
            raise ValueError("attempt_id values must be unique within one run.")
        attempts[attempt_id] = attempt
        call_ids_by_attempt[attempt_id] = _validate_v2_attempt(attempt, prefix)

    feedback_items = payload.get("critic_feedback")
    if not isinstance(feedback_items, list):
        raise ValueError("critic_feedback must be a list.")
    feedback_ids: set[str] = set()
    latest_structured_feedback: dict[str, Any] | None = None
    for index, feedback_value in enumerate(feedback_items):
        prefix = f"critic_feedback[{index}]"
        feedback = _require_mapping(feedback_value, prefix)
        feedback_id = feedback.get("feedback_id")
        target_attempt_id = feedback.get("target_attempt_id")
        if feedback_id in feedback_ids:
            raise ValueError("critic_feedback contains a duplicate feedback_id.")
        feedback_ids.add(feedback_id)
        if target_attempt_id not in attempts:
            raise ValueError(f"{prefix}.target_attempt_id does not identify an attempt.")
        critic_schema_version = feedback.get("critic_schema_version", 1)
        if not isinstance(critic_schema_version, int) or isinstance(critic_schema_version, bool):
            raise ValueError(f"{prefix}.critic_schema_version must be an integer.")
        if critic_schema_version < 1 or critic_schema_version > 3:
            raise ValueError(f"{prefix}.critic_schema_version is unsupported.")
        if critic_schema_version < 2:
            structured_fields = (
                "status",
                "legacy_issues",
                "preserve",
                "model_call_ids",
            )
            if any(key in feedback for key in structured_fields):
                raise ValueError(f"{prefix} uses critic_v1 fields with a legacy critic schema.")
            continue

        latest_structured_feedback = feedback
        status = feedback.get("status")
        attempt = attempts[target_attempt_id]
        evidence = attempt.get("critic_evidence")
        if status in {"pass", "revise"} and not isinstance(evidence, dict):
            raise ValueError(f"{prefix} requires attempt-correlated critic evidence.")

        model_call_ids = feedback.get("model_call_ids")
        if not isinstance(model_call_ids, list) or any(
            not isinstance(item, str) or not item for item in model_call_ids
        ):
            raise ValueError(f"{prefix}.model_call_ids must be a list of non-empty strings.")
        if len(set(model_call_ids)) != len(model_call_ids):
            raise ValueError(f"{prefix}.model_call_ids contains duplicates.")
        if status in {"pass", "revise"} and not model_call_ids:
            raise ValueError(f"{prefix} requires at least one successful Critic model call.")
        unknown_calls = set(model_call_ids) - call_ids_by_attempt[target_attempt_id]
        if unknown_calls:
            raise ValueError(f"{prefix}.model_call_ids references an unknown Critic call.")

        allowed_target_ids = _feedback_evidence_target_ids(feedback, prefix, target_attempt_id)
        for issue in feedback.get("issues", []):
            target_ids = set(issue.get("target_ids", []))
            if target_ids and not allowed_target_ids:
                raise ValueError(f"{prefix} has grounded targets without evidence target IDs.")
            if not target_ids.issubset(allowed_target_ids):
                raise ValueError(f"{prefix} contains a target ID absent from Critic evidence.")

    if payload.get("outcome") == "accepted" and latest_structured_feedback is not None:
        final_attempt_id = _selected_attempt_id(payload) or attempts_value[-1].get(
            "attempt_id"
        )
        if (
            latest_structured_feedback.get("target_attempt_id") != final_attempt_id
            or latest_structured_feedback.get("status") != "pass"
        ):
            raise ValueError(
                "An accepted structured artifact requires pass feedback for the final "
                "attempt."
            )


def _validate_v3_feedback(feedback: dict[str, Any], prefix: str) -> None:
    if feedback.get("critic_schema_version") != 3:
        raise ValueError(f"{prefix}.critic_schema_version must be 3.")
    status = feedback.get("status")
    _require_choice(status, f"{prefix}.status", {"pass", "revise", "invalid"})
    issues = feedback.get("issues")
    if not isinstance(issues, list) or len(issues) > 3:
        raise ValueError(f"{prefix}.issues must be an array of at most 3 typed issues.")
    issue_pairs: set[tuple[str, str]] = set()
    for index, issue_value in enumerate(issues):
        issue = _require_mapping(issue_value, f"{prefix}.issues[{index}]")
        issue_pair = _validate_v3_issue(issue, f"{prefix}.issues[{index}]", status)
        issue_pairs.add(issue_pair)

    _require_string_list(feedback.get("legacy_issues"), f"{prefix}.legacy_issues")
    metadata = _require_mapping(feedback.get("metadata"), f"{prefix}.metadata")
    model_call_ids = feedback.get("model_call_ids")
    if not isinstance(model_call_ids, list) or any(
        not isinstance(item, str) or not item for item in model_call_ids
    ):
        raise ValueError(f"{prefix}.model_call_ids must contain non-empty strings.")

    evaluations = feedback.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError(f"{prefix}.evaluations must be an array.")
    if status == "invalid":
        if evaluations:
            raise ValueError(f"{prefix}: invalid feedback cannot contain evaluations.")
        if not issues:
            raise ValueError(f"{prefix}: invalid feedback requires at least one issue.")
        if (
            feedback.get("score") != 0.0
            or feedback.get("is_valid") is not False
            or feedback.get("matches_instruction") is not False
        ):
            raise ValueError(f"{prefix}: invalid feedback requires score=0 and false flags.")
        return

    if len(evaluations) != len(_SCORECARD_PAIRS):
        raise ValueError(f"{prefix}.evaluations must contain all 18 category-type pairs.")
    evaluation_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for index, evaluation_value in enumerate(evaluations):
        evaluation = _require_mapping(
            evaluation_value,
            f"{prefix}.evaluations[{index}]",
        )
        pair = _validate_v3_evaluation(
            evaluation,
            f"{prefix}.evaluations[{index}]",
        )
        if pair in evaluation_by_pair:
            raise ValueError(f"{prefix}.evaluations contains a duplicate pair.")
        evaluation_by_pair[pair] = evaluation
    if set(evaluation_by_pair) != _SCORECARD_PAIRS:
        raise ValueError(f"{prefix}.evaluations must cover every scorecard pair exactly once.")

    threshold = metadata.get("score_threshold", 3.0)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 4.0
    ):
        raise ValueError(f"{prefix}.metadata.score_threshold must be between 0 and 4.")
    applicable = [item for item in evaluations if item.get("applicable") is True]
    if not applicable:
        raise ValueError(f"{prefix}.evaluations requires at least one applicable entry.")
    scores = [item.get("score") for item in applicable]
    if any(not isinstance(score, int) or isinstance(score, bool) for score in scores):
        raise ValueError(f"{prefix}.applicable evaluations require integer scores.")
    numeric_scores = [int(score) for score in scores]
    if float(feedback.get("score")) != float(min(numeric_scores)):
        raise ValueError(f"{prefix}.score must equal the minimum applicable score.")
    below_threshold = {
        pair
        for pair, evaluation in evaluation_by_pair.items()
        if evaluation.get("applicable") is True
        and int(evaluation["score"]) < float(threshold)
    }
    accepted = not below_threshold
    if status != ("pass" if accepted else "revise"):
        raise ValueError(f"{prefix}.status is inconsistent with score threshold.")
    if feedback.get("is_valid") is not True:
        raise ValueError(f"{prefix}: scorecard feedback requires is_valid=true.")
    if feedback.get("matches_instruction") is not accepted:
        raise ValueError(f"{prefix}.matches_instruction is inconsistent with its scores.")
    if accepted and issues:
        raise ValueError(f"{prefix}: passing scorecard feedback cannot contain issues.")
    if not accepted and not issues:
        raise ValueError(f"{prefix}: revising scorecard feedback requires issues.")
    if any(pair not in below_threshold for pair in issue_pairs):
        raise ValueError(f"{prefix}: issues must reference evaluations below threshold.")


def _validate_v3_evaluation(
    evaluation: dict[str, Any],
    prefix: str,
) -> tuple[str, str]:
    required = {"category", "type", "applicable", "score", "reason"}
    if set(evaluation) != required:
        raise ValueError(f"{prefix} must contain exactly the scorecard evaluation fields.")
    category = evaluation.get("category")
    issue_type = evaluation.get("type")
    if category not in _SCORECARD_TYPES or issue_type not in _SCORECARD_TYPES[category]:
        raise ValueError(f"{prefix}.type is invalid for its category.")
    applicable = evaluation.get("applicable")
    _require_bool(applicable, f"{prefix}.applicable")
    score = evaluation.get("score")
    if applicable:
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
            raise ValueError(f"{prefix}.score must be an integer from 0 to 4.")
    elif score is not None:
        raise ValueError(f"{prefix}.score must be null when not applicable.")
    _require_string(evaluation.get("reason"), f"{prefix}.reason", non_empty=True)
    return str(category), str(issue_type)


def _validate_v3_issue(
    issue: dict[str, Any],
    prefix: str,
    status: str,
) -> tuple[str, str]:
    required = {
        "category", "type", "scope", "target_ids", "observed", "expected", "fix",
    }
    if set(issue) != required:
        raise ValueError(f"{prefix} must contain exactly the scorecard issue fields.")
    category = issue.get("category")
    issue_type = issue.get("type")
    if category == "validity":
        if status != "invalid":
            raise ValueError(f"{prefix}: validity category is reserved for invalid feedback.")
    elif category not in _SCORECARD_TYPES or issue_type not in _SCORECARD_TYPES[category]:
        raise ValueError(f"{prefix}.type is invalid for its category.")
    _require_choice(issue.get("scope"), f"{prefix}.scope", {"global", "object", "part"})
    targets = issue.get("target_ids")
    if not isinstance(targets, list) or len(targets) > 4 or len(set(targets)) != len(targets):
        raise ValueError(f"{prefix}.target_ids is invalid.")
    if any(
        not isinstance(item, str)
        or len(item) != 5
        or item[0] not in "sged"
        or not item[1:].isdigit()
        for item in targets
    ):
        raise ValueError(f"{prefix}.target_ids contains an invalid ID.")
    missing_content = (
        category == "semantic"
        and issue_type in {"presence", "text_content"}
    )
    if (
        not targets
        and not missing_content
        and issue.get("scope") != "global"
        and category != "validity"
    ):
        raise ValueError(
            f"{prefix}: empty targets require missing visible content or a global issue."
        )
    if category == "validity" and (issue.get("scope") != "global" or targets):
        raise ValueError(f"{prefix}: validity issues must be global with no targets.")
    for key in ("type", "observed", "expected", "fix"):
        _require_string(issue.get(key), f"{prefix}.{key}", non_empty=True)
    return str(category), str(issue_type)


def _validate_v2_feedback(feedback: dict[str, Any], prefix: str) -> None:
    critic_schema_version = feedback.get("critic_schema_version")
    if (
        not isinstance(critic_schema_version, int)
        or isinstance(critic_schema_version, bool)
        or critic_schema_version < 2
    ):
        raise ValueError(f"{prefix}.critic_schema_version must be at least 2.")
    status = feedback.get("status")
    _require_choice(status, f"{prefix}.status", {"pass", "revise", "invalid"})
    issues = feedback.get("issues")
    if not isinstance(issues, list) or len(issues) > 3:
        raise ValueError(f"{prefix}.issues must be an array of at most 3 typed issues.")
    issue_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for index, issue_value in enumerate(issues):
        issue = _require_mapping(issue_value, f"{prefix}.issues[{index}]")
        issue_key = _validate_v2_issue(issue, f"{prefix}.issues[{index}]", status)
        if issue_key in issue_keys:
            raise ValueError(f"{prefix}.issues contains a duplicate typed issue.")
        issue_keys.add(issue_key)
    _require_string_list(feedback.get("legacy_issues"), f"{prefix}.legacy_issues")
    preserve = feedback.get("preserve")
    _require_string_list(preserve, f"{prefix}.preserve")
    if (
        len(preserve) > 3
        or len(set(preserve)) != len(preserve)
        or any(not item.strip() for item in preserve)
    ):
        raise ValueError(f"{prefix}.preserve must contain at most 3 unique non-empty strings.")
    metadata = feedback.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{prefix}.metadata must be an object.")
    if status == "pass":
        if issues or preserve:
            raise ValueError(f"{prefix}: pass requires empty issues and preserve.")
        if feedback.get("is_valid") is not True or feedback.get("matches_instruction") is not True:
            raise ValueError(f"{prefix}: only pass may represent acceptance.")
    elif status == "revise":
        if not issues or feedback.get("matches_instruction") is not False:
            raise ValueError(f"{prefix}: revise requires issues and cannot be accepted.")
    elif (
        not issues
        or feedback.get("is_valid") is not False
        or feedback.get("matches_instruction") is not False
    ):
        raise ValueError(f"{prefix}: invalid requires issues and false acceptance booleans.")


def _validate_v2_issue(
    issue: dict[str, Any],
    prefix: str,
    status: str,
) -> tuple[str, str, tuple[str, ...]]:
    required = {
        "category", "type", "severity", "scope", "target_ids",
        "observed", "expected", "fix",
    }
    if set(issue) != required:
        raise ValueError(f"{prefix} must contain exactly the CriticIssue fields.")
    category = issue.get("category")
    issue_type = issue.get("type")
    allowed_types = {
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
    if category == "validity":
        if status != "invalid":
            raise ValueError(f"{prefix}: validity category is reserved for invalid feedback.")
    elif category not in allowed_types or issue_type not in allowed_types[category]:
        raise ValueError(f"{prefix}.type is invalid for its category.")
    _require_choice(issue.get("severity"), f"{prefix}.severity", {"critical", "major", "minor"})
    _require_choice(issue.get("scope"), f"{prefix}.scope", {"global", "object", "part"})
    targets = issue.get("target_ids")
    if not isinstance(targets, list) or len(targets) > 4 or len(set(targets)) != len(targets):
        raise ValueError(f"{prefix}.target_ids is invalid.")
    if any(
        not isinstance(item, str)
        or len(item) != 5
        or item[0] not in "sged"
        or not item[1:].isdigit()
        for item in targets
    ):
        raise ValueError(f"{prefix}.target_ids contains an invalid ID.")
    missing_object = category == "content" and issue_type == "element_presence_or_count"
    if (
        not targets
        and not missing_object
        and issue.get("scope") != "global"
        and category != "validity"
    ):
        raise ValueError(f"{prefix}: empty targets require a missing object or global issue.")
    if category == "validity" and (
        issue.get("severity") != "critical" or issue.get("scope") != "global" or targets
    ):
        raise ValueError(f"{prefix}: validity issues must be critical and global.")
    for key in ("type", "observed", "expected", "fix"):
        _require_string(issue.get(key), f"{prefix}.{key}", non_empty=True)
    return str(category), str(issue_type), tuple(targets)


def _validate_v2_attempt(attempt: dict[str, Any], prefix: str) -> set[str]:
    attempt_id = attempt.get("attempt_id")
    evidence = attempt.get("critic_evidence")
    if evidence is not None:
        evidence = _require_mapping(evidence, f"{prefix}.critic_evidence")
        if evidence.get("attempt_id") != attempt_id:
            raise ValueError(f"{prefix}.critic_evidence.attempt_id does not match.")
        for name in ("png_ref", "labeled_svg_ref", "manifest_ref", "renderer"):
            _require_string(evidence.get(name), f"{prefix}.critic_evidence.{name}", non_empty=True)
        _require_optional_string(
            evidence.get("renderer_version"),
            f"{prefix}.critic_evidence.renderer_version",
        )
        for name in ("width", "height"):
            value = evidence.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{prefix}.critic_evidence.{name} must be a positive integer.")
        diagnostics = evidence.get("diagnostics", [])
        if not isinstance(diagnostics, list) or any(
            not isinstance(item, dict) for item in diagnostics
        ):
            raise ValueError(f"{prefix}.critic_evidence.diagnostics must be an object array.")
        similarity = evidence.get("similarity_evidence")
        if similarity is not None:
            _validate_similarity_evidence_payload(
                similarity,
                f"{prefix}.critic_evidence.similarity_evidence",
                attempt_id,
            )

    calls = attempt.get("critic_calls", [])
    if not isinstance(calls, list):
        raise ValueError(f"{prefix}.critic_calls must be a list.")
    call_ids: set[str] = set()
    for index, call_value in enumerate(calls):
        call_prefix = f"{prefix}.critic_calls[{index}]"
        call = _require_mapping(call_value, call_prefix)
        call_id = call.get("critic_call_id")
        _require_string(call_id, f"{call_prefix}.critic_call_id", non_empty=True)
        if call_id in call_ids:
            raise ValueError(f"{prefix}.critic_calls contains a duplicate call ID.")
        call_ids.add(call_id)
        _require_optional_string(call.get("feedback_id"), f"{call_prefix}.feedback_id")
        _require_nonnegative_int(call.get("retry_index"), f"{call_prefix}.retry_index")
        for name in ("prompt_ref", "response_format_ref", "validation_ref"):
            _require_string(call.get(name), f"{call_prefix}.{name}", non_empty=True)
        _require_optional_string(
            call.get("system_prompt_ref"), f"{call_prefix}.system_prompt_ref"
        )
        _require_optional_string(call.get("raw_output_ref"), f"{call_prefix}.raw_output_ref")
        _require_bool(call.get("validation_success"), f"{call_prefix}.validation_success")
        _require_optional_string(call.get("validation_error"), f"{call_prefix}.validation_error")
        if not isinstance(call.get("generation_parameters"), dict):
            raise ValueError(f"{call_prefix}.generation_parameters must be an object.")
    return call_ids


def _feedback_evidence_target_ids(
    feedback: dict[str, Any],
    prefix: str,
    attempt_id: str,
) -> set[str]:
    metadata = _require_mapping(feedback.get("metadata"), f"{prefix}.metadata")
    provenance = metadata.get("evidence_provenance")
    if not isinstance(provenance, list):
        raise ValueError(f"{prefix}.metadata.evidence_provenance must be a list.")
    result: set[str] = set()
    matched = False
    for index, record_value in enumerate(provenance):
        record = _require_mapping(record_value, f"{prefix}.metadata.evidence_provenance[{index}]")
        if record.get("attempt_id") != attempt_id:
            continue
        matched = True
        targets = record.get("target_ids", [])
        if not isinstance(targets, list) or any(not isinstance(item, str) for item in targets):
            raise ValueError(f"{prefix}.metadata.evidence_provenance target_ids is invalid.")
        similarity = record.get("similarity_evidence")
        if similarity is not None:
            _validate_similarity_evidence_payload(
                similarity,
                f"{prefix}.metadata.evidence_provenance[{index}].similarity_evidence",
                attempt_id,
            )
        result.update(targets)
    if feedback.get("status") in {"pass", "revise"} and not matched:
        raise ValueError(f"{prefix} lacks attempt-correlated evidence provenance.")
    return result


def _validate_similarity_evidence_payload(
    value: object,
    prefix: str,
    attempt_id: object,
) -> None:
    evidence = _require_mapping(value, prefix)
    if evidence.get("attempt_id") != attempt_id:
        raise ValueError(f"{prefix}.attempt_id does not match.")
    _require_choice(
        evidence.get("metric"),
        f"{prefix}.metric",
        {"siglip2_pair_probability"},
    )
    score = _require_number(evidence.get("score"), f"{prefix}.score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{prefix}.score must be between 0 and 1.")
    _require_number(evidence.get("raw_logit"), f"{prefix}.raw_logit")
    for name in (
        "model_id",
        "text_template",
        "text_input",
        "image_sha256",
        "device",
        "dtype",
    ):
        _require_string(evidence.get(name), f"{prefix}.{name}", non_empty=True)
    _require_optional_string(evidence.get("model_revision"), f"{prefix}.model_revision")
    image_sha256 = evidence.get("image_sha256")
    if not isinstance(image_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", image_sha256) is None:
        raise ValueError(f"{prefix}.image_sha256 must be a lowercase SHA-256.")
    latency = _require_number(
        evidence.get("latency_seconds"),
        f"{prefix}.latency_seconds",
    )
    if latency < 0.0:
        raise ValueError(f"{prefix}.latency_seconds must be non-negative.")


def _validate_v1_attempt(
    attempt: dict[str, Any],
    prefix: str,
    *,
    allow_v2_outcomes: bool = False,
) -> None:
    _require_string(attempt.get("attempt_id"), f"{prefix}.attempt_id", non_empty=True)
    mode = attempt.get("mode")
    _require_choice(mode, f"{prefix}.mode", {"initial", "revision"})
    status = attempt.get("status")
    _require_choice(
        status,
        f"{prefix}.status",
        {"succeeded", "failed"},
    )
    parent_attempt_id = attempt.get("parent_attempt_id")
    trigger_feedback_id = attempt.get("trigger_feedback_id")
    _require_optional_string(parent_attempt_id, f"{prefix}.parent_attempt_id")
    _require_optional_string(
        trigger_feedback_id,
        f"{prefix}.trigger_feedback_id",
    )
    if mode == "initial" and (
        parent_attempt_id is not None or trigger_feedback_id is not None
    ):
        raise ValueError(f"{prefix} initial attempt cannot have revision lineage.")
    if mode == "revision":
        _require_string(
            parent_attempt_id,
            f"{prefix}.parent_attempt_id",
            non_empty=True,
        )
        _require_string(
            trigger_feedback_id,
            f"{prefix}.trigger_feedback_id",
            non_empty=True,
        )

    svg_ref = attempt.get("svg_ref")
    _require_optional_string(svg_ref, f"{prefix}.svg_ref")
    if status == "succeeded":
        _require_string(svg_ref, f"{prefix}.svg_ref", non_empty=True)
    _require_optional_string(
        attempt.get("raw_output_ref"),
        f"{prefix}.raw_output_ref",
    )
    _require_optional_string(attempt.get("error"), f"{prefix}.error")
    outcome = attempt.get("outcome")
    outcomes = {"accepted", "rejected", "failed"}
    if allow_v2_outcomes:
        outcomes.update({"selected_best", "rolled_back", "critic_contract_failure"})
    _require_choice(
        outcome,
        f"{prefix}.outcome",
        outcomes,
    )
    if (status == "failed") != (outcome == "failed"):
        raise ValueError(f"{prefix} status and outcome are inconsistent.")
    _require_string(
        attempt.get("stop_reason"),
        f"{prefix}.stop_reason",
        non_empty=True,
    )
    _require_optional_string(attempt.get("prompt_version"), f"{prefix}.prompt_version")
    _require_string_list(attempt.get("context_item_ids"), f"{prefix}.context_item_ids")
    _require_string_list(
        attempt.get("truncated_context_item_ids"),
        f"{prefix}.truncated_context_item_ids",
    )
    _require_mapping(attempt.get("metadata"), f"{prefix}.metadata")
    if not isinstance(attempt.get("model_calls"), list):
        raise ValueError(f"{prefix}.model_calls must be a list.")


def _validate_v1_model_call(model_call: dict[str, Any], prefix: str) -> None:
    _require_string(
        model_call.get("model_call_id"),
        f"{prefix}.model_call_id",
        non_empty=True,
    )
    _require_string(
        model_call.get("raw_output_ref"),
        f"{prefix}.raw_output_ref",
        non_empty=True,
    )
    _require_optional_string(model_call.get("prompt_ref"), f"{prefix}.prompt_ref")
    _require_optional_string(
        model_call.get("system_prompt_ref"),
        f"{prefix}.system_prompt_ref",
    )
    _require_mapping(
        model_call.get("generation_parameters"),
        f"{prefix}.generation_parameters",
    )
    _require_mapping(model_call.get("metadata"), f"{prefix}.metadata")
    _require_optional_string(model_call.get("model_id"), f"{prefix}.model_id")
    _require_optional_string(
        model_call.get("model_revision"),
        f"{prefix}.model_revision",
    )
    _require_optional_string(
        model_call.get("finish_reason"),
        f"{prefix}.finish_reason",
    )
    _require_optional_nonnegative_int(
        model_call.get("prompt_tokens"),
        f"{prefix}.prompt_tokens",
    )
    _require_optional_nonnegative_int(
        model_call.get("completion_tokens"),
        f"{prefix}.completion_tokens",
    )
    _require_optional_nonnegative_number(
        model_call.get("latency_seconds"),
        f"{prefix}.latency_seconds",
    )


def _validate_v1_consistency(
    payload: dict[str, Any],
    attempts: list[GenerationAttemptRecord],
    svg_path: Path,
) -> None:
    revision_count = payload["revision_count"]
    actual_revision_count = sum(attempt.mode == "revision" for attempt in attempts)
    if revision_count != actual_revision_count:
        raise ValueError(
            "revision_count must equal the number of revision attempts."
        )

    selected_attempt_id = _selected_attempt_id(payload)
    selected_attempt = attempts[-1]
    selection_label = "final"
    if selected_attempt_id is not None:
        matching_attempts = [
            attempt
            for attempt in attempts
            if attempt.attempt_id == selected_attempt_id
        ]
        if not matching_attempts:
            raise ValueError(
                "metadata.selection.selected_attempt_id does not identify an attempt."
            )
        selected_attempt = matching_attempts[0]
        selection_label = "selected"
    if payload["outcome"] != selected_attempt.outcome:
        raise ValueError(f"Top-level outcome must match the {selection_label} attempt.")
    if payload["stop_reason"] != selected_attempt.stop_reason:
        raise ValueError(
            f"Top-level stop_reason must match the {selection_label} attempt."
        )
    if payload["outcome"] == "accepted" and not payload["is_valid"]:
        raise ValueError("An accepted artifact must have is_valid=true.")
    if (
        selected_attempt.status == "succeeded"
        and selected_attempt.svg_path is not None
        and svg_path.read_bytes() != selected_attempt.svg_path.read_bytes()
    ):
        raise ValueError(
            f"Top-level SVG must match the {selection_label} successful attempt SVG."
        )

    attempt_ids = [attempt.attempt_id for attempt in attempts]
    _require_unique_ids(attempt_ids, "attempt_id")
    if attempts[0].mode != "initial":
        raise ValueError("The first attempt must use mode=initial.")

    model_call_ids: list[str] = []
    for index, attempt in enumerate(attempts):
        model_call_ids.extend(call.model_call_id for call in attempt.model_calls)
        if index == 0:
            continue
        if attempt.mode != "revision":
            raise ValueError("Every attempt after the first must use mode=revision.")
        if attempt.parent_attempt_id != attempts[index - 1].attempt_id:
            raise ValueError(
                "Each revision parent_attempt_id must reference the previous attempt."
            )
    _require_unique_ids(model_call_ids, "model_call_id")

    feedback_items = payload["critic_feedback"]
    feedback_ids = [feedback["feedback_id"] for feedback in feedback_items]
    _require_unique_ids(feedback_ids, "feedback_id")
    feedback_by_id = {feedback["feedback_id"]: feedback for feedback in feedback_items}
    known_attempt_ids = set(attempt_ids)
    for feedback in feedback_items:
        if feedback["target_attempt_id"] not in known_attempt_ids:
            raise ValueError(
                "Each feedback target_attempt_id must reference an existing attempt."
            )

    _validate_v1_acceptance(payload, selected_attempt, feedback_items)

    for attempt in attempts[1:]:
        feedback = feedback_by_id.get(attempt.trigger_feedback_id)
        if feedback is None:
            raise ValueError(
                "Each revision trigger_feedback_id must reference existing feedback."
            )
        if feedback["target_attempt_id"] != attempt.parent_attempt_id:
            raise ValueError(
                "Revision feedback must target the revision parent attempt."
            )


def _validate_v1_acceptance(
    payload: dict[str, Any],
    final_attempt: GenerationAttemptRecord,
    feedback_items: list[dict[str, Any]],
) -> None:
    runtime = payload["runtime"]
    critic_enabled = runtime.get("enable_critic")
    if critic_enabled is not None:
        _require_bool(critic_enabled, "runtime.enable_critic")
    if critic_enabled is False and feedback_items:
        raise ValueError(
            "critic_feedback must be empty when runtime.enable_critic=false."
        )
    if payload["outcome"] != "accepted":
        return
    if not feedback_items:
        if critic_enabled is True:
            raise ValueError(
                "An accepted artifact with Critic enabled requires final feedback."
            )
        return

    final_feedback = feedback_items[-1]
    if final_feedback["target_attempt_id"] != final_attempt.attempt_id:
        raise ValueError("Final Critic feedback must target the final attempt.")
    if not final_feedback["is_valid"]:
        raise ValueError(
            "An accepted artifact requires final Critic feedback with is_valid=true."
        )
    if not final_feedback["matches_instruction"]:
        raise ValueError(
            "An accepted artifact requires final Critic feedback with "
            "matches_instruction=true."
        )

    if final_feedback.get("critic_schema_version", 1) >= 3:
        threshold = _v3_critic_score_threshold(runtime, final_feedback)
        applicable_scores = [
            evaluation.get("score")
            for evaluation in final_feedback.get("evaluations", [])
            if evaluation.get("applicable") is True
        ]
        if not applicable_scores or any(
            not isinstance(score, int) or score < threshold
            for score in applicable_scores
        ):
            raise ValueError(
                "Every applicable Critic score must meet the configured score threshold."
            )
        return

    acceptance_score = _v1_critic_acceptance_score(runtime)
    if final_feedback["score"] < acceptance_score:
        raise ValueError(
            "Final Critic feedback score must meet the configured acceptance threshold."
        )


def _v1_critic_acceptance_score(runtime: dict[str, Any]) -> float:
    generation_config = runtime.get("generation_config")
    if not isinstance(generation_config, dict):
        return 8.0
    orchestration = generation_config.get("orchestration")
    if not isinstance(orchestration, dict):
        return 8.0
    value = orchestration.get("critic_acceptance_score", 8.0)
    _require_score(value, "runtime.generation_config.orchestration.critic_acceptance_score")
    return float(value)


def _v3_critic_score_threshold(
    runtime: dict[str, Any],
    feedback: dict[str, Any],
) -> float:
    generation_config = runtime.get("generation_config")
    if isinstance(generation_config, dict):
        orchestration = generation_config.get("orchestration")
        if isinstance(orchestration, dict) and "critic_score_threshold" in orchestration:
            value = orchestration["critic_score_threshold"]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 4.0
            ):
                raise ValueError(
                    "runtime.generation_config.orchestration.critic_score_threshold "
                    "must be between 0 and 4."
                )
            return float(value)
    metadata = feedback.get("metadata")
    value = metadata.get("score_threshold", 3.0) if isinstance(metadata, dict) else 3.0
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 4.0
    ):
        raise ValueError("critic_feedback.metadata.score_threshold must be between 0 and 4.")
    return float(value)


def _selected_attempt_id(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    selection = metadata.get("selection")
    if selection is None:
        return None
    if not isinstance(selection, dict):
        raise ValueError("metadata.selection must be an object.")
    selected_id = selection.get("selected_attempt_id")
    _require_string(
        selected_id,
        "metadata.selection.selected_attempt_id",
        non_empty=True,
    )
    return str(selected_id)


def _require_unique_ids(values: list[str], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique within one run.")


def _require_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _require_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")


def _require_string(
    value: object,
    field_name: str,
    *,
    non_empty: bool = False,
) -> None:
    if not isinstance(value, str) or (non_empty and not value.strip()):
        suffix = "a non-empty string" if non_empty else "a string"
        raise ValueError(f"{field_name} must be {suffix}.")


def _require_optional_string(value: object, field_name: str) -> None:
    if value is not None:
        _require_string(value, field_name)


def _require_string_list(value: object, field_name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings.")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_optional_nonnegative_int(value: object, field_name: str) -> None:
    if value is not None:
        _require_nonnegative_int(value, field_name)


def _require_number(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _require_optional_nonnegative_number(value: object, field_name: str) -> None:
    if value is not None:
        number = _require_number(value, field_name)
        if number < 0:
            raise ValueError(f"{field_name} must be non-negative.")


def _require_score(value: object, field_name: str) -> None:
    score = _require_number(value, field_name)
    if not 0.0 <= score <= 10.0:
        raise ValueError(f"{field_name} must be between 0 and 10.")


def _require_choice(
    value: object,
    field_name: str,
    choices: set[str],
) -> None:
    _require_string(value, field_name)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {expected}.")


def _require_optional_choice(
    value: object,
    field_name: str,
    choices: set[str],
) -> None:
    if value is not None:
        _require_choice(value, field_name, choices)
