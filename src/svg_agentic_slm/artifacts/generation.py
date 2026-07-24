"""Utilities for loading generated SVG artifact bundles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        _validate_v1_payload(payload)

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
            _validate_v1_attempt(raw_attempt, field_prefix)
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
    if value > 1:
        raise ValueError(f"Unsupported generation artifact schema_version: {value}")
    return value


def _validate_v1_payload(payload: dict[str, Any]) -> None:
    _require_string(payload.get("run_id"), "run_id", non_empty=True)
    _require_string(payload.get("instruction"), "instruction", non_empty=True)
    _require_string(payload.get("svg_path"), "svg_path", non_empty=True)
    _require_bool(payload.get("is_valid"), "is_valid")
    _require_nonnegative_int(payload.get("revision_count"), "revision_count")
    _require_optional_string(payload.get("render_path"), "render_path")
    _require_choice(
        payload.get("outcome"),
        "outcome",
        {"accepted", "rejected", "failed"},
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


def _validate_v1_attempt(attempt: dict[str, Any], prefix: str) -> None:
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
    _require_choice(
        outcome,
        f"{prefix}.outcome",
        {"accepted", "rejected", "failed"},
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

    final_attempt = attempts[-1]
    if payload["outcome"] != final_attempt.outcome:
        raise ValueError("Top-level outcome must match the final attempt.")
    if payload["stop_reason"] != final_attempt.stop_reason:
        raise ValueError("Top-level stop_reason must match the final attempt.")
    if payload["outcome"] == "accepted" and not payload["is_valid"]:
        raise ValueError("An accepted artifact must have is_valid=true.")
    if (
        final_attempt.status == "succeeded"
        and final_attempt.svg_path is not None
        and svg_path.read_bytes() != final_attempt.svg_path.read_bytes()
    ):
        raise ValueError("Top-level SVG must match the final successful attempt SVG.")

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

    _validate_v1_acceptance(payload, final_attempt, feedback_items)

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
