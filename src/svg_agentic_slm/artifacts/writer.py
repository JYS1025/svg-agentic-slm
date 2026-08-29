"""Durable publication of generated SVG artifact bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from svg_agentic_slm.agents.schemas import CriticFeedback, GenerationResult
from svg_agentic_slm.artifacts.generation import parse_generation_artifact_payload

logger = logging.getLogger(__name__)


class GenerationArtifactRuntime(Protocol):
    """Runtime fields required to publish one generation artifact."""

    metadata_output_path: Path
    svg_output_path: Path
    render_output_path: Path | None
    generation_config: dict[str, Any]
    model_config: dict[str, Any]
    critic_model_config: dict[str, Any] | None
    similarity_model_config: dict[str, Any] | None
    rag_config: dict[str, Any]
    config_paths: dict[str, str]
    enable_rag: bool
    enable_critic: bool
    enable_render: bool
    critic_type: str | None
    render_config: dict[str, Any]
    run_id: str


@dataclass
class GenerationArtifacts:
    """Paths to files produced by a generation run."""

    svg_path: Path
    metadata_path: Path
    render_path: Path | None


def persist_generation_artifacts(
    result: GenerationResult,
    runtime: GenerationArtifactRuntime,
) -> GenerationArtifacts:
    """Serialize publication for one output stem."""
    with _artifact_publication_lock(runtime.metadata_output_path):
        return _persist_generation_artifacts_locked(result, runtime)


def _persist_generation_artifacts_locked(
    result: GenerationResult,
    runtime: GenerationArtifactRuntime,
) -> GenerationArtifacts:
    """Publish an immutable run bundle, then atomically replace its sidecar."""
    if not result.attempts:
        raise ValueError("GenerationResult must contain at least one attempt.")
    selected_attempt = _selected_attempt(result)
    if result.generated_svg != selected_attempt.svg:
        attempt_label = (
            "final" if selected_attempt is result.attempts[-1] else "selected"
        )
        raise ValueError(
            "GenerationResult.generated_svg must match the "
            f"{attempt_label} attempt SVG."
        )
    if result.critic_feedback and not result.feedback_events:
        raise ValueError(
            "Critic feedback must include feedback-to-attempt correlation events."
        )
    metadata_dir = runtime.metadata_output_path.parent.resolve()
    run_id = result.run_id or runtime.run_id
    bundle_token = build_bundle_token(run_id)
    bundle_root = runtime.metadata_output_path.with_suffix(".artifacts")
    bundle_dir = bundle_root / bundle_token
    staging_dir = bundle_root / f".{bundle_token}.{uuid4().hex}.tmp"
    metadata_published = False
    git_provenance = _git_provenance()

    if bundle_dir.exists():
        raise FileExistsError(f"Artifact bundle already exists: {bundle_dir}")

    try:
        staging_dir.mkdir(parents=True)
        staged_svg_path = staging_dir / "final.svg"
        published_svg_path = bundle_dir / "final.svg"
        _atomic_write_text(staged_svg_path, result.generated_svg)

        render_reference: str | None = None
        published_render_path: Path | None = None
        if result.render_path:
            source_render_path = Path(result.render_path)
            published_render_path = bundle_dir / f"render{source_render_path.suffix}"
            _copy_file_durable(
                source_render_path,
                staging_dir / published_render_path.name,
            )
            render_reference = _relative_artifact_reference(
                published_render_path,
                metadata_dir,
            )

        attempt_records = _persist_attempt_artifacts(
            result=result,
            attempt_dir=staging_dir / "attempts",
            published_attempt_dir=bundle_dir / "attempts",
            metadata_dir=metadata_dir,
        )
        feedback_records = _serialize_feedback_events(result)
        result_metadata = dict(result.metadata)
        generator_metadata = dict(result_metadata.get("generator", {}))
        generator_metadata["attempts"] = attempt_records
        result_metadata["generator"] = generator_metadata
        outcome = selected_attempt.metadata.get("outcome")
        stop_reason = selected_attempt.metadata.get("stop_reason")
        provenance = _build_provenance(
            result,
            runtime,
            attempt_records,
            git_provenance=git_provenance,
        )
        metadata_payload = {
            "schema_version": 3,
            "run_id": run_id,
            "instruction": result.instruction,
            "svg_path": _relative_artifact_reference(
                published_svg_path,
                metadata_dir,
            ),
            "is_valid": result.is_valid,
            "outcome": outcome,
            "stop_reason": stop_reason,
            "render_path": render_reference,
            "revision_count": result.revision_count,
            "critic_feedback": feedback_records,
            "runtime": {
                "enable_rag": runtime.enable_rag,
                "enable_critic": runtime.enable_critic,
                "enable_render": runtime.enable_render,
                "critic_type": runtime.critic_type,
                "config_paths": runtime.config_paths,
                "generation_config": runtime.generation_config,
                "model_config": runtime.model_config,
                "critic_model_config": (
                    _redact_sensitive_config(
                        getattr(runtime, "critic_model_config", None)
                    )
                    if runtime.enable_critic
                    and runtime.critic_type
                    in {"llm", "both", "vlm", "rule_vlm", "critic_v1"}
                    else None
                ),
                "similarity_model_config": (
                    _redact_sensitive_config(
                        getattr(runtime, "similarity_model_config", None)
                    )
                    if getattr(runtime, "similarity_scorer", None) is not None
                    else None
                ),
                "rag_config": runtime.rag_config if runtime.enable_rag else {},
                "render_config": runtime.render_config,
                "planned_artifacts": {
                    "svg_path": str(runtime.svg_output_path),
                    "metadata_path": str(runtime.metadata_output_path),
                    "render_path": (
                        str(runtime.render_output_path)
                        if runtime.render_output_path
                        else None
                    ),
                },
            },
            "metadata": result_metadata,
            "provenance": provenance,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        metadata_content = json.dumps(
            metadata_payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        serialized_payload = json.loads(metadata_content)

        _fsync_directory(staging_dir)
        os.replace(staging_dir, bundle_dir)
        _fsync_directory(bundle_dir.parent)
        parse_generation_artifact_payload(
            serialized_payload,
            runtime.metadata_output_path,
        )

        def mark_metadata_published() -> None:
            nonlocal metadata_published
            metadata_published = True

        _atomic_write_text(
            runtime.metadata_output_path,
            metadata_content,
            on_replace=mark_metadata_published,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if not metadata_published and bundle_dir.exists():
            shutil.rmtree(bundle_dir)

    exported_svg_path = runtime.svg_output_path
    try:
        _atomic_write_text(exported_svg_path, result.generated_svg)
    except OSError:
        logger.warning(
            "Published artifact bundle but could not update SVG export alias: %s",
            exported_svg_path,
            exc_info=True,
        )
        exported_svg_path = published_svg_path

    return GenerationArtifacts(
        svg_path=exported_svg_path,
        metadata_path=runtime.metadata_output_path,
        render_path=published_render_path,
    )


def _persist_attempt_artifacts(
    *,
    result: GenerationResult,
    attempt_dir: Path,
    published_attempt_dir: Path,
    metadata_dir: Path,
) -> list[dict[str, Any]]:
    if not result.attempts:
        return []

    attempt_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for attempt_index, attempt in enumerate(result.attempts):
        prefix = f"attempt-{attempt_index:03d}"
        svg_ref: str | None = None
        if attempt.svg:
            svg_path = attempt_dir / f"{prefix}.svg"
            _atomic_write_text(svg_path, attempt.svg)
            svg_ref = _relative_artifact_reference(
                published_attempt_dir / svg_path.name,
                metadata_dir,
            )

        model_calls: list[dict[str, Any]] = []
        for call_index, call in enumerate(attempt.model_calls):
            raw_path = attempt_dir / f"{prefix}.call-{call_index:03d}.raw.txt"
            _atomic_write_text(raw_path, call.response.text)
            prompt_ref = _write_optional_trace_text(
                attempt_dir / f"{prefix}.call-{call_index:03d}.prompt.txt",
                call.prompt,
                metadata_dir,
                published_attempt_dir,
            )
            system_prompt_ref = _write_optional_trace_text(
                attempt_dir / f"{prefix}.call-{call_index:03d}.system.txt",
                call.system_prompt,
                metadata_dir,
                published_attempt_dir,
            )
            model_calls.append(
                {
                    "model_call_id": call.model_call_id,
                    "prompt_ref": prompt_ref,
                    "prompt_sha256": _sha256_text(call.prompt),
                    "system_prompt_ref": system_prompt_ref,
                    "system_prompt_sha256": (
                        _sha256_text(call.system_prompt)
                        if call.system_prompt is not None
                        else None
                    ),
                    "generation_parameters": call.generation_parameters,
                    "raw_output_ref": _relative_artifact_reference(
                        published_attempt_dir / raw_path.name,
                        metadata_dir,
                    ),
                    "model_id": call.response.model_id,
                    "model_revision": call.response.model_revision,
                    "finish_reason": call.response.finish_reason,
                    "prompt_tokens": call.response.prompt_tokens,
                    "completion_tokens": call.response.completion_tokens,
                    "latency_seconds": call.response.latency_seconds,
                    "metadata": call.response.metadata,
                }
            )

        raw_output_ref = model_calls[-1]["raw_output_ref"] if model_calls else None
        if raw_output_ref is None and attempt.raw_output:
            raw_path = attempt_dir / f"{prefix}.raw.txt"
            _atomic_write_text(raw_path, attempt.raw_output)
            raw_output_ref = _relative_artifact_reference(
                published_attempt_dir / raw_path.name,
                metadata_dir,
            )

        evidence_record = _persist_critic_evidence(
            attempt=attempt,
            prefix=prefix,
            attempt_dir=attempt_dir,
            published_attempt_dir=published_attempt_dir,
            metadata_dir=metadata_dir,
        )
        critic_calls = _persist_critic_call_traces(
            result=result,
            attempt=attempt,
            prefix=prefix,
            attempt_dir=attempt_dir,
            published_attempt_dir=published_attempt_dir,
            metadata_dir=metadata_dir,
        )

        record: dict[str, Any] = {
            "attempt_id": attempt.attempt_id,
            "mode": attempt.mode,
            "parent_attempt_id": attempt.parent_attempt_id,
            "trigger_feedback_id": attempt.trigger_feedback_id,
            "svg_ref": svg_ref,
            "raw_output_ref": raw_output_ref,
            "status": attempt.status,
            "error": attempt.error,
            "outcome": attempt.metadata.get("outcome"),
            "stop_reason": attempt.metadata.get("stop_reason"),
            "prompt_version": attempt.prompt_version,
            "context_item_ids": attempt.context_item_ids,
            "truncated_context_item_ids": attempt.truncated_context_item_ids,
            "model_calls": model_calls,
            "metadata": attempt.metadata,
        }
        if evidence_record is not None:
            record["critic_evidence"] = evidence_record
        if critic_calls:
            record["critic_calls"] = critic_calls
        records.append(record)
    return records


def _persist_critic_evidence(
    *,
    attempt: Any,
    prefix: str,
    attempt_dir: Path,
    published_attempt_dir: Path,
    metadata_dir: Path,
) -> dict[str, Any] | None:
    """Persist optional attempt-correlated PNG and labeling evidence."""
    evidence = getattr(attempt, "critic_evidence", None)
    if evidence is None:
        return None
    if evidence.attempt_id != attempt.attempt_id:
        raise ValueError("Critic evidence attempt_id does not match attempt.")
    if evidence.labeling.attempt_id != attempt.attempt_id:
        raise ValueError("Critic labeling attempt_id does not match attempt.")
    if not isinstance(evidence.png, bytes) or not evidence.png:
        raise ValueError("Critic evidence PNG must contain bytes.")
    if not isinstance(evidence.labeling.labeled_svg, str) or not evidence.labeling.labeled_svg:
        raise ValueError("Critic evidence labeled SVG must be non-empty.")

    png_path = attempt_dir / f"{prefix}.critic.png"
    labeled_path = attempt_dir / f"{prefix}.labeled.svg"
    manifest_path = attempt_dir / f"{prefix}.manifest.json"
    _atomic_write_bytes(png_path, evidence.png)
    _atomic_write_text(labeled_path, evidence.labeling.labeled_svg)
    manifest = {
        str(key): _serialize_dataclass_value(value)
        for key, value in evidence.labeling.elements.items()
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
    )

    record: dict[str, Any] = {
        "attempt_id": evidence.attempt_id,
        "role": "critic_evidence_render",
        "png_ref": _relative_artifact_reference(
            published_attempt_dir / png_path.name,
            metadata_dir,
        ),
        "labeled_svg_ref": _relative_artifact_reference(
            published_attempt_dir / labeled_path.name,
            metadata_dir,
        ),
        "manifest_ref": _relative_artifact_reference(
            published_attempt_dir / manifest_path.name,
            metadata_dir,
        ),
        "renderer": evidence.renderer,
        "renderer_version": evidence.renderer_version,
        "width": evidence.width,
        "height": evidence.height,
    }
    diagnostics = getattr(evidence, "diagnostics", None)
    if diagnostics:
        record["diagnostics"] = [
            _serialize_dataclass_value(item) for item in diagnostics
        ]
    similarity_evidence = getattr(evidence, "similarity_evidence", None)
    if similarity_evidence is not None:
        record["similarity_evidence"] = _serialize_dataclass_value(
            similarity_evidence
        )
    return record


def _persist_critic_call_traces(
    *,
    result: GenerationResult,
    attempt: Any,
    prefix: str,
    attempt_dir: Path,
    published_attempt_dir: Path,
    metadata_dir: Path,
) -> list[dict[str, Any]]:
    """Persist successful and failed Critic calls correlated to an attempt."""
    traced_calls: list[tuple[str | None, Any]] = []
    for event in result.feedback_events:
        if event.target_attempt_id != attempt.attempt_id:
            continue
        for call in getattr(event.feedback, "model_calls", []):
            traced_calls.append((event.feedback_id, call))
    for call in getattr(attempt, "critic_error_calls", []):
        traced_calls.append((None, call))

    records: list[dict[str, Any]] = []
    for call_number, (feedback_id, call) in enumerate(traced_calls):
        call_prefix = f"{prefix}.critic-call-{call_number:03d}"
        response = call.response
        prompt_ref = _write_optional_trace_text(
            attempt_dir / f"{call_prefix}.prompt.txt",
            call.prompt,
            metadata_dir,
            published_attempt_dir,
        )
        system_prompt_ref = _write_optional_trace_text(
            attempt_dir / f"{call_prefix}.system.txt",
            call.system_prompt,
            metadata_dir,
            published_attempt_dir,
        )
        raw_output_ref = _write_optional_trace_text(
            attempt_dir / f"{call_prefix}.raw.txt",
            response.text,
            metadata_dir,
            published_attempt_dir,
        )
        response_format_path = attempt_dir / f"{call_prefix}.response-format.json"
        validation_path = attempt_dir / f"{call_prefix}.validation.json"
        _atomic_write_text(
            response_format_path,
            json.dumps(
                _redact_sensitive_config(call.response_format),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        _atomic_write_text(
            validation_path,
            json.dumps(
                {
                    "success": call.validation_success,
                    "error": call.validation_error,
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        record: dict[str, Any] = {
            "critic_call_id": call.critic_call_id,
            "feedback_id": feedback_id,
            "retry_index": call.retry_index,
            "prompt_ref": prompt_ref,
            "prompt_sha256": _sha256_text(call.prompt),
            "system_prompt_ref": system_prompt_ref,
            "system_prompt_sha256": (
                _sha256_text(call.system_prompt)
                if call.system_prompt is not None
                else None
            ),
            "raw_output_ref": raw_output_ref,
            "response_format_ref": _relative_artifact_reference(
                published_attempt_dir / response_format_path.name,
                metadata_dir,
            ),
            "validation_ref": _relative_artifact_reference(
                published_attempt_dir / validation_path.name,
                metadata_dir,
            ),
            "validation_success": call.validation_success,
            "validation_error": call.validation_error,
            "generation_parameters": _redact_sensitive_config(
                call.generation_parameters
            ),
            "model_id": response.model_id,
            "model_revision": response.model_revision,
            "finish_reason": response.finish_reason,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_seconds": response.latency_seconds,
            "metadata": _redact_sensitive_config(response.metadata),
        }
        time_to_first_token = getattr(response, "time_to_first_token_seconds", None)
        tokens_per_second = getattr(response, "tokens_per_second", None)
        if time_to_first_token is not None:
            record["time_to_first_token_seconds"] = time_to_first_token
        if tokens_per_second is not None:
            record["tokens_per_second"] = tokens_per_second
        records.append(record)
    return records


def _serialize_feedback_events(result: GenerationResult) -> list[dict[str, Any]]:
    if result.feedback_events:
        return [
            {
                "feedback_id": event.feedback_id,
                "target_attempt_id": event.target_attempt_id,
                **_serialize_feedback(event.feedback),
            }
            for event in result.feedback_events
        ]
    return [_serialize_feedback(feedback) for feedback in result.critic_feedback]


def _serialize_feedback(feedback: CriticFeedback) -> dict[str, Any]:
    critic_schema_version = getattr(feedback, "schema_version", 1)
    structured_issues = list(getattr(feedback, "structured_issues", []))
    payload: dict[str, Any] = {
        "score": feedback.score,
        "is_valid": feedback.is_valid,
        "matches_instruction": feedback.matches_instruction,
        "suggestions": feedback.suggestions,
        "critic_type": feedback.critic_type,
        "raw_response": feedback.raw_response,
        "critic_version": feedback.critic_version,
        "model_id": feedback.model_id,
        "model_revision": feedback.model_revision,
        "prompt_version": feedback.prompt_version,
    }
    if isinstance(critic_schema_version, int) and critic_schema_version >= 3:
        model_calls = list(getattr(feedback, "model_calls", []))
        payload.update(
            {
                "status": feedback.status,
                "evaluations": [
                    _serialize_dataclass_value(item) for item in feedback.evaluations
                ],
                "issues": [
                    _serialize_dataclass_value(item) for item in structured_issues
                ],
                "legacy_issues": list(feedback.issues),
                "critic_schema_version": critic_schema_version,
                "metadata": _redact_sensitive_config(feedback.metadata),
                "model_call_ids": [call.critic_call_id for call in model_calls],
            }
        )
    elif isinstance(critic_schema_version, int) and critic_schema_version >= 2:
        model_calls = list(getattr(feedback, "model_calls", []))
        payload.update(
            {
                "status": feedback.status,
                "issues": [
                    _serialize_dataclass_value(item) for item in structured_issues
                ],
                "legacy_issues": list(feedback.issues),
                "preserve": list(feedback.preserve),
                "critic_schema_version": critic_schema_version,
                "metadata": _redact_sensitive_config(feedback.metadata),
                "model_call_ids": [call.critic_call_id for call in model_calls],
            }
        )
    else:
        payload["issues"] = list(feedback.issues)
    return payload


def _serialize_dataclass_value(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("Structured Critic artifact values must be dataclasses or mappings.")


def _redact_sensitive_config(value: Any) -> Any:
    """Redact inline secrets while preserving environment-variable names."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_config_key(str(key))
                else _redact_sensitive_config(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    exact_names = {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "token",
        "password",
        "secret",
        "private_key",
    }
    suffixes = (
        "_api_key",
        "_credential",
        "_credentials",
        "_token",
        "_password",
        "_secret",
        "_private_key",
    )
    return normalized in exact_names or normalized.endswith(suffixes)


def _write_optional_trace_text(
    path: Path,
    content: str | None,
    metadata_dir: Path,
    published_dir: Path | None = None,
) -> str | None:
    if not content:
        return None
    _atomic_write_text(path, content)
    published_path = (published_dir / path.name) if published_dir else path
    return _relative_artifact_reference(published_path, metadata_dir)


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    on_replace: Callable[[], None] | None = None,
) -> None:
    """Atomically replace one text artifact without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        if on_replace is not None:
            on_replace()
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically publish binary evidence without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _artifact_publication_lock(metadata_path: Path) -> Iterator[None]:
    """Serialize publication with an atomic, self-cleaning lock directory."""
    resolved_metadata_path = metadata_path.resolve()
    lock_path = resolved_metadata_path.with_suffix(
        f"{resolved_metadata_path.suffix}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 120.0
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            # The owner may remove the directory between mkdir() and inspection.
            if not lock_path.exists():
                continue
            if not lock_path.is_dir():
                raise RuntimeError(
                    f"Legacy artifact lock file blocks publication: {lock_path}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for artifact lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock_path.rmdir()


def _copy_file_durable(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Rendered artifact not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_bundle_token(run_id: str) -> str:
    """Return a filesystem-safe run token with collision-resistant identity."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"{(token or 'run')[:64]}.{digest}"


def _selected_attempt(result: GenerationResult) -> Any:
    selection = result.metadata.get("selection", {})
    selected_id = (
        selection.get("selected_attempt_id")
        if isinstance(selection, dict)
        else None
    )
    if selected_id is None:
        return result.attempts[-1]
    for attempt in result.attempts:
        if attempt.attempt_id == selected_id:
            return attempt
    raise ValueError("metadata.selection.selected_attempt_id does not identify an attempt.")


def _build_provenance(
    result: GenerationResult,
    runtime: GenerationArtifactRuntime,
    attempt_records: list[dict[str, Any]],
    *,
    git_provenance: dict[str, Any],
) -> dict[str, Any]:
    config_hashes = {
        name: _sha256_file(Path(path))
        for name, path in runtime.config_paths.items()
        if Path(path).is_file()
    }
    generator_prompt_hashes = [
        {
            "attempt_id": attempt["attempt_id"],
            "model_call_id": call["model_call_id"],
            "prompt": call.get("prompt_sha256"),
            "system_prompt": call.get("system_prompt_sha256"),
        }
        for attempt in attempt_records
        for call in attempt.get("model_calls", [])
    ]
    critic_prompt_hashes = [
        {
            "attempt_id": attempt["attempt_id"],
            "critic_call_id": call["critic_call_id"],
            "prompt": call.get("prompt_sha256"),
            "system_prompt": call.get("system_prompt_sha256"),
        }
        for attempt in attempt_records
        for call in attempt.get("critic_calls", [])
    ]
    benchmark_hash = getattr(runtime, "benchmark_hash", None)
    if benchmark_hash is None:
        benchmark_hash = result.metadata.get("benchmark_sha256")
    effective_config = {
        "generation": runtime.generation_config,
        "model": runtime.model_config,
        "critic_model": runtime.critic_model_config,
        "rag": runtime.rag_config,
        "render": runtime.render_config,
    }
    return {
        "git": git_provenance,
        "execution_command": getattr(runtime, "execution_command", None),
        "config_sha256": config_hashes,
        "effective_config_sha256": _sha256_json(effective_config),
        "prompt_sha256": {
            "generator": generator_prompt_hashes,
            "critic": critic_prompt_hashes,
        },
        "benchmark_sha256": benchmark_hash,
    }


def _git_provenance() -> dict[str, Any]:
    source_checkout = Path(__file__).resolve().parents[3]
    git_cwd = source_checkout if (source_checkout / ".git").exists() else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=git_cwd,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            cwd=git_cwd,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha or None, "dirty": bool(status.strip())}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    )
    return _sha256_text(serialized)


def _relative_artifact_reference(path: Path, metadata_dir: Path) -> str:
    """Return a sidecar-relative path so artifact bundles remain portable."""
    return os.path.relpath(path.resolve(), start=metadata_dir)
