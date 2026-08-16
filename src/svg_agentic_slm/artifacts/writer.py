"""Durable publication of generated SVG artifact bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from svg_agentic_slm.agents.schemas import CriticFeedback, GenerationResult
from svg_agentic_slm.artifacts.generation import parse_generation_artifact_payload

logger = logging.getLogger(__name__)
_FILE_LOCK_API = import_module("msvcrt" if os.name == "nt" else "fcntl")


class GenerationArtifactRuntime(Protocol):
    """Runtime fields required to publish one generation artifact."""

    metadata_output_path: Path
    svg_output_path: Path
    render_output_path: Path | None
    generation_config: dict[str, Any]
    model_config: dict[str, Any]
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
    final_attempt = result.attempts[-1]
    if result.generated_svg != final_attempt.svg:
        raise ValueError(
            "GenerationResult.generated_svg must match the final attempt SVG."
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
        outcome = final_attempt.metadata.get("outcome")
        stop_reason = final_attempt.metadata.get("stop_reason")
        metadata_payload = {
            "schema_version": 1,
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
                    getattr(runtime, "critic_model_config", None)
                    if runtime.enable_critic and runtime.critic_type in {"llm", "both"}
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
                    "system_prompt_ref": system_prompt_ref,
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

        records.append(
            {
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
        )
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
    return {
        "score": feedback.score,
        "is_valid": feedback.is_valid,
        "matches_instruction": feedback.matches_instruction,
        "issues": feedback.issues,
        "suggestions": feedback.suggestions,
        "critic_type": feedback.critic_type,
        "raw_response": feedback.raw_response,
        "critic_version": feedback.critic_version,
        "model_id": feedback.model_id,
        "model_revision": feedback.model_revision,
        "prompt_version": feedback.prompt_version,
    }


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


@contextmanager
def _artifact_publication_lock(metadata_path: Path) -> Iterator[None]:
    resolved_metadata_path = metadata_path.resolve()
    lock_path = resolved_metadata_path.with_suffix(
        f"{resolved_metadata_path.suffix}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        _acquire_file_lock(lock_handle)
        try:
            yield
        finally:
            _release_file_lock(lock_handle)


def _acquire_file_lock(lock_handle: Any) -> None:
    if os.name != "nt":
        _FILE_LOCK_API.flock(lock_handle.fileno(), _FILE_LOCK_API.LOCK_EX)
        return

    lock_handle.seek(0)
    lock_handle.write(b"\0")
    lock_handle.flush()
    while True:
        lock_handle.seek(0)
        try:
            _FILE_LOCK_API.locking(
                lock_handle.fileno(),
                _FILE_LOCK_API.LK_NBLCK,
                1,
            )
            return
        except OSError:
            time.sleep(0.05)


def _release_file_lock(lock_handle: Any) -> None:
    if os.name != "nt":
        _FILE_LOCK_API.flock(lock_handle.fileno(), _FILE_LOCK_API.LOCK_UN)
        return

    lock_handle.seek(0)
    _FILE_LOCK_API.locking(
        lock_handle.fileno(),
        _FILE_LOCK_API.LK_UNLCK,
        1,
    )


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


def _relative_artifact_reference(path: Path, metadata_dir: Path) -> str:
    """Return a sidecar-relative path so artifact bundles remain portable."""
    return os.path.relpath(path.resolve(), start=metadata_dir)
