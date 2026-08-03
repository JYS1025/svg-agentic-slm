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
from dataclasses import asdict, dataclass
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
    report_path: Path | None = None
    events_path: Path | None = None


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
        metrics = _build_metrics(result)
        timeline = _build_timeline(result)
        events_path = staging_dir / "events.jsonl"
        report_path = staging_dir / "report.md"
        _atomic_write_text(events_path, "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in timeline))
        _atomic_write_text(report_path, _build_report(result, attempt_records, metrics))
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
            "metrics": metrics,
            "events_ref": _relative_artifact_reference(bundle_dir / "events.jsonl", metadata_dir),
            "report_ref": _relative_artifact_reference(bundle_dir / "report.md", metadata_dir),
            "runtime": {
                "enable_rag": runtime.enable_rag,
                "enable_critic": runtime.enable_critic,
                "enable_render": runtime.enable_render,
                "critic_type": runtime.critic_type,
                "config_paths": runtime.config_paths,
                "generation_config": runtime.generation_config,
                "model_config": runtime.model_config,
                "critic_model_config": getattr(runtime, "critic_model_config", {}),
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
        report_path=bundle_dir / "report.md",
        events_path=bundle_dir / "events.jsonl",
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

        evidence_record: dict[str, Any] | None = None
        if attempt.critic_evidence is not None:
            evidence = attempt.critic_evidence
            if evidence.attempt_id != attempt.attempt_id:
                raise ValueError("Critic evidence attempt_id does not match attempt.")
            png_path = attempt_dir / f"{prefix}.critic.png"
            labeled_path = attempt_dir / f"{prefix}.labeled.svg"
            manifest_path = attempt_dir / f"{prefix}.manifest.json"
            _atomic_write_bytes(png_path, evidence.png)
            _atomic_write_text(labeled_path, evidence.labeling.labeled_svg)
            _atomic_write_text(manifest_path, json.dumps(
                {key: asdict(value) for key, value in evidence.labeling.elements.items()},
                indent=2, ensure_ascii=False,
            ))
            evidence_record = {
                "attempt_id": evidence.attempt_id,
                "png_ref": _relative_artifact_reference(published_attempt_dir / png_path.name, metadata_dir),
                "labeled_svg_ref": _relative_artifact_reference(published_attempt_dir / labeled_path.name, metadata_dir),
                "manifest_ref": _relative_artifact_reference(published_attempt_dir / manifest_path.name, metadata_dir),
                "renderer": evidence.renderer, "renderer_version": evidence.renderer_version,
                "width": evidence.width, "height": evidence.height,
            }

        critic_calls: list[dict[str, Any]] = []
        matching_events = [event for event in result.feedback_events if event.target_attempt_id == attempt.attempt_id]
        call_number = 0
        traced_calls = [
            (event.feedback_id, call)
            for event in matching_events for call in event.feedback.model_calls
        ] + [(None, call) for call in attempt.critic_error_calls]
        for feedback_id, call in traced_calls:
            call_prefix = f"{prefix}.critic-call-{call_number:03d}"
            prompt_ref = _write_optional_trace_text(attempt_dir / f"{call_prefix}.prompt.txt", call.prompt, metadata_dir, published_attempt_dir)
            system_ref = _write_optional_trace_text(attempt_dir / f"{call_prefix}.system.txt", call.system_prompt, metadata_dir, published_attempt_dir)
            raw_ref = _write_optional_trace_text(attempt_dir / f"{call_prefix}.raw.txt", call.response.text, metadata_dir, published_attempt_dir)
            schema_path = attempt_dir / f"{call_prefix}.response-format.json"
            validation_path = attempt_dir / f"{call_prefix}.validation.json"
            _atomic_write_text(schema_path, json.dumps(call.response_format, indent=2, ensure_ascii=False))
            _atomic_write_text(validation_path, json.dumps({"success": call.validation_success, "error": call.validation_error}, indent=2, ensure_ascii=False))
            critic_calls.append({
                "critic_call_id": call.critic_call_id, "feedback_id": feedback_id,
                "retry_index": call.retry_index, "prompt_ref": prompt_ref,
                "system_prompt_ref": system_ref, "raw_output_ref": raw_ref,
                "response_format_ref": _relative_artifact_reference(published_attempt_dir / schema_path.name, metadata_dir),
                "validation_ref": _relative_artifact_reference(published_attempt_dir / validation_path.name, metadata_dir),
                "validation_success": call.validation_success,
                "validation_error": call.validation_error,
                "generation_parameters": call.generation_parameters,
                "model_id": call.response.model_id, "model_revision": call.response.model_revision,
                "finish_reason": call.response.finish_reason,
                "prompt_tokens": call.response.prompt_tokens,
                "completion_tokens": call.response.completion_tokens,
                "latency_seconds": call.response.latency_seconds,
                "metadata": call.response.metadata,
            })
            call_number += 1

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
                "critic_evidence": evidence_record,
                "critic_calls": critic_calls,
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
        "status": feedback.status,
        "structured_issues": [asdict(item) for item in feedback.structured_issues],
        "preserve": feedback.preserve,
        "critic_schema_version": feedback.schema_version,
        "metadata": feedback.metadata,
        "model_call_ids": [call.critic_call_id for call in feedback.model_calls],
    }


def _build_metrics(result: GenerationResult) -> dict[str, Any]:
    generator_calls = [call for attempt in result.attempts for call in attempt.model_calls]
    critic_calls = [call for feedback in result.critic_feedback for call in feedback.model_calls]
    critic_calls.extend(call for attempt in result.attempts for call in attempt.critic_error_calls)
    def total(values: list[int | None]) -> int:
        return sum(value or 0 for value in values)
    return {
        "total_latency_seconds": result.metadata.get("timing", {}).get("generation_latency_seconds"),
        "generator_call_count": len(generator_calls),
        "generator_prompt_tokens": total([call.response.prompt_tokens for call in generator_calls]),
        "generator_completion_tokens": total([call.response.completion_tokens for call in generator_calls]),
        "generator_latency_seconds": round(sum(call.response.latency_seconds or 0.0 for call in generator_calls), 6),
        "critic_call_count": len(critic_calls),
        "critic_retry_count": sum(call.retry_index > 0 for call in critic_calls),
        "critic_prompt_tokens": total([call.response.prompt_tokens for call in critic_calls]),
        "critic_completion_tokens": total([call.response.completion_tokens for call in critic_calls]),
        "critic_latency_seconds": round(sum(call.response.latency_seconds or 0.0 for call in critic_calls), 6),
        "revision_count": result.revision_count,
        "final_status": result.critic_feedback[-1].status if result.critic_feedback else None,
        "final_score": result.critic_feedback[-1].score if result.critic_feedback else None,
    }


def _build_timeline(result: GenerationResult) -> list[dict[str, Any]]:
    elapsed = 0.0
    events: list[dict[str, Any]] = [{"sequence": 0, "event": "run_started", "run_id": result.run_id, "elapsed_seconds": elapsed}]
    for attempt in result.attempts:
        elapsed += sum(call.response.latency_seconds or 0.0 for call in attempt.model_calls)
        events.append({"sequence": len(events), "event": "generator_completed", "attempt_id": attempt.attempt_id, "mode": attempt.mode, "status": attempt.status, "elapsed_seconds": round(elapsed, 6)})
        validation = attempt.metadata.get("validation", {})
        events.append({"sequence": len(events), "event": "validation_completed", "attempt_id": attempt.attempt_id, "valid": validation.get("is_valid"), "elapsed_seconds": round(elapsed, 6)})
        for feedback_event in [item for item in result.feedback_events if item.target_attempt_id == attempt.attempt_id]:
            for call in feedback_event.feedback.model_calls:
                elapsed += call.response.latency_seconds or 0.0
                events.append({"sequence": len(events), "event": "critic_call_completed", "attempt_id": attempt.attempt_id, "feedback_id": feedback_event.feedback_id, "critic_call_id": call.critic_call_id, "retry_index": call.retry_index, "validation_success": call.validation_success, "validation_error": call.validation_error, "latency_seconds": call.response.latency_seconds, "elapsed_seconds": round(elapsed, 6)})
            events.append({"sequence": len(events), "event": "critic_feedback_completed", "attempt_id": attempt.attempt_id, "feedback_id": feedback_event.feedback_id, "status": feedback_event.feedback.status, "score": feedback_event.feedback.score, "elapsed_seconds": round(elapsed, 6)})
        for call in attempt.critic_error_calls:
            elapsed += call.response.latency_seconds or 0.0
            events.append({"sequence": len(events), "event": "critic_call_failed", "attempt_id": attempt.attempt_id, "critic_call_id": call.critic_call_id, "retry_index": call.retry_index, "validation_error": call.validation_error, "latency_seconds": call.response.latency_seconds, "elapsed_seconds": round(elapsed, 6)})
    final = result.attempts[-1]
    events.append({"sequence": len(events), "event": "run_completed", "run_id": result.run_id, "outcome": final.metadata.get("outcome"), "stop_reason": final.metadata.get("stop_reason"), "elapsed_seconds": result.metadata.get("timing", {}).get("generation_latency_seconds", round(elapsed, 6))})
    return events


def _build_report(result: GenerationResult, attempts: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    final = result.attempts[-1]
    lines = ["# SVG generation report", "", f"- Run: `{result.run_id}`", f"- Instruction: {result.instruction}", f"- Outcome: `{final.metadata.get('outcome')}`", f"- Stop reason: `{final.metadata.get('stop_reason')}`", f"- Revisions: {result.revision_count}", "", "## Metrics", "", "```json", json.dumps(metrics, indent=2, ensure_ascii=False), "```"]
    for index, (attempt, record) in enumerate(zip(result.attempts, attempts, strict=True)):
        lines.extend(["", f"## Attempt {index}: `{attempt.attempt_id}`", "", f"- Mode: `{attempt.mode}`", f"- Status: `{attempt.status}`", f"- SVG: `{record.get('svg_ref')}`", f"- Generator calls: {len(attempt.model_calls)}", f"- Critic calls: {len(record.get('critic_calls', []))}"])
        feedback = next((event.feedback for event in result.feedback_events if event.target_attempt_id == attempt.attempt_id), None)
        if feedback:
            lines.extend([f"- Critic: `{feedback.status}` / {feedback.score}", "", "### Critic issues", ""])
            lines.extend([f"- **{issue.severity}** `{','.join(issue.target_ids) or 'global'}`: {issue.observed} → {issue.fix}" for issue in feedback.structured_issues] or ["- None"])
    return "\n".join(lines) + "\n"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary_path, path); _fsync_directory(path.parent)
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
