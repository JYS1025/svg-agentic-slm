"""Utilities for loading generated SVG artifact bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def load_generation_artifact(path: str | Path) -> GenerationArtifactRecord:
    """Load a generated artifact record from an SVG or metadata path."""
    path = Path(path)
    metadata_path = _resolve_metadata_path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    svg_path = _resolve_artifact_reference(
        payload.get("svg_path"),
        metadata_path=metadata_path,
        fallback=metadata_path.with_suffix(".svg"),
    )
    render_path_value = payload.get("render_path")
    render_path = (
        _resolve_artifact_reference(render_path_value, metadata_path=metadata_path)
        if render_path_value
        else None
    )

    return GenerationArtifactRecord(
        instruction=payload.get("instruction", ""),
        svg_path=svg_path,
        metadata_path=metadata_path,
        render_path=render_path,
        is_valid=payload.get("is_valid", False),
        revision_count=payload.get("revision_count", 0),
        critic_feedback=payload.get("critic_feedback", []),
        runtime=payload.get("runtime", {}),
        metadata=payload.get("metadata", {}),
        generated_at_utc=payload.get("generated_at_utc"),
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
) -> Path:
    """Resolve new sidecar-relative paths while retaining legacy cwd paths."""
    if value is None:
        if fallback is None:
            raise ValueError("Artifact path is missing from generation metadata.")
        return fallback
    if not isinstance(value, str):
        raise ValueError("Artifact paths in generation metadata must be strings.")

    path = Path(value)
    if path.is_absolute():
        return path

    sidecar_relative = metadata_path.parent / path
    if sidecar_relative.exists():
        return sidecar_relative
    if path.exists():
        return path
    return sidecar_relative
