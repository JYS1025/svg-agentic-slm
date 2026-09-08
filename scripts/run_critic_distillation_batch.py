#!/usr/bin/env python3
"""Generate an unfiltered, resumable Critic-distillation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from svg_agentic_slm.data.split_integrity import (
    read_manifest_records,
    validate_disjoint_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-manifest", type=Path, action="append", required=True)
    parser.add_argument("--sft-manifest", type=Path, action="append", required=True)
    parser.add_argument("--rag-manifest", type=Path, action="append", required=True)
    parser.add_argument("--eval-manifest", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    records = read_manifest_records(args.critic_manifest)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("Critic manifest contains no selected records.")
    for index, record in enumerate(records):
        instruction = record.get("instruction", record.get("description"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"Critic record {record['sample_id']!r} at index {index} needs instruction."
            )

    integrity = validate_disjoint_manifests(
        critic_paths=args.critic_manifest,
        protected_paths={
            "sft": args.sft_manifest,
            "rag": args.rag_manifest,
            "eval": args.eval_manifest,
        },
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "split_integrity.json", integrity)
    identity = _run_identity(args, integrity, records)
    run_manifest_path = output_dir / "batch_manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_identity") != identity:
            raise ValueError(
                "Output directory belongs to a different manifest/config selection; "
                "use a new directory."
            )
    else:
        _atomic_json(
            run_manifest_path,
            {
                "schema_version": 1,
                "purpose": "unfiltered_generator_outputs_for_critic_distillation",
                "selection_policy": "none; successful, invalid, and failed generations are retained",
                "critic_enabled": False,
                "rag_enabled": False,
                "record_count": len(records),
                "run_identity": identity,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "future_training_boundary": {
                    "critic_distillation": "data generation only; trainer/schema owned by Critic workstream",
                    "multi_turn": "not implemented; iteration count and transition format are undecided",
                    "grpo": "not implemented; reward definition is undecided",
                },
            },
        )
    if args.dry_run:
        print(f"Dry run passed for {len(records)} critic-distillation records: {output_dir}")
        return

    from svg_agentic_slm.agents.schemas import GenerationRequest
    from svg_agentic_slm.artifacts.generation import load_generation_artifact
    from svg_agentic_slm.factories.generation import (
        build_generation_runtime,
        persist_generation_artifacts,
    )

    results_path = output_dir / "results.jsonl"
    completed = _completed_ids(results_path, retry_failures=args.retry_failures)
    pending = [record for record in records if record["sample_id"] not in completed]
    if not pending:
        print(f"All {len(records)} records are already terminal: {results_path}")
        return

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    first = pending[0]
    first_stem = _artifact_stem(first["sample_id"])
    runtime = build_generation_runtime(
        config_path=args.config,
        model_config_path=args.model_config,
        prompt=_instruction(first),
        enable_rag=False,
        enable_critic=False,
        output_path=artifacts_dir / f"{first_stem}.svg",
        overrides={
            "generation": {
                "orchestration": {"enable_rag": False, "enable_critic": False},
                "render": {"enabled": False},
            }
        },
    )
    runtime.execution_command = list(sys.argv)
    try:
        for position, record in enumerate(pending, 1):
            sample_id = record["sample_id"]
            stem = _artifact_stem(sample_id)
            metadata_path = artifacts_dir / f"{stem}.json"
            if metadata_path.is_file():
                artifact = load_generation_artifact(metadata_path)
                output_record = _artifact_record(record, artifact, recovered=True)
            else:
                run_id = f"critic-distill-{stem}"
                request = GenerationRequest(
                    instruction=_instruction(record),
                    config_overrides=dict(runtime.request.config_overrides),
                    run_id=run_id,
                )
                item_runtime = replace(
                    runtime,
                    request=request,
                    svg_output_path=artifacts_dir / f"{stem}.svg",
                    metadata_output_path=metadata_path,
                    render_output_path=None,
                    run_id=run_id,
                )
                try:
                    result = runtime.orchestrator.run(request)
                    artifacts = persist_generation_artifacts(result=result, runtime=item_runtime)
                    artifact = load_generation_artifact(artifacts.metadata_path)
                    output_record = _artifact_record(record, artifact, recovered=False)
                except Exception as exc:
                    output_record = {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "instruction": _instruction(record),
                        "source_metadata": record.get("metadata", {}),
                        "status": "failed",
                        "is_valid": False,
                        "generated_svg": None,
                        "artifact_metadata_path": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recorded_at_utc": datetime.now(UTC).isoformat(),
                    }
            _append_jsonl_durable(results_path, output_record)
            print(f"[{position}/{len(pending)}] {sample_id}: {output_record['status']}")
    finally:
        runtime.model_backend.unload()


def _instruction(record: dict[str, Any]) -> str:
    return str(record.get("instruction", record.get("description"))).strip()


def _artifact_record(record: dict[str, Any], artifact: Any, *, recovered: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": record["sample_id"],
        "instruction": _instruction(record),
        "source_metadata": record.get("metadata", {}),
        "status": "generated",
        "is_valid": artifact.is_valid,
        "outcome": artifact.outcome,
        "stop_reason": artifact.stop_reason,
        "generated_svg": artifact.svg_path.read_text(encoding="utf-8"),
        "artifact_metadata_path": str(artifact.metadata_path),
        "artifact_svg_path": str(artifact.svg_path),
        "recovered_after_interruption": recovered,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }


def _completed_ids(path: Path, *, retry_failures: bool) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("sample_id"):
                raise ValueError(f"Invalid result record at {path}:{line_number}.")
            if row.get("status") != "failed" or not retry_failures:
                completed.add(str(row["sample_id"]))
    return completed


def _run_identity(args: argparse.Namespace, integrity: dict[str, Any], records: list[dict[str, Any]]) -> str:
    payload = {
        "config": _file_fingerprint(args.config),
        "model_config": _file_fingerprint(args.model_config) if args.model_config else None,
        "split_roles": integrity["roles"],
        "selected_sample_ids": [record["sample_id"] for record in records],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _artifact_stem(sample_id: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in sample_id)[:48]
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{safe or 'sample'}-{digest}"


def _append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
