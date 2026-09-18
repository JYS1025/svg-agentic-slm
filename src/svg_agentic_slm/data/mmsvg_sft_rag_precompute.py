"""Resumable offline MMSVG hybrid-RAG extraction for Generator SFT."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from svg_agentic_slm.rag.mmsvg_chroma_indexer import stable_record_id
from svg_agentic_slm.rag.mmsvg_hybrid import (
    HybridRetrievalRequest,
    HybridRetrievalTrace,
    MMSVGHybridRetriever,
    canonical_svg_sha256,
)
from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_SPLITS = ("train", "validation", "test")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPECTED_SPLIT_ROWS = {"train": 18_000, "validation": 1_000, "test": 1_000}
_INSTRUCTION_POLICIES = frozenset({"prepared", "r0", "r1", "r2"})


class _HybridRetriever(Protocol):
    def retrieve_with_trace(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_item_ids: set[str] | frozenset[str] | None = None,
        exclude_svg_sha256s: set[str] | frozenset[str] | None = None,
    ) -> tuple[list[RetrievedExample], HybridRetrievalTrace]: ...


@dataclass(frozen=True)
class MMSVGSFTRAGPrecomputeConfig:
    """Immutable input/output contract for the fixed MMSVG SFT retrieval run."""

    input_dir: Path
    output_dir: Path
    rag_config_path: Path
    selection_manifest_path: Path
    instruction_policy: str = "r1"
    limit: int | None = None
    resume: bool = True
    expected_sample_count: int | None = 20_000
    retrieval_batch_size: int = 512

    def __post_init__(self) -> None:
        if self.instruction_policy not in _INSTRUCTION_POLICIES:
            raise ValueError(
                "instruction_policy must be one of: prepared, r0, r1, r2."
            )
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided.")
        if self.expected_sample_count is not None and self.expected_sample_count <= 0:
            raise ValueError("expected_sample_count must be positive when provided.")
        if (
            isinstance(self.retrieval_batch_size, bool)
            or not isinstance(self.retrieval_batch_size, int)
            or self.retrieval_batch_size <= 0
        ):
            raise ValueError("retrieval_batch_size must be a positive integer.")


@dataclass(frozen=True)
class _Sample:
    sample_key: str
    split: str
    split_index: int
    global_index: int
    dataset_type: str
    record_id: str
    target_item_id: str
    instruction_source: str
    instruction: str
    description: str
    detail: str
    target_svg: str
    target_svg_sha256: str
    metadata_json: str
    source_file: str
    source_line: int


def precompute_mmsvg_sft_rag(
    config: MMSVGSFTRAGPrecomputeConfig,
    *,
    retriever: _HybridRetriever | None = None,
) -> dict[str, Any]:
    """Precompute exact Top-3 references with target ID/SVG leakage exclusion."""
    input_paths = _resolve_input_paths(config.input_dir)
    samples = _load_samples(input_paths, config.instruction_policy)
    input_provenance = _validate_input_inventory(samples, input_paths, config)
    selection_ids, selection_provenance = _load_selection_contract(
        config.selection_manifest_path,
        expected_sample_count=config.expected_sample_count,
    )
    _validate_fixed_selection(samples, selection_ids)

    rag_wrapper = load_yaml_config(config.rag_config_path)
    rag_config = rag_wrapper.get("rag", rag_wrapper)
    if not isinstance(rag_config, dict):
        raise ValueError("RAG configuration must be a mapping.")

    active_samples = samples[: config.limit] if config.limit is not None else samples
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "input": input_provenance,
        "selection": selection_provenance,
        "rag_config": {
            "path": str(config.rag_config_path.resolve()),
            "sha256": _file_sha256(config.rag_config_path),
            "effective_sha256": _json_sha256(rag_config),
        },
        "instruction_policy": config.instruction_policy,
        "limit": config.limit,
        "source_sample_count": len(samples),
        "selected_sample_count": len(active_samples),
        "algorithm": {
            "branch_top_k": 3,
            "final_top_k": 3,
            "self_exclusion": ["stable_item_id", "canonical_svg_sha256"],
            "selection": "overlap priority, then raw-prompt SigLIP2",
        },
        "execution": {
            "retrieval_batch_size": config.retrieval_batch_size,
            "cuda_oom_backoff": "halve failing batch until one sample",
        },
    }
    contract_sha256 = _json_sha256(run_contract)

    completed = _load_completed_output(config.output_dir, contract_sha256)
    if completed is not None:
        return completed

    work_dir = config.output_dir.parent / f".{config.output_dir.name}.resume"
    checkpoint_dir = work_dir / "checkpoints"
    _prepare_resume_directory(
        work_dir,
        checkpoint_dir,
        run_contract=run_contract,
        contract_sha256=contract_sha256,
        resume=config.resume,
    )

    pending_samples: list[_Sample] = []
    completed_count = 0
    for sample in active_samples:
        checkpoint_path = checkpoint_dir / f"{sample.global_index:08d}.json"
        if checkpoint_path.is_file():
            checkpoint = _read_json(checkpoint_path)
            _validate_checkpoint(sample, checkpoint)
            completed_count += 1
        else:
            pending_samples.append(sample)

    active_retriever = retriever
    if pending_samples and active_retriever is None:
        active_retriever = _build_hybrid_retriever(rag_config)

    started_at = time.perf_counter()
    processed_this_run = 0
    for start in range(0, len(pending_samples), config.retrieval_batch_size):
        sample_batch = pending_samples[start : start + config.retrieval_batch_size]
        if active_retriever is None:
            raise RuntimeError("Hybrid retriever was not initialized.")
        batch_results = _retrieve_sample_batch(active_retriever, sample_batch)
        if len(batch_results) != len(sample_batch):
            raise RuntimeError("Hybrid batch result count does not match the request count.")

        for sample, (examples, trace) in zip(
            sample_batch,
            batch_results,
            strict=True,
        ):
            checkpoint = _build_checkpoint(sample, examples, trace)
            _validate_checkpoint(sample, checkpoint)
            checkpoint_path = checkpoint_dir / f"{sample.global_index:08d}.json"
            _atomic_write_json(checkpoint_path, checkpoint)
            completed_count += 1
            processed_this_run += 1

        elapsed = max(time.perf_counter() - started_at, 1e-9)
        rate = processed_this_run / elapsed
        remaining = len(pending_samples) - processed_this_run
        eta_seconds = remaining / rate if rate > 0 else math.inf
        logger.info(
            "MMSVG SFT RAG precompute: %d/%d checkpointed; "
            "batch=%d, %.2f samples/s, ETA %.1f minutes.",
            completed_count,
            len(active_samples),
            len(sample_batch),
            rate,
            eta_seconds / 60,
        )

    manifest = _publish_artifacts(
        samples=active_samples,
        checkpoint_dir=checkpoint_dir,
        output_dir=config.output_dir,
        run_contract=run_contract,
        contract_sha256=contract_sha256,
    )
    try:
        shutil.rmtree(work_dir)
    except OSError as exc:
        logger.warning("Published successfully but could not remove resume data: %s", exc)
    return manifest


def _build_hybrid_retriever(rag_config: dict[str, Any]) -> MMSVGHybridRetriever:
    # Import lazily so importing the data module does not initialize the heavy
    # generation factory graph or create a data-package circular dependency.
    from svg_agentic_slm.factories.generation import build_rag_retriever

    built = build_rag_retriever(rag_config, index_chroma_corpus=False)
    if not isinstance(built, MMSVGHybridRetriever):
        raise TypeError(
            "Offline MMSVG SFT precompute requires "
            "rag.chromadb.strategy=mmsvg_dual_siglip2."
        )
    return built


def _retrieve_sample_batch(
    retriever: _HybridRetriever,
    samples: list[_Sample],
) -> list[tuple[list[RetrievedExample], HybridRetrievalTrace]]:
    requests = [
        HybridRetrievalRequest(
            query=sample.instruction,
            top_k=3,
            exclude_item_ids=frozenset({sample.target_item_id}),
            exclude_svg_sha256s=frozenset({sample.target_svg_sha256}),
        )
        for sample in samples
    ]
    batch_method = getattr(retriever, "retrieve_many_with_trace", None)
    if not callable(batch_method):
        return [
            retriever.retrieve_with_trace(
                request.query,
                top_k=request.top_k,
                exclude_item_ids=request.exclude_item_ids,
                exclude_svg_sha256s=request.exclude_svg_sha256s,
            )
            for request in requests
        ]

    try:
        results = list(batch_method(requests))
    except Exception as exc:
        if len(samples) <= 1 or not _is_cuda_out_of_memory(exc):
            raise
        midpoint = len(samples) // 2
        logger.warning(
            "CUDA OOM for retrieval batch=%d; retrying as %d + %d.",
            len(samples),
            midpoint,
            len(samples) - midpoint,
        )
        _release_cuda_cache()
        return _retrieve_sample_batch(
            retriever,
            samples[:midpoint],
        ) + _retrieve_sample_batch(
            retriever,
            samples[midpoint:],
        )
    if len(results) != len(samples):
        raise RuntimeError("retrieve_many_with_trace returned an unexpected result count.")
    return results


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = type(current).__name__.casefold()
        message = str(current).casefold()
        if "outofmemory" in name or "cuda out of memory" in message:
            return True
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)
    return False


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_input_paths(input_dir: Path) -> dict[str, Path]:
    resolved = input_dir.resolve()
    paths = {split: resolved / f"{split}.jsonl" for split in DEFAULT_SPLITS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared SFT split files are missing: " + ", ".join(missing)
        )
    return paths


def _load_samples(
    input_paths: dict[str, Path],
    instruction_policy: str,
) -> list[_Sample]:
    samples: list[_Sample] = []
    for split in DEFAULT_SPLITS:
        path = input_paths[split]
        split_index = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"metadata must be an object at {path}:{line_number}."
                    )

                declared_split = metadata.get("split")
                if declared_split is not None and str(declared_split) != split:
                    raise ValueError(
                        f"Split mismatch at {path}:{line_number}: "
                        f"{declared_split!r} != {split!r}."
                    )
                dataset_type = _required_text(
                    metadata.get("dataset_type"),
                    f"{path}:{line_number} metadata.dataset_type",
                ).casefold()
                if dataset_type not in {"icon", "illustration"}:
                    raise ValueError(
                        f"Unsupported dataset_type at {path}:{line_number}: "
                        f"{dataset_type!r}."
                    )
                record_id = _required_text(
                    metadata.get("record_id"),
                    f"{path}:{line_number} metadata.record_id",
                )
                sample_key = _validate_sha256(
                    metadata.get("stable_record_key_sha256"),
                    f"{path}:{line_number} metadata.stable_record_key_sha256",
                )
                description = _required_text(
                    row.get("description"),
                    f"{path}:{line_number} description",
                )
                detail = _required_text(
                    row.get("detail"),
                    f"{path}:{line_number} detail",
                )
                target_svg = _required_text(
                    row.get("output_svg"),
                    f"{path}:{line_number} output_svg",
                )
                target_hash = canonical_svg_sha256(target_svg)
                declared_hash = _validate_sha256(
                    metadata.get("canonical_svg_sha256"),
                    f"{path}:{line_number} metadata.canonical_svg_sha256",
                )
                if target_hash != declared_hash:
                    raise ValueError(
                        f"Canonical target SVG hash mismatch at {path}:{line_number}."
                    )
                instruction_source, instruction = _resolve_instruction(
                    row,
                    metadata,
                    instruction_policy,
                    path=path,
                    line_number=line_number,
                )
                samples.append(
                    _Sample(
                        sample_key=sample_key,
                        split=split,
                        split_index=split_index,
                        global_index=len(samples),
                        dataset_type=dataset_type,
                        record_id=record_id,
                        target_item_id=stable_record_id(dataset_type, record_id),
                        instruction_source=instruction_source,
                        instruction=instruction,
                        description=description,
                        detail=detail,
                        target_svg=target_svg,
                        target_svg_sha256=target_hash,
                        metadata_json=json.dumps(
                            metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        source_file=str(path),
                        source_line=line_number,
                    )
                )
                split_index += 1
    return samples


def _resolve_instruction(
    row: dict[str, Any],
    metadata: dict[str, Any],
    policy: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, str]:
    if policy == "prepared":
        source = _required_text(
            row.get("instruction_source"),
            f"{path}:{line_number} instruction_source",
        )
        instruction = _required_text(
            row.get("instruction"),
            f"{path}:{line_number} instruction",
        )
        return source, instruction

    source: str
    if policy == "r0":
        source = "description"
    elif policy == "r2":
        source = "detail"
    else:
        policies = metadata.get("instruction_policy")
        if not isinstance(policies, dict):
            raise ValueError(
                f"metadata.instruction_policy must be an object at "
                f"{path}:{line_number}."
            )
        value = policies.get("r1_detail_60_description_40")
        if isinstance(value, bool):
            source = "detail" if value else "description"
        elif isinstance(value, str) and value.strip() in {"description", "detail"}:
            # Accept already-produced legacy prepared rows during migration.
            source = value.strip()
        else:
            raise ValueError(
                f"R1 policy at {path}:{line_number} must be a boolean "
                "or legacy description/detail string."
            )
    return source, _required_text(row.get(source), f"{path}:{line_number} {source}")


def _validate_input_inventory(
    samples: list[_Sample],
    input_paths: dict[str, Path],
    config: MMSVGSFTRAGPrecomputeConfig,
) -> dict[str, Any]:
    if config.expected_sample_count is not None and len(samples) != config.expected_sample_count:
        raise RuntimeError(
            f"Prepared SFT input has {len(samples):,} rows; "
            f"expected {config.expected_sample_count:,}."
        )
    for name, values in (
        ("sample keys", [sample.sample_key for sample in samples]),
        ("target item IDs", [sample.target_item_id for sample in samples]),
        ("target canonical SVG hashes", [sample.target_svg_sha256 for sample in samples]),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"Prepared SFT input contains duplicate {name}.")

    split_counts = {
        split: sum(sample.split == split for sample in samples)
        for split in DEFAULT_SPLITS
    }
    if config.expected_sample_count == 20_000 and split_counts != _EXPECTED_SPLIT_ROWS:
        raise RuntimeError(
            f"Fixed 20K split must be {_EXPECTED_SPLIT_ROWS}; got {split_counts}."
        )
    domain_counts = {
        domain: sum(sample.dataset_type == domain for sample in samples)
        for domain in ("icon", "illustration")
    }
    if config.expected_sample_count == 20_000 and domain_counts != {
        "icon": 10_000,
        "illustration": 10_000,
    }:
        raise RuntimeError(
            "Fixed 20K input must contain exactly 10K icon and 10K illustration rows."
        )
    return {
        "files": {
            split: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "rows": split_counts[split],
            }
            for split, path in input_paths.items()
        },
        "total_rows": len(samples),
        "split_rows": split_counts,
        "domain_rows": domain_counts,
    }


def _load_selection_contract(
    manifest_path: Path,
    *,
    expected_sample_count: int | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    manifest = _read_json(manifest_path)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Selection manifest datasets must be a mapping.")
    result: dict[str, tuple[str, ...]] = {}
    file_provenance: dict[str, Any] = {}
    for domain in ("icon", "illustration"):
        entry = datasets.get(domain)
        if not isinstance(entry, dict):
            raise ValueError(f"Selection manifest is missing datasets.{domain}.")
        output_file = _required_text(
            entry.get("output_file"),
            f"selection datasets.{domain}.output_file",
        )
        if Path(output_file).name != output_file:
            raise ValueError("Selection output_file must be a basename.")
        ids_path = manifest_path.resolve().parent / output_file
        if not ids_path.is_file():
            raise FileNotFoundError(f"Selection ID file not found: {ids_path}")
        expected_sha = _validate_sha256(
            entry.get("output_sha256"),
            f"selection datasets.{domain}.output_sha256",
        )
        actual_sha = _file_sha256(ids_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Selection ID checksum mismatch: {ids_path}")
        values = [
            line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines()
        ]
        if any(not value for value in values):
            raise ValueError(f"Selection ID file contains a blank ID: {ids_path}")
        if len(values) != len(set(values)):
            raise RuntimeError(f"Selection ID file contains duplicate IDs: {ids_path}")
        declared_rows = int(entry.get("sample_unique_ids", len(values)))
        if len(values) != declared_rows:
            raise RuntimeError(
                f"Selection ID count mismatch for {domain}: "
                f"{len(values)} != {declared_rows}."
            )
        result[domain] = tuple(values)
        file_provenance[domain] = {
            "path": str(ids_path),
            "rows": len(values),
            "size_bytes": ids_path.stat().st_size,
            "sha256": actual_sha,
        }
    total = sum(len(values) for values in result.values())
    if expected_sample_count is not None and total != expected_sample_count:
        raise RuntimeError(
            f"Selection manifest contains {total:,} IDs; expected "
            f"{expected_sample_count:,}."
        )
    return result, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "files": file_provenance,
        "total_ids": total,
    }


def _validate_fixed_selection(
    samples: list[_Sample],
    selection_ids: dict[str, tuple[str, ...]],
) -> None:
    for domain in ("icon", "illustration"):
        ordered_ids = selection_ids[domain]
        selected_set = set(ordered_ids)
        actual_set = {
            sample.record_id
            for sample in samples
            if sample.dataset_type == domain
        }
        missing = selected_set - actual_set
        extra = actual_set - selected_set
        if missing or extra:
            raise RuntimeError(
                f"Prepared SFT {domain} IDs do not match the fixed selection: "
                f"missing={len(missing)}, extra={len(extra)}."
            )

        if len(ordered_ids) != 10_000:
            continue
        boundaries = {
            "train": (0, 9_000),
            "validation": (9_000, 9_500),
            "test": (9_500, 10_000),
        }
        for split, (start, end) in boundaries.items():
            expected_split = set(ordered_ids[start:end])
            actual_split = {
                sample.record_id
                for sample in samples
                if sample.dataset_type == domain and sample.split == split
            }
            split_missing = expected_split - actual_split
            split_extra = actual_split - expected_split
            if split_missing or split_extra:
                raise RuntimeError(
                    f"Prepared SFT {domain}/{split} IDs do not match the fixed "
                    "ID-file boundary: "
                    f"missing={len(split_missing)}, extra={len(split_extra)}."
                )


def _prepare_resume_directory(
    work_dir: Path,
    checkpoint_dir: Path,
    *,
    run_contract: dict[str, Any],
    contract_sha256: str,
    resume: bool,
) -> None:
    state_path = work_dir / "state.json"
    if work_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Resume directory exists and --no-resume was requested: {work_dir}"
            )
        if not state_path.is_file():
            raise RuntimeError(f"Resume directory has no state.json: {work_dir}")
        state = _read_json(state_path)
        if state.get("run_contract_sha256") != contract_sha256:
            raise RuntimeError(
                "Existing resume state belongs to different inputs or configuration."
            )
    else:
        work_dir.mkdir(parents=True)
        _atomic_write_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "run_contract_sha256": contract_sha256,
                "run_contract": run_contract,
            },
        )
    checkpoint_dir.mkdir(exist_ok=True)


def _build_checkpoint(
    sample: _Sample,
    examples: list[RetrievedExample],
    trace: HybridRetrievalTrace,
) -> dict[str, Any]:
    if not isinstance(trace, HybridRetrievalTrace):
        raise TypeError("retrieve_with_trace must return HybridRetrievalTrace.")
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_key": sample.sample_key,
        "target_item_id": sample.target_item_id,
        "target_svg_sha256": sample.target_svg_sha256,
        "query": sample.instruction,
        "trace": asdict(trace),
        "final_examples": [asdict(example) for example in examples],
    }


def _validate_checkpoint(sample: _Sample, checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Checkpoint schema mismatch for {sample.sample_key}.")
    expected = {
        "sample_key": sample.sample_key,
        "target_item_id": sample.target_item_id,
        "target_svg_sha256": sample.target_svg_sha256,
        "query": sample.instruction,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Checkpoint {key} mismatch for {sample.sample_key}.")

    trace = checkpoint.get("trace")
    references = checkpoint.get("final_examples")
    if not isinstance(trace, dict) or not isinstance(references, list):
        raise RuntimeError(f"Malformed checkpoint for {sample.sample_key}.")
    if len(references) != 3:
        raise RuntimeError(
            f"Hybrid retrieval for {sample.sample_key} returned "
            f"{len(references)} references instead of exactly 3."
        )
    if trace.get("query") != sample.instruction:
        raise RuntimeError(f"Trace query mismatch for {sample.sample_key}.")
    if set(trace.get("excluded_item_ids", [])) != {sample.target_item_id}:
        raise RuntimeError(f"Trace item-ID exclusion mismatch for {sample.sample_key}.")
    if set(trace.get("excluded_svg_sha256s", [])) != {sample.target_svg_sha256}:
        raise RuntimeError(f"Trace SVG exclusion mismatch for {sample.sample_key}.")

    reference_ids: list[str] = []
    reference_hashes: list[str] = []
    for expected_rank, reference in enumerate(references, 1):
        if not isinstance(reference, dict):
            raise RuntimeError(f"Malformed final reference for {sample.sample_key}.")
        item_id = _required_text(
            reference.get("item_id"),
            f"{sample.sample_key} final item_id",
        )
        if reference.get("rank") != expected_rank:
            raise RuntimeError(f"Non-contiguous final ranks for {sample.sample_key}.")
        svg = _required_text(reference.get("content"), f"{sample.sample_key} final SVG")
        svg_hash = canonical_svg_sha256(svg)
        metadata = reference.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Final reference metadata missing for {sample.sample_key}.")
        declared_hash = _validate_sha256(
            metadata.get("canonical_svg_sha256"),
            f"{sample.sample_key} final canonical SVG hash",
        )
        if svg_hash != declared_hash:
            raise RuntimeError(f"Final reference SVG hash mismatch for {sample.sample_key}.")
        score = float(reference.get("score"))
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite retrieval score for {sample.sample_key}.")
        if item_id == sample.target_item_id or svg_hash == sample.target_svg_sha256:
            raise RuntimeError(
                f"Target leakage reached final references for {sample.sample_key}."
            )
        reference_ids.append(item_id)
        reference_hashes.append(svg_hash)

    if len(set(reference_ids)) != 3 or len(set(reference_hashes)) != 3:
        raise RuntimeError(f"Final references are not unique for {sample.sample_key}.")
    if list(trace.get("final_item_ids", [])) != reference_ids:
        raise RuntimeError(f"Trace/final reference mismatch for {sample.sample_key}.")

    candidates = trace.get("candidates")
    if not isinstance(candidates, (list, tuple)):
        raise RuntimeError(f"Trace candidates missing for {sample.sample_key}.")
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError(f"Malformed candidate for {sample.sample_key}.")
        if candidate.get("item_id") == sample.target_item_id:
            raise RuntimeError(f"Target ID survived branch filtering for {sample.sample_key}.")
        if candidate.get("canonical_svg_sha256") == sample.target_svg_sha256:
            raise RuntimeError(f"Target SVG survived branch filtering for {sample.sample_key}.")
        if candidate.get("selected"):
            selected.append(candidate)
    selected.sort(key=lambda value: int(value["final_rank"]))
    if [value["item_id"] for value in selected] != reference_ids:
        raise RuntimeError(f"Selected candidate trace mismatch for {sample.sample_key}.")

    for branch_key in ("description_hits", "detail_hits"):
        hits = trace.get(branch_key)
        if not isinstance(hits, (list, tuple)):
            raise RuntimeError(f"Trace {branch_key} missing for {sample.sample_key}.")
        for hit in hits:
            if not isinstance(hit, dict):
                raise RuntimeError(f"Malformed retrieval hit for {sample.sample_key}.")
            is_target = (
                hit.get("item_id") == sample.target_item_id
                or hit.get("canonical_svg_sha256") == sample.target_svg_sha256
            )
            if is_target and not bool(hit.get("excluded")):
                raise RuntimeError(
                    f"Target hit was not marked excluded for {sample.sample_key}."
                )


def _publish_artifacts(
    *,
    samples: list[_Sample],
    checkpoint_dir: Path,
    output_dir: Path,
    run_contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory appeared during publication: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.publish-",
            dir=output_dir.parent,
        )
    )
    try:
        tables = _collect_artifact_rows(samples, checkpoint_dir)
        artifact_manifest: dict[str, Any] = {}
        for name, (schema, rows) in tables.items():
            path = stage / f"{name}.parquet"
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(
                table,
                path,
                compression="zstd",
                use_dictionary=True,
                version="2.6",
            )
            artifact_manifest[name] = {
                "file": path.name,
                "rows": table.num_rows,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "mmsvg-generator-sft-hybrid-rag-top3",
            "created_at": _utc_now(),
            "run_contract_sha256": contract_sha256,
            "run_contract": run_contract,
            "sample_rows": len(samples),
            "final_reference_rows": len(samples) * 3,
            "corpus_items_scope": (
                "unique payloads used by final Top-3 references; "
                "all six candidate scores remain in candidate_scores"
            ),
            "generator_join": {
                "sample_to_references": (
                    "samples.sample_key = final_references.sample_key"
                ),
                "reference_to_payload": (
                    "final_references.item_id = corpus_items.item_id"
                ),
            },
            "artifacts": artifact_manifest,
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _collect_artifact_rows(
    samples: list[_Sample],
    checkpoint_dir: Path,
) -> dict[str, tuple[pa.Schema, list[dict[str, Any]]]]:
    sample_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    corpus_items: dict[str, dict[str, Any]] = {}

    for sample in samples:
        checkpoint = _read_json(checkpoint_dir / f"{sample.global_index:08d}.json")
        _validate_checkpoint(sample, checkpoint)
        trace = checkpoint["trace"]
        references = checkpoint["final_examples"]
        sample_rows.append(
            {
                "sample_key": sample.sample_key,
                "split": sample.split,
                "split_index": sample.split_index,
                "global_index": sample.global_index,
                "target_item_id": sample.target_item_id,
                "dataset_type": sample.dataset_type,
                "record_id": sample.record_id,
                "instruction_source": sample.instruction_source,
                "instruction": sample.instruction,
                "description": sample.description,
                "detail": sample.detail,
                "target_svg": sample.target_svg,
                "target_svg_sha256": sample.target_svg_sha256,
                "metadata_json": sample.metadata_json,
                "source_file": sample.source_file,
                "source_line": sample.source_line,
            }
        )
        for branch_key in ("description_hits", "detail_hits"):
            for hit in trace[branch_key]:
                hit_rows.append(
                    {
                        "sample_key": sample.sample_key,
                        "branch": hit["branch"],
                        "raw_rank": hit["raw_rank"],
                        "item_id": hit["item_id"],
                        "embedding_score": hit["embedding_score"],
                        "eligible_rank": hit["eligible_rank"],
                        "canonical_svg_sha256": hit["canonical_svg_sha256"],
                        "excluded": hit["excluded"],
                        "exclusion_reason": hit["exclusion_reason"],
                    }
                )
        for candidate in trace["candidates"]:
            candidate_rows.append({"sample_key": sample.sample_key, **candidate})
        for reference in references:
            metadata = reference["metadata"]
            final_rows.append(
                {
                    "sample_key": sample.sample_key,
                    "final_rank": reference["rank"],
                    "item_id": reference["item_id"],
                    "canonical_svg_sha256": metadata["canonical_svg_sha256"],
                    "retrieval_score": reference["score"],
                    "score_kind": reference["score_kind"],
                    "selection_reason": metadata["selection_reason"],
                }
            )
            corpus_row = {
                "item_id": reference["item_id"],
                "dataset_type": _required_text(
                    metadata.get("dataset_type"),
                    f"{reference['item_id']} dataset_type",
                ),
                "record_id": _required_text(
                    metadata.get("record_id"),
                    f"{reference['item_id']} record_id",
                ),
                "shard": _required_text(
                    metadata.get("shard"),
                    f"{reference['item_id']} shard",
                ),
                "row_index_in_shard": _nonnegative_int(
                    metadata.get("row_index_in_shard"),
                    f"{reference['item_id']} row_index_in_shard",
                ),
                "source": reference["source"],
                "corpus_version": reference["corpus_version"],
                "description": reference["description"],
                "detail": reference["detail"],
                "svg": reference["content"],
                "canonical_svg_sha256": metadata["canonical_svg_sha256"],
                "image_sha256": metadata["image_sha256"],
            }
            existing = corpus_items.get(reference["item_id"])
            if existing is not None and existing != corpus_row:
                raise RuntimeError(
                    f"Corpus payload changed across queries for {reference['item_id']}."
                )
            corpus_items[reference["item_id"]] = corpus_row

    return {
        "samples": (_samples_schema(), sample_rows),
        "retrieval_hits": (_retrieval_hits_schema(), hit_rows),
        "candidate_scores": (_candidate_scores_schema(), candidate_rows),
        "final_references": (_final_references_schema(), final_rows),
        "corpus_items": (
            _corpus_items_schema(),
            [corpus_items[item_id] for item_id in sorted(corpus_items)],
        ),
    }


def _samples_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_key", pa.string()),
            ("split", pa.string()),
            ("split_index", pa.int64()),
            ("global_index", pa.int64()),
            ("target_item_id", pa.string()),
            ("dataset_type", pa.string()),
            ("record_id", pa.string()),
            ("instruction_source", pa.string()),
            ("instruction", pa.string()),
            ("description", pa.string()),
            ("detail", pa.string()),
            ("target_svg", pa.string()),
            ("target_svg_sha256", pa.string()),
            ("metadata_json", pa.string()),
            ("source_file", pa.string()),
            ("source_line", pa.int64()),
        ]
    )


def _retrieval_hits_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_key", pa.string()),
            ("branch", pa.string()),
            ("raw_rank", pa.int64()),
            ("item_id", pa.string()),
            ("embedding_score", pa.float64()),
            ("eligible_rank", pa.int64()),
            ("canonical_svg_sha256", pa.string()),
            ("excluded", pa.bool_()),
            ("exclusion_reason", pa.string()),
        ]
    )


def _candidate_scores_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_key", pa.string()),
            ("item_id", pa.string()),
            ("canonical_svg_sha256", pa.string()),
            ("description_rank", pa.int64()),
            ("description_score", pa.float64()),
            ("detail_rank", pa.int64()),
            ("detail_score", pa.float64()),
            ("is_overlap", pa.bool_()),
            ("siglip2_score", pa.float64()),
            ("siglip2_raw_logit", pa.float64()),
            ("image_sha256", pa.string()),
            ("eligible", pa.bool_()),
            ("exclusion_reason", pa.string()),
            ("selected", pa.bool_()),
            ("final_rank", pa.int64()),
            ("selection_reason", pa.string()),
        ]
    )


def _final_references_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_key", pa.string()),
            ("final_rank", pa.int64()),
            ("item_id", pa.string()),
            ("canonical_svg_sha256", pa.string()),
            ("retrieval_score", pa.float64()),
            ("score_kind", pa.string()),
            ("selection_reason", pa.string()),
        ]
    )


def _corpus_items_schema() -> pa.Schema:
    return pa.schema(
        [
            ("item_id", pa.string()),
            ("dataset_type", pa.string()),
            ("record_id", pa.string()),
            ("shard", pa.string()),
            ("row_index_in_shard", pa.int64()),
            ("source", pa.string()),
            ("corpus_version", pa.string()),
            ("description", pa.string()),
            ("detail", pa.string()),
            ("svg", pa.string()),
            ("canonical_svg_sha256", pa.string()),
            ("image_sha256", pa.string()),
        ]
    )


def _load_completed_output(
    output_dir: Path,
    contract_sha256: str,
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Existing output has no manifest: {output_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("run_contract_sha256") != contract_sha256:
        raise RuntimeError("Existing output belongs to a different run contract.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Existing output manifest has no artifacts mapping.")
    for entry in artifacts.values():
        if not isinstance(entry, dict):
            raise RuntimeError("Malformed existing artifact manifest.")
        path = output_dir / str(entry.get("file"))
        if not path.is_file() or _file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"Existing artifact checksum mismatch: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(entry.get("rows", -1)):
            raise RuntimeError(f"Existing artifact row-count mismatch: {path}")
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validate_sha256(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256.")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
