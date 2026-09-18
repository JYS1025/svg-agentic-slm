#!/usr/bin/env python3
"""Build a non-truncated Generator SFT cohort using exact model chat lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from svg_agentic_slm.prompts.system_prompts import (  # noqa: E402
    get_svg_generator_system_prompt,
)
from svg_agentic_slm.prompts.text_to_svg import (  # noqa: E402
    build_text_to_svg_prompt,
)
from svg_agentic_slm.train.sft_trainer import _select_instruction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/mmsvg_sft_20k"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument(
        "--instruction-mode",
        choices=("description_only", "mixed_60_detail_40_description", "detail_only"),
        default="description_only",
    )
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-12B-it-qat-q4_0-unquantized",
    )
    parser.add_argument(
        "--revision",
        default="b6ed86275a6a5735884e208bfed95b445a684ca2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_seq_length < 2 or args.batch_size < 1:
        parser.error("--max-seq-length must be >=2 and --batch-size must be positive")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing cohort: {output_dir}. Use a new directory."
        )
    paths = {split: input_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input splits: " + ", ".join(missing))

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Configured processor does not expose a tokenizer.")

    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        split_manifests = {}
        for split, source in paths.items():
            split_manifests[split] = _filter_split(
                source,
                temporary / f"{split}.jsonl",
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                instruction_mode=args.instruction_mode,
                batch_size=args.batch_size,
            )
            print(
                f"{split}: retained {split_manifests[split]['retained_count']}/"
                f"{split_manifests[split]['input_count']} records",
                flush=True,
            )
        manifest = {
            "schema_version": 1,
            "purpose": "exact_non_truncated_generator_sft_length_cohort",
            "model_id": args.model_id,
            "model_revision": args.revision,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocabulary_size": len(tokenizer),
            "instruction_mode": args.instruction_mode,
            "max_seq_length": args.max_seq_length,
            "serialization": "apply_chat_template_then_tokenizer_add_special_tokens_false",
            "truncation": False,
            "input_dir": str(input_dir),
            "splits": split_manifests,
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Published cohort atomically: {output_dir}")


def _filter_split(
    source: Path,
    destination: Path,
    *,
    tokenizer: Any,
    max_seq_length: int,
    instruction_mode: str,
    batch_size: int,
) -> dict[str, Any]:
    system_prompt = get_svg_generator_system_prompt()
    input_count = 0
    retained_count = 0
    dropped_count = 0
    retained_ids: list[str] = []
    dropped_ids: list[str] = []
    retained_lengths: list[int] = []
    dropped_lengths: list[int] = []
    batch: list[dict[str, Any]] = []
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()

    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        for raw_line in input_handle:
            source_digest.update(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {source}.")
            batch.append(record)
            input_count += 1
            if len(batch) < batch_size:
                continue
            retained, dropped = _process_batch(
                batch,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_seq_length=max_seq_length,
                instruction_mode=instruction_mode,
            )
            retained_count += _write_records(
                output_handle,
                retained,
                digest=output_digest,
                ids=retained_ids,
                lengths=retained_lengths,
            )
            dropped_count += _record_dropped(dropped, dropped_ids, dropped_lengths)
            batch.clear()
        if batch:
            retained, dropped = _process_batch(
                batch,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_seq_length=max_seq_length,
                instruction_mode=instruction_mode,
            )
            retained_count += _write_records(
                output_handle,
                retained,
                digest=output_digest,
                ids=retained_ids,
                lengths=retained_lengths,
            )
            dropped_count += _record_dropped(dropped, dropped_ids, dropped_lengths)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if retained_count + dropped_count != input_count or retained_count == 0:
        raise RuntimeError("Internal cohort accounting mismatch or empty retained split.")
    return {
        "source_path": str(source.resolve()),
        "source_sha256": source_digest.hexdigest(),
        "output_file": destination.name,
        "output_sha256": output_digest.hexdigest(),
        "input_count": input_count,
        "retained_count": retained_count,
        "dropped_count": dropped_count,
        "retained_fraction": retained_count / input_count,
        "retained_length": _length_summary(retained_lengths),
        "dropped_length": _length_summary(dropped_lengths),
        "retained_sample_ids_sha256": _ordered_string_hash(retained_ids),
        "dropped_sample_ids": dropped_ids,
        "dropped_sample_ids_sha256": _ordered_string_hash(dropped_ids),
    }


def _process_batch(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    system_prompt: str,
    max_seq_length: int,
    instruction_mode: str,
) -> tuple[list[tuple[dict[str, Any], int]], list[tuple[dict[str, Any], int]]]:
    rendered: list[str] = []
    for record in records:
        instruction = _select_instruction(record, instruction_mode)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_text_to_svg_prompt(instruction)},
            {"role": "assistant", "content": str(record.get("output_svg", ""))},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(text, str) or not text:
            raise RuntimeError("Tokenizer chat template returned invalid text.")
        rendered.append(text)
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    retained = []
    dropped = []
    for record, input_ids in zip(records, encoded):
        length = len(input_ids)
        copied = dict(record)
        metadata = dict(copied.get("metadata", {}))
        metadata["full_chat_token_length"] = length
        metadata["full_chat_tokenizer_model_id"] = tokenizer.name_or_path
        copied["metadata"] = metadata
        target = retained if length <= max_seq_length else dropped
        target.append((copied, length))
    return retained, dropped


def _write_records(
    handle: Any,
    records: list[tuple[dict[str, Any], int]],
    *,
    digest: Any,
    ids: list[str],
    lengths: list[int],
) -> int:
    for record, length in records:
        line = (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        handle.write(line)
        digest.update(line)
        ids.append(_sample_id(record))
        lengths.append(length)
    return len(records)


def _record_dropped(
    records: list[tuple[dict[str, Any], int]],
    ids: list[str],
    lengths: list[int],
) -> int:
    for record, length in records:
        ids.append(_sample_id(record))
        lengths.append(length)
    return len(records)


def _sample_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    dataset_id = str(metadata.get("dataset_id", ""))
    record_id = str(metadata.get("record_id", ""))
    if not dataset_id or not record_id:
        raise ValueError("Every record must provide metadata.dataset_id and metadata.record_id.")
    return f"{dataset_id}::{record_id}"


def _length_summary(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "p50": ordered[int(0.50 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "maximum": ordered[-1],
        "sum": sum(ordered),
    }


def _ordered_string_hash(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
