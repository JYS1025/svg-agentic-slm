"""SVGenius-specific candidate benchmark preparation.

SVGenius is intentionally isolated behind this adapter. Its Hugging Face SVG
table and GitHub text-to-SVG caption files have different schemas and revision
histories; no SVGenius-specific join logic belongs in the shared data pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.request import urlopen

from lxml import etree  # type: ignore[import-untyped]

from svg_agentic_slm.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkPreparationResult,
)
from svg_agentic_slm.data.jsonl import write_jsonl

SVGENIUS_DATASET_ID = "xiaoooobai/SVGenius"
SVGENIUS_HF_REVISION = "7d9cd059ee19ec86e12a87e67b82c168f0de65cb"
SVGENIUS_TASK_REVISION = "18d56d738827304e74bb5037c9f9a3445dbbda93"
SVGENIUS_DIFFICULTIES = ("easy", "medium", "hard")
SVGENIUS_ADAPTER_VERSION = "svgenius-text-to-svg-v2"
SVGENIUS_DATA_PARTITION = "candidate_unassigned"
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

CaptionLoader = Callable[[str, str], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class SVGeniusKnownExclusion:
    """One immutable, upstream-specific exception to the strict join."""

    exclusion_id: str
    hf_revision: str
    task_revision: str
    source_split: str
    asset_key: str
    reason_code: str
    reason: str

    def applies_to(
        self,
        *,
        hf_revision: str,
        task_revision: str,
        source_split: str,
    ) -> bool:
        """Return whether this exclusion applies to the selected upstreams."""
        return (
            self.hf_revision == hf_revision
            and self.task_revision == task_revision
            and self.source_split == source_split
        )

    def to_manifest(self) -> dict[str, str]:
        """Serialize the immutable exclusion policy."""
        return {
            "exclusion_id": self.exclusion_id,
            "hf_revision": self.hf_revision,
            "task_revision": self.task_revision,
            "source_split": self.source_split,
            "asset_key": self.asset_key,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


SVGENIUS_KNOWN_EXCLUSIONS = (
    SVGeniusKnownExclusion(
        exclusion_id="missing-caption-medium-page-38-ant-design-48353-icon-95",
        hf_revision=SVGENIUS_HF_REVISION,
        task_revision=SVGENIUS_TASK_REVISION,
        source_split="medium",
        asset_key="page_38_ant_design_48353_icon_95",
        reason_code="missing_official_text_to_svg_caption",
        reason=(
            "The pinned Hugging Face row has no matching caption in the pinned "
            "ZJU-REAL/SVGenius text-to-SVG task files."
        ),
    ),
)


class DatasetLoader(Protocol):
    """Loader contract for one pinned Hugging Face dataset split."""

    def __call__(
        self,
        *,
        dataset_id: str,
        split: str,
        revision: str,
        cache_dir: Path | None,
    ) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class SVGeniusPreparationConfig:
    """Pinned upstream inputs and local output settings."""

    output_dir: Path
    cache_dir: Path | None = None
    hf_revision: str = SVGENIUS_HF_REVISION
    task_revision: str = SVGENIUS_TASK_REVISION
    limit_per_difficulty: int | None = None

    def __post_init__(self) -> None:
        if self.limit_per_difficulty is not None and self.limit_per_difficulty <= 0:
            raise ValueError("limit_per_difficulty must be positive when provided.")
        _validate_pinned_revision(self.hf_revision, field_name="hf_revision")
        _validate_pinned_revision(self.task_revision, field_name="task_revision")


@dataclass(frozen=True)
class SVGeniusJoinStats:
    """Auditable row/caption join counts for one upstream difficulty split."""

    total_rows: int
    selected_rows: int
    total_captions: int
    joined: int
    missing_caption_rows: int
    known_excluded_rows: int
    unexpected_missing_caption_rows: int
    unused_captions: int

    def to_manifest(self) -> dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "selected_rows": self.selected_rows,
            "total_captions": self.total_captions,
            "joined": self.joined,
            "missing_caption_rows": self.missing_caption_rows,
            "known_excluded_rows": self.known_excluded_rows,
            "unexpected_missing_caption_rows": self.unexpected_missing_caption_rows,
            "unused_captions": self.unused_captions,
        }


class SVGeniusAdapter:
    """Join SVGenius caption tasks to the matching reference SVG rows."""

    name = SVGENIUS_ADAPTER_VERSION

    def __init__(
        self,
        *,
        dataset_loader: DatasetLoader | None = None,
        caption_loader: CaptionLoader | None = None,
    ) -> None:
        self._dataset_loader = dataset_loader or _load_huggingface_split
        self._caption_loader = caption_loader or _load_caption_tasks

    def prepare(self, config: SVGeniusPreparationConfig) -> BenchmarkPreparationResult:
        """Download, join, validate, and snapshot the text-to-SVG candidate."""
        examples: list[BenchmarkExample] = []
        records_by_split: dict[str, int] = {}
        join_stats_by_split: dict[str, dict[str, int]] = {}
        applied_exclusions: list[dict[str, Any]] = []
        applicable_exclusions = [
            exclusion
            for exclusion in SVGENIUS_KNOWN_EXCLUSIONS
            if exclusion.hf_revision == config.hf_revision
            and exclusion.task_revision == config.task_revision
        ]

        for difficulty in SVGENIUS_DIFFICULTIES:
            all_rows = list(
                self._dataset_loader(
                    dataset_id=SVGENIUS_DATASET_ID,
                    split=difficulty,
                    revision=config.hf_revision,
                    cache_dir=config.cache_dir,
                )
            )
            captions = list(self._caption_loader(difficulty, config.task_revision))
            rows = all_rows
            if config.limit_per_difficulty is not None:
                rows = rows[: config.limit_per_difficulty]

            split_examples, join_stats, split_exclusions = _join_split(
                rows=rows,
                total_rows=len(all_rows),
                captions=captions,
                difficulty=difficulty,
                hf_revision=config.hf_revision,
                task_revision=config.task_revision,
                require_complete_caption_set=config.limit_per_difficulty is None,
                known_exclusions=applicable_exclusions,
            )
            examples.extend(split_examples)
            applied_exclusions.extend(split_exclusions)
            records_by_split[difficulty] = len(split_examples)
            join_stats_by_split[difficulty] = join_stats.to_manifest()

        output_path = config.output_dir / "text_to_svg.jsonl"
        records = [
            example.to_jsonl_record(_sha256_text(example.reference_svg)) for example in examples
        ]
        write_jsonl(records, output_path)

        manifest_path = config.output_dir / "manifest.json"
        manifest = {
            "adapter": self.name,
            "manifest_schema_version": 2,
            "benchmark_status": "candidate_only",
            "dataset_id": SVGENIUS_DATASET_ID,
            "hf_revision": config.hf_revision,
            "task_repository": "ZJU-REAL/SVGenius",
            "task_revision": config.task_revision,
            "task": "text_to_svg",
            "source_splits": list(SVGENIUS_DIFFICULTIES),
            "excluded_source_splits": ["train"],
            "data_partition": SVGENIUS_DATA_PARTITION,
            "strict": True,
            "limit_per_difficulty": config.limit_per_difficulty,
            "num_records": len(records),
            "records_by_split": records_by_split,
            "join_stats": join_stats_by_split,
            "configured_known_exclusions": [
                exclusion.to_manifest() for exclusion in applicable_exclusions
            ],
            "applied_known_exclusions": applied_exclusions,
            "output_file": output_path.name,
            "output_sha256": _sha256_file(output_path),
            "memory_eligible": False,
            "reference_validation": "well_formed_xml_and_svg_root_only",
            "license_review_required": True,
            "license_note": (
                "The Hugging Face metadata header says Apache-2.0 while the "
                "dataset card body says MIT; resolve before final adoption."
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return BenchmarkPreparationResult(
            adapter=self.name,
            output_path=output_path,
            manifest_path=manifest_path,
            num_records=len(records),
            records_by_split=records_by_split,
        )


def _join_split(
    *,
    rows: Sequence[Mapping[str, Any]],
    total_rows: int,
    captions: Sequence[Mapping[str, Any]],
    difficulty: str,
    hf_revision: str,
    task_revision: str,
    require_complete_caption_set: bool,
    known_exclusions: Sequence[SVGeniusKnownExclusion],
) -> tuple[list[BenchmarkExample], SVGeniusJoinStats, list[dict[str, Any]]]:
    caption_by_key: dict[str, str] = {}
    for caption in captions:
        key = _asset_key(_require_string(caption, "image"))
        if key in caption_by_key:
            raise ValueError(f"Duplicate SVGenius caption asset key: {key}")
        caption_by_key[key] = _extract_instruction(caption)

    examples: list[BenchmarkExample] = []
    missing_caption_keys: list[str] = []
    unexpected_missing_caption_keys: list[str] = []
    applied_exclusions: list[dict[str, Any]] = []
    known_exclusion_by_key = {
        exclusion.asset_key: exclusion
        for exclusion in known_exclusions
        if exclusion.applies_to(
            hf_revision=hf_revision,
            task_revision=task_revision,
            source_split=difficulty,
        )
    }
    row_keys: set[str] = set()
    sample_ids: set[str] = set()
    for row in rows:
        filename = _require_string(row, "filename")
        key = _asset_key(filename)
        if key in row_keys:
            raise ValueError(f"Duplicate SVGenius SVG asset key: {key}")
        row_keys.add(key)

        row_difficulty = row.get("difficulty")
        if row_difficulty is not None and row_difficulty != difficulty:
            raise ValueError(
                f"SVGenius row '{key}' has difficulty {row_difficulty!r}, expected {difficulty!r}."
            )
        instruction = caption_by_key.get(key)
        if instruction is None:
            missing_caption_keys.append(key)
            exclusion = known_exclusion_by_key.get(key)
            if exclusion is None:
                unexpected_missing_caption_keys.append(key)
            else:
                upstream_id = row.get("id")
                applied_exclusions.append(
                    {
                        **exclusion.to_manifest(),
                        "filename": filename,
                        "upstream_id": (
                            str(upstream_id).strip() if upstream_id is not None else None
                        ),
                    }
                )
            continue
        if key in known_exclusion_by_key:
            raise ValueError(
                f"SVGenius known exclusion '{key}' unexpectedly has a matching "
                "caption at the pinned revisions; update the exclusion policy and "
                "adapter version."
            )

        reference_svg = _require_string(row, "svg_code")
        _validate_svg(reference_svg, sample_key=key)
        upstream_id = row.get("id")
        normalized_id = str(upstream_id).strip() if upstream_id is not None else ""
        sample_id = f"svgenius:{difficulty}:{normalized_id or key}"
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate SVGenius sample ID: {sample_id}")
        sample_ids.add(sample_id)
        examples.append(
            BenchmarkExample(
                benchmark_id="svgenius",
                sample_id=sample_id,
                instruction=instruction,
                reference_svg=reference_svg,
                source_split=difficulty,
                difficulty=difficulty,
                source=(
                    f"hf://datasets/{SVGENIUS_DATASET_ID}@{hf_revision}/{difficulty}/{filename}"
                ),
                source_revision=hf_revision,
                data_partition=SVGENIUS_DATA_PARTITION,
                task_source_revision=task_revision,
            )
        )

    unmatched_caption_keys = sorted(set(caption_by_key) - row_keys)
    if unexpected_missing_caption_keys:
        preview = ", ".join(unexpected_missing_caption_keys[:5])
        raise ValueError(
            f"{len(unexpected_missing_caption_keys)} unexpected SVGenius row(s) "
            f"had no matching caption: {preview}"
        )
    if require_complete_caption_set and unmatched_caption_keys:
        preview = ", ".join(unmatched_caption_keys[:5])
        raise ValueError(
            f"{len(unmatched_caption_keys)} SVGenius caption(s) had no matching SVG row: {preview}"
        )
    if len(examples) + len(applied_exclusions) != len(rows):
        raise ValueError(
            "SVGenius strict join did not preserve every selected row or account "
            "for it with a pinned exclusion."
        )
    return (
        examples,
        SVGeniusJoinStats(
            total_rows=total_rows,
            selected_rows=len(rows),
            total_captions=len(captions),
            joined=len(examples),
            missing_caption_rows=len(missing_caption_keys),
            known_excluded_rows=len(applied_exclusions),
            unexpected_missing_caption_rows=len(unexpected_missing_caption_keys),
            unused_captions=len(unmatched_caption_keys),
        ),
        applied_exclusions,
    )


def _validate_pinned_revision(revision: str, *, field_name: str) -> None:
    if not _COMMIT_SHA_PATTERN.fullmatch(revision):
        raise ValueError(f"{field_name} must be an immutable 40-character lowercase commit SHA.")


def _load_huggingface_split(
    *,
    dataset_id: str,
    split: str,
    revision: str,
    cache_dir: Path | None,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for SVGenius.") from exc
    return cast(
        Iterable[Mapping[str, Any]],
        load_dataset(
            dataset_id,
            split=split,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        ),
    )


def _load_caption_tasks(
    difficulty: str,
    task_revision: str,
) -> Sequence[Mapping[str, Any]]:
    url = (
        "https://raw.githubusercontent.com/ZJU-REAL/SVGenius/"
        f"{task_revision}/src/tasks/generation/text_svg/"
        f"{difficulty}_svg_captions.json"
    )
    try:
        with urlopen(url, timeout=60) as response:  # noqa: S310 - fixed trusted host
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Failed to download SVGenius caption tasks from {url}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"SVGenius caption payload for '{difficulty}' must be a list.")
    return payload


def _extract_instruction(record: Mapping[str, Any]) -> str:
    questions = record.get("question")
    if not isinstance(questions, list):
        raise ValueError("SVGenius caption record must contain a question list.")
    instructions = [item.strip() for item in questions if isinstance(item, str) and item.strip()]
    if len(instructions) != 1:
        raise ValueError("SVGenius text-to-SVG record must contain exactly one instruction.")
    return instructions[0]


def _require_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SVGenius field '{key}' must be a non-empty string.")
    return value.strip()


def _asset_key(value: str) -> str:
    return Path(value).stem


def _validate_svg(svg: str, *, sample_key: str) -> None:
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"SVGenius reference '{sample_key}' is not well-formed XML.") from exc
    if etree.QName(root).localname != "svg":
        raise ValueError(f"SVGenius reference '{sample_key}' does not have an SVG root.")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used explicitly by the experiment owner."""
    parser = argparse.ArgumentParser(
        description="Prepare the candidate SVGenius text-to-SVG snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/benchmarks/svgenius"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-revision", default=SVGENIUS_HF_REVISION)
    parser.add_argument("--task-revision", default=SVGENIUS_TASK_REVISION)
    parser.add_argument("--limit-per-difficulty", type=int)
    args = parser.parse_args(argv)

    result = SVGeniusAdapter().prepare(
        SVGeniusPreparationConfig(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            hf_revision=args.hf_revision,
            task_revision=args.task_revision,
            limit_per_difficulty=args.limit_per_difficulty,
        )
    )
    print(f"Prepared {result.num_records} records: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
