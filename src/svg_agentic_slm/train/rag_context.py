"""Fail-closed loading of precomputed MMSVG Top-3 references for SFT."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from svg_agentic_slm.rag.mmsvg_chroma_indexer import stable_record_id
from svg_agentic_slm.rag.mmsvg_hybrid import canonical_svg_sha256
from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.svg.validator import safe_svg_element_names

_FINAL_REFERENCE_SCHEMA = {
    "sample_key": "string",
    "final_rank": "int64",
    "item_id": "string",
    "canonical_svg_sha256": "string",
    "retrieval_score": "double",
    "score_kind": "string",
    "selection_reason": "string",
}

_CORPUS_ITEM_SCHEMA = {
    "item_id": "string",
    "dataset_type": "string",
    "record_id": "string",
    "shard": "string",
    "row_index_in_shard": "int64",
    "source": "string",
    "corpus_version": "string",
    "description": "string",
    "detail": "string",
    "svg": "string",
    "canonical_svg_sha256": "string",
    "image_sha256": "string",
}


@dataclass(frozen=True)
class _CorpusItem:
    item_id: str
    dataset_type: str
    record_id: str
    shard: str
    row_index_in_shard: int
    source: str
    corpus_version: str | None
    description: str
    detail: str
    svg: str
    canonical_svg_sha256: str
    image_sha256: str


@dataclass(frozen=True)
class _FinalReference:
    sample_key: str
    final_rank: int
    item_id: str
    canonical_svg_sha256: str
    retrieval_score: float
    score_kind: str
    selection_reason: str


class PrecomputedRAGContextStore:
    """Join normalized final-reference links to trusted corpus payloads."""

    def __init__(
        self,
        *,
        final_references_path: Path,
        corpus_items_path: Path,
        corpus_items: dict[str, _CorpusItem],
        references: dict[str, tuple[_FinalReference, ...]],
        final_references_sha256: str,
        corpus_items_sha256: str,
        manifest_path: Path,
        manifest_sha256: str,
        run_contract_sha256: str,
        instruction_policy: str,
        input_file_sha256s: dict[str, str],
    ) -> None:
        self.final_references_path = final_references_path
        self.corpus_items_path = corpus_items_path
        self._corpus_items = corpus_items
        self._references = references
        self._final_references_sha256 = final_references_sha256
        self._corpus_items_sha256 = corpus_items_sha256
        self.manifest_path = manifest_path
        self._manifest_sha256 = manifest_sha256
        self._run_contract_sha256 = run_contract_sha256
        self.instruction_policy = instruction_policy
        self._input_file_sha256s = dict(input_file_sha256s)

    @classmethod
    def from_parquet(
        cls,
        final_references_path: str | Path,
        corpus_items_path: str | Path,
    ) -> PrecomputedRAGContextStore:
        """Load and validate the two authoritative normalized artifacts."""
        final_path = Path(final_references_path).resolve()
        corpus_path = Path(corpus_items_path).resolve()
        manifest = _load_and_validate_manifest(final_path, corpus_path)
        corpus_rows = _read_parquet(
            corpus_path,
            expected_schema=_CORPUS_ITEM_SCHEMA,
            artifact_name="corpus_items",
        )
        final_rows = _read_parquet(
            final_path,
            expected_schema=_FINAL_REFERENCE_SCHEMA,
            artifact_name="final_references",
        )
        corpus_items = _load_corpus_items(corpus_rows)
        references = _load_final_references(final_rows, corpus_items)
        if len(references) != manifest["sample_rows"]:
            raise ValueError("Artifact sample count differs from manifest.json.")
        if sum(len(values) for values in references.values()) != manifest[
            "final_reference_rows"
        ]:
            raise ValueError("Final-reference count differs from manifest.json.")
        if len(corpus_items) != manifest["corpus_item_rows"]:
            raise ValueError("Corpus-item count differs from manifest.json.")
        return cls(
            final_references_path=final_path,
            corpus_items_path=corpus_path,
            corpus_items=corpus_items,
            references=references,
            final_references_sha256=_file_sha256(final_path),
            corpus_items_sha256=_file_sha256(corpus_path),
            manifest_path=manifest["path"],
            manifest_sha256=manifest["sha256"],
            run_contract_sha256=manifest["run_contract_sha256"],
            instruction_policy=manifest["instruction_policy"],
            input_file_sha256s=manifest["input_file_sha256s"],
        )

    def references_for_record(
        self,
        record: dict[str, Any],
    ) -> list[RetrievedExample]:
        """Return exact ranked Top-3 after rechecking target leakage."""
        sample_key, target_item_id, target_svg_hash = _record_identity(record)
        links = self._references.get(sample_key)
        if links is None:
            raise ValueError(
                f"Precomputed RAG artifacts contain no Top-3 for {sample_key!r}."
            )

        examples: list[RetrievedExample] = []
        for link in links:
            item = self._corpus_items[link.item_id]
            if item.item_id == target_item_id:
                raise ValueError(
                    f"SFT sample {sample_key!r} retrieved its target item ID "
                    f"at rank {link.final_rank}."
                )
            if item.canonical_svg_sha256 == target_svg_hash:
                raise ValueError(
                    f"SFT sample {sample_key!r} retrieved an identical canonical SVG "
                    f"through {item.item_id!r} at rank {link.final_rank}."
                )
            examples.append(
                RetrievedExample(
                    content=item.svg,
                    item_id=item.item_id,
                    source=item.source,
                    description=item.description,
                    detail=item.detail,
                    score=link.retrieval_score,
                    score_kind=link.score_kind,
                    rank=link.final_rank,
                    corpus_version=item.corpus_version,
                    metadata={
                        "retrieval_strategy": (
                            "mmsvg_description_detail_overlap_siglip2_v1"
                        ),
                        "canonical_svg_sha256": item.canonical_svg_sha256,
                        "selection_reason": link.selection_reason,
                        "is_overlap": link.selection_reason == "overlap",
                        "image_sha256": item.image_sha256,
                        "dataset_type": item.dataset_type,
                        "record_id": item.record_id,
                        "shard": item.shard,
                        "row_index_in_shard": item.row_index_in_shard,
                        "offline_precomputed": True,
                    },
                )
            )
        return examples

    def validate_records(self, records: list[dict[str, Any]]) -> None:
        """Validate artifact coverage and leakage before model allocation."""
        seen: set[str] = set()
        for record in records:
            sample_key, _item_id, _target_hash = _record_identity(record)
            if sample_key in seen:
                raise ValueError(f"Duplicate SFT sample_key: {sample_key!r}.")
            seen.add(sample_key)
            self.references_for_record(record)

    def validate_input_files(
        self,
        train_path: str | Path,
        validation_path: str | Path | None,
    ) -> None:
        """Prove retrieval queries came from the same prepared JSONL files."""
        paths = {"train": Path(train_path)}
        if validation_path is not None:
            paths["validation"] = Path(validation_path)
        for split, path in paths.items():
            expected = self._input_file_sha256s[split]
            if not path.is_file():
                raise FileNotFoundError(f"SFT {split} data file not found: {path}")
            if _file_sha256(path) != expected:
                raise ValueError(
                    f"SFT {split} JSONL checksum differs from the precompute contract."
                )

    def provenance(self) -> dict[str, Any]:
        """Return immutable artifact provenance for the training manifest."""
        return {
            "schema_version": 1,
            "expected_references_per_sample": 3,
            "sample_count": len(self._references),
            "corpus_item_count": len(self._corpus_items),
            "final_reference_count": sum(
                len(values) for values in self._references.values()
            ),
            "final_references_path": str(self.final_references_path),
            "final_references_sha256": self._final_references_sha256,
            "corpus_items_path": str(self.corpus_items_path),
            "corpus_items_sha256": self._corpus_items_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self._manifest_sha256,
            "run_contract_sha256": self._run_contract_sha256,
            "instruction_policy": self.instruction_policy,
            "input_file_sha256s": dict(self._input_file_sha256s),
            "leakage_guards": [
                "reference item_id != target stable item_id",
                "reference canonical SVG SHA-256 != target canonical SVG SHA-256",
            ],
        }


def _load_corpus_items(rows: list[dict[str, Any]]) -> dict[str, _CorpusItem]:
    output: dict[str, _CorpusItem] = {}
    for row_number, row in enumerate(rows):
        location = f"corpus_items row {row_number}"
        item_id = _required_text(row.get("item_id"), f"{location}.item_id")
        dataset_type = _required_text(
            row.get("dataset_type"), f"{location}.dataset_type"
        ).casefold()
        record_id = _required_text(row.get("record_id"), f"{location}.record_id")
        if item_id != stable_record_id(dataset_type, record_id):
            raise ValueError(
                f"{location} item_id does not match dataset_type:record_id."
            )
        if item_id in output:
            raise ValueError(f"Duplicate corpus item_id {item_id!r}.")

        svg = _required_text(row.get("svg"), f"{location}.svg")
        if safe_svg_element_names(svg, allow_fragment=True) is None:
            raise ValueError(f"{location} contains an unsafe or malformed SVG.")
        stored_hash = _sha256(
            row.get("canonical_svg_sha256"),
            f"{location}.canonical_svg_sha256",
        )
        if canonical_svg_sha256(svg) != stored_hash:
            raise ValueError(f"{location} canonical hash does not match svg.")

        row_index = row.get("row_index_in_shard")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
            raise ValueError(
                f"{location}.row_index_in_shard must be a non-negative integer."
            )
        output[item_id] = _CorpusItem(
            item_id=item_id,
            dataset_type=dataset_type,
            record_id=record_id,
            shard=_required_text(row.get("shard"), f"{location}.shard"),
            row_index_in_shard=row_index,
            source=_required_text(row.get("source"), f"{location}.source"),
            corpus_version=_optional_text(row.get("corpus_version")),
            description=_required_text(
                row.get("description"), f"{location}.description"
            ),
            detail=_required_text(row.get("detail"), f"{location}.detail"),
            svg=svg,
            canonical_svg_sha256=stored_hash,
            image_sha256=_sha256(
                row.get("image_sha256"), f"{location}.image_sha256"
            ),
        )
    if not output:
        raise ValueError("corpus_items.parquet is empty.")
    return output


def _load_final_references(
    rows: list[dict[str, Any]],
    corpus_items: dict[str, _CorpusItem],
) -> dict[str, tuple[_FinalReference, ...]]:
    grouped: dict[str, list[_FinalReference]] = {}
    for row_number, row in enumerate(rows):
        location = f"final_references row {row_number}"
        sample_key = _sha256(row.get("sample_key"), f"{location}.sample_key")
        item_id = _required_text(row.get("item_id"), f"{location}.item_id")
        rank = row.get("final_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (1, 2, 3):
            raise ValueError(f"{location}.final_rank must be 1, 2, or 3.")

        item = corpus_items.get(item_id)
        if item is None:
            raise ValueError(f"{location} references missing corpus item {item_id!r}.")
        link_hash = _sha256(
            row.get("canonical_svg_sha256"),
            f"{location}.canonical_svg_sha256",
        )
        if link_hash != item.canonical_svg_sha256:
            raise ValueError(f"{location} hash differs from corpus_items.")

        score = row.get("retrieval_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"{location}.retrieval_score must be finite.")
        reason = _required_text(
            row.get("selection_reason"), f"{location}.selection_reason"
        )
        if reason not in {"overlap", "siglip2"}:
            raise ValueError(
                f"{location}.selection_reason must be overlap or siglip2."
            )
        score_kind = _required_text(
            row.get("score_kind"), f"{location}.score_kind"
        )
        expected_score_kind = (
            "hybrid_overlap_priority"
            if reason == "overlap"
            else "siglip2_pair_probability"
        )
        if score_kind != expected_score_kind:
            raise ValueError(
                f"{location}.score_kind is inconsistent with selection_reason."
            )

        grouped.setdefault(sample_key, []).append(
            _FinalReference(
                sample_key=sample_key,
                final_rank=rank,
                item_id=item_id,
                canonical_svg_sha256=link_hash,
                retrieval_score=float(score),
                score_kind=score_kind,
                selection_reason=reason,
            )
        )

    output: dict[str, tuple[_FinalReference, ...]] = {}
    for sample_key, links in grouped.items():
        ordered = sorted(links, key=lambda value: value.final_rank)
        if [link.final_rank for link in ordered] != [1, 2, 3]:
            raise ValueError(
                f"SFT sample {sample_key!r} must have exactly ranks 1, 2, 3."
            )
        if len({link.item_id for link in ordered}) != 3:
            raise ValueError(f"SFT sample {sample_key!r} has duplicate item IDs.")
        if len({link.canonical_svg_sha256 for link in ordered}) != 3:
            raise ValueError(f"SFT sample {sample_key!r} has duplicate SVG hashes.")
        output[sample_key] = tuple(ordered)
    if not output:
        raise ValueError("final_references.parquet is empty.")
    return output


def _record_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SFT record is missing metadata.")
    sample_key = _sha256(
        metadata.get("stable_record_key_sha256"),
        "metadata.stable_record_key_sha256",
    )
    dataset_type = _required_text(
        metadata.get("dataset_type"), "metadata.dataset_type"
    ).casefold()
    record_id = _required_text(metadata.get("record_id"), "metadata.record_id")
    target_item_id = stable_record_id(dataset_type, record_id)
    target_svg = _required_text(record.get("output_svg"), "output_svg")
    target_hash = canonical_svg_sha256(target_svg)
    declared_hash = _sha256(
        metadata.get("canonical_svg_sha256"),
        "metadata.canonical_svg_sha256",
    )
    if target_hash != declared_hash:
        raise ValueError(
            f"SFT sample {sample_key!r} target hash does not match metadata."
        )
    return sample_key, target_item_id, target_hash


def _read_parquet(
    path: Path,
    *,
    expected_schema: dict[str, str],
    artifact_name: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} artifact not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for RAG-conditioned SFT.") from exc
    parquet = pq.ParquetFile(path)
    actual = {field.name: str(field.type) for field in parquet.schema_arrow}
    missing = sorted(set(expected_schema) - set(actual))
    if missing:
        raise ValueError(f"{artifact_name} is missing columns: {', '.join(missing)}")
    mismatches = [
        f"{name}={actual[name]} (expected {expected})"
        for name, expected in expected_schema.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise ValueError(
            f"{artifact_name} has incompatible types: {', '.join(mismatches)}"
        )
    return parquet.read(columns=list(expected_schema)).to_pylist()


def _load_and_validate_manifest(
    final_path: Path,
    corpus_path: Path,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for RAG-conditioned SFT.") from exc

    if final_path.parent != corpus_path.parent:
        raise ValueError("RAG Parquet artifacts must share one publication directory.")
    manifest_path = final_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RAG artifact manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RAG artifact manifest must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported RAG artifact manifest schema_version.")
    if payload.get("dataset") != "mmsvg-generator-sft-hybrid-rag-top3":
        raise ValueError("Unexpected RAG artifact dataset identity.")

    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("RAG artifact manifest is missing run_contract.")
    declared_contract_hash = _sha256(
        payload.get("run_contract_sha256"),
        "manifest.run_contract_sha256",
    )
    if _json_sha256(run_contract) != declared_contract_hash:
        raise ValueError("RAG artifact run_contract checksum mismatch.")
    instruction_policy = _required_text(
        run_contract.get("instruction_policy"),
        "manifest.run_contract.instruction_policy",
    )
    if instruction_policy not in {"prepared", "r0", "r1", "r2"}:
        raise ValueError("Unsupported precompute instruction policy.")
    input_contract = run_contract.get("input")
    if not isinstance(input_contract, dict):
        raise ValueError("RAG run_contract is missing input provenance.")
    input_files = input_contract.get("files")
    if not isinstance(input_files, dict):
        raise ValueError("RAG run_contract is missing input.files provenance.")
    input_file_sha256s: dict[str, str] = {}
    for split in ("train", "validation"):
        entry = input_files.get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"RAG run_contract is missing input.files.{split}.")
        input_file_sha256s[split] = _sha256(
            entry.get("sha256"),
            f"run_contract.input.files.{split}.sha256",
        )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("RAG artifact manifest is missing artifacts.")
    rows: dict[str, int] = {}
    for name, expected_path in (
        ("final_references", final_path),
        ("corpus_items", corpus_path),
    ):
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest is missing artifacts.{name}.")
        filename = _required_text(entry.get("file"), f"artifacts.{name}.file")
        if Path(filename).name != filename:
            raise ValueError(f"artifacts.{name}.file must be a basename.")
        if (manifest_path.parent / filename).resolve() != expected_path:
            raise ValueError(f"Configured {name} path differs from manifest.json.")
        declared_sha = _sha256(
            entry.get("sha256"), f"artifacts.{name}.sha256"
        )
        if _file_sha256(expected_path) != declared_sha:
            raise ValueError(f"{name} checksum differs from manifest.json.")
        declared_rows = entry.get("rows")
        if (
            isinstance(declared_rows, bool)
            or not isinstance(declared_rows, int)
            or declared_rows <= 0
        ):
            raise ValueError(f"artifacts.{name}.rows must be positive.")
        if pq.ParquetFile(expected_path).metadata.num_rows != declared_rows:
            raise ValueError(f"{name} row count differs from manifest.json.")
        rows[name] = declared_rows

    sample_rows = payload.get("sample_rows")
    final_reference_rows = payload.get("final_reference_rows")
    if (
        isinstance(sample_rows, bool)
        or not isinstance(sample_rows, int)
        or sample_rows <= 0
    ):
        raise ValueError("manifest.sample_rows must be positive.")
    if final_reference_rows != sample_rows * 3:
        raise ValueError("manifest.final_reference_rows must equal sample_rows * 3.")
    if rows["final_references"] != final_reference_rows:
        raise ValueError("Manifest final-reference row counts disagree.")

    return {
        "path": manifest_path,
        "sha256": _file_sha256(manifest_path),
        "run_contract_sha256": declared_contract_hash,
        "instruction_policy": instruction_policy,
        "input_file_sha256s": input_file_sha256s,
        "sample_rows": sample_rows,
        "final_reference_rows": final_reference_rows,
        "corpus_item_rows": rows["corpus_items"],
    }


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Optional text values must be strings or null.")
    return value.strip() or None


def _sha256(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
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
