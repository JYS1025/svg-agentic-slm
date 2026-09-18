"""Dual-index MMSVG retrieval with overlap priority and SigLIP2 reranking."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from svg_agentic_slm.models.image_text_similarity import (
    BaseImageTextSimilarityScorer,
    ImageTextSimilarityEvidence,
)
from svg_agentic_slm.rag.base import BaseRetriever
from svg_agentic_slm.rag.chroma_store import (
    ChromaQueryHit,
    ChromaRetriever,
    MMSVGCorpusRecord,
)
from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.svg.validator import safe_svg_element_names

logger = logging.getLogger(__name__)

MMSVG_HYBRID_METADATA_KEYS = frozenset(
    {
        "retrieval_strategy",
        "row_index_in_shard",
        "shard",
        "record_id",
        "dataset_type",
        "canonical_svg_sha256",
        "description_rank",
        "description_score",
        "detail_rank",
        "detail_score",
        "is_overlap",
        "siglip2_score",
        "siglip2_raw_logit",
        "siglip2_model_id",
        "siglip2_model_revision",
        "image_sha256",
        "selection_reason",
    }
)


@dataclass(frozen=True)
class HybridHitTrace:
    """Auditable outcome for one raw Chroma hit."""

    branch: str
    raw_rank: int
    item_id: str
    embedding_score: float
    eligible_rank: int | None
    canonical_svg_sha256: str | None
    excluded: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class HybridCandidateTrace:
    """Merged candidate features retained without copying image bytes."""

    item_id: str
    canonical_svg_sha256: str
    description_rank: int | None
    description_score: float | None
    detail_rank: int | None
    detail_score: float | None
    is_overlap: bool
    siglip2_score: float | None
    siglip2_raw_logit: float | None
    image_sha256: str
    eligible: bool
    exclusion_reason: str | None
    selected: bool
    final_rank: int | None
    selection_reason: str | None


@dataclass(frozen=True)
class HybridRetrievalTrace:
    """Complete deterministic trace for one hybrid retrieval request."""

    query: str
    search_k: int
    branch_top_k: int
    final_top_k: int
    excluded_item_ids: tuple[str, ...]
    excluded_svg_sha256s: tuple[str, ...]
    description_hits: tuple[HybridHitTrace, ...]
    detail_hits: tuple[HybridHitTrace, ...]
    candidates: tuple[HybridCandidateTrace, ...]
    final_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class HybridRetrievalRequest:
    """One ordered request for batched MMSVG hybrid retrieval."""

    query: str
    top_k: int = 3
    exclude_item_ids: frozenset[str] = frozenset()
    exclude_svg_sha256s: frozenset[str] = frozenset()


@dataclass
class _Candidate:
    record: MMSVGCorpusRecord
    canonical_svg_sha256: str
    description_hit: ChromaQueryHit | None = None
    detail_hit: ChromaQueryHit | None = None
    siglip2: ImageTextSimilarityEvidence | None = None

    @property
    def is_overlap(self) -> bool:
        return self.description_hit is not None and self.detail_hit is not None

    @property
    def best_embedding_rank(self) -> int:
        ranks = [hit.raw_rank for hit in (self.description_hit, self.detail_hit) if hit is not None]
        return min(ranks)

    @property
    def embedding_rank_sum(self) -> int:
        return sum(
            hit.raw_rank for hit in (self.description_hit, self.detail_hit) if hit is not None
        )


@dataclass(frozen=True)
class _NormalizedRequest:
    query: str
    top_k: int
    excluded_item_ids: frozenset[str]
    excluded_svg_sha256s: frozenset[str]


@dataclass
class _PendingRetrieval:
    request: _NormalizedRequest
    description_trace: list[HybridHitTrace]
    detail_trace: list[HybridHitTrace]
    unique_candidates: list[_Candidate]
    removed_duplicate_candidates: list[_Candidate]
    overlaps: list[_Candidate]
    remaining: list[_Candidate]
    remaining_slots: int


class MMSVGHybridRetriever(BaseRetriever):
    """Retrieve description Top-3 + detail Top-3, then select final Top-3.

    The same Qwen query embedding is reused for both compatible Chroma indexes.
    Exact target IDs and canonical target SVG hashes are removed before branch
    Top-3 eligibility and before overlap detection.
    """

    def __init__(
        self,
        *,
        description_retriever: ChromaRetriever,
        detail_retriever: ChromaRetriever,
        similarity_scorer: BaseImageTextSimilarityScorer,
        branch_top_k: int = 3,
        search_k: int = 16,
    ) -> None:
        if branch_top_k <= 0:
            raise ValueError("branch_top_k must be positive.")
        if search_k < branch_top_k:
            raise ValueError("search_k must be at least branch_top_k.")
        self._description_retriever = description_retriever
        self._detail_retriever = detail_retriever
        self._similarity_scorer = similarity_scorer
        self._branch_top_k = branch_top_k
        self._search_k = search_k
        self._contracts_validated = False

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        raise RuntimeError("MMSVG hybrid indexes are managed by the precomputed indexer.")

    def clear(self) -> None:
        raise RuntimeError("Refusing to clear managed MMSVG hybrid indexes.")

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedExample]:
        examples, _trace = self.retrieve_with_trace(query, top_k=top_k)
        return examples

    def retrieve_with_trace(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_item_ids: set[str] | frozenset[str] | None = None,
        exclude_svg_sha256s: set[str] | frozenset[str] | None = None,
    ) -> tuple[list[RetrievedExample], HybridRetrievalTrace]:
        """Retrieve with fail-closed exclusions and return an offline artifact trace."""
        request = self._normalize_request(
            HybridRetrievalRequest(
                query=query,
                top_k=top_k,
                exclude_item_ids=frozenset(exclude_item_ids or ()),
                exclude_svg_sha256s=frozenset(exclude_svg_sha256s or ()),
            )
        )
        query_embedding = self._description_retriever.encode_query(request.query)
        self._validate_index_contracts()
        description_hits = self._description_retriever.query_hits_by_embedding(
            query_embedding,
            n_results=self._search_k,
        )
        detail_hits = self._detail_retriever.query_hits_by_embedding(
            query_embedding,
            n_results=self._search_k,
        )
        pending = self._prepare_pending(request, description_hits, detail_hits)
        if pending.remaining and pending.remaining_slots:
            if not self._similarity_scorer.is_loaded():
                self._similarity_scorer.load_model()
            evidence = self._similarity_scorer.score_batch(
                request.query,
                [candidate.record.image_png for candidate in pending.remaining],
                attempt_ids=[candidate.record.item_id for candidate in pending.remaining],
            )
            self._assign_siglip2(pending.remaining, evidence)
            self._sort_remaining(pending.remaining)
        return self._finalize_pending(pending)

    def retrieve_many_with_trace(
        self,
        requests: Sequence[HybridRetrievalRequest],
    ) -> list[tuple[list[RetrievedExample], HybridRetrievalTrace]]:
        """Retrieve an ordered request batch with shared Qwen and SigLIP2 forwards."""
        if isinstance(requests, (str, bytes, bytearray)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a sequence of HybridRetrievalRequest.")
        ordered_requests = list(requests)
        if not ordered_requests:
            return []
        if not all(isinstance(request, HybridRetrievalRequest) for request in ordered_requests):
            raise TypeError("requests must contain only HybridRetrievalRequest values.")
        normalized_requests = [self._normalize_request(request) for request in ordered_requests]

        self._validate_index_contracts()
        query_embeddings = self._description_retriever.encode_queries(
            [request.query for request in normalized_requests]
        )
        if len(query_embeddings) != len(normalized_requests):
            raise RuntimeError("Qwen batch result count does not match requests.")
        description_hit_batches = self._description_retriever.query_hits_many_by_embeddings(
            query_embeddings,
            n_results=self._search_k,
        )
        detail_hit_batches = self._detail_retriever.query_hits_many_by_embeddings(
            query_embeddings,
            n_results=self._search_k,
        )
        if len(description_hit_batches) != len(normalized_requests):
            raise RuntimeError("Description Chroma batch result count does not match requests.")
        if len(detail_hit_batches) != len(normalized_requests):
            raise RuntimeError("Detail Chroma batch result count does not match requests.")

        pending_requests = [
            self._prepare_pending(request, description_hits, detail_hits)
            for request, description_hits, detail_hits in zip(
                normalized_requests,
                description_hit_batches,
                detail_hit_batches,
                strict=True,
            )
        ]
        pairs = [
            (pending, candidate)
            for pending in pending_requests
            if pending.remaining_slots
            for candidate in pending.remaining
        ]
        if pairs:
            if not self._similarity_scorer.is_loaded():
                self._similarity_scorer.load_model()
            evidence = self._similarity_scorer.score_pairs_batch(
                [pending.request.query for pending, _candidate in pairs],
                [candidate.record.image_png for _pending, candidate in pairs],
                attempt_ids=[candidate.record.item_id for _pending, candidate in pairs],
            )
            if len(evidence) != len(pairs):
                raise RuntimeError("SigLIP2 pair batch result count does not match candidates.")
            for (pending, candidate), item_evidence in zip(pairs, evidence, strict=True):
                self._assign_siglip2([candidate], [item_evidence])
            for pending in pending_requests:
                if pending.remaining and pending.remaining_slots:
                    self._sort_remaining(pending.remaining)
        return [self._finalize_pending(pending) for pending in pending_requests]

    def _normalize_request(
        self,
        request: HybridRetrievalRequest,
    ) -> _NormalizedRequest:
        normalized_query = request.query.strip()
        if not normalized_query:
            raise ValueError("query must be non-empty.")
        if request.top_k <= 0 or request.top_k > self._branch_top_k:
            raise ValueError("top_k must be between 1 and branch_top_k.")
        excluded_ids = frozenset(
            item_id.strip() for item_id in request.exclude_item_ids if item_id.strip()
        )
        excluded_hashes = frozenset(
            _validate_sha256(value) for value in request.exclude_svg_sha256s
        )
        return _NormalizedRequest(
            query=normalized_query,
            top_k=request.top_k,
            excluded_item_ids=excluded_ids,
            excluded_svg_sha256s=excluded_hashes,
        )

    def _prepare_pending(
        self,
        request: _NormalizedRequest,
        description_hits: list[ChromaQueryHit],
        detail_hits: list[ChromaQueryHit],
    ) -> _PendingRetrieval:
        description_candidates, description_trace = self._select_branch(
            branch="description",
            store=self._description_retriever,
            hits=description_hits,
            excluded_ids=request.excluded_item_ids,
            excluded_hashes=request.excluded_svg_sha256s,
        )
        detail_candidates, detail_trace = self._select_branch(
            branch="detail",
            store=self._detail_retriever,
            hits=detail_hits,
            excluded_ids=request.excluded_item_ids,
            excluded_hashes=request.excluded_svg_sha256s,
        )
        merged = self._merge_candidates(description_candidates, detail_candidates)
        unique_candidates, removed_duplicate_candidates = _deduplicate_candidate_svgs(merged)
        overlaps = sorted(
            (candidate for candidate in unique_candidates if candidate.is_overlap),
            key=lambda candidate: (
                candidate.embedding_rank_sum,
                candidate.best_embedding_rank,
                candidate.record.item_id,
            ),
        )
        remaining = [candidate for candidate in unique_candidates if not candidate.is_overlap]
        return _PendingRetrieval(
            request=request,
            description_trace=description_trace,
            detail_trace=detail_trace,
            unique_candidates=unique_candidates,
            removed_duplicate_candidates=removed_duplicate_candidates,
            overlaps=overlaps,
            remaining=remaining,
            remaining_slots=max(0, request.top_k - len(overlaps)),
        )

    @staticmethod
    def _assign_siglip2(
        candidates: Sequence[_Candidate],
        evidence: Sequence[ImageTextSimilarityEvidence],
    ) -> None:
        if len(evidence) != len(candidates):
            raise RuntimeError("SigLIP2 batch result count does not match candidates.")
        for candidate, item_evidence in zip(candidates, evidence, strict=True):
            if item_evidence.attempt_id != candidate.record.item_id:
                raise RuntimeError("SigLIP2 batch result order does not match candidates.")
            candidate.siglip2 = item_evidence

    @staticmethod
    def _sort_remaining(candidates: list[_Candidate]) -> None:
        candidates.sort(
            key=lambda candidate: (
                -float(candidate.siglip2.score if candidate.siglip2 else -1.0),
                candidate.best_embedding_rank,
                candidate.embedding_rank_sum,
                candidate.record.item_id,
            )
        )

    def _finalize_pending(
        self,
        pending: _PendingRetrieval,
    ) -> tuple[list[RetrievedExample], HybridRetrievalTrace]:
        selected = (pending.overlaps + pending.remaining[: pending.remaining_slots])[
            : pending.request.top_k
        ]
        selected_ids = {candidate.record.item_id for candidate in selected}
        final_ranks = {
            candidate.record.item_id: index for index, candidate in enumerate(selected, 1)
        }
        examples = [
            _to_retrieved_example(candidate, final_rank=index)
            for index, candidate in enumerate(selected, 1)
        ]
        candidate_trace = [
            _to_candidate_trace(
                candidate,
                selected=candidate.record.item_id in selected_ids,
                final_rank=final_ranks.get(candidate.record.item_id),
                eligible=True,
                exclusion_reason=None,
            )
            for candidate in pending.unique_candidates
        ]
        candidate_trace.extend(
            _to_candidate_trace(
                candidate,
                selected=False,
                final_rank=None,
                eligible=False,
                exclusion_reason="duplicate_candidate_svg",
            )
            for candidate in pending.removed_duplicate_candidates
        )
        trace = HybridRetrievalTrace(
            query=pending.request.query,
            search_k=self._search_k,
            branch_top_k=self._branch_top_k,
            final_top_k=pending.request.top_k,
            excluded_item_ids=tuple(sorted(pending.request.excluded_item_ids)),
            excluded_svg_sha256s=tuple(sorted(pending.request.excluded_svg_sha256s)),
            description_hits=tuple(pending.description_trace),
            detail_hits=tuple(pending.detail_trace),
            candidates=tuple(
                sorted(
                    candidate_trace,
                    key=lambda value: (
                        not value.eligible,
                        value.final_rank if value.final_rank is not None else 10**9,
                        value.item_id,
                    ),
                )
            ),
            final_item_ids=tuple(example.item_id for example in examples),
        )
        return examples, trace

    def _validate_index_contracts(self) -> None:
        if self._contracts_validated:
            return
        description_contract = self._description_retriever.index_contract()
        detail_contract = self._detail_retriever.index_contract()
        mismatches = [
            key
            for key in description_contract
            if description_contract.get(key) != detail_contract.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Description/detail Chroma indexes are incompatible: "
                + ", ".join(sorted(mismatches))
            )
        self._contracts_validated = True

    def _select_branch(
        self,
        *,
        branch: str,
        store: ChromaRetriever,
        hits: list[ChromaQueryHit],
        excluded_ids: frozenset[str],
        excluded_hashes: frozenset[str],
    ) -> tuple[list[_Candidate], list[HybridHitTrace]]:
        selected: list[_Candidate] = []
        traces: list[HybridHitTrace] = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        for hit in hits:
            if len(selected) >= self._branch_top_k:
                break
            exclusion_reason: str | None = None
            canonical_hash: str | None = None
            record: MMSVGCorpusRecord | None = None
            if hit.item_id in excluded_ids:
                exclusion_reason = "target_item_id"
            else:
                try:
                    record = store.hydrate_mmsvg_hit(hit)
                    canonical_hash = canonical_svg_sha256(record.svg)
                except Exception as exc:
                    logger.warning("Skipped invalid %s MMSVG hit %s: %s", branch, hit.item_id, exc)
                    exclusion_reason = "invalid_payload"
            if (
                exclusion_reason is None
                and canonical_hash is not None
                and canonical_hash in excluded_hashes
            ):
                exclusion_reason = "target_svg"
            if exclusion_reason is None and record is not None:
                if safe_svg_element_names(record.svg, allow_fragment=True) is None:
                    exclusion_reason = "unsafe_svg"
                elif hit.item_id in seen_ids:
                    exclusion_reason = "duplicate_item_id"
                elif canonical_hash in seen_hashes:
                    exclusion_reason = "duplicate_branch_svg"

            eligible_rank: int | None = None
            if exclusion_reason is None and record is not None and canonical_hash is not None:
                seen_ids.add(hit.item_id)
                seen_hashes.add(canonical_hash)
                eligible_rank = len(selected) + 1
                candidate = _Candidate(
                    record=record,
                    canonical_svg_sha256=canonical_hash,
                )
                if branch == "description":
                    candidate.description_hit = hit
                else:
                    candidate.detail_hit = hit
                selected.append(candidate)
            traces.append(
                HybridHitTrace(
                    branch=branch,
                    raw_rank=hit.raw_rank,
                    item_id=hit.item_id,
                    embedding_score=hit.score,
                    eligible_rank=eligible_rank,
                    canonical_svg_sha256=canonical_hash,
                    excluded=exclusion_reason is not None,
                    exclusion_reason=exclusion_reason,
                )
            )
        return selected, traces

    @staticmethod
    def _merge_candidates(
        description: list[_Candidate],
        detail: list[_Candidate],
    ) -> list[_Candidate]:
        merged: dict[str, _Candidate] = {}
        for branch_candidates in (description, detail):
            for candidate in branch_candidates:
                existing = merged.get(candidate.record.item_id)
                if existing is None:
                    merged[candidate.record.item_id] = candidate
                    continue
                if existing.canonical_svg_sha256 != candidate.canonical_svg_sha256:
                    raise RuntimeError(
                        "Dual indexes resolved one item ID to different SVG payloads."
                    )
                if existing.record.description != candidate.record.description:
                    raise RuntimeError(
                        "Dual indexes resolved one item ID to different descriptions."
                    )
                if existing.record.detail != candidate.record.detail:
                    raise RuntimeError("Dual indexes resolved one item ID to different details.")
                existing.description_hit = existing.description_hit or candidate.description_hit
                existing.detail_hit = existing.detail_hit or candidate.detail_hit
        return list(merged.values())


def canonical_svg_sha256(svg: str) -> str:
    """Hash canonical XML exactly as the MMSVG SFT preparation contract does."""
    from lxml import etree  # type: ignore[import-untyped]

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    canonical = etree.tostring(root, method="c14n", with_comments=False)
    return hashlib.sha256(canonical).hexdigest()


def _deduplicate_candidate_svgs(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], list[_Candidate]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            not candidate.is_overlap,
            candidate.embedding_rank_sum,
            candidate.best_embedding_rank,
            candidate.record.item_id,
        ),
    )
    kept: list[_Candidate] = []
    removed: list[_Candidate] = []
    seen_hashes: set[str] = set()
    for candidate in ordered:
        if candidate.canonical_svg_sha256 in seen_hashes:
            removed.append(candidate)
            continue
        seen_hashes.add(candidate.canonical_svg_sha256)
        kept.append(candidate)
    return kept, removed


def _to_retrieved_example(candidate: _Candidate, *, final_rank: int) -> RetrievedExample:
    evidence = candidate.siglip2
    selection_reason = "overlap" if candidate.is_overlap else "siglip2"
    embedding_scores = [
        hit.score for hit in (candidate.description_hit, candidate.detail_hit) if hit is not None
    ]
    score = max(embedding_scores) if evidence is None else evidence.score
    metadata = {
        "retrieval_strategy": "mmsvg_description_detail_overlap_siglip2_v1",
        "dataset_type": candidate.record.metadata.get("dataset_type"),
        "record_id": candidate.record.metadata.get("record_id"),
        "shard": candidate.record.metadata.get("shard"),
        "row_index_in_shard": candidate.record.metadata.get("row_index_in_shard"),
        "canonical_svg_sha256": candidate.canonical_svg_sha256,
        "description_rank": (
            candidate.description_hit.raw_rank if candidate.description_hit else None
        ),
        "description_score": (
            candidate.description_hit.score if candidate.description_hit else None
        ),
        "detail_rank": candidate.detail_hit.raw_rank if candidate.detail_hit else None,
        "detail_score": candidate.detail_hit.score if candidate.detail_hit else None,
        "is_overlap": candidate.is_overlap,
        "siglip2_score": evidence.score if evidence else None,
        "siglip2_raw_logit": evidence.raw_logit if evidence else None,
        "siglip2_model_id": evidence.model_id if evidence else None,
        "siglip2_model_revision": evidence.model_revision if evidence else None,
        "image_sha256": (
            evidence.image_sha256
            if evidence
            else hashlib.sha256(candidate.record.image_png).hexdigest()
        ),
        "selection_reason": selection_reason,
    }
    return RetrievedExample(
        content=candidate.record.svg,
        item_id=candidate.record.item_id,
        source=candidate.record.source,
        description=candidate.record.description,
        detail=candidate.record.detail,
        score=float(score),
        score_kind=(
            "hybrid_overlap_priority" if candidate.is_overlap else "siglip2_pair_probability"
        ),
        rank=final_rank,
        corpus_version=candidate.record.corpus_version,
        metadata=metadata,
    )


def _to_candidate_trace(
    candidate: _Candidate,
    *,
    selected: bool,
    final_rank: int | None,
    eligible: bool,
    exclusion_reason: str | None,
) -> HybridCandidateTrace:
    evidence = candidate.siglip2
    return HybridCandidateTrace(
        item_id=candidate.record.item_id,
        canonical_svg_sha256=candidate.canonical_svg_sha256,
        description_rank=(
            candidate.description_hit.raw_rank if candidate.description_hit else None
        ),
        description_score=(candidate.description_hit.score if candidate.description_hit else None),
        detail_rank=candidate.detail_hit.raw_rank if candidate.detail_hit else None,
        detail_score=candidate.detail_hit.score if candidate.detail_hit else None,
        is_overlap=candidate.is_overlap,
        siglip2_score=evidence.score if evidence else None,
        siglip2_raw_logit=evidence.raw_logit if evidence else None,
        image_sha256=(
            evidence.image_sha256
            if evidence
            else hashlib.sha256(candidate.record.image_png).hexdigest()
        ),
        eligible=eligible,
        exclusion_reason=exclusion_reason,
        selected=selected,
        final_rank=final_rank,
        selection_reason=(("overlap" if candidate.is_overlap else "siglip2") if selected else None),
    )


def _validate_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("exclude_svg_sha256s must contain lowercase SHA-256 values.")
    return normalized
