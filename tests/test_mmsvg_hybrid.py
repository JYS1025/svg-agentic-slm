"""Tests for description/detail MMSVG hybrid retrieval."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

from PIL import Image

from svg_agentic_slm.models.image_text_similarity import (
    SIGLIP2_PAIR_PROBABILITY,
    BaseImageTextSimilarityScorer,
    ImageTextSimilarityEvidence,
)
from svg_agentic_slm.rag.chroma_store import (
    ChromaQueryHit,
    MMSVGCorpusRecord,
)
from svg_agentic_slm.rag.mmsvg_hybrid import (
    HybridRetrievalRequest,
    MMSVGHybridRetriever,
    canonical_svg_sha256,
)


class _Store:
    def __init__(
        self,
        *,
        branch: str,
        hits: list[ChromaQueryHit],
        records: dict[str, MMSVGCorpusRecord],
    ) -> None:
        self.branch = branch
        self.hits = hits
        self.records = records
        self.encode_calls: list[str] = []
        self.query_embeddings: list[list[float]] = []
        self.encode_many_calls: list[list[str]] = []
        self.query_embedding_batches: list[list[list[float]]] = []

    def encode_query(self, query: str) -> list[float]:
        self.encode_calls.append(query)
        return [0.1, 0.2, 0.3, 0.4]

    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        self.encode_many_calls.append(list(queries))
        return [[float(index), 0.2, 0.3, 0.4] for index, _query in enumerate(queries, 1)]

    def index_contract(self) -> dict[str, Any]:
        return {
            "index_schema_version": 1,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": "a" * 40,
            "embedding_dimension": 4,
            "normalized_embeddings": True,
            "query_instruction": "retrieve SVG examples",
            "distance": "cosine",
            "sources": [{"dataset_type": "icon", "expected_rows": 10}],
        }

    def query_hits_by_embedding(
        self,
        query_embedding: list[float],
        *,
        n_results: int,
    ) -> list[ChromaQueryHit]:
        self.query_embeddings.append(query_embedding)
        return self.hits[:n_results]

    def query_hits_many_by_embeddings(
        self,
        query_embeddings: list[list[float]],
        *,
        n_results: int,
    ) -> list[list[ChromaQueryHit]]:
        self.query_embedding_batches.append([list(embedding) for embedding in query_embeddings])
        return [self.hits[:n_results] for _embedding in query_embeddings]

    def hydrate_mmsvg_hit(self, hit: ChromaQueryHit) -> MMSVGCorpusRecord:
        return self.records[hit.item_id]


class _Scorer(BaseImageTextSimilarityScorer):
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.loaded = False
        self.batch_calls: list[tuple[str, list[bytes], list[str]]] = []
        self.pair_batch_calls: list[tuple[list[str], list[bytes], list[str]]] = []

    def load_model(self) -> None:
        self.loaded = True

    def unload_model(self) -> None:
        self.loaded = False

    def is_loaded(self) -> bool:
        return self.loaded

    def score(
        self,
        instruction: str,
        image_png: bytes,
        *,
        attempt_id: str,
    ) -> ImageTextSimilarityEvidence:
        return self.score_batch(
            instruction,
            [image_png],
            attempt_ids=[attempt_id],
        )[0]

    def score_batch(
        self,
        instruction: str,
        image_pngs: list[bytes],
        *,
        attempt_ids: list[str],
    ) -> list[ImageTextSimilarityEvidence]:
        self.batch_calls.append((instruction, list(image_pngs), list(attempt_ids)))
        return [
            ImageTextSimilarityEvidence(
                attempt_id=item_id,
                metric=SIGLIP2_PAIR_PROBABILITY,
                score=self.scores[item_id],
                raw_logit=self.scores[item_id] * 10.0,
                model_id="google/siglip2-base-patch16-224",
                model_revision="b" * 40,
                text_template="{instruction}",
                text_input=instruction,
                image_sha256=hashlib.sha256(image_png).hexdigest(),
                device="cuda:1",
                dtype="bfloat16",
                latency_seconds=0.0,
            )
            for image_png, item_id in zip(image_pngs, attempt_ids, strict=True)
        ]

    def score_pairs_batch(
        self,
        instructions: list[str],
        image_pngs: list[bytes],
        *,
        attempt_ids: list[str],
    ) -> list[ImageTextSimilarityEvidence]:
        self.pair_batch_calls.append((list(instructions), list(image_pngs), list(attempt_ids)))
        return [
            ImageTextSimilarityEvidence(
                attempt_id=item_id,
                metric=SIGLIP2_PAIR_PROBABILITY,
                score=self.scores[item_id],
                raw_logit=self.scores[item_id] * 10.0,
                model_id="google/siglip2-base-patch16-224",
                model_revision="b" * 40,
                text_template="{instruction}",
                text_input=instruction,
                image_sha256=hashlib.sha256(image_png).hexdigest(),
                device="cuda:1",
                dtype="bfloat16",
                latency_seconds=0.0,
            )
            for instruction, image_png, item_id in zip(
                instructions, image_pngs, attempt_ids, strict=True
            )
        ]


def test_hybrid_excludes_target_id_and_same_svg_before_overlap_and_backfills() -> None:
    records = {
        "icon:target": _record("icon:target", "red"),
        "icon:target-copy": _record("icon:target-copy", "red"),
        "icon:overlap": _record("icon:overlap", "blue"),
        "icon:desc-high": _record("icon:desc-high", "green"),
        "icon:desc-low": _record("icon:desc-low", "yellow"),
        "icon:detail-high": _record("icon:detail-high", "purple"),
        "icon:detail-low": _record("icon:detail-low", "orange"),
    }
    description = _Store(
        branch="description",
        hits=[
            _hit("description", "icon:target", 1, 0.99),
            _hit("description", "icon:target-copy", 2, 0.98),
            _hit("description", "icon:overlap", 3, 0.80),
            _hit("description", "icon:desc-high", 4, 0.70),
            _hit("description", "icon:desc-low", 5, 0.60),
        ],
        records=records,
    )
    detail = _Store(
        branch="detail",
        hits=[
            _hit("detail", "icon:target", 1, 0.99),
            _hit("detail", "icon:overlap", 2, 0.88),
            _hit("detail", "icon:detail-high", 3, 0.77),
            _hit("detail", "icon:detail-low", 4, 0.66),
        ],
        records=records,
    )
    scorer = _Scorer(
        {
            "icon:desc-high": 0.90,
            "icon:desc-low": 0.10,
            "icon:detail-high": 0.80,
            "icon:detail-low": 0.20,
        }
    )
    retriever = MMSVGHybridRetriever(
        description_retriever=description,  # type: ignore[arg-type]
        detail_retriever=detail,  # type: ignore[arg-type]
        similarity_scorer=scorer,
        branch_top_k=3,
        search_k=8,
    )
    prompt = "a raw user prompt for a blue symbol"
    target_hash = canonical_svg_sha256(records["icon:target"].svg)

    examples, trace = retriever.retrieve_with_trace(
        prompt,
        top_k=3,
        exclude_item_ids={"icon:target"},
        exclude_svg_sha256s={target_hash},
    )

    assert [example.item_id for example in examples] == [
        "icon:overlap",
        "icon:desc-high",
        "icon:detail-high",
    ]
    assert all(example.item_id not in {"icon:target", "icon:target-copy"} for example in examples)
    assert description.encode_calls == [prompt]
    assert detail.encode_calls == []
    assert description.query_embeddings == [[0.1, 0.2, 0.3, 0.4]]
    assert detail.query_embeddings == [[0.1, 0.2, 0.3, 0.4]]
    assert len(scorer.batch_calls) == 1
    assert scorer.batch_calls[0][0] == prompt
    assert set(scorer.batch_calls[0][2]) == {
        "icon:desc-high",
        "icon:desc-low",
        "icon:detail-high",
        "icon:detail-low",
    }
    assert examples[0].metadata["selection_reason"] == "overlap"
    assert examples[0].metadata["siglip2_score"] is None
    assert examples[1].detail == records["icon:desc-high"].detail
    description_reasons = {(hit.item_id, hit.exclusion_reason) for hit in trace.description_hits}
    assert ("icon:target", "target_item_id") in description_reasons
    assert ("icon:target-copy", "target_svg") in description_reasons
    assert trace.final_item_ids == tuple(example.item_id for example in examples)


def test_all_overlaps_are_selected_before_any_siglip_candidate() -> None:
    records = {
        item_id: _record(item_id, color)
        for item_id, color in (
            ("icon:overlap-a", "red"),
            ("icon:overlap-b", "blue"),
            ("icon:desc-only", "green"),
            ("icon:detail-only", "yellow"),
        )
    }
    description = _Store(
        branch="description",
        hits=[
            _hit("description", "icon:overlap-b", 1, 0.9),
            _hit("description", "icon:overlap-a", 2, 0.8),
            _hit("description", "icon:desc-only", 3, 0.7),
        ],
        records=records,
    )
    detail = _Store(
        branch="detail",
        hits=[
            _hit("detail", "icon:overlap-a", 1, 0.9),
            _hit("detail", "icon:overlap-b", 2, 0.8),
            _hit("detail", "icon:detail-only", 3, 0.7),
        ],
        records=records,
    )
    scorer = _Scorer({"icon:desc-only": 0.01, "icon:detail-only": 0.99})
    retriever = MMSVGHybridRetriever(
        description_retriever=description,  # type: ignore[arg-type]
        detail_retriever=detail,  # type: ignore[arg-type]
        similarity_scorer=scorer,
        search_k=6,
    )

    examples, _trace = retriever.retrieve_with_trace("raw prompt", top_k=3)

    assert [example.item_id for example in examples] == [
        "icon:overlap-a",
        "icon:overlap-b",
        "icon:detail-only",
    ]
    assert [example.metadata["selection_reason"] for example in examples] == [
        "overlap",
        "overlap",
        "siglip2",
    ]


def test_same_canonical_candidate_svg_is_emitted_only_once() -> None:
    duplicate_svg = _svg("cyan")
    records = {
        "icon:desc-copy": _record("icon:desc-copy", "unused", svg=duplicate_svg),
        "icon:detail-copy": _record("icon:detail-copy", "unused", svg=duplicate_svg),
        "icon:desc-2": _record("icon:desc-2", "red"),
        "icon:desc-3": _record("icon:desc-3", "blue"),
        "icon:detail-2": _record("icon:detail-2", "green"),
        "icon:detail-3": _record("icon:detail-3", "yellow"),
    }
    description = _Store(
        branch="description",
        hits=[
            _hit("description", "icon:desc-copy", 1, 0.9),
            _hit("description", "icon:desc-2", 2, 0.8),
            _hit("description", "icon:desc-3", 3, 0.7),
        ],
        records=records,
    )
    detail = _Store(
        branch="detail",
        hits=[
            _hit("detail", "icon:detail-copy", 1, 0.9),
            _hit("detail", "icon:detail-2", 2, 0.8),
            _hit("detail", "icon:detail-3", 3, 0.7),
        ],
        records=records,
    )
    scorer = _Scorer({item_id: 0.5 for item_id in records})
    retriever = MMSVGHybridRetriever(
        description_retriever=description,  # type: ignore[arg-type]
        detail_retriever=detail,  # type: ignore[arg-type]
        similarity_scorer=scorer,
    )

    examples, trace = retriever.retrieve_with_trace("raw prompt", top_k=3)

    hashes = [canonical_svg_sha256(example.content) for example in examples]
    assert len(hashes) == len(set(hashes))
    removed = [candidate for candidate in trace.candidates if not candidate.eligible]
    assert len(removed) == 1
    assert removed[0].exclusion_reason == "duplicate_candidate_svg"


def test_retrieve_many_batches_all_model_boundaries_and_preserves_order() -> None:
    retriever, description, detail, scorer, records = _batch_case()
    target_hash = canonical_svg_sha256(records["icon:target"].svg)
    requests = [
        HybridRetrievalRequest(
            query="  first raw prompt  ",
            exclude_item_ids=frozenset({"icon:target"}),
            exclude_svg_sha256s=frozenset({target_hash}),
        ),
        HybridRetrievalRequest(
            query="second raw prompt",
            exclude_item_ids=frozenset({"icon:overlap"}),
        ),
    ]

    results = retriever.retrieve_many_with_trace(requests)

    assert len(results) == 2
    first_examples, first_trace = results[0]
    second_examples, second_trace = results[1]
    assert [example.item_id for example in first_examples] == [
        "icon:overlap",
        "icon:desc-high",
        "icon:detail-high",
    ]
    assert [example.item_id for example in second_examples] == [
        "icon:target",
        "icon:desc-high",
        "icon:detail-high",
    ]
    assert [first_trace.query, second_trace.query] == [
        "first raw prompt",
        "second raw prompt",
    ]
    assert first_trace.final_item_ids == tuple(example.item_id for example in first_examples)
    assert second_trace.final_item_ids == tuple(example.item_id for example in second_examples)
    assert description.encode_calls == []
    assert description.encode_many_calls == [["first raw prompt", "second raw prompt"]]
    assert detail.encode_many_calls == []
    expected_embeddings = [[1.0, 0.2, 0.3, 0.4], [2.0, 0.2, 0.3, 0.4]]
    assert description.query_embedding_batches == [expected_embeddings]
    assert detail.query_embedding_batches == [expected_embeddings]
    assert scorer.batch_calls == []
    assert len(scorer.pair_batch_calls) == 1
    instructions, _images, attempt_ids = scorer.pair_batch_calls[0]
    assert len(instructions) == len(attempt_ids) == 8
    assert instructions[:4] == ["first raw prompt"] * 4
    assert instructions[4:] == ["second raw prompt"] * 4
    assert set(attempt_ids[:4]) == {
        "icon:desc-high",
        "icon:desc-low",
        "icon:detail-high",
        "icon:detail-low",
    }
    assert set(attempt_ids[4:]) == {
        "icon:desc-high",
        "icon:desc-low",
        "icon:detail-high",
        "icon:detail-low",
    }
    assert ("icon:target", "target_item_id") in {
        (hit.item_id, hit.exclusion_reason) for hit in first_trace.description_hits
    }
    assert ("icon:overlap", "target_item_id") in {
        (hit.item_id, hit.exclusion_reason) for hit in second_trace.description_hits
    }


def test_one_request_batch_matches_single_examples_and_trace_exactly() -> None:
    batch_retriever, batch_description, _batch_detail, batch_scorer, records = _batch_case()
    target_hash = canonical_svg_sha256(records["icon:target"].svg)
    request = HybridRetrievalRequest(
        query="raw prompt",
        exclude_item_ids=frozenset({"icon:target"}),
        exclude_svg_sha256s=frozenset({target_hash}),
    )
    batch_result = batch_retriever.retrieve_many_with_trace([request])[0]

    single_retriever, single_description, _single_detail, single_scorer, _records = _batch_case()
    single_result = single_retriever.retrieve_with_trace(
        request.query,
        top_k=request.top_k,
        exclude_item_ids=request.exclude_item_ids,
        exclude_svg_sha256s=request.exclude_svg_sha256s,
    )

    assert batch_result == single_result
    assert batch_description.encode_many_calls == [["raw prompt"]]
    assert single_description.encode_calls == ["raw prompt"]
    assert len(batch_scorer.pair_batch_calls) == 1
    assert len(single_scorer.batch_calls) == 1


def _batch_case() -> tuple[
    MMSVGHybridRetriever,
    _Store,
    _Store,
    _Scorer,
    dict[str, MMSVGCorpusRecord],
]:
    records = {
        "icon:target": _record("icon:target", "red"),
        "icon:target-copy": _record("icon:target-copy", "red"),
        "icon:overlap": _record("icon:overlap", "blue"),
        "icon:desc-high": _record("icon:desc-high", "green"),
        "icon:desc-low": _record("icon:desc-low", "yellow"),
        "icon:detail-high": _record("icon:detail-high", "purple"),
        "icon:detail-low": _record("icon:detail-low", "orange"),
    }
    description = _Store(
        branch="description",
        hits=[
            _hit("description", "icon:target", 1, 0.99),
            _hit("description", "icon:target-copy", 2, 0.98),
            _hit("description", "icon:overlap", 3, 0.80),
            _hit("description", "icon:desc-high", 4, 0.70),
            _hit("description", "icon:desc-low", 5, 0.60),
        ],
        records=records,
    )
    detail = _Store(
        branch="detail",
        hits=[
            _hit("detail", "icon:target", 1, 0.99),
            _hit("detail", "icon:overlap", 2, 0.88),
            _hit("detail", "icon:detail-high", 3, 0.77),
            _hit("detail", "icon:detail-low", 4, 0.66),
        ],
        records=records,
    )
    scorer = _Scorer(
        {
            "icon:desc-high": 0.90,
            "icon:desc-low": 0.10,
            "icon:detail-high": 0.80,
            "icon:detail-low": 0.20,
        }
    )
    retriever = MMSVGHybridRetriever(
        description_retriever=description,  # type: ignore[arg-type]
        detail_retriever=detail,  # type: ignore[arg-type]
        similarity_scorer=scorer,
        branch_top_k=3,
        search_k=8,
    )
    return retriever, description, detail, scorer, records


def _hit(branch: str, item_id: str, rank: int, score: float) -> ChromaQueryHit:
    return ChromaQueryHit(
        item_id=item_id,
        document=f"{branch} document for {item_id}",
        metadata={},
        distance=1.0 - score,
        score=score,
        raw_rank=rank,
        document_field=branch,
    )


def _record(
    item_id: str,
    color: str,
    *,
    svg: str | None = None,
) -> MMSVGCorpusRecord:
    return MMSVGCorpusRecord(
        item_id=item_id,
        description=f"description for {item_id}",
        detail=f"detail for {item_id}",
        svg=svg or _svg(color),
        image_png=_png(color if color != "unused" else "white"),
        source=f"hf://test/{item_id}",
        corpus_version="revision",
        metadata={},
    )


def _svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="10" height="10" fill="{color}"/>'
        "</svg>"
    )


def _png(color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color).save(output, format="PNG")
    return output.getvalue()
