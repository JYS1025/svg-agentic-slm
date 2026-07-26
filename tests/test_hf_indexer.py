"""Offline tests for Hugging Face SVG row preparation and batching."""

from __future__ import annotations

from typing import Any

import pytest

from svg_agentic_slm.rag import hf_indexer
from svg_agentic_slm.rag.hf_indexer import (
    index_huggingface_svg_dataset,
    index_svg_rows,
    prepare_svg_document,
    validate_svg_for_reference,
)


class _FakeRetriever:
    def __init__(self, initial_count: int = 0) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.initial_count = initial_count
        self.batches: list[list[dict[str, Any]]] = []

    def count(self) -> int:
        return self.initial_count + len(self.documents)

    def missing_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            document
            for document in documents
            if document["metadata"]["point_identity"] not in self.documents
        ]

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        self.batches.append(list(documents))
        for document in documents:
            identity = document["metadata"]["point_identity"]
            self.documents[identity] = document


def _prepare(row: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
    return prepare_svg_document(
        row,
        dataset_id="org/svg-data",
        dataset_split="train",
        dataset_revision="rev-1",
        max_svg_chars=kwargs.get("max_svg_chars", 1_000),
        max_caption_chars=kwargs.get("max_caption_chars", 100),
    )


def test_prepare_svg_document_normalizes_dataset_metadata() -> None:
    row = {
        "Svg": (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<g><rect width="4" height="5"/><rect width="1" height="2"/></g>'
            "</svg>"
        ),
        "Filename": "shapes/boxes.svg",
        "caption_cogvlm": "  two nested boxes  ",
        "Label": "geometry|boxes",
    }

    document = _prepare(row)

    assert document is not None
    assert document["content"] == row["Svg"]
    metadata = document["metadata"]
    assert metadata["point_identity"] == ("hf://org/svg-data/train/shapes/boxes.svg")
    assert metadata["source"] == metadata["point_identity"]
    assert metadata["record_id"] == "shapes/boxes.svg"
    assert metadata["description"] == "two nested boxes"
    assert metadata["caption_field"] == "caption_cogvlm"
    assert metadata["tags"] == ["geometry", "boxes"]
    assert metadata["svg_elements"] == ["g", "rect"]
    assert "Description: two nested boxes" in metadata["search_text"]
    assert "SVG elements: g, rect" in metadata["search_text"]


def test_prepare_document_uses_tags_as_caption_and_truncates() -> None:
    document = _prepare(
        {
            "svg": "<svg><circle/></svg>",
            "id": 17,
            "tags": ["round", "minimal"],
        },
        max_caption_chars=8,
    )

    assert document is not None
    assert document["metadata"]["record_id"] == "17"
    assert document["metadata"]["description"] == "round mi"
    assert document["metadata"]["caption_field"] == ""


@pytest.mark.parametrize(
    "svg",
    [
        "<svg>",
        "<html><div/></html>",
        "<!DOCTYPE svg><svg/>",
        "<svg><script>alert(1)</script></svg>",
        "<svg><foreignObject><div/></foreignObject></svg>",
        '<svg><rect onclick="alert(1)"/></svg>',
        '<svg><use href="https://example.com/a.svg#shape"/></svg>',
        '<svg><image href="data:image/png;base64,AAAA"/></svg>',
        "<svg><style>.x { fill: url(https://example.com/a) }</style></svg>",
        '<svg><path fill="javascript:alert(1)"/></svg>',
        '<svg><image href="file:///etc/passwd"/></svg>',
        '<svg><image href="ftp://example.com/image.png"/></svg>',
        '<svg><rect style="fill:url(https://example.com/a.svg)"/></svg>',
        '<svg><rect fill="url(file:///tmp/a.svg)"/></svg>',
    ],
)
def test_validate_svg_rejects_malformed_or_active_content(svg: str) -> None:
    assert validate_svg_for_reference(svg) is None


def test_prepare_document_rejects_missing_caption_and_oversized_svg() -> None:
    assert _prepare({"svg": "<svg><circle/></svg>"}) is None
    assert (
        _prepare(
            {"svg": "<svg><circle/></svg>", "caption": "circle"},
            max_svg_chars=5,
        )
        is None
    )


def test_index_svg_rows_skips_invalid_rows_and_flushes_partial_batch() -> None:
    retriever = _FakeRetriever()
    rows = [
        {"svg": "<svg><circle/></svg>", "caption": "circle", "id": "one"},
        {"svg": "<svg><script/></svg>", "caption": "unsafe", "id": "bad"},
        {"svg": "<svg><rect/></svg>", "caption": "rectangle", "id": "two"},
        {"svg": "<svg><path/></svg>", "caption": "path", "id": "three"},
    ]

    result = index_svg_rows(
        retriever,
        rows,
        dataset_id="dataset",
        dataset_split="train",
        dataset_revision="main",
        index_limit=3,
        batch_size=2,
        max_svg_chars=1_000,
        max_caption_chars=100,
    )

    assert [len(batch) for batch in retriever.batches] == [2, 1]
    assert result.target_count == 3
    assert result.collection_count_before == 0
    assert result.collection_count_after == 3
    assert result.uploaded_this_run == 3
    assert result.scanned_this_run == 4
    assert result.skipped_this_run == 1


def test_index_svg_rows_honors_exact_target_with_unrelated_existing_points() -> None:
    retriever = _FakeRetriever(initial_count=2)
    rows = [
        {"svg": "<svg><circle/></svg>", "caption": "circle", "id": "one"},
        {"svg": "<svg><rect/></svg>", "caption": "rectangle", "id": "two"},
        {"svg": "<svg><path/></svg>", "caption": "path", "id": "three"},
    ]

    result = index_svg_rows(
        retriever,
        rows,
        dataset_id="dataset",
        dataset_split="train",
        dataset_revision="main",
        index_limit=3,
        batch_size=3,
        max_svg_chars=1_000,
        max_caption_chars=100,
    )

    assert result.collection_count_before == 2
    assert result.collection_count_after == 3
    assert result.uploaded_this_run == 1
    assert sum(len(batch) for batch in retriever.batches) == 1


def test_index_svg_rows_raises_when_no_valid_rows_are_stored() -> None:
    retriever = _FakeRetriever()

    with pytest.raises(RuntimeError, match="No valid SVG records"):
        index_svg_rows(
            retriever,
            [{"svg": "<svg><script/></svg>", "caption": "unsafe"}],
            dataset_id="dataset",
            dataset_split="train",
            dataset_revision="main",
            index_limit=1,
            batch_size=1,
            max_svg_chars=1_000,
            max_caption_chars=100,
        )


def test_huggingface_indexing_short_circuits_without_loading_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _FakeRetriever(initial_count=10)

    def fail_if_loaded(**kwargs: Any) -> Any:
        raise AssertionError("dataset loader must not run")

    monkeypatch.setattr(hf_indexer, "_load_streaming_dataset", fail_if_loaded)

    result = index_huggingface_svg_dataset(retriever, index_limit=10)

    assert result.collection_count_before == 10
    assert result.collection_count_after == 10
    assert result.uploaded_this_run == 0
    assert result.scanned_this_run == 0


def test_huggingface_indexing_passes_stream_options_to_injected_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _FakeRetriever()
    captured: dict[str, Any] = {}

    def fake_loader(**kwargs: Any) -> list[dict[str, str]]:
        captured.update(kwargs)
        return [
            {
                "svg": "<svg><circle/></svg>",
                "caption": "circle",
                "id": "one",
            }
        ]

    monkeypatch.setattr(hf_indexer, "_load_streaming_dataset", fake_loader)

    result = index_huggingface_svg_dataset(
        retriever,
        dataset_id="org/data",
        dataset_split="validation",
        dataset_revision="commit",
        index_limit=1,
        batch_size=1,
        shuffle_buffer=17,
        seed=9,
    )

    assert captured == {
        "dataset_id": "org/data",
        "dataset_split": "validation",
        "dataset_revision": "commit",
        "shuffle_buffer": 17,
        "seed": 9,
    }
    assert result.collection_count_after == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_limit": 0},
        {"batch_size": 0},
        {"max_svg_chars": 0},
        {"max_caption_chars": 0},
        {"shuffle_buffer": 0},
    ],
)
def test_huggingface_indexing_rejects_nonpositive_settings(
    kwargs: dict[str, int],
) -> None:
    settings = {
        "index_limit": 1,
        "batch_size": 1,
        "max_svg_chars": 100,
        "max_caption_chars": 100,
        "shuffle_buffer": 1,
    }
    settings.update(kwargs)

    with pytest.raises(ValueError, match="must be positive"):
        index_huggingface_svg_dataset(_FakeRetriever(), **settings)
