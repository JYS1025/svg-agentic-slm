"""Explicit embedding models shared by local RAG indexers and retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QWEN3_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_QWEN3_EMBEDDING_DIMENSION = 1024
DEFAULT_SVG_QUERY_INSTRUCTION = (
    "Given a user description of a desired SVG image, retrieve SVG examples "
    "matching the objects, attributes, colors, style, composition, and spatial "
    "relationships."
)


class Qwen3EmbeddingEncoder:
    """Lazily load the pinned Qwen3 embedding model.

    MMSVG documents are encoded as their raw descriptions without a prefix.
    Only retrieval queries receive the asymmetric Qwen instruction prefix.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN3_EMBEDDING_MODEL,
        revision: str = DEFAULT_QWEN3_EMBEDDING_REVISION,
        device: str = "cuda:0",
        batch_size: int = 256,
        max_seq_length: int = 512,
        expected_dimension: int = DEFAULT_QWEN3_EMBEDDING_DIMENSION,
        query_instruction: str = DEFAULT_SVG_QUERY_INSTRUCTION,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive.")
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive.")
        if not query_instruction.strip():
            raise ValueError("query_instruction must be non-empty.")

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.expected_dimension = expected_dimension
        self.query_instruction = query_instruction.strip()
        self._model: Any = None

    def encode_documents(self, texts: Sequence[str]) -> Any:
        """Encode raw corpus descriptions, with no instruction or field label."""
        values = _validate_texts(texts)
        return self._encode(values)

    def encode_queries(self, texts: Sequence[str]) -> Any:
        """Encode user descriptions with the SVG retrieval instruction."""
        values = _validate_texts(texts)
        prompt = f"Instruct: {self.query_instruction}\nQuery:"
        return self._encode(values, prompt=prompt)

    def _encode(self, texts: list[str], *, prompt: str | None = None) -> Any:
        model = self._ensure_model()
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if prompt is not None:
            kwargs["prompt"] = prompt
        vectors = model.encode(texts, **kwargs)
        if getattr(vectors, "ndim", None) != 2 or vectors.shape[1] != self.expected_dimension:
            shape = getattr(vectors, "shape", None)
            raise RuntimeError(
                f"Unexpected embedding shape {shape}; expected (*, {self.expected_dimension})."
            )
        return vectors.astype("float32", copy=False)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            import transformers
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Qwen3 embeddings require torch and sentence-transformers.") from exc

        model_kwargs: dict[str, Any] = {}
        if self.device.startswith("cuda"):
            dtype_key = (
                "dtype"
                if _major_minor_version(transformers.__version__) >= (4, 56)
                else "torch_dtype"
            )
            model_kwargs[dtype_key] = torch.bfloat16
        self._model = SentenceTransformer(
            self.model_name,
            revision=self.revision,
            device=self.device,
            model_kwargs=model_kwargs,
        )
        self._model.max_seq_length = self.max_seq_length
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = self._model.get_sentence_embedding_dimension
        dimension = dimension_getter()
        if dimension != self.expected_dimension:
            raise RuntimeError(
                f"Embedding model reports dimension {dimension}; "
                f"expected {self.expected_dimension}."
            )
        return self._model


def _validate_texts(texts: Sequence[str]) -> list[str]:
    values = [str(text).strip() for text in texts]
    if not values:
        raise ValueError("At least one text is required for embedding.")
    if any(not value for value in values):
        raise ValueError("Embedding texts must be non-empty.")
    return values


def _major_minor_version(version: str) -> tuple[int, int]:
    """Return a conservative major/minor pair for dependency API routing."""
    try:
        major, minor = version.split(".", maxsplit=2)[:2]
        return int(major), int(minor)
    except (TypeError, ValueError):
        return (0, 0)
