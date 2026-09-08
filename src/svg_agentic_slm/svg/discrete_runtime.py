"""Gemma tokenizer registration and constrained generation for the local SVG dialect."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from svg_agentic_slm.svg.codec_backends import (
    DEFAULT_CODEC_REGISTRY,
    LOCAL_GEMMA_BACKEND_ID,
    CodecCompatibilityError,
    CodecConfigurationError,
    GemmaOmniSVGInspiredBackend,
    NamedSpecialTokenSVGCodec,
)
from svg_agentic_slm.svg.official_discrete_cache import (
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    CachedOpenVGLabGemmaDialect,
)

LOCAL_GEMMA_CODEC_VOCABULARY_SIZE: Final = 44_468


@dataclass(frozen=True, slots=True)
class NamedTokenRegistration:
    backend_id: str
    base_vocabulary_size: int
    final_vocabulary_size: int
    added_token_count: int
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    vocabulary_sha256: str
    preserved_additional_special_tokens: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "base_vocabulary_size": self.base_vocabulary_size,
            "final_vocabulary_size": self.final_vocabulary_size,
            "added_token_count": self.added_token_count,
            "token_count": len(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "preserved_additional_special_tokens": list(
                self.preserved_additional_special_tokens
            ),
            "one_token_roundtrip_verified": True,
            "joined_sequence_roundtrip_verified": True,
            "replace_additional_special_tokens": False,
        }


def validate_gemma_codec_backend(
    backend_id: str,
    *,
    allow_legacy_toy_codec: bool = False,
) -> None:
    if backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID:
        return
    if backend_id == LOCAL_GEMMA_BACKEND_ID and not allow_legacy_toy_codec:
        raise CodecCompatibilityError(
            "The local OmniSVG-inspired codec is a legacy/toy dialect and requires "
            "allow_legacy_toy_codec=true",
            backend_id=backend_id,
            code="legacy_toy_codec_requires_opt_in",
        )
    metadata = DEFAULT_CODEC_REGISTRY.metadata(backend_id)
    if (
        not metadata.can_encode
        or not metadata.can_decode
        or metadata.token_kind != "named-special-token"
        or metadata.official_checkpoint_compatible
    ):
        raise CodecCompatibilityError(
            f"Codec backend {backend_id!r} cannot be used for Gemma discrete SFT",
            backend_id=backend_id,
            code="incompatible_gemma_codec_backend",
            context={
                "directions": list(metadata.directions),
                "token_kind": metadata.token_kind,
                "official_checkpoint_compatible": metadata.official_checkpoint_compatible,
            },
        )


def create_gemma_named_codec(
    backend_id: str = OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    *,
    grid_size: int = 200,
    allow_legacy_toy_codec: bool = False,
) -> NamedSpecialTokenSVGCodec:
    validate_gemma_codec_backend(
        backend_id,
        allow_legacy_toy_codec=allow_legacy_toy_codec,
    )
    codec = (
        CachedOpenVGLabGemmaDialect()
        if backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
        else DEFAULT_CODEC_REGISTRY.create(backend_id, grid_size=grid_size)
    )
    if not isinstance(codec, NamedSpecialTokenSVGCodec):
        raise CodecCompatibilityError(
            f"Codec backend {backend_id!r} does not expose named-token registration",
            backend_id=backend_id,
            code="missing_named_token_capability",
        )
    return codec


def _additional_special_tokens(tokenizer: Any, *, backend_id: str) -> tuple[str, ...]:
    collected: list[str] = []
    for attribute_name in ("additional_special_tokens", "extra_special_tokens"):
        values = getattr(tokenizer, attribute_name, None)
        if values is None:
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise CodecConfigurationError(
                f"Tokenizer {attribute_name} is not a sequence",
                backend_id=backend_id,
                code="invalid_tokenizer_special_tokens",
            )
        collected.extend(str(value) for value in values)
    return tuple(dict.fromkeys(collected))


def _tokenizer_ids(
    tokenizer: Any,
    tokens: Sequence[str],
    *,
    backend_id: str,
) -> tuple[int, ...]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if callable(get_vocab) else None
    if not isinstance(vocabulary, Mapping):
        raise CodecConfigurationError(
            "Tokenizer does not expose get_vocab() for codec preflight",
            backend_id=backend_id,
            code="tokenizer_vocabulary_unavailable",
        )
    ids: list[int] = []
    for token in tokens:
        token_id = vocabulary.get(token)
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise CodecConfigurationError(
                f"Registered codec token has no non-negative integer ID: {token}",
                backend_id=backend_id,
                code="codec_token_id_missing",
            )
        ids.append(token_id)
    return tuple(ids)


def _flat_integer_ids(
    value: Any,
    *,
    operation: str,
    backend_id: str,
) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes, bytearray)):
            if len(value) != 1:
                raise CodecConfigurationError(
                    f"Tokenizer returned multiple examples during {operation}",
                    backend_id=backend_id,
                    code="codec_token_roundtrip_failed",
                )
            value = value[0]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise CodecConfigurationError(
            f"Tokenizer returned invalid IDs during {operation}",
            backend_id=backend_id,
            code="codec_token_roundtrip_failed",
        )
    return tuple(value)


def register_named_codec_tokens(
    tokenizer: Any,
    codec: NamedSpecialTokenSVGCodec,
) -> NamedTokenRegistration:
    """Register and prove a backend-declared vocabulary before model allocation."""

    tokens = tuple(codec.vocabulary_tokens())
    backend_id = codec.metadata.backend_id
    codec_manifest = codec.codec_manifest()
    declared_count = codec_manifest.get("vocabulary_size")
    if declared_count is None and backend_id == LOCAL_GEMMA_BACKEND_ID:
        declared_count = LOCAL_GEMMA_CODEC_VOCABULARY_SIZE
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count <= 0
    ):
        raise CodecCompatibilityError(
            "Named codec manifest must declare a positive vocabulary_size",
            backend_id=backend_id,
            code="missing_codec_vocabulary_size",
        )
    if len(tokens) != declared_count or len(set(tokens)) != declared_count:
        raise CodecCompatibilityError(
            "Named codec vocabulary does not match its declared unique-token count",
            backend_id=backend_id,
            code="unexpected_codec_vocabulary",
            context={
                "declared": declared_count,
                "count": len(tokens),
                "unique": len(set(tokens)),
            },
        )
    if any(not isinstance(token, str) or not token for token in tokens):
        raise CodecCompatibilityError(
            "Local Gemma codec vocabulary contains an invalid string token",
            backend_id=backend_id,
            code="invalid_codec_vocabulary_token",
        )

    try:
        base_size = len(tokenizer)
    except Exception as exc:
        raise CodecConfigurationError(
            f"Tokenizer does not expose a vocabulary length: {exc}",
            backend_id=backend_id,
            code="tokenizer_length_unavailable",
        ) from exc
    before_specials = _additional_special_tokens(tokenizer, backend_id=backend_id)
    try:
        # Transformers 5 renamed the replacement flag. Supplying the complete,
        # de-duplicated list preserves existing additional specials without a
        # version-specific keyword.
        combined_specials = list(dict.fromkeys((*before_specials, *tokens)))
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": combined_specials},
        )
    except Exception as exc:
        raise CodecConfigurationError(
            f"Tokenizer codec-token registration failed: {exc}",
            backend_id=backend_id,
            code="codec_token_registration_failed",
        ) from exc
    final_size = len(tokenizer)
    if (
        isinstance(added, bool)
        or not isinstance(added, int)
        or added != declared_count
        or final_size - base_size != declared_count
    ):
        raise CodecCompatibilityError(
            "Tokenizer did not add the backend-declared codec token count",
            backend_id=backend_id,
            code="codec_token_registration_count_mismatch",
            context={
                "reported_added": added,
                "size_before": base_size,
                "size_after": final_size,
            },
        )
    after_specials = _additional_special_tokens(tokenizer, backend_id=backend_id)
    missing_specials = sorted(set(before_specials) - set(after_specials))
    if missing_specials:
        raise CodecCompatibilityError(
            "Codec registration removed existing additional special tokens",
            backend_id=backend_id,
            code="existing_special_tokens_replaced",
            context={"missing": missing_specials},
        )

    token_ids = _tokenizer_ids(tokenizer, tokens, backend_id=backend_id)
    if len(set(token_ids)) != declared_count:
        raise CodecCompatibilityError(
            "Codec strings do not map to the declared number of unique tokenizer IDs",
            backend_id=backend_id,
            code="codec_token_ids_not_unique",
        )
    joined = "".join(tokens)
    try:
        joined_ids = _flat_integer_ids(
            tokenizer.encode(joined, add_special_tokens=False),
            operation="joined codec roundtrip",
            backend_id=backend_id,
        )
    except AttributeError as exc:
        raise CodecConfigurationError(
            "Tokenizer does not expose encode() for codec preflight",
            backend_id=backend_id,
            code="tokenizer_encode_unavailable",
        ) from exc
    if joined_ids != token_ids:
        raise CodecCompatibilityError(
            "Concatenated codec strings are not tokenized as exactly one ID each",
            backend_id=backend_id,
            code="joined_codec_token_roundtrip_mismatch",
        )
    convert_ids = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert_ids) or list(convert_ids(list(token_ids))) != list(tokens):
        raise CodecCompatibilityError(
            "Codec tokenizer ID-to-token roundtrip failed",
            backend_id=backend_id,
            code="codec_id_to_token_roundtrip_mismatch",
        )

    compact_ids = json.dumps(token_ids, separators=(",", ":"))
    vocabulary_text = "\n".join(tokens)
    return NamedTokenRegistration(
        backend_id=codec.metadata.backend_id,
        base_vocabulary_size=base_size,
        final_vocabulary_size=final_size,
        added_token_count=added,
        token_ids=token_ids,
        token_ids_sha256=hashlib.sha256(compact_ids.encode("utf-8")).hexdigest(),
        vocabulary_sha256=hashlib.sha256(vocabulary_text.encode("utf-8")).hexdigest(),
        preserved_additional_special_tokens=before_specials,
    )


class DiscreteSVGGrammar:
    """Finite-state allowlist for the complete local svgd1 grammar."""

    def __init__(
        self,
        codec: GemmaOmniSVGInspiredBackend,
        id_to_token: Mapping[int, str],
    ) -> None:
        tokens = codec.vocabulary_tokens()
        if len(tokens) != LOCAL_GEMMA_CODEC_VOCABULARY_SIZE:
            raise CodecCompatibilityError(
                "Discrete grammar requires the canonical 44,468-token vocabulary",
                backend_id=codec.metadata.backend_id,
                code="unexpected_codec_vocabulary",
            )
        token_to_id = {token: token_id for token_id, token in id_to_token.items()}
        if len(token_to_id) != len(tokens) or any(token not in token_to_id for token in tokens):
            raise CodecCompatibilityError(
                "Discrete grammar token-ID mapping is incomplete",
                backend_id=codec.metadata.backend_id,
                code="incomplete_codec_id_mapping",
            )
        legacy = codec.legacy_codec
        grid_size = int(legacy.grid_size)
        rotation_start = 3 + 6 + 2
        coordinate_start = rotation_start + 360
        coordinate_end = coordinate_start + grid_size * grid_size
        color_start = coordinate_end

        self.sop_token_id = token_to_id[codec.sop_token]
        self.eop_token_id = token_to_id[codec.eop_token]
        self.eos_token_id = token_to_id[codec.eos_token]
        self._commands = {
            name: token_to_id[token] for name, token in legacy.command_tokens.items()
        }
        self._flags = tuple(token_to_id[token] for token in legacy.flag_tokens.values())
        self._rotations = tuple(token_to_id[token] for token in tokens[rotation_start:coordinate_start])
        self._coordinates = tuple(token_to_id[token] for token in tokens[coordinate_start:coordinate_end])
        self._coordinate_set = frozenset(self._coordinates)
        self._radius_coordinates = tuple(
            self._coordinates[x + y * grid_size]
            for y in range(1, grid_size)
            for x in range(1, grid_size)
        )
        self._colors = tuple(token_to_id[token] for token in tokens[color_start:])
        self._open_commands = tuple(
            self._commands[name] for name in ("M", "L", "C", "A", "Z", "F")
        )
        self._closed_commands = (self._commands["M"], self._commands["F"])

    def allowed_next(self, generated_ids: Sequence[int]) -> tuple[int, ...]:
        state = "start"
        for index, token_id in enumerate(generated_ids):
            allowed = self._allowed(state)
            if token_id not in allowed:
                raise CodecCompatibilityError(
                    f"Generated codec prefix violates grammar at token {index}",
                    backend_id=LOCAL_GEMMA_BACKEND_ID,
                    code="generated_codec_grammar_violation",
                    context={"token_id": token_id, "state": state},
                )
            state = self._advance(state, token_id)
        return self._allowed(state)

    def _allowed(self, state: str) -> tuple[int, ...]:
        if state == "start":
            return (self.sop_token_id,)
        if state == "first_command":
            return (self._commands["M"],)
        if state in {"m_point", "l_point", "c_point_1", "c_point_2", "c_point_3", "a_end"}:
            return self._coordinates
        if state == "open":
            return self._open_commands
        if state == "closed":
            return self._closed_commands
        if state == "a_radius":
            return self._radius_coordinates
        if state == "a_rotation":
            return self._rotations
        if state in {"a_large_arc", "a_sweep"}:
            return self._flags
        if state == "fill":
            return self._colors
        if state == "eop":
            return (self.eop_token_id,)
        if state == "after_path":
            return (self.sop_token_id, self.eos_token_id)
        if state == "done":
            return (self.eos_token_id,)
        raise RuntimeError(f"Unknown discrete grammar state: {state}")

    def _advance(self, state: str, token_id: int) -> str:
        if state == "start":
            return "first_command"
        if state == "first_command":
            return "m_point"
        if state == "m_point":
            return "open"
        if state == "l_point":
            return "open"
        if state == "c_point_1":
            return "c_point_2"
        if state == "c_point_2":
            return "c_point_3"
        if state == "c_point_3":
            return "open"
        if state == "a_radius":
            return "a_rotation"
        if state == "a_rotation":
            return "a_large_arc"
        if state == "a_large_arc":
            return "a_sweep"
        if state == "a_sweep":
            return "a_end"
        if state == "a_end":
            return "open"
        if state in {"open", "closed"}:
            if token_id == self._commands["M"]:
                return "m_point"
            if token_id == self._commands["L"]:
                return "l_point"
            if token_id == self._commands["C"]:
                return "c_point_1"
            if token_id == self._commands["A"]:
                return "a_radius"
            if token_id == self._commands["Z"]:
                return "closed"
            if token_id == self._commands["F"]:
                return "fill"
        if state == "fill":
            return "eop"
        if state == "eop":
            return "after_path"
        if state == "after_path":
            return "first_command" if token_id == self.sop_token_id else "done"
        if state == "done":
            return "done"
        raise RuntimeError(f"Invalid discrete grammar transition from {state}")


def discrete_generation_kwargs(
    grammar: DiscreteSVGGrammar,
    *,
    prompt_tokens: int,
    pad_token_id: int | None,
) -> dict[str, Any]:
    """Return Transformers generation constraints bound to one prompt length."""

    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        values = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        completion = [int(value) for value in values[prompt_tokens:]]
        return list(grammar.allowed_next(completion))

    return {
        "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        "eos_token_id": grammar.eos_token_id,
        "pad_token_id": grammar.eos_token_id if pad_token_id is None else int(pad_token_id),
    }


def as_local_gemma_codec(codec: NamedSpecialTokenSVGCodec) -> GemmaOmniSVGInspiredBackend:
    if not isinstance(codec, GemmaOmniSVGInspiredBackend):
        raise CodecCompatibilityError(
            "Expected the registered local Gemma codec backend",
            backend_id=codec.metadata.backend_id,
            code="unexpected_gemma_codec_type",
        )
    return cast(GemmaOmniSVGInspiredBackend, codec)
