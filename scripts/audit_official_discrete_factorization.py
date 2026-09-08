#!/usr/bin/env python3
"""Audit lossless factorized alternatives to the pinned OmniSVG 4B token cache.

This is CPU-only feasibility tooling.  It does not define or modify a production
codec.  Records are joined and provenance-checked by the existing immutable
cache loader before any experimental transformation is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from svg_agentic_slm.svg import official_discrete_cache as cache_module  # noqa: E402
from svg_agentic_slm.svg.official_discrete_cache import (  # noqa: E402
    CACHED_TARGET_FIELD,
    OPENVGLAB_BOS_ID,
    OPENVGLAB_EOS_ID,
    OPENVGLAB_MAX_PRODUCIBLE_ID,
    OPENVGLAB_MIN_PRODUCIBLE_ID,
    OPENVGLAB_NAMED_VOCABULARY_SIZE,
    OPENVGLAB_TOKENIZATION_CONFIG_SHA256,
    OPENVGLAB_TRAIN_COMMIT,
    PINNED_AUDIT_SHA256,
    PINNED_CACHE_SHA256,
    PINNED_INPUT_SHA256,
    PINNED_SPLIT_COUNTS,
    CachedOpenVGLabGemmaDialect,
    OpenVGLabCacheConfig,
    PinnedOpenVGLabTokenCache,
)

GRID_SIZE = 200
COMMAND_MIN = 151938
COMMAND_MAX = 151942
COORDINATE_MIN = 151943
COORDINATE_MAX = 191942
COLOR_MIN = 191946
COLOR_MAX = 196043
ARC_MIN = 196436
ARC_MAX = 196535
CONTEXT_THRESHOLDS = (2304, 4096, 8192)

COMMAND_ARGUMENT_KINDS: Mapping[int, tuple[str, ...]] = {
    151938: ("coordinate", "coordinate"),
    151939: ("coordinate",),
    151940: ("coordinate", "coordinate", "coordinate"),
    151941: ("coordinate", "arc", "arc", "arc", "coordinate"),
    151942: ("coordinate",),
}

TokenKind = Literal[
    "bos",
    "command",
    "coordinate",
    "color",
    "arc",
    "eos",
]
ComponentToken = tuple[str, int]

EXACT_TOKEN_CLASSES: Mapping[str, Mapping[str, int]] = {
    "command": {"minimum": COMMAND_MIN, "maximum": COMMAND_MAX, "count": 5},
    "coordinate": {
        "minimum": COORDINATE_MIN,
        "maximum": COORDINATE_MAX,
        "count": GRID_SIZE * GRID_SIZE,
    },
    "color": {"minimum": COLOR_MIN, "maximum": COLOR_MAX, "count": 4098},
    "arc": {"minimum": ARC_MIN, "maximum": ARC_MAX, "count": 100},
    "bos": {"minimum": OPENVGLAB_BOS_ID, "maximum": OPENVGLAB_BOS_ID, "count": 1},
    "eos": {"minimum": OPENVGLAB_EOS_ID, "maximum": OPENVGLAB_EOS_ID, "count": 1},
}

RESERVED_GAPS: tuple[Mapping[str, int], ...] = (
    {"minimum": 191943, "maximum": 191945, "count": 3},
    {"minimum": 196044, "maximum": 196435, "count": 392},
    {"minimum": 196536, "maximum": 196997, "count": 462},
)

VOCABULARY_FORMULAS: Mapping[str, Mapping[str, Any]] = {
    "current_contiguous_named_namespace": {
        "formula": "196999 - 151938 + 1",
        "terms": {"contiguous_upstream_ids": OPENVGLAB_NAMED_VOCABULARY_SIZE},
        "size": OPENVGLAB_NAMED_VOCABULARY_SIZE,
        "includes_reserved_encoder_gaps": True,
    },
    "atomic_exact_encoder_producible": {
        "formula": "5 command + 40000 joint_xy + 4098 color + 100 arc + 2 framing",
        "terms": {
            "command": 5,
            "joint_xy": 40000,
            "color": 4098,
            "arc": 100,
            "framing": 2,
        },
        "size": 44205,
    },
    "xy_factorized": {
        "formula": "5 command + 200 x + 200 y + 4098 color + 100 arc + 2 framing",
        "terms": {
            "command": 5,
            "x": 200,
            "y": 200,
            "color": 4098,
            "arc": 100,
            "framing": 2,
        },
        "size": 4605,
        "channel_namespaces_are_distinct": True,
    },
    "xy_rgb_explicit_path_end": {
        "formula": (
            "5 command + 200 x + 200 y + (16 r + 16 g + 16 b) + "
            "2 special_color + 1 path_end + 100 arc + 2 framing"
        ),
        "terms": {
            "command": 5,
            "x": 200,
            "y": 200,
            "r": 16,
            "g": 16,
            "b": 16,
            "special_color": 2,
            "path_end": 1,
            "arc": 100,
            "framing": 2,
        },
        "size": 558,
        "channel_namespaces_are_distinct": True,
    },
}

FACTORIZATION_CONTRACT: Mapping[str, Any] = {
    "schema_version": 1,
    "scope": "offline_feasibility_only_no_production_codec_claim",
    "coordinate_index": "joint_xy = y * 200 + x",
    "xy_order": ["x", "y"],
    "color_offsets": {
        "0": "none",
        "1": "currentColor",
        "2..4097": "12-bit RGB value = offset - 2, emitted in r,g,b order",
    },
    "rgb_channel_width_bits": 4,
    "path_end": "one explicit token after every special-color or RGB payload",
    "arc": "kept as one of 100 atomic parameter tokens",
    "command": "kept as one of 5 atomic opcode tokens",
    "framing": "BOS and EOS remain distinct tokens",
    "known_source_ambiguity": (
        "Upstream maps both gradient and quantized white to color offset 4097; "
        "cached token IDs cannot recover which source semantic produced it."
    ),
}

COMPONENT_SIZES: Mapping[str, int] = {
    "command": 5,
    "joint_xy": 40000,
    "x": 200,
    "y": 200,
    "atomic_color": 4098,
    "r": 16,
    "g": 16,
    "b": 16,
    "special_color": 2,
    "path_end": 1,
    "arc": 100,
}


class FactorizationAuditError(RuntimeError):
    """Raised when a cached sequence violates the audited factorization contract."""


@dataclass(frozen=True)
class SequenceFeatures:
    coordinate_count: int
    color_count: int
    rgb_color_count: int
    special_color_count: int
    command_count: int
    arc_count: int
    path_count: int


@dataclass
class SplitAccumulator:
    split: str
    record_count: int = 0
    lengths: dict[str, list[int]] = field(
        default_factory=lambda: {
            "atomic": [],
            "xy_factorized": [],
            "xy_rgb_explicit_path_end": [],
        }
    )
    components: dict[str, Counter[int]] = field(
        default_factory=lambda: {name: Counter() for name in COMPONENT_SIZES}
    )
    token_class_counts: Counter[str] = field(default_factory=Counter)
    path_count: int = 0
    id_roundtrip_xy_passed: int = 0
    id_roundtrip_xy_rgb_path_end_passed: int = 0
    ambiguous_white_or_gradient_count: int = 0
    record_order_digest: Any = field(default_factory=hashlib.sha256)
    atomic_sequence_digest: Any = field(default_factory=hashlib.sha256)
    xy_sequence_digest: Any = field(default_factory=hashlib.sha256)
    xy_rgb_path_end_sequence_digest: Any = field(default_factory=hashlib.sha256)

    def add(
        self,
        *,
        sample_id: str,
        training_ids: tuple[int, ...],
        features: SequenceFeatures,
        xy_tokens: tuple[ComponentToken, ...],
        xy_rgb_tokens: tuple[ComponentToken, ...],
    ) -> None:
        self.record_count += 1
        self.record_order_digest.update(sample_id.encode("utf-8"))
        self.record_order_digest.update(b"\0")
        _update_integer_sequence_digest(self.atomic_sequence_digest, training_ids)
        _update_component_sequence_digest(self.xy_sequence_digest, xy_tokens)
        _update_component_sequence_digest(self.xy_rgb_path_end_sequence_digest, xy_rgb_tokens)

        self.lengths["atomic"].append(len(training_ids))
        self.lengths["xy_factorized"].append(len(xy_tokens))
        self.lengths["xy_rgb_explicit_path_end"].append(len(xy_rgb_tokens))
        self.path_count += features.path_count
        _update_component_counters(self, training_ids)

    def merge(self, other: SplitAccumulator) -> None:
        self.record_order_digest.update(other.record_order_digest.digest())
        self.atomic_sequence_digest.update(other.atomic_sequence_digest.digest())
        self.xy_sequence_digest.update(other.xy_sequence_digest.digest())
        self.xy_rgb_path_end_sequence_digest.update(
            other.xy_rgb_path_end_sequence_digest.digest()
        )
        self.record_count += other.record_count
        for dialect, lengths in other.lengths.items():
            self.lengths[dialect].extend(lengths)
        for component, counter in other.components.items():
            self.components[component].update(counter)
        self.token_class_counts.update(other.token_class_counts)
        self.path_count += other.path_count
        self.id_roundtrip_xy_passed += other.id_roundtrip_xy_passed
        self.id_roundtrip_xy_rgb_path_end_passed += (
            other.id_roundtrip_xy_rgb_path_end_passed
        )
        self.ambiguous_white_or_gradient_count += other.ambiguous_white_or_gradient_count


def training_token_kind(token_id: int) -> TokenKind:
    """Return the exact class emitted by the pinned 4B training encoder."""

    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise FactorizationAuditError(f"Token ID must be an integer, observed {token_id!r}.")
    if token_id == OPENVGLAB_BOS_ID:
        return "bos"
    if token_id == OPENVGLAB_EOS_ID:
        return "eos"
    if COMMAND_MIN <= token_id <= COMMAND_MAX:
        return "command"
    if COORDINATE_MIN <= token_id <= COORDINATE_MAX:
        return "coordinate"
    if COLOR_MIN <= token_id <= COLOR_MAX:
        return "color"
    if ARC_MIN <= token_id <= ARC_MAX:
        return "arc"
    raise FactorizationAuditError(
        f"Token ID {token_id} is not in an exact encoder-producible range."
    )


def validate_framed_training_ids(training_ids: Sequence[int]) -> SequenceFeatures:
    """Validate framing and the exact command/argument/path grammar."""

    ids = tuple(training_ids)
    if len(ids) < 3:
        raise FactorizationAuditError("A framed sequence must contain an SVG body.")
    if ids[0] != OPENVGLAB_BOS_ID or ids[-1] != OPENVGLAB_EOS_ID:
        raise FactorizationAuditError("Sequence lacks exact BOS/EOS framing.")
    if OPENVGLAB_BOS_ID in ids[1:] or OPENVGLAB_EOS_ID in ids[:-1]:
        raise FactorizationAuditError("BOS/EOS may not occur inside the sequence.")

    body = ids[1:-1]
    index = 0
    command_count = 0
    coordinate_count = 0
    arc_count = 0
    path_count = 0
    commands_in_path = 0
    rgb_color_count = 0
    special_color_count = 0
    while index < len(body):
        token_id = body[index]
        kind = training_token_kind(token_id)
        if kind == "color":
            if commands_in_path == 0:
                raise FactorizationAuditError(
                    f"Color token at body index {index} has no preceding command."
                )
            color_offset = token_id - COLOR_MIN
            if color_offset < 2:
                special_color_count += 1
            else:
                rgb_color_count += 1
            path_count += 1
            commands_in_path = 0
            index += 1
            continue
        if kind != "command":
            raise FactorizationAuditError(
                f"Expected command or path-ending color at body index {index}, observed {kind}."
            )
        expected_arguments = COMMAND_ARGUMENT_KINDS[token_id]
        if index + len(expected_arguments) >= len(body):
            raise FactorizationAuditError(f"Truncated command at body index {index}.")
        for offset, expected_kind in enumerate(expected_arguments, start=1):
            observed_kind = training_token_kind(body[index + offset])
            if observed_kind != expected_kind:
                raise FactorizationAuditError(
                    f"Command at body index {index} expected {expected_kind} at offset "
                    f"{offset}, observed {observed_kind}."
                )
            if observed_kind == "coordinate":
                coordinate_count += 1
            elif observed_kind == "arc":
                arc_count += 1
        command_count += 1
        commands_in_path += 1
        index += 1 + len(expected_arguments)
    if commands_in_path:
        raise FactorizationAuditError("Final path has no color terminator.")
    if path_count == 0:
        raise FactorizationAuditError("Sequence contains no complete path.")
    return SequenceFeatures(
        coordinate_count=coordinate_count,
        color_count=rgb_color_count + special_color_count,
        rgb_color_count=rgb_color_count,
        special_color_count=special_color_count,
        command_count=command_count,
        arc_count=arc_count,
        path_count=path_count,
    )


def factorize_xy(training_ids: Sequence[int]) -> tuple[ComponentToken, ...]:
    """Replace each joint coordinate with distinct X then Y component tokens."""

    ids = tuple(training_ids)
    validate_framed_training_ids(ids)
    result: list[ComponentToken] = []
    for token_id in ids:
        kind = training_token_kind(token_id)
        if kind == "coordinate":
            joint_index = token_id - COORDINATE_MIN
            result.extend(
                (("x", joint_index % GRID_SIZE), ("y", joint_index // GRID_SIZE))
            )
        else:
            result.append(_atomic_component(kind, token_id))
    return tuple(result)


def invert_xy(tokens: Sequence[ComponentToken]) -> tuple[int, ...]:
    """Invert :func:`factorize_xy` and reject malformed component streams."""

    result: list[int] = []
    index = 0
    while index < len(tokens):
        kind, value = _component(tokens[index], index=index)
        if kind == "x":
            _require_component_value(kind, value, GRID_SIZE, index=index)
            if index + 1 >= len(tokens):
                raise FactorizationAuditError("X component lacks its following Y component.")
            next_kind, y = _component(tokens[index + 1], index=index + 1)
            if next_kind != "y":
                raise FactorizationAuditError("X component is not followed by Y.")
            _require_component_value(next_kind, y, GRID_SIZE, index=index + 1)
            result.append(COORDINATE_MIN + y * GRID_SIZE + value)
            index += 2
            continue
        result.append(_invert_atomic_component(kind, value, index=index))
        index += 1
    ids = tuple(result)
    validate_framed_training_ids(ids)
    return ids


def factorize_xy_rgb_path_end(
    training_ids: Sequence[int],
) -> tuple[ComponentToken, ...]:
    """Factorize joint XY and 12-bit RGB, with an explicit path terminator."""

    ids = tuple(training_ids)
    validate_framed_training_ids(ids)
    result: list[ComponentToken] = []
    for token_id in ids:
        kind = training_token_kind(token_id)
        if kind == "coordinate":
            joint_index = token_id - COORDINATE_MIN
            result.extend(
                (("x", joint_index % GRID_SIZE), ("y", joint_index // GRID_SIZE))
            )
            continue
        if kind == "color":
            color_offset = token_id - COLOR_MIN
            if color_offset < 2:
                result.append(("special_color", color_offset))
            else:
                rgb12 = color_offset - 2
                result.extend(
                    (
                        ("r", (rgb12 >> 8) & 0xF),
                        ("g", (rgb12 >> 4) & 0xF),
                        ("b", rgb12 & 0xF),
                    )
                )
            result.append(("path_end", 0))
            continue
        result.append(_atomic_component(kind, token_id))
    return tuple(result)


def invert_xy_rgb_path_end(tokens: Sequence[ComponentToken]) -> tuple[int, ...]:
    """Invert :func:`factorize_xy_rgb_path_end` at the cached token-ID level."""

    result: list[int] = []
    index = 0
    while index < len(tokens):
        kind, value = _component(tokens[index], index=index)
        if kind == "x":
            _require_component_value(kind, value, GRID_SIZE, index=index)
            if index + 1 >= len(tokens):
                raise FactorizationAuditError("X component lacks its following Y component.")
            next_kind, y = _component(tokens[index + 1], index=index + 1)
            if next_kind != "y":
                raise FactorizationAuditError("X component is not followed by Y.")
            _require_component_value(next_kind, y, GRID_SIZE, index=index + 1)
            result.append(COORDINATE_MIN + y * GRID_SIZE + value)
            index += 2
            continue
        if kind == "special_color":
            _require_component_value(kind, value, 2, index=index)
            _require_path_end(tokens, index + 1)
            result.append(COLOR_MIN + value)
            index += 2
            continue
        if kind == "r":
            _require_component_value(kind, value, 16, index=index)
            if index + 3 >= len(tokens):
                raise FactorizationAuditError("RGB payload lacks G, B, or PATH_END.")
            green_kind, green = _component(tokens[index + 1], index=index + 1)
            blue_kind, blue = _component(tokens[index + 2], index=index + 2)
            if green_kind != "g" or blue_kind != "b":
                raise FactorizationAuditError("RGB payload must use R, G, B order.")
            _require_component_value(green_kind, green, 16, index=index + 1)
            _require_component_value(blue_kind, blue, 16, index=index + 2)
            _require_path_end(tokens, index + 3)
            rgb12 = (value << 8) | (green << 4) | blue
            result.append(COLOR_MIN + 2 + rgb12)
            index += 4
            continue
        if kind in {"y", "g", "b", "path_end"}:
            raise FactorizationAuditError(f"Unexpected standalone {kind} component at {index}.")
        result.append(_invert_atomic_component(kind, value, index=index))
        index += 1
    ids = tuple(result)
    validate_framed_training_ids(ids)
    return ids


def _component(token: ComponentToken, *, index: int) -> ComponentToken:
    if (
        not isinstance(token, tuple)
        or len(token) != 2
        or not isinstance(token[0], str)
        or isinstance(token[1], bool)
        or not isinstance(token[1], int)
    ):
        raise FactorizationAuditError(f"Malformed component token at index {index}: {token!r}.")
    return token


def _require_component_value(kind: str, value: int, size: int, *, index: int) -> None:
    if not 0 <= value < size:
        raise FactorizationAuditError(
            f"{kind} component at index {index} is outside [0, {size - 1}]."
        )


def _require_path_end(tokens: Sequence[ComponentToken], index: int) -> None:
    if index >= len(tokens) or _component(tokens[index], index=index) != ("path_end", 0):
        raise FactorizationAuditError("Color payload lacks an explicit PATH_END token.")


def _atomic_component(kind: TokenKind, token_id: int) -> ComponentToken:
    if kind == "bos":
        return ("bos", 0)
    if kind == "eos":
        return ("eos", 0)
    if kind == "command":
        return ("command", token_id - COMMAND_MIN)
    if kind == "color":
        return ("atomic_color", token_id - COLOR_MIN)
    if kind == "arc":
        return ("arc", token_id - ARC_MIN)
    raise FactorizationAuditError(f"No atomic component mapping exists for {kind}.")


def _invert_atomic_component(kind: str, value: int, *, index: int) -> int:
    offsets = {
        "command": (COMMAND_MIN, 5),
        "atomic_color": (COLOR_MIN, 4098),
        "arc": (ARC_MIN, 100),
        "bos": (OPENVGLAB_BOS_ID, 1),
        "eos": (OPENVGLAB_EOS_ID, 1),
    }
    if kind not in offsets:
        raise FactorizationAuditError(f"Unexpected component {kind!r} at index {index}.")
    start, size = offsets[kind]
    _require_component_value(kind, value, size, index=index)
    return start + value


def _named_tokens_to_training_ids(tokens: Sequence[str]) -> tuple[int, ...]:
    prefix = "<svgovg4b:"
    ids: list[int] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, str) or not token.startswith(prefix) or not token.endswith(">"):
            raise FactorizationAuditError(
                f"Cached dialect returned malformed named token at index {index}: {token!r}."
            )
        number = token[len(prefix) : -1]
        if not number.isascii() or not number.isdigit():
            raise FactorizationAuditError(
                f"Cached dialect returned malformed upstream ID at index {index}: {token!r}."
            )
        ids.append(int(number))
    return tuple(ids)


def _record_sample_id(record: Mapping[str, Any]) -> str:
    payload = record.get(CACHED_TARGET_FIELD)
    if not isinstance(payload, Mapping) or payload.get("sample_id") is None:
        raise FactorizationAuditError("Verified cache record lacks a sample ID.")
    return str(payload["sample_id"])


def _update_component_counters(
    accumulator: SplitAccumulator, training_ids: Sequence[int]
) -> None:
    for token_id in training_ids:
        kind = training_token_kind(token_id)
        accumulator.token_class_counts[kind] += 1
        if kind == "command":
            accumulator.components["command"][token_id - COMMAND_MIN] += 1
        elif kind == "coordinate":
            joint_index = token_id - COORDINATE_MIN
            accumulator.components["joint_xy"][joint_index] += 1
            accumulator.components["x"][joint_index % GRID_SIZE] += 1
            accumulator.components["y"][joint_index // GRID_SIZE] += 1
        elif kind == "color":
            color_offset = token_id - COLOR_MIN
            accumulator.components["atomic_color"][color_offset] += 1
            accumulator.components["path_end"][0] += 1
            if color_offset < 2:
                accumulator.components["special_color"][color_offset] += 1
            else:
                rgb12 = color_offset - 2
                accumulator.components["r"][(rgb12 >> 8) & 0xF] += 1
                accumulator.components["g"][(rgb12 >> 4) & 0xF] += 1
                accumulator.components["b"][rgb12 & 0xF] += 1
                if color_offset == 4097:
                    accumulator.ambiguous_white_or_gradient_count += 1
        elif kind == "arc":
            accumulator.components["arc"][token_id - ARC_MIN] += 1


def _update_integer_sequence_digest(digest: Any, values: Sequence[int]) -> None:
    digest.update(len(values).to_bytes(8, "little"))
    for value in values:
        digest.update(value.to_bytes(4, "little"))


def _update_component_sequence_digest(
    digest: Any, values: Sequence[ComponentToken]
) -> None:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _nearest_rank(sorted_values: Sequence[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot summarize an empty sequence.")
    rank = max(1, int((len(sorted_values) * quantile) + 0.999999999999))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def length_summary(lengths: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("Cannot summarize empty lengths.")
    ordered = sorted(lengths)
    count = len(ordered)
    middle = count // 2
    median = (
        float(ordered[middle])
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "record_count": count,
        "total_tokens": sum(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": round(sum(ordered) / count, 6),
        "median": median,
        "p90_nearest_rank": _nearest_rank(ordered, 0.90),
        "p95_nearest_rank": _nearest_rank(ordered, 0.95),
        "p99_nearest_rank": _nearest_rank(ordered, 0.99),
        "context_thresholds": {
            str(threshold): _threshold_summary(ordered, threshold)
            for threshold in CONTEXT_THRESHOLDS
        },
    }


def _threshold_summary(lengths: Sequence[int], threshold: int) -> dict[str, Any]:
    at_or_below = sum(value <= threshold for value in lengths)
    count = len(lengths)
    return {
        "at_or_below_count": at_or_below,
        "at_or_below_fraction": round(at_or_below / count, 8),
        "above_count": count - at_or_below,
        "above_fraction": round((count - at_or_below) / count, 8),
    }


def _frequency_summary(counter: Counter[int], size: int) -> dict[str, Any]:
    dense = [counter.get(value, 0) for value in range(size)]
    observed = [(value, count) for value, count in enumerate(dense) if count]
    missing = [value for value, count in enumerate(dense) if not count]
    dense_payload = json.dumps(dense, separators=(",", ":")).encode("ascii")
    missing_payload = json.dumps(missing, separators=(",", ":")).encode("ascii")
    least = sorted(observed, key=lambda item: (item[1], item[0]))[:10]
    most = sorted(observed, key=lambda item: (-item[1], item[0]))[:10]
    return {
        "possible_values": size,
        "observed_values": len(observed),
        "coverage_fraction": round(len(observed) / size, 8),
        "total_occurrences": sum(dense),
        "minimum_nonzero_frequency": min((count for _, count in observed), default=0),
        "maximum_frequency": max(dense, default=0),
        "unobserved_value_count": len(missing),
        "unobserved_value_examples": missing[:20],
        "unobserved_values_sha256": hashlib.sha256(missing_payload).hexdigest(),
        "dense_histogram_sha256": hashlib.sha256(dense_payload).hexdigest(),
        "least_frequent_observed": [
            {"value": value, "count": count} for value, count in least
        ],
        "most_frequent_observed": [
            {"value": value, "count": count} for value, count in most
        ],
    }


def _paired_length_summary(
    atomic: Sequence[int], factorized: Sequence[int]
) -> dict[str, Any]:
    if len(atomic) != len(factorized) or not atomic:
        raise ValueError("Paired length inputs must be non-empty and equally sized.")
    deltas = [new - old for old, new in zip(atomic, factorized, strict=True)]
    ratios = [new / old for old, new in zip(atomic, factorized, strict=True)]
    return {
        "total_token_delta": sum(deltas),
        "mean_token_delta": round(sum(deltas) / len(deltas), 6),
        "maximum_token_delta": max(deltas),
        "total_length_ratio": round(sum(factorized) / sum(atomic), 8),
        "mean_per_record_length_ratio": round(sum(ratios) / len(ratios), 8),
        "maximum_per_record_length_ratio": round(max(ratios), 8),
    }


def _accumulator_report(accumulator: SplitAccumulator) -> dict[str, Any]:
    atomic = accumulator.lengths["atomic"]
    return {
        "record_count": accumulator.record_count,
        "digest_composition": (
            "direct_length_prefixed_record_stream"
            if accumulator.split != "all"
            else "sha256_of_ordered_train_validation_test_split_digest_bytes"
        ),
        "record_order_sha256": accumulator.record_order_digest.hexdigest(),
        "sequence_sha256": {
            "atomic": accumulator.atomic_sequence_digest.hexdigest(),
            "xy_factorized": accumulator.xy_sequence_digest.hexdigest(),
            "xy_rgb_explicit_path_end": (
                accumulator.xy_rgb_path_end_sequence_digest.hexdigest()
            ),
        },
        "token_class_occurrences": dict(sorted(accumulator.token_class_counts.items())),
        "complete_path_count": accumulator.path_count,
        "lengths": {
            dialect: length_summary(lengths)
            for dialect, lengths in accumulator.lengths.items()
        },
        "paired_inflation_from_atomic": {
            dialect: _paired_length_summary(atomic, accumulator.lengths[dialect])
            for dialect in ("xy_factorized", "xy_rgb_explicit_path_end")
        },
        "component_frequency_coverage": {
            name: _frequency_summary(accumulator.components[name], size)
            for name, size in COMPONENT_SIZES.items()
        },
        "invertibility": {
            "atomic_to_xy_to_atomic": {
                "passed_records": accumulator.id_roundtrip_xy_passed,
                "failed_records": (
                    accumulator.record_count - accumulator.id_roundtrip_xy_passed
                ),
            },
            "atomic_to_xy_rgb_path_end_to_atomic": {
                "passed_records": accumulator.id_roundtrip_xy_rgb_path_end_passed,
                "failed_records": (
                    accumulator.record_count
                    - accumulator.id_roundtrip_xy_rgb_path_end_passed
                ),
            },
        },
        "ambiguous_white_or_gradient_token_occurrences": (
            accumulator.ambiguous_white_or_gradient_count
        ),
    }


def _cross_split_coverage(
    train: SplitAccumulator, evaluation: SplitAccumulator
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for component, size in COMPONENT_SIZES.items():
        train_values = set(train.components[component])
        evaluation_counter = evaluation.components[component]
        unseen_values = sorted(set(evaluation_counter) - train_values)
        unseen_occurrences = sum(evaluation_counter[value] for value in unseen_values)
        total = sum(evaluation_counter.values())
        report[component] = {
            "possible_values": size,
            "evaluation_observed_values": len(evaluation_counter),
            "values_unseen_in_train_count": len(unseen_values),
            "values_unseen_in_train_examples": unseen_values[:20],
            "occurrences_unseen_in_train": unseen_occurrences,
            "unseen_occurrence_fraction": (
                round(unseen_occurrences / total, 8) if total else 0.0
            ),
        }
    return report


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    dialect: CachedOpenVGLabGemmaDialect | None = None,
) -> SplitAccumulator:
    """Analyze already provenance-joined cache records for one split."""

    codec = dialect or CachedOpenVGLabGemmaDialect()
    accumulator = SplitAccumulator(split=split)
    for record_index, record in enumerate(records):
        sample_id = _record_sample_id(record)
        try:
            named_tokens = codec.target_tokens_for_record(record)
            training_ids = _named_tokens_to_training_ids(named_tokens)
            features = validate_framed_training_ids(training_ids)
            xy_tokens = factorize_xy(training_ids)
            xy_rgb_tokens = factorize_xy_rgb_path_end(training_ids)
            if invert_xy(xy_tokens) != training_ids:
                raise FactorizationAuditError("XY factorization ID round-trip mismatch.")
            accumulator.id_roundtrip_xy_passed += 1
            if invert_xy_rgb_path_end(xy_rgb_tokens) != training_ids:
                raise FactorizationAuditError("XY+RGB factorization ID round-trip mismatch.")
            accumulator.id_roundtrip_xy_rgb_path_end_passed += 1

            expected_xy_length = len(training_ids) + features.coordinate_count
            expected_full_length = (
                len(training_ids)
                + features.coordinate_count
                + 3 * features.rgb_color_count
                + features.special_color_count
            )
            if len(xy_tokens) != expected_xy_length:
                raise FactorizationAuditError("XY factorized length formula mismatch.")
            if len(xy_rgb_tokens) != expected_full_length:
                raise FactorizationAuditError("XY+RGB factorized length formula mismatch.")
            accumulator.add(
                sample_id=sample_id,
                training_ids=training_ids,
                features=features,
                xy_tokens=xy_tokens,
                xy_rgb_tokens=xy_rgb_tokens,
            )
        except Exception as exc:
            if isinstance(exc, FactorizationAuditError):
                error = exc
            else:
                error = FactorizationAuditError(str(exc))
            raise FactorizationAuditError(
                f"{split} record {record_index} ({sample_id!r}) failed: {error}"
            ) from exc
    return accumulator


def run_audit(config: OpenVGLabCacheConfig) -> dict[str, Any]:
    """Load all pinned splits and return a deterministic feasibility report."""

    accumulators: dict[str, SplitAccumulator] = {}
    with PinnedOpenVGLabTokenCache(config) as cache:
        for split in ("train", "validation", "test"):
            records = cache.load_split(split)
            accumulators[split] = analyze_records(records, split=split)

    combined = SplitAccumulator(split="all")
    for split in ("train", "validation", "test"):
        combined.merge(accumulators[split])

    exact_size = sum(entry["count"] for entry in EXACT_TOKEN_CLASSES.values())
    gap_size = sum(entry["count"] for entry in RESERVED_GAPS)
    if exact_size != VOCABULARY_FORMULAS["atomic_exact_encoder_producible"]["size"]:
        raise FactorizationAuditError("Exact vocabulary formula is internally inconsistent.")
    if exact_size + gap_size != OPENVGLAB_NAMED_VOCABULARY_SIZE:
        raise FactorizationAuditError("Exact ranges plus gaps do not cover the named namespace.")
    if combined.record_count != sum(config.expected_split_counts.values()):
        raise FactorizationAuditError("Combined record count does not match the pinned contract.")

    contract_payload = json.dumps(
        FACTORIZATION_CONTRACT, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    cache_source = Path(inspect.getsourcefile(cache_module) or cache_module.__file__).resolve()
    this_source = Path(__file__).resolve()
    return {
        "audit": "official_discrete_factorization_feasibility",
        "schema_version": 1,
        "status": "PASS",
        "scope": "cpu_only_offline_audit_no_production_codec_change",
        "source_provenance": {
            "openvglab_training_commit": OPENVGLAB_TRAIN_COMMIT,
            "openvglab_tokenization_config_sha256": (
                OPENVGLAB_TOKENIZATION_CONFIG_SHA256
            ),
            "cache": {
                "path": str(config.cache_path),
                "sha256": config.expected_cache_sha256,
            },
            "audit_manifest": {
                "path": str(config.audit_manifest_path),
                "sha256": config.expected_audit_sha256,
            },
            "prepared_inputs": {
                split: {
                    "path": str(config.prepared_root / f"{split}.jsonl"),
                    "sha256": config.expected_input_sha256[split],
                    "record_count": config.expected_split_counts[split],
                }
                for split in ("train", "validation", "test")
            },
            "verified_loader_source": {
                "path": str(cache_source),
                "sha256": _sha256_file(cache_source),
            },
            "audit_script_source": {
                "path": str(this_source),
                "sha256": _sha256_file(this_source),
            },
        },
        "exact_encoder_producible_training_id_classes": EXACT_TOKEN_CLASSES,
        "reserved_contiguous_namespace_gaps": list(RESERVED_GAPS),
        "range_accounting": {
            "exact_encoder_producible": exact_size,
            "reserved_gap": gap_size,
            "current_contiguous_named_namespace": OPENVGLAB_NAMED_VOCABULARY_SIZE,
            "minimum_registered_upstream_id": OPENVGLAB_MIN_PRODUCIBLE_ID,
            "maximum_registered_upstream_id": OPENVGLAB_MAX_PRODUCIBLE_ID,
        },
        "factorization_contract": FACTORIZATION_CONTRACT,
        "factorization_contract_sha256": hashlib.sha256(contract_payload).hexdigest(),
        "vocabulary_size_formulas": VOCABULARY_FORMULAS,
        "context_thresholds_are_target_tokens_only": list(CONTEXT_THRESHOLDS),
        "splits": {
            split: _accumulator_report(accumulators[split])
            for split in ("train", "validation", "test")
        },
        "all_splits": _accumulator_report(combined),
        "heldout_component_coverage_against_train": {
            split: _cross_split_coverage(accumulators["train"], accumulators[split])
            for split in ("validation", "test")
        },
        "invertibility_scope": {
            "cached_token_id_roundtrip": True,
            "source_svg_semantic_roundtrip": False,
            "source_svg_semantic_limitation": FACTORIZATION_CONTRACT[
                "known_source_ambiguity"
            ],
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace ``path`` with one complete, deterministic JSON document."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=repository_root / "artifacts/omnisvg_train_4b_mmsvg_20k_tokens.sqlite3",
    )
    parser.add_argument(
        "--audit-manifest-path",
        type=Path,
        default=repository_root / "artifacts/omnisvg_train_4b_mmsvg_20k_token_audit.json",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Directory containing the pinned train/validation/test JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "artifacts/official_discrete_factorization_feasibility.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = OpenVGLabCacheConfig(
        cache_path=args.cache_path,
        audit_manifest_path=args.audit_manifest_path,
        prepared_root=args.prepared_root,
        expected_cache_sha256=PINNED_CACHE_SHA256,
        expected_audit_sha256=PINNED_AUDIT_SHA256,
        expected_input_sha256=PINNED_INPUT_SHA256,
        expected_split_counts=PINNED_SPLIT_COUNTS,
        allow_live_reencode=False,
    )
    report = run_audit(config)
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "records": report["all_splits"]["record_count"],
                "output": str(args.output.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
