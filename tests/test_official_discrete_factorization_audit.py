from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from scripts import audit_official_discrete_factorization as audit
from svg_agentic_slm.svg.official_discrete_cache import (
    CACHED_TARGET_FIELD,
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    OPENVGLAB_BOS_ID,
    OPENVGLAB_EOS_ID,
)


def _representative_ids() -> tuple[int, ...]:
    return (
        OPENVGLAB_BOS_ID,
        audit.COMMAND_MIN,
        audit.COORDINATE_MIN,
        audit.COORDINATE_MAX,
        audit.COMMAND_MIN + 2,
        audit.COORDINATE_MIN + 201,
        audit.COORDINATE_MIN + 402,
        audit.COORDINATE_MIN + 603,
        audit.COMMAND_MIN + 3,
        audit.COORDINATE_MIN + 804,
        audit.ARC_MIN,
        audit.ARC_MAX,
        audit.ARC_MIN + 1,
        audit.COORDINATE_MIN + 1005,
        audit.COMMAND_MAX,
        audit.COORDINATE_MIN + 1206,
        audit.COLOR_MIN + 2 + 0xABC,
        audit.COMMAND_MIN + 1,
        audit.COORDINATE_MIN + 1407,
        audit.COLOR_MIN + 1,
        OPENVGLAB_EOS_ID,
    )


def _cached_record(ids: tuple[int, ...], sample_id: str = "fixture") -> dict:
    raw = struct.pack(f"<{len(ids)}I", *ids)
    return {
        CACHED_TARGET_FIELD: {
            "backend_id": OFFICIAL_CACHED_GEMMA_BACKEND_ID,
            "cache_sha256": "fixture-hash",
            "expected_cache_sha256": "fixture-hash",
            "sample_id": sample_id,
            "framed_count": len(ids),
            "framed_uint32_le_zlib": zlib.compress(raw),
        }
    }


def test_exact_encoder_ranges_and_vocabulary_formulas() -> None:
    boundaries = {
        audit.COMMAND_MIN: "command",
        audit.COMMAND_MAX: "command",
        audit.COORDINATE_MIN: "coordinate",
        audit.COORDINATE_MAX: "coordinate",
        audit.COLOR_MIN: "color",
        audit.COLOR_MAX: "color",
        audit.ARC_MIN: "arc",
        audit.ARC_MAX: "arc",
        OPENVGLAB_BOS_ID: "bos",
        OPENVGLAB_EOS_ID: "eos",
    }
    assert {token_id: audit.training_token_kind(token_id) for token_id in boundaries} == {
        token_id: expected for token_id, expected in boundaries.items()
    }
    for reserved_id in (191943, 191945, 196044, 196435, 196536, 196997):
        with pytest.raises(audit.FactorizationAuditError, match="not in an exact"):
            audit.training_token_kind(reserved_id)

    formulas = audit.VOCABULARY_FORMULAS
    assert formulas["current_contiguous_named_namespace"]["size"] == 45062
    assert formulas["atomic_exact_encoder_producible"]["size"] == 44205
    assert formulas["xy_factorized"]["size"] == 4605
    assert formulas["xy_rgb_explicit_path_end"]["size"] == 558
    assert sum(gap["count"] for gap in audit.RESERVED_GAPS) == 857


def test_factorized_dialects_are_id_invertible_and_have_expected_lengths() -> None:
    ids = _representative_ids()
    features = audit.validate_framed_training_ids(ids)

    assert features.coordinate_count == 9
    assert features.rgb_color_count == 1
    assert features.special_color_count == 1
    assert features.path_count == 2

    xy = audit.factorize_xy(ids)
    xy_rgb = audit.factorize_xy_rgb_path_end(ids)
    assert len(ids) == 21
    assert len(xy) == 30
    assert len(xy_rgb) == 34
    assert audit.invert_xy(xy) == ids
    assert audit.invert_xy_rgb_path_end(xy_rgb) == ids
    assert ("r", 0xA) in xy_rgb
    assert ("g", 0xB) in xy_rgb
    assert ("b", 0xC) in xy_rgb
    assert xy_rgb.count(("path_end", 0)) == 2


def test_max_color_id_roundtrips_but_reports_source_semantic_ambiguity() -> None:
    ids = (
        OPENVGLAB_BOS_ID,
        audit.COMMAND_MIN + 1,
        audit.COORDINATE_MIN,
        audit.COLOR_MAX,
        OPENVGLAB_EOS_ID,
    )

    factorized = audit.factorize_xy_rgb_path_end(ids)
    assert ("r", 15) in factorized
    assert ("g", 15) in factorized
    assert ("b", 15) in factorized
    assert audit.invert_xy_rgb_path_end(factorized) == ids
    assert "gradient" in audit.FACTORIZATION_CONTRACT["known_source_ambiguity"]


def test_split_analysis_reports_lengths_component_coverage_and_roundtrips() -> None:
    ids = _representative_ids()
    accumulator = audit.analyze_records(
        [_cached_record(ids, "one"), _cached_record(ids, "two")], split="train"
    )
    report = audit._accumulator_report(accumulator)

    assert report["record_count"] == 2
    assert report["lengths"]["atomic"]["mean"] == 21.0
    assert report["lengths"]["xy_factorized"]["maximum"] == 30
    assert report["lengths"]["xy_rgb_explicit_path_end"]["maximum"] == 34
    assert report["lengths"]["xy_rgb_explicit_path_end"]["context_thresholds"][
        "2304"
    ]["above_count"] == 0
    assert report["component_frequency_coverage"]["x"]["total_occurrences"] == 18
    assert report["component_frequency_coverage"]["r"]["observed_values"] == 1
    assert report["component_frequency_coverage"]["path_end"][
        "total_occurrences"
    ] == 4
    assert report["invertibility"]["atomic_to_xy_to_atomic"]["passed_records"] == 2
    assert report["invertibility"]["atomic_to_xy_rgb_path_end_to_atomic"][
        "failed_records"
    ] == 0


def test_factorized_inverse_rejects_missing_explicit_path_end() -> None:
    tokens = list(audit.factorize_xy_rgb_path_end(_representative_ids()))
    tokens.remove(("path_end", 0))

    with pytest.raises(audit.FactorizationAuditError, match="PATH_END"):
        audit.invert_xy_rgb_path_end(tokens)


def test_atomic_json_writer_replaces_complete_document_and_cleans_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    audit.write_json_atomic(output, {"status": "PASS", "value": 7})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "PASS",
        "value": 7,
    }
    stable_contents = output.read_text(encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("fixture replace failure")

    monkeypatch.setattr(audit.os, "replace", fail_replace)
    with pytest.raises(OSError, match="fixture replace failure"):
        audit.write_json_atomic(output, {"status": "new"})

    assert output.read_text(encoding="utf-8") == stable_contents
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []
