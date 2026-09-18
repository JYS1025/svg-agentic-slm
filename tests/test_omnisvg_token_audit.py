"""Fixture-only tests for streaming MMSVG token-length auditing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from svg_agentic_slm.svg.codec_backends import (
    OPENVGLAB_TRAIN_4B_METADATA,
    CodecDecodeResult,
    CodecEncodeError,
    CodecEncodeResult,
    TokenValue,
)
from svg_agentic_slm.svg.token_audit import (
    AuditRecordError,
    FieldChatPrefixCounter,
    TokenAuditConfig,
    TokenAuditRunner,
    _normalize_chat_token_ids,
)


class _LengthFixtureCodec:
    @property
    def metadata(self):
        return OPENVGLAB_TRAIN_4B_METADATA

    def encode(self, svg: str) -> CodecEncodeResult:
        if svg == "fail":
            raise CodecEncodeError("fixture failure", code="fixture_failure")
        length = int(svg)
        body = tuple(range(length))
        return CodecEncodeResult(
            metadata=self.metadata,
            body_tokens=body,
            tokens=(196998, *body, 196999),
            source_sha256=hashlib.sha256(svg.encode()).hexdigest(),
            diagnostics={"masking_enabled": False},
        )

    def decode(self, tokens: list[TokenValue]) -> CodecDecodeResult:
        raise AssertionError("audit must not decode")


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    return payload


class _BatchEncodingFixture(dict):
    pass


class _TensorFixture:
    def __init__(self, value) -> None:
        self._value = value

    def tolist(self):
        return self._value


def test_chat_token_normalization_accepts_batch_encoding_and_single_tensor_batch() -> None:
    encoded = _BatchEncodingFixture(input_ids=_TensorFixture([[101, 102, 103]]))

    assert _normalize_chat_token_ids(encoded) == (101, 102, 103)
    assert _normalize_chat_token_ids({"input_ids": [201, 202]}) == (201, 202)

    with pytest.raises(AuditRecordError, match="multiple batches"):
        _normalize_chat_token_ids({"input_ids": [[1, 2], [3, 4]]})
    with pytest.raises(AuditRecordError, match="multiple batches"):
        _normalize_chat_token_ids({"input_ids": [True]})


def test_streaming_three_split_audit_writes_manifest_and_indexed_cache(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    payloads = {
        "train": _write_jsonl(
            prepared / "train.jsonl",
            [
                {"id": "a", "domain": "icon", "output_svg": "1", "chat_prefix": 5},
                {
                    "metadata": {"id": "b", "domain": "icon", "source": "mmsvg-icon"},
                    "output_svg": "2",
                    "chat_prefix": 5,
                },
            ],
        ),
        "validation": _write_jsonl(
            prepared / "validation.jsonl",
            [
                {
                    "id": "c",
                    "domain": "illustration",
                    "output_svg": "3",
                    "chat_prefix": 5,
                }
            ],
        ),
        "test": _write_jsonl(
            prepared / "test.jsonl",
            [
                {"id": "d", "domain": "icon", "output_svg": "4", "chat_prefix": 5},
                {
                    "metadata": {"source_id": "e", "domain": "icon", "source": "mmsvg"},
                    "output_svg": "fail",
                    "chat_prefix": 5,
                },
            ],
        ),
    }
    output = tmp_path / "audit" / "manifest.json"
    cache = tmp_path / "audit" / "tokens.sqlite3"

    manifest = TokenAuditRunner(
        codec=_LengthFixtureCodec(),
        chat_prefix_counter=FieldChatPrefixCounter("chat_prefix"),
        config=TokenAuditConfig(
            prepared_root=prepared,
            output_path=output,
            token_cache_path=cache,
            thresholds=(2, 4, 8),
        ),
    ).run()

    persisted = json.loads(output.read_text())
    assert persisted == manifest
    assert manifest["counts"] == {
        "records": 5,
        "successes": 4,
        "failures": 1,
        "by_split": {
            "train": {"records": 2, "successes": 2, "failures": 0},
            "validation": {"records": 1, "successes": 1, "failures": 0},
            "test": {"records": 2, "successes": 1, "failures": 1},
        },
    }
    assert manifest["metrics"]["body"]["p50"] == 2
    assert manifest["metrics"]["body"]["p95"] == 4
    assert manifest["metrics"]["body"]["p99"] == 4
    assert manifest["metrics"]["body"]["max"] == 4
    assert manifest["metrics"]["bos_eos"]["max"] == 6
    assert manifest["metrics"]["full_chat"]["max"] == 11
    assert manifest["metrics"]["body"]["greater_than_threshold"] == {
        "2": 2,
        "4": 0,
        "8": 0,
    }
    assert manifest["failures"][0]["error"]["code"] == "fixture_failure"
    assert manifest["failures"][0]["id"] == "e"
    assert manifest["failures"][0]["domain"] == "icon"
    assert manifest["failures"][0]["source"] == "mmsvg"
    assert manifest["determinism"]["masking_enabled"] is False
    for split, payload in payloads.items():
        assert manifest["inputs"][split]["sha256"] == hashlib.sha256(payload).hexdigest()

    with sqlite3.connect(cache) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sequences").fetchone() == (4,)
        assert connection.execute(
            "SELECT body_count, framed_count FROM sequences WHERE sample_id = 'd'"
        ).fetchone() == (4, 6)
        assert connection.execute(
            "SELECT domain, source FROM sequences WHERE sample_id = 'b'"
        ).fetchone() == ("icon", "mmsvg-icon")
        assert connection.execute(
            "SELECT value_json FROM metadata WHERE key = 'integer_encoding'"
        ).fetchone() == ('"uint32-little-endian-zlib"',)

    assert not list(output.parent.glob(".*.tmp"))
