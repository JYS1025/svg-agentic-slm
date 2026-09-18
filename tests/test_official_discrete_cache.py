from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import zlib
from pathlib import Path

import pytest

import svg_agentic_slm.svg.official_discrete_cache as cache_module
from svg_agentic_slm.svg.official_discrete_cache import (
    OPENVGLAB_BOS_ID,
    OPENVGLAB_EOS_ID,
    OPENVGLAB_NAMED_VOCABULARY_SIZE,
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    CachedOpenVGLabGemmaDialect,
    OfficialDiscreteCacheError,
    OpenVGLabCacheConfig,
    UnsupportedOfficialDiscreteOperation,
    load_all_cached_records,
    load_cached_splits_records,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack(ids: tuple[int, ...]) -> bytes:
    return zlib.compress(struct.pack(f"<{len(ids)}I", *ids), level=9)


def _codec_metadata() -> dict:
    return {
        "backend_id": "openvglab-omnisvg-train-4b-812489fd9d191e39fe94bc0c4027e5d0121e0fc6",
        "codec_name": "OpenVGLab OmniSVG released training encoder",
        "codec_version": "812489fd9d191e39fe94bc0c4027e5d0121e0fc6",
        "dialect": "openvglab-training-code-4b",
        "directions": ["encode"],
        "lossy": True,
        "model_family": "Qwen2.5-VL",
        "model_size": "4B",
        "notes": ["fixture"],
        "official_checkpoint_compatible": False,
        "provenance": "official-training-source",
        "source_revision": "812489fd9d191e39fe94bc0c4027e5d0121e0fc6",
        "source_url": "fixture",
        "token_kind": "absolute-integer-id",
    }


def _fixture(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True)
    input_hashes = {}
    counts = {"train": 2, "validation": 1, "test": 1}
    records = {
        "train": [
            {
                "metadata": {"record_id": "a" * 32},
                "instruction": "one",
                "output_svg": "<svg a='1'/>",
            },
            {"id": "b" * 32, "instruction": "two", "output_svg": "<svg b='2'/>"},
        ],
        "validation": [
            {"id": "c" * 32, "instruction": "three", "output_svg": "<svg c='3'/>"}
        ],
        "test": [
            {"id": "d" * 32, "instruction": "four", "output_svg": "<svg d='4'/>"}
        ],
    }
    for split, split_records in records.items():
        path = prepared / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in split_records),
            encoding="utf-8",
        )
        input_hashes[split] = _sha256(path)

    cache = tmp_path / "tokens.sqlite3"
    connection = sqlite3.connect(cache)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE sequences (
            sample_index INTEGER PRIMARY KEY, split TEXT NOT NULL,
            source_line INTEGER NOT NULL, sample_id TEXT, domain TEXT, source TEXT,
            body_count INTEGER NOT NULL, framed_count INTEGER NOT NULL,
            body_uint32_le_zlib BLOB NOT NULL, framed_uint32_le_zlib BLOB NOT NULL
        );
        CREATE INDEX sequences_split_sample_id ON sequences(split, sample_id);
        """
    )
    metadata = {
        "cache_schema_version": 1,
        "integer_encoding": "uint32-little-endian-zlib",
        "codec": _codec_metadata(),
    }
    connection.executemany(
        "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in metadata.items()],
    )
    sample_index = 0
    for split in ("train", "validation", "test"):
        for source_line, record in enumerate(records[split], start=1):
            body = (151938, 151943 + source_line)
            framed = (OPENVGLAB_BOS_ID, *body, OPENVGLAB_EOS_ID)
            connection.execute(
                """
                INSERT INTO sequences VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    sample_index,
                    split,
                    source_line,
                    record.get("id", record.get("metadata", {}).get("record_id")),
                    len(body),
                    len(framed),
                    _pack(body),
                    _pack(framed),
                ),
            )
            sample_index += 1
    connection.commit()
    connection.close()

    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "codec": _codec_metadata(),
                "inputs": {
                    split: {"sha256": digest} for split, digest in input_hashes.items()
                },
                "counts": {"failures": 0},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = OpenVGLabCacheConfig(
        cache_path=cache,
        audit_manifest_path=audit,
        prepared_root=prepared,
        expected_cache_sha256=_sha256(cache),
        expected_audit_sha256=_sha256(audit),
        expected_input_sha256=input_hashes,
        expected_split_counts=counts,
    )
    return config


def test_model_neutral_namespace_covers_boundaries_and_fails_decode_closed() -> None:
    dialect = CachedOpenVGLabGemmaDialect()
    tokens = dialect.vocabulary_tokens()

    assert dialect.metadata.backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
    assert len(tokens) == OPENVGLAB_NAMED_VOCABULARY_SIZE == 45062
    assert tokens[0] == "<svgovg4b:151938>"
    assert tokens[-2:] == ("<svgovg4b:196998>", "<svgovg4b:196999>")
    assert dialect.codec_manifest()["qwen_checkpoint_compatibility_claimed"] is False
    with pytest.raises(UnsupportedOfficialDiscreteOperation, match="cache-only"):
        dialect.encode("<svg/>")
    with pytest.raises(UnsupportedOfficialDiscreteOperation, match="No audited decoder"):
        dialect.decode(tokens[:2])


def test_pinned_cache_binds_split_line_sample_and_svg_identity(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    records = load_all_cached_records(config)
    dialect = CachedOpenVGLabGemmaDialect()

    assert {split: len(rows) for split, rows in records.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    target = dialect.target_for_record(records["train"][0])
    assert target == (
        "<svgovg4b:196998><svgovg4b:151938>"
        "<svgovg4b:151944><svgovg4b:196999>"
    )
    payload = records["train"][0]["_openvglab_4b_cached_target_v1"]
    assert payload["split"] == "train"
    assert payload["source_line"] == 1
    assert payload["sample_id"] == "a" * 32
    assert len(payload["svg_sha256"]) == 64


def test_selective_cache_never_touches_unrequested_test_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    forbidden_path = config.prepared_root / "test.jsonl"
    forbidden_path.unlink()
    filesystem_accesses: list[str] = []
    statements: list[str] = []

    for method_name in ("resolve", "stat", "is_file", "open"):
        original_method = getattr(Path, method_name)

        def guarded_path_method(self, *args, _name=method_name, _original=original_method, **kwargs):
            if self == forbidden_path:
                filesystem_accesses.append(_name)
                raise AssertionError(f"forbidden test split filesystem access via {_name}")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded_path_method)

    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(cache_module.sqlite3, "connect", traced_connect)

    records = load_cached_splits_records(config, ("train", "validation"))

    assert {split: len(rows) for split, rows in records.items()} == {
        "train": 2,
        "validation": 1,
    }
    assert filesystem_accesses == []
    assert not any("'test'" in statement for statement in statements)

    statements.clear()
    with pytest.raises(AssertionError, match="forbidden test split filesystem access"):
        load_cached_splits_records(config, ("test",))
    assert filesystem_accesses
    assert any("'test'" in statement for statement in statements)


def test_cache_and_input_drift_fail_closed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config.cache_path.write_bytes(config.cache_path.read_bytes() + b"drift")
    with pytest.raises(OfficialDiscreteCacheError, match="SQLite token cache SHA-256"):
        load_all_cached_records(config)

    config = _fixture(tmp_path / "second")
    train_path = config.prepared_root / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(OfficialDiscreteCacheError, match="blank line|input SHA-256"):
        load_all_cached_records(config)
