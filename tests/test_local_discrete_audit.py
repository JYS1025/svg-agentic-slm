from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from svg_agentic_slm.svg.official_discrete_cache import (
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    OPENVGLAB_NAMED_VOCABULARY_SIZE,
)
from svg_agentic_slm.svg.local_discrete_audit import (
    AuditDataConfig,
    AuditOutputConfig,
    AuditThresholdConfig,
    LocalDiscreteAuditConfig,
    TokenizerAuditConfig,
    load_local_discrete_audit_config,
    run_local_discrete_audit,
)


class _FakeCodec:
    def __init__(self) -> None:
        self.tokens = tuple(
            [f"<c{index:05d}>" for index in range(OPENVGLAB_NAMED_VOCABULARY_SIZE - 1)]
            + ["<codec-eos>"]
        )
        self.metadata = SimpleNamespace(backend_id=OFFICIAL_CACHED_GEMMA_BACKEND_ID)

    def vocabulary_tokens(self) -> tuple[str, ...]:
        return self.tokens

    def target_tokens_for_record(self, record) -> tuple[str, ...]:
        svg = record["svg"]
        match = re.search(r'data-length="(\d+)"', svg)
        if match is None:
            raise ValueError("missing fake data length")
        length = int(match.group(1))
        return (*self.tokens[: length - 1], self.tokens[-1])

    def target_for_record(self, record) -> str:
        return "".join(self.target_tokens_for_record(record))

    def codec_manifest(self) -> dict[str, object]:
        return {"vocabulary_size": len(self.tokens)}


class _FakeTokenizer:
    def __init__(self) -> None:
        self.vocabulary = {"<unk>": 0, "<chat-a>": 1, "<chat-b>": 2}

    def install(self, tokens: tuple[str, ...]) -> tuple[int, ...]:
        ids = tuple(range(10, 10 + len(tokens)))
        self.vocabulary.update(zip(tokens, ids, strict=True))
        return ids

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        pieces = re.findall(r"<[^>]+>", value)
        return [self.vocabulary[piece] for piece in pieces]


class _FakeResponseOnlyDataset:
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        records,
        *,
        tokenizer,
        instruction_mode,
        target_representation,
        max_seq_length,
        seed,
        codec,
    ) -> None:
        record = records[0]
        target_ids = tokenizer.encode(
            codec.target_for_record(record), add_special_tokens=False
        )
        labels = [-100, -100, *target_ids]
        if record.get("bad_label"):
            labels[-1] = 9
        self.sample = {"input_ids": [1, 2, *target_ids], "labels": labels}
        self.calls.append(
            {
                "instruction_mode": instruction_mode,
                "target_representation": target_representation,
                "max_seq_length": max_seq_length,
                "seed": seed,
            }
        )

    def __getitem__(self, index: int):
        assert index == 0
        return self.sample


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(tmp_path: Path, *, expected_total: int) -> LocalDiscreteAuditConfig:
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir(exist_ok=True)
    return LocalDiscreteAuditConfig(
        tokenizer=TokenizerAuditConfig(
            path=tokenizer_path.resolve(), revision="a" * 40
        ),
        data=AuditDataConfig(
            train_path=tmp_path / "train.jsonl",
            validation_path=tmp_path / "validation.jsonl",
            test_path=tmp_path / "test.jsonl",
        ),
        output=AuditOutputConfig(
            report_path=tmp_path / "report.json",
            longest_record_path=tmp_path / "longest.jsonl",
        ),
        thresholds=AuditThresholdConfig(
            codec_tokens=(2, 4),
            labeled_tokens=(2, 4),
            full_chat_tokens=(4, 6),
            maximum_codec_tokens=10,
            maximum_labeled_tokens=10,
            maximum_full_chat_tokens=12,
        ),
        expected_total_records=expected_total,
    )


def _dependencies():
    tokenizer = _FakeTokenizer()
    codec = _FakeCodec()

    def register(fake_tokenizer, fake_codec):
        ids = fake_tokenizer.install(fake_codec.vocabulary_tokens())
        digest = hashlib.sha256(
            json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return SimpleNamespace(
            token_ids=ids,
            token_ids_sha256=digest,
            vocabulary_sha256="f" * 64,
            added_token_count=len(ids),
        )

    def record_loader(config):
        loaded = {}
        for split, path in config.data.items():
            loaded[split] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return loaded

    return {
        "tokenizer_loader": lambda _config: tokenizer,
        "codec_factory": lambda _grid_size: codec,
        "registration_function": register,
        "dataset_factory": _FakeResponseOnlyDataset,
        "record_loader": record_loader,
    }


def test_config_requires_offline_absolute_pinned_tokenizer(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """
local_discrete_audit:
  tokenizer:
    path: ./tokenizer
    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    local_files_only: true
    trust_remote_code: false
  official_cache:
    cache_path: cache.sqlite3
    expected_cache_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    audit_manifest_path: audit.json
    expected_audit_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    prepared_root: .
    expected_input_sha256:
      train: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      validation: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
      test: eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
    expected_split_counts:
      train: 2
      validation: 1
      test: 1
    allow_live_reencode: false
  data:
    train_path: train.jsonl
    validation_path: validation.jsonl
    test_path: test.jsonl
  output:
    report_path: report.json
""",
        encoding="utf-8",
    )
    config = load_local_discrete_audit_config(config_path)
    assert config.tokenizer.path == (tmp_path / "tokenizer").resolve()
    assert config.tokenizer.local_files_only is True
    with pytest.raises(ValueError, match="40-hex"):
        TokenizerAuditConfig(path=tmp_path.resolve(), revision="main")
    with pytest.raises(ValueError, match="local_files_only"):
        TokenizerAuditConfig(
            path=tmp_path.resolve(), revision="b" * 40, local_files_only=False
        )


def test_streaming_audit_hashes_lengths_labels_and_longest_record(tmp_path: Path) -> None:
    _FakeResponseOnlyDataset.calls.clear()
    config = _config(tmp_path, expected_total=4)
    _write_jsonl(
        config.data.train_path,
        [
            {"id": "train-short", "description": "short", "svg": '<svg data-length="2"/>'},
            {"id": "train-mid", "description": "mid", "svg": '<svg data-length="3"/>'},
        ],
    )
    _write_jsonl(
        config.data.validation_path,
        [{"id": "validation", "description": "validation", "svg": '<svg data-length="4"/>'}],
    )
    _write_jsonl(
        config.data.test_path,
        [{"id": "test-long", "description": "long", "svg": '<svg data-length="5"/>'}],
    )

    report = run_local_discrete_audit(config, **_dependencies())

    assert report["status"] == "pass"
    assert report["offline"] is True and report["model_loaded"] is False
    assert report["codec"]["token_count"] == OPENVGLAB_NAMED_VOCABULARY_SIZE
    assert report["codec"]["maximum_registered_id"] == 10 + OPENVGLAB_NAMED_VOCABULARY_SIZE - 1
    statistics = report["overall"]["statistics"]
    codec_lengths = statistics["codec_token_length"]
    assert codec_lengths["count"] == 4
    assert codec_lengths["p50"] == 3
    assert codec_lengths["p95"] == 5
    assert codec_lengths["p99"] == 5
    assert codec_lengths["max"] == 5
    assert statistics["labeled_token_length"]["max"] == 5
    assert statistics["full_chat_token_length"]["max"] == 7
    assert statistics["maximum_observed_id"] == 10 + OPENVGLAB_NAMED_VOCABULARY_SIZE - 1
    assert report["longest_record"]["record_id"] == "test-long"
    assert report["inputs"]["train"]["input_sha256"] == hashlib.sha256(
        config.data.train_path.read_bytes()
    ).hexdigest()
    assert json.loads(config.output.report_path.read_text(encoding="utf-8"))["status"] == "pass"
    longest = json.loads(config.output.longest_record_path.read_text(encoding="utf-8"))
    assert longest["id"] == "test-long"
    assert longest["official_discrete_audit"]["full_chat_tokens"] == 7
    assert all(call["instruction_mode"] == "description_only" for call in _FakeResponseOnlyDataset.calls)
    assert all(call["seed"] == 42 for call in _FakeResponseOnlyDataset.calls)


def test_label_id_mismatch_is_recorded_without_network(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_total=3)
    _write_jsonl(
        config.data.train_path,
        [
            {
                "id": "bad",
                "description": "bad",
                "svg": '<svg data-length="2"/>',
                "bad_label": True,
            }
        ],
    )
    _write_jsonl(
        config.data.validation_path,
        [{"id": "valid", "description": "valid", "svg": '<svg data-length="2"/>'}],
    )
    _write_jsonl(
        config.data.test_path,
        [{"id": "valid-test", "description": "valid", "svg": '<svg data-length="2"/>'}],
    )

    report = run_local_discrete_audit(config, **_dependencies())

    assert report["status"] == "fail"
    assert report["failures"]["count"] == 1
    assert report["failures"]["details"][0]["code"] == "labeled_target_id_mismatch"
    assert report["inputs"]["train"]["failed_records"] == 1
    assert config.output.report_path.is_file()
