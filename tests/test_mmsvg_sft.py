"""Tests for immutable fixed-ID MMSVG SFT preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from svg_agentic_slm.data import mmsvg_sft


def test_fixed_ids_define_split_membership_and_manifest_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    _write_rows(
        Path(config.icon.paths[0]),
        [_row("i3"), _row("i2"), _row("not-selected"), _row("i1")],
    )
    _write_rows(
        Path(config.illustration.paths[0]),
        [_row("a3"), _row("a1"), _row("a2")],
    )
    monkeypatch.setattr(mmsvg_sft, "_validate_config", lambda _config: None)

    manifest = mmsvg_sft.prepare_mmsvg_sft_dataset(config)

    records = {
        split: [
            json.loads(line)
            for line in (config.output_dir / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for split in ("train", "validation", "test")
    }
    assert {row["metadata"]["record_id"] for row in records["train"]} == {"i1", "a1"}
    assert {row["metadata"]["record_id"] for row in records["validation"]} == {
        "i2",
        "a2",
    }
    assert {row["metadata"]["record_id"] for row in records["test"]} == {"i3", "a3"}
    for row in (row for split_rows in records.values() for row in split_rows):
        policy = row["metadata"]["instruction_policy"]
        assert isinstance(policy["r1_detail_60_description_40"], bool)
        expected_source = (
            "detail" if policy["r1_detail_60_description_40"] else "description"
        )
        assert row["metadata"]["mixed_instruction_source"] == expected_source

    icon_id_path = config.selected_ids.icon
    assert manifest["selected_id_files"]["icon"]["sha256"] == hashlib.sha256(
        icon_id_path.read_bytes()
    ).hexdigest()
    assert manifest["scan_statistics"]["icon"]["source_rows_scanned"] == 4
    assert manifest["scan_statistics"]["icon"]["fixed_id_rows_seen"] == 3
    assert manifest["scan_statistics"]["icon"]["eligible_not_selected"] == 0
    assert manifest["fixed_id_selection"]["replacement_sampling"] is False


def test_fixed_selection_fails_on_missing_or_rejected_without_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    rejected = _row("i2")
    rejected["detail"] = ""
    _write_rows(
        Path(config.icon.paths[0]),
        [_row("i1"), rejected, _row("not-selected"), _row("also-not-selected")],
    )
    _write_rows(
        Path(config.illustration.paths[0]),
        [_row("a1"), _row("a2"), _row("a3")],
    )
    config.selected_ids.icon.write_text("i1\ni2\nmissing\n", encoding="utf-8")
    monkeypatch.setattr(mmsvg_sft, "_validate_config", lambda _config: None)

    with pytest.raises(RuntimeError, match=r"missing=1.*rejected=1.*No replacement"):
        mmsvg_sft.prepare_mmsvg_sft_dataset(config)

    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("one\n\nthree\n", "Blank fixed"),
        ("one\none\nthree\n", "Duplicate fixed"),
        ("one\ntwo\n", "exactly 3"),
    ],
)
def test_fixed_id_file_validation_is_fail_closed(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "ids.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        mmsvg_sft._load_selected_ids(path, 3, "icon")


def test_duplicate_selected_id_in_raw_source_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    _write_rows(
        Path(config.icon.paths[0]),
        [_row("i1"), _row("i1", variant="duplicate"), _row("i2"), _row("i3")],
    )
    _write_rows(
        Path(config.illustration.paths[0]),
        [_row("a1"), _row("a2"), _row("a3")],
    )
    monkeypatch.setattr(mmsvg_sft, "_validate_config", lambda _config: None)

    with pytest.raises(RuntimeError, match="appears more than once"):
        mmsvg_sft.prepare_mmsvg_sft_dataset(config)


def test_exclusion_jsonl_preserves_unicode_line_separator(tmp_path: Path) -> None:
    caption = "first visual line second visual line"
    path = tmp_path / "exclusions.jsonl"
    path.write_text(
        json.dumps({"caption": caption}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exclusions = mmsvg_sft._load_exclusions((path,))

    assert mmsvg_sft._caption_hash(caption) in exclusions.caption_hashes


def _tiny_config(root: Path) -> mmsvg_sft.MMSVGPreparationConfig:
    icon_path = root / "icon.jsonl"
    illustration_path = root / "illustration.jsonl"
    icon_ids = root / "icon_ids.txt"
    illustration_ids = root / "illustration_ids.txt"
    icon_ids.write_text("i1\ni2\ni3\n", encoding="utf-8")
    illustration_ids.write_text("a1\na2\na3\n", encoding="utf-8")
    return mmsvg_sft.MMSVGPreparationConfig(
        icon=mmsvg_sft.SourceConfig((str(icon_path),), "test/icon", "icon-revision"),
        illustration=mmsvg_sft.SourceConfig(
            (str(illustration_path),),
            "test/illustration",
            "illustration-revision",
        ),
        output_dir=root / "output",
        selected_ids=mmsvg_sft.SelectedIdConfig(icon_ids, illustration_ids),
        seed=42,
        batch_size=2,
        split=mmsvg_sft.SplitConfig(1, 1, 1),
        filters=mmsvg_sft.FilterConfig(
            min_description_words=2,
            max_description_words=20,
            require_detail=True,
            min_svg_char_length=1,
            max_svg_char_length=1000,
            max_svg_token_length=None,
            require_renderable=False,
        ),
    )


def _row(record_id: str, *, variant: str = "base") -> dict[str, str]:
    return {
        "id": record_id,
        "description": f"draw item {record_id}",
        "detail": f"detailed item {record_id}",
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            f'<rect id="{record_id}-{variant}" width="1" height="1"/>'
            "</svg>"
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
