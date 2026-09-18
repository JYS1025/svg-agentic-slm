#!/usr/bin/env python3
"""Build local JSONL benchmark exclusions from downloaded Parquet files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MMSVGBENCH_PATH = (
    ROOT / "data/raw/mmsvgbench_text2svg/data/text2svg-00000-of-00001.parquet"
)
VECTORGYM_PATH = ROOT / "data/raw/vectorgym_test/data/test-00000-of-00001.parquet"
OUTPUT_DIR = ROOT / "data/processed/benchmark_exclusions"
MMSVGBENCH_REVISION = "bc1b0f0aebcf07009f861e0893f504965c1137c9"
VECTORGYM_REVISION = "7d4aafd16d2f42e66ca5acfc6d8f66f36a034b75"


def main() -> None:
    for path in (MMSVGBENCH_PATH, VECTORGYM_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage = Path(
        tempfile.mkdtemp(
            prefix=".benchmark-exclusions-",
            dir=OUTPUT_DIR.parent,
        )
    )
    try:
        mmsvgbench_rows = _mmsvgbench_rows()
        vectorgym_rows, malformed = _vectorgym_rows()
        outputs = {
            "mmsvgbench": _write_jsonl(stage / "mmsvgbench.jsonl", mmsvgbench_rows),
            "vectorgym_vg_text": _write_jsonl(
                stage / "vectorgym_vg_text.jsonl",
                vectorgym_rows,
            ),
        }
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "hash_contract": {
                "caption": "SHA-256 after casefold and whitespace collapse in SFT loader",
                "svg": "SHA-256 of strict lxml XML C14N without comments",
            },
            "sources": {
                "mmsvgbench": _source_manifest(
                    MMSVGBENCH_PATH,
                    dataset_id="OmniSVG/MMSVGBench",
                    revision=MMSVGBENCH_REVISION,
                    rows=300,
                ),
                "vectorgym": _source_manifest(
                    VECTORGYM_PATH,
                    dataset_id="VectorSpaceLab/VectorGym",
                    revision=VECTORGYM_REVISION,
                    rows=293,
                ),
            },
            "outputs": outputs,
            "vectorgym_malformed_original_svg_ids": malformed,
        }
        manifest_path = stage / "manifest.json"
        _write_json(manifest_path, manifest)
        if OUTPUT_DIR.exists():
            existing_manifest = OUTPUT_DIR / "manifest.json"
            if existing_manifest.is_file() and json.loads(
                existing_manifest.read_text(encoding="utf-8")
            ) == manifest:
                shutil.rmtree(stage)
                print(json.dumps(manifest, indent=2, sort_keys=True))
                return
            raise FileExistsError(
                f"Refusing to replace existing benchmark exclusions: {OUTPUT_DIR}"
            )
        os.replace(stage, OUTPUT_DIR)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _mmsvgbench_rows() -> list[dict[str, Any]]:
    table = pq.read_table(MMSVGBENCH_PATH, columns=["id", "text"])
    rows: list[dict[str, Any]] = []
    for row in table.to_pylist():
        item_id = _required_text(row.get("id"), "MMSVGBench id")
        caption = _required_text(row.get("text"), "MMSVGBench text")
        rows.append({"id": item_id, "caption": caption})
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 MMSVGBench rows, got {len(rows)}.")
    return rows


def _vectorgym_rows() -> tuple[list[dict[str, Any]], list[str]]:
    table = pq.read_table(
        VECTORGYM_PATH,
        columns=[
            "svg_id",
            "text_caption",
            "editing_prompt",
            "original_svg",
            "editing_target_svg",
        ],
    )
    rows: list[dict[str, Any]] = []
    malformed: list[str] = []
    for row in table.to_pylist():
        item_id = _required_text(row.get("svg_id"), "VectorGym svg_id")
        output: dict[str, Any] = {"id": item_id}
        caption = _optional_text(row.get("text_caption"))
        instruction = _optional_text(row.get("editing_prompt"))
        if caption is not None:
            output["caption"] = caption
        if instruction is not None:
            output["instruction"] = instruction
        hashes: list[str] = []
        original_svg = _optional_text(row.get("original_svg"))
        if original_svg is not None:
            try:
                hashes.append(_canonical_svg_sha256(original_svg))
            except Exception:
                malformed.append(item_id)
        target_svg = _optional_text(row.get("editing_target_svg"))
        if target_svg is not None:
            hashes.append(_canonical_svg_sha256(target_svg))
        if hashes:
            output["svg_hashes"] = list(dict.fromkeys(hashes))
        rows.append(output)
    if len(rows) != 293:
        raise RuntimeError(f"Expected 293 VectorGym rows, got {len(rows)}.")
    return rows, sorted(malformed)


def _canonical_svg_sha256(svg: str) -> str:
    from lxml import etree  # type: ignore[import-untyped]

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    canonical = etree.tostring(root, method="c14n", with_comments=False)
    return hashlib.sha256(canonical).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "file": path.name,
        "rows": len(rows),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _source_manifest(
    path: Path,
    *,
    dataset_id: str,
    revision: str,
    rows: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "revision": revision,
        "path": str(path.resolve()),
        "rows": rows,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    main()
