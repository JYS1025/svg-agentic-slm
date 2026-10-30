"""Hydrate the fixed critic rollout IDs into immutable OmniSVG text inputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree

DOMAINS = ("icon", "illustration")
DATASETS = {
    "icon": ("OmniSVG/MMSVG-Icon", "8b1980d64000138d9fd14c3bfbd592edcc4b0be9"),
    "illustration": ("OmniSVG/MMSVG-Illustration", "6d81c98ae9bc1f4e1fca80cea496a73cb7f150c1"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_stem(domain: str, record_id: str) -> str:
    return hashlib.sha256(f"{domain}:{record_id}".encode()).hexdigest()[:24]


def selected_detail_ids(domain: str, ids: list[str], seed: int) -> set[str]:
    ranked = sorted(
        ids,
        key=lambda record_id: (
            sha256_bytes(f"{seed}:{domain}:{record_id}".encode()),
            record_id,
        ),
    )
    return set(ranked[:3_000])


def caption_sha256(value: str) -> str:
    return sha256_bytes(" ".join(value.casefold().split()).encode("utf-8"))


def normalize_detail(value: str) -> tuple[str, bool]:
    """Join MMSVG details stored as a stringified Python list of sentences."""
    if not value.startswith("["):
        return value, False
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value, False
    if not isinstance(parsed, list) or not parsed or not all(isinstance(x, str) for x in parsed):
        return value, False
    return " ".join(x.strip() for x in parsed if x.strip()), True


def row_group_for_index(parquet: pq.ParquetFile, row_index: int) -> tuple[int, int]:
    offset = 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        if offset <= row_index < offset + count:
            return group, row_index - offset
        offset += count
    raise ValueError(f"Parquet row index {row_index} is out of range")


def prepare(ids_dir: Path, output_dir: Path, seed: int) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    ids_dir = ids_dir.resolve()
    audit_path = ids_dir / "selection_audit.parquet"
    audit = pq.read_table(
        audit_path,
        columns=[
            "domain",
            "record_id",
            "item_id",
            "shard",
            "row_index_in_shard",
            "canonical_svg_sha256",
            "caption_sha256",
            "detail_sha256",
        ],
    ).to_pylist()
    audit_by_key = {(row["domain"], row["record_id"]): row for row in audit}
    if len(audit_by_key) != 10_000 or len(audit) != 10_000:
        raise ValueError("Selection audit must contain 10,000 unique domain and record IDs")

    requested: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {"selection_audit.parquet": sha256_file(audit_path)}
    for domain in DOMAINS:
        name = f"mmsvg_{domain}_critic_rollout_5000_ids.txt"
        ids = (ids_dir / name).read_text(encoding="utf-8").splitlines()
        source_hashes[name] = sha256_file(ids_dir / name)
        if len(ids) != 5_000 or len(set(ids)) != 5_000:
            raise ValueError(f"{name} must contain 5,000 unique IDs")
        detail_ids = selected_detail_ids(domain, ids, seed)
        for record_id in ids:
            key = (domain, record_id)
            if key not in audit_by_key:
                raise ValueError(f"Missing selection audit row: {domain}:{record_id}")
            item = dict(audit_by_key[key])
            if item["item_id"] != f"{domain}:{record_id}":
                raise ValueError(f"Audit item ID mismatch: {key}")
            item["selected_text_field"] = "detail" if record_id in detail_ids else "description"
            requested.append(item)
    if len(requested) != len(audit_by_key):
        raise ValueError("ID lists and selection audit differ")

    by_shard: dict[Path, list[dict[str, object]]] = defaultdict(list)
    for item in requested:
        shard = Path(str(item["shard"])).resolve()
        if not shard.is_file() or "data/raw/mmsvg_" not in str(shard):
            raise FileNotFoundError(f"Invalid source shard: {shard}")
        by_shard[shard].append(item)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        hydrated: dict[tuple[str, str], dict[str, object]] = {}
        for shard, items in sorted(by_shard.items()):
            parquet = pq.ParquetFile(shard)
            by_group: dict[int, list[tuple[int, dict[str, object]]]] = defaultdict(list)
            for item in items:
                group, local_index = row_group_for_index(parquet, int(item["row_index_in_shard"]))
                by_group[group].append((local_index, item))
            for group, indexed in sorted(by_group.items()):
                table = parquet.read_row_group(
                    group, columns=["id", "description", "detail", "svg"]
                )
                for local_index, item in indexed:
                    raw = table.slice(local_index, 1).to_pylist()[0]
                    domain, record_id = str(item["domain"]), str(item["record_id"])
                    if str(raw["id"]) != record_id:
                        raise ValueError(f"Source pointer ID mismatch: {domain}:{record_id}")
                    description = str(raw["description"] or "").strip()
                    detail_raw = str(raw["detail"] or "").strip()
                    detail, detail_was_serialized_list = normalize_detail(detail_raw)
                    svg = str(raw["svg"] or "")
                    if not description or not detail or not svg.strip():
                        raise ValueError(f"Missing text or SVG: {domain}:{record_id}")
                    if caption_sha256(description) != item["caption_sha256"]:
                        raise ValueError(f"Description hash mismatch: {domain}:{record_id}")
                    if caption_sha256(detail_raw) != item["detail_sha256"]:
                        raise ValueError(f"Detail hash mismatch: {domain}:{record_id}")
                    parsed_svg = etree.fromstring(
                        svg.encode("utf-8"),
                        etree.XMLParser(
                            resolve_entities=False, no_network=True, load_dtd=False, recover=False
                        ),
                    )
                    canonical = etree.tostring(parsed_svg, method="c14n", with_comments=False)
                    if sha256_bytes(canonical) != item["canonical_svg_sha256"]:
                        raise ValueError(f"Canonical SVG hash mismatch: {domain}:{record_id}")
                    field = str(item["selected_text_field"])
                    stem = stable_stem(domain, record_id)
                    svg_rel = Path("reference") / stem[:2] / f"{stem}.svg"
                    svg_path = stage / svg_rel
                    svg_path.parent.mkdir(parents=True, exist_ok=True)
                    svg_bytes = svg.encode("utf-8")
                    svg_path.write_bytes(svg_bytes)
                    dataset_id, revision = DATASETS[domain]
                    hydrated[(domain, record_id)] = {
                        "schema_version": 1,
                        "sample_id": f"{domain}:{record_id}",
                        "domain": domain,
                        "record_id": record_id,
                        "dataset_id": dataset_id,
                        "dataset_revision": revision,
                        "description": description,
                        "detail": detail,
                        "detail_raw": detail_raw,
                        "detail_was_serialized_list": detail_was_serialized_list,
                        "selected_text_field": field,
                        "instruction": detail if field == "detail" else description,
                        "reference_svg_path": str(svg_rel),
                        "reference_svg_sha256": sha256_bytes(svg_bytes),
                        "audit_canonical_svg_sha256": item["canonical_svg_sha256"],
                        "audit_caption_sha256": item["caption_sha256"],
                        "audit_detail_sha256": item["detail_sha256"],
                        "source_pointer": {"shard": str(shard), "row": item["row_index_in_shard"]},
                    }
        manifest_path = stage / "records.jsonl"
        counts = {domain: {"detail": 0, "description": 0} for domain in DOMAINS}
        with manifest_path.open("w", encoding="utf-8") as handle:
            for item in requested:
                key = (str(item["domain"]), str(item["record_id"]))
                record = hydrated[key]
                counts[key[0]][str(record["selected_text_field"])] += 1
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if any(counts[d] != {"detail": 3_000, "description": 2_000} for d in DOMAINS):
            raise AssertionError(f"Unexpected instruction distribution: {counts}")
        manifest = {
            "schema_version": 1,
            "sample_count": len(requested),
            "counts": counts,
            "text_policy": "exact 3000 detail and 2000 description per domain by SHA-256 rank",
            "detail_normalization": (
                "join valid stringified lists of sentences with spaces; preserve detail_raw"
            ),
            "text_policy_seed": seed,
            "input_files_sha256": source_hashes,
            "records_sha256": sha256_file(manifest_path),
            "reference_image_policy": (
                "render reference_svg_path with the same renderer settings as candidate"
            ),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-dir", type=Path, default=Path("data/critic_rollout_sample_ids"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".work/omnisvg_critic_distill/inputs")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.ids_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
