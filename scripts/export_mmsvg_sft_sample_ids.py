#!/usr/bin/env python3
"""Export reproducible MMSVG SFT sample IDs without copying dataset payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATASETS = {
    "icon": {
        "repo_id": "OmniSVG/MMSVG-Icon",
        "revision": "8b1980d64000138d9fd14c3bfbd592edcc4b0be9",
        "source": "raw/mmsvg_icon_v2/data",
    },
    "illustration": {
        "repo_id": "OmniSVG/MMSVG-Illustration",
        "revision": "6d81c98ae9bc1f4e1fca80cea496a73cb7f150c1",
        "source": "raw/mmsvg_illustration_v2/data",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(source_dir: Path) -> tuple[list[str], int]:
    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found under {source_dir}")

    ids: list[str] = []
    for shard in shards:
        ids.extend(pq.read_table(shard, columns=["id"])["id"].to_pylist())

    if any(not isinstance(item, str) or not item or "\n" in item for item in ids):
        raise ValueError(f"Invalid ID found under {source_dir}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found under {source_dir}")
    return ids, len(shards)


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "MMSVG SFT sample IDs only",
        "sampling": {
            "method": "numpy.default_rng(seed).permutation(total_rows)[:sample_size]",
            "seed": args.seed,
            "sample_size_per_dataset": args.sample_size,
            "source_order": (
                "lexicographically sorted Parquet shard paths, then row order "
                "within each shard"
            ),
        },
        "datasets": {},
    }

    for kind, config in DATASETS.items():
        source_dir = args.data_dir / config["source"]
        ids, shard_count = read_ids(source_dir)
        if len(ids) < args.sample_size:
            raise ValueError(
                f"{kind} has {len(ids)} rows, fewer than {args.sample_size} requested"
            )

        indices = np.random.default_rng(args.seed).permutation(len(ids))[
            : args.sample_size
        ]
        selected = [ids[index] for index in indices]
        output_name = f"mmsvg_{kind}_sft_{args.sample_size}_ids.txt"
        output_path = args.output_dir / output_name
        output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

        manifest["datasets"][kind] = {
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "local_source": str(source_dir.resolve()),
            "source_shards": shard_count,
            "total_rows": len(ids),
            "sample_rows": len(selected),
            "sample_unique_ids": len(set(selected)),
            "output_file": output_name,
            "output_sha256": sha256(output_path),
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
