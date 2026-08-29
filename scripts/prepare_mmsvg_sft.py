#!/usr/bin/env python3
"""Prepare the balanced MMSVG SFT pool from Elice-local data files."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from svg_agentic_slm.data.mmsvg_sft import (
    SourceConfig,
    load_preparation_config,
    prepare_mmsvg_sft_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_mmsvg_sft.yaml")
    parser.add_argument("--icon", action="append", help="Override Icon path/glob; repeatable.")
    parser.add_argument(
        "--illustration", action="append", help="Override Illustration path/glob; repeatable."
    )
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument(
        "--rag-results",
        help=(
            "Optional metadata-only JSONL mapping record IDs to up to three unique retrieved IDs; "
            "SVG/context content is never included in SFT records."
        ),
    )
    args = parser.parse_args()

    config = load_preparation_config(args.config)
    if args.icon:
        config = replace(
            config,
            icon=SourceConfig(tuple(args.icon), config.icon.dataset_id, config.icon.dataset_revision),
        )
    if args.illustration:
        config = replace(
            config,
            illustration=SourceConfig(
                tuple(args.illustration),
                config.illustration.dataset_id,
                config.illustration.dataset_revision,
            ),
        )
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))
    if args.rag_results:
        config = replace(config, rag_results_path=Path(args.rag_results))
    manifest = prepare_mmsvg_sft_dataset(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
