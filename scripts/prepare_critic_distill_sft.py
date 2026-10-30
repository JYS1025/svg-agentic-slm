#!/usr/bin/env python3
"""Freeze validated OmniSVG critic teacher results into SFT JSONL splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from svg_agentic_slm.train.critic_distill_data import prepare_critic_distill_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path(".work/omnisvg_critic_distill/runs/full_10k"),
    )
    parser.add_argument(
        "--teacher-run",
        type=Path,
        default=Path(".work/omnisvg_critic_distill/teacher_runs/luna_critic_v3_nonblank"),
    )
    parser.add_argument(
        "--similarity-scores",
        type=Path,
        default=Path(".work/omnisvg_critic_distill/similarity/siglip2/scores.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/omnisvg_critic_distill"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_critic_distill_dataset(
        source_run=args.source_run,
        teacher_run=args.teacher_run,
        similarity_scores_path=args.similarity_scores,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
