#!/usr/bin/env python3
"""Precompute production SigLIP2 scores for current validated teacher records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from svg_agentic_slm.train.critic_similarity_data import precompute_critic_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-run",
        type=Path,
        default=Path(".work/omnisvg_critic_distill/teacher_runs/luna_critic_v3_nonblank"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".work/omnisvg_critic_distill/similarity/siglip2"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    result = precompute_critic_similarity(
        teacher_run=args.teacher_run,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        local_files_only=not args.allow_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
