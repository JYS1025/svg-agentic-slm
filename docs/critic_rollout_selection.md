# MMSVG critic rollout 10K selection

The export is a strict holdout: 5,000 Icon rows and 5,000 Illustration rows.
It excludes the Generator SFT 20K by both raw ID and canonical SVG SHA-256,
excludes benchmark IDs, caption/detail hashes, and SVG hashes, and globally
deduplicates the selected SVG hashes. It also excludes every unique item_id
and canonical SVG hash appearing in the Generator RAG
final_references.parquet (the completed artifact has 60,000 rows and 52,999
unique references).

The selector does not create embeddings. It batch-fetches the existing Qwen
description and detail vectors from their Chroma collections by stable
`domain:raw_id`, explicitly rejoins returned IDs, L2-normalizes the vectors,
and combines them as `0.4 * description + 0.6 * detail`. A seed-42 random
projection and 16-bit hyperplane buckets provide semantic coverage. Per domain,
1,000 rows are first assigned to explicit structural edge strata (long/complex,
many paths, colors, arcs/curves, rare commands, advanced styling, and minimal
SVGs); the other 4,000 maximize semantic-bucket Hamming coverage.

Chroma export uses explicit ID batches of 8,192. Returned rows are rejoined by
ID (their order is not assumed), and every vector must be finite, nonzero, and
exactly 1,024-dimensional.

Every pool row must pass `SVGValidator` and strict XML C14N. CairoSVG rendering
is deliberately limited to ranked finalists and deterministic backfills. This
keeps the full 1.159M-row scan feasible while guaranteeing that all published
IDs are renderable. Output publication is atomic and refuses to overwrite an
existing directory.

```bash
.venv/bin/python scripts/export_mmsvg_critic_rollout_ids.py \
  --data-root data/raw \
  --generator-id-dir data/sft_sample_ids \
  --generator-sft-dir data/processed/mmsvg_sft_20k \
  --final-references-path data/processed/mmsvg_sft_20k_rag/final_references.parquet \
  --description-chroma-dir data/chroma_db \
  --detail-chroma-dir data/chroma_db_detail \
  --output-dir data/critic_rollout_sample_ids
```

Artifacts are two raw-ID text files, `selection_audit.parquet`, and
`manifest.json`. The manifest pins source revisions and input hashes, records
all exclusion/rejection counts, the dual-Chroma contracts, projection hash,
quota fulfillment, output SHA-256 values, and render policy.
