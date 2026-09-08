# OmniSVG codec phase 1

Phase 1 keeps three representation boundaries explicit:

- `OmniSVGDiscreteCodec` remains the existing public local API.
- `omnisvg-inspired-discrete-svg` wraps that API as a bidirectional Gemma backend.
- `openvglab-omnisvg-train-4b-812489...` loads the official released training encoder as an encode-only analysis backend.

Neither backend claims compatibility with released OmniSVG checkpoints. The local dialect uses
named special tokens and an explicit SOP/F/EOP grammar. The released training encoder uses
absolute Qwen vocabulary IDs, but those IDs differ from the released inference decoder by one;
the released 8B encoder also has color-base and BOS/EOS conflicts. There is no public canonical
SVG-to-ID fixture or matching standalone codec checkpoint, so phase 1 does not provide an
official round trip.

## External checkout contract

The analysis backend imports, but does not install or copy, an external checkout of
[`OpenVGLab/OmniSVG-train`](https://github.com/OpenVGLab/OmniSVG-train) at exactly:

```text
812489fd9d191e39fe94bc0c4027e5d0121e0fc6
```

It fails before encoding unless all of these conditions hold:

- The supplied directory is the Git top level at the exact full commit.
- The tracked worktree is clean and `configs/tokenization.yaml` is tracked.
- That config has SHA-256 `bd56bc2bb9b39f614a9d553d2319332a18a075c4239bb71ec437822b20d97dd4`.
- `utils.config`, `utils.dataset`, and `deepsvg.svglib.svg` import from that checkout without module-name collisions.
- SVG parsing, tensorization, integer-ID validation, and exact `BOS + body + EOS` framing succeed.

Parser and tokenizer failures are errors. They never become empty token sequences. Masking,
dataset balancing, truncation, model loading, GPU setup, and training are outside this adapter.

## MMSVG 20K audit

The audit streams `train.jsonl`, `validation.jsonl`, and `test.jsonl` from the configured prepared
root. It records input hashes, failures, body/BOS-EOS/full-chat lengths, nearest-rank p50/p95/p99,
max samples, and counts above each threshold. Its JSON manifest is replaced atomically. The
optional SQLite cache stores successful sequences as zlib-compressed little-endian uint32 arrays
and indexes them by global sample index and `split + sample_id`.

Full-chat length is explicitly defined as the configured assistant-prefix length plus the framed
SVG IDs. The example uses a tokenizer directory that must already be local; Hugging Face loading
uses `local_files_only=True` and `trust_remote_code=False`. Text-only chat counting does not include
multimodal image-token expansion, and the manifest records that limitation. For prepared rows
that already contain a trusted prefix-token count, use `chat.mode: row_field` instead.

```bash
export OMNISVG_TRAIN_ROOT=/absolute/path/to/OmniSVG-train
export QWEN_TOKENIZER_ROOT=/absolute/path/to/local/Qwen2.5-VL-tokenizer
python -m svg_agentic_slm.svg.token_audit \
  --config configs/audit_omnisvg_train_4b.yaml
```

Any record failure is included in the atomic manifest and makes the CLI exit with status `2`.
Use `--no-token-cache` when only aggregate statistics and failure provenance are needed.
