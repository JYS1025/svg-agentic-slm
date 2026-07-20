# Generator Cycle 0 Status and Experiment Runbook

## 1. Purpose

This is the operational handoff for Generator Cycle 0. It separates:

1. code and contracts already implemented;
2. decisions agreed during Generator/RAG discussions;
3. external-owner reviews still required;
4. experiments that the local hardware owner must run.

The commands below are instructions for the experiment owner. Since the first
draft, the selected model has been downloaded and used for real generation, and
the SVGenius v2 candidate snapshot has been prepared locally. Full hardware
measurements and benchmark-backed quality evaluation have not been completed.

## 2. Completion Verdict

The Generator-owned Cycle 0 implementation is **code-complete as a review
candidate**. Project-wide Cycle 0 remains open and is not yet operationally
closed.

| Area | Status | Remaining gate |
|---|---|---|
| Branch baseline | Complete | `minjun/generator` was aligned with `origin/minjun/generator` at `f1a88c3` before local changes |
| Local model decision | Compatibility pin selected and real generation verified | Broader quality validation remains |
| CUDA llama.cpp backend | Implemented; native CUDA load/inference verified | Record VRAM and latency measurements |
| Rejected Google GGUF | Fully downloaded and checksum-consistent, but aborts at vocabulary token 237922 before tensor placement | Keep as incident evidence; do not retry as an OOM fix |
| Selected compatibility GGUF | Downloaded, loaded, and produced a valid non-placeholder SVG locally | Record E2 VRAM headroom and E3 latency evidence |
| Generator initial/revision contract | Implemented | Cross-team API review |
| Fake RAG/Critic integration | Implemented | Contract tests after merge/rebase |
| RAG metadata whitelist | Implemented | Whitelist is intentionally empty in Cycle 0 |
| Actual RAG retrieval | Not Generator-owned | RAG owner implementation |
| Actual LLM Critic | Not Generator-owned | Critic owner implementation |
| Revision orchestration | Implemented baseline | Orchestration owner review |
| Artifact schema version 1 | Implemented additively | Structure owner review |
| Strict SVG safety validation | Current validator remains lightweight | SVG Validation owner |
| SVGenius candidate preparation | Dataset-specific adapter v2 implements a revision-pinned known exclusion | Final benchmark/metric and license review remain |
| SVGenius candidate files | E7 smoke and E8 strict snapshot passed; 299 records with one audited exclusion | Candidate inspection complete; this is not a model-accuracy result |
| Dataset-backed batch evaluation | Not implemented | Evaluation owner; current evaluator scores existing artifacts only |
| Local latency/VRAM acceptance | Not measured | User-run hardware spike |

Remaining project-wide Cycle 0 closure gates are:

- local VRAM and latency evidence is recorded;
- RAG, Critic/Orchestration, and Structure owners review the shared fields;
- remaining work is assigned to its actual owner rather than absorbed into the
  Generator module.

### 2.1 Decisions Still Open

These are decisions or reviews, not work for the Generator owner to absorb:

| Owner | Open decision |
|---|---|
| Local hardware owner | Approve measured VRAM, TTFT, throughput, and end-to-end latency; choose an explicit partial-offload fallback only if full offload fails |
| RAG | Approve stable item/source ID construction, score semantics, corpus versioning, and the typed item contract |
| Critic/Orchestration | Approve feedback fields, calibrated acceptance threshold, stop/no-improvement policy, and failure handling |
| Structure/Artifact | Accept or revise schema version 1, file naming, atomic persistence, and intermediate retention |
| SVG Validation | Define strict XML, unsafe element/external reference, structural validity, and render-failure semantics |
| Evaluation | Select the final benchmark, freeze dataset-specific adapters/splits, choose metrics and success thresholds, and implement the batch runner |
| Memory/Research | Define curation/retention, enforce benchmark isolation, and set the evidence threshold for a “self-evolving” claim |

The detailed decision IDs and proposed defaults are maintained in the
[Cycle 0 Open Decision Register](./generator-cross-team-contract.md#7-cycle-0-open-decision-register).

## 3. Agreements to Date

### 3.1 Model and runtime

- Local baseline: `lmstudio-community/gemma-4-12B-it-QAT-GGUF`.
- File: `gemma-4-12B-it-QAT-Q4_0.gguf`.
- Immutable revision:
  `291406f49e16eff811c85ad8884d375f34138663`.
- Upstream model:
  `google/gemma-4-12B-it-qat-q4_0-unquantized`.
- Quantization provider and conversion runtime:
  `LM Studio Community`, `llama.cpp b9518`.
- Runtime: CUDA-enabled `llama-cpp-python==0.3.34`.
- Local placement: `n_gpu_layers=-1`, preferring full GPU offload.
- CPU offload must be configured explicitly; there is no silent fallback.
- Initial profile: `n_ctx=8192`, `n_batch=512`, flash attention and mmap.
- Generation baseline: 2,048 completion tokens and 12,000 characters of
  retrieved context.
- The current path is text-only and does not load the multimodal projector.
- GGUF metadata owns the chat template.
- The prior Google-hosted GGUF revision
  `9d1c295262db03ac01d47caa328b39ecc2fcdf10` is rejected because its native
  vocabulary load aborts. Its completed download and CUDA detection do not
  count as a successful E2.

### 3.2 Generator boundary

Initial generation and revision are separate operations:

```text
generate(request, context_items) -> GeneratorOutput

revise(request, previous_attempt, feedback_event, context_items)
    -> GeneratorOutput
```

- User instruction has higher priority than retrieved context.
- Generator owns prompt construction, context truncation, model invocation,
  SVG extraction/normalization, and attempt trace production.
- Generator does not own retrieval, Critic scoring, global stop policy, memory
  curation, benchmark execution, or artifact-store migration.
- Infrastructure failures raise explicitly.
- A response that cannot yield an SVG becomes a typed failed attempt;
  placeholder SVGs are never substituted.

### 3.3 RAG boundary

RAG returns `RetrievedExample[]`, not one preformatted prompt string.

Required typed fields are:

```text
item_id
source
content
description
score
score_kind
rank
kind
corpus_version
```

- `item_id` identifies the same logical item across retrieval runs.
- `source` and `score_kind` must be non-empty.
- RAG owns retrieval, deduplication, ranking, score interpretation, and corpus
  versioning.
- Generator owns prompt representation and context budget.
- The current orchestrator retrieves once per run and reuses context for
  revisions. Feedback-aware re-retrieval is a later measured variant.
- A rendered bitmap is not a Generator input. A future `render_ref` remains
  internal until explicitly admitted to the shared contract.

Current item kinds are:

```text
reference_svg
positive_experience
negative_lesson
correction_pair
```

Cycle 0 uses a strict metadata projection policy:

- all required information is represented by typed top-level fields;
- the free-form metadata whitelist is initially empty;
- vector-store-only metadata is dropped at the RAG adapter boundary;
- adding a key requires documenting its owner, meaning, and retention policy.

### 3.4 Critic and revision

Critic conceptually consumes:

```text
instruction + SVG attempt -> structured feedback
```

Each persisted event needs:

```text
feedback_id
target_attempt_id
score
issues
suggestions
is_valid
matches_instruction
critic type/version
model/prompt version when applicable
```

- `generate()` does not accept feedback.
- `revise()` requires the previous attempt and feedback targeting it.
- Orchestration correlates Critic output to the reviewed attempt.
- Current baseline is acceptance score `8.0` and at most two revisions.
- This is tunable and not a calibrated research threshold.

### 3.5 Identifier ownership

Identifiers are producer-owned:

```text
Factory/runtime: run_id
Generator:       attempt_id, model_call_id
Orchestration:   feedback_id
RAG:             item_id and source ID
```

### 3.6 Artifacts

Schema version 1 retains the legacy final artifact and additively records:

- attempt-level SVG files and raw model output;
- exact user/system prompt references and generation parameters;
- model, backend, prompt, and Critic versions;
- feedback-to-attempt correlation and revision lineage;
- accepted/rejected/failed outcome and stop reason;
- RAG provenance and whitelisted metadata.

Absence of `schema_version` remains legacy version 0.

### 3.7 Experience memory

The research direction is retrieval-conditioned experience reuse:

```text
static references
+ accepted examples
+ failed attempt -> feedback -> accepted correction
-> future retrieval
```

- Generator only emits the trace required to construct experience records.
- Memory curator owns eligibility, confidence, deduplication, and retention.
- RAG owns indexing and retrieval.
- Held-out benchmark runs use `memory_eligible=false`.
- A rejected SVG alone is not a negative lesson.
- “Self-evolving” requires held-out round-by-round improvement and ablations.

### 3.8 Benchmark

- SVGenius is a **candidate**, not the final project benchmark.
- All SVGenius mapping lives in `svg_agentic_slm.benchmarks.svgenius`.
- Each future dataset receives its own adapter.
- `source_split` preserves the upstream `easy`/`medium`/`hard` split, while
  `data_partition` separately records the still-unassigned evaluation role.
- The upstream `train` split is a union of those difficulty rows and is
  deliberately excluded to avoid duplicating the same 300 source rows.
- The Hugging Face table supplies SVG code/difficulty but no text prompt. The
  adapter joins it to the official GitHub caption tasks by asset filename.
- At the pinned revisions, medium source row
  `page_38_ant_design_48353_icon_95` has no caption in any pinned text-to-SVG
  task file. Adapter v2 accounts for exactly this revision-bound exclusion and
  continues to fail on every unexpected mismatch.
- Candidate records are always `data_partition=candidate_unassigned` and
  `memory_eligible=false`; the Evaluation owner assigns dev/test semantics only
  after final dataset selection.
- The license inconsistency must be resolved before final adoption.
- Dataset-backed evaluation remains unimplemented; prepared data is not a
  completed benchmark run.

## 4. Experiment Runbook

Run each section manually and save its terminal output. Do not enable actual
RAG or the LLM Critic in Cycle 0 because those remain placeholders.

### E0. Install and verify CUDA llama.cpp

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CUDAHOSTCXX=/usr/bin/g++-11 \
CMAKE_ARGS="-DGGML_CUDA=on \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11" \
FORCE_CMAKE=1 \
python -m pip install -e ".[local-gpu,dev]"
```

```bash
python - <<'PY'
import inspect
from pathlib import Path

import llama_cpp
from llama_cpp import llama_cpp as native
from svg_agentic_slm.models.llama_cpp_backend import LlamaCppModelBackend

print("llama_cpp_python:", llama_cpp.__version__)
print("gpu_offload_supported:", native.llama_supports_gpu_offload())
print("mmap_supported:", native.llama_supports_mmap())

actual_backend = Path(inspect.getfile(LlamaCppModelBackend)).resolve()
expected_backend = (
    Path.cwd() / "src/svg_agentic_slm/models/llama_cpp_backend.py"
).resolve()
print("backend_source:", actual_backend)
print("backend_signature:", inspect.signature(LlamaCppModelBackend.__init__))
if actual_backend != expected_backend:
    raise SystemExit(
        "The active Python environment is importing a stale non-editable install. "
        "Run: python -m pip install -e '.[local-gpu,dev]'"
    )
PY
```

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Pass: version `0.3.34`, GPU offload `True`, RTX 4080 detection, and
`backend_source` resolving to this checkout's `src/svg_agentic_slm/...` path.

### E1. Download the pinned model without inference

The downloader resumes interrupted partial downloads. The selected community
repository is public; authentication is still useful for rate limits:

```bash
hf auth whoami
```

If authentication is missing:

```bash
hf auth login
```

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

path = Path(hf_hub_download(
    repo_id="lmstudio-community/gemma-4-12B-it-QAT-GGUF",
    filename="gemma-4-12B-it-QAT-Q4_0.gguf",
    revision="291406f49e16eff811c85ad8884d375f34138663",
))
print("path:", path)
print("bytes:", path.stat().st_size)
print("GiB:", round(path.stat().st_size / 1024**3, 3))
PY
```

Pass: exact revision, approximately 6.98 GB, and no `.incomplete` path.

### E2. Load and inspect VRAM placement

Terminal A:

```bash
python - <<'PY'
from svg_agentic_slm.models.llama_cpp_backend import LlamaCppModelBackend

backend = LlamaCppModelBackend(
    model_id="lmstudio-community/gemma-4-12B-it-QAT-GGUF",
    filename="gemma-4-12B-it-QAT-Q4_0.gguf",
    model_revision="291406f49e16eff811c85ad8884d375f34138663",
    upstream_model_id="google/gemma-4-12B-it-qat-q4_0-unquantized",
    quantization="Q4_0",
    quantization_provider="LM Studio Community",
    conversion_runtime="llama.cpp b9518",
    n_ctx=8192,
    n_gpu_layers=-1,
    n_batch=512,
    flash_attn=True,
    use_mmap=True,
    verbose=True,
)
backend.load_model()
print("loaded:", backend.is_loaded())
input("Inspect nvidia-smi, then press Enter to unload...")
backend.unload_model()
PY
```

Terminal B:

```bash
nvidia-smi
```

Pass: vocabulary load completes, full offload succeeds, `loaded: True` prints,
and at least 1 GiB VRAM headroom remains as the provisional baseline. A native
assertion is a compatibility failure, not an OOM; do not respond by reducing
`n_gpu_layers`.

### E3. Real Generator-only smoke

```bash
svg-agentic-slm generate \
  "A blue circle centered inside a 256 by 256 canvas" \
  --config configs/generation.yaml \
  --no-render
```

Pass: real completion, successful SVG extraction, schema version 1, and model
revision/raw output/prompt/token/latency/attempt provenance.

### E4. Render and artifact-evaluation smoke

```bash
svg-agentic-slm generate \
  "A red triangle centered on a white square canvas" \
  --config configs/generation.yaml
```

```bash
svg-agentic-slm eval \
  --config configs/eval.yaml \
  --artifact-path outputs/generations
```

Pass: existing render, validity/render/latency report. This evaluator scores
artifacts; it does not execute a dataset.

### E5. Repeatability

Run twice:

```bash
svg-agentic-slm generate \
  "A black five-point star on a transparent canvas" \
  --config configs/generation.yaml \
  --no-render \
  --seed 42 \
  --temperature 0
```

Compare:

```bash
python - <<'PY'
import hashlib
from pathlib import Path

paths = sorted(
    Path("outputs/generations").glob("*.svg"),
    key=lambda path: path.stat().st_mtime_ns,
    reverse=True,
)[:2]
for path in paths:
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
PY
```

Pass: hashes match, or non-determinism is documented before benchmarking.

### E6. Contract-only tests

These do not download a model or dataset:

```bash
python -m pytest -q \
  tests/test_llama_cpp_backend.py \
  tests/test_generator.py \
  tests/test_orchestrator.py \
  tests/test_rag_contract.py \
  tests/test_generate_pipeline.py \
  tests/test_artifacts.py
```

Pass: revision correlation, fail-fast config, empty RAG metadata whitelist, and
artifact compatibility tests all pass.

### E7. Small SVGenius candidate snapshot

This downloads pinned candidate inputs and prepares three per difficulty:

```bash
python -m svg_agentic_slm.benchmarks.svgenius \
  --output-dir data/processed/benchmarks/svgenius-smoke \
  --limit-per-difficulty 3
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
from svg_agentic_slm.data.jsonl import read_jsonl

root = Path("data/processed/benchmarks/svgenius-smoke")
records = read_jsonl(root / "text_to_svg.jsonl")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
print("records:", len(records))
print("by_split:", manifest["records_by_split"])
print("join_stats:", manifest["join_stats"])
print("adapter:", manifest["adapter"])
print("manifest_schema_version:", manifest["manifest_schema_version"])
print("sha256:", manifest["output_sha256"])
print("first_instruction:", records[0]["instruction"])
print("memory_eligible:", records[0]["metadata"]["memory_eligible"])
PY
```

Pass: nine strict joins, adapter `svgenius-text-to-svg-v2`, manifest schema 2,
well-formed reference SVGs, pinned revisions, and
`candidate_unassigned`/`memory_eligible=false`. No exclusion is applied to the
three selected rows per difficulty.

### E8. Complete SVGenius candidate snapshot

Run only after E7:

```bash
python -m svg_agentic_slm.benchmarks.svgenius \
  --output-dir data/processed/benchmarks/svgenius
```

Inspect the exclusion audit:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("data/processed/benchmarks/svgenius/manifest.json").read_text(
        encoding="utf-8"
    )
)
print("adapter:", manifest["adapter"])
print("schema:", manifest["manifest_schema_version"])
print("strict:", manifest["strict"])
print("records:", manifest["num_records"])
print("by_split:", manifest["records_by_split"])
print("join_stats:", manifest["join_stats"])
print("configured_known_exclusions:", manifest["configured_known_exclusions"])
print("applied_known_exclusions:", manifest["applied_known_exclusions"])
PY
```

Pass: adapter `svgenius-text-to-svg-v2`, manifest schema 2, `strict=true`,
299 records (`easy=100`, `medium=99`, `hard=100`), exactly one applied
revision-pinned exclusion for `page_38_ant_design_48353_icon_95`, and zero
unexpected missing-caption rows. Adapter v2 exposes no broad
`--allow-unmatched` CLI bypass.

### E9. Candidate adapter unit tests

No network access:

```bash
python -m pytest -q tests/test_svgenius_adapter.py
```

## 5. Results to Record

| Field | Value |
|---|---|
| GPU and power profile | |
| CUDA/runtime version | |
| model revision/file | |
| `n_ctx`, `n_batch`, `n_gpu_layers` | |
| idle/loaded/peak VRAM | |
| model-load seconds | |
| first/warm generation latency | |
| prompt/completion tokens | |
| tokens per second | |
| SVG extraction/validation/render success | |
| deterministic hash match | |
| artifact schema audit | |

Approve numeric latency only after this evidence exists. The provisional hard
requirement is full offload with at least 1 GiB headroom; otherwise compare an
explicit partial-offload profile rather than hiding fallback.
