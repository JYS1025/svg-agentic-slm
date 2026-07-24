# Generator Cross-Team Contract and Research Assumptions

## Document Control

| Field | Value |
|---|---|
| Status | Implemented candidate for cross-team review |
| Applies from | Generator Cycle 0 |
| Baseline branch | `minjun/generator` |
| Baseline commit | `f1a88c3` |
| Primary owner | Generator workstream |
| Reviewers | Orchestration/Structure, RAG, Critic, SVG Validation, Evaluation owners |
| Last updated | 2026-07-20 |

This document records the assumptions established by the Generator workstream
that affect another workstream's implementation or the project's central
research direction. It is the cross-team contract, not a Generator
implementation diary.

The detailed execution plan remains in
[Generator Implementation Plan](./generator-implementation-plan.md). The
Cycle summary is in
[Generator Development Cycle Roadmap](./generator-cycle-roadmap.md), and the
Cycle 0 teammate-facing implementation handoff is in
[Generator Cycle 0 Cross-Team Handoff](./generator-cycle0-team-handoff.md).
The current CLI and artifact ownership boundary is described in
[Generate Command Workflow](./generate-command-workflow.md).

## 1. What Must Be Recorded Here

A Generator decision MUST be recorded in this document when at least one of the
following conditions applies:

1. It changes an input, output, schema, identifier, or failure contract consumed
   by another module.
2. It changes which workstream owns a decision or runtime behavior.
3. It changes artifact reproducibility, benchmark isolation, or data
   provenance.
4. It changes the central research hypothesis, ablation design, or the claims
   that the project may make.
5. It changes the minimum runtime or deployment assumptions shared by local and
   cloud execution.

Generator-internal refactoring, private helper APIs, and exploratory
hyperparameter results do not belong here unless they later satisfy one of the
conditions above.

## 2. Accepted Direction

The following direction is established for Cycle 0.

| ID | Decision | Status | Cross-team impact |
|---|---|---|---|
| D-001 | The local Generator default is the pinned `lmstudio-community/gemma-4-12B-it-QAT-GGUF` Q4_0 compatibility quant, derived from Google's QAT upstream and configured for CUDA-enabled llama.cpp full GPU offload. | Accepted | Model factory, runtime environment, config, and artifact provenance must preserve the distribution revision, upstream model ID, quantization provider, conversion runtime, and llama.cpp profile. |
| D-002 | Benchmark preparation uses one adapter per dataset; SVGenius is an evaluated candidate, not the final external benchmark. | Accepted direction | Evaluation may inspect the pinned SVGenius candidate without making it a project default, and must select and freeze the final dataset, subset, license decision, and metric protocol separately. |
| D-003 | Initial generation and feedback-based revision are separate Generator operations. | Accepted | Orchestration must call `generate()` for an initial draft and `revise()` for a critic-driven correction. |
| D-004 | Generator consumes RAG and Critic results through neutral typed contracts. | Accepted direction | Generator must not import a vector database client or depend on a concrete Critic implementation. |
| D-005 | Generator preserves attempt-level provenance but does not own the complete revision policy or artifact store. | Accepted | Orchestration decides continuation and acceptance; Structure persists the trace. |
| D-006 | Critic feedback and corrections may form an experience memory retrieved alongside static SVG examples. | Core research direction | RAG, Critic, Orchestration, Artifact, and Evaluation designs must retain the information needed to build and audit this memory. |
| D-007 | Existing final SVG and sidecar consumers remain compatible through additive schema evolution. Version 1 sidecars reference an immutable run bundle; version 0 retains legacy path handling. | Accepted | Structure publishes the complete bundle before the sidecar and keeps the explicit SVG output as a non-canonical export alias. |
| D-008 | RAG metadata crossing into Generator follows an explicit whitelist; the Cycle 0 free-form whitelist is empty. | Accepted | RAG projects vector-store metadata at its adapter boundary. New shared keys require documented semantics and retention ownership. |
| D-009 | The previously pinned Google-hosted GGUF revision is rejected for Cycle 0 because it deterministically aborts in native vocabulary loading. | Accepted incident decision | Do not treat retries, CUDA placement changes, or a `llama-cpp-python` reinstall as a fix. A repaired upstream file requires a new immutable revision and a fresh compatibility check before readoption. |

“Accepted direction” means the product/research choice is fixed, while a
deployment-specific identifier or schema may still require a Cycle 0 decision.

## 3. Normative Component Boundary

The conceptual data flow is:

```text
Prompt
  └── RAG.retrieve()
        └── retrieved context items
              └── Generator.generate()
                    └── SVG attempt
                          └── Critic.critique()
                                └── feedback event
                                      └── Generator.revise()
                                            └── revised SVG attempt
```

The target Generator interface is separated by operation:

```text
generate(request, context_items=()) -> GenerationOutput

revise(
    request,
    previous_svg,
    feedback,
    context_items=(),
) -> GenerationOutput
```

The exact Python dataclasses remain a Cycle 0 approval item, but the following
invariants are normative:

- `generate()` does not accept previous SVG or feedback.
- `revise()` requires both previous SVG and feedback.
- Both methods use the same model invocation, extraction, validation, and trace
  representation internally.
- RAG and Critic are optional; Generator-only execution remains valid.
- Generator returns a typed success or failure. It must not return a placeholder
  SVG when model loading or generation fails.
- Revision count, acceptance threshold, and stop conditions are not owned by
  Generator.

## 4. Cross-Team Requirements

### 4.1 RAG

RAG output must preserve item boundaries. A single preformatted context string
is insufficient as the long-term contract because it removes provenance.

Each context item must provide or make derivable:

- stable item/source identifier
- content to be included in the prompt
- retrieval score
- score semantics, such as similarity or distance
- rank
- corpus or index version
- item kind

The minimum item kinds are:

- `reference_svg`
- `positive_experience`
- `negative_lesson`
- `correction_pair`

RAG owns retrieval, ranking, indexing, and score interpretation. Generator owns
safe prompt rendering, context-budget enforcement, and reporting which items it
actually consumed.

All Cycle 0 shared information is represented by the typed fields above.
Arbitrary vector-store metadata does not cross into Generator. The default
adapter policy drops every free-form metadata key and logs the excluded names.
A field such as `render_ref` may be added later only through an explicit
whitelist change. Rendered images are not part of the current text-only
Generator input.

### 4.2 Critic

Critic continues to evaluate:

```text
instruction + SVG attempt -> structured feedback
```

A persisted feedback event must be attributable to exactly one SVG attempt.
The event therefore requires:

- `feedback_id`
- `target_attempt_id`
- structured score, issues, and suggestions
- critic type
- critic model revision, when model-based
- critic prompt version
- raw response reference, when retained

Critic owns feedback quality and calibration. Orchestration wraps or correlates
the returned payload with run and attempt identifiers. Generator only consumes
the structured feedback passed to `revise()`.

Every concrete Critic response must satisfy the shared runtime validator before
it can affect a CompositeCritic aggregate or the orchestration loop. In
particular, boolean-looking strings are invalid rather than truthy values;
scores must be finite and within `0..10`; issue/suggestion collections and
provenance fields must retain their declared string types.

### 4.3 Orchestration

Orchestration owns:

- run-level correlation
- critic-driven revision count
- acceptance threshold
- identical-output and no-improvement detection
- timeout and Critic failure policy
- final accepted/rejected/superseded outcome
- selection among multiple candidate branches

Orchestration must not call the initial `generate()` operation as an implicit
revision path. It calls `revise()` with the previous attempt and the feedback
that triggered the correction.

Identifiers follow one producer-ownership rule:

- Factory/runtime assembly creates `run_id`.
- Generator creates `attempt_id` and its internal `model_call_id`.
- Orchestration creates `feedback_id` when it correlates Critic output to an
  attempt.
- RAG creates or preserves the stable item/source ID.

The resulting hierarchy is:

```text
run_id
  └── attempt_id        # one generate() or revise() operation
        └── model_call_id  # initial call or Generator-internal formatting retry
```

This distinction prevents a malformed-output retry from being mistaken for a
critic revision.

### 4.4 Artifact and Structure

The current top-level final artifact remains readable:

- `instruction`
- `svg_path`
- `is_valid`
- `render_path`
- `revision_count`
- `critic_feedback`
- `runtime`
- `metadata`
- `generated_at_utc`

The proposed version 1 extension must be additive and include:

- `schema_version`
- raw model output reference per model call
- exact user/system prompt reference and generation parameters per model call
- extracted SVG reference per attempt
- feedback target attempt ID
- previous-to-revised lineage
- attempt outcome and stop reason
- model, tokenizer, quantization, prompt, and critic versions
- RAG item source, score, score semantics, rank, and kind
- benchmark partition and memory-ingestion eligibility

Generator produces its trace data. Structure owns file names, atomic writes,
sidecar migration, reader compatibility, and intermediate artifact retention.
Structure must validate a version 1 payload with the same parser used by readers
after its immutable files are promoted but before the sidecar commit marker is
published. Failed validation must remove the unpublished bundle and must not
replace an existing sidecar or export alias.
The canonical SVG must match the final successful attempt byte-for-byte.
Sidecar replacement is the exact commit point: failures before it remove the
new bundle, while failures after it retain that bundle and leave the
non-canonical SVG export alias unchanged.

Absence of `schema_version` is interpreted as legacy version 0. Version 1
readers should distinguish a valid legacy omission from an incomplete version
1 artifact.

### 4.5 SVG Validation

Generator requires a validator result that distinguishes at least:

- SVG extraction success
- well-formed XML
- structural validity
- unsafe element or external-reference failure
- render success, when rendering is enabled

Generator may retry formatting failures. It does not redefine validator
semantics. Orchestration combines validation and Critic results into the final
acceptance decision.

Until this contract is fixed, `is_valid` must not be treated as synonymous with
“accepted by the agentic loop.”

### 4.6 Evaluation

The final external benchmark has not been selected. SVGenius is only the first
dataset-specific candidate adapter. It must not become the project default
implicitly because preparation code exists. Before any dataset is adopted, the
Evaluation owner records:

- immutable dataset revision
- exact text-to-SVG task assets and split
- resolved license and source provenance
- sample inclusion/exclusion rules
- reference-based and prompt-based metrics
- held-out isolation policy

For the SVGenius candidate specifically, the public dataset card currently
exposes SVG code and complexity fields but does not by itself establish the
complete Generator evaluation protocol. Its dataset-specific adapter joins
those rows to the official caption tasks and marks the result
`candidate_only`. Evaluation must not silently treat that prepared snapshot as
the final text-to-SVG runner input.

The pinned medium split contains source asset
`page_38_ant_design_48353_icon_95`, but the pinned text-to-SVG task revision
contains no corresponding caption. Adapter `svgenius-text-to-svg-v2` therefore
permits exactly this exclusion only when the dataset revision, task revision,
split, and asset key all match. It records the configured and applied
exclusions in manifest schema 2. Any other missing row or unmatched caption
remains a strict preparation failure; adapter v2 exposes no broad
`--allow-unmatched` CLI bypass. The resulting candidate shape is 299
joined records (`100/99/100`), not a silently truncated 300-record snapshot.

Every additional candidate must have its own adapter. Dataset-specific field
mapping, split interpretation, joining, validation, and provenance rules must
not be added to the shared `data.preprocess` module.

Prepared records distinguish the immutable upstream `source_split` from the
project-owned `data_partition`. A candidate adapter emits
`data_partition=candidate_unassigned`; only the Evaluation owner may assign
development or held-out test semantics after selection.

Project smoke fixtures remain separate from the external benchmark and make no
leaderboard claim.

### 4.7 Model Runtime and Factory

The local default is:

| Setting | Value |
|---|---|
| Repository | `lmstudio-community/gemma-4-12B-it-QAT-GGUF` |
| File | `gemma-4-12B-it-QAT-Q4_0.gguf` |
| Revision | `291406f49e16eff811c85ad8884d375f34138663` |
| Upstream model | `google/gemma-4-12B-it-qat-q4_0-unquantized` |
| Quantization provider | LM Studio Community |
| Conversion runtime | `llama.cpp b9518` |
| Engine | `llama-cpp-python==0.3.34`, built with CUDA |
| GPU placement | `n_gpu_layers=-1` |
| Initial context | `n_ctx=8192` |
| Batch | `n_batch=512` |
| Attention | flash attention enabled |
| Modality | text-only; no multimodal projection is loaded |

This profile retains the selected Gemma 4 12B QAT Q4_0 model family and
approximately 6.98 GB file size while changing the GGUF distributor. Actual
12 GB GPU headroom still depends on KV cache and runtime allocations and must
be measured. CPU offload is an explicit fallback profile and must not activate
silently.

Local native-runtime verification on 2026-07-20 confirmed
`llama-cpp-python==0.3.34`, CUDA offload support, Gemma 4 architecture support,
and detection of the RTX 4080 Laptop GPU with 11,876 MiB reported VRAM. The
previous Google-hosted file was fully downloaded and its SHA-256 matched the
content-addressed cache, but loading aborted at vocabulary token 237922 before
tensor placement. The same assertion is reported with current standalone
llama.cpp, so this is an upstream file compatibility incident rather than an
OOM or backend configuration failure. The replacement compatibility pin has
loaded successfully and produced valid non-placeholder SVG output. E2 is still
incomplete as a hardware-acceptance record because TTFT, tokens/second, and
peak-VRAM/headroom evidence remain C0-10 work.

The backend is selected by config and implements `BaseModelBackend`; future
Transformers, vLLM, or remote cloud backends do not require Generator changes.
These runtimes are not interchangeable and must not all be recorded simply as
`Q4`.

Relevant official references:

- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- [Selected compatibility QAT Q4_0 checkpoint](https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF)
- [Rejected Google GGUF vocabulary-load incident](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/discussions/6)
- [Gemma 4 prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [SVGenius dataset card](https://huggingface.co/datasets/xiaoooobai/SVGenius)

## 5. Core Research Assumption: Feedback Experience Memory

The central research extension is to reuse Generator–Critic experience as
retrievable context:

```text
static SVG corpus
        +
accepted examples
        +
failed attempt → feedback → accepted correction
        ↓
RAG retrieval
        ↓
future Generator request
```

This direction affects other teams even though Generator does not implement the
memory store.

The required separation is:

- Generator preserves prompt, context, raw output, SVG attempt, feedback
  reference, and correction lineage.
- Critic produces structured, versioned feedback.
- Orchestration records the relationship and final outcome.
- Memory curator decides whether an event is trustworthy and reusable.
- RAG indexes and retrieves curated experiences.
- Evaluation prevents held-out benchmark events from entering persistent
  memory.

An `accepted` attempt is only a positive-memory candidate. A `rejected` attempt
is not automatically a useful negative example. Negative memory should
preferably be represented as a lesson or a failed→feedback→corrected tuple so
the Generator is not shown an unqualified bad SVG.

The initial research claim should be “retrieval-conditioned experience reuse”
or “experience-augmented generation.” The term “self-evolving” is reserved for
experiments that demonstrate reproducible improvement over rounds on held-out
prompts without benchmark leakage.

Minimum ablations for that claim are:

- no memory
- static RAG only
- positive experience only
- negative lesson or correction pair only
- static RAG plus experience memory
- memory round 0/1/2 under the same model, prompt, and evaluation split

## 6. Ownership Summary

| Workstream | Must provide | Must not be delegated to Generator |
|---|---|---|
| Generator | `generate()`/`revise()`, raw/extracted output, Generator provenance, consumed-context trace | RAG indexing, Critic scoring, global stop policy, memory curation |
| Model/Platform | loadable checkpoint profile, runtime dependencies, immutable model/tokenizer revisions | SVG extraction and revision semantics |
| RAG | typed context items, source/score/kind/version, static and experience retrieval | Generator prompt precedence and output cleanup |
| Critic | structured and versioned feedback | continuation, final acceptance, memory labeling |
| Orchestration | run correlation, feedback linkage, loop/branch/stop policy, final outcome | feedback quality and retrieval ranking |
| Structure/Artifact | schema version, persistence, paths, migration, retention | generation and critique quality |
| SVG Validation | strict structural and safety result | semantic acceptance |
| Evaluation | Final benchmark selection/freeze, dataset adapters, metrics, benchmark isolation, reports | Generator runtime implementation |
| Memory Curator | eligibility, confidence, deduplication, retention | model generation and RAG transport |

## 7. Cycle 0 Open Decision Register

The following items remain to be decided. Their priority is based on who is
blocked, not on implementation difficulty.

### 7.1 Must Be Decided Before the Shared Contract PR

| ID | Required decision | Proposed default | Decision owner / required review | Status |
|---|---|---|---|---|
| C0-01 | Exact `GenerationOutput`, context item, revision input, and feedback event dataclasses | Typed additive contracts; retain a compatibility adapter for the current string-returning Generator | Generator / RAG, Critic, Orchestration, Structure | Implemented; cross-team review required |
| C0-02 | Identifier ownership and retry granularity | Producer-owned IDs: Factory `run_id`; Generator `attempt_id`/`model_call_id`; Orchestration `feedback_id`; RAG item/source ID | Orchestration / Generator, Critic, Artifact | Accepted and implemented |
| C0-03 | Failure contract | Infrastructure failure raises explicitly; model-output/extraction failure returns a typed failed attempt; never substitute placeholder SVG | Generator / Factory, CLI, Orchestration, Evaluation | Accepted and implemented |
| C0-04 | Validation versus acceptance semantics | Validator reports code/safety facts; Orchestration records accepted/rejected outcome | Orchestration / Generator, SVG Validation, Critic, Evaluation | Accepted and implemented |
| C0-05 | Artifact schema version 1 and intermediate retention | Preserve current top-level fields; add versioned trace; retain raw/intermediate files for experiment runs | Structure / Generator, Evaluation, Memory | Accepted and implemented; locked immutable-bundle publication plus typed field, unique-ID, and lineage-correlation validation verified |
| C0-06 | RAG context budget, metadata, and conflict policy | Instruction has priority; kind-specific delimited items; empty free-form metadata whitelist; deterministic truncation recorded in trace | Generator / RAG, Evaluation | Accepted and implemented |

### 7.2 Must Be Decided Before the Real Model Backend PR

| ID | Required decision | Proposed default | Decision owner / required review | Status |
|---|---|---|---|---|
| C0-07 | Exact Gemma checkpoint and immutable revision | Compatibility QAT Q4_0 GGUF at the revision specified in section 4.7; preserve its Google upstream and conversion provenance | Model/Platform / Generator, Factory, Artifact | Accepted, configured, loaded, and used for real generation |
| C0-08 | Q4 format and inference engine | CUDA-enabled llama.cpp with full GPU offload; config-selectable backend | Model/Platform / Generator, Environment | Accepted; native CUDA, weight load, and real generation verified |
| C0-09 | Runtime lock | Pin CUDA-enabled llama.cpp Python binding, Hugging Face client, and tested CUDA build recipe | Model/Platform / CI, Generator | Local build verified; clean-environment/CI review required |
| C0-10 | Local memory and latency acceptance budget | GPU-resident preferred; at least 1 GiB VRAM headroom; CPU offload is a measured fallback. Record TTFT, tokens/second, and end-to-end latency in the first hardware spike, then approve numeric thresholds. | Generator / Evaluation, future Cloud deployment | Open; replacement inference works, but the hardware measurement record is incomplete |
| C0-11 | Chat template and thinking behavior | Backend uses the template stored in GGUF metadata; first SVG baseline does not request a reasoning mode | Generator / Model/Platform, Evaluation | Accepted and implemented |
| C0-12 | Input/context/output length profile | Start with 8K model context, 12,000 context characters, and 2,048 completion tokens; deterministic context truncation is recorded | Generator / RAG, Critic, Evaluation | Accepted as tunable baseline |

### 7.3 Must Be Decided Before Reporting Quality Results

| ID | Required decision | Proposed default | Decision owner / required review | Status |
|---|---|---|---|---|
| C0-13 | Final benchmark selection and frozen dataset contract | Compare candidates, then pin the chosen dataset revision, task subset, license decision, adapter version, and metrics. SVGenius remains candidate-only until that decision. | Evaluation / RAG, Artifact | Open |
| C0-14 | Primary metrics and success thresholds | Separate code validity, render success, semantic constraints, latency, and VRAM; do not collapse into one score | Evaluation / Generator, Critic | Open |
| C0-15 | Benchmark-memory isolation | `memory_eligible=false` for held-out runs; block ingestion by partition and run policy | Evaluation / Memory, RAG, Orchestration | Open |
| C0-16 | Research claim threshold | Require held-out round-by-round improvement and memory ablations before using “self-evolving” | Research/Evaluation / all research workstreams | Open |

C0-13 through C0-16 do not block the first Generator contract implementation,
but they block benchmark-backed model selection and research claims.

## 8. Cycle 0 Exit Criteria

Cycle 0 is complete when:

- D-001 through D-006 are represented in shared schemas or documented owner
  contracts.
- C0-01 through C0-06 have an accepted decision and contract tests with fake
  components.
- C0-07 through C0-12 define one reproducible local model profile.
- Structure owner has accepted or revised the artifact version 1 proposal.
- RAG and Critic owners can implement their modules without importing Generator
  internals.
- Generator can be tested with fake RAG and fake Critic components.
- Every remaining external dependency has an owner and does not silently expand
  the Generator workstream.

## 9. Change and Review Policy

- A change to an accepted cross-team decision requires an update to this
  document in the same pull request.
- A schema-breaking change requires a new schema version and migration note.
- A decision made in chat or a meeting is not considered integrated until its
  status and impact are recorded here.
- The responsible workstream records the review outcome as `Accepted`,
  `Rejected`, or `Deferred`, together with the decision date and replacement
  proposal when applicable.
- Experimental results belong in run reports; only adopted implications are
  promoted into this contract.
- Open items must retain an explicit owner/workstream and the downstream work
  they block.
