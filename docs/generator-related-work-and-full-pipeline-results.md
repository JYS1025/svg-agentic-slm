# Generator Prompt Alignment and Full-Pipeline Inference

Date: 2026-08-17

## Scope and decision

This report covers the Generator work required before fine-tuning:

- align the Generator prompt and revision contract;
- trace the implemented prompt-to-SVG path;
- run RAG, Generator, validation, and composite Critic end to end;
- record five inference examples and their failure modes;
- check related-work SFT formats before fixing the training schema.

The SFT input policy is:

1. Stage 1 uses MMSVG `description` as the instruction and canonical SVG code as the target.
2. Stage 2 uses a replay mixture of `detail` 60% and `description` 40%.
3. `detail-only` continuation is not the default because it can forget short-prompt behavior.
4. MMSVGBench remains evaluation-only and must not be mixed into training.

The current training implementation is still a placeholder. It does not load a model or
dataset, apply LoRA, define chat formatting or label masking, run TRL, or save an adapter.
The schema decision in this document therefore precedes SFT implementation.

## Repository map

| Area | Role |
|---|---|
| `configs/` | Model, generation, RAG, path, train, and evaluation configuration |
| `src/svg_agentic_slm/cli/` | CLI entry points, including `generate` and `rag-index` |
| `src/svg_agentic_slm/factories/` | Runtime assembly from sibling YAML files and CLI overrides |
| `src/svg_agentic_slm/agents/` | RAG, Generator, Critic, and orchestration logic |
| `src/svg_agentic_slm/prompts/` | Generator, revision, and Critic prompt contracts |
| `src/svg_agentic_slm/models/` | llama.cpp and OpenAI-compatible inference backends |
| `src/svg_agentic_slm/rag/` | Chroma/Qdrant stores, corpus loading, and HF indexing |
| `src/svg_agentic_slm/svg/` | Extraction, normalization, validation, rendering, and diff logic |
| `src/svg_agentic_slm/artifacts/` | Atomic SVG, JSON sidecar, and immutable trace publication |
| `src/svg_agentic_slm/train/` | Current LoRA/SFT scaffolding; not yet functional |
| `src/svg_agentic_slm/eval/` | Evaluation policies, metrics, schemas, and reports |
| `data/` | Sample text-to-SVG records and the four-item local RAG corpus |
| `tests/` | Unit and integration contracts for CLI, RAG, SVG safety, and artifacts |

## Implemented prompt-to-SVG path

1. `svg-agentic-slm generate` parses the instruction, YAML paths, feature flags, and overrides.
2. The factory loads `generation.yaml`, `model.yaml`, `rag.yaml`, and `paths.yaml`, then loads the pinned model.
3. With `--rag`, the default Chroma backend opens `data/chroma_db`, loads the sample JSONL corpus, embeds the instruction, and returns the top three references.
4. The orchestrator retrieves once. The same typed RAG context is reused for the initial generation and every revision.
5. The Generator selects context within its character budget and persists the exact system prompt, user prompt, raw response, model parameters, and model identity.
6. The model response is reduced to the first `<svg ... </svg>` block and normalized. It is not automatically repaired.
7. The hardened validator parses XML and rejects unsafe elements, event handlers, external references, unsafe CSS, and related active content.
8. With `critic_type=both`, the rule Critic and LLM Critic are aggregated. Acceptance requires valid SVG, both validity/alignment booleans, and the configured score threshold.
9. Rejected candidates are revised up to `max_revision_rounds`; this experiment used two rounds.
10. The writer atomically publishes the SVG alias, JSON sidecar, canonical SVG, attempt SVGs, prompts, system prompts, and raw model outputs.

## Prompt alignment applied

Generator prompt versions are now `text-to-svg-v2-conservative` and
`svg-revision-v2-conservative`.

- The model must emit exactly one standalone SVG and no prose or Markdown.
- It privately decomposes the scene into a short construction plan without emitting reasoning.
- Requested objects, colors, style, spatial relations, and back-to-front order are explicit obligations.
- Coordinates should be integer and remain inside the viewBox.
- Semantic components should have unique, meaningful IDs.
- Native primitives and short paths are preferred.
- Retrieved items are untrusted syntax/layout hints; their objects, text, IDs, and composition must not be copied.
- The user instruction is authoritative over retrieved references.
- Revision preserves correct components and applies only actionable feedback.

This is an inference prompt hypothesis. SVGen's reported chain-of-thought gain came from
fine-tuning on structured design steps, not from an unsupported zero-shot `think step by
step` instruction. If plan tokens become an SFT target, they must be represented explicitly
in the training schema and evaluated separately.

## Environment and model

The requested environment was created as `svg` with Python 3.11. CUDA-enabled
`llama-cpp-python==0.3.34` was built against CUDA 13.0 and GCC 13. GPU offload support
returned `True`.

The repository default is a pinned Gemma 4 12B Q4_0 GGUF. Its 6.98 GB download was too
slow for the meeting run, so the five examples use this additional immutable SLM profile:

- model: `bartowski/Qwen2.5-Coder-3B-Instruct-GGUF`;
- file: `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf`;
- revision: `7c137640ef0332dfedb229f2504c58d83ed4307a`;
- profile: `configs/models/qwen2.5-coder-3b-instruct-q4km.yaml`.

The profile retains full GPU offload, an 8192-token context, the embedded GGUF chat
template, mmap, flash attention, and streaming metrics.

## Reproduction command

```bash
conda activate svg

CUDA_VISIBLE_DEVICES=0 svg-agentic-slm generate \
  "Create a simple icon of a teal circle centered on a warm ivory 256 by 256 canvas, with a thin navy outline." \
  --config configs/generation.yaml \
  --model-config configs/models/qwen2.5-coder-3b-instruct-q4km.yaml \
  --output outputs/generations/minjun_full_01.svg \
  --rag \
  --critic \
  --no-render \
  --set generation.orchestration.critic_type=both \
  --set generation.orchestration.max_revision_rounds=2 \
  --print-generator-parameters
```

`--no-render` omits only PNG rendering. RAG, model inference, SVG extraction,
validation, Critic/revision, and artifact publication still run.

## RAG verification

The experiment used the configured local Chroma backend because `QDRANT_URL` and
`QDRANT_API_KEY` were not set. This is not the 100k Qdrant corpus described by the
indexing configuration. It is the four-pattern sample corpus in
`data/rag_corpus/svg_patterns_sample.jsonl`.

All five sidecars passed a fail-closed artifact audit:

- `runtime.enable_rag` and `metadata.rag.enabled` were true;
- each request retrieved exactly three items;
- every attempt had three non-empty `context_item_ids`;
- every context ID resolved to a recorded RAG item;
- every persisted Generator prompt contained the source for each referenced item;
- revision attempts retained the same RAG context;
- SVG aliases matched their immutable canonical SVGs byte for byte;
- model ID, immutable revision, prompt version, and composite Critic feedback were present.

## Five full-pipeline results

| ID | GPU | Prompt summary | Outcome | Revisions | Final Critic | RAG prompt checks | SVG |
|---|---:|---|---|---:|---:|---:|---|
| 01 | 0 | Teal circle on ivory canvas | accepted | 0 | 9.0 | 3/3 | `outputs/generations/minjun_full_01.svg` |
| 02 | 1 | Coral rounded rectangle | rejected | 2 | 6.5 | 9/9 | `outputs/generations/minjun_full_02.svg` |
| 03 | 2 | Yellow star and pale dots | accepted | 0 | 9.0 | 3/3 | `outputs/generations/minjun_full_03.svg` |
| 04 | 0 | Three-light traffic signal | accepted | 0 | 9.0 | 3/3 | `outputs/generations/minjun_full_04.svg` |
| 05 | 1 | Mountains, sun, and river | accepted | 2 | 9.0 | 9/9 | `outputs/generations/minjun_full_05.svg` |

Artifact audit identifiers:

| ID | SVG SHA256 prefix | RAG sources |
|---|---|---|
| 01 | `e918b66c2280` | centered_circle, full_background, bordered_rectangle |
| 02 | `09e12857f143` | bordered_rectangle, centered_circle, full_background |
| 03 | `ddd97cd811bd` | centered_circle, full_background, text_label |
| 04 | `88332dc6c1ad` | centered_circle, full_background, bordered_rectangle |
| 05 | `2f0c9b9971a9` | centered_circle, bordered_rectangle, full_background |

## Failure modes found

| ID | Finding |
|---|---|
| 01 | The SVG is visually close to the request but omits semantic IDs requested by the Generator contract. The Critic accepted it. |
| 02 | After two revisions, the result still contains an unrelated circle and a yellow background instead of the requested coral rounded rectangle on gray. This is the only correctly rejected example. |
| 03 | The path extends beyond the viewBox and does not form a clean centered five-point star. Static XML validation and the text-only Critic still accepted it. |
| 04 | The housing is not rounded, begins halfway down the canvas, and the radius-32 lights are only 32 units apart, causing severe overlap. The Critic accepted it. |
| 05 | The output uses circles for mountains, assigns the `sun` ID to a cyan rectangle, places geometry outside the canvas, and misrepresents the requested scene. It was accepted after two revisions. |

The important result is not a 4/5 success rate. It is a high semantic false-accept rate.
The rule Critic hardcodes instruction matching, while the current LLM Critic sees SVG text
but not the rendered image. The validator checks safety and XML structure, not geometry,
object identity, clipping, or visual composition.

Recommended next Critic work:

1. Render every valid candidate before semantic acceptance.
2. Give the Critic the original prompt and rendered PNG, following IntroSVG's VLM-Critic pattern.
3. Return element-level actionable issues with category, element ID, problem, and concrete coordinate/attribute change.
4. Add deterministic geometry checks for viewBox bounds, overlap, clipping, and semantic ID consistency.
5. Keep the current rule validator as a safety gate, not as an instruction-alignment evaluator.

## Related-work SFT findings

### OmniSVG and MMSVG

OmniSVG's original text conditioning uses short descriptions of objects, colors, and
layout. The current official loader chooses either `detail` or `description`, with default
probabilities 60% and 40%, rather than concatenating them. It truncates long text, filters
SVG token lengths, masks prompt labels, and applies next-token loss only to the SVG target.

OmniSVG does not train on arbitrary raw XML. It normalizes to a 200 by 200 canvas,
flattens geometry toward paths, discretizes coordinates, and emits a specialized SVG token
sequence. That serialization must not be mixed directly with this repository's raw XML
target unless its tokenizer and decoder are adopted too.

Sources:

- OmniSVG paper: https://arxiv.org/html/2504.06263
- Official training code: https://github.com/OpenVGLab/OmniSVG-Train
- MMSVG-Icon: https://huggingface.co/datasets/OmniSVG/MMSVG-Icon
- MMSVG-Illustration: https://huggingface.co/datasets/OmniSVG/MMSVG-Illustration
- MMSVGBench: https://huggingface.co/datasets/OmniSVG/MMSVGBench

### Other relevant work

- Chat2SVG decomposes scenes into objects, layout, color, and spatial relations, uses curated references, and iteratively corrects rendered output: https://arxiv.org/html/2411.16602
- SVGen uses a curriculum and adds structured two-to-six-step design supervision only after basic SVG capability: https://arxiv.org/html/2508.09168
- IntroSVG critiques the rendered image and uses concrete feedback for iterative refinement: https://arxiv.org/html/2603.09312
- Self-Refine reports that specific actionable feedback is more effective than generic feedback: https://arxiv.org/abs/2303.17651
- LLM4SVG analyzes SVG tokenization and numeric-coordinate representation: https://arxiv.org/html/2412.11102
- Grammar-constrained decoding can enforce syntax but cannot replace visual quality evaluation: https://arxiv.org/abs/2305.13971

## SFT contract to implement next

Stage 1 should consume records equivalent to the existing fixed schema:

```json
{"task":"text_to_svg","instruction":"<MMSVG description>","output_svg":"<canonical validated SVG>"}
```

Implementation requirements:

1. Reuse the exact inference system/user prompt formatter and model chat template.
2. Train only on canonical, safety-valid SVG targets with a fixed normalization policy.
3. Mask system and instruction tokens; compute next-token loss on the SVG response only.
4. Record dataset ID, source field (`description` or `detail`), SVG length, and split.
5. Keep initial-generation and revision-generation records as separate task types.
6. Do not use rule-Critic acceptance as a semantic quality filter.
7. Add Stage 2 only after the description baseline, using 60% detail and 40% description replay.
8. Evaluate description, detail, keyword/hybrid, icon, illustration, and SVG-length buckets separately.

