# Pipeline Operational Safety Contract

This document defines the runtime contracts shared by the RAG, Generator,
Critic, Orchestration, Validation, and Artifact workstreams. It covers
operational correctness only. Model quality, Critic calibration, and RAG
leakage evaluation remain separate research work.

## RAG context assembly

- Generator context budgets are tokenizer budgets (`max_context_tokens`), not
  character limits. A backend used with non-empty RAG context must implement
  `count_tokens` using the tokenizer of the served model.
- Retrieved non-SVG items are included or dropped as complete items. Reference
  SVGs may be reduced only at complete top-level SVG element boundaries; every
  included variant must remain a valid standalone SVG document.
- Each retrieved item records `fully_used`, `partially_used`, or `dropped`, its
  actual included token contribution, and included/total element counts.
  `context_token_count` records the tokenizer count of the final formatted RAG
  context.
- Do not add a character-slicing fallback. If token counting is unavailable for
  non-empty context, fail before calling the Generator.

## Candidate selection and stopping

- The Orchestrator preserves the highest-scoring structurally valid candidate.
  It never publishes the last revision solely because it was generated last.
- A score regression or invalid revision rolls back to the valid best-so-far.
  Improvements smaller than `min_critic_score_improvement` increment the
  no-improvement counter; reaching `max_no_improvement_rounds` stops revision
  and selects the best candidate.
- `metadata.selection.selected_attempt_id` is authoritative. CLI output,
  artifact `final.svg`, top-level outcome, and artifact consistency checks must
  all resolve this attempt. `last_attempt_id` remains diagnostic only.
- Every attempt must have its own outcome and stop reason. Schema v2 operation
  outcomes include `selected_best`, `rolled_back`, and
  `critic_contract_failure`; schema v1 compatibility remains read-only.

## Static SVG policy

- `svg.policy.STATIC_SVG_POLICY` is the single policy source used by the
  Generator system prompt and `SVGValidator`.
- Active elements, including animation and scripts, event-handler attributes,
  foreign content, data URLs, absolute URI schemes, and external references are
  rejected. Same-document `#fragment` references remain supported.
- Policy changes must update the shared policy object and add tests proving both
  prompt visibility and validator enforcement. Do not duplicate a separate
  prompt-only denylist.

## Critic contract failures

- The packaged `critic_output.schema.json` and semantic validation enforce the
  same status conditions: `pass` has no issues or preserve entries, while
  `revise` has at least one issue.
- The second model call is serialization-only format repair of the previous
  response. It must not request or create a new judgment.
- If repair still fails, raise `CriticTraceError` and retain all call traces.
  The Orchestrator records `critic_contract_failure`, emits no feedback event,
  and never sends this infrastructure failure to Generator revision.

## Artifacts and timing

- Provenance records Git SHA and dirty state, the execution command, hashes of
  resolved config files, per-call Generator/Critic prompt hashes, and a
  benchmark hash when a benchmark is present.
- Stage timing separates pipeline, RAG, Generator, Critic, validation, and
  rendering latency. Critic evidence rasterization is included in render
  latency and separately exposed as `critic_evidence_render_latency_seconds`;
  it is not double-counted as Critic model time.
- `metadata.render.role=user_output_render` identifies the optional user-facing
  render. Attempt evidence uses `role=critic_evidence_render`. RAG context state
  is recorded under `metadata.rag.context_usage` and is not render evidence.
- Artifact publication uses a per-sidecar atomic lock directory. It must be
  removed after both successful and failed publication. Existing regular lock
  files are treated as legacy blockers rather than deleted while ownership is
  unknown.

## Required regression coverage

Changes to these paths must retain tests for token-budgeted atomic context,
best-candidate rollback and stopping, shared static SVG restrictions, Critic
format repair and failure isolation, selected-attempt artifact round trips,
provenance fields, stage timing, and concurrent lock cleanup.
