# Generate Command Workflow

This document defines the current ownership boundary and integration contract
for the `svg-agentic-slm generate ...` path.

For the proposed Cycle 0 contracts and research assumptions that affect other
workstreams, see
[Generator Cross-Team Contract and Research Assumptions](./generator-cross-team-contract.md).
Cycle 0 completion gates and manual experiments are tracked in
[Generator Cycle 0 Status and Experiment Runbook](./generator-cycle0-status-and-runbook.md).

## Why This Exists

The project already had the architectural layers for generation, but the CLI
was still printing a hard-coded placeholder SVG. That made it hard for
different contributors to work in parallel, because `Generator`, `Critic`, and
`RAG` changes were not being exercised by the real command path.

The goal of this workflow is to keep the command path stable while each module
owner evolves their own implementation.

## Current Runtime Flow

The `generate` command now follows this sequence:

1. Read `generation.yaml`.
2. Resolve sibling configs for `model.yaml`, `rag.yaml`, and `paths.yaml`.
3. Assemble runtime dependencies in `src/svg_agentic_slm/factories/generation.py`.
4. Build a `GenerationRequest`.
5. Pre-compute planned artifact paths for SVG, metadata, and render output.
6. Call `SVGGenerationOrchestrator.run(...)`.
7. Publish the immutable artifact bundle and JSON sidecar through
   `src/svg_agentic_slm/artifacts/writer.py`.
8. Print a user-facing summary in the CLI.

Files involved:

- `src/svg_agentic_slm/cli/commands_generate.py`
- `src/svg_agentic_slm/cli/commands_render.py`
- `src/svg_agentic_slm/artifacts/generation.py`
- `src/svg_agentic_slm/artifacts/writer.py`
- `src/svg_agentic_slm/factories/generation.py`
- `src/svg_agentic_slm/agents/orchestrator.py`

`factories/generation.py` owns configuration resolution, component assembly,
and artifact path planning. `artifacts/writer.py` owns locking, immutable bundle
creation, strict pre-publication validation, atomic sidecar publication, and
the exported SVG alias. Existing imports of `GenerationArtifacts` and
`persist_generation_artifacts` from the factory remain supported.

## Ownership Boundary

### Structure / Integration Owner

This workstream owns:

- CLI command behavior
- config loading and resolution
- runtime assembly and dependency wiring
- output artifact conventions
- render artifact conventions
- artifact reader utilities
- orchestration metadata and integration tests

### Generator Owner

The generator is responsible for:

- turning a `GenerationRequest` plus optional context into SVG text
- respecting generation overrides from `request.config_overrides`
- rendering typed RAG context into the Generator-owned prompt
- extracting and normalizing SVG from raw model output
- returning attempt, model-call, prompt, and context provenance

The generator should not:

- read YAML directly
- decide output filenames
- write output files
- invoke the renderer directly

### Critic Owner

The critic is responsible for:

- returning `CriticFeedback`
- keeping critique logic inside the critic implementation
- preserving the runtime `CriticFeedback` contract: finite `0..10` score,
  concrete booleans, string issue/suggestion lists, and string provenance

The critic should not:

- mutate files
- depend on CLI behavior
- decide render output paths

### RAG Owner

The RAG layer is responsible for:

- retrieval backend behavior
- returning typed, stable, provenance-preserving examples through
  `RAGAgent.retrieve(...)`
- projecting vector-store metadata through the shared whitelist

The RAG layer should not:

- own CLI flags
- own output persistence
- own render artifact rules
- construct the Generator prompt

## Config Resolution Rules

The generate command accepts one explicit config path:

- `--config <path-to-generation.yaml>`

From that file, the runtime resolves sibling config files first:

- `model.yaml`
- `rag.yaml`
- `paths.yaml`

If a sibling file is missing, it falls back to the project-level `configs/`
directory.

This rule lets contributors create self-contained config bundles for tests,
experiments, or branches without rewriting the command.

## Artifact Contract

Each generate run writes:

- one exported `.svg` file for direct use
- one immutable companion run bundle containing the canonical SVG, traces, and
  successful render output
- one external render artifact when rendering is enabled, currently `.png` by default
- one `.json` sidecar with the same basename

If `--output` is not provided, the command writes to:

- `<outputs.generations>/<UTC timestamp>_<instruction slug>_<run id>.svg`
- `<outputs.renders>/<UTC timestamp>_<instruction slug>_<run id>.png`

The timestamp includes microseconds and the run ID is random, so concurrent
runs and prompts whose slugs normalize to the same value cannot overwrite one
another.

If `--output` is provided, the command writes:

- the SVG to the explicit path
- the metadata sidecar next to the SVG
- the render artifact next to the SVG with a run-qualified stem

An explicit generation output must either omit its extension or use `.svg`.
Other extensions are rejected before generation so the SVG cannot collide with
its JSON sidecar or be written under a misleading file type.

The canonical top-level `svg_path` and `render_path` point into the immutable
`<sidecar-stem>.artifacts/<run-id>/` bundle and are relative to the sidecar
location. Version 1 readers validate required scalar and nested record types,
require references to remain inside the sidecar directory, and reject missing,
absolute, or escaping references. Legacy version 0 readers retain compatibility
with older absolute and working-directory-relative sidecars.
The writer runs the same version 1 parser against the promoted bundle before
publishing the sidecar. A malformed payload removes the unpublished bundle and
cannot replace the previous commit marker or exported SVG alias.
The canonical SVG must be byte-identical to the final successful attempt SVG.
The sidecar replacement itself is the commit point: an error after replacement
preserves the referenced bundle, while the non-canonical SVG alias remains
unchanged because alias publication occurs only after durable completion.

The standalone `render` command follows the same renderer backend and format
rules, but it is strict:

- render success returns exit code `0`
- render failure returns exit code `1`
- an explicit `--format` must match an explicit output-file extension
- when no output path is given, `--format` determines the output extension

The JSON sidecar currently captures:

- instruction
- validation outcome
- render outcome
- critic feedback
- resolved config paths
- runtime feature flags
- raw result metadata from the orchestrator
- generation timestamp

## Orchestrator Metadata Contract

`SVGGenerationOrchestrator.run(...)` now records integration metadata in
`GenerationResult.metadata` for downstream tooling.

Current keys:

- `request`
- `rag`
- `validation`
- `render`
- `critic`
- `timing`

Module owners should prefer extending these nested sections instead of adding
top-level ad hoc keys unless a new category is genuinely needed.

The `render` section is now the canonical place for:

- render success or failure
- planned render path
- actual render path
- render width, height, and format
- backend failure details

The JSON sidecar is now readable through:

- `load_generation_artifact(...)`
- `list_generation_artifacts(...)`

These helpers are intended to be the shared entrypoint for evaluation,
reporting, and notebook tooling instead of ad hoc JSON parsing.

The `eval` workflow now consumes these helpers directly. The primary config
entry is:

- `configs/eval.yaml -> eval.artifact_path`

Supported sources are:

- a directory of `.json` sidecars
- a single `.json` sidecar
- a single `.svg` artifact with a matching `.json` sidecar

`eval.metrics` controls which aggregate metrics are computed. Unknown metric
names fail fast instead of silently producing incomplete reports. Render
success rate includes only artifacts where rendering was enabled, and a
recorded success counts only while the referenced render file exists.

The report directory precedence is:

1. explicit `--report-dir`
2. `--set eval.output_dir=...`
3. `eval.output_dir` from the evaluation config
4. `./outputs/eval_reports`

## Shared CLI Override Rules

`generate` and `eval` now share the same override contract:

- first-class convenience flags for common fields
- repeated `--set dotted.path=value` overrides for everything else

Examples:

- `svg-agentic-slm generate "..." --max-new-tokens 512 --temperature 0.2`
- `svg-agentic-slm generate "..." --set generation.top_p=0.8 --set model.model_id="custom-model"`
- `svg-agentic-slm eval --artifact-path outputs/generations --max-samples 25`
- `svg-agentic-slm eval --set eval.metrics='["svg_validity_rate"]'`

Values provided through `--set` are parsed as YAML scalars/collections, so all
of the following work:

- booleans: `true`, `false`
- numbers: `42`, `0.75`
- lists: `["a", "b"]`
- nulls: `null`

When a first-class flag and `--set` target the same config key, the first-class
flag wins.

## Working Rules For Teammates

When changing `Generator`, `Critic`, or `RAG`, keep these rules:

1. Preserve constructor-level dependency injection.
2. Do not move config loading into module logic.
3. Do not write files from agents or retrievers.
4. Keep outputs serializable through `GenerationResult` and `CriticFeedback`.
5. If a contract must change, update this document and the generate pipeline tests together.

## Current Extension Points

### Safe To Extend Now

- `LlamaCppModelBackend.load_model()` and `.generate()`
- `GeneratorAgent.generate()`
- `RuleBasedCritic.critique()`
- `LLMCritic.critique()`
- `ChromaRetriever.add_documents()` and `.retrieve()`
- artifact metadata payload shape inside the existing nested sections
- artifact reader consumers built on `svg_agentic_slm.artifacts`

### Coordinate Before Changing

- `GenerationRequest`
- `GenerationResult`
- `CriticFeedback`
- config file names and lookup rules
- artifact naming convention
- orchestrator call order

## Follow-Up Backlog

Recommended next structural tasks:

1. Split runtime assembly into per-component builders if model families multiply.
2. Add regression tests for `critic_type=llm` and `critic_type=both`.
3. Decide whether render failures should optionally become hard failures in strict CI modes.
4. Decide whether dataset-backed evaluation should be restored alongside artifact-backed evaluation later.

## Acceptance And Artifact Rules

- A Critic score cannot accept an attempt that failed SVG validation.
- `accepted` requires a structurally safe SVG and, when enabled, Critic
  `is_valid=true`, `matches_instruction=true`, and a score at or above the
  configured threshold.
- Critic output is validated both before each CompositeCritic aggregation and
  at the orchestration boundary. Concrete booleans, finite `0..10` scores,
  string issue/suggestion lists, and string provenance fields are required
  before feedback can affect aggregation, revision, or acceptance.
- The CLI prints the final `outcome` and `stop_reason`; evaluation reads the
  same fields without deriving acceptance from SVG validity.
- Schema version 1 readers validate and expose typed attempts and model-call
  traces. Legacy sidecars without `schema_version` remain readable as version 0.
- A complete immutable run bundle is promoted before the JSON sidecar is
  validated with the version 1 reader and atomically published. The sidecar is
  the commit marker; the exported SVG is updated only afterward and is not
  used as the canonical evaluation input.
- Writer and strict reader both require the canonical SVG to match the final
  successful attempt. If an error occurs after the sidecar replacement, the
  committed bundle is retained so the sidecar never references deleted files.
- Strict readers reject `accepted` Critic runs unless the final feedback targets
  the final attempt, satisfies both structural acceptance booleans, and meets
  the configured score threshold.
- Publication is serialized by sidecar path across threads and processes, so
  concurrent runs targeting one explicit `--output` cannot interleave sidecar
  and export-alias updates.

## SVG Safety Boundary

The validator accepts a reviewed allowlist of static SVG drawing, text,
gradient, masking, and filter elements. Script, style blocks, animation,
navigation, media, handlers, foreign namespaces, event attributes, DTDs,
processing instructions, absolute/external references, and CSS obfuscation are
rejected. Attribute namespaces are also deny-by-default; only reviewed
`xml:lang`, `xml:space`, and local-fragment `xlink:href` use is allowed. New SVG
elements or namespaced attributes must be reviewed before being added. Element
names are matched with the exact SVG/XML casing; invalid variants such as
`lineargradient` or `Rect` must not be normalized into allowed elements.
URI checks remove URL-parser-ignored ASCII tab and newline characters before
scheme detection, while ordinary multiline geometry attributes remain valid.

## Verification

The integration path should stay covered by:

- `tests/test_artifacts.py`
- `tests/test_generate_pipeline.py`
- `tests/test_render_cli.py`
- `tests/test_svg_renderer.py`
- existing orchestrator tests
- existing CLI import tests

Whenever the command workflow changes, update the tests before merging.
