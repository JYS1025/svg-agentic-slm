# SVG Safety Boundaries

## Purpose

This document defines the shared SVG safety policy used by generation, RAG,
validation, evaluation, and rendering. Security checks must not be recreated in
backend-specific modules.

## Shared Policy

`svg_agentic_slm.svg.validator.SVGValidator` owns document validation. It uses
a hardened `lxml` parser, a case-sensitive SVG element allowlist, namespace
checks, local-only references, and restricted inline CSS.

`safe_svg_element_names()` is the adapter for callers that also need structural
metadata. It returns normalized element names only after the same shared policy
passes.

- Complete documents use `allow_fragment=False`.
- Local corpus snippets use `allow_fragment=True` and are checked inside a
  temporary SVG root.
- Context-specific size limits remain with their owners. Generation and RAG
  indexing intentionally use different limits.
- Do not add a second element denylist, URL scanner, or XML parser in RAG code.

## Enforcement Points

The policy is enforced at these boundaries:

1. Every initial and revised Generator output is validated by the Orchestrator.
2. Hugging Face SVG rows are validated before Qdrant indexing.
3. Local Chroma fragments are validated before indexing.
4. Retrieved `reference_svg` items are validated by `RAGAgent` before crossing
   into Generator context. Non-SVG experience items are not parsed as SVG.
5. The Orchestrator skips rendering when generation or validation fails.
6. The standalone render command validates before selecting a backend.
7. `CairoSVGRenderer` validates again to protect direct API callers.

Validation failures are fail-closed. They must not be converted into warnings
that allow indexing, prompt injection, or rendering.

## Existing Vector Collections

Code changes do not remove records already stored in Chroma or Qdrant. The
retrieval boundary now filters unsafe records, which provides immediate
protection without deleting remote data.

When rebuilding a production corpus:

1. Index into a new versioned collection.
2. Compare accepted row counts and retrieval smoke tests with the old
   collection.
3. Switch configuration only after verification.
4. Keep the old collection for rollback until the new collection is accepted.
5. Never silently clear a shared remote collection from generation code.

## Compatibility Rules

The following public contracts must remain stable unless coordinated across
owners:

- `SVGValidator.validate(svg) -> SVGValidationResult`
- `validate_svg_for_reference(svg) -> list[str] | None`
- `BaseRetriever.add_documents()` and `.retrieve()`
- `RetrievedExample` and generation artifact schemas

Element names used in RAG search metadata remain lowercase, deduplicated, and
limited to 32 entries.

## Required Tests

Any safety-policy change must cover:

- valid complete SVG documents and local fragments;
- active elements and foreign namespaces;
- incorrect element casing;
- processing instructions and DTD/entity input;
- external, obfuscated, and multiline URL references;
- unsafe retrieved references;
- invalid SVG render suppression in the Orchestrator, CLI, and Renderer;
- the existing generation, artifact, RAG, and evaluation regression suites.
