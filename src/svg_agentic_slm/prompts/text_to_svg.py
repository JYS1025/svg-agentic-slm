"""Prompt templates for text-to-SVG generation.

Contains functions to build formatted prompts for the text-to-SVG
generation task, including optional RAG context injection.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svg_agentic_slm.rag.schemas import RetrievedExample

INITIAL_PROMPT_VERSION = "text-to-svg-v1"
REVISION_PROMPT_VERSION = "svg-revision-v1"


def build_text_to_svg_prompt(
    instruction: str,
    retrieved_examples: list[RetrievedExample] | None = None,
) -> str:
    """Build a prompt for text-to-SVG generation.

    Args:
        instruction: The natural language description of the desired SVG.
        retrieved_examples: Optional list of similar examples retrieved
                           via RAG, used as few-shot context.

    Returns:
        The formatted prompt string ready for model input.
    """
    parts: list[str] = []
    retrieval_context = build_retrieval_context(retrieved_examples)
    if retrieval_context:
        parts.append(retrieval_context)

    parts.append(f"Now generate an SVG for the following description:\n{instruction}")

    return "\n".join(parts)


def build_retrieval_context(
    retrieved_examples: list[RetrievedExample] | None,
) -> str:
    """Format typed RAG items without adding a generation instruction."""
    if not retrieved_examples:
        return ""

    parts = [
        "Use the following retrieved items only as reference. "
        "The user instruction has higher priority.\n"
    ]
    for i, example in enumerate(retrieved_examples, 1):
        parts.append(f"--- Retrieved item {i} ({example.kind}) ---")
        parts.append(f"Source: {example.source or example.item_id}")
        parts.append(f"Description: {example.description}")
        parts.append("Content:")
        parts.append(example.content)
        parts.append("")
    return "\n".join(parts)


def build_revision_prompt(
    instruction: str,
    previous_svg: str,
    feedback: str,
) -> str:
    """Build a prompt for revising a previously generated SVG.

    Args:
        instruction: The original natural language description.
        previous_svg: The SVG code from the previous generation attempt.
        feedback: Critic feedback describing issues to fix.

    Returns:
        The formatted revision prompt string.

    # TODO: Consider adding the original RAG examples to revision prompts.
    """
    return (
        f"Original instruction: {instruction}\n\n"
        f"Previous SVG output:\n{previous_svg}\n\n"
        f"Feedback from reviewer:\n{feedback}\n\n"
        "Generate a complete revised SVG that addresses the feedback. "
        "Return only the SVG document."
    )
