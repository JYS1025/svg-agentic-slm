"""Prompt templates for text-to-SVG generation.

Contains functions to build formatted prompts for the text-to-SVG
generation task, including optional RAG context injection.

# TODO: Add prompt versioning and template selection based on config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svg_agentic_slm.rag.schemas import RetrievedExample


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

    # Add few-shot examples from RAG if available
    if retrieved_examples:
        parts.append("Here are some similar SVG examples for reference:\n")
        for i, example in enumerate(retrieved_examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Description: {example.description}")
            parts.append(f"SVG: {example.content}")
            parts.append("")

    parts.append(f"Now generate an SVG for the following description:\n{instruction}")

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
        "Please generate an improved SVG that addresses the feedback above."
    )
