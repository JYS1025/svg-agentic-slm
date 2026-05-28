"""Prompt templates for critic agents.

Contains functions to build prompts for SVG critique tasks,
including both LLM-based and structured evaluation prompts.

# TODO: Add prompt versioning for critic prompt evolution tracking.
"""

from __future__ import annotations


def build_critic_prompt(
    instruction: str,
    svg_code: str,
) -> str:
    """Build a prompt for the LLM critic to evaluate an SVG.

    Args:
        instruction: The original natural language description.
        svg_code: The generated SVG code to evaluate.

    Returns:
        The formatted critic prompt.
    """
    return (
        f"Evaluate the following SVG against the given instruction.\n\n"
        f"Instruction: {instruction}\n\n"
        f"SVG Code:\n{svg_code}\n\n"
        "Provide your evaluation in the following format:\n"
        "Score: [1-10]\n"
        "Valid: [yes/no]\n"
        "Matches instruction: [yes/no/partially]\n"
        "Issues: [list any issues]\n"
        "Suggestions: [list improvement suggestions]\n"
    )


def build_comparison_prompt(
    instruction: str,
    svg_a: str,
    svg_b: str,
) -> str:
    """Build a prompt for comparing two SVG outputs.

    Args:
        instruction: The original instruction both SVGs were generated from.
        svg_a: First SVG code.
        svg_b: Second SVG code.

    Returns:
        The formatted comparison prompt.

    # TODO: Implement when needed for A/B evaluation experiments.
    """
    return (
        f"Compare the following two SVGs generated for the same instruction.\n\n"
        f"Instruction: {instruction}\n\n"
        f"SVG A:\n{svg_a}\n\n"
        f"SVG B:\n{svg_b}\n\n"
        "Which SVG better matches the instruction? Explain your reasoning.\n"
    )
