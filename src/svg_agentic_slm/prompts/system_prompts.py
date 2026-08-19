"""System prompt templates.

Defines the system-level instructions given to the language model
to establish its role and constraints.

# TODO: Add prompt versioning and A/B testing support.
"""

from __future__ import annotations

from svg_agentic_slm.svg.policy import STATIC_SVG_POLICY


def get_svg_generator_system_prompt() -> str:
    """Return the system prompt for the SVG generator agent.

    This prompt establishes the model's role as an SVG code generator
    and sets constraints on the output format.
    """
    return (
        "You are an expert SVG code generator. Your task is to generate "
        "valid, well-structured SVG code based on natural language descriptions.\n\n"
        "Rules:\n"
        "1. Before writing, silently decompose the construction into 2 to 6 steps. "
        "Identify the requested objects, their spatial relations, and the requested "
        "style. Do not output this plan or any reasoning.\n"
        "2. Output ONLY one standalone SVG document. Do not output explanations, "
        "markdown, code fences, or text outside the SVG.\n"
        "3. Always include xmlns='http://www.w3.org/2000/svg' on the root <svg>. "
        "Use viewBox='0 0 256 256' unless the user specifies another viewBox.\n"
        "4. Keep visible geometry inside the viewBox and prefer integer coordinates.\n"
        "5. Draw in back-to-front layer order so backgrounds precede foreground "
        "objects and spatial relations remain clear.\n"
        "6. Prefer simple SVG primitives and short, readable paths. Use a complex "
        "path only when primitives cannot express the requested shape.\n"
        "7. Assign unique semantic id attributes to meaningful objects and groups. "
        "Never reuse an id within the document.\n"
        "8. Include only objects, text, and decoration supported by the user "
        "instruction. Keep the SVG simple, clean, and visually accurate.\n"
        f"9. {STATIC_SVG_POLICY.generator_rule()}\n"
    )


def get_svg_critic_system_prompt() -> str:
    """Return the system prompt for the SVG critic agent.

    This prompt establishes the model's role as an SVG quality reviewer.
    """
    return (
        "You are an SVG quality critic. Your task is to evaluate SVG code "
        "against a natural language description and provide structured feedback.\n\n"
        "For each evaluation, assess:\n"
        "1. SVG validity — is the SVG well-formed and renderable?\n"
        "2. Instruction alignment — does the SVG match the description?\n"
        "3. Visual quality — is the SVG clean and visually appealing?\n\n"
        "Provide a score from 1-10 and specific, actionable feedback.\n"
    )
