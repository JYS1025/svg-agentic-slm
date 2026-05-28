"""System prompt templates.

Defines the system-level instructions given to the language model
to establish its role and constraints.

# TODO: Add prompt versioning and A/B testing support.
"""

from __future__ import annotations


def get_svg_generator_system_prompt() -> str:
    """Return the system prompt for the SVG generator agent.

    This prompt establishes the model's role as an SVG code generator
    and sets constraints on the output format.
    """
    return (
        "You are an expert SVG code generator. Your task is to generate "
        "valid, well-structured SVG code based on natural language descriptions.\n\n"
        "Rules:\n"
        "1. Output ONLY valid SVG code — no explanations, no markdown.\n"
        "2. Always include the xmlns attribute in the root <svg> element.\n"
        "3. Use a viewBox of '0 0 256 256' unless otherwise specified.\n"
        "4. Keep the SVG simple, clean, and visually accurate to the description.\n"
        "5. Do not include embedded scripts or external references.\n"
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
