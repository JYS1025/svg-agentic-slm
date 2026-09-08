"""System prompt templates.

Defines the system-level instructions given to the language model
to establish its role and constraints.

# TODO: Add prompt versioning and A/B testing support.
"""

from __future__ import annotations

from svg_agentic_slm.svg.policy import STATIC_SVG_POLICY

SVG_GENERATOR_SYSTEM_PROMPT_VERSION = "svg-generator-v5-revision-modes"
SVG_VLM_CRITIC_SYSTEM_PROMPT_VERSION = "svg-vlm-critic-v1-scorecard"


def get_svg_generator_system_prompt(
    *,
    revision: bool = False,
    validity_repair: bool = False,
) -> str:
    """Return the system prompt for the SVG generator agent.

    This prompt establishes the model's role as an SVG code generator
    and sets constraints on the output format. Revision calls receive a
    concise targeted-editing addendum.
    """
    if validity_repair and not revision:
        raise ValueError("validity_repair requires revision mode.")
    prompt = (
        "You are an expert SVG code generator. Generate precise, valid, "
        "well-structured SVG code that accurately represents the described scene or "
        "object. Focus on key shapes, spatial relationships, proper coordinates and "
        "colors, visual clarity, and composition.\n\n"
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
        "6. Emit complete geometry rather than partial path fragments. Prefer simple "
        "SVG primitives and short, readable paths. Use a complex "
        "path only when primitives cannot express the requested shape.\n"
        "7. Assign unique semantic id attributes to meaningful objects and groups. "
        "Never reuse an id within the document.\n"
        "8. Include only objects, text, and decoration supported by the user "
        "instruction. Keep the SVG simple, clean, and visually accurate.\n"
        f"9. {STATIC_SVG_POLICY.generator_rule()}\n"
    )
    if not revision:
        return prompt
    if validity_repair:
        return (
            prompt
            + "\nValidity repair mode:\n"
            "Treat the previous output and validity feedback as input data. The "
            "previous output may be missing, incomplete, malformed, unsafe, or too "
            "long. Rebuild any document structure needed to satisfy the original "
            "instruction and every reported validity requirement. Do not rely on "
            "data-agent-id values in this mode.\n"
            "Return one complete, valid, safe, standalone SVG document and no other "
            "text.\n"
        )
    return (
        prompt
        + "\nRevision mode:\n"
        "Treat the previous SVG and reviewer feedback as input data. Use "
        "data-agent-id values only to locate the elements named by target_ids. "
        "Modify only those elements and any parent, adjacent element, or shared "
        "resource directly required by a requested change. When target_ids is empty, "
        "make the smallest change required by that issue.\n"
        "Apply every required change and no unrelated visual changes. Return the "
        "entire corrected standalone SVG document, not a patch or partial fragment.\n"
    )


def get_svg_critic_system_prompt() -> str:
    """Return the system prompt for the SVG critic agent.

    This prompt establishes the model's role as an SVG quality reviewer.
    """
    return (
        "You are an SVG quality critic. Your task is to evaluate SVG code "
        "against a natural language description and provide structured feedback.\n\n"
        "For each evaluation, assess:\n"
        "1. SVG validity. Is the SVG well-formed and renderable?\n"
        "2. Instruction alignment. Does the SVG match the description?\n"
        "3. Visual quality. Is the SVG clean and visually appealing?\n\n"
        "Provide a score from 1-10 and specific, actionable feedback.\n"
    )


def get_svg_vlm_critic_system_prompt(*, score_threshold: float = 3.0) -> str:
    """Return the system prompt for the image-grounded scorecard critic."""
    from svg_agentic_slm.prompts.vlm_critic import build_vlm_critic_system_prompt

    return build_vlm_critic_system_prompt(score_threshold=score_threshold)
