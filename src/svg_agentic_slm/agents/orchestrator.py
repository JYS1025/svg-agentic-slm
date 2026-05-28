"""SVG generation orchestrator.

Coordinates the end-to-end SVG generation pipeline including
RAG retrieval, generation, validation, rendering, and critique.

All dependencies are injected through the constructor, making
the orchestrator testable and backend-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from svg_agentic_slm.agents.schemas import GenerationRequest, GenerationResult

if TYPE_CHECKING:
    from svg_agentic_slm.agents.base import BaseCritic, BaseGenerator
    from svg_agentic_slm.agents.rag_agent import RAGAgent
    from svg_agentic_slm.svg.base import BaseRenderer, BaseValidator

logger = logging.getLogger(__name__)


class SVGGenerationOrchestrator:
    """Orchestrates the SVG generation pipeline.

    Coordinates the following steps:
    1. Receive user prompt / GenerationRequest.
    2. (Optional) Retrieve similar examples via RAG.
    3. Call the generator agent to produce SVG.
    4. Validate the generated SVG.
    5. (Optional) Render SVG to raster image.
    6. (Optional) Call critic for feedback.
    7. (Optional) Revise based on critic feedback.
    8. Return GenerationResult.

    All components are injected, not instantiated internally.

    Args:
        generator: The SVG generator agent.
        validator: The SVG validator.
        renderer: The SVG renderer (optional).
        critic: The SVG critic agent (optional).
        rag_agent: The RAG retrieval agent (optional).
        max_revisions: Maximum number of critic-driven revision rounds.
        output_dir: Directory for saving outputs.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        validator: BaseValidator,
        renderer: BaseRenderer | None = None,
        critic: BaseCritic | None = None,
        rag_agent: RAGAgent | None = None,
        max_revisions: int = 2,
        output_dir: str | Path = "./outputs/generations",
    ) -> None:
        self._generator = generator
        self._validator = validator
        self._renderer = renderer
        self._critic = critic
        self._rag_agent = rag_agent
        self._max_revisions = max_revisions
        self._output_dir = Path(output_dir)

    def run(self, request: GenerationRequest) -> GenerationResult:
        """Run the full SVG generation pipeline.

        Args:
            request: The generation request.

        Returns:
            The generation result with SVG, validation, and feedback.

        TODO: Implement the full pipeline with RAG, critic, and revisions.
        TODO: Add timing/latency tracking.
        TODO: Add output saving.
        """
        logger.info("Starting generation pipeline for: %s", request.instruction[:80])

        result = GenerationResult(instruction=request.instruction)

        # Step 1: RAG retrieval (optional)
        context: str | None = None
        if self._rag_agent is not None:
            examples = self._rag_agent.retrieve(request.instruction)
            context = self._rag_agent.format_context(examples)
            logger.info("Retrieved %d RAG examples.", len(examples))

        # Step 2: Generate SVG
        generated_svg = self._generator.generate(request, context=context)
        result.generated_svg = generated_svg

        # Step 3: Validate
        validation = self._validator.validate(generated_svg)
        result.is_valid = validation.is_valid
        logger.info("Validation result: valid=%s, errors=%s", validation.is_valid, validation.errors)

        # Step 4: Render (optional)
        if self._renderer is not None:
            # TODO: Generate unique output filename
            # render_path = self._output_dir / "render.png"
            # render_result = self._renderer.render(generated_svg, render_path)
            # result.render_path = str(render_result.output_path) if render_result.success else None
            logger.info("[PLACEHOLDER] Would render SVG.")

        # Step 5: Critic feedback (optional)
        if self._critic is not None:
            feedback = self._critic.critique(request.instruction, generated_svg)
            result.critic_feedback.append(feedback)
            logger.info("Critic feedback: score=%.1f", feedback.score)

            # TODO: Implement revision loop
            # for revision in range(self._max_revisions):
            #     if feedback.score >= 8.0:
            #         break
            #     revised_svg = self._generator.generate_revision(...)
            #     ...

        logger.info("Pipeline complete. Valid=%s, Revisions=%d", result.is_valid, result.revision_count)
        return result
