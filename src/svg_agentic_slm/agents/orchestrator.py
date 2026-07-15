"""SVG generation orchestrator.

Coordinates the end-to-end SVG generation pipeline including
RAG retrieval, generation, validation, rendering, and critique.

All dependencies are injected through the constructor, making
the orchestrator testable and backend-agnostic.
"""

from __future__ import annotations

import logging
import time
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
        render_output_path: str | Path | None = None,
        render_width: int = 256,
        render_height: int = 256,
        render_format: str = "png",
    ) -> None:
        self._generator = generator
        self._validator = validator
        self._renderer = renderer
        self._critic = critic
        self._rag_agent = rag_agent
        self._max_revisions = max_revisions
        self._output_dir = Path(output_dir)
        self._render_output_path = Path(render_output_path) if render_output_path else None
        self._render_width = render_width
        self._render_height = render_height
        self._render_format = render_format

    def run(self, request: GenerationRequest) -> GenerationResult:
        """Run the full SVG generation pipeline.

        Args:
            request: The generation request.

        Returns:
            The generation result with SVG, validation, and feedback.

        TODO: Implement critic-driven revisions when the generator contract supports them.
        """
        logger.info("Starting generation pipeline for: %s", request.instruction[:80])
        started_at = time.perf_counter()

        result = GenerationResult(instruction=request.instruction)
        result.metadata["request"] = {
            "task": request.task,
            "config_overrides": request.config_overrides,
        }

        # Step 1: RAG retrieval (optional)
        context: str | None = None
        retrieved_count = 0
        if self._rag_agent is not None:
            examples = self._rag_agent.retrieve(request.instruction)
            context = self._rag_agent.format_context(examples)
            retrieved_count = len(examples)
            logger.info("Retrieved %d RAG examples.", len(examples))
        result.metadata["rag"] = {
            "enabled": self._rag_agent is not None,
            "retrieved_examples": retrieved_count,
        }

        # Step 2: Generate SVG
        generated_svg = self._generator.generate(request, context=context)
        result.generated_svg = generated_svg

        # Step 3: Validate
        validation = self._validator.validate(generated_svg)
        result.is_valid = validation.is_valid
        logger.info("Validation result: valid=%s, errors=%s", validation.is_valid, validation.errors)
        result.metadata["validation"] = {
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "has_svg_tag": validation.has_svg_tag,
            "has_closing_tag": validation.has_closing_tag,
            "is_well_formed_xml": validation.is_well_formed_xml,
        }

        # Step 4: Render (optional)
        render_success = False
        render_error: str | None = None
        if self._renderer is not None:
            render_path = self._render_output_path or (
                self._output_dir / "render.png"
            )
            render_result = self._renderer.render(
                generated_svg,
                render_path,
                width=self._render_width,
                height=self._render_height,
                output_format=self._render_format,
            )
            render_success = render_result.success
            render_error = render_result.error
            if render_result.success and render_result.output_path is not None:
                result.render_path = str(render_result.output_path)
            logger.info(
                "Render result: success=%s, path=%s, error=%s",
                render_result.success,
                render_result.output_path,
                render_result.error,
            )
        result.metadata["render"] = {
            "enabled": self._renderer is not None,
            "render_path": result.render_path,
            "planned_output_path": str(self._render_output_path) if self._render_output_path else None,
            "success": render_success,
            "error": render_error,
            "format": self._render_format,
            "width": self._render_width,
            "height": self._render_height,
        }

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
        result.metadata["critic"] = {
            "enabled": self._critic is not None,
            "feedback_count": len(result.critic_feedback),
        }

        result.metadata["timing"] = {
            "generation_latency_seconds": round(time.perf_counter() - started_at, 6),
        }

        logger.info("Pipeline complete. Valid=%s, Revisions=%d", result.is_valid, result.revision_count)
        return result
