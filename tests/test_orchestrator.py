"""Tests for the SVG generation orchestrator.

Verifies that the orchestrator can be instantiated with
stub/mock components and that the dependency injection
pattern works correctly.
"""

from __future__ import annotations

import hashlib

import pytest

from svg_agentic_slm.agents.base import BaseCritic, BaseGenerator
from svg_agentic_slm.agents.orchestrator import SVGGenerationOrchestrator
from svg_agentic_slm.agents.schemas import (
    CRITIC_ISSUE_TYPES,
    CriticEvaluation,
    CriticFeedback,
    CriticFeedbackEvent,
    CriticIssue,
    CriticTraceError,
    GenerationRequest,
    GenerationResult,
    GeneratorOutput,
)
from svg_agentic_slm.models.image_text_similarity import (
    ImageTextSimilarityEvidence,
)
from svg_agentic_slm.svg.base import BaseValidator
from svg_agentic_slm.svg.gates import SmokeRenderResult
from svg_agentic_slm.svg.schemas import SVGValidationResult


class StubGenerator(BaseGenerator):
    """Stub generator for testing."""

    @property
    def name(self) -> str:
        return "StubGenerator"

    def generate(
        self,
        request: GenerationRequest,
        context: str | None = None,
    ) -> str:
        return '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'


class StubValidator(BaseValidator):
    """Stub validator for testing."""

    def validate(self, svg_content: str) -> SVGValidationResult:
        return SVGValidationResult(is_valid=True, has_svg_tag=True, has_closing_tag=True)


class StubCritic(BaseCritic):
    """Stub critic for testing."""

    @property
    def name(self) -> str:
        return "StubCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(
            score=8.0,
            is_valid=True,
            matches_instruction=True,
            critic_type="stub",
        )


class RevisingGenerator(BaseGenerator):
    """Typed Generator that records the feedback correlation it receives."""

    def __init__(self) -> None:
        self.revision_feedback: CriticFeedbackEvent | None = None

    @property
    def name(self) -> str:
        return "RevisingGenerator"

    def generate(
        self,
        request: GenerationRequest,
        context=None,
    ) -> GeneratorOutput:
        return GeneratorOutput(
            attempt_id="attempt-initial",
            mode="initial",
            svg='<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
            raw_output="<svg><rect/></svg>",
            status="succeeded",
            prompt_version="test-v1",
        )

    def revise(
        self,
        request: GenerationRequest,
        previous: GeneratorOutput,
        feedback: CriticFeedbackEvent,
        context=None,
    ) -> GeneratorOutput:
        self.revision_feedback = feedback
        return GeneratorOutput(
            attempt_id="attempt-revised",
            mode="revision",
            svg='<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
            raw_output="<svg><circle/></svg>",
            status="succeeded",
            prompt_version="test-revision-v1",
            parent_attempt_id=previous.attempt_id,
            trigger_feedback_id=feedback.feedback_id,
        )


class SequentialCritic(BaseCritic):
    """Request one revision and then accept it."""

    def __init__(self) -> None:
        self.scores = iter([4.0, 9.0])

    @property
    def name(self) -> str:
        return "SequentialCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(
            score=next(self.scores),
            is_valid=True,
            matches_instruction=True,
            critic_type="sequential",
        )


class SequentialValidator(BaseValidator):
    """Reject the initial attempt, then accept the revision."""

    def __init__(self) -> None:
        self.results = iter([False, True])

    def validate(self, svg_content: str) -> SVGValidationResult:
        is_valid = next(self.results)
        return SVGValidationResult(
            is_valid=is_valid,
            has_svg_tag=True,
            has_closing_tag=True,
            is_well_formed_xml=is_valid,
            errors=[] if is_valid else ["invalid initial attempt"],
        )


class SequentialStructuredCritic(BaseCritic):
    """Reject with structured fields despite a high score, then accept."""

    def __init__(self) -> None:
        self.feedback = iter(
            [
                CriticFeedback(
                    score=9.0,
                    is_valid=True,
                    matches_instruction=False,
                    critic_type="structured",
                ),
                CriticFeedback(
                    score=9.0,
                    is_valid=True,
                    matches_instruction=True,
                    critic_type="structured",
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "SequentialStructuredCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return next(self.feedback)


class HighScoreInvalidCritic(BaseCritic):
    """Return a score above threshold while explicitly marking output invalid."""

    @property
    def name(self) -> str:
        return "HighScoreInvalidCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(
            score=9.0,
            is_valid=False,
            matches_instruction=True,
            critic_type="structured",
        )


class MalformedBooleanCritic(BaseCritic):
    """Violate the runtime contract despite matching the dataclass shape."""

    @property
    def name(self) -> str:
        return "MalformedBooleanCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(
            score=9.0,
            is_valid="false",  # type: ignore[arg-type]
            matches_instruction="false",  # type: ignore[arg-type]
            critic_type="malformed",
        )


class ScoreSequenceCritic(BaseCritic):
    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)

    @property
    def name(self) -> str:
        return "ScoreSequenceCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(
            score=next(self._scores),
            is_valid=True,
            matches_instruction=False,
            critic_type="score-sequence",
        )


class ContractFailingCritic(BaseCritic):
    @property
    def name(self) -> str:
        return "ContractFailingCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        raise CriticTraceError("critic response contract failed", [])


class RevisionContractFailingCritic(BaseCritic):
    def __init__(self) -> None:
        self._calls = 0

    @property
    def name(self) -> str:
        return "RevisionContractFailingCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        self._calls += 1
        if self._calls == 2:
            raise CriticTraceError("revision critic contract failed", [])
        return CriticFeedback(
            score=5.0,
            is_valid=True,
            matches_instruction=False,
            critic_type="contract-sequence",
        )


def _scorecard_feedback(
    *,
    scale_score: int,
    threshold: float = 3.0,
) -> CriticFeedback:
    evaluations = [
        CriticEvaluation(
            category=category,  # type: ignore[arg-type]
            type=issue_type,
            applicable=True,
            score=scale_score if (category, issue_type) == ("layout", "scale") else 4,
            reason="Visible scorecard assessment.",
        )
        for category, issue_types in CRITIC_ISSUE_TYPES.items()
        for issue_type in issue_types
    ]
    passing = scale_score >= threshold
    issues = [] if passing else [
        CriticIssue(
            category="layout",
            type="scale",
            scope="global",
            target_ids=[],
            observed="The complete object is too small.",
            expected="The object should occupy most of the canvas.",
            fix="Increase the complete object scale.",
        )
    ]
    return CriticFeedback(
        score=float(min(item.score for item in evaluations if item.score is not None)),
        is_valid=True,
        matches_instruction=passing,
        issues=[item.observed for item in issues],
        suggestions=[item.fix for item in issues],
        critic_type="scorecard",
        status="pass" if passing else "revise",
        evaluations=evaluations,
        structured_issues=issues,
        schema_version=3,
        metadata={"score_threshold": threshold},
    )


class SequentialScorecardCritic(BaseCritic):
    def __init__(self, scale_scores: list[int]) -> None:
        self._scale_scores = iter(scale_scores)

    @property
    def name(self) -> str:
        return "SequentialScorecardCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return _scorecard_feedback(scale_score=next(self._scale_scores))


def test_orchestrator_instantiation() -> None:
    """Test that the orchestrator can be created with stub components."""
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
    )
    assert orchestrator is not None


def test_orchestrator_with_all_components() -> None:
    """Test orchestrator with all optional components."""
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=StubCritic(),
    )
    assert orchestrator is not None


def test_orchestrator_run() -> None:
    """Test that the orchestrator can run a basic pipeline."""
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=StubCritic(),
    )

    request = GenerationRequest(instruction="Draw a red square.")
    result = orchestrator.run(request)

    assert isinstance(result, GenerationResult)
    assert result.instruction == "Draw a red square."
    assert result.generated_svg != ""
    assert result.is_valid


def test_orchestrator_correlates_feedback_and_revision() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=StubValidator(),
        critic=SequentialCritic(),
        max_revisions=2,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 1
    assert len(result.attempts) == 2
    assert len(result.feedback_events) == 2
    assert generator.revision_feedback is result.feedback_events[0]
    assert result.feedback_events[0].target_attempt_id == "attempt-initial"
    assert result.attempts[1].parent_attempt_id == "attempt-initial"
    assert result.attempts[1].metadata["outcome"] == "accepted"


def test_scorecard_accepts_only_when_every_applicable_score_meets_threshold() -> None:
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=SequentialScorecardCritic([3]),
        critic_score_threshold=3.0,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a square."))

    assert result.revision_count == 0
    assert result.feedback_events[0].feedback.status == "pass"
    assert result.attempts[0].metadata["outcome"] == "accepted"


def test_scorecard_revises_until_all_applicable_scores_meet_threshold() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=StubValidator(),
        critic=SequentialScorecardCritic([2, 3]),
        critic_score_threshold=3.0,
        max_revisions=1,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 1
    assert [event.feedback.status for event in result.feedback_events] == ["revise", "pass"]
    assert result.attempts[-1].metadata["outcome"] == "accepted"


def test_visual_critic_is_not_called_when_svg_validity_gate_fails() -> None:
    class InvalidValidator(BaseValidator):
        def validate(self, svg_content: str) -> SVGValidationResult:
            return SVGValidationResult(
                is_valid=False,
                has_svg_tag=True,
                has_closing_tag=True,
                is_well_formed_xml=False,
                errors=["Malformed SVG."],
            )

    class FailIfCalledCritic(BaseCritic):
        def __init__(self) -> None:
            self.calls = 0

        @property
        def name(self) -> str:
            return "FailIfCalledCritic"

        def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
            self.calls += 1
            raise AssertionError("The VLM Critic must not run after failed validity.")

    critic = FailIfCalledCritic()
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=InvalidValidator(),
        critic=critic,
        require_visual_evidence=True,
        max_revisions=0,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a square."))

    assert critic.calls == 0
    assert result.feedback_events[0].feedback.status == "invalid"
    assert result.feedback_events[0].feedback.critic_type == "critic_evidence_gate"


def test_visual_critic_receives_siglip2_evidence_for_the_exact_smoke_render() -> None:
    png = b"\x89PNG\r\n\x1a\nexact-smoke-render"

    class RecordingSmokeRenderGate:
        width = 320
        height = 240

        def evaluate(self, svg: str) -> SmokeRenderResult:
            assert svg.startswith("<svg")
            return SmokeRenderResult(
                success=True,
                png=png,
                renderer_version="test-renderer",
            )

    class RecordingSimilarityScorer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bytes, str]] = []

        def score(
            self,
            instruction: str,
            image_png: bytes,
            *,
            attempt_id: str,
        ) -> ImageTextSimilarityEvidence:
            self.calls.append((instruction, image_png, attempt_id))
            return ImageTextSimilarityEvidence(
                attempt_id=attempt_id,
                metric="siglip2_pair_probability",
                score=0.75,
                raw_logit=1.0986123,
                model_id="google/siglip2-base-patch16-224",
                model_revision="a" * 40,
                text_template="This is a photo of {instruction}.",
                text_input=f"This is a photo of {instruction}.",
                image_sha256=hashlib.sha256(image_png).hexdigest(),
                device="cuda:1",
                dtype="bfloat16",
                latency_seconds=0.125,
            )

    class RecordingVisualCritic(BaseCritic):
        def __init__(self) -> None:
            self.input = None

        @property
        def name(self) -> str:
            return "RecordingVisualCritic"

        def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
            raise AssertionError("The visual critic must receive CriticInput.")

        def critique_attempt(self, value) -> CriticFeedback:
            self.input = value
            return _scorecard_feedback(scale_score=4)

    scorer = RecordingSimilarityScorer()
    critic = RecordingVisualCritic()
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=critic,
        similarity_scorer=scorer,  # type: ignore[arg-type]
        smoke_render_gate=RecordingSmokeRenderGate(),  # type: ignore[arg-type]
        require_visual_evidence=True,
        max_revisions=0,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a square."))

    attempt = result.attempts[0]
    assert scorer.calls == [("Draw a square.", png, attempt.attempt_id)]
    assert critic.input.render_png is png
    assert critic.input.similarity_evidence is attempt.critic_evidence.similarity_evidence
    provenance = result.feedback_events[0].feedback.metadata["evidence_provenance"][0]
    assert provenance["similarity_evidence"]["score"] == 0.75
    assert provenance["similarity_evidence"]["image_sha256"] == hashlib.sha256(png).hexdigest()
    assert attempt.metadata["timing"]["similarity_latency_seconds"] == 0.125
    assert result.metadata["timing"]["similarity_latency_seconds"] >= 0.125


def test_generator_output_failure_receives_feedback_and_validity_revision() -> None:
    class InitialFailureGenerator(BaseGenerator):
        def __init__(self) -> None:
            self.revision_feedback: CriticFeedbackEvent | None = None

        @property
        def name(self) -> str:
            return "InitialFailureGenerator"

        def generate(self, request: GenerationRequest, context=None) -> GeneratorOutput:
            return GeneratorOutput(
                attempt_id="attempt-failed",
                mode="initial",
                svg="",
                raw_output="No SVG was produced.",
                status="failed",
                prompt_version="test-initial-failure",
                error="svg_extraction_failed",
            )

        def revise(
            self,
            request: GenerationRequest,
            previous: GeneratorOutput,
            feedback: CriticFeedbackEvent,
            context=None,
        ) -> GeneratorOutput:
            self.revision_feedback = feedback
            return GeneratorOutput(
                attempt_id="attempt-recovered",
                mode="revision",
                svg='<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
                raw_output="<svg><circle/></svg>",
                status="succeeded",
                prompt_version="test-validity-revision",
                parent_attempt_id=previous.attempt_id,
                trigger_feedback_id=feedback.feedback_id,
            )

    class CountingCritic(StubCritic):
        def __init__(self) -> None:
            self.calls = 0

        def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
            self.calls += 1
            return super().critique(instruction, svg_content)

    generator = InitialFailureGenerator()
    critic = CountingCritic()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=SequentialValidator(),
        critic=critic,
        max_revisions=1,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 1
    assert result.generated_svg == result.attempts[-1].svg
    assert result.feedback_events[0].feedback.status == "invalid"
    assert result.feedback_events[0].feedback.metadata["evidence_provenance"][0][
        "stage"
    ] == "generator_output_failure"
    assert generator.revision_feedback is result.feedback_events[0]
    assert critic.calls == 1


def test_orchestrator_revises_invalid_svg_even_when_critic_score_is_high() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=SequentialValidator(),
        critic=StubCritic(),
        max_revisions=1,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a valid circle."))

    assert result.revision_count == 1
    assert result.is_valid is True
    assert result.attempts[0].metadata["outcome"] == "rejected"
    assert result.attempts[-1].metadata["outcome"] == "accepted"


def test_orchestrator_does_not_render_an_invalid_svg() -> None:
    class FailIfCalledRenderer:
        called = False

        def render(self, *args, **kwargs):
            self.called = True
            raise AssertionError("renderer must not receive an invalid SVG")

    renderer = FailIfCalledRenderer()
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=SequentialValidator(),
        renderer=renderer,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a valid circle."))

    assert result.is_valid is False
    assert renderer.called is False
    assert result.metadata["render"]["success"] is False
    assert result.metadata["render"]["error"] == (
        "Render skipped because SVG validation failed."
    )


def test_orchestrator_honors_structured_critic_rejection() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=StubValidator(),
        critic=SequentialStructuredCritic(),
        max_revisions=1,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 1
    assert result.attempts[0].metadata["outcome"] == "rejected"
    assert result.attempts[-1].metadata["outcome"] == "accepted"


def test_orchestrator_does_not_accept_critic_marked_invalid() -> None:
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=HighScoreInvalidCritic(),
        max_revisions=0,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 0
    assert result.is_valid is True
    assert result.attempts[-1].metadata["outcome"] == "rejected"
    assert result.attempts[-1].metadata["stop_reason"] == "max_revisions_reached"


def test_orchestrator_rejects_malformed_critic_boolean_fields() -> None:
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=MalformedBooleanCritic(),
    )

    with pytest.raises(TypeError, match="is_valid must be a boolean"):
        orchestrator.run(GenerationRequest(instruction="Draw a circle."))


def test_orchestrator_observer_receives_exact_initial_generator_input() -> None:
    observed: list[tuple[GenerationRequest, list[object]]] = []
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
    )
    request = GenerationRequest(
        instruction="Draw a blue circle.",
        run_id="run-observer",
    )

    orchestrator.run(
        request,
        on_generator_input=lambda value, context: observed.append((value, context)),
    )

    assert len(observed) == 1
    assert observed[0][0] is request
    assert observed[0][1] == []


def test_orchestrator_rolls_back_when_revision_score_regresses() -> None:
    orchestrator = SVGGenerationOrchestrator(
        generator=RevisingGenerator(),
        validator=StubValidator(),
        critic=ScoreSequenceCritic([8.0, 4.0]),
        max_revisions=2,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.generated_svg == result.attempts[0].svg
    assert result.generated_svg != result.attempts[-1].svg
    assert result.metadata["selection"] == {
        "selected_attempt_id": "attempt-initial",
        "last_attempt_id": "attempt-revised",
        "rolled_back": True,
        "best_critic_score": 8.0,
        "no_improvement_rounds": 0,
    }
    assert result.attempts[0].metadata["stop_reason"] == (
        "critic_score_regressed_rollback"
    )
    assert result.attempts[-1].metadata["outcome"] == "rolled_back"


def test_orchestrator_stops_after_configured_no_improvement_rounds() -> None:
    orchestrator = SVGGenerationOrchestrator(
        generator=RevisingGenerator(),
        validator=StubValidator(),
        critic=ScoreSequenceCritic([5.0, 5.05]),
        max_revisions=2,
        max_no_improvement_rounds=1,
        min_critic_score_improvement=0.1,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.generated_svg == result.attempts[0].svg
    assert result.metadata["selection"]["no_improvement_rounds"] == 1
    assert result.attempts[0].metadata["stop_reason"] == (
        "no_critic_score_improvement_rollback"
    )


def test_orchestrator_does_not_revise_on_critic_contract_failure() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=StubValidator(),
        critic=ContractFailingCritic(),
        max_revisions=2,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 0
    assert result.critic_feedback == []
    assert result.feedback_events == []
    assert generator.revision_feedback is None
    assert result.generated_svg == result.attempts[0].svg
    assert result.attempts[0].metadata["outcome"] == "critic_contract_failure"
    assert result.attempts[0].metadata["stop_reason"] == "critic_contract_failure"
    assert result.metadata["critic_contract_failure"]["attempt_id"] == "attempt-initial"
    assert result.metadata["timing"]["pipeline_latency_seconds"] >= 0
    assert result.metadata["timing"]["critic_latency_seconds"] >= 0


def test_revision_contract_failure_rolls_back_without_more_feedback() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=StubValidator(),
        critic=RevisionContractFailingCritic(),
        max_revisions=2,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 1
    assert len(result.feedback_events) == 1
    assert result.generated_svg == result.attempts[0].svg
    assert result.metadata["selection"]["rolled_back"] is True
    assert result.attempts[0].metadata["outcome"] == "selected_best"
    assert result.attempts[-1].metadata["outcome"] == "critic_contract_failure"
    assert result.attempts[-1].metadata["stop_reason"] == "critic_contract_failure"
