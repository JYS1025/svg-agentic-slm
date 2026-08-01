"""Tests for the basic LLM critic contract."""

from __future__ import annotations

import pytest

from svg_agentic_slm.agents.llm_critic import LLMCritic
from svg_agentic_slm.models.schemas import ModelResponse


class _RecordingBackend:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.calls.append((prompt, kwargs))
        return ModelResponse(
            text=self.response_text,
            model_id="critic-model",
            model_revision="critic-revision",
        )


def test_llm_critic_calls_model_and_returns_typed_feedback() -> None:
    backend = _RecordingBackend(
        """
        {
          "score": 8.5,
          "is_valid": true,
          "matches_instruction": true,
          "issues": [],
          "suggestions": ["Increase the circle contrast."]
        }
        """
    )

    feedback = LLMCritic(backend).critique(
        "Draw a blue circle.",
        '<svg xmlns="http://www.w3.org/2000/svg"><circle fill="blue"/></svg>',
    )

    assert feedback.score == 8.5
    assert feedback.is_valid is True
    assert feedback.matches_instruction is True
    assert feedback.suggestions == ["Increase the circle contrast."]
    assert feedback.critic_type == "llm"
    assert feedback.critic_version == "llm-critic-v1"
    assert feedback.prompt_version == "critic-json-v1"
    assert feedback.model_id == "critic-model"
    assert feedback.model_revision == "critic-revision"
    assert backend.calls[0][1]["system_prompt"]
    assert "Draw a blue circle." in backend.calls[0][0]


def test_llm_critic_accepts_json_in_markdown_fence() -> None:
    backend = _RecordingBackend(
        """```json
        {
          "score": 4,
          "is_valid": true,
          "matches_instruction": false,
          "issues": ["The requested square is missing."],
          "suggestions": ["Replace the circle with a square."]
        }
        ```"""
    )

    feedback = LLMCritic(backend).critique("Draw a square.", "<svg><circle/></svg>")

    assert feedback.score == 4
    assert feedback.matches_instruction is False


def test_llm_critic_rejects_invalid_json() -> None:
    backend = _RecordingBackend("Score: 8, valid: yes")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        LLMCritic(backend).critique("Draw a circle.", "<svg/>")


def test_llm_critic_rejects_malformed_field_types() -> None:
    backend = _RecordingBackend(
        """
        {
          "score": 8,
          "is_valid": "yes",
          "matches_instruction": true,
          "issues": [],
          "suggestions": []
        }
        """
    )

    with pytest.raises(TypeError, match="is_valid must be a boolean"):
        LLMCritic(backend).critique("Draw a circle.", "<svg/>")
