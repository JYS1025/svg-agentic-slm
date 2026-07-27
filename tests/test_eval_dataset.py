"""Tests for dataset-backed generation and benchmark isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from svg_agentic_slm.agents.schemas import (
    GenerationResult,
    GeneratorOutput,
    ModelCallTrace,
)
from svg_agentic_slm.data.schemas import TextToSVGExample
from svg_agentic_slm.eval.evaluator import Evaluator
from svg_agentic_slm.eval.policy import BenchmarkRunPolicy
from svg_agentic_slm.eval.run_eval import run_evaluation
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.svg.schemas import SVGRenderResult
from svg_agentic_slm.svg.validator import SVGValidator


class _StubOrchestrator:
    def run(self, request) -> GenerationResult:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
        response = ModelResponse(
            text=svg,
            model_id="fake",
            prompt_tokens=10,
            completion_tokens=20,
            latency_seconds=0.5,
            time_to_first_token_seconds=0.1,
            tokens_per_second=50.0,
        )
        attempt = GeneratorOutput(
            attempt_id="attempt",
            mode="initial",
            svg=svg,
            raw_output=svg,
            status="succeeded",
            prompt_version="test",
            model_calls=[ModelCallTrace(model_call_id="call", response=response)],
            metadata={"stop_reason": "generator_only_complete"},
        )
        return GenerationResult(
            instruction=request.instruction,
            generated_svg=svg,
            is_valid=True,
            run_id=request.run_id,
            attempts=[attempt],
            metadata={"timing": {"generation_latency_seconds": 0.6}},
        )


class _StubRenderer:
    def render(
        self,
        svg_content: str,
        output_path: Path,
        width: int = 256,
        height: int = 256,
        output_format: str = "png",
    ) -> SVGRenderResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return SVGRenderResult(success=True, output_path=output_path)


class _FakeModelBackend:
    def load_model(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'
        return ModelResponse(
            text=svg,
            model_id="fake",
            prompt_tokens=5,
            completion_tokens=10,
            latency_seconds=0.2,
            time_to_first_token_seconds=0.05,
            tokens_per_second=40.0,
        )


def test_dataset_evaluator_generates_renders_and_records_performance(tmp_path: Path) -> None:
    evaluator = Evaluator(
        validator=SVGValidator(),
        orchestrator=_StubOrchestrator(),  # type: ignore[arg-type]
        renderer=_StubRenderer(),  # type: ignore[arg-type]
        prediction_output_dir=tmp_path / "predictions",
        render_output_dir=tmp_path / "renders",
    )

    result = evaluator.evaluate(
        [_example("sample:1")],
        metrics=[
            "generation_success_rate",
            "svg_validity_rate",
            "render_success_rate",
            "generation_latency",
            "time_to_first_token",
            "tokens_per_second",
        ],
    )

    assert result.num_samples == 1
    assert result.generation_success_rate == 1.0
    assert result.svg_validity_rate == 1.0
    assert result.render_success_rate == 1.0
    assert result.avg_generation_latency == 0.6
    assert result.avg_time_to_first_token == 0.1
    assert result.avg_tokens_per_second == 50.0
    assert result.metadata["run_policy"]["memory_write_blocked"] is True
    assert Path(result.per_sample_results[0]["svg_path"]).is_file()
    assert Path(result.per_sample_results[0]["render_path"]).is_file()


def test_held_out_policy_rejects_memory_eligible_records() -> None:
    example = _example("sample:unsafe")
    assert example.metadata is not None
    example.metadata["memory_eligible"] = True

    with pytest.raises(ValueError, match="memory_eligible=false"):
        BenchmarkRunPolicy(partition="held_out").validate(example)


def test_run_evaluation_executes_prepared_dataset_once_model_is_built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "svg_agentic_slm.factories.generation._build_model_backend",
        lambda model_config, generation_config: _FakeModelBackend(),
    )
    dataset_path = _write_benchmark_snapshot(tmp_path, [_example("sample:1"), _example("sample:2")])
    report_dir = tmp_path / "reports"
    _write_config_bundle(tmp_path, dataset_path, report_dir)

    result = run_evaluation(tmp_path / "eval.yaml")

    assert result.num_samples == 2
    assert result.metadata["evaluation_mode"] == "dataset"
    assert result.metadata["dataset_source"] == str(dataset_path)
    assert result.metadata["benchmark_manifest"]["benchmark_status"] == "adopted"
    assert result.generation_success_rate == 1.0
    assert result.svg_validity_rate == 1.0
    assert result.render_success_rate == 1.0
    assert result.metadata["acceptance_gate"]["passed"] is True
    assert len(list((report_dir / "predictions").glob("*.svg"))) == 2


def _example(sample_id: str) -> TextToSVGExample:
    return TextToSVGExample(
        task="text_to_svg",
        instruction="Draw a circle.",
        output_svg='<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
        metadata={
            "benchmark_id": "test",
            "sample_id": sample_id,
            "source_split": "easy",
            "difficulty": "easy",
            "source_revision": "a" * 40,
            "data_partition": "held_out_test",
            "memory_eligible": False,
        },
    )


def _write_benchmark_snapshot(
    root: Path, examples: list[TextToSVGExample]
) -> Path:
    dataset_dir = root / "benchmark"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "text_to_svg.jsonl"
    content = "".join(
        json.dumps(example.to_dict(), ensure_ascii=False) + "\n" for example in examples
    )
    dataset_path.write_text(content, encoding="utf-8")
    manifest = {
        "benchmark_status": "adopted",
        "license_review_required": False,
        "memory_eligible": False,
        "evaluation_metrics": [
            "generation_success_rate",
            "svg_validity_rate",
            "render_success_rate",
            "time_to_first_token",
            "tokens_per_second",
        ],
        "output_file": dataset_path.name,
        "num_records": len(examples),
        "output_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_path


def _write_config_bundle(root: Path, dataset_path: Path, report_dir: Path) -> None:
    files = {
        "eval.yaml": {
            "eval": {
                "dataset_path": str(dataset_path),
                "output_dir": str(report_dir),
                "metrics": [
                    "generation_success_rate",
                    "svg_validity_rate",
                    "render_success_rate",
                    "time_to_first_token",
                    "tokens_per_second",
                ],
                "partition": "held_out",
                "allow_memory_ingestion": False,
                "render_predictions": True,
                "acceptance_thresholds": {
                    "min_generation_success_rate": 1.0,
                    "min_svg_validity_rate": 1.0,
                    "min_render_success_rate": 1.0,
                    "max_avg_time_to_first_token": 1.0,
                    "min_avg_tokens_per_second": 20.0,
                },
            }
        },
        "generation.yaml": {
            "generation": {
                "max_new_tokens": 16,
                "temperature": 0,
                "do_sample": False,
                "render": {"enabled": False},
            }
        },
        "model.yaml": {"model": {"backend_type": "llama_cpp"}},
        "rag.yaml": {"rag": {}},
        "paths.yaml": {
            "paths": {
                "outputs": {
                    "generations": str(root / "generations"),
                    "renders": str(root / "renders"),
                }
            }
        },
    }
    for name, payload in files.items():
        (root / name).write_text(yaml.safe_dump(payload), encoding="utf-8")
