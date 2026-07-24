"""Tests for the config-driven generate command pipeline."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import CriticFeedback
from svg_agentic_slm.artifacts.generation import load_generation_artifact
from svg_agentic_slm.cli.app import app
from svg_agentic_slm.factories.generation import (
    CompositeCritic,
    GenerationArtifacts,
    _resolve_artifact_paths,
    build_generation_runtime,
    persist_generation_artifacts,
)
from svg_agentic_slm.models.schemas import ModelResponse

runner = CliRunner()


class _FakeModelBackend:
    """Network-free backend injected into config-driven pipeline tests."""

    def load_model(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(
            text=(
                '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">'
                '<circle cx="128" cy="128" r="64" fill="blue"/></svg>'
            ),
            model_id="fake-model",
            model_revision="test-revision",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )


class _MalformedBooleanCritic(BaseCritic):
    """Return a dataclass instance that violates its runtime field contract."""

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


@pytest.fixture(autouse=True)
def _inject_fake_model_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "svg_agentic_slm.factories.generation._build_model_backend",
        lambda model_config, generation_config: _FakeModelBackend(),
    )


def test_build_generation_runtime_from_sibling_configs(tmp_path: Path) -> None:
    """Runtime assembly should resolve sibling config files and feature flags."""
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)

    runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw a green triangle.",
        enable_rag=True,
        enable_critic=True,
    )

    assert runtime.request.instruction == "Draw a green triangle."
    assert runtime.request.config_overrides["max_new_tokens"] == 128
    assert runtime.enable_rag
    assert runtime.enable_critic
    assert runtime.critic_type == "rule"
    assert runtime.output_dir == output_dir
    assert runtime.config_paths["model"].endswith("model.yaml")
    assert runtime.config_paths["paths"].endswith("paths.yaml")


def test_composite_critic_validates_each_child_before_aggregation() -> None:
    critic = CompositeCritic([_MalformedBooleanCritic()])

    with pytest.raises(
        TypeError,
        match=r"CriticFeedback\.is_valid must be a boolean",
    ):
        critic.critique("Draw a circle.", "<svg/>")


def test_generate_command_persists_svg_and_metadata(tmp_path: Path) -> None:
    """The generate CLI should run the orchestrator and save output artifacts."""
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)

    result = runner.invoke(
        app,
        [
            "generate",
            "Draw a blue circle.",
            "--config",
            str(tmp_path / "generation.yaml"),
            "--critic",
        ],
    )

    assert result.exit_code == 0
    assert "Generated SVG:" in result.stdout
    assert "Valid SVG: True" in result.stdout
    assert "Outcome: accepted" in result.stdout

    svg_files = sorted(output_dir.glob("*.svg"))
    json_files = sorted(output_dir.glob("*.json"))

    assert len(svg_files) == 1
    assert len(json_files) == 1
    assert "<circle" in svg_files[0].read_text(encoding="utf-8")

    metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["run_id"].startswith("run_")
    assert metadata["instruction"] == "Draw a blue circle."
    assert metadata["outcome"] == "accepted"
    assert metadata["stop_reason"] == "critic_acceptance_threshold_met"
    canonical_svg = json_files[0].parent / metadata["svg_path"]
    assert canonical_svg.is_file()
    assert canonical_svg.read_text(encoding="utf-8") == svg_files[0].read_text(
        encoding="utf-8"
    )
    assert metadata["runtime"]["enable_critic"] is True
    assert metadata["runtime"]["enable_rag"] is False
    assert metadata["runtime"]["enable_render"] is False
    assert metadata["metadata"]["validation"]["is_valid"] is True
    assert metadata["critic_feedback"][0]["critic_type"] == "rule"
    assert metadata["critic_feedback"][0]["critic_version"] == "rule-svg-validation-v1"
    attempt = metadata["metadata"]["generator"]["attempts"][0]
    assert attempt["model_calls"]
    assert attempt["raw_output_ref"]
    assert (json_files[0].parent / attempt["raw_output_ref"]).exists()
    model_call = attempt["model_calls"][0]
    assert (json_files[0].parent / model_call["prompt_ref"]).exists()
    assert (json_files[0].parent / model_call["system_prompt_ref"]).exists()
    assert model_call["generation_parameters"]["max_new_tokens"] == 128
    assert metadata["critic_feedback"][0]["target_attempt_id"] == attempt["attempt_id"]
    assert not list(output_dir.rglob("*.tmp"))
    record = load_generation_artifact(json_files[0])
    assert record.outcome == "accepted"
    assert len(record.attempts) == 1


def test_generate_command_persists_render_output(tmp_path: Path) -> None:
    """The generate CLI should save a rendered artifact when rendering is enabled."""
    pytest.importorskip("cairosvg")

    output_dir = tmp_path / "outputs" / "generations"
    render_dir = tmp_path / "outputs" / "renders"
    _write_generation_config_bundle(
        tmp_path,
        output_dir,
        render_dir=render_dir,
        render_enabled=True,
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "Draw a yellow star.",
            "--config",
            str(tmp_path / "generation.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert "Render saved to:" in result.stdout

    png_files = sorted(render_dir.glob("*.png"))
    assert len(png_files) == 1
    assert png_files[0].stat().st_size > 0

    metadata_files = sorted(output_dir.glob("*.json"))
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    canonical_render = metadata_files[0].parent / metadata["render_path"]
    assert canonical_render.is_file()
    assert metadata["metadata"]["render"]["success"] is True
    assert metadata["runtime"]["planned_artifacts"]["render_path"] == str(png_files[0])


def test_artifact_republish_failure_preserves_previous_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)
    output_path = tmp_path / "result.svg"

    first_runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw the first circle.",
        output_path=output_path,
    )
    first_result = first_runtime.orchestrator.run(first_runtime.request)
    persist_generation_artifacts(first_result, first_runtime)
    original_metadata = first_runtime.metadata_output_path.read_text(encoding="utf-8")
    original_svg = load_generation_artifact(first_runtime.metadata_output_path).svg_path

    second_runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw a replacement square.",
        output_path=output_path,
    )
    second_result = second_runtime.orchestrator.run(second_runtime.request)
    second_result.generated_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'
    )
    second_result.attempts[-1].svg = second_result.generated_svg

    from svg_agentic_slm.artifacts import writer as artifact_writer

    original_atomic_write = artifact_writer._atomic_write_text

    def fail_metadata_publish(
        path: Path,
        content: str,
        *,
        on_replace=None,
    ) -> None:
        if path == second_runtime.metadata_output_path:
            raise OSError("simulated metadata publication failure")
        original_atomic_write(path, content, on_replace=on_replace)

    monkeypatch.setattr(artifact_writer, "_atomic_write_text", fail_metadata_publish)

    with pytest.raises(OSError, match="simulated metadata"):
        persist_generation_artifacts(second_result, second_runtime)

    assert second_runtime.metadata_output_path.read_text(encoding="utf-8") == original_metadata
    assert load_generation_artifact(second_runtime.metadata_output_path).svg_path == original_svg
    assert "<circle" in output_path.read_text(encoding="utf-8")
    assert not list(tmp_path.rglob("*.tmp"))


def test_post_replace_failure_keeps_committed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)
    output_path = tmp_path / "result.svg"

    first_runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw the first circle.",
        output_path=output_path,
    )
    first_result = first_runtime.orchestrator.run(first_runtime.request)
    persist_generation_artifacts(first_result, first_runtime)
    original_alias = output_path.read_text(encoding="utf-8")

    second_runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw the committed replacement.",
        output_path=output_path,
    )
    second_result = second_runtime.orchestrator.run(second_runtime.request)

    from svg_agentic_slm.artifacts import writer as artifact_writer

    original_fsync_directory = artifact_writer._fsync_directory

    def fail_after_sidecar_replace(path: Path) -> None:
        if path == second_runtime.metadata_output_path.parent:
            payload = json.loads(
                second_runtime.metadata_output_path.read_text(encoding="utf-8")
            )
            if payload["run_id"] == second_runtime.run_id:
                raise OSError("simulated post-replace durability failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        artifact_writer,
        "_fsync_directory",
        fail_after_sidecar_replace,
    )

    with pytest.raises(OSError, match="post-replace durability"):
        persist_generation_artifacts(second_result, second_runtime)

    committed = load_generation_artifact(second_runtime.metadata_output_path)
    assert committed.run_id == second_runtime.run_id
    assert committed.svg_path.is_file()
    assert output_path.read_text(encoding="utf-8") == original_alias
    bundle_root = second_runtime.metadata_output_path.with_suffix(".artifacts")
    assert len([path for path in bundle_root.iterdir() if path.is_dir()]) == 2


def test_artifact_writer_rejects_final_svg_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)
    output_path = tmp_path / "result.svg"
    runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw a circle.",
        output_path=output_path,
    )
    result = runtime.orchestrator.run(runtime.request)
    result.generated_svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'

    with pytest.raises(ValueError, match="must match the final attempt SVG"):
        persist_generation_artifacts(result, runtime)

    assert not runtime.metadata_output_path.exists()
    assert not output_path.exists()
    assert not runtime.metadata_output_path.with_suffix(".artifacts").exists()


def test_artifact_writer_rejects_invalid_schema_before_publication(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)
    output_path = tmp_path / "result.svg"
    first_runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw the accepted circle.",
        output_path=output_path,
    )
    first_result = first_runtime.orchestrator.run(first_runtime.request)
    persist_generation_artifacts(first_result, first_runtime)
    original_metadata = first_runtime.metadata_output_path.read_text(encoding="utf-8")
    original_svg = output_path.read_text(encoding="utf-8")
    original_record = load_generation_artifact(first_runtime.metadata_output_path)

    second_runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        prompt="Draw a malformed replacement.",
        output_path=output_path,
    )
    second_result = second_runtime.orchestrator.run(second_runtime.request)
    second_result.attempts.append(second_result.attempts[0])

    with pytest.raises(ValueError, match="attempt_id values must be unique"):
        persist_generation_artifacts(second_result, second_runtime)

    assert second_runtime.metadata_output_path.read_text(encoding="utf-8") == original_metadata
    assert output_path.read_text(encoding="utf-8") == original_svg
    current_record = load_generation_artifact(second_runtime.metadata_output_path)
    assert current_record.run_id == original_record.run_id
    assert current_record.svg_path == original_record.svg_path
    bundle_root = second_runtime.metadata_output_path.with_suffix(".artifacts")
    assert len([path for path in bundle_root.iterdir() if path.is_dir()]) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_artifact_publication_is_serialized_per_output_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from svg_agentic_slm.artifacts import writer as artifact_writer

    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def observe_locked_publication(result, runtime) -> GenerationArtifacts:
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return GenerationArtifacts(
            svg_path=tmp_path / "result.svg",
            metadata_path=tmp_path / "result.json",
            render_path=None,
        )

    monkeypatch.setattr(
        artifact_writer,
        "_persist_generation_artifacts_locked",
        observe_locked_publication,
    )
    runtime = SimpleNamespace(metadata_output_path=tmp_path / "result.json")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                artifact_writer.persist_generation_artifacts,
                object(),
                runtime,
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert maximum_active == 1


def test_generate_command_applies_cli_overrides(tmp_path: Path) -> None:
    """CLI overrides should update runtime config before generation."""
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)

    result = runner.invoke(
        app,
        [
            "generate",
            "Draw a green hexagon.",
            "--config",
            str(tmp_path / "generation.yaml"),
            "--max-new-tokens",
            "64",
            "--temperature",
            "0.25",
            "--seed",
            "11",
            "--no-render",
            "--set",
            "generation.top_p=0.8",
            "--set",
            "model.model_id=override-model",
        ],
    )

    assert result.exit_code == 0

    metadata_files = sorted(output_dir.glob("*.json"))
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["runtime"]["generation_config"]["max_new_tokens"] == 64
    assert metadata["runtime"]["generation_config"]["temperature"] == 0.25
    assert metadata["runtime"]["generation_config"]["seed"] == 11
    assert metadata["runtime"]["generation_config"]["top_p"] == 0.8
    assert metadata["runtime"]["model_config"]["model_id"] == "override-model"
    assert metadata["runtime"]["enable_render"] is False


def test_generate_command_rejects_non_svg_output_path(tmp_path: Path) -> None:
    """An explicit generation output must not collide with its JSON sidecar."""
    output_dir = tmp_path / "outputs" / "generations"
    _write_generation_config_bundle(tmp_path, output_dir)

    result = runner.invoke(
        app,
        [
            "generate",
            "Draw a circle.",
            "--config",
            str(tmp_path / "generation.yaml"),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    assert result.exit_code == 1
    assert "must use the .svg extension" in result.stdout
    assert not (tmp_path / "result.json").exists()


def test_generated_artifact_names_are_unique_for_non_ascii_prompts(tmp_path: Path) -> None:
    """Distinct runs must not overwrite each other when prompt slugs are identical."""
    first = _resolve_artifact_paths(
        generations_dir=tmp_path,
        renders_dir=tmp_path,
        instruction="파란 원을 그려줘",
        output_path=None,
        render_enabled=False,
        render_format="png",
    )
    second = _resolve_artifact_paths(
        generations_dir=tmp_path,
        renders_dir=tmp_path,
        instruction="빨간 사각형을 그려줘",
        output_path=None,
        render_enabled=False,
        render_format="png",
    )

    assert first["svg"] != second["svg"]


def _write_generation_config_bundle(
    config_dir: Path,
    output_dir: Path,
    *,
    render_dir: Path | None = None,
    render_enabled: bool = False,
) -> None:
    """Create a minimal self-contained config set for the generate command."""
    if render_dir is None:
        render_dir = config_dir / "outputs" / "renders"

    files = {
        "generation.yaml": {
            "generation": {
                "max_new_tokens": 128,
                "temperature": 0.1,
                "do_sample": False,
                "seed": 7,
                "orchestration": {
                    "enable_rag": False,
                    "enable_critic": False,
                    "max_revision_rounds": 2,
                    "critic_type": "rule",
                },
                "render": {
                    "enabled": render_enabled,
                    "backend": "cairosvg",
                    "output_format": "png",
                },
            }
        },
        "model.yaml": {
            "model": {
                "model_id": "test-model",
                "device_map": "cpu",
                "torch_dtype": "float32",
            }
        },
        "rag.yaml": {
            "rag": {
                "collection_name": "test_collection",
                "persist_directory": str(config_dir / "chroma"),
                "embedding_model": "test-embedding",
                "top_k": 2,
            }
        },
        "paths.yaml": {
            "paths": {
                "outputs": {
                    "generations": str(output_dir),
                    "renders": str(render_dir),
                }
            }
        },
    }

    for filename, payload in files.items():
        with open(config_dir / filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)
