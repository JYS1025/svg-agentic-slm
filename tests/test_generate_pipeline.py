"""Tests for the config-driven generate command pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from svg_agentic_slm.cli.app import app
from svg_agentic_slm.factories.generation import build_generation_runtime

runner = CliRunner()


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

    svg_files = sorted(output_dir.glob("*.svg"))
    json_files = sorted(output_dir.glob("*.json"))

    assert len(svg_files) == 1
    assert len(json_files) == 1
    assert "Placeholder" in svg_files[0].read_text(encoding="utf-8")

    metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert metadata["instruction"] == "Draw a blue circle."
    assert metadata["runtime"]["enable_critic"] is True
    assert metadata["runtime"]["enable_rag"] is False
    assert metadata["runtime"]["enable_render"] is False
    assert metadata["metadata"]["validation"]["is_valid"] is True
    assert metadata["critic_feedback"][0]["critic_type"] == "rule"


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
    assert metadata["render_path"] == str(png_files[0])
    assert metadata["metadata"]["render"]["success"] is True
    assert metadata["runtime"]["planned_artifacts"]["render_path"] == str(png_files[0])


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
