"""Tests for configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from svg_agentic_slm.factories.generation import _build_model_backend
from svg_agentic_slm.models.gemma_loader import GemmaModelBackend
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.llama_cpp_backend import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    LlamaCppModelBackend,
)
from svg_agentic_slm.utils.config import load_yaml_config, merge_configs


def test_load_yaml_config(tmp_path: Path) -> None:
    """Test loading a valid YAML config."""
    config_data = {"model": {"model_id": "test-model", "backend_type": "huggingface"}}
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    loaded = load_yaml_config(config_path)
    assert loaded == config_data


def test_load_yaml_config_not_found() -> None:
    """Test that missing config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("/nonexistent/config.yaml")


def test_load_empty_yaml(tmp_path: Path) -> None:
    """Test that an empty YAML file returns an empty dict."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    loaded = load_yaml_config(config_path)
    assert loaded == {}


def test_load_yaml_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Configuration files must contain a mapping at the document root."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml_config(config_path)


def test_merge_configs() -> None:
    """Test merging multiple config dictionaries."""
    a = {"key1": "a", "key2": "a"}
    b = {"key2": "b", "key3": "b"}

    merged = merge_configs(a, b)
    assert merged == {"key1": "a", "key2": "b", "key3": "b"}


def test_generation_config_rejects_unknown_option() -> None:
    with pytest.raises(ValueError, match="unknown_option"):
        GenerationConfig.from_dict({"unknown_option": True})


def test_model_backend_factory_rejects_unknown_option() -> None:
    with pytest.raises(ValueError, match="n_gpu_layer"):
        _build_model_backend(
            {"backend_type": "llama_cpp", "n_gpu_layer": -1},
            GenerationConfig(),
        )


def test_model_backend_factory_uses_compatibility_checkpoint_defaults() -> None:
    backend = _build_model_backend(
        {"backend_type": "llama_cpp"},
        GenerationConfig(),
    )

    assert backend.model_id == DEFAULT_MODEL_ID
    assert backend.filename == DEFAULT_MODEL_FILE
    assert backend.model_revision == DEFAULT_MODEL_REVISION


@pytest.mark.parametrize(
    ("backend_type", "expected_class"),
    [
        ("llama_cpp", LlamaCppModelBackend),
        ("gemma", GemmaModelBackend),
    ],
)
def test_model_backend_factory_selects_class_and_forwards_common_options(
    backend_type: str,
    expected_class: type[LlamaCppModelBackend],
) -> None:
    generation_config = GenerationConfig(temperature=0.25)

    backend = _build_model_backend(
        {
            "backend_type": backend_type,
            "model_id": "example/model",
            "filename": "model.gguf",
            "n_ctx": 4096,
            "n_gpu_layers": 24,
            "n_batch": 128,
            "flash_attn": False,
            "use_mmap": False,
            "verbose": True,
            "chat_format": "gemma",
        },
        generation_config,
    )

    assert type(backend) is expected_class
    assert backend.model_id == "example/model"
    assert backend.filename == "model.gguf"
    assert backend.n_ctx == 4096
    assert backend.n_gpu_layers == 24
    assert backend.n_batch == 128
    assert backend.flash_attn is False
    assert backend.use_mmap is False
    assert backend.verbose is True
    assert backend.chat_format == "gemma"
    assert backend.generation_config is generation_config
