"""Tests for configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_merge_configs() -> None:
    """Test merging multiple config dictionaries."""
    a = {"key1": "a", "key2": "a"}
    b = {"key2": "b", "key3": "b"}

    merged = merge_configs(a, b)
    assert merged == {"key1": "a", "key2": "b", "key3": "b"}
