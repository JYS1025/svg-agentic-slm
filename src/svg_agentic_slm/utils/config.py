"""YAML configuration loading utilities.

Provides functions to load and merge YAML configuration files.
All configurable values should live in YAML files under configs/,
not hard-coded in source files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def load_env(env_path: str | Path | None = None) -> None:
    """Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file. If None, searches for .env
                  in the current directory and parent directories.
    """
    load_dotenv(dotenv_path=env_path)


def get_env(key: str, default: str | None = None) -> str | None:
    """Get an environment variable with an optional default.

    Args:
        key: Environment variable name.
        default: Default value if the variable is not set.

    Returns:
        The value of the environment variable, or the default.
    """
    return os.environ.get(key, default)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries, with later values taking precedence.

    Performs a shallow merge at the top level. For nested merging,
    consider using a dedicated library or recursive merge.

    Args:
        *configs: Configuration dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    merged: dict[str, Any] = {}
    for config in configs:
        merged.update(config)
    return merged
