"""Project path helpers.

Provides functions to resolve commonly used directories
(data, outputs, checkpoints, logs) relative to the project root.
Avoids hard-coded paths in source files.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory.

    Resolution order:
    1. PROJECT_ROOT environment variable, if set.
    2. Walk up from this file to find the directory containing pyproject.toml.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If the project root cannot be determined.
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        "Cannot determine project root. Set PROJECT_ROOT env var "
        "or ensure pyproject.toml exists in a parent directory."
    )


def get_data_dir(subdir: str | None = None) -> Path:
    """Return the data directory, optionally with a subdirectory.

    Args:
        subdir: Optional subdirectory under data/ (e.g., 'raw', 'processed').

    Returns:
        Resolved path to the data directory.
    """
    data_dir = get_project_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    return data_dir


def get_output_dir(subdir: str | None = None) -> Path:
    """Return the outputs directory, optionally with a subdirectory.

    Args:
        subdir: Optional subdirectory under outputs/ (e.g., 'generations').

    Returns:
        Resolved path to the output directory.
    """
    output_dir = get_project_root() / "outputs"
    if subdir:
        output_dir = output_dir / subdir
    return output_dir


def get_checkpoint_dir() -> Path:
    """Return the checkpoints directory."""
    return get_project_root() / "checkpoints"


def get_log_dir() -> Path:
    """Return the logs directory."""
    return get_project_root() / "logs"


def get_config_dir() -> Path:
    """Return the configs directory."""
    return get_project_root() / "configs"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.

    Returns:
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
