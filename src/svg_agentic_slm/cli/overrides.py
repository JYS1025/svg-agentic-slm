"""Shared CLI override helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


def parse_override_items(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated KEY=VALUE CLI overrides into a nested dictionary."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected dotted.path=value format."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Override key cannot be empty.")
        value = yaml.safe_load(raw_value)
        set_nested_override(overrides, key, value)
    return overrides


def set_nested_override(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a dotted override path inside a nested dictionary."""
    current = target
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("Override path cannot be empty.")

    for part in parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            current[part] = {}
            next_value = current[part]
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot assign nested override under non-mapping key '{part}'."
            )
        current = next_value
    current[parts[-1]] = value


def merge_nested_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries without mutating the inputs."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
