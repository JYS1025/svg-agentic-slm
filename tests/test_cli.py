"""Tests for CLI module imports.

Verifies that the CLI module can be imported without errors.
Does not test actual command execution (which would require
model loading and other dependencies).
"""

from __future__ import annotations


def test_cli_app_imports() -> None:
    """Test that the CLI app can be imported."""
    from svg_agentic_slm.cli.app import app
    assert app is not None


def test_cli_commands_import() -> None:
    """Test that all CLI command modules can be imported."""
    from svg_agentic_slm.cli import commands_generate
    from svg_agentic_slm.cli import commands_validate
    from svg_agentic_slm.cli import commands_render
    from svg_agentic_slm.cli import commands_train
    from svg_agentic_slm.cli import commands_eval

    assert commands_generate is not None
    assert commands_validate is not None
    assert commands_render is not None
    assert commands_train is not None
    assert commands_eval is not None
