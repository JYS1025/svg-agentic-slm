#!/usr/bin/env python3
"""Smoke test to verify the project skeleton is functional.

Runs basic checks to ensure imports work, configs load,
and placeholder components can be instantiated.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for direct script execution
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))


def check_imports() -> bool:
    """Verify that all major modules can be imported."""
    print("Checking imports...")
    try:
        import svg_agentic_slm
        from svg_agentic_slm.utils.config import load_yaml_config
        from svg_agentic_slm.utils.logging import setup_logging, get_logger
        from svg_agentic_slm.utils.paths import get_project_root
        from svg_agentic_slm.utils.seed import set_seed
        from svg_agentic_slm.models.base import BaseModelBackend
        from svg_agentic_slm.models.gemma_loader import GemmaModelBackend
        from svg_agentic_slm.models.generation_config import GenerationConfig
        from svg_agentic_slm.agents.base import BaseGenerator, BaseCritic
        from svg_agentic_slm.agents.schemas import GenerationRequest, GenerationResult
        from svg_agentic_slm.agents.orchestrator import SVGGenerationOrchestrator
        from svg_agentic_slm.svg.validator import SVGValidator
        from svg_agentic_slm.svg.schemas import SVGValidationResult
        from svg_agentic_slm.data.jsonl import read_jsonl, write_jsonl
        from svg_agentic_slm.data.schemas import TextToSVGExample
        from svg_agentic_slm.rag.base import BaseRetriever
        from svg_agentic_slm.rag.schemas import RetrievedExample
        from svg_agentic_slm.eval.schemas import EvaluationResult
        print("  ✓ All imports successful.")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_config_loading() -> bool:
    """Verify that YAML configs can be loaded."""
    print("Checking config loading...")
    from svg_agentic_slm.utils.config import load_yaml_config
    from svg_agentic_slm.utils.paths import get_project_root

    try:
        config_dir = get_project_root() / "configs"
        for config_file in config_dir.glob("*.yaml"):
            config = load_yaml_config(config_file)
            print(f"  ✓ Loaded {config_file.name}: {list(config.keys())}")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def check_validator() -> bool:
    """Verify that the SVG validator works."""
    print("Checking SVG validator...")
    from svg_agentic_slm.svg.validator import SVGValidator

    validator = SVGValidator()

    # Valid SVG
    valid_svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
    result = validator.validate(valid_svg)
    assert result.is_valid, f"Expected valid, got errors: {result.errors}"
    print("  ✓ Valid SVG correctly identified.")

    # Invalid SVG
    invalid_svg = "This is not SVG"
    result = validator.validate(invalid_svg)
    assert not result.is_valid, "Expected invalid SVG to fail."
    print("  ✓ Invalid SVG correctly rejected.")

    return True


def check_jsonl() -> bool:
    """Verify that JSONL read/write works."""
    print("Checking JSONL utilities...")
    from svg_agentic_slm.data.jsonl import read_jsonl
    from svg_agentic_slm.utils.paths import get_project_root

    sample_path = get_project_root() / "data" / "examples" / "text_to_svg_sample.jsonl"
    if sample_path.exists():
        records = read_jsonl(sample_path)
        print(f"  ✓ Read {len(records)} records from sample JSONL.")
        return True
    else:
        print(f"  ⚠ Sample JSONL not found at {sample_path}")
        return True  # Not a failure, just missing sample data


def main() -> None:
    """Run all smoke tests."""
    print("=" * 60)
    print("SVG Agentic SLM — Smoke Test")
    print("=" * 60)
    print()

    checks = [
        check_imports,
        check_config_loading,
        check_validator,
        check_jsonl,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"  ✗ Check failed with exception: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} checks passed.")
    if all(results):
        print("All smoke tests passed! ✓")
    else:
        print("Some smoke tests failed. ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
