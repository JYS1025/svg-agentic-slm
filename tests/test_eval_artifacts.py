"""Tests for artifact-backed evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from svg_agentic_slm.cli.app import app
from svg_agentic_slm.eval.report import generate_report
from svg_agentic_slm.eval.run_eval import run_evaluation

runner = CliRunner()


def test_run_evaluation_reads_generation_artifacts(tmp_path: Path) -> None:
    """Artifact-backed evaluation should aggregate metrics from saved outputs."""
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    _write_artifact_bundle(
        artifact_dir,
        stem="sample_a",
        instruction="Draw a blue circle.",
        svg_content='<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>',
        render_success=True,
        latency=0.25,
        outcome="accepted",
    )
    _write_artifact_bundle(
        artifact_dir,
        stem="sample_b",
        instruction="Broken svg.",
        svg_content="not svg",
        render_success=False,
        latency=0.75,
        outcome="rejected",
    )

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "artifact_path": str(artifact_dir),
                    "output_dir": str(report_dir),
                    "metrics": [
                        "svg_validity_rate",
                        "render_success_rate",
                        "generation_latency",
                        "simple_instruction_alignment",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_evaluation(config_path)

    assert result.num_samples == 2
    assert result.svg_validity_rate == 0.5
    assert result.render_success_rate == 0.5
    assert result.avg_generation_latency == 0.5
    assert result.avg_instruction_alignment == 1.0
    assert result.metadata["evaluation_mode"] == "artifacts"
    assert result.metadata["artifact_source"] == str(artifact_dir)
    assert result.metadata["outcome_counts"] == {"accepted": 1, "rejected": 1}
    assert len(result.per_sample_results) == 2


def test_generate_report_includes_per_sample_results(tmp_path: Path) -> None:
    """JSON reports should preserve per-sample artifact-backed details."""
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    _write_artifact_bundle(
        artifact_dir,
        stem="sample_a",
        instruction="Draw a square.",
        svg_content='<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>',
        render_success=True,
        latency=0.2,
    )

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump({"eval": {"artifact_path": str(artifact_dir)}}),
        encoding="utf-8",
    )

    result = run_evaluation(config_path)
    report_path = generate_report(result, output_dir=report_dir, report_name="artifact_eval")
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["num_samples"] == 1
    assert len(payload["per_sample_results"]) == 1
    assert payload["per_sample_results"][0]["instruction"] == "Draw a square."
    assert payload["per_sample_results"][0]["render_success"] is True


def test_eval_command_applies_cli_overrides(tmp_path: Path) -> None:
    """The eval CLI should apply artifact and max-sample overrides consistently."""
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    _write_artifact_bundle(
        artifact_dir,
        stem="sample_a",
        instruction="Draw a red square.",
        svg_content='<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>',
        render_success=True,
        latency=0.4,
    )
    _write_artifact_bundle(
        artifact_dir,
        stem="sample_b",
        instruction="Draw a red triangle.",
        svg_content=(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<polygon points="0,0 10,0 5,10"/></svg>'
        ),
        render_success=True,
        latency=0.8,
    )

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump({"eval": {"artifact_path": str(tmp_path / "unused")}}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(config_path),
            "--report-dir",
            str(report_dir),
            "--artifact-path",
            str(artifact_dir),
            "--max-samples",
            "1",
            "--set",
            "eval.metrics=['svg_validity_rate']",
        ],
    )

    assert result.exit_code == 0
    assert "Evaluation Results (1 samples):" in result.stdout

    report_path = report_dir / "eval_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["num_samples"] == 1
    assert payload["metadata"]["artifact_source"] == str(artifact_dir)
    assert payload["metadata"]["requested_max_samples"] == 1
    assert payload["metadata"]["metrics"] == ["svg_validity_rate"]
    assert payload["render_success_rate"] == 0.0


def test_eval_command_uses_configured_report_directory(tmp_path: Path) -> None:
    """The config output directory should apply when --report-dir is omitted."""
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "configured-reports"
    _write_artifact_bundle(
        artifact_dir,
        stem="sample",
        instruction="Draw a square.",
        svg_content='<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
        render_success=False,
        latency=0.1,
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"eval": {"artifact_path": str(artifact_dir), "output_dir": str(report_dir)}}
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (report_dir / "eval_report.json").exists()


def test_evaluation_excludes_disabled_rendering_from_success_rate(tmp_path: Path) -> None:
    """A skipped render is not a failed render attempt."""
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_bundle(
        artifact_dir,
        stem="rendered",
        instruction="Draw a square.",
        svg_content='<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
        render_success=True,
        latency=0.1,
    )
    _write_artifact_bundle(
        artifact_dir,
        stem="not_rendered",
        instruction="Draw a circle.",
        svg_content='<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
        render_success=False,
        latency=0.1,
        render_enabled=False,
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump({"eval": {"artifact_path": str(artifact_dir)}}),
        encoding="utf-8",
    )

    result = run_evaluation(config_path)

    assert result.render_success_rate == 1.0
    assert result.metadata["render_attempt_count"] == 1
    disabled_sample = next(
        sample
        for sample in result.per_sample_results
        if sample["instruction"] == "Draw a circle."
    )
    assert disabled_sample["render_success"] is None


def _write_artifact_bundle(
    artifact_dir: Path,
    *,
    stem: str,
    instruction: str,
    svg_content: str,
    render_success: bool,
    latency: float,
    render_enabled: bool = True,
    outcome: str | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    svg_path = artifact_dir / f"{stem}.svg"
    metadata_path = artifact_dir / f"{stem}.json"
    svg_path.write_text(svg_content, encoding="utf-8")
    render_path = artifact_dir / f"{stem}.png"
    if render_enabled and render_success:
        render_path.write_bytes(b"render")

    payload = {
        "instruction": instruction,
        "svg_path": str(svg_path),
        "render_path": str(render_path) if render_success else None,
        "is_valid": render_success,
        "outcome": outcome,
        "stop_reason": "test_complete" if outcome else None,
        "revision_count": 0,
        "critic_feedback": [],
        "runtime": {"enable_render": render_enabled},
        "metadata": {
            "render": {
                "enabled": render_enabled,
                "success": render_success,
            },
            "timing": {
                "generation_latency_seconds": latency,
            },
        },
        "generated_at_utc": "2026-07-10T12:00:00Z",
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
