#!/usr/bin/env python3
"""Inspect live staged critic-SFT metrics and regenerate a combined loss graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from svg_agentic_slm.train.critic_training_logging import (
    collapse_resumed_metrics,
    convergence_summary,
    read_metrics,
    render_loss_svg,
)


def collect(pipeline_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stages_root = pipeline_dir / "stages"
    combined: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    step_offset = 0
    for stage in sorted(stages_root.glob("*")):
        metrics_path = stage / "output/logs/metrics.jsonl"
        status_path = stage / "output/logs/status.json"
        records = collapse_resumed_metrics(read_metrics(metrics_path))
        maximum = 0
        for record in records:
            local_step = int(record.get("global_step", 0))
            maximum = max(maximum, local_step)
            combined.append(
                {
                    **record,
                    "pipeline_step": step_offset + local_step,
                    "stage_directory": stage.name,
                }
            )
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        summaries[stage.name] = {
            "status": status.get("status", "not_started"),
            "global_step": status.get("global_step", maximum),
            "epoch": status.get("epoch"),
            "best_metric": status.get("best_metric"),
            "last_train_loss": next(
                (
                    record["loss"]
                    for record in reversed(records)
                    if isinstance(record.get("loss"), (int, float))
                ),
                None,
            ),
            "last_eval_loss": next(
                (
                    record["eval_loss"]
                    for record in reversed(records)
                    if isinstance(record.get("eval_loss"), (int, float))
                ),
                None,
            ),
            "convergence": convergence_summary(records),
            "metrics_path": str(metrics_path),
        }
        step_offset += maximum
    pipeline_manifest = pipeline_dir / "pipeline_manifest.json"
    pipeline = json.loads(pipeline_manifest.read_text()) if pipeline_manifest.exists() else {}
    return combined, {"pipeline": pipeline, "stages": summaries}


def update(pipeline_dir: Path) -> dict[str, Any]:
    records, summary = collect(pipeline_dir)
    graph = pipeline_dir / "loss_curve.svg"
    render_loss_svg(records, graph, title=f"Critic distillation pipeline — {pipeline_dir.name}")
    summary["loss_graph"] = str(graph)
    summary["metric_records"] = len(records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "plot", "watch"))
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    pipeline_dir = args.pipeline_dir.resolve()
    if args.command in {"status", "plot"}:
        print(json.dumps(update(pipeline_dir), indent=2, sort_keys=True))
        return
    try:
        while True:
            summary = update(pipeline_dir)
            print("\033[2J\033[H", end="")
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
