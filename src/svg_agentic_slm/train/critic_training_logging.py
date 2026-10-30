"""Durable JSONL metrics and dependency-free loss plots for critic SFT."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(value, dict):
                records.append(value)
    return records


def collapse_resumed_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the latest contiguous attempt when a run resumes from a checkpoint.

    Metrics are intentionally append-only, so restarting from checkpoint ``N``
    leaves the earlier attempt's records for ``N`` and later in the JSONL file.
    A decreasing ``global_step`` (or ``pipeline_step`` for a combined graph)
    marks that restart. The newer attempt replaces every previous record at or
    beyond its resume step; records without a usable step are retained because
    they cannot be attributed to either attempt.
    """
    collapsed: list[dict[str, Any]] = []
    previous_step: float | None = None
    for record in records:
        raw_step = record.get("pipeline_step", record.get("global_step"))
        if not isinstance(raw_step, (int, float)) or not math.isfinite(float(raw_step)):
            collapsed.append(record)
            continue
        step = float(raw_step)
        if previous_step is not None and step < previous_step:
            collapsed = [
                prior
                for prior in collapsed
                if not isinstance(
                    prior.get("pipeline_step", prior.get("global_step")),
                    (int, float),
                )
                or float(prior.get("pipeline_step", prior.get("global_step"))) < step
            ]
        collapsed.append(record)
        previous_step = step
    return collapsed


def convergence_summary(records: list[dict[str, Any]], *, window: int = 10) -> dict[str, Any]:
    points = [
        (float(row["global_step"]), float(row["loss"]))
        for row in records
        if isinstance(row.get("loss"), (int, float))
        and isinstance(row.get("global_step"), (int, float))
        and math.isfinite(float(row["loss"]))
    ][-window:]
    if len(points) < 4:
        return {
            "classification": "insufficient_history",
            "points": len(points),
            "relative_slope_per_step": None,
        }
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
        if denominator
        else 0.0
    )
    relative = slope / mean_y if mean_y else 0.0
    if relative < -0.002:
        classification = "still_decreasing"
    elif relative > 0.002:
        classification = "increasing_or_unstable"
    else:
        classification = "plateau_candidate"
    return {
        "classification": classification,
        "points": len(points),
        "window_first_step": points[0][0],
        "window_last_step": points[-1][0],
        "window_first_loss": points[0][1],
        "window_last_loss": points[-1][1],
        "slope_per_step": slope,
        "relative_slope_per_step": relative,
        "warning": "A flat training-loss window alone does not prove generalization convergence.",
    }


def render_loss_svg(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Render train/eval loss series without adding a plotting dependency."""
    series = collapse_resumed_metrics(series)
    width, height = 1100, 620
    left, right, top, bottom = 80, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    train_points: list[tuple[float, float]] = []
    eval_points: list[tuple[float, float]] = []
    for row in series:
        step = row.get("pipeline_step", row.get("global_step"))
        if not isinstance(step, (int, float)):
            continue
        if isinstance(row.get("loss"), (int, float)):
            train_points.append((float(step), float(row["loss"])))
        if isinstance(row.get("eval_loss"), (int, float)):
            eval_points.append((float(step), float(row["eval_loss"])))
    all_points = train_points + eval_points
    if not all_points:
        output_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620">'
            f'<text x="40" y="80" font-family="sans-serif">{title}: no loss data yet</text>'
            "</svg>\n"
        )
        return
    x_min, x_max = min(x for x, _ in all_points), max(x for x, _ in all_points)
    y_min, y_max = min(y for _, y in all_points), max(y for _, y in all_points)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0
    y_margin = max((y_max - y_min) * 0.08, 1e-4)
    y_min = max(0.0, y_min - y_margin)
    y_max += y_margin

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + (x - x_min) / (x_max - x_min) * plot_width
        py = top + (y_max - y) / (y_max - y_min) * plot_height
        return px, py

    def polyline(points: list[tuple[float, float]], color: str) -> str:
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, points))
        return (
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="22">{title}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
                'stroke="#e5e7eb"/>',
                f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" '
                f'font-family="monospace" font-size="12">{value:.4f}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
            'stroke="#111827"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#111827"/>',
            polyline(train_points, "#2563eb") if train_points else "",
            polyline(eval_points, "#dc2626") if eval_points else "",
            f'<text x="{left}" y="{height-25}" font-family="sans-serif" font-size="13">'
            f'step {x_min:g} → {x_max:g}</text>',
            f'<line x1="{width-250}" y1="28" x2="{width-220}" y2="28" '
            'stroke="#2563eb" stroke-width="3"/>',
            f'<text x="{width-212}" y="33" font-family="sans-serif" font-size="13">train</text>',
            f'<line x1="{width-140}" y1="28" x2="{width-110}" y2="28" '
            'stroke="#dc2626" stroke-width="3"/>',
            f'<text x="{width-102}" y="33" font-family="sans-serif" font-size="13">eval</text>',
            "</svg>",
        ]
    )
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, output_path)


class DetailedLoggingCallback(TrainerCallback):
    """Transformers-compatible callback writing live metrics, status, and SVG."""

    def __init__(self, output_dir: str | Path, run_context: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.status_path = self.output_dir / "status.json"
        self.plot_path = self.output_dir / "loss_curve.svg"
        self.run_context = dict(run_context)

    def _world_zero(self, state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _status(self, state: Any, status: str, **extra: Any) -> None:
        if not self._world_zero(state):
            return
        records = read_metrics(self.metrics_path)
        records = collapse_resumed_metrics(records)
        value = {
            **self.run_context,
            "status": status,
            "updated_at": _utc_now(),
            "global_step": int(getattr(state, "global_step", 0)),
            "epoch": getattr(state, "epoch", None),
            "best_metric": getattr(state, "best_metric", None),
            "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
            "convergence": convergence_summary(records),
            **extra,
        }
        _atomic_json(self.status_path, value)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del control, kwargs
        if self._world_zero(state):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                pass
        self._status(state, "running", output_dir=str(args.output_dir))

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del control, kwargs
        if not self._world_zero(state) or logs is None:
            return
        record: dict[str, Any] = {
            **self.run_context,
            "event": "log",
            "timestamp": _utc_now(),
            "global_step": int(state.global_step),
            "epoch": state.epoch,
            "estimated_samples_seen": int(
                state.global_step
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * max(1, args.world_size)
            ),
            **logs,
        }
        try:
            import torch

            if torch.cuda.is_available():
                record["gpu_memory_allocated_gib"] = round(
                    torch.cuda.memory_allocated() / 1024**3, 4
                )
                record["gpu_memory_reserved_gib"] = round(
                    torch.cuda.memory_reserved() / 1024**3, 4
                )
                record["gpu_peak_memory_allocated_gib"] = round(
                    torch.cuda.max_memory_allocated() / 1024**3, 4
                )
        except (ImportError, RuntimeError):
            pass
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records = read_metrics(self.metrics_path)
        render_loss_svg(
            records,
            self.plot_path,
            title=f"Critic SFT loss — {self.run_context.get('stage_id', 'stage')}",
        )
        self._status(state, "running")

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        self._status(state, "running", last_event="checkpoint_saved")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        self._status(state, "completed")

    def finalize(self, state: Any, **extra: Any) -> None:
        self._status(state, "completed", **extra)


__all__ = [
    "DetailedLoggingCallback",
    "collapse_resumed_metrics",
    "convergence_summary",
    "read_metrics",
    "render_loss_svg",
]
