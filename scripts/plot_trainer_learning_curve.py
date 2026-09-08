#!/usr/bin/env python3
"""Plot a Hugging Face Trainer learning curve using only Pillow and stdlib."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageFont


Color = tuple[int, int, int]
Box = tuple[int, int, int, int]


INK: Color = (25, 35, 52)
MUTED: Color = (101, 113, 133)
GRID: Color = (221, 226, 235)
PANEL: Color = (248, 250, 253)
BLUE: Color = (32, 104, 226)
BLUE_LIGHT: Color = (157, 190, 245)
ORANGE: Color = (225, 101, 38)
GREEN: Color = (24, 142, 92)
WHITE: Color = (255, 255, 255)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _trailing_mean(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coords(
    box: Box,
    *,
    x_max: float,
    y_min: float,
    y_max: float,
    logarithmic: bool,
) -> tuple[Callable[[float], int], Callable[[float], int]]:
    left, top, right, bottom = box

    def x(value: float) -> int:
        return round(left + (right - left) * value / x_max)

    if logarithmic:
        low = math.log(y_min)
        high = math.log(y_max)

        def y(value: float) -> int:
            clamped = min(max(value, y_min), y_max)
            ratio = (math.log(clamped) - low) / (high - low)
            return round(bottom - (bottom - top) * ratio)

    else:

        def y(value: float) -> int:
            clamped = min(max(value, y_min), y_max)
            ratio = (clamped - y_min) / (y_max - y_min)
            return round(bottom - (bottom - top) * ratio)

    return x, y


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: Box,
    *,
    x_max: int,
    x_ticks: Iterable[int],
    y_ticks: Iterable[float],
    y_min: float,
    y_max: float,
    logarithmic: bool,
    title: str,
) -> tuple[Callable[[float], int], Callable[[float], int]]:
    left, top, right, bottom = box
    draw.rounded_rectangle((left - 80, top - 58, right + 34, bottom + 62), 18, fill=PANEL)
    draw.text((left - 60, top - 48), title, fill=INK, font=_font(23, bold=True))
    x, y = _coords(
        box,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        logarithmic=logarithmic,
    )
    for tick in x_ticks:
        px = x(tick)
        draw.line((px, top, px, bottom), fill=GRID, width=2)
        label = f"{tick:,}"
        bbox = draw.textbbox((0, 0), label, font=_font(17))
        draw.text((px - (bbox[2] - bbox[0]) / 2, bottom + 13), label, fill=MUTED, font=_font(17))
    for tick in y_ticks:
        py = y(tick)
        draw.line((left, py, right, py), fill=GRID, width=2)
        label = f"{tick:g}"
        bbox = draw.textbbox((0, 0), label, font=_font(17))
        draw.text((left - 16 - (bbox[2] - bbox[0]), py - 10), label, fill=MUTED, font=_font(17))
    draw.line((left, top, left, bottom), fill=MUTED, width=2)
    draw.line((left, bottom, right, bottom), fill=MUTED, width=2)
    return x, y


def _line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: Color,
    width: int,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trainer_state", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--title", default="SFT Learning Curve")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()

    state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    history: list[dict[str, Any]] = state["log_history"]
    train = sorted((item for item in history if "loss" in item), key=lambda item: item["step"])
    validation = sorted(
        (item for item in history if "eval_loss" in item), key=lambda item: item["step"]
    )
    tests = [item for item in history if "test_loss" in item]
    summaries = [item for item in history if "train_loss" in item]
    if not train or not validation:
        raise ValueError("trainer_state must contain both loss and eval_loss records")

    max_steps = int(state.get("max_steps") or state["global_step"])
    train_steps = [int(item["step"]) for item in train]
    train_loss = [float(item["loss"]) for item in train]
    smooth_loss = _trailing_mean(train_loss, window=5)
    eval_steps = [int(item["step"]) for item in validation]
    eval_loss = [float(item["eval_loss"]) for item in validation]
    test_loss = float(tests[-1]["test_loss"]) if tests else None
    aggregate_train_loss = float(summaries[-1]["train_loss"]) if summaries else None
    best_step = int(state.get("best_global_step") or eval_steps[eval_loss.index(min(eval_loss))])
    best_eval = float(state.get("best_metric") or min(eval_loss))

    image = Image.new("RGB", (1600, 1040), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), args.title, fill=INK, font=_font(42, bold=True))
    subtitle = args.subtitle or f"Final Trainer state · {state['global_step']:,}/{max_steps:,} steps"
    draw.text((72, 98), subtitle, fill=MUTED, font=_font(21))
    badge = "COMPLETE"
    badge_box = (1390, 48, 1530, 92)
    draw.rounded_rectangle(badge_box, 20, fill=GREEN)
    draw.text((1411, 59), badge, fill=WHITE, font=_font(18, bold=True))

    cards = [
        ("Best validation", f"{best_eval:.5f}"),
        ("Best checkpoint", f"step {best_step:,}"),
        ("Test loss", f"{test_loss:.5f}" if test_loss is not None else "n/a"),
    ]
    for index, (label, value) in enumerate(cards):
        x0 = 810 + index * 230
        draw.text((x0, 98), label, fill=MUTED, font=_font(16))
        draw.text((x0, 121), value, fill=INK, font=_font(24, bold=True))

    x_ticks = sorted(set([0, *eval_steps, max_steps]))
    top_box = (135, 210, 1525, 535)
    x_top, y_top = _draw_panel(
        draw,
        top_box,
        x_max=max_steps,
        x_ticks=x_ticks,
        y_ticks=(0.2, 0.3, 0.5, 1.0, 2.0),
        y_min=0.17,
        y_max=max(2.5, max(train_loss) * 1.05),
        logarithmic=True,
        title="Full run · loss (log scale)",
    )
    raw_points = [(x_top(step), y_top(loss)) for step, loss in zip(train_steps, train_loss)]
    smooth_points = [(x_top(step), y_top(loss)) for step, loss in zip(train_steps, smooth_loss)]
    eval_points = [(x_top(step), y_top(loss)) for step, loss in zip(eval_steps, eval_loss)]
    _line(draw, raw_points, fill=BLUE_LIGHT, width=2)
    _line(draw, smooth_points, fill=BLUE, width=5)
    _line(draw, eval_points, fill=ORANGE, width=4)
    for px, py in eval_points:
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=ORANGE, outline=WHITE, width=2)

    bottom_box = (135, 655, 1525, 930)
    x_bottom, y_bottom = _draw_panel(
        draw,
        bottom_box,
        x_max=max_steps,
        x_ticks=x_ticks,
        y_ticks=(0.18, 0.22, 0.26, 0.30, 0.34),
        y_min=0.17,
        y_max=0.34,
        logarithmic=False,
        title="Convergence detail · loss (linear scale)",
    )
    zoom_train = [(step, loss) for step, loss in zip(train_steps, train_loss) if step >= eval_steps[0]]
    zoom_smooth = [(step, loss) for step, loss in zip(train_steps, smooth_loss) if step >= eval_steps[0]]
    _line(draw, [(x_bottom(s), y_bottom(v)) for s, v in zoom_train], fill=BLUE_LIGHT, width=2)
    _line(draw, [(x_bottom(s), y_bottom(v)) for s, v in zoom_smooth], fill=BLUE, width=5)
    bottom_eval = [(x_bottom(step), y_bottom(loss)) for step, loss in zip(eval_steps, eval_loss)]
    _line(draw, bottom_eval, fill=ORANGE, width=4)
    for (px, py), value in zip(bottom_eval, eval_loss):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=ORANGE, outline=WHITE, width=2)
        label = f"{value:.4f}"
        bbox = draw.textbbox((0, 0), label, font=_font(16, bold=True))
        label_x = min(px + 10, bottom_box[2] - (bbox[2] - bbox[0]))
        draw.text((label_x, py - 27), label, fill=ORANGE, font=_font(16, bold=True))

    legend_y = 984
    draw.line((110, legend_y, 160, legend_y), fill=BLUE_LIGHT, width=3)
    draw.text((170, legend_y - 11), "Train loss (every 10 steps)", fill=MUTED, font=_font(17))
    draw.line((450, legend_y, 500, legend_y), fill=BLUE, width=6)
    draw.text((510, legend_y - 11), "Train trailing mean (5 points)", fill=MUTED, font=_font(17))
    draw.line((845, legend_y, 895, legend_y), fill=ORANGE, width=5)
    draw.ellipse((866, legend_y - 7, 880, legend_y + 7), fill=ORANGE)
    draw.text((905, legend_y - 11), "Validation loss (epoch end)", fill=MUTED, font=_font(17))
    draw.text((1335, legend_y - 11), "Optimizer step", fill=MUTED, font=_font(17))

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".summary.json")
    image.save(png_path, format="PNG", optimize=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "epoch", "split", "loss", "grad_norm", "learning_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        for item in train:
            writer.writerow(
                {
                    "step": item["step"],
                    "epoch": item.get("epoch"),
                    "split": "train",
                    "loss": item["loss"],
                    "grad_norm": item.get("grad_norm"),
                    "learning_rate": item.get("learning_rate"),
                }
            )
        for item in validation:
            writer.writerow(
                {
                    "step": item["step"],
                    "epoch": item.get("epoch"),
                    "split": "validation",
                    "loss": item["eval_loss"],
                    "grad_norm": "",
                    "learning_rate": "",
                }
            )
        for item in tests[-1:]:
            writer.writerow(
                {
                    "step": item["step"],
                    "epoch": item.get("epoch"),
                    "split": "test",
                    "loss": item["test_loss"],
                    "grad_norm": "",
                    "learning_rate": "",
                }
            )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.trainer_state.resolve()),
        "source_sha256": _sha256(args.trainer_state),
        "status": "complete" if state["global_step"] >= max_steps else "incomplete",
        "global_step": state["global_step"],
        "max_steps": max_steps,
        "epoch": state.get("epoch"),
        "train_points": len(train),
        "validation_points": len(validation),
        "validation_curve": [
            {"step": step, "loss": loss} for step, loss in zip(eval_steps, eval_loss)
        ],
        "best_global_step": best_step,
        "best_validation_loss": best_eval,
        "test_loss": test_loss,
        "aggregate_train_loss": aggregate_train_loss,
        "last_10_logged_train_loss_mean": mean(train_loss[-10:]),
        "validation_improvement_first_to_best_percent": 100.0
        * (eval_loss[0] - best_eval)
        / eval_loss[0],
        "outputs": {"png": str(png_path.resolve()), "csv": str(csv_path.resolve())},
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
