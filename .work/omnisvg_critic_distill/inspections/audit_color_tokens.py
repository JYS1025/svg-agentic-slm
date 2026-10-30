"""Snapshot generated OmniSVG outputs and copy missing-color examples."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import yaml
from huggingface_hub import hf_hub_download
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / ".work/omnisvg_critic_distill"
RUN = WORK / "runs/full_10k"
INPUTS = WORK / "inputs/records.jsonl"
OUT_PARENT = WORK / "inspections"
MODEL_REVISION = "40138c74a343eae6c0d1149e0dbe95b163db0665"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_latest() -> tuple[dict[str, dict], dict[str, dict]]:
    latest: dict[str, dict] = {}
    shards: dict[str, dict] = {}
    for path in sorted(RUN.glob("results.gpu*.jsonl")):
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
        shards[path.name] = {"captured_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for line in data.splitlines():
            row = json.loads(line)
            latest[row["sample_id"]] = row
    return latest, shards


def is_blank_white(path: Path) -> bool:
    with Image.open(path) as image:
        return image.convert("RGB").getextrema() == ((255, 255), (255, 255), (255, 255))


def evenly_spaced(rows: list[dict], n: int) -> list[dict]:
    ordered = sorted(rows, key=lambda r: r["sample_index"])
    if len(ordered) <= n:
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]


def main() -> None:
    config_path = hf_hub_download(
        "OmniSVG/OmniSVG", "config.yaml", revision=MODEL_REVISION, local_files_only=True
    )
    colors = yaml.safe_load(Path(config_path).read_text())["colors"]
    low = int(colors["color_start_offset"])
    high = int(colors["color_end_offset"])
    latest, shards = load_latest()
    captured_at = datetime.now(timezone.utc).isoformat()
    counts = Counter()
    analyzed: list[dict] = []

    for row in latest.values():
        counts[f"status_{row['status']}"] += 1
        if row["status"] != "generated":
            continue
        svg_path = RUN / row["candidate_svg_path"]
        png_path = RUN / row["candidate_png_path"]
        token_path = RUN / row["generated_token_ids_path"]
        assert digest(svg_path) == row["candidate_svg_sha256"]
        assert digest(png_path) == row["candidate_png_sha256"]
        token_ids = json.loads(token_path.read_text())
        color_ids = [token for token in token_ids if low <= token < high]
        paths = [el for el in ElementTree.parse(svg_path).getroot().iter() if el.tag.endswith("}path")]
        blank = is_blank_white(png_path)
        item = {
            "sample_id": row["sample_id"],
            "sample_index": row["sample_index"],
            "domain": row["domain"],
            "selected_text_field": row["selected_text_field"],
            "candidate_svg_path": row["candidate_svg_path"],
            "candidate_png_path": row["candidate_png_path"],
            "generated_token_ids_path": row["generated_token_ids_path"],
            "generated_token_count": len(token_ids),
            "color_token_count": len(color_ids),
            "candidate_blank_white": blank,
            "svg_path_count": len(paths),
            "all_svg_paths_fill_none": bool(paths) and all(
                el.get("fill") == "none" for el in paths
            ),
        }
        analyzed.append(item)
        counts["generated"] += 1
        if not color_ids:
            counts["zero_color_tokens"] += 1
            if blank:
                counts["zero_color_and_blank"] += 1
            else:
                counts["zero_color_but_visible"] += 1
        elif blank:
            counts["has_color_but_blank"] += 1
        if blank:
            counts["candidate_blank_white"] += 1
        if item["all_svg_paths_fill_none"]:
            counts["all_svg_paths_fill_none"] += 1
    assert counts["generated"] > 0

    name = datetime.now(timezone.utc).strftime("color_token_snapshot_%Y%m%dT%H%M%SZ")
    output = OUT_PARENT / name
    output.mkdir(parents=True, exist_ok=False)
    missing = [row for row in analyzed if row["color_token_count"] == 0 and row["candidate_blank_white"]]
    normal = [row for row in analyzed if row["color_token_count"] > 0 and not row["candidate_blank_white"]]
    examples = {
        "missing_color": evenly_spaced(missing, 5),
        "has_color_visible": evenly_spaced(normal, 5),
    }
    selected_ids = {row["sample_id"] for group in examples.values() for row in group}
    records = {}
    with INPUTS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["sample_id"] in selected_ids:
                records[record["sample_id"]] = record

    readme = [
        "# OmniSVG color-token audit",
        "",
        f"Snapshot: {captured_at}",
        f"Generated outputs: {counts['generated']} (failed statuses excluded)",
        f"No color token + blank render: {counts['zero_color_and_blank']} "
        f"({counts['zero_color_and_blank'] / counts['generated'] * 100:.2f}%)",
        f"All blank candidate renders: {counts['candidate_blank_white']} "
        f"({counts['candidate_blank_white'] / counts['generated'] * 100:.2f}%)",
        "",
        "`missing_color/`: no model color token; upstream decoder assigns fill=none.",
        "`has_color_visible/`: one or more color tokens and a nonblank candidate PNG.",
        "Each example contains the exact generated SVG, rendered PNG, generated token IDs,",
        "selected text instruction, and metadata. Source artifacts are copied unchanged.",
        "",
    ]
    for group, chosen in examples.items():
        readme.extend([f"## {group}", ""])
        for number, row in enumerate(chosen, 1):
            folder = output / group / f"{number:02d}_index_{row['sample_index']:05d}"
            folder.mkdir(parents=True)
            shutil.copy2(RUN / row["candidate_svg_path"], folder / "candidate.svg")
            shutil.copy2(RUN / row["candidate_png_path"], folder / "candidate.png")
            shutil.copy2(RUN / row["generated_token_ids_path"], folder / "generated_token_ids.json")
            (folder / "prompt.txt").write_text(records[row["sample_id"]]["instruction"] + "\n")
            (folder / "metadata.json").write_text(json.dumps(row, indent=2) + "\n")
            readme.append(
                f"- [{folder.name}/{row['sample_id']}]({group}/{folder.name}/candidate.svg) "
                f"— color tokens: {row['color_token_count']}; "
                f"[PNG]({group}/{folder.name}/candidate.png)"
            )
        readme.append("")
    report = {
        "captured_at": captured_at,
        "source_run": str(RUN),
        "source_shards": shards,
        "color_token_id_range_inclusive": [low, high - 1],
        "counts": dict(counts),
        "zero_color_and_blank_percent": round(
            counts["zero_color_and_blank"] / counts["generated"] * 100, 4
        ),
        "all_blank_candidate_percent": round(
            counts["candidate_blank_white"] / counts["generated"] * 100, 4
        ),
        "examples": examples,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps({"output": str(output), "counts": dict(counts),
                      "zero_color_and_blank_percent": report["zero_color_and_blank_percent"],
                      "all_blank_candidate_percent": report["all_blank_candidate_percent"]}, indent=2))


if __name__ == "__main__":
    main()
