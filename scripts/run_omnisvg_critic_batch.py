"""Resumable, two-GPU OmniSVG-3B text-to-SVG inference for critic inputs."""

# ruff: noqa: UP017 -- this runner uses Python 3.10; datetime.UTC needs 3.11.

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from run_omnisvg_critic_smoke import (
    BASE_ID,
    BASE_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
    UPSTREAM_COMMIT,
    render,
    sha256_file,
)

DEFAULT_INPUT = Path(".work/omnisvg_critic_distill/inputs/records.jsonl")
DEFAULT_UPSTREAM = Path(".work/omnisvg_critic_distill/upstream")
DEFAULT_OUTPUT = Path(".work/omnisvg_critic_distill/runs/full_10k")
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def input_records(path: Path, start: int, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    end = None if limit is None else start + limit
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if end is not None and index >= end:
                break
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("sample_id"):
                raise ValueError(f"Invalid input at line {index + 1}")
            value["sample_index"] = index
            records.append(value)
    if not records or (limit is not None and len(records) != limit):
        raise ValueError(f"Expected {limit or 'at least one'} inputs from index {start}")
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Input sample IDs are not unique")
    return records


def prompt_text(instruction: str) -> str:
    return (
        f"Task: text-to-svg\nDescription: {instruction}\n"
        "Generate SVG code based on the above description."
    )


def chat_prompt(processor: Any, instruction: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt_text(instruction)}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def stable_stem(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def relative_artifacts(sample_id: str, attempt: int) -> Path:
    stem = stable_stem(sample_id)
    return Path("artifacts") / stem[:2] / stem / f"attempt-{attempt:03d}"


def latest_results(run_dir: Path, gpu: int) -> dict[str, dict[str, Any]]:
    path = run_dir / f"results.gpu{gpu}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                # A status reader can race with the one writer for this shard.
                # The worker repairs this final fragment before appending on resume.
                break
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"Missing sample ID at {path}:{line_number}")
            latest[sample_id] = row
    return latest


def repair_result_tail(path: Path) -> None:
    """Drop an incomplete final JSONL line after an interrupted write."""
    if not path.exists():
        return
    with path.open("rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if not size:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        content = handle.read()
        boundary = content.rfind(b"\n") + 1
        atomic_bytes(path.with_name(f"{path.name}.partial.{int(time.time())}"), content[boundary:])
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())


def generated_artifacts_valid(run_dir: Path, row: dict[str, Any], input_dir: Path) -> bool:
    if row.get("status") != "generated":
        return False
    try:
        for key in ("candidate_svg", "candidate_png", "reference_png"):
            path = run_dir / row[f"{key}_path"]
            if not path.is_file() or sha256_file(path) != row[f"{key}_sha256"]:
                return False
        reference = input_dir / row["reference_svg_path"]
        return reference.is_file() and sha256_file(reference) == row["reference_svg_sha256"]
    except (KeyError, OSError):
        return False


def settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    input_path = args.input_manifest.resolve()
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}, found {commit}")
    if subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("Pinned upstream checkout has tracked modifications")
    config_path = Path(
        hf_hub_download(MODEL_ID, "config.yaml", revision=MODEL_REVISION, local_files_only=True)
    )
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(value < 0 for value in gpus):
        raise ValueError("--gpus must be unique nonnegative comma-separated indices")
    return {
        "schema_version": SCHEMA_VERSION,
        "input_manifest": str(input_path),
        "input_sha256": sha256_file(input_path),
        "start_index": args.start_index,
        "limit": args.limit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model_id": BASE_ID,
        "base_model_revision": BASE_REVISION,
        "upstream": str(upstream),
        "upstream_commit": commit,
        "model_config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "smoke_helpers_sha256": sha256_file(
            Path(__file__).with_name("run_omnisvg_critic_smoke.py")
        ),
        "gpus": gpus,
        "batch_size": args.batch_size,
        "render_workers": args.render_workers,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "attention": "sdpa",
            "dtype": "bfloat16",
        },
        "render": {"width": 512, "height": 512, "background": "#ffffff", "renderer": "CairoSVG"},
    }


def planned_for_gpu(
    records: list[dict[str, Any]], gpus: list[int], gpu: int, input_dir: Path
) -> list[dict[str, Any]]:
    position = gpus.index(gpu)
    own = [row for index, row in enumerate(records) if index % len(gpus) == position]
    # Similar reference SVG sizes often need similar output lengths, reducing
    # padding waste during batched autoregressive decoding. The order is fixed
    # so batch membership and sampling seeds stay stable on resume.
    return sorted(
        own,
        key=lambda row: (
            row["domain"],
            (input_dir / row["reference_svg_path"]).stat().st_size,
            len(row["instruction"]),
            row["sample_index"],
        ),
    )


def progress_snapshot(
    run_dir: Path,
    gpu: int,
    planned: int,
    latest: dict[str, dict[str, Any]],
    phase: str,
    batch_id: int | None = None,
    model_ready_at: str | None = None,
    completed_at_model_ready: int | None = None,
) -> None:
    counts = Counter(row["status"] for row in latest.values())
    atomic_json(
        run_dir / f"progress.gpu{gpu}.json",
        {
            "gpu": gpu,
            "phase": phase,
            "planned": planned,
            "completed": len(latest),
            "generated": counts["generated"],
            "failed": len(latest) - counts["generated"],
            "batch_id": batch_id,
            "model_ready_at": model_ready_at,
            "completed_at_model_ready": completed_at_model_ready,
            "updated_at": utc_now(),
        },
    )


_POSTPROCESS_TOKENIZER: Any = None
_POSTPROCESS_CONFIG: dict[str, Any] | None = None


def init_postprocess(upstream: str, config_path: str, config: dict[str, Any]) -> None:
    """Initialize the pinned SVG decoder once in each CPU worker."""
    global _POSTPROCESS_TOKENIZER, _POSTPROCESS_CONFIG

    sys.path.insert(0, upstream)
    from tokenizer import SVGTokenizer  # type: ignore[import-not-found]

    _POSTPROCESS_TOKENIZER = SVGTokenizer(config_path)
    _POSTPROCESS_CONFIG = config


def postprocess_ready() -> bool:
    return _POSTPROCESS_TOKENIZER is not None and _POSTPROCESS_CONFIG is not None


def postprocess_artifacts(
    body: list[int], reference_path: str, artifact_dir: str
) -> dict[str, Any]:
    """Decode OmniSVG tokens and render both SVGs in a separate CPU process."""
    output = Path(artifact_dir)
    reference = Path(reference_path)
    payload: dict[str, Any] = {}
    try:
        candidate_svg = decode_tokens(body, _POSTPROCESS_TOKENIZER, _POSTPROCESS_CONFIG)
        atomic_bytes(output / "candidate.svg", candidate_svg.encode("utf-8"))
        payload["candidate_svg_sha256"] = sha256_file(output / "candidate.svg")
    except Exception as exc:
        payload.update(status="decode_failed", error_type=type(exc).__name__, error=str(exc))
        return payload
    try:
        render(candidate_svg, output / "candidate.png")
        payload["candidate_png_sha256"] = sha256_file(output / "candidate.png")
    except Exception as exc:
        payload.update(
            status="candidate_render_failed", error_type=type(exc).__name__, error=str(exc)
        )
        return payload
    try:
        render(reference.read_text(encoding="utf-8"), output / "reference.png")
        payload["reference_png_sha256"] = sha256_file(output / "reference.png")
    except Exception as exc:
        payload.update(
            status="reference_render_failed", error_type=type(exc).__name__, error=str(exc)
        )
        return payload
    payload["reference_svg_sha256"] = sha256_file(reference)
    payload["status"] = "generated"
    return payload


def load_runtime(settings: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], Path]:
    import torch
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This worker requires a CUDA GPU with BF16 support")
    model_dir = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    base_dir = Path(snapshot_download(BASE_ID, revision=BASE_REVISION, local_files_only=True))
    config_path = Path(
        hf_hub_download(MODEL_ID, "config.yaml", revision=MODEL_REVISION, local_files_only=True)
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    processor = AutoProcessor.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=False, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.resize_token_embeddings(196042, mean_resizing=False)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(k.startswith("transformer.") for k in state)
    ):
        raise ValueError("Unexpected OmniSVG checkpoint state keys")
    model.load_state_dict({k[len("transformer.") :]: v for k, v in state.items()}, strict=True)
    del state
    model.config.bos_token_id = int(config["bos_token_id"])
    model.config.eos_token_id = int(config["eos_token_id"])
    model.config.pad_token_id = int(config["pad_token_id"])
    model.to("cuda:0").eval()
    return processor, model, config, config_path


def decode_tokens(body: list[int], svg_tokenizer: Any, config: dict[str, Any]) -> str:
    import torch

    wrapper = torch.tensor(
        [[int(config["bos_token_id"]), *body, int(config["eos_token_id"])]], dtype=torch.long
    )
    pixels, colors = svg_tokenizer.process_generated_tokens(wrapper)
    paths = svg_tokenizer.raster_svg(pixels)
    if not paths or not paths[0]:
        raise ValueError("OmniSVG decoder produced no SVG paths")
    return svg_tokenizer.apply_colors_to_svg(paths[0], colors).to_str()


def generate_batch(
    rows: list[dict[str, Any]],
    batch_seed: int,
    processor: Any,
    model: Any,
    config: dict[str, Any],
    max_new_tokens: int,
) -> tuple[list[list[int]], int, float]:
    import torch

    prompts = [chat_prompt(processor, str(row["instruction"])) for row in rows]
    inputs = processor(text=prompts, padding=True, truncation=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")
    torch.manual_seed(batch_seed)
    torch.cuda.manual_seed_all(batch_seed)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            use_cache=True,
            bos_token_id=int(config["bos_token_id"]),
            eos_token_id=int(config["eos_token_id"]),
            pad_token_id=int(config["pad_token_id"]),
        )
    elapsed = time.monotonic() - started
    if len(output) != len(rows):
        raise ValueError("Model output batch size does not match input batch size")
    bodies: list[list[int]] = []
    eos = int(config["eos_token_id"])
    for sequence in output[:, input_ids.shape[1] :].cpu().tolist():
        if eos in sequence:
            sequence = sequence[: sequence.index(eos)]
        bodies.append(sequence)
    return bodies, int(input_ids.shape[1]), elapsed


def batch_with_oom_split(
    rows: list[dict[str, Any]],
    seed: int,
    processor: Any,
    model: Any,
    config: dict[str, Any],
    max_new_tokens: int,
) -> list[tuple[dict[str, Any], list[int] | None, int, float, str | None]]:
    import torch

    try:
        bodies, prompt_tokens, seconds = generate_batch(
            rows, seed, processor, model, config, max_new_tokens
        )
        return [(row, body, prompt_tokens, seconds, None) for row, body in zip(rows, bodies)]
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        if len(rows) == 1:
            return [(rows[0], None, 0, 0.0, f"CUDA out of memory: {exc}")]
        middle = len(rows) // 2
        return batch_with_oom_split(
            rows[:middle], seed * 2 + 1, processor, model, config, max_new_tokens
        ) + batch_with_oom_split(
            rows[middle:], seed * 2 + 2, processor, model, config, max_new_tokens
        )


def worker_locked(run_dir: Path, gpu: int, retry_failures: bool) -> None:
    settings = read_json(run_dir / "run_manifest.json")["settings"]
    records = input_records(
        Path(settings["input_manifest"]), settings["start_index"], settings["limit"]
    )
    input_dir = Path(settings["input_manifest"]).parent
    own = planned_for_gpu(records, settings["gpus"], gpu, input_dir)
    output_path = run_dir / f"results.gpu{gpu}.jsonl"
    repair_result_tail(output_path)
    latest = latest_results(run_dir, gpu)
    own_ids = {row["sample_id"] for row in own}
    if set(latest) - own_ids:
        raise ValueError(f"GPU {gpu} has results for samples outside its shard")
    if not own:
        progress_snapshot(run_dir, gpu, 0, latest, "finished")
        return
    repair_ids = {
        sample_id
        for sample_id, row in latest.items()
        if row["status"] == "generated" and not generated_artifacts_valid(run_dir, row, input_dir)
    }
    if repair_ids:
        print(f"gpu={gpu} reprocessing {len(repair_ids)} damaged artifacts", flush=True)
    progress_snapshot(run_dir, gpu, len(own), latest, "loading_model")
    processor, model, config, config_path = load_runtime(settings)
    model_ready_at = utc_now()
    completed_at_model_ready = len(latest)
    batch_size = int(settings["batch_size"])
    pending: dict[Any, dict[str, Any]] = {}
    failures_this_run = 0
    started = time.monotonic()

    def progress(phase: str, batch_id: int | None = None) -> None:
        progress_snapshot(
            run_dir,
            gpu,
            len(own),
            latest,
            phase,
            batch_id,
            model_ready_at,
            completed_at_model_ready,
        )

    progress("running")

    def complete(row: dict[str, Any]) -> None:
        nonlocal failures_this_run
        append_jsonl(output_path, row)
        latest[row["sample_id"]] = row
        if row["status"] == "generation_failed":
            failures_this_run += 1
        progress("running", row.get("batch_id"))

    def drain_one() -> None:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            basis = pending.pop(future)
            try:
                basis.update(future.result())
            except BrokenProcessPool as exc:
                raise RuntimeError("CPU postprocessing pool stopped unexpectedly") from exc
            except Exception as exc:
                basis.update(
                    status="postprocess_failed", error_type=type(exc).__name__, error=str(exc)
                )
            complete(basis)

    with ProcessPoolExecutor(
        max_workers=int(settings["render_workers"]),
        mp_context=get_context("spawn"),
        initializer=init_postprocess,
        initargs=(settings["upstream"], str(config_path), config),
    ) as postprocess_pool:
        # Start CPU decoder imports while the GPU computes its first batch.
        warmup = postprocess_pool.submit(postprocess_ready)
        for batch_id, offset in enumerate(range(0, len(own), batch_size)):
            batch = own[offset : offset + batch_size]
            needed = [
                row
                for row in batch
                if row["sample_id"] not in latest
                or (retry_failures and latest[row["sample_id"]]["status"] != "generated")
                or row["sample_id"] in repair_ids
            ]
            if not needed:
                continue
            if failures_this_run >= 25:
                raise RuntimeError("Stopping after 25 model generation failures on this worker")
            progress("generating", batch_id)
            # Generate the full fixed batch, even if only some rows need retry.
            # This preserves sampling RNG and output for the pending rows.
            generated = batch_with_oom_split(
                batch,
                int(settings["seed"]) + gpu * 1_000_000 + batch_id,
                processor,
                model,
                config,
                int(settings["max_new_tokens"]),
            )
            for row, body, prompt_tokens, generation_seconds, error in generated:
                if row not in needed:
                    continue
                sample_id = row["sample_id"]
                attempt = int(latest.get(sample_id, {}).get("attempt", -1)) + 1
                rel_dir = relative_artifacts(sample_id, attempt)
                artifact_dir = run_dir / rel_dir
                artifact_dir.mkdir(parents=True, exist_ok=True)
                basis: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "sample_index": row["sample_index"],
                    "domain": row["domain"],
                    "selected_text_field": row["selected_text_field"],
                    "gpu": gpu,
                    "batch_id": batch_id,
                    "attempt": attempt,
                    "seed": int(settings["seed"]) + gpu * 1_000_000 + batch_id,
                    "prompt_tokens_padded": prompt_tokens,
                    "generation_seconds_for_batch": round(generation_seconds, 3),
                    "max_new_tokens": settings["max_new_tokens"],
                    "reference_svg_path": row["reference_svg_path"],
                    "artifact_dir": str(rel_dir),
                    "created_at": utc_now(),
                }
                if error is not None or body is None:
                    basis.update(
                        status="generation_failed", error_type="CudaOutOfMemory", error=error
                    )
                    complete(basis)
                    continue
                basis["generated_tokens"] = len(body)
                atomic_bytes(
                    artifact_dir / "generated_token_ids.json", (json.dumps(body) + "\n").encode()
                )
                basis["generated_token_ids_path"] = str(rel_dir / "generated_token_ids.json")
                basis["candidate_svg_path"] = str(rel_dir / "candidate.svg")
                basis["candidate_png_path"] = str(rel_dir / "candidate.png")
                basis["reference_png_path"] = str(rel_dir / "reference.png")
                reference_path = input_dir / row["reference_svg_path"]
                future = postprocess_pool.submit(
                    postprocess_artifacts, body, str(reference_path), str(artifact_dir)
                )
                pending[future] = basis
            while len(pending) >= int(settings["render_workers"]) * 16:
                drain_one()
            elapsed = time.monotonic() - started
            print(
                f"gpu={gpu} batch={batch_id} completed={len(latest)}/{len(own)} "
                f"postprocess_queued={len(pending)} elapsed={elapsed:.1f}s",
                flush=True,
            )
        progress("postprocessing")
        while pending:
            drain_one()
        if not warmup.result():
            raise RuntimeError("CPU postprocessing initialization failed")
    progress("finished")
    print(f"gpu={gpu} finished completed={len(latest)}/{len(own)}", flush=True)


def worker(run_dir: Path, gpu: int, retry_failures: bool) -> None:
    with (run_dir / f"worker.gpu{gpu}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU {gpu} already has an active worker") from exc
        worker_locked(run_dir, gpu, retry_failures)


def status(run_dir: Path, verify: bool = False) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    settings = manifest["settings"]
    records = input_records(
        Path(settings["input_manifest"]), settings["start_index"], settings["limit"]
    )
    expected = {row["sample_id"] for row in records}
    combined: dict[str, dict[str, Any]] = {}
    workers: dict[str, Any] = {}
    for gpu in settings["gpus"]:
        latest = latest_results(run_dir, gpu)
        if set(latest) & set(combined):
            raise ValueError("The same sample appears in multiple GPU result shards")
        combined.update(latest)
        progress_path = run_dir / f"progress.gpu{gpu}.json"
        workers[str(gpu)] = (
            read_json(progress_path) if progress_path.exists() else {"phase": "waiting"}
        )
    if set(combined) - expected:
        raise ValueError("Results contain sample IDs outside the selected input set")
    counts = Counter(row["status"] for row in combined.values())
    completed = len(combined)
    started = datetime.fromisoformat(manifest["created_at"])
    summary_path = run_dir / "summary.json"
    saved_summary = read_json(summary_path) if summary_path.is_file() else None
    stop_time = (
        datetime.fromisoformat(saved_summary["finished_at"])
        if saved_summary and saved_summary.get("finished_at")
        else datetime.now(timezone.utc)
    )
    elapsed = max(0.0, (stop_time - started).total_seconds())
    rates = []
    new_completed_total = 0
    for worker_progress in workers.values():
        if worker_progress.get("phase") == "finished":
            continue
        ready_at = worker_progress.get("model_ready_at")
        baseline = worker_progress.get("completed_at_model_ready")
        if ready_at is None or baseline is None:
            continue
        duration = (datetime.now(timezone.utc) - datetime.fromisoformat(ready_at)).total_seconds()
        new_completed = worker_progress.get("completed", 0) - baseline
        new_completed_total += max(0, new_completed)
        if duration > 0 and new_completed > 0:
            rates.append(new_completed / duration)
    remaining = len(records) - completed
    initial_batch_count = min(len(records), settings["batch_size"] * len(settings["gpus"]))
    eta = (
        0
        if remaining == 0
        else round(remaining / sum(rates))
        if rates and new_completed_total >= initial_batch_count
        else None
    )
    output: dict[str, Any] = {
        "planned": len(records),
        "completed": completed,
        "remaining": remaining,
        "generated": counts["generated"],
        "failed": completed - counts["generated"],
        "by_status": dict(counts),
        "elapsed_seconds": round(elapsed),
        "eta_seconds_rough": eta,
        "workers": workers,
        "finished": saved_summary is not None,
    }
    if verify:
        if completed != len(records):
            raise ValueError(f"Only {completed}/{len(records)} samples have terminal results")
        for row in combined.values():
            if row["status"] != "generated":
                continue
            if not generated_artifacts_valid(run_dir, row, Path(settings["input_manifest"]).parent):
                raise ValueError(f"Missing or corrupt artifacts for {row['sample_id']}")
    return output


def finalize(run_dir: Path) -> dict[str, Any]:
    summary = status(run_dir, verify=True)
    settings = read_json(run_dir / "run_manifest.json")["settings"]
    latest: dict[str, dict[str, Any]] = {}
    for gpu in settings["gpus"]:
        latest.update(latest_results(run_dir, gpu))
    ordered = sorted(latest.values(), key=lambda row: row["sample_index"])
    merged = run_dir / "results.jsonl"
    atomic_bytes(
        merged,
        b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in ordered
        ),
    )
    results_hash = sha256_file(merged)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous.get("results_sha256") == results_hash:
            return previous
    summary["results_sha256"] = results_hash
    summary["finished_at"] = utc_now()
    summary["finished"] = True
    atomic_json(summary_path, summary)
    return summary


def compact_status(snapshot: dict[str, Any]) -> str:
    phases = []
    for gpu, worker in snapshot["workers"].items():
        batch = f"[batch {worker['batch_id']}]" if worker.get("batch_id") is not None else ""
        phases.append(
            f"gpu{gpu}={worker['phase']}{batch}"
            f"({worker.get('completed', 0)}/{worker.get('planned', '?')})"
        )
    worker_phases = " ".join(phases)
    eta = snapshot["eta_seconds_rough"]
    eta_text = "pending" if eta is None else f"{eta}s"
    return (
        f"{utc_now()} {snapshot['completed']}/{snapshot['planned']} complete "
        f"generated={snapshot['generated']} failed={snapshot['failed']} "
        f"ETA~{eta_text} {worker_phases}"
    )


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.render_workers < 1 or args.start_index < 0:
        raise ValueError("Batch size and render workers must be positive; start index nonnegative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be positive")
    if args.max_new_tokens < 1 or args.max_new_tokens > 1537:
        raise ValueError("max-new-tokens must be in the checkpoint's supported 1..1537 range")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another coordinator owns {run_dir}") from exc
        settings = settings_from_args(args)
        selected = input_records(Path(settings["input_manifest"]), args.start_index, args.limit)
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("settings") != settings:
                raise ValueError("Run settings changed; choose a new run directory")
        else:
            atomic_json(
                manifest_path,
                {"created_at": utc_now(), "planned": len(selected), "settings": settings},
            )
        summary_path = run_dir / "summary.json"
        if not args.retry_failures:
            try:
                existing = status(run_dir, verify=True)
            except ValueError:
                pass
            else:
                if existing["completed"] == len(selected):
                    print(json.dumps(finalize(run_dir), indent=2))
                    return
        summary_path.unlink(missing_ok=True)
        processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda _signal, _frame: sys.exit(143))
        try:
            for gpu in settings["gpus"]:
                log_path = run_dir / f"gpu{gpu}.log"
                log = log_path.open("ab", buffering=0)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--run-dir",
                    str(run_dir),
                    "--gpu",
                    str(gpu),
                ]
                if args.retry_failures:
                    command.append("--retry-failures")
                process = subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
                processes.append((gpu, process, log))
            while any(process.poll() is None for _, process, _ in processes):
                errors = [
                    (gpu, process.returncode)
                    for gpu, process, _ in processes
                    if process.poll() is not None and process.returncode
                ]
                if errors:
                    raise RuntimeError(
                        f"GPU worker failure {errors}; inspect gpu*.log and rerun to resume"
                    )
                snapshot = status(run_dir)
                print(
                    f"{snapshot['completed']}/{snapshot['planned']} complete "
                    f"({snapshot['generated']} generated, {snapshot['failed']} failed), "
                    f"ETA~{snapshot['eta_seconds_rough']}s"
                    if snapshot["eta_seconds_rough"] is not None
                    else "ETA pending",
                    flush=True,
                )
                time.sleep(args.progress_interval)
            errors = [
                (gpu, process.returncode) for gpu, process, _ in processes if process.returncode
            ]
            if errors:
                raise RuntimeError(
                    f"GPU worker failure {errors}; inspect gpu*.log and rerun to resume"
                )
            print(json.dumps(finalize(run_dir), indent=2))
        except BaseException:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process, _ in processes:
                process.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            for _, _, log in processes:
                log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run", help="start or resume sharded inference")
    run_parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    run_parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--gpus", default="0,1")
    run_parser.add_argument("--batch-size", type=int, default=16)
    run_parser.add_argument("--render-workers", type=int, default=2)
    run_parser.add_argument("--max-new-tokens", type=int, default=1537)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--start-index", type=int, default=0)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--retry-failures", action="store_true")
    run_parser.add_argument("--progress-interval", type=float, default=15)
    worker_parser = subcommands.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--run-dir", type=Path, required=True)
    worker_parser.add_argument("--gpu", type=int, required=True)
    worker_parser.add_argument("--retry-failures", action="store_true")
    status_parser = subcommands.add_parser("status", help="show live progress")
    status_parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    status_parser.add_argument("--watch", type=float, default=0)
    status_parser.add_argument("--compact", action="store_true", help="one line per update")
    audit_parser = subcommands.add_parser("audit", help="verify and merge terminal results")
    audit_parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "worker":
        worker(args.run_dir.resolve(), args.gpu, args.retry_failures)
    elif args.command == "audit":
        print(json.dumps(finalize(args.run_dir.resolve()), indent=2))
    else:
        while True:
            snapshot = status(args.run_dir.resolve())
            print(
                compact_status(snapshot) if args.compact else json.dumps(snapshot, indent=2),
                flush=True,
            )
            if args.watch <= 0 or (args.run_dir / "summary.json").exists():
                break
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
