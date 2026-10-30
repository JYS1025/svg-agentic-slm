"""Run one pinned Hugging Face OmniSVG-3B text-to-SVG inference sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

MODEL_ID = "OmniSVG/OmniSVG"
MODEL_REVISION = "40138c74a343eae6c0d1149e0dbe95b163db0665"
BASE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
UPSTREAM_COMMIT = "0d1e5e4e8516c341d432de5f211c0afdbecfb773"
SYSTEM_PROMPT = (
    "You are a multimodal SVG generation assistant capable of generating SVG code "
    "from both text descriptions and images."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_record(path: Path, index: int) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"No input record at zero-based index {index}")


def render(svg: str, destination: Path) -> None:
    import cairosvg

    root = ElementTree.fromstring(svg)
    if not root.tag.endswith("svg"):
        raise ValueError("Generated document is not SVG")
    for element in root.iter():
        if element.tag.split("}")[-1] in {"script", "foreignObject", "image"}:
            raise ValueError("SVG contains unsupported active or external content")
        for attr, value in element.attrib.items():
            if attr.split("}")[-1] == "href" and not value.startswith("#"):
                raise ValueError("SVG contains an external reference")
            if "url(" in value and "url(#" not in value:
                raise ValueError("SVG contains an external URL")
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=512,
        output_height=512,
        background_color="#ffffff",
        unsafe=False,
    )
    destination.write_bytes(png)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import transformers
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this smoke test")
    upstream = args.upstream.resolve()
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"OmniSVG source commit mismatch: {commit}")
    sys.path.insert(0, str(upstream))
    from tokenizer import SVGTokenizer  # type: ignore[import-not-found]

    record = load_record(args.input_manifest, args.sample_index)
    instruction = str(record["instruction"])
    if not instruction.strip():
        raise ValueError("Selected instruction is empty")
    reference_path = args.input_manifest.parent / str(record["reference_svg_path"])
    if sha256_file(reference_path) != record["reference_svg_sha256"]:
        raise ValueError("Reference SVG hash mismatch")
    if args.output_dir.exists():
        raise FileExistsError(f"Smoke output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"
    start = time.monotonic()
    try:
        model_dir = Path(
            snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        )
        base_dir = Path(snapshot_download(BASE_ID, revision=BASE_REVISION, local_files_only=True))
        config_path = Path(
            hf_hub_download(MODEL_ID, "config.yaml", revision=MODEL_REVISION, local_files_only=True)
        )
        weight_path = model_dir / "pytorch_model.bin"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_config = config["model"]
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:0")

        processor = AutoProcessor.from_pretrained(
            base_dir, local_files_only=True, trust_remote_code=False, use_fast=False
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Task: text-to-svg\nDescription: {instruction}\n"
                            "Generate SVG code based on the above description."
                        ),
                    }
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        )
        # Every added embedding is overwritten by the OmniSVG checkpoint. Avoid
        # the expensive mean/covariance initialization of the temporary rows.
        model.resize_token_embeddings(196042, mean_resizing=False)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("OmniSVG checkpoint has no state dictionary")
        prefix = "transformer."
        if not all(key.startswith(prefix) for key in state):
            raise ValueError("Unexpected OmniSVG checkpoint key prefix")
        state = {key[len(prefix) :]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        del state
        model.config.bos_token_id = int(model_config["bos_token_id"])
        model.config.eos_token_id = int(model_config["eos_token_id"])
        model.config.pad_token_id = int(model_config["pad_token_id"])
        model.to(device).eval()

        max_new_tokens = args.max_new_tokens or int(model_config["max_length"])
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,
                use_cache=True,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )
        body = generated[0, input_ids.shape[1] :].cpu().tolist()
        eos = int(model_config["eos_token_id"])
        if eos in body:
            body = body[: body.index(eos)]
        (args.output_dir / "generated_token_ids.json").write_text(
            json.dumps(body) + "\n", encoding="utf-8"
        )
        wrapper = torch.tensor([[int(model_config["bos_token_id"]), *body, eos]], dtype=torch.long)
        svg_tokenizer = SVGTokenizer(str(config_path))
        pixels, colors = svg_tokenizer.process_generated_tokens(wrapper)
        paths = svg_tokenizer.raster_svg(pixels)
        if not paths or not paths[0]:
            raise ValueError("OmniSVG decoder produced no SVG paths")
        svg = svg_tokenizer.apply_colors_to_svg(paths[0], colors).to_str()
        (args.output_dir / "candidate.svg").write_text(svg, encoding="utf-8")
        render(svg, args.output_dir / "candidate.png")
        reference_svg = reference_path.read_text(encoding="utf-8")
        (args.output_dir / "reference.svg").write_text(reference_svg, encoding="utf-8")
        render(reference_svg, args.output_dir / "reference.png")
        result: dict[str, object] = {
            "status": "generated",
            "sample_id": record["sample_id"],
            "selected_text_field": record["selected_text_field"],
            "input_manifest_sha256": sha256_file(args.input_manifest),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_model_id": BASE_ID,
            "base_model_revision": BASE_REVISION,
            "upstream_commit": commit,
            "config_sha256": sha256_file(config_path),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": len(body),
            "max_new_tokens": max_new_tokens,
            "generation": {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.05,
            },
            "render": {
                "width": 512,
                "height": 512,
                "background": "#ffffff",
                "renderer": "CairoSVG",
            },
            "candidate_svg_sha256": sha256_file(args.output_dir / "candidate.svg"),
            "candidate_png_sha256": sha256_file(args.output_dir / "candidate.png"),
            "reference_svg_sha256": sha256_file(args.output_dir / "reference.svg"),
            "reference_png_sha256": sha256_file(args.output_dir / "reference.png"),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "sample_id": record["sample_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        status_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
    status_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path(".work/omnisvg_critic_distill/inputs/records.jsonl"),
    )
    parser.add_argument(
        "--upstream", type=Path, default=Path(".work/omnisvg_critic_distill/upstream")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".work/omnisvg_critic_distill/runs/smoke_0000")
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
