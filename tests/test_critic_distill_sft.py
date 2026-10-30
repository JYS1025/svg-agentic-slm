from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from svg_agentic_slm.train.critic_distill_data import (
    _split_rows,
    _validate_candidate_hash_split_disjointness,
)
from svg_agentic_slm.train.critic_multimodal_cache import (
    CriticMultimodalCollator,
    load_or_build_critic_cache,
)
from svg_agentic_slm.train.critic_training_logging import (
    DetailedLoggingCallback,
    collapse_resumed_metrics,
)
from svg_agentic_slm.train.train_critic_distill import (
    _adapter_targets_match,
    resolve_qwen_language_lora_targets,
)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.process_calls = 0
        self.texts = []
        self.template_texts = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert tokenize is False
        assert kwargs["enable_thinking"] is False
        self.template_texts.extend(
            item["text"]
            for message in messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if item.get("type") == "text"
        )
        if add_generation_prompt:
            return "PREFIX"
        return "PREFIX" + messages[-1]["content"] + "EOS"

    def __call__(self, *, text, images, return_tensors):
        assert return_tensors == "pt"
        assert images.size == (4, 4)
        self.process_calls += 1
        self.texts.append(text)
        ids = torch.tensor([[ord(character) for character in text]], dtype=torch.long)
        return {
            "input_ids": ids,
            "mm_token_type_ids": torch.ones_like(ids),
            "pixel_values": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


def _cache_record(image_path: Path, sample_id: str) -> dict[str, object]:
    import hashlib

    return {
        "sample_id": sample_id,
        "instruction": "Draw a circle.",
        "labeled_svg": '<svg><circle id="s0001"/></svg>',
        "allowed_target_ids": ["s0001"],
        "similarity_score": 0.625,
        "target_json": '{"evaluations":[],"issues":[]}',
        "candidate_image_path": str(image_path),
        "candidate_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }


def test_multimodal_cache_warm_hit_does_not_run_processor(tmp_path: Path) -> None:
    image_path = tmp_path / "candidate.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    split_paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(_cache_record(image_path, split)) + "\n")
        split_paths[split] = path
    processor = _Processor()
    datasets, manifest = load_or_build_critic_cache(
        processor=processor,
        split_paths=split_paths,
        cache_root=tmp_path / "cache",
        model_id="Qwen/Qwen3.5-4B",
        revision="a" * 40,
        max_seq_length=1024,
    )
    assert processor.process_calls == 6
    assert any("<auxiliary_siglip2_score>" in text for text in processor.template_texts)
    assert any("Score: 0.625000" in text for text in processor.template_texts)
    feature = datasets["train"][0]
    assert feature["pixel_values"].dtype == torch.bfloat16
    assert feature["labels"][: len("PREFIX")].tolist() == [-100] * len("PREFIX")
    assert feature["labels"][len("PREFIX")] != -100
    batch = CriticMultimodalCollator(0)([feature, feature])
    assert batch["pixel_values"].shape == (4, 3)
    assert batch["image_grid_thw"].shape == (2, 3)

    warm_processor = _Processor()
    warm, warm_manifest = load_or_build_critic_cache(
        processor=warm_processor,
        split_paths=split_paths,
        cache_root=tmp_path / "cache",
        model_id="Qwen/Qwen3.5-4B",
        revision="a" * 40,
        max_seq_length=1024,
    )
    assert warm_processor.process_calls == 0
    assert warm_manifest["fingerprint"] == manifest["fingerprint"]
    assert warm["test"][0]["input_ids"].tolist() == datasets["test"][0]["input_ids"].tolist()


def test_split_preserves_rare_passes_without_duplication() -> None:
    rows = [
        {
            "candidate_svg_sha256": f"{index:064x}",
            "sample_index": index,
            "passes_threshold": index < 11,
        }
        for index in range(211)
    ]
    splits = _split_rows(rows, seed=42)
    pass_counts = {
        key: sum(row["passes_threshold"] for row in value)
        for key, value in splits.items()
    }
    assert pass_counts == {
        "train": 9,
        "validation": 1,
        "test": 1,
    }
    identities = [id(row) for values in splits.values() for row in values]
    assert len(identities) == len(rows) == len(set(identities))


def test_repeated_candidate_hash_is_allowed_within_one_split() -> None:
    repeated = "a" * 64
    splits = {
        "train": [
            {"candidate_svg_sha256": repeated},
            {"candidate_svg_sha256": repeated},
        ],
        "validation": [{"candidate_svg_sha256": "b" * 64}],
        "test": [{"candidate_svg_sha256": "c" * 64}],
    }

    _validate_candidate_hash_split_disjointness(splits)


def test_repeated_candidate_hash_cannot_cross_splits() -> None:
    repeated = "a" * 64
    splits = {
        "train": [{"candidate_svg_sha256": repeated}],
        "validation": [{"candidate_svg_sha256": repeated}],
        "test": [],
    }

    with pytest.raises(ValueError, match="cross dataset split boundaries"):
        _validate_candidate_hash_split_disjointness(splits)


def test_qwen_lora_target_resolver_covers_hybrid_language_only() -> None:
    class Hybrid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})
            self.language_model = torch.nn.ModuleDict(
                {
                    "q_proj": torch.nn.Linear(4, 4),
                    "k_proj": torch.nn.Linear(4, 4),
                    "v_proj": torch.nn.Linear(4, 4),
                    "o_proj": torch.nn.Linear(4, 4),
                    "in_proj_qkv": torch.nn.Linear(4, 4),
                    "in_proj_z": torch.nn.Linear(4, 4),
                    "in_proj_b": torch.nn.Linear(4, 4),
                    "in_proj_a": torch.nn.Linear(4, 4),
                    "out_proj": torch.nn.Linear(4, 4),
                    "gate_proj": torch.nn.Linear(4, 4),
                    "up_proj": torch.nn.Linear(4, 4),
                    "down_proj": torch.nn.Linear(4, 4),
                }
            )
            self.lm_head = torch.nn.Linear(4, 4)

    targets = resolve_qwen_language_lora_targets(Hybrid())
    assert len(targets) == 12
    assert all("language_model" in target for target in targets)
    assert all("visual" not in target for target in targets)
    assert all(not target.endswith("lm_head") for target in targets)


def test_saved_peft_leaf_targets_match_resolved_full_topology() -> None:
    resolved = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.k_proj",
    ]

    assert _adapter_targets_match({"q_proj", "k_proj"}, resolved)
    assert _adapter_targets_match(set(resolved), resolved)
    assert not _adapter_targets_match({"q_proj"}, resolved)


def test_detailed_logging_writes_live_metrics_status_and_svg(tmp_path: Path) -> None:
    class Args:
        output_dir = str(tmp_path / "checkpoints")
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 4
        world_size = 2

    class State:
        is_world_process_zero = True
        global_step = 10
        epoch = 0.5
        best_metric = None
        best_model_checkpoint = None

    callback = DetailedLoggingCallback(
        tmp_path / "logs", {"pipeline_id": "p1", "stage_id": "s1"}
    )
    callback.on_train_begin(Args(), State(), object())
    for step, loss in enumerate((2.0, 1.8, 1.6, 1.5), 1):
        State.global_step = step * 10
        callback.on_log(Args(), State(), object(), {"loss": loss, "learning_rate": 1e-4})
    callback.finalize(State(), test_metrics={"test_loss": 1.4})

    records = [
        json.loads(line)
        for line in (tmp_path / "logs/metrics.jsonl").read_text().splitlines()
    ]
    status = json.loads((tmp_path / "logs/status.json").read_text())
    assert records[-1]["estimated_samples_seen"] == 4 * 10 * 1 * 4 * 2
    assert status["status"] == "completed"
    assert status["convergence"]["classification"] == "still_decreasing"
    assert (tmp_path / "logs/loss_curve.svg").read_text().startswith("<svg")


def test_collapse_resumed_metrics_replaces_the_abandoned_checkpoint_branch() -> None:
    records = [
        {"global_step": 10, "loss": 1.0},
        {"global_step": 20, "loss": 0.9},
        {"global_step": 20, "eval_loss": 0.95},
        {"global_step": 30, "loss": 0.8},
        {"global_step": 20, "loss": 0.85},
        {"global_step": 30, "loss": 0.75},
    ]

    assert collapse_resumed_metrics(records) == [
        {"global_step": 10, "loss": 1.0},
        {"global_step": 20, "loss": 0.85},
        {"global_step": 30, "loss": 0.75},
    ]


def test_collapse_resumed_metrics_uses_pipeline_steps_for_combined_graphs() -> None:
    records = [
        {"global_step": 88, "pipeline_step": 88, "loss": 1.0},
        {"global_step": 1, "pipeline_step": 89, "loss": 0.9},
    ]

    assert collapse_resumed_metrics(records) == records
