"""CPU-only production-contract tests for selected-row discrete SVG training."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from svg_agentic_slm.models.transformers_text_backend import TransformersTextBackend
from svg_agentic_slm.svg.codec_backends import (
    LOCAL_GEMMA_BACKEND_ID,
    OPENVGLAB_TRAIN_4B_BACKEND_ID,
    CodecCompatibilityError,
)
from svg_agentic_slm.svg.discrete_runtime import (
    LOCAL_GEMMA_CODEC_VOCABULARY_SIZE,
    DiscreteSVGGrammar,
    create_gemma_named_codec,
    discrete_generation_kwargs,
    register_named_codec_tokens,
    validate_gemma_codec_backend,
)
from svg_agentic_slm.svg.official_discrete_cache import (
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    UnsupportedOfficialDiscreteOperation,
)
from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.train.sft_trainer import (
    SFTConfig,
    _configure_discrete_lora,
    _filter_official_train_records,
    _load_adapter_weight_only,
    _select_target,
)


def test_official_cache_is_default_and_legacy_local_requires_explicit_opt_in() -> None:
    official = create_gemma_named_codec()
    assert official.metadata.backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
    with pytest.raises(UnsupportedOfficialDiscreteOperation, match="cache-only"):
        official.encode("<svg/>")
    with pytest.raises(CodecCompatibilityError, match="allow_legacy_toy_codec"):
        create_gemma_named_codec(LOCAL_GEMMA_BACKEND_ID)

    codec = create_gemma_named_codec(
        LOCAL_GEMMA_BACKEND_ID,
        allow_legacy_toy_codec=True,
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        '<path d="M 10 10 L 50 50 Z" fill="#369"/></svg>'
    )

    encoded = codec.encode(svg)
    decoded = codec.decode(encoded.tokens)

    assert codec.metadata.official_checkpoint_compatible is False
    assert len(codec.vocabulary_tokens()) == LOCAL_GEMMA_CODEC_VOCABULARY_SIZE
    assert decoded.svg.startswith("<svg")
    with pytest.raises(CodecCompatibilityError, match="cannot be used"):
        validate_gemma_codec_backend(OPENVGLAB_TRAIN_4B_BACKEND_ID)


def test_real_fast_tokenizer_registration_preserves_specials_and_roundtrips() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    tokenizer_object = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "base": 1, "<existing-special>": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer_object.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        unk_token="[UNK]",
        additional_special_tokens=["<existing-special>"],
    )
    codec = create_gemma_named_codec(
        LOCAL_GEMMA_BACKEND_ID,
        allow_legacy_toy_codec=True,
    )

    registration = register_named_codec_tokens(tokenizer, codec)

    assert registration.added_token_count == LOCAL_GEMMA_CODEC_VOCABULARY_SIZE
    assert len(registration.token_ids) == LOCAL_GEMMA_CODEC_VOCABULARY_SIZE
    assert len(set(registration.token_ids)) == LOCAL_GEMMA_CODEC_VOCABULARY_SIZE
    assert "<existing-special>" in tokenizer.extra_special_tokens
    assert registration.to_manifest()["joined_sequence_roundtrip_verified"] is True


def test_registration_uses_backend_declared_vocabulary_size() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    tokenizer_object = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    tokenizer_object.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        unk_token="[UNK]",
    )

    class DeclaredCodec:
        metadata = SimpleNamespace(backend_id="declared-test-codec")

        def vocabulary_tokens(self):
            return ("<declared:0>", "<declared:1>", "<declared:2>")

        def codec_manifest(self):
            return {"vocabulary_size": 3}

    registration = register_named_codec_tokens(tokenizer, DeclaredCodec())
    assert registration.added_token_count == 3
    assert len(registration.token_ids) == 3


def test_legacy_discrete_lora_config_becomes_selected_tied_rows() -> None:
    legacy = LoRAConfig(modules_to_save=["embed_tokens", "lm_head"])
    ids = list(range(100_000, 100_000 + LOCAL_GEMMA_CODEC_VOCABULARY_SIZE))

    configured = _configure_discrete_lora(legacy, ids)

    assert configured.modules_to_save == []
    assert configured.trainable_token_indices == ids
    assert configured.ensure_weight_tying is True


def _tiny_gemma_config():
    transformers = pytest.importorskip("transformers")
    return transformers.models.gemma4.configuration_gemma4.Gemma4TextConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=32,
        layer_types=["full_attention", "sliding_attention"],
        final_logit_softcapping=12.0,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=67,
        tie_word_embeddings=True,
    )


def _tiny_gemma(state_dict=None):
    transformers = pytest.importorskip("transformers")
    model = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(
        _tiny_gemma_config()
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def test_peft_selected_rows_change_base_rows_stay_frozen_and_reload_matches(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")
    torch.manual_seed(11)
    pristine = _tiny_gemma()
    pristine_state = copy.deepcopy(pristine.state_dict())

    torch.manual_seed(12)
    base = _tiny_gemma(pristine_state)
    base.resize_token_embeddings(70)
    assert (
        base.get_input_embeddings().weight.data_ptr()
        == base.get_output_embeddings().weight.data_ptr()
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            trainable_token_indices=[67, 68, 69],
            ensure_weight_tying=True,
            task_type="CAUSAL_LM",
        ),
    )
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    assert type(input_layer).__name__ == "TrainableTokensWrapper"
    assert type(output_layer).__name__ == "TrainableTokensWrapper"
    input_adapter = input_layer.token_adapter
    output_adapter = output_layer.token_adapter
    assert (
        output_adapter.tied_adapter is input_adapter
        or input_adapter.tied_adapter is output_adapter
    )
    input_delta_ids = {
        id(parameter) for parameter in input_adapter.trainable_tokens_delta.values()
    }
    output_delta_ids = {
        id(parameter) for parameter in output_adapter.trainable_tokens_delta.values()
    }
    assert len(input_delta_ids) == 1
    assert input_delta_ids == output_delta_ids
    delta = next(
        parameter
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name
    )
    delta_before = delta.detach().clone()
    frozen_base_row = input_adapter.base_layer.weight[1].detach().clone()

    input_ids = torch.tensor([[1, 67, 68, 69, 2]])
    model.train()
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    assert delta.grad is not None and torch.count_nonzero(delta.grad).item() > 0
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )
    optimizer.step()
    assert not torch.equal(delta.detach(), delta_before)
    torch.testing.assert_close(input_adapter.base_layer.weight[1], frozen_base_row)

    model.eval()
    expected_logits = model(input_ids=input_ids).logits.detach()
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir, save_embedding_layers=False)
    adapter_state = safetensors.load_file(adapter_dir / "adapter_model.safetensors")
    assert any("trainable_tokens_delta" in key for key in adapter_state)
    assert not any("base_layer.weight" in key for key in adapter_state)
    assert not any(key.endswith("embed_tokens.weight") for key in adapter_state)
    assert not any(key.endswith("lm_head.weight") for key in adapter_state)

    torch.manual_seed(99)
    reload_base = _tiny_gemma(pristine_state)
    reload_base.resize_token_embeddings(70)
    reloaded = peft.PeftModel.from_pretrained(reload_base, adapter_dir).eval()
    actual_logits = reloaded(input_ids=input_ids).logits.detach()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)


def test_adapter_weight_only_warm_start_is_audited_trainable_and_not_resume(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SFTConfig(init_adapter_from="adapter", resume_from_checkpoint="checkpoint")

    def adapter_config(token_ids, target_modules=None):
        return peft.LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=target_modules or ["q_proj", "v_proj"],
            trainable_token_indices=token_ids,
            ensure_weight_tying=True,
            task_type="CAUSAL_LM",
        )

    base_model_id = "test/tiny-gemma"
    source_base = _tiny_gemma()
    source_base.config._name_or_path = base_model_id
    source_base.resize_token_embeddings(70)
    source = peft.get_peft_model(source_base, adapter_config([67, 68, 69]))
    source.peft_config["default"].base_model_name_or_path = base_model_id
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.125)
    adapter_dir = tmp_path / "adapter"
    source.save_pretrained(adapter_dir, save_embedding_layers=False)

    reload_base = _tiny_gemma()
    reload_base.config._name_or_path = base_model_id
    reload_base.resize_token_embeddings(70)
    expanded_targets = sorted(
        name
        for name, _module in reload_base.named_modules()
        if name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    )
    loaded, audit = _load_adapter_weight_only(
        reload_base,
        adapter_path=adapter_dir,
        expected_peft_config=adapter_config([67, 68, 69], expanded_targets),
        expected_base_model_id=base_model_id,
        expected_base_model_revision="test-revision",
        codec_backend_id="test-codec",
        expected_codec_token_ids=[67, 68, 69],
    )

    assert audit["mode"] == "adapter_weights_only"
    assert audit["post_load_tensor_equality_verified"] is True
    assert audit["optimizer_loaded"] is False
    assert audit["scheduler_loaded"] is False
    assert audit["trainer_state_loaded"] is False
    assert audit["adapter_config_sha256"]
    assert audit["adapter_model_sha256"]
    assert audit["canonical_target_module_suffixes"] == ["q_proj", "v_proj"]
    assert audit["resolved_target_module_count"] == len(expanded_targets)
    assert audit["resolved_target_modules_sha256"]
    assert any(parameter.requires_grad for parameter in loaded.parameters())

    mismatch_base = _tiny_gemma()
    mismatch_base.resize_token_embeddings(70)
    with pytest.raises(ValueError, match="topology mismatch"):
        _load_adapter_weight_only(
            mismatch_base,
            adapter_path=adapter_dir,
            expected_peft_config=adapter_config([67, 68], expanded_targets),
            expected_base_model_id=base_model_id,
            expected_base_model_revision="test-revision",
            codec_backend_id="test-codec",
            expected_codec_token_ids=[67, 68],
        )

    unauthorized_targets = [
        *expanded_targets,
        next(
            name
            for name, _module in mismatch_base.named_modules()
            if name.rsplit(".", 1)[-1] == "k_proj"
        ),
    ]
    with pytest.raises(ValueError, match="target_modules mismatch"):
        _load_adapter_weight_only(
            mismatch_base,
            adapter_path=adapter_dir,
            expected_peft_config=adapter_config([67, 68, 69], unauthorized_targets),
            expected_base_model_id=base_model_id,
            expected_base_model_revision="test-revision",
            codec_backend_id="test-codec",
            expected_codec_token_ids=[67, 68, 69],
        )


def test_grammar_allowlist_codec_eos_stopping_and_strict_decode() -> None:
    torch = pytest.importorskip("torch")
    codec = create_gemma_named_codec(
        LOCAL_GEMMA_BACKEND_ID,
        allow_legacy_toy_codec=True,
    )
    tokens = codec.vocabulary_tokens()
    id_to_token = {10_000 + index: token for index, token in enumerate(tokens)}
    token_to_id = {token: token_id for token_id, token in id_to_token.items()}
    grammar = DiscreteSVGGrammar(codec, id_to_token)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        '<path d="M 10 10 L 50 50 Z" fill="#369"/></svg>'
    )
    encoded = codec.encode(svg)
    completion_ids = [token_to_id[str(token)] for token in encoded.tokens]

    prefix: list[int] = []
    for token_id in completion_ids:
        assert token_id in grammar.allowed_next(prefix)
        prefix.append(token_id)
    kwargs = discrete_generation_kwargs(grammar, prompt_tokens=2, pad_token_id=0)
    assert kwargs["eos_token_id"] == token_to_id[codec.eos_token]
    assert kwargs["prefix_allowed_tokens_fn"](
        0, torch.tensor([7, 8, completion_ids[0]])
    ) == list(grammar.allowed_next(completion_ids[:1]))
    assert codec.decode(encoded.tokens).svg.startswith("<svg")
    with pytest.raises(CodecCompatibilityError, match="violates grammar"):
        grammar.allowed_next([completion_ids[0], 999])


def test_cached_target_selection_never_calls_live_encode() -> None:
    class CacheOnlyCodec:
        def target_for_record(self, record):
            assert record["id"] == "cached"
            return "<cached:bos><cached:eos>"

        def encode(self, _svg):
            raise AssertionError("live encode must not be called")

    target = _select_target(
        {"id": "cached", "output_svg": "<svg/>"},
        "omnisvg_discrete",
        CacheOnlyCodec(),
    )
    assert target == "<cached:bos><cached:eos>"


def test_official_cached_backend_fails_closed_for_generation(tmp_path: Path) -> None:
    with pytest.raises(CodecCompatibilityError, match="no audited decoder"):
        TransformersTextBackend(
            adapter_path=tmp_path / "adapter",
            tokenizer_path=tmp_path / "tokenizer",
            output_format="discrete_svg",
            codec_manifest_path=tmp_path / "codec_manifest.json",
        )


def test_legacy_discrete_backend_loaded_state_uses_initialized_codec_grammar(
    tmp_path: Path,
) -> None:
    backend = TransformersTextBackend(
        adapter_path=tmp_path / "adapter",
        tokenizer_path=tmp_path / "tokenizer",
        output_format="discrete_svg",
        codec_manifest_path=tmp_path / "codec_manifest.json",
        codec_backend_id=LOCAL_GEMMA_BACKEND_ID,
        allow_legacy_toy_codec=True,
    )
    backend._processor = object()
    backend._tokenizer = object()
    backend._model = object()
    backend._codec_id_to_token = {1: "<codec:test>"}
    backend._codec_manifest_sha256 = "a" * 64

    assert not backend.is_loaded()
    backend._codec_grammar = object()  # type: ignore[assignment]
    assert backend.is_loaded()


def test_official_train_filter_uses_nested_identity_and_requested_order() -> None:
    records = [
        {"id": "sample-a", "metadata": {"record_id": "sample-a"}},
        {"id": "sample-b", "metadata": {"record_id": "sample-b"}},
    ]
    selected = _filter_official_train_records(records, ["sample-b", "sample-a"])
    assert selected == [records[1], records[0]]

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        _filter_official_train_records(records, ["sample-a", "sample-a"])
    with pytest.raises(ValueError, match="missing from the pinned cache join"):
        _filter_official_train_records(records, ["missing"])
    with pytest.raises(ValueError, match="identity mismatch"):
        _filter_official_train_records(
            [{"id": "top", "metadata": {"record_id": "nested"}}],
            ["nested"],
        )
