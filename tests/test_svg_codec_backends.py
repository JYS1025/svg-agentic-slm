"""Network-free tests for explicit local and pinned OpenVGLab codec backends."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from svg_agentic_slm.svg.codec_backends import (
    LOCAL_GEMMA_BACKEND_ID,
    OPENVGLAB_TRAIN_4B_BACKEND_ID,
    OPENVGLAB_TRAIN_COMMIT,
    CodecCompatibilityError,
    CodecEncodeError,
    CodecImportError,
    CodecRegistry,
    GemmaOmniSVGInspiredBackend,
    GitCheckoutState,
    OpenVGLabRuntime,
    OpenVGLabTrainingEncoder4B,
    UnsupportedCodecOperation,
)


class _FakeLegacyCodec:
    def encode_svg(self, svg: str) -> list[str]:
        return ["<svgd1:sop>", f"<svgd1:source:{len(svg)}>", "<svgd1:eos>"]

    def decode_tokens(self, tokens: list[str]) -> str:
        return f'<svg xmlns="http://www.w3.org/2000/svg" data-count="{len(tokens)}"/>'


class _FakeConfig:
    @classmethod
    def from_yaml(cls, path: str, *, model_size: str):
        assert Path(path).name == "tokenization.yaml"
        assert model_size == "4B"
        return cls()


class _FakeTokenizer:
    def __init__(self, config: _FakeConfig) -> None:
        assert isinstance(config, _FakeConfig)

    def tokenize_svg_tensors(self, paths, colors) -> list[int]:
        assert paths == ["paths"]
        assert colors == ["colors"]
        return [151938, 151943, 191946]

    def add_special_tokens(self, body) -> list[int]:
        return [196998, *list(body), 196999]


class _FakeParsedSvg:
    def to_tensor(self, *, concat_groups: bool, PAD_VAL: int):
        assert concat_groups is False
        assert PAD_VAL == 0
        return ["paths"], ["colors"]


class _FakeSvgType:
    @classmethod
    def from_str(cls, svg: str):
        if svg == "bad":
            raise ValueError("fixture parse failure")
        return _FakeParsedSvg()


def _external_fixture(tmp_path: Path):
    config_path = tmp_path / "configs" / "tokenization.yaml"
    config_path.parent.mkdir(parents=True)
    config_bytes = b"fixture: pinned\n"
    config_path.write_bytes(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()

    def git_state(root: Path) -> GitCheckoutState:
        return GitCheckoutState(
            commit=OPENVGLAB_TRAIN_COMMIT,
            top_level=root,
            clean_tracked_worktree=True,
            tokenization_config_tracked=True,
        )

    def runtime_loader(root: Path) -> OpenVGLabRuntime:
        assert root == tmp_path.resolve()
        return OpenVGLabRuntime(_FakeConfig, _FakeTokenizer, _FakeSvgType)

    return digest, git_state, runtime_loader


def test_local_gemma_wrapper_preserves_bidirectional_legacy_contract() -> None:
    backend = GemmaOmniSVGInspiredBackend(_FakeLegacyCodec())

    encoded = backend.encode('<svg xmlns="http://www.w3.org/2000/svg"/>')
    decoded = backend.decode(encoded.tokens)

    assert backend.metadata.backend_id == LOCAL_GEMMA_BACKEND_ID
    assert backend.metadata.directions == ("encode", "decode")
    assert backend.metadata.official_checkpoint_compatible is False
    assert encoded.body_tokens is None
    assert decoded.svg.startswith("<svg")


def test_registry_is_explicit_and_rejects_unknown_backends() -> None:
    backend = GemmaOmniSVGInspiredBackend(_FakeLegacyCodec())
    registry = CodecRegistry()
    registry.register(backend.metadata, lambda: backend)

    assert registry.create(LOCAL_GEMMA_BACKEND_ID) is backend
    assert registry.backend_ids() == (LOCAL_GEMMA_BACKEND_ID,)

    with pytest.raises(Exception, match="Unknown codec backend"):
        registry.create("missing")


def test_pinned_training_encoder_is_strict_encode_only(tmp_path: Path) -> None:
    digest, git_state, runtime_loader = _external_fixture(tmp_path)
    backend = OpenVGLabTrainingEncoder4B(
        tmp_path,
        expected_config_sha256=digest,
        git_state_resolver=git_state,
        runtime_loader=runtime_loader,
    )

    result = backend.encode('<svg xmlns="http://www.w3.org/2000/svg"/>')

    assert backend.metadata.backend_id == OPENVGLAB_TRAIN_4B_BACKEND_ID
    assert backend.metadata.directions == ("encode",)
    assert backend.metadata.official_checkpoint_compatible is False
    assert result.body_tokens == (151938, 151943, 191946)
    assert result.tokens == (196998, 151938, 151943, 191946, 196999)
    assert result.diagnostics["masking_enabled"] is False
    with pytest.raises(UnsupportedCodecOperation, match="encode-only"):
        backend.decode(result.tokens)


def test_pinned_training_encoder_rejects_revision_and_config_drift(tmp_path: Path) -> None:
    digest, git_state, runtime_loader = _external_fixture(tmp_path)

    def wrong_git_state(root: Path) -> GitCheckoutState:
        state = git_state(root)
        return GitCheckoutState("0" * 40, state.top_level, True, True)

    with pytest.raises(CodecCompatibilityError, match="expected"):
        OpenVGLabTrainingEncoder4B(
            tmp_path,
            expected_config_sha256=digest,
            git_state_resolver=wrong_git_state,
            runtime_loader=runtime_loader,
        )

    with pytest.raises(CodecCompatibilityError, match="SHA-256"):
        OpenVGLabTrainingEncoder4B(
            tmp_path,
            expected_config_sha256="0" * 64,
            git_state_resolver=git_state,
            runtime_loader=runtime_loader,
        )


def test_pinned_training_encoder_fails_closed_on_import_and_parse(tmp_path: Path) -> None:
    digest, git_state, runtime_loader = _external_fixture(tmp_path)

    def failing_loader(root: Path) -> OpenVGLabRuntime:
        raise ImportError(f"missing dependency under {root}")

    with pytest.raises(CodecImportError, match="missing dependency"):
        OpenVGLabTrainingEncoder4B(
            tmp_path,
            expected_config_sha256=digest,
            git_state_resolver=git_state,
            runtime_loader=failing_loader,
        )

    backend = OpenVGLabTrainingEncoder4B(
        tmp_path,
        expected_config_sha256=digest,
        git_state_resolver=git_state,
        runtime_loader=runtime_loader,
    )
    with pytest.raises(CodecEncodeError, match="parse failed") as exc_info:
        backend.encode("bad")
    assert exc_info.value.code == "svg_parse_failed"
