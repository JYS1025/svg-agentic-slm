"""Typed SVG codec contracts and explicitly separated OmniSVG dialect backends.

This module intentionally does not change ``OmniSVGDiscreteCodec``.  The local
Gemma dialect is wrapped as-is, while the released OpenVGLab training encoder is
loaded from an externally managed, revision-pinned checkout for analysis only.
"""

from __future__ import annotations

import hashlib
import importlib
import operator
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

TokenValue: TypeAlias = int | str

OPENVGLAB_TRAIN_COMMIT = "812489fd9d191e39fe94bc0c4027e5d0121e0fc6"
OPENVGLAB_TOKENIZATION_CONFIG_SHA256 = (
    "bd56bc2bb9b39f614a9d553d2319332a18a075c4239bb71ec437822b20d97dd4"
)
OPENVGLAB_SOURCE_URL = (
    "https://github.com/OpenVGLab/OmniSVG-train/tree/"
    f"{OPENVGLAB_TRAIN_COMMIT}"
)
LOCAL_GEMMA_BACKEND_ID = "omnisvg-inspired-discrete-svg"
OPENVGLAB_TRAIN_4B_BACKEND_ID = (
    "openvglab-omnisvg-train-4b-812489fd9d191e39fe94bc0c4027e5d0121e0fc6"
)


@dataclass(frozen=True, slots=True)
class CodecMetadata:
    """Serializable identity and capability contract for an SVG codec backend."""

    backend_id: str
    codec_name: str
    codec_version: str
    dialect: str
    provenance: str
    directions: tuple[str, ...]
    token_kind: str
    model_family: str | None
    model_size: str | None
    source_url: str | None
    source_revision: str | None
    official_checkpoint_compatible: bool
    lossy: bool
    notes: tuple[str, ...] = ()

    @property
    def can_encode(self) -> bool:
        return "encode" in self.directions

    @property
    def can_decode(self) -> bool:
        return "decode" in self.directions

    def to_manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "codec_name": self.codec_name,
            "codec_version": self.codec_version,
            "dialect": self.dialect,
            "provenance": self.provenance,
            "directions": list(self.directions),
            "token_kind": self.token_kind,
            "model_family": self.model_family,
            "model_size": self.model_size,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "official_checkpoint_compatible": self.official_checkpoint_compatible,
            "lossy": self.lossy,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class CodecEncodeResult:
    """Result of encoding one SVG.

    ``tokens`` is the backend's fully framed sequence. ``body_tokens`` is
    ``None`` when a legacy backend cannot expose framing without changing its
    public behavior.
    """

    metadata: CodecMetadata
    tokens: tuple[TokenValue, ...]
    body_tokens: tuple[TokenValue, ...] | None
    source_sha256: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def framed_length(self) -> int:
        return len(self.tokens)

    @property
    def body_length(self) -> int | None:
        return None if self.body_tokens is None else len(self.body_tokens)


@dataclass(frozen=True, slots=True)
class CodecDecodeResult:
    """Result of decoding one token sequence."""

    metadata: CodecMetadata
    svg: str
    consumed_tokens: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class CodecError(RuntimeError):
    """Base error with a stable machine-readable code and context."""

    default_code = "codec_error"

    def __init__(
        self,
        message: str,
        *,
        backend_id: str | None = None,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.backend_id = backend_id
        self.code = code or self.default_code
        self.context = dict(context or {})

    def to_manifest(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "backend_id": self.backend_id,
            "context": self.context,
        }


class CodecConfigurationError(CodecError):
    default_code = "codec_configuration_error"


class CodecCompatibilityError(CodecError):
    default_code = "codec_compatibility_error"


class CodecImportError(CodecError):
    default_code = "codec_import_error"


class CodecEncodeError(CodecError):
    default_code = "codec_encode_error"


class CodecDecodeError(CodecError):
    default_code = "codec_decode_error"


class UnsupportedCodecOperation(CodecError):
    default_code = "unsupported_codec_operation"


@runtime_checkable
class SVGCodec(Protocol):
    """Object-oriented contract shared by local and external SVG codecs."""

    @property
    def metadata(self) -> CodecMetadata: ...

    def encode(self, svg: str) -> CodecEncodeResult: ...

    def decode(self, tokens: Sequence[TokenValue]) -> CodecDecodeResult: ...


@runtime_checkable
class NamedSpecialTokenSVGCodec(SVGCodec, Protocol):
    """Capability required by a tokenizer-registered Gemma SVG dialect."""

    def vocabulary_tokens(self) -> list[str]: ...

    def codec_manifest(self) -> dict[str, object]: ...


LOCAL_GEMMA_METADATA = CodecMetadata(
    backend_id=LOCAL_GEMMA_BACKEND_ID,
    codec_name="OmniSVG-inspired discrete SVG",
    codec_version="1.0.0",
    dialect="svgd1-local-gemma",
    provenance="project-local",
    directions=("encode", "decode"),
    token_kind="named-special-token",
    model_family="Gemma",
    model_size=None,
    source_url=None,
    source_revision=None,
    official_checkpoint_compatible=False,
    lossy=True,
    notes=(
        "Backward-compatible wrapper around OmniSVGDiscreteCodec.",
        "Its explicit SOP/F/EOP grammar is not the released OpenVGLab ID grammar.",
    ),
)


class GemmaOmniSVGInspiredBackend:
    """Bidirectional adapter around the existing public local codec API."""

    def __init__(self, legacy_codec: Any | None = None, *, grid_size: int = 200) -> None:
        if legacy_codec is None:
            from svg_agentic_slm.svg.discrete_codec import OmniSVGDiscreteCodec

            legacy_codec = OmniSVGDiscreteCodec(grid_size=grid_size)
        elif grid_size != 200:
            raise CodecConfigurationError(
                "grid_size cannot be combined with an injected legacy codec",
                backend_id=LOCAL_GEMMA_BACKEND_ID,
                code="ambiguous_codec_construction",
            )
        self._legacy_codec = legacy_codec

    @property
    def metadata(self) -> CodecMetadata:
        return LOCAL_GEMMA_METADATA

    @property
    def legacy_codec(self) -> Any:
        return self._legacy_codec

    @property
    def sop_token(self) -> str:
        return str(self._legacy_codec.sop_token)

    @property
    def eop_token(self) -> str:
        return str(self._legacy_codec.eop_token)

    @property
    def eos_token(self) -> str:
        return str(self._legacy_codec.eos_token)

    def vocabulary_tokens(self) -> list[str]:
        return list(self._legacy_codec.vocabulary_tokens())

    def codec_manifest(self) -> dict[str, object]:
        return dict(self._legacy_codec.manifest())

    def encode(self, svg: str) -> CodecEncodeResult:
        if not isinstance(svg, str) or not svg.strip():
            raise CodecEncodeError(
                "SVG input must be a non-empty string",
                backend_id=self.metadata.backend_id,
                code="invalid_svg_input",
            )
        try:
            tokens = tuple(self._legacy_codec.encode_svg(svg))
        except Exception as exc:
            raise CodecEncodeError(
                f"Local Gemma codec encode failed: {exc}",
                backend_id=self.metadata.backend_id,
                code="local_encode_failed",
            ) from exc
        if not tokens:
            raise CodecEncodeError(
                "Local Gemma codec returned an empty sequence",
                backend_id=self.metadata.backend_id,
                code="empty_token_sequence",
            )
        if not all(isinstance(token, str) for token in tokens):
            raise CodecEncodeError(
                "Local Gemma codec returned a non-string token",
                backend_id=self.metadata.backend_id,
                code="invalid_token_type",
            )
        return CodecEncodeResult(
            metadata=self.metadata,
            tokens=tokens,
            body_tokens=None,
            source_sha256=hashlib.sha256(svg.encode("utf-8")).hexdigest(),
            diagnostics={
                "legacy_public_api": "OmniSVGDiscreteCodec.encode_svg",
                "framing_exposed": False,
            },
        )

    def decode(self, tokens: Sequence[TokenValue]) -> CodecDecodeResult:
        token_list = list(tokens)
        if not token_list or not all(isinstance(token, str) for token in token_list):
            raise CodecDecodeError(
                "Local Gemma decode requires a non-empty string-token sequence",
                backend_id=self.metadata.backend_id,
                code="invalid_token_sequence",
            )
        try:
            svg = self._legacy_codec.decode_tokens(token_list)
        except Exception as exc:
            raise CodecDecodeError(
                f"Local Gemma codec decode failed: {exc}",
                backend_id=self.metadata.backend_id,
                code="local_decode_failed",
            ) from exc
        if not isinstance(svg, str) or not svg.strip():
            raise CodecDecodeError(
                "Local Gemma codec returned an empty SVG",
                backend_id=self.metadata.backend_id,
                code="empty_svg_output",
            )
        return CodecDecodeResult(
            metadata=self.metadata,
            svg=svg,
            consumed_tokens=len(token_list),
            diagnostics={"legacy_public_api": "OmniSVGDiscreteCodec.decode_tokens"},
        )


OPENVGLAB_TRAIN_4B_METADATA = CodecMetadata(
    backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
    codec_name="OpenVGLab OmniSVG released training encoder",
    codec_version=OPENVGLAB_TRAIN_COMMIT,
    dialect="openvglab-training-code-4b",
    provenance="official-training-source",
    directions=("encode",),
    token_kind="absolute-integer-id",
    model_family="Qwen2.5-VL",
    model_size="4B",
    source_url=OPENVGLAB_SOURCE_URL,
    source_revision=OPENVGLAB_TRAIN_COMMIT,
    official_checkpoint_compatible=False,
    lossy=True,
    notes=(
        "Encode-only analysis backend; no round-trip decoder is claimed.",
        "Released training IDs are offset from the released inference decoder/checkpoint IDs.",
        "Masking and dataset-side truncation are not invoked.",
    ),
)


@dataclass(frozen=True, slots=True)
class GitCheckoutState:
    commit: str
    top_level: Path
    clean_tracked_worktree: bool
    tokenization_config_tracked: bool


@dataclass(frozen=True, slots=True)
class OpenVGLabRuntime:
    tokenization_config_type: Any
    tokenizer_type: Any
    svg_type: Any


GitStateResolver: TypeAlias = Callable[[Path], GitCheckoutState]
RuntimeLoader: TypeAlias = Callable[[Path], OpenVGLabRuntime]
_IMPORT_LOCK = threading.RLock()


def _run_git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodecCompatibilityError(
            f"Unable to inspect external OpenVGLab checkout: {exc}",
            backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
            code="git_inspection_failed",
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise CodecCompatibilityError(
            f"Unable to inspect external OpenVGLab checkout: {detail}",
            backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
            code="git_inspection_failed",
        )
    return completed.stdout.strip()


def resolve_git_checkout_state(root: Path) -> GitCheckoutState:
    commit = _run_git(root, "rev-parse", "HEAD")
    top_level = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    try:
        _run_git(root, "ls-files", "--error-unmatch", "--", "configs/tokenization.yaml")
        config_tracked = True
    except CodecCompatibilityError:
        config_tracked = False
    return GitCheckoutState(
        commit=commit,
        top_level=top_level,
        clean_tracked_worktree=not bool(status),
        tokenization_config_tracked=config_tracked,
    )


def _module_is_under_root(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def load_openvglab_runtime(root: Path) -> OpenVGLabRuntime:
    """Load the pinned checkout without installing or downloading it."""

    required_modules = ("utils.config", "utils.dataset", "deepsvg.svglib.svg")
    guarded_prefixes = ("utils", "deepsvg")
    with _IMPORT_LOCK:
        for name, module in tuple(sys.modules.items()):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in guarded_prefixes):
                if module is not None and not _module_is_under_root(module, root):
                    raise CodecImportError(
                        f"Module name collision for {name!r}; refusing ambiguous upstream import",
                        backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                        code="upstream_module_collision",
                        context={"module": name, "module_file": getattr(module, "__file__", None)},
                    )

        modules_before = set(sys.modules)
        sys.path.insert(0, str(root))
        try:
            imported = {name: importlib.import_module(name) for name in required_modules}
        except Exception as exc:
            for name in set(sys.modules) - modules_before:
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in guarded_prefixes
                ):
                    sys.modules.pop(name, None)
            raise CodecImportError(
                f"Unable to import pinned OpenVGLab encoder dependencies: {exc}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="upstream_import_failed",
            ) from exc
        finally:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass

        for name, module in imported.items():
            if not _module_is_under_root(module, root):
                raise CodecImportError(
                    f"Imported {name!r} from outside the pinned checkout",
                    backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                    code="upstream_import_origin_mismatch",
                    context={"module": name, "module_file": getattr(module, "__file__", None)},
                )
        try:
            return OpenVGLabRuntime(
                tokenization_config_type=imported["utils.config"].TokenizationConfig,
                tokenizer_type=imported["utils.dataset"].SVGTokenizer,
                svg_type=imported["deepsvg.svglib.svg"].SVG,
            )
        except AttributeError as exc:
            raise CodecImportError(
                f"Pinned OpenVGLab encoder API is missing: {exc}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="upstream_api_missing",
            ) from exc


def _integer_ids(value: Any, *, field_name: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise CodecEncodeError(
            f"OpenVGLab {field_name} is not a flat sequence",
            backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
            code="invalid_upstream_token_sequence",
        )
    result: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool):
            raise CodecEncodeError(
                f"OpenVGLab {field_name}[{index}] is boolean, not an integer ID",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="invalid_upstream_token_type",
            )
        try:
            token_id = operator.index(token)
        except TypeError as exc:
            raise CodecEncodeError(
                f"OpenVGLab {field_name}[{index}] is not an integer ID",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="invalid_upstream_token_type",
            ) from exc
        if not 0 <= token_id < 197000:
            raise CodecEncodeError(
                f"OpenVGLab {field_name}[{index}]={token_id} is outside [0, 197000)",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="upstream_token_out_of_range",
            )
        result.append(token_id)
    return tuple(result)


class OpenVGLabTrainingEncoder4B:
    """Strict adapter for the released 4B training encoder at one exact commit."""

    def __init__(
        self,
        upstream_root: str | Path,
        *,
        expected_commit: str = OPENVGLAB_TRAIN_COMMIT,
        expected_config_sha256: str = OPENVGLAB_TOKENIZATION_CONFIG_SHA256,
        git_state_resolver: GitStateResolver = resolve_git_checkout_state,
        runtime_loader: RuntimeLoader = load_openvglab_runtime,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
            raise CodecConfigurationError(
                "expected_commit must be a full lowercase 40-character Git commit",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="mutable_or_invalid_revision",
            )
        if re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None:
            raise CodecConfigurationError(
                "expected_config_sha256 must be a lowercase SHA-256 digest",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="invalid_config_digest",
            )

        root = Path(upstream_root).expanduser().resolve()
        if not root.is_dir():
            raise CodecConfigurationError(
                f"External OpenVGLab root does not exist: {root}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="upstream_root_missing",
            )
        config_path = root / "configs" / "tokenization.yaml"
        if not config_path.is_file():
            raise CodecCompatibilityError(
                f"Pinned tokenization config is missing: {config_path}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="tokenization_config_missing",
            )

        state = git_state_resolver(root)
        if state.top_level.resolve() != root:
            raise CodecCompatibilityError(
                "External root is not the top level of the pinned Git checkout",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="git_root_mismatch",
                context={"expected_root": str(root), "actual_root": str(state.top_level)},
            )
        if state.commit != expected_commit:
            raise CodecCompatibilityError(
                f"OpenVGLab checkout is {state.commit}, expected {expected_commit}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="git_commit_mismatch",
                context={"expected": expected_commit, "actual": state.commit},
            )
        if not state.clean_tracked_worktree:
            raise CodecCompatibilityError(
                "OpenVGLab checkout has tracked modifications",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="dirty_upstream_checkout",
            )
        if not state.tokenization_config_tracked:
            raise CodecCompatibilityError(
                "configs/tokenization.yaml is not tracked by the pinned checkout",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="untracked_tokenization_config",
            )

        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_digest != expected_config_sha256:
            raise CodecCompatibilityError(
                "Pinned tokenization config SHA-256 does not match",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="tokenization_config_digest_mismatch",
                context={"expected": expected_config_sha256, "actual": config_digest},
            )

        try:
            runtime = runtime_loader(root)
        except CodecError:
            raise
        except Exception as exc:
            raise CodecImportError(
                f"Unable to load the pinned OpenVGLab runtime: {exc}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="upstream_import_failed",
            ) from exc
        try:
            tokenization_config = runtime.tokenization_config_type.from_yaml(
                str(config_path), model_size="4B"
            )
            tokenizer = runtime.tokenizer_type(tokenization_config)
        except Exception as exc:
            raise CodecImportError(
                f"Unable to initialize the pinned OpenVGLab tokenizer: {exc}",
                backend_id=OPENVGLAB_TRAIN_4B_BACKEND_ID,
                code="upstream_initialization_failed",
            ) from exc

        self._root = root
        self._config_path = config_path
        self._config_sha256 = config_digest
        self._runtime = runtime
        self._tokenizer = tokenizer

    @property
    def metadata(self) -> CodecMetadata:
        return OPENVGLAB_TRAIN_4B_METADATA

    def encode(self, svg: str) -> CodecEncodeResult:
        if not isinstance(svg, str) or not svg.strip():
            raise CodecEncodeError(
                "SVG input must be a non-empty string",
                backend_id=self.metadata.backend_id,
                code="invalid_svg_input",
            )
        try:
            parsed = self._runtime.svg_type.from_str(svg)
        except Exception as exc:
            raise CodecEncodeError(
                f"OpenVGLab SVG parse failed: {exc}",
                backend_id=self.metadata.backend_id,
                code="svg_parse_failed",
            ) from exc
        if parsed is None:
            raise CodecEncodeError(
                "OpenVGLab SVG parser returned no object",
                backend_id=self.metadata.backend_id,
                code="svg_parse_returned_none",
            )
        try:
            paths, colors = parsed.to_tensor(concat_groups=False, PAD_VAL=0)
            raw_body = self._tokenizer.tokenize_svg_tensors(paths, colors)
        except Exception as exc:
            raise CodecEncodeError(
                f"OpenVGLab SVG tensorization/tokenization failed: {exc}",
                backend_id=self.metadata.backend_id,
                code="svg_tokenization_failed",
            ) from exc

        body = _integer_ids(raw_body, field_name="body tokens")
        if not body:
            raise CodecEncodeError(
                "OpenVGLab encoder returned an empty SVG body",
                backend_id=self.metadata.backend_id,
                code="empty_token_sequence",
            )
        try:
            raw_framed = self._tokenizer.add_special_tokens(raw_body)
        except Exception as exc:
            raise CodecEncodeError(
                f"OpenVGLab special-token framing failed: {exc}",
                backend_id=self.metadata.backend_id,
                code="special_token_framing_failed",
            ) from exc
        framed = _integer_ids(raw_framed, field_name="framed tokens")
        if len(framed) != len(body) + 2 or framed[1:-1] != body:
            raise CodecEncodeError(
                "OpenVGLab framing is not exactly BOS + body + EOS",
                backend_id=self.metadata.backend_id,
                code="unexpected_special_token_framing",
            )
        if framed[0] != 196998 or framed[-1] != 196999:
            raise CodecEncodeError(
                "OpenVGLab 4B BOS/EOS IDs do not match the pinned training config",
                backend_id=self.metadata.backend_id,
                code="unexpected_bos_eos_ids",
                context={"bos": framed[0], "eos": framed[-1]},
            )
        return CodecEncodeResult(
            metadata=self.metadata,
            tokens=framed,
            body_tokens=body,
            source_sha256=hashlib.sha256(svg.encode("utf-8")).hexdigest(),
            diagnostics={
                "upstream_root": str(self._root),
                "upstream_commit": OPENVGLAB_TRAIN_COMMIT,
                "tokenization_config": str(self._config_path.relative_to(self._root)),
                "tokenization_config_sha256": self._config_sha256,
                "model_size": "4B",
                "masking_enabled": False,
                "bos_token_id": 196998,
                "eos_token_id": 196999,
            },
        )

    def decode(self, tokens: Sequence[TokenValue]) -> CodecDecodeResult:
        del tokens
        raise UnsupportedCodecOperation(
            "The pinned OpenVGLab training backend is encode-only; no compatible official "
            "round-trip decoder is published",
            backend_id=self.metadata.backend_id,
            code="official_training_backend_encode_only",
        )


CodecFactory: TypeAlias = Callable[..., SVGCodec]


@dataclass(frozen=True, slots=True)
class CodecRegistration:
    metadata: CodecMetadata
    factory: CodecFactory


class CodecRegistry:
    """Explicit registry that does not instantiate optional external backends eagerly."""

    def __init__(self) -> None:
        self._registrations: dict[str, CodecRegistration] = {}

    def register(
        self,
        metadata: CodecMetadata,
        factory: CodecFactory,
        *,
        replace: bool = False,
    ) -> None:
        if metadata.backend_id in self._registrations and not replace:
            raise CodecConfigurationError(
                f"Codec backend already registered: {metadata.backend_id}",
                backend_id=metadata.backend_id,
                code="duplicate_codec_registration",
            )
        self._registrations[metadata.backend_id] = CodecRegistration(metadata, factory)

    def create(self, backend_id: str, **kwargs: Any) -> SVGCodec:
        try:
            registration = self._registrations[backend_id]
        except KeyError as exc:
            raise CodecConfigurationError(
                f"Unknown codec backend: {backend_id}",
                backend_id=backend_id,
                code="unknown_codec_backend",
                context={"available": list(self.backend_ids())},
            ) from exc
        codec = registration.factory(**kwargs)
        if codec.metadata.backend_id != backend_id:
            raise CodecConfigurationError(
                "Codec factory returned a backend with the wrong identity",
                backend_id=backend_id,
                code="codec_factory_identity_mismatch",
                context={"actual": codec.metadata.backend_id},
            )
        return codec

    def metadata(self, backend_id: str) -> CodecMetadata:
        try:
            return self._registrations[backend_id].metadata
        except KeyError as exc:
            raise CodecConfigurationError(
                f"Unknown codec backend: {backend_id}",
                backend_id=backend_id,
                code="unknown_codec_backend",
                context={"available": list(self.backend_ids())},
            ) from exc

    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))


DEFAULT_CODEC_REGISTRY = CodecRegistry()
DEFAULT_CODEC_REGISTRY.register(LOCAL_GEMMA_METADATA, GemmaOmniSVGInspiredBackend)
DEFAULT_CODEC_REGISTRY.register(OPENVGLAB_TRAIN_4B_METADATA, OpenVGLabTrainingEncoder4B)


def create_svg_codec(backend_id: str, **kwargs: Any) -> SVGCodec:
    """Construct a registered backend without changing legacy codec call sites."""

    return DEFAULT_CODEC_REGISTRY.create(backend_id, **kwargs)
