"""CPU-only experimental bridge for the pinned OmniSVG 4B token dialects.

This module is deliberately not imported by the production inference path. The
released training encoder emits every SVG body token one ID below the official
inference tokenizer, while both dialects use BOS=196998 and EOS=196999. The
bridge therefore performs this exact transformation only::

    Gemma registered ID -> named token -> training ID -> body ID + 1

Decode remains disabled until one record from the pinned audited cache has
passed strict grammar validation, official reconstruction, and CPU rendering.
This is experimental validation tooling, not a production compatibility claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import struct
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from svg_agentic_slm.svg.official_discrete_cache import (
    CACHED_TARGET_FIELD,
    OPENVGLAB_BOS_ID,
    OPENVGLAB_EOS_ID,
    OPENVGLAB_MAX_PRODUCIBLE_ID,
    OPENVGLAB_MIN_PRODUCIBLE_ID,
    OPENVGLAB_NAMED_VOCABULARY_SIZE,
    OPENVGLAB_TRAIN_COMMIT,
    CachedOpenVGLabGemmaDialect,
    OpenVGLabCacheConfig,
    PinnedOpenVGLabTokenCache,
)

OMNISVG_INFERENCE_COMMIT = "308b49b3e3df29c8b5424bbaa7f1f7c6ff218e8d"
OMNISVG_INFERENCE_CONFIG_BLOB = "3d7e7f1e210241178c247f0e3dc421b4f045fb18"
OMNISVG_INFERENCE_TOKENIZER_BLOB = "0fd445a38cbb0969f30bc09f19a51ecc425e509a"
GEMMA_REGISTERED_ID_MIN = 262144
GEMMA_REGISTERED_ID_MAX = (
    GEMMA_REGISTERED_ID_MIN + OPENVGLAB_NAMED_VOCABULARY_SIZE - 1
)
EXPERIMENTAL_DIALECT = (
    "openvglab-training-4b-body-id-plus-one-to-omnisvg-inference-4b"
)
EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL TEST-ONLY OmniSVG decode: the training and inference token "
    "dialects differ by +1 for every body token. Production inference remains "
    "unsupported; results require cached-sample and render validation."
)

_NAMED_TOKEN = re.compile(r"<svgovg4b:([0-9]+)>")
_INFERENCE_COMMAND_ARGUMENTS: Mapping[int, tuple[str, ...]] = {
    151939: ("coordinate", "coordinate"),
    151940: ("coordinate",),
    151941: ("coordinate", "coordinate", "coordinate"),
    151942: ("coordinate", "arc", "arc", "arc", "coordinate"),
    151943: ("coordinate",),
}
# The official inference parser exposes broad class boundaries, but the pinned
# training encoder cannot produce every ID inside those gaps.  Keep this bridge
# stricter than the upstream parser so reserved, never-supervised rows cannot be
# accepted as geometry or silently rendered as the decoder's gray fallback.
_PRODUCIBLE_INFERENCE_COORDINATE_MIN = 151944
_PRODUCIBLE_INFERENCE_COORDINATE_MAX = 191943
_PRODUCIBLE_INFERENCE_COLOR_MIN = 191947
_PRODUCIBLE_INFERENCE_COLOR_MAX = 196044
_PRODUCIBLE_INFERENCE_ARC_MIN = 196437
_PRODUCIBLE_INFERENCE_ARC_MAX = 196536


class ExperimentalOmniSVGDecodeWarning(UserWarning):
    """Warn that this path is test-only and dialect-shifted."""


class ExperimentalOmniSVGDecodeError(RuntimeError):
    """Fail-closed error with a stable code and diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True)
class DecodeArtifact:
    svg: str
    png: bytes
    registered_ids: tuple[int, ...]
    training_ids: tuple[int, ...]
    inference_ids: tuple[int, ...]
    command_count: int
    path_count: int
    original_registered_ids: tuple[int, ...]
    original_training_ids: tuple[int, ...]
    recovery: bool
    recovery_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialInferenceRuntime:
    tokenizer: Any
    torch: Any
    cairosvg: Any


def _fail(message: str, *, code: str, **context: Any) -> None:
    raise ExperimentalOmniSVGDecodeError(message, code=code, context=context)


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
        _fail(f"Unable to inspect official inference checkout: {exc}", code="git_failed")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        _fail(
            f"Unable to inspect official inference checkout: {detail}",
            code="git_failed",
        )
    return completed.stdout.strip()


def _validate_inference_checkout(root: Path) -> dict[str, str]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        _fail(
            f"Pinned official inference checkout does not exist: {resolved}",
            code="inference_checkout_missing",
        )
    commit = _run_git(resolved, "rev-parse", "HEAD")
    top_level = Path(_run_git(resolved, "rev-parse", "--show-toplevel")).resolve()
    status = _run_git(resolved, "status", "--porcelain", "--untracked-files=no")
    config_blob = _run_git(resolved, "rev-parse", "HEAD:config.yaml")
    tokenizer_blob = _run_git(resolved, "rev-parse", "HEAD:tokenizer.py")
    expected = {
        "commit": OMNISVG_INFERENCE_COMMIT,
        "config_blob": OMNISVG_INFERENCE_CONFIG_BLOB,
        "tokenizer_blob": OMNISVG_INFERENCE_TOKENIZER_BLOB,
    }
    observed = {
        "commit": commit,
        "config_blob": config_blob,
        "tokenizer_blob": tokenizer_blob,
    }
    if top_level != resolved:
        _fail("Inference root is not the Git top level", code="inference_root_mismatch")
    if status:
        _fail("Inference checkout has tracked modifications", code="dirty_inference_checkout")
    if observed != expected:
        _fail(
            "Official inference revision or decoder blobs differ from the pinned contract",
            code="inference_provenance_mismatch",
            expected=expected,
            observed=observed,
        )
    return {"root": str(resolved), **observed}


def _module_is_under(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _load_official_runtime(root: Path) -> OfficialInferenceRuntime:
    for name, module in tuple(sys.modules.items()):
        if name == "deepsvg" or name.startswith("deepsvg."):
            if module is not None and not _module_is_under(module, root):
                _fail(
                    f"Refusing conflicting {name!r} import",
                    code="upstream_module_collision",
                    module_file=getattr(module, "__file__", None),
                )
    try:
        import cairosvg
        import torch
    except ImportError as exc:
        _fail(
            "Experimental decode requires CPU-capable torch and CairoSVG",
            code="experimental_dependency_missing",
            dependency=str(exc),
        )

    module_name = f"_omnisvg_inference_tokenizer_{OMNISVG_INFERENCE_COMMIT[:12]}"
    tokenizer_path = root / "tokenizer.py"
    spec = importlib.util.spec_from_file_location(module_name, tokenizer_path)
    if spec is None or spec.loader is None:
        _fail("Unable to create tokenizer import spec", code="tokenizer_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(
            f"Unable to import pinned official tokenizer: {exc}",
            code="tokenizer_import_failed",
        )
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

    tokenizer = module.SVGTokenizer(str(root / "config.yaml"), model_size="4B")
    expected_constants = {
        "BOS_TOKEN_ID": 196998,
        "EOS_TOKEN_ID": 196999,
        "BASE_OFFSET": 151936,
        "CMD_TOKEN_START": 151939,
        "CMD_TOKEN_END": 151944,
        "COORD_TOKEN_START": 151944,
        "COLOR_COORD_BOUNDARY": 191947,
        "ARC_PARAM_START": 196436,
    }
    observed_constants = {
        name: int(getattr(tokenizer, name)) for name in expected_constants
    }
    if observed_constants != expected_constants:
        _fail(
            "Official 4B tokenizer constants differ from the audited bridge",
            code="inference_dialect_mismatch",
            expected=expected_constants,
            observed=observed_constants,
        )
    return OfficialInferenceRuntime(tokenizer=tokenizer, torch=torch, cairosvg=cairosvg)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registered_mapping(tokenizer_json: Path) -> dict[int, str]:
    path = tokenizer_json.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"Unable to read saved tokenizer JSON: {exc}", code="tokenizer_json_invalid")
    entries = payload.get("added_tokens") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        _fail("Saved tokenizer JSON lacks added_tokens", code="tokenizer_mapping_missing")
    mapping: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        token_id = entry.get("id")
        content = entry.get("content")
        if isinstance(token_id, int) and not isinstance(token_id, bool) and isinstance(content, str):
            mapping[token_id] = content
    for offset, upstream_id in enumerate(
        range(OPENVGLAB_MIN_PRODUCIBLE_ID, OPENVGLAB_MAX_PRODUCIBLE_ID + 1)
    ):
        registered_id = GEMMA_REGISTERED_ID_MIN + offset
        expected_token = f"<svgovg4b:{upstream_id}>"
        if mapping.get(registered_id) != expected_token:
            _fail(
                "Saved tokenizer does not contain the exact ordered OmniSVG namespace",
                code="registered_mapping_mismatch",
                registered_id=registered_id,
                expected_token=expected_token,
                observed_token=mapping.get(registered_id),
            )
    return {
        token_id: mapping[token_id]
        for token_id in range(GEMMA_REGISTERED_ID_MIN, GEMMA_REGISTERED_ID_MAX + 1)
    }


def _registered_to_training_ids(
    registered_ids: Sequence[int],
    registered_id_to_token: Mapping[int, str],
) -> tuple[int, ...]:
    if isinstance(registered_ids, (str, bytes)) or not registered_ids:
        _fail("Generated IDs must be a non-empty integer sequence", code="invalid_id_sequence")
    result: list[int] = []
    for index, registered_id in enumerate(registered_ids):
        if isinstance(registered_id, bool) or not isinstance(registered_id, int):
            _fail(
                "Generated sequence contains a non-integer ID",
                code="invalid_registered_id_type",
                index=index,
            )
        if not GEMMA_REGISTERED_ID_MIN <= registered_id <= GEMMA_REGISTERED_ID_MAX:
            _fail(
                "Generated ID is outside the registered OmniSVG namespace",
                code="registered_id_out_of_range",
                index=index,
                registered_id=registered_id,
            )
        token = registered_id_to_token.get(registered_id)
        if token is None:
            _fail(
                "Generated ID has no registered OmniSVG token mapping",
                code="unmapped_registered_id",
                index=index,
                registered_id=registered_id,
            )
        match = _NAMED_TOKEN.fullmatch(token)
        if match is None:
            _fail(
                "Registered ID maps to a malformed OmniSVG token string",
                code="invalid_named_token",
                index=index,
                token=token,
            )
        upstream_id = int(match.group(1))
        expected = OPENVGLAB_MIN_PRODUCIBLE_ID + (
            registered_id - GEMMA_REGISTERED_ID_MIN
        )
        if upstream_id != expected:
            _fail(
                "Registered token order differs from the trained tokenizer contract",
                code="registered_token_order_mismatch",
                index=index,
                expected_upstream_id=expected,
                observed_upstream_id=upstream_id,
            )
        result.append(upstream_id)
    framed = tuple(result)
    if len(framed) < 3:
        _fail("OmniSVG sequence has no body", code="empty_svg_body")
    if framed[0] != OPENVGLAB_BOS_ID or framed[-1] != OPENVGLAB_EOS_ID:
        _fail("OmniSVG sequence lacks exact BOS/EOS framing", code="invalid_framing")
    if OPENVGLAB_BOS_ID in framed[1:] or OPENVGLAB_EOS_ID in framed[:-1]:
        _fail("OmniSVG BOS/EOS occurs inside the sequence", code="embedded_framing_token")
    return framed


def _token_kind(token_id: int) -> str:
    if 151939 <= token_id < 151944:
        return "command"
    if (
        _PRODUCIBLE_INFERENCE_COORDINATE_MIN
        <= token_id
        <= _PRODUCIBLE_INFERENCE_COORDINATE_MAX
    ):
        return "coordinate"
    if _PRODUCIBLE_INFERENCE_COLOR_MIN <= token_id <= _PRODUCIBLE_INFERENCE_COLOR_MAX:
        return "color"
    if _PRODUCIBLE_INFERENCE_ARC_MIN <= token_id <= _PRODUCIBLE_INFERENCE_ARC_MAX:
        return "arc"
    _fail(
        "Shifted token has no official inference-token class",
        code="invalid_inference_token",
        token_id=token_id,
    )
    raise AssertionError("unreachable")


def _validate_inference_grammar(
    body_ids: tuple[int, ...],
    *,
    allow_trim_incomplete_final_path: bool = False,
) -> tuple[tuple[int, ...], int, int, dict[str, Any]]:
    index = 0
    command_count = 0
    path_count = 0
    commands_in_path = 0
    last_proven_color_end = 0
    retained_command_count = 0
    while index < len(body_ids):
        token_id = body_ids[index]
        kind = _token_kind(token_id)
        if kind == "color":
            if commands_in_path == 0:
                _fail(
                    "Color token has no preceding path commands",
                    code="orphan_color_token",
                    index=index,
                )
            path_count += 1
            commands_in_path = 0
            index += 1
            last_proven_color_end = index
            retained_command_count = command_count
            continue
        if kind != "command":
            _fail(
                "Expected a command or path-ending color token",
                code="unexpected_body_token",
                index=index,
                token_id=token_id,
                token_kind=kind,
            )
        expected_arguments = _INFERENCE_COMMAND_ARGUMENTS[token_id]
        if index + len(expected_arguments) >= len(body_ids):
            _fail("Truncated SVG command", code="truncated_command", index=index)
        for offset, expected_kind in enumerate(expected_arguments, start=1):
            argument = body_ids[index + offset]
            observed_kind = _token_kind(argument)
            if observed_kind != expected_kind:
                _fail(
                    "SVG command argument has the wrong token class",
                    code="invalid_command_argument",
                    index=index + offset,
                    expected_kind=expected_kind,
                    observed_kind=observed_kind,
                    token_id=argument,
                )
        command_count += 1
        commands_in_path += 1
        index += 1 + len(expected_arguments)
    if commands_in_path:
        if not allow_trim_incomplete_final_path or last_proven_color_end == 0:
            _fail("Final path lacks a color terminator", code="unterminated_path")
        retained_body = body_ids[:last_proven_color_end]
        trimmed_count = len(body_ids) - len(retained_body)
        return (
            retained_body,
            retained_command_count,
            path_count,
            {
                "recovery": True,
                "original_strict_error": "unterminated_path",
                "original_framed_token_count": len(body_ids) + 2,
                "original_body_token_count": len(body_ids),
                "retained_framed_token_count": len(retained_body) + 2,
                "retained_body_token_count": len(retained_body),
                "trimmed_body_token_count": trimmed_count,
                "retained_command_count": retained_command_count,
                "retained_path_count": path_count,
                "trim_policy": "drop-only-suffix-after-last-proven-color-terminator",
            },
        )
    if not path_count:
        _fail("Sequence contains no complete SVG path", code="empty_svg_paths")
    return (
        body_ids,
        command_count,
        path_count,
        {
            "recovery": False,
            "original_strict_error": None,
            "original_framed_token_count": len(body_ids) + 2,
            "original_body_token_count": len(body_ids),
            "retained_framed_token_count": len(body_ids) + 2,
            "retained_body_token_count": len(body_ids),
            "trimmed_body_token_count": 0,
            "retained_command_count": command_count,
            "retained_path_count": path_count,
        },
    )


def _validate_svg(svg: str) -> None:
    if not isinstance(svg, str) or not svg.strip():
        _fail("Official decoder returned empty SVG", code="empty_svg_output")
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        _fail(f"Official decoder returned invalid XML: {exc}", code="invalid_svg_xml")
    if root.tag.rsplit("}", 1)[-1] != "svg":
        _fail("Official decoder XML root is not svg", code="invalid_svg_root")


def _decode(
    registered_ids: Sequence[int],
    *,
    mapping: Mapping[int, str],
    runtime: OfficialInferenceRuntime,
    render_size: int,
    allow_trim_incomplete_final_path: bool = False,
) -> DecodeArtifact:
    original_registered_ids = tuple(registered_ids)
    original_training_ids = _registered_to_training_ids(original_registered_ids, mapping)
    original_inference_body = tuple(
        token_id + 1 for token_id in original_training_ids[1:-1]
    )
    inference_body, command_count, path_count, recovery = (
        _validate_inference_grammar(
            original_inference_body,
            allow_trim_incomplete_final_path=allow_trim_incomplete_final_path,
        )
    )
    retained_body_count = len(inference_body)
    registered_ids = (
        original_registered_ids[0],
        *original_registered_ids[1 : retained_body_count + 1],
        original_registered_ids[-1],
    )
    training_ids = (
        original_training_ids[0],
        *original_training_ids[1 : retained_body_count + 1],
        original_training_ids[-1],
    )
    inference_ids = (OPENVGLAB_BOS_ID, *inference_body, OPENVGLAB_EOS_ID)
    tensor = runtime.torch.tensor([inference_ids], dtype=runtime.torch.long, device="cpu")
    try:
        pixels = runtime.tokenizer.process_generated_tokens(tensor)
        svg_tensors, colors = runtime.tokenizer.raster_svg(pixels)
        paths = svg_tensors[0] if svg_tensors else []
        if len(paths) != path_count or len(colors) != path_count:
            _fail(
                "Official decoder silently dropped paths or colors",
                code="decoder_path_count_mismatch",
                expected_paths=path_count,
                decoded_paths=len(paths),
                decoded_colors=len(colors),
            )
        svg = runtime.tokenizer.apply_colors_to_svg(paths, colors).to_str()
    except ExperimentalOmniSVGDecodeError:
        raise
    except Exception as exc:
        _fail(f"Official inference decoder failed: {exc}", code="official_decode_failed")
    _validate_svg(svg)
    try:
        png = runtime.cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=render_size,
            output_height=render_size,
        )
    except Exception as exc:
        _fail(f"Decoded SVG failed CPU rendering: {exc}", code="render_failed")
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        _fail("Renderer did not return a PNG", code="invalid_render_output")
    return DecodeArtifact(
        svg=svg,
        png=png,
        registered_ids=tuple(registered_ids),
        training_ids=training_ids,
        inference_ids=inference_ids,
        command_count=command_count,
        path_count=path_count,
        original_registered_ids=original_registered_ids,
        original_training_ids=original_training_ids,
        recovery=bool(recovery["recovery"]),
        recovery_diagnostics=recovery,
    )


def _normalized_rgba_mae(first_png: bytes, second_png: bytes) -> float:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        _fail(
            f"Cached render comparison requires numpy and Pillow: {exc}",
            code="benchmark_dependency_missing",
        )
    first = np.asarray(Image.open(io.BytesIO(first_png)).convert("RGBA"), dtype=np.float32)
    second = np.asarray(Image.open(io.BytesIO(second_png)).convert("RGBA"), dtype=np.float32)
    if first.shape != second.shape:
        _fail("Cached renders have different shapes", code="render_shape_mismatch")
    return float(np.abs(first - second).mean() / 255.0)


def _ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def _write_artifact(directory: Path, name: str, artifact: DecodeArtifact) -> None:
    (directory / f"{name}.svg").write_text(artifact.svg, encoding="utf-8")
    (directory / f"{name}.png").write_bytes(artifact.png)


def run_experimental_decode(
    *,
    experimental_enable: bool,
    upstream_root: Path,
    tokenizer_json: Path,
    cache_config: OpenVGLabCacheConfig,
    split: str,
    source_line: int,
    output_dir: Path,
    generated_ids_json: Path | None = None,
    render_size: int = 512,
    allow_trim_incomplete_final_path: bool = False,
) -> dict[str, Any]:
    """Run one mandatory cached gate, then optionally decode generated Gemma IDs."""

    if experimental_enable is not True:
        _fail(
            "Experimental OmniSVG decode requires an explicit opt-in flag",
            code="experimental_opt_in_required",
        )
    if not isinstance(render_size, int) or isinstance(render_size, bool) or render_size < 16:
        _fail("render_size must be an integer >= 16", code="invalid_render_size")
    if not isinstance(allow_trim_incomplete_final_path, bool):
        _fail(
            "allow_trim_incomplete_final_path must be boolean",
            code="invalid_recovery_option",
        )
    warnings.warn(EXPERIMENTAL_WARNING, ExperimentalOmniSVGDecodeWarning, stacklevel=2)
    provenance = _validate_inference_checkout(upstream_root)
    mapping = _load_registered_mapping(tokenizer_json)
    runtime = _load_official_runtime(upstream_root.expanduser().resolve())

    with PinnedOpenVGLabTokenCache(cache_config) as cache:
        records = cache.load_split(split)
    if source_line < 1 or source_line > len(records):
        _fail(
            "Cached gate source line is outside the split",
            code="cached_gate_line_out_of_range",
            split=split,
            source_line=source_line,
        )
    record = records[source_line - 1]
    payload = record.get(CACHED_TARGET_FIELD)
    if not isinstance(payload, Mapping) or payload.get("source_line") != source_line:
        _fail("Cached gate record provenance is malformed", code="cached_gate_invalid")
    source_svg = record.get("output_svg")
    if not isinstance(source_svg, str) or not source_svg:
        _fail("Cached gate record lacks source SVG", code="cached_gate_invalid")
    if hashlib.sha256(source_svg.encode("utf-8")).hexdigest() != payload.get("svg_sha256"):
        _fail("Cached gate source SVG hash mismatch", code="cached_gate_svg_hash_mismatch")

    dialect = CachedOpenVGLabGemmaDialect()
    named_tokens = dialect.target_tokens_for_record(record)
    token_to_registered = {token: token_id for token_id, token in mapping.items()}
    try:
        gate_registered_ids = tuple(token_to_registered[token] for token in named_tokens)
    except KeyError as exc:
        _fail(
            f"Cached target token is absent from saved tokenizer: {exc}",
            code="cached_gate_mapping_mismatch",
        )
    gate = _decode(
        gate_registered_ids,
        mapping=mapping,
        runtime=runtime,
        render_size=render_size,
        allow_trim_incomplete_final_path=False,
    )
    try:
        source_png = runtime.cairosvg.svg2png(
            bytestring=source_svg.encode("utf-8"),
            output_width=render_size,
            output_height=render_size,
        )
    except Exception as exc:
        _fail(f"Cached source SVG failed rendering: {exc}", code="source_render_failed")
    rgba_mae = _normalized_rgba_mae(gate.png, source_png)

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        _fail(
            f"Output directory already exists: {destination}",
            code="output_exists",
        )
    destination.mkdir(parents=True)
    _write_artifact(destination, "gate_decoded", gate)
    (destination / "gate_source.svg").write_text(source_svg, encoding="utf-8")
    (destination / "gate_source.png").write_bytes(source_png)

    candidate: DecodeArtifact | None = None
    if generated_ids_json is not None:
        try:
            generated_payload = json.loads(
                generated_ids_json.expanduser().resolve().read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(f"Unable to read generated IDs JSON: {exc}", code="generated_ids_invalid")
        if not isinstance(generated_payload, list):
            _fail("Generated IDs JSON root must be an array", code="generated_ids_invalid")
        candidate = _decode(
            generated_payload,
            mapping=mapping,
            runtime=runtime,
            render_size=render_size,
            allow_trim_incomplete_final_path=allow_trim_incomplete_final_path,
        )
        _write_artifact(destination, "candidate_decoded", candidate)
        if candidate.recovery:
            (destination / "candidate_recovered_framed_gemma_ids.json").write_text(
                json.dumps(list(candidate.registered_ids), separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experimental": True,
        "warning": EXPERIMENTAL_WARNING,
        "dialect": EXPERIMENTAL_DIALECT,
        "production_inference_enabled": False,
        "training_source_revision": OPENVGLAB_TRAIN_COMMIT,
        "inference_source": provenance,
        "tokenizer_json": str(tokenizer_json.expanduser().resolve()),
        "tokenizer_json_sha256": _sha256_file(tokenizer_json.expanduser().resolve()),
        "registered_id_min": GEMMA_REGISTERED_ID_MIN,
        "registered_id_max": GEMMA_REGISTERED_ID_MAX,
        "mapping_contract": "registered-id lookup then strict <svgovg4b:N> parse",
        "body_bridge": "add exactly 1 to each non-BOS/EOS training ID",
        "cached_gate": {
            "passed": True,
            "split": split,
            "source_line": source_line,
            "sample_id": payload.get("sample_id"),
            "cache_sha256": cache_config.expected_cache_sha256,
            "audit_sha256": cache_config.expected_audit_sha256,
            "registered_ids_sha256": _ids_sha256(gate.registered_ids),
            "training_ids_sha256": _ids_sha256(gate.training_ids),
            "inference_ids_sha256": _ids_sha256(gate.inference_ids),
            "framed_token_count": len(gate.training_ids),
            "body_token_count": len(gate.training_ids) - 2,
            "command_count": gate.command_count,
            "path_count": gate.path_count,
            "decoded_svg_sha256": hashlib.sha256(gate.svg.encode("utf-8")).hexdigest(),
            "decoded_png_sha256": hashlib.sha256(gate.png).hexdigest(),
            "source_svg_sha256": hashlib.sha256(source_svg.encode("utf-8")).hexdigest(),
            "source_png_sha256": hashlib.sha256(source_png).hexdigest(),
            "rgba_mae_normalized": rgba_mae,
            "render_size": render_size,
        },
        "candidate": None,
    }
    if candidate is not None:
        manifest["candidate"] = {
            "generated_ids_path": str(generated_ids_json.expanduser().resolve()),
            "recovery": candidate.recovery,
            "recovery_diagnostics": dict(candidate.recovery_diagnostics),
            "original_registered_ids_sha256": _ids_sha256(
                candidate.original_registered_ids
            ),
            "original_training_ids_sha256": _ids_sha256(
                candidate.original_training_ids
            ),
            "registered_ids_sha256": _ids_sha256(candidate.registered_ids),
            "training_ids_sha256": _ids_sha256(candidate.training_ids),
            "inference_ids_sha256": _ids_sha256(candidate.inference_ids),
            "framed_token_count": len(candidate.training_ids),
            "command_count": candidate.command_count,
            "path_count": candidate.path_count,
            "decoded_svg_sha256": hashlib.sha256(candidate.svg.encode("utf-8")).hexdigest(),
            "decoded_png_sha256": hashlib.sha256(candidate.png).hexdigest(),
            "recovered_framed_ids_path": (
                str(destination / "candidate_recovered_framed_gemma_ids.json")
                if candidate.recovery
                else None
            ),
        }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental-enable-official-4b-plus-one-decode", action="store_true")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--source-line", type=int, default=874)
    parser.add_argument("--generated-ids-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--allow-trim-incomplete-final-path", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_experimental_decode(
        experimental_enable=args.experimental_enable_official_4b_plus_one_decode,
        upstream_root=args.upstream_root,
        tokenizer_json=args.tokenizer_json,
        cache_config=OpenVGLabCacheConfig(
            cache_path=args.cache_path,
            audit_manifest_path=args.audit_manifest,
            prepared_root=args.prepared_root,
        ),
        split=args.split,
        source_line=args.source_line,
        generated_ids_json=args.generated_ids_json,
        output_dir=args.output_dir,
        render_size=args.render_size,
        allow_trim_incomplete_final_path=args.allow_trim_incomplete_final_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
