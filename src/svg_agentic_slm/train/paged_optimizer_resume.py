"""Safe, state-preserving recovery of bitsandbytes paged optimizer checkpoints.

PyTorch serialization preserves tensor attributes such as ``is_paged`` but does
not recreate the CUDA managed-memory allocation that bitsandbytes originally
used.  ``Optimizer.load_state_dict`` then maps tensors to current parameters via
``Tensor.to``, which can discard those attributes before a post-load hook sees
them.  This module captures paging metadata and numeric hashes from the raw
state dict passed to the standard loader, lets that loader perform its normal
parameter mapping, and transactionally replaces only the captured buffers with
fresh allocations from the current bitsandbytes optimizer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PagedStateAllocator = Callable[[Any, Any, Any, str], Any]

_PAGED_STATE_KEYS = frozenset({"state1", "state2"})
_BNB_QUANT_STATE_WRAPPER_KEY = "__bnb_optimizer_quant_state__"
_BNB_QUANT_STATE_KEYS = frozenset(
    {
        "qmap1",
        "qmap2",
        "max1",
        "max2",
        "new_max1",
        "new_max2",
        "state1",
        "state2",
        "gnorm_vec",
        "absmax1",
        "absmax2",
        "unorm_vec",
    }
)
_MAX_BNB_QUANT_STATE_WRAPPER_DEPTH = 1
_REHYDRATED_IDS_ATTR = "_svg_agentic_slm_rehydrated_paged_state_ids"


class PagedOptimizerResumeError(RuntimeError):
    """Raised before optimizer state is committed when safe recovery is impossible."""


@dataclass(frozen=True)
class PagedOptimizerResumeAudit:
    """JSON-serializable provenance for one optimizer resume operation."""

    schema_version: int
    enabled: bool
    status: str
    source_checkpoint: str | None
    optimizer_class: str | None
    detected_paged_tensors: int
    rehydrated_paged_tensors: int
    verified_paged_tensors: int
    rehydrated_bytes: int
    nonpaged_state_entries_preserved: int
    state_key_counts: dict[str, int]
    source_page_device_ids: tuple[int | None, ...]
    target_page_device_ids: tuple[int, ...]
    source_state_sha256: str | None
    rehydrated_state_sha256: str | None
    numeric_values_preserved: bool
    storage_strategy: str | None
    checkpoint_mutated: bool = False

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_page_device_ids"] = list(self.source_page_device_ids)
        payload["target_page_device_ids"] = list(self.target_page_device_ids)
        return payload


@dataclass(frozen=True)
class _PagedCandidate:
    path: str
    state: dict[str, Any]
    key: str
    parameter: Any
    source: Any
    expected_page_device_id: int
    source_page_device_id: int | None
    serialized_sha256: str


@dataclass(frozen=True)
class _SerializedPagedTensor:
    path: str
    group_index: int
    parameter_index: int
    key: str
    shape: tuple[int, ...]
    dtype: Any
    source_page_device_id: int | None
    source_sha256: str


@dataclass(frozen=True)
class _SerializedPagedState:
    tensors: tuple[_SerializedPagedTensor, ...]


@dataclass(frozen=True)
class _PreparedReplacement:
    candidate: _PagedCandidate
    destination: Any
    source_sha256: str
    destination_sha256: str


def initial_paged_optimizer_resume_manifest(
    *,
    enabled: bool,
    checkpoint: str | Path | None,
) -> dict[str, Any]:
    """Return provenance used before Trainer invokes its checkpoint loader."""

    status = "pending" if checkpoint is not None else "fresh_run"
    return _empty_audit(
        enabled=enabled,
        status=status,
        checkpoint=checkpoint,
        optimizer=None,
    ).to_manifest()


def unwrap_optimizer(optimizer: Any) -> Any:
    """Unwrap common optimizer wrappers without depending on Accelerate."""

    current = optimizer
    visited: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        if is_bitsandbytes_paged_optimizer(current):
            return current
        wrapped = getattr(current, "optimizer", None)
        if wrapped is None or wrapped is current:
            break
        current = wrapped
    return current


def is_bitsandbytes_paged_optimizer(optimizer: Any) -> bool:
    """Identify a paged bitsandbytes optimizer, including subclasses."""

    if optimizer is None or getattr(optimizer, "is_paged", False) is not True:
        return False
    return any(
        cls.__module__ == "bitsandbytes" or cls.__module__.startswith("bitsandbytes.")
        for cls in type(optimizer).__mro__
    )


def rehydrate_paged_optimizer_state(
    optimizer: Any,
    *,
    checkpoint: str | Path,
    parameter_names: Mapping[int, str] | None = None,
    allocator: PagedStateAllocator | None = None,
    _serialized_state: _SerializedPagedState | None = None,
) -> PagedOptimizerResumeAudit:
    """Replace deserialized paged state with fresh CUDA managed-memory buffers.

    All allocations, copies, metadata validation, and byte checksums complete
    before any optimizer state reference is replaced.  On failure, temporary
    page-manager registrations are removed and the loaded state is untouched.
    """

    import torch

    core = unwrap_optimizer(optimizer)
    if not is_bitsandbytes_paged_optimizer(core):
        raise PagedOptimizerResumeError(
            "Paged state rehydration requires a bitsandbytes optimizer with is_paged=True."
        )
    if not isinstance(getattr(core, "state", None), Mapping):
        raise PagedOptimizerResumeError("Paged optimizer state must be a mapping.")

    candidates: list[_PagedCandidate] = []
    already_rehydrated = 0
    nonpaged_entries = 0
    current_rehydrated_ids = getattr(core, _REHYDRATED_IDS_ATTR, frozenset())
    if not isinstance(current_rehydrated_ids, (set, frozenset)):
        raise PagedOptimizerResumeError("Invalid in-memory paged-state recovery marker.")

    serialized_by_location = (
        {
            (item.group_index, item.parameter_index, item.key): item
            for item in _serialized_state.tensors
        }
        if _serialized_state is not None
        else {}
    )
    if _serialized_state is not None and len(serialized_by_location) != len(
        _serialized_state.tensors
    ):
        raise PagedOptimizerResumeError(
            "Serialized optimizer paging metadata contains duplicate state locations."
        )
    resolved_serialized_locations: set[tuple[int, int, str]] = set()

    for group_index, group in enumerate(getattr(core, "param_groups", [])):
        parameters = group.get("params", []) if isinstance(group, Mapping) else []
        for parameter_index, parameter in enumerate(parameters):
            state = core.state.get(parameter)
            if not isinstance(state, dict):
                if state is not None:
                    raise PagedOptimizerResumeError(
                        "Each paged optimizer parameter state must be a mutable dictionary."
                    )
                continue
            parameter_name = (
                parameter_names.get(id(parameter)) if parameter_names is not None else None
            )
            parameter_label = parameter_name or f"group{group_index}.param{parameter_index}"
            for key, value in state.items():
                location = (group_index, parameter_index, key)
                serialized = serialized_by_location.get(location)
                if _serialized_state is not None:
                    if serialized is None:
                        nonpaged_entries += 1
                        continue
                    resolved_serialized_locations.add(location)
                    if not torch.is_tensor(value):
                        raise PagedOptimizerResumeError(
                            f"Mapped paged optimizer state {serialized.path} is not a tensor."
                        )
                    _validate_source_tensor(value, parameter, path=serialized.path)
                    if value.dtype != serialized.dtype or tuple(value.shape) != serialized.shape:
                        raise PagedOptimizerResumeError(
                            f"Mapped paged optimizer state metadata changed for {serialized.path}."
                        )
                    mapped_sha256 = _tensor_sha256(value)
                    if mapped_sha256 != serialized.source_sha256:
                        raise PagedOptimizerResumeError(
                            f"Mapped paged optimizer numeric state changed for {serialized.path}."
                        )
                    candidates.append(
                        _PagedCandidate(
                            path=serialized.path,
                            state=state,
                            key=key,
                            parameter=parameter,
                            source=value,
                            expected_page_device_id=_parameter_cuda_device_id(
                                parameter, parameter_label
                            ),
                            source_page_device_id=serialized.source_page_device_id,
                            serialized_sha256=serialized.source_sha256,
                        )
                    )
                    continue
                if not (torch.is_tensor(value) and getattr(value, "is_paged", False) is True):
                    nonpaged_entries += 1
                    continue
                if key not in _PAGED_STATE_KEYS:
                    raise PagedOptimizerResumeError(
                        f"Unexpected paged optimizer state key {key!r} for {parameter_label}."
                    )
                if id(value) in current_rehydrated_ids:
                    already_rehydrated += 1
                    continue
                _validate_source_tensor(value, parameter, path=f"{parameter_label}.{key}")
                expected_device_id = _parameter_cuda_device_id(parameter, parameter_label)
                candidates.append(
                    _PagedCandidate(
                        path=f"{parameter_label}.{key}",
                        state=state,
                        key=key,
                        parameter=parameter,
                        source=value,
                        expected_page_device_id=expected_device_id,
                        source_page_device_id=getattr(value, "page_deviceid", None),
                        serialized_sha256=_tensor_sha256(value),
                    )
                )

    if _serialized_state is not None:
        missing = set(serialized_by_location).difference(resolved_serialized_locations)
        if missing:
            locations = ", ".join(
                item.path
                for item in _serialized_state.tensors
                if (item.group_index, item.parameter_index, item.key) in missing
            )
            raise PagedOptimizerResumeError(
                f"Standard optimizer loading did not map captured paged state: {locations}."
            )

    detected = len(candidates) + already_rehydrated
    if already_rehydrated and candidates:
        raise PagedOptimizerResumeError(
            "Optimizer contains a partially rehydrated paged state; refusing mixed storage."
        )
    if already_rehydrated:
        entries = [
            (candidate_path, value)
            for candidate_path, value in _iter_paged_state(core, parameter_names)
        ]
        aggregate = _aggregate_tensor_checksum(entries)
        return PagedOptimizerResumeAudit(
            schema_version=2,
            enabled=True,
            status="already_rehydrated",
            source_checkpoint=_resolved_checkpoint(checkpoint),
            optimizer_class=_qualified_class_name(core),
            detected_paged_tensors=detected,
            rehydrated_paged_tensors=0,
            verified_paged_tensors=already_rehydrated,
            rehydrated_bytes=0,
            nonpaged_state_entries_preserved=nonpaged_entries,
            state_key_counts=dict(sorted(Counter(path.rsplit(".", 1)[-1] for path, _ in entries).items())),
            source_page_device_ids=tuple(
                sorted({getattr(value, "page_deviceid", None) for _, value in entries}, key=_none_first)
            ),
            target_page_device_ids=tuple(
                sorted({int(getattr(value, "page_deviceid")) for _, value in entries})
            ),
            source_state_sha256=aggregate,
            rehydrated_state_sha256=aggregate,
            numeric_values_preserved=True,
            storage_strategy="fresh_cuda_managed_allocation_copy",
        )
    if not candidates:
        return _empty_audit(
            enabled=True,
            status="no_serialized_paged_state",
            checkpoint=checkpoint,
            optimizer=core,
            nonpaged_entries=nonpaged_entries,
        )

    page_manager = getattr(core, "page_mng", None)
    registered = getattr(page_manager, "paged_tensors", None)
    if not isinstance(registered, list):
        raise PagedOptimizerResumeError(
            "Paged optimizer does not expose a mutable page_mng.paged_tensors registry."
        )
    registry_start = len(registered)
    allocate = allocator or _allocate_with_bitsandbytes
    prepared: list[_PreparedReplacement] = []
    try:
        for candidate in candidates:
            destination = allocate(core, candidate.parameter, candidate.source, candidate.key)
            _validate_destination_tensor(destination, candidate, registered)
            source_sha256 = _tensor_sha256(candidate.source)
            if source_sha256 != candidate.serialized_sha256:
                raise PagedOptimizerResumeError(
                    f"Paged state changed while preparing {candidate.path}."
                )
            with torch.no_grad():
                destination.copy_(candidate.source, non_blocking=False)
            destination_sha256 = _tensor_sha256(destination)
            if candidate.serialized_sha256 != destination_sha256:
                raise PagedOptimizerResumeError(
                    f"Paged state checksum mismatch after copying {candidate.path}."
                )
            prepared.append(
                _PreparedReplacement(
                    candidate=candidate,
                    destination=destination,
                    source_sha256=source_sha256,
                    destination_sha256=destination_sha256,
                )
            )
    except Exception as exc:
        del registered[registry_start:]
        if isinstance(exc, PagedOptimizerResumeError):
            raise
        raise PagedOptimizerResumeError(
            "Failed to prepare paged optimizer state; loaded state was not modified."
        ) from exc

    for replacement in prepared:
        replacement.candidate.state[replacement.candidate.key] = replacement.destination
    setattr(core, _REHYDRATED_IDS_ATTR, frozenset(id(item.destination) for item in prepared))

    source_entries = [
        (item.candidate.path, item.candidate.serialized_sha256) for item in prepared
    ]
    destination_entries = [
        (item.candidate.path, item.destination_sha256) for item in prepared
    ]
    source_aggregate = _aggregate_digest_entries(source_entries)
    destination_aggregate = _aggregate_digest_entries(destination_entries)
    if source_aggregate != destination_aggregate:
        raise AssertionError("Internal paged optimizer aggregate checksum mismatch.")
    audit = PagedOptimizerResumeAudit(
        schema_version=2,
        enabled=True,
        status="rehydrated",
        source_checkpoint=_resolved_checkpoint(checkpoint),
        optimizer_class=_qualified_class_name(core),
        detected_paged_tensors=len(prepared),
        rehydrated_paged_tensors=len(prepared),
        verified_paged_tensors=len(prepared),
        rehydrated_bytes=sum(_tensor_nbytes(item.candidate.source) for item in prepared),
        nonpaged_state_entries_preserved=nonpaged_entries,
        state_key_counts=dict(
            sorted(Counter(item.candidate.key for item in prepared).items())
        ),
        source_page_device_ids=tuple(
            sorted(
                {item.candidate.source_page_device_id for item in prepared},
                key=_none_first,
            )
        ),
        target_page_device_ids=tuple(
            sorted({item.candidate.expected_page_device_id for item in prepared})
        ),
        source_state_sha256=source_aggregate,
        rehydrated_state_sha256=destination_aggregate,
        numeric_values_preserved=True,
        storage_strategy=(
            "pre_load_metadata_capture_standard_mapping_fresh_cuda_managed_allocation_copy"
            if _serialized_state is not None
            else "fresh_cuda_managed_allocation_copy"
        ),
    )
    setattr(core, "_svg_agentic_slm_paged_resume_audit", audit.to_manifest())
    return audit


class PagedOptimizerResumeMixin:
    """HF Trainer mixin that captures raw paging metadata before checkpoint load."""

    _svg_enable_paged_optimizer_resume_rehydration = True
    _svg_paged_optimizer_state_allocator: PagedStateAllocator | None = None
    _svg_paged_optimizer_resume_audit: dict[str, Any] | None = None

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        enabled = self._svg_enable_paged_optimizer_resume_rehydration
        if not isinstance(enabled, bool):
            raise PagedOptimizerResumeError(
                "Paged optimizer resume rehydration toggle must be boolean."
            )
        optimizer_before_load = getattr(self, "optimizer", None)
        core_before_load = unwrap_optimizer(optimizer_before_load)
        model = getattr(self, "model", None)
        parameter_names = (
            {id(parameter): name for name, parameter in model.named_parameters()}
            if model is not None and hasattr(model, "named_parameters")
            else None
        )
        should_capture = (
            checkpoint is not None
            and enabled
            and is_bitsandbytes_paged_optimizer(core_before_load)
        )
        captured: list[_SerializedPagedState] = []
        if should_capture:
            with _capture_optimizer_load_state_dict(
                optimizer_before_load,
                core_before_load,
                parameter_names,
            ) as captured:
                super()._load_optimizer_and_scheduler(checkpoint)
        else:
            super()._load_optimizer_and_scheduler(checkpoint)

        optimizer = getattr(self, "optimizer", None)
        core = unwrap_optimizer(optimizer)
        if checkpoint is None:
            audit = _empty_audit(
                enabled=enabled,
                status="fresh_run",
                checkpoint=None,
                optimizer=core,
            )
        elif not enabled:
            audit = _empty_audit(
                enabled=False,
                status="disabled",
                checkpoint=checkpoint,
                optimizer=core,
            )
        elif not is_bitsandbytes_paged_optimizer(core):
            audit = _empty_audit(
                enabled=True,
                status="not_paged_bitsandbytes",
                checkpoint=checkpoint,
                optimizer=core,
            )
        else:
            if not should_capture or core is not core_before_load:
                raise PagedOptimizerResumeError(
                    "Paged optimizer checkpoint state was loaded without a safe raw-state "
                    "interception boundary."
                )
            if not captured:
                if getattr(core, "state", None):
                    raise PagedOptimizerResumeError(
                        "Paged optimizer state was populated without an intercepted "
                        "load_state_dict call."
                    )
                audit = _empty_audit(
                    enabled=True,
                    status="optimizer_state_not_loaded",
                    checkpoint=checkpoint,
                    optimizer=core,
                )
            else:
                audit = rehydrate_paged_optimizer_state(
                    core,
                    checkpoint=checkpoint,
                    parameter_names=parameter_names,
                    allocator=self._svg_paged_optimizer_state_allocator,
                    _serialized_state=captured[0],
                )
        manifest = audit.to_manifest()
        args = getattr(self, "args", None)
        manifest["process_index"] = getattr(args, "process_index", None)
        manifest["world_size"] = getattr(args, "world_size", None)
        self._svg_paged_optimizer_resume_audit = manifest
        logger.info("Paged optimizer resume audit: %s", json.dumps(manifest, sort_keys=True))


@contextmanager
def _capture_optimizer_load_state_dict(
    optimizer: Any,
    core: Any,
    parameter_names: Mapping[int, str] | None,
) -> Iterator[list[_SerializedPagedState]]:
    """Intercept Trainer's already-deserialized payload without loading another pickle.

    Transformers retains responsibility for checkpoint path validation and safe
    deserialization (including its ``weights_only`` policy).  Only compact
    descriptors and hashes survive the call, so raw checkpoint tensors are not
    duplicated while the standard loader maps them to current parameters.
    """

    load_state_dict = getattr(optimizer, "load_state_dict", None)
    if not callable(load_state_dict):
        raise PagedOptimizerResumeError(
            "Paged optimizer wrapper does not expose load_state_dict for interception."
        )
    try:
        namespace = vars(optimizer)
    except TypeError as exc:
        raise PagedOptimizerResumeError(
            "Paged optimizer wrapper cannot install a safe load_state_dict interception."
        ) from exc
    missing = object()
    previous_instance_value = namespace.get("load_state_dict", missing)
    captured: list[_SerializedPagedState] = []

    def intercepted_load_state_dict(state_dict: Mapping[str, Any]) -> Any:
        if captured:
            raise PagedOptimizerResumeError(
                "Optimizer load_state_dict was invoked more than once for one checkpoint."
            )
        captured.append(
            _capture_serialized_paged_state(core, state_dict, parameter_names)
        )
        return load_state_dict(state_dict)

    try:
        setattr(optimizer, "load_state_dict", intercepted_load_state_dict)
    except Exception as exc:
        raise PagedOptimizerResumeError(
            "Paged optimizer wrapper rejected safe load_state_dict interception."
        ) from exc
    try:
        yield captured
    finally:
        if previous_instance_value is missing:
            delattr(optimizer, "load_state_dict")
        else:
            setattr(optimizer, "load_state_dict", previous_instance_value)


def _capture_serialized_paged_state(
    optimizer: Any,
    optimizer_state_dict: Mapping[str, Any],
    parameter_names: Mapping[int, str] | None,
) -> _SerializedPagedState:
    """Capture paging identity using the same positional mapping as Optimizer."""

    import torch

    if not isinstance(optimizer_state_dict, Mapping):
        raise PagedOptimizerResumeError("Serialized optimizer state must be a mapping.")
    saved_state = optimizer_state_dict.get("state")
    saved_groups = optimizer_state_dict.get("param_groups")
    current_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, Sequence):
        raise PagedOptimizerResumeError(
            "Serialized optimizer state requires state and param_groups mappings."
        )
    if not isinstance(current_groups, Sequence) or len(saved_groups) != len(current_groups):
        raise PagedOptimizerResumeError(
            "Serialized optimizer parameter-group count does not match the current optimizer."
        )

    captured: list[_SerializedPagedTensor] = []
    for group_index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups, strict=True)
    ):
        if not isinstance(saved_group, Mapping) or not isinstance(current_group, Mapping):
            raise PagedOptimizerResumeError("Optimizer parameter groups must be mappings.")
        saved_parameters = saved_group.get("params")
        current_parameters = current_group.get("params")
        if not isinstance(saved_parameters, Sequence) or not isinstance(
            current_parameters, Sequence
        ):
            raise PagedOptimizerResumeError(
                "Optimizer parameter groups require positional params sequences."
            )
        if len(saved_parameters) != len(current_parameters):
            raise PagedOptimizerResumeError(
                f"Serialized optimizer group {group_index} parameter count changed."
            )
        for parameter_index, (saved_parameter_id, parameter) in enumerate(
            zip(saved_parameters, current_parameters, strict=True)
        ):
            saved_parameter_state = saved_state.get(saved_parameter_id, {})
            if not isinstance(saved_parameter_state, Mapping):
                raise PagedOptimizerResumeError(
                    "Each serialized optimizer parameter state must be a mapping."
                )
            parameter_name = (
                parameter_names.get(id(parameter)) if parameter_names is not None else None
            )
            parameter_label = parameter_name or f"group{group_index}.param{parameter_index}"
            logical_parameter_state = _unwrap_serialized_bnb_quant_state(
                saved_parameter_state,
                parameter_label=parameter_label,
            )
            for key, value in logical_parameter_state.items():
                if not (torch.is_tensor(value) and getattr(value, "is_paged", False) is True):
                    continue
                if key not in _PAGED_STATE_KEYS:
                    raise PagedOptimizerResumeError(
                        f"Unexpected serialized paged optimizer state key {key!r} "
                        f"for {parameter_label}."
                    )
                path = f"{parameter_label}.{key}"
                _validate_source_tensor(value, parameter, path=path)
                page_device_id = getattr(value, "page_deviceid", None)
                if page_device_id is not None:
                    if isinstance(page_device_id, bool):
                        raise PagedOptimizerResumeError(
                            f"Serialized paged state {path} has an invalid page device."
                        )
                    try:
                        page_device_id = int(page_device_id)
                    except (TypeError, ValueError) as exc:
                        raise PagedOptimizerResumeError(
                            f"Serialized paged state {path} has an invalid page device."
                        ) from exc
                captured.append(
                    _SerializedPagedTensor(
                        path=path,
                        group_index=group_index,
                        parameter_index=parameter_index,
                        key=key,
                        shape=tuple(value.shape),
                        dtype=value.dtype,
                        source_page_device_id=page_device_id,
                        source_sha256=_tensor_sha256(value),
                    )
                )
    return _SerializedPagedState(tensors=tuple(captured))


def _unwrap_serialized_bnb_quant_state(
    parameter_state: Mapping[str, Any],
    *,
    parameter_label: str,
) -> dict[str, Any]:
    """Return bitsandbytes' logical state without mutating its serialized envelope."""

    if _BNB_QUANT_STATE_WRAPPER_KEY not in parameter_state:
        return dict(parameter_state)
    wrapper = parameter_state[_BNB_QUANT_STATE_WRAPPER_KEY]
    quant_state = _decode_bnb_quant_state_wrapper(
        wrapper,
        parameter_label=parameter_label,
        depth=1,
        active_container_ids={id(parameter_state)},
    )
    logical_state = {
        key: value
        for key, value in parameter_state.items()
        if key != _BNB_QUANT_STATE_WRAPPER_KEY
    }
    collisions = set(logical_state).intersection(quant_state)
    if collisions:
        names = ", ".join(sorted(str(key) for key in collisions))
        raise PagedOptimizerResumeError(
            f"Serialized bitsandbytes quant state for {parameter_label} collides with "
            f"top-level keys: {names}."
        )
    logical_state.update(quant_state)
    return logical_state


def _decode_bnb_quant_state_wrapper(
    wrapper: Any,
    *,
    parameter_label: str,
    depth: int,
    active_container_ids: set[int],
) -> dict[str, Any]:
    """Decode only Optimizer8bit's fixed dict wrapper schema."""

    import torch

    if id(wrapper) in active_container_ids:
        raise PagedOptimizerResumeError(
            f"Serialized bitsandbytes quant state for {parameter_label} contains a cycle."
        )
    if depth > _MAX_BNB_QUANT_STATE_WRAPPER_DEPTH:
        raise PagedOptimizerResumeError(
            f"Serialized bitsandbytes quant state for {parameter_label} exceeds wrapper depth."
        )
    if type(wrapper) is not dict:
        raise PagedOptimizerResumeError(
            f"Serialized bitsandbytes quant state for {parameter_label} must use a plain dict."
        )
    if not wrapper:
        raise PagedOptimizerResumeError(
            f"Serialized bitsandbytes quant state for {parameter_label} is empty."
        )

    active_container_ids.add(id(wrapper))
    decoded: dict[str, Any] = {}
    try:
        for key, value in wrapper.items():
            if key == _BNB_QUANT_STATE_WRAPPER_KEY:
                nested = _decode_bnb_quant_state_wrapper(
                    value,
                    parameter_label=parameter_label,
                    depth=depth + 1,
                    active_container_ids=active_container_ids,
                )
                collisions = set(decoded).intersection(nested)
                if collisions:
                    raise PagedOptimizerResumeError(
                        f"Serialized bitsandbytes quant state for {parameter_label} "
                        "contains duplicate nested keys."
                    )
                decoded.update(nested)
                continue
            if key not in _BNB_QUANT_STATE_KEYS:
                raise PagedOptimizerResumeError(
                    f"Serialized bitsandbytes quant state for {parameter_label} has "
                    f"unknown key {key!r}."
                )
            if isinstance(value, Mapping) and id(value) in active_container_ids:
                raise PagedOptimizerResumeError(
                    f"Serialized bitsandbytes quant state for {parameter_label} "
                    "contains a cycle."
                )
            if not torch.is_tensor(value):
                raise PagedOptimizerResumeError(
                    f"Serialized bitsandbytes quant state {parameter_label}.{key} "
                    "must be a tensor."
                )
            decoded[key] = value
    finally:
        active_container_ids.remove(id(wrapper))
    return decoded


def _empty_audit(
    *,
    enabled: bool,
    status: str,
    checkpoint: str | Path | None,
    optimizer: Any,
    nonpaged_entries: int = 0,
) -> PagedOptimizerResumeAudit:
    return PagedOptimizerResumeAudit(
        schema_version=2,
        enabled=enabled,
        status=status,
        source_checkpoint=_resolved_checkpoint(checkpoint),
        optimizer_class=_qualified_class_name(optimizer) if optimizer is not None else None,
        detected_paged_tensors=0,
        rehydrated_paged_tensors=0,
        verified_paged_tensors=0,
        rehydrated_bytes=0,
        nonpaged_state_entries_preserved=nonpaged_entries,
        state_key_counts={},
        source_page_device_ids=(),
        target_page_device_ids=(),
        source_state_sha256=None,
        rehydrated_state_sha256=None,
        numeric_values_preserved=status
        in {"fresh_run", "no_serialized_paged_state", "optimizer_state_not_loaded"},
        storage_strategy=None,
    )


def _allocate_with_bitsandbytes(optimizer: Any, parameter: Any, source: Any, key: str) -> Any:
    del key
    allocator = getattr(optimizer, "get_state_buffer", None)
    if not callable(allocator):
        raise PagedOptimizerResumeError(
            "Paged bitsandbytes optimizer does not expose get_state_buffer()."
        )
    return allocator(parameter, dtype=source.dtype)


def _validate_source_tensor(source: Any, parameter: Any, *, path: str) -> None:
    import torch

    if source.layout != torch.strided:
        raise PagedOptimizerResumeError(f"Paged state {path} must be a dense strided tensor.")
    if tuple(source.shape) != tuple(parameter.shape) or source.numel() != parameter.numel():
        raise PagedOptimizerResumeError(
            f"Paged state {path} shape does not match its current parameter."
        )


def _validate_destination_tensor(
    destination: Any,
    candidate: _PagedCandidate,
    registered: list[Any],
) -> None:
    import torch

    if not torch.is_tensor(destination) or destination is candidate.source:
        raise PagedOptimizerResumeError(
            f"Allocator did not return fresh tensor storage for {candidate.path}."
        )
    if destination.data_ptr() == candidate.source.data_ptr():
        raise PagedOptimizerResumeError(
            f"Allocator reused deserialized storage for {candidate.path}."
        )
    if tuple(destination.shape) != tuple(candidate.source.shape):
        raise PagedOptimizerResumeError(
            f"Rehydrated paged state shape mismatch for {candidate.path}."
        )
    if destination.dtype != candidate.source.dtype:
        raise PagedOptimizerResumeError(
            f"Rehydrated paged state dtype mismatch for {candidate.path}."
        )
    if getattr(destination, "is_paged", False) is not True:
        raise PagedOptimizerResumeError(
            f"Allocator returned non-paged storage for {candidate.path}."
        )
    page_device_id = getattr(destination, "page_deviceid", None)
    if page_device_id != candidate.expected_page_device_id:
        raise PagedOptimizerResumeError(
            f"Rehydrated page device mismatch for {candidate.path}: "
            f"expected {candidate.expected_page_device_id}, got {page_device_id!r}."
        )
    if not any(item is destination for item in registered):
        raise PagedOptimizerResumeError(
            f"Rehydrated buffer for {candidate.path} is absent from the page manager."
        )


def _parameter_cuda_device_id(parameter: Any, label: str) -> int:
    device = getattr(parameter, "device", None)
    if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) is None:
        raise PagedOptimizerResumeError(
            f"Paged state parameter {label} is not on an explicit CUDA device."
        )
    return int(device.index)


def _iter_paged_state(
    optimizer: Any,
    parameter_names: Mapping[int, str] | None,
) -> list[tuple[str, Any]]:
    import torch

    entries: list[tuple[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group.get("params", [])):
            state = optimizer.state.get(parameter, {})
            label = (
                parameter_names.get(id(parameter)) if parameter_names is not None else None
            ) or f"group{group_index}.param{parameter_index}"
            for key, value in state.items():
                if torch.is_tensor(value) and getattr(value, "is_paged", False) is True:
                    entries.append((f"{label}.{key}", value))
    return entries


def _tensor_sha256(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().contiguous().cpu()
    byte_view = cpu.reshape(-1).view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _aggregate_tensor_checksum(entries: list[tuple[str, Any]]) -> str:
    return _aggregate_digest_entries(
        [(path, _tensor_sha256(tensor)) for path, tensor in entries]
    )


def _aggregate_digest_entries(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path, value_digest in sorted(entries):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(value_digest))
    return digest.hexdigest()


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _resolved_checkpoint(checkpoint: str | Path | None) -> str | None:
    return str(Path(checkpoint).expanduser().resolve()) if checkpoint is not None else None


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _none_first(value: int | None) -> tuple[bool, int]:
    return (value is not None, value if value is not None else -1)
