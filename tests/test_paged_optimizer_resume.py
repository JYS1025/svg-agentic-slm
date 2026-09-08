from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from svg_agentic_slm.train.paged_optimizer_resume import (
    PagedOptimizerResumeError,
    PagedOptimizerResumeMixin,
    rehydrate_paged_optimizer_state,
)
from svg_agentic_slm.train.sft_trainer import SFTConfig


class _FakeParameter:
    def __init__(self, shape: tuple[int, ...], device_index: int) -> None:
        self.shape = shape
        self.device = SimpleNamespace(type="cuda", index=device_index)

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


class _FakePagedOptimizer:
    is_paged = True

    def __init__(
        self,
        parameter: _FakeParameter | list[_FakeParameter],
        state: dict[str, Any] | None = None,
    ) -> None:
        parameters = parameter if isinstance(parameter, list) else [parameter]
        self.param_groups = [{"params": parameters, "lr": 2e-4}]
        if state is not None and len(parameters) != 1:
            raise ValueError("Direct fake state requires one parameter.")
        self.state = {parameters[0]: state} if state is not None else {}
        self.page_mng = SimpleNamespace(paged_tensors=[])

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Model generic Optimizer mapping, including loss of tensor attributes."""

        saved_group = state_dict["param_groups"][0]
        mapped_optimizer_state: dict[_FakeParameter, dict[str, Any]] = {}
        for saved_parameter_id, parameter in zip(
            saved_group["params"], self.param_groups[0]["params"], strict=True
        ):
            serialized_state = state_dict["state"][saved_parameter_id]
            saved_state = {
                key: value
                for key, value in serialized_state.items()
                if key != "__bnb_optimizer_quant_state__"
            }
            saved_state.update(serialized_state.get("__bnb_optimizer_quant_state__", {}))
            mapped_state: dict[str, Any] = {}
            for key, value in saved_state.items():
                if torch.is_tensor(value):
                    mapped = torch.empty_like(value).copy_(value)
                    for attribute in ("is_paged", "page_deviceid"):
                        if hasattr(mapped, attribute):
                            delattr(mapped, attribute)
                    mapped_state[key] = mapped
                else:
                    mapped_state[key] = value
            mapped_optimizer_state[parameter] = mapped_state
        self.state = mapped_optimizer_state
        self.param_groups[0].update(
            {key: value for key, value in saved_group.items() if key != "params"}
        )


_FakePagedOptimizer.__module__ = "bitsandbytes.optim.fake"


class _FakeOptimizerWrapper:
    def __init__(self, optimizer: _FakePagedOptimizer) -> None:
        self.optimizer = optimizer

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)


def _serialized_paged_tensor(values: list[int], *, page_deviceid: int) -> torch.Tensor:
    source = torch.tensor(values, dtype=torch.uint8)
    source.is_paged = True
    source.page_deviceid = page_deviceid
    payload = io.BytesIO()
    torch.save(source, payload)
    payload.seek(0)
    loaded = torch.load(payload, map_location="cpu", weights_only=True)
    assert loaded.is_paged is True
    assert loaded.page_deviceid == page_deviceid
    assert loaded.data_ptr() != source.data_ptr()
    return loaded


def _allocator(call_log: list[str]):
    def allocate(
        optimizer: _FakePagedOptimizer,
        parameter: _FakeParameter,
        source: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        call_log.append(key)
        destination = torch.empty_like(source)
        destination.is_paged = True
        destination.page_deviceid = parameter.device.index
        optimizer.page_mng.paged_tensors.append(destination)
        return destination

    return allocate


def _optimizer_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return _optimizer_state_dicts([state])


def _optimizer_state_dicts(states: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_ids = list(range(len(states)))
    return {
        "state": dict(zip(parameter_ids, states, strict=True)),
        "param_groups": [{"params": parameter_ids, "lr": 2e-4}],
    }


def test_serialized_paged_state_is_rehydrated_without_changing_numeric_state() -> None:
    parameter = _FakeParameter((4,), device_index=2)
    state1 = _serialized_paged_tensor([1, 2, 3, 4], page_deviceid=7)
    state2 = _serialized_paged_tensor([9, 8, 7, 6], page_deviceid=7)
    qmap = torch.tensor([0.25, 0.5])
    absmax = torch.tensor([3.0])
    state = {"step": 41, "state1": state1, "state2": state2, "qmap1": qmap, "absmax1": absmax}
    optimizer = _FakePagedOptimizer(parameter, state)
    calls: list[str] = []

    audit = rehydrate_paged_optimizer_state(
        optimizer,
        checkpoint="checkpoint-41",
        parameter_names={id(parameter): "adapter.weight"},
        allocator=_allocator(calls),
    ).to_manifest()

    assert calls == ["state1", "state2"]
    assert state["state1"] is not state1
    assert state["state2"] is not state2
    assert torch.equal(state["state1"], state1)
    assert torch.equal(state["state2"], state2)
    assert state["state1"].page_deviceid == 2
    assert state["state2"].page_deviceid == 2
    assert state["step"] == 41
    assert state["qmap1"] is qmap
    assert state["absmax1"] is absmax
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert audit["status"] == "rehydrated"
    assert audit["detected_paged_tensors"] == 2
    assert audit["rehydrated_paged_tensors"] == 2
    assert audit["verified_paged_tensors"] == 2
    assert audit["rehydrated_bytes"] == 8
    assert audit["nonpaged_state_entries_preserved"] == 3
    assert audit["state_key_counts"] == {"state1": 1, "state2": 1}
    assert audit["source_page_device_ids"] == [7]
    assert audit["target_page_device_ids"] == [2]
    assert audit["source_state_sha256"] == audit["rehydrated_state_sha256"]
    assert len(audit["source_state_sha256"]) == 64
    assert audit["numeric_values_preserved"] is True
    assert audit["checkpoint_mutated"] is False

    first_pointers = (state["state1"].data_ptr(), state["state2"].data_ptr())
    repeated = rehydrate_paged_optimizer_state(
        optimizer,
        checkpoint="checkpoint-41",
        parameter_names={id(parameter): "adapter.weight"},
        allocator=_allocator(calls),
    ).to_manifest()
    assert calls == ["state1", "state2"]
    assert repeated["status"] == "already_rehydrated"
    assert (state["state1"].data_ptr(), state["state2"].data_ptr()) == first_pointers


def test_rehydration_is_transactional_and_fails_closed() -> None:
    parameter = _FakeParameter((4,), device_index=0)
    state1 = _serialized_paged_tensor([1, 2, 3, 4], page_deviceid=0)
    state2 = _serialized_paged_tensor([5, 6, 7, 8], page_deviceid=0)
    state = {"step": 5, "state1": state1, "state2": state2}
    optimizer = _FakePagedOptimizer(parameter, state)

    def bad_allocator(
        optimizer: _FakePagedOptimizer,
        parameter: _FakeParameter,
        source: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        shape = source.shape if key == "state1" else (source.numel() + 1,)
        destination = torch.empty(shape, dtype=source.dtype)
        destination.is_paged = True
        destination.page_deviceid = parameter.device.index
        optimizer.page_mng.paged_tensors.append(destination)
        return destination

    with pytest.raises(PagedOptimizerResumeError, match="shape mismatch"):
        rehydrate_paged_optimizer_state(
            optimizer,
            checkpoint="checkpoint-5",
            allocator=bad_allocator,
        )

    assert state["state1"] is state1
    assert state["state2"] is state2
    assert optimizer.page_mng.paged_tensors == []


class _BaseTrainer:
    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        self.delegated_checkpoint = checkpoint
        if checkpoint is not None:
            state_dict = getattr(self, "serialized_optimizer_state", None)
            if state_dict is not None:
                self.optimizer.load_state_dict(state_dict)


class _ResumeSubject(PagedOptimizerResumeMixin, _BaseTrainer):
    pass


def test_trainer_mixin_runs_only_for_enabled_paged_resume() -> None:
    parameter = _FakeParameter((2,), device_index=1)
    serialized_state1 = _serialized_paged_tensor([4, 2], page_deviceid=0)
    serialized = _optimizer_state_dict({"step": 2, "state1": serialized_state1})
    subject = _ResumeSubject()
    core = _FakePagedOptimizer(parameter)
    subject.optimizer = _FakeOptimizerWrapper(core)
    subject.serialized_optimizer_state = serialized
    subject.model = None
    subject.args = SimpleNamespace(process_index=1, world_size=3)
    calls: list[str] = []
    allocate = _allocator(calls)

    def assert_metadata_was_lost_before_allocation(
        optimizer: _FakePagedOptimizer,
        current_parameter: _FakeParameter,
        source: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        assert getattr(source, "is_paged", False) is False
        assert not hasattr(source, "page_deviceid")
        return allocate(optimizer, current_parameter, source, key)

    subject._svg_paged_optimizer_state_allocator = assert_metadata_was_lost_before_allocation

    subject._load_optimizer_and_scheduler("checkpoint-2")

    assert subject.delegated_checkpoint == "checkpoint-2"
    assert calls == ["state1"]
    assert serialized_state1.is_paged is True
    assert serialized_state1.page_deviceid == 0
    assert core.state[parameter]["state1"].is_paged is True
    assert core.state[parameter]["state1"].page_deviceid == 1
    assert subject._svg_paged_optimizer_resume_audit["status"] == "rehydrated"
    assert subject._svg_paged_optimizer_resume_audit["schema_version"] == 2
    assert subject._svg_paged_optimizer_resume_audit["storage_strategy"].startswith(
        "pre_load_metadata_capture"
    )
    assert subject._svg_paged_optimizer_resume_audit["process_index"] == 1
    assert subject._svg_paged_optimizer_resume_audit["world_size"] == 3

    disabled_serialized_state = {
        "step": 2,
        "state1": _serialized_paged_tensor([4, 2], page_deviceid=0),
    }
    disabled = _ResumeSubject()
    disabled_core = _FakePagedOptimizer(parameter)
    disabled.optimizer = _FakeOptimizerWrapper(disabled_core)
    disabled.serialized_optimizer_state = _optimizer_state_dict(disabled_serialized_state)
    disabled.model = None
    disabled.args = SimpleNamespace(process_index=0, world_size=1)
    disabled._svg_enable_paged_optimizer_resume_rehydration = False
    disabled._svg_paged_optimizer_state_allocator = _allocator(calls)
    disabled._load_optimizer_and_scheduler("checkpoint-2")
    assert disabled._svg_paged_optimizer_resume_audit["status"] == "disabled"
    assert getattr(disabled_core.state[parameter]["state1"], "is_paged", False) is False
    assert disabled_serialized_state["state1"].page_deviceid == 0
    assert calls == ["state1"]


def test_world_one_checkpoint_rehydrates_independently_on_two_ranks() -> None:
    serialized_state1 = _serialized_paged_tensor([1, 3], page_deviceid=0)
    serialized_state2 = _serialized_paged_tensor([5, 7], page_deviceid=0)
    serialized = _optimizer_state_dict(
        {"step": 375, "state1": serialized_state1, "state2": serialized_state2}
    )
    audits: list[dict[str, Any]] = []
    destinations: list[tuple[torch.Tensor, torch.Tensor]] = []

    for rank in range(2):
        parameter = _FakeParameter((2,), device_index=rank)
        core = _FakePagedOptimizer(parameter)
        wrapper = _FakeOptimizerWrapper(core)
        subject = _ResumeSubject()
        subject.optimizer = wrapper
        subject.serialized_optimizer_state = serialized
        subject.model = None
        subject.args = SimpleNamespace(process_index=rank, world_size=2)
        calls: list[str] = []
        subject._svg_paged_optimizer_state_allocator = _allocator(calls)

        subject._load_optimizer_and_scheduler("checkpoint-375")

        state = core.state[parameter]
        assert calls == ["state1", "state2"]
        assert state["state1"].page_deviceid == rank
        assert state["state2"].page_deviceid == rank
        assert torch.equal(state["state1"], serialized_state1)
        assert torch.equal(state["state2"], serialized_state2)
        assert "load_state_dict" not in wrapper.__dict__
        audits.append(subject._svg_paged_optimizer_resume_audit)
        destinations.append((state["state1"], state["state2"]))

    assert serialized_state1.is_paged is True
    assert serialized_state2.is_paged is True
    assert serialized_state1.page_deviceid == 0
    assert serialized_state2.page_deviceid == 0
    assert destinations[0][0] is not destinations[1][0]
    assert [audit["process_index"] for audit in audits] == [0, 1]
    assert [audit["world_size"] for audit in audits] == [2, 2]
    assert [audit["source_page_device_ids"] for audit in audits] == [[0], [0]]
    assert [audit["target_page_device_ids"] for audit in audits] == [[0], [1]]
    assert audits[0]["source_state_sha256"] == audits[1]["source_state_sha256"]
    assert all(
        audit["source_state_sha256"] == audit["rehydrated_state_sha256"]
        for audit in audits
    )
    assert all(audit["checkpoint_mutated"] is False for audit in audits)


def test_actual_bnb_nested_wrapper_cardinality_rehydrates_320_paged_tensors() -> None:
    wrapper_key = "__bnb_optimizer_quant_state__"
    parameters = [_FakeParameter((2,), device_index=0) for _ in range(656)]
    serialized_states: list[dict[str, Any]] = []
    for parameter_index in range(656):
        state1 = torch.tensor([parameter_index % 251, 1], dtype=torch.uint8)
        state2 = torch.tensor([2, parameter_index % 251], dtype=torch.uint8)
        if parameter_index < 160:
            state1.is_paged = True
            state1.page_deviceid = 0
            state2.is_paged = True
            state2.page_deviceid = 0
        serialized_states.append(
            {
                "step": 375,
                wrapper_key: {
                    "state1": state1,
                    "state2": state2,
                    "qmap1": torch.tensor([0.25]),
                    "qmap2": torch.tensor([0.5]),
                    "absmax1": torch.tensor([1.0]),
                    "absmax2": torch.tensor([2.0]),
                },
            }
        )
    serialized = _optimizer_state_dicts(serialized_states)
    core = _FakePagedOptimizer(parameters)
    subject = _ResumeSubject()
    subject.optimizer = _FakeOptimizerWrapper(core)
    subject.serialized_optimizer_state = serialized
    subject.model = None
    subject.args = SimpleNamespace(process_index=0, world_size=2)
    calls: list[str] = []
    subject._svg_paged_optimizer_state_allocator = _allocator(calls)

    subject._load_optimizer_and_scheduler("checkpoint-375")

    audit = subject._svg_paged_optimizer_resume_audit
    assert len(serialized_states) == 656
    assert sum(len(state[wrapper_key]) for state in serialized_states) == 3936
    assert audit["detected_paged_tensors"] == 320
    assert audit["rehydrated_paged_tensors"] == 320
    assert audit["verified_paged_tensors"] == 320
    assert audit["state_key_counts"] == {"state1": 160, "state2": 160}
    assert audit["nonpaged_state_entries_preserved"] == 4272
    assert audit["source_page_device_ids"] == [0]
    assert audit["target_page_device_ids"] == [0]
    assert audit["source_state_sha256"] == audit["rehydrated_state_sha256"]
    assert calls.count("state1") == 160
    assert calls.count("state2") == 160
    assert wrapper_key in serialized_states[0]
    assert "state1" not in serialized_states[0]
    assert serialized_states[0][wrapper_key]["state1"].is_paged is True
    assert core.state[parameters[0]]["state1"].is_paged is True


@pytest.mark.parametrize(
    ("malformation", "message"),
    [("unknown", "unknown key"), ("cycle", "cycle"), ("depth", "wrapper depth")],
)
def test_bnb_quant_state_wrapper_fails_closed(
    malformation: str,
    message: str,
) -> None:
    wrapper_key = "__bnb_optimizer_quant_state__"
    wrapper: dict[str, Any] = {}
    if malformation == "unknown":
        wrapper["future_state"] = torch.tensor([1], dtype=torch.uint8)
    elif malformation == "cycle":
        wrapper["state1"] = wrapper
    else:
        wrapper[wrapper_key] = {"state1": torch.tensor([1], dtype=torch.uint8)}
    parameter = _FakeParameter((1,), device_index=0)
    core = _FakePagedOptimizer(parameter)
    subject = _ResumeSubject()
    subject.optimizer = _FakeOptimizerWrapper(core)
    subject.serialized_optimizer_state = _optimizer_state_dict(
        {"step": 1, wrapper_key: wrapper}
    )
    subject.model = None
    subject.args = SimpleNamespace(process_index=0, world_size=1)

    with pytest.raises(PagedOptimizerResumeError, match=message):
        subject._load_optimizer_and_scheduler("checkpoint-1")

    assert core.state == {}


def test_fresh_run_does_not_inspect_or_allocate_optimizer_state() -> None:
    subject = _ResumeSubject()
    subject.optimizer = object()
    subject.args = SimpleNamespace(process_index=0, world_size=1)
    subject._load_optimizer_and_scheduler(None)
    assert subject.delegated_checkpoint is None
    assert subject._svg_paged_optimizer_resume_audit["status"] == "fresh_run"


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_rehydration_config_toggle_requires_boolean(invalid: Any) -> None:
    with pytest.raises(ValueError, match="rehydrate_paged_optimizer_state_on_resume"):
        SFTConfig(rehydrate_paged_optimizer_state_on_resume=invalid)  # type: ignore[arg-type]


def test_rehydration_config_defaults_enabled() -> None:
    assert SFTConfig().rehydrate_paged_optimizer_state_on_resume is True
