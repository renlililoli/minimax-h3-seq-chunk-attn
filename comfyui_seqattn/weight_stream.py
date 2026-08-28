from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import comfy.memory_management
import comfy.model_management
import comfy.ops
import comfy.pinned_memory
import comfy_aimdo.model_vbar
import torch

RecordCallback = Callable[[dict[str, Any]], None]
StageCompute = Callable[[int], None]


class StageAuxiliary(Protocol):
    def prepare(self, index: int) -> Any: ...

    def wait_ready(self, state: Any) -> None: ...

    def compute_end(self, state: Any) -> None: ...

    def release(self, state: Any) -> None: ...

    def metrics(self, state: Any, event: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _stage_modules(stage) -> Iterable:
    if isinstance(stage, (list, tuple)):
        for module in stage:
            yield from _stage_modules(module)
        return
    yield stage


def _vbar_modules(stage) -> list:
    modules = []
    seen = set()
    for module in _stage_modules(stage):
        for child in module.modules():
            if not hasattr(child, "_v") or id(child) in seen:
                continue
            seen.add(id(child))
            modules.append(child)
    return modules


def _materialize_loaded_weight_pins(modules: list) -> int:
    materialized_bytes = 0
    for module in modules:
        parameters = [
            parameter
            for parameter in (
                getattr(module, "weight", None),
                getattr(module, "bias", None),
            )
            if parameter is not None
        ]
        if not parameters:
            continue
        if (
            comfy.pinned_memory.get_pin(module, subset="weights") is not None
            or comfy.pinned_memory.get_pin(module, subset="weights-loaded")
            is not None
        ):
            continue
        size = comfy.memory_management.vram_aligned_size(parameters)
        comfy.pinned_memory.pin_memory(
            module, subset="weights-loaded", size=size
        )
        pin = comfy.pinned_memory.get_pin(module, subset="weights-loaded")
        if pin is None:
            raise RuntimeError(
                "SeqAttn could not allocate a loaded-weight pin for "
                f"{type(module).__name__}"
            )
        comfy.model_management.cast_to_gathered(
            parameters, pin, non_blocking=False, stream=None
        )
        materialized_bytes += size
    return materialized_bytes


def _registerable_size(modules: list) -> int:
    size = 0
    for module in modules:
        parameters = [
            parameter
            for parameter in (
                getattr(module, "weight", None),
                getattr(module, "bias", None),
            )
            if parameter is not None
        ]
        size += comfy.memory_management.vram_aligned_size(parameters)
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(module, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                size += lowvram_fn.memory_required()
    return size


def _cleanup_modules(modules: list) -> int:
    unpinned_bytes = 0
    for module in modules:
        prefetch = getattr(module, "_prefetch", None)
        if prefetch is None:
            continue
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(module, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                lowvram_fn.clear_prepared()
        if prefetch["signature"] is not None:
            comfy_aimdo.model_vbar.vbar_unpin(module._v)
            unpinned_bytes += int(module._v[2])
        delattr(module, "_prefetch")
    return unpinned_bytes


def _evict_modules(modules: list, requested_bytes: int) -> int:
    vbars = {}
    for module in modules:
        vbar = module._v[0]
        vbars[id(vbar)] = vbar
    freed_bytes = 0
    remaining = requested_bytes
    for vbar in vbars.values():
        if remaining <= 0:
            break
        requested = remaining
        reported = int(vbar.free_memory(requested))
        freed = min(max(reported, 0), requested)
        freed_bytes += freed
        remaining -= freed
    return freed_bytes


@dataclass
class BlockWeightState:
    index: int
    modules: list
    stream: Any
    registerable_bytes: int
    staged_bytes: int
    prepared_at: float
    auxiliary_state: Any = None
    ready: bool = False
    released: bool = False


class BlockWeightStreamer:
    """Bounded ComfyUI VBAR adapter for layer or module-group stages."""

    def __init__(
        self,
        blocks: list,
        device: torch.device,
        *,
        record: RecordCallback | None = None,
        auxiliary: StageAuxiliary | None = None,
    ) -> None:
        self.blocks = blocks
        self.device = torch.device(device)
        self.record = record
        self.auxiliary = auxiliary
        self.started_at = time.perf_counter()
        self.states: dict[int, BlockWeightState] = {}
        self.staged_indices: set[int] = set()
        self.max_staged_blocks = 0
        self.closed = False

    def _vbars(self) -> list:
        vbars = {}
        for state in self.states.values():
            for module in state.modules:
                vbar = module._v[0]
                vbars[id(vbar)] = vbar
        return list(vbars.values())

    def _loaded_bytes(self) -> int:
        return sum(int(vbar.loaded_size()) for vbar in self._vbars())

    def _emit(self, event: str, state: BlockWeightState, **fields) -> None:
        if self.record is None:
            return
        auxiliary_fields = (
            {}
            if self.auxiliary is None
            else self.auxiliary.metrics(state.auxiliary_state, event)
        )
        self.record(
            {
                "event": event,
                "block_index": state.index,
                "seconds": time.perf_counter() - self.started_at,
                "staged_blocks": sorted(self.staged_indices),
                "staged_block_count": len(self.staged_indices),
                "vbar_loaded_mib": self._loaded_bytes() / 2**20,
                **auxiliary_fields,
                **fields,
            }
        )

    def prepare(self, index: int) -> BlockWeightState:
        if self.closed:
            raise RuntimeError("block weight streamer is closed")
        if index in self.states:
            raise RuntimeError(f"block {index} has already been prepared")
        if len(self.staged_indices) >= 2:
            raise RuntimeError(
                "SeqAttn weight pipeline exceeded current-plus-next staging"
            )

        modules = _vbar_modules(self.blocks[index])
        if not modules:
            raise RuntimeError(
                f"stage {index} has no Dynamic VBAR weights; "
                "run ComfyUI with DynamicVRAM enabled"
            )
        collisions = [
            type(module).__name__ for module in modules if hasattr(module, "_prefetch")
        ]
        if collisions:
            raise RuntimeError(
                f"stage {index} contains weights owned by another prefetch: "
                + ", ".join(collisions)
            )

        stream = None
        auxiliary_state = None
        try:
            registerable_bytes = _registerable_size(modules)
            materialized_bytes = _materialize_loaded_weight_pins(modules)
            stream = comfy.ops.cast_modules_with_vbar(
                modules,
                None,
                self.device,
                None,
                True,
            )
            if not comfy.model_management.args.fast_disk:
                comfy.model_management.ensure_pin_registerable(registerable_bytes)
            staged_bytes = sum(
                int(module._v[2])
                for module in modules
                if module._prefetch["signature"] is not None
            )
            if self.auxiliary is not None:
                auxiliary_state = self.auxiliary.prepare(index)
        except Exception:
            if stream is not None:
                stream.synchronize()
            unpinned_bytes = _cleanup_modules(modules)
            _evict_modules(modules, unpinned_bytes)
            if self.auxiliary is not None and auxiliary_state is not None:
                self.auxiliary.release(auxiliary_state)
            raise
        state = BlockWeightState(
            index=index,
            modules=modules,
            stream=stream,
            registerable_bytes=registerable_bytes,
            staged_bytes=staged_bytes,
            prepared_at=time.perf_counter(),
            auxiliary_state=auxiliary_state,
        )
        self.states[index] = state
        self.staged_indices.add(index)
        self.max_staged_blocks = max(self.max_staged_blocks, len(self.staged_indices))
        self._emit(
            "prepare",
            state,
            module_count=len(modules),
            registerable_mib=registerable_bytes / 2**20,
            materialized_mib=materialized_bytes / 2**20,
            staged_mib=staged_bytes / 2**20,
        )
        return state

    def wait_ready(self, state: BlockWeightState) -> None:
        if state.released:
            raise RuntimeError(f"block {state.index} was released before use")
        blocked_started = time.perf_counter()
        if state.stream is not None:
            state.stream.synchronize()
        if self.auxiliary is not None:
            self.auxiliary.wait_ready(state.auxiliary_state)
        blocked_seconds = time.perf_counter() - blocked_started
        state.ready = True
        self._emit(
            "ready",
            state,
            wait_seconds=time.perf_counter() - state.prepared_at,
            blocked_seconds=blocked_seconds,
        )

    def compute_start(self, state: BlockWeightState) -> None:
        if not state.ready:
            raise RuntimeError(f"block {state.index} compute started before ready")
        self._emit("compute_start", state)

    def compute_end(self, state: BlockWeightState) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        if self.auxiliary is not None:
            self.auxiliary.compute_end(state.auxiliary_state)
        self._emit("compute_end", state)

    def release(self, state: BlockWeightState) -> None:
        if state.released:
            return
        unpinned_bytes = _cleanup_modules(state.modules)
        freed_bytes = _evict_modules(state.modules, unpinned_bytes)
        if self.auxiliary is not None:
            self.auxiliary.release(state.auxiliary_state)
        state.released = True
        self.staged_indices.discard(state.index)
        self._emit(
            "release",
            state,
            unpinned_mib=unpinned_bytes / 2**20,
            evicted_mib=freed_bytes / 2**20,
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        for state in self.states.values():
            if state.released:
                continue
            if state.stream is not None:
                state.stream.synchronize()
            self.release(state)
        if self.auxiliary is not None:
            self.auxiliary.close()
        self.closed = True


def run_weight_stages(
    stages: list,
    device: torch.device,
    compute: StageCompute,
    *,
    record: RecordCallback | None = None,
    auxiliary: StageAuxiliary | None = None,
) -> int:
    if not stages:
        if auxiliary is not None:
            auxiliary.close()
        return 0
    streamer = BlockWeightStreamer(
        stages,
        device,
        record=record,
        auxiliary=auxiliary,
    )
    try:
        current = streamer.prepare(0)
        for index in range(len(stages)):
            streamer.wait_ready(current)
            next_state = (
                streamer.prepare(index + 1) if index + 1 < len(stages) else None
            )
            streamer.compute_start(current)
            compute(index)
            streamer.compute_end(current)
            streamer.release(current)
            current = next_state
        return streamer.max_staged_blocks
    finally:
        streamer.close()


__all__ = ["BlockWeightState", "BlockWeightStreamer", "run_weight_stages"]
