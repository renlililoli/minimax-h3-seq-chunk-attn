from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import comfy.lora
import comfy.lora_convert
import comfy.ops
import torch
import torch.nn.functional as F
from comfy.weight_adapter.lora import LoRAAdapter

StageId = str | tuple[str, int]
EMBEDDING_STAGE = "embedding"
FINAL_STAGE = "final"


@dataclass(frozen=True)
class AdapterIdentity:
    name: str
    path: str
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, name: str, path: str) -> "AdapterIdentity":
        stat = os.stat(path)
        return cls(
            name=str(name),
            path=os.path.realpath(path),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )

    @property
    def signature(self) -> tuple:
        return (self.path, self.size, self.mtime_ns)


@dataclass(frozen=True)
class LinearLoRA:
    """Model-independent, CPU-resident Linear LoRA representation."""

    adapter: AdapterIdentity
    target: str
    down: torch.Tensor
    up: torch.Tensor
    alpha: float | None
    rank: int
    in_features: int
    out_features: int
    dtype: torch.dtype
    strength: float

    @property
    def scale(self) -> float:
        alpha_scale = 1.0 if self.alpha is None else self.alpha / self.rank
        return float(self.strength) * float(alpha_scale)

    @property
    def signature(self) -> tuple:
        return (
            self.target,
            self.adapter.signature,
            float(self.strength),
            self.alpha,
            self.rank,
            self.in_features,
            self.out_features,
            self.dtype,
        )


@dataclass(frozen=True)
class LinearLoRABundle:
    adapter: AdapterIdentity
    strength: float
    layers: tuple[LinearLoRA, ...]

    @property
    def signature(self) -> tuple:
        return (
            self.adapter.signature,
            float(self.strength),
            tuple(layer.signature for layer in self.layers),
        )


@dataclass(frozen=True)
class H3TargetSpec:
    path: str
    stage: StageId
    in_features: int
    out_features: int
    compute_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class H3LoRAState:
    bundles: tuple[LinearLoRABundle, ...] = ()
    target_specs: tuple[H3TargetSpec, ...] = ()

    def append(
        self,
        bundle: LinearLoRABundle,
        target_specs: tuple[H3TargetSpec, ...],
    ) -> "H3LoRAState":
        if self.target_specs and self.target_specs != target_specs:
            raise ValueError("MiniMax H3 LoRA target layout changed between chained nodes")
        return H3LoRAState(
            bundles=(*self.bundles, bundle),
            target_specs=target_specs,
        )

    @property
    def signature(self) -> tuple:
        return tuple(bundle.signature for bundle in self.bundles)

    @property
    def adapter_count(self) -> int:
        return len(self.bundles)

    @property
    def target_count(self) -> int:
        return len({layer.target for bundle in self.bundles for layer in bundle.layers})

    def plan_for(
        self,
        stage: StageId,
        *,
        activation_dtype: torch.dtype,
    ) -> "LoRAStagePlan":
        specs = {spec.path: spec for spec in self.target_specs if spec.stage == stage}
        by_target: dict[str, list[StagedAdapterSource]] = defaultdict(list)
        for bundle in self.bundles:
            for layer in bundle.layers:
                spec = specs.get(layer.target)
                if spec is None or layer.scale == 0.0:
                    continue
                by_target[layer.target].append(
                    StagedAdapterSource(
                        adapter_name=bundle.adapter.name,
                        target=layer.target,
                        down=layer.down,
                        up=layer.up,
                        scale=layer.scale,
                        compute_dtype=spec.compute_dtype or activation_dtype,
                        signature=layer.signature,
                    )
                )
        return LoRAStagePlan(
            stage=stage,
            adapters_by_target=tuple(
                (target, tuple(adapters)) for target, adapters in sorted(by_target.items())
            ),
        )


def _linear_spec(named_modules: dict[str, Any], path: str, stage: StageId, dtype=None):
    module = named_modules.get(path)
    if module is None:
        return None
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        raise TypeError(f"supported MiniMax H3 LoRA target {path} is not Linear-like")
    return H3TargetSpec(path, stage, in_features, out_features, dtype)


def build_h3_target_specs(model) -> tuple[H3TargetSpec, ...]:
    """Build supported call sites without materializing a Dynamic VBAR state_dict."""

    named_modules = dict(model.named_modules())
    requested: list[tuple[str, StageId, torch.dtype | None]] = [
        ("video_patch_proj", EMBEDDING_STAGE, torch.float32),
        ("audio_patch_proj", EMBEDDING_STAGE, torch.float32),
        ("condition_proj", EMBEDDING_STAGE, None),
    ]
    if hasattr(model, "time_embedder"):
        requested.extend(
            [
                ("time_embedder.proj_in", EMBEDDING_STAGE, torch.float32),
                ("time_embedder.proj_out", EMBEDDING_STAGE, torch.float32),
            ]
        )
    for index, _block in enumerate(model.token_refiner.blocks):
        prefix = f"token_refiner.blocks.{index}"
        requested.extend(
            (f"{prefix}.{suffix}", EMBEDDING_STAGE, None)
            for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
        )
    adaln_dtype = torch.float32 if model.use_adaln_curves else None
    for index, _block in enumerate(model.blocks):
        prefix = f"blocks.{index}"
        stage = ("block", index)
        requested.extend(
            [
                (f"{prefix}.adaln_proj.linear", stage, adaln_dtype),
                (f"{prefix}.attn.qkv_proj", stage, None),
                (f"{prefix}.attn.out_proj", stage, None),
                (f"{prefix}.mlp.fc1", stage, None),
                (f"{prefix}.mlp.fc2", stage, None),
            ]
        )
    requested.extend(
        [
            ("final_layer.adaln_proj.linear", FINAL_STAGE, adaln_dtype),
            ("final_layer.video_out", FINAL_STAGE, torch.float32),
            ("final_layer.audio_out", FINAL_STAGE, torch.float32),
        ]
    )
    specs = [
        spec
        for path, stage, dtype in requested
        if (spec := _linear_spec(named_modules, path, stage, dtype)) is not None
    ]
    return tuple(specs)


def validate_h3_int8_convrot_base(model, specs: tuple[H3TargetSpec, ...]) -> None:
    if len(model.blocks) != 50:
        raise ValueError(
            "MiniMaxH3SeqAttnLoRA requires the supported 50-block MiniMax-H3 model"
        )
    named_modules = dict(model.named_modules())
    invalid = []
    for index in range(len(model.blocks)):
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            path = f"blocks.{index}.{suffix}"
            module = named_modules[path]
            weight = getattr(module, "weight", None)
            params = getattr(weight, "_params", None)
            if (
                getattr(module, "quant_format", None) != "int8_tensorwise"
                or not bool(getattr(params, "convrot", False))
            ):
                invalid.append(path)
    patched = []
    for spec in specs:
        module = named_modules[spec.path]
        if getattr(module, "weight_function", ()) or getattr(module, "bias_function", ()):
            patched.append(spec.path)
    if invalid:
        preview = ", ".join(invalid[:4])
        raise ValueError(
            "MiniMaxH3SeqAttnLoRA v1 requires an INT8 tensorwise ConvRot DiT base; "
            f"unsupported modules include {preview}"
        )
    if patched:
        preview = ", ".join(patched[:4])
        raise ValueError(
            "MiniMaxH3SeqAttnLoRA requires unpatched base Linear modules; "
            f"ordinary ComfyUI patches are present on {preview}"
        )


def _lora_key_map(specs: tuple[H3TargetSpec, ...]) -> dict[str, str]:
    key_map = {}
    for spec in specs:
        for prefix in ("", "diffusion_model.", "transformer.", "base_model.model."):
            key_map[f"{prefix}{spec.path}"] = spec.path
    return key_map


def parse_h3_linear_lora(
    state_dict: dict[str, torch.Tensor],
    *,
    identity: AdapterIdentity,
    strength: float,
    specs: tuple[H3TargetSpec, ...],
) -> LinearLoRABundle:
    converted = comfy.lora_convert.convert_lora(state_dict)
    try:
        loaded = comfy.lora.load_lora(converted, _lora_key_map(specs), log_missing=False)
    except (KeyError, RuntimeError, ValueError) as error:
        raise ValueError(f"invalid Linear LoRA tensor set in {identity.name}: {error}") from error

    consumed: set[str] = set()
    layers = []
    spec_by_path = {spec.path: spec for spec in specs}
    for target, adapter in loaded.items():
        if not isinstance(adapter, LoRAAdapter):
            raise ValueError(
                f"{identity.name} target {target} uses unsupported adapter type "
                f"{type(adapter).__name__}; only ordinary Linear LoRA is supported"
            )
        consumed.update(adapter.loaded_keys)
        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        if dora_scale is not None:
            raise ValueError(f"{identity.name} target {target} uses unsupported DoRA")
        if mid is not None:
            raise ValueError(f"{identity.name} target {target} uses unsupported convolutional mid")
        if reshape is not None:
            raise ValueError(f"{identity.name} target {target} uses unsupported reshape")
        if up.ndim != 2 or down.ndim != 2:
            raise ValueError(f"{identity.name} target {target} is not a Linear A/B adapter")
        spec = spec_by_path.get(target)
        if spec is None:
            raise ValueError(f"{identity.name} target {target} is not a supported H3 call site")
        rank = int(down.shape[0])
        expected_down = (rank, spec.in_features)
        expected_up = (spec.out_features, rank)
        if tuple(down.shape) != expected_down or tuple(up.shape) != expected_up:
            raise ValueError(
                f"{identity.name} target {target} shape mismatch: down {tuple(down.shape)}, "
                f"up {tuple(up.shape)}, expected {expected_down} and {expected_up}"
            )
        if not (down.dtype.is_floating_point and up.dtype.is_floating_point):
            raise ValueError(f"{identity.name} target {target} must use floating tensors")
        down_cpu = down.detach().to(device="cpu").contiguous()
        up_cpu = up.detach().to(device="cpu").contiguous()
        layers.append(
            LinearLoRA(
                adapter=identity,
                target=target,
                down=down_cpu,
                up=up_cpu,
                alpha=None if alpha is None else float(alpha),
                rank=rank,
                in_features=spec.in_features,
                out_features=spec.out_features,
                dtype=down_cpu.dtype,
                strength=float(strength),
            )
        )
    unconsumed = sorted(set(converted) - consumed)
    if unconsumed:
        preview = ", ".join(unconsumed[:6])
        suffix = "" if len(unconsumed) <= 6 else f" (+{len(unconsumed) - 6} more)"
        raise ValueError(
            f"{identity.name} has {len(unconsumed)} unmapped or unsupported tensors: "
            f"{preview}{suffix}"
        )
    if not layers:
        raise ValueError(f"{identity.name} contains no supported MiniMax-H3 Linear LoRA targets")
    return LinearLoRABundle(identity, float(strength), tuple(sorted(layers, key=lambda x: x.target)))


@dataclass(frozen=True)
class StagedAdapterSource:
    adapter_name: str
    target: str
    down: torch.Tensor
    up: torch.Tensor
    scale: float
    compute_dtype: torch.dtype
    signature: tuple


@dataclass(frozen=True)
class LoRAStagePlan:
    stage: StageId
    adapters_by_target: tuple[tuple[str, tuple[StagedAdapterSource, ...]], ...]

    @property
    def target_count(self) -> int:
        return len(self.adapters_by_target)

    @property
    def adapter_count(self) -> int:
        return len(
            {
                adapter.adapter_name
                for _target, adapters in self.adapters_by_target
                for adapter in adapters
            }
        )

    @property
    def packed_bytes(self) -> int:
        offset = 0
        for _target, adapters in self.adapters_by_target:
            for adapter in adapters:
                for tensor in (adapter.down, adapter.up):
                    offset = _align(offset, 64)
                    offset += tensor.numel() * _element_size(adapter.compute_dtype)
        return offset


@dataclass(frozen=True)
class DeviceLinearLoRA:
    adapter_name: str
    target: str
    down: torch.Tensor
    up: torch.Tensor
    scale: float
    signature: tuple


@dataclass
class _StageSlot:
    index: int
    host: torch.Tensor
    device: torch.Tensor
    stream: Any
    active: bool = False


@dataclass
class LoRAStageState:
    index: int
    slot: _StageSlot
    adapters_by_target: dict[str, tuple[DeviceLinearLoRA, ...]]
    staged_bytes: int
    target_count: int
    adapter_count: int
    prepared_at: float
    ready: bool = False
    released: bool = False
    wait_seconds: float = 0.0


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class LoRAStageStreamer:
    """Two-slot pinned-host/GPU streamer for current-plus-next LoRA stages."""

    def __init__(self, plans: list[LoRAStagePlan], device: torch.device) -> None:
        self.plans = plans
        self.device = torch.device(device)
        self.capacity = max((plan.packed_bytes for plan in plans), default=0)
        self.slot_count = min(2, max(1, len(plans))) if self.capacity else 0
        self.slots: list[_StageSlot] = []
        self.states: dict[int, LoRAStageState] = {}
        self.closed = False
        if not self.capacity:
            return
        required = self.capacity * self.slot_count
        try:
            for index in range(self.slot_count):
                host = torch.empty(
                    self.capacity,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self.device.type == "cuda",
                )
                target = torch.empty(self.capacity, dtype=torch.uint8, device=self.device)
                stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
                self.slots.append(_StageSlot(index, host, target, stream))
        except RuntimeError as error:
            self.slots.clear()
            raise RuntimeError(
                "MiniMax H3 LoRA staging allocation failed; "
                f"requires {required} bytes for {self.slot_count} slots"
            ) from error

    def _free_slot(self) -> _StageSlot:
        for slot in self.slots:
            if not slot.active:
                return slot
        raise RuntimeError("MiniMax H3 LoRA exceeded the two staged-slot limit")

    @staticmethod
    def _typed_view(buffer: torch.Tensor, offset: int, tensor: torch.Tensor, dtype):
        size = tensor.numel() * _element_size(dtype)
        return buffer[offset : offset + size].view(dtype).view(tensor.shape), size

    def prepare(self, index: int) -> LoRAStageState | None:
        if self.closed:
            raise RuntimeError("LoRA stage streamer is closed")
        plan = self.plans[index]
        if not plan.adapters_by_target:
            return None
        if index in self.states:
            raise RuntimeError(f"LoRA stage {index} was already prepared")
        slot = self._free_slot()
        slot.active = True
        offset = 0
        device_adapters: dict[str, list[DeviceLinearLoRA]] = defaultdict(list)
        for target, adapters in plan.adapters_by_target:
            for adapter in adapters:
                views = []
                for tensor in (adapter.down, adapter.up):
                    offset = _align(offset, 64)
                    host_view, size = self._typed_view(
                        slot.host, offset, tensor, adapter.compute_dtype
                    )
                    host_view.copy_(tensor, non_blocking=False)
                    device_view, _ = self._typed_view(
                        slot.device, offset, tensor, adapter.compute_dtype
                    )
                    views.append(device_view)
                    offset += size
                device_adapters[target].append(
                    DeviceLinearLoRA(
                        adapter.adapter_name,
                        target,
                        views[0],
                        views[1],
                        adapter.scale,
                        adapter.signature,
                    )
                )
        if self.device.type == "cuda":
            with torch.cuda.stream(slot.stream):
                slot.device[:offset].copy_(slot.host[:offset], non_blocking=True)
        else:
            slot.device[:offset].copy_(slot.host[:offset])
        state = LoRAStageState(
            index=index,
            slot=slot,
            adapters_by_target={key: tuple(value) for key, value in device_adapters.items()},
            staged_bytes=offset,
            target_count=plan.target_count,
            adapter_count=plan.adapter_count,
            prepared_at=time.perf_counter(),
        )
        self.states[index] = state
        return state

    def wait_ready(self, state: LoRAStageState | None) -> None:
        if state is None:
            return
        started = time.perf_counter()
        if state.slot.stream is not None:
            state.slot.stream.synchronize()
        state.wait_seconds = time.perf_counter() - started
        state.ready = True

    def adapters_for(self, index: int) -> dict[str, tuple[DeviceLinearLoRA, ...]]:
        state = self.states.get(index)
        if state is None:
            return {}
        if not state.ready or state.released:
            raise RuntimeError(f"LoRA stage {index} is not ready for compute")
        return state.adapters_by_target

    def compute_end(self, state: LoRAStageState | None) -> None:
        if state is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def release(self, state: LoRAStageState | None) -> None:
        if state is None or state.released:
            return
        state.adapters_by_target.clear()
        state.released = True
        state.slot.active = False

    def metrics(self, state: LoRAStageState | None, event: str) -> dict[str, Any]:
        if state is None:
            return {
                "lora_staged_mib": 0.0,
                "lora_target_count": 0,
                "lora_adapter_count": 0,
                "lora_wait_seconds": 0.0,
            }
        return {
            "lora_staged_mib": state.staged_bytes / 2**20,
            "lora_target_count": state.target_count,
            "lora_adapter_count": state.adapter_count,
            "lora_wait_seconds": state.wait_seconds if event in {"ready", "release"} else 0.0,
            "lora_slot": state.slot.index,
        }

    def close(self) -> None:
        if self.closed:
            return
        for state in self.states.values():
            if state.slot.stream is not None:
                state.slot.stream.synchronize()
            self.release(state)
        self.slots.clear()
        self.closed = True


@contextmanager
def prepared_lora_stage(plan: LoRAStagePlan, device: torch.device):
    if not plan.adapters_by_target:
        yield {}
        return
    streamer = LoRAStageStreamer([plan], device)
    state = None
    try:
        state = streamer.prepare(0)
        streamer.wait_ready(state)
        yield streamer.adapters_for(0)
        streamer.compute_end(state)
    finally:
        streamer.release(state)
        streamer.close()


def linear_with_lora(base_linear, x: torch.Tensor, adapters=()) -> torch.Tensor:
    output = base_linear(x)
    for adapter in adapters:
        if adapter.scale == 0.0:
            continue
        hidden = F.linear(x, adapter.down)
        output.add_(F.linear(hidden, adapter.up), alpha=adapter.scale)
    return output


def _input_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "swiglu":
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate).mul_(value)
    if activation == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"unsupported LoRA input activation {activation}")


def linear_input_act_with_lora(
    base_linear,
    x: torch.Tensor,
    activation: str,
    adapters=(),
) -> torch.Tensor:
    output = comfy.ops.linear_input_act(base_linear, x, activation)
    if not adapters:
        return output
    activated = _input_activation(x, activation)
    for adapter in adapters:
        if adapter.scale == 0.0:
            continue
        hidden = F.linear(activated, adapter.down)
        output.add_(F.linear(hidden, adapter.up), alpha=adapter.scale)
    return output


__all__ = [
    "AdapterIdentity",
    "DeviceLinearLoRA",
    "EMBEDDING_STAGE",
    "FINAL_STAGE",
    "H3LoRAState",
    "H3TargetSpec",
    "LinearLoRA",
    "LinearLoRABundle",
    "LoRAStagePlan",
    "LoRAStageState",
    "LoRAStageStreamer",
    "build_h3_target_specs",
    "linear_input_act_with_lora",
    "linear_with_lora",
    "parse_h3_linear_lora",
    "prepared_lora_stage",
    "validate_h3_int8_convrot_base",
]
