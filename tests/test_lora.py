from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from comfyui_seqattn import lora as lora_mod


def _identity(name: str = "adapter.safetensors") -> lora_mod.AdapterIdentity:
    return lora_mod.AdapterIdentity(name, f"/models/loras/{name}", 1234, 5678)


def _layer(
    *,
    target: str = "blocks.0.attn.out_proj",
    strength: float = 1.0,
    alpha: float | None = 2.0,
) -> lora_mod.LinearLoRA:
    down = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    up = torch.tensor([[0.25], [1.5]], dtype=torch.float32)
    return lora_mod.LinearLoRA(
        adapter=_identity(),
        target=target,
        down=down,
        up=up,
        alpha=alpha,
        rank=1,
        in_features=3,
        out_features=2,
        dtype=down.dtype,
        strength=strength,
    )


def _device_adapter(layer: lora_mod.LinearLoRA) -> lora_mod.DeviceLinearLoRA:
    return lora_mod.DeviceLinearLoRA(
        adapter_name=layer.adapter.name,
        target=layer.target,
        down=layer.down,
        up=layer.up,
        scale=layer.scale,
        signature=layer.signature,
    )


def test_parse_generic_linear_lora_and_reject_unconsumed_keys():
    target = "blocks.0.attn.out_proj"
    specs = (lora_mod.H3TargetSpec(target, ("block", 0), 3, 2),)
    state_dict = {
        f"{target}.lora_A.weight": torch.tensor([[1.0, 2.0, 3.0]]),
        f"{target}.lora_B.weight": torch.tensor([[4.0], [5.0]]),
        f"{target}.alpha": torch.tensor(0.5),
    }

    bundle = lora_mod.parse_h3_linear_lora(
        state_dict,
        identity=_identity(),
        strength=-2.0,
        specs=specs,
    )

    assert len(bundle.layers) == 1
    layer = bundle.layers[0]
    assert layer.target == target
    assert layer.rank == 1
    assert layer.alpha == 0.5
    assert layer.scale == -1.0
    assert layer.down.device.type == "cpu"
    assert layer.down.is_contiguous()
    assert layer.up.is_contiguous()

    with pytest.raises(ValueError, match="unmapped or unsupported"):
        lora_mod.parse_h3_linear_lora(
            {**state_dict, "unknown.lora_A.weight": torch.ones(1, 1)},
            identity=_identity(),
            strength=1.0,
            specs=specs,
        )


def test_parse_rejects_shape_mismatch_and_non_lora_patch():
    target = "blocks.0.attn.out_proj"
    specs = (lora_mod.H3TargetSpec(target, ("block", 0), 3, 2),)
    with pytest.raises(ValueError, match="shape mismatch"):
        lora_mod.parse_h3_linear_lora(
            {
                f"{target}.lora_A.weight": torch.ones(1, 4),
                f"{target}.lora_B.weight": torch.ones(2, 1),
            },
            identity=_identity(),
            strength=1.0,
            specs=specs,
        )
    with pytest.raises(ValueError, match="only ordinary Linear LoRA"):
        lora_mod.parse_h3_linear_lora(
            {f"{target}.diff": torch.ones(2, 3)},
            identity=_identity(),
            strength=1.0,
            specs=specs,
        )


def test_linear_lora_matches_explicit_single_and_multi_adapter_math():
    base = torch.nn.Linear(3, 2, bias=True)
    x = torch.tensor([[0.5, -1.0, 2.0]])
    first = _device_adapter(_layer(strength=0.75))
    second_layer = _layer(strength=-0.25, alpha=None)
    second = _device_adapter(second_layer)

    actual = lora_mod.linear_with_lora(base, x, (first, second))
    expected = base(x)
    for adapter in (first, second):
        expected = expected + F.linear(F.linear(x, adapter.down), adapter.up) * adapter.scale

    torch.testing.assert_close(actual, expected)


def test_linear_input_activation_lora_uses_same_swiglu_input(monkeypatch):
    base = torch.nn.Linear(2, 2, bias=False)
    x = torch.tensor([[0.5, -1.0, 2.0, 3.0]])
    layer = _layer(target="blocks.0.mlp.fc2", strength=0.5, alpha=None)
    layer = lora_mod.LinearLoRA(
        **{
            **layer.__dict__,
            "down": torch.tensor([[2.0, -1.0]]),
            "up": torch.tensor([[0.25], [1.5]]),
            "in_features": 2,
            "out_features": 2,
        }
    )
    adapter = _device_adapter(layer)

    def reference_base(linear, value, activation):
        assert activation == "swiglu"
        gate, payload = value.chunk(2, dim=-1)
        return linear(F.silu(gate) * payload)

    monkeypatch.setattr(lora_mod.comfy.ops, "linear_input_act", reference_base)
    activated = F.silu(x[:, :2]) * x[:, 2:]
    expected = base(activated) + F.linear(
        F.linear(activated, adapter.down), adapter.up
    ) * adapter.scale

    actual = lora_mod.linear_input_act_with_lora(base, x, "swiglu", (adapter,))

    torch.testing.assert_close(actual, expected)


def test_state_plans_preserve_order_skip_zero_strength_and_stage_two_slots():
    first = _layer(strength=1.0)
    second = _layer(strength=-0.5)
    zero = _layer(target="blocks.1.attn.out_proj", strength=0.0)
    specs = (
        lora_mod.H3TargetSpec(first.target, ("block", 0), 3, 2),
        lora_mod.H3TargetSpec(zero.target, ("block", 1), 3, 2),
    )
    state = lora_mod.H3LoRAState(
        bundles=(
            lora_mod.LinearLoRABundle(first.adapter, first.strength, (first,)),
            lora_mod.LinearLoRABundle(second.adapter, second.strength, (second,)),
            lora_mod.LinearLoRABundle(zero.adapter, zero.strength, (zero,)),
        ),
        target_specs=specs,
    )
    plan0 = state.plan_for(("block", 0), activation_dtype=torch.float32)
    plan1 = state.plan_for(("block", 1), activation_dtype=torch.float32)
    plan2 = lora_mod.LoRAStagePlan(("block", 2), plan0.adapters_by_target)

    assert [adapter.scale for adapter in plan0.adapters_by_target[0][1]] == [2.0, -1.0]
    assert not plan1.adapters_by_target

    streamer = lora_mod.LoRAStageStreamer([plan0, plan1, plan2], torch.device("cpu"))
    current = streamer.prepare(0)
    assert streamer.prepare(1) is None
    final = streamer.prepare(2)
    with pytest.raises(RuntimeError, match="two staged-slot"):
        streamer._free_slot()
    streamer.wait_ready(current)
    staged = streamer.adapters_for(0)[first.target]
    torch.testing.assert_close(staged[0].down, first.down)
    torch.testing.assert_close(staged[1].up, second.up)
    streamer.release(current)
    streamer.wait_ready(final)
    streamer.release(final)
    streamer.close()
    assert streamer.closed


def test_int8_convrot_validation_rejects_wrong_base_and_existing_patches():
    class LinearLike(torch.nn.Module):
        def __init__(self, *, valid=True, patched=False):
            super().__init__()
            self.in_features = 1
            self.out_features = 1
            self.weight = torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
            self.weight._params = SimpleNamespace(convrot=valid)
            self.quant_format = "int8_tensorwise" if valid else "bf16"
            self.weight_function = (object(),) if patched else ()
            self.bias_function = ()

    class Block(torch.nn.Module):
        def __init__(self, *, valid=True):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv_proj = LinearLike(valid=valid)
            self.attn.out_proj = LinearLike()
            self.mlp = torch.nn.Module()
            self.mlp.fc1 = LinearLike()
            self.mlp.fc2 = LinearLike()

    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([Block() for _ in range(50)])
    specs = (lora_mod.H3TargetSpec("blocks.0.attn.out_proj", ("block", 0), 1, 1),)
    lora_mod.validate_h3_int8_convrot_base(model, specs)

    model.blocks[3].attn.qkv_proj.quant_format = "bf16"
    with pytest.raises(ValueError, match="INT8 tensorwise ConvRot"):
        lora_mod.validate_h3_int8_convrot_base(model, specs)
    model.blocks[3].attn.qkv_proj.quant_format = "int8_tensorwise"
    model.blocks[3].attn.qkv_proj.weight._params.convrot = True
    model.blocks[0].attn.out_proj.weight_function = (object(),)
    with pytest.raises(ValueError, match="unpatched base Linear"):
        lora_mod.validate_h3_int8_convrot_base(model, specs)
