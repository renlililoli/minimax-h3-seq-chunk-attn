from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_seqattn import minimax_h3 as streaming


def _qkv_module(quant_format, *, convrot=None, groupsize=64):
    params = None
    if convrot is not None:
        params = SimpleNamespace(convrot=convrot, convrot_groupsize=groupsize)
    return SimpleNamespace(
        quant_format=quant_format,
        weight=SimpleNamespace(_params=params) if params is not None else object(),
    )


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (_qkv_module("nvfp4", convrot=True), "quant_format"),
        (_qkv_module(None, convrot=True), "quant_format"),
        (_qkv_module("int8_tensorwise", convrot=False), "without ConvRot"),
        (_qkv_module("int8_tensorwise"), "unknown packed"),
    ],
)
def test_recompute_rejects_unsupported_qkv_formats(module, message):
    blocks = [SimpleNamespace(attn=SimpleNamespace(qkv_proj=module))]

    with pytest.raises(RuntimeError, match=message):
        streaming._validate_recompute_blocks(blocks)


def test_recompute_accepts_int8_tensorwise_convrot():
    blocks = [
        SimpleNamespace(
            attn=SimpleNamespace(
                qkv_proj=_qkv_module("int8_tensorwise", convrot=True)
            )
        )
        for _ in range(50)
    ]

    streaming._validate_recompute_blocks(blocks)


def test_int8_q_and_kv_row_scale_bias_slices_match_full_projection(monkeypatch):
    qdata = torch.arange(48, dtype=torch.int8).reshape(12, 4)
    scale = torch.linspace(0.25, 1.25, 12)
    bias = torch.linspace(-0.5, 0.5, 12)
    active = {
        "weight": SimpleNamespace(
            _qdata=qdata,
            _params=SimpleNamespace(
                scale=scale,
                convrot=True,
                convrot_groupsize=4,
            ),
        ),
        "bias": bias,
    }
    calls = []

    def int8_linear(tile, weight, row_scale, row_bias, **kwargs):
        calls.append((weight.clone(), row_scale.clone(), row_bias.clone(), kwargs))
        result = tile.float() @ weight.float().transpose(0, 1)
        result.mul_(row_scale.float().reshape(1, -1))
        result.add_(row_bias.float())
        return result.to(tile.dtype)

    monkeypatch.setattr(streaming.comfy_kitchen, "int8_linear", int8_linear)
    tile = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    full = streaming._project_int8_rows(active, tile, 0, 12)
    q = streaming._project_int8_rows(active, tile, 0, 4)
    kv = streaming._project_int8_rows(active, tile, 4, 12)

    torch.testing.assert_close(torch.cat((q, kv), dim=-1), full, rtol=0, atol=0)
    torch.testing.assert_close(calls[1][0], qdata[:4])
    torch.testing.assert_close(calls[1][1], scale[:4])
    torch.testing.assert_close(calls[1][2], bias[:4])
    torch.testing.assert_close(calls[2][0], qdata[4:])
    torch.testing.assert_close(calls[2][1], scale[4:])
    torch.testing.assert_close(calls[2][2], bias[4:])
    assert all(call[3]["convrot"] is True for call in calls)


def test_single_qk_partial_rope_matches_materialized_kernel():
    torch.manual_seed(17)
    tokens = 3
    heads = 2
    head_dim = 8
    projected = torch.randn(tokens, heads * head_dim)
    position_ids = torch.arange(tokens).reshape(tokens, 1)
    norm = SimpleNamespace(weight=torch.randn(head_dim), eps=1e-6)
    block = SimpleNamespace(
        attn=SimpleNamespace(heads=heads, head_dim=head_dim)
    )
    model = SimpleNamespace(
        rope_freqs=lambda ids, _device: ids.float().repeat(1, 4) * 0.1
    )

    actual = streaming._single_qk_with_rope(
        block,
        projected,
        position_ids,
        model,
        norm=norm,
    )
    tensor = projected.view(1, tokens, heads, head_dim)
    rope = streaming._rope_for_tile(block, position_ids, model, projected)
    expected, _ = streaming.comfy.quant_ops.ck.rms_rope_split_half(
        tensor,
        tensor,
        rope,
        norm.weight,
        norm.weight,
        epsilon=norm.eps,
        rot_dim=rope.shape[-3] * 2,
    )

    torch.testing.assert_close(actual, expected[0])


def test_attention_epilogue_reads_explicit_residual_host():
    features = 3
    block = SimpleNamespace(
        attn=SimpleNamespace(out_proj=lambda value: value),
        norm2=lambda value: value,
        mlp=lambda value: torch.zeros_like(value),
    )
    zeros = torch.zeros((1, features))
    ones = torch.ones((1, features))
    modulation = (zeros, zeros, ones, zeros, zeros, zeros)
    ops = streaming._consumer_ops(
        SimpleNamespace(),
        block,
        [(0, 2, 0)],
        modulation,
    )
    residual = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    attention = torch.full((2, features), 0.5)
    expected = residual.clone().add_(attention)

    result = ops.attention_epilogue(attention, residual, 0, 2)

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(residual, expected)


@pytest.mark.parametrize(("blocks", "expected_buffer"), [(1, 1), (2, 0), (3, 1)])
def test_recompute_schedule_returns_odd_even_ping_pong_buffer(
    monkeypatch, blocks, expected_buffer
):
    buffers = [torch.full((2, 2), 0.0), torch.full((2, 2), -1.0)]

    class FakeRunner:
        def run_block(self, source, destination, *_args, **_kwargs):
            destination.copy_(source + 1)
            return destination

    def resident_stages(stages, _device, compute, *, record=None):
        del record
        for index in range(len(stages)):
            compute(index)
        return 1

    monkeypatch.setattr(streaming, "run_weight_stages", resident_stages)
    result = streaming._run_dit_blocks(
        runner=FakeRunner(),
        blocks=[object()] * blocks,
        device=torch.device("cpu"),
        sequence_meta=object(),
        current_hidden=buffers[0],
        scratch_hidden=buffers[1],
        execution_mode="recompute",
        parts_for=lambda _index: (object(), object()),
        softmax_scale=1.0,
        record=None,
    )

    assert result is buffers[expected_buffer]
    torch.testing.assert_close(result, torch.full((2, 2), float(blocks)))
