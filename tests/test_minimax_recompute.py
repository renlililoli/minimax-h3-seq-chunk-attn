from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from seqattn_core.dit.minimax_h3 import H3DenoisingStep

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


@pytest.mark.parametrize(
    ("segments", "seq_len", "expected"),
    [
        (((0, 3, "text"), (3, 7, "audio"), (7, 15, "video")), 15, 3),
        (
            (
                (0, 3, "text"),
                (3, 7, "cond"),
                (7, 11, "audio"),
                (11, 19, "video"),
            ),
            19,
            7,
        ),
        (
            (
                (0, 3, "text"),
                (3, 7, "ref_img"),
                (7, 11, "ref_audio"),
                (11, 15, "ref_img"),
                (15, 19, "audio"),
                (19, 27, "video"),
            ),
            27,
            15,
        ),
    ],
)
def test_conditioning_prefix_ends_before_target_audio(segments, seq_len, expected):
    layout = SimpleNamespace(segments=segments, seq_len=seq_len)

    assert streaming._conditioning_prefix_tokens(layout) == expected


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (1.0, 0),
        (0.75, 1),
        (0.6, 1),
        (0.5, 2),
        (0.0, 3),
        (1.05, 0),
    ],
)
def test_denoising_step_uses_sampler_schedule(current, expected):
    step = streaming._resolve_denoising_step(
        {
            "sigmas": torch.tensor([current, current]),
            "sample_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
        }
    )

    assert step == H3DenoisingStep(step_index=expected, total_steps=4)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "sampler metadata"),
        (
            {
                "sigmas": torch.tensor([0.5, 0.25]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
            "one current sigma",
        ),
        (
            {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.5, 0.0]),
            },
            "strictly descending",
        ),
        (
            {
                "sigmas": torch.tensor([-0.1]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
            "outside the sampler schedule",
        ),
    ],
)
def test_denoising_step_rejects_ambiguous_metadata(options, message):
    with pytest.raises(RuntimeError, match=message):
        streaming._resolve_denoising_step(options)


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


def test_q_and_kv_lora_row_slices_match_full_delta():
    torch.manual_seed(23)
    tile = torch.randn(3, 5)
    adapters = [
        SimpleNamespace(
            down=torch.randn(2, 5),
            up=torch.randn(12, 2),
            scale=0.75,
        ),
        SimpleNamespace(
            down=torch.randn(3, 5),
            up=torch.randn(12, 3),
            scale=-0.25,
        ),
    ]
    full = torch.zeros(3, 12)
    q = torch.zeros(3, 4)
    kv = torch.zeros(3, 8)

    streaming._add_lora_rows_(full, tile, adapters, 0, 12)
    streaming._add_lora_rows_(q, tile, adapters, 0, 4)
    streaming._add_lora_rows_(kv, tile, adapters, 4, 12)

    torch.testing.assert_close(torch.cat((q, kv), dim=-1), full)


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
        0,
        [(0, 2, 0)],
        modulation,
        {},
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
    calls = []

    class FakeRunner:
        def run_block(self, source, destination, *_args, **kwargs):
            calls.append(kwargs)
            destination.copy_(source + 1)
            return destination

    def resident_stages(
        stages,
        _device,
        compute,
        *,
        record=None,
        auxiliary=None,
    ):
        del record
        try:
            for index in range(len(stages)):
                state = None if auxiliary is None else auxiliary.prepare(index)
                if auxiliary is not None:
                    auxiliary.wait_ready(state)
                compute(index)
                if auxiliary is not None:
                    auxiliary.compute_end(state)
                    auxiliary.release(state)
            return 1
        finally:
            if auxiliary is not None:
                auxiliary.close()

    monkeypatch.setattr(streaming, "run_weight_stages", resident_stages)
    denoising_step = H3DenoisingStep(step_index=2, total_steps=4)
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
        denoising_step=denoising_step,
    )

    assert result is buffers[expected_buffer]
    torch.testing.assert_close(result, torch.full((2, 2), float(blocks)))
    assert [call["block_index"] for call in calls] == list(range(blocks))
    assert all(call["denoising_step"] is denoising_step for call in calls)


def test_materialized_schedule_propagates_block_and_denoising_step(monkeypatch):
    calls = []

    class FakeRunner:
        def run_block_(self, hidden, *_args, **kwargs):
            calls.append(kwargs)
            hidden.add_(1)
            return hidden

    def resident_stages(
        stages,
        _device,
        compute,
        *,
        record=None,
        auxiliary=None,
    ):
        del record, auxiliary
        for index in range(len(stages)):
            compute(index)
        return 1

    monkeypatch.setattr(streaming, "run_weight_stages", resident_stages)
    denoising_step = H3DenoisingStep(step_index=2, total_steps=4)
    hidden = torch.zeros((2, 2))
    result = streaming._run_dit_blocks(
        runner=FakeRunner(),
        blocks=[object()] * 3,
        device=torch.device("cpu"),
        sequence_meta=object(),
        current_hidden=hidden,
        scratch_hidden=None,
        execution_mode="materialized",
        parts_for=lambda _index: (object(), object()),
        softmax_scale=1.0,
        record=None,
        denoising_step=denoising_step,
    )

    assert result is hidden
    torch.testing.assert_close(result, torch.full((2, 2), 3.0))
    assert [call["block_index"] for call in calls] == [0, 1, 2]
    assert all(call["denoising_step"] is denoising_step for call in calls)
