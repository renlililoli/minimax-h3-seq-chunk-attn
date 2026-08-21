from __future__ import annotations

import comfy.ops
import pytest
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model

from comfyui_seqattn import minimax_h3 as streaming
from comfyui_seqattn import runtime as runtime_mod


def _tiny_model(
    device,
    *,
    use_curves=False,
    num_layers=1,
    token_refiner_num_layers=0,
    text_dim=256,
):
    model = MiniMaxH3Model(
        hidden_size=256,
        num_layers=num_layers,
        token_refiner_num_layers=token_refiner_num_layers,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 2, 2),
        text_dim=text_dim,
        timestep_input_dim=16,
        time_embed_hidden_size=32,
        time_embed_dim=32,
        rope_inv_freq_len=16,
        adaln_curve_grid=8 if use_curves else None,
        dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    for parameter in model.parameters():
        parameter.normal_(mean=0.0, std=0.02)
        parameter.requires_grad_(False)
    model.rope.inv_freq.copy_(torch.linspace(0.001, 0.05, 16, device=device))
    if use_curves:
        model.adaln_t_table.normal_(mean=0.0, std=0.02)
    return model


def _assert_outputs_close(actual, expected):
    for streamed, native in zip(actual, expected):
        assert torch.isfinite(streamed).all()
        torch.testing.assert_close(
            streamed.float(), native.float(), rtol=2e-2, atol=2e-2
        )
        cosine = torch.nn.functional.cosine_similarity(
            streamed.float().flatten(), native.float().flatten(), dim=0
        )
        assert cosine.item() >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_one_block_native_streaming_parity():
    torch.manual_seed(7)
    device = torch.device("cuda")
    model = _tiny_model(device)

    video = torch.randn((1, 2, 1, 4, 4), device=device, dtype=torch.bfloat16)
    audio = torch.randn((1, 4, 2, 2), device=device, dtype=torch.bfloat16)
    context = torch.randn((1, 3, 256), device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([600.0], device=device)

    native = model._forward(
        [video, audio], timestep, context, transformer_options={}
    )
    runtime = runtime_mod.SeqAttnRuntime(
        runtime_mod.SeqAttnSettings(
            activation_workspace_mib=128,
            kv_chunk_tokens=64,
            projection_chunk_tokens=4,
        )
    )
    actual = streaming.streaming_minimax_h3_forward(
        model,
        runtime,
        [video, audio],
        timestep,
        context,
        transformer_options={},
    )

    _assert_outputs_close(actual, native)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_refined_conditioning_cached_across_denoising_calls():
    torch.manual_seed(9)
    device = torch.device("cuda")
    model = _tiny_model(
        device,
        token_refiner_num_layers=1,
        text_dim=128,
    )
    video = torch.randn((1, 2, 1, 4, 4), device=device, dtype=torch.bfloat16)
    audio = torch.randn((1, 4, 2, 2), device=device, dtype=torch.bfloat16)
    context = torch.randn((1, 3, 128), device=device, dtype=torch.bfloat16)
    changed_context = context.clone()
    changed_context[0, 0, 0] += 1
    timesteps = [
        torch.tensor([700.0], device=device),
        torch.tensor([500.0], device=device),
        torch.tensor([300.0], device=device),
    ]

    expected = [
        model._forward(
            [video, audio],
            timesteps[0],
            context,
            transformer_options={},
        ),
        model._forward(
            [video, audio],
            timesteps[1],
            context,
            transformer_options={},
        ),
        model._forward(
            [video, audio],
            timesteps[2],
            changed_context,
            transformer_options={},
        ),
    ]
    calls = {"condition_proj": 0, "token_refiner": 0}

    def count_condition_proj(*_args):
        calls["condition_proj"] += 1

    def count_token_refiner(*_args):
        calls["token_refiner"] += 1

    hooks = [
        model.condition_proj.register_forward_hook(count_condition_proj),
        model.token_refiner.register_forward_hook(count_token_refiner),
    ]
    runtime = runtime_mod.SeqAttnRuntime(
        runtime_mod.SeqAttnSettings(
            activation_workspace_mib=128,
            kv_chunk_tokens=64,
            projection_chunk_tokens=4,
        )
    )
    try:
        actual = [
            streaming.streaming_minimax_h3_forward(
                model,
                runtime,
                [video, audio],
                timesteps[0],
                context,
                transformer_options={"uuids": ["prompt-a"]},
            ),
            streaming.streaming_minimax_h3_forward(
                model,
                runtime,
                [video, audio],
                timesteps[1],
                context,
                transformer_options={"uuids": ["prompt-a"]},
            ),
            streaming.streaming_minimax_h3_forward(
                model,
                runtime,
                [video, audio],
                timesteps[2],
                changed_context,
                transformer_options={"uuids": ["prompt-b"]},
            ),
        ]
    finally:
        for hook in hooks:
            hook.remove()

    for actual_output, expected_output in zip(actual, expected):
        _assert_outputs_close(actual_output, expected_output)
    assert calls == {"condition_proj": 2, "token_refiner": 2}
    cached = runtime._refined_conditioning
    assert cached is not None
    assert cached.device.type == "cpu"
    assert cached.dtype == torch.bfloat16
    assert cached.is_pinned()
    assert runtime.refined_conditioning_cache_stats == {
        "hits": 1,
        "misses": 2,
        "stores": 2,
        "bypasses": 0,
        "entries": 1,
        "host_bytes": cached.numel() * cached.element_size(),
    }

    clone = runtime.clone()
    assert clone.refined_conditioning_cache_stats["entries"] == 0
    runtime.clear()
    assert runtime._refined_conditioning is None
    assert runtime.refined_conditioning_cache_stats == {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "bypasses": 0,
        "entries": 0,
        "host_bytes": 0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_curve_adaln_t2va_fl2va_and_ref2va_layout_parity():
    torch.manual_seed(11)
    device = torch.device("cuda")
    model = _tiny_model(device, use_curves=True)
    video = torch.randn((1, 2, 1, 4, 4), device=device, dtype=torch.bfloat16)
    audio = torch.randn((1, 4, 2, 2), device=device, dtype=torch.bfloat16)
    context = torch.randn((1, 3, 256), device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([350.0], device=device)
    cond_video = torch.randn_like(video)
    cond_video_last = torch.randn_like(video)
    cond_audio = torch.randn_like(audio)
    payloads = [
        {
            "text_token_tags": torch.tensor([1, 0, 2]),
            "seed": 17,
        },
        {
            "keyframes": [{"resolved_frame_index": 0}],
            "frame_count": 5,
            "cond_video_latents": [cond_video],
            "text_token_tags": torch.tensor([1, 0, 2]),
            "seed": 19,
        },
        {
            "keyframes": [{"resolved_frame_index": 4}],
            "frame_count": 5,
            "cond_video_latents": [cond_video_last],
            "text_token_tags": torch.tensor([1, 0, 2]),
            "seed": 20,
        },
        {
            "keyframes": [
                {"resolved_frame_index": 0},
                {"resolved_frame_index": 4},
            ],
            "frame_count": 5,
            "cond_video_latents": [cond_video, cond_video_last],
            "text_token_tags": torch.tensor([1, 0, 2]),
            "seed": 21,
        },
        {
            "refs": [
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
            "cond_video_latents": [cond_video],
            "cond_audio_latents": [cond_audio],
            "text_token_tags": torch.tensor([1, 0, 2]),
            "seed": 23,
        },
    ]

    runtime = runtime_mod.SeqAttnRuntime(
        runtime_mod.SeqAttnSettings(
            activation_workspace_mib=128,
            kv_chunk_tokens=64,
            projection_chunk_tokens=4,
        )
    )
    for payload in payloads:
        native = model._forward(
            [video, audio],
            timestep,
            context,
            transformer_options={},
            minimax_payload=payload,
        )
        actual = streaming.streaming_minimax_h3_forward(
            model,
            runtime,
            [video, audio],
            timestep,
            context,
            transformer_options={},
            minimax_payload=payload,
        )
        _assert_outputs_close(actual, native)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_full_50_block_tiny_forward_parity():
    torch.manual_seed(29)
    device = torch.device("cuda")
    model = _tiny_model(device, num_layers=50)
    video = torch.randn((1, 2, 1, 4, 4), device=device, dtype=torch.bfloat16)
    audio = torch.randn((1, 4, 2, 2), device=device, dtype=torch.bfloat16)
    context = torch.randn((1, 3, 256), device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device)
    native = model._forward(
        [video, audio], timestep, context, transformer_options={}
    )
    runtime = runtime_mod.SeqAttnRuntime(
        runtime_mod.SeqAttnSettings(
            activation_workspace_mib=128,
            kv_chunk_tokens=64,
            projection_chunk_tokens=4,
        )
    )
    actual = streaming.streaming_minimax_h3_forward(
        model,
        runtime,
        [video, audio],
        timestep,
        context,
        transformer_options={},
    )
    _assert_outputs_close(actual, native)


@pytest.mark.parametrize("tokens", [257, 1023, 3072])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_projected_attention_full_output_parity(tokens):
    torch.manual_seed(1000 + tokens)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_size = 256
    heads = 2
    head_dim = 128
    hidden = torch.randn(
        (tokens, hidden_size), dtype=dtype, device="cpu", pin_memory=True
    )
    qkv_proj = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=False).to(
        device=device, dtype=dtype
    )
    out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(
        device=device, dtype=dtype
    )

    def project_qkv(tile, start, stop):
        del start, stop
        qkv = qkv_proj(tile).view(-1, 3, heads, head_dim)
        return tuple(qkv[:, index].contiguous() for index in range(3))

    def output_projector(tile, start, stop):
        del start, stop
        return out_proj(tile)

    runtime = runtime_mod.SeqAttnRuntime(
        runtime_mod.SeqAttnSettings(
            activation_workspace_mib=256,
            kv_chunk_tokens=256,
            projection_chunk_tokens=257,
        )
    )
    runner = runtime.runner_for(
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    actual = runner(
        hidden,
        torch.tensor([0, tokens], dtype=torch.int32),
        project_qkv=project_qkv,
        output_projector=output_projector,
        output_features=hidden_size,
    )

    hidden_gpu = hidden.to(device)
    qkv = qkv_proj(hidden_gpu).view(tokens, 3, heads, head_dim)
    expected_attn = torch.nn.functional.scaled_dot_product_attention(
        qkv[:, 0].transpose(0, 1).unsqueeze(0),
        qkv[:, 1].transpose(0, 1).unsqueeze(0),
        qkv[:, 2].transpose(0, 1).unsqueeze(0),
        scale=head_dim**-0.5,
    )
    expected = out_proj(
        expected_attn.squeeze(0).transpose(0, 1).reshape(tokens, hidden_size)
    ).cpu()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=2e-2, atol=7e-2
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine.item() >= 0.999
