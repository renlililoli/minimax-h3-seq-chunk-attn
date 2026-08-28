from __future__ import annotations

import os

import comfy.sd
import pytest
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model

from comfyui_seqattn import minimax_h3 as streaming
from comfyui_seqattn import runtime as runtime_mod

CHECKPOINT_ENV = "MINIMAX_H3_QUANTIZED_CHECKPOINT"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not os.environ.get(CHECKPOINT_ENV),
    reason=f"set {CHECKPOINT_ENV} to a ComfyUI H3 quantized checkpoint",
)
@pytest.mark.parametrize("execution_mode", ["materialized", "recompute"])
@torch.inference_mode()
def test_real_quantized_block_native_streaming_parity(monkeypatch, execution_mode):
    def run_resident_stages(stages, device, compute, *, record=None):
        del device, record
        for index in range(len(stages)):
            compute(index)
        return min(len(stages), 2)

    monkeypatch.setattr(streaming, "run_weight_stages", run_resident_stages)
    checkpoint = os.environ[CHECKPOINT_ENV]
    patcher = comfy.sd.load_diffusion_model(checkpoint)
    model = patcher.model.diffusion_model
    assert isinstance(model, MiniMaxH3Model)
    assert model.dtype == torch.bfloat16
    block = model.blocks[0]
    assert block.attn.qkv_proj.quant_format in {"int8_tensorwise", "nvfp4"}
    model.blocks = torch.nn.ModuleList([block])

    device = torch.device("cuda")
    torch.manual_seed(41)
    video = torch.randn((1, 24, 1, 4, 4), device=device, dtype=torch.bfloat16)
    audio = torch.randn((1, 32, 2, 2), device=device, dtype=torch.bfloat16)
    context = torch.randn(
        (1, 3, model.hidden_size), device=device, dtype=torch.bfloat16
    )
    timestep = torch.tensor([500.0], device=device)

    native = model._forward(
        [video, audio], timestep, context, transformer_options={}
    )
    runtime = runtime_mod.SeqAttnRuntime(
        runtime_mod.SeqAttnSettings(
            execution_mode=execution_mode,
            q_chunk_tokens=32,
            kv_chunk_tokens=64,
            qkv_tile_tokens=4,
            mlp_tile_tokens=4,
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

    for streamed, expected in zip(actual, native):
        assert torch.isfinite(streamed).all()
        torch.testing.assert_close(
            streamed.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
        cosine = torch.nn.functional.cosine_similarity(
            streamed.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine.item() >= 0.999

    assert len(runtime._dit_runners) == 1
    runner = next(iter(runtime._dit_runners.values()))
    del actual
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    reused = streaming.streaming_minimax_h3_forward(
        model,
        runtime,
        [video, audio],
        timestep,
        context,
        transformer_options={},
    )
    torch.cuda.synchronize(device)

    assert next(iter(runtime._dit_runners.values())) is runner
    for streamed, expected in zip(reused, native):
        torch.testing.assert_close(
            streamed.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
    output_bytes = sum(item.numel() * item.element_size() for item in reused)
    assert torch.cuda.memory_allocated(device) <= baseline + output_bytes + 16 * 2**20
