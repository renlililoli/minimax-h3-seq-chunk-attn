from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

import comfy.sd
from comfy.ldm.minimax.model import MiniMaxH3Model


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_ENV = "MINIMAX_H3_QUANTIZED_CHECKPOINT"


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "comfyui_seqattn_quant_test",
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


package = _load_package()
streaming = sys.modules["comfyui_seqattn_quant_test.minimax_h3"]
runtime_mod = sys.modules["comfyui_seqattn_quant_test.runtime"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not os.environ.get(CHECKPOINT_ENV),
    reason=f"set {CHECKPOINT_ENV} to a ComfyUI H3 quantized checkpoint",
)
@torch.inference_mode()
def test_real_quantized_block_native_streaming_parity():
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
            activation_workspace_mib=256,
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

    for streamed, expected in zip(actual, native):
        assert torch.isfinite(streamed).all()
        torch.testing.assert_close(
            streamed.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
        cosine = torch.nn.functional.cosine_similarity(
            streamed.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine.item() >= 0.999
