from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from seqattn_core.dit.minimax_h3 import H3SequenceMeta

from comfyui_seqattn import qwen


class _RMSNorm(torch.nn.Module):
    def __init__(self, features: int, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(features, device=device, dtype=torch.bfloat16) * 0.02 + 1
        )
        self.eps = 1e-6

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(value.dtype)


class _Embedding(torch.nn.Embedding):
    def forward(self, input_ids, out_dtype=None):
        output = super().forward(input_ids)
        return output if out_dtype is None else output.to(out_dtype)


class _DecoderAttention(torch.nn.Module):
    def __init__(self, hidden, q_heads, kv_heads, head_dim, device):
        super().__init__()
        self.num_heads = q_heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(
            hidden, q_heads * head_dim, bias=False, device=device, dtype=torch.bfloat16
        )
        self.k_proj = torch.nn.Linear(
            hidden, kv_heads * head_dim, bias=False, device=device, dtype=torch.bfloat16
        )
        self.v_proj = torch.nn.Linear(
            hidden, kv_heads * head_dim, bias=False, device=device, dtype=torch.bfloat16
        )
        self.o_proj = torch.nn.Linear(
            q_heads * head_dim, hidden, bias=False, device=device, dtype=torch.bfloat16
        )
        self.q_norm = _RMSNorm(head_dim, device)
        self.k_norm = _RMSNorm(head_dim, device)


class _DecoderMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate, device):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            hidden, intermediate, bias=False, device=device, dtype=torch.bfloat16
        )
        self.up_proj = torch.nn.Linear(
            hidden, intermediate, bias=False, device=device, dtype=torch.bfloat16
        )
        self.down_proj = torch.nn.Linear(
            intermediate, hidden, bias=False, device=device, dtype=torch.bfloat16
        )
        self.activation = F.silu


class _DecoderLayer(torch.nn.Module):
    def __init__(self, hidden, q_heads, kv_heads, head_dim, intermediate, device):
        super().__init__()
        self.self_attn = _DecoderAttention(
            hidden, q_heads, kv_heads, head_dim, device
        )
        self.mlp = _DecoderMLP(hidden, intermediate, device)
        self.input_layernorm = _RMSNorm(hidden, device)
        self.post_attention_layernorm = _RMSNorm(hidden, device)


class _TinyDecoderModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        hidden = 128
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed_tokens = _Embedding(
            512, hidden, device=device, dtype=torch.bfloat16
        )
        self.layers = torch.nn.ModuleList(
            [
                _DecoderLayer(hidden, 8, 1, 16, 256, device)
                for _ in range(3)
            ]
        )

    def compute_freqs_cis(self, position_ids, device):
        from comfy.text_encoders.llama import precompute_freqs_cis

        return precompute_freqs_cis(
            16,
            position_ids,
            5_000_000.0,
            rope_dims=[3, 3, 2],
            interleaved_mrope=True,
            device=device,
        )


class _VisionMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate, device):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(
            hidden, intermediate, device=device, dtype=torch.bfloat16
        )
        self.linear_fc2 = torch.nn.Linear(
            intermediate, hidden, device=device, dtype=torch.bfloat16
        )

    def forward(self, value):
        return self.linear_fc2(F.gelu(self.linear_fc1(value), approximate="tanh"))


class _VisionAttention(torch.nn.Module):
    def __init__(self, hidden, heads, device):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.qkv = torch.nn.Linear(
            hidden, hidden * 3, device=device, dtype=torch.bfloat16
        )
        self.proj = torch.nn.Linear(
            hidden, hidden, device=device, dtype=torch.bfloat16
        )


class _VisionBlock(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(
            144, device=device, dtype=torch.bfloat16
        )
        self.norm2 = torch.nn.LayerNorm(
            144, device=device, dtype=torch.bfloat16
        )
        self.attn = _VisionAttention(144, 2, device)
        self.mlp = _VisionMLP(144, 288, device)


def _initialize(module):
    for parameter in module.parameters():
        if parameter.ndim == 1:
            parameter.data.fill_(1)
        else:
            parameter.data.normal_(mean=0, std=0.02)
        parameter.requires_grad_(False)
    return module.eval()


def _resident_stages(stages, _device, compute, *, record=None):
    del record
    for index in range(len(stages)):
        compute(index)
    return min(len(stages), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_packed_vision_block_parity_with_non_divisible_tiles():
    from comfy.text_encoders.llama import apply_rope

    torch.manual_seed(17)
    device = torch.device("cuda")
    block = _initialize(_VisionBlock(device))
    hidden_gpu = torch.randn((9, 144), device=device, dtype=torch.bfloat16)
    hidden_host = hidden_gpu.cpu().pin_memory()
    angles = torch.randn((9, 36), dtype=torch.float32)
    cu = torch.tensor([0, 4, 9], dtype=torch.int32)

    normalized = block.norm1(hidden_gpu)
    q, k, v = (
        block.attn.qkv(normalized)
        .reshape(9, 3, 2, 72)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    emb = torch.cat((angles.to(device), angles.to(device)), dim=-1)
    sin = emb.sin().unsqueeze(-2)
    q, k = apply_rope(
        q,
        k,
        (emb.cos().unsqueeze(-2), sin[..., :36], -sin[..., 36:]),
    )
    outputs = []
    for start, stop in ((0, 4), (4, 9)):
        outputs.append(
            F.scaled_dot_product_attention(
                q[start:stop].transpose(0, 1).unsqueeze(0),
                k[start:stop].transpose(0, 1).unsqueeze(0),
                v[start:stop].transpose(0, 1).unsqueeze(0),
                scale=72**-0.5,
            )
            .transpose(1, 2)
            .reshape(stop - start, 144)
        )
    expected = hidden_gpu + block.attn.proj(torch.cat(outputs))
    expected = expected + block.mlp(block.norm2(expected))

    runtime = qwen.QwenEncodeRuntime(
        qwen.QwenSeqAttnSettings(
            q_chunk_tokens=128,
            kv_chunk_tokens=128,
            qkv_tile_tokens=4,
            mlp_tile_tokens=4,
        ),
        device,
    )
    runner = runtime.runner(
        tokens=9,
        hidden_features=144,
        q_heads=2,
        kv_heads=2,
        head_dim=72,
    )
    projection, ops = qwen._vision_block_parts(block, angles, device)
    runner.run_block_(
        hidden_host,
        H3SequenceMeta(cu_seqlens=cu),
        projection,
        ops,
        softmax_scale=72**-0.5,
        causal=False,
    )

    torch.testing.assert_close(
        hidden_host.float(), expected.cpu().float(), rtol=3e-2, atol=3e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_decoder_causal_gqa_mrope_and_three_cpu_deepstack_injections(monkeypatch):
    from comfy.text_encoders.llama import apply_rope

    torch.manual_seed(23)
    device = torch.device("cuda")
    model = _initialize(_TinyDecoderModel(device))
    decoder = SimpleNamespace(model=model)
    embedding = torch.randn((2, 128), dtype=torch.bfloat16)
    visual_entry = {"type": "image", "data": torch.empty((1, 32, 32, 3))}
    layout = qwen.QwenPresentationLayout(
        spans=(
            qwen.QwenInputSpan("token", 0, 1, 11),
            qwen.QwenInputSpan("token", 1, 2, 21),
            qwen.QwenInputSpan("visual", 2, 3, visual_entry),
            qwen.QwenInputSpan("token", 3, 4, 22),
            qwen.QwenInputSpan("token", 4, 5, 12),
            qwen.QwenInputSpan("embedding", 5, 7, embedding),
        ),
        attention_mask=(1,) * 7,
        token_tags=(1, 0, 0, 0, 1, 1, 1),
        total_rows=7,
        visual_rows=1,
    )
    visual = qwen.PreparedVisual(
        layout.visual_spans[0],
        torch.tensor([[1, 2, 2]], dtype=torch.long),
        0,
        4,
        0,
        1,
    )
    merged_visual = torch.randn((1, 128), dtype=torch.bfloat16).pin_memory()
    deepstack = [
        torch.randn((1, 128), dtype=torch.bfloat16).pin_memory()
        for _ in range(3)
    ]
    positions = qwen._decoder_position_ids(layout, [visual]).to(device)
    expected = torch.empty((7, 128), device=device, dtype=torch.bfloat16)
    expected[0] = model.embed_tokens(torch.tensor(11, device=device))
    expected[1] = model.embed_tokens(torch.tensor(21, device=device))
    expected[2] = merged_visual[0].to(device)
    expected[3] = model.embed_tokens(torch.tensor(22, device=device))
    expected[4] = model.embed_tokens(torch.tensor(12, device=device))
    expected[5:7] = embedding.to(device)

    for index, layer in enumerate(model.layers):
        residual = expected
        tile = layer.input_layernorm(expected)
        attention = layer.self_attn
        q = attention.q_norm(attention.q_proj(tile).view(7, 8, 16))
        k = attention.k_norm(attention.k_proj(tile).view(7, 1, 16))
        v = attention.v_proj(tile).view(7, 1, 16)
        freqs = model.compute_freqs_cis(positions, device)
        q, k = apply_rope(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            freqs,
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
            scale=16**-0.5,
        )
        expected = residual + attention.o_proj(
            attended.transpose(1, 2).reshape(7, 128)
        )
        residual = expected
        tile = layer.post_attention_layernorm(expected)
        expected = residual + layer.mlp.down_proj(
            layer.mlp.activation(layer.mlp.gate_proj(tile))
            * layer.mlp.up_proj(tile)
        )
        expected[2].add_(deepstack[index][0].to(device))

    monkeypatch.setattr(qwen, "run_weight_stages", _resident_stages)
    monkeypatch.setattr(
        qwen,
        "_embed_token_rows_cpu",
        lambda module, token_ids, dtype: module(token_ids.to(device)).to(
            device="cpu", dtype=dtype
        ),
    )
    runtime = qwen.QwenEncodeRuntime(
        qwen.QwenSeqAttnSettings(
            q_chunk_tokens=128,
            kv_chunk_tokens=128,
            qkv_tile_tokens=3,
            mlp_tile_tokens=2,
        ),
        device,
    )
    actual, _ = qwen._encode_decoder(
        decoder,
        layout,
        merged_visual,
        [item.clone().pin_memory() for item in deepstack],
        [visual],
        runtime,
    )
    assert runtime.active_runner_count == 0

    torch.testing.assert_close(
        actual.float(), expected.cpu().float(), rtol=3e-2, atol=3e-2
    )
    cosine = F.cosine_similarity(actual.float().flatten(), expected.cpu().float().flatten(), dim=0)
    assert cosine.item() > 0.999
