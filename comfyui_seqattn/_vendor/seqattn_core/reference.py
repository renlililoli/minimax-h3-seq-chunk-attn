from __future__ import annotations

import math

import torch

from .validation import validate_host_qkv


@torch.inference_mode()
def streaming_attention_reference(
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    q_chunk_tokens: int,
    kv_chunk_tokens: int,
    device: torch.device | str = "cpu",
    softmax_scale: float | None = None,
    causal: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically simple FP32 online-softmax implementation.

    This path is deliberately written in PyTorch and optimized for auditability,
    not speed.  It is the correctness oracle for the Triton backend.
    """

    q_bounds, k_bounds = validate_host_qkv(
        q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k
    )
    if q_chunk_tokens <= 0 or kv_chunk_tokens <= 0:
        raise ValueError("q_chunk_tokens and kv_chunk_tokens must be positive")
    device = torch.device(device)
    scale = q_cpu.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    if out is None:
        out = torch.empty_like(q_cpu)
    if out.shape != q_cpu.shape or out.device.type != "cpu" or out.dtype != q_cpu.dtype:
        raise ValueError("out must be a CPU tensor matching q shape and dtype")

    group_size = q_cpu.shape[1] // k_cpu.shape[1]
    for q_start, q_stop, k_start, k_stop in zip(
        q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
    ):
        q_length = q_stop - q_start
        k_length = k_stop - k_start
        causal_shift = k_length - q_length
        for q_tile_start in range(q_start, q_stop, q_chunk_tokens):
            q_tile_stop = min(q_tile_start + q_chunk_tokens, q_stop)
            q_local_start = q_tile_start - q_start
            q = q_cpu[q_tile_start:q_tile_stop].to(device=device).transpose(0, 1).float()
            heads, query_tokens, head_dim = q.shape
            running_max = torch.full(
                (heads, query_tokens), -math.inf, dtype=torch.float32, device=device
            )
            running_sum = torch.zeros_like(running_max)
            running_out = torch.zeros(
                (heads, query_tokens, head_dim), dtype=torch.float32, device=device
            )
            q_positions = torch.arange(
                q_local_start,
                q_local_start + query_tokens,
                device=device,
            )

            for kv_tile_start in range(k_start, k_stop, kv_chunk_tokens):
                kv_tile_stop = min(kv_tile_start + kv_chunk_tokens, k_stop)
                kv_local_start = kv_tile_start - k_start
                k = k_cpu[kv_tile_start:kv_tile_stop].to(device=device)
                v = v_cpu[kv_tile_start:kv_tile_stop].to(device=device)
                if group_size != 1:
                    k = k.repeat_interleave(group_size, dim=1)
                    v = v.repeat_interleave(group_size, dim=1)
                k = k.transpose(0, 1).float()
                v = v.transpose(0, 1).float()
                scores = torch.matmul(q, k.transpose(-1, -2)).mul_(scale)
                if causal:
                    k_positions = torch.arange(
                        kv_local_start,
                        kv_local_start + k.shape[1],
                        device=device,
                    )
                    valid = k_positions.unsqueeze(0) <= (
                        q_positions.unsqueeze(1) + causal_shift
                    )
                    scores.masked_fill_(~valid.unsqueeze(0), -math.inf)

                tile_max = scores.amax(dim=-1)
                merged_max = torch.maximum(running_max, tile_max)
                valid_rows = torch.isfinite(merged_max)
                old_scale = torch.where(
                    valid_rows,
                    torch.exp(running_max - merged_max),
                    torch.ones_like(merged_max),
                )
                probabilities = torch.where(
                    torch.isfinite(scores),
                    torch.exp(scores - merged_max.unsqueeze(-1)),
                    torch.zeros_like(scores),
                )
                running_sum.mul_(old_scale).add_(probabilities.sum(dim=-1))
                running_out.mul_(old_scale.unsqueeze(-1)).add_(
                    torch.matmul(probabilities, v)
                )
                running_max = torch.where(valid_rows, merged_max, running_max)

            normalized = torch.where(
                running_sum.unsqueeze(-1) > 0,
                running_out / running_sum.unsqueeze(-1),
                torch.zeros_like(running_out),
            )
            out[q_tile_start:q_tile_stop].copy_(
                normalized.transpose(0, 1).to(dtype=q_cpu.dtype, device="cpu")
            )
    return out
