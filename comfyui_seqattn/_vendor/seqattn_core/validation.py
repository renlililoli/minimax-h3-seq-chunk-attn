from __future__ import annotations

import torch


def validate_host_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[list[int], list[int]]:
    if any(t.device.type != "cpu" for t in (q, k, v)):
        raise ValueError("seqattn expects CPU-backed q, k, and v tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must use packed [tokens, heads, head_dim] layout")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes, got {k.shape} and {v.shape}")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q and k/v must use the same head_dim")
    if q.shape[1] % k.shape[1]:
        raise ValueError("the number of query heads must be divisible by the number of KV heads")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must use the same dtype")
    if not all(t.is_contiguous() for t in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous")

    def bounds(name: str, cu: torch.Tensor, total: int) -> list[int]:
        if cu.device.type != "cpu" or cu.ndim != 1 or cu.numel() < 2:
            raise ValueError(f"{name} must be a one-dimensional CPU tensor")
        result = cu.to(dtype=torch.int64).tolist()
        if result[0] != 0 or result[-1] != total:
            raise ValueError(f"{name} must span [0, {total}], got {result}")
        if any(stop < start for start, stop in zip(result[:-1], result[1:])):
            raise ValueError(f"{name} must be non-decreasing")
        return result

    q_bounds = bounds("cu_seqlens_q", cu_seqlens_q, q.shape[0])
    k_bounds = bounds("cu_seqlens_k", cu_seqlens_k, k.shape[0])
    if len(q_bounds) != len(k_bounds):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")
    for index, ((qs, qe), (ks, ke)) in enumerate(
        zip(zip(q_bounds[:-1], q_bounds[1:]), zip(k_bounds[:-1], k_bounds[1:]))
    ):
        if qe > qs and ke == ks:
            raise ValueError(f"sequence {index} has queries but no keys")
    return q_bounds, k_bounds


def require_pinned_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not all(t.is_pinned() for t in (q, k, v)):
        raise ValueError(
            "asynchronous streaming requires pinned q, k, and v; call pin_memory() "
            "or set require_pinned=False for the synchronous fallback"
        )
