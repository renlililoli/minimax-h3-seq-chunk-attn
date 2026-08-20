from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only development hosts
    triton = None
    tl = None


def triton_is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _streaming_attention_update_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        k_scale_ptr,
        v_scale_ptr,
        max_ptr,
        sum_ptr,
        acc_ptr,
        q_tokens,
        kv_tokens,
        q_local_offset,
        kv_local_offset,
        causal_shift,
        scale_log2,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ksg,
        stride_ksh,
        stride_vsg,
        stride_vsh,
        storage_token_offset,
        stride_st,
        stride_sh,
        stride_at,
        stride_ah,
        stride_ad,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        INITIALIZE: tl.constexpr,
        KV_QUANTIZED: tl.constexpr,
        QUANT_GROUP_TOKENS: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        query_head = tl.program_id(1)
        kv_head = query_head // GROUP_SIZE
        offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_d = tl.arange(0, BLOCK_D)
        q_mask = offsets_m < q_tokens
        d_mask = offsets_d < HEAD_DIM

        q_offsets = (
            offsets_m[:, None] * stride_qt
            + query_head * stride_qh
            + offsets_d[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

        state_offsets = offsets_m * stride_st + query_head * stride_sh
        acc_offsets = (
            offsets_m[:, None] * stride_at
            + query_head * stride_ah
            + offsets_d[None, :] * stride_ad
        )
        if INITIALIZE:
            running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
            running_sum = tl.zeros((BLOCK_M,), tl.float32)
            accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        else:
            running_max = tl.load(max_ptr + state_offsets, mask=q_mask, other=-float("inf"))
            running_sum = tl.load(sum_ptr + state_offsets, mask=q_mask, other=0.0)
            accumulator = tl.load(
                acc_ptr + acc_offsets,
                mask=q_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

        for start_n in range(0, kv_tokens, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = offsets_n < kv_tokens
            k_offsets = (
                offsets_n[:, None] * stride_kt
                + kv_head * stride_kh
                + offsets_d[None, :] * stride_kd
            )
            v_offsets = (
                offsets_n[:, None] * stride_vt
                + kv_head * stride_vh
                + offsets_d[None, :] * stride_vd
            )
            k = tl.load(k_ptr + k_offsets, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
            if KV_QUANTIZED:
                scale_group = (storage_token_offset + offsets_n) // QUANT_GROUP_TOKENS
                k_scale_offsets = scale_group * stride_ksg + kv_head * stride_ksh
                v_scale_offsets = scale_group * stride_vsg + kv_head * stride_vsh
                k_scale = tl.load(k_scale_ptr + k_scale_offsets, mask=kv_mask, other=1.0)
                v_scale = tl.load(v_scale_ptr + v_scale_offsets, mask=kv_mask, other=1.0)
                k = (k.to(tl.float32) * k_scale[:, None]).to(q.dtype)
                v = (v.to(tl.float32) * v_scale[:, None]).to(q.dtype)
            logits = tl.dot(q, tl.trans(k)) * scale_log2
            valid = q_mask[:, None] & kv_mask[None, :]
            if CAUSAL:
                q_positions = q_local_offset + offsets_m
                k_positions = kv_local_offset + offsets_n
                valid &= k_positions[None, :] <= (
                    q_positions[:, None] + causal_shift
                )
            logits = tl.where(valid, logits, -float("inf"))

            tile_max = tl.max(logits, axis=1)
            merged_max = tl.maximum(running_max, tile_max)
            row_valid = merged_max != -float("inf")
            alpha = tl.where(
                row_valid,
                tl.exp2(running_max - merged_max),
                1.0,
            )
            probabilities = tl.where(
                valid,
                tl.exp2(logits - merged_max[:, None]),
                0.0,
            )
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(q.dtype), v
            )
            running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
            running_max = tl.where(row_valid, merged_max, running_max)

        tl.store(max_ptr + state_offsets, running_max, mask=q_mask)
        tl.store(sum_ptr + state_offsets, running_sum, mask=q_mask)
        tl.store(
            acc_ptr + acc_offsets,
            accumulator,
            mask=q_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _finalize_attention_kernel(
        acc_ptr,
        sum_ptr,
        out_ptr,
        q_tokens,
        stride_at,
        stride_ah,
        stride_ad,
        stride_st,
        stride_sh,
        stride_ot,
        stride_oh,
        stride_od,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dim_offsets = tl.arange(0, BLOCK_D)
        total_rows = q_tokens * HEADS
        row_mask = row_offsets < total_rows
        token_offsets = row_offsets // HEADS
        head_offsets = row_offsets - token_offsets * HEADS
        state_offsets = token_offsets * stride_st + head_offsets * stride_sh
        value_offsets = (
            token_offsets[:, None] * stride_at
            + head_offsets[:, None] * stride_ah
            + dim_offsets[None, :] * stride_ad
        )
        out_offsets = (
            token_offsets[:, None] * stride_ot
            + head_offsets[:, None] * stride_oh
            + dim_offsets[None, :] * stride_od
        )
        value_mask = row_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM)
        normalizer = tl.load(sum_ptr + state_offsets, mask=row_mask, other=0.0)
        accumulator = tl.load(acc_ptr + value_offsets, mask=value_mask, other=0.0)
        output = tl.where(
            normalizer[:, None] > 0,
            accumulator / normalizer[:, None],
            0.0,
        )
        tl.store(out_ptr + out_offsets, output, mask=value_mask)


def update_attention_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    accumulator: torch.Tensor,
    *,
    q_tokens: int,
    kv_tokens: int,
    q_local_offset: int,
    kv_local_offset: int,
    causal_shift: int,
    softmax_scale: float,
    causal: bool,
    initialize: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> None:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    head_dim = q.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(q_tokens, block_m), q.shape[1])
    # KV_QUANTIZED=False specializations never read the scale pointers, their
    # strides, or storage_token_offset, but Triton requires every positional
    # argument at each call site. Passing k/v and zeros keeps one kernel
    # signature for both variants.
    _streaming_attention_update_kernel[grid](
        q,
        k,
        v,
        k,
        v,
        running_max,
        running_sum,
        accumulator,
        q_tokens,
        kv_tokens,
        q_local_offset,
        kv_local_offset,
        causal_shift,
        softmax_scale * 1.4426950408889634,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        0,
        0,
        0,
        0,
        0,
        *running_max.stride(),
        *accumulator.stride(),
        GROUP_SIZE=q.shape[1] // k.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        INITIALIZE=initialize,
        KV_QUANTIZED=False,
        QUANT_GROUP_TOKENS=64,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def update_attention_state_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    accumulator: torch.Tensor,
    *,
    q_tokens: int,
    kv_tokens: int,
    q_local_offset: int,
    kv_local_offset: int,
    storage_token_offset: int,
    causal_shift: int,
    softmax_scale: float,
    causal: bool,
    initialize: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    quant_group_tokens: int = 64,
) -> None:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    if k.dtype != torch.int8 or v.dtype != torch.int8:
        raise ValueError("quantized K/V buffers must use int8")
    if k_scales.dtype != torch.float16 or v_scales.dtype != torch.float16:
        raise ValueError("quantized K/V scales must use float16")
    head_dim = q.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(q_tokens, block_m), q.shape[1])
    _streaming_attention_update_kernel[grid](
        q,
        k,
        v,
        k_scales,
        v_scales,
        running_max,
        running_sum,
        accumulator,
        q_tokens,
        kv_tokens,
        q_local_offset,
        kv_local_offset,
        causal_shift,
        softmax_scale * 1.4426950408889634,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *k_scales.stride(),
        *v_scales.stride(),
        storage_token_offset,
        *running_max.stride(),
        *accumulator.stride(),
        GROUP_SIZE=q.shape[1] // k.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        INITIALIZE=initialize,
        KV_QUANTIZED=True,
        QUANT_GROUP_TOKENS=quant_group_tokens,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def finalize_attention(
    accumulator: torch.Tensor,
    running_sum: torch.Tensor,
    output: torch.Tensor,
    *,
    q_tokens: int,
) -> None:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    head_dim = output.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    block_rows = 8
    grid = (triton.cdiv(q_tokens * output.shape[1], block_rows),)
    _finalize_attention_kernel[grid](
        accumulator,
        running_sum,
        output,
        q_tokens,
        *accumulator.stride(),
        *running_sum.stride(),
        *output.stride(),
        HEADS=output.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=4,
    )
