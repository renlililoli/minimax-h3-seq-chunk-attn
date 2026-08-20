from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .config import StreamingAttentionConfig


def _align_down(value: int, alignment: int) -> int:
    return value - value % alignment


_PORTABLE_KERNEL = (64, 64, 4, 2)
_BLACKWELL_D128_KERNEL = (128, 64, 8, 3)


def _resolve_kernel_config(
    config: StreamingAttentionConfig,
    *,
    device: torch.device,
    head_dim: int,
    dtype: torch.dtype,
) -> StreamingAttentionConfig:
    values = (config.block_m, config.block_n, config.num_warps, config.num_stages)
    if any(value is not None for value in values):
        base = _PORTABLE_KERNEL
    else:
        base = _PORTABLE_KERNEL
        if (
            device.type == "cuda"
            and torch.cuda.is_available()
            and dtype in {torch.float16, torch.bfloat16}
            and head_dim == 128
        ):
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 12:
                base = _BLACKWELL_D128_KERNEL
    block_m, block_n, num_warps, num_stages = (
        default if value is None else value for value, default in zip(values, base)
    )
    return replace(
        config,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@dataclass(frozen=True)
class AttentionPlan:
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    max_q_tokens: int
    max_kv_tokens: int
    q_chunk_tokens: int
    kv_chunk_tokens: int
    num_kv_buffers: int
    num_output_buffers: int
    output_mode: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    estimated_workspace_bytes: int
    workspace_budget_bytes: int | None

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads


def estimate_workspace_bytes(
    *,
    q_tokens: int,
    kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_kv_buffers: int,
    num_output_buffers: int,
    output_mode: str = "host",
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    q_bytes = q_tokens * q_heads * head_dim * element_size
    # FP32 accumulator plus FP32 running max and normalizer.
    state_bytes = q_tokens * q_heads * (head_dim + 2) * 4
    output_bytes = (
        num_output_buffers * q_tokens * q_heads * head_dim * element_size
        if output_mode == "host"
        else 0
    )
    kv_bytes = num_kv_buffers * 2 * kv_tokens * kv_heads * head_dim * element_size
    # Events and stream objects are small; the fixed margin mainly absorbs
    # allocator alignment and Triton launch scratch without overstating usable
    # query capacity.
    fixed_margin = 32 * 2**20
    return q_bytes + state_bytes + output_bytes + kv_bytes + fixed_margin


def _largest_q_chunk_that_fits(
    *,
    max_q_tokens: int,
    kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    config: StreamingAttentionConfig,
) -> int | None:
    minimum_q = min(config.block_m, max_q_tokens)
    if config.workspace_budget_bytes is None:
        return max_q_tokens
    minimum = estimate_workspace_bytes(
        q_tokens=minimum_q,
        kv_tokens=kv_tokens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        num_kv_buffers=config.num_kv_buffers,
        num_output_buffers=config.num_output_buffers,
        output_mode=config.output_mode,
    )
    if minimum > config.workspace_budget_bytes:
        return None
    low = minimum_q
    high = max_q_tokens
    best = minimum_q
    while low <= high:
        candidate = _align_down((low + high) // 2, config.block_m)
        candidate = max(candidate, minimum_q)
        needed = estimate_workspace_bytes(
            q_tokens=candidate,
            kv_tokens=kv_tokens,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            num_kv_buffers=config.num_kv_buffers,
            num_output_buffers=config.num_output_buffers,
            output_mode=config.output_mode,
        )
        if needed <= config.workspace_budget_bytes:
            best = candidate
            low = candidate + config.block_m
        else:
            high = candidate - config.block_m
    return min(best, max_q_tokens)


def _candidate_cost(
    *,
    q_chunk: int,
    kv_chunk: int,
    max_q_tokens: int,
    max_kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> float:
    """Simple transfer/state/launch model used for joint chunk selection."""

    element_size = torch.empty((), dtype=dtype).element_size()
    q_passes = math.ceil(max_q_tokens / q_chunk)
    kv_tiles = math.ceil(max_kv_tokens / kv_chunk)
    kv_bytes = 2 * max_kv_tokens * kv_heads * head_dim * element_size * q_passes
    state_row_bytes = q_heads * (head_dim + 2) * 4
    state_bytes = 2 * max_q_tokens * state_row_bytes * kv_tiles
    launches = q_passes * (kv_tiles + 1)

    # 8K is a robust transfer/kernel overlap point on current PCIe systems.
    # Penalize both smaller launch-heavy tiles and larger tiles that reduce
    # ring-buffer overlap without hard-coding a shape-specific winner.
    tile_log2 = math.log2(kv_chunk / 8192)
    overlap_factor = 1.0 + 0.08 * max(-tile_log2, 0.0) + 0.65 * max(tile_log2, 0.0)
    return kv_bytes * overlap_factor / 24e9 + state_bytes / 1.2e12 + launches * 8e-6


def build_plan(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
    max_q_tokens: int,
    max_kv_tokens: int,
    config: StreamingAttentionConfig | None = None,
) -> AttentionPlan:
    config = StreamingAttentionConfig() if config is None else config
    config.validate()
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("q_heads must be a positive multiple of kv_heads")
    if head_dim <= 0 or head_dim > 256 or head_dim % 8:
        raise ValueError("head_dim must be a multiple of 8 and at most 256")
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if max_q_tokens <= 0 or max_kv_tokens <= 0:
        raise ValueError("max_q_tokens and max_kv_tokens must be positive")
    config = _resolve_kernel_config(
        config,
        device=device,
        head_dim=head_dim,
        dtype=dtype,
    )

    if config.kv_chunk_tokens is None:
        kv_candidates = sorted(
            {
                max(config.block_n, min(max_kv_tokens, candidate))
                for candidate in (4096, 8192, 16384)
            }
        )
    else:
        kv_candidates = [min(config.kv_chunk_tokens, max_kv_tokens)]

    candidates: list[tuple[float, int, int]] = []
    for kv_candidate in kv_candidates:
        kv_candidate = max(
            config.block_n,
            _align_down(kv_candidate, config.block_n),
        )
        kv_candidate = min(kv_candidate, max_kv_tokens)
        if config.q_chunk_tokens is None:
            q_candidate = _largest_q_chunk_that_fits(
                max_q_tokens=max_q_tokens,
                kv_tokens=kv_candidate,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                config=config,
            )
            if q_candidate is None:
                continue
        else:
            q_candidate = min(config.q_chunk_tokens, max_q_tokens)
            q_candidate = max(
                min(config.block_m, max_q_tokens),
                _align_down(q_candidate, config.block_m),
            )
        candidates.append(
            (
                _candidate_cost(
                    q_chunk=q_candidate,
                    kv_chunk=kv_candidate,
                    max_q_tokens=max_q_tokens,
                    max_kv_tokens=max_kv_tokens,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                ),
                q_candidate,
                kv_candidate,
            )
        )

    if not candidates:
        minimum_kv = min(kv_candidates)
        minimum = estimate_workspace_bytes(
            q_tokens=min(config.block_m, max_q_tokens),
            kv_tokens=minimum_kv,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            num_kv_buffers=config.num_kv_buffers,
            num_output_buffers=config.num_output_buffers,
            output_mode=config.output_mode,
        )
        raise ValueError(
            "workspace budget is too small for one query block and the requested KV buffers: "
            f"need at least {minimum / 2**20:.1f} MiB"
        )
    _, q_chunk, kv_chunk = min(candidates)

    estimated = estimate_workspace_bytes(
        q_tokens=q_chunk,
        kv_tokens=kv_chunk,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        num_kv_buffers=config.num_kv_buffers,
        num_output_buffers=config.num_output_buffers,
        output_mode=config.output_mode,
    )
    if config.workspace_budget_bytes is not None and estimated > config.workspace_budget_bytes:
        raise ValueError(
            f"requested chunks need {estimated / 2**20:.1f} MiB, exceeding the "
            f"{config.workspace_budget_bytes / 2**20:.1f} MiB workspace budget"
        )

    return AttentionPlan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        max_q_tokens=max_q_tokens,
        max_kv_tokens=max_kv_tokens,
        q_chunk_tokens=q_chunk,
        kv_chunk_tokens=kv_chunk,
        num_kv_buffers=config.num_kv_buffers,
        num_output_buffers=config.num_output_buffers,
        output_mode=config.output_mode,
        block_m=config.block_m,
        block_n=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        estimated_workspace_bytes=estimated,
        workspace_budget_bytes=config.workspace_budget_bytes,
    )
