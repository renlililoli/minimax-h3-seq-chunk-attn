from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamingAttentionConfig:
    """Execution policy for CPU-backed exact attention.

    ``workspace_budget_bytes`` covers operator-owned CUDA buffers only.  It is
    intentionally distinct from a whole-process VRAM limit, because CUDA
    context, model weights, and allocations owned by the caller remain outside
    this package.
    """

    workspace_budget_bytes: int | None = None
    q_chunk_tokens: int | None = None
    kv_chunk_tokens: int | None = None
    num_kv_buffers: int = 2
    num_output_buffers: int = 1
    output_mode: str = "host"
    # Leaving all launch parameters unset enables a conservative device/shape
    # preset. Setting any one parameter switches to the portable 64x64/4/2
    # baseline and applies the supplied values as overrides.
    block_m: int | None = None
    block_n: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None
    backend: str = "auto"
    require_pinned: bool = True
    pin_output: bool = True
    enable_nvtx: bool = False

    def validate(self) -> None:
        if self.workspace_budget_bytes is not None and self.workspace_budget_bytes <= 0:
            raise ValueError("workspace_budget_bytes must be positive")
        if self.q_chunk_tokens is not None and self.q_chunk_tokens <= 0:
            raise ValueError("q_chunk_tokens must be positive")
        if self.kv_chunk_tokens is not None and self.kv_chunk_tokens <= 0:
            raise ValueError("kv_chunk_tokens must be positive")
        if self.num_kv_buffers not in {1, 2, 3}:
            raise ValueError("num_kv_buffers must be 1, 2, or 3")
        if self.num_output_buffers not in {1, 2}:
            raise ValueError("num_output_buffers must be 1 or 2")
        if self.output_mode not in {"host", "device_consumer"}:
            raise ValueError("output_mode must be 'host' or 'device_consumer'")
        if self.block_m is not None and self.block_m not in {16, 32, 64, 128}:
            raise ValueError("block_m must be one of 16, 32, 64, 128")
        if self.block_n is not None and self.block_n not in {16, 32, 64, 128}:
            raise ValueError("block_n must be one of 16, 32, 64, 128")
        if self.num_warps is not None and self.num_warps not in {2, 4, 8}:
            raise ValueError("num_warps must be 2, 4, or 8")
        if self.num_stages is not None and self.num_stages not in {1, 2, 3, 4}:
            raise ValueError("num_stages must be within [1, 4]")
        if self.backend not in {"auto", "triton", "reference"}:
            raise ValueError(f"unsupported backend: {self.backend}")


@dataclass(frozen=True)
class PagedAttentionConfig:
    """Execution and memory policy for paged CPU/NVMe attention.

    The host budget covers allocations owned by the operator. Caller-owned
    tensors passed through ``MemoryPageSource`` are deliberately excluded.
    """

    attention: StreamingAttentionConfig = field(
        default_factory=lambda: StreamingAttentionConfig(
            workspace_budget_bytes=2 * 2**30,
            output_mode="host",
        )
    )
    host_memory_budget_bytes: int = 8 * 2**30
    pinned_staging_budget_bytes: int = 1 * 2**30
    direct_io_bounce_budget_bytes: int = 512 * 2**20
    metadata_margin_bytes: int = 128 * 2**20
    page_target_bytes: int = 16 * 2**20
    io_workers: int = 4
    io_queue_depth: int = 4
    num_output_buffers: int = 2
    cache_hot_fraction: float = 0.8
    direct_io: bool = True
    kv_storage_dtype: str = "bf16"
    quant_group_tokens: int = 64

    def validate(self) -> None:
        self.attention.validate()
        if self.host_memory_budget_bytes <= 0:
            raise ValueError("host_memory_budget_bytes must be positive")
        for name, value in (
            ("pinned_staging_budget_bytes", self.pinned_staging_budget_bytes),
            ("direct_io_bounce_budget_bytes", self.direct_io_bounce_budget_bytes),
            ("metadata_margin_bytes", self.metadata_margin_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        reserved = (
            self.pinned_staging_budget_bytes
            + self.direct_io_bounce_budget_bytes
            + self.metadata_margin_bytes
        )
        if reserved >= self.host_memory_budget_bytes:
            raise ValueError(
                "pinned, bounce, and metadata reservations must leave room for the DRAM cache"
            )
        if self.page_target_bytes <= 0:
            raise ValueError("page_target_bytes must be positive")
        if self.io_workers <= 0:
            raise ValueError("io_workers must be positive")
        if self.io_queue_depth <= 0:
            raise ValueError("io_queue_depth must be positive")
        if self.num_output_buffers not in {1, 2}:
            raise ValueError("num_output_buffers must be 1 or 2")
        if not 0.0 <= self.cache_hot_fraction <= 1.0:
            raise ValueError("cache_hot_fraction must be within [0, 1]")
        if self.kv_storage_dtype not in {"bf16", "fp16", "fp32", "int8"}:
            raise ValueError("kv_storage_dtype must be bf16, fp16, fp32, or int8")
        if self.quant_group_tokens != 64:
            raise ValueError("V1 INT8 storage requires quant_group_tokens=64")


@dataclass(frozen=True)
class ProjectionPipelineConfig:
    """Execution policy for CPU-hidden -> QKV -> attention -> output projection.

    Exact self-attention has a global K/V readiness barrier.  Projection chunks
    are pipelined internally, then attention and output projection run as a
    second pipeline without materializing raw attention output on the CPU.
    """

    projection_chunk_tokens: int = 2048
    num_projection_buffers: int = 2
    require_pinned_hidden: bool = True
    pin_qkv: bool = True
    pin_output: bool = True
    enable_nvtx: bool = False

    def validate(self) -> None:
        if self.projection_chunk_tokens <= 0:
            raise ValueError("projection_chunk_tokens must be positive")
        if self.num_projection_buffers not in {1, 2, 3}:
            raise ValueError("num_projection_buffers must be 1, 2, or 3")
