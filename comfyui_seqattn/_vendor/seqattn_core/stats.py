from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StreamingAttentionStats:
    backend: str = ""
    wall_seconds: float = 0.0
    q_chunks: int = 0
    kv_tiles: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    estimated_workspace_bytes: int = 0
    q_chunk_tokens: int = 0
    kv_chunk_tokens: int = 0
    max_resident_q_tokens: int = 0

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass
class ProjectedAttentionStats:
    backend: str = ""
    wall_seconds: float = 0.0
    projection_seconds: float = 0.0
    attention_output_seconds: float = 0.0
    projection_chunks: int = 0
    projection_hidden_h2d_bytes: int = 0
    projection_qkv_d2h_bytes: int = 0
    raw_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes: int = 0
    attention: StreamingAttentionStats = field(default_factory=StreamingAttentionStats)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["total_h2d_bytes"] = (
            self.projection_hidden_h2d_bytes + self.attention.h2d_bytes
        )
        result["total_d2h_bytes"] = (
            self.projection_qkv_d2h_bytes + self.attention.d2h_bytes
        )
        return result


@dataclass
class PagedAttentionStats:
    backend: str = ""
    kv_storage_dtype: str = ""
    wall_seconds: float = 0.0
    nvme_read_seconds: float = 0.0
    nvme_write_seconds: float = 0.0
    simulated_read_seconds: float = 0.0
    simulated_write_seconds: float = 0.0
    simulated_read_service_seconds: float = 0.0
    simulated_write_service_seconds: float = 0.0
    simulated_read_queue_seconds: float = 0.0
    simulated_write_queue_seconds: float = 0.0
    io_queue_wait_seconds: float = 0.0
    cache_lookup_seconds: float = 0.0
    pinned_copy_seconds: float = 0.0
    gpu_kernel_seconds: float = 0.0
    output_write_seconds: float = 0.0
    quantization_seconds: float = 0.0
    q_pages: int = 0
    q_chunks: int = 0
    kv_page_scans: int = 0
    kv_pages: int = 0
    state_spills: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    nvme_logical_read_bytes: int = 0
    nvme_physical_read_bytes: int = 0
    nvme_logical_write_bytes: int = 0
    nvme_physical_write_bytes: int = 0
    simulated_logical_read_bytes: int = 0
    simulated_physical_read_bytes: int = 0
    simulated_logical_write_bytes: int = 0
    simulated_physical_write_bytes: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    q_chunk_tokens: int = 0
    kv_chunk_tokens: int = 0
    estimated_workspace_bytes: int = 0
    operator_host_allocated_bytes: int = 0
    operator_host_peak_bytes: int = 0
    pinned_peak_bytes: int = 0
    direct_io_bounce_peak_bytes: int = 0
    dram_cache_peak_bytes: int = 0
    host_memory_budget_bytes: int = 0

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
