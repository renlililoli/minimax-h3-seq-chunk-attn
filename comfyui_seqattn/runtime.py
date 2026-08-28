from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import torch
from seqattn_core import (
    H3MaterializedRunner,
    H3RecomputeRunner,
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    RecomputedAttentionRunner,
    StreamingAttentionConfig,
    build_plan,
)

from .config import load_attention_stage_config


@dataclass(frozen=True)
class SeqAttnSettings:
    execution_mode: str = "materialized"
    q_chunk_tokens: int = 5760
    kv_chunk_tokens: int = 4096
    qkv_tile_tokens: int = 4096
    mlp_tile_tokens: int = 4096

    @classmethod
    def from_config(
        cls,
        *,
        q_chunk_tokens: int,
        kv_chunk_tokens: int,
    ) -> SeqAttnSettings:
        config = load_attention_stage_config("minimax_h3")
        return cls(
            execution_mode=config.execution_mode,
            q_chunk_tokens=int(q_chunk_tokens),
            kv_chunk_tokens=int(kv_chunk_tokens),
            qkv_tile_tokens=config.qkv_tile_tokens,
            mlp_tile_tokens=config.mlp_tile_tokens,
        )

    def validate(self) -> None:
        if self.execution_mode not in {"materialized", "recompute"}:
            raise ValueError("execution_mode must be 'materialized' or 'recompute'")
        for name, value in (
            ("q_chunk_tokens", self.q_chunk_tokens),
            ("kv_chunk_tokens", self.kv_chunk_tokens),
            ("qkv_tile_tokens", self.qkv_tile_tokens),
            ("mlp_tile_tokens", self.mlp_tile_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class _SeqAttnRuntimeMetrics:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_refined_conditioning_cache_stats: dict[str, int] | None = None
        self.refined_conditioning_forward_calls = 0
        self.refined_conditioning_applicable_calls = 0
        self.refined_conditioning_hits = 0
        self.refined_conditioning_misses = 0
        self.refined_conditioning_stores = 0
        self.refined_conditioning_bypasses = 0
        self.refined_conditioning_peak_host_bytes = 0

    def record_refined_conditioning_cache_stats(self, stats: dict[str, int]) -> None:
        with self.lock:
            self.last_refined_conditioning_cache_stats = dict(stats)

    def last_cache_stats(self) -> dict[str, int] | None:
        with self.lock:
            stats = self.last_refined_conditioning_cache_stats
            return None if stats is None else dict(stats)

    def record_forward(self, applicable: bool) -> None:
        with self.lock:
            self.refined_conditioning_forward_calls += 1
            self.refined_conditioning_applicable_calls += int(applicable)

    def record_cache_event(self, event: str, host_bytes: int = 0) -> None:
        with self.lock:
            attribute = f"refined_conditioning_{event}"
            setattr(self, attribute, getattr(self, attribute) + 1)
            self.refined_conditioning_peak_host_bytes = max(
                self.refined_conditioning_peak_host_bytes, int(host_bytes)
            )

    def lifetime_cache_stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "forward_calls": self.refined_conditioning_forward_calls,
                "applicable_calls": self.refined_conditioning_applicable_calls,
                "passthrough_calls": (
                    self.refined_conditioning_forward_calls
                    - self.refined_conditioning_applicable_calls
                ),
                "hits": self.refined_conditioning_hits,
                "misses": self.refined_conditioning_misses,
                "stores": self.refined_conditioning_stores,
                "bypasses": self.refined_conditioning_bypasses,
                "peak_host_bytes": self.refined_conditioning_peak_host_bytes,
            }


class SeqAttnRuntime:
    """Per-patched-model runner and refined-conditioning cache.

    ProjectedAttentionRunner is single-flight, and ComfyUI can retain a patched
    MODEL across queues. The lock covers one complete H3 forward while the
    caches reuse expensive pinned allocations and prompt-only refiner work
    across denoising steps.
    """

    def __init__(
        self,
        settings: SeqAttnSettings,
        *,
        _metrics: _SeqAttnRuntimeMetrics | None = None,
        weight_schedule_records: list[dict] | None = None,
        weight_schedule_lock: threading.RLock | None = None,
    ):
        settings.validate()
        self.settings = settings
        self.lock = threading.RLock()
        self._runners: dict[tuple, ProjectedAttentionRunner] = {}
        self._dit_runners: dict[tuple, H3MaterializedRunner | H3RecomputeRunner] = {}
        self._refined_conditioning_key: tuple | None = None
        self._refined_conditioning: torch.Tensor | None = None
        self._refined_conditioning_hits = 0
        self._refined_conditioning_misses = 0
        self._refined_conditioning_stores = 0
        self._refined_conditioning_bypasses = 0
        self._metrics = _metrics or _SeqAttnRuntimeMetrics()
        self._weight_schedule_records = (
            [] if weight_schedule_records is None else weight_schedule_records
        )
        self._weight_schedule_lock = (
            threading.RLock()
            if weight_schedule_lock is None
            else weight_schedule_lock
        )

    def clone(self) -> "SeqAttnRuntime":
        return SeqAttnRuntime(
            self.settings,
            _metrics=self._metrics,
            weight_schedule_records=self._weight_schedule_records,
            weight_schedule_lock=self._weight_schedule_lock,
        )

    @property
    def cache_size(self) -> int:
        return len(self._runners) + len(self._dit_runners)

    @property
    def refined_conditioning_cache_stats(self) -> dict[str, int]:
        with self.lock:
            cached = self._refined_conditioning
            return {
                "hits": self._refined_conditioning_hits,
                "misses": self._refined_conditioning_misses,
                "stores": self._refined_conditioning_stores,
                "bypasses": self._refined_conditioning_bypasses,
                "entries": int(cached is not None),
                "host_bytes": (
                    cached.numel() * cached.element_size()
                    if cached is not None
                    else 0
                ),
            }

    @property
    def last_refined_conditioning_cache_stats(self) -> dict[str, int] | None:
        return self._metrics.last_cache_stats()

    @property
    def lifetime_refined_conditioning_cache_stats(self) -> dict[str, int]:
        return self._metrics.lifetime_cache_stats()

    def record_refined_conditioning_forward(self, applicable: bool) -> None:
        self._metrics.record_forward(applicable)

    @property
    def weight_schedule_records(self) -> list[dict]:
        with self._weight_schedule_lock:
            return list(self._weight_schedule_records)

    def record_weight_schedule(self, record: dict) -> None:
        with self._weight_schedule_lock:
            self._weight_schedule_records.append(record)

    def clear(self) -> None:
        with self.lock:
            stats = self.refined_conditioning_cache_stats
            if any(stats[key] for key in ("hits", "misses", "stores", "bypasses")):
                self._metrics.record_refined_conditioning_cache_stats(stats)
            self._runners.clear()
            self._dit_runners.clear()
            self._refined_conditioning_key = None
            self._refined_conditioning = None
            self._refined_conditioning_hits = 0
            self._refined_conditioning_misses = 0
            self._refined_conditioning_stores = 0
            self._refined_conditioning_bypasses = 0

    def refined_conditioning_for(self, key: tuple) -> torch.Tensor | None:
        with self.lock:
            if (
                self._refined_conditioning is not None
                and self._refined_conditioning_key == key
            ):
                self._refined_conditioning_hits += 1
                self._metrics.record_cache_event("hits")
                return self._refined_conditioning
            self._refined_conditioning_misses += 1
            self._metrics.record_cache_event("misses")
            return None

    def store_refined_conditioning(
        self, key: tuple, conditioning: torch.Tensor
    ) -> None:
        if conditioning.device.type != "cpu":
            raise ValueError("refined conditioning cache must be CPU-resident")
        if conditioning.dtype != torch.bfloat16:
            raise ValueError("refined conditioning cache must use BF16")
        if not conditioning.is_pinned():
            raise ValueError("refined conditioning cache must use pinned memory")
        if not conditioning.is_contiguous():
            raise ValueError("refined conditioning cache must be contiguous")
        with self.lock:
            self._refined_conditioning_key = key
            self._refined_conditioning = conditioning
            self._refined_conditioning_stores += 1
            self._metrics.record_cache_event(
                "stores", conditioning.numel() * conditioning.element_size()
            )

    def record_refined_conditioning_bypass(self) -> None:
        with self.lock:
            self._refined_conditioning_bypasses += 1
            self._metrics.record_cache_event("bypasses")

    def runner_for(
        self,
        *,
        tokens: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> ProjectedAttentionRunner:
        device = torch.device(device)
        key = (
            tokens,
            heads,
            head_dim,
            dtype,
            device.type,
            device.index,
            self.settings,
        )
        runner = self._runners.get(key)
        if runner is not None:
            return runner

        attention_config = StreamingAttentionConfig(
            q_chunk_tokens=self.settings.q_chunk_tokens,
            kv_chunk_tokens=self.settings.kv_chunk_tokens,
            output_mode="device_consumer",
            backend=None,
            require_pinned=True,
            pin_output=True,
        )
        plan = build_plan(
            q_heads=heads,
            kv_heads=heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            max_q_tokens=tokens,
            max_kv_tokens=tokens,
            config=attention_config,
        )
        runner = ProjectedAttentionRunner(
            plan,
            attention_config=attention_config,
            pipeline_config=ProjectionPipelineConfig(
                projection_chunk_tokens=self.settings.qkv_tile_tokens,
                require_pinned_hidden=True,
                pin_qkv=True,
                pin_output=True,
            ),
        )
        self._runners[key] = runner
        return runner

    def dit_runner_for(
        self,
        *,
        tokens: int,
        hidden_features: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> H3MaterializedRunner | H3RecomputeRunner:
        device = torch.device(device)
        enable_nvtx = os.environ.get("SEQATTN_ENABLE_NVTX") == "1"
        key = (
            tokens,
            hidden_features,
            heads,
            head_dim,
            dtype,
            device.type,
            device.index,
            self.settings,
        )
        runner = self._dit_runners.get(key)
        if runner is not None:
            return runner

        attention_config = StreamingAttentionConfig(
            q_chunk_tokens=self.settings.q_chunk_tokens,
            kv_chunk_tokens=self.settings.kv_chunk_tokens,
            output_mode="device_consumer",
            backend=None,
            require_pinned=True,
            pin_output=True,
            enable_nvtx=enable_nvtx,
        )
        plan = build_plan(
            q_heads=heads,
            kv_heads=heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            max_q_tokens=tokens,
            max_kv_tokens=tokens,
            config=attention_config,
        )
        if self.settings.execution_mode == "materialized":
            projected = ProjectedAttentionRunner(
                plan,
                attention_config=attention_config,
                pipeline_config=ProjectionPipelineConfig(
                    projection_chunk_tokens=self.settings.qkv_tile_tokens,
                    require_pinned_hidden=True,
                    pin_qkv=True,
                    pin_output=True,
                    enable_nvtx=enable_nvtx,
                ),
            )
            runner = H3MaterializedRunner(
                projected,
                hidden_features=hidden_features,
                mlp_chunk_tokens=self.settings.mlp_tile_tokens,
                num_final_output_buffers=2,
            )
        else:
            recomputed = RecomputedAttentionRunner(
                plan,
                hidden_features=hidden_features,
                attention_config=attention_config,
                require_pinned_hidden=True,
                enable_nvtx=enable_nvtx,
            )
            runner = H3RecomputeRunner(
                recomputed,
                mlp_chunk_tokens=self.settings.mlp_tile_tokens,
                num_final_output_buffers=2,
            )
        self._dit_runners[key] = runner
        return runner


__all__ = ["SeqAttnRuntime", "SeqAttnSettings"]
