from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from ._vendor.seqattn_core import (
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
)


@dataclass(frozen=True)
class SeqAttnSettings:
    activation_workspace_mib: int = 1024
    kv_chunk_tokens: int = 4096
    planner_mode: str = "fit"
    projection_chunk_tokens: int = 2048

    def validate(self) -> None:
        if self.activation_workspace_mib <= 0:
            raise ValueError("activation_workspace_mib must be positive")
        if self.kv_chunk_tokens <= 0:
            raise ValueError("kv_chunk_tokens must be positive")
        if self.planner_mode != "fit":
            raise ValueError("only planner_mode='fit' is supported")
        if self.projection_chunk_tokens <= 0:
            raise ValueError("projection_chunk_tokens must be positive")


class SeqAttnRuntime:
    """Per-patched-model runner and refined-conditioning cache.

    ProjectedAttentionRunner is single-flight, and ComfyUI can retain a patched
    MODEL across queues. The lock covers one complete H3 forward while the
    caches reuse expensive pinned allocations and prompt-only refiner work
    across denoising steps.
    """

    def __init__(self, settings: SeqAttnSettings):
        settings.validate()
        self.settings = settings
        self.lock = threading.RLock()
        self._runners: dict[tuple, ProjectedAttentionRunner] = {}
        self._refined_conditioning_key: tuple | None = None
        self._refined_conditioning: torch.Tensor | None = None
        self._refined_conditioning_hits = 0
        self._refined_conditioning_misses = 0
        self._refined_conditioning_stores = 0
        self._refined_conditioning_bypasses = 0

    def clone(self) -> "SeqAttnRuntime":
        return SeqAttnRuntime(self.settings)

    @property
    def cache_size(self) -> int:
        return len(self._runners)

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

    def clear(self) -> None:
        with self.lock:
            self._runners.clear()
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
                return self._refined_conditioning
            self._refined_conditioning_misses += 1
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

    def record_refined_conditioning_bypass(self) -> None:
        with self.lock:
            self._refined_conditioning_bypasses += 1

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
            workspace_budget_bytes=self.settings.activation_workspace_mib * 2**20,
            kv_chunk_tokens=self.settings.kv_chunk_tokens,
            output_mode="device_consumer",
            backend="auto",
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
                projection_chunk_tokens=self.settings.projection_chunk_tokens,
                require_pinned_hidden=True,
                pin_qkv=True,
                pin_output=True,
            ),
        )
        self._runners[key] = runner
        return runner


__all__ = ["SeqAttnRuntime", "SeqAttnSettings"]
