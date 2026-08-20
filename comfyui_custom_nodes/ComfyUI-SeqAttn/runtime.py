from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from seqattn import (
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
)


@dataclass(frozen=True)
class SeqAttnSettings:
    activation_workspace_mib: int = 4096
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
    """Per-patched-model runner cache.

    ProjectedAttentionRunner is single-flight, and ComfyUI can retain a patched
    MODEL across queues. The lock covers one complete H3 forward while the
    cache reuses its expensive pinned Q/K/V allocations across denoising steps.
    """

    def __init__(self, settings: SeqAttnSettings):
        settings.validate()
        self.settings = settings
        self.lock = threading.RLock()
        self._runners: dict[tuple, ProjectedAttentionRunner] = {}
        self.profile_enabled = False
        self.profile_records: list[dict] = []

    def clone(self) -> "SeqAttnRuntime":
        clone = SeqAttnRuntime(self.settings)
        clone.profile_enabled = self.profile_enabled
        clone.profile_records = self.profile_records
        return clone

    def enable_profile(self) -> None:
        self.profile_enabled = True
        self.profile_records.clear()

    def record_profile(self, record: dict) -> None:
        if self.profile_enabled:
            self.profile_records.append(record)

    @property
    def cache_size(self) -> int:
        return len(self._runners)

    def clear(self) -> None:
        self._runners.clear()

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
            enable_nvtx=self.profile_enabled,
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
                enable_nvtx=self.profile_enabled,
            ),
        )
        self._runners[key] = runner
        return runner


__all__ = ["SeqAttnRuntime", "SeqAttnSettings"]
