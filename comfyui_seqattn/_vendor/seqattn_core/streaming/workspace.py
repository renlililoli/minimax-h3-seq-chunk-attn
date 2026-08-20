from __future__ import annotations

import torch

from ..planner import AttentionPlan


class CudaWorkspace:
    def __init__(self, plan: AttentionPlan) -> None:
        device = plan.device
        self.q = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=plan.dtype,
            device=device,
        )
        self.k = [
            torch.empty(
                (plan.kv_chunk_tokens, plan.kv_heads, plan.head_dim),
                dtype=plan.dtype,
                device=device,
            )
            for _ in range(plan.num_kv_buffers)
        ]
        self.v = [torch.empty_like(tensor) for tensor in self.k]
        self.running_max = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads), dtype=torch.float32, device=device
        )
        self.running_sum = torch.empty_like(self.running_max)
        self.accumulator = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.output = (
            [torch.empty_like(self.q) for _ in range(plan.num_output_buffers)]
            if plan.output_mode == "host"
            else []
        )
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.q_ready = torch.cuda.Event()
        self.q_free = torch.cuda.Event()
        self.q_has_pending_compute = False
        self.kv_ready = [torch.cuda.Event() for _ in self.k]
        self.kv_free = [torch.cuda.Event() for _ in self.k]
        self.kv_has_pending_compute = [False for _ in self.k]
        self.output_ready = [torch.cuda.Event() for _ in range(plan.num_output_buffers)]
        self.output_free = [torch.cuda.Event() for _ in range(plan.num_output_buffers)]
        self.output_has_pending_copy = [False for _ in range(plan.num_output_buffers)]


__all__ = ["CudaWorkspace"]
