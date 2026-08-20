from __future__ import annotations

import torch

from ..config import ProjectionPipelineConfig


class ProjectionWorkspace:
    def __init__(
        self,
        *,
        hidden_features: int,
        dtype: torch.dtype,
        device: torch.device,
        config: ProjectionPipelineConfig,
    ) -> None:
        self.hidden_features = hidden_features
        self.hidden = [
            torch.empty(
                (config.projection_chunk_tokens, hidden_features),
                dtype=dtype,
                device=device,
            )
            for _ in range(config.num_projection_buffers)
        ]
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.input_ready = [torch.cuda.Event() for _ in self.hidden]
        self.projected_ready = [torch.cuda.Event() for _ in self.hidden]
        self.copy_done = [torch.cuda.Event() for _ in self.hidden]
        self.busy = [False for _ in self.hidden]
        self.keepalive: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.hidden
        ]


__all__ = ["ProjectionWorkspace"]
