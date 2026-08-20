from __future__ import annotations

import torch

from ..kernels import triton_is_available


def resolve_backend(
    name: str, dtype: torch.dtype, device: torch.device, *, head_dim: int | None = None
) -> str:
    if name == "auto":
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            backend = "triton" if triton_is_available() else "reference"
        else:
            backend = "reference"
    else:
        backend = name
        if backend == "triton":
            if device.type != "cuda":
                raise ValueError("the Triton backend requires a CUDA device")
            if dtype not in {torch.float16, torch.bfloat16}:
                raise ValueError("the Triton backend requires float16 or bfloat16 inputs")
            if not triton_is_available():
                raise RuntimeError("the Triton backend is not available")
    if backend == "triton" and head_dim is not None and head_dim < 16:
        raise ValueError(
            "the Triton backend requires head_dim >= 16 (tl.dot needs BLOCK_D >= 16)"
        )
    return backend


__all__ = ["resolve_backend"]
