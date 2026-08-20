from collections.abc import Callable

import torch

QKVProjector = Callable[
    [torch.Tensor, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
OutputProjector = Callable[[torch.Tensor, int, int], torch.Tensor]


__all__ = ["OutputProjector", "QKVProjector"]
