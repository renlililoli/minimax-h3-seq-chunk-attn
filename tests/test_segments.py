from __future__ import annotations

import torch

from comfyui_seqattn import minimax_h3 as minimax


def test_chunk_modulation_matches_full_segments():
    source = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    shift = torch.tensor([[1.0] * 6, [2.0] * 6, [3.0] * 6])
    scale = torch.tensor([[0.1] * 6, [0.2] * 6, [0.3] * 6])
    segments = [(0, 2, 0), (2, 5, 1), (5, 8, 2)]

    expected = source.clone()
    for start, stop, row in segments:
        expected[start:stop].mul_(1.0 + scale[row]).add_(shift[row])

    actual = torch.empty_like(source)
    for start, stop in ((0, 3), (3, 7), (7, 8)):
        actual[start:stop] = minimax._modulate_tile(
            source[start:stop].clone(), shift, scale, segments, start, stop
        )
    torch.testing.assert_close(actual, expected)


def test_chunk_gate_matches_full_segments():
    residual = torch.arange(40, dtype=torch.float32).reshape(8, 5)
    update = torch.full_like(residual, 2.0)
    gate = torch.tensor([[0.5] * 5, [1.5] * 5])
    segments = [(0, 3, 0), (3, 8, 1)]

    expected = residual.clone()
    for start, stop, row in segments:
        expected[start:stop].addcmul_(update[start:stop], gate[row])

    actual = torch.empty_like(residual)
    for start, stop in ((0, 4), (4, 8)):
        actual[start:stop] = minimax._gate_tile(
            residual[start:stop].clone(),
            update[start:stop],
            gate,
            segments,
            start,
            stop,
        )
    torch.testing.assert_close(actual, expected)
