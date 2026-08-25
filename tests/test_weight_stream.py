from __future__ import annotations

import inspect

import pytest
import torch

from comfyui_seqattn import minimax_h3
from comfyui_seqattn import weight_stream as weight_stream_mod


def _plain_blocks(count: int) -> list[torch.nn.Module]:
    return [torch.nn.Sequential(torch.nn.Linear(2, 2)) for _ in range(count)]


def test_plain_blocks_require_dynamic_vbar(monkeypatch):
    monkeypatch.setattr(
        weight_stream_mod.comfy.ops,
        "cast_modules_with_vbar",
        lambda *_args, **_kwargs: pytest.fail("plain modules must not use VBAR"),
    )
    streamer = weight_stream_mod.BlockWeightStreamer(
        _plain_blocks(1),
        torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="no Dynamic VBAR weights"):
        streamer.prepare(0)


def test_weight_streamer_enforces_current_plus_next_limit(monkeypatch):
    class FakeVbar:
        def loaded_size(self):
            return 0

        def free_memory(self, _size):
            return 0

    class FakeModule:
        def __init__(self):
            self._v = (FakeVbar(), None, 0)

    blocks = _plain_blocks(3)
    modules = {id(block): FakeModule() for block in blocks}

    monkeypatch.setattr(
        weight_stream_mod,
        "_vbar_modules",
        lambda block: [modules[id(block)]],
    )
    monkeypatch.setattr(weight_stream_mod, "_registerable_size", lambda _modules: 0)

    def prepare_modules(prepared, *_args, **_kwargs):
        for module in prepared:
            module._prefetch = {"signature": None}
        return None

    monkeypatch.setattr(
        weight_stream_mod.comfy.ops,
        "cast_modules_with_vbar",
        prepare_modules,
    )
    monkeypatch.setattr(
        weight_stream_mod.comfy.model_management,
        "ensure_pin_registerable",
        lambda _size: None,
    )

    streamer = weight_stream_mod.BlockWeightStreamer(
        blocks,
        torch.device("cpu"),
    )
    current = streamer.prepare(0)
    next_state = streamer.prepare(1)
    with pytest.raises(RuntimeError, match="current-plus-next"):
        streamer.prepare(2)

    streamer.release(current)
    final = streamer.prepare(2)
    assert streamer.max_staged_blocks == 2
    streamer.release(next_state)
    streamer.release(final)
    streamer.close()


def test_minimax_forward_does_not_use_comfyui_prefetch_queue():
    source = inspect.getsource(minimax_h3)
    assert "comfy.model_prefetch" not in source
    assert "make_prefetch_queue" not in source
    assert "prefetch_queue_pop" not in source
