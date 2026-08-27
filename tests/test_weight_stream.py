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


def test_run_weight_stages_uses_current_plus_next_order(monkeypatch):
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
        weight_stream_mod.comfy.ops, "cast_modules_with_vbar", prepare_modules
    )
    monkeypatch.setattr(
        weight_stream_mod.comfy.model_management,
        "ensure_pin_registerable",
        lambda _size: None,
    )
    events = []
    computed = []
    max_staged = weight_stream_mod.run_weight_stages(
        blocks,
        torch.device("cpu"),
        computed.append,
        record=lambda event: events.append(
            (event["event"], event["block_index"], event["staged_blocks"])
        ),
    )

    assert computed == [0, 1, 2]
    assert max_staged == 2
    assert events == [
        ("prepare", 0, [0]),
        ("ready", 0, [0]),
        ("prepare", 1, [0, 1]),
        ("compute_start", 0, [0, 1]),
        ("compute_end", 0, [0, 1]),
        ("release", 0, [1]),
        ("ready", 1, [1]),
        ("prepare", 2, [1, 2]),
        ("compute_start", 1, [1, 2]),
        ("compute_end", 1, [1, 2]),
        ("release", 1, [2]),
        ("ready", 2, [2]),
        ("compute_start", 2, [2]),
        ("compute_end", 2, [2]),
        ("release", 2, []),
    ]


def test_stage_group_collects_unique_vbar_modules(monkeypatch):
    module_a = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module_b = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module_a[0]._v = object()
    module_b[0]._v = object()

    modules = weight_stream_mod._vbar_modules((module_a, [module_b, module_a]))

    assert modules == [module_a[0], module_b[0]]


@pytest.mark.parametrize(
    ("reported", "expected_requests", "expected_evicted_mib"),
    [
        (-2**63, [4096, 4096], 0.0),
        (2**63, [4096], 4096 / 2**20),
    ],
)
def test_weight_streamer_bounds_invalid_vbar_free_counts(
    monkeypatch, reported, expected_requests, expected_evicted_mib
):
    requests = []

    class FakeVbar:
        def loaded_size(self):
            return 0

        def free_memory(self, size):
            requests.append(size)
            return reported

    class FakeModule:
        def __init__(self, vbar):
            self._v = (vbar, None, 2048)
            self._prefetch = {"signature": object()}

    modules = [FakeModule(FakeVbar()), FakeModule(FakeVbar())]
    monkeypatch.setattr(
        weight_stream_mod.comfy_aimdo.model_vbar,
        "vbar_unpin",
        lambda _v: None,
    )
    events = []
    streamer = weight_stream_mod.BlockWeightStreamer(
        _plain_blocks(1), torch.device("cpu"), record=events.append
    )
    state = weight_stream_mod.BlockWeightState(
        index=0,
        modules=modules,
        stream=None,
        registerable_bytes=4096,
        staged_bytes=4096,
        prepared_at=0.0,
    )
    streamer.states[0] = state
    streamer.staged_indices.add(0)

    streamer.release(state)

    assert requests == expected_requests
    assert events[-1]["evicted_mib"] == expected_evicted_mib
    assert state.released


def test_minimax_forward_does_not_use_comfyui_prefetch_queue():
    source = inspect.getsource(minimax_h3)
    assert "comfy.model_prefetch" not in source
    assert "make_prefetch_queue" not in source
    assert "prefetch_queue_pop" not in source
