from __future__ import annotations

import pytest
import torch

from comfyui_seqattn import runtime as runtime_mod


def test_settings_validation():
    runtime_mod.SeqAttnSettings().validate()
    with pytest.raises(ValueError, match="planner_mode"):
        runtime_mod.SeqAttnSettings(planner_mode="fixed").validate()
    with pytest.raises(ValueError, match="positive"):
        runtime_mod.SeqAttnSettings(activation_workspace_mib=0).validate()


def test_runtime_clone_isolated_and_clear(monkeypatch):
    created = []

    class FakeRunner:
        def __init__(self, plan, attention_config, pipeline_config):
            created.append((plan, attention_config, pipeline_config))

    monkeypatch.setattr(runtime_mod, "ProjectedAttentionRunner", FakeRunner)
    monkeypatch.setattr(runtime_mod, "build_plan", lambda **kwargs: kwargs)

    runtime = runtime_mod.SeqAttnRuntime(runtime_mod.SeqAttnSettings())
    first = runtime.runner_for(
        tokens=257,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    assert runtime.runner_for(
        tokens=257,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    ) is first
    assert len(created) == 1

    clone = runtime.clone()
    assert clone.settings == runtime.settings
    assert clone.cache_size == 0
    assert runtime.cache_size == 1
    runtime.clear()
    assert runtime.cache_size == 0
