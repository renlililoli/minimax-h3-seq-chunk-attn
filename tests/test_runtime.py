from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_seqattn import minimax_h3 as streaming
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
    assert (
        clone.lifetime_refined_conditioning_cache_stats
        == runtime.lifetime_refined_conditioning_cache_stats
    )
    assert clone.refined_conditioning_cache_stats["entries"] == 0
    assert runtime.cache_size == 1
    runtime.clear()
    assert runtime.cache_size == 0
    assert runtime.refined_conditioning_cache_stats["entries"] == 0
    assert runtime.last_refined_conditioning_cache_stats is None


def test_refined_conditioning_fallback_key_uses_tensor_content_and_metadata():
    model = SimpleNamespace(
        hidden_size=8,
        condition_proj=torch.nn.Linear(4, 8),
        token_refiner=torch.nn.Sequential(torch.nn.Linear(8, 8)),
    )
    first = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    same = first.clone()
    changed = first.clone()
    changed[0, 0] += 1

    first_key = streaming._refined_conditioning_cache_key(model, first, {})
    same_key = streaming._refined_conditioning_cache_key(model, same, {})
    changed_key = streaming._refined_conditioning_cache_key(model, changed, {})
    float_key = streaming._refined_conditioning_cache_key(
        model, first.float(), {}
    )

    assert first_key == same_key
    assert first_key != changed_key
    assert first_key != float_key
    assert (
        streaming._refined_conditioning_cache_key(
            model, first, {"patches": {"attention": object()}}
        )
        is None
    )
    assert (
        streaming._refined_conditioning_cache_key(
            model, first, {"optimized_attention_override": object()}
        )
        is None
    )
