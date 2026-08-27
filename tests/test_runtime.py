from __future__ import annotations

from types import SimpleNamespace

import pytest
import seqattn_core
import torch

import comfyui_seqattn
from comfyui_seqattn import minimax_h3 as streaming
from comfyui_seqattn import runtime as runtime_mod


def test_seqattn_dependency_version():
    assert seqattn_core.__version__ == "0.3.0a3"
    assert comfyui_seqattn.__version__ == "0.4.2"


def test_settings_validation_and_toml_tiles(tmp_path, monkeypatch):
    runtime_mod.SeqAttnSettings().validate()
    with pytest.raises(ValueError, match="positive"):
        runtime_mod.SeqAttnSettings(q_chunk_tokens=0).validate()

    config_path = tmp_path / "seqattn.toml"
    config_path.write_text(
        "[minimax_h3]\n"
        "qkv_tile_tokens = 1024\n"
        "mlp_tile_tokens = 512\n"
    )
    monkeypatch.setenv("SEQATTN_CONFIG", str(config_path))
    settings = runtime_mod.SeqAttnSettings.from_config(
        q_chunk_tokens=3840,
        kv_chunk_tokens=4096,
    )
    assert settings.q_chunk_tokens == 3840
    assert settings.kv_chunk_tokens == 4096
    assert settings.qkv_tile_tokens == 1024
    assert settings.mlp_tile_tokens == 512


def test_runtime_clone_isolated_and_clear(monkeypatch):
    created = []
    created_dit = []

    class FakeRunner:
        def __init__(self, plan, attention_config, pipeline_config):
            created.append((plan, attention_config, pipeline_config))

    class FakeDiTRunner:
        def __init__(self, projected, **kwargs):
            created_dit.append((projected, kwargs))

    monkeypatch.setattr(runtime_mod, "ProjectedAttentionRunner", FakeRunner)
    monkeypatch.setattr(runtime_mod, "H3DiTRunner", FakeDiTRunner)
    monkeypatch.setattr(runtime_mod, "build_plan", lambda **kwargs: kwargs)

    runtime = runtime_mod.SeqAttnRuntime(runtime_mod.SeqAttnSettings())
    first = runtime.runner_for(
        tokens=257,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    assert created[0][1].backend is None
    assert runtime.runner_for(
        tokens=257,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    ) is first
    assert len(created) == 1

    dit = runtime.dit_runner_for(
        tokens=257,
        hidden_features=256,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    assert created[1][1].backend is None
    assert runtime.dit_runner_for(
        tokens=257,
        hidden_features=256,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    ) is dit
    assert len(created_dit) == 1

    clone = runtime.clone()
    assert clone.settings == runtime.settings
    assert clone.cache_size == 0
    assert (
        clone.lifetime_refined_conditioning_cache_stats
        == runtime.lifetime_refined_conditioning_cache_stats
    )
    assert clone.refined_conditioning_cache_stats["entries"] == 0
    assert runtime.cache_size == 2
    runtime.record_weight_schedule({"event": "root"})
    clone.record_weight_schedule({"event": "clone"})
    assert runtime.weight_schedule_records == [
        {"event": "root"},
        {"event": "clone"},
    ]
    assert clone.weight_schedule_records == runtime.weight_schedule_records
    clone.clear()
    assert runtime.weight_schedule_records == [
        {"event": "root"},
        {"event": "clone"},
    ]
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
