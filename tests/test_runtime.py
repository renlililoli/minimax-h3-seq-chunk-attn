from __future__ import annotations

from types import SimpleNamespace

import pytest
import seqattn_core
import torch

import comfyui_seqattn
from comfyui_seqattn import lora as lora_mod
from comfyui_seqattn import minimax_h3 as streaming
from comfyui_seqattn import runtime as runtime_mod


def test_seqattn_dependency_version():
    assert seqattn_core.__version__ == "0.4.0a1"
    assert comfyui_seqattn.__version__ == "0.4.4"


def test_settings_validation_and_toml_tiles(tmp_path, monkeypatch):
    runtime_mod.SeqAttnSettings().validate()
    assert runtime_mod.SeqAttnSettings().execution_mode == "materialized"
    with pytest.raises(ValueError, match="positive"):
        runtime_mod.SeqAttnSettings(q_chunk_tokens=0).validate()
    with pytest.raises(ValueError, match="execution_mode"):
        runtime_mod.SeqAttnSettings(execution_mode="fallback").validate()

    config_path = tmp_path / "seqattn.toml"
    config_path.write_text(
        "[minimax_h3]\n"
        "execution_mode = 'recompute'\n"
        "attention_mode = 'sol_streaming'\n"
        "projection_tile_tokens = 1024\n"
        "ffn_tile_tokens = 512\n"
        "sol_tau = 0.75\n"
        "sol_first_dense_step_fraction = 0.1\n"
        "sol_first_dense_layers = 3\n"
    )
    monkeypatch.setenv("SEQATTN_CONFIG", str(config_path))
    settings = runtime_mod.SeqAttnSettings.from_config(
        q_chunk_tokens=3840,
        kv_chunk_tokens=4096,
    )
    assert settings.q_chunk_tokens == 3840
    assert settings.execution_mode == "recompute"
    assert settings.attention_mode == "sol_streaming"
    assert settings.kv_chunk_tokens == 4096
    assert settings.projection_tile_tokens == 1024
    assert settings.ffn_tile_tokens == 512
    assert settings.sol_tau == 0.75
    assert settings.sol_first_dense_step_fraction == 0.1
    assert settings.sol_first_dense_layers == 3


def test_runtime_clone_isolated_and_clear(monkeypatch):
    created = []
    created_dit = []

    class FakeRunner:
        def __init__(self, plan, pipeline_config):
            created.append((plan, pipeline_config))

    def fake_build_h3_runner(plan, **kwargs):
        runner = SimpleNamespace(plan=plan, kwargs=kwargs)
        created_dit.append(runner)
        return runner

    monkeypatch.setattr(runtime_mod, "ProjectedAttentionRunner", FakeRunner)
    monkeypatch.setattr(runtime_mod, "build_h3_runner", fake_build_h3_runner)
    monkeypatch.setattr(runtime_mod, "build_attention_plan", lambda **kwargs: kwargs)

    runtime = runtime_mod.SeqAttnRuntime(runtime_mod.SeqAttnSettings())
    first = runtime.runner_for(
        tokens=257,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    assert created[0][0]["config"].backend is None
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
    assert created_dit[0].plan["config"].backend is None
    assert created_dit[0].kwargs["config"] == runtime.settings.h3_config()
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
    assert clone.lora_state is runtime.lora_state
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


def test_dit_runtime_propagates_and_caches_h3_config(monkeypatch):
    created = []

    def fake_build_h3_runner(plan, **kwargs):
        runner = SimpleNamespace(plan=plan, kwargs=kwargs)
        created.append(runner)
        return runner

    monkeypatch.setattr(runtime_mod, "build_h3_runner", fake_build_h3_runner)
    monkeypatch.setattr(runtime_mod, "build_attention_plan", lambda **kwargs: kwargs)

    settings = runtime_mod.SeqAttnSettings(
        execution_mode="recompute",
        attention_mode="sol_streaming",
        projection_tile_tokens=1024,
        ffn_tile_tokens=512,
        sol_tau=0.75,
        sol_first_dense_step_fraction=0.1,
        sol_first_dense_layers=3,
    )
    runtime = runtime_mod.SeqAttnRuntime(settings)
    first = runtime.dit_runner_for(
        tokens=257,
        hidden_features=256,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    second = runtime.dit_runner_for(
        tokens=257,
        hidden_features=256,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )

    assert first is second
    assert len(created) == 1
    assert created[0].kwargs["hidden_features"] == 256
    assert created[0].kwargs["num_output_buffers"] == 2
    assert created[0].kwargs["config"] == settings.h3_config()
    assert runtime.clone().settings.execution_mode == "recompute"


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
    lora_key = streaming._refined_conditioning_cache_key(
        model, first, {}, (("adapter", 1.0),)
    )

    assert first_key == same_key
    assert first_key != changed_key
    assert first_key != float_key
    assert first_key != lora_key
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


def test_runtime_with_lora_state_isolated_but_shares_adapter_tensors():
    identity = lora_mod.AdapterIdentity("a", "/a", 1, 2)
    down = torch.ones(1, 2)
    up = torch.ones(3, 1)
    layer = lora_mod.LinearLoRA(
        identity,
        "blocks.0.attn.out_proj",
        down,
        up,
        None,
        1,
        2,
        3,
        torch.float32,
        1.0,
    )
    state = lora_mod.H3LoRAState(
        (lora_mod.LinearLoRABundle(identity, 1.0, (layer,)),),
        (lora_mod.H3TargetSpec(layer.target, ("block", 0), 2, 3),),
    )
    runtime = runtime_mod.SeqAttnRuntime(runtime_mod.SeqAttnSettings())
    patched = runtime.with_lora_state(state)

    assert patched is not runtime
    assert patched.lora_state is state
    assert patched.lora_state.bundles[0].layers[0].down is down
    assert patched.cache_size == 0
