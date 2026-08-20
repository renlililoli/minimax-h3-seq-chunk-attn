from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_runtime_module():
    spec = importlib.util.spec_from_file_location("seqattn_comfy_runtime", PACKAGE_DIR / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime_mod = _load_runtime_module()


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
