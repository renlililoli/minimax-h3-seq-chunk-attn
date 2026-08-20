from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_nodes():
    spec = importlib.util.spec_from_file_location(
        "comfyui_seqattn_nodes_test",
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules["comfyui_seqattn_nodes_test.nodes"]


nodes = _load_nodes()


class FakePatcher:
    def __init__(self, diffusion_model):
        self.model = SimpleNamespace(diffusion_model=diffusion_model)
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.callbacks = {}

    def clone(self):
        clone = FakePatcher(self.model.diffusion_model)
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        clone.wrappers = {
            kind: {key: values.copy() for key, values in keyed.items()}
            for kind, keyed in self.wrappers.items()
        }
        clone.callbacks = {
            kind: {key: values.copy() for key, values in keyed.items()}
            for kind, keyed in self.callbacks.items()
        }
        for callback in self.get_all_callbacks(
            comfy.patcher_extension.CallbacksMP.ON_CLONE
        ):
            callback(self, clone)
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def add_callback_with_key(self, callback_type, key, callback):
        self.callbacks.setdefault(callback_type, {}).setdefault(key, []).append(callback)

    def remove_callbacks_with_key(self, callback_type, key):
        self.callbacks.get(callback_type, {}).pop(key, None)

    def get_all_callbacks(self, callback_type):
        result = []
        for callbacks in self.callbacks.get(callback_type, {}).values():
            result.extend(callbacks)
        return result


def test_model_type_validation():
    with pytest.raises(TypeError, match="MiniMaxH3Model"):
        nodes.patch_minimax_h3_model(
            FakePatcher(object()),
            activation_workspace_mib=4096,
            kv_chunk_tokens=4096,
            planner_mode="fit",
        )


def test_patch_is_clone_isolated_and_idempotent():
    diffusion_model = object.__new__(MiniMaxH3Model)
    original = FakePatcher(diffusion_model)
    first = nodes.patch_minimax_h3_model(
        original,
        activation_workspace_mib=4096,
        kv_chunk_tokens=4096,
        planner_mode="fit",
    )
    second = nodes.patch_minimax_h3_model(
        first,
        activation_workspace_mib=2048,
        kv_chunk_tokens=8192,
        planner_mode="fit",
    )

    assert nodes.STATE_KEY not in original.model_options["transformer_options"]
    wrappers = second.wrappers[
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    ][nodes.STATE_KEY]
    assert wrappers == [nodes.minimax_h3_seqattn_wrapper]
    first_runtime = first.model_options["transformer_options"][nodes.STATE_KEY]
    second_runtime = second.model_options["transformer_options"][nodes.STATE_KEY]
    assert first_runtime is not second_runtime
    assert second_runtime.settings.activation_workspace_mib == 2048


def test_clone_and_cleanup_callbacks_manage_runtime():
    patched = nodes.patch_minimax_h3_model(
        FakePatcher(object.__new__(MiniMaxH3Model)),
        activation_workspace_mib=4096,
        kv_chunk_tokens=4096,
        planner_mode="fit",
    )
    cloned = patched.clone()
    runtime = patched.model_options["transformer_options"][nodes.STATE_KEY]
    cloned_runtime = cloned.model_options["transformer_options"][nodes.STATE_KEY]
    assert cloned_runtime is not runtime
    runtime._runners["sentinel"] = object()
    for callback in patched.get_all_callbacks(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP
    ):
        callback(patched)
    assert runtime.cache_size == 0
