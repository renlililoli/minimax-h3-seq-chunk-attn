from __future__ import annotations

from types import SimpleNamespace

import comfy.patcher_extension
import pytest
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from comfyui_seqattn import nodes


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


def test_video_vae_tile_size_validation_and_patch():
    vae = SimpleNamespace(
        first_stage_model=object.__new__(MiniMaxH3VideoVAE),
        encode=lambda value: value,
        decode=lambda value: value,
    )
    vae.first_stage_model.vae_ratio = 16
    vae.first_stage_model.tile_size = 256

    original_encode = vae.encode
    patched = nodes.patch_minimax_h3_video_vae(
        vae, tile_size=192, workspace_mib=512
    )

    assert patched is vae
    assert vae.first_stage_model.tile_size == 192
    assert vae.encode is not original_encode
    first_patched_encode = vae.encode
    nodes.patch_minimax_h3_video_vae(
        vae, tile_size=160, workspace_mib=768
    )
    assert vae.encode is first_patched_encode
    assert vae.first_stage_model.tile_size == 160
    with pytest.raises(ValueError, match="divisible"):
        nodes.patch_minimax_h3_video_vae(
            vae, tile_size=190, workspace_mib=512
        )
    nodes.unpatch_minimax_h3_video_vae(vae)
    assert vae.encode is original_encode


def test_video_vae_tile_size_rejects_other_vae_types():
    with pytest.raises(TypeError, match="MiniMax H3 video VAE"):
        nodes.patch_minimax_h3_video_vae(
            SimpleNamespace(first_stage_model=object()),
            tile_size=192,
            workspace_mib=512,
        )
