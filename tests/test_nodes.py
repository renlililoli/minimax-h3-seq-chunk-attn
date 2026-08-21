from __future__ import annotations

import contextlib
import types
from types import SimpleNamespace

import comfy.patcher_extension
import pytest
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from comfyui_seqattn import nodes
from comfyui_seqattn import vae as vae_mod


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


class FakeVAEPatcher:
    def __init__(self, model):
        self.model = model

    def clone(self):
        return FakeVAEPatcher(self.model)


def _fake_video_vae():
    model = object.__new__(MiniMaxH3VideoVAE)
    model.vae_ratio = 16
    model.tile_size = 256
    model.latents_mean = torch.tensor([1.0, -2.0])
    model.latents_std = torch.tensor([2.0, 4.0])
    vae = SimpleNamespace(
        first_stage_model=model,
        patcher=FakeVAEPatcher(model),
        encode=lambda value: value,
        decode=lambda value: value,
        device=torch.device("cpu"),
        output_device=torch.device("cpu"),
        process_output=lambda value: value,
    )
    return vae


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
    vae = _fake_video_vae()

    original_encode = vae.encode
    patched = nodes.patch_minimax_h3_video_vae(
        vae, tile_size=192, workspace_mib=512
    )

    assert patched is not vae
    assert patched.patcher is not vae.patcher
    assert vae.first_stage_model.tile_size == 256
    assert vae.encode is original_encode
    assert patched.encode is not original_encode
    repatched = nodes.patch_minimax_h3_video_vae(
        patched, tile_size=160, workspace_mib=768
    )
    assert repatched is not patched
    assert repatched.encode is not patched.encode
    assert getattr(repatched, vae_mod.STATE_KEY).tile_size == 160
    assert vae.first_stage_model.tile_size == 256
    with pytest.raises(ValueError, match="divisible"):
        nodes.patch_minimax_h3_video_vae(
            vae, tile_size=190, workspace_mib=512
        )
    vae_mod.unpatch_minimax_h3_video_vae(patched)
    assert vae.encode is original_encode


def test_video_vae_single_frame_decode_matches_native_and_restores_tile(
    monkeypatch,
):
    vae = _fake_video_vae()
    model = vae.first_stage_model
    captured = {}

    def native_decode(self, latent):
        captured["latent"] = latent.clone()
        captured["tile_size"] = self.tile_size
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent)
        return latent * std + mean

    object.__setattr__(model, "decode", types.MethodType(native_decode, model))
    patched = nodes.patch_minimax_h3_video_vae(
        vae, tile_size=192, workspace_mib=512
    )
    controller = getattr(patched, vae_mod.STATE_KEY)
    monkeypatch.setattr(controller, "_load", lambda: None)

    import comfy.model_management as model_management

    monkeypatch.setattr(
        model_management,
        "cuda_device_context",
        lambda _device: contextlib.nullcontext(),
    )
    latent = torch.tensor([[[[[0.25]]], [[[0.5]]]]])
    expected = native_decode(model, latent).movedim(1, -1)
    actual = patched.decode(latent)

    torch.testing.assert_close(captured["latent"], latent)
    torch.testing.assert_close(actual, expected)
    assert captured["tile_size"] == 192
    assert model.tile_size == 256


def test_video_vae_patched_branches_are_isolated():
    vae = _fake_video_vae()
    first = nodes.patch_minimax_h3_video_vae(
        vae, tile_size=192, workspace_mib=512
    )
    second = nodes.patch_minimax_h3_video_vae(
        vae, tile_size=160, workspace_mib=768
    )

    assert first is not second
    assert first.patcher is not second.patcher
    assert getattr(first, vae_mod.STATE_KEY).tile_size == 192
    assert getattr(second, vae_mod.STATE_KEY).tile_size == 160
    vae_mod.unpatch_minimax_h3_video_vae(first)
    assert not hasattr(first, vae_mod.STATE_KEY)
    assert hasattr(second, vae_mod.STATE_KEY)
    assert vae.first_stage_model.tile_size == 256


def test_video_vae_tile_size_rejects_other_vae_types():
    with pytest.raises(TypeError, match="MiniMax H3 video VAE"):
        nodes.patch_minimax_h3_video_vae(
            SimpleNamespace(first_stage_model=object()),
            tile_size=192,
            workspace_mib=512,
        )
