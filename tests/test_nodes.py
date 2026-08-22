from __future__ import annotations

import contextlib
from types import SimpleNamespace

import comfy.patcher_extension
import pytest
import torch
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
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
    model.latents_mean = torch.tensor([0.2, 0.3, 0.4])
    model.latents_std = torch.tensor([0.1, 0.2, 0.3])
    model.pixel_mean = torch.zeros((1, 3, 1, 1, 1))
    model.pixel_std = torch.ones((1, 3, 1, 1, 1))
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

    def adaptive_decode(latent):
        captured["denormalized_latent"] = latent.clone()
        captured["tile_size"] = model.tile_size
        return latent

    object.__setattr__(model, "_adaptive_decode", adaptive_decode)
    patched = nodes.patch_minimax_h3_video_vae(
        vae, tile_size=192, workspace_mib=512
    )
    controller = getattr(patched, vae_mod.STATE_KEY)
    load_inference_modes = []
    monkeypatch.setattr(
        controller,
        "_load",
        lambda: load_inference_modes.append(torch.is_inference_mode_enabled()),
    )

    import comfy.model_management as model_management

    monkeypatch.setattr(
        model_management,
        "cuda_device_context",
        lambda _device: contextlib.nullcontext(),
    )
    latent = torch.tensor([[[[[0.5]]], [[[0.25]]], [[[-0.5]]]]])
    expected = model.decode(latent).movedim(1, -1)
    actual = patched.decode(latent)

    latent_mean = model.latents_mean.view(1, -1, 1, 1, 1)
    latent_std = model.latents_std.view(1, -1, 1, 1, 1)
    torch.testing.assert_close(
        captured["denormalized_latent"], latent * latent_std + latent_mean
    )
    torch.testing.assert_close(actual, expected)
    assert captured["tile_size"] == 192
    assert model.tile_size == 256
    assert load_inference_modes == [False]


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
    disabled = nodes.MiniMaxH3VAEStreaming.execute(
        first,
        tile_size=128,
        workspace_mib=256,
        enabled=False,
    ).result[0]
    assert disabled is first
    assert hasattr(first, vae_mod.STATE_KEY)
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


def test_ref2va_seqattn_schema_preserves_native_input_contract():
    native = MiniMaxH3ReferenceToVideo.define_schema()
    seqattn = nodes.MiniMaxH3ReferenceToVideoSeqAttn.define_schema()

    assert seqattn.node_id == "MiniMaxH3ReferenceToVideoSeqAttn"
    assert [item.id for item in seqattn.inputs] == [
        item.id for item in native.inputs
    ]
    assert [item.id for item in seqattn.outputs] == [
        item.id for item in native.outputs
    ]


def test_ref2va_qwen_rejection_happens_before_any_vae_encode(monkeypatch):
    events = []

    class RejectingClip:
        def tokenize(self, prompt, *, minimax_ref_items):
            events.append(("tokenize", prompt, len(minimax_ref_items)))
            return "tokens"

        def encode_from_tokens_scheduled(self, tokens):
            events.append(("qwen", tokens))
            raise RuntimeError("Qwen input rejected before encode")

    class RecordingVAE:
        def encode(self, _value):
            events.append(("vae",))
            raise AssertionError("VAE encode must not run after Qwen rejection")

    monkeypatch.setattr(nodes, "_empty_av_latent", lambda *_args: ("latent", 5))
    monkeypatch.setattr(nodes, "_resize", lambda value, *_args: value)
    image = torch.zeros((1, 32, 32, 3))

    with pytest.raises(RuntimeError, match="Qwen input rejected before encode"):
        nodes.MiniMaxH3ReferenceToVideoSeqAttn.execute(
            RejectingClip(),
            RecordingVAE(),
            RecordingVAE(),
            "prompt",
            32,
            32,
            5,
            ref_images={"ref_image_0": image},
        )

    assert events == [("tokenize", "prompt", 1), ("qwen", "tokens")]


def test_ref2va_seqattn_preserves_reference_order_and_payload(monkeypatch):
    events = []
    captured = {}

    class RecordingClip:
        def tokenize(self, _prompt, *, minimax_ref_items):
            captured["ref_item_types"] = [
                item["type"] for item in minimax_ref_items
            ]
            events.append("tokenize")
            return "tokens"

        def encode_from_tokens_scheduled(self, _tokens):
            events.append("qwen")
            return "conditioning"

    class RecordingVideoVAE:
        def encode(self, value):
            kind = "image" if value.shape[0] == 1 else "video"
            events.append(f"video_vae:{kind}")
            latent_t = 1 if kind == "image" else 2
            return torch.zeros((1, 24, latent_t, 2, 2))

    class RecordingAudioVAE:
        audio_sample_rate = 32000

        def encode(self, _value):
            events.append("audio_vae")
            return torch.zeros((1, 32, 2, 3))

    def set_values(conditioning, values):
        captured["conditioning"] = conditioning
        captured["refs"] = values["minimax_refs"]
        return "conditioned"

    monkeypatch.setattr(nodes, "_empty_av_latent", lambda *_args: ("latent", 5))
    monkeypatch.setattr(nodes, "_resize", lambda value, *_args: value)
    monkeypatch.setattr(nodes, "adapt_canvas", lambda width, height: (width, height))
    monkeypatch.setattr(nodes.node_helpers, "conditioning_set_values", set_values)

    image = torch.zeros((1, 32, 32, 3))
    video = torch.zeros((5, 32, 32, 3))
    audio = {
        "waveform": torch.zeros((1, 2, 16)),
        "sample_rate": 32000,
    }
    output = nodes.MiniMaxH3ReferenceToVideoSeqAttn.execute(
        RecordingClip(),
        RecordingVideoVAE(),
        RecordingAudioVAE(),
        "prompt",
        32,
        32,
        5,
        ref_images={"ref_image_0": image},
        ref_videos={"ref_video_0": video},
        ref_video_audios={"ref_video_audio_0": audio},
        ref_audios={"ref_audio_0": audio},
    )

    assert output.result == ("conditioned", "latent")
    assert events == [
        "tokenize",
        "qwen",
        "video_vae:image",
        "video_vae:video",
        "audio_vae",
        "audio_vae",
    ]
    assert captured["ref_item_types"] == ["image", "audio", "video", "audio"]
    assert captured["conditioning"] == "conditioning"
    assert [block["kind"] for block in captured["refs"]] == [
        "image",
        "video_audio",
        "audio",
    ]
