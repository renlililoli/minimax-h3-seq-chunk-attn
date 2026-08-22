from __future__ import annotations

import math

import comfy.patcher_extension
import node_helpers
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_api.v0_0_2 import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _resize,
    adapt_canvas,
)
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ReferenceToVideo as NativeMiniMaxH3ReferenceToVideo,
)

from .minimax_h3 import streaming_minimax_h3_forward
from .qwen import patch_minimax_h3_qwen_clip
from .runtime import SeqAttnRuntime, SeqAttnSettings
from .vae import patch_minimax_h3_video_vae

STATE_KEY = "minimax_h3_seqattn"


def _diffusion_model(model_patcher):
    diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
    if not isinstance(diffusion_model, MiniMaxH3Model):
        actual = type(diffusion_model).__name__ if diffusion_model is not None else "None"
        raise TypeError(
            "MiniMaxH3SeqAttn requires a native ComfyUI MiniMaxH3Model; "
            f"received {actual}"
        )
    return diffusion_model


def minimax_h3_seqattn_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None and len(args) >= 4:
        transformer_options = args[3]
    if not isinstance(transformer_options, dict):
        raise ValueError("MiniMax H3 SeqAttn could not resolve transformer_options")
    runtime = transformer_options.get(STATE_KEY)
    if not isinstance(runtime, SeqAttnRuntime):
        raise RuntimeError("MiniMax H3 SeqAttn runtime is missing from transformer_options")
    return streaming_minimax_h3_forward(
        executor.class_obj, runtime, *args, **kwargs
    )


def _runtime_from_patcher(model_patcher):
    return model_patcher.model_options.get("transformer_options", {}).get(STATE_KEY)


def _on_model_clone(original, clone):
    runtime = _runtime_from_patcher(original)
    if isinstance(runtime, SeqAttnRuntime):
        clone.model_options.setdefault("transformer_options", {})[STATE_KEY] = (
            runtime.clone()
        )


def _on_model_cleanup(model_patcher):
    runtime = _runtime_from_patcher(model_patcher)
    if isinstance(runtime, SeqAttnRuntime):
        runtime.clear()


def patch_minimax_h3_model(
    model,
    *,
    activation_workspace_mib: int,
    kv_chunk_tokens: int,
    planner_mode: str,
):
    _diffusion_model(model)
    patched = model.clone()
    runtime = SeqAttnRuntime(
        SeqAttnSettings(
            activation_workspace_mib=int(activation_workspace_mib),
            kv_chunk_tokens=int(kv_chunk_tokens),
            planner_mode=planner_mode,
        )
    )
    patched.model_options.setdefault("transformer_options", {})[STATE_KEY] = runtime
    patched.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, STATE_KEY
    )
    patched.remove_callbacks_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLONE, STATE_KEY
    )
    patched.remove_callbacks_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP, STATE_KEY
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        STATE_KEY,
        minimax_h3_seqattn_wrapper,
    )
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLONE, STATE_KEY, _on_model_clone
    )
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP, STATE_KEY, _on_model_cleanup
    )
    return patched


class MiniMaxH3SeqAttn(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SeqAttn",
            display_name="MiniMax H3 SeqAttn",
            description="Exact CPU-backed streaming attention for native ComfyUI MiniMax H3 models with BF16 activations, including ComfyUI quantized weights.",
            category="model/patch/minimax",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "activation_workspace_mib",
                    default=1024,
                    min=256,
                    max=65536,
                    step=256,
                ),
                io.Combo.Input(
                    "kv_chunk_tokens",
                    options=[2048, 4096, 8192, 16384],
                    default=4096,
                ),
                io.Combo.Input("planner_mode", options=["fit"], default="fit"),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        activation_workspace_mib: int,
        kv_chunk_tokens: int,
        planner_mode: str,
        enabled: bool,
    ) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)
        patched = patch_minimax_h3_model(
            model,
            activation_workspace_mib=activation_workspace_mib,
            kv_chunk_tokens=kv_chunk_tokens,
            planner_mode=planner_mode,
        )
        return io.NodeOutput(patched)


class MiniMaxH3QwenBF16Offload(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3QwenBF16Offload",
            display_name="MiniMax H3 Qwen BF16 Offload",
            description="Bounds native MiniMax-H3 Qwen conditioning memory with BF16 activations, input preflight, and selectable dual-stream or extreme offload.",
            category="model/patch/minimax",
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input(
                    "activation_limit_mib",
                    default=5888,
                    min=1024,
                    max=65536,
                    step=256,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_conditioning_rows",
                    default=25000,
                    min=1024,
                    max=100000,
                    step=1024,
                    advanced=True,
                ),
                io.Int.Input(
                    "preflight_safety_mib",
                    default=128,
                    min=0,
                    max=8192,
                    step=128,
                    advanced=True,
                ),
                io.Combo.Input(
                    "offload_mode",
                    options=["prefetch", "extreme"],
                    default="prefetch",
                ),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(
        cls,
        clip,
        activation_limit_mib: int,
        max_conditioning_rows: int,
        preflight_safety_mib: int,
        offload_mode: str,
        enabled: bool,
    ) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(clip)
        patched = patch_minimax_h3_qwen_clip(
            clip,
            activation_limit_mib=activation_limit_mib,
            max_conditioning_rows=max_conditioning_rows,
            preflight_safety_mib=preflight_safety_mib,
            offload_mode=offload_mode,
        )
        return io.NodeOutput(patched)


class MiniMaxH3VAEStreaming(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3VAEStreaming",
            display_name="MiniMax H3 VAE Streaming",
            description="Reduces native MiniMax-H3 video VAE spatial tiles to bound keyframe encode and video decode activation memory.",
            category="model/patch/minimax",
            is_experimental=True,
            inputs=[
                io.Vae.Input("vae"),
                io.Int.Input(
                    "tile_size",
                    default=192,
                    min=128,
                    max=512,
                    step=16,
                ),
                io.Int.Input(
                    "workspace_mib",
                    default=512,
                    min=256,
                    max=8192,
                    step=256,
                    advanced=True,
                ),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(
        cls,
        vae,
        tile_size: int,
        workspace_mib: int,
        enabled: bool,
    ) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(vae)
        return io.NodeOutput(
            patch_minimax_h3_video_vae(
                vae, tile_size=tile_size, workspace_mib=workspace_mib
            )
        )


class MiniMaxH3ReferenceToVideoSeqAttn(NativeMiniMaxH3ReferenceToVideo):
    """Native Ref2VA conditioning with Qwen encode before reference VAE work."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "MiniMaxH3ReferenceToVideoSeqAttn"
        schema.display_name = "MiniMax H3 Reference to Video (SeqAttn)"
        schema.description = (
            "Reference-to-video conditioning that validates and encodes the Qwen "
            "presentation before running reference image, video, or audio VAEs."
        )
        schema.is_experimental = True
        return schema

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)
        ref_items = []
        prepared_images = []
        prepared_videos = []
        prepared_audios = []

        for image in (ref_images or {}).values():
            if image is None:
                continue
            image_height, image_width = image.shape[1], image.shape[2]
            if ref_image_size == "match":
                scale = min(
                    1.0,
                    math.sqrt(
                        (width * height) / (image_width * image_height)
                    ),
                )
            else:
                scale = min(
                    1.0,
                    REF_IMAGE_SHORT_EDGE / min(image_width, image_height),
                )
            target_width = max(
                CANVAS_MULTIPLE,
                round(image_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            target_height = max(
                CANVAS_MULTIPLE,
                round(image_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            resized = _resize(
                image[:1], target_width, target_height, "disabled"
            )
            prepared_images.append((resized, target_height, target_width))
            ref_items.append({"type": "image", "data": resized})

        video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            suffix = name.rsplit("_", 1)[-1]
            soundtrack = video_audios.get(f"ref_video_audio_{suffix}")
            video_height, video_width = video_frames.shape[1:3]
            canvas_width, canvas_height = adapt_canvas(video_width, video_height)
            if video_width * video_height < canvas_width * canvas_height:
                canvas_width = max(
                    CANVAS_MULTIPLE,
                    round(video_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
                )
                canvas_height = max(
                    CANVAS_MULTIPLE,
                    round(video_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
                )
            frames = _resize(
                video_frames, canvas_width, canvas_height, "disabled"
            )
            frames = frames[:frame_count]
            aligned_frames = int(frames.shape[0])
            if aligned_frames < 5:
                raise ValueError(
                    "MiniMax H3 reference videos need at least 5 frames "
                    "(~0.2s at 24 fps)"
                )
            while aligned_frames % 17 != 5:
                aligned_frames -= 1
            frames = frames[:aligned_frames]
            prepared_videos.append(
                (frames, canvas_height, canvas_width, soundtrack)
            )
            if soundtrack is not None:
                ref_items.append({"type": "audio"})
            sample_indices = list(range(0, frames.shape[0], FPS // 2))
            ref_items.append(
                {
                    "type": "video",
                    "data": frames[sample_indices],
                    "timestamps": [
                        index / 2.0 for index in range(len(sample_indices))
                    ],
                }
            )

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            prepared_audios.append(audio)
            ref_items.append({"type": "audio"})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        ref_blocks = []
        for resized, target_height, target_width in prepared_images:
            encoded = vae.encode(resized)
            ref_blocks.append(
                {
                    "kind": "image",
                    "latent_h": target_height // 16,
                    "latent_w": target_width // 16,
                    "latent": encoded,
                }
            )

        for frames, canvas_height, canvas_width, soundtrack in prepared_videos:
            encoded = vae.encode(frames)
            audio_latent, ref_audio_t = None, 0
            if soundtrack is not None:
                audio_latent, ref_audio_t = cls._encode_ref_audio(
                    audio_vae, soundtrack
                )
            ref_blocks.append(
                {
                    "kind": "video_audio" if ref_audio_t else "video",
                    "latent_t": encoded.shape[2],
                    "latent_h": canvas_height // 16,
                    "latent_w": canvas_width // 16,
                    "ref_audio_t": ref_audio_t,
                    "latent": encoded,
                    "audio_latent": audio_latent,
                }
            )

        for audio in prepared_audios:
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, audio)
            ref_blocks.append(
                {
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent,
                }
            )

        if ref_blocks:
            cond = node_helpers.conditioning_set_values(
                cond, {"minimax_refs": ref_blocks}
            )
        return io.NodeOutput(cond, latent)


class SeqAttnExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3SeqAttn,
            MiniMaxH3QwenBF16Offload,
            MiniMaxH3VAEStreaming,
            MiniMaxH3ReferenceToVideoSeqAttn,
        ]


async def comfy_entrypoint():
    return SeqAttnExtension()


__all__ = [
    "MiniMaxH3SeqAttn",
    "MiniMaxH3QwenBF16Offload",
    "MiniMaxH3ReferenceToVideoSeqAttn",
    "MiniMaxH3VAEStreaming",
    "SeqAttnExtension",
    "comfy_entrypoint",
    "minimax_h3_seqattn_wrapper",
    "patch_minimax_h3_model",
    "patch_minimax_h3_video_vae",
]
