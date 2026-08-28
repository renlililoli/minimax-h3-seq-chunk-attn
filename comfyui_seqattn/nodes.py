from __future__ import annotations

import math

import comfy.patcher_extension
import comfy.utils
import folder_paths
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

from .lora import (
    AdapterIdentity,
    H3LoRAState,
    build_h3_target_specs,
    parse_h3_linear_lora,
    validate_h3_int8_convrot_base,
)
from .minimax_h3 import streaming_minimax_h3_forward
from .qwen import patch_minimax_h3_qwen_seqattn_clip
from .runtime import SeqAttnRuntime, SeqAttnSettings
from .vae import patch_minimax_h3_video_vae

STATE_KEY = "minimax_h3_seqattn"
LORA_STATE_KEY = "minimax_h3_seqattn_lora"


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


def _lora_state_from_patcher(model_patcher) -> H3LoRAState:
    transformer_options = model_patcher.model_options.get("transformer_options", {})
    state = transformer_options.get(LORA_STATE_KEY)
    if isinstance(state, H3LoRAState):
        return state
    runtime = transformer_options.get(STATE_KEY)
    if isinstance(runtime, SeqAttnRuntime):
        return runtime.lora_state
    return H3LoRAState()


def _validate_unpatched_model_patcher(model_patcher) -> None:
    patch_keys = set(getattr(model_patcher, "patches", {}))
    patch_keys.update(getattr(model_patcher, "weight_wrapper_patches", {}))
    if patch_keys:
        preview = ", ".join(sorted(str(key) for key in patch_keys)[:4])
        raise ValueError(
            "MiniMaxH3SeqAttnLoRA requires an unpatched model; ordinary "
            f"ComfyUI weight patches are present on {preview}"
        )


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
    model, *, q_chunk_tokens: int, kv_chunk_tokens: int
):
    _diffusion_model(model)
    lora_state = _lora_state_from_patcher(model)
    patched = model.clone()
    runtime = SeqAttnRuntime(
        SeqAttnSettings.from_config(
            q_chunk_tokens=q_chunk_tokens,
            kv_chunk_tokens=kv_chunk_tokens,
        ),
        lora_state=lora_state,
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options[STATE_KEY] = runtime
    if lora_state.bundles:
        transformer_options[LORA_STATE_KEY] = lora_state
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


def patch_minimax_h3_lora_model(model, *, lora_name: str, strength_model: float):
    diffusion_model = _diffusion_model(model)
    _validate_unpatched_model_patcher(model)
    specs = build_h3_target_specs(diffusion_model)
    validate_h3_int8_convrot_base(diffusion_model, specs)
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state_dict, _metadata = comfy.utils.load_torch_file(
        lora_path,
        safe_load=True,
        return_metadata=True,
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"{lora_name} did not contain a tensor state dictionary")
    identity = AdapterIdentity.from_path(lora_name, lora_path)
    bundle = parse_h3_linear_lora(
        state_dict,
        identity=identity,
        strength=float(strength_model),
        specs=specs,
    )
    lora_state = _lora_state_from_patcher(model).append(bundle, specs)

    patched = model.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options[LORA_STATE_KEY] = lora_state
    runtime = transformer_options.get(STATE_KEY)
    if isinstance(runtime, SeqAttnRuntime):
        transformer_options[STATE_KEY] = runtime.with_lora_state(lora_state)
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
                    "q_chunk_tokens",
                    default=5760,
                    min=128,
                    max=262144,
                    step=128,
                ),
                io.Combo.Input(
                    "kv_chunk_tokens",
                    options=[2048, 4096, 8192, 16384],
                    default=4096,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        q_chunk_tokens: int,
        kv_chunk_tokens: int,
    ) -> io.NodeOutput:
        patched = patch_minimax_h3_model(
            model,
            q_chunk_tokens=q_chunk_tokens,
            kv_chunk_tokens=kv_chunk_tokens,
        )
        return io.NodeOutput(patched)


class MiniMaxH3SeqAttnLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SeqAttnLoRA",
            display_name="MiniMax H3 SeqAttn LoRA",
            description=(
                "Stages ordinary Linear LoRA adapters beside the MiniMax-H3 "
                "INT8 ConvRot base without merging or requantizing its weights."
            ),
            category="model/patch/minimax",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                ),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-20.0,
                    max=20.0,
                    step=0.01,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        lora_name: str,
        strength_model: float,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            patch_minimax_h3_lora_model(
                model,
                lora_name=lora_name,
                strength_model=strength_model,
            )
        )


class MiniMaxH3QwenSeqAttn(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3QwenSeqAttn",
            display_name="MiniMax H3 Qwen SeqAttn",
            description="Exact CPU-backed streaming vision and causal GQA conditioning for the native MiniMax-H3 Qwen3-VL-32B encoder.",
            category="model/patch/minimax",
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input(
                    "q_chunk_tokens",
                    default=5760,
                    min=128,
                    max=262144,
                    step=128,
                ),
                io.Combo.Input(
                    "kv_chunk_tokens",
                    options=[2048, 4096, 8192, 16384],
                    default=4096,
                ),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(
        cls, clip, q_chunk_tokens: int, kv_chunk_tokens: int
    ) -> io.NodeOutput:
        return io.NodeOutput(
            patch_minimax_h3_qwen_seqattn_clip(
                clip,
                q_chunk_tokens=q_chunk_tokens,
                kv_chunk_tokens=kv_chunk_tokens,
            )
        )


class MiniMaxH3VAEStreaming(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3VAEStreaming",
            display_name="MiniMax H3 VAE Streaming",
            description="Reduces native MiniMax-H3 video VAE spatial tiles to bound keyframe encode and video decode activation memory.",
            category="model/patch/minimax",
            is_experimental=True,
            inputs=[io.Vae.Input("vae")],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae) -> io.NodeOutput:
        return io.NodeOutput(patch_minimax_h3_video_vae(vae))


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
            MiniMaxH3SeqAttnLoRA,
            MiniMaxH3QwenSeqAttn,
            MiniMaxH3VAEStreaming,
            MiniMaxH3ReferenceToVideoSeqAttn,
        ]


async def comfy_entrypoint():
    return SeqAttnExtension()


__all__ = [
    "MiniMaxH3SeqAttn",
    "MiniMaxH3SeqAttnLoRA",
    "MiniMaxH3QwenSeqAttn",
    "MiniMaxH3ReferenceToVideoSeqAttn",
    "MiniMaxH3VAEStreaming",
    "SeqAttnExtension",
    "comfy_entrypoint",
    "minimax_h3_seqattn_wrapper",
    "patch_minimax_h3_model",
    "patch_minimax_h3_lora_model",
    "patch_minimax_h3_video_vae",
]
