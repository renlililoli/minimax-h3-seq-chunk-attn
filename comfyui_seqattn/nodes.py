from __future__ import annotations

import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_api.v0_0_2 import ComfyExtension, io

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


class SeqAttnExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3SeqAttn,
            MiniMaxH3QwenBF16Offload,
            MiniMaxH3VAEStreaming,
        ]


async def comfy_entrypoint():
    return SeqAttnExtension()


__all__ = [
    "MiniMaxH3SeqAttn",
    "MiniMaxH3QwenBF16Offload",
    "MiniMaxH3VAEStreaming",
    "SeqAttnExtension",
    "comfy_entrypoint",
    "minimax_h3_seqattn_wrapper",
    "patch_minimax_h3_model",
    "patch_minimax_h3_video_vae",
]
