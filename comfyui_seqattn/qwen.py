from __future__ import annotations

import math
import numbers
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass

import torch

QWEN_STATE_KEY = "minimax_h3_qwen_bf16_offload"

_BASELINE_TOTAL_ROWS = 11325
_BASELINE_VISUAL_ROWS = 11088
_BASELINE_PLANNED_ACTIVATION_MIB = 3159.71240234375
_ENCODE_POLICY_LOCK = threading.RLock()


@dataclass(frozen=True)
class QwenMemorySettings:
    activation_limit_mib: int = 5888
    max_conditioning_rows: int = 25000
    preflight_safety_mib: int = 128
    offload_mode: str = "prefetch"

    def validate(self) -> None:
        if self.activation_limit_mib <= 0:
            raise ValueError("activation_limit_mib must be positive")
        if self.max_conditioning_rows <= 0:
            raise ValueError("max_conditioning_rows must be positive")
        if self.preflight_safety_mib < 0:
            raise ValueError("preflight_safety_mib cannot be negative")
        if self.offload_mode not in {"prefetch", "extreme"}:
            raise ValueError("offload_mode must be 'prefetch' or 'extreme'")

    @property
    def offload_streams(self) -> int:
        return 2 if self.offload_mode == "prefetch" else 0


def qwen_vision_merged_rows(data: torch.Tensor) -> int:
    """Predict Qwen3-VL merged rows without running the vision tower."""
    if not isinstance(data, torch.Tensor) or data.ndim != 4:
        raise ValueError("Qwen visual input must have shape [T, H, W, C]")
    _, height, width, channels = data.shape
    if channels != 3 or height <= 0 or width <= 0:
        raise ValueError(f"invalid Qwen visual input shape: {list(data.shape)}")

    factor = 16 * 2
    min_pixels = 3136
    max_pixels = 12845056
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return (resized_height // factor) * (resized_width // factor)


def _qwen_token_batches(tokens) -> list:
    if isinstance(tokens, dict):
        if len(tokens) != 1:
            raise ValueError(f"expected one Qwen token stream, found {len(tokens)}")
        tokens = next(iter(tokens.values()))
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Qwen tokenizer returned no token batches")
    return tokens


def inspect_qwen_input_tokens(tokens) -> dict[str, int]:
    non_visual_rows = 0
    visual_rows = 0
    image_rows = 0
    video_rows = 0
    embedding_rows = 0

    for batch in _qwen_token_batches(tokens):
        for weighted_entry in batch:
            entry = weighted_entry[0] if isinstance(weighted_entry, tuple) else weighted_entry
            if isinstance(entry, numbers.Integral):
                non_visual_rows += 1
                continue
            if isinstance(entry, torch.Tensor):
                if entry.ndim == 0:
                    raise ValueError("Qwen embedding input must have a row axis")
                embedding_rows += int(entry.numel() // entry.shape[-1])
                continue
            if not isinstance(entry, dict):
                raise TypeError(
                    f"unsupported Qwen token entry type: {type(entry).__name__}"
                )

            entry_type = entry.get("type")
            if entry_type == "embedding":
                data = entry.get("data")
                if not isinstance(data, torch.Tensor) or data.ndim == 0:
                    raise ValueError("Qwen embedding entry has no valid tensor")
                embedding_rows += int(data.numel() // data.shape[-1])
                continue
            if entry_type != "image":
                raise ValueError(f"cannot preflight Qwen entry type {entry_type!r}")

            rows = qwen_vision_merged_rows(entry.get("data"))
            visual_rows += rows
            if entry.get("minimax_video_block", False):
                video_rows += rows
            else:
                image_rows += rows

    return {
        "non_visual_rows": non_visual_rows,
        "image_rows": image_rows,
        "video_rows": video_rows,
        "visual_rows": visual_rows,
        "embedding_rows": embedding_rows,
        "total_rows": non_visual_rows + visual_rows + embedding_rows,
    }


def estimate_qwen_activation(
    plan: dict[str, int],
    settings: QwenMemorySettings,
    *,
    hidden_features: int,
    intermediate_features: int,
) -> tuple[float, float]:
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    hidden_row_mib = hidden_features * element_size / 2**20
    intermediate_row_mib = intermediate_features * element_size / 2**20
    presentation_row_mib = 3 * hidden_row_mib + 2 * intermediate_row_mib
    visual_row_mib = 3 * hidden_row_mib
    baseline_fixed_mib = (
        _BASELINE_PLANNED_ACTIVATION_MIB
        - presentation_row_mib * _BASELINE_TOTAL_ROWS
        - visual_row_mib * _BASELINE_VISUAL_ROWS
    )
    estimated_mib = (
        baseline_fixed_mib
        + presentation_row_mib * plan["total_rows"]
        + visual_row_mib * plan["visual_rows"]
    )
    # The measured baseline already includes one causal mask and both the
    # per-reference and concatenated DeepStack tensors. Add only growth above
    # that point so smaller inputs remain conservatively overestimated.
    additional_causal_mask_mib = max(
        0.0,
        (plan["total_rows"] ** 2 - _BASELINE_TOTAL_ROWS**2)
        * element_size
        / 2**20,
    )
    additional_retained_deepstack_mib = (
        max(0, plan["visual_rows"] - _BASELINE_VISUAL_ROWS)
        * 3
        * hidden_row_mib
    )
    estimated_mib += additional_causal_mask_mib + additional_retained_deepstack_mib
    return estimated_mib, estimated_mib + settings.preflight_safety_mib


class QwenBF16Controller:
    """Apply the validated BF16, layer-serial MiniMax-H3 Qwen policy."""

    def __init__(self, clip, settings: QwenMemorySettings):
        self.clip = clip
        self.settings = settings
        self._local = threading.local()
        self._active_layer = None
        self._active_device = None
        self._module_names = {
            id(module): name for name, module in clip.cond_stage_model.named_modules()
        }
        self.decoder = self._find_decoder()
        self.clip_model = self._find_clip_model()
        self.layers = self.decoder.model.layers

    def install(self) -> None:
        self.clip.patcher.set_model_compute_dtype(torch.bfloat16)
        self._install_activation_patches()
        self._install_layer_patches()
        self._wrap_clip_encoding()
        setattr(self.clip, QWEN_STATE_KEY, self)

    def _find_decoder(self):
        candidates = []
        for module in self.clip.cond_stage_model.modules():
            model = getattr(module, "model", None)
            layers = getattr(model, "layers", None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                candidates.append(module)
        if len(candidates) != 1:
            raise TypeError(
                "MiniMax H3 Qwen BF16 Offload requires the native MiniMax-H3 "
                f"Qwen encoder; found {len(candidates)} decoder candidates"
            )
        return candidates[0]

    def _find_clip_model(self):
        candidates = [
            module
            for module in self.clip.cond_stage_model.modules()
            if getattr(module, "transformer", None) is self.decoder
            and hasattr(module, "process_tokens")
        ]
        if len(candidates) != 1:
            raise TypeError(
                "MiniMax H3 Qwen BF16 Offload requires the native MiniMax-H3 "
                f"CLIP wrapper; found {len(candidates)} candidates"
            )
        return candidates[0]

    def _patch_method(self, module, name: str, method) -> None:
        module_name = self._module_names.get(id(module))
        if module_name is None:
            raise RuntimeError(f"cannot locate Qwen module for patched method {name}")
        path = f"{module_name}.{name}" if module_name else name
        self.clip.patcher.add_object_patch(path, types.MethodType(method, module))

    @staticmethod
    def _cast_nested(value, dtype):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype=dtype)
        if isinstance(value, tuple):
            return tuple(QwenBF16Controller._cast_nested(item, dtype) for item in value)
        if isinstance(value, list):
            return [QwenBF16Controller._cast_nested(item, dtype) for item in value]
        return value

    def _install_activation_patches(self) -> None:
        import comfy.text_encoders.minimax as minimax_text_encoder
        import comfy.text_encoders.qwen_vl as qwen_vl

        controller = self
        decoder = self.decoder
        visual = decoder.visual
        dtype = torch.bfloat16

        def preprocess_embed(this, embed, device):
            if embed.get("type") != "image":
                return None, None
            if embed.get("minimax_video_block", False):
                image, grid = minimax_text_encoder.process_video_block(embed["data"])
            else:
                image, grid = qwen_vl.process_qwen2vl_images(
                    embed["data"],
                    patch_size=16,
                    image_mean=[0.5, 0.5, 0.5],
                    image_std=[0.5, 0.5, 0.5],
                )
            merged, deepstack = this.visual(image.to(device=device, dtype=dtype), grid)
            return merged.to(dtype=dtype), {
                "grid": grid,
                "deepstack": [item.to(dtype=dtype) for item in deepstack],
            }

        self._patch_method(decoder, "preprocess_embed", preprocess_embed)

        original_fast_pos = visual.fast_pos_embed_interpolate
        original_rot_pos = visual.rot_pos_emb

        def fast_pos_embed(this, grid_thw):
            return original_fast_pos(grid_thw).to(dtype=dtype)

        def rot_pos_embed(this, grid_thw):
            return original_rot_pos(grid_thw).to(dtype=dtype)

        self._patch_method(visual, "fast_pos_embed_interpolate", fast_pos_embed)
        self._patch_method(visual, "rot_pos_emb", rot_pos_embed)

        decoder_model = decoder.model
        original_compute_freqs = decoder_model.compute_freqs_cis

        def compute_freqs_cis(this, position_ids, device):
            return controller._cast_nested(
                original_compute_freqs(position_ids, device), dtype
            )

        self._patch_method(decoder_model, "compute_freqs_cis", compute_freqs_cis)

        def process_tokens(this, tokens, device):
            end_token = this.special_tokens.get("end", None)
            pad_token = this.special_tokens.get("pad", -1)
            cmp_token = pad_token if end_token is None else end_token
            embeds_out = []
            attention_masks = []
            num_tokens = []
            embeds_info = []

            for token_batch in tokens:
                attention_mask = []
                token_ids = []
                other_embeds = []
                eos = False
                left_pad = False
                for index, entry in enumerate(token_batch):
                    if isinstance(entry, numbers.Integral):
                        token = int(entry)
                        if index == 0 and token == pad_token:
                            left_pad = True
                        if eos or (left_pad and token == pad_token):
                            attention_mask.append(0)
                        else:
                            attention_mask.append(1)
                            left_pad = False
                        token_ids.append(token)
                        if not eos and token == cmp_token and not left_pad:
                            if end_token is None:
                                attention_mask[-1] = 0
                            eos = True
                    else:
                        other_embeds.append((index, entry))

                token_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
                token_embeds = this.transformer.get_input_embeddings()(
                    token_tensor, out_dtype=dtype
                )
                index_offset = 0
                for original_index, embed_spec in other_embeds:
                    if torch.is_tensor(embed_spec):
                        embed_spec = {"type": "embedding", "data": embed_spec}
                    embed_type = embed_spec.get("type")
                    if embed_type == "embedding":
                        embed = embed_spec.get("data")
                        extra = None
                    else:
                        embed, extra = this.transformer.preprocess_embed(
                            embed_spec, device=device
                        )
                    if embed is None:
                        index_offset -= 1
                        continue
                    insert_at = index_offset + original_index
                    embed = embed.view(1, -1, embed.shape[-1]).to(
                        device=device, dtype=dtype
                    )
                    if embed.shape[-1] != token_embeds.shape[-1]:
                        raise RuntimeError(
                            "Qwen embedding width mismatch: "
                            f"{embed.shape[-1]} != {token_embeds.shape[-1]}"
                        )
                    embed_tokens = embed.shape[1]
                    token_embeds = torch.cat(
                        [
                            token_embeds[:, :insert_at],
                            embed,
                            token_embeds[:, insert_at:],
                        ],
                        dim=1,
                    )
                    attention_mask = (
                        attention_mask[:insert_at]
                        + [1] * embed_tokens
                        + attention_mask[insert_at:]
                    )
                    index_offset += embed_tokens - 1
                    embeds_info.append(
                        {
                            "type": embed_type,
                            "index": insert_at,
                            "size": embed_tokens,
                            "extra": extra,
                        }
                    )

                embeds_out.append(token_embeds)
                attention_masks.append(attention_mask)
                num_tokens.append(sum(attention_mask))

            return (
                torch.cat(embeds_out),
                torch.tensor(attention_masks, device=device, dtype=torch.long),
                num_tokens,
                embeds_info,
            )

        self._patch_method(self.clip_model, "process_tokens", process_tokens)

        def clip_forward(this, tokens):
            device = (
                this.transformer.get_input_embeddings().weight.device
                if this.execution_device is None
                else this.execution_device
            )
            embeds, attention_mask, num_tokens, embeds_info = this.process_tokens(
                tokens, device
            )
            attention_mask_model = attention_mask if this.enable_attention_masks else None
            if isinstance(this.layer, list):
                intermediate_output = this.layer
            elif this.layer == "all":
                intermediate_output = "all"
            else:
                intermediate_output = this.layer_idx
            outputs = this.transformer(
                None,
                attention_mask_model,
                embeds=embeds,
                num_tokens=num_tokens,
                intermediate_output=intermediate_output,
                final_layer_norm_intermediate=this.layer_norm_hidden_state,
                dtype=dtype,
                embeds_info=embeds_info,
            )
            hidden = outputs[0] if this.layer == "last" else outputs[1]
            hidden = hidden.to(dtype=dtype)
            if this.zero_out_masked:
                hidden *= attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = None
            if len(outputs) >= 3:
                if (
                    not this.return_projected_pooled
                    and len(outputs) >= 4
                    and outputs[3] is not None
                ):
                    pooled = outputs[3].to(dtype=dtype)
                elif outputs[2] is not None:
                    pooled = outputs[2].to(dtype=dtype)
            extra = {}
            if this.return_attention_masks:
                extra["attention_mask"] = attention_mask
            if extra:
                return hidden, pooled, extra
            return hidden, pooled

        self._patch_method(self.clip_model, "forward", clip_forward)

    @staticmethod
    def _first_tensor(args, kwargs):
        for value in args:
            if isinstance(value, torch.Tensor):
                return value
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _cuda_parameter_mib(self) -> float:
        total_bytes = 0
        for parameter in self.clip.cond_stage_model.parameters():
            if parameter.device.type == "cuda":
                total_bytes += parameter.numel() * parameter.element_size()
        return total_bytes / 2**20

    def _planned_activation_mib(self, layer, hidden_states) -> float:
        if hidden_states is None:
            return 0.0
        device = hidden_states.device
        allocated_mib = torch.cuda.memory_allocated(device) / 2**20
        boundary_mib = max(0.0, allocated_mib - self._cuda_parameter_mib())
        hidden_mib = hidden_states.numel() * hidden_states.element_size() / 2**20
        intermediate_features = int(layer.mlp.gate_proj.out_features)
        intermediate_mib = (
            hidden_states.shape[0]
            * hidden_states.shape[1]
            * intermediate_features
            * hidden_states.element_size()
            / 2**20
        )
        return boundary_mib + 2 * hidden_mib + 2 * intermediate_mib

    def _before_layer(self, index, layer, args, kwargs) -> None:
        if self._active_layer is not None:
            raise RuntimeError(
                f"Qwen decoder layers overlapped: {self._active_layer} and {index}"
            )
        hidden_states = self._first_tensor(args, kwargs)
        if hidden_states is None or hidden_states.device.type != "cuda":
            raise RuntimeError("MiniMax H3 Qwen BF16 Offload requires CUDA execution")
        device = hidden_states.device
        if self.settings.offload_mode == "extreme":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        planned_mib = self._planned_activation_mib(layer, hidden_states)
        if planned_mib > self.settings.activation_limit_mib:
            raise RuntimeError(
                "Qwen activation plan exceeds limit before decoder layer "
                f"{index}: {planned_mib:.2f} MiB > "
                f"{self.settings.activation_limit_mib} MiB"
            )
        self._active_layer = index
        self._active_device = device

    def _after_layer(self) -> None:
        try:
            if (
                self.settings.offload_mode == "extreme"
                and self._active_device is not None
            ):
                torch.cuda.synchronize(self._active_device)
        finally:
            self._active_layer = None
            self._active_device = None
            if self.settings.offload_mode == "extreme":
                torch.cuda.empty_cache()

    def _install_layer_patches(self) -> None:
        controller = self
        for index, layer in enumerate(self.layers):
            mlp = layer.mlp

            def low_peak_mlp(this, hidden_states):
                gate = this.activation(this.gate_proj(hidden_states))
                up = this.up_proj(hidden_states)
                gate.mul_(up)
                del up
                return this.down_proj(gate)

            self._patch_method(mlp, "forward", low_peak_mlp)
            original_forward = layer.forward

            def serial_forward(
                this,
                *args,
                _index=index,
                _layer=layer,
                _original=original_forward,
                **kwargs,
            ):
                hidden_states = controller._first_tensor(args, kwargs)
                controller._before_layer(_index, _layer, args, kwargs)
                try:
                    output = _original(*args, **kwargs)
                    layer_hidden = output[0]
                    if (
                        hidden_states is not None
                        and layer_hidden is not hidden_states
                        and layer_hidden.shape == hidden_states.shape
                        and layer_hidden.dtype == hidden_states.dtype
                    ):
                        hidden_states.copy_(layer_hidden)
                        output = (hidden_states, *output[1:])
                    return output
                finally:
                    controller._after_layer()

            self._patch_method(layer, "forward", serial_forward)

    def preflight(self, tokens) -> dict[str, int | float]:
        plan = inspect_qwen_input_tokens(tokens)
        hidden_features = int(self.decoder.get_input_embeddings().weight.shape[-1])
        intermediate_features = int(self.layers[0].mlp.gate_proj.out_features)
        estimated_mib, estimated_with_safety_mib = estimate_qwen_activation(
            plan,
            self.settings,
            hidden_features=hidden_features,
            intermediate_features=intermediate_features,
        )
        plan.update(
            {
                "estimated_activation_mib": estimated_mib,
                "estimated_with_safety_mib": estimated_with_safety_mib,
            }
        )
        failures = []
        if plan["total_rows"] > self.settings.max_conditioning_rows:
            failures.append(
                f"{plan['total_rows']} rows exceed hard limit "
                f"{self.settings.max_conditioning_rows}"
            )
        if estimated_with_safety_mib > self.settings.activation_limit_mib:
            failures.append(
                f"estimated activation {estimated_mib:.2f} MiB plus "
                f"{self.settings.preflight_safety_mib} MiB safety exceeds "
                f"{self.settings.activation_limit_mib} MiB"
            )
        if failures:
            raise RuntimeError("Qwen input rejected before encode: " + "; ".join(failures))
        return plan

    @contextmanager
    def _encoding_policy(self):
        import comfy.model_management as model_management

        with _ENCODE_POLICY_LOCK:
            original_vram_state = model_management.vram_state
            original_streams = model_management.NUM_STREAMS
            model_management.vram_state = model_management.VRAMState.NO_VRAM
            model_management.NUM_STREAMS = self.settings.offload_streams
            try:
                yield
            finally:
                model_management.NUM_STREAMS = original_streams
                model_management.vram_state = original_vram_state

    def _run_encode(self, tokens, operation):
        if getattr(self._local, "encoding", False):
            return operation()
        self.preflight(tokens)
        self._local.encoding = True
        try:
            with self._encoding_policy():
                return operation()
        finally:
            self._local.encoding = False

    def _wrap_clip_encoding(self) -> None:
        controller = self
        original_encode = self.clip.encode_from_tokens
        original_scheduled = self.clip.encode_from_tokens_scheduled

        def encode_from_tokens(this, tokens, *args, **kwargs):
            return controller._run_encode(
                tokens, lambda: original_encode(tokens, *args, **kwargs)
            )

        def encode_from_tokens_scheduled(this, tokens, *args, **kwargs):
            return controller._run_encode(
                tokens, lambda: original_scheduled(tokens, *args, **kwargs)
            )

        self.clip.encode_from_tokens = types.MethodType(encode_from_tokens, self.clip)
        self.clip.encode_from_tokens_scheduled = types.MethodType(
            encode_from_tokens_scheduled, self.clip
        )


def patch_minimax_h3_qwen_clip(
    clip,
    *,
    activation_limit_mib: int,
    max_conditioning_rows: int,
    preflight_safety_mib: int,
    offload_mode: str,
):
    settings = QwenMemorySettings(
        activation_limit_mib=int(activation_limit_mib),
        max_conditioning_rows=int(max_conditioning_rows),
        preflight_safety_mib=int(preflight_safety_mib),
        offload_mode=offload_mode,
    )
    settings.validate()
    load_device = torch.device(clip.patcher.load_device)
    if load_device.type != "cuda":
        raise ValueError(
            "MiniMax H3 Qwen BF16 Offload requires CLIPLoader device='default' "
            "with an NVIDIA CUDA load device"
        )
    patched = clip.clone()
    controller = QwenBF16Controller(patched, settings)
    controller.install()
    return patched


__all__ = [
    "QWEN_STATE_KEY",
    "QwenBF16Controller",
    "QwenMemorySettings",
    "estimate_qwen_activation",
    "inspect_qwen_input_tokens",
    "patch_minimax_h3_qwen_clip",
    "qwen_vision_merged_rows",
]
