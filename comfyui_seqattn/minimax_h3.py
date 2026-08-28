from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager, nullcontext

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as native_minimax
import comfy.model_management
import comfy.quant_ops
import torch
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    MiniMaxH3Model,
    PackedLayout,
    pack_audio,
    patchify_video,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from seqattn_core import H3BlockOps, H3SequenceMeta

from .lora import (
    EMBEDDING_STAGE,
    FINAL_STAGE,
    LoRAStageStreamer,
    linear_input_act_with_lora,
    linear_with_lora,
    prepared_lora_stage,
)
from .runtime import SeqAttnRuntime
from .weight_stream import run_weight_stages


def _pinned_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _lease(module):
    lease = getattr(module, "computation_lease", None)
    return nullcontext(module) if lease is None else lease(allow_preparing=True)


@contextmanager
def _lease_group(*modules):
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(_lease(module))
        yield


def _segment_intersections(
    segments: Iterable[tuple[int, int, int]], start: int, stop: int
):
    for seg_start, seg_stop, row in segments:
        a = max(start, seg_start)
        b = min(stop, seg_stop)
        if a < b:
            yield a - start, b - start, row


def _modulate_tile(
    tile: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: list[tuple[int, int, int]],
    start: int,
    stop: int,
) -> torch.Tensor:
    for a, b, row in _segment_intersections(segments, start, stop):
        tile[a:b].mul_(1.0 + scale[row].to(tile.dtype)).add_(shift[row].to(tile.dtype))
    return tile


def _gate_tile(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    segments: list[tuple[int, int, int]],
    start: int,
    stop: int,
) -> torch.Tensor:
    for a, b, row in _segment_intersections(segments, start, stop):
        residual[a:b].addcmul_(update[a:b], gate[row].to(update.dtype))
    return residual


def _audio_velocity(
    audio_rows: torch.Tensor,
    audio_x: torch.Tensor,
    sigma_v: torch.Tensor,
    shift_v: float,
    shift_a: float,
) -> torch.Tensor:
    audio_velocity = -unpack_audio(audio_rows).to(audio_x.dtype)
    slope_fn = getattr(native_minimax, "time_shift_slope", None)
    if slope_fn is not None:
        slope = slope_fn(sigma_v, shift_v, shift_a).to(audio_velocity.dtype)
        audio_velocity.mul_(slope)
    return audio_velocity


def _qkv_with_rope(
    block,
    tile: torch.Tensor,
    position_ids: torch.Tensor,
    model,
    adapters,
    target: str,
):
    tokens = tile.shape[0]
    heads = block.attn.heads
    head_dim = block.attn.head_dim
    q, k, v = linear_with_lora(
        block.attn.qkv_proj,
        tile,
        adapters.get(target, ()),
    ).split(heads * head_dim, dim=-1)
    q = q.view(1, tokens, heads, head_dim)
    k = k.view(1, tokens, heads, head_dim)
    v = v.view(tokens, heads, head_dim)

    angles = model.rope_freqs(position_ids, tile.device)
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    c, s = torch.cos(ang), torch.sin(ang)
    rope = torch.stack([c, -s, s, c], dim=-1).reshape(
        1, tokens, 1, half, 2, 2
    ).to(tile.dtype)
    qw = comfy.model_management.cast_to(block.attn.q_norm.weight, device=tile.device)
    kw = comfy.model_management.cast_to(block.attn.k_norm.weight, device=tile.device)
    rot = rope.shape[-3] * 2
    if comfy.model_management.in_training:
        q, k = comfy.quant_ops.ck.rms_rope_split_half(
            q,
            k,
            rope,
            qw,
            kw,
            epsilon=block.attn.q_norm.eps,
            rot_dim=rot,
        )
    else:
        comfy.quant_ops.ck.rms_rope_split_half_(
            q,
            k,
            rope,
            qw,
            kw,
            epsilon=block.attn.q_norm.eps,
            rot_dim=rot,
        )
    return q[0], k[0], v


def _copy_projected_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    projection,
    *,
    start: int,
    chunk_tokens: int,
    dtype: torch.dtype,
    adapters=(),
) -> None:
    device = source.device
    for offset in range(0, source.shape[0], chunk_tokens):
        stop = min(offset + chunk_tokens, source.shape[0])
        projected = linear_with_lora(
            projection,
            source[offset:stop],
            adapters,
        ).to(dtype)
        destination[start + offset : start + stop].copy_(projected, non_blocking=True)
        projected.record_stream(torch.cuda.current_stream(device))


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None


def _refiner_parameter_signature(model: MiniMaxH3Model) -> tuple:
    signature = []
    for module in (model.condition_proj, model.token_refiner):
        tensors = list(module.parameters()) + list(module.buffers())
        signature.extend(
            (
                id(tensor),
                _tensor_version(tensor),
                tuple(tensor.shape),
                tensor.dtype,
            )
            for tensor in tensors
        )
    return tuple(signature)


def _tensor_content_digest(tensor: torch.Tensor) -> bytes:
    host = tensor.detach().contiguous()
    if host.device.type != "cpu":
        host = host.cpu()
    raw = host.view(torch.uint8).numpy()
    return hashlib.blake2b(memoryview(raw), digest_size=16).digest()


def _refined_conditioning_cache_key(
    model: MiniMaxH3Model,
    text_states: torch.Tensor,
    transformer_options: dict,
    lora_signature: tuple = (),
) -> tuple | None:
    if (
        transformer_options.get("patches")
        or transformer_options.get("patches_replace")
        or "optimized_attention_override" in transformer_options
    ):
        return None

    uuids = transformer_options.get("uuids")
    if uuids:
        conditioning_identity = (
            "comfy-uuids",
            tuple(str(value) for value in uuids),
            tuple(transformer_options.get("cond_or_uncond", ())),
        )
    else:
        conditioning_identity = (
            "content",
            _tensor_content_digest(text_states),
        )

    device = text_states.device
    return (
        conditioning_identity,
        tuple(text_states.shape),
        tuple(text_states.stride()),
        text_states.dtype,
        device.type,
        device.index,
        _tensor_version(text_states),
        model.hidden_size,
        _refiner_parameter_signature(model),
        lora_signature,
    )


def _time_embedder_with_lora(model, t: torch.Tensor, adapters):
    embedder = model.time_embedder
    if not (
        adapters.get("time_embedder.proj_in")
        or adapters.get("time_embedder.proj_out")
    ):
        return embedder(t)
    half = embedder.freq_dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = linear_with_lora(
        embedder.proj_in,
        emb,
        adapters.get("time_embedder.proj_in", ()),
    )
    return linear_with_lora(
        embedder.proj_out,
        torch.nn.functional.silu(hidden),
        adapters.get("time_embedder.proj_out", ()),
    )


def _adaln_with_lora(module, t_emb: torch.Tensor, adapters, target: str):
    if not adapters.get(target):
        return module(t_emb)
    value = torch.nn.functional.silu(t_emb) if module.apply_silu else t_emb
    value = linear_with_lora(module.linear, value, adapters.get(target, ()))
    value = value.view(value.shape[0] * module.modalities, module.expand * module.hidden)
    return value.chunk(module.expand, dim=-1)


def _refiner_attention(block, x: torch.Tensor, transformer_options: dict, adapters, prefix: str):
    attention = block.attn
    tokens = x.shape[0]
    qkv = linear_with_lora(
        attention.qkv_proj,
        x,
        adapters.get(f"{prefix}.attn.qkv_proj", ()),
    )
    q, k, v = qkv.split(attention.heads * attention.head_dim, dim=-1)
    q = attention.q_norm(q.view(tokens, attention.heads, attention.head_dim))
    k = attention.k_norm(k.view(tokens, attention.heads, attention.head_dim))
    v = v.view(tokens, attention.heads, attention.head_dim)
    output = native_minimax.optimized_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attention.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return linear_with_lora(
        attention.out_proj,
        output.squeeze(0),
        adapters.get(f"{prefix}.attn.out_proj", ()),
    )


def _refine_tokens(model, x: torch.Tensor, transformer_options: dict, adapters):
    if not any(target.startswith("token_refiner.blocks.") for target in adapters):
        return model.token_refiner(x, transformer_options=transformer_options)
    for index, block in enumerate(model.token_refiner.blocks):
        prefix = f"token_refiner.blocks.{index}"
        attention = _refiner_attention(
            block,
            block.norm1(x),
            transformer_options,
            adapters,
            prefix,
        )
        x = attention.add_(x)
        hidden = linear_with_lora(
            block.mlp.fc1,
            block.norm2(x),
            adapters.get(f"{prefix}.mlp.fc1", ()),
        )
        update = linear_input_act_with_lora(
            block.mlp.fc2,
            hidden,
            "swiglu",
            adapters.get(f"{prefix}.mlp.fc2", ()),
        )
        x = update.add_(x)
    return model.token_refiner.final_norm(x)


def _embed_packed_hidden(
    model: MiniMaxH3Model,
    runtime: SeqAttnRuntime,
    layout: PackedLayout,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    context: torch.Tensor,
    payload: dict,
    transformer_options: dict,
    dtype: torch.dtype,
    chunk_tokens: int,
    adapters,
) -> torch.Tensor:
    device = video_x.device
    hidden = _pinned_empty((layout.seq_len, model.hidden_size), dtype)

    video_rows = patchify_video(video_x.to(torch.float32), model.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = model._cond_video_rows(payload, device)
    cond_audio_rows = model._cond_audio_rows(payload, device)

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    if cond_video_rows is None:
        all_video_rows = video_rows
    else:
        all_video_rows = torch.empty(
            (img_update.numel(), video_rows.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    if cond_audio_rows is None:
        all_audio_rows = audio_rows
    else:
        all_audio_rows = torch.empty(
            (audio_update.numel(), audio_rows.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    text_states = context[0]
    runtime.record_refined_conditioning_forward(
        text_states.shape[-1] != model.hidden_size
    )
    refined_cache_key = None
    publish_refined_cache = False
    if text_states.shape[-1] != model.hidden_size:
        refined_cache_key = _refined_conditioning_cache_key(
            model,
            text_states,
            transformer_options,
            runtime.lora_state.signature,
        )
        if refined_cache_key is None:
            runtime.record_refined_conditioning_bypass()
            refined = _refine_tokens(
                model,
                linear_with_lora(
                    model.condition_proj,
                    text_states,
                    adapters.get("condition_proj", ()),
                ),
                transformer_options,
                adapters,
            )
            text_states = _pinned_empty(refined.shape, torch.bfloat16)
            text_states.copy_(refined.to(torch.bfloat16), non_blocking=True)
            refined.record_stream(torch.cuda.current_stream(device))
        else:
            cached = runtime.refined_conditioning_for(refined_cache_key)
            if cached is not None:
                text_states = cached
            else:
                refined = _refine_tokens(
                    model,
                    linear_with_lora(
                        model.condition_proj,
                        text_states,
                        adapters.get("condition_proj", ()),
                    ),
                    transformer_options,
                    adapters,
                )
                text_states = _pinned_empty(refined.shape, torch.bfloat16)
                text_states.copy_(refined.to(torch.bfloat16), non_blocking=True)
                refined.record_stream(torch.cuda.current_stream(device))
                publish_refined_cache = True

    video_offset = 0
    audio_offset = 0
    deferred_text_segments = []
    for start, stop, kind in layout.segments:
        count = stop - start
        if kind == "text":
            if text_states.device.type == "cpu":
                deferred_text_segments.append((start, stop))
            else:
                hidden[start:stop].copy_(text_states.to(dtype), non_blocking=True)
        elif kind in ("cond", "ref_img", "video"):
            rows = all_video_rows[video_offset : video_offset + count]
            _copy_projected_rows(
                hidden,
                rows,
                model.video_patch_proj,
                start=start,
                chunk_tokens=chunk_tokens,
                dtype=dtype,
                adapters=adapters.get("video_patch_proj", ()),
            )
            video_offset += count
        else:
            rows = all_audio_rows[audio_offset : audio_offset + count]
            _copy_projected_rows(
                hidden,
                rows,
                model.audio_patch_proj,
                start=start,
                chunk_tokens=chunk_tokens,
                dtype=dtype,
                adapters=adapters.get("audio_patch_proj", ()),
            )
            audio_offset += count
    torch.cuda.synchronize(device)
    if publish_refined_cache:
        runtime.store_refined_conditioning(refined_cache_key, text_states)
    for start, stop in deferred_text_segments:
        hidden[start:stop].copy_(text_states)
    return hidden


def _time_and_modality(
    model: MiniMaxH3Model,
    layout: PackedLayout,
    timestep: torch.Tensor,
    payload: dict,
    transformer_options: dict,
    device: torch.device,
    dtype: torch.dtype,
    adapters,
):
    shift_v = float(
        transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video)
    )
    shift_a = float(
        transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio)
    )
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
    has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted(
        {t_v, t_a}
        | ({seg_t["cond"]} if has_vis_cond else set())
        | ({seg_t["ref_audio"]} if has_aud_cond else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "ref_audio": 2,
    }

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, stop, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, stop - start + 1):
                if index == stop - start or tags[index] != tags[run_start]:
                    mod_segments.append(
                        (start + run_start, start + index, row_base + int(tags[run_start]))
                    )
                    run_start = index
        else:
            mod_segments.append((start, stop, row_base + seg_tag[kind]))

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if model.use_adaln_curves:
        table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = _time_embedder_with_lora(model, t_vals, adapters).to(dtype)
    return sigma_v, shift_v, shift_a, seg_t, t_row, mod_segments, t_emb


def _block_ops(
    model: MiniMaxH3Model,
    block,
    block_index: int,
    hidden: torch.Tensor,
    layout: PackedLayout,
    mod_segments: list[tuple[int, int, int]],
    t_emb: torch.Tensor,
    adapters,
) -> H3BlockOps:
    device = t_emb.device
    prefix = f"blocks.{block_index}"
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _adaln_with_lora(
            block.adaln_proj,
            t_emb,
            adapters,
            f"{prefix}.adaln_proj.linear",
        )
    )
    position_ids = layout.position_ids

    def project_qkv(tile: torch.Tensor, start: int, stop: int):
        tile = block.norm1(tile)
        tile = _modulate_tile(
            tile, shift_msa, scale_msa, mod_segments, start, stop
        )
        return _qkv_with_rope(
            block,
            tile,
            position_ids[start:stop],
            model,
            adapters,
            f"{prefix}.attn.qkv_proj",
        )

    def attention_epilogue(attention: torch.Tensor, start: int, stop: int):
        update = linear_with_lora(
            block.attn.out_proj,
            attention,
            adapters.get(f"{prefix}.attn.out_proj", ()),
        )
        residual = hidden[start:stop].to(device, non_blocking=True)
        return _gate_tile(
            residual, update, gate_msa, mod_segments, start, stop
        )

    def mlp(post_attention: torch.Tensor, start: int, stop: int):
        residual = post_attention
        tile = block.norm2(residual)
        tile = _modulate_tile(
            tile, shift_mlp, scale_mlp, mod_segments, start, stop
        )
        fc1_adapters = adapters.get(f"{prefix}.mlp.fc1", ())
        fc2_adapters = adapters.get(f"{prefix}.mlp.fc2", ())
        if not fc1_adapters and not fc2_adapters:
            update = block.mlp(tile)
        else:
            hidden_mlp = linear_with_lora(block.mlp.fc1, tile, fc1_adapters)
            update = linear_input_act_with_lora(
                block.mlp.fc2,
                hidden_mlp,
                "swiglu",
                fc2_adapters,
            )
        return _gate_tile(
            residual, update, gate_mlp, mod_segments, start, stop
        )

    return H3BlockOps(
        project_qkv=project_qkv,
        attention_epilogue=attention_epilogue,
        mlp=mlp,
        qkv_lease=lambda: _lease_group(block.attn.qkv_proj),
        consumer_lease=lambda: _lease_group(
            block.attn.out_proj,
            block.mlp.fc1,
            block.mlp.fc2,
        ),
    )


def _final_rows(
    model: MiniMaxH3Model,
    hidden: torch.Tensor,
    t_emb: torch.Tensor,
    segment: tuple[int, int, int],
    projection,
    projection_target: str,
    chunk_tokens: int,
    adapters,
) -> torch.Tensor:
    start, stop, row = segment
    device = t_emb.device
    shift, scale = _adaln_with_lora(
        model.final_layer.adaln_proj,
        t_emb,
        adapters,
        "final_layer.adaln_proj.linear",
    )
    output = torch.empty(
        (stop - start, projection.out_features),
        dtype=torch.float32,
        device=device,
    )
    for offset in range(start, stop, chunk_tokens):
        end = min(offset + chunk_tokens, stop)
        tile = hidden[offset:end].to(device, non_blocking=True)
        tile = (
            model.final_layer.norm(tile) * (1.0 + scale[row]) + shift[row]
        ).to(torch.float32)
        output[offset - start : end - start] = linear_with_lora(
            projection,
            tile,
            adapters.get(projection_target, ()),
        )
    return output


@torch.inference_mode()
def streaming_minimax_h3_forward(
    model: MiniMaxH3Model,
    runtime: SeqAttnRuntime,
    x,
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict | None = None,
    minimax_payload: dict | None = None,
    **kwargs,
):
    del kwargs
    transformer_options = transformer_options or {}
    payload = minimax_payload or {}
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax H3 SeqAttn requires CUDA")
    video_x, audio_x = x[0], x[1]
    if video_x.device.type != "cuda":
        raise ValueError("MiniMax H3 SeqAttn requires CUDA-resident input latents")
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 SeqAttn supports batch size 1")
    dtype = context.dtype
    if dtype != torch.bfloat16:
        raise ValueError(
            "MiniMax H3 SeqAttn requires BF16 activations; ComfyUI-native "
            f"quantized or BF16 weights are supported, got activation dtype {dtype}"
        )
    if transformer_options.get("patches_replace", {}).get("dit"):
        raise ValueError(
            "MiniMax H3 SeqAttn does not yet support patches_replace['dit']"
        )

    original_t, original_h, original_w = video_x.shape[2:]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    latent_t, latent_h, latent_w = video_x.shape[2:]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    signature = (text_len, latent_t, latent_h, latent_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = PackedLayout(
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )

    with runtime.lock:
        embedding_plan = runtime.lora_state.plan_for(
            EMBEDDING_STAGE,
            activation_dtype=dtype,
        )
        with prepared_lora_stage(embedding_plan, video_x.device) as adapters:
            (
                sigma_v,
                shift_v,
                shift_a,
                seg_t,
                t_row,
                mod_segments,
                t_emb,
            ) = _time_and_modality(
                model,
                layout,
                timestep,
                payload,
                transformer_options,
                video_x.device,
                dtype,
                adapters,
            )
            hidden_a = _embed_packed_hidden(
                model,
                runtime,
                layout,
                video_x,
                audio_x,
                context,
                payload,
                transformer_options,
                dtype,
                runtime.settings.qkv_tile_tokens,
                adapters,
            )
        sequence_meta = H3SequenceMeta(
            cu_seqlens=torch.tensor([0, layout.seq_len], dtype=torch.int32)
        )

        blocks = list(model.blocks)
        if blocks:
            first_block = blocks[0]
            runner = runtime.dit_runner_for(
                tokens=layout.seq_len,
                hidden_features=model.hidden_size,
                heads=first_block.attn.heads,
                head_dim=first_block.attn.head_dim,
                dtype=hidden_a.dtype,
                device=video_x.device,
            )
            lora_streamer = LoRAStageStreamer(
                [
                    runtime.lora_state.plan_for(
                        ("block", index),
                        activation_dtype=dtype,
                    )
                    for index in range(len(blocks))
                ],
                video_x.device,
            )

            def compute_block(index: int) -> None:
                block = blocks[index]
                adapters = lora_streamer.adapters_for(index)
                runner.run_block_(
                    hidden_a,
                    sequence_meta,
                    _block_ops(
                        model,
                        block,
                        index,
                        hidden_a,
                        layout,
                        mod_segments,
                        t_emb,
                        adapters,
                    ),
                    softmax_scale=first_block.attn.head_dim**-0.5,
                )

            run_weight_stages(
                blocks,
                video_x.device,
                compute_block,
                record=runtime.record_weight_schedule,
                auxiliary=lora_streamer,
            )

        video_seg = next(
            (a, b, t_row[seg_t["video"]])
            for a, b, kind in layout.segments
            if kind == "video"
        )
        audio_seg = next(
            (a, b, t_row[seg_t["audio"]])
            for a, b, kind in layout.segments
            if kind == "audio"
        )
        chunk = runtime.settings.qkv_tile_tokens
        final_plan = runtime.lora_state.plan_for(
            FINAL_STAGE,
            activation_dtype=dtype,
        )
        with prepared_lora_stage(final_plan, video_x.device) as adapters:
            video_rows = _final_rows(
                model,
                hidden_a,
                t_emb,
                video_seg,
                model.final_layer.video_out,
                "final_layer.video_out",
                chunk,
                adapters,
            )
            audio_rows = _final_rows(
                model,
                hidden_a,
                t_emb,
                audio_seg,
                model.final_layer.audio_out,
                "final_layer.audio_out",
                chunk,
                adapters,
            )

        video_out = unpatchify_video(
            video_rows,
            latent_t,
            latent_h // 2,
            latent_w // 2,
            model.latents_dim,
            model.patch_size,
        )
        video_out = video_out[:, :, :original_t, :original_h, :original_w]
        return [
            -video_out.to(video_x.dtype),
            _audio_velocity(audio_rows, audio_x, sigma_v, shift_v, shift_a),
        ]


__all__ = ["streaming_minimax_h3_forward"]
