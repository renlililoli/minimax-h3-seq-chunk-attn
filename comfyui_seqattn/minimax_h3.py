from __future__ import annotations

import hashlib
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager, nullcontext

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as native_minimax
import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy_kitchen
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
from seqattn_core import (
    H3BlockOps,
    H3MaterializedProjection,
    H3RecomputeProjection,
    H3SequenceMeta,
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


def _rope_for_tile(block, position_ids: torch.Tensor, model, tile: torch.Tensor):
    tokens = tile.shape[0]
    angles = model.rope_freqs(position_ids, tile.device)
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    c, s = torch.cos(ang), torch.sin(ang)
    rope = torch.stack([c, -s, s, c], dim=-1).reshape(
        1, tokens, 1, half, 2, 2
    ).to(tile.dtype)
    return rope


def _qkv_with_rope(block, tile: torch.Tensor, position_ids: torch.Tensor, model):
    tokens = tile.shape[0]
    heads = block.attn.heads
    head_dim = block.attn.head_dim
    q, k, v = block.attn.qkv_proj(tile).split(heads * head_dim, dim=-1)
    q = q.view(1, tokens, heads, head_dim)
    k = k.view(1, tokens, heads, head_dim)
    v = v.view(tokens, heads, head_dim)
    rope = _rope_for_tile(block, position_ids, model, tile)
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


def _single_qk_with_rope(
    block,
    projected: torch.Tensor,
    position_ids: torch.Tensor,
    model,
    *,
    norm,
) -> torch.Tensor:
    tokens = projected.shape[0]
    tensor = projected.view(1, tokens, block.attn.heads, block.attn.head_dim)
    rope = _rope_for_tile(block, position_ids, model, projected)
    weight = comfy.model_management.cast_to(norm.weight, device=projected.device)
    tensor = torch.nn.functional.rms_norm(
        tensor,
        (block.attn.head_dim,),
        weight=weight,
        eps=norm.eps,
    )
    rot = rope.shape[-3] * 2
    rotated = tensor[..., :rot]
    pairs = (
        rotated.reshape(*rotated.shape[:-1], 2, -1)
        .movedim(-2, -1)
        .unsqueeze(-2)
        .to(rope.dtype)
    )
    rotated = (rope[..., 0] * pairs[..., 0] + rope[..., 1] * pairs[..., 1])
    rotated = rotated.movedim(-1, -2).reshape_as(tensor[..., :rot]).to(tensor.dtype)
    return torch.cat((rotated, tensor[..., rot:]), dim=-1)[0]


def _recompute_support_error(qkv_module) -> str | None:
    quant_format = getattr(qkv_module, "quant_format", None)
    if quant_format != "int8_tensorwise":
        return f"quant_format={quant_format!r}"
    params = getattr(getattr(qkv_module, "weight", None), "_params", None)
    if params is None or not hasattr(params, "convrot"):
        return "unknown packed INT8 layout"
    if not bool(params.convrot):
        return "int8_tensorwise without ConvRot"
    if not hasattr(params, "convrot_groupsize"):
        return "INT8 ConvRot weight has no group-size metadata"
    return None


def _validate_recompute_blocks(blocks: list) -> None:
    unsupported = []
    for index, block in enumerate(blocks):
        reason = _recompute_support_error(block.attn.qkv_proj)
        if reason is not None:
            unsupported.append(f"block {index}: {reason}")
    if unsupported:
        details = "; ".join(unsupported[:4])
        if len(unsupported) > 4:
            details += f"; and {len(unsupported) - 4} more"
        raise RuntimeError(
            "MiniMax H3 recompute requires int8_tensorwise QKV weights with "
            f"ConvRot for every DiT block ({details}). Use execution_mode='materialized'."
        )


@contextmanager
def _recompute_qkv_lease(qkv_module, device: torch.device, dtype: torch.dtype, active: dict):
    weight, bias, offload = comfy.ops.cast_bias_weight(
        qkv_module,
        input=None,
        dtype=qkv_module.weight.dtype,
        device=device,
        bias_dtype=dtype,
        offloadable=True,
        compute_dtype=dtype,
        want_requant=True,
    )
    params = getattr(weight, "_params", None)
    if not hasattr(weight, "_qdata") or params is None:
        raise RuntimeError("recompute QKV lease did not produce a quantized INT8 weight")
    if not bool(getattr(params, "convrot", False)):
        raise RuntimeError("recompute QKV lease requires ConvRot INT8 weights")
    active.update(weight=weight, bias=bias)
    try:
        yield
    finally:
        active.clear()
        comfy.ops.uncast_bias_weight(qkv_module, weight, bias, offload)


def _project_int8_rows(
    active: dict,
    tile: torch.Tensor,
    row_start: int,
    row_stop: int,
) -> torch.Tensor:
    weight = active.get("weight")
    if weight is None:
        raise RuntimeError("recompute QKV projector called outside its weight lease")
    qdata = weight._qdata
    params = weight._params
    if qdata.ndim != 2 or not 0 <= row_start < row_stop <= qdata.shape[0]:
        raise RuntimeError(f"unexpected recompute QKV weight shape: {tuple(qdata.shape)}")
    scale = params.scale
    if scale.numel() != 1:
        if scale.shape[0] != qdata.shape[0]:
            raise RuntimeError(f"unexpected recompute QKV scale shape: {tuple(scale.shape)}")
        scale = scale[row_start:row_stop].contiguous()
    bias = active["bias"]
    if bias is not None:
        bias = bias[row_start:row_stop].contiguous()
    return comfy_kitchen.int8_linear(
        tile.contiguous(),
        qdata[row_start:row_stop].contiguous(),
        scale,
        bias,
        out_dtype=tile.dtype,
        convrot=True,
        convrot_groupsize=int(params.convrot_groupsize),
    )


def _copy_projected_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    projection,
    *,
    start: int,
    chunk_tokens: int,
    dtype: torch.dtype,
) -> None:
    device = source.device
    for offset in range(0, source.shape[0], chunk_tokens):
        stop = min(offset + chunk_tokens, source.shape[0])
        projected = projection(source[offset:stop]).to(dtype)
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
    )


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
            model, text_states, transformer_options
        )
        if refined_cache_key is None:
            runtime.record_refined_conditioning_bypass()
            refined = model.token_refiner(
                model.condition_proj(text_states),
                transformer_options=transformer_options,
            )
            text_states = _pinned_empty(refined.shape, torch.bfloat16)
            text_states.copy_(refined.to(torch.bfloat16), non_blocking=True)
            refined.record_stream(torch.cuda.current_stream(device))
        else:
            cached = runtime.refined_conditioning_for(refined_cache_key)
            if cached is not None:
                text_states = cached
            else:
                refined = model.token_refiner(
                    model.condition_proj(text_states),
                    transformer_options=transformer_options,
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
        t_emb = model.time_embedder(t_vals).to(dtype)
    return sigma_v, shift_v, shift_a, seg_t, t_row, mod_segments, t_emb


def _consumer_ops(
    model: MiniMaxH3Model,
    block,
    mod_segments: list[tuple[int, int, int]],
    modulation: tuple[torch.Tensor, ...],
) -> H3BlockOps:
    del model
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    device = shift_msa.device

    def attention_epilogue(
        attention: torch.Tensor,
        residual_hidden_host: torch.Tensor,
        start: int,
        stop: int,
    ):
        update = block.attn.out_proj(attention)
        residual = residual_hidden_host[start:stop].to(device, non_blocking=True)
        return _gate_tile(residual, update, gate_msa, mod_segments, start, stop)

    def mlp(post_attention: torch.Tensor, start: int, stop: int):
        residual = post_attention
        tile = block.norm2(residual)
        tile = _modulate_tile(tile, shift_mlp, scale_mlp, mod_segments, start, stop)
        update = block.mlp(tile)
        return _gate_tile(residual, update, gate_mlp, mod_segments, start, stop)

    return H3BlockOps(
        attention_epilogue=attention_epilogue,
        mlp=mlp,
        consumer_lease=lambda: _lease_group(
            block.attn.out_proj,
            block.mlp.fc1,
            block.mlp.fc2,
        ),
    )


def _materialized_block_parts(
    model: MiniMaxH3Model,
    block,
    layout: PackedLayout,
    mod_segments: list[tuple[int, int, int]],
    t_emb: torch.Tensor,
) -> tuple[H3MaterializedProjection, H3BlockOps]:
    modulation = block.adaln_proj(t_emb)
    shift_msa, scale_msa = modulation[:2]
    position_ids = layout.position_ids

    def project_qkv(tile: torch.Tensor, start: int, stop: int):
        tile = block.norm1(tile)
        tile = _modulate_tile(tile, shift_msa, scale_msa, mod_segments, start, stop)
        return _qkv_with_rope(block, tile, position_ids[start:stop], model)

    return (
        H3MaterializedProjection(
            project_qkv,
            weight_lease=lambda: _lease_group(block.attn.qkv_proj),
        ),
        _consumer_ops(model, block, mod_segments, modulation),
    )


def _recompute_block_parts(
    model: MiniMaxH3Model,
    block,
    layout: PackedLayout,
    mod_segments: list[tuple[int, int, int]],
    t_emb: torch.Tensor,
) -> tuple[H3RecomputeProjection, H3BlockOps]:
    modulation = block.adaln_proj(t_emb)
    shift_msa, scale_msa = modulation[:2]
    position_ids = layout.position_ids
    attention_features = block.attn.heads * block.attn.head_dim
    active_qkv = {}

    def normalized(tile: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        tile = block.norm1(tile)
        return _modulate_tile(tile, shift_msa, scale_msa, mod_segments, start, stop)

    def project_q(
        tile: torch.Tensor,
        destination_q: torch.Tensor,
        start: int,
        stop: int,
    ) -> None:
        tile = normalized(tile, start, stop)
        q = _project_int8_rows(active_qkv, tile, 0, attention_features)
        q = _single_qk_with_rope(
            block,
            q,
            position_ids[start:stop],
            model,
            norm=block.attn.q_norm,
        )
        destination_q.copy_(q)

    def project_kv(
        tile: torch.Tensor,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        start: int,
        stop: int,
    ) -> None:
        tile = normalized(tile, start, stop)
        kv = _project_int8_rows(
            active_qkv,
            tile,
            attention_features,
            3 * attention_features,
        )
        k, v = kv.split(attention_features, dim=-1)
        k = _single_qk_with_rope(
            block,
            k,
            position_ids[start:stop],
            model,
            norm=block.attn.k_norm,
        )
        destination_k.copy_(k)
        destination_v.copy_(
            v.view(tile.shape[0], block.attn.heads, block.attn.head_dim)
        )

    return (
        H3RecomputeProjection(
            project_q,
            project_kv,
            weight_lease=lambda: _recompute_qkv_lease(
                block.attn.qkv_proj,
                t_emb.device,
                t_emb.dtype,
                active_qkv,
            ),
        ),
        _consumer_ops(model, block, mod_segments, modulation),
    )


def _run_dit_blocks(
    *,
    runner,
    blocks: list,
    device: torch.device,
    sequence_meta: H3SequenceMeta,
    current_hidden: torch.Tensor,
    scratch_hidden: torch.Tensor | None,
    execution_mode: str,
    parts_for,
    softmax_scale: float,
    record,
) -> torch.Tensor:
    def compute_block(index: int) -> None:
        nonlocal current_hidden, scratch_hidden
        projection, ops = parts_for(index)
        if execution_mode == "recompute":
            if scratch_hidden is None:
                raise RuntimeError("recompute requires a scratch hidden buffer")
            result = runner.run_block(
                current_hidden,
                scratch_hidden,
                sequence_meta,
                projection,
                ops,
                softmax_scale=softmax_scale,
            )
            current_hidden, scratch_hidden = result, current_hidden
        else:
            current_hidden = runner.run_block_(
                current_hidden,
                sequence_meta,
                projection,
                ops,
                softmax_scale=softmax_scale,
            )

    run_weight_stages(blocks, device, compute_block, record=record)
    return current_hidden


def _final_rows(
    model: MiniMaxH3Model,
    hidden: torch.Tensor,
    t_emb: torch.Tensor,
    segment: tuple[int, int, int],
    projection,
    chunk_tokens: int,
) -> torch.Tensor:
    start, stop, row = segment
    device = t_emb.device
    shift, scale = model.final_layer.adaln_proj(t_emb)
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
        output[offset - start : end - start] = projection(tile)
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
        )
        sequence_meta = H3SequenceMeta(
            cu_seqlens=torch.tensor([0, layout.seq_len], dtype=torch.int32)
        )

        blocks = list(model.blocks)
        if runtime.settings.execution_mode == "recompute":
            _validate_recompute_blocks(blocks)
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
            current_hidden = hidden_a
            scratch_hidden = (
                _pinned_empty(tuple(hidden_a.shape), hidden_a.dtype)
                if runtime.settings.execution_mode == "recompute"
                else None
            )

            def parts_for(index: int):
                block = blocks[index]
                if runtime.settings.execution_mode == "recompute":
                    return _recompute_block_parts(
                        model,
                        block,
                        layout,
                        mod_segments,
                        t_emb,
                    )
                return _materialized_block_parts(
                    model,
                    block,
                    layout,
                    mod_segments,
                    t_emb,
                )

            hidden_a = _run_dit_blocks(
                runner=runner,
                blocks=blocks,
                device=video_x.device,
                sequence_meta=sequence_meta,
                current_hidden=current_hidden,
                scratch_hidden=scratch_hidden,
                execution_mode=runtime.settings.execution_mode,
                parts_for=parts_for,
                softmax_scale=first_block.attn.head_dim**-0.5,
                record=runtime.record_weight_schedule,
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
        video_rows = _final_rows(
            model, hidden_a, t_emb, video_seg, model.final_layer.video_out, chunk
        )
        audio_rows = _final_rows(
            model, hidden_a, t_emb, audio_seg, model.final_layer.audio_out, chunk
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
