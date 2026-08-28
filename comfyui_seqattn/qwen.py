from __future__ import annotations

import gc
import math
import numbers
import threading
import types
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from seqattn_core import (
    H3BlockOps,
    H3MaterializedProjection,
    H3MaterializedRunner,
    H3SequenceMeta,
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
)

from .config import load_attention_stage_config
from .weight_stream import run_weight_stages

QWEN_SEQATTN_STATE_KEY = "minimax_h3_qwen_seqattn"

_ENCODE_LOCK = threading.RLock()
_PAD_TOKEN = 151643


def qwen_vision_merged_rows(data: torch.Tensor) -> int:
    if not isinstance(data, torch.Tensor) or data.ndim != 4:
        raise ValueError("Qwen visual input must have shape [T, H, W, C]")
    _, height, width, channels = data.shape
    if channels != 3 or height <= 0 or width <= 0:
        raise ValueError(f"invalid Qwen visual input shape: {list(data.shape)}")
    factor = 32
    min_pixels = 3136
    max_pixels = 12845056
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(
            factor, math.floor(height / scale / factor) * factor
        )
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return (resized_height // factor) * (resized_width // factor)


@dataclass(frozen=True)
class QwenSeqAttnSettings:
    q_chunk_tokens: int = 5760
    kv_chunk_tokens: int = 4096
    qkv_tile_tokens: int = 4096
    mlp_tile_tokens: int = 4096

    @classmethod
    def from_config(
        cls, *, q_chunk_tokens: int, kv_chunk_tokens: int
    ) -> QwenSeqAttnSettings:
        config = load_attention_stage_config("minimax_h3_qwen")
        return cls(
            q_chunk_tokens=int(q_chunk_tokens),
            kv_chunk_tokens=int(kv_chunk_tokens),
            qkv_tile_tokens=config.qkv_tile_tokens,
            mlp_tile_tokens=config.mlp_tile_tokens,
        )

    def validate(self) -> None:
        for name, value in (
            ("q_chunk_tokens", self.q_chunk_tokens),
            ("kv_chunk_tokens", self.kv_chunk_tokens),
            ("qkv_tile_tokens", self.qkv_tile_tokens),
            ("mlp_tile_tokens", self.mlp_tile_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class QwenInputSpan:
    kind: str
    start: int
    stop: int
    entry: object


@dataclass(frozen=True)
class QwenPresentationLayout:
    spans: tuple[QwenInputSpan, ...]
    attention_mask: tuple[int, ...]
    token_tags: tuple[int, ...]
    total_rows: int
    visual_rows: int

    @property
    def visual_spans(self) -> tuple[QwenInputSpan, ...]:
        return tuple(span for span in self.spans if span.kind == "visual")


@dataclass(frozen=True)
class PreparedVisual:
    layout_span: QwenInputSpan
    grid: torch.Tensor
    raw_start: int
    raw_stop: int
    merged_start: int
    merged_stop: int


@dataclass
class PreparedVisionBatch:
    patches: torch.Tensor
    grids: torch.Tensor
    cu_seqlens: torch.Tensor
    visuals: list[PreparedVisual]


def _token_batches(tokens) -> list:
    if isinstance(tokens, dict):
        if len(tokens) != 1:
            raise ValueError(f"expected one Qwen token stream, found {len(tokens)}")
        tokens = next(iter(tokens.values()))
    if not isinstance(tokens, list) or len(tokens) != 1:
        raise ValueError("MiniMax H3 Qwen SeqAttn supports batch size 1")
    return tokens


def _entry_value(entry):
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def _embedding_rows(entry) -> int:
    data = entry if isinstance(entry, torch.Tensor) else entry.get("data")
    if not isinstance(data, torch.Tensor) or data.ndim == 0:
        raise ValueError("Qwen embedding input must have a row axis")
    return int(data.numel() // data.shape[-1])


def build_qwen_presentation_layout(tokens) -> QwenPresentationLayout:
    batch = _token_batches(tokens)[0]
    spans = []
    attention_mask = []
    cursor = 0
    visual_rows = 0
    eos = False
    left_pad = False

    for index, weighted_entry in enumerate(batch):
        entry = _entry_value(weighted_entry)
        if isinstance(entry, numbers.Integral):
            token = int(entry)
            if index == 0 and token == _PAD_TOKEN:
                left_pad = True
            active = not eos and not (left_pad and token == _PAD_TOKEN)
            mask_value = int(active)
            if active:
                left_pad = False
            if not eos and token == _PAD_TOKEN and not left_pad:
                mask_value = 0
                eos = True
            spans.append(QwenInputSpan("token", cursor, cursor + 1, token))
            attention_mask.append(mask_value)
            cursor += 1
            continue

        if isinstance(entry, torch.Tensor):
            rows = _embedding_rows(entry)
            spans.append(QwenInputSpan("embedding", cursor, cursor + rows, entry))
        elif isinstance(entry, dict) and entry.get("type") == "embedding":
            rows = _embedding_rows(entry)
            spans.append(QwenInputSpan("embedding", cursor, cursor + rows, entry))
        elif isinstance(entry, dict) and entry.get("type") == "image":
            rows = qwen_vision_merged_rows(entry.get("data"))
            spans.append(QwenInputSpan("visual", cursor, cursor + rows, entry))
            visual_rows += rows
        else:
            entry_type = entry.get("type") if isinstance(entry, dict) else None
            raise ValueError(f"cannot stream Qwen entry type {entry_type!r}")
        attention_mask.extend([1] * rows)
        cursor += rows

    token_tags = [1] * cursor
    for span in spans:
        if span.kind != "visual":
            continue
        for row in range(max(0, span.start - 1), min(cursor, span.stop + 1)):
            token_tags[row] = 0

    return QwenPresentationLayout(
        spans=tuple(spans),
        attention_mask=tuple(attention_mask),
        token_tags=tuple(token_tags),
        total_rows=cursor,
        visual_rows=visual_rows,
    )


def packed_vision_cu_seqlens(layout: QwenPresentationLayout) -> torch.Tensor:
    bounds = [0]
    for span in layout.visual_spans:
        bounds.append(bounds[-1] + (span.stop - span.start) * 4)
    return torch.tensor(bounds, dtype=torch.int32)


def inject_deepstack_cpu_(
    hidden: torch.Tensor,
    deepstack: torch.Tensor,
    visuals: list[PreparedVisual],
    *,
    chunk_tokens: int,
) -> None:
    if hidden.device.type != "cpu" or deepstack.device.type != "cpu":
        raise ValueError("DeepStack CPU injection requires CPU tensors")
    if hidden.dtype != deepstack.dtype or hidden.shape[1] != deepstack.shape[1]:
        raise ValueError("DeepStack feature layout must match hidden")
    for visual in visuals:
        rows = visual.layout_span.stop - visual.layout_span.start
        if rows != visual.merged_stop - visual.merged_start:
            raise ValueError("DeepStack visual span mapping is inconsistent")
        for offset in range(0, rows, chunk_tokens):
            stop = min(offset + chunk_tokens, rows)
            destination = hidden[
                visual.layout_span.start + offset : visual.layout_span.start + stop
            ]
            source = deepstack[
                visual.merged_start + offset : visual.merged_start + stop
            ]
            torch.add(destination, source, out=destination)


def _pinned_empty(shape, dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


@contextmanager
def _lease_group(*modules):
    with ExitStack() as stack:
        seen = set()
        for root in modules:
            if root is None:
                continue
            descendants = root.modules() if hasattr(root, "modules") else (root,)
            for module in descendants:
                lease = getattr(module, "computation_lease", None)
                if lease is None or id(module) in seen:
                    continue
                seen.add(id(module))
                stack.enter_context(lease(allow_preparing=True))
        yield


class QwenEncodeRuntime:
    def __init__(self, settings: QwenSeqAttnSettings, device: torch.device):
        self.settings = settings
        self.device = torch.device(device)
        self.runners: list[H3MaterializedRunner] = []
        self.closed = False

    def runner(
        self,
        *,
        tokens: int,
        hidden_features: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> H3MaterializedRunner:
        attention_config = StreamingAttentionConfig(
            q_chunk_tokens=self.settings.q_chunk_tokens,
            kv_chunk_tokens=self.settings.kv_chunk_tokens,
            output_mode="device_consumer",
            backend=None,
            require_pinned=True,
            pin_output=True,
        )
        plan = build_plan(
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=self.device,
            max_q_tokens=tokens,
            max_kv_tokens=tokens,
            config=attention_config,
        )
        projected = ProjectedAttentionRunner(
            plan,
            attention_config=attention_config,
            pipeline_config=ProjectionPipelineConfig(
                projection_chunk_tokens=self.settings.qkv_tile_tokens,
                require_pinned_hidden=True,
                pin_qkv=True,
                pin_output=True,
            ),
        )
        runner = H3MaterializedRunner(
            projected,
            hidden_features=hidden_features,
            mlp_chunk_tokens=self.settings.mlp_tile_tokens,
            num_final_output_buffers=2,
        )
        self.runners.append(runner)
        return runner

    def release_runner(self, runner: H3MaterializedRunner) -> None:
        try:
            self.runners.remove(runner)
        except ValueError:
            return

    @property
    def active_runner_count(self) -> int:
        return len(self.runners)

    def close(self) -> None:
        self.runners.clear()
        self.closed = True


def _prepare_visual_batch(layout: QwenPresentationLayout) -> PreparedVisionBatch | None:
    visual_spans = layout.visual_spans
    if not visual_spans:
        return None

    import comfy.text_encoders.minimax as minimax_text_encoder
    import comfy.text_encoders.qwen_vl as qwen_vl

    prepared = []
    total_raw_rows = 0
    merged_cursor = 0
    patch_features = None
    for span in visual_spans:
        entry = span.entry
        data = entry["data"].detach().to(device="cpu")
        if entry.get("minimax_video_block", False):
            patches, grid = minimax_text_encoder.process_video_block(data)
        else:
            patches, grid = qwen_vl.process_qwen2vl_images(
                data,
                patch_size=16,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        patches = patches.to(device="cpu", dtype=torch.float32).contiguous()
        grid = grid.to(device="cpu", dtype=torch.long).contiguous()
        merged_rows = span.stop - span.start
        if patches.shape[0] != merged_rows * 4:
            raise RuntimeError(
                "Qwen visual preprocessing row mismatch: "
                f"{patches.shape[0]} raw rows for {merged_rows} merged rows"
            )
        patch_features = patches.shape[1]
        prepared.append((span, patches, grid, total_raw_rows, merged_cursor))
        total_raw_rows += patches.shape[0]
        merged_cursor += merged_rows

    patches_host = _pinned_empty((total_raw_rows, patch_features), torch.float32)
    grids = []
    visuals = []
    for span, patches, grid, raw_start, merged_start in prepared:
        raw_stop = raw_start + patches.shape[0]
        merged_stop = merged_start + (span.stop - span.start)
        patches_host[raw_start:raw_stop].copy_(patches)
        grids.append(grid)
        visuals.append(
            PreparedVisual(
                layout_span=span,
                grid=grid,
                raw_start=raw_start,
                raw_stop=raw_stop,
                merged_start=merged_start,
                merged_stop=merged_stop,
            )
        )
    return PreparedVisionBatch(
        patches=patches_host,
        grids=torch.cat(grids, dim=0),
        cu_seqlens=packed_vision_cu_seqlens(layout),
        visuals=visuals,
    )


def _vision_position_plan(visual, grids: torch.Tensor):
    grid_rows = [(int(t), int(h), int(w)) for t, h, w in grids.tolist()]
    side = visual.num_grid_per_side
    merge = visual.spatial_merge_size
    all_indices = [[] for _ in range(4)]
    all_weights = [[] for _ in range(4)]
    angle_chunks = []
    inv_freq = visual.rotary_pos_emb.inv_freq.detach().float().cpu()
    max_hw = max(max(height, width) for _, height, width in grid_rows)
    freq_table = torch.outer(torch.arange(max_hw, dtype=torch.float32), inv_freq)

    for frames, height, width in grid_rows:
        h = torch.linspace(0, side - 1, height)
        w = torch.linspace(0, side - 1, width)
        hf, wf = h.int(), w.int()
        hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
        dh, dw = h - hf, w - wf
        base_h, base_hc = hf * side, hc * side
        indices = [
            (base_h[:, None] + wf[None, :]).flatten(),
            (base_h[:, None] + wc[None, :]).flatten(),
            (base_hc[:, None] + wf[None, :]).flatten(),
            (base_hc[:, None] + wc[None, :]).flatten(),
        ]
        weights = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]
        rows = torch.arange(height).view(height, 1).expand(height, width)
        cols = torch.arange(width).view(1, width).expand(height, width)
        order_shape = (height // merge, merge, width // merge, merge)
        order = (
            torch.arange(height * width)
            .view(order_shape)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        coords = torch.stack((rows.flatten()[order], cols.flatten()[order]), dim=-1)
        angles = freq_table[coords].flatten(1).repeat(frames, 1)
        angle_chunks.append(angles)
        for index in range(4):
            ordered_indices = indices[index][order].repeat(frames)
            ordered_weights = weights[index][order].repeat(frames)
            all_indices[index].append(ordered_indices)
            all_weights[index].append(ordered_weights)

    indices = torch.stack([torch.cat(items) for items in all_indices])
    weights = torch.stack([torch.cat(items) for items in all_weights])
    angles = torch.cat(angle_chunks).contiguous()
    return indices, weights, angles


def _vision_embed_to_host(
    visual,
    batch: PreparedVisionBatch,
    hidden: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    settings: QwenSeqAttnSettings,
    device: torch.device,
) -> None:
    chunk = settings.qkv_tile_tokens
    with _lease_group(visual.patch_embed, visual.pos_embed):
        for start in range(0, hidden.shape[0], chunk):
            stop = min(start + chunk, hidden.shape[0])
            patches = batch.patches[start:stop].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            index_tile = indices[:, start:stop].to(device=device)
            weight_tile = weights[:, start:stop].to(
                device=device, dtype=torch.bfloat16
            )
            patch_hidden = visual.patch_embed(patches).to(torch.bfloat16)
            pos = visual.pos_embed(index_tile).to(torch.bfloat16)
            patch_hidden.add_((pos * weight_tile[:, :, None]).sum(dim=0))
            hidden[start:stop].copy_(patch_hidden, non_blocking=True)
    torch.cuda.current_stream(device).synchronize()


def _vision_block_parts(block, angles_host, device):
    from comfy.text_encoders.llama import apply_rope

    heads = block.attn.num_heads
    head_dim = block.attn.head_dim

    def project_qkv(tile: torch.Tensor, start: int, stop: int):
        tokens = stop - start
        q, k, v = (
            block.attn.qkv(block.norm1(tile))
            .reshape(tokens, 3, heads, head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        angles = angles_host[start:stop].to(device=device)
        emb = torch.cat((angles, angles), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        half = sin.shape[-1] // 2
        q, k = apply_rope(q, k, (cos, sin[..., :half], -sin[..., half:]))
        return q, k, v

    def attention_epilogue(
        attention: torch.Tensor,
        residual_hidden_host: torch.Tensor,
        start: int,
        stop: int,
    ):
        residual = residual_hidden_host[start:stop].to(device=device, non_blocking=True)
        return residual.add(block.attn.proj(attention.reshape(stop - start, -1)))

    def mlp(post_attention: torch.Tensor, _start: int, _stop: int):
        return post_attention.add(block.mlp(block.norm2(post_attention)))

    return (
        H3MaterializedProjection(
            project_qkv,
            weight_lease=lambda: _lease_group(block.attn.qkv),
        ),
        H3BlockOps(
            attention_epilogue=attention_epilogue,
            mlp=mlp,
            consumer_lease=lambda: _lease_group(
                block.attn.proj,
                block.mlp.linear_fc1,
                block.mlp.linear_fc2,
            ),
        ),
    )


def _merge_vision_to_host(
    merger,
    hidden: torch.Tensor,
    *,
    output_features: int,
    tile_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    merge_unit = 4
    rows = hidden.shape[0] // merge_unit
    output = _pinned_empty((rows, output_features), torch.bfloat16)
    with _lease_group(merger):
        for start in range(0, rows, tile_tokens):
            stop = min(start + tile_tokens, rows)
            source = hidden[start * merge_unit : stop * merge_unit].to(
                device=device, non_blocking=True
            )
            merged = merger(source).to(torch.bfloat16)
            output[start:stop].copy_(merged, non_blocking=True)
    torch.cuda.current_stream(device).synchronize()
    return output


def _encode_vision(
    decoder,
    layout: QwenPresentationLayout,
    runtime: QwenEncodeRuntime,
):
    batch = _prepare_visual_batch(layout)
    if batch is None:
        return None, [], [], 0
    visual = decoder.visual
    device = runtime.device
    indices, weights, angles = _vision_position_plan(visual, batch.grids)
    hidden = _pinned_empty(
        (batch.patches.shape[0], visual.hidden_size), torch.bfloat16
    )
    runner = runtime.runner(
        tokens=hidden.shape[0],
        hidden_features=visual.hidden_size,
        q_heads=visual.num_heads,
        kv_heads=visual.num_heads,
        head_dim=visual.hidden_size // visual.num_heads,
    )
    sequence_meta = H3SequenceMeta(cu_seqlens=batch.cu_seqlens)
    deepstack_by_layer = {}
    merged = None
    deepstack_indexes = list(visual.deepstack_visual_indexes)
    stages = [(visual.patch_embed, visual.pos_embed)]
    for index, block in enumerate(visual.blocks):
        group = [block]
        if index in deepstack_indexes:
            group.append(visual.deepstack_merger_list[deepstack_indexes.index(index)])
        stages.append(tuple(group))
    stages.append((visual.merger,))

    def compute(stage_index: int):
        nonlocal merged
        if stage_index == 0:
            _vision_embed_to_host(
                visual, batch, hidden, indices, weights, runtime.settings, device
            )
            return
        block_index = stage_index - 1
        if block_index < len(visual.blocks):
            block = visual.blocks[block_index]
            projection, ops = _vision_block_parts(block, angles, device)
            runner.run_block_(
                hidden,
                sequence_meta,
                projection,
                ops,
                softmax_scale=(visual.hidden_size // visual.num_heads) ** -0.5,
                causal=False,
            )
            if block_index in deepstack_indexes:
                merger = visual.deepstack_merger_list[
                    deepstack_indexes.index(block_index)
                ]
                deepstack_by_layer[block_index] = _merge_vision_to_host(
                    merger,
                    hidden,
                    output_features=decoder.model.config.hidden_size,
                    tile_tokens=runtime.settings.qkv_tile_tokens,
                    device=device,
                )
            return
        merged = _merge_vision_to_host(
            visual.merger,
            hidden,
            output_features=decoder.model.config.hidden_size,
            tile_tokens=runtime.settings.qkv_tile_tokens,
            device=device,
        )

    try:
        max_staged = run_weight_stages(stages, device, compute)
    finally:
        runtime.release_runner(runner)
    deepstack = [deepstack_by_layer[index] for index in deepstack_indexes]
    return merged, deepstack, batch.visuals, max_staged


def _embed_token_rows_cpu(
    module, token_ids: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    import comfy.quant_ops as quant_ops

    weight = module.weight
    if (
        not isinstance(weight, quant_ops.QuantizedTensor)
        or getattr(module, "quant_format", None) != "int8_tensorwise"
        or len(getattr(module, "weight_function", ())) != 0
    ):
        raise RuntimeError(
            "MiniMax H3 Qwen CPU embedding requires the native unpatched "
            "int8_tensorwise embedding table"
        )
    layout = quant_ops.get_layout_class(module.layout_type)
    return layout.dequantize_embedding(
        weight._qdata,
        weight._params,
        token_ids.to(device="cpu"),
    ).to(device="cpu", dtype=dtype)


def _fill_decoder_hidden(
    decoder,
    layout: QwenPresentationLayout,
    visuals: list[PreparedVisual],
    merged_visual: torch.Tensor | None,
    hidden: torch.Tensor,
) -> None:
    token_spans = [span for span in layout.spans if span.kind == "token"]
    if token_spans:
        token_ids = torch.tensor([int(span.entry) for span in token_spans], dtype=torch.long)
        token_rows = torch.tensor([span.start for span in token_spans], dtype=torch.long)
        unique_ids, inverse = torch.unique(
            token_ids, sorted=False, return_inverse=True
        )
        unique_rows = _embed_token_rows_cpu(
            decoder.model.embed_tokens, unique_ids, hidden.dtype
        )
        embedded_host = _pinned_empty((len(token_spans), hidden.shape[1]), hidden.dtype)
        embedded_host.copy_(unique_rows.index_select(0, inverse))
        hidden.index_copy_(0, token_rows, embedded_host)

    for span in layout.spans:
        if span.kind != "embedding":
            continue
        entry = span.entry
        data = entry if isinstance(entry, torch.Tensor) else entry["data"]
        data = data.detach().reshape(-1, data.shape[-1]).to(
            device="cpu", dtype=hidden.dtype
        )
        if data.shape != hidden[span.start : span.stop].shape:
            raise RuntimeError("Qwen embedding width does not match hidden size")
        hidden[span.start : span.stop].copy_(data)

    if merged_visual is not None:
        for visual in visuals:
            hidden[visual.layout_span.start : visual.layout_span.stop].copy_(
                merged_visual[visual.merged_start : visual.merged_stop]
            )


def _decoder_position_ids(layout, visuals):
    if not visuals:
        return torch.arange(layout.total_rows, dtype=torch.float32).unsqueeze(0)
    import comfy.text_encoders.qwen_vl as qwen_vl

    embeds_info = [
        {
            "type": "image",
            "index": visual.layout_span.start,
            "size": visual.layout_span.stop - visual.layout_span.start,
            "extra": {"grid": visual.grid},
        }
        for visual in visuals
    ]
    return qwen_vl.qwen2vl_mrope_position_ids(
        embeds_info, layout.total_rows, torch.device("cpu")
    )


def _decoder_block_parts(layer, model, position_ids, device):
    from comfy.text_encoders.llama import apply_rope

    attention = layer.self_attn
    q_heads = attention.num_heads
    kv_heads = attention.num_kv_heads
    head_dim = attention.head_dim

    def project_qkv(tile: torch.Tensor, start: int, stop: int):
        tokens = stop - start
        tile = layer.input_layernorm(tile)
        q = attention.q_proj(tile).view(tokens, q_heads, head_dim)
        k = attention.k_proj(tile).view(tokens, kv_heads, head_dim)
        v = attention.v_proj(tile).view(tokens, kv_heads, head_dim)
        if attention.q_norm is not None:
            q = attention.q_norm(q)
        if attention.k_norm is not None:
            k = attention.k_norm(k)
        positions = position_ids[:, start:stop].to(device=device)
        freqs = model.compute_freqs_cis(positions, device)
        q, k = apply_rope(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            freqs,
        )
        return q[0].transpose(0, 1), k[0].transpose(0, 1), v

    def attention_epilogue(
        attention_output: torch.Tensor,
        residual_hidden_host: torch.Tensor,
        start: int,
        stop: int,
    ):
        update = attention.o_proj(attention_output.reshape(stop - start, -1))
        residual = residual_hidden_host[start:stop].to(device=device, non_blocking=True)
        return residual.add(update)

    def mlp(post_attention: torch.Tensor, _start: int, _stop: int):
        residual = post_attention
        tile = layer.post_attention_layernorm(post_attention)
        gate = layer.mlp.activation(layer.mlp.gate_proj(tile))
        gate.mul_(layer.mlp.up_proj(tile))
        return residual.add(layer.mlp.down_proj(gate))

    return (
        H3MaterializedProjection(
            project_qkv,
            weight_lease=lambda: _lease_group(
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
            ),
        ),
        H3BlockOps(
            attention_epilogue=attention_epilogue,
            mlp=mlp,
            consumer_lease=lambda: _lease_group(
                attention.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ),
        ),
    )


def _encode_decoder(
    decoder,
    layout: QwenPresentationLayout,
    merged_visual,
    deepstack,
    visuals,
    runtime: QwenEncodeRuntime,
) -> tuple[torch.Tensor, int]:
    model = decoder.model
    device = runtime.device
    hidden = _pinned_empty(
        (layout.total_rows, model.config.hidden_size), torch.bfloat16
    )
    _fill_decoder_hidden(
        decoder,
        layout,
        visuals,
        merged_visual,
        hidden,
    )
    position_ids = _decoder_position_ids(layout, visuals)
    first_attention = model.layers[0].self_attn
    runner = runtime.runner(
        tokens=layout.total_rows,
        hidden_features=model.config.hidden_size,
        q_heads=first_attention.num_heads,
        kv_heads=first_attention.num_kv_heads,
        head_dim=first_attention.head_dim,
    )
    sequence_meta = H3SequenceMeta(
        cu_seqlens=torch.tensor([0, layout.total_rows], dtype=torch.int32)
    )
    stages = [(layer,) for layer in model.layers]

    def compute(stage_index: int):
        layer_index = stage_index
        layer = model.layers[stage_index]
        projection, ops = _decoder_block_parts(layer, model, position_ids, device)
        runner.run_block_(
            hidden,
            sequence_meta,
            projection,
            ops,
            softmax_scale=first_attention.head_dim**-0.5,
            causal=True,
        )
        if layer_index < len(deepstack):
            inject_deepstack_cpu_(
                hidden,
                deepstack[layer_index],
                visuals,
                chunk_tokens=runtime.settings.mlp_tile_tokens,
            )
            deepstack[layer_index] = None

    try:
        max_staged = run_weight_stages(stages, device, compute)
    finally:
        runtime.release_runner(runner)
    return hidden, max_staged


def _validate_conditioning_hidden(hidden: torch.Tensor) -> None:
    if not bool(torch.isfinite(hidden).all()):
        raise RuntimeError(
            "MiniMax H3 Qwen SeqAttn produced non-finite conditioning"
        )
    if int(torch.count_nonzero(hidden)) == 0:
        raise RuntimeError(
            "MiniMax H3 Qwen SeqAttn produced all-zero conditioning"
        )


class QwenSeqAttnController:
    def __init__(self, clip, settings: QwenSeqAttnSettings):
        self.clip = clip
        self.settings = settings
        self._local = threading.local()
        self._module_names = {
            id(module): name for name, module in clip.cond_stage_model.named_modules()
        }
        self.decoder = self._find_decoder()
        self.clip_model = self._find_clip_model()
        self._active_runtime: QwenEncodeRuntime | None = None
        self.last_encode_stats: dict[str, int | bool] | None = None
        self._validate_model_contract()

    def _find_decoder(self):
        candidates = []
        for module in self.clip.cond_stage_model.modules():
            model = getattr(module, "model", None)
            layers = getattr(model, "layers", None)
            visual = getattr(module, "visual", None)
            if isinstance(layers, torch.nn.ModuleList) and visual is not None:
                candidates.append(module)
        if len(candidates) != 1:
            raise TypeError(
                "MiniMax H3 Qwen SeqAttn requires the native MiniMax-H3 "
                f"Qwen encoder; found {len(candidates)} candidates"
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
                "MiniMax H3 Qwen SeqAttn requires the native MiniMax-H3 "
                f"CLIP wrapper; found {len(candidates)} candidates"
            )
        return candidates[0]

    def _validate_model_contract(self) -> None:
        config = self.decoder.model.config
        visual = self.decoder.visual
        expected = {
            "hidden_size": 5120,
            "num_hidden_layers": 50,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
        }
        actual = {name: int(getattr(config, name)) for name in expected}
        vision_contract = (
            int(visual.hidden_size) == 1152
            and int(visual.num_heads) == 16
            and int(visual.spatial_merge_size) == 2
            and len(visual.blocks) == 27
            and list(visual.deepstack_visual_indexes) == [8, 16, 24]
            and len(visual.deepstack_merger_list) == 3
        )
        if (
            actual != expected
            or not vision_contract
            or self.decoder.model.norm is not None
        ):
            raise TypeError(
                "MiniMax H3 Qwen SeqAttn currently supports only the native "
                "Qwen3-VL-32B 50-layer conditioning encoder"
            )

    def _patch_method(self, module, name: str, method) -> None:
        module_name = self._module_names.get(id(module))
        if module_name is None:
            raise RuntimeError(f"cannot locate Qwen module for patched method {name}")
        path = f"{module_name}.{name}" if module_name else name
        self.clip.patcher.add_object_patch(path, types.MethodType(method, module))

    def install(self) -> None:
        self.clip.patcher.set_model_compute_dtype(torch.bfloat16)
        controller = self

        def streaming_forward(this, tokens):
            hidden, attention_mask, token_tags = controller.encode(tokens)
            controller.decoder.last_token_tags = token_tags
            if this.zero_out_masked:
                hidden.mul_(attention_mask.unsqueeze(-1).to(hidden.dtype))
            extra = {}
            if this.return_attention_masks:
                extra["attention_mask"] = attention_mask
            if extra:
                return hidden.unsqueeze(0), None, extra
            return hidden.unsqueeze(0), None

        self._patch_method(self.clip_model, "forward", streaming_forward)
        self._wrap_clip_encoding()
        setattr(self.clip, QWEN_SEQATTN_STATE_KEY, self)

    def preflight(self, tokens) -> QwenPresentationLayout:
        layout = build_qwen_presentation_layout(tokens)
        if layout.total_rows == 0:
            raise RuntimeError("Qwen input rejected before encode: no conditioning rows")
        return layout

    @contextmanager
    def _encoding_policy(self):
        import comfy.model_management as model_management

        with _ENCODE_LOCK:
            original_vram_state = model_management.vram_state
            original_streams = model_management.NUM_STREAMS
            model_management.vram_state = model_management.VRAMState.NO_VRAM
            model_management.NUM_STREAMS = 0
            try:
                yield
            finally:
                model_management.NUM_STREAMS = original_streams
                model_management.vram_state = original_vram_state

    def encode(self, tokens):
        if not torch.cuda.is_available():
            raise RuntimeError("MiniMax H3 Qwen SeqAttn requires CUDA")
        layout = self.preflight(tokens)
        device = torch.device(self.clip.patcher.load_device)
        runtime = QwenEncodeRuntime(self.settings, device)
        self._active_runtime = runtime
        vision_max_staged = 0
        decoder_max_staged = 0
        try:
            merged_visual, deepstack, visuals, vision_max_staged = _encode_vision(
                self.decoder, layout, runtime
            )
            hidden, decoder_max_staged = _encode_decoder(
                self.decoder,
                layout,
                merged_visual,
                deepstack,
                visuals,
                runtime,
            )
            _validate_conditioning_hidden(hidden)
            attention_mask = torch.tensor(layout.attention_mask, dtype=torch.long)
            token_tags = torch.tensor(layout.token_tags, dtype=torch.long)
            return hidden, attention_mask, token_tags
        finally:
            torch.cuda.synchronize(device)
            runtime.close()
            self._active_runtime = None
            gc.collect()
            self.last_encode_stats = {
                "runtime_released": runtime.closed,
                "vision_max_staged": vision_max_staged,
                "decoder_max_staged": decoder_max_staged,
            }

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


def patch_minimax_h3_qwen_seqattn_clip(
    clip, *, q_chunk_tokens: int, kv_chunk_tokens: int
):
    settings = QwenSeqAttnSettings.from_config(
        q_chunk_tokens=q_chunk_tokens,
        kv_chunk_tokens=kv_chunk_tokens,
    )
    settings.validate()
    load_device = torch.device(clip.patcher.load_device)
    if load_device.type != "cuda":
        raise ValueError(
            "MiniMax H3 Qwen SeqAttn requires CLIPLoader device='default' "
            "with an NVIDIA CUDA load device"
        )
    patched = clip.clone()
    controller = QwenSeqAttnController(patched, settings)
    controller.install()
    return patched


__all__ = [
    "PreparedVisual",
    "QWEN_SEQATTN_STATE_KEY",
    "QwenEncodeRuntime",
    "QwenInputSpan",
    "QwenPresentationLayout",
    "QwenSeqAttnController",
    "QwenSeqAttnSettings",
    "build_qwen_presentation_layout",
    "inject_deepstack_cpu_",
    "packed_vision_cu_seqlens",
    "patch_minimax_h3_qwen_seqattn_clip",
    "qwen_vision_merged_rows",
]
