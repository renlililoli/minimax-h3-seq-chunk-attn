from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_seqattn import qwen


def _image(rows: int = 1):
    side = 32 * int(rows**0.5)
    return torch.empty((1, side, side, 3), device="meta")


def test_settings_validate_low_vram_defaults():
    settings = qwen.QwenSeqAttnSettings()
    settings.validate()
    assert settings == qwen.QwenSeqAttnSettings(
        q_chunk_tokens=5760,
        kv_chunk_tokens=4096,
        qkv_tile_tokens=4096,
        mlp_tile_tokens=4096,
    )


def test_settings_read_qwen_tiles_from_shared_config(tmp_path, monkeypatch):
    config = tmp_path / "seqattn.toml"
    config.write_text(
        "[minimax_h3_qwen]\n"
        "qkv_tile_tokens = 96\n"
        "mlp_tile_tokens = 48\n"
    )
    monkeypatch.setenv("SEQATTN_CONFIG", str(config))

    settings = qwen.QwenSeqAttnSettings.from_config(
        q_chunk_tokens=320,
        kv_chunk_tokens=640,
    )

    assert settings == qwen.QwenSeqAttnSettings(
        q_chunk_tokens=320,
        kv_chunk_tokens=640,
        qkv_tile_tokens=96,
        mlp_tile_tokens=48,
    )


def test_preflight_has_no_presentation_row_limit():
    controller = object.__new__(qwen.QwenSeqAttnController)
    controller.settings = qwen.QwenSeqAttnSettings()
    embedding = torch.empty((1, 30000, 5120), device="meta")
    tokens = [[({"type": "embedding", "data": embedding}, 1.0)]]

    layout = controller.preflight(tokens)

    assert layout.total_rows == 30000


def test_layout_preserves_visual_order_tags_and_embedding_rows():
    image_a = _image()
    image_b = _image()
    embedding = torch.empty((1, 3, 5120), device="meta")
    tokens = {
        "qwen3vl_32b": [
            [
                (101, 1.0),
                (151652, 1.0),
                ({"type": "image", "data": image_a}, 1.0),
                (151653, 1.0),
                (202, 1.0),
                ({"type": "embedding", "data": embedding}, 1.0),
                (151652, 1.0),
                ({"type": "image", "data": image_b}, 1.0),
                (151653, 1.0),
            ]
        ]
    }

    layout = qwen.build_qwen_presentation_layout(tokens)

    assert layout.total_rows == 17
    assert layout.visual_rows == 8
    assert [(span.start, span.stop) for span in layout.visual_spans] == [
        (2, 6),
        (12, 16),
    ]
    assert layout.token_tags == (
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert qwen.packed_vision_cu_seqlens(layout).tolist() == [0, 16, 32]


def test_each_two_frame_video_block_is_a_packed_vision_sequence():
    first = {
        "type": "image",
        "data": torch.empty((2, 32, 64, 3), device="meta"),
        "minimax_video_block": True,
    }
    second = {
        "type": "image",
        "data": torch.empty((2, 64, 64, 3), device="meta"),
        "minimax_video_block": True,
    }

    layout = qwen.build_qwen_presentation_layout(
        [[(151652, 1.0), (first, 1.0), (151653, 1.0), (second, 1.0)]]
    )

    assert [(span.start, span.stop) for span in layout.visual_spans] == [
        (1, 7),
        (8, 12),
    ]
    assert qwen.packed_vision_cu_seqlens(layout).tolist() == [0, 24, 40]


def test_deepstack_cpu_injection_maps_each_visual_span():
    first = qwen.QwenInputSpan("visual", 1, 3, object())
    second = qwen.QwenInputSpan("visual", 5, 6, object())
    visuals = [
        qwen.PreparedVisual(first, torch.empty(0), 0, 8, 0, 2),
        qwen.PreparedVisual(second, torch.empty(0), 8, 12, 2, 3),
    ]
    hidden = torch.zeros((7, 4), dtype=torch.bfloat16)
    deepstack = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)

    qwen.inject_deepstack_cpu_(hidden, deepstack, visuals, chunk_tokens=1)

    torch.testing.assert_close(hidden[1:3], deepstack[:2])
    torch.testing.assert_close(hidden[5:6], deepstack[2:3])
    assert torch.count_nonzero(hidden[[0, 3, 4, 6]]) == 0


def test_encode_releases_ephemeral_runtime(monkeypatch):
    controller = object.__new__(qwen.QwenSeqAttnController)
    controller.settings = qwen.QwenSeqAttnSettings()
    controller.clip = SimpleNamespace(
        patcher=SimpleNamespace(load_device=torch.device("cuda"))
    )
    controller.decoder = object()
    controller._active_runtime = None
    controller.last_encode_stats = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("Qwen encode must not force allocator cache cleanup"),
    )
    monkeypatch.setattr(
        qwen,
        "_encode_vision",
        lambda *_args: (None, [], [], 1),
    )
    monkeypatch.setattr(
        qwen,
        "_encode_decoder",
        lambda *_args: (torch.zeros((1, 5120), dtype=torch.bfloat16), 1),
    )

    hidden, attention_mask, token_tags = controller.encode([[1]])

    assert hidden.shape == (1, 5120)
    assert attention_mask.tolist() == [1]
    assert token_tags.tolist() == [1]
    assert controller._active_runtime is None
    assert controller.last_encode_stats == {
        "runtime_released": True,
        "vision_max_staged": 1,
        "decoder_max_staged": 1,
    }
