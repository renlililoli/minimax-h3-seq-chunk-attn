from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_seqattn import qwen


def test_settings_validation():
    qwen.QwenMemorySettings().validate()
    with pytest.raises(ValueError, match="activation_limit_mib"):
        qwen.QwenMemorySettings(activation_limit_mib=0).validate()
    with pytest.raises(ValueError, match="max_conditioning_rows"):
        qwen.QwenMemorySettings(max_conditioning_rows=0).validate()
    with pytest.raises(ValueError, match="preflight_safety_mib"):
        qwen.QwenMemorySettings(preflight_safety_mib=-1).validate()
    with pytest.raises(ValueError, match="offload_mode"):
        qwen.QwenMemorySettings(offload_mode="fast").validate()
    assert qwen.QwenMemorySettings(offload_mode="prefetch").offload_streams == 2
    assert qwen.QwenMemorySettings(offload_mode="extreme").offload_streams == 0


def test_768p_video_blocks_match_measured_presentation():
    video_block = torch.empty((2, 768, 1344, 3), device="meta")
    assert qwen.qwen_vision_merged_rows(video_block) == 1008

    entries = [(1, 1.0) for _ in range(237)]
    entries.extend(
        (
            {
                "type": "image",
                "data": video_block,
                "minimax_video_block": True,
            },
            1.0,
        )
        for _ in range(11)
    )
    plan = qwen.inspect_qwen_input_tokens({"qwen3vl_32b": [entries]})

    assert plan["non_visual_rows"] == 237
    assert plan["video_rows"] == 11088
    assert plan["total_rows"] == 11325


def test_mixed_text_image_video_and_embedding_rows():
    image = torch.empty((1, 768, 1344, 3), device="meta")
    video_block = torch.empty((2, 768, 1344, 3), device="meta")
    embedding = torch.empty((1, 5, 5120), device="meta")
    entries = [(1, 1.0) for _ in range(10)]
    entries.extend(
        [
            ({"type": "image", "data": image}, 1.0),
            (
                {
                    "type": "image",
                    "data": video_block,
                    "minimax_video_block": True,
                },
                1.0,
            ),
            ({"type": "embedding", "data": embedding}, 1.0),
        ]
    )

    plan = qwen.inspect_qwen_input_tokens({"qwen3vl_32b": [entries]})

    assert plan == {
        "non_visual_rows": 10,
        "image_rows": 1008,
        "video_rows": 1008,
        "visual_rows": 2016,
        "embedding_rows": 5,
        "total_rows": 2031,
    }


@pytest.mark.parametrize(
    ("total_rows", "visual_rows", "expected_mib", "fits_default_limit"),
    [
        (21388, 21168, 5655.748626708984, True),
        (22405, 22176, 5928.87060546875, False),
    ],
)
def test_bf16_default_limit_matches_measured_prefetch_boundary(
    total_rows, visual_rows, expected_mib, fits_default_limit
):
    settings = qwen.QwenMemorySettings()
    estimated, with_safety = qwen.estimate_qwen_activation(
        {"total_rows": total_rows, "visual_rows": visual_rows},
        settings,
        hidden_features=5120,
        intermediate_features=25600,
    )

    assert estimated == pytest.approx(expected_mib)
    assert (with_safety <= settings.activation_limit_mib) is fits_default_limit


def test_bf16_measured_baseline_reproduces_recorded_plan():
    estimated, _ = qwen.estimate_qwen_activation(
        {"total_rows": 11325, "visual_rows": 11088},
        qwen.QwenMemorySettings(),
        hidden_features=5120,
        intermediate_features=25600,
    )

    assert estimated == pytest.approx(3159.71240234375)


def test_25000_rows_is_a_hard_cap_not_a_memory_guarantee():
    settings = qwen.QwenMemorySettings()
    _, with_safety = qwen.estimate_qwen_activation(
        {"total_rows": 25000, "visual_rows": 25000},
        settings,
        hidden_features=5120,
        intermediate_features=25600,
    )

    assert with_safety > settings.activation_limit_mib


def test_preflight_rejects_oversized_plan_before_encode(monkeypatch):
    monkeypatch.setattr(
        qwen,
        "inspect_qwen_input_tokens",
        lambda _tokens: {
            "non_visual_rows": 229,
            "image_rows": 22176,
            "video_rows": 0,
            "visual_rows": 22176,
            "embedding_rows": 0,
            "total_rows": 22405,
        },
    )
    controller = object.__new__(qwen.QwenBF16Controller)
    controller.settings = qwen.QwenMemorySettings()
    controller.decoder = SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(
            weight=torch.empty((1, 5120), device="meta")
        )
    )
    controller.layers = [
        SimpleNamespace(mlp=SimpleNamespace(gate_proj=SimpleNamespace(out_features=25600)))
    ]
    controller._local = SimpleNamespace(encoding=False)
    encoded = False

    def operation():
        nonlocal encoded
        encoded = True

    with pytest.raises(RuntimeError, match="Qwen input rejected before encode"):
        controller._run_encode(object(), operation)

    assert not encoded


def test_patch_requires_cuda_clip_loader():
    clip = SimpleNamespace(patcher=SimpleNamespace(load_device=torch.device("cpu")))
    with pytest.raises(ValueError, match="device='default'"):
        qwen.patch_minimax_h3_qwen_clip(
            clip,
            activation_limit_mib=5888,
            max_conditioning_rows=25000,
            preflight_safety_mib=128,
            offload_mode="prefetch",
        )


@pytest.mark.parametrize(
    ("mode", "expected_streams"), [("prefetch", 2), ("extreme", 0)]
)
def test_encoding_policy_selects_and_restores_streams(mode, expected_streams):
    import comfy.model_management as model_management

    controller = object.__new__(qwen.QwenBF16Controller)
    controller.settings = qwen.QwenMemorySettings(offload_mode=mode)
    original_vram_state = model_management.vram_state
    original_streams = model_management.NUM_STREAMS

    with controller._encoding_policy():
        assert model_management.vram_state == model_management.VRAMState.NO_VRAM
        assert model_management.NUM_STREAMS == expected_streams

    assert model_management.vram_state == original_vram_state
    assert model_management.NUM_STREAMS == original_streams
