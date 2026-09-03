from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPOSITORY_ROOT / "workflows"
FL2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF2VA_WORKFLOW = "minimax_h3_seqattn_ref2va.json"
REF2VA_LONG_WORKFLOW = "minimax_h3_seqattn_ref2va_long_2step.json"
SOL_TURBO_WORKFLOWS = {
    "minimax_h3_seqattn_fl2va_sol_4step_lora.json": (
        FL2VA_MODEL,
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        4,
    ),
    "minimax_h3_seqattn_ref2va_sol_4step_lora.json": (
        REF2VA_MODEL,
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        4,
    ),
}
TURBO_WORKFLOWS = {
    "minimax_h3_seqattn_fl2va_turbo_4step_lora.json": (
        FL2VA_MODEL,
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        4,
    ),
    "minimax_h3_seqattn_fl2va_turbo_8step_lora.json": (
        FL2VA_MODEL,
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        8,
    ),
    "minimax_h3_seqattn_ref2va_turbo_4step_lora.json": (
        REF2VA_MODEL,
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        4,
    ),
    **SOL_TURBO_WORKFLOWS,
}

WORKFLOW_KEYFRAME_LINKS = {
    "minimax_h3_seqattn_t2va.json": (None, None),
    "minimax_h3_seqattn_first_frame.json": (278, None),
    "minimax_h3_seqattn_last_frame.json": (None, 282),
    "minimax_h3_seqattn_fl2va.json": (278, 282),
}


def _nodes_by_type(workflow):
    return {node["type"]: node for node in workflow["nodes"]}


def test_fl2va_workflows_are_standalone_and_use_safe_defaults():
    for filename, expected_keyframe_links in WORKFLOW_KEYFRAME_LINKS.items():
        workflow = json.loads((WORKFLOW_DIR / filename).read_text())
        nodes = _nodes_by_type(workflow)

        assert nodes["UNETLoader"]["widgets_values"][0] == FL2VA_MODEL
        assert nodes["MiniMaxH3SeqAttn"]["widgets_values"] == [5760, 4096]
        assert nodes["MiniMaxH3QwenSeqAttn"]["widgets_values"] == [5760, 4096]
        assert nodes["MiniMaxH3VAEStreaming"]["widgets_values"] == []

        conditioning = nodes["MiniMaxH3ImageToVideo"]
        assert [item["name"] for item in conditioning["inputs"]] == [
            "clip",
            "vae",
            "prompt",
            "width",
            "height",
            "length",
            "first_frame",
            "last_frame",
        ]
        inputs = {item["name"]: item.get("link") for item in conditioning["inputs"]}
        assert inputs["vae"] == 287
        assert tuple(inputs[name] for name in ("first_frame", "last_frame")) == (
            expected_keyframe_links
        )


def test_fl2va_workflow_links_match_node_slots():
    for filename in (
        *WORKFLOW_KEYFRAME_LINKS,
        REF2VA_WORKFLOW,
        REF2VA_LONG_WORKFLOW,
        *TURBO_WORKFLOWS,
    ):
        workflow = json.loads((WORKFLOW_DIR / filename).read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        for link_id, source_id, source_slot, target_id, target_slot, link_type in links.values():
            source_type = nodes[source_id]["outputs"][source_slot]["type"]
            target_input = nodes[target_id]["inputs"][target_slot]
            assert link_type in source_type.split(",")
            assert link_type in target_input["type"].split(",")
            assert target_input["link"] == link_id

        for node in nodes.values():
            for source_slot, output in enumerate(node.get("outputs", [])):
                for link_id in output.get("links") or []:
                    assert links[link_id][1:3] == [node["id"], source_slot]


def test_turbo_workflows_use_dedicated_staged_lora_node():
    for filename, (base_model, lora_name, steps) in TURBO_WORKFLOWS.items():
        workflow = json.loads((WORKFLOW_DIR / filename).read_text())
        nodes = _nodes_by_type(workflow)
        links = {link[0]: link for link in workflow["links"]}

        loader = nodes["UNETLoader"]
        lora = nodes["MiniMaxH3SeqAttnLoRA"]
        seqattn = nodes["MiniMaxH3SeqAttn"]
        assert loader["widgets_values"][0] == base_model
        assert lora["widgets_values"] == [lora_name, 1]
        assert nodes["BasicScheduler"]["widgets_values"][1] == steps

        loader_link = links[lora["inputs"][0]["link"]]
        seqattn_link = links[seqattn["inputs"][0]["link"]]
        assert loader_link[1:5] == [loader["id"], 0, lora["id"], 0]
        assert seqattn_link[1:5] == [lora["id"], 0, seqattn["id"], 0]


def test_sol_turbo_workflows_use_the_gpu1_deployment_contract():
    for filename in SOL_TURBO_WORKFLOWS:
        workflow = json.loads((WORKFLOW_DIR / filename).read_text())
        nodes = _nodes_by_type(workflow)
        metadata = workflow["extra"]["seqattn"]

        assert nodes["MiniMaxH3SeqAttn"]["widgets_values"] == [15360, 4096]
        assert nodes["MiniMaxH3QwenSeqAttn"]["widgets_values"] == [5760, 4096]
        assert nodes["BasicScheduler"]["widgets_values"][1] == 4
        assert metadata == {
            "attention_mode": "sol_streaming",
            "config": "docker/seqattn-sol.toml",
            "q_chunk_tokens": 15360,
            "kv_chunk_tokens": 4096,
        }
        note = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "MarkdownNote"
            and "Sol deployment" in node["widgets_values"][0]
        )
        assert 'attention_mode = "sol_streaming"' in note["widgets_values"][0]


def test_ref2va_workflow_uses_streaming_vae_and_matching_references():
    workflow = json.loads((WORKFLOW_DIR / REF2VA_WORKFLOW).read_text())
    nodes = _nodes_by_type(workflow)

    assert nodes["UNETLoader"]["widgets_values"][0] == REF2VA_MODEL
    assert nodes["MiniMaxH3SeqAttn"]["widgets_values"] == [5760, 4096]
    assert nodes["MiniMaxH3QwenSeqAttn"]["widgets_values"] == [5760, 4096]
    assert nodes["MiniMaxH3VAEStreaming"]["widgets_values"] == []

    conditioning = nodes["MiniMaxH3ReferenceToVideoSeqAttn"]
    inputs = {item["name"]: item.get("link") for item in conditioning["inputs"]}
    assert inputs["vae"] == 287
    assert inputs["ref_images.ref_image_0"] == 278
    assert inputs["ref_images.ref_image_1"] == 282
    assert inputs["ref_videos.ref_video_0"] is None
    assert inputs["ref_video_audios.ref_video_audio_0"] is None
    assert inputs["ref_audios.ref_audio_0"] is None

    assert nodes["VAEDecode"]["inputs"][1]["link"] == 286
    video_vae_loader = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "VAELoader"
        and node["widgets_values"][0] == "minimax_h3_video_vae_fp16.safetensors"
    )
    assert video_vae_loader["outputs"][0]["links"] == [285]

    prompt = nodes["PrimitiveStringMultiline"]["widgets_values"][0]
    assert "<Picture 1>" in prompt
    assert "<Picture 2>" in prompt
    assert "<Audio 1>" not in prompt


def test_long_ref2va_workflow_is_a_two_step_video_ui_example():
    workflow = json.loads((WORKFLOW_DIR / REF2VA_LONG_WORKFLOW).read_text())
    nodes = _nodes_by_type(workflow)

    assert nodes["UNETLoader"]["widgets_values"][0] == REF2VA_MODEL
    assert nodes["MiniMaxH3SeqAttn"]["widgets_values"] == [5760, 4096]
    assert nodes["MiniMaxH3QwenSeqAttn"]["widgets_values"] == [5760, 4096]
    assert nodes["MiniMaxH3VAEStreaming"]["widgets_values"] == []
    assert nodes["BasicScheduler"]["widgets_values"][1] == 2
    assert nodes["PrimitiveFloat"]["widgets_values"][0] == 10.125
    assert nodes["LoadVideo"]["widgets_values"] == ["ref2va_input.mp4"]

    conditioning = nodes["MiniMaxH3ReferenceToVideoSeqAttn"]
    inputs = {item["name"]: item.get("link") for item in conditioning["inputs"]}
    assert inputs["ref_images.ref_image_0"] is None
    assert inputs["ref_images.ref_image_1"] is None
    assert inputs["ref_videos.ref_video_0"] == 289
    assert inputs["ref_video_audios.ref_video_audio_0"] == 290
    assert conditioning["widgets_values"][1:4] == [1344, 768, 243]

    prompt = nodes["PrimitiveStringMultiline"]["widgets_values"][0]
    assert "<Video 1>" in prompt
    assert "<Audio 1>" in prompt
