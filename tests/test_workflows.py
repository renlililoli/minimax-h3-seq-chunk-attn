from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPOSITORY_ROOT / "workflows"
FL2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"

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
        assert nodes["MiniMaxH3SeqAttn"]["widgets_values"] == [
            1024,
            4096,
            "fit",
            True,
        ]
        assert nodes["MiniMaxH3QwenBF16Offload"]["widgets_values"] == [
            5888,
            25000,
            128,
            "prefetch",
            True,
        ]
        assert nodes["MiniMaxH3VAEStreaming"]["widgets_values"] == [
            192,
            512,
            True,
        ]

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
    for filename in WORKFLOW_KEYFRAME_LINKS:
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
