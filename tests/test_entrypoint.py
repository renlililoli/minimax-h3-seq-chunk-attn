from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_repository_entrypoint():
    module_name = "comfyui_minimax_h3_seqattn_entrypoint_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPOSITORY_ROOT / "__init__.py",
        submodule_search_locations=[str(REPOSITORY_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_root_is_a_comfyui_v3_extension():
    module = _load_repository_entrypoint()
    extension = asyncio.run(module.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    assert [node.__name__ for node in nodes] == ["MiniMaxH3SeqAttn"]
