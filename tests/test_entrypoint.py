from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

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


def _load_repository_entrypoint_without_package_context():
    module_name = "comfyui_minimax_h3_seqattn_top_level_entrypoint_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPOSITORY_ROOT / "__init__.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_root_is_a_comfyui_v3_extension():
    module = _load_repository_entrypoint()
    extension = asyncio.run(module.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    assert [node.__name__ for node in nodes] == [
        "MiniMaxH3SeqAttn",
        "MiniMaxH3SeqAttnLoRA",
        "MiniMaxH3QwenSeqAttn",
        "MiniMaxH3VAEStreaming",
        "MiniMaxH3ReferenceToVideoSeqAttn",
    ]


def test_repository_root_supports_top_level_import():
    module = _load_repository_entrypoint_without_package_context()
    assert callable(module.comfy_entrypoint)


def test_repository_entrypoint_rejects_other_comfyui_versions(monkeypatch):
    module = _load_repository_entrypoint()
    monkeypatch.setattr("comfyui_version.__version__", "0.33.1")

    with pytest.raises(RuntimeError, match="requires ComfyUI 0.30.0"):
        asyncio.run(module.comfy_entrypoint())


def test_repository_entrypoint_rejects_other_comfyui_commits(monkeypatch):
    module = _load_repository_entrypoint()
    monkeypatch.setattr("comfyui_version.__version__", "0.30.0")
    monkeypatch.setattr(module, "_comfyui_git_commit", lambda _path: "wrong")

    with pytest.raises(RuntimeError, match="requires ComfyUI commit"):
        asyncio.run(module.comfy_entrypoint())
