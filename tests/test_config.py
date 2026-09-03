from __future__ import annotations

from pathlib import Path

import pytest

from comfyui_seqattn import config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_missing_default_config_uses_compiled_stage_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQATTN_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert config.load_attention_stage_config(
        "minimax_h3_qwen"
    ) == config.AttentionStageConfig()
    assert config.load_vae_stage_config() == config.VAEStageConfig()


def test_python_310_declares_tomli_runtime_dependency():
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = config.tomllib.load(handle)["project"]["dependencies"]

    assert "tomli>=1.1; python_version < '3.11'" in dependencies


def test_explicit_missing_config_fails(tmp_path, monkeypatch):
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv("SEQATTN_CONFIG", str(missing))

    with pytest.raises(FileNotFoundError, match="SEQATTN_CONFIG"):
        config.load_attention_stage_config("minimax_h3_qwen")


def test_invalid_toml_fails(tmp_path, monkeypatch):
    path = tmp_path / "seqattn.toml"
    path.write_text("[minimax_h3\n")
    monkeypatch.setenv("SEQATTN_CONFIG", str(path))

    with pytest.raises(config.tomllib.TOMLDecodeError):
        config.load_attention_stage_config("minimax_h3_qwen")


def test_each_selected_stage_reads_only_its_section(tmp_path, monkeypatch):
    path = tmp_path / "seqattn.toml"
    path.write_text(
        "minimax_h3 = 'unused-invalid-section'\n\n"
        "[minimax_h3_qwen]\n"
        "qkv_tile_tokens = 64\n"
        "mlp_tile_tokens = 32\n"
    )
    monkeypatch.setenv("SEQATTN_CONFIG", str(path))

    assert config.load_attention_stage_config(
        "minimax_h3_qwen"
    ) == config.AttentionStageConfig(
        qkv_tile_tokens=64,
        mlp_tile_tokens=32,
    )


def test_minimax_h3_config_is_owned_by_seqattn_core():
    with pytest.raises(ValueError, match="unsupported SeqAttn stage"):
        config.load_attention_stage_config("minimax_h3")


def test_qwen_rejects_execution_mode(tmp_path, monkeypatch):
    path = tmp_path / "seqattn.toml"
    path.write_text("[minimax_h3_qwen]\nexecution_mode = 'recompute'\n")
    monkeypatch.setenv("SEQATTN_CONFIG", str(path))

    with pytest.raises(ValueError, match="Qwen always uses materialized"):
        config.load_attention_stage_config("minimax_h3_qwen")


@pytest.mark.parametrize(
    "document, match",
    [
        ("[minimax_h3_qwen]\nmlp_tile_tokens = 0\n", "mlp_tile_tokens"),
        ("[minimax_h3_vae]\ntile_size = 64\n", "tile_size"),
        ("[minimax_h3_vae]\nworkspace_mib = 128\n", "workspace_mib"),
    ],
)
def test_invalid_selected_stage_values_fail(tmp_path, monkeypatch, document, match):
    path = tmp_path / "seqattn.toml"
    path.write_text(document)
    monkeypatch.setenv("SEQATTN_CONFIG", str(path))

    loader = (
        config.load_vae_stage_config
        if "minimax_h3_vae" in document
        else lambda: config.load_attention_stage_config("minimax_h3_qwen")
    )
    with pytest.raises(ValueError, match=match):
        loader()
