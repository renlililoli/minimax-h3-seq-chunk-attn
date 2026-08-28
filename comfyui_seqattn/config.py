from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class AttentionStageConfig:
    execution_mode: str = "materialized"
    qkv_tile_tokens: int = 4096
    mlp_tile_tokens: int = 4096


@dataclass(frozen=True)
class VAEStageConfig:
    tile_size: int = 192
    workspace_mib: int = 512


def _default_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "seqattn" / "config.toml"


def _load_document() -> dict:
    configured_path = os.environ.get("SEQATTN_CONFIG")
    path = Path(configured_path).expanduser() if configured_path else _default_config_path()
    if not path.exists():
        if configured_path:
            raise FileNotFoundError(f"SEQATTN_CONFIG does not exist: {path}")
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _section(document: dict, name: str) -> dict:
    section = document.get(name, {})
    if not isinstance(section, dict):
        raise TypeError(f"seqattn config [{name}] must be a TOML table")
    return section


def _positive_int(section: dict, section_name: str, name: str, default: int) -> int:
    value = section.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section_name}.{name} must be a positive integer")
    return value


def load_attention_stage_config(section_name: str) -> AttentionStageConfig:
    if section_name not in {"minimax_h3", "minimax_h3_qwen"}:
        raise ValueError(f"unsupported SeqAttn stage config: {section_name}")
    defaults = AttentionStageConfig()
    section = _section(_load_document(), section_name)
    if section_name == "minimax_h3_qwen" and "execution_mode" in section:
        raise ValueError(
            "minimax_h3_qwen.execution_mode is unsupported; "
            "Qwen always uses materialized execution"
        )
    execution_mode = section.get("execution_mode", defaults.execution_mode)
    if not isinstance(execution_mode, str) or execution_mode not in {
        "materialized",
        "recompute",
    }:
        raise ValueError(
            f"{section_name}.execution_mode must be 'materialized' or 'recompute'"
        )
    return AttentionStageConfig(
        execution_mode=execution_mode,
        qkv_tile_tokens=_positive_int(
            section, section_name, "qkv_tile_tokens", defaults.qkv_tile_tokens
        ),
        mlp_tile_tokens=_positive_int(
            section, section_name, "mlp_tile_tokens", defaults.mlp_tile_tokens
        ),
    )


def load_vae_stage_config() -> VAEStageConfig:
    section_name = "minimax_h3_vae"
    defaults = VAEStageConfig()
    section = _section(_load_document(), section_name)
    tile_size = _positive_int(section, section_name, "tile_size", defaults.tile_size)
    workspace_mib = _positive_int(
        section, section_name, "workspace_mib", defaults.workspace_mib
    )
    if tile_size < 128:
        raise ValueError("minimax_h3_vae.tile_size must be at least 128")
    if workspace_mib < 256:
        raise ValueError("minimax_h3_vae.workspace_mib must be at least 256")
    return VAEStageConfig(tile_size=tile_size, workspace_mib=workspace_mib)


__all__ = [
    "AttentionStageConfig",
    "VAEStageConfig",
    "load_attention_stage_config",
    "load_vae_stage_config",
]
