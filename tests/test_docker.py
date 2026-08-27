from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_compose_builds_the_runtime_stage():
    lines = (REPOSITORY_ROOT / "docker" / "compose.yaml").read_text().splitlines()
    build_start = lines.index("    build:")
    ports_start = lines.index("    ports:")

    assert "      target: runtime" in lines[build_start:ports_start]


def test_documented_production_builds_select_the_runtime_stage():
    readme = (REPOSITORY_ROOT / "docker" / "README.md").read_text()

    assert readme.count("  --target runtime \\") == 2
    assert "docker compose --env-file docker/.env --file docker/compose.yaml up --build" in readme
