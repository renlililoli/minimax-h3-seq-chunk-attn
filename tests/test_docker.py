from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_compose_builds_the_runtime_stage():
    lines = (REPOSITORY_ROOT / "docker" / "compose.yaml").read_text().splitlines()
    build_start = lines.index("    build:")
    ports_start = lines.index("    ports:")

    assert "      target: runtime" in lines[build_start:ports_start]
    assert '    user: "${HOST_UID}:${HOST_GID}"' in lines[build_start:ports_start]
    assert "      HOME: /opt/ComfyUI/user" in lines


def test_environment_template_declares_host_user_mapping():
    environment = (REPOSITORY_ROOT / "docker" / ".env.example").read_text()

    assert "HOST_UID=1000" in environment
    assert "HOST_GID=1000" in environment
    assert "COMFYUI_TEMP_DIRECTORY=/opt/ComfyUI/user" in environment


def test_entrypoint_places_temp_files_under_the_writable_user_mount():
    entrypoint = (REPOSITORY_ROOT / "docker" / "entrypoint.sh").read_text()

    assert '--temp-directory "${COMFYUI_TEMP_DIRECTORY:-/opt/ComfyUI/user}"' in entrypoint


def test_runtime_allows_the_host_user_to_read_the_pinned_comfyui_revision():
    dockerfile = (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text()

    assert "git config --system --add safe.directory /opt/ComfyUI" in dockerfile


def test_documented_production_builds_select_the_runtime_stage():
    readme = (REPOSITORY_ROOT / "docker" / "README.md").read_text()

    assert readme.count("  --target runtime \\") == 2
    assert "docker compose --env-file docker/.env --file docker/compose.yaml up --build" in readme
