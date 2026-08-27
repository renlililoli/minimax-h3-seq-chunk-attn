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

    assert (
        "PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:"
        "38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e"
        in environment
    )
    assert "COMFYUI_BASE_IMAGE=" not in environment
    assert "HOST_UID=1000" in environment
    assert "HOST_GID=1000" in environment
    assert "COMFYUI_TEMP_DIRECTORY=/opt/ComfyUI/user" in environment


def test_entrypoint_places_temp_files_under_the_writable_user_mount():
    entrypoint = (REPOSITORY_ROOT / "docker" / "entrypoint.sh").read_text()

    assert '--temp-directory "${COMFYUI_TEMP_DIRECTORY:-/opt/ComfyUI/user}"' in entrypoint


def test_runtime_allows_the_host_user_to_read_the_pinned_comfyui_revision():
    dockerfile = (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text()

    assert "git config --system --add safe.directory /opt/ComfyUI" in dockerfile


def test_runtime_rebuilds_comfyui_from_the_public_pytorch_base():
    dockerfile = (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text()

    assert (
        "ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:"
        "38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e"
        in dockerfile
    )
    assert "FROM ${PYTORCH_BASE_IMAGE} AS runtime" in dockerfile
    assert "comfyui@sha256:4708ab49" not in dockerfile
    assert '"torch==${TORCH_VERSION}"' in dockerfile
    assert 'git -C /opt/ComfyUI fetch --depth 1 origin "${COMFYUI_COMMIT}"' in dockerfile
    assert "git -C custom_nodes/ComfyUI-Manager fetch --depth 1 origin" in dockerfile
    assert '"${COMFYUI_MANAGER_COMMIT}"' in dockerfile

    compose = (REPOSITORY_ROOT / "docker" / "compose.yaml").read_text()
    assert "PYTORCH_BASE_IMAGE:" in compose
    assert "COMFYUI_BASE_IMAGE:" not in compose


def test_third_party_notices_cover_docker_inputs():
    notices = (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text()

    assert "nvcr.io/nvidia/pytorch" in notices
    assert "38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e" in notices
    assert "d47c9346190397e1c316bc5a82155faaf9f5d700" in notices
    assert "NVIDIA Software License Agreement" in notices
    assert "GNU General Public License v3.0" in notices


def test_readme_does_not_publish_the_retired_qwen_benchmark():
    readme = (REPOSITORY_ROOT / "README.md").read_text()

    assert "Historical 0.4.0 Validation" not in readme
    assert "historical-040-validation" not in readme
    assert "community_v040_ref2va_video_20step_20260825" not in readme


def test_documented_production_builds_select_the_runtime_stage():
    readme = (REPOSITORY_ROOT / "docker" / "README.md").read_text()

    assert readme.count("  --target runtime \\") == 2
    assert "docker compose --env-file docker/.env --file docker/compose.yaml up --build" in readme


def test_release_build_ignores_local_agent_and_archive_state():
    ignored = set((REPOSITORY_ROOT / ".gitignore").read_text().splitlines())

    assert {
        "/.agents/",
        "/.claude/",
        "/.codex/",
        "/.ruff_cache/",
        "/.worktree-archive/",
        "/outputs/",
        "/sessions/",
    } <= ignored
