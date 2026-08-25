import subprocess
from pathlib import Path

SUPPORTED_COMFYUI_VERSION = "0.30.0"
SUPPORTED_COMFYUI_COMMIT = "9a9fdb10ed144ce760d9682cb247526ea23cc525"


def _comfyui_git_commit(comfyui_version_file: str) -> str | None:
    repository = Path(comfyui_version_file).resolve().parent
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _validate_comfyui_version() -> None:
    import comfyui_version as version_module

    if version_module.__version__ != SUPPORTED_COMFYUI_VERSION:
        raise RuntimeError(
            "MiniMax H3 SeqAttn requires ComfyUI "
            f"{SUPPORTED_COMFYUI_VERSION} at commit {SUPPORTED_COMFYUI_COMMIT}; "
            f"found ComfyUI {version_module.__version__}."
        )
    commit = _comfyui_git_commit(version_module.__file__)
    if commit is not None and commit != SUPPORTED_COMFYUI_COMMIT:
        raise RuntimeError(
            "MiniMax H3 SeqAttn requires ComfyUI commit "
            f"{SUPPORTED_COMFYUI_COMMIT}; found {commit}."
        )


async def comfy_entrypoint():
    _validate_comfyui_version()
    if __package__:
        from .comfyui_seqattn.nodes import comfy_entrypoint as load_extension
    else:
        from comfyui_seqattn.nodes import comfy_entrypoint as load_extension

    return await load_extension()

__all__ = [
    "SUPPORTED_COMFYUI_COMMIT",
    "SUPPORTED_COMFYUI_VERSION",
    "comfy_entrypoint",
]
