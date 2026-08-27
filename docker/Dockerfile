# syntax=docker/dockerfile:1.7

ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e
FROM ${PYTORCH_BASE_IMAGE} AS runtime

ARG COMFYUI_VERSION=0.30.0
ARG COMFYUI_COMMIT=9a9fdb10ed144ce760d9682cb247526ea23cc525
ARG COMFYUI_REPOSITORY=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFYUI_MANAGER_COMMIT=d47c9346190397e1c316bc5a82155faaf9f5d700
ARG COMFYUI_MANAGER_REPOSITORY=https://github.com/ltdrdata/ComfyUI-Manager.git
ARG TORCH_VERSION=2.10.0+cu128
ARG TORCHVISION_VERSION=0.25.0+cu128
ARG TORCHAUDIO_VERSION=2.10.0+cu128
ARG TORCH_CUDA_VERSION=12.8
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG COMFY_AIMDO_VERSION=0.4.11
ARG SEQATTN_CORE_COMMIT=86049c058a4dfb26da408e79ca2c95677ebbd250

LABEL org.opencontainers.image.title="MiniMax H3 SeqAttn for ComfyUI"
LABEL org.opencontainers.image.description="Pinned ComfyUI 0.30.0 and CUDA 12.8 runtime for NVIDIA RTX 50-series GPUs"
LABEL org.opencontainers.image.source="https://github.com/renlililoli/minimax-h3-seq-chunk-attn"
LABEL io.seqattn.comfyui.version="${COMFYUI_VERSION}"
LABEL io.seqattn.comfyui.commit="${COMFYUI_COMMIT}"
LABEL io.seqattn.comfyui-manager.commit="${COMFYUI_MANAGER_COMMIT}"
LABEL io.seqattn.seqattn-core.commit="${SEQATTN_CORE_COMMIT}"

USER root

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip uninstall --yes torch torchvision torchaudio \
    && python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"

RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin "${COMFYUI_REPOSITORY}" \
    && git -C /opt/ComfyUI fetch --depth 1 origin "${COMFYUI_COMMIT}" \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = "${COMFYUI_COMMIT}"

WORKDIR /opt/ComfyUI

RUN grep -viE '^(torch|torchvision|torchaudio)([<>= ]|$)' \
        requirements.txt > /tmp/comfyui-requirements.txt \
    && python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --requirement /tmp/comfyui-requirements.txt \
    && rm /tmp/comfyui-requirements.txt

RUN git init custom_nodes/ComfyUI-Manager \
    && git -C custom_nodes/ComfyUI-Manager remote add origin \
        "${COMFYUI_MANAGER_REPOSITORY}" \
    && git -C custom_nodes/ComfyUI-Manager fetch --depth 1 origin \
        "${COMFYUI_MANAGER_COMMIT}" \
    && git -C custom_nodes/ComfyUI-Manager checkout --detach FETCH_HEAD \
    && test "$(git -C custom_nodes/ComfyUI-Manager rev-parse HEAD)" = \
        "${COMFYUI_MANAGER_COMMIT}" \
    && python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --requirement custom_nodes/ComfyUI-Manager/requirements.txt

RUN git config --system --add safe.directory /opt/ComfyUI

RUN COMFYUI_VERSION="${COMFYUI_VERSION}" \
    COMFYUI_COMMIT="${COMFYUI_COMMIT}" \
    COMFYUI_MANAGER_COMMIT="${COMFYUI_MANAGER_COMMIT}" \
    TORCH_VERSION="${TORCH_VERSION}" \
    TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION}" \
    COMFY_AIMDO_VERSION="${COMFY_AIMDO_VERSION}" \
    python - <<'PY'
import os
import subprocess
from importlib.metadata import version

import comfyui_version
import torch


def require(actual: str, expected: str, name: str) -> None:
    if actual != expected:
        raise SystemExit(f"{name} mismatch: expected {expected}, found {actual}")


commit = subprocess.check_output(
    ["git", "-c", "safe.directory=/opt/ComfyUI", "-C", "/opt/ComfyUI", "rev-parse", "HEAD"],
    text=True,
).strip()
require(comfyui_version.__version__, os.environ["COMFYUI_VERSION"], "ComfyUI version")
require(commit, os.environ["COMFYUI_COMMIT"], "ComfyUI commit")
require(torch.__version__, os.environ["TORCH_VERSION"], "PyTorch version")
require(torch.version.cuda or "", os.environ["TORCH_CUDA_VERSION"], "PyTorch CUDA")
require(version("comfy-aimdo"), os.environ["COMFY_AIMDO_VERSION"], "comfy-aimdo version")

manager_commit = subprocess.check_output(
    [
        "git",
        "-C",
        "/opt/ComfyUI/custom_nodes/ComfyUI-Manager",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()
require(
    manager_commit,
    os.environ["COMFYUI_MANAGER_COMMIT"],
    "ComfyUI-Manager commit",
)
PY

RUN install -d /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn
COPY __init__.py pyproject.toml README.md CHANGELOG.md LICENSE THIRD_PARTY_NOTICES.md \
    /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn/
COPY comfyui_seqattn/ \
    /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn/comfyui_seqattn/
COPY workflows/ /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn/workflows/
COPY docker/seqattn.toml /etc/seqattn/config.toml
COPY docker/entrypoint.sh /opt/seqattn/bin/start-comfyui
RUN chmod 0755 /opt/seqattn/bin/start-comfyui

WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn
RUN python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        "seqattn-core[dit] @ https://github.com/renlililoli/stream-attn/archive/${SEQATTN_CORE_COMMIT}.tar.gz" \
    && python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        .

RUN SEQATTN_CORE_COMMIT="${SEQATTN_CORE_COMMIT}" python - <<'PY'
from importlib.metadata import version

import comfy_aimdo.control
import comfy_aimdo.model_vbar
import seqattn_core

from comfyui_seqattn import __version__ as node_version

print(f"MiniMax H3 SeqAttn {node_version}")
print(f"seqattn-core {version('seqattn-core')}")
print(f"seqattn module {seqattn_core.__file__}")
PY

ENV CUDA_MODULE_LOADING=LAZY \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    SEQATTN_CONFIG=/etc/seqattn/config.toml

WORKDIR /opt/ComfyUI
EXPOSE 8188
VOLUME ["/opt/ComfyUI/models", "/opt/ComfyUI/input", "/opt/ComfyUI/output", "/opt/ComfyUI/user"]

CMD ["/opt/seqattn/bin/start-comfyui"]

FROM runtime AS checks

ARG BUILD_VERSION=1.4.0
ARG PYTEST_VERSION=8.1.1
ARG RUFF_VERSION=0.16.4

RUN python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-deps \
        "build==${BUILD_VERSION}" \
        "pytest==${PYTEST_VERSION}" \
        "ruff==${RUFF_VERSION}"

COPY docker/check.sh /opt/seqattn/bin/check-repo
RUN chmod 0755 /opt/seqattn/bin/check-repo

WORKDIR /workspace
CMD ["/opt/seqattn/bin/check-repo"]
