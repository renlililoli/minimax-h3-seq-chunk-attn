# Third-party notices

## ComfyUI

The MiniMax-H3 integration in `comfyui_seqattn/minimax_h3.py`,
`comfyui_seqattn/qwen.py`, and `comfyui_seqattn/vae.py` adapts the MiniMax-H3
model execution interfaces provided by ComfyUI.

- Upstream: https://github.com/Comfy-Org/ComfyUI
- License: GNU General Public License v3.0
- Tested baseline: ComfyUI 0.30.0, commit `9a9fdb10`

The complete GPL-3.0 license text is provided in `LICENSE`.

## ComfyUI-Manager

The Docker image includes ComfyUI-Manager at a fixed commit. ComfyUI-Manager is
not included in this custom node's source distribution or wheel.

- Upstream: https://github.com/ltdrdata/ComfyUI-Manager
- Commit: `d47c9346190397e1c316bc5a82155faaf9f5d700`
- License: GNU General Public License v3.0

The Manager checkout in the built image contains its upstream `LICENSE.txt`.

## NVIDIA PyTorch Container

The optional RTX 50-series Docker image starts from the publicly pullable
NVIDIA PyTorch `26.01-py3` container. This base is not included in the custom
node's source distribution or wheel.

- Catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Image: `nvcr.io/nvidia/pytorch`
- Digest: `sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e`
- NVIDIA Software License Agreement:
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/
- NVIDIA AI Product-Specific Terms:
  https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/

Review the NVIDIA terms before building, using, or redistributing the Docker
image. Pulling the base image anonymously does not remove those terms.

## stream-attn / seqattn_core

This custom node depends on the separately packaged `seqattn-core` runtime from
the `stream-attn` project. The dependency is pinned to an immutable upstream
commit so the community package does not carry or maintain a second copy of the
SeqAttn sources.

- Upstream: https://github.com/renlililoli/stream-attn
- Commit: `5a52f7ea8e83d9187ed39d03e66eccc305eaaaf3`
- Package version: `0.3.0a4`
- Install extra: `dit`
- License: Apache License 2.0

The dependency distributes its own Apache-2.0 license and package metadata.

## Model weights

No MiniMax-H3, Qwen3-VL, VAE, or other model weights are distributed with this
custom node. Users obtain model files separately under their respective terms.
