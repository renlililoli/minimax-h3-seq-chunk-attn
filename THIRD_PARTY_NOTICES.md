# Third-party notices

## ComfyUI

The MiniMax-H3 integration in `comfyui_seqattn/minimax_h3.py` adapts the
MiniMax-H3 model execution interfaces provided by ComfyUI.

- Upstream: https://github.com/Comfy-Org/ComfyUI
- License: GNU General Public License v3.0
- Tested baseline: ComfyUI 0.30.0, commit `9a9fdb10`

The complete GPL-3.0 license text is provided in `LICENSE`.

## stream-attn / seqattn_core

This custom node depends on the separately packaged `seqattn-core` runtime from
the `stream-attn` project. The dependency is pinned to an immutable upstream
commit so the community package does not carry or maintain a second copy of the
SeqAttn sources.

- Upstream: https://github.com/renlililoli/stream-attn
- Commit: `2641fdf87cbd158026533a7bfb457ce0f951c965`
- Upstream tag: `v0.3.0-alpha.2`
- Package version: `0.3.0a2`
- License: Apache License 2.0

The dependency distributes its own Apache-2.0 license and package metadata.

## Model weights

No MiniMax-H3, Qwen3-VL, VAE, or other model weights are distributed with this
custom node. Users obtain model files separately under their respective terms.
