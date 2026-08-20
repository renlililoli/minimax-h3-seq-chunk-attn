# Third-party notices

## ComfyUI

The MiniMax-H3 integration in `comfyui_seqattn/minimax_h3.py` adapts the
MiniMax-H3 model execution interfaces provided by ComfyUI.

- Upstream: https://github.com/Comfy-Org/ComfyUI
- License: GNU General Public License v3.0
- Tested baseline: ComfyUI 0.30.0, commit `9a9fdb10`

The complete GPL-3.0 license text is provided in `LICENSE`.

## stream-attn / seqattn_core

`comfyui_seqattn/_vendor/seqattn_core` is vendored from the `stream-attn`
project so ComfyUI Manager installations do not depend on Git submodules or an
unrelated package that currently occupies the `seqattn` name on PyPI.

- Upstream: https://github.com/renlililoli/stream-attn
- Commit: `9eb3dfb1c7d9df1d84ec96b6b088896864ed06d5`
- Upstream version: `0.3.0`
- License: Apache License 2.0

The Apache-2.0 license text is provided in `LICENSES/Apache-2.0.txt`. The
community package contains only the projection and streaming-attention runtime
dependency closure. Those source modules are kept at the recorded upstream
commit; package exports are narrowed to the symbols used by this custom node.

## Model weights

No MiniMax-H3, Qwen3-VL, VAE, or other model weights are distributed with this
custom node. Users obtain model files separately under their respective terms.
