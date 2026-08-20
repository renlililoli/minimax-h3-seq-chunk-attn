# MiniMax H3 SeqAttn for ComfyUI

Exact CPU-backed streaming attention for native ComfyUI MiniMax-H3 models.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-%3E%3D%200.30.0-111827)](#requirements)
[![Platform](https://img.shields.io/badge/Linux-NVIDIA%20CUDA-76b900)](#requirements)
[![Weights](https://img.shields.io/badge/INT8%20DiT-NVFP4%20Text-7c3aed)](#models)
[![Capacity](https://img.shields.io/badge/157K%20tokens-8%20GiB%20validated-16a34a)](#usage)

This custom node lets MiniMax-H3 keep long-sequence hidden states and complete
Q/K/V tensors in pinned CPU memory while using a bounded GPU working set. It
uses exact dense attention and works with existing ComfyUI checkpoints without
conversion.

Supported layouts: T2VA, FL2VA, and Ref2VA. A 157,196-token, 243-frame,
1344x768 Ref2VA denoise step has been validated below an 8 GiB process target.

## Install

### ComfyUI Manager

Search for **MiniMax H3 SeqAttn**, install it, and restart ComfyUI.

### Manual

```bash
cd /path/to/ComfyUI/custom_nodes
git clone --branch community/comfyui-minimax-h3-seqattn \
  https://github.com/renlililoli/minimax-h3-seq-chunk-attn.git \
  ComfyUI-MiniMaxH3-SeqAttn
```

No submodules or additional Python packages are required.

## Requirements

- ComfyUI `>= 0.30.0`
- Linux and NVIDIA CUDA
- Python `>= 3.10`
- Batch size 1
- Sufficient CPU DRAM for full hidden and Q/K/V storage

The attention activation path uses BF16 regardless of checkpoint storage
precision. LoRA, diffusion-model replacement patches, NVMe activation backing,
and multi-GPU execution are not currently supported.

## Models

The bundled Ref2VA workflow uses these files from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

```text
ComfyUI/models/
|-- diffusion_models/
|   `-- minimax_h3_ref2va_pruned_int8_convrot.safetensors
|-- text_encoders/
|   `-- qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
`-- vae/
    |-- minimax_h3_video_vae_fp16.safetensors
    `-- minimax_h3_audio_vae_fp32.safetensors
```

Model weights are not included with this node.

## Usage

Import [`workflows/minimax_h3_seqattn_ref2va.json`](workflows/minimax_h3_seqattn_ref2va.json)
or add **MiniMax H3 SeqAttn** immediately after a native MiniMax-H3 diffusion
model loader and connect its `MODEL` output to the sampler or guider.

| Setting | Default | Description |
|---|---:|---|
| `activation_workspace_mib` | `1024` | GPU workspace owned by SeqAttn |
| `kv_chunk_tokens` | `4096` | K/V tokens transferred per tile |
| `planner_mode` | `fit` | Fits resident query tiles to the workspace |
| `enabled` | `true` | Enables or bypasses the patch |

The workspace value is not a whole-process VRAM limit. Lower it if other
ComfyUI nodes or models leave less GPU headroom.

## License

The custom node is GPL-3.0. The bundled SeqAttn runtime is Apache-2.0. See
[`LICENSE`](LICENSE), [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

See the
[original development branch](https://github.com/renlililoli/minimax-h3-seq-chunk-attn/tree/feature/comfyui-minimax-h3-seqattn)
for project history.
