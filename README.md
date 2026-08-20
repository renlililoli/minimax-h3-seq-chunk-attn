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

## 8 GiB Ref2VA Demo

The standalone community package completed a real **20-step MiniMax-H3
Ref2VA generation** at 1344x768 with 243 reference frames and 243 output
frames. The combined text, reference-video, audio, and target-video layout is
157,196 tokens. Whole-process GPU memory peaked at **7,982 MiB** on an RTX
5090.

### Generated Output

[![Animated preview of the generated 20-step Ref2VA output](assets/benchmark/seqattn_ref2va_8g_20step_1344x768_243f_preview.webp)](assets/benchmark/seqattn_ref2va_8g_20step_1344x768_243f.mp4)

Animated 8 fps preview. Click it to open the full-resolution 24 fps MP4.

### Reference Video

[![Animated preview of the Ref2VA reference video](assets/benchmark/ref2va_reference_1344x768_243f_preview.webp)](assets/benchmark/ref2va_reference_1344x768_243f.mp4)

Animated 8 fps preview. Click it to open the full-resolution 24 fps MP4 with
the original AAC audio track.

| Clean community-package run | Result |
|---|---:|
| Status | **20/20 denoise steps completed** |
| Whole-process GPU peak | **7,982 MiB NVML** |
| GPU headroom to 8,192 MiB target | **210 MiB** |
| Denoise time | **5,536.855 s / 92m 16.855s** |
| Mean denoise time | **276.843 s / 4m 36.843s per step** |
| Complete pipeline | **5,691.244 s / 94m 51.244s** |
| CPU RSS peak | **59.76 GiB** |
| Output | **H.264, 1344x768, 243 frames, 24 fps, 10.125 s** |

The same 157,196-token workload previously caused the historical native
ComfyUI path to OOM in its first QKV projection on a 32 GB RTX 5090, after a
31,590 MiB sampled process peak. SeqAttn trades CPU DRAM and runtime for the
ability to complete the long-video workload with a bounded GPU working set;
this is a capacity result, not a native-attention speedup claim.

<details>
<summary><strong>Prompt and validation details</strong></summary>

```text
Use <Video 1> as the exact motion, camera, subject, and scene reference. Continue it as one coherent cinematic shot with natural motion, stable identity, photorealistic detail, synchronized ambient sound, no cuts, no text, no logos.
```

- Model: MiniMax-H3 Ref2VA INT8 ConvRot DiT
- Text encoder: Qwen3-VL 32B NVFP4 AWQ with GPU layer offload
- SeqAttn workspace: 1,024 MiB
- K/V tile: 4,096 tokens
- Seed: 0
- GPU/CPU memory sampling interval: 20 ms
- The generated benchmark MP4 contains video only; the reference MP4 also has
  an AAC audio stream.
- Measurements are from one run on August 20, 2026 UTC and have no error bars.

</details>

The full protocol, phase timings, packed-token layout, all persisted GPU/CPU
memory measurements, checksums, raw JSON, and measurement limitations are in
the
[development-branch technical report](https://github.com/renlililoli/minimax-h3-seq-chunk-attn/blob/feature/comfyui-minimax-h3-seqattn/docs/comfyui_minimax_h3_seqattn_8g_20step_20260820.md).

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
