# MiniMax H3 SeqAttn for ComfyUI

Exact CPU-backed streaming attention for native ComfyUI MiniMax-H3 models.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-%3E%3D%200.30.0-111827)](#requirements)
[![Platform](https://img.shields.io/badge/Linux-NVIDIA%20CUDA-76b900)](#requirements)
[![Weights](https://img.shields.io/badge/INT8%20DiT-NVFP4%20Text-7c3aed)](#models)
[![Capacity](https://img.shields.io/badge/157K%20tokens-8%20GiB%20validated-16a34a)](#usage)

This package bounds both major MiniMax-H3 activation paths. The SeqAttn model
patch keeps long-sequence hidden states and complete Q/K/V tensors in pinned
CPU memory. The Qwen BF16 patch runs text and visual conditioning with bounded
BF16 activations and layer-offloaded weights. Both work with existing ComfyUI
checkpoints without conversion. MiniMax-H3 text projection and token refinement
run once per sampling job; the refined conditioning is then reused from pinned
CPU memory for the remaining denoise steps.

Supported layouts: T2VA, FL2VA, and Ref2VA. A 157,302-token, 243-frame,
1344x768 Ref2VA denoise step has been validated below an 8 GiB process target.

## 8 GiB Ref2VA Demo

The standalone community package completed a real **20-step MiniMax-H3
Ref2VA generation** at 1344x768 with 243 reference frames and 243 output
frames. The combined text, reference-video, audio, and target-video layout is
157,302 tokens. Whole-process GPU memory peaked at **7,696 MiB** on an RTX
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
| Whole-process GPU peak | **7,696 MiB NVML** |
| GPU headroom to 8,192 MiB target | **496 MiB** |
| Denoise time | **5,875.663 s / 97m 55.663s** |
| Mean denoise time | **293.783 s / 4m 53.783s per step** |
| Complete pipeline | **5,983.785 s / 99m 43.785s** |
| CPU RSS peak | **61.19 GiB** |
| Output | **H.264, 1344x768, 243 frames, 24 fps, 10.125 s** |

A comparable historical 157K-token native ComfyUI run OOMed in its first QKV
projection on a 32 GB RTX 5090 after a 31,590 MiB sampled process peak. SeqAttn
trades CPU DRAM and runtime for the ability to complete the long-video workload
with a bounded GPU working set; this is a capacity result, not a
native-attention speedup claim.

<details>
<summary><strong>Prompt and validation details</strong></summary>

```text
Transform every visible element in <Video 1> into a cohesive, high-quality Japanese anime world. Render every person, face, body, hairstyle, piece of clothing, object, prop, vehicle, building, interior, landscape, sky, surface, texture, reflection, shadow, light, and background detail as hand-drawn 2D animation with clean expressive line art, polished cel shading, vivid anime color design, and consistent stylization. No photorealistic, live-action, or realistic-looking element may remain in any frame. Preserve the exact actions, poses, identities, object layout, timing, framing, camera motion, scene continuity, and composition of the reference video. Keep one continuous stable shot with coherent details. No cuts, no text, no subtitles, no logos.
```

- Model: MiniMax-H3 Ref2VA INT8 ConvRot DiT
- Text encoder: Qwen3-VL 32B NVFP4 AWQ with community `prefetch` offload
- SeqAttn workspace: 1,024 MiB
- K/V tile: 4,096 tokens
- Seed: 0
- GPU/CPU memory sampling interval: 20 ms
- Fresh install: ComfyUI `9a9fdb1`, clean community-branch checkout
- Complete memory trace: 88,208 process samples
- The generated benchmark MP4 contains video only; the reference MP4 also has
  an AAC audio stream.
- Measurements are from one run on August 21, 2026 UTC and have no error bars.

</details>

## Qwen Conditioning

Add **MiniMax H3 Qwen BF16 Offload** after the MiniMax `CLIPLoader` and before
the MiniMax conditioning node. The bundled workflow already includes it.

The node converts token, vision, and decoder activations to BF16, uses an
in-place decoder MLP, reuses hidden-state storage between layers, and rejects
oversized text/image/video presentations before the vision tower runs.

| Setting | Default | Description |
|---|---:|---|
| `offload_mode` | `prefetch` | `prefetch` uses two asynchronous weight streams; `extreme` disables asynchronous prefetch for the lowest transient weight footprint |
| `activation_limit_mib` | `5888` | Per-layer Qwen activation-plan limit |
| `max_conditioning_rows` | `25000` | Hard limit for the complete Qwen presentation |
| `preflight_safety_mib` | `128` | Reserve added to the calibrated preflight estimate |

The fresh 20-step demo conditioned 11,340 Qwen rows. On an RTX 5090, the default
`prefetch` policy completed all 50 decoder layers in 13.28 seconds at a 5,242
MiB text-window process peak. A 21.4K-row multi-reference probe completed in
19.60 seconds at 7,724 MiB under the same 8 GiB target. Preflight accounts for
the quadratic causal mask and retained DeepStack features; the 25K-row value is
an absolute input cap, not a guarantee that every 25K-row composition fits in
8 GiB.

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

The attention and Qwen activation paths use BF16 regardless of checkpoint
storage precision. The Qwen node requires `CLIPLoader` device `default`. LoRA,
diffusion-model replacement patches, NVMe activation backing, and multi-GPU
execution are not currently supported.

## Models

The bundled workflows use these files from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

```text
ComfyUI/models/
|-- diffusion_models/
|   |-- minimax_h3_fl2va_pruned_int8_convrot.safetensors
|   `-- minimax_h3_ref2va_pruned_int8_convrot.safetensors
|-- text_encoders/
|   `-- qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
`-- vae/
    |-- minimax_h3_video_vae_fp16.safetensors
    `-- minimax_h3_audio_vae_fp32.safetensors
```

Model weights are not included with this node.

## Usage

Import the workflow matching the generation mode:

| Mode | Workflow | Inputs |
|---|---|---|
| T2VA | [`minimax_h3_seqattn_t2va.json`](workflows/minimax_h3_seqattn_t2va.json) | Prompt |
| First-frame video | [`minimax_h3_seqattn_first_frame.json`](workflows/minimax_h3_seqattn_first_frame.json) | Prompt + first frame |
| Last-frame video | [`minimax_h3_seqattn_last_frame.json`](workflows/minimax_h3_seqattn_last_frame.json) | Prompt + last frame |
| FL2VA | [`minimax_h3_seqattn_fl2va.json`](workflows/minimax_h3_seqattn_fl2va.json) | Prompt + first and last frames |
| Ref2VA | [`minimax_h3_seqattn_ref2va.json`](workflows/minimax_h3_seqattn_ref2va.json) | Prompt + image/video/audio references |

The four T2VA/FL2VA workflows use the same FL2VA checkpoint. The first frame
anchors frame 0; the last frame anchors the final aligned output frame. To
patch an existing workflow, add **MiniMax H3 SeqAttn** immediately after the
diffusion-model loader and **MiniMax H3 Qwen BF16 Offload** immediately after
the MiniMax `CLIPLoader`. For bounded keyframe encoding and video decoding,
pass the video VAE through **MiniMax H3 VAE Streaming**; the bundled workflows
use a validated 192-pixel tile and 512 MiB activation workspace. This also
streams long VAE inputs and decoded frames through CPU memory.

All four FL2VA checkpoint modes were validated at 1344x768 with 56 output
frames under an 8 GiB process target on an RTX 5090:

| Mode | Validation | Packed tokens | GPU peak | Denoise |
|---|---:|---:|---:|---:|
| T2VA | 20/20 steps | 17,460 | 7,430 MiB | 322.038 s |
| First frame | 1/1 step | 19,390 | 7,332 MiB | 50.240 s |
| Last frame | 1/1 step | 19,388 | 7,332 MiB | 40.758 s |
| First + last frames | 20/20 steps | 21,419 | 7,272 MiB | 400.612 s |

The complete FL2VA run averaged 20.031 seconds per denoise step. Measurements
are from single runs on August 21, 2026 UTC and have no error bars.

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
