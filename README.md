# MiniMax H3 SeqAttn for ComfyUI

Exact CPU-backed streaming attention for native ComfyUI MiniMax-H3 models.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-%3E%3D%200.30.0-111827)](#requirements)
[![Platform](https://img.shields.io/badge/Linux-NVIDIA%20CUDA-76b900)](#requirements)
[![Weights](https://img.shields.io/badge/INT8%20DiT-NVFP4%20Text-7c3aed)](#models)
[![Capacity](https://img.shields.io/badge/81K%20tokens-8%20GiB%20validated-16a34a)](#validated-run)

This package bounds both major MiniMax-H3 activation paths. The SeqAttn model
patch keeps one long-sequence hidden state and complete Q/K/V tensors in pinned
CPU memory. Attention output, output projection, residual updates, and the
complete MLP remain on GPU in bounded tiles, so only the final block hidden
state returns to CPU. DiT weights use a strict current-block plus next-block
pipeline instead of ComfyUI's native DiT prefetch queue. The Qwen BF16 patch
runs text and visual conditioning with bounded BF16 activations and
layer-offloaded weights. Both work with existing ComfyUI checkpoints without
conversion. MiniMax-H3 text projection and token refinement run once per
sampling job; the refined conditioning is then reused from pinned CPU memory
for the remaining denoise steps.

Supported layouts: T2VA, FL2VA, and Ref2VA. The current `0.4.0` validation is a
complete 20-step, 81,180-token Ref2VA run at 1344x768 with 124 output frames.
Older `0.3.x` measurements are intentionally excluded from the current-version
tables below.

## Validated Run

The `0.4.0` fused DiT path completed a real **20-step MiniMax-H3 Ref2VA edit**
at 1344x768 with 124 reference frames and 124 output frames. It used DynamicVRAM
for model weights, a strict current-block plus next-block SeqAttn prefetch
pipeline, and an 8,192 MiB whole-process target.

### Generated Output

[![Animated preview of the 0.4.0 fused-DiT 20-step Ref2VA output](assets/benchmark/seqattn_ref2va_8g_20step_1344x768_124f_preview.webp)](assets/benchmark/seqattn_ref2va_8g_20step_1344x768_124f.mp4)

Animated 8 fps preview. Click it to open the full-resolution 24 fps MP4.

### Reference Video

[![Animated preview of the 5-second Ref2VA reference clip](assets/benchmark/ref2va_reference_1344x768_124f_preview.webp)](assets/benchmark/ref2va_reference_1344x768_124f.mp4)

Animated 8 fps preview. Click it to open the full-resolution 24 fps MP4. This
clip contains the exact first 124 reference frames used by the run.

| `0.4.0` community-package run | Result |
|---|---:|
| Status | **20/20 denoise steps completed** |
| Packed sequence | **81,180 tokens** |
| Whole-process GPU peak | **7,708 MiB NVML** |
| GPU headroom to 8,192 MiB target | **484 MiB** |
| Denoise GPU peak | **4,386 MiB NVML** |
| Denoise steady state | **4,274-4,276 MiB NVML** |
| Denoise time | **1,812.935 s / 30m 12.935s** |
| First forward with compile/warmup | **252.666 s** |
| Following 19 steady forwards | **81.033 s mean, 80.925-81.144 s range** |
| Complete pipeline | **2,073.534 s / 34m 33.534s** |
| CPU RSS peak | **32,666 MiB** |
| Output | **H.264 + AAC, 1344x768, 124 frames, 24 fps, 5.167 s** |

![Phase-aware process GPU memory profile for the validated 0.4.0 run](docs/assets/community_v040_ref2va_video_20step_20260825_memory.png)

The plot separates Torch allocations, unused Torch reservation, and the
remaining process allocation reported by NVML. The last category includes
DynamicVRAM/AIMDO mappings, VBAR-resident weights, CUDA context memory, and
other non-Torch CUDA allocations; the available trace cannot split those
subsources further. See the
[full experiment record](docs/community_v040_ref2va_video_20step_20260825.md)
for per-phase values and raw artifacts.

<details>
<summary><strong>Prompt and validation details</strong></summary>

```text
Use <Video 1> as the exact reference for the original environment, existing subjects, object layout, camera trajectory, framing, perspective, lens behavior, lighting, colors, materials, timing, and scene continuity. Keep the video photorealistic and preserve all original people and objects in their original roles. Add one new, clearly visible adult woman without replacing or obscuring the original main subjects.

The added woman has shoulder-length dark hair and wears a vivid red jacket, a plain white shirt, black trousers, and dark shoes. Keep her face, hairstyle, clothing, body proportions, and identity fully consistent in every frame. Place her naturally within the scene at the correct scale, depth, and perspective, with physically plausible contact shadows, reflections, occlusion, and lighting that match the original footage.

At the beginning, she enters smoothly from the right edge of the frame and walks at a relaxed natural pace toward the center-right midground. During the middle of the shot, she slows down, stops beside the main area of interest, looks toward the principal object or activity already present in the scene, and clearly points toward it with her left hand. During the final part of the shot, she lowers her pointing hand, turns her head and upper body toward the camera, smiles naturally, and gives one clear friendly wave with her right hand. Her walking, stopping, pointing, turning, and waving must form one continuous believable action with stable anatomy and no sudden position changes.

Do not alter the visual style, weather, time of day, architecture, machinery, background, camera motion, or actions of the original subjects. Do not add any other new person. Do not create duplicate limbs, identity changes, flicker, teleportation, unintended cuts, text, subtitles, logos, or watermarks.
```

- Model: MiniMax-H3 Ref2VA INT8 ConvRot DiT
- Text encoder: Qwen3-VL 32B NVFP4 AWQ with community `prefetch` offload
- Qwen conditioning: 6,174 rows
- Qwen estimated activation: 2,358.12 MiB plus 128 MiB safety
- Query chunk: 5,760 tokens
- K/V tile: 4,096 tokens
- QKV projection tile: 4,096 tokens
- MLP tile: 4,096 tokens
- Seed: 0
- GPU/CPU memory sampling interval: 20 ms
- Weight scheduler: 20 forwards, 1,000 blocks, 5,000 lifecycle records
- Maximum staged blocks: 2
- VBAR-loaded peak: 320 MiB
- Measurements are from one run on August 25, 2026 UTC and have no error bars.
- The run used physical GPU 1 with CPU and memory bound to NUMA node 7. This is
  a single-node capacity/stability result, not the calibrated 56 GB/s
  interleaved host-memory performance result.

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

The validated `0.4.0` run conditioned 6,174 rows. Preflight estimated
2,358.12 MiB of activation storage, or 2,486.12 MiB with the configured safety
reserve. The complete conditioning phase took 160.709 seconds and contained
the run's 7,708 MiB whole-process peak. Preflight accounts for the quadratic
causal mask and retained DeepStack features; the 25K-row value is an absolute
input cap, not a guarantee that every 25K-row composition fits in 8 GiB.

## Install

### ComfyUI Manager

Search for **MiniMax H3 SeqAttn**, install it, and restart ComfyUI.

### Manual

```bash
cd /path/to/ComfyUI/custom_nodes
git clone --branch community/comfyui-minimax-h3-seqattn \
  https://github.com/renlililoli/minimax-h3-seq-chunk-attn.git \
  ComfyUI-MiniMaxH3-SeqAttn
cd ComfyUI-MiniMaxH3-SeqAttn
python -m pip install -e .
```

No Git submodules are required. Installation resolves the pinned
`seqattn-core[dit]` runtime directly from its upstream alpha.3 release commit.

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

For command-line, two-step end-to-end checks after a fresh installation, see
the bundled [`examples/`](examples/README.md) directory. It includes one-click
T2VA, FL2VA, image-reference Ref2VA, and video-reference Ref2VA scripts plus
recorded outputs, memory traces, and validation metadata. Its summary separates
the current `0.4.0` Ref2VA result from retained historical `0.3.x` fixtures.

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

The bundled Ref2VA workflow uses **MiniMax H3 Reference to Video (SeqAttn)**.
It preserves the native reference ordering and payload, but completes Qwen
preflight and text/visual encoding before any reference image, video, or audio
VAE encode. Oversized multimodal prompts therefore fail before expensive VAE
work begins.

The workflow files for all modes use the same `0.4.0` fused DiT integration.
The current release-level end-to-end performance and memory claim is limited
to the Ref2VA run documented above; older split-path T2VA and FL2VA timings are
not carried forward as fused-path results.

| Setting | Default | Description |
|---|---:|---|
| `q_chunk_tokens` | `5760` | Resident query tokens; select from the calibrated host-memory roofline |
| `kv_chunk_tokens` | `4096` | K/V tokens transferred per tile |
| `enabled` | `true` | Enables or bypasses the patch |

QKV projection and MLP tiles are deployment configuration, not workflow node
inputs. The default is 4,096 tokens for both. Override them with the shared
SeqAttn TOML file selected by `SEQATTN_CONFIG`, or
`~/.config/seqattn/config.toml`:

```toml
[minimax_h3]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096
```

The node does not impose a whole-process VRAM limit or silently shrink the
resident query chunk.

## License

The custom node is GPL-3.0. Its external `seqattn-core` dependency is
Apache-2.0. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The independent SeqAttn runtime is installed through the pinned
`seqattn-core[dit]` dependency; this community branch contains only the ComfyUI
integration and workflows.
