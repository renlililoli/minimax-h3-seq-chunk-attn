# ComfyUI MiniMax-H3 SeqAttn

> Run long MiniMax-H3 video sequences with exact CPU-backed streaming
> attention and existing ComfyUI quantized weights.

[![ComfyUI](https://img.shields.io/badge/runtime-ComfyUI-111827?style=for-the-badge)](#run-with-comfyui)
[![GPU](https://img.shields.io/badge/tested-RTX%205090-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](#strict-8-gib-result)
[![Weights](https://img.shields.io/badge/weights-INT8%20%2B%20NVFP4-7c3aed?style=for-the-badge)](#supported-models)
[![Capacity](https://img.shields.io/badge/157K%20tokens-8%20GiB%20passed-16a34a?style=for-the-badge)](#strict-8-gib-result)

This branch adds a native ComfyUI custom node that replaces MiniMax-H3's
full-sequence attention and activation path with `seqattn`. Model hidden
states and complete Q/K/V tensors are backed by pinned CPU memory while the
GPU holds a bounded working set for projection, attention, MLP, and output
assembly.

The attention remains exact dense attention. Existing ComfyUI checkpoints do
not need to be converted.

## Strict 8 GiB result

The validated Ref2VA workload uses a real 243-frame reference video and a
243-frame target at 1344x768 and 24 fps. Its packed sequence contains 157,196
tokens:

| Segment | Tokens |
|---|---:|
| Text | 11,234 |
| Reference video | 72,576 |
| Audio | 810 |
| Target video | 72,576 |
| **Total** | **157,196** |

| Exact workload | SeqAttn ComfyUI, strict 8 GiB | Native ComfyUI, RTX 5090 32 GB |
|---|---:|---:|
| One denoise step | **Completed in 369.39 s** | OOM in the first QKV projection |
| Process GPU peak | **7,982 MiB** | 31,590 MiB before failure |
| Immediate allocation | 1,024 MiB planned workspace | Requested another 6.30 GiB |
| Compute-only target estimate | Measured | 232-294 s/step, extrapolated |

Native ComfyUI uses `NORMAL_VRAM`, DynamicVRAM, and two-stream asynchronous
weight offload in this comparison. It cannot materialize the full-sequence QKV
result on a device with 31.36 GiB usable memory. SeqAttn uses 74.7% less
process GPU memory than the native pre-OOM peak and completes the step below
8 GiB.

Because native cannot finish the exact workload, its target time is estimated
from completed 26K-49K-token runs. The measured SeqAttn step is approximately
1.25x-1.59x slower than that compute-only estimate. This is a capacity result,
not a claimed speedup over native attention.

A separate validation completed denoise, VAE decode, and a 243-frame MP4 under
the same 8 GiB target. Cached GPU layer offload reduced Qwen3-VL text
conditioning from 553.17 s on CPU to 24.17 s without retaining the text
encoder during DiT denoising.

See the [complete experiment report](docs/comfyui_minimax_h3_8g_vs_native_20260820.md)
for the protocol, native scaling points, profiler breakdown, artifacts, and
measurement limitations. The canonical summary is also available as
[JSON](docs/comfyui_minimax_h3_8g_vs_native_20260820_results.json).

## Supported models

The current integration supports:

- Native ComfyUI `MiniMaxH3Model`.
- T2VA, FL2VA, and Ref2VA packed layouts.
- Comfy-Org pruned BF16 or ComfyUI-native quantized DiT checkpoints.
- INT8 ConvRot MiniMax-H3 DiT weights.
- NVFP4-AWQ or INT8 Qwen3-VL text encoders.
- BF16 attention activations, regardless of checkpoint storage precision.
- CUDA on Linux with batch size 1.

The tested downloaded model set uses:

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

## Run with ComfyUI

Clone the repository with the `seqattn` submodule:

```bash
git clone --branch feature/comfyui-minimax-h3-seqattn --recurse-submodules \
  git@github.com:renlililoli/minimax-h3-seq-chunk-attn.git
cd minimax-h3-seq-chunk-attn
```

The provided image extends a CUDA 12.8 ComfyUI base image named
`comfyui:cu128`. Build and start the service with an existing ComfyUI models
directory mounted into the container:

```bash
COMFYUI_MODELS_DIR=/path/to/ComfyUI/models \
COMFYUI_OUTPUT_DIR=/path/to/ComfyUI/output \
docker compose -f docker-compose.comfyui.yml up --build
```

Open ComfyUI on port `8188`. The bundled
[`minimax_h3_seqattn_ref2va.json`](workflows/minimax_h3_seqattn_ref2va.json)
workflow is mounted into ComfyUI's workflow directory by the Compose service.

For a manual installation, install `extern/seqattn` into ComfyUI's Python
environment and place
[`ComfyUI-SeqAttn`](comfyui_custom_nodes/ComfyUI-SeqAttn) under
`ComfyUI/custom_nodes/`.

## Node usage

Insert **MiniMax H3 SeqAttn** immediately after the MiniMax-H3 diffusion model
loader and connect its `MODEL` output to the sampler.

| Setting | Meaning | Validated 8 GiB value |
|---|---|---:|
| `activation_workspace_mib` | GPU workspace owned by SeqAttn | `1024` |
| `kv_chunk_tokens` | K/V tokens transferred per tile | `4096` |
| `planner_mode` | Resident-query planner | `fit` |
| `enabled` | Return patched or original model | `true` |

`activation_workspace_mib` is not a whole-process VRAM limit. Interactive
ComfyUI also holds CUDA context, model layers, VAE tensors, allocator cache,
and other node state. The strict 8 GiB result uses the benchmark runner's
allocator-aware process budget in addition to the 1,024 MiB SeqAttn workspace.

The custom node rejects non-MiniMax models. LoRA, diffusion-model replacement
patches, NVMe activation backing, and multi-GPU execution are outside the
current supported path.

## Reproduce the benchmark

The benchmark directly executes ComfyUI's native loaders, MiniMax-H3
conditioning, sampler, and VAEs with the downloaded ComfyUI checkpoints:

```bash
workspace/benchmarks/run_comfyui_ref2va_8g.sh streaming
workspace/benchmarks/run_comfyui_ref2va_8g.sh native
```

The scripts default to the model and reference-video mounts used by the test
host. Adjust those mounts before running on another machine. Select the GPU and
output directory with `COMFYUI_BENCH_GPU` and `COMFYUI_BENCH_OUTPUT_DIR`.

Capture an Nsight Systems denoise profile with:

```bash
workspace/benchmarks/run_comfyui_ref2va_profile.sh
```

The measured forward spends 190.74 s, or 62.0%, in streamed attention; QKV
projection takes 50.58 s and MLP takes 54.63 s. The full analysis is in the
[ComfyUI SeqAttn profile](docs/comfyui_minimax_h3_seqattn_profile_20260820.md).

## Repository layout

| Path | Purpose |
|---|---|
| `comfyui_custom_nodes/ComfyUI-SeqAttn` | ComfyUI extension and tests |
| `extern/seqattn` | Exact CPU-backed streaming attention operator |
| `workflows` | Importable ComfyUI workflows |
| `workspace/benchmarks` | Strict-memory runner, profiler, and result tooling |
| `docs` | ComfyUI experiment reports and machine-readable summaries |
| `Dockerfile.comfyui-seqattn` | ComfyUI image with SeqAttn installed |
| `docker-compose.comfyui.yml` | GPU service and model/output mounts |

## Limitations

- CPU DRAM capacity and PCIe bandwidth replace GPU activation capacity as the
  main resource constraints.
- Long sequences are slower than native GPU attention when native fits in HBM.
- The 8 GiB measurements are single-run system characterization on an RTX
  5090, not cross-GPU performance guarantees.
- The native 157K target time is extrapolated because native OOMs before
  completing one step.
- The current path uses CPU DRAM for activations; it does not page activations
  to NVMe.
