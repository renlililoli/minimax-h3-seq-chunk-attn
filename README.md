# MiniMax H3 SeqAttn for ComfyUI

Bounded CPU-backed dense and streamed Sol attention for native ComfyUI
MiniMax-H3 models.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-0.30.0%20pinned-111827)](#requirements)
[![Platform](https://img.shields.io/badge/Linux-NVIDIA%20CUDA-76b900)](#requirements)
[![Weights](https://img.shields.io/badge/INT8%20DiT-NVFP4%20Text-7c3aed)](#models)

This package bounds both major MiniMax-H3 activation paths. The DiT and Qwen
SeqAttn patches keep long-sequence hidden state and complete Q/K/V tensors in
pinned CPU memory while attention, projections, residual updates, mergers, and
MLPs execute in bounded GPU tiles. Both stages use a strict current-layer plus
next-layer Dynamic VBAR weight pipeline instead of ComfyUI's native prefetch
queue. The independent video VAE patch streams spatial tiles and long inputs.
All three patches work with existing ComfyUI checkpoints without conversion.
MiniMax-H3 text projection and token refinement run once per sampling job; the
refined conditioning is then reused from pinned CPU memory for the remaining
denoise steps.

Supported layouts are T2VA, FL2VA, and Ref2VA. The bundled workflows are
functional templates for the current Qwen, DiT, and VAE streaming paths; they
are not performance or maximum-capacity claims.

## Streamed Sol Attention

Dense attention remains the default. Set
`minimax_h3.attention_mode = "sol_streaming"` in the shared SeqAttn TOML to
select the approximate streamed Sol algorithm implemented by `seqattn-core`.
The node does not expose this deployment choice as a workflow input.

The adapter keeps MiniMax-H3's complete packed sequence as one self-attention
segment. Text, first/last-frame conditioning, and Ref2VA image/video/audio
references form an exact prefix ending before the target audio segment; target
audio and video rows are eligible for Sol routing. The current denoising
position comes from ComfyUI's complete sampler sigma schedule rather than from
forward-call counting, so CFG and intermediate multistep evaluations do not
shift the configured dense warmup or leading layers. Missing or inconsistent
schedule metadata is an error when Sol is selected.

Sol V1 requires CUDA SM80 or newer, Triton, BF16, non-causal self-attention,
head dimension 128, and equal query/KV head counts. These constraints apply to
the MiniMax-H3 DiT only. Qwen remains dense and materialized.

The supported ComfyUI baseline is fixed to **version `0.30.0`, commit
`9a9fdb10ed144ce760d9682cb247526ea23cc525`**. Newer ComfyUI releases are not
part of the `0.4.x` compatibility contract, even if individual code paths may
continue to work.

## Qwen Conditioning

Add **MiniMax H3 Qwen SeqAttn** after the MiniMax `CLIPLoader` and before the
MiniMax conditioning node. Connecting it explicitly selects the streaming Qwen
implementation; wiring the loader directly to the conditioning node keeps the
native ComfyUI implementation. The bundled low-memory workflows connect it.
The previous activation-estimate-based Qwen BF16 offload node has been removed.

The SeqAttn node keeps complete vision and decoder hidden states, Q/K/V, and
DeepStack features in pinned CPU memory. GPU execution is limited to configured
projection, attention, merger, output-projection, residual, and MLP tiles. It
uses packed non-causal vision attention and causal decoder GQA without building
a quadratic causal mask. DeepStack is injected directly into CPU hidden after
each of the first three decoder layers. Decoder token embedding stays on the CPU:
only unique token rows are gathered and dequantized from the mmap-backed INT8
table before they are written into the final pinned hidden tensor.

Qwen uses the same current-plus-next Dynamic VBAR weight pipeline as the DiT
path. Before a stage is prefetched to the GPU for the first time, its checkpoint
data is synchronously materialized into the ComfyUI loaded-weight host pin. This
keeps cold-start execution identical to later cached encodes instead of consuming
an incomplete first VBAR transfer. It does not impose an artificial presentation
length limit; pinned host allocations, attention work, and runtime still scale
with the actual text, image, and video input. Its node exposes the
deployment-sensitive resident Q chunk and transferred K/V chunk independently
from the DiT node:

| Setting | Default | Description |
|---|---:|---|
| `q_chunk_tokens` | `5760` | Resident Q rows for Qwen vision and decoder attention |
| `kv_chunk_tokens` | `4096` | K/V rows transferred per attention tile |

QKV projection and MLP tile sizes come from the shared SeqAttn TOML:

```toml
[minimax_h3_qwen]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096
```

The shipped Qwen Q/KV and tile values currently reuse the measured RTX 5090
DiT configuration. They are starting values, not a claim that Qwen has been
independently tuned to its performance optimum.

## Install

### ComfyUI Manager

Search for **MiniMax H3 SeqAttn**, install it, and restart ComfyUI.

### Manual

```bash
cd /path/to/ComfyUI
git fetch origin 9a9fdb10ed144ce760d9682cb247526ea23cc525
git checkout --detach 9a9fdb10ed144ce760d9682cb247526ea23cc525

cd /path/to/ComfyUI/custom_nodes
git clone --branch community/comfyui-minimax-h3-seqattn \
  https://github.com/renlililoli/minimax-h3-seq-chunk-attn.git \
  ComfyUI-MiniMaxH3-SeqAttn
cd ComfyUI-MiniMaxH3-SeqAttn
python -m pip install -e .
```

No Git submodules are required. Installation resolves the pinned
`seqattn-core[dit,sparse]` `0.4.0a1` runtime directly from its immutable
upstream commit.

## Requirements

- ComfyUI `0.30.0`, commit
  `9a9fdb10ed144ce760d9682cb247526ea23cc525`
- Linux and NVIDIA CUDA
- Python `>= 3.10`
- Batch size 1
- Sufficient CPU DRAM for full hidden and Q/K/V storage

For the current release, 8 GiB of VRAM provides ample headroom for typical
bundled workflows, while 12 GiB is recommended for the best overall
experience. System RAM of 64 GiB is usually sufficient. Very long or
high-resolution Ref2VA inputs can require more host memory; the 243-frame long
Ref2VA workflow template is intentionally a stress case and may exceed a single
64 GiB NUMA node during video decode.

The attention and Qwen activation paths use BF16 regardless of checkpoint
storage precision. The Qwen node requires `CLIPLoader` device `default`.
Ordinary Linear LoRA is supported through the dedicated **MiniMax H3 SeqAttn
LoRA** node described below. Diffusion-model replacement patches, NVMe
activation backing, and multi-GPU execution are not currently supported.

The extension rejects other reported ComfyUI versions during entrypoint
loading with an explicit compatibility error. This avoids silently treating a
changed internal MiniMax-H3 sampling contract as a supported environment.

For an RTX 50-series container, use the pinned
[`docker/Dockerfile`](docker/Dockerfile) and
[`docker/README.md`](docker/README.md). The image starts from a publicly
pullable, digest-pinned NVIDIA NGC PyTorch base, installs the validated stable
CUDA 12.8 Torch wheels, checks out the pinned ComfyUI commit, verifies the
DynamicVRAM runtime during the build, and installs this node with its fixed
`seqattn-core` revision. Docker base-image and bundled Manager terms are listed
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### AIMDO Startup Order

Normal Web UI users do not need a separate AIMDO bootstrap. Start the pinned
ComfyUI checkout through its standard `main.py` entrypoint and leave
DynamicVRAM enabled. ComfyUI initializes `comfy_aimdo.control` before importing
PyTorch and the dynamic model patcher, so workflows queued after the UI starts
use the correct initialization order.

This guarantee does not apply to custom Python launchers that bypass
`main.py`. Such launchers must initialize `comfy_aimdo.control` before
importing `torch`, `nodes`, `comfy.model_patcher`, or any module that imports
`comfy_aimdo.host_buffer`. Otherwise `host_buffer` can cache an uninitialized
native-library handle and fail when `ModelPatcherDynamic` creates its first
host buffer.

## Models

The bundled workflows use these files from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

```text
ComfyUI/models/
|-- diffusion_models/
|   |-- minimax_h3_fl2va_pruned_int8_convrot.safetensors
|   `-- minimax_h3_ref2va_pruned_int8_convrot.safetensors
|-- loras/
|   |-- minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
|   |-- minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
|   `-- minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
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
| Ref2VA long 2-step validation | [`minimax_h3_seqattn_ref2va_long_2step.json`](workflows/minimax_h3_seqattn_ref2va_long_2step.json) | User-provided 243-frame reference video + audio |
| FL2VA Turbo 4-step LoRA | [`minimax_h3_seqattn_fl2va_turbo_4step_lora.json`](workflows/minimax_h3_seqattn_fl2va_turbo_4step_lora.json) | Prompt + optional first/last frames |
| FL2VA Sol Turbo 4-step LoRA | [`minimax_h3_seqattn_fl2va_sol_4step_lora.json`](workflows/minimax_h3_seqattn_fl2va_sol_4step_lora.json) | Prompt + optional first/last frames; Sol-configured service |
| FL2VA Turbo 8-step LoRA | [`minimax_h3_seqattn_fl2va_turbo_8step_lora.json`](workflows/minimax_h3_seqattn_fl2va_turbo_8step_lora.json) | Prompt + optional first/last frames |
| Ref2VA Turbo 4-step LoRA | [`minimax_h3_seqattn_ref2va_turbo_4step_lora.json`](workflows/minimax_h3_seqattn_ref2va_turbo_4step_lora.json) | Prompt + image/video/audio references |
| Ref2VA Sol Turbo 4-step LoRA | [`minimax_h3_seqattn_ref2va_sol_4step_lora.json`](workflows/minimax_h3_seqattn_ref2va_sol_4step_lora.json) | Prompt + image/video/audio references; Sol-configured service |

### Validation Evidence

#### 1080p-Class FL2VA Sol Warm Start

On September 4, 2026, an RTX 5090 service completed a 10.0-second FL2VA Sol
Turbo 4-step LoRA workflow at 1080p-class resolution (`1920x1088` aligned
output, 243 frames at 24 FPS). Before the measured run, the same service
completed one full 1080p-class four-step denoising run as warmup.

The measured sampler reported these cumulative and per-step denoising times:

| Step | Configured routing | Cumulative | Step elapsed |
|---:|---|---:|---:|
| 1 | Dense | 3:32 | about 212 s |
| 2 | Sol streaming | 4:13 | about 41 s |
| 3 | Sol streaming | 6:13 | about 120 s |
| 4 | Sol streaming | 8:15 | about 122 s |

The complete denoising stage took 8:15 and the full workflow took 11:50. The
step values are differences between the sampler's whole-second cumulative
timestamps, so each value is approximate to about one second. The run used
materialized execution, MiniMax-H3 Q/KV chunks `15360`/`4096`, Qwen Q/KV
chunks `5760`/`4096`, the `simple` scheduler, and `res_multistep` sampler. Sol
was configured with a 20% leading dense-step fraction and two leading dense
layers, so step 1 was dense while steps 2-4 used Sol routing outside those
leading layers.

This is observed functional evidence, not a controlled throughput benchmark.
Prompt text, input media, and generated content are omitted. The sanitized
record is in
[`docs/results/2026-09-04-fl2va-1080p-sol-warm-start.md`](docs/results/2026-09-04-fl2va-1080p-sol-warm-start.md).

#### 10-Second FL2VA Sol vs. Dense

On September 3, 2026, the two later completed 10.0-second runs from each RTX
5090 service were selected so every compared sample had all four cumulative
step boundaries. The final `4/4` elapsed values measure denoising rather than
the full `Prompt executed` workflow time.

| Attention mode | GPU | Denoising samples | Median | Range |
|---|---:|---:|---:|---:|
| Sol streaming | 1 | 189 s, 189 s | 189 s (3:09) | 189 s |
| Dense | 2 | 299 s, 297 s | 298 s (4:58) | 297-299 s |

The per-step values below are differences between consecutive whole-second
cumulative sampler timestamps:

| Step | Sol samples | Sol median | Dense samples | Dense median | Dense/Sol |
|---:|---:|---:|---:|---:|---:|
| 1 | 73 s, 73 s | 73 s | 74 s, 74 s | 74 s | 1.01x |
| 2 | 21 s, 21 s | 21 s | 75 s, 74 s | 74.5 s | 3.55x |
| 3 | 47 s, 48 s | 47.5 s | 75 s, 75 s | 75 s | 1.58x |
| 4 | 48 s, 47 s | 47.5 s | 75 s, 74 s | 74.5 s | 1.57x |

The overall Sol median was 109 seconds lower, a 36.6% reduction in denoising
elapsed time; equivalently, the Dense/Sol median ratio was 1.58x. All four
runs used 1.0 MP at 9:16, the `simple` scheduler, four steps, materialized
execution, MiniMax-H3 Q/KV chunks `15360`/`4096`, and Qwen Q/KV chunks
`5760`/`4096`. The Dense samples therefore also used a 15,360-token Q chunk
and are not a default-Q comparison. Runtime mode was classified from each
service's mounted TOML, not from the submitted workflow filename.

The services used separate GPUs but shared the host CPU and pinned-memory
path, and some runs overlapped. Treat these measurements as observed
functional evidence, not a controlled throughput benchmark. No completed
10-second Ref2VA sample was collected. Prompt text and video content remain
omitted; the sanitized run record is in
[`docs/results/2026-09-03-fl2va-10s-sol-vs-dense.md`](docs/results/2026-09-03-fl2va-10s-sol-vs-dense.md).

#### Earlier 0.4.3 Validation

These ComfyUI `Prompt executed` wall-clock times were recorded on August 28,
2026, with an RTX 5090, ComfyUI `0.30.0` at commit `9a9fdb10`, CUDA 12.8,
PyTorch `2.10.0+cu128`, and node version `0.4.3`. The runs used the runtime code
released as `seqattn-core` `0.3.0a4`; the final upstream pin differs only by a
release-documentation correction. Prompt text, reference media, and generated
content are intentionally omitted.

| Validation run | Elapsed wall time |
|---|---:|
| FL2VA Turbo 4-step, cold | 358.31 s |
| FL2VA Turbo 4-step, repeated | 58.31 s, 52.70 s, 53.36 s |
| Ref2VA Turbo 4-step, cold | 251.76 s |
| Ref2VA Turbo 4-step, repeated | 54.54 s |
| Long Ref2VA validation, 10 sampler steps | 1081.37 s (18:01.37) |
| Long Ref2VA rerun, 8 sampler steps | 891.83 s (14:51.83) |

These are functional workflow timings, not throughput benchmarks or
cross-system performance claims.

The four T2VA/FL2VA workflows use the same FL2VA checkpoint. The first frame
anchors frame 0; the last frame anchors the final aligned output frame. To
patch an existing workflow, add **MiniMax H3 SeqAttn** immediately after the
diffusion-model loader and **MiniMax H3 Qwen SeqAttn** immediately after the
MiniMax `CLIPLoader`. For bounded keyframe encoding and video decoding, pass
the video VAE through **MiniMax H3 VAE Streaming**. Each patch is independent:
connect it to select streaming for that stage, or wire around it to keep the
native ComfyUI implementation. There is no node-level `enabled` switch and no
silent native fallback after a streaming error.

### LoRA

Use **MiniMax H3 SeqAttn LoRA** instead of ComfyUI's standard LoRA loader for
the diffusion model. The recommended order is:

```text
UNETLoader -> MiniMax H3 SeqAttn LoRA -> MiniMax H3 SeqAttn
```

Multiple adapters can be chained by adding more SeqAttn LoRA nodes. Applying
the LoRA node after **MiniMax H3 SeqAttn** is also supported, but the dedicated
workflows keep all adapter selection directly after the model loader. Each
node has an independent signed `strength_model`; a zero-strength adapter stays
in the model signature and cache identity but is omitted from GPU staging.

The v1 adapter path requires the supported 50-block INT8 tensorwise ConvRot
MiniMax-H3 base. It accepts ordinary Linear A/B LoRA tensors in ComfyUI or PEFT
naming, including per-layer alpha. It rejects DoRA, convolutional or reshaped
adapters, diff/set patches, unsupported targets, shape mismatches, unconsumed
tensors, and models already carrying ordinary ComfyUI weight patches. The base
quantized tensors are never merged or requantized.

LoRA tensors remain CPU-resident and stream beside the current and next DiT
weight stages through two reusable pinned-host slots and two reusable GPU
slots. For the three Turbo adapters listed above, the largest stage is about
71.75 MiB, so the two-slot maximum is about 143.5 MiB of pinned host memory and
143.5 MiB of GPU memory in addition to the base streaming path.

The bundled Ref2VA workflow uses **MiniMax H3 Reference to Video (SeqAttn)**.
It preserves the native reference ordering and payload, but completes Qwen
layout validation and text/visual encoding before any reference image, video,
or audio VAE encode. It does not select a backend itself; the supplied `CLIP`
and video `VAE` objects determine which implementations run.

The workflow files for all modes use the current fused DiT and Qwen SeqAttn
integrations. A workflow file by itself is not release evidence; performance,
whole-process memory, and maximum-capacity claims require a preserved result
with matching package and environment metadata.

The long Ref2VA workflow is a UI-driven memory and integration stress test,
not a recommended quality preset. It generates 243 frames at 1344x768 from the
selected 243-frame reference and intentionally uses only two denoise steps.
The repository does not include reference media; place a compatible video in
the ComfyUI `input/` directory and select it in `LoadVideo`. The workflow file
by itself is not release evidence; only a preserved result with matching
package and environment metadata should be cited as validation. This case can
exceed one 64 GiB NUMA node during video decode even though GPU memory stays
low; on a multi-node host, allow enough memory nodes for the complete Qwen,
DiT, and VAE pipeline instead of binding the container to a single 64 GiB
memory node.

Calibrate `q_chunk_tokens` for each deployed stage, GPU, backend, CPU affinity, and
NUMA memory policy using the independent
[SeqAttn chunk-size calibration guide](https://github.com/renlililoli/stream-attn/blob/main/docs/q_chunk_calibration.md).
The shipped DiT `5760` value matches the validated RTX 5090 single-node path at
about 37 GB/s concurrent pinned H2D bandwidth. The same GPU used `3840` after
interleaving pinned pages across two populated memory nodes reproduced about
56.7 GB/s. These are DiT calibration results, not Qwen SeqAttn measurements.
Do not select Q from nominal PCIe bandwidth or advertised GPU peak TFLOPS; the
guide measures the effective concurrent bandwidth and resident attention
throughput used by the roofline.

The DiT and Qwen patch nodes each expose their own Q/KV chunks as workflow
inputs. This makes backend selection and the main performance parameter
explicit per stage. QKV projection and MLP tiles, plus video VAE settings, are
deployment configuration. Set them in the shared SeqAttn TOML file selected by
`SEQATTN_CONFIG`, or
`~/.config/seqattn/config.toml`. An explicitly selected missing or invalid file
is an error; if the default user file is absent, the compiled defaults below
are used.

```toml
[attention]
backend = "auto"

[minimax_h3]
execution_mode = "materialized" # or "recompute"
attention_mode = "dense" # or "sol_streaming"
projection_tile_tokens = 4096
ffn_tile_tokens = 4096
sol_tau = 1.0
sol_first_dense_step_fraction = 0.2
sol_first_dense_layers = 2

[minimax_h3_qwen]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096

[minimax_h3_vae]
tile_size = 192
workspace_mib = 512
```

The `[minimax_h3]` table is owned and validated by `seqattn-core` when the DiT
patch node constructs its runtime. `materialized` remains the default;
`recompute` is limited to the validated INT8 tensorwise ConvRot MiniMax-H3
path. `attention_mode = "dense"` is exact; `"sol_streaming"` applies the
configured leading dense-step fraction and leading dense-layer count before
using approximate routing. Qwen remains materialized-only and does not accept
these MiniMax-H3 settings in `[minimax_h3_qwen]`.
The patch nodes do not impose a whole-process VRAM limit, silently shrink the
resident query chunk, or fall back to native execution.

## License

The custom node is GPL-3.0. Its external `seqattn-core` dependency is
Apache-2.0. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The independent SeqAttn runtime is installed through the pinned
`seqattn-core[dit,sparse]` dependency; this community branch contains only the
ComfyUI integration and workflows.
