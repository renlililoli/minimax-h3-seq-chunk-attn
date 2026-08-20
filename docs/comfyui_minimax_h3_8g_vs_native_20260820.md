# ComfyUI MiniMax-H3: strict 8 GiB SeqAttn vs native

Experiment date: August 20, 2026 UTC

## Executive summary

This experiment evaluates whether MiniMax-H3 Ref2VA can process a real
10.125-second, 1344x768 video-reference workload within an 8 GiB whole-process
GPU target using the ComfyUI weights and execution path.

The target workload contains 157,196 packed tokens. It includes both a
72,576-token target video and a 72,576-token reference video. This is materially
larger than a 10-second text-to-video or image-reference workload because the
reference video contributes a second full spatiotemporal token sequence.

The main result is:

- SeqAttn ComfyUI completes one denoise step in 369.39 seconds with a 7,982 MiB
  process peak.
- The old native ComfyUI environment OOMs in the first QKV projection on an RTX
  5090 32 GB, after reaching a sampled 31,590 MiB process peak.
- Completed native short-sequence points imply an estimated 232-294 seconds for
  the target step if native full-sequence execution had sufficient memory.
- The current 8 GiB implementation therefore pays an estimated 1.25x-1.59x
  step-time cost to execute a workload that native cannot complete on 32 GB.

The native target time is an extrapolation. Native did not complete the target
step, and the 66.87 seconds before its OOM is not a performance result.

## Target workload

| Property | Value |
|---|---:|
| Model | MiniMax-H3 Ref2VA INT8 ConvRot DiT |
| Text encoder | Qwen3-VL 32B NVFP4 AWQ |
| Output size | 1344x768 |
| Output frames | 243 at 24 fps |
| Output duration | 10.125 seconds |
| Reference | Real 243-frame video |
| Reference audio | Disabled for the native comparison |
| Denoise steps | 1 |
| Seed | 0 |
| SeqAttn workspace | 1,024 MiB |
| SeqAttn K/V tile | 4,096 tokens |
| Compute dtype | BF16 |
| Weight backing | CPU DRAM |
| Physical GPU | NVIDIA GeForce RTX 5090, 31.36 GiB usable |

The exact packed layout measured by the ComfyUI benchmark is:

| Segment | Token range | Tokens |
|---|---:|---:|
| Text | 0-11,234 | 11,234 |
| Reference video | 11,234-83,810 | 72,576 |
| Audio latent | 83,810-84,620 | 810 |
| Target video | 84,620-157,196 | 72,576 |
| **Total** | | **157,196** |

The video latent shape is `[1, 24, 72, 48, 84]`; the audio latent shape is
`[1, 32, 2, 405]`.

## Implementations compared

### SeqAttn ComfyUI

The custom node patches the MiniMax-H3 DiT so complete sequence activations can
be backed by CPU DRAM. GPU memory holds a planned working set for embedding,
QKV projection, streamed attention, output projection, MLP, residual state, and
the final layer.

The measured point uses:

- strict 8,192 MiB whole-process target;
- a 1,024 MiB SeqAttn activation workspace;
- 4,096-token K/V tiles;
- GPU layer-by-layer text-encoder offload;
- complete text-encoder unload before loading the DiT;
- CPU-streamed video VAE input chunks.

### Old native ComfyUI

The native baseline is not the benchmark script's forced `LOW_VRAM` mode. It is
the original environment from `../minimax-h3`:

| Component | Version / mode |
|---|---|
| Container | `comfyui:cu128` |
| ComfyUI | 0.30.0, revision `9a9fdb10` |
| PyTorch | 2.10.0+cu128 |
| comfy-kitchen | 0.2.26 |
| comfy-aimdo | 0.4.11 |
| VRAM state | `NORMAL_VRAM` |
| Weight offload | DynamicVRAM, asynchronous, two streams |

This is the correct historical ComfyUI baseline because it preserves the model
loading and weight-paging behavior used by the previous successful MiniMax-H3
jobs.

## Primary comparison

| Exact 157,196-token point | SeqAttn, strict 8 GiB | Old native ComfyUI, 32 GB |
|---|---:|---:|
| Status | **Success** | **OOM** |
| Completed denoise steps | **1/1** | 0/1 |
| Process GPU peak | **7,982 MiB** | 31,590 MiB before failure |
| Torch allocated peak | 4,225.50 MiB | 27,994 MiB reported by OOM summary |
| Torch reserved peak | 5,766 MiB | 30,816 MiB reported by OOM summary |
| Denoise interval | **369.391 s** | Not completed |
| Time to failure | — | 66.87 s for the complete prompt |
| Failure site | — | First block `qkv_proj`, INT8 linear output assembly |
| Failed allocation | — | 6.30 GiB |

The native failure reported:

```text
Currently allocated: 26.83 GiB
Requested:            6.30 GiB
Device limit:        31.36 GiB
```

The process-level reduction from the native pre-OOM peak to the successful
SeqAttn peak is:

```text
1 - 7,982 / 31,590 = 74.7%
```

Native fails before producing a valid step time. Dividing 369.39 seconds by
66.87 seconds would compare a completed step against an incomplete failure
path and is invalid.

## Successful 8 GiB runs

Two same-shape SeqAttn runs serve different purposes:

| Run | Text encoder | Denoise | Process peak | Decode/output |
|---|---:|---:|---:|---|
| Canonical generated-media run | CPU | 419.091 s | 7,696 MiB | 243-frame MP4 completed |
| Optimized conditioning run | GPU layer offload | **369.391 s** | 7,982 MiB | Decode skipped |

The second run isolates the current practical text-to-denoise path and proves
that GPU text encoding and the subsequent DiT can share the same strict 8 GiB
process sequentially. The first run proves that the pipeline can also decode
and write the real video under the target.

The optimized result is stored at:

```text
workspace/benchmarks/results/comfyui_ref2va_8g_20260820/
comfyui_ref2va_streaming_8g_gpu_text_768x1344_f243_s1_20260820T070448Z.json
```

A compact, repository-tracked machine-readable summary is available at
[`comfyui_minimax_h3_8g_vs_native_20260820_results.json`](comfyui_minimax_h3_8g_vs_native_20260820_results.json).

The generated-media result and MP4 are:

```text
workspace/benchmarks/results/comfyui_ref2va_8g_20260820/
comfyui_ref2va_streaming_8g_ws1024_768x1344_f243_s1_20260820T060432Z.json
comfyui_ref2va_streaming_8g_ws1024_768x1344_f243_s1_20260820T060432Z.mp4
```

## Native short-sequence scaling experiment

Because the exact native target OOMs, shorter runs were used to estimate the
compute-only native target cost. Resolution, reference video, prompt, model,
seed, and one-step Euler sampler remained fixed. Only the output/reference
frame count changed. VAE decode was removed from the measured path.

A temporary output node reported the exact `PackedLayout.seq_len` without
changing the model or attention implementation.

### Completed points

| Frames | Packed tokens | Native step | Sampled GPU peak | Use in fit |
|---:|---:|---:|---:|---|
| 39 | 26,411 | 12.31 s | 28,298 MiB | Yes |
| 56 | 37,563 | 19.15 s | 31,722 MiB | Yes, warm run |
| 73 | 48,717 | 28.21 s | 29,576 MiB | Yes |
| 90 | 58,853 | 172.49 s | 31,494 MiB | No: memory cliff |

The first cold 56-frame run took 44.19 seconds and peaked at 26,920 MiB. It was
excluded in favor of the repeated warm 19.15-second point. The 90-frame point
was also excluded: its 6.1x time jump relative to 73 frames coincides with a
31.5 GiB peak and represents DynamicVRAM paging or allocation pressure rather
than smooth compute scaling.

The non-monotonic sampled memory peaks are expected for a stateful DynamicVRAM
server with changing cached model residency. These memory values are useful for
identifying the cliff, not for fitting a linear memory model.

## Native target-time estimate

Two estimates bound the likely compute-only native step time.

### Quadratic plus linear fit

The stable points were fitted with:

```text
T(N) = aN^2 + bN + c

a = 8.91785888e-9 seconds/token^2
b = 4.28317938e-5 seconds/token
c = 4.95819799 seconds
```

At `N = 157,196`:

```text
T(157,196) = 232.06 seconds
```

This model represents quadratic dense-attention work plus approximately linear
projection, MLP, transfer, and fixed costs.

### Pure quadratic scaling

Using the largest stable point and assuming the entire step scales as `N^2`:

```text
28.21 * (157,196 / 48,717)^2 = 293.71 seconds
```

The resulting estimate is therefore:

```text
Native compute-only target: 232-294 seconds/step
SeqAttn strict-8-GiB measured: 369.39 seconds/step
SeqAttn / native estimate: 1.25x-1.59x
Center estimate: approximately 1.4x
```

This estimate intentionally excludes the 90-frame memory-cliff point. If the
real native target inherited that paging behavior, it would be much slower,
but in practice it OOMs before such a step can complete.

## Denoise profile

The instrumented 157,196-token SeqAttn run completed the sampler interval in
419.04 seconds. The instrumented MiniMax transformer forward accounts for
307.66 seconds:

| Component | Time | Share of instrumented forward |
|---|---:|---:|
| Streamed attention output | 190.74 s | 62.0% |
| QKV projection | 50.58 s | 16.4% |
| MLP | 54.63 s | 17.8% |
| Packed embedding | 2.50 s | 0.8% |
| Final projection | 0.15 s | <0.1% |
| AdaLN | 0.05 s | <0.1% |

The 1,024 MiB workspace holds 18,432 query tokens. The 157,196-token sequence
therefore requires nine query passes. Each pass scans the complete K/V sequence
in 4,096-token tiles:

```text
39 K/V tiles/pass * 9 query passes = 351 K/V tiles/block
351 * 50 blocks = 17,550 K/V tiles/step
```

Measured logical traffic for the 50-block step is approximately 2.02 TiB H2D
and 0.38 TiB D2H. Repeated K/V scans are the largest current cost.

An exact-shape standalone projected benchmark uses the same planner:

| Path | Projected attention time |
|---|---:|
| Standalone SeqAttn projected pipeline | 3.642 s |
| ComfyUI steady projected attention | 4.603 s |

The ComfyUI path is 26.4% slower at this boundary. The core attention kernel
differs by only about 6%; most of the gap is in the projection callback and
integration path.

The detailed profile is in
[`comfyui_minimax_h3_seqattn_profile_20260820.md`](comfyui_minimax_h3_seqattn_profile_20260820.md).

## Text conditioning

CPU-only Qwen3-VL conditioning took 553.17 seconds, longer than one denoise
step. The final implementation gives the encoder a CUDA execution device while
forcing ComfyUI `NO_VRAM`, so weights execute layer-by-layer and do not remain
resident before the DiT starts.

| Conditioning path | Time | Process peak |
|---|---:|---:|
| CPU | 553.17 s | below target |
| GPU layer offload, first run | 43.92 s | 7,982 MiB |
| GPU layer offload, OS-cache warm | **24.17 s** | 7,982 MiB |

The cached GPU-offload path is 22.9x faster than CPU conditioning. After the
encoder unload, the same process completes the 369.39-second denoise without
exceeding 8,192 MiB.

## Investigated but rejected comparisons

### Forced LOW_VRAM benchmark as a native baseline

The benchmark script's `native` mode forces ComfyUI `LOW_VRAM` and applies an
allocator-aware synthetic cap. It is useful for proving that full-sequence QKV
cannot fit under 8 GiB, but it does not reproduce the historical ComfyUI
DynamicVRAM runtime.

The strict-8-GiB native point OOMed after 96.77 seconds while requesting
1.57 GiB. A separate 32,000 MiB run OOMed after 115.64 seconds while requesting
12.60 GiB. A 1280x704, 124-frame attempt also OOMed in this forced mode.

These runs are retained as implementation diagnostics, not used as the formal
old-ComfyUI comparison.

### Historical "10-second 720p" outputs

All MP4 files under the old `../minimax-h3/data/output` directory were decoded
to verify their actual metadata. The existing 243-frame, 10.125-second outputs
are 864x480. Existing outputs near 720p are 124 frames, or 5.167 seconds. The
source named `t1_seg5_10.mp4` is a segment spanning seconds 5 through 10; it is
5.086 seconds long, not a 10-second generated output.

This does not imply that MiniMax-H3 cannot generate 720p video. It establishes
that the saved artifacts do not provide an equivalent 243-frame, 720p,
video-reference native baseline for this experiment.

### DiffSynth native result

The repository also contains a DiffSynth 158,208-token Ref2VA result where an
unrestricted 32 GiB native run succeeds. That is a valid result for the
DiffSynth pipeline, but it is not the old ComfyUI runtime evaluated here. The
packing, reference-audio path, sampler integration, model loading, and native
execution implementation differ. Both results remain documented, but they
must not be merged into a single native baseline.

## Reproduction

Build the ComfyUI integration image:

```bash
docker compose -f docker-compose.comfyui.yml build
```

Run the strict-memory points in isolated containers:

```bash
workspace/benchmarks/run_comfyui_ref2va_8g.sh native
workspace/benchmarks/run_comfyui_ref2va_8g.sh streaming
```

Use GPU text-encoder offload:

```bash
COMFYUI_TEXT_ENCODER_MODE=gpu-offload \
  workspace/benchmarks/run_comfyui_ref2va_8g.sh streaming --skip-decode
```

Profile the denoise step:

```bash
workspace/benchmarks/run_comfyui_ref2va_profile.sh
```

The old native comparison requires the historical `comfyui:cu128` image and
the default server startup without `--lowvram`. Verify that startup logs contain:

```text
Set vram state to: NORMAL_VRAM
Using async weight offloading with 2 streams
DynamicVRAM support detected and enabled
```

Then submit the same Ref2VA workflow with 1344x768, 243 frames, one denoise
step, seed 0, the same reference video and prompt, and no reference audio.

## Measurement boundaries and limitations

- All measurements are single-run system-characterization points without error
  bars.
- Process GPU peaks use NVML sampling. The old native target and scaling runs
  were sampled at 200 ms; the strict-memory benchmark samples at 20 ms.
- The 369.39-second SeqAttn point uses `res_multistep`; native scaling uses
  one-step Euler. Both execute one model forward, but sampler wrappers are not
  byte-for-byte identical.
- The native 232-294-second target is an extrapolation, not a completed target
  run.
- The experiment evaluates capacity and runtime behavior, not visual quality.
- CPU DRAM, PCIe bandwidth, OS page cache, and NUMA placement affect streamed
  performance.
- The strict 8 GiB target has only 210 MiB process-level headroom in the final
  GPU-text run; allocator and NVML checks should remain enabled.

## Conclusion

The practical value of the ComfyUI integration is capacity rather than raw
speed. The old native runtime reaches the physical 32 GB boundary and fails in
the first full-sequence QKV projection. SeqAttn completes the same 157,196-token
step with a 7,982 MiB process peak.

The cost is measurable: approximately 1.25x-1.59x over an extrapolated
memory-unconstrained native step, plus CPU DRAM and PCIe traffic. The next
optimization target is reducing repeated K/V scans and projection callback
overhead while retaining the 8 GiB capacity guarantee.
