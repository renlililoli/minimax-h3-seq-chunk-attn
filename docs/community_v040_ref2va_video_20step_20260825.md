# Community 0.4.0 Ref2VA 20-Step Validation

Date: August 25, 2026 UTC

This document records the release-level capacity and stability validation for
the fused MiniMax-H3 DiT path in the ComfyUI community package. It is the only
performance and memory dataset used for the `0.4.0` claims in the root README.
Historical `0.3.x` split-path results are not combined with this run.

## Result

The run completed successfully with 20 denoise forwards and produced a valid
124-frame H.264/AAC MP4.

| Item | Value |
|---|---:|
| Resolution | 1344x768 |
| Reference/output frames | 124 / 124 |
| Packed tokens | 81,180 |
| Denoise steps | 20/20 |
| Whole-process NVML peak | 7,708 MiB |
| Headroom to 8,192 MiB target | 484 MiB |
| Denoise NVML peak | 4,386 MiB |
| Denoise steady NVML level | 4,274-4,276 MiB |
| CPU RSS peak | 32,666.40 MiB |
| Denoise time | 1,812.935 s |
| Complete process time | 2,073.534 s |
| Output | 1344x768, 124 frames, 24 fps, 5.167 s |

The first DiT forward took 252.666 seconds because it included compilation and
warmup. The remaining 19 forwards averaged 81.033 seconds, with a range of
80.925-81.144 seconds. The narrow steady range and the flat 4,276 MiB trace
show no step-over-step GPU-memory growth.

## Memory Profile

![Phase-aware GPU memory profile](assets/community_v040_ref2va_video_20step_20260825_memory.png)

The plotted source categories are derived at each 20 ms sample:

- **Torch allocated** is `torch.cuda.memory_allocated()`.
- **Torch reserved, unused** is reserved minus allocated memory.
- **Non-Torch / AIMDO / CUDA context** is NVML process memory minus Torch
  reserved memory.

The third category includes DynamicVRAM/AIMDO mappings, VBAR-resident weights,
the CUDA context, codec allocations, and any other non-Torch CUDA allocation.
The trace does not expose a reliable per-sample split inside that category, so
the graph does not claim more precise attribution. The weight-scheduler trace
separately records a 320 MiB peak for loaded VBAR weights.

| Phase | Seconds | NVML peak | Torch allocated | Reserved unused | Non-Torch/AIMDO/context |
|---|---:|---:|---:|---:|---:|
| Video VAE load | 0.117 | 500 MiB | 0 MiB | 2 MiB | 498 MiB |
| Qwen load | 0.789 | 500 MiB | 0 MiB | 2 MiB | 498 MiB |
| Reference decode | 0.706 | 500 MiB | 0 MiB | 2 MiB | 498 MiB |
| Conditioning | 160.709 | 7,708 MiB | 1,228 MiB | 1,142 MiB | 5,338 MiB |
| DiT load | 0.094 | 884 MiB | 9 MiB | 17 MiB | 858 MiB |
| Denoise | 1,812.935 | 4,386 MiB | 1,158 MiB | 1,650 MiB | 1,578 MiB |
| Video VAE reload | 0.139 | 1,492 MiB | 10 MiB | 14 MiB | 1,468 MiB |
| Video decode | 62.662 | 6,002 MiB | 479 MiB | 247 MiB | 5,276 MiB |
| Audio VAE load | 0.090 | 1,492 MiB | 11 MiB | 13 MiB | 1,468 MiB |
| Audio decode | 6.847 | 1,492 MiB | 11 MiB | 13 MiB | 1,468 MiB |
| Media write | 5.898 | 1,492 MiB | 11 MiB | 13 MiB | 1,468 MiB |
| Media probe | 0.116 | 1,492 MiB | 11 MiB | 13 MiB | 1,468 MiB |

The source columns are the composition at the sample where that phase reached
its NVML peak. They are not independent maxima and therefore add back to the
reported phase peak.

## DiT Configuration

```text
q_chunk_tokens   = 5760
kv_chunk_tokens  = 4096
qkv_tile_tokens  = 4096
mlp_tile_tokens  = 4096
```

The scheduler produced exactly 5,000 lifecycle records: 1,000 each for
`prepare`, `ready`, `compute_start`, `compute_end`, and `release`. It recorded
20 forwards, 50 blocks per forward, a maximum of two staged blocks, and only
0.194 seconds of accumulated ready-blocked time across all 1,000 blocks.

This run used physical GPU 1, CPU set `224-255,480-511`, and memory node 7. The
single-node host-memory path was paired with `q_chunk_tokens=5760`, matching the
previous approximately 37 GB/s roofline calibration. It is valid capacity and
long-run stability evidence. It is not the final high-bandwidth performance
comparison: that comparison must use interleaved nodes 5 and 7 with the
corresponding `q_chunk_tokens=3840` calibration.

## Qwen Preflight

The Qwen input limit remained enabled and unchanged:

```text
activation_limit_mib   = 5888
max_conditioning_rows  = 25000
preflight_safety_mib   = 128
offload_mode           = prefetch
```

The actual presentation contained 6,174 rows, including 6,048 visual rows.
Preflight estimated 2,358.12 MiB of activation storage and 2,486.12 MiB after
the safety reserve. The observed memory reduction came from correct
DynamicVRAM/headroom enforcement, not a change to Qwen encode computation.

## Reproduction Record

The standalone runner arguments were:

```text
--scenario ref2va_video
--width 1344
--height 768
--frames 124
--steps 20
--seed 0
--target-vram-mib 8192
--reserve-vram-gib 3.0
--aimdo-watermark-margin-mib 2560
--sample-interval-ms 20
--audio-device cpu
```

The raw result metadata reports base checkout `105600c` because the tested
`0.4.0` changes were present in the mounted working tree before their final
commit. The result, trace, scheduler record, generated media, and source diff
were kept together; the final integration commit contains that tested source.

## Artifacts

- [Result JSON](results/community_v040_ref2va_video_20step_20260825_result.json)
- [20 ms memory trace](results/community_v040_ref2va_video_20step_20260825_memory_trace.csv.gz)
- [Weight lifecycle trace](results/community_v040_ref2va_video_20step_20260825_weight_schedule.json.gz)
- [Generated output](../assets/benchmark/seqattn_ref2va_8g_20step_1344x768_124f.mp4)
- [Plot script](../tools/plot_memory_profile.py)

Regenerate the figure from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-seqattn \
python3 tools/plot_memory_profile.py \
  --result docs/results/community_v040_ref2va_video_20step_20260825_result.json \
  --trace docs/results/community_v040_ref2va_video_20step_20260825_memory_trace.csv.gz \
  --output docs/assets/community_v040_ref2va_video_20step_20260825_memory.png
```
