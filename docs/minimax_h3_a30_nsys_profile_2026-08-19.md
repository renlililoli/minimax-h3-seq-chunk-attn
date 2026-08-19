# MiniMax-H3 SeqAttn A30 denoise profile

Date: 2026-08-19

## Scope

This profile isolates one warmed-up denoise step from a two-step run. The first
step warms the model and CUDA paths; CUDA Profiler API capture starts only for
step 2.

| Item | Value |
| --- | --- |
| GPU | NVIDIA A30, 24 GiB, PCIe Gen3 x16 |
| Driver / CUDA / PyTorch | 580.173.02 / 12.6 / 2.7.0+cu126 |
| Model | MiniMax-H3-NF4 |
| Shape | 480x832, 124 frames, 15,104 combined tokens |
| Denoise steps | 2, with step 2 captured |
| Memory policy | strict 6,144 MiB target |
| Attention | SeqAttn Triton backend |
| Projection / Q / KV chunks | 2,048 / 11,136 resolved / 4,096 tokens |
| SeqAttn workspace | 1,024 MiB requested, 1,071,652,864 bytes resolved |

The captured NVTX denoise range is **29.282 s**. Unprofiled steady-state runs
for the same point are approximately **28.74-29.10 s/step**. The benchmark JSON
records 40.742 s for captured step 2 because `cudaProfilerStop()` and report
drain occur before the outer timer finishes. That value is profiler overhead,
not model latency.

At this operating point, a 50-step DiT pass is approximately 24 minutes before
decode, consistent with the separately measured full 50-step run.

## Denoise wall-time breakdown

The following values are host NVTX range durations. Nested asynchronous GPU
operations can finish after their launching NVTX range, so these rows describe
the pipeline's wall-time ownership rather than mutually exclusive GPU time.

| Stage, 50 blocks | Wall time | Denoise share | Mean/block |
| --- | ---: | ---: | ---: |
| Attention | 10.548 s | 36.0% | 210.96 ms |
| MLP D1 | 6.170 s | 21.1% | 123.40 ms |
| QKV projection | 4.870 s | 16.6% | 97.39 ms |
| AdaLN | 2.498 s | 8.5% | 49.96 ms |
| MLP D2 | 0.832 s | 2.8% | 16.63 ms |
| Final layer | 0.052 s | 0.2% | 51.55 ms total |
| Remaining block/input/control work | 4.313 s | 14.7% | - |

Per-block wall time is stable after block 0. Block 0 is 494.6 ms; most blocks
are 573-600 ms; block 49 is the largest at 608.2 ms. There is no sustained
growth with block index, so allocator growth or an accumulating cache is not
the primary explanation.

Nsight's GPU-projected attribution is 10.438 s for attention, 5.828 s for MLP
D1, 4.518 s for MLP D2, 4.492 s for QKV projection, and 1.714 s for AdaLN.
These projected values overlap and must not be added to obtain wall time.

## Device timeline overlap

Clipping all kernel and copy intervals to the 29.282 s denoise range gives:

| Concurrent activity | Union time | Denoise share |
| --- | ---: | ---: |
| Kernel only | 9.014 s | 30.8% |
| H2D only | 9.208 s | 31.4% |
| D2H only | 2.727 s | 9.3% |
| Kernel + H2D | 2.660 s | 9.1% |
| Kernel + D2H | 2.320 s | 7.9% |
| H2D + D2H | 0.287 s | 1.0% |
| No GPU kernel/copy activity | 3.066 s | 10.5% |

At least one GPU engine is active for 26.216 s, or 89.5% of the denoise range.
Kernel union time is 13.994 s, H2D union time is 12.155 s, and D2H union time
is 5.335 s. These overlap, so adding them would overstate elapsed time.

There are 19,430 individual idle gaps, mostly launch-scale gaps: median 1.12 us.
However, 419 gaps exceed 1 ms, 65 exceed 10 ms, and the largest is 211.5 ms.
The long tail is large enough to matter even though most gaps are harmless.

## CUDA kernels

Total kernel engine time is approximately 14.03 s.

| Kernel group | GPU time | Calls | Kernel-time share |
| --- | ---: | ---: | ---: |
| SeqAttn `_streaming_attention_update_kernel` | 6.849 s | 450 | 48.8% |
| Ampere BF16 GEMM | 3.948 s | 1,300 | 28.1% |
| NF4 BF16 dequantization | 0.564 s | 1,350 | 4.0% |
| `indexSelect` | 0.419 s | 2,170 | 3.0% |
| SeqAttn `_finalize_attention_kernel` | 0.040 s | 150 | 0.3% |

The attention update kernel averages 15.22 ms per launch. Its 450 launches
correspond to nine updates per DiT block at the current sequence and chunk
sizes. It is the largest single compute target, but reducing it alone cannot
remove the H2D-only and synchronization portions of the timeline.

## PCIe traffic

| Direction | Bytes | Engine time | Effective throughput | Operations |
| --- | ---: | ---: | ---: | ---: |
| H2D | 123.463 GB | 12.155 s | 10.16 GB/s | 5,951 |
| D2H | 69.639 GB | 5.335 s | 13.05 GB/s | 2,161 |

The GPU is attached through PCIe Gen3 x16. D2H is relatively close to the
practical limit, while H2D leaves more headroom and is split across several
streams. Logical streaming counters account for about 106.993 GB H2D and
69.478 GB D2H per step. The additional traced H2D traffic is primarily weights
and transfers outside the logical SeqAttn activation counters.

Within the host NVTX ranges, attention is almost continuously device-active
(10.393 of 10.548 s). QKV projection has 0.529 s with no kernel/copy activity.
AdaLN has 1.100 s and MLP D1 has 0.748 s without device activity inside their
host ranges. MLP D1's visible kernel, H2D, and D2H intervals are effectively
serial in this trace, making it a strong pipeline-overlap target. Because GPU
work launched in one range can finish in a later range, use these numbers to
locate timeline gaps, not as exclusive operation attribution.

## CUDA API waits and launch pressure

| CUDA API | Aggregate host duration | Calls |
| --- | ---: | ---: |
| `cudaStreamSynchronize` | 16.834 s | 3,243 |
| `cudaDeviceSynchronize` | 4.062 s | 153 |
| `cudaMemcpyAsync` | 3.646 s | 8,123 |
| `cudaEventSynchronize` | 0.168 s | 400 |
| `cudaLaunchKernel` | 0.261 s | 33,430 |

The aggregate API durations are not additive wall time and may overlap across
host threads. Synchronization time also includes legitimate waiting for GPU
work. Its value is that it identifies serialization boundaries: attention
contains 344 stream synchronizations totaling 9.990 s, and MLP D1 contains 700
totaling 4.263 s. There are also 103 device synchronizations totaling 4.061 s
outside the five inner stage ranges. The trace should therefore be read as a
copy/compute pipeline with frequent blocking boundaries, not as a purely
compute-bound attention workload.

## Optimization order

1. Reduce or better pipeline H2D traffic. H2D-only time is 9.208 s and total
   traced H2D volume is 123.5 GB per step. Prefetch the next weight/activation
   chunk earlier, reuse resident tensors where the 6 GiB cap permits it, and
   verify that weight leases do not force a device-wide boundary.
2. Remove coarse synchronization. Audit the 103 `cudaDeviceSynchronize` calls
   outside the inner phase ranges and replace correctness-independent waits
   with event dependencies. Then reduce per-chunk `cudaStreamSynchronize`
   boundaries so D1/D2 and adjacent blocks can overlap copies with compute.
3. Tune the SeqAttn update kernel and launch geometry. It consumes 6.849 s, or
   48.8% of kernel time. Sweep KV chunk and resolved Q chunk under the same
   6 GiB cap, tracking update-launch count, update time, and end-to-end step
   time together.
4. Reduce launch fragmentation in projections and elementwise work. One step
   launches 33,430 runtime kernels, while NF4 dequantization alone launches
   1,350 kernels. Fusion or larger projection chunks may help, but only if they
   do not increase memory beyond the strict target or damage copy overlap.
5. Re-profile one warmed step after each change and validate final candidates
   with an unprofiled 5-step run. Nsight timings are diagnostic; unprofiled
   steady-state latency remains the acceptance metric.

## Artifacts

- Nsight report: `workspace/benchmarks/artifacts/a30_h3_seqattn_15k_step2_api_20260819.nsys-rep`
- SQLite export: `workspace/benchmarks/artifacts/a30_h3_seqattn_15k_step2_api_20260819.sqlite`
- Benchmark JSON: `workspace/benchmarks/results/a30_nsys_seqattn_15k_s2_api_480x832_f124_s2_20260819T041255Z.json`

CPU sampling was unavailable because the host has `perf_event_paranoid=4`.
A30 hardware counter sampling was also unavailable due to GPU performance
counter permissions. CUDA API, kernel, memcpy, NVTX, and stream-overlap data
remain valid for this analysis.

## Fused MLP follow-up

The highest-priority MLP optimization was implemented after this profile. The
new default `--streaming-mlp-mode fused` acquires FC1 and FC2 computation leases
together and executes `FC1 -> SiLU/gate -> FC2 -> residual/gate` for each tile.
Only the final hidden tile returns to CPU. The previous implementation remains
available as `--streaming-mlp-mode split` for comparison and fallback.

At the same 480x832, 124-frame, 15,104-token, strict-6-GiB point, independent
unprofiled two-step runs produced:

| MLP mode | Step 1 | Step 2 | NVML peak | Torch reserved peak |
| --- | ---: | ---: | ---: | ---: |
| Fused | 26.615 s | 24.753 s | 5,032 MiB | 4,728 MiB |
| Split | 30.929 s | 29.015 s | 4,988 MiB | 4,684 MiB |

The warmed step improved by **4.262 s, or 14.7%**, while remaining under the
6,144 MiB target. The fused path increases the measured peaks by only 44 MiB.

Logical traffic per step falls by **50.773 GB**:

- 21.378 GB MLP-intermediate D2H removed.
- 21.378 GB MLP-intermediate H2D removed.
- 8.017 GB duplicate residual H2D removed by retaining the input tile for the
  final residual/gate operation.

A separate full 50-block, one-step correctness run saved both final latent
tensors. Fused and split video latents were bitwise identical, as were fused
and split audio latents; both maximum absolute differences were zero.

Follow-up artifacts:

- Fused performance JSON: `workspace/benchmarks/results/a30_mlp_fused_15k_s2_480x832_f124_s2_20260819T044700Z.json`
- Split performance JSON: `workspace/benchmarks/results/a30_mlp_split_15k_s2_480x832_f124_s2_20260819T044822Z.json`
- Fused correctness JSON: `workspace/benchmarks/results/a30_mlp_fused_correctness_480x832_f124_s1_20260819T045003Z.json`
- Split correctness JSON: `workspace/benchmarks/results/a30_mlp_split_correctness_480x832_f124_s1_20260819T045105Z.json`

## RTX 5090 launch-profile idea on A30

The RTX 5090 study found that Blackwell `sm_120`, BF16/FP16, D=128 benefits
from changing the update-kernel profile from `64x64 / 4 warps / 2 stages` to
`128x64 / 8 warps / 3 stages`. That parameter tuple does **not** transfer to
Ampere A30.

Unprofiled standalone A/B results were:

| A30 workload | Portable `64x64/4/2` | 5090 `128x64/8/3` | Result |
| --- | ---: | ---: | ---: |
| H3 15,104 tokens, 42 heads, 1 GiB workspace | 136.13 ms | 195.23 ms | 43.4% slower |
| 61,312 tokens, 56 heads, 2 GiB workspace | 2.433 s | 3.650 s | 50.0% slower |

The architecture-specific profiling method is useful, however. A complete A30
sweep for the H3 shape found:

| A30 profile | Standalone mean | Relative to portable | Sample parity |
| --- | ---: | ---: | --- |
| `64x64 / 4 / 2` | 136.11 ms | baseline | bitwise |
| `64x64 / 4 / 1` | 127.91 ms | 6.0% faster | bitwise |
| `128x32 / 4 / 1` | 122.72 ms | 9.8% faster | max abs 6.1e-5 |

The standalone winner was not the end-to-end winner. With fused MLP enabled,
the warmed H3 denoise step was:

| Profile | Step 2 | Relative to portable fused |
| --- | ---: | ---: |
| Portable `64x64/4/2` | 24.753 s | baseline |
| Safe A30 `64x64/4/1` | **23.807 s** | **3.8% faster** |
| Aggressive `128x32/4/1` | 24.127 s | 2.5% faster |

The safe A30 profile remains below the 6 GiB target and its full one-step video
and audio latents are bitwise identical to the portable profile. Combined with
fused MLP, warmed step latency is 17.9% lower than the legacy split-MLP plus
portable-kernel path (23.807 s versus 29.015 s).

The H3 integration now exposes optional launch overrides without changing the
default planner behavior: `--seqattn-block-m`, `--seqattn-block-n`,
`--seqattn-num-warps`, and `--seqattn-num-stages`. An automatic Ampere preset
should only be added after testing more token counts, head layouts, causal
modes, dtypes, and chunk sizes.

Launch-profile artifacts:

- H3 direct 5090-profile A/B: `workspace/benchmarks/results/a30_h3_15k_kernel_profile_ab_20260819.json`
- 61K direct 5090-profile A/B: `workspace/benchmarks/results/a30_61k_kernel_profile_ab_20260819.json`
- H3 full sweep: `workspace/benchmarks/results/a30_h3_15k_kernel_full_sweep_20260819.json`
- H3 top-profile retest: `workspace/benchmarks/results/a30_h3_15k_kernel_top_retest_20260819.json`
- Safe-profile end-to-end: `workspace/benchmarks/results/a30_mlp_fused_ampere_s1_15k_s2_480x832_f124_s2_20260819T051245Z.json`
- Aggressive-profile end-to-end: `workspace/benchmarks/results/a30_mlp_fused_ampere_m128n32s1_15k_s2_480x832_f124_s2_20260819T051414Z.json`

## RTX 5090 262K full-DiT follow-up

The architecture-specific conclusion was subsequently validated on the RTX
5090 at a much larger model scale. The README 720p workload was extended to
957 frames, producing 262,720 packed tokens. One BF16 Q/K/V set is 10.523GiB;
Q/K/V plus attention output is 14.031GiB.

Both runs used the complete 50-block MiniMax-H3 DiT, one denoise step, a 2GiB
SeqAttn workspace, a 4,096-token K/V chunk, and an 8,192MiB whole-process GPU
target on the same physical GPU3:

| RTX 5090 configuration | Denoise step | CPU RSS peak | PID NVML peak |
|---|---:|---:|---:|
| Legacy `64x64/4/2` + split MLP | 806.465 s | 66,048 MiB | **7,564 MiB** |
| Auto Blackwell `128x64/8/3` + fused MLP | **570.980 s** | **57,769 MiB** | 7,866 MiB |

The combined current path is **29.20% faster** and reduces peak RSS by
8,279MiB. Fused MLP removes 833.076GiB of logical PCIe traffic per denoise
step. Both configurations remain below the strict 8GiB GPU target.

This 5090 result does not change the A30 recommendation above: Blackwell's
`128x64/8/3` profile is not portable to Ampere. It confirms that keeping the
launch profile architecture-specific while sharing the fused MLP optimization
is the correct policy.

Artifacts:

- Legacy JSON: `workspace/benchmarks/results/rtx5090_streaming_full_dit_20260819/long262k_old_split_m64n64w4s2_gpu3_720x1280_f957_s1_20260819T053432Z.json`
- Current JSON: `workspace/benchmarks/results/rtx5090_streaming_full_dit_20260819/long262k_new_auto_fused_gpu3_720x1280_f957_s1_20260819T054902Z.json`
- Full report: `docs/minimax_h3_rtx5090_262k_streaming_optimization_2026-08-19.md`
