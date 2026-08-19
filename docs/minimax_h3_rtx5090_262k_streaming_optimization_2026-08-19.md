# MiniMax-H3 262K streaming optimization comparison

## Summary

The current MiniMax-H3 streaming path was compared with the previous portable
SeqAttn launch profile and split MLP implementation on one physical RTX 5090.
Both runs executed one complete 50-block DiT denoise forward with a 2GiB
SeqAttn workspace and an 8GiB whole-process GPU-memory target.

The input is the README 720p workload extended only along the time axis:

| Item | Value |
|---|---:|
| Requested video | 720x1280, 957 frames |
| Model-aligned video | 736x1280, 957 frames |
| Duration at 24 fps | 39.875 seconds |
| Packed sequence | **262,720 tokens** |
| DiT blocks / denoise steps | **50 / 1** |
| BF16 Q/K/V size | **10.523 GiB** |
| BF16 Q/K/V/output size | **14.031 GiB** |
| SeqAttn HBM workspace | **2,048 MiB** |
| Whole-process GPU target | **8,192 MiB** |

The full Q/K/V activation is therefore already 2.523GiB larger than 8GiB;
including attention output raises the isolated attention activation footprint
to 14.031GiB. This is a materially larger capacity point than the previous
132,288-token README example, whose full Q/K/V is about 5.299GiB.

## Compared configurations

| Configuration | SeqAttn update kernel | MLP path |
|---|---|---|
| Previous | explicit `64x64`, 4 warps, 2 stages | `split`: FC1 intermediate round-trips through CPU |
| Current | automatic Blackwell `128x64`, 8 warps, 3 stages | `fused`: FC1, gate/SiLU, FC2 and residual remain tile-local on GPU |

Everything else was held constant: checkpoint, prompt, seed, CPU weight
backing, physical GPU, CPU affinity, projection chunk, 4,096-token K/V chunk,
2GiB SeqAttn workspace, and the 8GiB process target.

## Results

| Metric | Previous | Current | Change |
|---|---:|---:|---:|
| Complete 50-block denoise step | 806.465 s | **570.980 s** | **29.20% faster** |
| Old/current latency ratio | 1.412x | **1.000x** | **1.412x speedup** |
| Complete benchmark pipeline | 818.109 s | **583.017 s** | **28.74% faster** |
| CPU RSS peak | 66,048 MiB | **57,769 MiB** | **8,279 MiB / 12.54% lower** |
| PID-level NVML peak | **7,564 MiB** | 7,866 MiB | +302 MiB; both below 8GiB |
| Torch allocated peak | 6,661 MiB | **6,596 MiB** | 65 MiB lower |
| Torch reserved peak | **6,874 MiB** | 7,176 MiB | +302 MiB |
| Resident Q chunk | 26,048 tokens | 25,984 tokens | both require 11 Q passes |

The current path saves 235.485 seconds in one full DiT forward. The GPU-memory
tradeoff is small and remains within the configured boundary: the measured
PID peak has 326MiB of headroom below 8,192MiB.

## Logical transfer reduction

| Logical traffic per denoise step | Previous | Current | Reduction |
|---|---:|---:|---:|
| H2D | 4,912.582 GiB | **4,430.275 GiB** | **482.307 GiB** |
| D2H | 1,139.999 GiB | **789.230 GiB** | **350.769 GiB** |
| Total | 6,052.581 GiB | **5,219.505 GiB** | **833.076 GiB** |

Attention H2D is identical at 4,033.030GiB in both runs. The transfer reduction
comes from the fused MLP path:

- 350.769GiB of full FC1 intermediate D2H is removed;
- 350.769GiB of full FC1 intermediate H2D is removed;
- 131.538GiB of duplicate residual H2D is removed.

These are logical operator counters, not measured PCIe wire throughput. They
include traffic across all 50 DiT blocks.

## Sequence-length trend

The same old-to-current configuration comparison was also run at shorter
lengths. The shorter points use warmed step 2; the 262K capacity point uses its
only full step.

| Packed tokens | Previous | Current | Improvement |
|---:|---:|---:|---:|
| 14,912 | 26.314 s | **20.431 s** | **22.36%** |
| 30,976 | 28.982 s | **22.813 s** | **21.29%** |
| 61,056 | 65.990 s | **50.230 s** | **23.88%** |
| 262,720 | 806.465 s | **570.980 s** | **29.20%** |

The larger gain at 262K is directionally consistent with both optimizations:
the Blackwell update-kernel specialization matters more as dense-attention
work grows, while fused MLP removes traffic proportional to sequence length.
The 262K row is still one observation, so it should not be treated as a
statistical estimate.

## Interpretation

1. The current streaming implementation is materially faster at a model scale
   where attention activations alone exceed 8GiB. The gain is not limited to a
   standalone attention microbenchmark.
2. Fused MLP is essential for host-memory efficiency at this length. It lowers
   peak RSS by about 8.1GiB while removing 833GiB of logical PCIe traffic.
3. The automatic Blackwell kernel does not enlarge the resident Q set in this
   experiment; both plans use 11 Q passes. Its contribution is improved update
   kernel execution rather than reduced K/V scan count.
4. The current path uses about 302MiB more PID-level GPU memory than the old
   path, but both satisfy the same strict 8GiB process target.
5. This comparison is between two CPU-DRAM-backed streaming implementations.
   It is not a native FlashAttention comparison and does not involve the paged
   or NVMe runtime.

## Measurement limits

- Each 262K configuration was measured once on the same exclusive physical
  GPU3 and the same NUMA-local CPU set.
- The run skips VAE decode and media writing; the measured step is the complete
  50-block DiT forward, not a complete generated-video pipeline.
- Final latents were not saved for this long performance run. Correctness is
  covered by the existing fused/split latent parity runs and SeqAttn tests, but
  this specific 262K pair is a performance/capacity result.

## Artifacts

- Previous JSON:
  `workspace/benchmarks/results/rtx5090_streaming_full_dit_20260819/long262k_old_split_m64n64w4s2_gpu3_720x1280_f957_s1_20260819T053432Z.json`
- Current JSON:
  `workspace/benchmarks/results/rtx5090_streaming_full_dit_20260819/long262k_new_auto_fused_gpu3_720x1280_f957_s1_20260819T054902Z.json`
- Benchmark runner: `workspace/benchmarks/minimax_h3_baseline.py`
