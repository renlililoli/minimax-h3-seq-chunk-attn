# ComfyUI MiniMax-H3 SeqAttn profile, 2026-08-20

## Workload

- MiniMax-H3 Ref2VA INT8 ConvRot DiT
- Qwen3-VL 32B NVFP4 AWQ conditioning
- 1344x768, 243 frames, 157,196 packed tokens
- one `res_multistep` denoise step
- strict 8,192 MiB process budget
- SeqAttn workspace 1,024 MiB, KV tile 4,096 tokens

The profiled denoise completed in 419.04 seconds. Torch peaked at 3,703.54 MiB
allocated and 5,226 MiB reserved; the complete process peaked at 7,708 MiB.

## Measured denoise costs

The instrumented MiniMax forward accounts for 307.66 seconds. The gap to the
419.04-second sampler interval includes model paging and sampler/model wrapper
work outside the instrumented transformer forward.

| Component | Total | Share of instrumented forward |
|---|---:|---:|
| streamed attention output | 190.74 s | 62.0% |
| QKV projection | 50.58 s | 16.4% |
| MLP | 54.63 s | 17.8% |
| packed embedding | 2.50 s | 0.8% |
| final projection | 0.15 s | <0.1% |
| AdaLN | 0.05 s | <0.1% |

Block 0 took 25.80 seconds because it includes CUDA/Triton warm-up. The other
blocks were tightly grouped around a 5.68-second median.

## Primary inefficiency

The 1,024 MiB workspace fits 18,432 query tokens, producing nine query chunks.
Each query chunk scans the complete CPU-resident K/V sequence in 4,096-token
tiles. Per block this is 351 K/V tiles; across 50 blocks it is 17,550 tiles.

Measured logical traffic across the 50 blocks:

- total H2D: 2.02 TiB
- total D2H: 0.38 TiB
- attention K/V H2D alone: 1.95 TiB
- QKV projection output D2H: 0.31 TiB

The repeated K/V scans dominate the streamed attention time. Weight prefetch
bookkeeping is not the bottleneck; measured `prefetch_seconds` totals less than
one millisecond because transfers are consumed asynchronously elsewhere.

## Optimization order

1. Change the loop/data layout so each K/V tile is uploaded once and reused by
   multiple resident query chunks, or retain compressed K/V tiles on GPU.
2. Increase effective query residency without violating the 8 GiB cap, reducing
   the current nine complete K/V scans.
3. Pipeline QKV projection output with attention consumption to avoid writing
   the full 6.30 GiB QKV result to host and reading it back.
4. Reduce MLP hidden-state CPU/GPU round trips after the attention path is fixed.
5. Add a warm-up/cached Triton path so block 0 does not pay an extra 20 seconds.

## Text conditioning

CPU Qwen3-VL encoding took 553.17 seconds, longer than one denoise step. A
`gpu-offload` mode now gives the text encoder a CUDA execution device while
forcing ComfyUI `NO_VRAM`, so weights move layer-by-layer and do not stay
resident. The first isolated run encoded in 43.92 seconds; a second run with the
checkpoint in the OS page cache encoded in 24.17 seconds, or 12.6x to 22.9x
faster than CPU. Both runs peaked at 7,982 MiB. The second run then completed
the real one-step SeqAttn denoise in 369.39 seconds in the same process, proving
that the Qwen-to-DiT unload boundary remains within the 8,192 MiB limit.

The 210 MiB process-level margin is small. This mode should keep the allocator
cap and NVML process sampling enabled; ordinary ComfyUI `LOW_VRAM` is not an
equivalent substitute because it may retain too many Qwen weights for the
visual DeepStack activation.
