# MiniMax-H3 sequence streaming

> Exact 132K-token MiniMax-H3 inference with a bounded GPU working set.

[![seqattn](https://img.shields.io/badge/operator-seqattn-22d3ee?style=for-the-badge)](https://github.com/renlililoli/stream-attn)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](#current-results)
[![Precision](https://img.shields.io/badge/weights-NF4-8b5cf6?style=for-the-badge)](#current-results)
[![Status](https://img.shields.io/badge/15K%20full%20generation-passed-22c55e?style=for-the-badge)](#completed-15k-comparison)

This repository integrates the standalone
[`seqattn`](https://github.com/renlililoli/stream-attn) operator with
MiniMax-H3 in DiffSynth-Studio.  Complete Q/K/V activations live in CPU DRAM;
GPU HBM holds a statically planned resident query working set and streamed K/V
tiles.  The implementation preserves exact dense attention semantics while
trading latency and PCIe traffic for a much lower GPU capacity requirement.

<p align="center">
  <img src="https://raw.githubusercontent.com/renlililoli/stream-attn/main/docs/assets/minimax-h3-live-overview.svg" alt="MiniMax-H3 132K-token live benchmark" width="100%">
</p>

## Current results

| Experiment | Native / prior path | `seqattn` | Improvement |
|---|---:|---:|---:|
| 15,104-token full generation | 20,970 MiB | **4,748 MiB under 6GiB** | **77.36% / 4.42× lower** |
| 15,104-token 50-step DiT | **227.83 s** | 764.31 s | 3.35× capacity tradeoff |
| 132,288-token 50-step attempt | **OOM after 14 steps** | **50/50 DiT steps completed** | native cannot finish denoise |
| 132,288-token GPU peak | 30,876 MiB | **about 7,166 MiB** | **about 4.31× lower** |
| 132,288-token completed capacity probe | — | **5,968 MiB** | succeeds below 8GiB |
| 61,312-token projected-pipeline peak | 7,108 MiB | **3,848 MiB** | **45.9% lower** |
| 61,312-token projected-pipeline latency | 919.79 ms | **843.44 ms** | **8.3% faster** |
| 61,056-token full H3 PCIe traffic | 917.6 GiB | **836.1 GiB** | **81.5 GiB less/step** |

The completed 132K capacity probe executes all 50 DiT blocks for one denoise
step under an 8,192MiB whole-process target.  It takes 236.39 seconds and peaks
at 5,968MiB PID-level NVML memory.  The input requested as 720×1280 and 480
frames is aligned by H3 to 736×1280 and 481 frames: 20.04 seconds at 24fps and
132,288 packed tokens.

The completed 6GiB run provides the end-to-end success point: 480×832, 124
frames, 15,104 packed tokens, 50 denoise steps, both VAE decoders, and MP4 mux.
It peaks at 4,748MiB PID-level NVML memory and finishes the pipeline in 798.94
seconds.  The validated output is H.264 832×480 with 124 frames and AAC stereo
audio.  A sequential native run on the same physical GPU completes the same
workload in 305.04 seconds but peaks at 20,970MiB.  Both generated videos and
the complete protocol are in the [15K native-vs-seqattn report](docs/minimax_h3_native_vs_seqattn_15k.md).

## Completed 15K comparison

The formal 15K video-generation points were run sequentially on the same
physical RTX 5090 with the same image, NF4 checkpoint, prompt, seed, input, and
scheduler.  Native DiffSynth is unrestricted; `seqattn` has a strict 6GiB
whole-process target.  Both points completed all 50 denoise steps, both VAE
decoders, and MP4 mux.

| Completed run · August 18, 2026 UTC | Native DiffSynth | `seqattn` · 6GiB |
|---|---:|---:|
| Completed workflow | 50 steps + dual VAE + MP4 | 50 steps + dual VAE + MP4 |
| 50-step DiT | **227.833 s** | 764.313 s |
| Median step | **4.428 s** | 14.637 s |
| Full pipeline | **305.038 s** | 798.938 s |
| PID-level NVML peak | 20,970 MiB | **4,748 MiB** |
| CPU RSS peak | **35,164 MiB** | 38,697 MiB |

Native is 3.35× faster for the 50-step DiT when the sequence fits.  `seqattn`
uses only 22.64% of the native PID peak, stays 1,396MiB below its 6GiB target,
and completes the same generated-media workflow.  This is a capacity result:
CPU RSS and PCIe traffic increase, and the implementation does not claim a
speed advantage over native FlashAttention at this sequence length.

| Generated media | Link |
|---|---|
| Native DiffSynth | [▶ 832×480, 124-frame MP4](docs/media/minimax_h3_15k_native.mp4) |
| `seqattn` · 6GiB | [▶ 832×480, 124-frame MP4](docs/media/minimax_h3_15k_seqattn_6gb.mp4) |

The separate 132,288-token stress test has a different boundary: native OOMs
while starting step 15, whereas `seqattn` completes all 50 DiT steps in
11,941.56 seconds with an approximately 7,166MiB DiT peak.  Its subsequent
Video VAE assembly OOM is retained as a failure; it is not reported as a full
end-to-end video success.

## Native memory residency

The native run also uses CPU/DRAM weight offload; it is not keeping the whole
checkpoint in GPU memory.  During the denoise loop only the DiT is active.  The
text encoder and both VAEs are offloaded and are not part of the 30,876MiB peak.

| During native DiT denoising | CPU DRAM | GPU HBM |
|---|---|---|
| Model weights | Inactive models and offloaded DiT NF4 backing weights | Prepared/current DiT layers |
| Attention activations | — | Full hidden, residual, Q, K, V and attention output |
| MLP activations | — | Full `fc1`, gate, up and product tensors |
| Runtime | Python/checkpoint backing | CUDA context, FA workspace and Torch reserved cache |

<p align="center">
  <img src="https://raw.githubusercontent.com/renlililoli/stream-attn/main/docs/assets/minimax-h3-native-residency.svg" alt="Native MiniMax-H3 memory residency" width="100%">
</p>

For 132,288 BF16 tokens, full QKV is approximately 5.299GiB and the MLP `fc1`
output is 7.065GiB.  The native failure requests 3.53GiB for the complete
`SiLU(gate) * up` result.  Thus the immediate OOM is a sequence-activation
allocation, not simultaneous residency of the encoder, DiT, and decoders.

## Repository layout

| Path | Purpose |
|---|---|
| `extern/seqattn` | Standalone Triton exact out-of-core attention operator |
| `extern/DiffSynth-Studio` | MiniMax-H3 integration branch |
| `workspace/benchmarks` | Strict-memory runner, PID NVML sampler, JSON/trace output |
| `docs` | Experiment protocol, failure history, interpretation, and limits |

Clone with both implementations:

```bash
git clone --recurse-submodules git@github.com:renlililoli/minimax-h3-seq-chunk-attn.git
```

The Compose setup mounts both submodules and installs `seqattn` as an editable
package in the DiffSynth image.

## Why this differs from Stream-CQSA

Both systems move complete Q/K/V outside GPU memory, but the execution model is
different.  Stream-CQSA recursively creates overlapping combinatorial
subsequences.  `seqattn` uses regular resident-Q × streamed-K/V scheduling:

```text
H2D = |Q| + resident_q_passes × (|K| + |V|)
D2H = |projected output|
```

This gives deterministic K/V reuse within each resident query set, avoids a
CPU FP32 full-sequence numerator accumulator, performs stable online-softmax
merging in HBM, and supports direct attention→out-projection consumption.  The
standalone [`seqattn` README](https://github.com/renlililoli/stream-attn)
contains the full complexity comparison, API, limitations, and operator
benchmarks.

## Measurement boundary

- Memory source of record: current-PID NVML samples every 2ms for 132K runs.
- Weights: MiniMax-H3 FL2VA NF4; compute: BF16.
- Offload backing: CPU DRAM, never disk.
- Logical H2D/D2H values are instrumented traffic, not link-level counters.
- Completed points are single-run system characterization without error bars.
- Capacity is the current advantage; native FlashAttention remains faster when
  the complete sequence fits comfortably in HBM.

Detailed reports:

- [15K native vs. strict-6GiB `seqattn`, with generated videos](docs/minimax_h3_native_vs_seqattn_15k.md)
- [8GB / 61K end-to-end experiment](docs/minimax_h3_8gb_61k_end_to_end_experiment.md)
- [Exclusive-GPU native rerun procedure](docs/exclusive_gpu_benchmark.md)
- [Standalone H3 integration report](https://github.com/renlililoli/stream-attn/blob/main/docs/minimax_h3_integration.md)
