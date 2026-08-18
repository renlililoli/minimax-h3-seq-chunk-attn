# MiniMax-H3 sequence streaming

> Exact 132K-token MiniMax-H3 inference with a bounded GPU working set.

[![seqattn](https://img.shields.io/badge/operator-seqattn-22d3ee?style=for-the-badge)](https://github.com/renlililoli/stream-attn)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](#current-results)
[![Precision](https://img.shields.io/badge/weights-NF4-8b5cf6?style=for-the-badge)](#current-results)
[![Status](https://img.shields.io/badge/50--step%20soak-running-f59e0b?style=for-the-badge)](#live-50-step-comparison)

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
| 132,288-token 50-step attempt | **OOM after 14 steps** | **14 steps and continuing** | bounded activation memory |
| 132,288-token GPU peak | 30,876 MiB | **7,164 MiB** | **4.31× lower** |
| 132,288-token completed capacity probe | — | **5,968 MiB** | succeeds below 8GiB |
| 61,312-token projected-pipeline peak | 7,108 MiB | **3,848 MiB** | **45.9% lower** |
| 61,312-token projected-pipeline latency | 919.79 ms | **843.44 ms** | **8.3% faster** |
| 61,056-token full H3 PCIe traffic | 917.6 GiB | **836.1 GiB** | **81.5 GiB less/step** |

The completed 132K capacity probe executes all 50 DiT blocks for one denoise
step under an 8,192MiB whole-process target.  It takes 236.39 seconds and peaks
at 5,968MiB PID-level NVML memory.  The input requested as 720×1280 and 480
frames is aligned by H3 to 736×1280 and 481 frames: 20.04 seconds at 24fps and
132,288 packed tokens.

## Live 50-step comparison

The formal video-generation experiment runs simultaneously on two separate
RTX 5090 GPUs with the same image, NF4 checkpoint, prompt, seed, input, and
scheduler.  Native DiffSynth is unrestricted; `seqattn` has a strict 8GiB
whole-process target.  Current numbers are a live snapshot, not a completed
video claim.

| Live snapshot · August 18, 2026 UTC | Native DiffSynth | `seqattn` |
|---|---:|---:|
| Completed steps | **14 / 50, then OOM** | **14 / 50, still running** |
| Mean step through step 14 | **140.068 s** | about 224.31 s |
| PID-level NVML peak | 30,876 MiB | **7,164 MiB** |
| Step-end memory | 30,876 MiB | **4,432 MiB** |

Native is about 1.60× faster per successful step, but the unrestricted process
OOMs while starting step 15: it requests another 3.53GiB with only 1.20GiB free
on the 31.36GiB device.  `seqattn` uses only 23.2% of the native peak and stays
flat through the matching 14-step checkpoint inside an 8GiB target.  Final
`seqattn` numbers will be published only after all 50 steps, Video VAE decode,
Audio VAE decode, and MP4 mux complete.

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

- [8GB / 61K end-to-end experiment](docs/minimax_h3_8gb_61k_end_to_end_experiment.md)
- [Standalone H3 integration report](https://github.com/renlililoli/stream-attn/blob/main/docs/minimax_h3_integration.md)
