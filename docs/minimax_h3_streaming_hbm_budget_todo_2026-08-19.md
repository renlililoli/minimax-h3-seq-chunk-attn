# MiniMax-H3 streaming activation HBM budget

Date: August 19, 2026

## Goal

Make the HBM workspace requested by the MiniMax-H3 streaming integration match
the peak GPU memory attributable to denoise activations. Model weights, the CUDA
context, allocator bookkeeping, and non-DiT pipeline state are explicitly outside
this planner and remain separately observable. A `2 GiB` activation workspace
therefore has one narrow, stable meaning and does not claim a 2 GiB process peak.

The legacy `seqattn_workspace_mib` parameter continues to limit only CUDA buffers
owned by the standalone attention runner. The new
`streaming_activation_workspace_mib` parameter enables the H3-wide activation
planner and derives the SeqAttn workspace and all phase-specific chunk sizes.

## Scope boundary

Included in the activation budget:

- GPU-resident DiT inputs that remain live during the streaming call;
- embedding and RoPE projection tiles;
- QKV projection pipeline buffers;
- SeqAttn Q/KV/state buffers;
- the device-output projection, residual, and AdaLN gate tile;
- fused/split MLP tiles;
- final-layer tiles and selected video/audio output tensors;
- a fixed allocator and small-metadata margin.

Excluded from the activation budget:

- prepared and transient model weights;
- CUDA context and driver allocations;
- PyTorch allocator reserved-but-unused memory;
- VAE, text encoder, scheduler, and other non-DiT pipeline state.

Weight residency remains DiffSynth's responsibility. Coordinating it with a
whole-process VRAM target is a separate optional feature and must not change the
activation planner's meaning.

## Legacy audit before the activation planner

A single-GPU audit used physical RTX 5090 GPU3, one benchmark process, one
262K-class denoise step, one DiT block, BF16 activations, fused MLP, and a
2,048 MiB SeqAttn workspace.

| Measurement | Current value |
|---|---:|
| SeqAttn planned workspace | 2,142,609,408 bytes / 2,043.35 MiB |
| Resident Q chunk | 25,984 tokens |
| Torch peak allocated | 6,346.29 MiB |
| Torch peak reserved | 6,966 MiB |
| PID NVML peak | 7,656 MiB |
| PID NVML at step end | 4,128 MiB |

Artifact:

```text
workspace/benchmarks/results/hbm_workspace_audit_20260819/
current_262k_1block_gpu3_720x1280_f957_s1_20260819T073100Z.json
```

The host was not globally exclusive, but no second SeqAttn benchmark shared
DRAM bandwidth and GPU3 was dedicated to this run.

## Mismatches found by the legacy audit

### 1. Embedding is not activation-streamed

`MiniMaxH3DiT._embed()` projects all video tokens into a complete GPU
`video_embed`, then allocates another complete `[tokens, 5376]` GPU
`embeddings` tensor before copying the result to pinned CPU memory. At the
audited length, one complete BF16 hidden tensor is about 2.63 GiB. The two
full-sequence allocations dominate the early denoise peak before SeqAttn runs.

### 2. Device-output temporaries are outside the SeqAttn plan

The old H3 integration consumed finalized attention on GPU, but built a
SeqAttn plan with `output_mode="host"` and two raw-output buffers. The output
projection callback additionally allocates projected attention, a residual
tile, gathered AdaLN gate rows, and pointwise intermediates that are not
charged to `seqattn_workspace_mib`.

For the current 25,984-token Q chunk:

| Allocation class | Approximate BF16 size |
|---|---:|
| SeqAttn Q buffer | 355.25 MiB |
| SeqAttn FP32 online-softmax state | 721.60 MiB |
| Two SeqAttn raw-output buffers | 710.50 MiB |
| SeqAttn K/V ring | 224.00 MiB |
| Projected output tile | 266.44 MiB |
| Residual tile | 266.44 MiB |
| Gathered gate rows | 266.44 MiB |

The last three rows alone added at least 799 MiB outside the attention planner;
the old expression could hold further pointwise temporaries concurrently.

### 3. Projection and MLP share one unrelated chunk control

`projection_chunk_size` controls QKV projection, fused MLP, and the final
layer. Their activation shapes and memory costs differ substantially. In the
default H3 shape, a 2,048-token fused MLP tile materializes a 112 MiB FC1
output plus a 56 MiB gated intermediate, while QKV projection has a different
set of normalization, modulation, QKV, Q/K norm, and RoPE temporaries.

### 4. Whole-process weight residency is coordinated only heuristically

DiffSynth's wrapper residency threshold and the PyTorch allocator cap are
configured separately from the SeqAttn workspace. The wrapper checks current
device usage before preparing a layer but does not reserve the next layer's
size or the model-owned activation peak. Changing the attention workspace can
therefore change the whole-process OOM margin without changing the layer
residency policy.

## Implemented

1. Added a streaming embedding path that writes chunked video/audio/text
   projections directly into pinned CPU hidden storage and never constructs a
   full `[tokens, hidden_size]` GPU tensor.
2. Added an H3 activation-workspace plan. It charges persistent
   projection buffers, output-consumer bytes per resident Q token, MLP/final
   layer tiles, RoPE/index tensors, and a fixed allocator margin before choosing
   the attention Q chunk.
3. Switched the H3 adapter to SeqAttn's device-consumer mode and used
   in-place gate/residual modulation where numerically equivalent.
4. Split projection, MLP, embedding, and final-layer chunk controls. The planner
   reduces each independently when the requested cap does not fit.
5. Moved long-sequence RoPE and AdaLN indices to CPU backing and now transfers
   only the current tile to GPU.
6. Reports planned phase peaks, persistent input bytes, selected chunk sizes,
   SeqAttn workspace, Torch allocated/reserved memory, DiT CUDA storage, and
   PID-level NVML memory as distinct quantities.

## Deferred optional feature: weight residency

DiffSynth currently decides whether to prepare a layer using its existing
`vram_limit` heuristic. A future, separate feature may derive that threshold
from a whole-process target and reserve the largest transient layer. It should:

1. expose a separate process/weight budget API rather than reuse the activation
   workspace parameter;
2. account for CUDA context, prepared weights, transient computation copies,
   allocator safety margin, and the already-planned activation workspace;
3. report prepared and transient weight peaks independently;
4. remain optional so SeqAttn's activation-memory guarantee does not depend on
   a particular model-offload implementation.

## 262K activation-budget audit

RTX 5090 GPU3, one process, 262K-class sequence, one DiT block, one denoise
step, BF16, fused MLP, 4K KV tile:

| Activation budget | Planned peak | SeqAttn workspace | Resident Q | Denoise time | Torch peak allocated | Step PID NVML peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 1016.0 MiB | 574.3 MiB | 7,680 | 26.85 s | 2,173.7 MiB | 3,248 MiB |
| 2 GiB | 2041.9 MiB | 1163.1 MiB | 21,888 | 27.65 s | 2,762.5 MiB | 3,928 MiB |
| 4 GiB | 4093.7 MiB | 2340.7 MiB | 50,304 | 27.43 s | 4,813.5 MiB | 6,388 MiB |

Torch and NVML columns include weights and runtime overhead and therefore are
not expected to equal the activation budget. The JSON artifacts retain the
separate DiT CUDA-storage measurement needed to interpret those totals.

Artifacts:

```text
workspace/benchmarks/results/hbm_workspace_planner_20260819/
```

## Acceptance criteria

- At 524,288 tokens, verify that no full-sequence GPU hidden, Q, K, V,
  attention output, or MLP intermediate exists outside the declared activation
  workspace. The completed audit in this document covers 262K tokens.
- Across 1/2/4 GiB workspace settings, every planned activation phase remains
  within the configured budget and selected chunks grow monotonically.
- Whole-process Torch/NVML memory and DiT weight storage are reported separately;
  they are not presented as activation-budget violations.
- Fused and split MLP modes, device-consumer and host-output attention modes,
  and streamed versus non-streamed embedding have numerical parity tests.
- Formal performance runs use one GPU process at a time to avoid host DRAM
  bandwidth contention.
