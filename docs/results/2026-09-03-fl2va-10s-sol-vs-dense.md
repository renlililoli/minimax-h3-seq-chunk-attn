# FL2VA 10-Second Sol vs. Dense Denoising Record

Date: September 3, 2026

This record contains sanitized timing evidence for the two later completed
10.0-second FL2VA Turbo 4-step LoRA runs from each service. Prompt text, input
media, and generated video content are intentionally omitted.

## Metric and Selection

The denoising value is the elapsed time shown on the final ComfyUI sampler
`4/4` progress line. Per-step elapsed values are differences between
consecutive cumulative progress timestamps. The timestamps have whole-second
resolution, so the derived step values are approximate to about one second.
These values are not the full `Prompt executed` wall time.

Included runs met all of these conditions:

- duration input `10.0` seconds;
- FL2VA Turbo 4-step LoRA workflow;
- final sampler progress reached `4/4`;
- scheduler `simple` with four steps;
- 1.0 MP resolution at 9:16;
- materialized MiniMax-H3 execution;
- MiniMax-H3 Q/KV chunks `15360`/`4096`;
- Qwen Q/KV chunks `5760`/`4096`;
- all four cumulative step boundaries were present and monotonic.

Only the two later completed runs per mode were selected so the whole-run and
per-step comparisons use the same complete sample set. Five-second jobs, a
failed Dense prompt, `0.01`-second cache-only history entries, and jobs without
a final `4/4` sampler line were excluded. No completed 10-second Ref2VA sample
was available.

## Results

### Sol Streaming, GPU 1

The service mounted `docker/seqattn-sol.toml` with
`attention_mode = "sol_streaming"`, `sol_tau = 1.0`,
`sol_first_dense_step_fraction = 0.2`, and `sol_first_dense_layers = 2`.

| Prompt ID | Output | Full workflow | Cumulative steps | Step elapsed |
|---|---|---:|---:|---:|
| `4dd15bea-b79c-469c-b9c1-85bfaac0fad7` | `00005` | 261.64 s | 1:13 / 1:34 / 2:21 / 3:09 | 73 / 21 / 47 / 48 s |
| `9f98eaa7-3304-41d8-88a3-3909d65e4134` | `00006` | 263.98 s | 1:13 / 1:34 / 2:22 / 3:09 | 73 / 21 / 48 / 47 s |

Median denoising time: 189 seconds (3:09). Both samples were 189 seconds.

### Dense, GPU 2

The service mounted `docker/seqattn.toml` with
`attention_mode = "dense"`.

| Prompt ID | Output | Full workflow | Cumulative steps | Step elapsed |
|---|---|---:|---:|---:|
| `d5648cea-e0eb-4634-9ef7-f7d5a19c0a36` | `00008` | 344.34 s | 1:14 / 2:29 / 3:44 / 4:59 | 74 / 75 / 75 / 75 s |
| `1a8c5ba9-2597-4984-8f14-2f708567a0ae` | `00009` | 341.79 s | 1:14 / 2:28 / 3:43 / 4:57 | 74 / 74 / 75 / 74 s |

Median denoising time: 298 seconds (4:58). Range: 297-299 seconds.

The submitted workflow retained the Sol-oriented filename and a 15,360-token
MiniMax-H3 Q chunk on the Dense service. The output name therefore does not
identify the runtime backend, and these results are not a default-Q Dense
comparison. Backend classification comes from the mounted service TOML.

### Per-Step Comparison

| Step | Sol samples | Sol median | Dense samples | Dense median | Dense/Sol |
|---:|---:|---:|---:|---:|---:|
| 1 | 73 s, 73 s | 73 s | 74 s, 74 s | 74 s | 1.01x |
| 2 | 21 s, 21 s | 21 s | 75 s, 74 s | 74.5 s | 3.55x |
| 3 | 47 s, 48 s | 47.5 s | 75 s, 75 s | 75 s | 1.58x |
| 4 | 48 s, 47 s | 47.5 s | 75 s, 74 s | 74.5 s | 1.57x |

With a four-step schedule and `sol_first_dense_step_fraction = 0.2`, Sol step
1 is dense. Steps 2-4 use Sol routing while the configured first two H3 blocks
remain dense. The Sol whole-run median was 109 seconds lower than the Dense
median, a 36.6% elapsed-time reduction. The Dense/Sol median ratio was 1.58x.

## Environment

| Component | Value |
|---|---|
| Community node | `0.4.4` at `886c95a16173606377a23b6545a3beca15ce9685` |
| SeqAttn core | `0.4.0a1` at `d8c51ef1e347d76f237478949679a976d8179bde` |
| ComfyUI | `0.30.0` at `9a9fdb10ed144ce760d9682cb247526ea23cc525` |
| PyTorch | `2.10.0+cu128` |
| NVIDIA driver | `595.84` |
| GPU | NVIDIA GeForce RTX 5090 |
| Container image | `sha256:56f247ce123e6458f6743301f1344344d7dff2af53107105429cc7adcce18f0b` |
| Sol GPU UUID | `GPU-28e1c1eb-f738-21b5-909c-b025a2281165` |
| Dense GPU UUID | `GPU-a9f0c52d-3d52-fd1a-71b9-44f2a59366f7` |

Both services used the same image, model, LoRA, workflow parameters, and host.
They ran on separate GPUs but shared CPU resources and the pinned host-memory
path, and some samples overlapped. This is preserved functional evidence from
completed workflows, not a controlled throughput benchmark.
