# 1080p-Class FL2VA Sol Warm-Start Record

Date: September 4, 2026

This record contains sanitized timing evidence for a completed 10.0-second
FL2VA Sol Turbo 4-step LoRA workflow. Prompt text, input media, and generated
video content are intentionally omitted.

## Warm-Start Definition

Before the measured run, the same RTX 5090 service completed one full
1080p-class four-step denoising run as warmup.

## Output and Configuration

| Setting | Value |
|---|---|
| Sanitized output filename | `MiniMax_H3_Sol_FL2VA_4step_00011_.mp4` |
| Output geometry | `1920x1088`, aligned 1080p-class |
| Frames and rate | 243 frames at 24 FPS |
| Encoded duration | 10.125 s |
| Duration input | 10.0 s |
| Resolution selector | 2.0 MP, 16:9 |
| Workflow | FL2VA Sol Turbo 4-step LoRA |
| Conditioning | First frame connected; last frame disconnected |
| Scheduler and sampler | `simple`; `res_multistep` |
| Denoising steps | 4 |
| Execution and attention | `materialized`; `sol_streaming` |
| MiniMax-H3 Q/KV chunks | `15360` / `4096` |
| Qwen Q/KV chunks | `5760` / `4096` |
| LoRA strength | 1.0 |

The output geometry, frame count, frame rate, duration, and workflow settings
were read from the completed MP4 and its embedded ComfyUI metadata.

## Denoising Timing

The sampler's cumulative elapsed values and their consecutive differences are:

| Step | Configured routing | Cumulative | Step elapsed |
|---:|---|---:|---:|
| 1 | Dense | 3:32 | about 212 s |
| 2 | Sol streaming | 4:13 | about 41 s |
| 3 | Sol streaming | 6:13 | about 120 s |
| 4 | Sol streaming | 8:15 | about 122 s |

The cumulative values are displayed at whole-second resolution. The derived
step values are therefore approximate to about one second and sum to the 8:15
denoising total. The full `Prompt executed` workflow time was 11:50.

With `sol_first_dense_step_fraction = 0.2`, step 1 is dense in a four-step
schedule. Steps 2-4 use Sol routing while `sol_first_dense_layers = 2` keeps
the first two H3 blocks dense.

## Environment

| Component | Value |
|---|---|
| Community node | `0.4.4` at `886c95a16173606377a23b6545a3beca15ce9685` |
| SeqAttn core | `0.4.0a1` at `d8c51ef1e347d76f237478949679a976d8179bde` |
| ComfyUI | `0.30.0` at `9a9fdb10ed144ce760d9682cb247526ea23cc525` |
| PyTorch | `2.10.0+cu128` |
| Triton | `3.6.0` |
| NVIDIA driver | `595.84` |
| GPU | NVIDIA GeForce RTX 5090, physical GPU 2 |
| GPU UUID | `GPU-a9f0c52d-3d52-fd1a-71b9-44f2a59366f7` |
| Container image | `sha256:56f247ce123e6458f6743301f1344344d7dff2af53107105429cc7adcce18f0b` |

This is preserved functional evidence from one completed workflow, not a
controlled throughput benchmark or a cross-system performance claim.
