# Two-Step End-to-End Examples

These scripts run real MiniMax-H3 workloads through the installed community
node. Every scenario performs two denoise steps, streaming video VAE
encode/decode, full SeqAttn DiT execution, audio decode, MP4 writing, and PyAV
read-back validation.

## Install From Scratch

Install the node into the pinned ComfyUI checkout:

```bash
cd /path/to/ComfyUI
git fetch origin 9a9fdb10ed144ce760d9682cb247526ea23cc525
git checkout --detach 9a9fdb10ed144ce760d9682cb247526ea23cc525

cd /path/to/ComfyUI/custom_nodes
git clone --branch community/comfyui-minimax-h3-seqattn \
  https://github.com/renlililoli/minimax-h3-seq-chunk-attn.git \
  ComfyUI-MiniMaxH3-SeqAttn
cd ComfyUI-MiniMaxH3-SeqAttn
python -m pip install -e .
```

Place these files under `ComfyUI/models/`:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

Run one scenario from the custom-node directory:

```bash
export COMFYUI_DIR=/path/to/ComfyUI
./examples/run_t2va_2step.sh
./examples/run_fl2va_2step.sh
./examples/run_ref2va_images_2step.sh
./examples/run_ref2va_video_2step.sh
```

Run all four sequentially:

```bash
export COMFYUI_DIR=/path/to/ComfyUI
./examples/run_all_2step.sh
```

The scripts also infer `COMFYUI_DIR` automatically when the repository is
installed at `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn`.

The runner verifies that `COMFYUI_DIR` is exactly commit
`9a9fdb10ed144ce760d9682cb247526ea23cc525` and stops before loading models if
the checkout differs. It initializes DynamicVRAM before importing PyTorch, so
the scripts do not require a separate AIMDO bootstrap command.

## Scenarios

| Script | Mode | Input | Output scale |
|---|---|---|---:|
| `run_t2va_2step.sh` | T2VA | Text prompt | 1344x768, 56 frames |
| `run_fl2va_2step.sh` | FL2VA | Bundled first and last frames | 1344x768, 56 frames |
| `run_ref2va_images_2step.sh` | Ref2VA | Two bundled reference images | 1344x768, 124 frames |
| `run_ref2va_video_2step.sh` | Ref2VA | Bundled 124-frame reference video | 1344x768, 124 frames |

Useful overrides are passed directly to the Python runner:

```bash
./examples/run_t2va_2step.sh --seed 123 --target-vram-mib 8192
./examples/run_ref2va_video_2step.sh --prompt "Use <Video 1> ..."
```

Audio decoding defaults to CPU so the complete pipeline, including media
post-processing, stays within the 8 GiB GPU target. Pass `--audio-device cuda`
when additional GPU memory is available.

## Clean-Install Docker Validation

Maintainers can mount the package read-only into a clean ComfyUI image. The
container receives only this package, the standard model tree, and the result
directory:

```bash
SEQATTN_EXAMPLE_GPU=1 \
COMFYUI_MODELS_DIR=/path/to/models \
./examples/run_all_2step_docker.sh
```

Override the image with `SEQATTN_EXAMPLE_IMAGE`. The image must contain the
pinned ComfyUI commit above; the runner rejects floating or mismatched
checkouts. The clean-install runner installs this package and its pinned
`seqattn-core` dependency into the ephemeral container before executing the
read-only mounted package while reusing the image's PyTorch and Triton. Select
scenarios with a space-separated
`SEQATTN_EXAMPLE_SCENARIOS` value.

## Recorded Results

All checked-in results below were regenerated from clean package installs on
August 25, 2026 UTC with package version `0.4.1`, ComfyUI `0.30.0` at commit
`9a9fdb10ed144ce760d9682cb247526ea23cc525`, an RTX 5090, PyTorch
2.10.0+cu128, seed 0, and an 8,192 MiB process target. These are functional
two-step checks, not throughput benchmarks.

| Scenario | Version | Packed tokens | GPU peak | CPU RSS peak | Denoise | Validated media |
|---|---:|---:|---:|---:|---:|---|
| T2VA | `0.4.1` | 17,375 | 6,366 MiB | 23.16 GiB | 213.823 s | 56 frames, 24 fps, AAC |
| FL2VA | `0.4.1` | 21,422 | 6,820 MiB | 24.31 GiB | 213.738 s | 56 frames, 24 fps, AAC |
| Ref2VA images | `0.4.1` | 41,824 | 6,820 MiB | 26.36 GiB | 250.413 s | 124 frames, 24 fps, AAC |
| Ref2VA video | `0.4.1` | 81,180 | 7,708 MiB | 31.88 GiB | 354.346 s | 124 frames, 24 fps, AAC |

Every run recorded two forwards, 100 blocks, and 500 weight-scheduler lifecycle
events with at most two staged blocks. The shared settings were
`q_chunk_tokens=5760` and 4,096 tokens for the K/V, QKV projection, and MLP
tiles. The per-scenario JSON remains the source of truth for phase timings,
Qwen preflight, refiner counters, and memory details.

## Result Files

Each `examples/results/<scenario>/` directory contains:

- `output.mp4`: generated H.264 video with AAC stereo audio.
- `result.json`: configuration, prompt, environment, packed layout, Qwen
  preflight, phase timings, refiner counts, memory peaks, and media probe.
- `memory_trace.csv.gz`: 20 ms process-level GPU/CPU and PyTorch allocator
  samples for the complete run.
- `weight_schedule.json.gz`: compressed block weight lifecycle events when the
  result was generated by the fused DiT path.

[`results/summary.json`](results/summary.json) provides a compact
machine-readable index for all four current results. The per-scenario
`result.json` files remain the source of truth.
