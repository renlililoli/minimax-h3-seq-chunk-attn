# Two-Step End-to-End Examples

These scripts run real MiniMax-H3 workloads through the installed community
node. Every scenario performs two denoise steps, streaming video VAE
encode/decode, full SeqAttn DiT execution, audio decode, MP4 writing, and PyAV
read-back validation.

## Install From Scratch

Install the node into a current ComfyUI checkout:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone --branch community/comfyui-minimax-h3-seqattn \
  https://github.com/renlililoli/minimax-h3-seq-chunk-attn.git \
  ComfyUI-MiniMaxH3-SeqAttn
cd ComfyUI-MiniMaxH3-SeqAttn
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

Override the image with `SEQATTN_EXAMPLE_IMAGE`. Select scenarios with a
space-separated `SEQATTN_EXAMPLE_SCENARIOS` value.

## Recorded Results

The following clean-container runs completed on August 22, 2026 UTC using an
RTX 5090, PyTorch 2.10.0+cu128, seed 0, and an 8,192 MiB process target. These
are functional two-step checks, not quality or throughput benchmarks.

| Scenario | Packed tokens | GPU peak | CPU RSS peak | Denoise | Wall time | Validated media |
|---|---:|---:|---:|---:|---:|---|
| T2VA | 17,375 | 7,344 MiB | 45.68 GiB | 101.296 s | 165.979 s | 56 frames, 24 fps, AAC |
| FL2VA | 21,422 | 7,326 MiB | 47.35 GiB | 141.388 s | 208.115 s | 56 frames, 24 fps, AAC |
| Ref2VA images | 41,824 | 7,392 MiB | 50.42 GiB | 158.371 s | 253.196 s | 124 frames, 24 fps, AAC |
| Ref2VA video | 81,180 | 7,756 MiB | 55.85 GiB | 272.410 s | 398.962 s | 124 frames, 24 fps, AAC |

For all four runs, ComfyUI executed `condition_proj` and `token_refiner`
exactly once during sampling preparation. Both denoise forwards then received
the already refined 5,376-wide conditioning and correctly bypassed SeqAttn's
compatibility fallback cache.

## Result Files

Each `examples/results/<scenario>/` directory contains:

- `output.mp4`: generated H.264 video with AAC stereo audio.
- `result.json`: configuration, prompt, environment, packed layout, Qwen
  preflight, phase timings, refiner counts, memory peaks, and media probe.
- `memory_trace.csv.gz`: 50 ms process-level GPU/CPU and PyTorch allocator
  samples for the complete run.

[`results/summary.json`](results/summary.json) provides a compact
machine-readable index. The per-scenario `result.json` files remain the source
of truth.
