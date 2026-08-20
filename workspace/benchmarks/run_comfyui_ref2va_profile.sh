#!/usr/bin/env bash
set -euo pipefail

gpu=${COMFYUI_BENCH_GPU:-1}
output_dir=${COMFYUI_BENCH_OUTPUT_DIR:-workspace/benchmarks/results/comfyui_ref2va_profile_20260820}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$output_dir"
chmod 0777 "$output_dir"

docker run --rm --gpus "device=$gpu" \
  --ipc host \
  -e BENCH_NVML_GPU_INDEX="$gpu" \
  -e PYTHONPATH=/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v "$PWD:/workspace" \
  -v /scratch/grzhu/weights/video/minimax/merged:/opt/ComfyUI/models:ro \
  -v /scratch/grzhu/weights/video/minimax/assets:/models/minimax/assets:ro \
  comfyui-seqattn:test \
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="/workspace/$output_dir/seqattn_h3_${stamp}" \
  python /workspace/workspace/benchmarks/comfyui_minimax_h3_ref2va_8g.py \
    --mode streaming \
    --source /models/minimax/assets/h3_direct_768p.mp4 \
    --output-dir "/workspace/$output_dir" \
    --height 768 --width 1344 --frames 243 --steps 1 --seed 0 \
    --target-vram-mib 8192 \
    --activation-workspace-mib 1024 \
    --kv-chunk-tokens 4096 \
    --profile-denoise \
    --cuda-profiler-capture \
    --skip-decode
