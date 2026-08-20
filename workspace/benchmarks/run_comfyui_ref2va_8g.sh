#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_comfyui_ref2va_8g.sh native|streaming}
case "$mode" in
  native|streaming) ;;
  *) echo "mode must be native or streaming" >&2; exit 2 ;;
esac
shift

gpu=${COMFYUI_BENCH_GPU:-1}
output_dir=${COMFYUI_BENCH_OUTPUT_DIR:-workspace/benchmarks/results/comfyui_ref2va_8g_20260820}
text_encoder_mode=${COMFYUI_TEXT_ENCODER_MODE:-gpu-offload}
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
  python /workspace/workspace/benchmarks/comfyui_minimax_h3_ref2va_8g.py \
    --mode "$mode" \
    --source /models/minimax/assets/h3_direct_768p.mp4 \
    --output-dir "/workspace/$output_dir" \
    --height 768 --width 1344 --frames 243 --steps 1 --seed 0 \
    --target-vram-mib 8192 \
    --activation-workspace-mib 1024 \
    --kv-chunk-tokens 4096 \
    --text-encoder-mode "$text_encoder_mode" \
    "$@"
