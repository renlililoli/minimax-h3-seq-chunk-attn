#!/usr/bin/env bash
set -euo pipefail

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
package_root=$(cd "$example_dir/.." && pwd)
image=${SEQATTN_EXAMPLE_IMAGE:-comfyui:cu128}
gpu=${SEQATTN_EXAMPLE_GPU:-0}
models_dir=${COMFYUI_MODELS_DIR:-}
output_root=${SEQATTN_EXAMPLE_OUTPUT_DIR:-$example_dir/results}
scenario_list=${SEQATTN_EXAMPLE_SCENARIOS:-t2va fl2va ref2va_images ref2va_video}
install_dir=/opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SeqAttn

if [[ -z "$models_dir" ]]; then
  echo "set COMFYUI_MODELS_DIR to a directory containing diffusion_models/, text_encoders/, and vae/" >&2
  exit 2
fi

models_dir=$(cd "$models_dir" && pwd)
mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd)

required=(
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
)
if [[ " $scenario_list " == *" t2va "* || " $scenario_list " == *" fl2va "* ]]; then
  required+=(diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors)
fi
if [[ " $scenario_list " == *" ref2va_images "* || " $scenario_list " == *" ref2va_video "* ]]; then
  required+=(diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
fi
for path in "${required[@]}"; do
  if [[ ! -f "$models_dir/$path" ]]; then
    echo "missing model: $models_dir/$path" >&2
    exit 2
  fi
done

for scenario in $scenario_list; do
  mkdir -p "$output_root/$scenario"
  docker run --rm \
    --gpus "device=$gpu" \
    --ipc host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER=seqattn \
    -e LOGNAME=seqattn \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}" \
    -e MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-131072}" \
    -e MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-0}" \
    -e SEQATTN_EXAMPLE_NVML_GPU_INDEX=0 \
    -v "$package_root:$install_dir:ro" \
    -v "$models_dir:/opt/ComfyUI/models:ro" \
    -v "$output_root:/results" \
    "$image" \
    python "$install_dir/examples/run_2step_example.py" \
      --scenario "$scenario" \
      --comfyui-dir /opt/ComfyUI \
      --output-dir "/results/$scenario" \
      "$@"
done
