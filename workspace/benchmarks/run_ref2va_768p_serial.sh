#!/usr/bin/env bash
set -euo pipefail

phase=${1:-two-step}
container=${REF2VA_CONTAINER:-diffsynth-long-gpu3}
physical_gpu=${REF2VA_PHYSICAL_GPU:-3}
cpu_set=${REF2VA_CPU_SET:-160-191,416-447}
output_dir=${REF2VA_OUTPUT_DIR:-/workspace/benchmarks/results/ref2va_768p_20260819}
host_output_dir=${REF2VA_HOST_OUTPUT_DIR:-workspace/benchmarks/results/ref2va_768p_20260819}
lock_file=${REF2VA_LOCK_FILE:-/tmp/minimax-h3-ref2va-gpu3.lock}

mkdir -p "$host_output_dir"
chmod 0777 "$host_output_dir"
exec 9>"$lock_file"
flock -n 9 || {
  echo "another Ref2VA benchmark owns $lock_file" >&2
  exit 1
}

common=(
  --height 768
  --width 1344
  --frames 243
  --seed 0
  --source /models/minimax/assets/h3_direct_768p.mp4
  --model-dir /models/MiniMax-H3-NF4
  --processor-dir /models/minimax/processor
  --output-dir "$output_dir"
  --offload-device cpu
  --projection-chunk-size 8192
  --attention-kv-block-size 4096
  --sample-interval-ms 20
  --save-latents
  --save-memory-trace
)

run_point() {
  local tag=$1
  local mode=$2
  local steps=$3
  shift 3
  local log="$host_output_dir/${tag}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $tag" | tee "$log"
  docker exec \
    -e BENCH_NVML_GPU_INDEX="$physical_gpu" \
    -e BENCH_PHYSICAL_GPU_INDEX="$physical_gpu" \
    -e PYTHONPATH=/opt/DiffSynth-Studio:/workspace \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$container" \
    taskset -c "$cpu_set" \
    python -m benchmarks.minimax_h3_bench.ref2va_point \
    --tag "$tag" --mode "$mode" --steps "$steps" \
    "${common[@]}" "$@" 2>&1 | tee -a "$log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] finished $tag" | tee -a "$log"
}

case "$phase" in
  two-step)
    run_point ref2va_streaming_8g_2step streaming 2 \
      --target-vram-mib 8192 --activation-workspace-mib 4096
    run_point ref2va_native_8g_2step native 2 --target-vram-mib 8192
    run_point ref2va_native_32g_2step native 2 --vram-reserve-gib 5
    ;;
  full)
    run_point ref2va_streaming_8g_full50 streaming 50 \
      --target-vram-mib 8192 --activation-workspace-mib 4096
    ;;
  *)
    echo "usage: $0 {two-step|full}" >&2
    exit 2
    ;;
esac
