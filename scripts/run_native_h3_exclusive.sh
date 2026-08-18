#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

gpu_index=""
cpu_bind=""
mem_bind=""
image="diffsynth:cu128"
container_name="h3-native-exclusive"
tag="exclusive_720p20s_native_unlimited"
weights_dir="/scratch/grzhu/weights/video"
sample_interval_ms=2
dry_run=0

usage() {
  cat <<'EOF'
Run the native MiniMax-H3 720p/20s/50-step experiment on an administrator-
reserved EXCLUSIVE_PROCESS GPU.

Usage:
  scripts/run_native_h3_exclusive.sh [options]

Options:
  --gpu INDEX              Physical GPU index reserved by the administrator
  --cpu-bind LIST          numactl CPU list; inferred for known GPUs
  --mem-bind NODE          numactl memory node; inferred for GPU 1/3
  --image IMAGE            Docker image (default: diffsynth:cu128)
  --container NAME         Dedicated container name
  --tag TAG                Benchmark result tag
  --weights-dir PATH       Host model directory
  --sample-interval-ms N   PID NVML interval (default: 2)
  --dry-run                Validate and print the docker command only
  -h, --help               Show this help

The script deliberately does not set PYTORCH_CUDA_ALLOC_CONF. This reproduces
the original native DiffSynth allocator behavior. Run an allocator ablation as
a separate experiment rather than silently changing this baseline.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu_index=$2; shift 2 ;;
    --cpu-bind) cpu_bind=$2; shift 2 ;;
    --mem-bind) mem_bind=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --container) container_name=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --weights-dir) weights_dir=$2; shift 2 ;;
    --sample-interval-ms) sample_interval_ms=$2; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n $gpu_index ]] || die "--gpu is required; run scripts/find_idle_gpu.sh first"
[[ $gpu_index =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ $sample_interval_ms =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid sampling interval"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available"
command -v docker >/dev/null 2>&1 || die "docker is not available"
command -v numactl >/dev/null 2>&1 || die "numactl is not available on the host"
[[ -d $weights_dir ]] || die "weights directory does not exist: $weights_dir"

if [[ -z $cpu_bind || -z $mem_bind ]]; then
  case "$gpu_index" in
    0)
      [[ -n $cpu_bind ]] || cpu_bind="64-95,320-351"
      [[ -n $mem_bind ]] || mem_bind="2"
      ;;
    1)
      [[ -n $cpu_bind ]] || cpu_bind="224-255,480-511"
      [[ -n $mem_bind ]] || mem_bind="7"
      ;;
    2)
      [[ -n $cpu_bind ]] || cpu_bind="192-223,448-479"
      [[ -n $mem_bind ]] || mem_bind="6"
      ;;
    3)
      [[ -n $cpu_bind ]] || cpu_bind="160-191,416-447"
      [[ -n $mem_bind ]] || mem_bind="5"
      ;;
    *)
      die "unknown topology for GPU $gpu_index; pass --cpu-bind and --mem-bind"
      ;;
  esac
fi

gpu_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null) \
  || die "GPU $gpu_index does not exist or is not visible"
compute_mode=$(nvidia-smi -i "$gpu_index" --query-gpu=compute_mode --format=csv,noheader,nounits)
if [[ $compute_mode != "Exclusive Process" && $compute_mode != "Exclusive_Process" && $compute_mode != "EXCLUSIVE_PROCESS" ]]; then
  die "GPU $gpu_index is in '$compute_mode' mode; ask an administrator to run: sudo scripts/admin_gpu_exclusive.sh enable $gpu_index"
fi

existing_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid { print }')
if [[ -n $existing_processes ]]; then
  printf 'GPU %s already has a CUDA process:\n%s\n' "$gpu_index" "$existing_processes" >&2
  die "exclusive GPU must be idle before launch"
fi

if docker inspect "$container_name" >/dev/null 2>&1; then
  die "container name already exists: $container_name (inspect/remove it explicitly)"
fi

results_dir="$repo_root/workspace/benchmarks/results"
mkdir -p "$results_dir"
started_utc=$(date -u +%Y%m%dT%H%M%SZ)
audit_csv="$results_dir/${tag}_gpu${gpu_index}_${started_utc}_gpu_audit.csv"
audit_summary="$results_dir/${tag}_gpu${gpu_index}_${started_utc}_gpu_audit.txt"
log_path="$results_dir/${tag}_gpu${gpu_index}_${started_utc}.log"

docker_cmd=(
  docker run --detach
  --name "$container_name"
  --gpus "device=$gpu_index"
  --shm-size 128g
  --cap-add SYS_NICE
  --workdir /workspace
  --volume "$repo_root/extern/DiffSynth-Studio:/opt/DiffSynth-Studio"
  --volume "$repo_root/extern/seqattn:/opt/seqattn"
  --volume "$repo_root/workspace:/workspace"
  --volume "$weights_dir:/models"
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPATH=/opt/DiffSynth-Studio:/workspace
  --env BENCH_NVML_GPU_INDEX=0
  --env BENCH_PHYSICAL_GPU_INDEX="$gpu_index"
  "$image"
  numactl --physcpubind="$cpu_bind" --membind="$mem_bind"
  python -m benchmarks.minimax_h3_baseline
  --height 720 --width 1280 --frames 480 --steps 50 --seed 0
  --tag "$tag"
  --offload-device cpu
  --sample-interval-ms "$sample_interval_ms"
  --save-memory-trace
)

printf 'GPU reservation:\n'
nvidia-smi -i "$gpu_index" \
  --query-gpu=index,uuid,name,compute_mode,memory.used,memory.total \
  --format=csv,noheader
printf 'CPU bind: %s\nMemory node: %s\n' "$cpu_bind" "$mem_bind"
printf 'Audit CSV: %s\nBenchmark log: %s\n' "$audit_csv" "$log_path"
printf 'Docker command:\n'
printf '  %q' "${docker_cmd[@]}"
printf '\n'

if [[ $dry_run -eq 1 ]]; then
  exit 0
fi

container_id=$("${docker_cmd[@]}")
printf 'container id: %s\n' "$container_id"

cleanup_hint() {
  printf '\nContainer retained for inspection: %s\n' "$container_name"
  printf 'Logs: docker logs %q\n' "$container_name"
  printf 'Remove after inspection: docker rm %q\n' "$container_name"
}
trap cleanup_hint EXIT

printf 'timestamp_utc,gpu_index,gpu_uuid,compute_mode,memory_used_mib,memory_total_mib,utilization_gpu_pct,container_host_pid,compute_processes,foreign_process_detected\n' > "$audit_csv"
contaminated=0

while [[ $(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true) == "true" ]]; do
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  container_pid=$(docker inspect --format '{{.State.Pid}}' "$container_name")
  gpu_row=$(nvidia-smi -i "$gpu_index" \
    --query-gpu=compute_mode,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits)
  IFS=',' read -r mode used total util <<< "$gpu_row"
  mode=${mode# }
  used=${used// /}
  total=${total// /}
  util=${util// /}
  process_rows=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid { printf "%s:%s:%s;", $2, $3, $4 }')
  foreign=0
  if [[ -n $process_rows ]]; then
    while IFS= read -r process_pid; do
      [[ -z $process_pid ]] && continue
      if [[ $process_pid != "$container_pid" ]]; then
        foreign=1
        contaminated=1
      fi
    done < <(printf '%s' "$process_rows" | tr ';' '\n' | cut -d: -f1)
  fi
  quoted_processes=${process_rows//\"/\"\"}
  printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s\n' \
    "$timestamp" "$gpu_index" "$gpu_uuid" "$mode" "$used" "$total" "$util" \
    "$container_pid" "$quoted_processes" "$foreign" >> "$audit_csv"
  sleep 1
done

exit_code=$(docker inspect --format '{{.State.ExitCode}}' "$container_name")
docker logs "$container_name" > "$log_path" 2>&1
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  printf 'status=%s\n' "$([[ $exit_code -eq 0 ]] && printf success || printf failed)"
  printf 'container_exit_code=%s\n' "$exit_code"
  printf 'foreign_process_detected=%s\n' "$contaminated"
  printf 'gpu_index=%s\n' "$gpu_index"
  printf 'gpu_uuid=%s\n' "$gpu_uuid"
  printf 'compute_mode=%s\n' "$compute_mode"
  printf 'started_utc=%s\n' "$started_utc"
  printf 'finished_utc=%s\n' "$finished_utc"
  printf 'audit_csv=%s\n' "$audit_csv"
  printf 'benchmark_log=%s\n' "$log_path"
} > "$audit_summary"

printf '\nBenchmark finished with container exit code %s.\n' "$exit_code"
printf 'Foreign process detected: %s\n' "$contaminated"
printf 'Audit summary: %s\n' "$audit_summary"
printf 'Benchmark result line:\n'
rg 'BENCH_RESULT' "$log_path" || true

if [[ $contaminated -ne 0 ]]; then
  die "GPU audit detected a foreign compute PID; do not use this run as an exclusive result"
fi
exit "$exit_code"
