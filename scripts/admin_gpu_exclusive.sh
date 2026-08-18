#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo scripts/admin_gpu_exclusive.sh enable GPU_INDEX
  sudo scripts/admin_gpu_exclusive.sh disable GPU_INDEX
  scripts/admin_gpu_exclusive.sh status GPU_INDEX

enable refuses to modify a GPU that already has a compute process. It changes
the NVIDIA compute mode to EXCLUSIVE_PROCESS, which permits only one CUDA
context on that physical GPU.

disable restores the normal DEFAULT compute mode. It also refuses to run while
a compute process is active.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 2 ]] || { usage >&2; exit 2; }
action=$1
gpu_index=$2
[[ $gpu_index =~ ^[0-9]+$ ]] || die "GPU_INDEX must be a non-negative integer"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available"

gpu_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null) \
  || die "GPU $gpu_index does not exist or is not visible"

compute_processes() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid { print }'
}

show_status() {
  nvidia-smi -i "$gpu_index" \
    --query-gpu=index,uuid,name,compute_mode,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  local processes
  processes=$(compute_processes)
  if [[ -n $processes ]]; then
    printf 'compute processes:\n%s\n' "$processes"
  else
    printf 'compute processes: none\n'
  fi
}

require_idle() {
  local processes
  processes=$(compute_processes)
  if [[ -n $processes ]]; then
    printf 'GPU %s is not idle:\n%s\n' "$gpu_index" "$processes" >&2
    die "refusing to change compute mode on an active GPU"
  fi
}

case "$action" in
  status)
    show_status
    ;;
  enable)
    [[ ${EUID:-$(id -u)} -eq 0 ]] || die "enable requires root; run with sudo"
    require_idle
    nvidia-smi -i "$gpu_index" --compute-mode=EXCLUSIVE_PROCESS
    mode=$(nvidia-smi -i "$gpu_index" --query-gpu=compute_mode --format=csv,noheader,nounits)
    [[ $mode == "Exclusive Process" || $mode == "Exclusive_Process" || $mode == "EXCLUSIVE_PROCESS" ]] \
      || die "driver did not enter EXCLUSIVE_PROCESS mode (reported: $mode)"
    printf '\nGPU %s is reserved for one CUDA process.\n' "$gpu_index"
    show_status
    printf '\nRestore after the experiment with:\n'
    printf '  sudo %q disable %q\n' "$0" "$gpu_index"
    ;;
  disable)
    [[ ${EUID:-$(id -u)} -eq 0 ]] || die "disable requires root; run with sudo"
    require_idle
    nvidia-smi -i "$gpu_index" --compute-mode=DEFAULT
    show_status
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
