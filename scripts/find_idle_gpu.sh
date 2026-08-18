#!/usr/bin/env bash
set -euo pipefail

command -v nvidia-smi >/dev/null 2>&1 || {
  printf 'error: nvidia-smi is not available\n' >&2
  exit 1
}

declare -A active
while IFS=', ' read -r uuid pid process_name used_mib; do
  [[ -z ${uuid:-} ]] && continue
  active["$uuid"]+=" pid=$pid process=$process_name memory=${used_mib}MiB;"
done < <(nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)

printf '%-5s %-38s %-19s %-12s %-12s %s\n' \
  INDEX UUID MODE MEMORY UTILIZATION STATUS

idle_indices=()
while IFS=', ' read -r index uuid mode used total utilization; do
  [[ -z ${index:-} ]] && continue
  process_text=${active[$uuid]:-}
  if [[ -z $process_text ]]; then
    status=IDLE
    idle_indices+=("$index")
  else
    status="ACTIVE:$process_text"
  fi
  printf '%-5s %-38s %-19s %5s/%-6s %-12s %s\n' \
    "$index" "$uuid" "$mode" "$used" "$total" "$utilization" "$status"
done < <(nvidia-smi \
  --query-gpu=index,uuid,compute_mode,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits)

printf '\nIdle candidates are a point-in-time observation, not a reservation.\n'
if [[ ${#idle_indices[@]} -eq 0 ]]; then
  printf 'No idle GPU is available.\n'
  exit 3
fi

printf 'Ask an administrator to reserve one candidate, for example:\n'
for index in "${idle_indices[@]}"; do
  printf '  sudo scripts/admin_gpu_exclusive.sh enable %q\n' "$index"
done
printf 'Then launch immediately with:\n'
printf '  scripts/run_native_h3_exclusive.sh --gpu GPU_INDEX\n'
