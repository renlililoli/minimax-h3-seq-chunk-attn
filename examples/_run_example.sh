#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <scenario> [runner arguments...]" >&2
  exit 2
fi

scenario=$1
shift

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
package_root=$(cd "$example_dir/.." && pwd)

if [[ -n "${COMFYUI_DIR:-}" ]]; then
  comfyui_dir=$(cd "$COMFYUI_DIR" && pwd)
elif [[ $(basename "$(dirname "$package_root")") == custom_nodes ]]; then
  comfyui_dir=$(cd "$package_root/../.." && pwd)
elif [[ -d /opt/ComfyUI ]]; then
  comfyui_dir=/opt/ComfyUI
else
  echo "set COMFYUI_DIR to the ComfyUI installation directory" >&2
  exit 2
fi

output_root=${SEQATTN_EXAMPLE_OUTPUT_DIR:-$example_dir/results}
output_dir=$output_root/$scenario
mkdir -p "$output_dir"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-0}

exec "${PYTHON:-python3}" "$example_dir/run_2step_example.py" \
  --scenario "$scenario" \
  --comfyui-dir "$comfyui_dir" \
  --output-dir "$output_dir" \
  "$@"
