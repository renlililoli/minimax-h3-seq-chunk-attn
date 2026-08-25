#!/usr/bin/env bash
set -euo pipefail

args=(
  python
  main.py
  --listen "${COMFYUI_LISTEN:-0.0.0.0}"
  --port "${COMFYUI_PORT:-8188}"
)

if [[ -n "${COMFYUI_RESERVE_VRAM_GIB:-}" ]]; then
  args+=(--reserve-vram "$COMFYUI_RESERVE_VRAM_GIB")
fi
if [[ -n "${COMFYUI_VRAM_HEADROOM_GIB:-}" ]]; then
  args+=(--vram-headroom "$COMFYUI_VRAM_HEADROOM_GIB")
fi
if [[ "${COMFYUI_ENABLE_DYNAMIC_VRAM:-1}" == "1" ]]; then
  args+=(--enable-dynamic-vram)
else
  args+=(--disable-dynamic-vram)
fi
if [[ "${COMFYUI_DISABLE_NVML_PRESSURE:-0}" == "1" ]]; then
  args+=(--disable-nvml-pressure)
fi
if [[ "${COMFYUI_DISABLE_ASYNC_OFFLOAD:-0}" == "1" ]]; then
  args+=(--disable-async-offload)
elif [[ -n "${COMFYUI_ASYNC_OFFLOAD_STREAMS:-}" ]]; then
  args+=(--async-offload "$COMFYUI_ASYNC_OFFLOAD_STREAMS")
fi
if [[ -n "${COMFYUI_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$COMFYUI_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

exec "${args[@]}"
