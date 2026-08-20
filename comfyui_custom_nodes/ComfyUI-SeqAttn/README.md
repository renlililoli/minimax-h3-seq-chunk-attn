# ComfyUI-SeqAttn

Native ComfyUI MiniMax-H3 integration for exact CPU-backed SeqAttn.

## Scope

- ComfyUI `MiniMaxH3Model` only
- Comfy-Org pruned BF16 or ComfyUI-native quantized DiT checkpoints
- ComfyUI NVFP4/INT8 text encoders and INT8 ConvRot H3 DiT weights
- CUDA, Linux, batch size 1
- T2VA, FL2VA, and Ref2VA packed layouts
- Exact dense attention with pinned CPU activation/QKV backing

The attention activation path remains BF16 regardless of weight storage. The
already-downloaded ComfyUI set is supported: NVFP4-AWQ Qwen3-VL text encoder,
INT8 ConvRot H3 DiT, and FP16 video VAE.

LoRA, DiT replacement patches, NVMe activation backing, and multi-GPU
execution are rejected or outside the initial scope.

Insert **MiniMax H3 SeqAttn** immediately after the diffusion model loader and
connect its `MODEL` output to the sampler. `activation_workspace_mib` is the
SeqAttn-owned CUDA workspace budget; it is not a whole-process VRAM limit.

## Container

```bash
docker compose -f docker-compose.comfyui.yml up --build
```

Set `COMFYUI_MODELS_DIR` to an existing ComfyUI models directory before
starting the service. The UI listens on port 8188.

For the already-downloaded MiniMax-H3 model set on this machine:

```bash
COMFYUI_MODELS_DIR=/scratch/grzhu/weights/video/minimax/merged \
docker compose -f docker-compose.comfyui.yml up --build
```

That directory contains the Ref2VA/FL2VA INT8 ConvRot DiTs, the NVFP4-AWQ
text encoder, and both MiniMax-H3 VAEs expected by the bundled workflow.
