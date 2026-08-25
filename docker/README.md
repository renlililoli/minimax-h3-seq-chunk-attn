# RTX 50-Series ComfyUI Image

This image extends the exact ComfyUI environment used for the checked-in
`0.4.1` examples:

- ComfyUI `0.30.0`, commit `9a9fdb10ed144ce760d9682cb247526ea23cc525`
- PyTorch `2.10.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- `comfy-aimdo` `0.4.11`
- `seqattn-core` commit `86049c058a4dfb26da408e79ca2c95677ebbd250`

The default base is pinned by digest rather than by a floating tag. The build
also checks every version above before installing this node, so a mismatched
base image fails immediately.

## Build

Copy the environment template and edit the absolute host paths, GPU, and port:

```bash
cp docker/.env.example docker/.env
```

Runtime environment variables are kept in `docker/.env`; structured SeqAttn
settings are kept in `docker/seqattn.toml`. Build-time version and commit pins
remain in the Dockerfile because changing them creates a different supported
software image rather than a runtime configuration.

Build through Compose from the repository root:

```bash
docker compose --env-file docker/.env --file docker/compose.yaml build
```

The equivalent direct build is:

```bash
docker build \
  --file docker/Dockerfile \
  --tag minimax-h3-seqattn:comfyui-0.30.0-cu128-rtx50 \
  .
```

The default `FROM` reference is the locally validated
`comfyui@sha256:4708ab49a718640950f5cd698172d4800718d3b62e961f79d20866c115a8cff5`.
To use a mirrored copy of the same base image, override the reference while
retaining the build-time version checks:

```bash
docker build \
  --file docker/Dockerfile \
  --build-arg COMFYUI_BASE_IMAGE=registry.example/comfyui@sha256:... \
  --tag minimax-h3-seqattn:comfyui-0.30.0-cu128-rtx50 \
  .
```

Model weights are intentionally not copied into the image.

## Configuration Boundaries

`docker/.env` contains the deployment-level settings:

- image name and pinned base image reference;
- GPU selection, host port, and bind-mount paths;
- ComfyUI DynamicVRAM, NVML pressure, async offload, reserve, and headroom;
- CUDA module loading, PyTorch allocator, libc allocator, cache, and NVTX
  switches.

`docker/seqattn.toml` contains the structured runtime settings:

```toml
[attention]
backend = "auto"

[minimax_h3]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096
```

`q_chunk_tokens` and `kv_chunk_tokens` remain inputs on the ComfyUI SeqAttn
node, so they are serialized with the workflow. The RTX 5090 single-NUMA
default is Q `5760` and K/V `4096`; do not change Q solely because the Docker
image changed.

The empty `COMFYUI_RESERVE_VRAM_GIB` and `COMFYUI_VRAM_HEADROOM_GIB` values are
intentional. They leave whole-process memory policy to the pinned ComfyUI
runtime. Set them only when the deployment must reserve memory for another
process.

## Verify RTX 50-Series Access

```bash
docker run --rm \
  --gpus device=0 \
  --ipc=host \
  minimax-h3-seqattn:comfyui-0.30.0-cu128-rtx50 \
  python -c 'import torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())'
```

An RTX 5090 reports compute capability `(12, 0)`.

## Run Repository Checks

Build the separate `checks` target. It extends the exact runtime image with a
pinned Ruff binary; pytest and build already come from the fixed base image.
The production image does not contain the additional lint tool.

```bash
docker build \
  --file docker/Dockerfile \
  --target checks \
  --tag minimax-h3-seqattn:checks \
  .
```

Run all static checks, tests, whitespace checks, and package builds against the
current worktree mounted read-only:

```bash
docker run --rm \
  --gpus device=0 \
  --ipc=host \
  --volume "$PWD:/workspace:ro" \
  minimax-h3-seqattn:checks
```

The script disables Ruff and pytest repository caches and writes package
artifacts only to a temporary directory inside the disposable container.

## Run ComfyUI

With Compose:

```bash
docker compose --env-file docker/.env --file docker/compose.yaml up
```

Or directly:

```bash
docker run --rm \
  --gpus device=0 \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --publish 8188:8188 \
  --volume /path/to/models:/opt/ComfyUI/models \
  --volume /path/to/input:/opt/ComfyUI/input \
  --volume /path/to/output:/opt/ComfyUI/output \
  --volume /path/to/user:/opt/ComfyUI/user \
  minimax-h3-seqattn:comfyui-0.30.0-cu128-rtx50
```

Open `http://localhost:8188`. Starting through the image's default `main.py`
command preserves ComfyUI's required AIMDO-before-PyTorch initialization order.

On multi-socket hosts, add `--cpuset-cpus` and `--cpuset-mems` for the CPU and
memory node local to the selected GPU. Keep the model mount read-only if the
workflow does not download or modify weights.
