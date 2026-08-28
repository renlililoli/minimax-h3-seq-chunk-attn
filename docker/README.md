# RTX 50-Series ComfyUI Image

This image reconstructs the pinned ComfyUI compatibility stack used for the
`0.4.x` validated runs from a publicly pullable NVIDIA NGC base:

- NVIDIA PyTorch `26.01-py3`, pinned by immutable multi-architecture digest
- ComfyUI `0.30.0`, commit `9a9fdb10ed144ce760d9682cb247526ea23cc525`
- PyTorch `2.10.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- `comfy-aimdo` `0.4.11`
- ComfyUI-Manager commit `d47c9346190397e1c316bc5a82155faaf9f5d700`
- `seqattn-core` commit `5a52f7ea8e83d9187ed39d03e66eccc305eaaaf3`

The default base is
`nvcr.io/nvidia/pytorch@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e`,
the public `26.01-py3` image pinned by digest rather than by a floating tag.
The Dockerfile replaces its preview Torch build with the validated stable
CUDA 12.8 wheels, checks out the exact ComfyUI and Manager commits, and checks
the resulting versions before installing this node.

## Build

Copy the environment template and edit the absolute host paths, GPU, and port:

```bash
cp docker/.env.example docker/.env
id -u
id -g
```

Set `HOST_UID` and `HOST_GID` in `docker/.env` to the two values printed above.
The container runs as that user so writable bind mounts also work on NFS homes
with root squashing. Create the writable directories before the first start;
Docker should not create them as root on the user's behalf:

```bash
mkdir -p /path/to/ComfyUI/input /path/to/ComfyUI/output /path/to/ComfyUI/user
```

Runtime environment variables are kept in `docker/.env`; structured SeqAttn
settings are kept in `docker/seqattn.toml`. Build-time version and commit pins
remain in the Dockerfile because changing them creates a different supported
software image rather than a runtime configuration.

The SeqAttn pin is declared only after the fixed Torch and ComfyUI stack has
been verified, and core installation happens before node source is copied.
Changing the core pin therefore preserves the framework cache; changing node
source preserves the core layer. The repository `.dockerignore` also excludes
local worktrees, generated output, agent state, and deployment secrets.

Build through Compose from the repository root:

```bash
docker compose --env-file docker/.env --file docker/compose.yaml build
```

The equivalent direct build is:

```bash
docker build \
  --file docker/Dockerfile \
  --target runtime \
  --tag minimax-h3-seqattn:comfyui-0.30.0-cu128-rtx50 \
  .
```

The default `FROM` reference can be pulled anonymously from NVIDIA NGC. To use
a mirrored copy of the same NGC image, override the reference while retaining
the build-time version checks:

```bash
docker build \
  --file docker/Dockerfile \
  --target runtime \
  --build-arg PYTORCH_BASE_IMAGE=registry.example/nvidia-pytorch@sha256:... \
  --tag minimax-h3-seqattn:comfyui-0.30.0-cu128-rtx50 \
  .
```

Public pull access does not replace the NVIDIA container license terms. Review
the NVIDIA Software License Agreement and NVIDIA AI Product-Specific Terms
listed in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before
redistributing the built image.

Model weights are intentionally not copied into the image.

## Configuration Boundaries

`docker/.env` contains the deployment-level settings:

- image name and pinned public NVIDIA PyTorch base reference;
- GPU selection, host UID/GID, host port, and bind-mount paths;
- ComfyUI DynamicVRAM, NVML pressure, async offload, reserve, and headroom;
- CUDA module loading, PyTorch allocator, libc allocator, cache, and NVTX
  switches.

`docker/seqattn.toml` contains the structured runtime settings:

```toml
[attention]
backend = "auto"

[minimax_h3]
execution_mode = "materialized" # or "recompute"
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096

[minimax_h3_qwen]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096

[minimax_h3_vae]
tile_size = 192
workspace_mib = 512
```

Connecting a Qwen, DiT, or video VAE patch node selects streaming for that
stage; wiring around it keeps the native implementation. The Qwen and DiT
nodes each expose their own Q/KV chunks, while the MiniMax-H3 execution mode,
projection/MLP tiles, and VAE settings come from this TOML file. Qwen remains
materialized-only. The shipped Qwen values currently reuse the RTX 5090 DiT
values and are not presented as an independently calibrated Qwen optimum.

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

Build the separate `checks` target. It extends the exact runtime image with
pinned Ruff, pytest, and build packages. The production image does not add the
lint tool.

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
docker compose --env-file docker/.env --file docker/compose.yaml up --build
```

The `--build` flag makes a source checkout pick up local node changes instead
of silently reusing an older image with the same tag.

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
