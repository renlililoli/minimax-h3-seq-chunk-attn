# ComfyUI Community Node Agent Guide

This branch is the release source for the **MiniMax H3 SeqAttn** ComfyUI
community node. It is a consumer of the independent `seqattn-core` package; it
must not become a second implementation of the core runtime.

## Repository Role

The branch is `community/comfyui-minimax-h3-seqattn` and is the default branch
of `renlililoli/minimax-h3-seq-chunk-attn`. This worktree owns:

- the ComfyUI V3 extension entrypoint and nodes;
- MiniMax-H3 model patching and framework callbacks;
- Dynamic VBAR current-block plus next-block weight scheduling;
- Qwen BF16 conditioning/preflight and refined-conditioning caching;
- streamed MiniMax-H3 video VAE encode/decode;
- T2VA, first/last-frame, FL2VA, and Ref2VA workflows;
- the RTX 50-series image;
- GitHub Release and Comfy Registry publication.

The independent core lives at `renlililoli/stream-attn`. Changes to kernels,
planners, generic projected attention, paged storage, or H3 runners belong
there first. Consume a reviewed immutable core commit through
`seqattn-core[dit]`.

As of August 28, 2026, this branch is version `0.4.3`, pins ComfyUI `0.30.0` at
commit `9a9fdb10ed144ce760d9682cb247526ea23cc525`, and pins SeqAttn commit
`5a52f7ea8e83d9187ed39d03e66eccc305eaaaf3`. Verify current metadata before a
release rather than assuming these values remain current.

## Start Safely

```bash
git status --short --branch
git log -5 --oneline --decorate
git tag --sort=-version:refname | head
```

Do not reset unrelated edits or generated evidence. Commit only files for the
current task. Do not develop community releases from the root integration
feature branch; that worktree may use vendored experimental code and has a
different purpose.

## Architecture Contracts

### SeqAttn ownership

Import the released `seqattn_core` package directly. Do not add a vendored
`seqattn_core` copy to this branch.

The core owns bounded attention/DiT execution. This adapter supplies
`H3BlockOps` callbacks for ComfyUI model operations and owns framework-specific
weight preparation and cleanup.

### Weight schedule

The streaming DiT path owns the complete 50-block schedule:

```text
prepare block 0
for block N:
    prepare block N+1 asynchronously
    wait block N ready
    compute block N
    wait for compute completion
    unpin and evict block N
release remaining staged weights
```

At most two blocks may be staged. The streaming path must not call
`comfy.model_prefetch.make_prefetch_queue()` or
`comfy.model_prefetch.prefetch_queue_pop()`. Tests enforce this boundary.

Reuse Dynamic VBAR as the checkpoint's lazy weight representation, but do not
give ComfyUI ownership of the streaming block order. Avoid instrumentation such
as `state_dict()` or `module_size()` on Dynamic VBAR models because it can
materialize lazy weight pages and invalidate memory measurements.

### Activation placement

- Full hidden state and complete Q/K/V are pinned CPU tensors between phases.
- QKV projection, attention, output projection, residual updates, and MLP work
  execute in bounded GPU tiles.
- Only completed block hidden tiles return to the host.
- The implementation does not impose a hidden whole-process VRAM fraction or
  silently reduce configured chunks.
- The aggressive hidden-to-QKV recompute design is deferred. Do not implement
  it without an explicit design and benchmark task.

### Configuration ownership

Workflow/node inputs:

- `q_chunk_tokens`: HBM-resident Q super-block;
- `kv_chunk_tokens`: streamed K/V tile.

Shared SeqAttn TOML configuration:

```toml
[attention]
backend = "auto"

[minimax_h3]
execution_mode = "materialized" # or "recompute"
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096
```

The file is selected through `SEQATTN_CONFIG` or the default user config path.
Execution mode is deployment configuration, never a node input or workflow
widget. Qwen remains materialized-only.
Do not reintroduce an attention `workspace` UI concept. The VAE node has a
separate `workspace_mib` input and is not part of this rule.

Q chunk calibration is topology-specific. Preserve the GPU, backend, CPU set,
NUMA memory policy, and measured concurrent pinned-memory bandwidth. Do not
replace a measured Q chunk with a value derived only from advertised PCIe or
GPU peak numbers.

### AIMDO import order

Normal Web UI startup must go through the pinned ComfyUI `main.py`, which
initializes `comfy_aimdo.control` before Torch. Custom Python launchers must
initialize AIMDO before importing `torch`, `nodes`, `comfy.model_patcher`, or
`comfy_aimdo.host_buffer` consumers.

### Compatibility

The entrypoint deliberately rejects ComfyUI versions/commits outside the pinned
compatibility contract. Do not widen the range based only on import success.
To support a new ComfyUI release, inspect its MiniMax-H3 sampling, DynamicVRAM,
model patcher, VAE, and node contracts, then run the bundled workflows
end-to-end from a fresh installation.

## Development Workflow

1. Identify whether the change is core, adapter, workflow, Docker, or release
   metadata. Move core work to the SeqAttn repository.
2. Add focused tests beside the affected module.
3. Keep workflows and test assertions synchronized when node inputs/defaults
   change.
4. For operator changes, run synthetic/single-block parity before end-to-end
   denoising.
5. For user-facing behavior, test a fresh installation rather than relying on
   dependencies already present in a development image.
6. Preserve UI workflow runs with matching package and environment metadata
   before citing them in summary tables or README claims.
7. Preserve failures and explain invalid runs rather than replacing them with
   undocumented reruns.

Do not run a full pipeline when the requested question is limited to a
synthetic block. Conversely, do not claim user installation works based only
on unit tests or an editable checkout.

## Checks

CI installs the exact ComfyUI commit and runs:

```bash
ruff check __init__.py comfyui_seqattn tests
python -m build
pytest -q
```

The release-equivalent local check uses the pinned Docker checks target:

```bash
docker build --file docker/Dockerfile --target checks \
  --tag minimax-h3-seqattn:checks .
docker run --rm --gpus device=0 --ipc=host \
  --volume "$PWD:/workspace:ro" \
  minimax-h3-seqattn:checks
```

The checks target runs Ruff, pytest, whitespace validation, and builds the
source/wheel from a clean staged tree. Do not publish if CI or this check fails.

## End-to-End Validation

The supported validation scenarios are:

- T2VA;
- FL2VA;
- Ref2VA with images;
- Ref2VA with video.

Validate each by opening the bundled workflow in the pinned ComfyUI Web UI and
running it end-to-end from a fresh installation. The workflow files live in
`workflows/`; the long Ref2VA case additionally documents its host-memory and
NUMA requirements in `README.md`.

Preserve a validated run as a result record with matching package and
environment metadata before citing it. A release-level performance claim
requires a documented longer run with exact hardware, process memory,
steady-state timing, and artifact provenance; a two-step UI run is a functional
check, not a throughput claim.

## Docker and Deployment Configuration

`docker/Dockerfile` contains build-time compatibility pins: base-image digest,
ComfyUI version/commit, PyTorch/CUDA, AIMDO, and SeqAttn commit.

`docker/.env` contains deployment variables such as GPU, ports, mounts,
DynamicVRAM switches, allocator settings, caches, and diagnostics.

`docker/seqattn.toml` contains structured SeqAttn settings. Do not move
build-time compatibility pins into runtime configuration, and do not bake model
weights into the image.

Keep `SEQATTN_CORE_COMMIT` below the fixed Torch and ComfyUI layers, and install
the core before copying node source. This preserves the expensive framework
cache when only the core pin or adapter source changes.

## Version and Dependency Updates

For every node version change, keep these synchronized:

- `pyproject.toml` project version;
- `comfyui_seqattn/__init__.py` `__version__`;
- `CHANGELOG.md`;
- tests that assert node/core versions;
- README version labels when new artifacts were generated.

For every SeqAttn pin change, update the same immutable commit everywhere:

- dependency URL in `pyproject.toml`;
- `SEQATTN_CORE_COMMIT` in `docker/Dockerfile`;
- `THIRD_PARTY_NOTICES.md`;
- Docker and installation documentation;
- version/pin tests.

Do not pin the community package to a moving branch. Release or otherwise
identify the core commit first, then use the full SHA.

## Community Release Procedure

1. Confirm the intended SeqAttn commit is pushed and installable with `[dit]`.
2. Synchronize versions, dependency pins, changelog, compatibility pins, tests,
   and current documentation.
3. Run focused tests, Docker checks, and the UI workflow end-to-end checks
   required by the change.
4. Commit and push `community/comfyui-minimax-h3-seqattn`.
5. Wait for the GitHub CI run for that exact commit to succeed.
6. Create an annotated tag at the tested commit and push it:

   ```bash
   git tag -a v0.4.3 -m 'MiniMax H3 SeqAttn 0.4.3'
   git push publish v0.4.3
   ```

7. Write a non-empty GitHub Release body. Include user-visible changes,
   compatibility changes, and the validated configuration; do not paste stale
   benchmark numbers.
8. Publish the GitHub Release:

   ```bash
   gh release create v0.4.3 \
     --repo renlililoli/minimax-h3-seq-chunk-attn \
     --verify-tag \
     --title 'MiniMax H3 SeqAttn 0.4.3' \
     --notes-file /path/to/release-notes.md
   ```

The Registry workflow runs on `release.published`, not on tag push. It reads
the GitHub Release body and passes it to `comfy node publish` as the version
changelog. A tag alone is not a Registry release.

Registry metadata rules:

- `[project].description` is the canonical node introduction; every publish
  sends it to Registry, so do not replace it with an unrelated short summary.
- The GitHub Release body is the Registry version's Updates content.
- An empty Release body must block publication.
- If an already published Release body is edited, manually dispatch
  `.github/workflows/publish.yml` with its tag to resync the existing Registry
  changelog. Manual dispatch updates changelog only; it does not republish the
  package archive.

After publishing, verify:

```bash
gh run list --workflow publish.yml --limit 5
curl -sS https://api.comfy.org/nodes/minimax-h3-seqattn/versions/X.Y.Z
```

Confirm the tag target, GitHub Release state, Registry version, dependency URL,
download URL, changelog, node description, and exact ComfyUI compatibility.
Report Registry security status separately from upload success.

Never move a published tag. If package contents change, increment the version
and publish a new release.

## Completion Report

When finishing work, identify the branch and commit, list checks that actually
ran, name any skipped validation, and distinguish source changes from new
generated evidence. For a release, include the tag, target commit, GitHub
Release result, Registry workflow run, and Registry metadata verification.
