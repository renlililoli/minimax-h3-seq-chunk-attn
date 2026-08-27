# Changelog

## 0.4.2 - 2026-08-27

- Add a separate MiniMax-H3 Qwen SeqAttn node with CPU-backed vision and
  decoder hidden/Q/K/V, packed non-causal vision attention, causal decoder GQA,
  tiled vision mergers, and CPU DeepStack injection.
- Generalize the DiT current-plus-next Dynamic VBAR weight pipeline to support
  Qwen vision and decoder module-group stages.
- Materialize first-use loaded-weight host pins before VBAR prefetch so cold-start
  Qwen, vision, and DiT stages cannot execute with incomplete GPU weights.
- Gather and dequantize only unique decoder token-embedding rows on the CPU, and
  reject non-finite or all-zero conditioning before ComfyUI can cache it.
- Remove the old activation-estimate-based Qwen BF16 offload node and switch
  bundled workflows to the single streaming encode path.
- Make Qwen, DiT, and video VAE streaming independent patch nodes. Connecting
  a node selects streaming for that stage; bypassing it keeps the native
  ComfyUI implementation.
- Keep Q/KV chunks explicit and independent on the Qwen and DiT nodes; load
  projection/MLP tiles and VAE tile/workspace settings from the shared SeqAttn
  TOML configuration.
- Build the Compose service from the production `runtime` stage and document
  `up --build`, preventing stale or checks-stage images from starting as
  ComfyUI.
- Run the Compose container as the configured host UID/GID so writable input,
  output, user, and cache mounts work on NFS homes with root squashing.
- Keep `0.4.0` and `0.4.1` measurements explicitly historical instead of
  applying them to the new Qwen SeqAttn implementation, and document current
  8 GiB, recommended 12 GiB, and typical 64 GiB system-memory guidance.
- Exclude local agent sessions, archived worktrees, and generated output state
  from clean release staging.
- Remove the bundled command-line examples and their checked-in results; UI
  workflows are the supported end-to-end validation path.

## 0.4.1 - 2026-08-25

- Pin the supported ComfyUI runtime to version `0.30.0`, exact validation
  commit `9a9fdb10ed144ce760d9682cb247526ea23cc525`, across package metadata,
  CI, documentation, and end-to-end examples.
- Avoid importing the removed `time_shift_slope` helper unconditionally and
  preserve both MiniMax-H3 audio velocity contracts for future compatibility
  work, without claiming support beyond the pinned ComfyUI baseline.
- Regenerate the T2VA, FL2VA, image-reference Ref2VA, and video-reference
  Ref2VA examples from clean installs, including media, memory traces, and
  fused weight-scheduler traces.

## 0.4.0 - 2026-08-25

- Replace the old split MiniMax-H3 denoise path with the fused block runtime
  published by `seqattn-core[dit]` `0.3.0a3`.
- Keep one pinned CPU hidden tensor between blocks while attention output,
  output projection, residual updates, and the complete MLP remain on GPU in
  bounded tiles.
- Replace ComfyUI's DiT prefetch queue with a strict current-block plus
  next-block VBAR pipeline owned by the SeqAttn integration.
- Replace workspace/planner node controls with explicit Q and K/V chunk sizes;
  load QKV and MLP tile sizes from the shared SeqAttn TOML configuration.
- Preserve the community Qwen, VAE, Ref2VA, cache-lifetime, and example paths.
- Validate the complete 1344x768, 124-frame, 81,180-token Ref2VA pipeline for
  20 steps at a 7,708 MiB whole-process peak, with a flat 4,276 MiB denoise
  steady state and a strict two-block weight staging limit.

## 0.3.2 - 2026-08-24

- Replace the vendored SeqAttn runtime with a direct dependency on
  `seqattn-core` `0.3.0a2` from upstream tag `v0.3.0-alpha.2`.
- Pin the dependency to upstream commit `2641fdf` so community installations
  and development use one authoritative SeqAttn source revision.

## 0.3.1 - 2026-08-22

- Fix import ordering so the published CI static checks pass.
- Keep the package runtime version synchronized with project metadata.

## 0.3.0 - 2026-08-22

- Add a SeqAttn-aware Ref2VA conditioning node with Qwen preflight before
  reference VAE work and streaming VAE support for image/video references.
- Fix streamed VAE model loading when ComfyUI unloads the preceding Qwen model
  during inference-mode execution.
- Record upstream MiniMax-H3 refiner execution and SeqAttn compatibility-cache
  behavior across denoise forwards.
- Add one-click two-step T2VA, FL2VA, image-reference Ref2VA, and
  video-reference Ref2VA examples with real outputs and memory traces.
- Validate all four end-to-end examples at 1344x768 under an 8 GiB process
  target, including H.264/AAC media read-back checks.

## 0.2.0 - 2026-08-21

- Add MiniMax-H3 Qwen BF16 conditioning with input preflight and layer offload.
- Use dual-stream weight prefetch by default and retain synchronous extreme mode.
- Account for quadratic causal-mask and retained DeepStack memory before encode.
- Update the bundled Ref2VA workflow to patch both Qwen and DiT activations.

## 0.1.0 - 2026-08-20

- Add native ComfyUI MiniMax-H3 SeqAttn model patch node.
- Support T2VA, FL2VA, and Ref2VA packed layouts with BF16 activations.
- Support ComfyUI INT8 ConvRot DiT and NVFP4/INT8 text-encoder workflows.
- Vendor the exact CPU-backed attention runtime for Manager installations.
- Add an importable Ref2VA workflow.
