# Changelog

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
