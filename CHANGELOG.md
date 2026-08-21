# Changelog

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
