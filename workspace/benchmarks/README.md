# MiniMax-H3 original-path benchmark

This benchmark exercises the unmodified full-sequence MiniMax-H3 DiT path in
DiffSynth-Studio. It can independently select CPU/DRAM or disk weight offload
and can apply a hard PyTorch allocator cap to emulate a smaller GPU.

## 8 GiB, CPU/DRAM offload, DiT-only result

Run date: 2026-08-17 UTC

- DiffSynth-Studio commit: `6343deda06b3e09efc9b1ce23c135c35a341d143`
- Model: `MiniMax-H3-NF4`, FL2VA
- Physical GPU: RTX 5090 32 GB; CUDA allocator hard-capped to 8 GiB
- Weight backing store: CPU DRAM (not disk)
- DiffSynth layer residency limit: 4 GiB
- Activation/headroom reserve: 4 GiB
- Shape: 480x832, 124 frames
- Seed: 0
- Denoising: 5 steps
- Text encoding: completed before the measured denoise interval
- Video/audio decode: skipped with no-op outputs

Results:

| Metric | Value |
|---|---:|
| Denoise total | 44.767 s |
| Mean per step | 8.953 s |
| Min / median / max | 8.613 / 9.057 / 9.170 s |
| Denoise peak allocated VRAM | 5,278 MiB |
| Denoise peak reserved VRAM | 6,276 MiB |
| Whole-process `nvidia-smi` peak | 6,908 MiB |
| Process peak RSS | 34,656 MiB |

Per-step times: `9.1266, 8.8013, 9.1697, 8.6131, 9.0568` seconds.

Reproduction command:

```bash
docker compose exec -T --user 1091:1102 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  diffsynth python /workspace/benchmarks/minimax_h3_baseline.py \
  --height 480 --width 832 --frames 124 --steps 5 --seed 0 \
  --tag sim8_cpu_dit5 \
  --simulated-vram-gib 8 --vram-reserve-gib 4 \
  --offload-device cpu --skip-decode --no-media
```

Structured output:

- `results/sim8_cpu_dit5_480x832_f124_s5_20260817T034204Z.json`
- `results/sim8_cpu_dit5_gpu.csv`

## Notes

The standard NF4 inference example uses CPU/DRAM offload. The separately named
`model_inference_low_vram` example uses disk offload. This benchmark defaults to
CPU offload and exposes `--offload-device disk` only for explicit comparisons.

With an 8 GiB cap and a 6 GiB layer-residency limit, the original full-sequence
configuration OOMed in the MLP at `silu(gate) * up`. Reducing layer residency to
4 GiB left sufficient room for full-sequence activations and completed normally.
# MiniMax-H3 sequence-streaming benchmarks

正式协议代码位于 `minimax_h3_bench/`。核心约束是：每个点独立进程、NVML-aware
allocator、CPU/DRAM weight offload、原子 JSON，以及只从 JSON 聚合结果。
当前 PID 显存由持久 `pynvml` handle 采样；只有 NVML 不可用时才回退到
`nvidia-smi`。

canonical artifact 可在容器内生成。正式数据点的 controller 必须在仓库宿主机运行，
由宿主机读取两个 git commit，并通过 `docker compose exec -e` 注入容器：

```bash
python -m benchmarks.minimax_h3_bench.canonical \
  --frames 22 39 56 73 90 124 158 192 226 260 \
  --output /workspace/benchmarks/artifacts/canonical_480x832_t256.pt

python3 -m workspace.benchmarks.minimax_h3_bench.run_point \
  --output workspace/benchmarks/results/attention_streaming_3072.json \
  --container-output /workspace/benchmarks/results/attention_streaming_3072.json \
  --timeout-seconds 1800 -- \
  docker compose exec -T \
  -e PYTHONPATH=/opt/DiffSynth-Studio:/workspace \
  diffsynth numactl --physcpubind=64-95,320-351 --membind=3 \
  python -m benchmarks.minimax_h3_bench.attention_point \
  --mode streaming --tokens 3072 --used-tokens 3060 --target-vram-mib 4096
```

不要在容器内运行 controller：容器只挂载代码/结果目录，不挂载 `.git`，因此容器内
的 `git rev-parse` 不是实验 provenance 的可信来源。

本节点用 `numa_copy_probe.py` 实测后锁定 memory node 3。NUMA probe 只用于确定
环境配置，不属于模型主结果。

`attention_correctness.py` 强制 Torch SDPA math backend。`select_chunks.py` 实现
3968MiB 过滤、1% tie window 和大 tile 优先规则，但 winner 仍必须独立复测。

`minimax_h3_baseline.py` 是早期 pipeline smoke/validation driver，历史 JSON 保留用于
开发记录；它的输出不能与正式协议 JSON 混合聚合。
