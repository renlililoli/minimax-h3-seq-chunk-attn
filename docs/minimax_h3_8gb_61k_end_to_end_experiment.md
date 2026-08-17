# MiniMax-H3 V0：8GB、61K Tokens 端到端视频生成实验说明

> 文档状态：正式 50-step 实验结果待确认
>
> 最近一次 5090 节点观测：2026-08-17 08:03 UTC（北京时间 16:03）
>
> 当前结论：在相同 8GB 整进程显存预算和 61,312-token 输入下，full-sequence
> baseline 在首个 DiT step 因 MLP activation OOM；Streaming V0 已完成完整
> 50-block 单步验证。正式 50-step 生成在最近一次 5090 节点观测时仍在运行，尚无
> 可确认的最终 VAE decode 和 MP4 结果。

## 1. 实验目的

本实验不是 attention 微基准，也不是缩减层数的工程 smoke test。实验使用真实
MiniMax-H3 NF4 checkpoint 和完整 FL2VA pipeline，构造普通 8GB GPU 无法通过
full-sequence DiT 完成、而 CPU-backed activation streaming 可以继续工作的长序列
视频生成案例。

实验需要回答：

1. 在严格的 8GB 整进程显存预算下，full-sequence baseline 是否因序列 activation
   OOM？
2. Streaming V0 是否能在相同 checkpoint、输入、seed 和权重 offload 策略下完成
   完整 50-block DiT forward？
3. Streaming 的实际 GPU 显存、CPU RAM 和延迟代价是多少？
4. Streaming 是否最终能完成 50 denoise steps、Video VAE decode、Audio VAE decode
   和 MP4 mux？
5. Text Encoder 和 VAE Decoder 为什么没有与 full-sequence DiT 相同的 activation
   OOM 行为？

预期能够支持的结论为：

> 在相同 8GB 整进程显存预算下，61,312-token full-sequence MiniMax-H3 DiT 因完整
> 序列 MLP activation OOM；V0 将 QKV、attention output 和 MLP intermediate 转移到
> CPU DRAM，并按 tile 在 GPU 上计算，从而完成相同输入的完整 DiT forward。端到端
> 成功仍以 50-step 任务完成真实 VAE decode 并生成 MP4 为准。

## 2. 实验工作流

### 2.1 当前是 FL2VA，不是 Ref2VA

本实验没有提供 reference image、reference video、reference audio 或 keyframe：

```text
references=None
keyframes=None
retake_video=None
retake_audio=None
```

因此使用 `MiniMaxH3Unit_PackedSequenceBuilder._build_packed_fl2va`：

```text
Text prompt
  → Text Encoder
  → 初始 video/audio noise latents
  → FL2VA packed sequence
  → 50-step joint video/audio DiT denoise
  → Video VAE decode
  → Audio VAE decode
  → MP4 + audio mux
```

没有 keyframe 时，packed layout 为：

```text
[text | target audio | target video | padding]
```

本实验不能作为 Ref2VA 已实测成功的证据。Ref2VA 还需单独覆盖 reference VAE
encoding、reference token 增长及 reference-target parity。

### 2.2 Prompt

```text
A girl is very happy, she is speaking in english:
“I enjoy working with Diffsynth-Studio, it's a perfect framework.”
```

### 2.3 完整模型路径

- MiniMax-H3 DiT：完整 50 blocks，不截层。
- Text Encoder：真实 NF4 Text Encoder。
- Video VAE：真实 NF4 Video VAE，启用空间 tiled 和 temporal chunk decode。
- Audio VAE：真实 NF4 Audio VAE。
- Denoise steps：50。
- CFG scale：默认 1.0，每个 step 只执行 positive forward。
- Seed：0。
- 最终输出：带音频 MP4。

## 3. 输入尺寸与真实 Packed Tokens

### 3.1 视频配置

```text
width:       832
height:      480
frames:      515
text length: 256
```

515 满足 pipeline 的合法帧数规则 `17n + 5`，无需额外 frame snapping。

### 3.2 Latent shape 与 token 公式

真实 pipeline 公式：

```text
video_latent_t = ((frames - 5) // 17) * 5 + 2
latent_h       = height // 16
latent_w       = width // 16
audio_latent_t = round(frames / 24 * 40)
```

代入本实验参数：

```text
video_latent_t = 152
latent_h       = 30
latent_w       = 52
audio_latent_t = 858
```

Video patch 后每个 temporal latent 的 token 数：

```text
(30 // 2) * (52 // 2) = 15 * 26 = 390
```

有效 token 数：

```text
text  = 256
audio = 858 * 2 = 1,716
video = 152 * 390 = 59,280
used  = 256 + 1,716 + 59,280 = 61,252
```

按 64-token 对齐：

```text
packed tokens = 61,312
cu_seqlens     = [0, 61,252, 61,312]
segments       = [61,252, 60]
```

这些数值由真实 `MiniMaxH3Unit_PackedSequenceBuilder` 生成，不是手工构造任意 hidden。
Canonical manifest：

```text
workspace/benchmarks/artifacts/canonical_480x832_f515.json
```

## 4. 硬件、软件与提交版本

### 4.1 硬件和 NUMA

```text
Physical GPU: NVIDIA GeForce RTX 5090, 32GB
Logical experiment budget: 8192MiB
GPU index: 0
CPU affinity: 64-95,320-351
DRAM NUMA node: 3
```

GPU 0 的 PCIe affinity 对应 CPU node 2，但该 node 在本机没有可分配 DRAM。通过
512MiB pinned-memory H2D/D2H probe，node 3 的带宽明显高于 node 1：

| Memory node | H2D | D2H |
|---|---:|---:|
| 1 | 15.09 GB/s | 14.53 GB/s |
| 3 | 36.02 GB/s | 32.06 GB/s |
| 1,3 | 36.12 GB/s | 32.67 GB/s |

正式 worker 使用：

```bash
numactl --physcpubind=64-95,320-351 --membind=3
```

### 4.2 软件与 commits

```text
Torch:        2.10.0+cu128
CUDA runtime: 12.8
bitsandbytes: 0.50.1
baseline attention: FlashAttention 2
streaming attention: PyTorch FP32 online softmax

Main repository:
9ee99a4f7d6eb8c19270953b543eb6aa7b115ddf

DiffSynth submodule:
2090652a25590daa4a09681c3ceb9a650b78de7a
```

## 5. 8GB 显存预算定义

不能简单把 PyTorch allocator 设置为物理 GPU 的 `8 / 32`。CUDA context、
bitsandbytes 和其他非 PyTorch allocation 同样占用显存，因此使用整进程
NVML-aware 预算：

1. 初始化 CUDA context；
2. 通过 NVML 查询当前 PID 的 context baseline；
3. 保留 128MiB safety margin；
4. 将剩余空间设为 PyTorch allocator allowance；
5. DiffSynth `vram_limit` 固定为目标预算的 50%；
6. 通过持久 `pynvml` handle 持续采样当前 PID 显存。

本机实测：

```text
target_vram_mib      = 8192
context_mib          = 500
safety_margin_mib    = 128
allocator_limit_mib  = 8192 - 500 - 128 = 7564
diffsynth_vram_limit = 4GiB
```

状态只允许：

```text
success / oom / budget_exceeded / timeout / numerical_failure / runtime_error
```

即使 PyTorch forward 成功，只要当前 PID 的 NVML 峰值超过 8192MiB，也必须记为
`budget_exceeded`。

## 6. 权重和模型驻留策略

所有模型使用 NF4 checkpoint，权重 backing store 为 CPU DRAM：

```text
offload_device:     cpu
computation_device: cuda
computation_dtype:  BF16
```

Text Encoder、DiT、Video VAE 和 Audio VAE 不会同时完整驻留 GPU：

```text
Text Encoder 阶段
  → offload Text Encoder
  → onload/lease DiT

DiT 完成
  → offload DiT
  → onload Video VAE

Video decode 完成
  → offload Video VAE
  → onload Audio VAE
```

因此“模型总参数很大”不等于“所有模型权重同时占用 8GB GPU”。本实验测试的主要矛盾
是单个 DiT block 中随序列长度增长的 activation 峰值。

## 7. Baseline 与 Streaming 实现差异

### 7.1 Full-sequence baseline

Baseline 保持当前实际路径：

```text
full hidden on GPU
full Q/K/V on GPU
FlashAttention 2
full out projection
full MLP fc1 output
full SiLU gate/up product
full residual
```

对于 61,312 tokens：

```text
hidden size = 5,376
FFN size    = 14,336
fc1 output  = 28,672
dtype       = BF16
```

理论 activation 大小：

```text
完整 hidden:
61,312 * 5,376 * 2 bytes ≈ 628MiB

完整 fc1 output:
61,312 * 28,672 * 2 bytes ≈ 3.28GiB
```

在 `silu(gate) * up` 时还需 gate/up、SiLU output、乘法 output、residual、当前层权重
和 CUDA workspace，因此 full-sequence 工作集会超过 8GB。

### 7.2 Streaming V0

V0 每个 block 的阶段：

| Phase | 内容 |
|---|---|
| A | chunked norm、AdaLN、QKV projection，Q/K/V 写入 pinned CPU RAM |
| B | Q/KV tiled FP32 online-softmax attention，KV 多次从 CPU 重扫 |
| C | attention output tile、out projection、gate、residual，结果写回 CPU |
| D1 | chunked fc1、gate/up、SiLU，完整 MLP intermediate 存 CPU |
| D2 | chunked fc2、gate、residual，next hidden 存 CPU |

正式配置：

```text
projection_chunk = 2048
q_block          = 4096
kv_block         = 1024
```

单个 fc1 GPU tile：

```text
2,048 * 28,672 * 2 bytes ≈ 112MiB
```

完整 MLP intermediate 转移到 CPU：

```text
61,312 * 14,336 * 2 bytes ≈ 1.64GiB CPU RAM
```

V0 没有消除 activation，而是把完整序列 activation 从 GPU 转移到 CPU DRAM，用
时间、CPU RAM 和 PCIe traffic 换取较低 GPU 峰值。

## 8. 61K Attention Tile 选择

在 full-DiT 测试前，对 61,312-token attention 测试四组 tile：

| Q block | KV block | 状态 | 时间 | NVML peak | Torch reserved |
|---:|---:|---|---:|---:|---:|
| 2048 | 512 | success | 9.357s | 1688MiB | 1068MiB |
| 2048 | 1024 | success | 9.116s | 2640MiB | 2020MiB |
| 4096 | 512 | success | 8.763s | 2740MiB | 2120MiB |
| 4096 | 1024 | success | 8.318s | 4532MiB | 3912MiB |

选择 attention latency 最低的 `4096/1024`。没有继续增大 tile，因为完整 DiT 单步
NVML 峰值已达到 7704MiB，只剩约 488MiB 物理余量，需要为 allocator 波动和最终
VAE decode 保留空间。

## 9. 已完成结果

### 9.1 Baseline：50-step 配置，首步 OOM

正式 baseline 命令配置了 50 steps；首个 DiT forward 已 OOM，后续 steps 和 VAE
decode 没有执行。

```text
status:                    oom
nvml_process_peak_mib:     7432
torch peak allocated:      6504.8MiB
torch peak reserved:       6810MiB
cpu_rss_peak_mib:          18001.8MiB
failed allocation request: 836MiB
failure location:          MLP silu(gate) * up
```

物理 GPU 仍有空闲显存，是因为实验进程被限制为 8GB；PyTorch allowance 约为
7.39GiB，其余用于 CUDA context 和 safety margin。

结果文件：

```text
workspace/benchmarks/results/
final8_baseline_full_f515_s50_480x832_f515_s50_20260817T074648Z.json
```

### 9.2 Streaming：完整 50-block 单步成功

```text
status:                    success
denoise step:              450.329s
pipeline total:            456.993s
nvml_process_peak_mib:     7704
torch peak allocated:      5371.7MiB
torch peak reserved:       7082MiB
cpu_rss_peak_mib:          41499.3MiB
```

该结果证明相同 61,312-token DiT forward 在 V0 下可以完成，但不等同于 50-step
端到端视频已经成功。

结果文件：

```text
workspace/benchmarks/results/
probe8_streaming_full_f515_q4096_kv1024_480x832_f515_s1_20260817T073717Z.json
```

### 9.3 对比摘要

| 项目 | Baseline | Streaming V0 |
|---|---:|---:|
| Packed tokens | 61,312 | 61,312 |
| VRAM budget | 8192MiB | 8192MiB |
| Weight offload | CPU/DRAM | CPU/DRAM |
| DiT blocks | 50 | 50 |
| 首个 forward | OOM | success |
| NVML peak | 7432MiB 后申请失败 | 7704MiB |
| CPU RSS peak | 17.6GiB | 40.5GiB |
| 单步时间 | 未完成 | 450.3s |

不能把 Streaming 表述为“观测峰值数值一定比 baseline 更低”。Baseline 在 7432MiB
时失败，是因为下一次 836MiB allocation 无法满足；其完成该操作所需工作集超过
8GB。Streaming 的意义是拆分大 allocation 并转移到 CPU，使完整 forward 在
7704MiB 实际峰值下完成。

## 10. 为什么 Encoder/Decoder 不一定 OOM

### 10.1 Text Encoder

Text Encoder 虽然参数量大，但：

1. 权重为 NF4，并采用 CPU/DRAM layer-wise offload；
2. 文本长度约 256 tokens，而 DiT packed length 是 61,312 tokens；
3. Text Encoder 完成后会在进入 DiT 前 offload。

```text
61,312 / 256 ≈ 240x
```

因此 Text Encoder 面临的是大权重问题，而不是 61k-token full-sequence MLP
activation 问题。

### 10.2 当前没有 Video/Audio Encoder

本实验是纯文本 FL2VA，没有输入 reference、keyframe 或 retake media。初始
video/audio latents 由 seed 0 的噪声生成，不需要 Video VAE Encoder 或 Audio VAE
Encoder。

### 10.3 Video VAE Decoder

进入 Video VAE decode 前，DiT 会被 offload。Video VAE 默认使用：

```text
tiled=True
tile_size=256
tile_overlap=64
```

空间使用 tile，时间维也按 latent token chunks 解码，因此不会创建类似
`[61,312, 28,672]` 的全序列 MLP 张量。

但 decoder 仍可能 OOM。完整 515-frame RGB 输出约为：

```text
BF16: 约 1.15GiB
FP32: 约 2.30GiB
```

再加上 VAE feature tile、overlap、权重和 BF16/FP32 conversion buffer，仍可能超过
8GB。只有正式任务完成 decode 后，才能声称端到端成功。

### 10.4 Audio VAE Decoder

Audio VAE 在 Video VAE 完成并 offload 后单独加载。其序列和通道规模远小于
61k-token DiT MLP，一般不会成为主 GPU 峰值，但最终仍以实际运行 JSON 为准。

## 11. 正式 50-step Streaming 任务

### 11.1 运行命令

```bash
docker compose exec -T -d \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/DiffSynth-Studio:/workspace \
  diffsynth bash -lc '
    exec numactl \
      --physcpubind=64-95,320-351 \
      --membind=3 \
      python /workspace/benchmarks/minimax_h3_baseline.py \
        --height 480 --width 832 --frames 515 --steps 50 --seed 0 \
        --tag final8_streaming_full_f515_s50_q4096_kv1024 \
        --target-vram-mib 8192 \
        --offload-device cpu \
        --activation-streaming \
        --projection-chunk-size 2048 \
        --attention-q-block-size 4096 \
        --attention-kv-block-size 1024 \
        > /workspace/benchmarks/results/final8_streaming_full_f515_s50.log 2>&1'
```

命令没有设置：

```text
--dit-layers
--skip-decode
--no-media
```

因此使用完整 50 blocks，并在 denoise 后执行真实 Video/Audio VAE decode 和 MP4 mux。

### 11.2 最近一次可确认的状态快照

截至 2026-08-17 08:03 UTC，在 RTX 5090 节点上观测到：

```text
container PID: 7947
elapsed:       15m22s
completed:     1 / 50 denoise steps
step 1:        457.903s
process state: running
```

该信息是时间点快照，不表示任务在文档更新时仍然存活。后续会话如果位于其他宿主机，
不能通过本机 Docker 状态推断 5090 节点上任务成功或失败；必须回到原 5090 节点检查
最终 JSON、日志和 MP4。当前尚未取得这些最终产物，因此本报告仍将端到端结果标记为
“待确认”。

按单步约 450–458 秒估算：

```text
50-step denoise ≈ 6.3 hours
```

该估算不包含最终 VAE decode 和 MP4 mux。实时日志：

```text
workspace/benchmarks/results/final8_streaming_full_f515_s50.log
```

预期最终文件：

```text
workspace/benchmarks/results/
final8_streaming_full_f515_s50_q4096_kv1024_480x832_f515_s50_<timestamp>.json

workspace/benchmarks/results/
final8_streaming_full_f515_s50_q4096_kv1024_480x832_f515_s50_<timestamp>.mp4
```

## 12. 端到端成功标准

正式 Streaming 结果必须同时满足：

1. JSON `status == success`；
2. 完成 50 个 `BENCH_STEP`；
3. `nvml_process_peak_mib <= 8192`；
4. Video VAE decode 完成；
5. Audio VAE decode 完成；
6. MP4 文件存在且大小大于 0；
7. MP4 可被 ffprobe 读取；
8. 视频帧数和音频 stream 存在；
9. JSON 中 `media_path`、`media_size_bytes`、decode timing 完整；
10. 不得把单步 probe 当作 50-step 完成结果。

最终验证命令：

```bash
ffprobe -v error \
  -show_entries format=duration,size \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of json \
  <output.mp4>
```

## 13. 最终应报告的指标

### 13.1 Baseline

```text
status
failure phase
failed allocation bytes
NVML process peak
Torch allocated/reserved peak
CPU RSS peak
是否进入第一个 denoise step
是否执行 decode
```

### 13.2 Streaming

```text
status
50-step total denoise time
per-step times
pipeline total time
NVML process peak
Torch allocated/reserved peak
CPU RSS peak
Video VAE decode time
Audio VAE decode time
MP4 write time
MP4 size/duration/frame count/audio stream
```

### 13.3 允许和禁止的表述

允许：

> 在相同 8GB 进程预算和 61,312-token FL2VA 输入下，full-sequence baseline 在首个
> DiT step 因 MLP activation OOM；Streaming V0 完成完整 50-block 单步，并在最终
> 50-step 任务成功后完成真实 Video/Audio VAE decode 和 MP4 生成。

正式任务完成前禁止：

> Streaming 已完成 61k-token、50-step、端到端视频生成。

## 14. 结论边界与风险

1. 当前案例是 FL2VA，不是 Ref2VA。
2. V0 是 PyTorch reference，不是高性能 fused kernel。
3. 61k attention 的计算量近似按序列长度平方增长，延迟非常高。
4. CPU RSS 已达到约 40.5GiB，不能只报告 GPU 显存收益。
5. KV 重扫会产生大量 PCIe H2D 流量。
6. Streaming 完整单步 NVML 峰值 7704MiB，距离 8192MiB 只有约 488MiB，decoder
   仍存在 OOM 或 budget-exceeded 风险。
7. Baseline 的 7432MiB 是失败前观测峰值，不是完成 forward 所需显存；其下一次
   836MiB allocation 已使所需工作集超过 8GB。
8. 单次运行没有误差条，只能使用 measured latency、observed peak 和 system
   characterization 等措辞。

## 15. 后续实验

正式 50-step FL2VA 完成后，建议：

1. 验证 MP4 和音频 stream；
2. 汇总 baseline/streaming 端到端表格；
3. 执行 Nsight Systems 单 block profile；
4. 增加 Ref2VA image reference parity；
5. 增加 Ref2VA video+audio reference 真实生成；
6. 对 reference encoder、DiT 和 decoder 分阶段记录 NVML peak；
7. 优化 final layer，避免完整 hidden 回到 GPU；
8. 将 V0 online-softmax 替换为更高效的 fused streaming kernel。
