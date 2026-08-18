# MiniMax-H3：8GB、61K Tokens 序列 Streaming 端到端实验说明

> 文档状态：BF16 fused streaming、跨 step 显存修复和 5-step soak 已完成；最终
> 50-step + VAE decode 正式任务正在执行
>
> 实验日期：2026-08-17 UTC
>
> 当前结论：在相同 8GB 整进程显存预算和 61,312-token 输入下，full-sequence
> baseline 在首个 DiT forward 的 full Q/K/V 路径 OOM；优化后的 Streaming 已完成
> 完整 50-block、61,312-token 的 5-step soak，NVML 峰值 6374MiB。正式 50-step
> 生成仍须以最终 JSON、双 VAE decode 和 MP4/ffprobe 结果为准。

## 1. 实验目的

本实验不是 attention 微基准，也不是缩减层数的工程 smoke test。实验使用真实
MiniMax-H3 NF4 checkpoint 和完整 FL2VA pipeline，构造普通 8GB GPU 无法通过
full-sequence DiT 完成、而 CPU-backed activation streaming 可以继续工作的长序列
视频生成案例。

实验需要回答：

1. 在严格的 8GB 整进程显存预算下，full-sequence baseline 是否因序列 activation
   OOM？
2. Fused Streaming 是否能在相同 checkpoint、输入、seed 和权重 offload 策略下完成
   完整 50-block DiT forward？
3. Streaming 的实际 GPU 显存、CPU RAM 和延迟代价是多少？
4. Streaming 是否最终能完成 50 denoise steps、Video VAE decode、Audio VAE decode
   和 MP4 mux？
5. Text Encoder 和 VAE Decoder 为什么没有与 full-sequence DiT 相同的 activation
   OOM 行为？

当前已经能够支持的结论为：

> 在相同 8GB 整进程显存预算下，61,312-token full-sequence MiniMax-H3 DiT 因完整
> 序列 QKV/MLP activation OOM；Streaming 将 QKV、attention output 和 MLP
> intermediate 转移到 CPU DRAM，并按 tile 在 GPU 上计算，从而完成相同输入的完整
> DiT forward 和 5-step soak。端到端成功仍以 50-step 任务完成真实 VAE decode 并
> 生成 MP4 为准。

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
streaming attention: BF16 FlashAttention 2 tiles + FP32 LSE/Triton merge
streaming fallback: pure PyTorch FP32 online softmax

Main repository branch: feature/minimax-h3-sequence-streaming
DiffSynth submodule commit: 653709f
seqattn submodule commit: 6d58f86
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

### 6.1 原生 full-sequence 实验的 CPU/GPU 驻留关系

这里的原生版本并不是“完全不 offload、把整个 checkpoint 放进显存”的弱 baseline。
其完整实验配置同样为：

```text
offload_device = cpu
disk offload   = disabled
activation_streaming = false
```

DiffSynth 按 pipeline 阶段切换 active model：

| 阶段 | GPU active model | CPU DRAM 中的非 active 大模型 |
|---|---|---|
| Text/prompt 处理 | Text Encoder | DiT、Video VAE、Audio VAE |
| 50-step denoise | **DiT** | Text Encoder、Video VAE、Audio VAE |
| Video decode | Video VAE | DiT、Text Encoder、Audio VAE |
| Audio decode | Audio VAE | DiT、Text Encoder、Video VAE |

因此原生 30GB 以上的 DiT 峰值不能解释为“Encoder + DiT + 两个 VAE 权重同时常驻”。
实际 OOM 发生在第 15 个 denoise step 开始后的 DiT MLP 中，此时两个 VAE decode 还
没有开始。

原生 denoise 阶段的驻留关系为：

| 对象 | CPU DRAM | GPU HBM |
|---|---|---|
| NF4 checkpoint / 非 active 模型权重 | backing store | 不常驻 |
| 未进入 preparing 状态的 DiT layer 权重 | backing store | 不常驻 |
| 当前/prepared DiT layer 权重 | CPU 有 backing | 当前计算副本或 prepared 副本 |
| Video/audio latent、packed hidden | — | 完整序列 tensor |
| Q/K/V | — | 完整序列 Q/K/V |
| Attention output | — | 完整序列 output，再执行 out projection |
| MLP `fc1` / gate / up / product | — | 完整序列 intermediate |
| CUDA context、FA workspace | — | runtime 占用 |
| Torch reserved/cache | — | 属于进程 NVML 显存，即使暂未 allocated |

132,288-token、BF16、H3 真实维度下，几个关键 activation 的孤立尺寸为：

| Tensor | 计算 | 大小 |
|---|---:|---:|
| hidden / residual | `132288 × 5376 × 2B` | 1.325GiB |
| 单个 Q/K/V | `132288 × 56 × 128 × 2B` | 1.766GiB |
| 合并 QKV | 上项 × 3 | 5.299GiB |
| attention output | `132288 × 56 × 128 × 2B` | 1.766GiB |
| MLP `fc1` output | `132288 × 2 × 14336 × 2B` | 7.065GiB |
| gate 或 up | `132288 × 14336 × 2B` | 3.532GiB |
| `SiLU(gate) * up` 结果 | `132288 × 14336 × 2B` | 3.532GiB |

原生正式实验的 traceback 正好停在：

```python
hidden = nn.functional.silu(gate) * up
```

失败申请为 3.53GiB，与完整 `[132288, 14336]` BF16 MLP product 的理论尺寸一致。
所以直接触发 OOM 的是 full-sequence activation；与此同时，进程总显存还包含当前权重、
临时 weight cast/rebuild、FlashAttention workspace、CUDA context 和 allocator cache。

高频 benchmark 在原生 step 14 边界记录到：

```text
DiT named CUDA parameter/buffer storage: 1030.8MiB
Torch allocated:                           4222.7MiB
Torch reserved:                          30254.0MiB
PID NVML:                                30876.0MiB
```

其中 1030.8MiB 只统计可从 DiT `parameters()/buffers()` 枚举到的去重 CUDA storage，
不包括完整序列 activation、临时计算权重、workspace、CUDA context 或 allocator cache。
因此它不能拿来代替 NVML 进程峰值。

Streaming 与原生的核心差别不是模型阶段调度，而是 sequence activation 的驻留位置：

| Sequence state | 原生 full-sequence | Streaming / seqattn |
|---|---|---|
| hidden/residual | 完整 GPU tensor | 完整 pinned CPU tensor + GPU chunk |
| Q/K/V | 完整 GPU tensor | 完整 pinned CPU backing + resident Q / streamed KV |
| attention output | 完整 GPU tensor | GPU tile 直接进入 out projection/gate/residual |
| MLP intermediate | 完整 GPU tensor | 完整 CPU intermediate + GPU fc1/fc2 chunk |

这也解释了为什么 Streaming 的 CPU RSS 更高，但 step-end GPU 显存可以维持在约
4.43GiB、within-step 峰值维持在约 7.16GiB。

Streaming 的权重驻留还增加了两个跨 denoise step 约束：

1. 第一个完整 denoise step 允许 DiffSynth 按原 `vram_limit=4GiB` 策略建立 GPU
   权重工作集；
2. 第一步结束后冻结该工作集，后续 step 不再把新的 CPU-backed layer 变为 GPU
   常驻；
3. 每个 streaming step 结束后执行 `empty_cache()`，释放未占用 allocator cache；
4. 下一次独立 pipeline generation 开始时重新允许建立工作集。

这不是 disk offload，也不是把 `vram_limit` 降低到 4GB 以下。权重 backing store 仍为
CPU DRAM；冻结只用于避免 50-step 过程中常驻权重集合和 allocator reserved 逐步扩大。

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

### 7.2 Streaming 实现

Streaming 每个 block 的阶段：

| Phase | 内容 |
|---|---|
| A | chunked norm、AdaLN、QKV projection，Q/K/V 写入 pinned CPU RAM |
| B | Q/KV tiled BF16 FlashAttention 2；FP32 LSE/output 跨 KV tile 合并 |
| C | attention output tile、out projection、gate、residual，结果写回 CPU |
| D1 | chunked fc1、gate/up、SiLU，完整 MLP intermediate 存 CPU |
| D2 | chunked fc2、gate、residual，next hidden 存 CPU |

正式配置：

```text
projection_chunk = 2048
q_block          = 16384
kv_block         = 4096
```

单个 fc1 GPU tile：

```text
2,048 * 28,672 * 2 bytes ≈ 112MiB
```

完整 MLP intermediate 转移到 CPU：

```text
61,312 * 14,336 * 2 bytes ≈ 1.64GiB CPU RAM
```

Phase B 使用两组 GPU KV buffer、独立 CUDA copy stream 和 event：当前 KV tile 计算时
预取下一 tile。每个 FlashAttention tile 使用 BF16 Q/K/V，返回 FP32 softmax LSE；
Triton kernel 将 tile output 和 LSE 合并进 FP32 running output，最后转回 BF16 并
D2H。`auto` backend 在 CUDA + BF16/FP16 + FlashAttention/Triton 可用时选择该路径，
否则回退纯 PyTorch reference。

Streaming 没有消除 activation，而是把完整序列 activation 从 GPU 转移到 CPU DRAM，
用 CPU RAM、PCIe traffic 和额外延迟换取较低 GPU 峰值。

## 8. 61K Full-DiT Tile 选择与加速

使用完整 50-block DiT、61,312 tokens、1 step、8GB budget 测试 fused backend：

| Q block | KV block | 状态 | Denoise step | NVML peak |
|---:|---:|---|---:|---:|
| 4096 | 1024 | success | 94.461s | 5646MiB |
| 8192 | 2048 | success | 79.809s | 5700MiB |
| 16384 | 2048 | success | 77.272s | 5746MiB |
| 16384 | 4096 | success | 72.843s | 5744MiB |

最终锁定：

```text
projection_chunk = 2048
q_block          = 16384
kv_block         = 4096
```

旧 pure-PyTorch V0 的同形状单步为 450.329s；锁定配置的开发测量为 72.843s，约
`6.18x` 加速。后续无 instrumentation 的 5-step soak 中，首步 66.905s，step 2–5
稳定在约 61.9s。不同 run 的绝对延迟会受到主机 DRAM/NUMA 和 GPU 状态影响，因此
正文同时保留原始 JSON，而不只报告单一最佳数字。

## 9. 已完成结果

### 9.1 Baseline：首个 full-sequence forward OOM

最终代码重跑的 baseline 配置了 50 steps，但首个 DiT forward 未完成：

```text
status:                    oom
completed denoise steps:   0
nvml_process_peak_mib:     8078
torch peak allocated:      6634.4MiB
torch peak reserved:       7456MiB
cpu_rss_peak_mib:          18002.6MiB
failed allocation request: 836MiB
failure location:          full Q/K/V path, k_norm RMSNorm
```

物理 32GB GPU 仍显示有空闲，是因为实验进程被限制为 8GB；PyTorch allowance 约
7.39GiB。Baseline 在完整 Q/K/V 阶段已经没有足够空间创建下一个 836MiB tensor，
因此没有进入后续 denoise step 或 VAE decode。

结果文件：

```text
workspace/benchmarks/results/
final8_baseline_full_f515_s50_recheck_480x832_f515_s50_20260817T124244Z.json
```

### 9.2 从 reference V0 到 fused backend

同为完整 50-block、61,312-token、1-step：

| 实现 | Denoise step | NVML peak | 说明 |
|---|---:|---:|---|
| PyTorch FP32 online-softmax V0 | 450.329s | 7704MiB | reference |
| BF16 Flash2 + Triton LSE merge | 72.843s | 5744MiB | 开发调参点 |
| 最终 instrumented run | 67.307s | 5610MiB | phase timing，不作主延迟 |

开发调参点相对旧 V0 约 `6.18x` 加速。最终 steady-state 5-step 中 step 2–5 约
61.9s；跨独立进程的绝对差异可能来自 DRAM/NUMA、权重工作集建立和 GPU 状态。

### 9.3 旧多步失败与修复

两次旧 50-step 尝试均保留为失败证据：

1. pure-PyTorch V0 在完成 5 steps 后因 allocator/residency headroom 耗尽而 OOM；
2. 初版 fused backend 完成 47 steps，第 48 个 forward 的 FlashAttention tile 在申请
   224MiB 时 OOM，NVML 峰值 8166MiB；其单步从约 72s 逐渐升到 100–110s。

最终修复包括：

- final layer 只对 video/audio output positions 分 chunk 投影，不再把完整
  `[61,312, 5,376]` hidden 搬回 GPU；
- 第一个完整 denoise step 后冻结 DiffSynth GPU 权重工作集；
- 后续 step 禁止新 layer 进入 preparing/GPU 常驻状态；
- 每个 streaming step 结束后释放未占用 CUDA allocator cache；
- benchmark 在失败路径也保留已完成 step 的逐步显存和时间。

### 9.4 最终 5-step 稳定性 soak

```text
status:                    success
step times:                66.905, 61.863, 61.897, 61.885, 61.967s
mean step:                 62.904s
nvml_process_peak_mib:     6374
torch peak allocated:      5536.5MiB
torch peak reserved:       5684MiB
cpu_rss_peak_mib:          43616.7MiB
logical H2D per step:      611.88GiB
logical D2H per step:      305.69GiB
logical CPU activation:    3.67GiB
```

step-end NVML 为：

```text
4490, 3832, 3952, 3774, 3696MiB
```

step-end Torch reserved 为：

```text
3800, 3142, 3262, 3084, 3006MiB
```

这组数据没有旧实现的逐 step 显存阶梯增长；`empty_cache()` A/B 对平均延迟的影响约
0.08%，但将 5-step NVML 峰值从 6516MiB 降到 6374MiB。

结果文件：

```text
workspace/benchmarks/results/
soak8_flash2_frozen_emptycache_full_f515_s5_q16384_kv4096_480x832_f515_s5_20260817T123616Z.json
```

### 9.5 单步 phase 与搬运开销

Instrumented 单步的 phase wall time：

| Phase | Seconds | 占 67.307s |
|---|---:|---:|
| A | 10.037 | 14.9% |
| B | 30.229 | 44.9% |
| C | 4.194 | 6.2% |
| D1 | 9.604 | 14.3% |
| D2 | 7.046 | 10.5% |
| final layer | 0.120 | 0.2% |

Nsight Systems 独立 profiling run 的 CUDA memcpy 汇总：

```text
profile step wall: 74.209s
H2D copy-engine:   21.358s, 688.3GB
D2H copy-engine:   12.849s, 334.6GB
copy-engine sum:   34.207s, 46.1% of profiled wall
```

46.1% 是 copy-engine busy time / profiled wall，不是“去掉搬运后的可直接加速比例”。
KV 双缓冲允许部分 H2D 与 FlashAttention kernel 重叠，而且 H2D/D2H 与计算的重叠关系
必须结合 timeline 理解。

### 9.6 数值 parity

在 baseline 可以运行的 19K-token、完整 50-block、1-step 点上：

| Output | Relative L2 | Max abs | Cosine |
|---|---:|---:|---:|
| Video | 0.017715 | 0.21875 | 0.999730 |
| Audio | 0.027160 | 0.03125 | 0.999714 |

BF16 fused streaming 不追求 bitwise 一致；cosine 均大于 0.9997，未观测到 NaN/Inf。

### 9.7 当前对比摘要

| 项目 | Full-sequence baseline | Fused Streaming |
|---|---:|---:|
| Packed tokens | 61,312 | 61,312 |
| VRAM budget | 8192MiB | 8192MiB |
| Weight backing | CPU/DRAM | CPU/DRAM |
| DiT blocks | 50 | 50 |
| 首个 forward | OOM | success |
| 5-step soak | 不可执行 | success |
| NVML peak | 8078MiB 后申请失败 | 6374MiB |
| CPU RSS peak | 17.6GiB | 42.6GiB |

Baseline 的 8078MiB 是失败前观测峰值，不是完成 forward 所需显存。Streaming 的
收益在于拆分无法满足的大 allocation，并将完整序列 activation 转移到 CPU；代价是
约 43GB RSS、每步约 918GiB 逻辑双向流量和显著延迟。

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
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/opt/DiffSynth-Studio:/workspace \
  diffsynth bash -lc '
    exec numactl \
      --physcpubind=64-95,320-351 \
      --membind=3 \
      python /workspace/benchmarks/minimax_h3_baseline.py \
        --height 480 --width 832 --frames 515 --steps 50 --seed 0 \
        --tag final8_flash2_frozen_emptycache_full_f515_s50_q16384_kv4096 \
        --target-vram-mib 8192 \
        --offload-device cpu \
        --activation-streaming \
        --projection-chunk-size 2048 \
        --attention-q-block-size 16384 \
        --attention-kv-block-size 4096 \
        --streaming-attention-backend flash2_lse \
        > /workspace/benchmarks/results/final8_flash2_frozen_emptycache_full_f515_s50.log 2>&1'
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
workspace/benchmarks/results/final8_flash2_frozen_emptycache_full_f515_s50.log
```

预期最终文件：

```text
workspace/benchmarks/results/
final8_flash2_frozen_emptycache_full_f515_s50_q16384_kv4096_480x832_f515_s50_<timestamp>.json

workspace/benchmarks/results/
final8_flash2_frozen_emptycache_full_f515_s50_q16384_kv4096_480x832_f515_s50_<timestamp>.mp4
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
> DiT forward 的 full Q/K/V 路径 OOM；Fused Streaming 完成完整 50-block 和
> 5-step soak，并在最终
> 50-step 任务成功后完成真实 Video/Audio VAE decode 和 MP4 生成。

正式任务完成前禁止：

> Streaming 已完成 61k-token、50-step、端到端视频生成。

## 14. 结论边界与风险

1. 当前案例是 FL2VA，不是 Ref2VA。
2. 当前 backend 已使用 BF16 FlashAttention 2 和 Triton LSE merge，但 projection、
   MLP 与 CPU/GPU pipeline 仍不是端到端单 kernel 实现。
3. 61k attention 的计算量近似按序列长度平方增长，延迟非常高。
4. CPU RSS 已达到约 42.6GiB，不能只报告 GPU 显存收益。
5. KV 重扫会产生大量 PCIe H2D 流量。
6. 5-step Streaming NVML 峰值 6374MiB，距离 8192MiB 约 1818MiB；最终 decoder
   仍须以正式任务实测为准。
7. Baseline 的 8078MiB 是失败前观测峰值，不是完成 forward 所需显存；其下一次
   836MiB allocation 已使所需工作集超过 8GB。
8. 单次运行没有误差条，只能使用 measured latency、observed peak 和 system
   characterization 等措辞。

## 15. 后续实验

正式 50-step FL2VA 完成后，建议：

1. 增加 Ref2VA image reference parity；
2. 增加 Ref2VA video+audio reference 真实生成；
3. 对 reference encoder、DiT 和 decoder 分阶段记录 NVML peak；
4. 将 projection/MLP 小算子进一步 fuse，减少 Python launch 和中间 tensor；
5. 复用持久 copy stream/event/buffer，减少每 block 对象创建；
6. 在保持固定 tile 的前提下研究更深的 copy/compute pipeline；
7. 增加 4GB/6GB/8GB 与更多序列长度的独立进程扫长。
