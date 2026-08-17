# MiniMax-H3 V0 序列 Chunk 化对比实验方案

> 状态：V0 与 Stage A benchmark 基础设施已实现；正式扫长与 Stage B 完整模型实验待执行
> 对比对象：当前 full-sequence 路径 vs. CPU-backed streaming V0
> 主要目标：验证固定显存预算下的长序列可扩展性

## 1. 核心论点

本实验面向论文或技术报告，主要验证以下结论：

> 在相同 NF4 权重、CPU/DRAM offload 和 GPU 显存预算下，当前 full-sequence 实现的显存随序列长度增长并最终 OOM；V0 chunked streaming 的 GPU 峰值主要由 chunk/tile 大小决定，可以在 4GB 显存内处理显著更长的序列，同时保持原有 attention 和 segment 语义。

V0 是用于验证数学正确性和 out-of-core 可行性的纯 PyTorch reference，不主张比 FlashAttention 2 baseline 更快。报告必须同时展示它的延迟、CPU RAM 和 PCIe 流量代价，不能只展示显存收益。

## 2. 公平对比协议

### 2.1 固定条件

- 模型：MiniMax-H3 FL2VA NF4。
- 计算精度：BF16；FP32 仅用于 attention 正确性 reference。
- 权重后端：CPU/DRAM offload，不使用 disk offload。
- Baseline：当前未修改的 full-sequence DiT，attention backend 固定为 FlashAttention 2。
- Streaming：纯 PyTorch online-softmax V0，包含 chunked QKV、streaming attention、chunked out projection 和 two-pass MLP。
- 两种方案使用相同 checkpoint、输入、seed、显存预算和 DiffSynth 权重驻留策略。
- 只测 DiT，不把文本编码和视频/音频 VAE decode 计入结果。
- 每个数据点使用独立进程，避免 allocator、权重驻留和 page cache 状态相互污染。

Baseline 使用 FlashAttention 2 是为了与当前实际最强路径比较，而不是刻意选择较弱的 Torch SDPA。Torch SDPA 只作为数值正确性的 reference。

### 2.2 4GB 显存预算

不能简单把 32GB RTX 5090 的 PyTorch allocator 上限设置为 4GB，因为 CUDA context、bitsandbytes 和其他非 PyTorch 分配也会占用设备内存。每个 benchmark 进程采用以下步骤：

1. 初始化 CUDA context。
2. 使用 NVML 读取当前进程的非 PyTorch GPU 占用 `context_mib`。
3. 设置 PyTorch allocator 上限：

```text
allocator_limit_mib = 4096 - context_mib - 128
```

其中 128MiB 是安全余量。

4. DiffSynth `vram_limit` 固定为目标预算的 50%，主实验为 2GiB。
5. 运行中持续采样当前 PID 的 NVML 显存；实际进程峰值超过 4096MiB 即记为失败。

结果状态只允许：

```text
success / oom / budget_exceeded / timeout / numerical_failure / runtime_error
```

OOM 和 timeout 是有效实验结果，不得删除或用外推值代替。

### 2.3 硬件与运行环境

- GPU：RTX 5090，固定 GPU 0。
- GPU 0 的 PCIe/CPU affinity 位于 NUMA node 2，对应 CPU
  `64-95,320-351`。该节点在当前主机是 0MiB memory-only 拓扑中的 CPU node，不能
  作为 `membind` 目标。独立的 512MiB pinned-copy probe 显示 node 3 是实际高带宽
  DRAM 路径（H2D/D2H 约 36.0/32.1 GB/s；node 1 约 15.1/14.5 GB/s）。
- 进程使用：

```text
numactl --physcpubind=64-95,320-351 --membind=3
```

- CPU affinity 对应 `64-95,320-351`。
- Docker Compose 必须授予 `SYS_NICE` capability，否则 `set_mempolicy` 会返回
  `Operation not permitted`，该数据点应记为 `runtime_error` 而不是在未绑定状态下继续。
- 每个数据点先用 3072 tokens 做一次不计时 warmup，再执行一次正式测量。
- warmup 后执行 `torch.cuda.empty_cache()`、同步 GPU 并重置 peak-memory counters。
- 单个数据点最长运行 30 分钟，超过后记为 `timeout`。
- 当前选择单次正式运行，因此不绘制误差条，也不使用统计显著性表述。

warmup 与正式测量必须在同一个数据点进程中使用相同 mode、dtype、head/block
配置。warmup 固定为 3,072 tokens，但不计入正式时间和 peak counter；CPU page
cache 不作为可控变量。每个正式点仍由 controller 启动独立 worker 进程。

## 3. 合成输入与序列长度

实验只使用合成输入，但保持 MiniMax-H3 实际 packed sequence 的模态比例和 shape 规则：

- 固定目标分辨率：480×832。
- 固定 text length：256。
- 视频 latent 空间尺寸：`H=30, W=52`。
- audio/video token 数由真实 pipeline 公式生成。
- 通过帧数改变 temporal length，不直接构造不符合模型布局的任意 token 数。
- `cu_seqlens`、padding segment、token tags 和 position IDs 均通过现有 `MiniMaxH3Unit_PackedSequenceBuilder` 生成。

主序列长度矩阵：

| 帧数 | Video latent T | Packed tokens |
|---:|---:|---:|
| 22 | 7 | 3,072 |
| 39 | 12 | 5,120 |
| 56 | 17 | 7,104 |
| 73 | 22 | 9,088 |
| 90 | 27 | 11,136 |
| 124 | 37 | 15,104 |
| 158 | 47 | 19,136 |
| 192 | 57 | 23,168 |
| 226 | 67 | 27,200 |
| 260 | 77 | 31,168 |

所有随机 hidden、audio latent、prompt embedding 和 timestep tensor 使用固定 seed 0。

## 4. 三级实验矩阵

### 4.1 Attention 正确性与扫长微基准

#### 正确性

使用 Torch SDPA 作为 reference，比较 V0 online softmax：

- 序列长度：257、1023、3072。
- dtype：FP32 和 BF16。
- segment 数：1、2、5。
- 必须覆盖：
  - segment 长度不能整除 Q/KV block；
  - 最后一个 Q/KV tile 不完整；
  - packed padding segment；
  - 多 segment 不能发生跨 segment attention。

FP32 通过条件：

```text
relative_l2 <= 1e-5
max_abs_error <= 1e-4
```

BF16 报告以下指标，不要求 bitwise 一致：

```text
relative L2
max absolute error
cosine similarity
```

#### 性能与显存扫长

使用第 3 节全部 10 个 token 档位：

- Baseline：Q/K/V 完整驻留 GPU，调用 FlashAttention 2。
- Streaming：Q/K/V 位于 pinned CPU RAM，逐 tile H2D，结果逐 tile D2H。
- Baseline 计时不包含随机输入生成。
- Streaming 分别报告 kernel 时间、数据传输时间和包含传输的端到端时间。

记录：

- wall time；
- Torch allocated/reserved peak；
- 当前进程 NVML peak；
- CPU RSS 和 pinned-memory peak；
- H2D/D2H 字节数；
- tokens/s；
- success/OOM/timeout。

### 4.2 单 Transformer Block 对比与 chunk 消融

使用真实 MiniMax-H3 block 维度和真实 NF4 权重，覆盖：

```text
AdaLN/norm
QKV projection
attention
out projection + residual
MLP fc1 + SiLU gate
MLP fc2 + residual
```

主扫长使用第 3 节全部 10 个 token 档位。

#### Chunk 参数锁定

在主实验开始前，只允许在 15,104 tokens、4GB 预算下做一次参数选择：

```text
q_block:                 512 / 1024 / 2048
kv_block:                256 / 512
projection_and_mlp_chunk: 512 / 1024 / 2048
```

共 18 个组合。选择规则固定为：

1. 过滤 NVML 峰值超过 3968MiB 或运行失败的组合。
2. 在剩余组合中选择单 block latency 最低者。
3. latency 差异不超过最快配置的 1% 时视为并列，再依次选择更大的
   `q_block`、`kv_block`、projection/MLP chunk。
4. 参数一旦选定，锁死用于全部序列长度和完整 DiT 实验，不允许逐点调参。
5. 被选中的配置必须在新的独立进程中复测一次，复测成功后才进入主实验。

Streaming 单 block 必须记录：

| Phase | 内容 |
|---|---|
| A | chunked QKV projection |
| B | online-softmax streaming attention |
| C | out projection、gate 和 residual |
| D1 | fc1、gate/up、SiLU、CPU intermediate |
| D2 | fc2、gate、residual 和 next hidden |

每个 phase 分别记录耗时、H2D/D2H 字节和阶段峰值显存。

### 4.3 完整 DiT 合成推理

完整 50-block DiT 每个数据点只执行一个 denoise forward，不执行文本 encode 或 VAE decode。

代表性序列长度：

| Packed tokens | 用途 |
|---:|---|
| 7,104 | 短序列 |
| 15,104 | 当前 480×832、124 帧标准任务 |
| 23,168 | 长序列 |
| 31,168 | V0 强化目标 |

在 15,104 tokens 上增加显存预算敏感性实验：

```text
4GB / 6GB / 8GB
```

每档的 DiffSynth `vram_limit` 固定为目标预算的 50%，并使用相同的 NVML-aware allocator 限制方法。

## 5. 指标与图表

### 5.1 主图：Peak GPU Memory vs. Sequence Length

- Baseline 和 Streaming 各一条曲线。
- 横轴使用 packed token 数，而不是只写视频帧数。
- 添加 4GB 水平线。
- OOM/TIMEOUT 使用独立标记，不能与成功点连线。
- 同时绘制 Torch reserved peak 和 NVML process peak，论文正文使用 NVML peak。

该图用于支撑“baseline 随序列增长、streaming 由 tile 决定”的主要论点。

### 5.2 Latency vs. Sequence Length

- 使用对数纵轴。
- 同时展示 baseline 和 Streaming。
- 表格中增加相对 slowdown：

```text
streaming_latency / baseline_latency
```

- Baseline OOM 后不计算 slowdown。

### 5.3 Maximum Supported Sequence Length vs. VRAM Budget

- 显存预算：4GB、6GB、8GB。
- 每种方案只报告实际成功运行的最大 token 数。
- 不对未测长度做插值或外推。

### 5.4 Streaming Phase Breakdown

在 15,104 和 31,168 tokens 上绘制 Phase A/B/C/D1/D2 堆叠柱状图，并在柱顶标出：

```text
总 H2D
总 D2H
CPU activation peak
```

### 5.5 正确性表

分别报告 attention、单 block 和完整 DiT 的：

```text
relative L2
max absolute error
cosine similarity
```

完整 DiT parity 只在 baseline 能运行的最小两个代表点上比较。

## 6. Profiling 与结果格式

### 6.1 NVTX/Nsight

在 15,104-token 单 block 上额外执行一次 Nsight Systems profiling：

```text
--trace=cuda,nvtx
```

代码必须为 Phase A/B/C/D1/D2 添加 NVTX range，用于检查：

- H2D/D2H 是否与预期 phase 对应；
- pinned-memory copy 是否为异步；
- copy 与 compute 是否存在重叠；
- KV 重扫带来的实际 PCIe 流量。

Nsight run 只用于解释性能，不作为主延迟数字。

### 6.2 JSON 输出

每个数据点输出独立 JSON，至少包含：

```text
status
mode: baseline | streaming
scope: attention | block | dit
frames / packed_tokens / segment_lengths
dtype / seed
q_block / kv_block / projection_chunk / mlp_chunk
target_vram_mib / allocator_limit_mib / vram_limit_gib
torch_peak_allocated_mib / torch_peak_reserved_mib
nvml_process_peak_mib
cpu_rss_peak_mib / pinned_memory_peak_mib
total_seconds / phase_seconds
h2d_bytes / d2h_bytes
attention_backend
model_commit / torch / cuda / bitsandbytes / gpu / numa
error_metrics
failure_message
```

聚合脚本只读取 JSON 生成 CSV、Markdown 表格和图片，不从 stdout 解析核心指标。

## 7. 成功标准与结论边界

V0 只有满足以下条件，才能支持预期论文结论：

1. 4GB 实际进程显存限制下，Streaming 完成 31,168-token attention 和单-block 测试。
2. 完整 DiT 至少完成 23,168 tokens；31,168 tokens 是强化目标。
3. 从 3,072 到 31,168 tokens，Streaming 的 NVML GPU 峰值变化不超过 512MiB。
4. CPU activation RAM 可以随序列长度增长，但必须被完整记录。
5. Baseline 在相同预算下具有更低的最大可运行序列长度。
6. FP32 attention parity 达到预设阈值，BF16 无异常数值漂移。
7. 报告 V0 的真实 slowdown、CPU RAM 和 PCIe 成本。

若 V0 未达到 4GB 或 31,168-token 目标：

- 保留所有失败点；
- 不允许删除不利数据或在主扫长中重新选择 tile；
- 结论降级为“降低显存增长率”或“提高最大可运行序列长度”；
- 不得表述为“显存与序列长度无关”。

由于当前实验选择单次正式运行，最终报告只能使用系统 characterization、observed peak 和 measured latency 等措辞，不能使用 statistically significant、confidence interval 或稳定性分布等表述。

## 8. 执行分阶段与实现约束

为避免开发期 50-block 运行阻塞验证，实验分为两个阶段：

- Stage A（当前必须完成）：attention 正确性/扫长、真实单 block、5-block
  implementation validation。5-block 结果必须显式标记为 validation，不能写成
  full DiT 或用于完整模型结论。
- Stage B（长时任务）：只运行第 4.3 节的代表性 50-block 点和预算敏感性点。

主延迟 run 与 instrumented run 分开。主 run 不启用同步 phase timer；instrumented
run 启用 A/B/C/D1/D2 NVTX 和同步边界，仅用于 phase breakdown 与 Nsight 解释。

传输字节采用代码路径的 logical accounting，并至少区分 weight、QKV、activation、
attention output、MLP intermediate 和 metadata。当前 V0 插桩直接覆盖 activation/QKV/
attention output/MLP intermediate；权重流量由 DiffSynth lease 层单独记录或在结果中
明确标为 unavailable，不能把 PCIe profiler 数字和 logical bytes 混为同一指标。

Streaming 最后一层当前仍把完整 hidden materialize 到 GPU，因此必须单列
`final_layer` phase 和峰值；在优化它之前，不得把完整 DiT 的峰值表述为完全由 tile
决定。
