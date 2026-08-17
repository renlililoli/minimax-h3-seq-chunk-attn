# MiniMax-H3 Out-of-Core（Ring-style Streaming）推理方案设计

> 状态：V0 reference 已实现（分支 `feature/minimax-h3-sequence-streaming`）
> 适用代码：`extern/DiffSynth-Studio`（diffsynth 2.1.2，commit 6343ded）
> 目标硬件：单 GPU（RTX 5090 32GB）+ 大容量 CPU RAM

## 1. 背景与动机

目标：**不保留完整 K/V 在显存，也不把「FFN chunking」当主线**，而是实现一个单 GPU、CPU-backed 的 Ring-Attention / out-of-core Transformer：

> 整条 sequence 的 hidden/QKV 放在 CPU RAM；GPU 任意时刻只有一个 Q block、一个 KV block、online-softmax 状态和当前算子权重。

显存占用由 **tile size 决定**，而不随 sequence length 线性增长。

### 现状代码事实（已核实）

- `MiniMaxH3Attention.forward`（`diffsynth/models/minimax_h3_dit.py:136`）对完整 `x` 一次性做 fused `qkv_proj`（`nn.Linear(5376, 3*56*128)`），随后 q/k RMSNorm、RoPE、attention、`out_proj`，**全序列 activation 都在 GPU**：

```text
GPU:  full hidden → full QKV → attention → full hidden
```

- `_sdpa_varlen_attention`（`minimax_h3_dit.py:70`）已经按 `cu_seqlens` 对每个 packed segment 独立做 attention（逐 segment 调 `attention_forward`）。streaming 实现必须保持同样的 segment 语义，不能把所有 token 当成一条序列。
- `diffsynth/core/attention/attention.py` 的统一入口 `attention_forward()` 只返回最终 output，支持 Torch SDPA / FlashAttention 2/3/4 / SageAttention / xFormers / FlexAttention，但**不暴露做跨 KV tile merge 所需的 softmax LSE / normalization state**。因此 streaming attention 不强塞进 `attention_forward()`，而是新建独立模块。
- VRAM 管理：`configs/vram_management_module_maps.py` 把 `MiniMaxH3DiT` 的所有 `nn.Linear` 包成 `AutoWrappedLinear`。disk offload 时 `computation()` 每次 forward 都 `load_from_disk(..., assign=False)` **重新加载一次权重**（`diffsynth/core/vram/layers.py:377`）。若直接对 hidden 做 chunk 循环，每块权重会被反复加载 → 需要 lease 机制让权重一次加载、多个 chunk 复用。
- `MiniMaxH3DiT.forward`（`minimax_h3_dit.py:374`）经 `gradient_checkpoint_forward` 逐 block 调用；streaming 模式与 gradient checkpointing 互斥。
- 调用链：pipeline `__call__`（`diffsynth/pipelines/minimax_h3_audio_video.py:76`）→ `inputs_shared` → `model_fn_minimax_h3`（:820）→ `dit(...)`（:886）。新参数沿此链路透传。
- `_embed` 产生的 `[seq_len, 5376]` embeddings（15k token 约 154MB）是一次性张量，streaming 模式下算完即整体搬 CPU。
- `token_refiner` 只处理文本短序列，保持原有 GPU 路径不变。
- `rope_freqs`（`[1, S, rot_dim]`）、`t_emb`、AdaLN 参数都很小，保留在 GPU，按 chunk 切片即可。

## 2. 核心数据流

```text
CPU RAM
┌───────────────────────────────┐
│ hidden[N,5376]                │
│ Q[N,56,128]                   │
│ K[N,56,128]                   │
│ V[N,56,128]                   │
└───────────────────────────────┘
              │
              │ tiles
              ▼
GPU
┌───────────────────────────────┐
│ Q tile                        │
│ K tile                        │
│ V tile                        │
│ running m / l / O             │
└───────────────────────────────┘
```

这与 Ring Attention 的数学核心相同；区别只是 KV block 的传输路径：

```text
Ring Attention:  GPU0 → GPU1 → GPU2 → ...
本方案:          CPU RAM → GPU → CPU RAM
```

因此不直接搬某个 Ring Attention 库，而是复用它的 **blockwise online-softmax 算法**（exact，非近似）。

## 3. 单 Block 的四阶段执行

### Phase A：QKV projection streaming

`hidden` 完整留 CPU，每次只搬 `hidden[start:end]` 到 GPU：

```python
for start in range(0, N, proj_chunk):
    x = hidden_cpu[start:end].to("cuda")

    h = norm1(x)
    h = modulate(h)                      # AdaLN shift/scale，index_select 按 chunk 做

    qkv = qkv_proj(h)
    q, k, v = split(qkv)
    q = q_norm(q); k = k_norm(k)
    q = apply_rope(q, rope[start:end])
    k = apply_rope(k, rope[start:end])

    Q_cpu[start:end].copy_(q, non_blocking=True)
    K_cpu[start:end].copy_(k, non_blocking=True)
    V_cpu[start:end].copy_(v, non_blocking=True)
    # GPU 上这批 qkv 随即释放
```

projection 阶段最大 activation 是 `O(C_proj·d)` 而不是 `O(N·d)`。

**为什么 CPU 保存完整 Q/K/V，而不是只存 KV、重算 Q？** H3 的 `qkv_proj` 是 fused 的（NF4 量化下不宜拆 packed weight）。一次 projection 把 Q/K/V 全部放 CPU，15k token 约：

```text
Q ≈ 205 MiB, K ≈ 205 MiB, V ≈ 205 MiB  →  合计 ~615 MiB host RAM
```

完全可接受，且避免 projection 跑两遍。

### Phase B：Ring-style streaming attention（exact）

对每个 Q block `Q_i`，初始化 `m_i = -∞, l_i = 0, O_i = 0`，然后逐个扫描 KV blocks：

```text
             KV0
              ↓
Q_i ────── attention → update m,l,O
             KV1
              ↓
Q_i ────── attention → update m,l,O
             ...
             KVn
              ↓
         final O_i
```

Online softmax 更新公式（对新到的 `K_j, V_j`）：

```text
S_ij = Q_i K_j^T / √d
m_ij = max_j S_ij                       # 当前 tile 行最大值
m_i' = max(m_i, m_ij)
P_ij = exp(S_ij - m_i')
α    = exp(m_i - m_i')                  # 旧 accumulator 缩放
l_i' = α·l_i + Σ_j P_ij
O_i' = α·O_i + P_ij V_j
```

全部 KV 扫完：`Attention_i = O_i / l_i`。

**必须按 `cu_seqlens` 分 segment**：外层先逐 segment，segment 内再切 Q/KV tile，Q tile 不跨 segment。

### Phase C：out projection 不恢复成 full GPU tensor

```python
for q_tile:
    attn = streaming_attention_tile(...)   # Phase B
    h = out_proj(attn)
    x_attn = residual_tile + gate * h      # AdaLN gate + residual，逐 chunk 在 GPU 做
    post_attn_cpu[start:end] = x_attn.cpu()
```

完整的 attention output 永远不在 GPU 上物化。

### Phase D：MLP 两 pass out-of-core

不让 `[N, 28672]` 出现，且不同时把 `fc1` 和 `fc2` 权重放 GPU：

```text
Phase D1:  CPU post_attention ──chunk──→ GPU fc1 → gate/up → SiLU → CPU mlp_intermediate[N,14336]
           （15k token BF16 约 410 MiB host RAM）
Phase D2:  释放 fc1，加载 fc2：
           CPU mlp_intermediate ──chunk──→ GPU fc2 → gate + residual → CPU next_hidden
```

### 完整执行图

```text
              CPU hidden
                   │ chunks
                   ▼
        ┌────────────────────┐
        │ Phase A: QKV proj  │
        └────────────────────┘
           │      │      │
           ▼      ▼      ▼
        CPU Q   CPU K   CPU V

            ┌────────────────┐
            │ Phase B        │
            │ Ring attention │   Q tile → KV0/KV1/.../KVn → update m,l,O
            └────────────────┘
                │
             out_proj → residual + gate
                ▼
       CPU post_attn_hidden
        （del Q/K/V，复用 buffer）

          ┌─────────────┐
          │ Phase D1    │  fc1 stream
          └─────────────┘
                ▼
       CPU mlp_intermediate[N,14336]

          ┌─────────────┐
          │ Phase D2    │  fc2 stream
          └─────────────┘
                ▼
         CPU next_hidden

hidden_cpu, next_hidden_cpu = next_hidden_cpu, hidden_cpu   # 进入下一个 block
```

## 4. 内存预算（约 15k tokens，BF16）

### Host RAM

Attention 阶段：

```text
hidden      ~154 MB
Q/K/V       ~615 MB
post_attn   ~154 MB
────────────────────
~923 MB
```

MLP 阶段（attention 结束后 `del Q/K/V` 复用 buffer）：

```text
post_attn       ~154 MB
mlp_intermediate ~410 MB
next_hidden     ~154 MB
────────────────────
~718 MB
```

整个 activation host working set **约 1GB**，相当现实。

### GPU working set（streaming attention 自身）

例：`q_block=2048, kv_block=512, heads=56, d=128, BF16`：

```text
Q tile              ~28 MiB
K+V tile            ~14 MiB
fp32 accumulator O  ~56 MiB
score tile [56,2048,512]   BF16 ~112 MiB / FP32 ~224 MiB（V0 会显式物化）
```

合计约 **200–350MB 级**，而非完整序列的数 GB 峰值。显存基本不随 sequence length 增长，由 tile size 决定——这是冲击 4GB 以下显存的关键。

## 5. 性能瓶颈：PCIe 带宽

这条路线最大的敌人不是 RAM，而是 PCIe。每个 Q block 都要重扫一次完整 K/V：

```text
K+V ≈ 410 MB（15k token）
q_block = 2048 → 约 8 个 Q blocks
每个 Transformer block: 410MB × 8 ≈ 3.3GB H2D
50 blocks: ≈ 165GB / denoise step
PCIe 4.0 x16 实际约 24GB/s → 约 6.9s（仅 KV 传输理论下限）
```

**优化重点：让 Q block 尽量大。**

```text
q_block = 8192 → 约 2 个 Q blocks
KV 流量: 410MB × 2 ≈ 820MB / block
50 blocks ≈ 41GB / denoise step → PCIe 4 理论约 1.7s
```

但纯 PyTorch reference 会显式物化 `QK^T` score tile（`q=8192, kv=512, heads=56` 已非常巨大），Q block 做不大。

结论：本方案**显存极强，性能很可能显著下降**；性能优化的关键在于 fused kernel + 大 Q block。

## 6. 实现路线

### V0：纯 PyTorch reference

- `q_block = 512 / 1024 / 2048`，`kv_block = 256 / 512`。
- 目的只有两个：验证数学完全一致；验证整条 out-of-core pipeline 能跑。

### V1：fused streaming kernel（Triton 或 CUDA）

```text
Q tile × K/V tile → QK → softmax update → PV，不把 score tile 写到 HBM
```

Q block 可显著增大（4096 / 8192），这才是真正的性能版。

## 7. 落地改动清单（4 个核心文件）

### 7.1 新建 `diffsynth/core/attention/streaming.py`

```python
def streaming_attention_reference(
    q_cpu, k_cpu, v_cpu,        # [N, H, D]，CPU pinned 张量
    cu_seqlens,                 # int32 segment 边界
    q_block_size, kv_block_size,
    device, scale,
) -> torch.Tensor:              # out_cpu [N, H, D]
```

- API 直接围绕 online state（m/l/O）设计；外层按 `cu_seqlens` 逐 segment，segment 内切 Q/KV tile。
- 以后若要利用 FA2/3 的 LSE 接口，再专门做 `streaming_flash_attention(...)`。

### 7.2 `diffsynth/core/vram/layers.py`：weight lease

disk 量化模块目前在 `forward()` 时临时构建 computation module，每个 chunk 都重载权重。增加：

```python
with qkv_proj.computation_lease():
    for hidden_chunk:
        qkv_proj(hidden_chunk)
```

使权重 SSD → GPU 每 block 只发生一次。一个 block 的 weight schedule：

```text
qkv weight → 所有 projection chunks → release
attention tiles（无权重）
out_proj weight → 所有 Q chunks → release
fc1 → 所有 chunks → release
fc2 → 所有 chunks → release
```

`qkv_proj`/`out_proj`/`fc1`/`fc2` 均为 `nn.Linear`，只需在 `AutoWrappedLinear` 上实现；无 vram 管理的普通 `nn.Linear` 由调用侧用 `getattr(layer, "computation_lease", nullcontext)` 兼容。

### 7.3 `diffsynth/models/minimax_h3_dit.py`：out-of-core 执行调度

- `MiniMaxH3Attention` 增加 chunked projection（Phase A）与 streaming attention + chunked out_proj（Phase B/C）方法。
- `MiniMaxH3MLP` 增加两 pass streaming forward（Phase D1/D2）。
- `MiniMaxH3DiTBlock` 增加 `forward_streaming`，串联四个阶段，CPU buffer 复用，全程 `torch.inference_mode()`。
- `MiniMaxH3DiT.forward` 新增 `activation_streaming` 及 block size 参数；开启时 `_embed` 输出搬 CPU（pinned）、绕过 `gradient_checkpoint_forward` 直接逐 block streaming，最后搬回 GPU 走原有 `final_layer`。

### 7.4 `diffsynth/pipelines/minimax_h3_audio_video.py`：参数暴露

```python
pipe(
    ...,
    activation_streaming=True,
    projection_chunk_size=2048,
    attention_q_block_size=2048,      # V0 保守值；V1 fused kernel 后可到 8192
    attention_kv_block_size=512,
    activation_offload_device="cpu",
    streaming_attention_backend="torch",   # 后续可切 "triton"
)
```

沿 `__call__` → `inputs_shared` → `model_fn_minimax_h3` → `dit(...)` 透传。

## 8. 约束与边界

- **仅推理**：输入 `requires_grad=True` 直接报错；与 `use_gradient_checkpointing*` 互斥（assert）。
- **exact，非近似**：online softmax 数学上与 full attention 等价，需用 varlen 多 segment 随机数据与 `_sdpa_varlen_attention` 做 parity 验证。
- token_refiner（文本短序列）保持原 GPU 路径。
- CPU buffer 用 pinned memory 加速 H2D/D2H；attention 结束后立即 `del Q/K/V` 复用 host RAM。
- 不改现有 attention 后端；所有新参数默认关闭，默认行为不变。
- 最终架构形态：

```text
Disk:  weights
CPU:   hidden / Q / K / V / mlp_intermediate / next_hidden
GPU:   当前算子权重 + Q tile + KV tile + online-softmax 状态 + MLP tile
```

一旦 V0 reference 跑通，后续主要工程问题不再是「能不能跑」，而是**如何把 CPU↔GPU traffic 和 attention kernel 做快**。

### V0 实现状态（2026-08-17）

- 已实现按 `cu_seqlens` 分 segment 的 FP32 online-softmax reference。
- 已实现 QKV projection、out projection 和 two-pass MLP 的 sequence chunk 调度。
- 已实现普通 Linear 与 NF4 quantized wrapper 的可重入 computation lease。
- pipeline 参数默认关闭，原有 full-sequence 路径保持不变。
- 单元测试覆盖 attention parity、完整 DiT block parity 和 lease 生命周期。
- 真实 NF4 前 5 层、480×832、124 帧在 4GiB allocator 硬上限下完成 1 个 denoise step；
  `projection/q/kv = 512/512/256` 时，PyTorch allocated 峰值约 1865MiB，reserved
  峰值约 2134MiB。该结果用于 V0 工程验证，不代表完整 50 层性能。

## 9. 评测方案

V0 的 baseline、正确性验证、4GB 显存模拟、长序列扫长和 chunk 消融实验见：

- [MiniMax-H3 V0 序列 Chunk 化对比实验方案](./minimax_h3_streaming_benchmark_plan.md)
- [MiniMax-H3 V0：8GB、61K Tokens 端到端视频生成实验说明](./minimax_h3_8gb_61k_end_to_end_experiment.md)
