# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository role

This repository is an integration and experiment harness, not an installable root Python package. It combines:

- `extern/DiffSynth-Studio`: patched MiniMax-H3 model and pipeline integration.
- `extern/seqattn`: standalone exact attention runtime (a separate git submodule/package).
- `workspace/benchmarks`: end-to-end and formal benchmark drivers plus generated evidence.
- `scripts`: GPU reservation and native-run operations.
- Root Docker files: the canonical integrated runtime.

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:renlililoli/minimax-h3-seq-chunk-attn.git
```

Both implementation directories are git submodules. Inspect parent and submodule status before changing revisions; do not reset or update a submodule implicitly. The `seqattn` submodule URL requires GitHub SSH access.

## Environment and setup

The canonical integrated environment is Docker Compose:

```bash
docker compose build diffsynth
docker compose up -d diffsynth
docker compose exec diffsynth bash
```

`Dockerfile.diffsynth` installs both submodules editable. `docker-compose.yml` mounts their source trees and `workspace`, reserves GPU 0, requests 128 GiB shared memory, and maps `/scratch/grzhu/weights/video` to `/models`. It depends on a locally available `comfyui:cu128` image, NVIDIA Container Toolkit, and the host-specific weights path.

Local standalone setup:

```bash
cd extern/seqattn
pip install -e '.[cuda,benchmark,dev]'

cd ../DiffSynth-Studio
pip install -e .
```

Both packages require Python 3.10+. `seqattn` additionally targets Linux, CUDA, PyTorch 2.5+, and Triton 3.1+.

Download the NF4 model components with `bash workspace/scripts/download-nf4.sh`; the script writes to the host-specific `/scratch/grzhu/weights/video` hierarchy.

## Tests and quality checks

There is no root test suite. Run checks in the relevant submodule.

### `seqattn`

```bash
cd extern/seqattn
pytest -q
pytest -q tests/test_planner.py
pytest -q tests/test_planner.py::test_name
ruff check .
ruff format --check .
```

Ruff is configured for Python 3.10 with a 100-column line length. CPU/reference/planner tests work without CUDA; Triton and pipeline cases skip without CUDA/Triton, and direct-I/O cases may skip on filesystems without `O_DIRECT`.

Useful focused suites are `test_reference.py`, `test_triton.py`, `test_pipeline.py`, `test_paged_triton.py`, and `test_nvme.py`.

### DiffSynth integration

```bash
cd extern/DiffSynth-Studio
pytest -q
pytest -q tests/models/test_minimax_h3_streaming.py
pytest -q tests/models/test_minimax_h3_streaming.py::test_name
```

Other relevant suites are:

- `tests/core/attention/test_streaming.py`
- `tests/models/test_minimax_h3_video_vae.py`
- `tests/core/vram/test_computation_lease.py`

These tests are present but DiffSynth does not declare pytest or a pytest configuration; install test dependencies separately. CUDA, FlashAttention, and optional `seqattn` cases may skip.

To build either submodule package after installing `build`, run `python -m build` from that submodule. Do not run it at the repository root.

## Architecture and data flow

### MiniMax-H3 integration

`extern/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py` is the top-level runtime. It preprocesses inputs, constructs the packed multimodal sequence, runs the denoise loop, updates video/audio latents, and invokes the VAEs. The packed layout is:

```text
[text | conditions/references | audio | video | padding]
```

It aligns total length to 64 and carries modality positions, tags, RoPE coordinates, and `cu_seqlens` into `model_fn_minimax_h3` and the DiT.

`extern/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py` owns the 50-block DiT and both native and activation-streaming paths. In streaming mode it places the full packed hidden tensor in pinned CPU DRAM and reuses one compatible `seqattn` projected-attention runner across blocks.

Each streaming block proceeds as follows:

1. Hidden chunks move H2D; model callbacks perform normalization/AdaLN, fused QKV projection, Q/K normalization, and RoPE.
2. Full Q/K/V are stored in persistent pinned CPU buffers.
3. A global K/V readiness barrier occurs; exact global self-attention cannot finalize queries before all keys and values exist.
4. Resident Q super-blocks and streamed K/V tiles run Triton online softmax. Final GPU attention tiles pass directly through the model-owned output projection, gate, and residual callback before projected hidden returns to CPU.
5. The fused MLP keeps `fc1`, SiLU/gate, `fc2`, and residual/gate work tile-local on GPU; only completed hidden tiles return to the next CPU-backed hidden tensor.
6. The chunked final layer extracts only audio/video output rows.

After denoise step 1, DiffSynth freezes weight “preparing” so later steps do not grow the resident GPU weight set. Native and streaming modes both use CPU-backed weight offload; the principal difference is sequence-activation placement.

`diffsynth/core/attention/streaming.py` contains older generic Torch online-softmax and FlashAttention-LSE streaming backends. Do not route `seqattn` through `iter_streaming_attention`: `seqattn` is integrated at the model adapter so model-owned projection and output callbacks remain fused.

`diffsynth/models/minimax_h3_video_vae.py` implements a separate temporal-streaming VAE decode path. Its frame planning and chunk assembly are downstream of, and independent from, DiT sequence attention.

### Standalone `seqattn`

Public imports live in `extern/seqattn/src/seqattn`, which must remain a pure compatibility facade. Put new implementation code in `extern/seqattn/src/seqattn_core`; `tests/test_module_layout.py` enforces this boundary.

Key subsystems:

- `config.py`: memory-policy and execution dataclasses.
- `planner.py`: jointly selects resident-Q and streamed-K/V chunks under an operator workspace budget.
- `streaming/{runner,executor,workspace}.py`: contiguous pinned-DRAM execution and persistent resources.
- `projection/runner.py`: chunked QKV production, global barrier, attention, and caller-owned output-projection fusion.
- `kernels/streaming.py`: Triton kernels.
- `paged/` and `storage/`: bounded host-cache and memory/simulated-NVMe/physical-`O_DIRECT` storage tiers.

The MiniMax-H3 integration currently uses the contiguous pinned-DRAM projected path, not the paged/NVMe runtime.

Important invariants:

- Streaming is inference-only, CPU-activation-offload-only, and incompatible with gradient checkpointing.
- Preserve packed `cu_seqlens`; query and K/V tiles must not cross segment boundaries.
- `workspace_budget_bytes` limits only `seqattn`-owned HBM. It excludes CUDA context, model weights, caller activations, and other process allocations.
- Runners own persistent buffers, streams, and events; they are single-flight. Reuse a runner for a compatible shape/configuration rather than rebuilding it per block.
- Full K/V readiness is required for exact dense global attention.

See `extern/seqattn/docs/architecture.md` and `docs/minimax_h3_out_of_core_streaming.md` for design detail, but prefer current source and the root/`seqattn` READMEs when older documents describe the pre-`seqattn` V0 implementation.

## Running benchmarks

### End-to-end MiniMax-H3

`workspace/benchmarks/minimax_h3_baseline.py` drives both native and streaming runs despite its historical filename. It loads NF4 components, configures weight backing and memory limits, samples NVML/RSS, times denoise/VAE phases, and atomically writes JSON. `decode_saved_latents.py` can recover VAE decode and muxing without rerunning denoising.

A container smoke run:

```bash
docker compose exec -T --user 1091:1102 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  diffsynth python /workspace/benchmarks/minimax_h3_baseline.py \
  --height 480 --width 832 --frames 124 --steps 5 --seed 0 \
  --tag sim8_cpu_dit5 --simulated-vram-gib 8 --vram-reserve-gib 4 \
  --offload-device cpu --skip-decode --no-media
```

Use explicit tags and preserve each run's JSON metadata; do not mix end-to-end,
capacity, profiler, and formal microbenchmark measurements.

### Formal microbenchmarks

The formal protocol is `workspace/benchmarks/minimax_h3_bench`. Generate canonical tensors inside the container, but run `run_point` on the host:

```bash
# Inside the container
python -m benchmarks.minimax_h3_bench.canonical \
  --frames 22 39 56 73 90 124 158 192 226 260 \
  --output /workspace/benchmarks/artifacts/canonical_480x832_t256.pt

# On the host
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

Do not run the controller inside the container: `.git` is not mounted there, while the host controller injects revision provenance and guarantees a result JSON on success, failure, or timeout. See `workspace/benchmarks/README.md` for the protocol and each module’s `--help` for options.

Measurement rules:

- Treat JSON as the source of truth; stdout is diagnostic.
- Retain OOM, timeout, decode failure, and budget exceedance outcomes.
- Do not mix formal latency, phase-instrumented, Nsight, capacity-probe, or full-media measurements.
- Run comparison points in independent processes.
- Simulated NVMe is not physical-storage evidence; physical claims require measured local storage and `--formal-local-nvme`.
- A chunk selected by `select_chunks.py` still requires an independent winner retest.

### Standalone operator

After installing the benchmark extra, use `seqattn-bench`, `seqattn-pipeline-bench`, and `seqattn-paged-bench`. Example:

```bash
seqattn-bench --mode seqattn --tokens 61312 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --workspace-mib 4096 --target-vram-mib 8192 \
  --kv-chunk 4096 --repeats 1 \
  --output benchmark-results/seqattn_61312.json
```

Sweeps live in `extern/seqattn/benchmarks`; run them from that submodule. `extern/seqattn/scripts/profile_nsys.sh` runs an NVTX/Nsight profile, whose timings must not be reported as primary latency measurements.

### Exclusive native GPU runs

Follow `docs/exclusive_gpu_benchmark.md`:

```bash
scripts/find_idle_gpu.sh
sudo scripts/admin_gpu_exclusive.sh enable GPU_INDEX
scripts/admin_gpu_exclusive.sh status GPU_INDEX
scripts/run_native_h3_exclusive.sh --gpu GPU_INDEX --dry-run
scripts/run_native_h3_exclusive.sh --gpu GPU_INDEX
sudo scripts/admin_gpu_exclusive.sh disable GPU_INDEX
```

Only reservation changes require root. Do not silently add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the native baseline; allocator experiments must be separately named.

## Generated data and provenance

`workspace/benchmarks/results` and `workspace/benchmarks/artifacts` contain large generated evidence, not source. Avoid rewriting or committing outputs unintentionally.

Be explicit about which repository a git operation targets: the parent, upstream/patched DiffSynth, or standalone `seqattn`. Formal parent provenance may not identify locally modified submodule code, so record the actual submodule revisions when interpreting benchmark results.
