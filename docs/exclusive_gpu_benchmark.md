# Exclusive-GPU MiniMax-H3 benchmark

This procedure separates administrator-only GPU reservation from the actual
benchmark.  The normal user never needs `sudo` during model execution.

## Why this is necessary

`CUDA_VISIBLE_DEVICES` and Docker `--gpus device=N` select a GPU but do not
reserve it.  Another user can still create a CUDA context on the same physical
device.  NVIDIA `EXCLUSIVE_PROCESS` compute mode is driver-enforced: after the
benchmark creates its context, a second compute process is rejected.

The benchmark also records a one-second whole-device audit containing compute
mode, total device memory, utilization, the Docker host PID, and every compute
PID reported by NVML.  The result is marked contaminated if a PID other than
the benchmark container appears.

## 1. Normal user: find a current idle candidate

GPU availability changes continuously on a shared node.  Do not permanently
assume that GPU 1 or GPU 3 is idle.  Run:

```bash
cd /home/grzhu/project/video/seq_offload
scripts/find_idle_gpu.sh
```

This lists compute mode, device memory, utilization, and every compute process.
An `IDLE` result is only a point-in-time observation and is not yet a
reservation.

Known topology defaults encoded by the runner are:

| GPU | CPU affinity | NUMA node |
|---:|---|---:|
| 0 | `64-95,320-351` | 2 |
| 1 | `224-255,480-511` | 7 |
| 2 | `192-223,448-479` | 6 |
| 3 | `160-191,416-447` | 5 |

NUMA nodes are read from each GPU's PCI sysfs `numa_node`; the CPU lists come
from `nvidia-smi topo -m`.

## 2. Administrator: reserve the selected idle GPU

For example, if the current idle candidate is GPU 1:

```bash
sudo scripts/admin_gpu_exclusive.sh enable 1
```

The script refuses to change compute mode if any compute process is already on
the target GPU.  A successful output must report:

```text
compute mode: Exclusive_Process
compute processes: none
```

Status can be checked without root:

```bash
scripts/admin_gpu_exclusive.sh status 1
```

`EXCLUSIVE_PROCESS` allows only one CUDA context, but the first process to open
the still-idle device wins.  Launch the benchmark immediately after the
administrator command.  If another process wins the race, the user runner
will refuse to start or CUDA context initialization will fail.

## 3. Normal user: validate and run

First perform a dry run:

```bash
scripts/run_native_h3_exclusive.sh --gpu 1 --dry-run
```

Then launch the exact native 720p/20s/50-step experiment:

```bash
scripts/run_native_h3_exclusive.sh --gpu 1
```

The runner requires all of the following before launch:

- target GPU is in `EXCLUSIVE_PROCESS` mode;
- target GPU has no existing compute process;
- the dedicated Docker container name does not already exist;
- model directory and image exist;
- CPU and memory affinity are defined.

The native reproduction intentionally does **not** set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  This keeps allocator
behavior identical to the original native run.  If fragmentation remains the
suspected cause after the exclusive run, run an explicitly named allocator
ablation separately.

## 4. Artifacts and validity

The script writes three host-visible files under
`workspace/benchmarks/results/`:

```text
<tag>_gpu<index>_<timestamp>.log
<tag>_gpu<index>_<timestamp>_gpu_audit.csv
<tag>_gpu<index>_<timestamp>_gpu_audit.txt
```

The model benchmark additionally writes its normal JSON and 2ms PID-level
memory trace.  A run is usable as an exclusive result only if:

```text
compute_mode=Exclusive_Process
foreign_process_detected=0
```

The CSV is independent evidence that the device stayed exclusive.  PID-level
NVML from the Python benchmark remains the source of truth for the benchmark
process itself; the audit provides whole-device and foreign-process evidence.

## 5. Administrator: restore the GPU

After the container exits and the user has removed it, restore normal sharing:

```bash
sudo scripts/admin_gpu_exclusive.sh disable 1
```

The restore command also refuses to change mode while a compute process is
active.

## 6. Interpreting the rerun

- If native completes 50 steps exclusively, the previous OOM must not be used
  as a native capacity conclusion; investigate shared state and rerun both
  modes under the same exclusive protocol.
- If native OOMs again with `foreign_process_detected=0`, shared-GPU contention
  is ruled out for that run.  Compare its per-step Torch allocated/reserved and
  PID NVML trace with the previous result.
- If `reserved >> allocated` still grows before OOM, run a separate
  `expandable_segments:True` allocator ablation.  Do not silently replace the
  native baseline with the ablation.
