from __future__ import annotations

import os
import subprocess
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_module_storage_summary(module) -> dict:
    """Return unique CUDA storage held directly by a module."""
    seen = set()
    storage_bytes = 0
    tensor_count = 0
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor is None or tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        storage_bytes += storage.nbytes()
        tensor_count += 1

    wrapper_states = Counter()
    preparing_enabled = Counter()
    for child in module.modules():
        attributes = vars(child)
        if "state" not in attributes or "preparing_enabled" not in attributes:
            continue
        wrapper_states[str(attributes["state"])] += 1
        preparing_enabled[str(bool(attributes["preparing_enabled"]))] += 1
    return {
        "dit_cuda_storage_mib": storage_bytes / 2**20,
        "dit_cuda_tensor_count": tensor_count,
        "dit_wrapper_state_counts": dict(wrapper_states),
        "dit_wrapper_preparing_enabled_counts": dict(preparing_enabled),
    }


class StepTimer:
    def __init__(
        self,
        iterable,
        sampler,
        memory_probe=None,
        profile_nvtx: bool = False,
        profile_capture_step: int | None = None,
    ):
        self.iterable = iterable
        self.sampler = sampler
        self.memory_probe = memory_probe
        self.profile_nvtx = profile_nvtx
        self.profile_capture_step = profile_capture_step
        self.step_seconds = []
        self.step_memory = []
        self.peak_allocated_mib = None
        self.peak_reserved_mib = None

    def __iter__(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for item in self.iterable:
            synchronize()
            self.sampler.begin_window()
            started = time.perf_counter()
            step_index = len(self.step_seconds) + 1
            step_range = (
                torch.cuda.nvtx.range(f"h3:denoise_step_{step_index:02d}")
                if self.profile_nvtx
                else nullcontext()
            )
            capture_step = self.profile_capture_step == step_index
            if capture_step:
                torch.cuda.profiler.start()
            try:
                with step_range:
                    yield item
                synchronize()
            finally:
                if capture_step:
                    torch.cuda.profiler.stop()
            elapsed = time.perf_counter() - started
            step_peaks = self.sampler.end_window()
            self.step_seconds.append(elapsed)
            nvml_mib, rss_mib = self.sampler.sample()
            memory = {
                "step": len(self.step_seconds),
                "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
                "nvml_process_mib": nvml_mib,
                "cpu_rss_mib": rss_mib,
                "nvml_process_peak_mib": step_peaks["nvml_process_peak_mib"],
                "cpu_rss_peak_mib": step_peaks["cpu_rss_peak_mib"],
                "nvml_sample_count": step_peaks["sample_count"],
            }
            if self.memory_probe is not None:
                memory.update(self.memory_probe())
            self.step_memory.append(memory)
            print(f"BENCH_STEP {len(self.step_seconds)} {elapsed:.6f}s", flush=True)
            print(
                "BENCH_STEP_MEMORY "
                f"{memory['step']} "
                f"allocated={memory['torch_allocated_mib']:.1f}MiB "
                f"reserved={memory['torch_reserved_mib']:.1f}MiB "
                f"nvml_end={memory['nvml_process_mib']:.1f}MiB "
                f"nvml_peak={memory['nvml_process_peak_mib']:.1f}MiB "
                f"samples={memory['nvml_sample_count']} "
                f"rss={memory['cpu_rss_mib']:.1f}MiB "
                f"dit_cuda={memory.get('dit_cuda_storage_mib', 0.0):.1f}MiB",
                flush=True,
            )
        synchronize()
        self.peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
        self.peak_reserved_mib = torch.cuda.max_memory_reserved() / 2**20


def timed_method(
    obj,
    method_name: str,
    samples: list[float],
    *,
    capture_key: str | None = None,
    captured_latents: dict | None = None,
    capture_path: str | Path | None = None,
) -> None:
    original = getattr(obj, method_name)
    resolved_capture_path = Path(capture_path) if capture_path is not None else None

    def wrapper(*args, **kwargs):
        if capture_key is not None and captured_latents is not None and args:
            captured = args[0].detach().to("cpu")
            captured_latents[capture_key] = captured
            if resolved_capture_path is not None:
                temporary_path = resolved_capture_path.with_suffix(
                    resolved_capture_path.suffix + ".tmp"
                )
                torch.save(captured, temporary_path)
                os.replace(temporary_path, resolved_capture_path)
                print(
                    f"BENCH_TENSOR_SAVED {capture_key} {resolved_capture_path}",
                    flush=True,
                )
        synchronize()
        started = time.perf_counter()
        output = original(*args, **kwargs)
        synchronize()
        samples.append(time.perf_counter() - started)
        return output

    setattr(obj, method_name, wrapper)


def gpu_info() -> list[dict]:
    query = "index,name,uuid,driver_version,memory.total,power.limit"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": float(fields[4]),
                "power_limit_w": float(fields[5]),
            }
        )
    return rows


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": sum(values) / len(values),
        "min_seconds": ordered[0],
        "median_seconds": ordered[len(ordered) // 2],
        "max_seconds": ordered[-1],
        "per_step_seconds": values,
    }
