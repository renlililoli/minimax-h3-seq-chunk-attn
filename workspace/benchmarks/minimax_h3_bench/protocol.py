from __future__ import annotations

import json
import csv
import gzip
import importlib.metadata
import os
import platform
import resource
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import pynvml
except ImportError:  # pragma: no cover - fallback for non-benchmark dev hosts
    pynvml = None


STATUSES = {
    "success", "oom", "budget_exceeded", "timeout",
    "numerical_failure", "runtime_error",
}


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_nvml_handles = {}


def _pid_gpu_memory_mib(pid: int) -> float:
    physical_gpu_index = int(os.environ.get("BENCH_NVML_GPU_INDEX", "0"))
    if pynvml is not None:
        try:
            if physical_gpu_index not in _nvml_handles:
                pynvml.nvmlInit()
                _nvml_handles[physical_gpu_index] = pynvml.nvmlDeviceGetHandleByIndex(
                    physical_gpu_index
                )
            processes = list(
                pynvml.nvmlDeviceGetComputeRunningProcesses(
                    _nvml_handles[physical_gpu_index]
                )
            )
            return sum(
                float(process.usedGpuMemory) / 2**20
                for process in processes
                if process.pid == pid and process.usedGpuMemory is not None
            )
        except pynvml.NVMLError:
            pass
    query = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=False,
    )
    if query.returncode:
        return 0.0
    total = 0.0
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            total += float(fields[1])
    return total


def _rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError):
        pass
    return 0.0


class ProcessSampler:
    def __init__(
        self,
        pid: int | None = None,
        interval_seconds: float = 0.02,
        record_trace: bool = False,
    ):
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.pid = os.getpid() if pid is None else pid
        self.interval_seconds = interval_seconds
        self.nvml_peak_mib = 0.0
        self.rss_peak_mib = 0.0
        self.sample_count = 0
        self._window_active = False
        self._window_index = 0
        self._window_nvml_peak_mib = 0.0
        self._window_rss_peak_mib = 0.0
        self._window_sample_count = 0
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._trace_started_ns = time.perf_counter_ns()
        self._record_trace = record_trace
        self._trace = []

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_seconds)

    def sample(self) -> tuple[float, float]:
        with self._sample_lock:
            nvml_mib = _pid_gpu_memory_mib(self.pid)
            rss_mib = _rss_mib(self.pid)
            self.sample_count += 1
            self.nvml_peak_mib = max(self.nvml_peak_mib, nvml_mib)
            self.rss_peak_mib = max(self.rss_peak_mib, rss_mib)
            if self._window_active:
                self._window_nvml_peak_mib = max(
                    self._window_nvml_peak_mib, nvml_mib
                )
                self._window_rss_peak_mib = max(
                    self._window_rss_peak_mib, rss_mib
                )
                self._window_sample_count += 1
            if self._record_trace:
                self._trace.append((
                    (time.perf_counter_ns() - self._trace_started_ns) / 1_000_000.0,
                    self._window_index if self._window_active else 0,
                    nvml_mib,
                    rss_mib,
                ))
            return nvml_mib, rss_mib

    def begin_window(self) -> None:
        """Start a sampling window used to attribute peaks to one denoise step."""
        with self._sample_lock:
            if self._window_active:
                raise RuntimeError("a process-sampling window is already active")
            self._window_active = True
            self._window_index += 1
            self._window_nvml_peak_mib = 0.0
            self._window_rss_peak_mib = 0.0
            self._window_sample_count = 0

    def end_window(self) -> dict[str, float | int]:
        """Finish the current window after taking a synchronized final sample."""
        self.sample()
        with self._sample_lock:
            if not self._window_active:
                raise RuntimeError("no process-sampling window is active")
            result = {
                "nvml_process_peak_mib": self._window_nvml_peak_mib,
                "cpu_rss_peak_mib": self._window_rss_peak_mib,
                "sample_count": self._window_sample_count,
            }
            self._window_active = False
            return result

    def write_trace_csv_gz(self, path: str | Path) -> int:
        """Write raw samples without inflating the main benchmark JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("elapsed_ms", "denoise_step", "nvml_process_mib", "cpu_rss_mib"))
            writer.writerows(self._trace)
        return len(self._trace)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self.sample()


@dataclass(frozen=True)
class VramBudget:
    target_mib: int
    context_mib: float
    safety_margin_mib: int
    allocator_limit_mib: float
    allocator_fraction: float
    vram_limit_gib: float


def initialize_vram_budget(target_mib: int, safety_margin_mib: int = 128) -> VramBudget:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a VRAM-budgeted benchmark")
    torch.cuda.init()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    context_mib = _pid_gpu_memory_mib(os.getpid())
    physical_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    allocator_limit_mib = target_mib - context_mib - safety_margin_mib
    if allocator_limit_mib <= 0:
        raise RuntimeError(
            f"CUDA context ({context_mib:.1f} MiB) leaves no allocator budget under {target_mib} MiB"
        )
    fraction = min(1.0, allocator_limit_mib / physical_mib)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return VramBudget(
        target_mib=target_mib,
        context_mib=context_mib,
        safety_margin_mib=safety_margin_mib,
        allocator_limit_mib=allocator_limit_mib,
        allocator_fraction=fraction,
        vram_limit_gib=target_mib / 2048.0,
    )


def classify_exception(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "oom"
    if "numerical" in message or "nan" in message or "inf" in message:
        return "numerical_failure"
    return "runtime_error"


def environment_metadata() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    return {
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_driver": driver[0].strip() if driver else None,
        "bitsandbytes": package_version("bitsandbytes"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_commit": os.environ.get("MINIMAX_H3_MODEL_COMMIT") or git("rev-parse", "HEAD"),
        "diffsynth_commit": os.environ.get("MINIMAX_H3_DIFFSYNTH_COMMIT") or git(
            "-C", "extern/DiffSynth-Studio", "rev-parse", "HEAD"
        ),
        "model_dirty": os.environ.get("MINIMAX_H3_MODEL_DIRTY"),
        "diffsynth_dirty": os.environ.get("MINIMAX_H3_DIFFSYNTH_DIRTY"),
        "container_image": os.environ.get("MINIMAX_H3_CONTAINER_IMAGE"),
        "numa_policy": subprocess.run(
            ["numactl", "--show"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def finish_result(result: dict, sampler: ProcessSampler, budget: VramBudget | None) -> dict:
    result["nvml_process_peak_mib"] = sampler.nvml_peak_mib
    result["cpu_rss_peak_mib"] = sampler.rss_peak_mib
    if torch.cuda.is_available():
        result["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    if budget is not None:
        result.update({
            "target_vram_mib": budget.target_mib,
            "allocator_limit_mib": budget.allocator_limit_mib,
            "context_mib": budget.context_mib,
            "vram_limit_gib": budget.vram_limit_gib,
        })
        if result.get("status") == "success" and sampler.nvml_peak_mib > budget.target_mib:
            result["status"] = "budget_exceeded"
            result["failure_message"] = (
                f"NVML process peak {sampler.nvml_peak_mib:.1f} MiB exceeded "
                f"target {budget.target_mib} MiB"
            )
    if result["status"] not in STATUSES:
        raise ValueError(f"invalid benchmark status: {result['status']}")
    return result
