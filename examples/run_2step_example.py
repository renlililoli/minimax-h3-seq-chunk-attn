from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import gzip
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import pynvml
except ImportError:  # pragma: no cover - ComfyUI images normally include it
    pynvml = None

DEFAULT_TARGET_VRAM_MIB = 8192
DEFAULT_RESERVE_VRAM_GIB = 3.0
DEFAULT_AIMDO_WATERMARK_MARGIN_MIB = 2560.0
VRAM_BUDGET_SAFETY_MIB = 384


def initialize_aimdo_before_torch(argv: list[str]) -> float | None:
    if "-h" in argv or "--help" in argv:
        return None
    if pynvml is None:
        raise RuntimeError(
            "pynvml is required to initialize DynamicVRAM before torch"
        )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--target-vram-mib", type=int, default=DEFAULT_TARGET_VRAM_MIB
    )
    parser.add_argument(
        "--reserve-vram-gib", type=float, default=DEFAULT_RESERVE_VRAM_GIB
    )
    parser.add_argument(
        "--aimdo-watermark-margin-mib",
        type=float,
        default=DEFAULT_AIMDO_WATERMARK_MARGIN_MIB,
    )
    args, _ = parser.parse_known_args(argv)

    pynvml.nvmlInit()
    index = int(os.environ.get("SEQATTN_EXAMPLE_NVML_GPU_INDEX", "0"))
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    total_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**20
    requested_reserve_mib = args.reserve_vram_gib * 1024.0
    target_derived_headroom_mib = max(
        0.0,
        total_mib
        - args.target_vram_mib
        + VRAM_BUDGET_SAFETY_MIB
        + args.aimdo_watermark_margin_mib,
    )
    effective_headroom_mib = max(
        requested_reserve_mib, target_derived_headroom_mib
    )

    import comfy_aimdo.control as aimdo_control

    headroom_bytes = int(effective_headroom_mib * 2**20)
    try:
        initialized = aimdo_control.init(
            simple_vram_headroom=headroom_bytes,
            nvml_pressure=True,
        )
    except TypeError:
        initialized = aimdo_control.init(simple_vram_headroom=headroom_bytes)
    if not initialized:
        raise RuntimeError(
            "ComfyUI DynamicVRAM control initialization failed before torch import"
        )
    return effective_headroom_mib


EARLY_AIMDO_HEADROOM_MIB = (
    initialize_aimdo_before_torch(sys.argv[1:])
    if __name__ == "__main__"
    else None
)

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def release_unused_host_memory() -> None:
    gc.collect()
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if callable(host_empty_cache):
        host_empty_cache()
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_COMFYUI_COMMIT = "9a9fdb10ed144ce760d9682cb247526ea23cc525"
ASSET_ROOT = PACKAGE_ROOT / "assets" / "benchmark"

SCENARIOS = {
    "t2va": {
        "width": 1344,
        "height": 768,
        "frames": 56,
        "model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "prompt": (
            "A cinematic sunrise over a quiet coastal town. Warm light moves "
            "across tiled roofs while seabirds circle above the harbor and a "
            "small sailboat crosses the water. One continuous stable camera "
            "move, natural motion, realistic detail, no cuts, no text, no logos."
        ),
    },
    "fl2va": {
        "width": 1344,
        "height": 768,
        "frames": 56,
        "model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "first_frame": ASSET_ROOT / "fl2va_first_frame_1344x768.png",
        "last_frame": ASSET_ROOT / "fl2va_last_frame_1344x768.png",
        "prompt": (
            "Create a smooth continuous transition from the supplied first "
            "frame to the supplied last frame. Preserve the woman's identity, "
            "the pink fairy-tale garden, lighting, perspective, and camera "
            "continuity. Natural motion, stable anatomy, no cuts, no text, "
            "no logos."
        ),
    },
    "ref2va_images": {
        "width": 1344,
        "height": 768,
        "frames": 124,
        "model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "ref_images": [
            ASSET_ROOT / "fl2va_first_frame_1344x768.png",
            ASSET_ROOT / "fl2va_last_frame_1344x768.png",
        ],
        "prompt": (
            "Use <Picture 1> and <Picture 2> as identity, wardrobe, palette, "
            "and environment references. The same woman walks through the "
            "pink enchanted garden, pauses beside a crystal flower, then "
            "turns toward the camera. One coherent five-second shot, stable "
            "identity, natural motion, no cuts, no text, no logos."
        ),
    },
    "ref2va_video": {
        "width": 1344,
        "height": 768,
        "frames": 124,
        "model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "ref_video": ASSET_ROOT / "ref2va_reference_1344x768_124f.mp4",
        "prompt": (
            "Use <Video 1> as the exact reference for camera motion, timing, "
            "existing subjects, scene geometry, and continuity. Preserve the "
            "original shot while adding one clearly visible woman in a red "
            "jacket who enters from the right, points toward the main object, "
            "then turns and waves. Photorealistic, stable identity, no cuts, "
            "no text, no logos."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real two-step MiniMax H3 community example."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--comfyui-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-vram-mib", type=int, default=DEFAULT_TARGET_VRAM_MIB
    )
    parser.add_argument(
        "--reserve-vram-gib",
        type=float,
        default=DEFAULT_RESERVE_VRAM_GIB,
        help=(
            "Minimum ComfyUI/AIMDO reserve. The effective reserve is raised "
            "as needed to keep process VRAM under --target-vram-mib."
        ),
    )
    parser.add_argument(
        "--aimdo-watermark-margin-mib",
        type=float,
        default=DEFAULT_AIMDO_WATERMARK_MARGIN_MIB,
        help="Additional reserve for transient DynamicVRAM fault batches.",
    )
    parser.add_argument("--sample-interval-ms", type=float, default=50.0)
    parser.add_argument("--prompt")
    parser.add_argument("--audio-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-audio-decode", action="store_true")
    return parser.parse_args()


def infer_comfyui_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    environment = os.environ.get("COMFYUI_DIR")
    if environment:
        return Path(environment).resolve()
    if PACKAGE_ROOT.parent.name == "custom_nodes":
        return PACKAGE_ROOT.parent.parent.resolve()
    candidate = Path("/opt/ComfyUI")
    if candidate.is_dir():
        return candidate
    raise ValueError(
        "cannot infer ComfyUI; set COMFYUI_DIR or pass --comfyui-dir"
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_revision(path: Path) -> str:
    repository = path.resolve()
    top_level = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "--show-toplevel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if top_level.returncode != 0:
        return "unknown"
    if Path(top_level.stdout.strip()).resolve() != repository:
        return "unknown"
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


class MemorySampler:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.rows = []
        self.nvml_peak_mib = 0.0
        self.rss_peak_mib = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._handle = None
        if pynvml is not None:
            pynvml.nvmlInit()
            index = int(os.environ.get("SEQATTN_EXAMPLE_NVML_GPU_INDEX", "0"))
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def _gpu_mib(self) -> float:
        if self._handle is None:
            return 0.0
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
            return sum(
                float(process.usedGpuMemory) / 2**20
                for process in processes
                if process.pid == self.pid and process.usedGpuMemory is not None
            )
        except pynvml.NVMLError:
            return 0.0

    def _rss_mib(self) -> float:
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        except (FileNotFoundError, PermissionError):
            pass
        return 0.0

    def sample(self) -> dict[str, float]:
        gpu_mib = self._gpu_mib()
        rss_mib = self._rss_mib()
        allocated_mib = (
            torch.cuda.memory_allocated() / 2**20
            if torch.cuda.is_initialized()
            else 0.0
        )
        reserved_mib = (
            torch.cuda.memory_reserved() / 2**20
            if torch.cuda.is_initialized()
            else 0.0
        )
        row = {
            "elapsed_ms": (time.perf_counter() - self._started) * 1000.0,
            "nvml_process_mib": gpu_mib,
            "cpu_rss_mib": rss_mib,
            "torch_allocated_mib": allocated_mib,
            "torch_reserved_mib": reserved_mib,
        }
        with self._lock:
            self.rows.append(row)
            self.nvml_peak_mib = max(self.nvml_peak_mib, gpu_mib)
            self.rss_peak_mib = max(self.rss_peak_mib, rss_mib)
        return row

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self.sample()

    def interval_stats(self, start_ms: float, end_ms: float) -> dict:
        with self._lock:
            rows = [
                row
                for row in self.rows
                if start_ms <= row["elapsed_ms"] <= end_ms
            ]
        return {
            "sample_count": len(rows),
            "nvml_process_peak_mib": max(
                (row["nvml_process_mib"] for row in rows), default=0.0
            ),
            "cpu_rss_peak_mib": max(
                (row["cpu_rss_mib"] for row in rows), default=0.0
            ),
        }

    def write_csv_gz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self.rows)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def initialize_vram_budget(target_mib: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.init()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    physical_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    nvml_total_mib = physical_mib
    context_mib = 0.0
    if pynvml is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            int(os.environ.get("SEQATTN_EXAMPLE_NVML_GPU_INDEX", "0"))
        )
        nvml_total_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**20
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == os.getpid() and process.usedGpuMemory is not None:
                context_mib += float(process.usedGpuMemory) / 2**20
    safety_mib = VRAM_BUDGET_SAFETY_MIB
    allocator_limit_mib = target_mib - context_mib - safety_mib
    if allocator_limit_mib <= 0:
        raise RuntimeError("CUDA context leaves no memory under target")
    fraction = min(1.0, allocator_limit_mib / physical_mib)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {
        "target_mib": target_mib,
        "context_mib": context_mib,
        "safety_mib": safety_mib,
        "allocator_limit_mib": allocator_limit_mib,
        "allocator_fraction": fraction,
        "physical_mib": physical_mib,
        "nvml_total_mib": nvml_total_mib,
    }


def install_budget_aware_free_memory(model_management, budget: dict) -> None:
    original = model_management.get_free_memory

    def get_free_memory(device=None, torch_free_too=False):
        device = model_management.get_torch_device() if device is None else device
        if getattr(device, "type", None) != "cuda":
            return original(device, torch_free_too)
        stats = torch.cuda.memory_stats(device)
        active = stats["active_bytes.all.current"]
        reserved = stats["reserved_bytes.all.current"]
        limit = int(budget["allocator_limit_mib"] * 2**20)
        free_torch = max(0, reserved - active)
        free_unreserved = max(0, limit - reserved)
        total = free_torch + free_unreserved
        return (total, free_torch) if torch_free_too else total

    model_management.get_free_memory = get_free_memory


def load_image(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).div_(255.0).unsqueeze(0)


def load_video(path: Path, frames: int) -> tuple[torch.Tensor, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    decoded = []
    source_fps = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            decoded.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if len(decoded) == frames:
                break
    if len(decoded) < frames:
        raise ValueError(f"decoded {len(decoded)} frames, expected {frames}")
    tensor = torch.stack(decoded).to(torch.float32).div_(255.0)
    return tensor, {
        "path": str(path),
        "frames": len(decoded),
        "height": int(tensor.shape[1]),
        "width": int(tensor.shape[2]),
        "source_fps": source_fps,
    }


def probe_media(path: Path) -> dict:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"output has no video stream: {path}")
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        result = {
            "duration_seconds": (
                float(container.duration / av.time_base)
                if container.duration is not None
                else None
            ),
            "video": {
                "codec": video_stream.codec_context.name,
                "width": video_stream.codec_context.width,
                "height": video_stream.codec_context.height,
                "fps": (
                    float(video_stream.average_rate)
                    if video_stream.average_rate is not None
                    else None
                ),
                "container_frames": int(video_stream.frames),
            },
            "audio": None,
        }
        if audio_stream is not None:
            result["audio"] = {
                "codec": audio_stream.codec_context.name,
                "sample_rate": audio_stream.codec_context.sample_rate,
                "channels": audio_stream.codec_context.channels,
            }

    with av.open(str(path)) as container:
        result["video"]["decoded_frames"] = sum(
            1 for _frame in container.decode(video=0)
        )
    return result


def build_packed_layout(PackedLayout, positive, latent, frame_count: int) -> dict:
    video_latent, audio_latent = latent["samples"].unbind()
    conditioning, metadata = positive[0]
    layout = PackedLayout(
        int(conditioning.shape[1]),
        int(video_latent.shape[2]),
        int(video_latent.shape[3]),
        int(video_latent.shape[4]),
        int(audio_latent.shape[-1]),
        keyframes=metadata.get("minimax_keyframes"),
        refs=metadata.get("minimax_refs"),
        frame_count=metadata.get("minimax_frame_count", frame_count),
    )
    return {
        "tokens": int(layout.seq_len),
        "segments": [
            {"start": int(start), "stop": int(stop), "kind": kind}
            for start, stop, kind in layout.segments
        ],
        "latent_shapes": [
            [int(value) for value in item.shape]
            for item in latent["samples"].unbind()
        ],
    }


def main() -> int:
    args = parse_args()
    scenario = dict(SCENARIOS[args.scenario])
    width = args.width or scenario["width"]
    height = args.height or scenario["height"]
    frames = args.frames or scenario["frames"]
    prompt = args.prompt or scenario["prompt"]
    comfyui_dir = infer_comfyui_dir(args.comfyui_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    trace_path = output_dir / "memory_trace.csv.gz"
    weight_schedule_path = output_dir / "weight_schedule.json.gz"
    video_path = output_dir / "output.mp4"

    result = {
        "status": "running",
        "scenario": args.scenario,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "width": width,
            "height": height,
            "frames": frames,
            "steps": args.steps,
            "seed": args.seed,
            "target_vram_mib": args.target_vram_mib,
            "aimdo_watermark_margin_mib": args.aimdo_watermark_margin_mib,
            "audio_device": args.audio_device,
            "prompt": prompt,
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "package_root": str(PACKAGE_ROOT),
            "package_commit": git_revision(PACKAGE_ROOT),
            "comfyui_dir": str(comfyui_dir),
            "comfyui_commit": git_revision(comfyui_dir),
        },
        "phases": [],
    }
    video_path.unlink(missing_ok=True)
    trace_path.unlink(missing_ok=True)
    weight_schedule_path.unlink(missing_ok=True)
    atomic_write_json(result_path, result)

    sampler = None
    runtime = None
    original_cwd = Path.cwd()
    exit_code = 0
    try:
        actual_comfyui_commit = result["environment"]["comfyui_commit"]
        if actual_comfyui_commit != SUPPORTED_COMFYUI_COMMIT:
            raise RuntimeError(
                "MiniMax H3 SeqAttn examples require ComfyUI commit "
                f"{SUPPORTED_COMFYUI_COMMIT}; found {actual_comfyui_commit}"
            )
        budget = initialize_vram_budget(args.target_vram_mib)
        requested_reserve_mib = args.reserve_vram_gib * 1024.0
        aimdo_watermark_margin_mib = args.aimdo_watermark_margin_mib
        target_derived_headroom_mib = max(
            0.0,
            budget["nvml_total_mib"]
            - args.target_vram_mib
            + budget["safety_mib"]
            + aimdo_watermark_margin_mib,
        )
        effective_headroom_mib = max(
            requested_reserve_mib, target_derived_headroom_mib
        )
        result["memory_budget"] = budget
        result["memory_budget"].update(
            {
                "requested_reserve_mib": requested_reserve_mib,
                "target_derived_aimdo_headroom_mib": (
                    target_derived_headroom_mib
                ),
                "aimdo_watermark_margin_mib": aimdo_watermark_margin_mib,
                "effective_aimdo_headroom_mib": effective_headroom_mib,
                "aimdo_device_extra_headroom_mib": 0.0,
            }
        )
        sampler = MemorySampler(args.sample_interval_ms / 1000.0)
        sampler.start()

        sys.argv = [
            sys.argv[0],
            "--lowvram",
            "--reserve-vram",
            str(effective_headroom_mib / 1024.0),
        ]
        sys.path.insert(0, str(PACKAGE_ROOT))
        sys.path.insert(0, str(comfyui_dir))
        os.chdir(comfyui_dir)

        import comfy.memory_management as memory_management
        import comfy.model_management as model_management
        import comfy.model_patcher as model_patcher
        import comfy_aimdo.control as aimdo_control
        import nodes as comfy_nodes
        from comfy.ldm.minimax.model import PackedLayout
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider,
            BasicScheduler,
            KSamplerSelect,
            RandomNoise,
            SamplerCustomAdvanced,
        )
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        from comfy_extras.nodes_video import CreateVideo

        from comfyui_seqattn.nodes import (
            STATE_KEY,
            MiniMaxH3ReferenceToVideoSeqAttn,
            patch_minimax_h3_model,
        )
        from comfyui_seqattn.qwen import (
            QWEN_STATE_KEY,
            patch_minimax_h3_qwen_clip,
        )
        from comfyui_seqattn.vae import patch_minimax_h3_video_vae

        if EARLY_AIMDO_HEADROOM_MIB is None:
            raise RuntimeError(
                "run the example as a script so DynamicVRAM initializes "
                "before torch import"
            )
        if abs(EARLY_AIMDO_HEADROOM_MIB - effective_headroom_mib) > 1.0:
            raise RuntimeError(
                "DynamicVRAM early headroom does not match the runtime budget"
            )

        devices = model_management.get_all_torch_devices()
        try:
            aimdo_devices_initialized = aimdo_control.init_devices(
                (device.index, 0) for device in devices
            )
        except TypeError:
            aimdo_devices_initialized = aimdo_control.init_devices(
                device.index for device in devices
            )
        if not aimdo_devices_initialized:
            raise RuntimeError("ComfyUI DynamicVRAM device initialization failed")
        model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
        memory_management.aimdo_enabled = True
        result["dynamic_vram"] = {
            "devices_initialized": True,
            "devices": [str(device) for device in devices],
            "simple_headroom_mib": effective_headroom_mib,
            "device_extra_headroom_mib": 0.0,
            "core_model_patcher": model_patcher.CoreModelPatcher.__name__,
            "aimdo_enabled": bool(memory_management.aimdo_enabled),
        }

        install_budget_aware_free_memory(model_management, budget)

        def phase(name, function):
            torch.cuda.synchronize()
            start = time.perf_counter()
            start_ms = (start - sampler._started) * 1000.0
            try:
                return function()
            finally:
                torch.cuda.synchronize()
                end = time.perf_counter()
                end_ms = (end - sampler._started) * 1000.0
                result["phases"].append(
                    {
                        "name": name,
                        "seconds": end - start,
                        "start_elapsed_ms": start_ms,
                        "end_elapsed_ms": end_ms,
                        **sampler.interval_stats(start_ms, end_ms),
                    }
                )

        video_vae = phase(
            "video_vae_load",
            lambda: comfy_nodes.VAELoader().load_vae(
                "minimax_h3_video_vae_fp16.safetensors"
            )[0],
        )
        video_vae = patch_minimax_h3_video_vae(
            video_vae, tile_size=192, workspace_mib=512
        )

        previous_vram_state = model_management.vram_state
        model_management.vram_state = model_management.VRAMState.NORMAL_VRAM
        try:
            clip = phase(
                "qwen_load",
                lambda: comfy_nodes.CLIPLoader().load_clip(
                    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    type="minimax",
                    device="default",
                )[0],
            )
        finally:
            model_management.vram_state = previous_vram_state
        clip = patch_minimax_h3_qwen_clip(
            clip,
            activation_limit_mib=5888,
            max_conditioning_rows=25000,
            preflight_safety_mib=128,
            offload_mode="prefetch",
        )
        qwen_controller = getattr(clip, QWEN_STATE_KEY)
        original_preflight = qwen_controller.preflight

        def recording_preflight(tokens, _original_preflight=original_preflight):
            plan = _original_preflight(tokens)
            result["qwen_preflight"] = plan
            return plan

        qwen_controller.preflight = recording_preflight

        conditioning_inputs = {}
        if args.scenario == "fl2va":
            conditioning_inputs["first_frame"] = load_image(scenario["first_frame"])
            conditioning_inputs["last_frame"] = load_image(scenario["last_frame"])
            result["inputs"] = {
                "first_frame": str(scenario["first_frame"]),
                "last_frame": str(scenario["last_frame"]),
            }
        elif args.scenario == "ref2va_images":
            images = [load_image(path) for path in scenario["ref_images"]]
            conditioning_inputs["ref_images"] = {
                f"ref_image_{index}": image
                for index, image in enumerate(images)
            }
            result["inputs"] = {
                "ref_images": [str(path) for path in scenario["ref_images"]]
            }
        elif args.scenario == "ref2va_video":
            reference, metadata = phase(
                "reference_video_decode",
                lambda: load_video(scenario["ref_video"], frames),
            )
            conditioning_inputs["ref_videos"] = {"ref_video_0": reference}
            result["inputs"] = {"ref_video": metadata}

        def create_conditioning(
            _clip=clip,
            _video_vae=video_vae,
            _conditioning_inputs=conditioning_inputs,
        ):
            if args.scenario in {"t2va", "fl2va"}:
                return MiniMaxH3ImageToVideo.execute(
                    _clip,
                    _video_vae,
                    prompt,
                    width,
                    height,
                    frames,
                    **_conditioning_inputs,
                ).result
            return MiniMaxH3ReferenceToVideoSeqAttn.execute(
                _clip,
                _video_vae,
                None,
                prompt,
                width,
                height,
                frames,
                **_conditioning_inputs,
            ).result

        positive, latent = phase("conditioning", create_conditioning)
        conditioning_tensor = positive[0][0]
        result["conditioning"] = {
            "shape": [int(value) for value in conditioning_tensor.shape],
            "dtype": str(conditioning_tensor.dtype),
            "device": str(conditioning_tensor.device),
        }
        result["packed_sequence"] = build_packed_layout(
            PackedLayout, positive, latent, frames
        )
        model_management.unload_all_models()
        del (
            clip,
            conditioning_inputs,
            original_preflight,
            qwen_controller,
            recording_preflight,
            video_vae,
        )
        model_management.soft_empty_cache()
        release_unused_host_memory()

        model = phase(
            "diffusion_model_load",
            lambda: comfy_nodes.UNETLoader().load_unet(
                scenario["model"], "default"
            )[0],
        )
        model = patch_minimax_h3_model(
            model,
            q_chunk_tokens=5760,
            kv_chunk_tokens=4096,
        )
        runtime = model.model_options["transformer_options"][STATE_KEY]
        diffusion_model = model.model.diffusion_model
        result["diffusion_model"] = {
            "hidden_size": int(diffusion_model.hidden_size),
            "input_conditioning_width": int(conditioning_tensor.shape[-1]),
            "upstream_refiner_required": (
                int(conditioning_tensor.shape[-1]) != int(diffusion_model.hidden_size)
            ),
            "patcher_type": type(model).__name__,
            "is_dynamic": bool(model.is_dynamic()),
            "current_patcher_type": (
                None
                if model.model.current_patcher is None
                else type(model.model.current_patcher).__name__
            ),
        }
        if not model.is_dynamic():
            raise RuntimeError("MiniMax-H3 DiT did not load with DynamicVRAM")
        upstream_refiner_calls = {"condition_proj": 0, "token_refiner": 0}

        def count_condition_proj(_module, _inputs, _output):
            upstream_refiner_calls["condition_proj"] += 1

        def count_token_refiner(_module, _inputs, _output):
            upstream_refiner_calls["token_refiner"] += 1

        refiner_hooks = [
            diffusion_model.condition_proj.register_forward_hook(
                count_condition_proj
            ),
            diffusion_model.token_refiner.register_forward_hook(
                count_token_refiner
            ),
        ]
        noise = RandomNoise.execute(args.seed)[0]
        guider = BasicGuider.execute(model, positive)[0]
        sampler_object = KSamplerSelect.execute("res_multistep")[0]
        sigmas = BasicScheduler.execute(model, "simple", args.steps, 1.0)[0]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            sampled, _ = phase(
                "denoise",
                lambda _guider=guider, _latent=latent: SamplerCustomAdvanced.execute(
                    noise, _guider, sampler_object, sigmas, _latent
                ).result,
            )
        finally:
            for hook in refiner_hooks:
                hook.remove()
        result["denoise"] = {
            "steps": args.steps,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        result["refined_conditioning_cache"] = (
            runtime.lifetime_refined_conditioning_cache_stats
        )
        result["upstream_refiner"] = {
            **upstream_refiner_calls,
            "expected_calls_per_sampling_job": 1,
        }

        del guider, model, positive, latent
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        release_unused_host_memory()

        video_latent = sampled["samples"].unbind()[0]
        video_vae = phase(
            "video_vae_reload",
            lambda: comfy_nodes.VAELoader().load_vae(
                "minimax_h3_video_vae_fp16.safetensors"
            )[0],
        )
        video_vae = patch_minimax_h3_video_vae(
            video_vae, tile_size=192, workspace_mib=512
        )
        decoded = phase(
            "video_decode",
            lambda _video_vae=video_vae, _video_latent=video_latent: (
                _video_vae.decode(_video_latent)
            ),
        )
        if decoded.ndim == 5:
            decoded = decoded[0]

        del video_latent, video_vae
        model_management.unload_all_models()
        model_management.soft_empty_cache()

        audio = None
        if not args.skip_audio_decode:
            audio_latent = sampled["samples"].unbind()[-1]
            audio_vae = phase(
                "audio_vae_load",
                lambda: comfy_nodes.VAELoader().load_vae(
                    "minimax_h3_audio_vae_fp32.safetensors"
                )[0],
            )
            if args.audio_device == "cpu":
                cpu = torch.device("cpu")
                register_load_device = getattr(
                    audio_vae.patcher, "register_load_device", None
                )
                if register_load_device is not None:
                    register_load_device(cpu)
                audio_vae.device = cpu
                audio_vae.output_device = cpu
                audio_vae.patcher.load_device = cpu
                audio_vae.patcher.offload_device = cpu
            def decode_audio():
                with torch.inference_mode():
                    return VAEDecodeAudio.execute(
                        audio_vae, {"samples": audio_latent}
                    ).result[0]

            audio = phase("audio_decode", decode_audio)

        video = CreateVideo.execute(
            decoded, fps=24.0, audio=audio, bit_depth=8
        ).result[0]
        phase("media_write", lambda: video.save_to(str(video_path), crf=23))
        media_probe = phase("media_probe", lambda: probe_media(video_path))
        result["output"] = {
            "path": str(video_path),
            "size_bytes": video_path.stat().st_size,
            "video_shape": [int(value) for value in decoded.shape],
            "fps": 24,
            "audio_sample_rate": audio["sample_rate"] if audio else None,
            "probe": media_probe,
        }
        result["status"] = "success"
    except BaseException as error:
        exit_code = 1
        message = str(error).lower()
        result["status"] = (
            "oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in message
            else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        os.chdir(original_cwd)
        if runtime is not None:
            weight_records = runtime.weight_schedule_records
            if weight_records:
                atomic_write_json_gz(weight_schedule_path, weight_records)
                event_counts = Counter(
                    record["event"] for record in weight_records
                )
                result["seqattn_weight_schedule"] = {
                    "path": str(weight_schedule_path),
                    "record_count": len(weight_records),
                    "event_counts": dict(sorted(event_counts.items())),
                    "forward_count": sum(
                        record["event"] == "prepare"
                        and record["block_index"] == 0
                        for record in weight_records
                    ),
                    "max_staged_blocks": max(
                        record["staged_block_count"]
                        for record in weight_records
                    ),
                    "vbar_loaded_peak_mib": max(
                        record["vbar_loaded_mib"]
                        for record in weight_records
                    ),
                    "ready_blocked_seconds": sum(
                        record.get("blocked_seconds", 0.0)
                        for record in weight_records
                        if record["event"] == "ready"
                    ),
                    "settings": {
                        "q_chunk_tokens": runtime.settings.q_chunk_tokens,
                        "kv_chunk_tokens": runtime.settings.kv_chunk_tokens,
                        "qkv_tile_tokens": runtime.settings.qkv_tile_tokens,
                        "mlp_tile_tokens": runtime.settings.mlp_tile_tokens,
                    },
                }
        if sampler is not None:
            sampler.stop()
            sampler.write_csv_gz(trace_path)
            result["memory"] = {
                "nvml_process_peak_mib": sampler.nvml_peak_mib,
                "cpu_rss_peak_mib": sampler.rss_peak_mib,
                "sample_count": len(sampler.rows),
                "trace_path": str(trace_path),
                "sample_interval_ms": args.sample_interval_ms,
            }
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(result_path, result)
        print(f"EXAMPLE_RESULT {result_path}", flush=True)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "scenario": args.scenario,
                    "packed_tokens": result.get("packed_sequence", {}).get("tokens"),
                    "nvml_process_peak_mib": result.get("memory", {}).get(
                        "nvml_process_peak_mib"
                    ),
                    "output": result.get("output", {}).get("path"),
                },
                indent=2,
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
