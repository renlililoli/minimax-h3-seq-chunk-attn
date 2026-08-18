#!/usr/bin/env python3
"""Reproducible baseline runner for the unmodified DiffSynth MiniMax-H3 path."""

import argparse
from collections import Counter
import json
import os
import platform
import resource
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.core.attention.streaming import StreamingStats
from diffsynth.utils.data.audio_video import write_video_audio
from benchmarks.minimax_h3_bench.protocol import (
    ProcessSampler,
    atomic_write_json,
    classify_exception,
    initialize_vram_budget,
)


PROMPT = (
    "A girl is very happy, she is speaking in english: “I enjoy working with "
    "Diffsynth-Studio, it's a perfect framework.”"
)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_module_storage_summary(module):
    """Return CUDA storage held directly by a module at a step boundary."""
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
    def __init__(self, iterable, sampler, memory_probe=None):
        self.iterable = iterable
        self.sampler = sampler
        self.memory_probe = memory_probe
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
            yield item
            synchronize()
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
    method_name,
    samples,
    *,
    capture_key=None,
    captured_latents=None,
):
    original = getattr(obj, method_name)

    def wrapper(*args, **kwargs):
        if capture_key is not None and captured_latents is not None and args:
            captured_latents[capture_key] = args[0].detach().to("cpu")
        synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        synchronize()
        samples.append(time.perf_counter() - started)
        return result

    setattr(obj, method_name, wrapper)


def gpu_info():
    query = (
        "index,name,uuid,driver_version,memory.total,power.limit"
    )
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


def stats(values):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--dit-layers",
        type=int,
        default=None,
        help="Keep only the first N DiT blocks for implementation smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", default="/models/MiniMax-H3-NF4")
    parser.add_argument("--processor-dir", default="/models/minimax/processor")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Replace video/audio VAE decode with tiny no-op outputs; useful for DiT-only benchmarks.",
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Save the final video/audio latents captured at the decode boundary.",
    )
    parser.add_argument(
        "--offload-device",
        choices=("cpu", "disk"),
        default="cpu",
        help="Weight backing store. The standard inference example uses CPU; low_vram uses disk.",
    )
    parser.add_argument(
        "--simulated-vram-gib",
        type=float,
        default=None,
        help="Hard-cap the PyTorch CUDA allocator to this fraction of physical VRAM.",
    )
    parser.add_argument(
        "--target-vram-mib",
        type=int,
        default=None,
        help="NVML-aware whole-process budget; preferred for formal 4/6/8GB runs.",
    )
    parser.add_argument(
        "--vram-reserve-gib",
        type=float,
        default=2.0,
        help="VRAM kept outside DiffSynth's layer residency budget.",
    )
    parser.add_argument("--activation-streaming", action="store_true")
    parser.add_argument("--projection-chunk-size", type=int, default=2048)
    parser.add_argument("--attention-q-block-size", type=int, default=2048)
    parser.add_argument("--attention-kv-block-size", type=int, default=512)
    parser.add_argument(
        "--streaming-attention-backend",
        choices=("auto", "torch", "flash2_lse", "seqattn"),
        default="auto",
    )
    parser.add_argument(
        "--seqattn-workspace-mib",
        type=int,
        default=1024,
        help="CUDA workspace owned by seqattn; excludes model weights and caller allocations.",
    )
    parser.add_argument(
        "--seqattn-q-chunk-tokens",
        type=int,
        default=None,
        help="Optional explicit resident query chunk; default lets seqattn plan from workspace.",
    )
    parser.add_argument(
        "--instrument-streaming",
        action="store_true",
        help="Collect synchronized A/B/C/D1/D2 timing and logical transfer counters.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=20.0,
        help="NVML/RSS sampling interval used for global and per-step peaks.",
    )
    parser.add_argument(
        "--save-memory-trace",
        action="store_true",
        help="Save every NVML/RSS sample with its denoise-step index as CSV.gz.",
    )
    args = parser.parse_args()

    if args.target_vram_mib is not None and args.simulated_vram_gib is not None:
        raise ValueError("target-vram-mib and simulated-vram-gib are mutually exclusive")
    if args.sample_interval_ms <= 0:
        raise ValueError("sample-interval-ms must be positive")

    physical_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    physical_vram_gib = physical_vram_bytes / 2**30
    budget = None
    if args.target_vram_mib is not None:
        budget = initialize_vram_budget(args.target_vram_mib)
        effective_vram_gib = args.target_vram_mib / 1024.0
    else:
        effective_vram_gib = args.simulated_vram_gib or physical_vram_gib
    if effective_vram_gib > physical_vram_gib:
        raise ValueError(
            f"simulated VRAM ({effective_vram_gib} GiB) exceeds physical VRAM "
            f"({physical_vram_gib:.3f} GiB)"
        )
    if budget is None and effective_vram_gib <= args.vram_reserve_gib:
        raise ValueError("simulated VRAM must exceed the requested reserve")
    if budget is None and args.simulated_vram_gib is not None:
        torch.cuda.set_per_process_memory_fraction(
            effective_vram_gib / physical_vram_gib,
            device=0,
        )
    layer_vram_limit_gib = (
        budget.vram_limit_gib
        if budget is not None
        else effective_vram_gib - args.vram_reserve_gib
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{args.tag}_{args.height}x{args.width}_f{args.frames}_s{args.steps}_{stamp}"
    json_path = output_dir / f"{run_name}.json"
    media_path = output_dir / f"{run_name}.mp4"

    result = {
        "status": "running",
        "run_name": run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args),
        "memory_policy": {
            "physical_vram_gib": physical_vram_gib,
            "effective_vram_gib": effective_vram_gib,
            "pytorch_allocator_fraction": effective_vram_gib / physical_vram_gib,
            "diffsynth_vram_limit_gib": layer_vram_limit_gib,
            "reserve_gib": args.vram_reserve_gib,
        },
        "prompt": PROMPT,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "benchmark_nvml_gpu_index": int(
                os.environ.get("BENCH_NVML_GPU_INDEX", "0")
            ),
            "physical_gpu_index": int(
                os.environ.get(
                    "BENCH_PHYSICAL_GPU_INDEX",
                    os.environ.get("BENCH_NVML_GPU_INDEX", "0"),
                )
            ),
            "gpu": gpu_info(),
        },
    }

    progress = None
    video_decode_samples = []
    audio_decode_samples = []
    captured_latents = {}
    sampler = ProcessSampler(
        interval_seconds=args.sample_interval_ms / 1000.0,
        record_trace=args.save_memory_trace,
    )
    sampler.__enter__()
    try:
        model_dir = Path(args.model_dir)
        required = [
            "minimax-h3-fl2va-nf4.safetensors",
            "minimax-h3-text-encoder-nf4.safetensors",
            "video_vae_nf4.safetensors",
            "audio_vae_nf4.safetensors",
        ]
        for filename in required:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(model_dir / filename)

        backing_dtype = "disk" if args.offload_device == "disk" else torch.bfloat16
        vram_config = {
            "offload_dtype": backing_dtype,
            "offload_device": args.offload_device,
            "onload_dtype": backing_dtype,
            "onload_device": args.offload_device,
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        synchronize()
        load_started = time.perf_counter()
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=str(model_dir / required[0]), **vram_config),
                ModelConfig(path=str(model_dir / required[1]), **vram_config),
                ModelConfig(path=str(model_dir / required[2]), **vram_config),
                ModelConfig(path=str(model_dir / required[3]), **vram_config),
            ],
            processor_config=ModelConfig(path=args.processor_dir),
            vram_limit=layer_vram_limit_gib,
        )
        synchronize()
        result["model_load_seconds"] = time.perf_counter() - load_started
        result["model_load_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["model_load_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        print(f"BENCH_MODEL_LOAD {result['model_load_seconds']:.6f}s", flush=True)

        if args.dit_layers is not None:
            if args.dit_layers <= 0 or args.dit_layers > len(pipe.dit.blocks):
                raise ValueError(
                    f"dit-layers must be within [1, {len(pipe.dit.blocks)}]"
                )
            pipe.dit.blocks = torch.nn.ModuleList(list(pipe.dit.blocks[:args.dit_layers]))
            result["effective_dit_layers"] = len(pipe.dit.blocks)

        if args.skip_decode:
            def skip_video_decode(latents, *unused_args, **unused_kwargs):
                if args.save_latents:
                    captured_latents["video"] = latents.detach().to("cpu")
                return torch.zeros(
                    (1, 3, 1, 1, 1), device=latents.device, dtype=latents.dtype
                )

            def skip_audio_decode(latents, *unused_args, **unused_kwargs):
                if args.save_latents:
                    captured_latents["audio"] = latents.detach().to("cpu")
                return torch.zeros(
                    (1, 2, 1), device=latents.device, dtype=torch.float32
                )

            pipe.video_vae.decode_video = skip_video_decode
            pipe.audio_vae.decode_audio = skip_audio_decode
        else:
            timed_method(
                pipe.video_vae,
                "decode_video",
                video_decode_samples,
                capture_key="video" if args.save_latents else None,
                captured_latents=captured_latents,
            )
            timed_method(
                pipe.audio_vae,
                "decode_audio",
                audio_decode_samples,
                capture_key="audio" if args.save_latents else None,
                captured_latents=captured_latents,
            )

        torch.cuda.reset_peak_memory_stats()
        synchronize()
        inference_started = time.perf_counter()
        def progress_factory(iterable):
            nonlocal progress
            progress = StepTimer(
                iterable,
                sampler,
                memory_probe=lambda: cuda_module_storage_summary(pipe.dit),
            )
            return progress

        streaming_stats = StreamingStats(
            instrument_phases=args.instrument_streaming
        ) if args.activation_streaming else None
        video, audio = pipe(
            prompt=PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            progress_bar_cmd=progress_factory,
            activation_streaming=args.activation_streaming,
            projection_chunk_size=args.projection_chunk_size,
            attention_q_block_size=args.attention_q_block_size,
            attention_kv_block_size=args.attention_kv_block_size,
            streaming_attention_backend=args.streaming_attention_backend,
            seqattn_workspace_mib=args.seqattn_workspace_mib,
            seqattn_q_chunk_tokens=args.seqattn_q_chunk_tokens,
            streaming_stats=streaming_stats,
        )
        synchronize()
        result["pipeline_seconds"] = time.perf_counter() - inference_started
        result["denoise"] = stats(progress.step_seconds if progress is not None else [])
        if progress is not None:
            result["denoise_peak_allocated_mib"] = progress.peak_allocated_mib
            result["denoise_peak_reserved_mib"] = progress.peak_reserved_mib
            result["denoise_step_memory"] = progress.step_memory
        result["video_decode_calls_seconds"] = video_decode_samples
        result["audio_decode_calls_seconds"] = audio_decode_samples
        result["inference_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["inference_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        if streaming_stats is not None:
            result["streaming_stats"] = streaming_stats.as_dict()
        result["output"] = {
            "video_frames": len(video),
            "audio_shape": list(audio.shape),
            "audio_dtype": str(audio.dtype),
        }

        if args.save_latents:
            latent_path = output_dir / f"{run_name}_latents.pt"
            torch.save(captured_latents, latent_path)
            result["latent_path"] = str(latent_path)

        if not args.no_media and not args.skip_decode:
            save_started = time.perf_counter()
            write_video_audio(
                video=video,
                audio=audio,
                output_path=str(media_path),
                fps=24,
                audio_sample_rate=32000,
            )
            result["media_write_seconds"] = time.perf_counter() - save_started
            result["media_path"] = str(media_path)
            result["media_size_bytes"] = media_path.stat().st_size

        result["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["status"] = "success"
    except BaseException as exc:
        result["status"] = classify_exception(exc)
        result["failure_message"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        sampler.__exit__(None, None, None)
        if args.save_latents and captured_latents and "latent_path" not in result:
            latent_path = output_dir / f"{run_name}_latents.pt"
            torch.save(captured_latents, latent_path)
            result["latent_path"] = str(latent_path)
        if progress is not None:
            result["denoise"] = stats(progress.step_seconds)
            result["denoise_peak_allocated_mib"] = progress.peak_allocated_mib
            result["denoise_peak_reserved_mib"] = progress.peak_reserved_mib
            result["denoise_step_memory"] = progress.step_memory
        if video_decode_samples:
            result["video_decode_calls_seconds"] = video_decode_samples
        if audio_decode_samples:
            result["audio_decode_calls_seconds"] = audio_decode_samples
        if torch.cuda.is_available():
            result["observed_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["observed_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        result["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["nvml_process_peak_mib"] = sampler.nvml_peak_mib
        result["cpu_rss_peak_mib"] = sampler.rss_peak_mib
        result["memory_sampling"] = {
            "interval_ms": args.sample_interval_ms,
            "sample_count": sampler.sample_count,
        }
        if args.save_memory_trace:
            trace_path = output_dir / f"{run_name}_memory_trace.csv.gz"
            result["memory_sampling"]["trace_sample_count"] = (
                sampler.write_trace_csv_gz(trace_path)
            )
            result["memory_sampling"]["trace_path"] = str(trace_path)
        if budget is not None:
            result["memory_policy"].update({
                "target_vram_mib": budget.target_mib,
                "context_mib": budget.context_mib,
                "allocator_limit_mib": budget.allocator_limit_mib,
                "pytorch_allocator_fraction": budget.allocator_fraction,
                "diffsynth_vram_limit_gib": budget.vram_limit_gib,
                "safety_margin_mib": budget.safety_margin_mib,
            })
            if result["status"] == "success" and sampler.nvml_peak_mib > budget.target_mib:
                result["status"] = "budget_exceeded"
                result["failure_message"] = (
                    f"NVML process peak {sampler.nvml_peak_mib:.1f} MiB exceeded "
                    f"target {budget.target_mib} MiB"
                )
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(json_path, result)
        print(f"BENCH_RESULT {json_path}", flush=True)


if __name__ == "__main__":
    main()
