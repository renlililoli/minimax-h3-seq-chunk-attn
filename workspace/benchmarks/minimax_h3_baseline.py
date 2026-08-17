#!/usr/bin/env python3
"""Reproducible baseline runner for the unmodified DiffSynth MiniMax-H3 path."""

import argparse
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
from benchmarks.minimax_h3_bench.protocol import atomic_write_json, classify_exception


PROMPT = (
    "A girl is very happy, she is speaking in english: “I enjoy working with "
    "Diffsynth-Studio, it's a perfect framework.”"
)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StepTimer:
    def __init__(self, iterable):
        self.iterable = iterable
        self.step_seconds = []
        self.peak_allocated_mib = None
        self.peak_reserved_mib = None

    def __iter__(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for item in self.iterable:
            synchronize()
            started = time.perf_counter()
            yield item
            synchronize()
            elapsed = time.perf_counter() - started
            self.step_seconds.append(elapsed)
            print(f"BENCH_STEP {len(self.step_seconds)} {elapsed:.6f}s", flush=True)
        synchronize()
        self.peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
        self.peak_reserved_mib = torch.cuda.max_memory_reserved() / 2**20


def timed_method(obj, method_name, samples):
    original = getattr(obj, method_name)

    def wrapper(*args, **kwargs):
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
        "--instrument-streaming",
        action="store_true",
        help="Collect synchronized A/B/C/D1/D2 timing and logical transfer counters.",
    )
    args = parser.parse_args()

    physical_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    physical_vram_gib = physical_vram_bytes / 2**30
    effective_vram_gib = args.simulated_vram_gib or physical_vram_gib
    if effective_vram_gib > physical_vram_gib:
        raise ValueError(
            f"simulated VRAM ({effective_vram_gib} GiB) exceeds physical VRAM "
            f"({physical_vram_gib:.3f} GiB)"
        )
    if effective_vram_gib <= args.vram_reserve_gib:
        raise ValueError("simulated VRAM must exceed the requested reserve")
    if args.simulated_vram_gib is not None:
        torch.cuda.set_per_process_memory_fraction(
            effective_vram_gib / physical_vram_gib,
            device=0,
        )
    layer_vram_limit_gib = effective_vram_gib - args.vram_reserve_gib

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
            "gpu": gpu_info(),
        },
    }

    progress = None
    video_decode_samples = []
    audio_decode_samples = []
    captured_latents = {}
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
            timed_method(pipe.video_vae, "decode_video", video_decode_samples)
            timed_method(pipe.audio_vae, "decode_audio", audio_decode_samples)

        torch.cuda.reset_peak_memory_stats()
        synchronize()
        inference_started = time.perf_counter()
        def progress_factory(iterable):
            nonlocal progress
            progress = StepTimer(iterable)
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
            streaming_stats=streaming_stats,
        )
        synchronize()
        result["pipeline_seconds"] = time.perf_counter() - inference_started
        result["denoise"] = stats(progress.step_seconds if progress is not None else [])
        if progress is not None:
            result["denoise_peak_allocated_mib"] = progress.peak_allocated_mib
            result["denoise_peak_reserved_mib"] = progress.peak_reserved_mib
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
        if progress is not None:
            result["denoise"] = stats(progress.step_seconds)
            result["denoise_peak_allocated_mib"] = progress.peak_allocated_mib
            result["denoise_peak_reserved_mib"] = progress.peak_reserved_mib
        if video_decode_samples:
            result["video_decode_calls_seconds"] = video_decode_samples
        if audio_decode_samples:
            result["audio_decode_calls_seconds"] = audio_decode_samples
        if torch.cuda.is_available():
            result["observed_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["observed_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        result["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(json_path, result)
        print(f"BENCH_RESULT {json_path}", flush=True)


if __name__ == "__main__":
    main()
