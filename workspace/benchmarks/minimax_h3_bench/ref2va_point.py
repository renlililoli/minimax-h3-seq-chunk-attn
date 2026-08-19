from __future__ import annotations

import argparse
import os
import platform
import resource
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import torch
from diffsynth.core.attention.streaming import StreamingStats
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

from .protocol import (
    ProcessSampler,
    atomic_write_json,
    classify_exception,
    finish_result,
    initialize_vram_budget,
)
from .telemetry import (
    StepTimer,
    cuda_module_storage_summary,
    gpu_info,
    stats,
    synchronize,
    timed_method,
)

PROMPT = """subject_definitions:
<Video 1> is the source video for a reference-driven video editing task.
<Audio 1> is the synchronized soundtrack of <Video 1>.

summary:
[source-preserving video and audio edit] Re-render <Video 1> as a polished cinematic master while preserving its subjects, actions, camera motion, timing, composition, and scene continuity. Reuse <Audio 1> as the synchronized soundtrack.

retention_analysis:
<Video 1>: fully_preserved - retain the source identity, motion, framing, shot order, and temporal structure.
<Audio 1>: fully_preserved - retain the synchronized source soundtrack without changing its timing.

detailed_description:
Preserve the complete visual content and movement of <Video 1>. Improve fine detail, natural contrast, highlight roll-off, color consistency, and temporal stability. Keep the original duration and camera behavior. Do not introduce new subjects, cuts, text, logos, or scene changes.

overall_soundscape:
Use <Audio 1> as the synchronized soundtrack for the full video."""


def decode_audio_with_pyav(
    path: str, *, duration_seconds: float, sample_rate: int
) -> torch.Tensor:
    container = av.open(path)
    try:
        stream = next(
            (stream for stream in container.streams if stream.type == "audio"), None
        )
        if stream is None:
            raise ValueError(f"no audio stream found in {path}")
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=sample_rate,
        )
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    finally:
        container.close()
    if not chunks:
        raise ValueError(f"audio decoder returned no samples for {path}")
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).to(torch.float32)
    expected_samples = round(duration_seconds * sample_rate)
    if waveform.shape[-1] < expected_samples:
        waveform = torch.nn.functional.pad(
            waveform, (0, expected_samples - waveform.shape[-1])
        )
    return waveform[:, :expected_samples].contiguous()


class UnitProfiler:
    def __init__(self, original, sampler: ProcessSampler, result: dict):
        self.original = original
        self.sampler = sampler
        self.result = result
        self.records = []

    def __call__(self, unit, pipe, inputs_shared, inputs_posi, inputs_nega):
        synchronize()
        start_ms = self.sampler.elapsed_ms()
        started = time.perf_counter()
        outputs = self.original(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
        synchronize()
        end_ms = self.sampler.elapsed_ms()
        nvml_mib, rss_mib = self.sampler.sample()
        record = {
            "name": type(unit).__name__,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "seconds": time.perf_counter() - started,
            "torch_allocated_end_mib": torch.cuda.memory_allocated() / 2**20,
            "torch_reserved_end_mib": torch.cuda.memory_reserved() / 2**20,
            "nvml_process_end_mib": nvml_mib,
            "cpu_rss_end_mib": rss_mib,
        }
        self.records.append(record)
        shared, posi, _nega = outputs
        if record["name"] == "MiniMaxH3Unit_PackedSequenceBuilder":
            self.result["sequence"] = sequence_summary(shared, posi)
            print(
                "BENCH_SEQUENCE "
                f"tokens={self.result['sequence']['packed_tokens']} "
                f"used={self.result['sequence']['used_tokens']}",
                flush=True,
            )
        return outputs


def sequence_summary(shared: dict, posi: dict) -> dict:
    packed = posi["packed"]
    video_latents = shared["video_latents"]
    audio_latents = shared["audio_latents"]
    ref_blocks = shared.get("ref_blocks") or []
    text_tokens = int(posi["prompt_embeds"].shape[0])

    target_video_tokens = (
        int(video_latents.shape[2])
        * (int(video_latents.shape[3]) // 2)
        * (int(video_latents.shape[4]) // 2)
    )
    target_audio_tokens = int(audio_latents.shape[0]) * int(audio_latents.shape[-1])
    reference_video_tokens = 0
    reference_audio_tokens = 0
    references = []
    for block in ref_blocks:
        visual_tokens = 0
        if block["kind"] in {"image", "video", "video_audio"}:
            visual_tokens = (
                int(block["latent_t"])
                * (int(block["latent_h"]) // 2)
                * (int(block["latent_w"]) // 2)
            )
        audio_tokens = int(block.get("ref_audio_t", 0)) * int(audio_latents.shape[0])
        reference_video_tokens += visual_tokens
        reference_audio_tokens += audio_tokens
        references.append(
            {
                "kind": block["kind"],
                "latent_t": int(block.get("latent_t", 0)),
                "latent_h": int(block.get("latent_h", 0)),
                "latent_w": int(block.get("latent_w", 0)),
                "audio_t": int(block.get("ref_audio_t", 0)),
                "video_tokens": visual_tokens,
                "audio_tokens": audio_tokens,
            }
        )

    used_tokens = int(packed["cu_seqlens"][1])
    packed_tokens = int(packed["seq_len"])
    breakdown_sum = (
        text_tokens
        + reference_video_tokens
        + reference_audio_tokens
        + target_video_tokens
        + target_audio_tokens
    )
    if breakdown_sum != used_tokens:
        raise RuntimeError(
            f"packed token accounting mismatch: {breakdown_sum} != {used_tokens}"
        )
    return {
        "text_tokens": text_tokens,
        "reference_video_tokens": reference_video_tokens,
        "reference_audio_tokens": reference_audio_tokens,
        "target_video_tokens": target_video_tokens,
        "target_audio_tokens": target_audio_tokens,
        "used_tokens": used_tokens,
        "padding_tokens": packed_tokens - used_tokens,
        "packed_tokens": packed_tokens,
        "cu_seqlens": [int(value) for value in packed["cu_seqlens"].tolist()],
        "target_video_latent_shape": [int(value) for value in video_latents.shape],
        "target_audio_latent_shape": [int(value) for value in audio_latents.shape],
        "references": references,
    }


def enrich_phase_peaks(records: list[dict], sampler: ProcessSampler) -> None:
    for record in records:
        record.update(sampler.trace_stats(record["start_ms"], record["end_ms"]))


def measure_phase(name: str, sampler: ProcessSampler, records: list[dict], fn):
    synchronize()
    start_ms = sampler.elapsed_ms()
    started = time.perf_counter()
    output = fn()
    synchronize()
    end_ms = sampler.elapsed_ms()
    nvml_mib, rss_mib = sampler.sample()
    records.append(
        {
            "name": name,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "seconds": time.perf_counter() - started,
            "torch_allocated_end_mib": torch.cuda.memory_allocated() / 2**20,
            "torch_reserved_end_mib": torch.cuda.memory_reserved() / 2**20,
            "nvml_process_end_mib": nvml_mib,
            "cpu_rss_end_mib": rss_mib,
        }
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instrumented MiniMax-H3 Ref2VA capacity and performance point."
    )
    parser.add_argument("--mode", choices=("native", "streaming"), required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dit-layers", type=int, default=None)
    parser.add_argument(
        "--source",
        default="/models/minimax/assets/h3_direct_768p.mp4",
    )
    parser.add_argument("--model-dir", default="/models/MiniMax-H3-NF4")
    parser.add_argument("--processor-dir", default="/models/minimax/processor")
    parser.add_argument("--output-dir", default="/workspace/benchmarks/results/ref2va")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-vram-mib", type=int, default=None)
    parser.add_argument("--vram-reserve-gib", type=float, default=5.0)
    parser.add_argument("--offload-device", choices=("cpu", "disk"), default="cpu")
    parser.add_argument("--activation-workspace-mib", type=int, default=4096)
    parser.add_argument("--projection-chunk-size", type=int, default=8192)
    parser.add_argument("--attention-kv-block-size", type=int, default=4096)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--save-memory-trace", action="store_true")
    args = parser.parse_args()
    if args.sample_interval_ms <= 0:
        parser.error("--sample-interval-ms must be positive")
    if args.activation_workspace_mib <= 0:
        parser.error("--activation-workspace-mib must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"{args.tag}_{args.mode}_{args.height}x{args.width}_"
        f"f{args.frames}_s{args.steps}_{stamp}"
    )
    json_path = output_dir / f"{run_name}.json"
    media_path = output_dir / f"{run_name}.mp4"
    trace_path = output_dir / f"{run_name}_memory_trace.csv.gz"
    latent_path = output_dir / f"{run_name}_latents.pt"

    physical_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    physical_vram_gib = physical_vram_bytes / 2**30
    budget = None
    if args.target_vram_mib is not None:
        budget = initialize_vram_budget(args.target_vram_mib)
        layer_vram_limit_gib = budget.vram_limit_gib
    else:
        layer_vram_limit_gib = physical_vram_gib - args.vram_reserve_gib
    if layer_vram_limit_gib <= 0:
        raise ValueError("VRAM reserve leaves no DiffSynth layer-residency budget")

    result = {
        "status": "running",
        "run_name": run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args),
        "prompt": PROMPT,
        "memory_policy": {
            "physical_vram_gib": physical_vram_gib,
            "target_vram_mib": args.target_vram_mib,
            "diffsynth_vram_limit_gib": layer_vram_limit_gib,
            "activation_workspace_mib": (
                args.activation_workspace_mib if args.mode == "streaming" else None
            ),
            "scope": "whole-process target plus separately reported activation plan",
        },
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
    phases = []
    progress = None
    captured_latents = {}
    core_timings = {
        "reference_video_encode_seconds": [],
        "reference_audio_encode_seconds": [],
        "text_encoder_seconds": [],
        "video_decode_seconds": [],
        "audio_decode_seconds": [],
    }
    sampler = ProcessSampler(
        interval_seconds=args.sample_interval_ms / 1000.0,
        record_trace=args.save_memory_trace,
    )
    sampler.__enter__()
    try:
        model_dir = Path(args.model_dir)
        required = [
            "minimax-h3-ref2va-nf4.safetensors",
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

        def load_pipeline():
            return MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=str(model_dir / filename), **vram_config)
                    for filename in required
                ],
                processor_config=ModelConfig(path=args.processor_dir),
                vram_limit=layer_vram_limit_gib,
            )

        pipe = measure_phase("model_load", sampler, phases, load_pipeline)
        if args.dit_layers is not None:
            if args.dit_layers <= 0 or args.dit_layers > len(pipe.dit.blocks):
                raise ValueError(
                    f"dit-layers must be within [1, {len(pipe.dit.blocks)}]"
                )
            pipe.dit.blocks = torch.nn.ModuleList(
                list(pipe.dit.blocks[: args.dit_layers])
            )
        result["effective_dit_layers"] = len(pipe.dit.blocks)

        def load_reference():
            return read_video_audio(
                args.source,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                fps=24,
                audio_sample_rate=pipe.audio_vae.sample_rate,
            )

        ref_video, ref_audio, ref_sample_rate = measure_phase(
            "reference_media_read", sampler, phases, load_reference
        )
        if len(ref_video) != args.frames:
            raise ValueError(
                f"reference produced {len(ref_video)} frames, expected {args.frames}"
            )
        audio_decoder = "diffsynth_torchcodec"
        if ref_audio is None:
            ref_sample_rate = pipe.audio_vae.sample_rate
            ref_audio = measure_phase(
                "reference_audio_pyav_fallback",
                sampler,
                phases,
                lambda: decode_audio_with_pyav(
                    args.source,
                    duration_seconds=len(ref_video) / 24.0,
                    sample_rate=ref_sample_rate,
                ),
            )
            audio_decoder = "pyav_fallback"
        result["reference_media"] = {
            "path": args.source,
            "decoded_frames": len(ref_video),
            "frame_size": list(ref_video[0].size),
            "fps": 24,
            "duration_seconds": len(ref_video) / 24.0,
            "audio_shape": [int(value) for value in ref_audio.shape],
            "audio_sample_rate": int(ref_sample_rate),
            "audio_decoder": audio_decoder,
        }

        timed_method(
            pipe.video_vae,
            "encode_video",
            core_timings["reference_video_encode_seconds"],
        )
        timed_method(
            pipe.audio_vae,
            "encode_audio",
            core_timings["reference_audio_encode_seconds"],
        )
        timed_method(
            pipe.text_encoder,
            "forward",
            core_timings["text_encoder_seconds"],
        )
        timed_method(
            pipe.video_vae,
            "decode_video",
            core_timings["video_decode_seconds"],
            capture_key="video",
            captured_latents=captured_latents,
            capture_path=latent_path.with_name(f"{run_name}_video_latents.pt")
            if args.save_latents
            else None,
        )
        timed_method(
            pipe.audio_vae,
            "decode_audio",
            core_timings["audio_decode_seconds"],
            capture_key="audio",
            captured_latents=captured_latents,
            capture_path=latent_path.with_name(f"{run_name}_audio_latents.pt")
            if args.save_latents
            else None,
        )

        unit_profiler = UnitProfiler(pipe.unit_runner, sampler, result)
        pipe.unit_runner = unit_profiler
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def progress_factory(iterable):
            nonlocal progress
            progress = StepTimer(
                iterable,
                sampler,
                memory_probe=lambda: cuda_module_storage_summary(pipe.dit),
            )
            return progress

        streaming_stats = (
            StreamingStats(instrument_phases=True) if args.mode == "streaming" else None
        )

        def execute_pipeline():
            return pipe(
                prompt=PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                seed=args.seed,
                references=[
                    {
                        "type": "video_audio",
                        "video": ref_video,
                        "audio": ref_audio,
                        "sample_rate": ref_sample_rate,
                    }
                ],
                progress_bar_cmd=progress_factory,
                activation_streaming=args.mode == "streaming",
                projection_chunk_size=args.projection_chunk_size,
                attention_kv_block_size=args.attention_kv_block_size,
                streaming_attention_backend=(
                    "seqattn" if args.mode == "streaming" else "auto"
                ),
                streaming_activation_workspace_mib=(
                    args.activation_workspace_mib if args.mode == "streaming" else None
                ),
                streaming_mlp_mode="fused",
                streaming_stats=streaming_stats,
            )

        video, audio = measure_phase(
            "pipeline_total", sampler, phases, execute_pipeline
        )
        result["pipeline_units"] = unit_profiler.records
        result["denoise"] = stats(progress.step_seconds if progress else [])
        result["denoise_step_memory"] = progress.step_memory if progress else []
        result["core_timings"] = core_timings
        if streaming_stats is not None:
            result["streaming_stats"] = streaming_stats.as_dict()
        result["output"] = {
            "video_frames": len(video),
            "frame_size": list(video[0].size),
            "audio_shape": [int(value) for value in audio.shape],
            "audio_dtype": str(audio.dtype),
        }

        if args.save_latents:
            temporary_path = latent_path.with_suffix(".pt.tmp")
            torch.save(captured_latents, temporary_path)
            os.replace(temporary_path, latent_path)
            result["latent_path"] = str(latent_path)

        if not args.no_media:
            measure_phase(
                "media_write",
                sampler,
                phases,
                lambda: write_video_audio(
                    video=video,
                    audio=audio,
                    output_path=str(media_path),
                    fps=24,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                ),
            )
            result["media_path"] = str(media_path)
            result["media_size_bytes"] = media_path.stat().st_size
        result["status"] = "success"
    except Exception as exc:  # noqa: BLE001 - benchmark must persist failures
        result["status"] = classify_exception(exc)
        result["failure_message"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        sampler.__exit__(None, None, None)
        enrich_phase_peaks(phases, sampler)
        result["phases"] = phases
        if progress is not None:
            result["denoise"] = stats(progress.step_seconds)
            result["denoise_step_memory"] = progress.step_memory
        result["core_timings"] = core_timings
        result["observed_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / 2**20
        )
        result["observed_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        result["max_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        result["memory_sampling"] = {
            "interval_ms": args.sample_interval_ms,
            "sample_count": sampler.sample_count,
        }
        if args.save_memory_trace:
            result["memory_sampling"]["trace_sample_count"] = (
                sampler.write_trace_csv_gz(trace_path)
            )
            result["memory_sampling"]["trace_path"] = str(trace_path)
        finish_result(result, sampler, budget)
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(json_path, result)
        print(f"BENCH_RESULT {json_path}", flush=True)


if __name__ == "__main__":
    main()
