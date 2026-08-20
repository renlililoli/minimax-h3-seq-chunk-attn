from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import torch

from workspace.benchmarks.minimax_h3_bench.protocol import (
    ProcessSampler,
    atomic_write_json,
    classify_exception,
    finish_result,
    initialize_vram_budget,
)


PROMPT = (
    "Use <Video 1> as the exact motion, camera, subject, and scene reference. "
    "Continue it as one coherent cinematic shot with natural motion, stable identity, "
    "photorealistic detail, synchronized ambient sound, no cuts, no text, no logos."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "streaming"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-vram-mib", type=int, default=8192)
    parser.add_argument("--activation-workspace-mib", type=int, default=1024)
    parser.add_argument("--kv-chunk-tokens", type=int, default=4096)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    parser.add_argument("--comfy-reserve-vram-gib", type=float, default=3.0)
    parser.add_argument(
        "--text-encoder-mode",
        choices=("cpu", "gpu-offload"),
        default="cpu",
    )
    parser.add_argument("--stop-after-text-conditioning", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--profile-denoise", action="store_true")
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    return parser.parse_args()


def load_package(name: str, package_dir: str):
    init_path = Path(package_dir) / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name, init_path, submodule_search_locations=[str(init_path.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decode_reference_video(path: str, frames: int) -> tuple[torch.Tensor, dict]:
    decoded = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        source_rate = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            decoded.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if len(decoded) == frames:
                break
    if len(decoded) != frames:
        raise ValueError(f"reference decoded {len(decoded)} frames, expected {frames}")
    images = torch.stack(decoded).to(torch.float32).div_(255.0)
    return images, {
        "path": path,
        "decoded_frames": len(decoded),
        "height": int(images.shape[1]),
        "width": int(images.shape[2]),
        "source_fps": source_rate,
    }


def save_video(path: Path, images: torch.Tensor, fps: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(images.shape[2])
        stream.height = int(images.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for image in images:
            array = image.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(array), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def nested_shapes(samples) -> list[list[int]]:
    return [[int(value) for value in item.shape] for item in samples.unbind()]


@torch.inference_mode()
def encode_minimax_video_cpu_streamed(vae, pixel_samples: torch.Tensor) -> torch.Tensor:
    """Run the stock MiniMax VAE temporal chunks without staging all frames on CUDA."""
    import comfy.model_management as model_management

    pixel_samples = vae.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
    if pixel_samples.ndim < 5:
        pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
    x = vae.process_input(pixel_samples).to(vae.vae_dtype)
    model = vae.first_stage_model

    model_management.load_models_gpu(
        [vae.patcher], memory_required=512 * 2**20, force_full_load=vae.disable_offload
    )
    clip_length = int(model.clip_length)
    if x.shape[2] % clip_length:
        pad_size = (-x.shape[2]) % clip_length
        x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)], dim=2)

    moments = []
    with model_management.cuda_device_context(vae.device):
        for start in range(0, x.shape[2], clip_length):
            clip = x[:, :, start : start + clip_length].to(vae.device)
            encoded = model._adaptive_encode(clip)
            moments.append(encoded.to(vae.output_device, copy=True))
            del clip, encoded
            model_management.soft_empty_cache()

    moments = torch.cat(moments, dim=2)
    if model.token_drop > 0:
        moments = moments[:, :, : -model.token_drop]
    mean = torch.chunk(moments.float(), 2, dim=1)[0]
    latent_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(mean)
    latent_std = model.latents_std.view(1, -1, 1, 1, 1).to(mean)
    return ((mean - latent_mean) / latent_std).to(dtype=vae.vae_output_dtype())


@torch.inference_mode()
def decode_minimax_video_cpu_streamed(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode stock MiniMax temporal chunks into a CPU frame canvas."""
    import comfy.model_management as model_management

    model = vae.first_stage_model
    model_management.load_models_gpu(
        [vae.patcher], memory_required=512 * 2**20, force_full_load=vae.disable_offload
    )

    latent_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(latent)
    latent_std = model.latents_std.view(1, -1, 1, 1, 1).to(latent)
    z = latent * latent_std + latent_mean
    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    pseudo_total_tokens = z.shape[2] + model.token_drop
    pad_tokens = (-pseudo_total_tokens) % model.tokens_chunk_size
    pseudo_total_tokens += pad_tokens
    num_chunks = pseudo_total_tokens // model.tokens_chunk_size - int(
        model.token_drop > 0
    )
    if num_chunks < 1:
        pad_tokens += model.tokens_chunk_size
        num_chunks += 1
    if pad_tokens:
        z = torch.cat(
            [z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2
        )

    output_frames = model._decode_temporal_frame_plan(
        z.shape[2], num_chunks, pad_tokens
    )
    output = None
    overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal output, write_pos
        frame_count = int(part.shape[2])
        if frame_count <= 0:
            return
        if output is None:
            output = torch.empty(
                (part.shape[0], 3, output_frames, part.shape[-2], part.shape[-1]),
                dtype=torch.float32,
                device="cpu",
            )
        copy_frames = min(frame_count, output_frames - write_pos)
        if copy_frames <= 0:
            return
        pixels = part[:, :, :copy_frames].float()
        pixels.mul_(model.pixel_std.to(pixels)).add_(model.pixel_mean.to(pixels))
        pixels.clamp_(0.0, 1.0)
        output[:, :, write_pos : write_pos + copy_frames].copy_(
            pixels.to("cpu"), non_blocking=False
        )
        write_pos += copy_frames

    with model_management.cuda_device_context(vae.device):
        for index in range(num_chunks):
            start = index * model.tokens_chunk_size
            stop = start + model.tokens_chunk_size + model.token_overlap
            clip_z = z[:, :, start:stop].to(
                device=vae.device, dtype=vae.vae_dtype
            )
            clip_dec = model._adaptive_decode(clip_z)

            for split in range(split_count):
                frame_start = split * chunk_dec
                frame_stop = min(frame_start + chunk_dec, clip_dec.shape[2])
                part = clip_dec[:, :, frame_start:frame_stop]
                part = part[:, :, model.frame_pre_padding :]
                if split == 0:
                    if overlap is not None:
                        part = model.blend(
                            overlap, part, model.frame_overlap, dim=-3
                        )
                        overlap = None
                    write_part(part)
                else:
                    overlap = part.contiguous()

            if index == num_chunks - 1 and overlap is not None:
                write_part(overlap)
                overlap = None
            del clip_dec, clip_z
            model_management.soft_empty_cache()

    if output is None or write_pos != output_frames:
        raise RuntimeError(
            f"streamed VAE decode wrote {write_pos} of {output_frames} frames"
        )
    return output[0].movedim(0, -1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"comfyui_ref2va_{args.mode}_8g_ws{args.activation_workspace_mib}_"
        f"{args.height}x{args.width}_f{args.frames}_s{args.steps}_{stamp}"
    )
    result_path = output_dir / f"{run_name}.json"
    latent_path = output_dir / f"{run_name}_latent.pt"
    video_path = output_dir / f"{run_name}.mp4"

    budget = initialize_vram_budget(args.target_vram_mib, safety_margin_mib=384)
    result = {
        "status": "running",
        "run_name": run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args),
        "prompt": PROMPT,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "comfyui_vram_mode": "lowvram",
            "text_encoder_device": args.text_encoder_mode,
        },
        "phases": [],
    }
    process_sampler = ProcessSampler(
        interval_seconds=args.sample_interval_ms / 1000.0
    )
    process_sampler.__enter__()

    def phase(name, function):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_allocated = torch.cuda.memory_allocated() / 2**20
        try:
            return function()
        finally:
            torch.cuda.synchronize()
            result["phases"].append({
                "name": name,
                "seconds": time.perf_counter() - start,
                "end_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "start_allocated_mib": start_allocated,
            })

    try:
        # ComfyUI parses flags at import time. Select its real low-VRAM model
        # manager without exposing the benchmark-specific arguments to it.
        sys.argv = [
            sys.argv[0],
            "--lowvram",
            "--reserve-vram",
            str(args.comfy_reserve_vram_gib),
        ]
        sys.path.insert(0, "/opt/ComfyUI")
        os.chdir("/opt/ComfyUI")

        import nodes
        import comfy.model_management as model_management
        from comfy.ldm.minimax.model import PackedLayout
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider,
            BasicScheduler,
            KSamplerSelect,
            RandomNoise,
            SamplerCustomAdvanced,
        )
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import (
            _empty_av_latent,
            _resize,
            adapt_canvas,
        )

        original_get_free_memory = model_management.get_free_memory

        def budget_aware_get_free_memory(dev=None, torch_free_too=False):
            dev = model_management.get_torch_device() if dev is None else dev
            if getattr(dev, "type", None) != "cuda":
                return original_get_free_memory(dev, torch_free_too)
            stats = torch.cuda.memory_stats(dev)
            active = stats["active_bytes.all.current"]
            reserved = stats["reserved_bytes.all.current"]
            allocator_limit = int(budget.allocator_limit_mib * 2**20)
            free_torch = max(0, reserved - active)
            free_unreserved = max(0, allocator_limit - reserved)
            free_total = free_unreserved + free_torch
            return (free_total, free_torch) if torch_free_too else free_total

        model_management.get_free_memory = budget_aware_get_free_memory
        result["memory_policy"] = {
            "allocator_limit_mib": budget.allocator_limit_mib,
            "comfy_reserve_vram_gib": args.comfy_reserve_vram_gib,
            "comfy_free_memory_source": "allocator-budget-aware",
        }

        reference_frames, reference_metadata = phase(
            "reference_video_decode",
            lambda: decode_reference_video(args.source, args.frames),
        )
        result["reference_video"] = reference_metadata

        video_vae = phase(
            "video_vae_load",
            lambda: nodes.VAELoader().load_vae(
                "minimax_h3_video_vae_fp16.safetensors"
            )[0],
        )
        audio_vae = phase(
            "audio_vae_load",
            lambda: nodes.VAELoader().load_vae(
                "minimax_h3_audio_vae_fp32.safetensors"
            )[0],
        )
        normal_vram_state = model_management.vram_state
        if args.text_encoder_mode == "gpu-offload":
            # CLIPLoader chooses its load device from the global VRAM state.
            # NORMAL_VRAM gives the patcher a CUDA execution device while the
            # allocator-aware free-memory check keeps this 32B model initially
            # resident on CPU.
            model_management.vram_state = model_management.VRAMState.NORMAL_VRAM
        try:
            clip = phase(
                "text_encoder_load",
                lambda: nodes.CLIPLoader().load_clip(
                    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    type="minimax",
                    device=(
                        "default"
                        if args.text_encoder_mode == "gpu-offload"
                        else "cpu"
                    ),
                )[0],
            )
        finally:
            model_management.vram_state = normal_vram_state

        # The stock VAE already chunks its math, but its ComfyUI wrapper stages
        # the complete 243-frame BF16 tensor on CUDA first. Preserve the same
        # VAE chunks while streaming their inputs from CPU under the 8 GiB cap.
        video_vae.encode = lambda images: encode_minimax_video_cpu_streamed(
            video_vae, images
        )

        latent, frame_count = _empty_av_latent(
            args.width, args.height, args.frames
        )
        source_h, source_w = reference_frames.shape[1:3]
        canvas_w, canvas_h = adapt_canvas(source_w, source_h)
        if source_w * source_h < canvas_w * canvas_h:
            canvas_w = max(32, round(source_w / 32) * 32)
            canvas_h = max(32, round(source_h / 32) * 32)
        resized_reference = _resize(
            reference_frames, canvas_w, canvas_h, "disabled"
        )[:frame_count]
        valid_frames = int(resized_reference.shape[0])
        while valid_frames % 17 != 5:
            valid_frames -= 1
        resized_reference = resized_reference[:valid_frames]
        qwen_reference = resized_reference[::12].clone()
        reference_latent = phase(
            "reference_video_vae_encode",
            lambda: video_vae.encode(resized_reference),
        )
        del reference_frames, resized_reference
        model_management.unload_all_models()
        model_management.soft_empty_cache()

        ref_items = [{
            "type": "video",
            "data": qwen_reference,
            "timestamps": [index / 2.0 for index in range(len(qwen_reference))],
        }]
        ref_blocks = [{
            "kind": "video",
            "latent_t": int(reference_latent.shape[2]),
            "latent_h": canvas_h // 16,
            "latent_w": canvas_w // 16,
            "ref_audio_t": 0,
            "latent": reference_latent,
            "audio_latent": None,
        }]

        def encode_text_conditioning():
            tokens = clip.tokenize(PROMPT, minimax_ref_items=ref_items)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            return node_helpers.conditioning_set_values(
                conditioning, {"minimax_refs": ref_blocks}
            )

        if args.text_encoder_mode == "gpu-offload":
            # Force layer-at-a-time execution. Ordinary LOW_VRAM can retain
            # several GiB of weights and leave too little room for Qwen3-VL's
            # visual DeepStack activations under the strict 8 GiB budget.
            model_management.vram_state = model_management.VRAMState.NO_VRAM
        process_sampler.begin_window()
        try:
            positive = phase("text_conditioning", encode_text_conditioning)
        finally:
            result["text_conditioning_memory"] = process_sampler.end_window()
            model_management.vram_state = normal_vram_state
        del qwen_reference, clip
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        torch.cuda.synchronize()
        post_text_nvml_mib, post_text_rss_mib = process_sampler.sample()
        result["post_text_unload_memory"] = {
            "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "nvml_process_mib": post_text_nvml_mib,
            "cpu_rss_mib": post_text_rss_mib,
        }

        if args.stop_after_text_conditioning:
            result["status"] = "success"
            return

        model = phase(
            "diffusion_model_load",
            lambda: nodes.UNETLoader().load_unet(
                "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default"
            )[0],
        )
        if args.mode == "streaming":
            load_package(
                "comfyui_seqattn", "/opt/ComfyUI/custom_nodes/ComfyUI-SeqAttn"
            )
            from comfyui_seqattn.nodes import STATE_KEY, patch_minimax_h3_model

            model = patch_minimax_h3_model(
                model,
                activation_workspace_mib=args.activation_workspace_mib,
                kv_chunk_tokens=args.kv_chunk_tokens,
                planner_mode="fit",
            )
            profile_runtime = model.model_options["transformer_options"][STATE_KEY]
            if args.profile_denoise:
                profile_runtime.enable_profile()
        else:
            profile_runtime = None

        video_latent, audio_latent = latent["samples"].unbind()
        cond_tensor, cond_metadata = positive[0]
        layout = PackedLayout(
            int(cond_tensor.shape[1]),
            int(video_latent.shape[2]),
            int(video_latent.shape[3]),
            int(video_latent.shape[4]),
            int(audio_latent.shape[-1]),
            refs=cond_metadata.get("minimax_refs"),
            frame_count=args.frames,
        )
        result["packed_sequence"] = {
            "tokens": int(layout.seq_len),
            "segments": [
                {"start": int(start), "stop": int(stop), "kind": kind}
                for start, stop, kind in layout.segments
            ],
            "latent_shapes": nested_shapes(latent["samples"]),
        }

        noise = RandomNoise.execute(args.seed)[0]
        guider = BasicGuider.execute(model, positive)[0]
        sampler_object = KSamplerSelect.execute("res_multistep")[0]
        sigmas = BasicScheduler.execute(model, "simple", args.steps, 1.0)[0]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        denoise_start = time.perf_counter()
        try:
            if args.cuda_profiler_capture:
                torch.cuda.profiler.start()
            try:
                sampled, _ = SamplerCustomAdvanced.execute(
                    noise, guider, sampler_object, sigmas, latent
                ).result
            finally:
                if args.cuda_profiler_capture:
                    torch.cuda.profiler.stop()
            torch.cuda.synchronize()
            result["denoise"] = {
                "status": "success",
                "seconds": time.perf_counter() - denoise_start,
                "steps": args.steps,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
            if profile_runtime is not None and args.profile_denoise:
                result["seqattn_profile"] = profile_runtime.profile_records
        except Exception as exc:
            result["denoise"] = {
                "status": classify_exception(exc),
                "seconds": time.perf_counter() - denoise_start,
                "failure_message": f"{type(exc).__name__}: {exc}",
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
            raise

        torch.save(sampled, latent_path)
        result["latent_path"] = str(latent_path)
        result["latent_size_bytes"] = latent_path.stat().st_size

        if not args.skip_decode:
            del guider, model, positive, latent
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            output_video_latent = sampled["samples"].unbind()[0]
            decoded = phase(
                "video_vae_decode",
                lambda: decode_minimax_video_cpu_streamed(
                    video_vae, output_video_latent
                ),
            )
            phase("video_write", lambda: save_video(video_path, decoded, fps=24))
            result["video_path"] = str(video_path)
            result["video_size_bytes"] = video_path.stat().st_size
            result["output_video_shape"] = [int(value) for value in decoded.shape]

        result["status"] = "success"
    except Exception as exc:
        result["status"] = classify_exception(exc)
        result["failure_message"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        process_sampler.__exit__(None, None, None)
        finish_result(result, process_sampler, budget)
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(result_path, result)
        print(f"BENCH_RESULT {result_path}", flush=True)
        print(json.dumps({
            "status": result["status"],
            "text_conditioning": next(
                (
                    item
                    for item in result.get("phases", [])
                    if item["name"] == "text_conditioning"
                ),
                None,
            ),
            "text_conditioning_memory": result.get("text_conditioning_memory"),
            "denoise": result.get("denoise"),
            "packed_tokens": result.get("packed_sequence", {}).get("tokens"),
            "nvml_process_peak_mib": result.get("nvml_process_peak_mib"),
            "result_path": str(result_path),
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
