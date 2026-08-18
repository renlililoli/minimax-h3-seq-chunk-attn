#!/usr/bin/env python3
"""Decode MiniMax-H3 latents saved before the VAE boundary.

This runner deliberately does not execute the text encoder or DiT.  It exists so
an expensive denoise result remains useful if media decoding or muxing fails.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from diffsynth.core import ModelConfig
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from diffsynth.utils.data import save_video
from diffsynth.utils.data.audio_video import write_video_audio


def atomic_write_json(path, value):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def load_latents(path):
    value = torch.load(path, map_location="cpu")
    if torch.is_tensor(value):
        return {"video": value}
    if not isinstance(value, dict):
        raise TypeError(f"Expected Tensor or dict, got {type(value).__name__}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("latent_path")
    parser.add_argument("output_path")
    parser.add_argument("--model-dir", default="/models/MiniMax-H3-NF4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio-sample-rate", type=int, default=32000)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--vram-limit-gib", type=float, default=24.0)
    args = parser.parse_args()

    latent_path = Path(args.latent_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".decode.json")
    latents = load_latents(latent_path)
    if "video" not in latents:
        raise KeyError("The latent file does not contain a 'video' tensor")

    result = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "latent_path": str(latent_path),
        "latent_keys": sorted(latents),
        "latent_shapes": {
            key: list(value.shape)
            for key, value in latents.items()
            if torch.is_tensor(value)
        },
        "output_path": str(output_path),
    }
    atomic_write_json(json_path, result)

    model_dir = Path(args.model_dir)
    common_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    model_configs = [
        ModelConfig(
            path=str(model_dir / "video_vae_nf4.safetensors"),
            **common_config,
        )
    ]
    if "audio" in latents:
        model_configs.append(
            ModelConfig(
                path=str(model_dir / "audio_vae_nf4.safetensors"),
                **common_config,
            )
        )

    try:
        load_started = time.perf_counter()
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            processor_config=None,
            vram_limit=args.vram_limit_gib,
        )
        result["model_load_seconds"] = time.perf_counter() - load_started

        pipe.load_models_to_device(["video_vae"])
        decode_started = time.perf_counter()
        video_tensor = pipe.video_vae.decode_video(
            latents["video"].to("cuda"),
            dtype=torch.bfloat16,
            tiled=True,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            postprocess_device="cpu",
        )
        torch.cuda.synchronize()
        result["video_decode_seconds"] = time.perf_counter() - decode_started
        video = pipe.vae_output_to_video(video_tensor, min_value=0, max_value=1)
        result["video_frames"] = len(video)

        if "audio" in latents:
            pipe.load_models_to_device(["audio_vae"])
            audio_started = time.perf_counter()
            waveform = pipe.audio_vae.decode_audio(
                latents["audio"].to("cuda"), dtype=torch.bfloat16
            )
            torch.cuda.synchronize()
            result["audio_decode_seconds"] = time.perf_counter() - audio_started
            audio = pipe.output_audio_format_check(waveform)
            write_video_audio(
                video=video,
                audio=audio,
                output_path=str(output_path),
                fps=args.fps,
                audio_sample_rate=args.audio_sample_rate,
            )
            result["has_audio"] = True
        else:
            save_video(video, str(output_path), fps=args.fps)
            result["has_audio"] = False

        result["output_size_bytes"] = output_path.stat().st_size
        result["status"] = "success"
    except BaseException as exc:
        result["status"] = "failure"
        result["failure_message"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if torch.cuda.is_available():
            result["torch_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 2**20
            )
            result["torch_peak_reserved_mib"] = (
                torch.cuda.max_memory_reserved() / 2**20
            )
        atomic_write_json(json_path, result)
        print(f"DECODE_RESULT {json_path}", flush=True)


if __name__ == "__main__":
    main()
