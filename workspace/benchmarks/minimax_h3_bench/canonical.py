from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
from .protocol import atomic_write_json


def dimensions(frames: int, height: int, width: int) -> tuple[int, int, int, int]:
    video_t = ((frames - 5) // 17) * 5 + 2
    latent_h, latent_w = height // 16, width // 16
    audio_t = round(frames / 24.0 * 40.0)
    return video_t, latent_h, latent_w, audio_t


def build_manifest(frames: int, height: int = 480, width: int = 832, text_len: int = 256) -> dict:
    video_t, latent_h, latent_w, audio_t = dimensions(frames, height, width)
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    packed = builder._build_packed_fl2va(
        text_len, video_t, latent_h, latent_w, audio_t, keyframe_indices=[]
    )
    return {
        "frames": frames,
        "height": height,
        "width": width,
        "text_len": text_len,
        "video_latent_t": video_t,
        "video_latent_h": latent_h,
        "video_latent_w": latent_w,
        "audio_latent_t": audio_t,
        "packed_tokens": int(packed["seq_len"]),
        "segment_lengths": torch.diff(packed["cu_seqlens"]).tolist(),
        "packed": packed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = [build_manifest(f, args.height, args.width, args.text_len) for f in args.frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    os.close(fd)
    try:
        torch.save(manifests, temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    printable = [{k: v for k, v in item.items() if k != "packed"} for item in manifests]
    atomic_write_json(args.output.with_suffix(".json"), printable)


if __name__ == "__main__":
    main()
