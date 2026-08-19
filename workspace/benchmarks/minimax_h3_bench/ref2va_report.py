from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .protocol import atomic_write_json

RUN_PREFIXES = {
    "streaming_8g": "ref2va_streaming_8g_2step_streaming_",
    "native_8g": "ref2va_native_8g_2step_native_",
    "native_32g": "ref2va_native_32g_2step_native_",
}
COLORS = {
    "streaming_8g": "#0f766e",
    "native_8g": "#dc2626",
    "native_32g": "#f59e0b",
}
LABELS = {
    "streaming_8g": "Streaming 8 GiB",
    "native_8g": "Native 8 GiB",
    "native_32g": "Native 32 GiB",
}


def latest_result(directory: Path, prefix: str) -> tuple[Path, dict]:
    candidates = sorted(directory.glob(f"{prefix}*.json"))
    if not candidates:
        raise FileNotFoundError(f"no result matching {prefix}*.json in {directory}")
    path = candidates[-1]
    return path, json.loads(path.read_text())


def host_path(path: str, workspace: Path) -> Path:
    if path.startswith("/workspace/"):
        return workspace / path.removeprefix("/workspace/")
    return Path(path)


def numerical_metrics(streaming: dict, native: dict, workspace: Path) -> dict | None:
    if streaming.get("status") != "success" or native.get("status") != "success":
        return None
    streaming_path = host_path(streaming["latent_path"], workspace)
    native_path = host_path(native["latent_path"], workspace)
    streaming_latents = torch.load(
        streaming_path, map_location="cpu", weights_only=True
    )
    native_latents = torch.load(native_path, map_location="cpu", weights_only=True)
    output = {}
    for name in sorted(native_latents):
        reference = native_latents[name].float().reshape(-1)
        actual = streaming_latents[name].float().reshape(-1)
        difference = actual - reference
        output[name] = {
            "shape": list(native_latents[name].shape),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference)
            ),
            "max_abs": float(difference.abs().max()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(actual, reference, dim=0)
            ),
        }
    return output


def phase_seconds(result: dict, name: str) -> float:
    return sum(
        float(record["seconds"])
        for record in result.get("phases", [])
        if record["name"] == name
    )


def result_summary(result: dict) -> dict:
    denoise = result.get("denoise") or {}
    return {
        "status": result["status"],
        "failure_message": result.get("failure_message"),
        "packed_tokens": (result.get("sequence") or {}).get("packed_tokens"),
        "pipeline_seconds": phase_seconds(result, "pipeline_total"),
        "denoise_seconds": denoise.get("total_seconds"),
        "step_seconds": denoise.get("per_step_seconds"),
        "model_load_seconds": phase_seconds(result, "model_load"),
        "reference_media_read_seconds": phase_seconds(result, "reference_media_read")
        + phase_seconds(result, "reference_audio_pyav_fallback"),
        "media_write_seconds": phase_seconds(result, "media_write"),
        "video_decode_seconds": sum(
            result.get("core_timings", {}).get("video_decode_seconds", [])
        ),
        "audio_decode_seconds": sum(
            result.get("core_timings", {}).get("audio_decode_seconds", [])
        ),
        "nvml_process_peak_mib": result.get("nvml_process_peak_mib"),
        "cpu_rss_peak_mib": result.get("cpu_rss_peak_mib"),
        "torch_peak_allocated_mib": result.get("torch_peak_allocated_mib"),
        "torch_peak_reserved_mib": result.get("torch_peak_reserved_mib"),
        "media_path": result.get("media_path"),
        "trace_path": (result.get("memory_sampling") or {}).get("trace_path"),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0f172a",
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )


def comparison_chart(results: dict[str, dict], output: Path) -> None:
    configure_style()
    keys = list(RUN_PREFIXES)
    labels = [LABELS[key] for key in keys]
    colors = [COLORS[key] for key in keys]
    gpu = [float(results[key].get("nvml_process_peak_mib") or 0) / 1024 for key in keys]
    cpu = [float(results[key].get("cpu_rss_peak_mib") or 0) / 1024 for key in keys]
    denoise = [
        float((results[key].get("denoise") or {}).get("total_seconds") or 0)
        for key in keys
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(keys))
    width = 0.36
    axes[0].bar(
        [value - width / 2 for value in x], gpu, width, label="GPU NVML", color=colors
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        cpu,
        width,
        label="CPU RSS",
        color=colors,
        alpha=0.42,
        hatch="//",
    )
    axes[0].axhline(
        8, color="#dc2626", linestyle="--", linewidth=1.5, label="8 GiB target"
    )
    axes[0].set_title("Whole-process memory peak")
    axes[0].set_ylabel("GiB")
    axes[0].set_xticks(list(x), labels, rotation=12)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(labels, denoise, color=colors)
    axes[1].set_title("Two-step denoise wall time")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=12)
    for index, result in enumerate(results.values()):
        if result["status"] != "success":
            axes[1].text(
                index,
                max(denoise + [1]) * 0.04,
                result["status"].upper(),
                ha="center",
                color="#991b1b",
                fontweight="bold",
            )
    fig.suptitle(
        "MiniMax-H3 Ref2VA · 1344×768 · 243 frames · 147K-class sequence",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def read_trace(path: Path) -> tuple[list[float], list[float], list[float]]:
    elapsed, nvml, rss = [], [], []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            elapsed.append(float(row["elapsed_ms"]) / 1000.0)
            nvml.append(float(row["nvml_process_mib"]) / 1024.0)
            rss.append(float(row["cpu_rss_mib"]) / 1024.0)
    return elapsed, nvml, rss


def timeline_chart(results: dict[str, dict], workspace: Path, output: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for key in ("streaming_8g", "native_32g"):
        trace = (results[key].get("memory_sampling") or {}).get("trace_path")
        if not trace:
            continue
        elapsed, nvml, rss = read_trace(host_path(trace, workspace))
        axes[0].plot(elapsed, nvml, label=LABELS[key], color=COLORS[key], linewidth=1.6)
        axes[1].plot(elapsed, rss, label=LABELS[key], color=COLORS[key], linewidth=1.6)
    axes[0].axhline(
        8, color="#dc2626", linestyle="--", linewidth=1.3, label="8 GiB target"
    )
    axes[0].set_title("PID GPU memory")
    axes[0].set_ylabel("GiB")
    axes[1].set_title("Process CPU RSS")
    axes[1].set_ylabel("GiB")
    axes[1].set_xlabel("End-to-end elapsed time (seconds)")
    for axis in axes:
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.legend(frameon=False)
    fig.suptitle("Ref2VA end-to-end memory timeline", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--comparison-svg", type=Path, required=True)
    parser.add_argument("--timeline-svg", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = {}
    paths = {}
    for key, prefix in RUN_PREFIXES.items():
        path, result = latest_result(args.input_dir, prefix)
        loaded[key] = result
        paths[key] = str(path)

    summary = {
        "source_results": paths,
        "runs": {key: result_summary(result) for key, result in loaded.items()},
        "sequence": loaded["streaming_8g"].get("sequence"),
        "numerical": numerical_metrics(
            loaded["streaming_8g"], loaded["native_32g"], args.workspace
        ),
    }
    atomic_write_json(args.summary, summary)
    comparison_chart(loaded, args.comparison_svg)
    timeline_chart(loaded, args.workspace, args.timeline_svg)
    os.chmod(args.comparison_svg, 0o644)
    os.chmod(args.timeline_svg, 0o644)


if __name__ == "__main__":
    main()
