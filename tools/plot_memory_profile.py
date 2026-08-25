#!/usr/bin/env python3
"""Render a phase-aware GPU memory profile from the example runner trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SOURCE_COLORS = {
    "torch_allocated": "#2563eb",
    "torch_reserved_free": "#94a3b8",
    "non_torch": "#f59e0b",
}

PHASE_LABELS = {
    "video_vae_load": "Video VAE load",
    "qwen_load": "Qwen load",
    "reference_video_decode": "Reference decode",
    "conditioning": "Conditioning",
    "diffusion_model_load": "DiT load",
    "denoise": "Denoise",
    "video_vae_reload": "Video VAE reload",
    "video_decode": "Video decode",
    "audio_vae_load": "Audio VAE load",
    "audio_decode": "Audio decode",
    "media_write": "Media write",
    "media_probe": "Media probe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    columns = {
        "elapsed_ms": [],
        "nvml_process_mib": [],
        "torch_allocated_mib": [],
        "torch_reserved_mib": [],
    }
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for name in columns:
                columns[name].append(float(row[name]))
    return {name: np.asarray(values) for name, values in columns.items()}


def memory_sources(trace: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    total = trace["nvml_process_mib"]
    allocated = np.minimum(trace["torch_allocated_mib"], total)
    reserved = np.maximum(
        allocated,
        np.minimum(trace["torch_reserved_mib"], total),
    )
    reserved_free = reserved - allocated
    non_torch = total - reserved
    return total, allocated, reserved_free, non_torch


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    trace = load_trace(args.trace)
    total, allocated, reserved_free, non_torch = memory_sources(trace)
    elapsed_seconds = trace["elapsed_ms"] / 1000.0
    elapsed_minutes = elapsed_seconds / 60.0
    phases = result["phases"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#1e293b",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    figure = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(2.4, 1.7, 1.25))
    timeline = figure.add_subplot(grid[0])
    phase_peaks = figure.add_subplot(grid[1])
    denoise_zoom = figure.add_subplot(grid[2])

    timeline.stackplot(
        elapsed_minutes,
        allocated,
        reserved_free,
        non_torch,
        colors=[
            SOURCE_COLORS["torch_allocated"],
            SOURCE_COLORS["torch_reserved_free"],
            SOURCE_COLORS["non_torch"],
        ],
        alpha=0.88,
        linewidth=0,
    )
    timeline.plot(elapsed_minutes, total, color="#111827", linewidth=0.8)
    timeline.axhline(
        result["configuration"]["target_vram_mib"],
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label="8,192 MiB target",
    )
    for index, phase in enumerate(phases, start=1):
        start = phase["start_elapsed_ms"] / 60000.0
        end = phase["end_elapsed_ms"] / 60000.0
        timeline.axvspan(
            start,
            end,
            color="#0f172a" if index % 2 else "#64748b",
            alpha=0.035,
            linewidth=0,
        )
        midpoint = (start + end) / 2.0
        timeline.text(
            midpoint,
            1.012 + 0.035 * (index % 2),
            str(index),
            transform=timeline.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )

    source_legend = [
        Patch(
            facecolor=SOURCE_COLORS["torch_allocated"],
            label="Torch allocated",
        ),
        Patch(
            facecolor=SOURCE_COLORS["torch_reserved_free"],
            label="Torch reserved, unused",
        ),
        Patch(
            facecolor=SOURCE_COLORS["non_torch"],
            label="Non-Torch / AIMDO / CUDA context",
        ),
    ]
    handles, labels = timeline.get_legend_handles_labels()
    timeline.legend(
        source_legend + handles,
        [item.get_label() for item in source_legend] + labels,
        loc="upper right",
        frameon=False,
        ncol=2,
    )
    timeline.set_title("Whole-process GPU memory timeline")
    timeline.set_ylabel("Process GPU memory (MiB)")
    timeline.set_xlabel("Elapsed time (minutes)")
    timeline.set_ylim(0, result["configuration"]["target_vram_mib"] * 1.08)
    timeline.grid(axis="y", color="#e2e8f0", linewidth=0.7)

    peak_allocated = []
    peak_reserved_free = []
    peak_non_torch = []
    phase_names = []
    phase_peak_totals = []
    for phase in phases:
        mask = (
            (trace["elapsed_ms"] >= phase["start_elapsed_ms"])
            & (trace["elapsed_ms"] <= phase["end_elapsed_ms"])
        )
        indices = np.flatnonzero(mask)
        if not len(indices):
            peak_index = int(np.argmin(np.abs(trace["elapsed_ms"] - phase["end_elapsed_ms"])))
        else:
            peak_index = int(indices[np.argmax(total[indices])])
        peak_allocated.append(allocated[peak_index])
        peak_reserved_free.append(reserved_free[peak_index])
        peak_non_torch.append(non_torch[peak_index])
        phase_peak_totals.append(total[peak_index])
        phase_names.append(PHASE_LABELS.get(phase["name"], phase["name"]))

    positions = np.arange(len(phases))
    phase_peaks.bar(
        positions,
        peak_allocated,
        color=SOURCE_COLORS["torch_allocated"],
    )
    phase_peaks.bar(
        positions,
        peak_reserved_free,
        bottom=peak_allocated,
        color=SOURCE_COLORS["torch_reserved_free"],
    )
    phase_peaks.bar(
        positions,
        peak_non_torch,
        bottom=np.asarray(peak_allocated) + np.asarray(peak_reserved_free),
        color=SOURCE_COLORS["non_torch"],
    )
    for position, peak in zip(positions, phase_peak_totals, strict=True):
        phase_peaks.text(
            position,
            peak + 120,
            f"{peak:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e293b",
        )
    phase_peaks.axhline(
        result["configuration"]["target_vram_mib"],
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
    )
    phase_peaks.set_title("GPU memory composition at each phase's NVML peak sample")
    phase_peaks.set_ylabel("Process GPU memory (MiB)")
    phase_peaks.set_xticks(positions, phase_names, rotation=35, ha="right")
    phase_peaks.set_ylim(0, result["configuration"]["target_vram_mib"] * 1.12)
    phase_peaks.grid(axis="y", color="#e2e8f0", linewidth=0.7)

    denoise = next(phase for phase in phases if phase["name"] == "denoise")
    denoise_mask = (
        (trace["elapsed_ms"] >= denoise["start_elapsed_ms"])
        & (trace["elapsed_ms"] <= denoise["end_elapsed_ms"])
    )
    denoise_elapsed = (
        trace["elapsed_ms"][denoise_mask] - denoise["start_elapsed_ms"]
    ) / 60000.0
    denoise_zoom.plot(
        denoise_elapsed,
        total[denoise_mask],
        color="#111827",
        linewidth=1.0,
        label="NVML process total",
    )
    denoise_zoom.plot(
        denoise_elapsed,
        trace["torch_reserved_mib"][denoise_mask],
        color=SOURCE_COLORS["torch_reserved_free"],
        linewidth=0.9,
        label="Torch reserved",
    )
    denoise_zoom.plot(
        denoise_elapsed,
        trace["torch_allocated_mib"][denoise_mask],
        color=SOURCE_COLORS["torch_allocated"],
        linewidth=0.9,
        label="Torch allocated",
    )
    steady_value = float(np.median(total[denoise_mask][len(total[denoise_mask]) // 5 :]))
    denoise_zoom.axhline(
        steady_value,
        color="#16a34a",
        linestyle=":",
        linewidth=1.2,
        label=f"Steady NVML median {steady_value:,.0f} MiB",
    )
    denoise_zoom.set_title("Denoise memory stability across all 20 forwards")
    denoise_zoom.set_ylabel("Memory (MiB)")
    denoise_zoom.set_xlabel("Denoise elapsed time (minutes)")
    denoise_zoom.grid(color="#e2e8f0", linewidth=0.7)
    denoise_zoom.legend(loc="lower right", frameon=False, ncol=2)

    phase_key = "   ".join(
        f"{index}: {PHASE_LABELS.get(phase['name'], phase['name'])}"
        for index, phase in enumerate(phases, start=1)
    )
    figure.suptitle(
        "MiniMax-H3 Ref2VA 0.4.0 fused DiT: 1344x768, 124 frames, 20 steps",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.002,
        phase_key,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#475569",
        wrap=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
