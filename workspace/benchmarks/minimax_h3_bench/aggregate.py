from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "status", "mode", "scope", "frames", "packed_tokens", "dtype", "seed",
    "q_block", "kv_block", "projection_chunk", "target_vram_mib",
    "torch_peak_allocated_mib", "torch_peak_reserved_mib", "nvml_process_peak_mib",
    "cpu_rss_peak_mib", "pinned_memory_peak_mib", "total_seconds", "h2d_bytes",
    "d2h_bytes", "attention_backend", "failure_message",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        paths.extend(sorted(item.glob("*.json")) if item.is_dir() else [item])
    rows = [json.loads(path.read_text()) for path in paths]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
