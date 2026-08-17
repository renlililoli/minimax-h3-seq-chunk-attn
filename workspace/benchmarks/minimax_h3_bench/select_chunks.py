from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--max-nvml-mib", type=float, default=3968)
    parser.add_argument("--tie-percent", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for path in args.inputs:
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    rows = [json.loads(path.read_text()) for path in paths]
    eligible = [
        row for row in rows
        if row.get("status") == "success"
        and row.get("nvml_process_peak_mib", float("inf")) <= args.max_nvml_mib
    ]
    if not eligible:
        raise SystemExit("no eligible chunk configuration")
    fastest = min(row["total_seconds"] for row in eligible)
    tied = [row for row in eligible if row["total_seconds"] <= fastest * (1 + args.tie_percent / 100)]
    winner = max(
        tied,
        key=lambda row: (row.get("q_block", 0), row.get("kv_block", 0), row.get("projection_chunk", 0)),
    )
    payload = {
        "selection_rule": {
            "max_nvml_mib": args.max_nvml_mib,
            "tie_percent": args.tie_percent,
            "eligible_count": len(eligible),
            "tied_count": len(tied),
            "winner_requires_retest": True,
        },
        "winner": winner,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
