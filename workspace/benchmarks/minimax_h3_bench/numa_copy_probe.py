from __future__ import annotations

import argparse
import json
import time

import torch


def measure(direction: str, cpu: torch.Tensor, gpu: torch.Tensor, repeats: int) -> float:
    for _ in range(2):
        if direction == "h2d":
            gpu.copy_(cpu, non_blocking=True)
        else:
            cpu.copy_(gpu, non_blocking=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        if direction == "h2d":
            gpu.copy_(cpu, non_blocking=True)
        else:
            cpu.copy_(gpu, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return cpu.numel() * cpu.element_size() * repeats / elapsed / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    elements = args.mib * 2**20 // 2
    cpu = torch.empty(elements, dtype=torch.bfloat16, pin_memory=True)
    cpu.fill_(1)
    gpu = torch.empty_like(cpu, device="cuda")
    print(json.dumps({
        "mib": args.mib,
        "repeats": args.repeats,
        "h2d_gbps": measure("h2d", cpu, gpu, args.repeats),
        "d2h_gbps": measure("d2h", cpu, gpu, args.repeats),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
