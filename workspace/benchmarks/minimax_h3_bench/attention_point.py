from __future__ import annotations

import argparse
import importlib
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch

from diffsynth.core.attention.attention import attention_forward
from diffsynth.core.attention.streaming import StreamingStats, streaming_attention_reference

from .protocol import (
    ProcessSampler, atomic_write_json, classify_exception, environment_metadata,
    finish_result, initialize_vram_budget,
)

attention_module = importlib.import_module("diffsynth.core.attention.attention")


def make_cpu_qkv(tokens: int, heads: int, head_dim: int, seed: int, pin: bool):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = [torch.randn(tokens, heads, head_dim, generator=generator, dtype=torch.bfloat16) for _ in range(3)]
    if pin:
        tensors = [tensor.pin_memory() for tensor in tensors]
    return tensors


def run_attention(mode, q_cpu, k_cpu, v_cpu, cu, q_block, kv_block, stats=None):
    if mode == "streaming":
        phase = nullcontext() if stats is None else stats.phase("B", "cuda")
        with phase:
            return streaming_attention_reference(
                q_cpu, k_cpu, v_cpu, cu, q_block, kv_block, "cuda",
                stats=stats,
            )
    if not attention_module.FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError("formal baseline requires FlashAttention 2")
    attention_module.ATTENTION_IMPLEMENTATION = "flash_attention_2"
    outputs = []
    for start, stop in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if start == stop:
            continue
        q = q_cpu[start:stop].to("cuda").transpose(0, 1).unsqueeze(0)
        k = k_cpu[start:stop].to("cuda").transpose(0, 1).unsqueeze(0)
        v = v_cpu[start:stop].to("cuda").transpose(0, 1).unsqueeze(0)
        outputs.append(attention_forward(q, k, v).squeeze(0).transpose(0, 1).cpu())
    return torch.cat(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "streaming"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--used-tokens", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-block", type=int, default=2048)
    parser.add_argument("--kv-block", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-vram-mib", type=int, default=4096)
    parser.add_argument("--warmup-tokens", type=int, default=3072)
    parser.add_argument("--instrument-phases", action="store_true")
    parser.add_argument("--output", type=Path, default=os.environ.get("MINIMAX_H3_RESULT_JSON"))
    args = parser.parse_args()
    if args.output is None:
        parser.error("--output or MINIMAX_H3_RESULT_JSON is required")

    result = {
        "status": "runtime_error", "mode": args.mode, "scope": "attention",
        "frames": args.frames, "packed_tokens": args.tokens,
        "segment_lengths": None, "dtype": "bfloat16", "seed": args.seed,
        "q_block": args.q_block, "kv_block": args.kv_block,
        "projection_chunk": None, "mlp_chunk": None,
        "attention_backend": "flash_attention_2" if args.mode == "baseline" else "torch_online_softmax_v0",
        "failure_message": None,
    }
    budget = None
    try:
        budget = initialize_vram_budget(args.target_vram_mib)
        used = args.tokens if args.used_tokens is None else args.used_tokens
        if not 0 < used <= args.tokens:
            raise ValueError("used-tokens must be within (0, tokens]")
        cu = torch.tensor([0, used, args.tokens], dtype=torch.int32)
        result["segment_lengths"] = torch.diff(cu).tolist()
        pin = args.mode == "streaming"

        warm_tokens = min(args.warmup_tokens, args.tokens)
        warm_q, warm_k, warm_v = make_cpu_qkv(warm_tokens, args.heads, args.head_dim, args.seed, pin)
        run_attention(
            args.mode, warm_q, warm_k, warm_v,
            torch.tensor([0, warm_tokens], dtype=torch.int32), args.q_block, args.kv_block,
        )
        torch.cuda.synchronize()
        del warm_q, warm_k, warm_v
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        q_cpu, k_cpu, v_cpu = make_cpu_qkv(args.tokens, args.heads, args.head_dim, args.seed, pin)
        stats = StreamingStats(instrument_phases=args.instrument_phases) if pin else None
        with ProcessSampler() as sampler:
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = run_attention(
                args.mode, q_cpu, k_cpu, v_cpu, cu, args.q_block, args.kv_block, stats,
            )
            torch.cuda.synchronize()
            result["total_seconds"] = time.perf_counter() - started
        if not torch.isfinite(output).all():
            result["status"] = "numerical_failure"
            result["failure_message"] = "attention output contains NaN or Inf"
        else:
            result["status"] = "success"
        result["tokens_per_second"] = args.tokens / result["total_seconds"]
        if stats is not None:
            result.update(stats.as_dict())
            qkv_bytes = sum(t.numel() * t.element_size() for t in (q_cpu, k_cpu, v_cpu))
            result["cpu_activation_peak_bytes"] = qkv_bytes + output.numel() * output.element_size()
            result["pinned_memory_peak_bytes"] = qkv_bytes
            result["pinned_memory_peak_mib"] = qkv_bytes / 2**20
        else:
            result.update({"h2d_bytes": 3 * q_cpu.numel() * q_cpu.element_size(), "d2h_bytes": output.numel() * output.element_size()})
            result["pinned_memory_peak_mib"] = 0.0
        finish_result(result, sampler, budget)
    except BaseException as error:
        result["status"] = classify_exception(error)
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        if "sampler" not in locals():
            sampler = ProcessSampler()
        finish_result(result, sampler, budget)
    result["environment"] = environment_metadata()
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
