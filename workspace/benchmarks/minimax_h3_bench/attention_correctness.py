from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from diffsynth.core.attention.streaming import streaming_attention_reference
from .protocol import atomic_write_json


def boundaries(length: int, segments: int) -> torch.Tensor:
    # Deliberately uneven, with a small final segment for tile edge coverage.
    weights = torch.arange(1, segments + 1, dtype=torch.float64)
    cuts = [0]
    for index in range(1, segments):
        cuts.append(round(length * float(weights[:index].sum() / weights.sum())))
    cuts.append(length)
    return torch.tensor(cuts, dtype=torch.int32)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f, expected_f = actual.float(), expected.float()
    delta = actual_f - expected_f
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected_f)),
        "max_absolute_error": float(delta.abs().max()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(
            actual_f.flatten(), expected_f.flatten(), dim=0
        )),
    }


def reference(q, k, v, cu, scale):
    output = torch.empty_like(q)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for start, stop in zip(cu[:-1].tolist(), cu[1:].tolist()):
            tile = torch.nn.functional.scaled_dot_product_attention(
                q[start:stop].transpose(0, 1).unsqueeze(0),
                k[start:stop].transpose(0, 1).unsqueeze(0),
                v[start:stop].transpose(0, 1).unsqueeze(0),
                scale=scale,
            )
            output[start:stop] = tile.squeeze(0).transpose(0, 1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[257, 1023, 3072])
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--q-block", type=int, default=128)
    parser.add_argument("--kv-block", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    args = parser.parse_args()
    # The V0 recurrence is FP32, but CUDA matmul may otherwise silently use
    # TF32 while the SDPA math reference does not.  Correctness runs disable it
    # explicitly; performance runs retain the environment default.
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for length, segment_count, dtype in itertools.product(
        args.lengths, args.segments, (torch.float32, torch.bfloat16)
    ):
        torch.manual_seed(0)
        q = torch.randn(length, args.heads, args.head_dim, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = boundaries(length, segment_count)
        scale = args.head_dim ** -0.5
        expected = reference(q, k, v, cu, scale).cpu()
        actual = streaming_attention_reference(
            q.cpu(), k.cpu(), v.cpu(), cu, args.q_block, args.kv_block, "cuda", scale
        )
        error = metrics(actual, expected)
        passed = True
        if dtype == torch.float32:
            passed = error["relative_l2"] <= 1e-5 and error["max_absolute_error"] <= 1e-4
        rows.append({
            "length": length, "segments": segment_count, "segment_lengths": torch.diff(cu).tolist(),
            "dtype": str(dtype).removeprefix("torch."), "error_metrics": error, "passed": passed,
        })
    status = "success" if all(row["passed"] for row in rows) else "numerical_failure"
    atomic_write_json(
        args.output,
        {"status": status, "scope": "attention_correctness", "cases": rows},
    )


if __name__ == "__main__":
    main()
