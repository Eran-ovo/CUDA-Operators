"""Benchmark tiled D128 attention, v6, and PyTorch Flash SDPA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import ai_infra_ops
from bench_final import balanced_orders, benchmark_case, gpu_snapshot, run_optional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--n", type=int, nargs="+", default=[31, 128, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.heads, args.warmup, args.iters, args.rounds, *args.n) < 1:
        parser.error("batch, heads, lengths, warmup, iters and rounds must be positive")

    torch.manual_seed(42)
    rows = []
    before = gpu_snapshot()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for n in args.n:
            shape = (args.batch, args.heads, n, 128)
            q = torch.randn(shape, device="cuda", dtype=torch.float16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            for causal in (False, True):
                implementations = [
                    ("v6", lambda: ai_infra_ops.flashattention_v6(q, k, v, causal)),
                    ("d128", lambda: ai_infra_ops.flashattention_d128(q, k, v, causal)),
                    ("sdpa_flash", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)),
                ]
                rows.extend(benchmark_case(
                    suite="flash_d128", case=str(n), shape=str(shape),
                    mode="causal" if causal else "full",
                    implementations=implementations, baseline="sdpa_flash",
                    warmup=args.warmup, iters=args.iters, rounds=args.rounds,
                ))

    if args.output is not None:
        repo = Path(__file__).resolve().parent
        result = {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_commit": run_optional(["git", "rev-parse", "HEAD"], repo),
            "git_dirty": bool(run_optional(["git", "status", "--porcelain"], repo)),
            "warmup": args.warmup,
            "rounds": args.rounds,
            "order": [
                [name for name, _ in order]
                for order in balanced_orders(implementations, args.rounds)
            ],
            "gpu_before": before,
            "gpu_after": gpu_snapshot(),
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
