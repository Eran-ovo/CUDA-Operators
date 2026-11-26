#!/usr/bin/env python3
"""Compare packed D64 GQA with segmented and packed PyTorch Flash Attention."""

import argparse
import json
from itertools import accumulate
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import ai_infra_ops
from bench_final import balanced_orders, benchmark_case, gpu_snapshot, run_optional


def segmented_attention(q, k, v, lengths):
    return torch.cat([
        F.scaled_dot_product_attention(
            qs.transpose(0, 1).unsqueeze(0),
            ks.transpose(0, 1).unsqueeze(0),
            vs.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
        )[0].transpose(0, 1)
        for qs, ks, vs in zip(q.split(lengths), k.split(lengths), v.split(lengths), strict=True)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    warmup, iters, rounds = 20, 100, 12
    rows = []
    before = gpu_snapshot()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for lengths in ((1, 17, 65, 128), (128, 512, 1024), (256, 384, 512, 640), (1024, 2048)):
            num_tokens = sum(lengths)
            q = torch.randn(num_tokens, 14, 64, dtype=torch.float16, device=device)
            k = torch.randn(num_tokens, 2, 64, dtype=q.dtype, device=device)
            v = torch.randn_like(k)
            offsets = tuple(accumulate(lengths, initial=0))
            cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
            plan = ai_infra_ops.FlashAttentionVarlenPlan(offsets, device)
            max_length = max(lengths)
            implementations = [
                ("segmented_sdpa", lambda: segmented_attention(q, k, v, lengths)),
                ("varlen_sdpa", lambda: torch.ops.aten._flash_attention_forward(
                    q, k, v, cu_seqlens, cu_seqlens, max_length, max_length, 0.0, True, False,
                )[0]),
                ("varlen_cuda", lambda: ai_infra_ops.flashattention_varlen(q, k, v, plan)),
            ]
            rows.extend(benchmark_case(
                suite="flash_varlen_d64", case="+".join(map(str, lengths)),
                shape=f"[{num_tokens},14:2,64]", mode="causal",
                implementations=implementations, baseline="segmented_sdpa",
                warmup=warmup, iters=iters, rounds=rounds,
            ))
    repo = Path(__file__).resolve().parent
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": run_optional(["git", "rev-parse", "HEAD"], repo),
        "git_dirty": bool(run_optional(["git", "status", "--porcelain"], repo)),
        "warmup": warmup,
        "rounds": rounds,
        "order": [
            [name for name, _ in order] for order in balanced_orders(implementations, rounds)
        ],
        "gpu_before": before,
        "gpu_after": gpu_snapshot(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
