#!/usr/bin/env python3
"""FlashAttention v8/v9 的受控流水实验。

v8 和 v9 使用同一套 D=64 fragment、MMA 和 online softmax。唯一差异是：
v8 在每个 KV tile 前等待 K/V 搬运完成；v9 在当前 tile 的 PV 计算期间，
异步预取下一 tile 的 K/V。脚本先做逐元素一致性检查，再用交替顺序的多轮
中位数测量，避免错误的 fragment 或 stage 映射被误判成加速。
"""

import argparse
from statistics import median

import torch

import ai_infra_ops


def measure(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def compare(v8_fn, v9_fn, warmup, iterations, rounds):
    for fn in (v8_fn, v9_fn):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[], []]
    functions = (v8_fn, v9_fn)
    for round_id in range(rounds):
        order = (0, 1) if round_id % 2 == 0 else (1, 0)
        for index in order:
            samples[index].append(measure(functions[index], iterations))
    return median(samples[0]), median(samples[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--n", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    if args.rounds <= 0 or args.rounds % 2:
        parser.error("--rounds must be a positive even number")

    torch.manual_seed(42)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("B,H,N,mode,kv_tiles,v8_ms,v9_ms,v9_over_v8,v9_speedup,max_diff")

    for n in args.n:
        q = torch.randn(args.batch, args.heads, n, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for causal in (False, True):
            v8_fn = lambda c=causal: ai_infra_ops.flashattention_v8(q, k, v, c)
            v9_fn = lambda c=causal: ai_infra_ops.flashattention_v9(q, k, v, c)

            out8 = v8_fn()
            out9 = v9_fn()
            max_diff = (out8.float() - out9.float()).abs().max().item()
            if max_diff != 0.0:
                raise AssertionError(f"v9 differs from v8: N={n}, causal={causal}, diff={max_diff}")

            t8, t9 = compare(v8_fn, v9_fn, args.warmup, args.iters, args.rounds)
            print(
                f"{args.batch},{args.heads},{n},{'causal' if causal else 'full'},"
                f"{(n + 63) // 64},{t8:.6f},{t9:.6f},{t9 / t8:.3f}x,"
                f"{t8 / t9:.3f}x,{max_diff:.1e}"
            )


if __name__ == "__main__":
    main()
