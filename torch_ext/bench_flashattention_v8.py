#!/usr/bin/env python3
"""FlashAttention v7/v8 受控对照实验。

v7 与 v8 的数学路径完全相同，v8 只替换 Q/K/V 数据通路：
  - v7: scalar global load -> [D+1] shared -> scalar fragment load
  - v8: 16B cp.async -> [D+8] shared -> ldmatrix fragment load

使用交替测试顺序和多轮中位数，降低笔记本 GPU 温度、频率和固定测试顺序
对结果的影响。脚本同时检查两版输出逐元素一致，避免把错误的 fragment mapping
误判成性能提升。
"""

import argparse
from statistics import median

import torch

import ai_infra_ops


def measure_cuda(fn, iterations):
    """使用同一 CUDA stream 上的 Event 测量平均 device 时间。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def compare_pair(v7_fn, v8_fn, warmup, iterations, rounds):
    for fn in (v7_fn, v8_fn):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[], []]
    functions = (v7_fn, v8_fn)
    for round_id in range(rounds):
        # 偶数轮 v7 先跑，奇数轮 v8 先跑；每版处于相同的顺序位置。
        order = (0, 1) if round_id % 2 == 0 else (1, 0)
        for index in order:
            samples[index].append(measure_cuda(functions[index], iterations))
    return median(samples[0]), median(samples[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--n", type=int, nargs="+", default=[128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    if args.rounds <= 0 or args.rounds % 2:
        parser.error("--rounds must be a positive even number")

    torch.manual_seed(42)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("B,H,N,mode,kv_tiles,v7_ms,v8_ms,v8_over_v7,v8_speedup,max_diff")

    for n in args.n:
        q = torch.randn(
            args.batch,
            args.heads,
            n,
            64,
            device="cuda",
            dtype=torch.float16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        for causal in (False, True):
            v7_fn = lambda c=causal: ai_infra_ops.flashattention_v7(q, k, v, c)
            v8_fn = lambda c=causal: ai_infra_ops.flashattention_v8(q, k, v, c)

            # ldmatrix mapping 必须保持完全相同的 MMA 输入顺序；当前两版理论上
            # 应逐元素一致，而不只是落在一个宽松浮点容差内。
            out7 = v7_fn()
            out8 = v8_fn()
            max_diff = (out8.float() - out7.float()).abs().max().item()
            if max_diff != 0.0:
                raise AssertionError(
                    f"v8 differs from v7 at N={n}, causal={causal}: {max_diff}"
                )

            t7, t8 = compare_pair(
                v7_fn, v8_fn, args.warmup, args.iters, args.rounds
            )
            print(
                f"{args.batch},{args.heads},{n},"
                f"{'causal' if causal else 'full'},{(n + 63) // 64},"
                f"{t7:.6f},{t8:.6f},{t8 / t7:.3f}x,{t7 / t8:.3f}x,"
                f"{max_diff:.1e}"
            )


if __name__ == "__main__":
    main()
