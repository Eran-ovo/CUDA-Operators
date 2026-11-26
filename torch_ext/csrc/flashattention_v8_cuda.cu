// FlashAttention v8：把 GEMM v8/v10 的“布局与加载共同设计”迁移到 D=64。
//
// 与 v7 相比，本文件刻意保持以下内容不变：
//   1. 一个 CTA 处理 64 行 Q，4 个 warp 各处理 16 行；
//   2. Q fragment 跨所有 KV tile 缓存在寄存器中；
//   3. QK^T、online softmax、P@V 的计算顺序；
//   4. P fragment 从寄存器直接送入第二个 MMA，不写回 shared memory。
//
// 本版本只替换 Q/K/V 的数据通路：
//   global memory --cp.async(16B)--> padded shared memory --ldmatrix--> registers
//
// 为什么不能在 v7 的 [64][65] shared layout 上直接使用 cp.async？
//   65 个 half 的行跨度是 130B，除第 0 行外，大多数行首不是 16B 对齐；
//   cp.async 的 16B 形式要求源地址和目标地址满足相应对齐。因此这里采用
//   D+8=72 half 的物理行宽：144B 既是 16B 的整数倍，也让 ldmatrix 读取
//   相邻 8 行时每行错开 4 个 shared-memory bank。
//
// 第一版没有做 K/V 双缓冲。这样可以先回答一个清晰问题：仅减少 global->
// shared 和 shared->register 指令，是否已经有效。若直接叠加双缓冲，性能变化
// 将同时包含寄存器、shared memory、occupancy 和流水重叠，难以准确归因。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "ops.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#ifndef FLASHATTENTION_V8_KERNEL
#define FLASHATTENTION_V8_KERNEL flash_fp16_mma_v8_kernel
#endif

#ifndef FLASHATTENTION_V8_FORWARD
#define FLASHATTENTION_V8_FORWARD flashattention_v8_forward
#endif

namespace flashattention_v8 {

constexpr int D = 64;
constexpr int BQ = 64;
constexpr int BC = 64;
constexpr int MMA_K = 16;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;

// 每行有效数据仍为 64 half，尾部 8 half 只用于改变行跨度。
// 72 half = 144B，因此每一行都保持 16B 对齐。
constexpr int SMEM_STRIDE = D + 8;
constexpr int COPY_BYTES = 16;
constexpr int COPY_HALFS = COPY_BYTES / sizeof(__half);  // 8 half
constexpr int SEGMENTS_PER_ROW = D / COPY_HALFS;         // 8 segments
constexpr int SEGMENTS_PER_TILE = BC * SEGMENTS_PER_ROW; // 512 segments

static_assert(D % MMA_K == 0, "D must be a multiple of mma K");
static_assert(BC % MMA_K == 0, "BC must be a multiple of mma K");
static_assert(SMEM_STRIDE * static_cast<int>(sizeof(__half)) % COPY_BYTES == 0,
              "every shared-memory row must be 16-byte aligned");

__device__ __forceinline__ unsigned shared_u32(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// src_size=0 是 cp.async 的硬件 zero-fill 路径：不读取 global memory，
// 并将目标 16B 全部写成 0。尾 KV tile 因而不会发生越界读取。
// 即使 valid=false，也把 src 指向一个合法地址；这避免在 C++ 层构造远超
// allocation 末尾的指针，真正是否读取仍由 src_size 决定。
__device__ __forceinline__ void cp_async_16_zfill(
    void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned dst_shared = shared_u32(dst);
    const int src_size = valid ? COPY_BYTES : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(dst_shared), "l"(src), "r"(src_size) : "memory");
#else
    // 当前性能目标是 sm_86；保留同步 fallback 使源码仍能为旧架构编译。
    if (valid) {
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
    } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" :: : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
#endif
}

// 一个 CTA 合作搬运一个 64x64 tile。每个线程负责 4 个连续的 16B segment。
// shared memory 的 padding 不参与 segment 编号，因此不会误写 padding 区域。
__device__ __forceinline__ void copy_tile_async(
    const __half* src,
    __half (*dst)[SMEM_STRIDE],
    int row_offset,
    int valid_rows,
    int tid,
    size_t row_stride)
{
    #pragma unroll
    for (int segment = tid; segment < SEGMENTS_PER_TILE;
         segment += THREADS) {
        const int row = segment / SEGMENTS_PER_ROW;
        const int col = (segment % SEGMENTS_PER_ROW) * COPY_HALFS;
        const int global_row = row_offset + row;
        const bool valid = global_row < valid_rows;

        // 对无效行传入 src 本身作为合法占位地址；src_size=0 保证不读取它。
        const __half* src_segment =
            valid ? src + static_cast<size_t>(global_row) * row_stride + col : src;
        cp_async_16_zfill(&dst[row][col], src_segment, valid);
    }
}

// ldmatrix.x4 装载四个 8x8 fp16 子矩阵，组合成 mma 的 16x16 A fragment。
__device__ __forceinline__ void ldmatrix_x4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
    const void* row_addr)
{
    const unsigned addr = shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    unsigned& r0, unsigned& r1, const void* row_addr)
{
    const unsigned addr = shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    unsigned& r0, unsigned& r1, const void* row_addr)
{
    const unsigned addr = shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_regs(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack_half2(float x, float y) {
    const __half2 packed = __floats2half2_rn(x, y);
    return *reinterpret_cast<const unsigned*>(&packed);
}

// Q 是 row-major 的 16x16 A tile。四组 8 个 lane 分别给 ldmatrix
// 提供四个 8x8 子矩阵的行首地址；返回的 4 个寄存器可直接送入 MMA。
__device__ __forceinline__ void load_q_fragment(
    unsigned (&frag)[4],
    const __half (*tile)[SMEM_STRIDE],
    int warp_row,
    int d0,
    int lane)
{
    const int matrix = lane / 8;
    const int row = warp_row + (matrix & 1) * 8 + lane % 8;
    const int col = d0 + (matrix / 2) * 8;
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], &tile[row][col]);
}

// QK^T 的 B 操作数在数学上是 [D, BC]，而 shared 中保存的是原始
// K[BC, D]，也就是已经转置后的物理形状。两个 8x8 子矩阵分别覆盖
// d0+[0,8) 和 d0+[8,16)，所以这里使用非 trans 的 ldmatrix.x2。
__device__ __forceinline__ void load_k_as_b_fragment(
    unsigned& b0,
    unsigned& b1,
    const __half (*tile)[SMEM_STRIDE],
    int key_col,
    int d0,
    int lane)
{
    const int addr_lane = lane & 15;
    const int matrix = addr_lane / 8;
    const int row = key_col + addr_lane % 8;
    const int col = d0 + matrix * 8;
    ldmatrix_x2(b0, b1, &tile[row][col]);
}

// PV 的 B 操作数是 V[BC, D]。它与 GEMM 中 row-major B tile 的情形一致：
// 16 个 lane 给出 16 个 K 行的地址，.trans 将其重排为 row.col MMA
// 所需的 B fragment。
__device__ __forceinline__ void load_v_as_b_fragment(
    unsigned& b0,
    unsigned& b1,
    const __half (*tile)[SMEM_STRIDE],
    int key_row,
    int output_col,
    int lane)
{
    const int addr_lane = lane & 15;
    ldmatrix_x2_trans(
        b0, b1, &tile[key_row + addr_lane][output_col]);
}

template <bool IS_CAUSAL, bool PACKED = false>
__global__ void FLASHATTENTION_V8_KERNEL(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N,
    int query_heads,
    int kv_heads,
    const int32_t* __restrict__ tiles)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int group_row = lane / 4;
    const int group_col = lane % 4;

    const int warp_row = warp_id * MMA_M;

    int q_offset = blockIdx.x * BQ;
    size_t q_stride = D;
    size_t kv_stride = D;
    if constexpr (PACKED) {
        // One x-block per real query tile and one y-block per query head.
        // Q/O and K/V are token-major, with no expanded KV heads.
        const int32_t* tile = tiles + static_cast<size_t>(blockIdx.x) * 3;
        const size_t start = tile[0];
        N = tile[1];
        q_offset = tile[2];
        q_stride = static_cast<size_t>(query_heads) * D;
        kv_stride = static_cast<size_t>(kv_heads) * D;
        const size_t q_base = start * q_stride + blockIdx.y * D;
        const size_t kv_base = start * kv_stride + (blockIdx.y / (query_heads / kv_heads)) * D;
        Q += q_base;
        K += kv_base;
        V += kv_base;
        O += q_base;
    } else {
        const size_t base = static_cast<size_t>(blockIdx.y) * N * D;
        Q += base;
        K += base;
        V += base;
        O += base;
    }

    // Q 缓存为寄存器 fragment 后，QK_tile 即可被每一轮 K tile 覆盖。
#ifdef FLASHATTENTION_V8_PARTIAL_PIPELINE
    // v9：K 在 QK 阶段结束后即可覆盖，V 需要等当前 PV 阶段结束，因此只
    // 为 V 保留两个 stage。总 shared memory 为 K(9.2KB)+V(18.4KB)。
    __shared__ __align__(16) __half QK_tile[BC][SMEM_STRIDE];
    __shared__ __align__(16) __half V_tile[2][BC][SMEM_STRIDE];
#else
    // v8：单 stage，先完成 K/V 搬运，再执行当前 tile 的全部计算。
    // 两块 shared tile 合计 2*64*72*2 = 18 KiB。
    __shared__ __align__(16) __half QK_tile[BQ][SMEM_STRIDE];
    __shared__ __align__(16) __half V_tile[BC][SMEM_STRIDE];
#endif

    // Q 只搬一次。尾 Q block 中不存在的行由 cp.async zero-fill。
    copy_tile_async(Q, QK_tile, q_offset, N, tid, q_stride);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    unsigned q_frag[D / MMA_K][4];
    #pragma unroll
    for (int d0 = 0; d0 < D; d0 += MMA_K) {
        load_q_fragment(q_frag[d0 / MMA_K], QK_tile, warp_row, d0, lane);
    }

    // 所有 warp 都必须先完成 Q 的 ldmatrix，才能让 K 覆盖 QK_tile。
    __syncthreads();

    float running_max[2] = {-1e20f, -1e20f};
    float running_sum[2] = {0.0f, 0.0f};
    float output_acc[2][8][4];
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                output_acc[row][tile][i] = 0.0f;
            }
        }
    }

    constexpr float SCALE = 0.125f;  // 1/sqrt(64)
    const int q_last = min(q_offset + BQ - 1, N - 1);
    const int num_kv_tiles =
        IS_CAUSAL ? (q_last / BC + 1) : ((N + BC - 1) / BC);

#ifdef FLASHATTENTION_V8_PARTIAL_PIPELINE
    int v_read_stage = 0;
#endif
    for (int kv = 0; kv < num_kv_tiles; ++kv) {
        const int key_offset = kv * BC;

        // K 和 V 属于同一个 async group；wait_group 0 返回后，两块 tile
        // 都已写入 shared memory。随后的 block barrier 保证所有线程可见。
#ifdef FLASHATTENTION_V8_PARTIAL_PIPELINE
        if (kv == 0) {
            // 第一个 tile 在 Q fragment 初始化之后才首次装载 K/V。
            copy_tile_async(K, QK_tile, key_offset, N, tid, kv_stride);
            copy_tile_async(V, V_tile[v_read_stage], key_offset, N, tid, kv_stride);
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
        }
#else
        copy_tile_async(K, QK_tile, key_offset, N, tid, kv_stride);
        copy_tile_async(V, V_tile, key_offset, N, tid, kv_stride);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
#endif

        float scores[8][4];
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                scores[tile][i] = 0.0f;
            }
        }

        // GEMM 1：Q[16,64] @ K^T[64,64]。
        #pragma unroll
        for (int d0 = 0; d0 < D; d0 += MMA_K) {
            const unsigned* a = q_frag[d0 / MMA_K];
            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                unsigned b0, b1;
                load_k_as_b_fragment(
                    b0, b1, QK_tile, tile * MMA_N, d0, lane);
                mma_m16n8k16_regs(
                    scores[tile][0], scores[tile][1],
                    scores[tile][2], scores[tile][3],
                    a[0], a[1], a[2], a[3], b0, b1);
            }
        }

#ifdef FLASHATTENTION_V8_PARTIAL_PIPELINE
        // QK 的最后一个 ldmatrix 已经结束，但其他 warp 可能仍在读取 K。
        // 只有 block barrier 之后，才能让所有线程异步覆盖下一 tile 的 K。
        __syncthreads();
        if (kv + 1 < num_kv_tiles) {
            const int next_key_offset = (kv + 1) * BC;
            const int next_v_stage = v_read_stage ^ 1;
            // 这两组 copy 会与下面的 mask、softmax 和 PV 计算重叠：K 写入
            // 当前 QK_tile（QK 已经完成），V 写入未被当前 PV 使用的 stage。
            copy_tile_async(K, QK_tile, next_key_offset, N, tid, kv_stride);
            copy_tile_async(
                V, V_tile[next_v_stage], next_key_offset, N, tid, kv_stride);
            cp_async_commit();
        }
#endif

        // tail 与 causal mask 必须在 max/sum 之前应用。zero-fill 只保证
        // 越界 K 的点积为 0；若不改成 -inf，它仍会错误进入 softmax 分母。
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const int key0 = key_offset + tile * MMA_N + group_col * 2;
            const int key1 = key0 + 1;
            const int query0 = q_offset + warp_row + group_row;
            const int query1 = query0 + 8;

            if (key0 >= N || (IS_CAUSAL && key0 > query0))
                scores[tile][0] = -INFINITY;
            if (key1 >= N || (IS_CAUSAL && key1 > query0))
                scores[tile][1] = -INFINITY;
            if (key0 >= N || (IS_CAUSAL && key0 > query1))
                scores[tile][2] = -INFINITY;
            if (key1 >= N || (IS_CAUSAL && key1 > query1))
                scores[tile][3] = -INFINITY;
        }

        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                scores[tile][i] *= SCALE;
            }
        }

        const float old_max0 = running_max[0];
        const float old_max1 = running_max[1];
        const float old_sum0 = running_sum[0];
        const float old_sum1 = running_sum[1];

        float tile_max0 = -1e20f;
        float tile_max1 = -1e20f;
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            tile_max0 = fmaxf(
                tile_max0, fmaxf(scores[tile][0], scores[tile][1]));
            tile_max1 = fmaxf(
                tile_max1, fmaxf(scores[tile][2], scores[tile][3]));
        }
        tile_max0 = fmaxf(
            tile_max0, __shfl_xor_sync(0xffffffff, tile_max0, 1));
        tile_max0 = fmaxf(
            tile_max0, __shfl_xor_sync(0xffffffff, tile_max0, 2));
        tile_max1 = fmaxf(
            tile_max1, __shfl_xor_sync(0xffffffff, tile_max1, 1));
        tile_max1 = fmaxf(
            tile_max1, __shfl_xor_sync(0xffffffff, tile_max1, 2));

        const float new_max0 = fmaxf(old_max0, tile_max0);
        const float new_max1 = fmaxf(old_max1, tile_max1);
        const float alpha0 = expf(old_max0 - new_max0);
        const float alpha1 = expf(old_max1 - new_max1);

        float tile_sum0 = 0.0f;
        float tile_sum1 = 0.0f;
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            scores[tile][0] = expf(scores[tile][0] - new_max0);
            scores[tile][1] = expf(scores[tile][1] - new_max0);
            scores[tile][2] = expf(scores[tile][2] - new_max1);
            scores[tile][3] = expf(scores[tile][3] - new_max1);
            tile_sum0 += scores[tile][0] + scores[tile][1];
            tile_sum1 += scores[tile][2] + scores[tile][3];
        }
        tile_sum0 += __shfl_xor_sync(0xffffffff, tile_sum0, 1);
        tile_sum0 += __shfl_xor_sync(0xffffffff, tile_sum0, 2);
        tile_sum1 += __shfl_xor_sync(0xffffffff, tile_sum1, 1);
        tile_sum1 += __shfl_xor_sync(0xffffffff, tile_sum1, 2);

        const float new_sum0 = old_sum0 * alpha0 + tile_sum0;
        const float new_sum1 = old_sum1 * alpha1 + tile_sum1;

        // online softmax 的 running max 改变后，旧输出必须乘 alpha。
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                output_acc[0][tile][i] *= alpha0;
                output_acc[1][tile][i] *= alpha1;
            }
        }

        // GEMM 2：P[16,64] @ V[64,64]。
        // P 仍按 v5/v7 的方式直接从 fp32 score fragment 转成 fp16 A fragment；
        // V fragment 则改由 ldmatrix.x2.trans 从 shared memory 装载。
        #pragma unroll
        for (int key_row = 0; key_row < BC; key_row += MMA_K) {
            const int score_tile = key_row / MMA_N;
            const unsigned pa0 = pack_half2(
                scores[score_tile][0], scores[score_tile][1]);
            const unsigned pa1 = pack_half2(
                scores[score_tile][2], scores[score_tile][3]);
            const unsigned pa2 = pack_half2(
                scores[score_tile + 1][0], scores[score_tile + 1][1]);
            const unsigned pa3 = pack_half2(
                scores[score_tile + 1][2], scores[score_tile + 1][3]);

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                unsigned vb0, vb1;
#ifdef FLASHATTENTION_V8_PARTIAL_PIPELINE
                load_v_as_b_fragment(
                    vb0, vb1, V_tile[v_read_stage], key_row,
                    tile * MMA_N, lane);
#else
                load_v_as_b_fragment(
                    vb0, vb1, V_tile, key_row, tile * MMA_N, lane);
#endif
                mma_m16n8k16_regs(
                    output_acc[0][tile][0], output_acc[0][tile][1],
                    output_acc[1][tile][0], output_acc[1][tile][1],
                    pa0, pa1, pa2, pa3, vb0, vb1);
            }
        }

        running_max[0] = new_max0;
        running_max[1] = new_max1;
        running_sum[0] = new_sum0;
        running_sum[1] = new_sum1;

        // 下一轮会覆盖两块 shared tile；必须等所有 warp 完成当前 ldmatrix。
#ifdef FLASHATTENTION_V8_PARTIAL_PIPELINE
        if (kv + 1 < num_kv_tiles) {
            // 等待刚才与 softmax/PV 重叠的 K/V copy。barrier 将 async
            // 写入转为整个 block 对下一轮 ldmatrix 可见的 shared 数据。
            cp_async_wait_all();
            __syncthreads();
            v_read_stage ^= 1;
        }
#else
        __syncthreads();
#endif
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int output_col = tile * MMA_N;
        const int output_row0 = q_offset + warp_row + group_row;
        const int output_row1 = output_row0 + 8;
        const float inv_sum0 = 1.0f / running_sum[0];
        const float inv_sum1 = 1.0f / running_sum[1];

        if (output_row0 < N) {
            O[output_row0 * q_stride + output_col + group_col * 2] =
                __float2half(output_acc[0][tile][0] * inv_sum0);
            O[output_row0 * q_stride + output_col + group_col * 2 + 1] =
                __float2half(output_acc[0][tile][1] * inv_sum0);
        }
        if (output_row1 < N) {
            O[output_row1 * q_stride + output_col + group_col * 2] =
                __float2half(output_acc[1][tile][0] * inv_sum1);
            O[output_row1 * q_stride + output_col + group_col * 2 + 1] =
                __float2half(output_acc[1][tile][1] * inv_sum1);
        }
    }
}

}  // namespace flashattention_v8

torch::Tensor FLASHATTENTION_V8_FORWARD(torch::Tensor q,
                                        torch::Tensor k,
                                        torch::Tensor v,
                                        bool causal) {
    using namespace flashattention_v8;

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf &&
                    k.dtype() == torch::kHalf &&
                    v.dtype() == torch::kHalf,
                "q/k/v must be float16");
    TORCH_CHECK(q.dim() == 2 || q.dim() == 4,
                "q/k/v must be [N, 64] or [B, H, N, 64]");
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim(),
                "q/k/v must have the same rank");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q/k/v must have the same shape");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q/k/v must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(q.device());

    const bool batched = q.dim() == 4;
    const int64_t n64 = batched ? q.size(2) : q.size(0);
    TORCH_CHECK(n64 > 0 && n64 <= INT_MAX, "N must be in [1, INT_MAX]");
    TORCH_CHECK(q.size(-1) == D, "head dim must be 64");

    int64_t batch_heads = 1;
    if (batched) {
        const int64_t batch = q.size(0);
        const int64_t heads = q.size(1);
        TORCH_CHECK(batch > 0 && heads > 0, "B and H must be positive");
        TORCH_CHECK(heads <= 65535 && batch <= 65535 / heads,
                    "B*H must not exceed CUDA grid.y limit 65535");
        batch_heads = batch * heads;
    }

    // contiguous 只描述 stride，不保证带 storage_offset 的 tensor 起点仍为
    // 16B 对齐。v8 的 cp.async fast path 对三个真实数据指针逐一检查；不满足
    // 时回退到已验证的 v7，而不是冒险执行未对齐的异步拷贝。
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(q.data_ptr<at::Half>()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(k.data_ptr<at::Half>()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(v.data_ptr<at::Half>()) % 16 == 0;
    if (!aligned) {
        // 沿用引入 v8 之前 production dispatch 的保守边界：短序列使用 v5，
        // Q fragment 有足够复用机会的长序列使用 v7。这样 v8 的 fast path
        // 不会改变少见的 misaligned-storage 输入原本选择的稳定实现。
        constexpr int64_t V7_MIN_N = 1024;
        if (n64 >= V7_MIN_N) {
            return flashattention_v7_forward(q, k, v, causal);
        }
        return flashattention_v5_forward(q, k, v, causal);
    }

    const int n = static_cast<int>(n64);
    auto output = torch::empty_like(q);
    const dim3 grid((n + BQ - 1) / BQ,
                    static_cast<unsigned>(batch_heads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (causal) {
        FLASHATTENTION_V8_KERNEL<true><<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), n, 0, 0, nullptr);
    } else {
        FLASHATTENTION_V8_KERNEL<false><<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), n, 0, 0, nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

#ifndef FLASHATTENTION_V8_PARTIAL_PIPELINE
FlashAttentionVarlenPlan::FlashAttentionVarlenPlan(
    const std::vector<int64_t>& cu_seqlens, c10::Device device)
{
    using flashattention_v8::BQ;
    TORCH_CHECK(device.is_cuda(), "varlen plan requires a CUDA device");
    TORCH_CHECK(cu_seqlens.size() >= 2 && cu_seqlens.front() == 0,
                "cu_seqlens must start at zero and contain at least one sequence");
    TORCH_CHECK(cu_seqlens.back() > 0 && cu_seqlens.back() <= INT_MAX,
                "the packed token count must be in [1, INT_MAX]");
    total_tokens = cu_seqlens.back();

    std::vector<int32_t> entries;
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
        const int64_t start = cu_seqlens[i - 1];
        const int64_t end = cu_seqlens[i];
        TORCH_CHECK(end > start && end <= total_tokens,
                    "cu_seqlens must be strictly increasing");
        const int64_t length = end - start;
        for (int64_t offset = 0; offset < length; offset += BQ) {
            entries.insert(entries.end(), {
                static_cast<int32_t>(start), static_cast<int32_t>(length),
                static_cast<int32_t>(offset),
            });
        }
    }
    tiles = torch::tensor(entries, torch::TensorOptions().dtype(torch::kInt32))
                .view({-1, 3}).to(device);
}

torch::Tensor flashattention_varlen_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, const FlashAttentionVarlenPlan& plan)
{
    using namespace flashattention_v8;
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes(),
                "expected q [T,Hq,64] and k/v [T,Hkv,64]");
    TORCH_CHECK(q.size(0) == plan.total_tokens && k.size(0) == plan.total_tokens &&
                q.size(2) == D && k.size(2) == D, "packed token count or head dimension mismatch");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) > 0 && q.size(1) <= 65535 &&
                q.size(1) % k.size(1) == 0, "Hq must be a positive multiple of Hkv");
    for (const auto& tensor : {q, k, v}) {
        TORCH_CHECK(tensor.device() == plan.tiles.device() && tensor.scalar_type() == torch::kHalf &&
                    tensor.is_contiguous(), "packed inputs must be contiguous FP16 on the plan device");
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % COPY_BYTES == 0,
                    "packed inputs must be 16-byte aligned");
    }
    c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty_like(q);
    const dim3 grid(static_cast<unsigned>(plan.tiles.size(0)), static_cast<unsigned>(q.size(1)));
    auto stream = at::cuda::getCurrentCUDAStream();
    // The plan can be released while its kernel is queued on a different stream.
    c10::cuda::CUDACachingAllocator::recordStream(plan.tiles.storage().data_ptr(), stream);
    FLASHATTENTION_V8_KERNEL<true, true><<<grid, THREADS, 0, stream.stream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        0, static_cast<int>(q.size(1)), static_cast<int>(k.size(1)),
        plan.tiles.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
#endif

#undef FLASHATTENTION_V8_KERNEL
#undef FLASHATTENTION_V8_FORWARD
