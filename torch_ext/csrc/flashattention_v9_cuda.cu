// FlashAttention v9：在 v8 的 16B cp.async + ldmatrix 路径上加入部分双缓冲。
//
// v9 通过 include v8 的 kernel 实现来保持 fragment、mask 和 online softmax
// 完全一致，只打开 FLASHATTENTION_V8_PARTIAL_PIPELINE 这个编译期变体：
//   - K 使用一个 shared tile；QK 阶段结束后即可覆盖它；
//   - V 使用两个 shared tile，当前 PV 读取 stage 0 时，下一 V 写入 stage 1；
//   - 下一轮 K/V 的 cp.async 与当前 tile 的 PV MMA 重叠。
//
// 只做部分双缓冲是有意的：完整复制 K 和 V 会把 shared memory 从18.4KB
// 增加到36.9KB，在 sm_86 上很可能把 block residency 压到1。K 在当前 QK
// 完成后已经不再被读取，复用 K buffer 可以用更低的 occupancy 成本获得
// 异步搬运机会。

#define FLASHATTENTION_V8_PARTIAL_PIPELINE
#define FLASHATTENTION_V8_KERNEL flash_fp16_mma_v9_kernel
#define FLASHATTENTION_V8_FORWARD flashattention_v9_forward

// v8 的实现文件包含完整的 host wrapper；宏重命名后，这个 translation unit
// 只额外导出 v9 符号，不会覆盖或改变 v8 的独立入口。
#include "flashattention_v8_cuda.cu"

#undef FLASHATTENTION_V8_FORWARD
#undef FLASHATTENTION_V8_KERNEL
#undef FLASHATTENTION_V8_PARTIAL_PIPELINE
