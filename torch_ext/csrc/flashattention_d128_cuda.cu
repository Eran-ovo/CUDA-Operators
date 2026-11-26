// D128 uses 32 query rows and 32 KV rows to bound shared memory and score fragments.
// Reuse the v8 data movement and online softmax with separate compile-time dimensions.
#include "flashattention_d128.h"

#define FLASHATTENTION_V8_D128
#define FLASHATTENTION_V8_NAMESPACE flashattention_d128
#define FLASHATTENTION_V8_KERNEL flash_fp16_mma_d128_kernel
#define FLASHATTENTION_V8_FORWARD flashattention_d128_forward
#include "flashattention_v8_cuda.cu"
