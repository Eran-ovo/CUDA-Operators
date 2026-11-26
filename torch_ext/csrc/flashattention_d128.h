#pragma once
#include <torch/extension.h>

torch::Tensor flashattention_d128_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
