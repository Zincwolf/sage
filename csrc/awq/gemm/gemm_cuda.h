// Source: https://github.com/mit-han-lab/llm-awq (d6e797a); MIT.
#include <torch/extension.h>

torch::Tensor gemm_forward_cuda_new(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scales, torch::Tensor _zeros);
