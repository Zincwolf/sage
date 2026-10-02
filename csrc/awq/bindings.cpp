// Wraps https://github.com/mit-han-lab/llm-awq quantization_new (MIT; per-file notices retained).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "gemm/gemm_cuda.h"
#include "gemv/gemv_cuda.h"

// Guard the frozen kernels' unchecked tile loads; no padding or dense fallback.
torch::Tensor awq_forward(torch::Tensor x, torch::Tensor weight,
                          torch::Tensor scales, torch::Tensor zeros) {
  TORCH_CHECK(x.is_cuda() && x.dim() >= 1 && x.is_contiguous(),
              "AWQ requires contiguous CUDA input");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "AWQ requires FP16 or BF16 activations");
  const auto k = x.size(-1);
  TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == at::kShort,
              "AWQ qweight must be a 2-D int16 tensor");
  const auto n = weight.size(0) * 4;
  TORCH_CHECK(k > 0 && k % 256 == 0 && n > 0 && n % 128 == 0 && weight.size(1) == k,
              "AWQ requires K divisible by 256 and N divisible by 128");
  const auto groups = ((k / 128 + 7) / 8) * 8;
  for (const auto& tensor : {weight, scales, zeros}) {
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                "AWQ buffers must be contiguous and on the input device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "AWQ buffers require 16-byte alignment");
  }
  for (const auto& tensor : {scales, zeros}) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == groups && tensor.size(1) == n &&
                tensor.scalar_type() == x.scalar_type(), "Invalid AWQ scale/zero layout or dtype");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "AWQ input requires 16-byte alignment");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto m = x.numel() / k;
  if (m == 0) {
    auto shape = x.sizes().vec();
    shape.back() = n;
    return torch::empty(shape, x.options());
  }
  auto output = m < 8 ? gemv_forward_cuda_new(x, weight, scales, zeros, m, n, k, 128)
                      : gemm_forward_cuda_new(x, weight, scales, zeros);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &awq_forward);
}
