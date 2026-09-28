#include "sph_gemv_cuda.h"
#include "sage_gemv_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor sage_gemv_forward_cuda(
    torch::Tensor input, torch::Tensor codes, torch::Tensor radii,
    torch::Tensor grid, torch::Tensor book, torch::Tensor permutation,
    int64_t out_features) {
  // custom_op supplies dispatch/fake semantics, but does not validate raw pointer layouts.
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
              input.is_contiguous() && input.dim() >= 1,
              "SAGE requires contiguous CUDA FP16 input");
  const c10::cuda::CUDAGuard guard(input.device());
  for (const auto &tensor : {codes, radii, grid, book, permutation})
    TORCH_CHECK(tensor.device() == input.device() && tensor.is_contiguous(),
                "SAGE buffers must be contiguous and on the input device");
  const int64_t k = input.size(-1);
  const int64_t tiles = (out_features + 15) / 16;
  TORCH_CHECK(k > 0 && k % 128 == 0 && out_features > 0,
              "SAGE requires positive N and K divisible by 128");
  TORCH_CHECK(book.dim() == 3 && book.size(0) == 4 && book.size(2) == 2 &&
              (book.size(1) == 64 || book.size(1) == 128 || book.size(1) == 256),
              "SAGE codebook must have shape [4, 64/128/256, 2]");
  const int entries = book.size(1);
  TORCH_CHECK(codes.scalar_type() == at::kInt &&
              codes.sizes() == at::IntArrayRef({tiles, k / 128, 16, 8}) &&
              radii.scalar_type() == at::kInt &&
              radii.sizes() == at::IntArrayRef({tiles, k / 128, 16}) &&
              grid.scalar_type() == at::kHalf &&
              grid.sizes() == at::IntArrayRef({tiles, k / 128, 16, 2}) &&
              book.scalar_type() == at::kHalf &&
              permutation.scalar_type() == at::kInt && permutation.dim() == 1 &&
              (permutation.numel() == 0 || permutation.numel() == k),
              "Invalid SAGE packed layout");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(grid.data_ptr()) & 3) == 0 &&
              (reinterpret_cast<uintptr_t>(book.data_ptr()) & 3) == 0,
              "SAGE grid and codebook require 4-byte aligned storage");
  auto shape = input.sizes().vec();
  shape.back() = out_features;
  auto output = torch::empty(shape, input.options());
  const int64_t rows = input.numel() / k;
  if (rows == 0)
    return output;
  TORCH_CHECK(rows <= 65535 && k <= INT_MAX && out_features <= INT_MAX,
              "SAGE launch dimensions exceed supported range");
  const bool gather = rows < 16 && sage_gemv_fuses_permutation(out_features, k);
  if (permutation.numel() && !gather)
    input = at::index_select(input, -1, permutation);
  if (rows < 16) {
    sage_gemv_cuda(
        reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t *>(codes.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(radii.data_ptr<int32_t>()),
        reinterpret_cast<const half2 *>(grid.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t *>(book.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        gather && permutation.numel() ? permutation.data_ptr<int32_t>() : nullptr,
        rows, out_features, k, entries, at::cuda::getCurrentCUDAStream());
  } else {
    const int parts = sage_gemm_splits(
        rows, out_features, k, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    auto workspace = torch::empty(
        {parts > 1 ? parts * rows * out_features : 0}, input.options().dtype(at::kFloat));
    sage_gemm_cuda(
        reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t *>(codes.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t *>(radii.data_ptr<int32_t>()),
        reinterpret_cast<const half2 *>(grid.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t *>(book.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        workspace.data_ptr<float>(), parts,
        rows, out_features, k, entries, at::cuda::getCurrentCUDAStream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

extern int sage_gemv_tuning;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_sage_tuning", [](int value) { sage_gemv_tuning = value; });
  m.def("sph_gemv_forward_cuda", &sph_gemv_forward_cuda);
  m.def("sage_gemv_forward_cuda", &sage_gemv_forward_cuda);
}
