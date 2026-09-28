#include "sph_gemv_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxGroups = 128;
constexpr int kCodebookSize = 64;
constexpr int kVectorSize = 8;

template <int OutputTile>
__global__ void sph_gemv_kernel(const half *__restrict__ input,
                                const uint16_t *__restrict__ packed_codes,
                                const uint32_t *__restrict__ radius_extra,
                                const half *__restrict__ radius_scales,
                                const half *__restrict__ radius_mins,
                                const half *__restrict__ codebook,
                                half *__restrict__ output, int rows,
                                int out_features, int in_features, int groups) {
  // Store four half2 coordinates as [coordinate][direction].  A warp then
  // indexes 32 banks by direction, instead of only eight banks with the
  // natural [direction][coordinate] layout.
  __shared__ uint32_t shared_codebook[kVectorSize / 2][kCodebookSize];
  __shared__ half shared_scales[OutputTile][kMaxGroups];
  __shared__ half shared_mins[OutputTile][kMaxGroups];
  __shared__ uint32_t shared_extra[OutputTile][kMaxGroups];
  __shared__ float warp_sums[OutputTile][kBlockSize / 32];

  const int output_base = blockIdx.x * OutputTile;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (tid < kCodebookSize * (kVectorSize / 2)) {
    const int pair = tid / kCodebookSize;
    const int direction = tid - pair * kCodebookSize;
    shared_codebook[pair][direction] =
        reinterpret_cast<const uint32_t *>(codebook)[direction * 4 + pair];
  }

  for (int linear = tid; linear < OutputTile * groups;
       linear += kBlockSize) {
    const int tile = linear / groups;
    const int group = linear - tile * groups;
    const int out = output_base + tile;
    if (out < out_features) {
      const int64_t index = static_cast<int64_t>(out) * groups + group;
      shared_scales[tile][group] = radius_scales[index];
      shared_mins[tile][group] = radius_mins[index];
      shared_extra[tile][group] = radius_extra[index];
    } else {
      shared_scales[tile][group] = __float2half(0.0f);
      shared_mins[tile][group] = __float2half(0.0f);
      shared_extra[tile][group] = 0;
    }
  }
  __syncthreads();

  const int vectors_per_row = in_features / kVectorSize;
  const half *row_input = input + static_cast<int64_t>(row) * in_features;
  float accum[OutputTile] = {0.0f};

  for (int vector = tid; vector < vectors_per_row; vector += kBlockSize) {
    half2 values[kVectorSize / 2];
    const half2 *x =
        reinterpret_cast<const half2 *>(row_input + vector * kVectorSize);
#pragma unroll
    for (int pair = 0; pair < kVectorSize / 2; ++pair)
      values[pair] = x[pair];

    const int group = vector >> 4; // 16 vectors = 128 scalar weights.
    const int vector_in_group = vector & 15;
#pragma unroll
    for (int tile = 0; tile < OutputTile; ++tile) {
      const int out = output_base + tile;
      if (out >= out_features)
        continue;
      const uint16_t code =
          packed_codes[static_cast<int64_t>(out) * vectors_per_row + vector];
      const uint32_t sign_bits = code & 0xffu;
      const uint32_t direction_index = (code >> 8) & 0x3fu;
      const uint32_t radius_low = (code >> 14) & 0x3u;
      const uint32_t radius_high =
          (shared_extra[tile][group] >> (2 * vector_in_group)) & 0x3u;
      const uint32_t radius_code = radius_low | (radius_high << 2);

      half2 direction_sum = __float2half2_rn(0.0f);
#pragma unroll
      for (int pair = 0; pair < kVectorSize / 2; ++pair) {
        uint32_t direction_bits = shared_codebook[pair][direction_index];
        const uint32_t sign_mask =
            (((sign_bits >> (pair * 2)) & 1u) << 15) |
            (((sign_bits >> (pair * 2 + 1)) & 1u) << 31);
        direction_bits ^= sign_mask;
        const half2 direction =
            *reinterpret_cast<const half2 *>(&direction_bits);
        direction_sum =
            __hadd2(direction_sum, __hmul2(direction, values[pair]));
      }
      const float2 direction_parts = __half22float2(direction_sum);
      const float direction_dot = direction_parts.x + direction_parts.y;
      const half radius_half =
          __hfma(__uint2half_rn(radius_code), shared_scales[tile][group],
                 shared_mins[tile][group]);
      const float radius = __half2float(radius_half);
      accum[tile] = fmaf(radius, direction_dot, accum[tile]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int tile = 0; tile < OutputTile; ++tile)
      accum[tile] += __shfl_down_sync(0xffffffff, accum[tile], offset);
  }
  if (lane == 0) {
#pragma unroll
    for (int tile = 0; tile < OutputTile; ++tile)
      warp_sums[tile][warp] = accum[tile];
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int tile = 0; tile < OutputTile; ++tile) {
      float value = lane < (kBlockSize / 32) ? warp_sums[tile][lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (lane == 0 && output_base + tile < out_features)
        output[static_cast<int64_t>(row) * out_features + output_base + tile] =
            __float2half(value);
    }
  }
}

template <int OutputsPerWarp>
__global__ void sph_gemv_warp_kernel(
    const half *__restrict__ input,
    const uint16_t *__restrict__ packed_codes,
    const uint32_t *__restrict__ radius_extra,
    const half *__restrict__ radius_scales,
    const half *__restrict__ radius_mins,
    const half *__restrict__ codebook,
    half *__restrict__ output, int rows, int out_features, int in_features,
    int groups) {
  __shared__ uint32_t shared_codebook[kVectorSize / 2][kCodebookSize];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int warps = kBlockSize / 32;
  constexpr int outputs_per_block = warps * OutputsPerWarp;
  const int output_base = blockIdx.x * outputs_per_block +
                          warp * OutputsPerWarp;
  const int row = blockIdx.y;

  if (tid < kCodebookSize * (kVectorSize / 2)) {
    const int pair = tid / kCodebookSize;
    const int direction = tid - pair * kCodebookSize;
    shared_codebook[pair][direction] =
        reinterpret_cast<const uint32_t *>(codebook)[direction * 4 + pair];
  }
  __syncthreads();

  const int vectors_per_row = in_features / kVectorSize;
  const half *row_input = input + static_cast<int64_t>(row) * in_features;
  float accum[OutputsPerWarp] = {0.0f};

  for (int vector = lane; vector < vectors_per_row; vector += 32) {
    half2 values[kVectorSize / 2];
    const half2 *x = reinterpret_cast<const half2 *>(
        row_input + vector * kVectorSize);
#pragma unroll
    for (int pair = 0; pair < kVectorSize / 2; ++pair)
      values[pair] = x[pair];

    const int group = vector >> 4;
    const int vector_in_group = vector & 15;
#pragma unroll
    for (int output_index = 0; output_index < OutputsPerWarp;
         ++output_index) {
      const int out = output_base + output_index;
      if (out >= out_features)
        continue;
      const uint16_t code =
          packed_codes[static_cast<int64_t>(out) * vectors_per_row + vector];
      const uint32_t sign_bits = code & 0xffu;
      const uint32_t direction_index = (code >> 8) & 0x3fu;
      const uint32_t radius_low = (code >> 14) & 0x3u;
      const uint32_t extra =
          radius_extra[static_cast<int64_t>(out) * groups + group];
      const uint32_t radius_high =
          (extra >> (2 * vector_in_group)) & 0x3u;
      const uint32_t radius_code = radius_low | (radius_high << 2);

      half2 direction_sum = __float2half2_rn(0.0f);
#pragma unroll
      for (int pair = 0; pair < kVectorSize / 2; ++pair) {
        uint32_t direction_bits = shared_codebook[pair][direction_index];
        const uint32_t sign_mask =
            (((sign_bits >> (pair * 2)) & 1u) << 15) |
            (((sign_bits >> (pair * 2 + 1)) & 1u) << 31);
        direction_bits ^= sign_mask;
        const half2 direction =
            *reinterpret_cast<const half2 *>(&direction_bits);
        direction_sum =
            __hadd2(direction_sum, __hmul2(direction, values[pair]));
      }
      const float2 direction_parts = __half22float2(direction_sum);
      const half radius_half = __hfma(
          __uint2half_rn(radius_code),
          radius_scales[static_cast<int64_t>(out) * groups + group],
          radius_mins[static_cast<int64_t>(out) * groups + group]);
      accum[output_index] =
          fmaf(__half2float(radius_half),
               direction_parts.x + direction_parts.y,
               accum[output_index]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int output_index = 0; output_index < OutputsPerWarp;
         ++output_index)
      accum[output_index] +=
          __shfl_down_sync(0xffffffff, accum[output_index], offset);
  }
  if (lane == 0) {
#pragma unroll
    for (int output_index = 0; output_index < OutputsPerWarp;
         ++output_index) {
      const int out = output_base + output_index;
      if (out < out_features)
        output[static_cast<int64_t>(row) * out_features + out] =
            __float2half(accum[output_index]);
    }
  }
}

// AWQ-style mapping: a block produces eight output channels.  Threads are
// split over four interleaves and each thread consumes 32 consecutive scalar
// inputs (four SPH vectors) from a single radius group.  This amortizes the
// radius metadata loads across four vectors and keeps two output accumulators
// in one half2-friendly lane pair.
__global__ void sph_gemv_interleaved_kernel(
    const half *__restrict__ input,
    const uint16_t *__restrict__ packed_codes,
    const uint32_t *__restrict__ radius_extra,
    const half *__restrict__ radius_scales,
    const half *__restrict__ radius_mins,
    const half *__restrict__ codebook,
    half *__restrict__ output, int rows, int out_features, int in_features,
    int groups) {
  constexpr int kInterleave = 4;
  constexpr int kOutputsPerInterleave = 2;
  constexpr int kOutputsPerBlock =
      kInterleave * kOutputsPerInterleave;
  constexpr int kScalarsPerThread = 32;
  constexpr int kVectorsPerThread = kScalarsPerThread / kVectorSize;
  constexpr int kThreadsPerChunk = 2;
  constexpr int kInputForwardStep =
      kBlockSize * kScalarsPerThread / kInterleave;

  __shared__ uint32_t shared_codebook[kVectorSize / 2][kCodebookSize];
  __shared__ float block_sums[kBlockSize / 32][kOutputsPerBlock];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int interleave = (tid / kThreadsPerChunk) % kInterleave;
  const int input_offset =
      (tid / (kThreadsPerChunk * kInterleave)) * 64 +
      (tid % kThreadsPerChunk) * kScalarsPerThread;
  const int output_base = blockIdx.x * kOutputsPerBlock;
  const int row = blockIdx.y;

  if (tid < kCodebookSize * (kVectorSize / 2)) {
    const int pair = tid / kCodebookSize;
    const int direction = tid - pair * kCodebookSize;
    shared_codebook[pair][direction] =
        reinterpret_cast<const uint32_t *>(codebook)[direction * 4 + pair];
  }
  __syncthreads();

  const half *row_input = input + static_cast<int64_t>(row) * in_features;
  const int vectors_per_row = in_features / kVectorSize;
  half partial[kOutputsPerInterleave] = {
      __float2half(0.0f), __float2half(0.0f)};

  for (int scalar_base = input_offset; scalar_base < in_features;
       scalar_base += kInputForwardStep) {
    half2 values[kScalarsPerThread / 2];
    const half *input_ptr = row_input + scalar_base;
#pragma unroll
    for (int load = 0; load < kScalarsPerThread / 8; ++load)
      reinterpret_cast<float4 *>(values)[load] =
          reinterpret_cast<const float4 *>(input_ptr)[load];

    const int vector_base = scalar_base / kVectorSize;
    const int group = scalar_base / 128;
#pragma unroll
    for (int output_index = 0;
         output_index < kOutputsPerInterleave; ++output_index) {
      const int out = output_base + interleave + output_index * kInterleave;
      if (out >= out_features)
        continue;
      const int64_t metadata_index =
          static_cast<int64_t>(out) * groups + group;
      const uint32_t extra = radius_extra[metadata_index];
      const half radius_scale = radius_scales[metadata_index];
      const half radius_min = radius_mins[metadata_index];
      const uint64_t four_codes = *reinterpret_cast<const uint64_t *>(
          packed_codes + static_cast<int64_t>(out) * vectors_per_row +
          vector_base);

#pragma unroll
      for (int vector_in_thread = 0;
           vector_in_thread < kVectorsPerThread; ++vector_in_thread) {
        const uint16_t code = static_cast<uint16_t>(
            four_codes >> (16 * vector_in_thread));
        const uint32_t sign_bits = code & 0xffu;
        const uint32_t direction_index = (code >> 8) & 0x3fu;
        const uint32_t radius_low = (code >> 14) & 0x3u;
        const int vector_in_group =
            (vector_base + vector_in_thread) & 15;
        const uint32_t radius_high =
            (extra >> (2 * vector_in_group)) & 0x3u;
        const uint32_t radius_code = radius_low | (radius_high << 2);

        half2 direction_sum = __float2half2_rn(0.0f);
#pragma unroll
        for (int pair = 0; pair < kVectorSize / 2; ++pair) {
          uint32_t direction_bits =
              shared_codebook[pair][direction_index];
          const uint32_t sign_mask =
              (((sign_bits >> (pair * 2)) & 1u) << 15) |
              (((sign_bits >> (pair * 2 + 1)) & 1u) << 31);
          direction_bits ^= sign_mask;
          const half2 direction =
              *reinterpret_cast<const half2 *>(&direction_bits);
          direction_sum = __hadd2(
              direction_sum,
              __hmul2(direction,
                      values[vector_in_thread * (kVectorSize / 2) + pair]));
        }
        const half direction_dot =
            __hadd(direction_sum.x, direction_sum.y);
        const half radius = __hfma(__uint2half_rn(radius_code),
                                   radius_scale, radius_min);
        partial[output_index] =
            __hfma(radius, direction_dot, partial[output_index]);
      }
    }
  }

  float reduced[kOutputsPerInterleave];
#pragma unroll
  for (int output_index = 0; output_index < kOutputsPerInterleave;
       ++output_index) {
    reduced[output_index] = __half2float(partial[output_index]);
    reduced[output_index] +=
        __shfl_xor_sync(0xffffffff, reduced[output_index], 16);
    reduced[output_index] +=
        __shfl_xor_sync(0xffffffff, reduced[output_index], 8);
    reduced[output_index] +=
        __shfl_xor_sync(0xffffffff, reduced[output_index], 1);
  }
  if (lane == 0 || lane == 2 || lane == 4 || lane == 6) {
#pragma unroll
    for (int output_index = 0; output_index < kOutputsPerInterleave;
         ++output_index)
      block_sums[warp][output_index * kInterleave + lane / 2] =
          reduced[output_index];
  }
  __syncthreads();

  for (int output_index = tid; output_index < kOutputsPerBlock;
       output_index += kBlockSize) {
    float value = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < kBlockSize / 32;
         ++source_warp)
      value += block_sums[source_warp][output_index];
    const int out = output_base + output_index;
    if (out < out_features)
      output[static_cast<int64_t>(row) * out_features + out] =
          __float2half(value);
  }
}

} // namespace

torch::Tensor sph_gemv_forward_cuda(torch::Tensor input,
                                    torch::Tensor packed_codes,
                                    torch::Tensor radius_extra,
                                    torch::Tensor radius_scales,
                                    torch::Tensor radius_mins,
                                    torch::Tensor codebook,
                                    int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && packed_codes.is_cuda() &&
                  radius_extra.is_cuda() && radius_scales.is_cuda() &&
                  radius_mins.is_cuda() && codebook.is_cuda(),
              "sph_gemv expects CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  radius_scales.scalar_type() == torch::kFloat16 &&
                  radius_mins.scalar_type() == torch::kFloat16 &&
                  codebook.scalar_type() == torch::kFloat16,
              "sph_gemv currently supports FP16 tensors only");
  TORCH_CHECK(packed_codes.scalar_type() == torch::kInt16 &&
                  radius_extra.scalar_type() == torch::kInt32,
              "invalid SPH packed tensor dtypes");
  TORCH_CHECK(input.is_contiguous() && packed_codes.is_contiguous() &&
                  radius_extra.is_contiguous() &&
                  radius_scales.is_contiguous() && radius_mins.is_contiguous() &&
                  codebook.is_contiguous(),
              "sph_gemv expects contiguous tensors");
  TORCH_CHECK(group_size == 128,
              "sph_gemv currently requires group_size=128");

  const int64_t in_features = input.size(-1);
  const int64_t out_features = packed_codes.size(0);
  const int64_t rows = input.numel() / in_features;
  const int64_t groups = in_features / group_size;
  TORCH_CHECK(in_features % 128 == 0 && groups <= kMaxGroups,
              "invalid SPH input dimension");
  TORCH_CHECK(packed_codes.dim() == 2 &&
                  packed_codes.size(1) == in_features / kVectorSize,
              "invalid SPH packed code shape");
  TORCH_CHECK(radius_extra.sizes() ==
                      torch::IntArrayRef({out_features, groups}) &&
                  radius_scales.sizes() ==
                      torch::IntArrayRef({out_features, groups}) &&
                  radius_mins.sizes() ==
                      torch::IntArrayRef({out_features, groups}),
              "invalid SPH radius metadata shape");
  TORCH_CHECK(codebook.sizes() ==
                  torch::IntArrayRef({kCodebookSize, kVectorSize}),
              "SPH codebook must have shape [64, 8]");

  auto shape = input.sizes().vec();
  shape.back() = out_features;
  auto output = torch::empty(shape, input.options());
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const half *input_ptr = reinterpret_cast<const half *>(input.data_ptr());
  const uint16_t *code_ptr =
      reinterpret_cast<const uint16_t *>(packed_codes.data_ptr());
  const uint32_t *extra_ptr =
      reinterpret_cast<const uint32_t *>(radius_extra.data_ptr());
  const half *scale_ptr =
      reinterpret_cast<const half *>(radius_scales.data_ptr());
  const half *min_ptr = reinterpret_cast<const half *>(radius_mins.data_ptr());
  const half *codebook_ptr =
      reinterpret_cast<const half *>(codebook.data_ptr());
  half *output_ptr = reinterpret_cast<half *>(output.data_ptr());
  constexpr int tile = 8;
  const dim3 grid((out_features + tile - 1) / tile, rows);
  sph_gemv_interleaved_kernel<<<grid, kBlockSize, 0, stream>>>(
      input_ptr, code_ptr, extra_ptr, scale_ptr, min_ptr, codebook_ptr,
      output_ptr, rows, out_features, in_features, groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
