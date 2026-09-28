#include "sage_gemv_cuda.h"

// Negative selects the production dispatch; 0--7 retain the original tuning sweep.
int sage_gemv_tuning = -1;

namespace {

template <int Entries, int K, int Threads, int Outputs, bool Gather, bool HalfAccum,
          bool Aligned, int RowsPerThread, bool VectorBook, int MinBlocks>
__global__ void __launch_bounds__(Threads, MinBlocks) sage_gemv_kernel(
    const half *__restrict__ input, const uint32_t *__restrict__ codes,
    const uint32_t *__restrict__ radii, const half2 *__restrict__ grid,
    const uint32_t *__restrict__ book, half *__restrict__ output,
    const int32_t *__restrict__ permutation, int n, int dynamic_k) {
  constexpr int ParallelOutputs = Outputs / RowsPerThread;
  constexpr int Splits = Threads / (4 * ParallelOutputs);
  constexpr int DirectionBits = Entries == 64 ? 6 : (Entries == 128 ? 7 : 8);
  const int k = K ? K : dynamic_k;
  const int groups = k / 128;
  __shared__ __align__(16) uint32_t shared_book[4][Entries];
  __shared__ float partial[Splits][Outputs];
  extern __shared__ __align__(16) half gathered[];
  const unsigned tid = threadIdx.x;
  const unsigned chunk = tid % 4;
  const unsigned out_local = (tid / 4) % ParallelOutputs;
  const unsigned split = tid / (4 * ParallelOutputs);
  const unsigned out_base = blockIdx.x * Outputs;
  // The checkpoint stays planar. Only the CTA-local copy is interleaved for LDS.128.
#pragma unroll
  for (int base = 0; base < 4 * Entries; base += Threads) {
    const unsigned i = base + tid;
    const unsigned target = VectorBook ? (i % Entries) * 4 + i / Entries : i;
    if (i < 4 * Entries)
      reinterpret_cast<uint32_t *>(shared_book)[target] = book[i];
  }
  const half *x = input + static_cast<int64_t>(blockIdx.y) * k;
  if constexpr (Gather) {
    for (int i = tid; i < k; i += Threads)
      gathered[i] = x[permutation ? permutation[i] : i];
    x = gathered;
  }
  __syncthreads();
  float sum[RowsPerThread] = {};
  half hsum[RowsPerThread] = {};
  for (int group = split; group < groups; group += Splits) {
    half2 values[16];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const half *src = x + group * 128 + chunk * 32 + v * 8;
      if constexpr (Aligned || Gather)
        reinterpret_cast<uint4 *>(values)[v] = *reinterpret_cast<const uint4 *>(src);
      else {
#pragma unroll
        for (int c = 0; c < 8; ++c)
          reinterpret_cast<half *>(values)[v * 8 + c] = src[c];
      }
    }
#pragma unroll
    for (int o = 0; o < RowsPerThread; ++o) {
      const unsigned row = out_base % 16 + out_local + o * ParallelOutputs;
      const int64_t index = (static_cast<int64_t>(out_base / 16) * groups + group) * 16 + row;
      const uint2 words = *reinterpret_cast<const uint2 *>(codes + index * 8 + chunk * 2);
      const uint32_t extra = radii[index];
      const half2 endpoints = grid[index];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t code = ((v < 2 ? words.x : words.y) >> (16 * (v & 1))) & 0xffffu;
        const int vector = chunk * 4 + v;
        const uint32_t r = (code >> (8 + DirectionBits)) |
            (((extra >> (2 * vector)) & 3u) << (8 - DirectionBits));
        half2 dot2 = __float2half2_rn(0.0f);
        uint4 direction;
        if constexpr (VectorBook)
          direction = reinterpret_cast<const uint4 *>(shared_book)[(code >> 8) & (Entries - 1)];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t bits;
          if constexpr (VectorBook) bits = reinterpret_cast<uint32_t *>(&direction)[p];
          else bits = shared_book[p][(code >> 8) & (Entries - 1)];
          bits ^= (((code >> (2 * p)) & 1u) << 15) |
                  (((code >> (2 * p + 1)) & 1u) << 31);
          dot2 = __hfma2(*reinterpret_cast<half2 *>(&bits), values[v * 4 + p], dot2);
        }
        if constexpr (HalfAccum) {
          const half radius = __hfma(__uint2half_rn(r), endpoints.y, endpoints.x);
          hsum[o] = __hfma(radius, __hadd(dot2.x, dot2.y), hsum[o]);
        } else {
          const float2 e = __half22float2(endpoints);
          const float radius = fmaf(static_cast<float>(r), e.y, e.x);
          const float2 dot = __half22float2(dot2);
          sum[o] = fmaf(radius, dot.x + dot.y, sum[o]);
        }
      }
    }
  }
#pragma unroll
  for (int o = 0; o < RowsPerThread; ++o) {
    if constexpr (HalfAccum) sum[o] = __half2float(hsum[o]);
    for (int offset = 2; offset; offset /= 2)
      sum[o] += __shfl_down_sync(0xffffffff, sum[o], offset, 4);
    if (chunk == 0) partial[split][out_local + o * ParallelOutputs] = sum[o];
  }
  __syncthreads();
  if (tid < Outputs && out_base + tid < n) {
    float total = 0.0f;
#pragma unroll
    for (int s = 0; s < Splits; ++s) total += partial[s][tid];
    output[static_cast<int64_t>(blockIdx.y) * n + out_base + tid] = __float2half_rn(total);
  }
}

} // namespace

bool sage_gemv_fuses_permutation(int n, int k) {
  return k <= 4096;
}

void sage_gemv_cuda(const half *input, const uint32_t *codes,
                    const uint32_t *radii, const half2 *grid,
                    const uint32_t *book,
                    half *output, const int32_t *permutation,
                    int rows, int n, int k, int entries, cudaStream_t stream) {
#define CALL(E, K, T, O, G, H, R, V, B) \
  if (G || (reinterpret_cast<uintptr_t>(input) & 15) == 0) { \
    sage_gemv_kernel<E, K, T, O, G, H, true, R, V, B><<<dim3((n + O - 1) / O, rows), T, G ? k * sizeof(half) : 0, stream>>>( \
        input, codes, radii, grid, book, output, permutation, n, k); \
  } else { \
    sage_gemv_kernel<E, K, T, O, G, H, false, R, V, B><<<dim3((n + O - 1) / O, rows), T, 0, stream>>>( \
        input, codes, radii, grid, book, output, permutation, n, k); \
  }
#define AUTO(E) \
  if (k == 2560) { \
    if (permutation) { CALL(E, 2560, 320, 8, true, false, 1, true, 4); } \
    else if (n >= 4096) { CALL(E, 2560, 256, 16, false, false, 2, true, 5); } \
    else { CALL(E, 2560, 320, 8, false, false, 1, true, 4); } \
  } else if (k == 4096) { \
    if (permutation) { CALL(E, 4096, 256, 8, true, false, 1, true, 5); } \
    else if (rows == 1 && n == 2560) { CALL(E, 4096, 128, 4, false, false, 1, true, 8); } \
    else { CALL(E, 4096, 128, 8, false, false, 1, true, 8); } \
  } else if (rows == 1 && n == 2560 && !permutation) { \
    CALL(E, 9728, 128, 4, false, false, 1, true, 8); \
  } else { CALL(E, 9728, 320, 8, false, false, 1, true, 4); }
  // Preordered inputs favor wider output tiles at large N. For single-row
  // K4096 or K9728 with N2560, four outputs double the CTA count to improve
  // utilization. Keep the short-K fused-gather configurations for eager.
  if (sage_gemv_tuning < 0 && (k == 2560 || k == 4096 || k == 9728)) {
    if (entries == 64) { AUTO(64); }
    else if (entries == 128) { AUTO(128); }
    else { AUTO(256); }
    return;
  }
#undef AUTO
#define FIXED_CASE(E, T, O, H) \
  if (k == 2560) { \
    if (permutation) { CALL(E, 2560, T, O, true, H, 2, false, 1); } \
    else { CALL(E, 2560, T, O, false, H, 2, false, 1); } \
  } else if (k == 4096) { \
    if (permutation) { CALL(E, 4096, T, O, true, H, 2, false, 1); } \
    else { CALL(E, 4096, T, O, false, H, 2, false, 1); } \
  } else { CALL(E, 9728, T, O, false, H, 2, false, 1); }
#define FIXED(E) \
  switch (sage_gemv_tuning) { \
    case 0: { FIXED_CASE(E, 128, 8, true); break; } \
    case 1: { FIXED_CASE(E, 256, 8, true); break; } \
    case 2: { FIXED_CASE(E, 128, 16, true); break; } \
    case 3: { FIXED_CASE(E, 256, 16, true); break; } \
    case 4: { FIXED_CASE(E, 128, 8, false); break; } \
    case 5: default: { FIXED_CASE(E, 256, 8, false); break; } \
    case 6: { FIXED_CASE(E, 128, 16, false); break; } \
    case 7: { FIXED_CASE(E, 256, 16, false); break; } \
  }
  if (k == 2560 || k == 4096 || k == 9728) {
    if (entries == 64) { FIXED(64); }
    else if (entries == 128) { FIXED(128); }
    else { FIXED(256); }
    return;
  }
#define FALLBACK(E) \
  if (sage_gemv_fuses_permutation(n, k)) { CALL(E, 0, 256, 8, true, false, 2, false, 1); } \
  else { CALL(E, 0, 256, 8, false, false, 2, false, 1); }
  if (entries == 64) { FALLBACK(64); }
  else if (entries == 128) { FALLBACK(128); }
  else { FALLBACK(256); }
#undef FALLBACK
#undef FIXED
#undef FIXED_CASE
#undef CALL
}
