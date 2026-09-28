// Prefill GEMM: 32x64 output tile for short K, 64x64 tile for K >= 4096.
// The 64x64 tile halves the packed-weight decode per output element, but its
// split-K policy would over-split these shapes, so the original split heuristic
// is kept for both kernels.
#include "sage_gemv_cuda.h"

#include <mma.h>
#include <algorithm>

namespace sage_base {
using namespace nvcuda;

// Four warps cover a 32x64 output tile. Each 128-wide quantization group
// is decoded into two shared-memory slabs; the next group's loads overlap MMA.
// Small grids split K, accumulating partial results in FP32 before one rounding.
constexpr int kM = 32;
constexpr int kN = 64;
constexpr int kK = 64;
constexpr int kAStride = 128 + 8;
constexpr int kBStride = kK + 8;
constexpr int kThreads = 128;
// The decoded tile is written 16 bytes at a time by 32 lanes that cover only
// eight distinct 128-byte rows, so the second slab starts one bank quad later:
// slab and row then map to distinct quads and the stores stay conflict free.
constexpr int kBPad = 16;
constexpr int kABytes = kM * kAStride * 2;
constexpr int kBWords = 2 * (kN * kBStride / 2 + kBPad);
constexpr int kBBytes = kBWords * 4;

struct Shared {
  alignas(32) char ab[kABytes + kBBytes];
  uint32_t book[4][256];
};

template <bool kAligned, bool kSplit, int Entries>
__global__ void __launch_bounds__(kThreads) sage_gemm_kernel_base(
    const half *__restrict__ input, const uint32_t *__restrict__ codes,
    const uint32_t *__restrict__ radii, const half2 *__restrict__ grid,
    const uint32_t *__restrict__ book, half *__restrict__ output,
    float *__restrict__ workspace, int parts, int rows, int n, int k) {
  __shared__ Shared s;
  half *sa = reinterpret_cast<half *>(s.ab);
  half *sb = reinterpret_cast<half *>(s.ab + kABytes);
  constexpr int kBSlab = kN * kBStride + 2 * kBPad;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int mbase = blockIdx.y * kM;
  const int nbase = blockIdx.x * kN;
  const int groups = k >> 7;
  constexpr int dbits = Entries == 64 ? 6 : (Entries == 128 ? 7 : 8);
  constexpr int cmask = Entries - 1;
  for (int i = tid; i < 4 * Entries; i += kThreads)
    s.book[i / Entries][i % Entries] = book[i];
  __syncthreads();

  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[j], 0.0f);

  // One activation row and one 16-byte column chunk per thread for each of the
  // four 8-row slices of the tile.
  const int a_row = tid >> 4;
  const int a_col = (tid & 15) * 8;
  // One warp owns one 16-row packed tile, i.e. 16 output columns.
  const int panel_tile = (nbase >> 4) + warp;
  const int panel_row = nbase + warp * 16 + (lane >> 1);
  const bool panel_ok = panel_row < n;
  const int cword = (lane & 1) * 4;

  uint4 areg[4];
  uint4 creg = make_uint4(0, 0, 0, 0);
  uint32_t rword = 0;
  float2 endpoints = make_float2(0.0f, 0.0f);

  auto load_a = [&](int g) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = mbase + a_row + j * 8;
      const half *src = input + static_cast<int64_t>(row) * k + g * 128 + a_col;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (row < rows) {
        if constexpr (kAligned) {
          value = *reinterpret_cast<const uint4 *>(src);
        } else {
          half *dst = reinterpret_cast<half *>(&value);
#pragma unroll
          for (int c = 0; c < 8; ++c) dst[c] = src[c];
        }
      }
      areg[j] = value;
    }
  };

  auto store_a = [&]() {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      *reinterpret_cast<uint4 *>(&sa[(a_row + j * 8) * kAStride + a_col]) = areg[j];
  };

  auto load_codes = [&](int g) {
    const int64_t base = (static_cast<int64_t>(panel_tile) * groups + g) * 16 + (lane >> 1);
    creg = *reinterpret_cast<const uint4 *>(codes + base * 8 + cword);
    rword = radii[base];
    endpoints = __half22float2(grid[base]);
  };

  auto store_b = [&]() {
    const uint32_t words[4] = {creg.x, creg.y, creg.z, creg.w};
    const int row = warp * 16 + (lane >> 1);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const uint32_t code = (words[q] >> (16 * v)) & 0xffffu;
        const int field = 2 * ((lane & 1) * 4 + q) + v;
        const uint32_t rbits =
            (code >> (8 + dbits)) | (((rword >> (2 * field)) & 3u) << (8 - dbits));
        const float radius = fmaf(static_cast<float>(rbits), endpoints.y, endpoints.x);
        uint4 value;
        half *dst = reinterpret_cast<half *>(&value);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t bits = s.book[p][(code >> 8) & cmask];
          bits ^= (((code >> (2 * p)) & 1u) << 15) | (((code >> (2 * p + 1)) & 1u) << 31);
          const float2 d = __half22float2(*reinterpret_cast<half2 *>(&bits));
          dst[2 * p] = __float2half_rn(d.x * radius);
          dst[2 * p + 1] = __float2half_rn(d.y * radius);
        }
        const int slab = lane & 1;
        *reinterpret_cast<uint4 *>(
            &sb[slab * kBSlab + row * kBStride + 16 * q + 8 * v]) = value;
      }
    }
  };

  auto do_mma = [&](int sl) {
#pragma unroll
    for (int step = 0; step < kK; step += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
      wmma::load_matrix_sync(
          af, &sa[(warp_m * 16) * kAStride + sl * kK + step], kAStride);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(
            bf[j], &sb[sl * kBSlab + (warp_n * 32 + j * 16) * kBStride + step], kBStride);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[j], af, bf[j], acc[j]);
    }
  };

  const int begin = kSplit ? groups * blockIdx.z / parts : 0;
  const int end = kSplit ? groups * (blockIdx.z + 1) / parts : groups;
  load_a(begin);
  if (panel_ok) load_codes(begin);
  store_a();
  store_b();
  __syncthreads();
  for (int g = begin; g < end; ++g) {
    if (g + 1 < end) {
      load_a(g + 1);
      if (panel_ok) load_codes(g + 1);
    }
    do_mma(0);
    do_mma(1);
    __syncthreads();
    if (g + 1 < end) {
      store_a();
      store_b();
    }
    __syncthreads();
  }

  // The weight buffer is idle now; reuse it to stage the fp32 accumulators.
  float(*staging)[kBStride * 2] = reinterpret_cast<float(*)[kBStride * 2]>(sb);
#pragma unroll
  for (int j = 0; j < 2; ++j)
    wmma::store_matrix_sync(&staging[warp_m * 16][warp_n * 32 + j * 16], acc[j], kBStride * 2,
                            wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < kM * kN; i += kThreads) {
    const int row = i / kN;
    const int column = i % kN;
    const int global_row = mbase + row;
    const int global_column = nbase + column;
    if (global_row < rows && global_column < n) {
      const int64_t index = static_cast<int64_t>(global_row) * n + global_column;
      if constexpr (!kSplit)
        output[index] = __float2half_rn(staging[row][column]);
      else
        workspace[static_cast<int64_t>(blockIdx.z) * rows * n + index] = staging[row][column];
    }
  }
}

__global__ void reduce_gemm_base(const float *workspace, half *output, int64_t count, int parts) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    float sum = 0.0f;
    for (int p = 0; p < parts; ++p) sum += workspace[p * count + index];
    output[index] = __float2half_rn(sum);
  }
}


int sage_gemm_splits_base(int rows, int n, int k, int sm_count) {
  const int64_t blocks = static_cast<int64_t>((n + kN - 1) / kN) * ((rows + kM - 1) / kM);
  if (blocks >= sm_count && k < 4096) return 1;
  return static_cast<int>(std::min<int64_t>(
      std::max(1, k / 512), std::min<int64_t>(8, (3 * sm_count + blocks - 1) / blocks)));
}

void sage_gemm_cuda_base(const half *input, const uint32_t *codes,
                    const uint32_t *radii, const half2 *grid,
                    const uint32_t *book, half *output, float *workspace, int parts,
                    int rows, int n, int k, int entries, cudaStream_t stream) {
  const dim3 blocks((n + kN - 1) / kN, (rows + kM - 1) / kM, parts);
  const bool aligned = (reinterpret_cast<uintptr_t>(input) & 15) == 0;
#define DISPATCH(A, S, E) \
  sage_gemm_kernel_base<A, S, E><<<blocks, kThreads, 0, stream>>>( \
      input, codes, radii, grid, book, output, workspace, parts, rows, n, k)
#define LAUNCH(E) \
  if (aligned) { \
    if (parts > 1) { DISPATCH(true, true, E); } \
    else { DISPATCH(true, false, E); } \
  } else { \
    if (parts > 1) { DISPATCH(false, true, E); } \
    else { DISPATCH(false, false, E); } \
  }
  if (entries == 64) { LAUNCH(64); }
  else if (entries == 128) { LAUNCH(128); }
  else { LAUNCH(256); }
#undef LAUNCH
#undef DISPATCH
  if (parts > 1) {
    const int64_t count = static_cast<int64_t>(rows) * n;
    reduce_gemm_base<<<(count + 255) / 256, 256, 0, stream>>>(workspace, output, count, parts);
  }
}
}

namespace sage_m64 {
using namespace nvcuda;

// Four warps cover a 64x64 output tile. Each 128-wide quantization group is
// decoded into two shared-memory slabs; the next group's loads overlap MMA.
// The activation tile is staged one 64-wide half at a time so that the block
// stays under a third of the SM's shared memory and three CTAs stay resident.
// Small grids split K, accumulating partial results in FP32 before one rounding.
constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 64;
constexpr int kAStride = kK + 8;
constexpr int kBStride = kK + 8;
constexpr int kThreads = 128;
// The decoded tile is written 16 bytes at a time by 32 lanes that cover only
// eight distinct 128-byte rows, so the second slab starts one bank quad later:
// slab and row then map to distinct quads and the stores stay conflict free.
constexpr int kBPad = 16;
constexpr int kABytes = kM * kAStride * 2;
constexpr int kBWords = 2 * (kN * kBStride / 2 + kBPad);
constexpr int kBBytes = kBWords * 4;

template<int Entries> struct Shared {
  alignas(32) char ab[kABytes + kBBytes];
  uint32_t book[4][Entries];
};

template <bool kAligned, bool kSplit, int Entries>
__global__ void __launch_bounds__(kThreads) sage_gemm_kernel_m64(
    const half *__restrict__ input, const uint32_t *__restrict__ codes,
    const uint32_t *__restrict__ radii, const half2 *__restrict__ grid,
    const uint32_t *__restrict__ book, half *__restrict__ output,
    float *__restrict__ workspace, int parts, int rows, int n, int k) {
  __shared__ Shared<Entries> s;
  half *sa = reinterpret_cast<half *>(s.ab);
  half *sb = reinterpret_cast<half *>(s.ab + kABytes);
  constexpr int kBSlab = kN * kBStride + 2 * kBPad;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int mbase = blockIdx.y * kM;
  const int nbase = blockIdx.x * kN;
  const int groups = k >> 7;
  constexpr int dbits = Entries == 64 ? 6 : (Entries == 128 ? 7 : 8);
  constexpr int cmask = Entries - 1;
  for (int i = tid; i < 4 * Entries; i += kThreads)
    s.book[i / Entries][i % Entries] = book[i];
  __syncthreads();

  const int warp_m = (warp / 2) * 2;
  const int warp_n = warp % 2;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  // One activation row and one 16-byte column chunk per thread for each of the
  // four 16-row slices of the half-width A slab.
  const int a_row = tid >> 3;
  const int a_col = (tid & 7) * 8;
  // One warp owns one 16-row packed tile, i.e. 16 output columns.
  const int panel_tile = (nbase >> 4) + warp;
  const int panel_row = nbase + warp * 16 + (lane >> 1);
  const bool panel_ok = warp < 4 && panel_row < n;
  const int cword = (lane & 1) * 4;

  uint4 acur[4];
  uint4 anext[4];
  uint4 creg = make_uint4(0, 0, 0, 0);
  uint32_t rword = 0;
  float2 endpoints = make_float2(0.0f, 0.0f);

  auto load_a = [&](int g, int part, uint4 *dst) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = mbase + a_row + j * (kM / 4);
      const half *src = input + static_cast<int64_t>(row) * k + g * 128 + part * kK + a_col;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (row < rows) {
        if constexpr (kAligned) {
          value = *reinterpret_cast<const uint4 *>(src);
        } else {
          half *out = reinterpret_cast<half *>(&value);
#pragma unroll
          for (int c = 0; c < 8; ++c) out[c] = src[c];
        }
      }
      dst[j] = value;
    }
  };

  auto store_a = [&](const uint4 *src) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      *reinterpret_cast<uint4 *>(&sa[(a_row + j * (kM / 4)) * kAStride + a_col]) = src[j];
  };

  auto load_codes = [&](int g) {
    const int64_t base = (static_cast<int64_t>(panel_tile) * groups + g) * 16 + (lane >> 1);
    creg = *reinterpret_cast<const uint4 *>(codes + base * 8 + cword);
    rword = radii[base];
    endpoints = __half22float2(grid[base]);
  };

  auto store_b = [&]() {
    if (warp >= 4) return;
    const uint32_t words[4] = {creg.x, creg.y, creg.z, creg.w};
    const int row = warp * 16 + (lane >> 1);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const uint32_t code = (words[q] >> (16 * v)) & 0xffffu;
        const int field = 2 * ((lane & 1) * 4 + q) + v;
        const uint32_t rbits =
            (code >> (8 + dbits)) | (((rword >> (2 * field)) & 3u) << (8 - dbits));
        const float radius = fmaf(static_cast<float>(rbits), endpoints.y, endpoints.x);
        uint4 value;
        half *dst = reinterpret_cast<half *>(&value);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t bits = s.book[p][(code >> 8) & cmask];
          bits ^= (((code >> (2 * p)) & 1u) << 15) | (((code >> (2 * p + 1)) & 1u) << 31);
          const float2 d = __half22float2(*reinterpret_cast<half2 *>(&bits));
          dst[2 * p] = __float2half_rn(d.x * radius);
          dst[2 * p + 1] = __float2half_rn(d.y * radius);
        }
        const int slab = lane & 1;
        *reinterpret_cast<uint4 *>(
            &sb[slab * kBSlab + row * kBStride + 16 * q + 8 * v]) = value;
      }
    }
  };

  auto do_mma = [&](int sl) {
#pragma unroll
    for (int step = 0; step < kK; step += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(af[i], &sa[(warp_m + i) * 16 * kAStride + step], kAStride);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(bf[j], &sb[sl * kBSlab + (warp_n * 2 + j) * 16 * kBStride + step], kBStride);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
  };

  const int begin = kSplit ? groups * blockIdx.z / parts : 0;
  const int end = kSplit ? groups * (blockIdx.z + 1) / parts : groups;
  load_a(begin, 0, acur);
  if (panel_ok) load_codes(begin);
  store_a(acur);
  store_b();
  __syncthreads();
  for (int g = begin; g < end; ++g) {
    load_a(g, 1, anext);
    do_mma(0);
    __syncthreads();
    store_a(anext);
    if (g + 1 < end) {
      load_a(g + 1, 0, acur);
      if (panel_ok) load_codes(g + 1);
    }
    __syncthreads();
    do_mma(1);
    __syncthreads();
    if (g + 1 < end) {
      store_a(acur);
      store_b();
    }
    __syncthreads();
  }

  // The weight buffer is idle now; reuse it to stage the fp32 accumulators.
  float(*staging)[kN] = reinterpret_cast<float(*)[kN]>(sb);
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&staging[(warp_m + i) * 16][(warp_n * 2 + j) * 16], acc[i][j], kN, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < kM * kN; i += kThreads) {
    const int row = i / kN;
    const int column = i % kN;
    const int global_row = mbase + row;
    const int global_column = nbase + column;
    if (global_row < rows && global_column < n) {
      const int64_t index = static_cast<int64_t>(global_row) * n + global_column;
      if constexpr (!kSplit)
        output[index] = __float2half_rn(staging[row][column]);
      else
        workspace[static_cast<int64_t>(blockIdx.z) * rows * n + index] = staging[row][column];
    }
  }
}

__global__ void reduce_gemm_m64(const float *workspace, half *output, int64_t count, int parts) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    float sum = 0.0f;
    for (int p = 0; p < parts; ++p) sum += workspace[p * count + index];
    output[index] = __float2half_rn(sum);
  }
}


int sage_gemm_splits_m64(int rows, int n, int k, int sm_count) {
  const int64_t blocks = static_cast<int64_t>((n + kN - 1) / kN) * ((rows + kM - 1) / kM);
  if (blocks >= sm_count && k < 4096) return 1;
  return static_cast<int>(std::min<int64_t>(
      std::max(1, k / 512), std::min<int64_t>(8, (3 * sm_count + blocks - 1) / blocks)));
}

void sage_gemm_cuda_m64(const half *input, const uint32_t *codes,
                    const uint32_t *radii, const half2 *grid,
                    const uint32_t *book, half *output, float *workspace, int parts,
                    int rows, int n, int k, int entries, cudaStream_t stream) {
  const dim3 blocks((n + kN - 1) / kN, (rows + kM - 1) / kM, parts);
  const bool aligned = (reinterpret_cast<uintptr_t>(input) & 15) == 0;
#define DISPATCH(A, S, E) \
  sage_gemm_kernel_m64<A, S, E><<<blocks, kThreads, 0, stream>>>( \
      input, codes, radii, grid, book, output, workspace, parts, rows, n, k)
#define LAUNCH(E) \
  if (aligned) { \
    if (parts > 1) { DISPATCH(true, true, E); } \
    else { DISPATCH(true, false, E); } \
  } else { \
    if (parts > 1) { DISPATCH(false, true, E); } \
    else { DISPATCH(false, false, E); } \
  }
  if (entries == 64) { LAUNCH(64); }
  else if (entries == 128) { LAUNCH(128); }
  else { LAUNCH(256); }
#undef LAUNCH
#undef DISPATCH
  if (parts > 1) {
    const int64_t count = static_cast<int64_t>(rows) * n;
    reduce_gemm_m64<<<(count + 255) / 256, 256, 0, stream>>>(workspace, output, count, parts);
  }
}
}

int sage_gemm_splits(int rows, int n, int k, int sm_count) {
  return sage_base::sage_gemm_splits_base(rows, n, k, sm_count);
}

void sage_gemm_cuda(const half *input, const uint32_t *codes,
                    const uint32_t *radii, const half2 *grid,
                    const uint32_t *book, half *output, float *workspace, int parts,
                    int rows, int n, int k, int entries, cudaStream_t stream) {
  // The 64x64 tile decodes each packed weight for 64 output rows instead of 32,
  // so it pays off wherever it still has enough blocks. Large K always does;
  // below K=4096 it depends on the shape, so compare the reduced grid to the
  // SM count rather than gating on K alone.
  static const int sm_count = [] {
    int device = 0;
    int value = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
    return value;
  }();
  const int64_t blocks_m64 = static_cast<int64_t>((n + 63) / 64) * ((rows + 63) / 64);
  if (k >= 4096 || (rows >= 256 && blocks_m64 >= sm_count))
    sage_m64::sage_gemm_cuda_m64(input, codes, radii, grid, book, output, workspace,
                                 parts, rows, n, k, entries, stream);
  else
    sage_base::sage_gemm_cuda_base(input, codes, radii, grid, book, output, workspace,
                                   parts, rows, n, k, entries, stream);
}
