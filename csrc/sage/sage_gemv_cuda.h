#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// codes [ceil(N/16), K/128, 16, 8], radii [ceil(N/16), K/128, 16],
// grid [ceil(N/16), K/128, 16, 2] (min, step), book [4, entries, 2].
// entries=64/128/256 selects 6/7/8 direction bits and 4/3/2 radius bits.
// Main codes carry radius low bits above the direction; radii carry two high bits.
// GEMM input is contiguous in quantized channel order. GEMV can gather from
// original channel order when sage_gemv_fuses_permutation returns true; otherwise
// the caller gathers. All activation processing remains in the timed forward.
bool sage_gemv_fuses_permutation(int n, int k);

void sage_gemv_cuda(const half *input, const uint32_t *codes,
                    const uint32_t *radii, const half2 *grid,
                    const uint32_t *book, half *output, const int32_t *permutation,
                    int rows, int n, int k, int entries, cudaStream_t stream);

int sage_gemm_splits(int rows, int n, int k, int sm_count);

void sage_gemm_cuda(const half *input, const uint32_t *codes,
                    const uint32_t *radii, const half2 *grid,
                    const uint32_t *book, half *output, float *workspace, int parts,
                    int rows, int n, int k, int entries, cudaStream_t stream);
