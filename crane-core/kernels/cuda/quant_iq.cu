// SPDX-License-Identifier: MIT
//
// llama.cpp i-quant weight kernels for CUDA: IQ4_NL, IQ4_XS, IQ2_S, IQ3_XXS,
// IQ3_S and Q2_0 -- the counterpart of `kernels/metal/quant_iq.metal` and
// `kernels/sycl/quant_iq.cpp` (block layouts: `crane-core/src/quantized/
// iquant.rs`). The codebooks come from `kernels/sycl/iq_grids.h`, so every
// backend reads one copy of the tables.
//
// Every format splits a row into 32-value chunks that decode on their own, so
// both entry points are written against one per-type `decode32`:
//
// - `iq_matvec`: one warp per output row; lanes stride over the row's chunks,
//   decode, dot against the f32 activations and warp-reduce. With an expert-id
//   buffer it is the `MoE` "matmul by id": input row `p` uses expert
//   `ids[p]`'s slice of a packed `[experts, rows, cols]` tensor and activation
//   row `p / x_div`, so routing never leaves the device.
// - `iq_dequant`: one thread per chunk, optionally over a list of experts,
//   feeding prefill-sized matmuls.
//
// Activations are not quantized to int8 here (the IQ4_XS-only dp4a path in
// `quant_iq4.cu` does that); the dot product is plain f32. Decoders follow the
// CPU reference's float operation order, so `iq_dequant` is bit-identical to it.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdint>

// The shared header declares `inline constexpr` tables; in device code they
// must live in global memory, so `inline` is swapped for `__device__` here.
#define inline __device__
#include "../sycl/iq_grids.h"
#undef inline

using namespace crane_iq;

#define QK_K 256
#define ROWS_PER_BLOCK 4  // output rows (warps) per thread block

enum { IQ4_NL = 0, IQ4_XS = 1, IQ2_S = 2, IQ3_XXS = 3, IQ3_S = 4, Q2_0 = 5 };

template <int TY> __device__ __forceinline__ int block_values() {
    return TY == IQ4_NL ? 32 : TY == Q2_0 ? 64 : QK_K;
}
template <int TY> __device__ __forceinline__ int block_bytes() {
    switch (TY) {
    case IQ4_NL: return 2 + 16;
    case IQ4_XS: return 2 + 2 + QK_K / 64 + QK_K / 2;
    case IQ2_S: return 2 + QK_K / 4 + QK_K / 16;
    case IQ3_XXS: return 2 + 3 * QK_K / 8;
    case IQ3_S: return 2 + 13 * QK_K / 32 + QK_K / 64;
    default: return 2 + 64 / 4;  // Q2_0
    }
}

__device__ const float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f,
};

// Little-endian f16 from 2 bytes, no alignment assumed.
__device__ __forceinline__ float load_half(const uint8_t * p) {
    const uint16_t bits = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    return __half2float(*reinterpret_cast<const half *>(&bits));
}

__device__ __forceinline__ float grid_byte(uint64_t entry, int j) {
    return float((entry >> (8 * j)) & 0xff);
}
__device__ __forceinline__ float signed_val(float v, uint32_t signs, int j) {
    return (signs >> j) & 1 ? -v : v;
}
// ggml `ksigns_iq2xs`: seven sign bits plus one keeping the negatives even.
__device__ __forceinline__ uint32_t ksigns(uint32_t bits7) {
    const uint32_t b = bits7 & 127;
    return b | ((__popc(b) & 1) << 7);
}

// Decodes chunk `c` (values 32c..32c+31) of `row` into `w`.
template <int TY> __device__ __forceinline__ void decode32(const uint8_t * row, int c, float * w) {
    if (TY == IQ4_NL) {
        const uint8_t * blk = row + size_t(c) * block_bytes<TY>();
        const float d = load_half(blk);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            w[j] = d * kvalues_iq4nl[blk[2 + j] & 0xf];
            w[j + 16] = d * kvalues_iq4nl[blk[2 + j] >> 4];
        }
    } else if (TY == Q2_0) {
        const uint8_t * blk = row + size_t(c / 2) * block_bytes<TY>();
        const float d = load_half(blk);
        const uint8_t * qs = blk + 2 + 8 * (c % 2);
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            w[j] = d * (float((qs[j / 4] >> (2 * (j % 4))) & 3) - 1.f);
        }
    } else {
        const uint8_t * blk = row + size_t(c / 8) * block_bytes<TY>();
        const int ib = c % 8;
        const float d = load_half(blk);
        if (TY == IQ4_XS) {
            const int scales_h = int(blk[2]) | (int(blk[3]) << 8);
            const int lo = (blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            const int hi = (scales_h >> (2 * ib)) & 3;
            const float dl = d * float((lo | (hi << 4)) - 32);
            const uint8_t * q = blk + 4 + QK_K / 64 + 16 * ib;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                w[j] = dl * kvalues_iq4nl[q[j] & 0xf];
                w[j + 16] = dl * kvalues_iq4nl[q[j] >> 4];
            }
        } else if (TY == IQ2_S) {
            const uint8_t * qs = blk + 2;
            const uint8_t * signs = qs + QK_K / 8;
            const uint8_t * qh = blk + 2 + QK_K / 4;
            const uint8_t sc = blk[2 + QK_K / 4 + QK_K / 32 + ib];
            const float db0 = d * (0.5f + float(sc & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + float(sc >> 4)) * 0.25f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = qs[4 * ib + l] | ((int(qh[ib]) << (8 - 2 * l)) & 0x300);
                const uint64_t g = iq2s_grid[idx];
                const uint32_t s = signs[4 * ib + l];
                const float db = l < 2 ? db0 : db1;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    w[8 * l + j] = signed_val(db * grid_byte(g, j), s, j);
                }
            }
        } else if (TY == IQ3_XXS) {
            const uint8_t * qs = blk + 2 + 8 * ib;
            const uint8_t * sas = blk + 2 + QK_K / 4 + 4 * ib;
            const uint32_t aux = uint32_t(sas[0]) | (uint32_t(sas[1]) << 8) |
                                 (uint32_t(sas[2]) << 16) | (uint32_t(sas[3]) << 24);
            const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t s = ksigns(aux >> (7 * l));
                const uint32_t g1 = iq3xxs_grid[qs[2 * l]];
                const uint32_t g2 = iq3xxs_grid[qs[2 * l + 1]];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    w[8 * l + j] = signed_val(db * grid_byte(g1, j), s, j);
                    w[8 * l + 4 + j] = signed_val(db * grid_byte(g2, j), s, j + 4);
                }
            }
        } else {  // IQ3_S
            const uint8_t * qs = blk + 2 + 8 * ib;
            const int qh = blk[2 + QK_K / 4 + ib];
            const uint8_t * signs = blk + 2 + QK_K / 4 + QK_K / 32 + 4 * ib;
            const int sc =
                (blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            const float db = d * float(1 + 2 * sc);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t g1 = iq3s_grid[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
                const uint32_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    w[8 * l + j] = signed_val(db * grid_byte(g1, j), signs[l], j);
                    w[8 * l + 4 + j] = signed_val(db * grid_byte(g2, j), signs[l], j + 4);
                }
            }
        }
    }
}

template <int TY> __device__ __forceinline__ size_t row_bytes(int cols) {
    return size_t(cols / block_values<TY>()) * block_bytes<TY>();
}

__device__ __forceinline__ void store_out(float * p, float v) { *p = v; }
__device__ __forceinline__ void store_out(half * p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_out(__nv_bfloat16 * p, float v) { *p = __float2bfloat16(v); }

// `output[p, r] = dot(W_e[r], input[p / x_div])` for `p < pairs`, where `W_e`
// is expert `e = ids[p]` of `packed`, or the only matrix without ids
// (`has_ids == 0`). Grid (ceil(out_rows / ROWS_PER_BLOCK), pairs),
// ROWS_PER_BLOCK * 32 threads.
template <int TY, typename T>
__device__ __forceinline__ void iq_matvec_impl(
    const uint8_t * packed, const uint32_t * ids, const float * input, T * output,
    uint64_t expert_stride, int x_div, int out_rows, int cols, int has_ids) {
    const int pair = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
    // Uniform across the warp, so the full-mask reduction below stays legal.
    if (row >= out_rows) {
        return;
    }
    const size_t expert = has_ids ? size_t(ids[pair]) : 0;
    const uint8_t * r = packed + expert * expert_stride + size_t(row) * row_bytes<TY>(cols);
    const float * x = input + size_t(pair / x_div) * cols;
    const int chunks = cols / 32;
    float acc = 0.f;
    float w[32];
    for (int c = lane; c < chunks; c += 32) {
        decode32<TY>(r, c, w);
        const float * xc = x + 32 * c;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            acc += w[j] * xc[j];
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        store_out(output + size_t(pair) * out_rows + row, acc);
    }
}

// Decodes `n_mats` matrices of `n_rows` x `cols` into `[n_mats, n_rows,
// cols]`: matrix `m` is expert `ids[m]` of `packed`, or the only one without
// ids. One thread per 32-value chunk; grid (ceil(cols / 32 / 256), n_mats *
// n_rows), 256 threads.
template <int TY, typename T>
__device__ __forceinline__ void iq_dequant_impl(
    const uint8_t * packed, const uint32_t * ids, T * output, uint64_t expert_stride,
    int n_mats, int n_rows, int cols, int has_ids) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t mat_row = blockIdx.y;
    if (c >= cols / 32 || mat_row >= size_t(n_mats) * size_t(n_rows)) {
        return;
    }
    const size_t m = mat_row / size_t(n_rows);
    const size_t row = mat_row % size_t(n_rows);
    const size_t expert = has_ids ? size_t(ids[m]) : 0;
    float w[32];
    decode32<TY>(packed + expert * expert_stride + row * row_bytes<TY>(cols), c, w);
    T * out = output + mat_row * size_t(cols) + 32 * size_t(c);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        store_out(out + j, w[j]);
    }
}

#define IQ_KERNELS(TY, NAME, T, TNAME)                                                       \
    extern "C" __global__ void matvec_##NAME##_##TNAME(                                      \
        const uint8_t * packed, const uint32_t * ids, const float * input, T * output,       \
        uint64_t expert_stride, int x_div, int out_rows, int cols, int has_ids) {            \
        iq_matvec_impl<TY, T>(packed, ids, input, output, expert_stride, x_div, out_rows,    \
                              cols, has_ids);                                                \
    }                                                                                        \
    extern "C" __global__ void dequant_##NAME##_##TNAME(                                     \
        const uint8_t * packed, const uint32_t * ids, T * output, uint64_t expert_stride,    \
        int n_mats, int n_rows, int cols, int has_ids) {                                     \
        iq_dequant_impl<TY, T>(packed, ids, output, expert_stride, n_mats, n_rows, cols,     \
                               has_ids);                                                     \
    }

#define IQ_ALL_TYPES(T, TNAME)               \
    IQ_KERNELS(IQ4_NL, iq4_nl, T, TNAME)     \
    IQ_KERNELS(IQ4_XS, iq4_xs, T, TNAME)     \
    IQ_KERNELS(IQ2_S, iq2_s, T, TNAME)       \
    IQ_KERNELS(IQ3_XXS, iq3_xxs, T, TNAME)   \
    IQ_KERNELS(IQ3_S, iq3_s, T, TNAME)       \
    IQ_KERNELS(Q2_0, q2_0, T, TNAME)

IQ_ALL_TYPES(float, f32)
IQ_ALL_TYPES(half, f16)
IQ_ALL_TYPES(__nv_bfloat16, bf16)
