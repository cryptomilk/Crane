// SPDX-License-Identifier: MIT
//
// llama.cpp IQ4_XS / IQ4_NL weight kernels (see crane-core/src/quantized/iquant.rs
// for the block layouts). Entry points:
//
// - `iq4_quantize_q8_*` + `iq4_xs_matvec_q8_{1,4}_*`: IQ4_XS decode. The
//   activations are quantized to int8 and dotted with the weights via `dp4a`
//   (llama.cpp's approach), one warp per output row; weights are decoded
//   straight from the packed bytes, never materialised.
// - `iq4_nl_matvec{1,4}_f32`: IQ4_NL decode, a simpler float kernel (IQ4_NL
//   only shows up as a fallback type for rows not divisible by 256).
// - `*_dequant_{f32,f16,bf16}`: expands a range of weight rows for prefill,
//   where the caller follows up with a regular (cuBLAS) matmul.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define QK_K 256
#define QK4_NL 32
#define IQ4_XS_BLOCK_BYTES 136
#define IQ4_NL_BLOCK_BYTES 18
#define IQ4_WARPS 4

__device__ const float iq4_kvalues[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f,
};

__device__ __forceinline__ float iq4_half(const uint8_t * p) {
    const uint16_t bits = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    return __half2float(*reinterpret_cast<const half *>(&bits));
}

__device__ __forceinline__ float iq4_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One lane's share of a 256-weight group: its sub-block scale, 4 packed bytes
// (8 nibbles), and the column of its first low/high nibble within the row.
struct iq4_lane_group {
    float dl;
    uint32_t q;
    int col;  // low nibbles cover col..col+3, high nibbles col+16..col+19
    bool valid;
};

// IQ4_XS: group g is block g. Lane l takes sub-block l/4 and bytes 4*(l%4)..+3
// of its 16 packed bytes.
__device__ __forceinline__ iq4_lane_group iq4_xs_lane(const uint8_t * row, int g, int lane) {
    const uint8_t * blk = row + size_t(g) * IQ4_XS_BLOCK_BYTES;
    const int ib = lane >> 2;
    const int j0 = (lane & 3) * 4;
    const uint16_t scales_h = uint16_t(blk[2]) | (uint16_t(blk[3]) << 8);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
    iq4_lane_group out;
    out.dl = iq4_half(blk) * float(ls - 32);
    // Blocks are 136 bytes (a multiple of 4) and the qs payload starts at byte
    // 8, so this word load is aligned whenever the tensor base is.
    out.q = *reinterpret_cast<const uint32_t *>(blk + 8 + 16 * ib + j0);
    out.col = g * QK_K + 32 * ib + j0;
    out.valid = true;
    return out;
}

// IQ4_NL: group g spans blocks 8g..8g+7 (the last group may be partial). Lane l
// takes block 8g + l/4 and bytes 4*(l%4)..+3 of its 16 packed bytes. 18-byte
// blocks only guarantee 2-byte alignment, hence the half-word loads.
__device__ __forceinline__ iq4_lane_group iq4_nl_lane(const uint8_t * row, int g, int lane, int blocks) {
    iq4_lane_group out;
    const int b = g * 8 + (lane >> 2);
    out.valid = b < blocks;
    if (!out.valid) {
        out.dl = 0.f;
        out.q = 0;
        out.col = 0;
        return out;
    }
    const uint8_t * blk = row + size_t(b) * IQ4_NL_BLOCK_BYTES;
    const int j0 = (lane & 3) * 4;
    const uint16_t * qs = reinterpret_cast<const uint16_t *>(blk + 2 + j0);
    out.dl = iq4_half(blk);
    out.q = uint32_t(qs[0]) | (uint32_t(qs[1]) << 16);
    out.col = b * QK4_NL + j0;
    return out;
}

template <int NR, bool XS>
__device__ __forceinline__ void iq4_matvec(
    const uint8_t * packed, const float * input, float * output,
    int input_rows, int output_rows, int cols) {
    __shared__ float kv[16];
    if (threadIdx.x < 16) kv[threadIdx.x] = iq4_kvalues[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int out = blockIdx.x * IQ4_WARPS + warp;
    const int row0 = blockIdx.y * NR;
    if (out >= output_rows || row0 >= input_rows) return;

    const int blocks = XS ? cols / QK_K : cols / QK4_NL;
    const int groups = XS ? blocks : (blocks + 7) / 8;
    const size_t row_bytes = size_t(blocks) * (XS ? IQ4_XS_BLOCK_BYTES : IQ4_NL_BLOCK_BYTES);
    const uint8_t * row = packed + size_t(out) * row_bytes;

    float sums[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) sums[r] = 0.f;

#pragma unroll 2
    for (int g = 0; g < groups; ++g) {
        const iq4_lane_group lg = XS ? iq4_xs_lane(row, g, lane) : iq4_nl_lane(row, g, lane, blocks);
        if (!lg.valid) continue;
        float w_lo[4], w_hi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t byte = (lg.q >> (8 * k)) & 0xff;
            w_lo[k] = kv[byte & 0xf];
            w_hi[k] = kv[byte >> 4];
        }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            if (row0 + r < input_rows) {
                const float * x = input + size_t(row0 + r) * cols;
                const float4 lo = *reinterpret_cast<const float4 *>(x + lg.col);
                const float4 hi = *reinterpret_cast<const float4 *>(x + lg.col + 16);
                sums[r] += lg.dl * (w_lo[0] * lo.x + w_lo[1] * lo.y + w_lo[2] * lo.z + w_lo[3] * lo.w +
                                    w_hi[0] * hi.x + w_hi[1] * hi.y + w_hi[2] * hi.z + w_hi[3] * hi.w);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float s = iq4_warp_sum(sums[r]);
        if (lane == 0 && row0 + r < input_rows) output[size_t(row0 + r) * output_rows + out] = s;
    }
}

// ---------------------------------------------------------------------------
// Integer IQ4_XS matvec (the llama.cpp `vec_dot_iq4_xs_q8_1` approach).
//
// The activations are first quantized to int8 with one f32 scale per 32
// values (`iq4_quantize_q8_*`). The weight nibbles are expanded to their int8
// codebook values with `__byte_perm`, and each lane accumulates with `__dp4a`.
// Compared with a float dot product this cuts activation traffic ~4x (it was
// ~7x the weight bytes) and drops the per-weight shared-memory lookups.
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ T iq4_cast(float v);
template <> __device__ __forceinline__ float iq4_cast<float>(float v) { return v; }
template <> __device__ __forceinline__ half iq4_cast<half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 iq4_cast<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T> __device__ __forceinline__ float iq4_load_f32(const T * p);
template <> __device__ __forceinline__ float iq4_load_f32<float>(const float * p) { return *p; }
template <> __device__ __forceinline__ float iq4_load_f32<half>(const half * p) { return __half2float(*p); }
template <> __device__ __forceinline__ float iq4_load_f32<__nv_bfloat16>(const __nv_bfloat16 * p) {
    return __bfloat162float(*p);
}

// One warp per 32 activations: xq = round(x / d), d = max|x| / 127.
template <typename T>
__device__ __forceinline__ void iq4_quantize_q8(const T * x, int8_t * xq, float * xd, int n_blocks) {
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk >= n_blocks) return;
    const float v = iq4_load_f32<T>(x + size_t(blk) * 32 + lane);
    float amax = fabsf(v);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float d = amax / 127.f;
    xq[size_t(blk) * 32 + lane] = int8_t(amax == 0.f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

#define IQ4_QUANTIZE_Q8(name, T)                                                         \
    extern "C" __global__ void name(const T * x, int8_t * xq, float * xd, int n_blocks) {  \
        iq4_quantize_q8<T>(x, xq, xd, n_blocks);                                         \
    }

IQ4_QUANTIZE_Q8(iq4_quantize_q8_f32, float)
IQ4_QUANTIZE_Q8(iq4_quantize_q8_f16, half)
IQ4_QUANTIZE_Q8(iq4_quantize_q8_bf16, __nv_bfloat16)

// The IQ4 codebook as int8, packed four per word for `__byte_perm`.
#define IQ4_KV_0 0xBFAD9881u  // -127 -104  -83  -65
#define IQ4_KV_1 0xF6EADDCFu  //  -49  -35  -22  -10
#define IQ4_KV_2 0x26190D01u  //    1   13   25   38
#define IQ4_KV_3 0x71594535u  //   53   69   89  113

// Map four nibbles (one per byte of `n`, each 0..15) to four int8 codebook
// values packed in one word.
__device__ __forceinline__ int iq4_lookup4(uint32_t n) {
    const uint32_t low3 = n & 0x07070707u;
    const uint32_t sel = (low3 & 0xfu) | ((low3 >> 4) & 0xf0u) | ((low3 >> 8) & 0xf00u) | ((low3 >> 12) & 0xf000u);
    const uint32_t lo = __byte_perm(IQ4_KV_0, IQ4_KV_1, sel);
    const uint32_t hi = __byte_perm(IQ4_KV_2, IQ4_KV_3, sel);
    const uint32_t mask = ((n & 0x08080808u) >> 3) * 0xffu;
    return int((lo & ~mask) | (hi & mask));
}

#define IQ4_XS_UNROLL 4

// One lane's contribution for one block and NR activation rows.
template <int NR>
__device__ __forceinline__ void iq4_xs_q8_accumulate(
    uint2 head, uint32_t q, int ib, int col, const int8_t * xq, const float * xd,
    int row0, int input_rows, int cols, float * sums) {
    const uint16_t d_bits = uint16_t(head.x & 0xffff);
    const uint32_t scales_h = head.x >> 16;
    const uint32_t scales_l = (head.y >> (8 * (ib >> 1))) & 0xff;
    const int ls = ((scales_l >> (4 * (ib & 1))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = __half2float(*reinterpret_cast<const half *>(&d_bits)) * float(ls - 32);
    const int w_lo = iq4_lookup4(q & 0x0f0f0f0fu);
    const int w_hi = iq4_lookup4((q >> 4) & 0x0f0f0f0fu);
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        if (row0 + r < input_rows) {
            const int8_t * x = xq + size_t(row0 + r) * cols + col;
            const int sumi = __dp4a(w_hi, *reinterpret_cast<const int *>(x + 16),
                                    __dp4a(w_lo, *reinterpret_cast<const int *>(x), 0));
            sums[r] += dl * xd[(size_t(row0 + r) * cols + col) >> 5] * float(sumi);
        }
    }
}

// Lane l reads payload bytes 4l..4l+3 of each block (coalesced), and issues
// the loads for IQ4_XS_UNROLL blocks before using any, with a single 8-byte
// load for each block header, so wide rows are not latency-bound.
template <int NR, typename TO>
__device__ __forceinline__ void iq4_xs_matvec_q8(
    const uint8_t * packed, const int8_t * xq, const float * xd, TO * output,
    int input_rows, int output_rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int out = blockIdx.x * IQ4_WARPS + warp;
    const int row0 = blockIdx.y * NR;
    if (out >= output_rows || row0 >= input_rows) return;

    const int blocks = cols / QK_K;
    const uint8_t * row = packed + size_t(out) * blocks * IQ4_XS_BLOCK_BYTES;
    const int ib = lane >> 2;
    const int j0 = (lane & 3) * 4;
    const int qs_off = 8 + 16 * ib + j0;
    const int col_off = 32 * ib + j0;

    float sums[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) sums[r] = 0.f;

    int b = 0;
    for (; b + IQ4_XS_UNROLL <= blocks; b += IQ4_XS_UNROLL) {
        uint2 head[IQ4_XS_UNROLL];
        uint32_t q[IQ4_XS_UNROLL];
#pragma unroll
        for (int u = 0; u < IQ4_XS_UNROLL; ++u) {
            const uint8_t * blk = row + size_t(b + u) * IQ4_XS_BLOCK_BYTES;
            head[u] = *reinterpret_cast<const uint2 *>(blk);
            q[u] = *reinterpret_cast<const uint32_t *>(blk + qs_off);
        }
#pragma unroll
        for (int u = 0; u < IQ4_XS_UNROLL; ++u) {
            iq4_xs_q8_accumulate<NR>(head[u], q[u], ib, (b + u) * QK_K + col_off, xq, xd,
                                     row0, input_rows, cols, sums);
        }
    }
    for (; b < blocks; ++b) {
        const uint8_t * blk = row + size_t(b) * IQ4_XS_BLOCK_BYTES;
        iq4_xs_q8_accumulate<NR>(*reinterpret_cast<const uint2 *>(blk),
                                 *reinterpret_cast<const uint32_t *>(blk + qs_off), ib,
                                 b * QK_K + col_off, xq, xd, row0, input_rows, cols, sums);
    }

#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float s = iq4_warp_sum(sums[r]);
        if (lane == 0 && row0 + r < input_rows) output[size_t(row0 + r) * output_rows + out] = iq4_cast<TO>(s);
    }
}

// Output is written directly in the caller's dtype to save a cast launch.
#define IQ4_XS_MATVEC_Q8(name, nr, TO)                                                  \
    extern "C" __global__ void name(                                                    \
        const uint8_t * packed, const int8_t * xq, const float * xd, TO * output,       \
        int input_rows, int output_rows, int cols) {                                    \
        iq4_xs_matvec_q8<nr, TO>(packed, xq, xd, output, input_rows, output_rows, cols);\
    }

IQ4_XS_MATVEC_Q8(iq4_xs_matvec_q8_1_f32, 1, float)
IQ4_XS_MATVEC_Q8(iq4_xs_matvec_q8_4_f32, 4, float)
IQ4_XS_MATVEC_Q8(iq4_xs_matvec_q8_1_f16, 1, half)
IQ4_XS_MATVEC_Q8(iq4_xs_matvec_q8_4_f16, 4, half)
IQ4_XS_MATVEC_Q8(iq4_xs_matvec_q8_1_bf16, 1, __nv_bfloat16)
IQ4_XS_MATVEC_Q8(iq4_xs_matvec_q8_4_bf16, 4, __nv_bfloat16)

#define IQ4_MATVEC(name, nr, xs)                                                        \
    extern "C" __global__ void name(                                                    \
        const uint8_t * packed, const float * input, float * output,                    \
        int input_rows, int output_rows, int cols) {                                    \
        iq4_matvec<nr, xs>(packed, input, output, input_rows, output_rows, cols);       \
    }

IQ4_MATVEC(iq4_nl_matvec1_f32, 1, false)
IQ4_MATVEC(iq4_nl_matvec4_f32, 4, false)


template <typename T> __device__ __forceinline__ void iq4_store4(T * p, const float * v);
template <> __device__ __forceinline__ void iq4_store4<float>(float * p, const float * v) {
    *reinterpret_cast<float4 *>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
template <> __device__ __forceinline__ void iq4_store4<half>(half * p, const float * v) {
    const __half2 a = __floats2half2_rn(v[0], v[1]);
    const __half2 b = __floats2half2_rn(v[2], v[3]);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t *>(&a);
    packed.y = *reinterpret_cast<const uint32_t *>(&b);
    *reinterpret_cast<uint2 *>(p) = packed;
}
template <> __device__ __forceinline__ void iq4_store4<__nv_bfloat16>(__nv_bfloat16 * p, const float * v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]);
    const __nv_bfloat162 b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t *>(&a);
    packed.y = *reinterpret_cast<const uint32_t *>(&b);
    *reinterpret_cast<uint2 *>(p) = packed;
}

// One warp per 256 weights (an IQ4_XS block, or eight IQ4_NL blocks) with the
// matvec lane mapping, so both the packed reads and the 4-wide output stores
// are coalesced. `n_blocks` counts 256-value blocks (IQ4_XS) or 32-value
// blocks (IQ4_NL) from `packed`, which points at the first requested row.
template <typename T, bool XS>
__device__ __forceinline__ void iq4_dequant(const uint8_t * packed, T * output, int n_blocks) {
    const int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (XS && g >= n_blocks) return;
    const iq4_lane_group lg = XS ? iq4_xs_lane(packed, g, lane) : iq4_nl_lane(packed, g, lane, n_blocks);
    if (!lg.valid) return;
    float lo[4], hi[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t byte = (lg.q >> (8 * k)) & 0xff;
        lo[k] = lg.dl * iq4_kvalues[byte & 0xf];
        hi[k] = lg.dl * iq4_kvalues[byte >> 4];
    }
    iq4_store4<T>(output + lg.col, lo);
    iq4_store4<T>(output + lg.col + 16, hi);
}

#define IQ4_DEQUANT(name, T, xs)                                                        \
    extern "C" __global__ void name(const uint8_t * packed, T * output, int n_blocks) { \
        iq4_dequant<T, xs>(packed, output, n_blocks);                                   \
    }

IQ4_DEQUANT(iq4_xs_dequant_f32, float, true)
IQ4_DEQUANT(iq4_xs_dequant_f16, half, true)
IQ4_DEQUANT(iq4_xs_dequant_bf16, __nv_bfloat16, true)
IQ4_DEQUANT(iq4_nl_dequant_f32, float, false)
IQ4_DEQUANT(iq4_nl_dequant_f16, half, false)
IQ4_DEQUANT(iq4_nl_dequant_bf16, __nv_bfloat16, false)
