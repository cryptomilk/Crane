// SPDX-License-Identifier: MIT
//
// Hyper-connection mixer kernels (Qwen4-Exp) for CUDA, driven by
// `ops/hyper_connection.rs` -- the counterpart of
// `kernels/{sycl/hyper_connection.cpp,metal/hyper_connection.metal}`.
//
// The residual is `groups` parallel streams of `group` values per row
// (`[rows, groups * group]`). Each kernel replaces a chain of small candle ops
// that decode runs 97 times per token; all accumulate in f32. F32, F16 and
// BF16 tensors are supported.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define HC_NORM_THREADS 256

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// out = x / rms(x over each group) * alpha[g]; one block of HC_NORM_THREADS
// threads per (row, group).
template <typename T>
__device__ void hc_norm_impl(const T * x, const T * alpha, T * out, int groups, int group,
                             float eps) {
    __shared__ float partial[HC_NORM_THREADS / 32];
    const size_t rg = blockIdx.x;
    const int g = int(rg % size_t(groups));
    const T * xr = x + rg * group;
    float ss = 0.f;
    for (int j = threadIdx.x; j < group; j += HC_NORM_THREADS) {
        const float v = to_f(xr[j]);
        ss += v * v;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x >> 5] = ss;
    }
    __syncthreads();
    ss = 0.f;
#pragma unroll
    for (int i = 0; i < HC_NORM_THREADS / 32; ++i) {
        ss += partial[i];
    }
    const float inv = rsqrtf(ss / float(group) + eps);
    const T * a = alpha + size_t(g) * group;
    T * o = out + rg * group;
    for (int j = threadIdx.x; j < group; j += HC_NORM_THREADS) {
        o[j] = from_f<T>(to_f(xr[j]) * inv * to_f(a[j]));
    }
}

// out[i] = silu(low[i] * scale).
template <typename T>
__device__ void hc_low_impl(const T * low, T * out, size_t n, float scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = to_f(low[i]) * scale;
    out[i] = from_f<T>(v * sigmoidf(v));
}

// out[r, j] = mean over g of sigmoid(gate[r, g, j]) * normed[r, g, j];
// one thread per (r, j).
template <typename T>
__device__ void hc_mix_impl(const T * gate, const T * normed, T * out, size_t rows, int groups,
                            int group) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * group) {
        return;
    }
    const size_t r = idx / group;
    const size_t j = idx % group;
    const size_t base = r * size_t(groups) * group + j;
    float acc = 0.f;
    for (int g = 0; g < groups; ++g) {
        const size_t k = base + size_t(g) * group;
        acc += sigmoidf(to_f(gate[k])) * to_f(normed[k]);
    }
    out[idx] = from_f<T>(acc / float(groups));
}

// out[r, g, j] = streams[r, g, j] + block[r, j] * 2 * sigmoid(logits[r, g] * scale);
// one thread per output element.
template <typename T>
__device__ void hc_combine_impl(const T * streams, const T * block, const T * logits, T * out,
                                size_t rows, int groups, int group, float scale) {
    const size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= rows * groups * group) {
        return;
    }
    const size_t j = k % group;
    const size_t rg = k / group;  // r * groups + g
    const size_t r = rg / groups;
    const float w = 2.f * sigmoidf(to_f(logits[rg]) * scale);
    out[k] = from_f<T>(to_f(streams[k]) + to_f(block[r * group + j]) * w);
}

#define HC_KERNELS(T, NAME)                                                                   \
    extern "C" __global__ void hc_norm_##NAME(const T * x, const T * alpha, T * out,          \
                                              int groups, int group, float eps) {             \
        hc_norm_impl<T>(x, alpha, out, groups, group, eps);                                   \
    }                                                                                         \
    extern "C" __global__ void hc_low_##NAME(const T * low, T * out, size_t n, float scale) { \
        hc_low_impl<T>(low, out, n, scale);                                                   \
    }                                                                                         \
    extern "C" __global__ void hc_mix_##NAME(const T * gate, const T * normed, T * out,       \
                                             size_t rows, int groups, int group) {            \
        hc_mix_impl<T>(gate, normed, out, rows, groups, group);                               \
    }                                                                                         \
    extern "C" __global__ void hc_combine_##NAME(const T * streams, const T * block,          \
                                                 const T * logits, T * out, size_t rows,      \
                                                 int groups, int group, float scale) {        \
        hc_combine_impl<T>(streams, block, logits, out, rows, groups, group, scale);          \
    }

HC_KERNELS(float, f32)
HC_KERNELS(half, f16)
HC_KERNELS(__nv_bfloat16, bf16)
