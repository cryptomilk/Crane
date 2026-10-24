// Activation-type helpers shared by the GDN kernels (T = __nv_bfloat16 or float).
#pragma once

#include <cuda_bf16.h>

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}
