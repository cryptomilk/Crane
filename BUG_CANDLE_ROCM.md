# Bug: `candle-rocm-kernels` HIP shim conflicts with ROCm 7.14's HIP headers

Affects: [huggingface/candle#3801](https://github.com/huggingface/candle/pull/3801)
("Feat/rocm backend"), branch `feat/rocm-backend` of `xmiksay/candle`, commit
`fe15b633887106b3ef08d2e464ddc7bb15e0a82d`.

## Summary

JIT-compiling the `quantized` kernel module (via `candle_core::rocm_backend`'s
`hipcc`-at-runtime pipeline) fails on ROCm 7.14.0 with:

```
error: redefinition of 'atomicAdd'
```

for both `__half` and `__hip_bfloat16`.

## Environment

- GPU: AMD Radeon RX 9070 XT (`gfx1201`, RDNA4)
- ROCm/HIP: 7.14.0, `HIP_VERSION` `7.14.60850` (image:
  `rocm/dev-ubuntu-24.04:7.14.0-full`)
- Consumer: [Crane](https://github.com/lucasjinreal/Crane) (`crane-serve`),
  built with `--features rocm` against the pinned commit above, running
  Qwen 3.5 9B (GGUF, `Q5_K_M`)

## Root cause

`candle-rocm-kernels/src/hip_shim/hip_compat.h`'s `CANDLE_HIP_ATOMIC_ADD_16`
macro is built on the documented assumption:

> HIP has atomicAdd for float/double/int, but not for the 16-bit float types.

That was true against whatever ROCm version this crate was developed and
tested against (ROCm 7.2.4 / `gfx1101`, per the crate's own README), but is
no longer true on ROCm 7.14: `/opt/rocm/include/hip/amd_detail/amd_hip_fp16.h`
and `amd_hip_bf16.h` **already define `atomicAdd`** for `__half` and
`__hip_bfloat16` themselves:

```cpp
// amd_hip_fp16.h:868-869
#if defined(__clang__) && defined(__HIP__)
inline __device__ __half atomicAdd(__half* const address, const __half value) {
```

```cpp
// amd_hip_bf16.h:1902
inline __device__ __hip_bfloat16 atomicAdd(__hip_bfloat16* address, __hip_bfloat16 value) {
```

The guard on the `__half` overload — `#if defined(__clang__) && defined(__HIP__)`
— is a **compiler-family check, not a ROCm-version/feature gate**: it is
unconditionally true for however `candle-rocm-kernels` itself compiles
(`hipcc`, i.e. clang, targeting HIP), so the shim's own definitions collide
with AMD's whenever both are visible in the same translation unit — which
they always are, since the shim's `hip_compat.h` is force-included ahead of
every kernel source and itself includes `hip/hip_fp16.h`/`hip/hip_bf16.h`
first.

There is no HIP feature-test macro available to detect this cleanly from the
shim's side (the guard AMD uses is about the compiler, not about whether the
function already exists). A `HIP_VERSION` threshold check in the shim is
probably the least-bad fix, but the exact ROCm release where AMD added these
overloads is unknown — only two data points are available: needed on ROCm
7.2.4 (`gfx1101`), conflicts on ROCm 7.14.60850 (`gfx1201`).

## Full compile error

```
Error: Kernel compilation failed: hipcc failed for `quantized` (gfx1201):
In file included from <built-in>:2:
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:46:1: warning: attribute declaration must precede definition [-Wignored-attributes]
   46 | CANDLE_HIP_ATOMIC_ADD_16(__half, __half_as_ushort, __ushort_as_half)
      | ^
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:26:16: note: expanded from macro 'CANDLE_HIP_ATOMIC_ADD_16'
   26 |     __device__ __forceinline__ TYPE atomicAdd(TYPE *address, TYPE val) {       \
      |                ^
/opt/rocm/include/hip/amd_detail/host_defines.h:336:47: note: expanded from macro '__forceinline__'
  336 | #define __forceinline__ inline __attribute__((always_inline))
      |                                               ^
/opt/rocm/include/hip/amd_detail/amd_hip_fp16.h:869:26: note: previous definition is here
  869 | inline __device__ __half atomicAdd(__half* const address, const __half value) {
      |                          ^
In file included from <built-in>:2:
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:46:1: error: redefinition of 'atomicAdd'
   46 | CANDLE_HIP_ATOMIC_ADD_16(__half, __half_as_ushort, __ushort_as_half)
      | ^
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:26:37: note: expanded from macro 'CANDLE_HIP_ATOMIC_ADD_16'
   26 |     __device__ __forceinline__ TYPE atomicAdd(TYPE *address, TYPE val) {       \
      |                                     ^
/opt/rocm/include/hip/amd_detail/amd_hip_fp16.h:869:26: note: previous definition is here
  869 | inline __device__ __half atomicAdd(__half* const address, const __half value) {
      |                          ^
In file included from <built-in>:2:
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:47:1: warning: attribute declaration must precede definition [-Wignored-attributes]
   47 | CANDLE_HIP_ATOMIC_ADD_16(__hip_bfloat16, __bfloat16_as_ushort, __ushort_as_bfloat16)
      | ^
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:26:16: note: expanded from macro 'CANDLE_HIP_ATOMIC_ADD_16'
   26 |     __device__ __forceinline__ TYPE atomicAdd(TYPE *address, TYPE val) {       \
      |                ^
/opt/rocm/include/hip/amd_detail/host_defines.h:336:47: note: expanded from macro '__forceinline__'
  336 | #define __forceinline__ inline __attribute__((always_inline))
      |                                               ^
/opt/rocm/include/hip/amd_detail/amd_hip_bf16.h:1902:34: note: previous definition is here
 1902 | inline __device__ __hip_bfloat16 atomicAdd(__hip_bfloat16* address, __hip_bfloat16 value) {
      |                                  ^
In file included from <built-in>:2:
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:47:1: error: redefinition of 'atomicAdd'
   47 | CANDLE_HIP_ATOMIC_ADD_16(__hip_bfloat16, __bfloat16_as_ushort, __ushort_as_bfloat16)
      | ^
/cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h:26:37: note: expanded from macro 'CANDLE_HIP_ATOMIC_ADD_16'
   26 |     __device__ __forceinline__ TYPE atomicAdd(TYPE *address, TYPE val) {       \
      |                                     ^
/opt/rocm/include/hip/amd_detail/amd_hip_bf16.h:1902:34: note: previous definition is here
 1902 | inline __device__ __hip_bfloat16 atomicAdd(__hip_bfloat16* address, __hip_bfloat16 value) {
      |                                  ^
2 warnings and 2 errors generated when compiling for gfx1201.
failed to execute:/opt/rocm/lib/llvm/bin/clang++  --offload-arch=gfx1201  --cuda-device-only -O3 -std=c++17 -D__CUDA_ARCH__=890 -DRDNA3 -include /cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim/hip_compat.h -I /cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/hip_shim -I /cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9 -o "/cache/candle-rocm/gfx1201-7.14/quantized_485fa0ba843694ca.1.bundle" -x hip /cache/candle-rocm/gfx1201-7.14/src-cb33cedcb69151b9/quantized_485fa0ba843694ca.cu
```

## Relevant source

`candle-rocm-kernels/src/hip_shim/hip_compat.h` (as of the pinned commit):

```cpp
// HIP has atomicAdd for float/double/int, but not for the 16-bit float types.
// reduce.cu's SUM_OP instantiates sum_f16 and sum_bf16, both of which need one.
// Implemented with a 32-bit CAS over the containing aligned word, which is the
// same trick CUDA's own pre-sm_70 fallback uses.
#define CANDLE_HIP_ATOMIC_ADD_16(TYPE, TO_BITS, FROM_BITS)                     \
    __device__ __forceinline__ TYPE atomicAdd(TYPE *address, TYPE val) {       \
        ...
    }

CANDLE_HIP_ATOMIC_ADD_16(__half, __half_as_ushort, __ushort_as_half)
CANDLE_HIP_ATOMIC_ADD_16(__hip_bfloat16, __bfloat16_as_ushort, __ushort_as_bfloat16)

#undef CANDLE_HIP_ATOMIC_ADD_16
```

## Status

Not yet reported upstream. Local workaround/patch not yet applied either —
next step is deciding between:

1. Reporting this on PR #3801 for the branch owner to fix (unknown timeline).
2. A local patch on Crane's side: vendor a patched copy of
   `candle-rocm-kernels` and use a Cargo `[patch]` entry to substitute it in
   place of the one pulled transitively via the `candle-core` git dependency
   (`candle-rocm-kernels` is excluded from the upstream repo's own Cargo
   workspace, so it resolves as a path dependency of `candle-core` within
   the same git checkout — patching just this one file requires overriding
   the whole crate via `[patch]`, not a smaller surgical override).
