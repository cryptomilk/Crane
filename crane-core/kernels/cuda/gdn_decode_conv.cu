// Single-token GDN convolution + SiLU. The BF16 kernel preserves the reference's
// BF16 rounding; the F32 kernel (GGUF decode) rounds nothing through BF16.

#include <math.h>

#include "gdn_act.cuh"

template <typename T>
__device__ void gdn_decode_conv_silu(
    const T *input,       // [channels]: new token's values
    const T *state,       // [channels, 4]: cached history
    const float *weights, // [channels, 4]: preconverted weights
    T *output,            // [channels]: convolution + SiLU
    T *next_state,        // [channels, 4]: updated history
    int channels)
{
    // Give each GPU thread one channel.
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels)
    {
        return;
    }

    // Drop the oldest cached value and append the new token.
    const int base = channel * 4;
    const float x0 = to_f(state[base + 1]);
    const float x1 = to_f(state[base + 2]);
    const float x2 = to_f(state[base + 3]);
    const float x3 = to_f(input[channel]);

    const float p0 = __fmul_rn(x0, weights[base + 0]);
    const float p1 = __fmul_rn(x1, weights[base + 1]);
    const float p2 = __fmul_rn(x2, weights[base + 2]);
    const float p3 = __fmul_rn(x3, weights[base + 3]);

    const float sum02 = __fadd_rn(p0, p2);
    const float sum13 = __fadd_rn(p1, p3);

    const float sum = __fadd_rn(sum02, sum13);

    // Round like the activation dtype before the FP32 SiLU (a no-op for F32).
    const float value = to_f(from_f<T>(sum));
    const float denom = 1 + expf(-value);

    const float silu = __fdiv_rn(value, denom);

    output[channel] = from_f<T>(silu);

    // Carry the window forward without changing its representation.
    next_state[base + 0] = state[base + 1];
    next_state[base + 1] = state[base + 2];
    next_state[base + 2] = state[base + 3];
    next_state[base + 3] = input[channel];
}

#define DECODE_CONV_KERNEL(T, NAME)                                                      \
    extern "C" __global__ void gdn_decode_conv_silu_##NAME(                              \
        const T *input, const T *state, const float *weights, T *output, T *next_state, \
        int channels)                                                                    \
    {                                                                                    \
        gdn_decode_conv_silu<T>(input, state, weights, output, next_state, channels);    \
    }

DECODE_CONV_KERNEL(float, f32)
DECODE_CONV_KERNEL(__nv_bfloat16, bf16)
