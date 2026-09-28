// llama.cpp IQ4_XS / IQ4_NL weight kernels for the Metal backend — the
// counterpart of `kernels/sycl/quant_iq4.cpp` (see
// `crane-core/src/quantized/iquant.rs` for the block layouts). Compiled to an
// `MTLLibrary` at runtime by `ops/quant_iq/metal.rs` via
// `Device::new_library_with_source` (no `.metallib` to ship), and dispatched
// through candle's `CommandsGuard` like the other native kernels.
//
// Activations are not quantized to int8 for a `dp4a`-style dot product (Metal
// has no portable `simd_sum`-friendly int4 dot across the whole family) —
// both entry points here decode weights on the fly and accumulate a plain
// float dot product, one thread per (output row, input row) pair. F32, F16 and
// (when `__HAVE_BFLOAT__` is defined, i.e. Metal 3.0+) BF16 outputs are
// supported.

#include <metal_stdlib>
using namespace metal;

constant int QK_K    = 256;
constant int QK4_NL  = 32;
constant int IQ4_XS_BLOCK_BYTES = 2 + 2 + QK_K / 64 + QK_K / 2;  // 136
constant int IQ4_NL_BLOCK_BYTES = 2 + QK4_NL / 2;                //  18

// IQ4_NL / IQ4_XS codebook (mirrors `KVALUES_IQ4NL` in iquant.rs).
constant float KVALUES[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f,
};

// Decode a little-endian f16 (2 bytes, no alignment assumed).
inline float load_f16(const device uchar *p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    half h = as_type<half>(bits);
    return float(h);
}

// Decode IQ4_XS super-block `blk` (136 bytes, 256 values) into `out`.
inline void iq4_xs_decode(const device uchar *blk, thread float *out) {
    const float d = load_f16(blk);
    const ushort scales_h = (ushort)blk[2] | ((ushort)blk[3] << 8);
    const device uchar *scales_l = blk + 4;
    const device uchar *qs       = blk + 4 + QK_K / 64;
    for (int sb = 0; sb < QK_K / 32; ++sb) {
        const int   lo = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0xf;
        const int   hi = (scales_h >> (2 * sb)) & 3;
        const float dl = d * float((lo | (hi << 4)) - 32);
        const device uchar *q = qs + 16 * sb;
        thread float *y = out + 32 * sb;
        for (int j = 0; j < 16; ++j) {
            y[j]     = dl * KVALUES[q[j] & 0xf];
            y[j+16]  = dl * KVALUES[q[j] >> 4];
        }
    }
}

// Decode IQ4_NL block `blk` (18 bytes, 32 values) into `out`.
inline void iq4_nl_decode(const device uchar *blk, thread float *out) {
    const float d = load_f16(blk);
    const device uchar *qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        out[j]    = d * KVALUES[qs[j] & 0xf];
        out[j+16] = d * KVALUES[qs[j] >> 4];
    }
}

// Full row dot product against `x` (`cols` f32 activations), decoding blocks
// on the fly. Used by the decode-sized matvec kernel.
inline float iq4_row_dot(const device uchar *row, const device float *x,
                         int cols, bool xs) {
    const int bs = xs ? QK_K   : QK4_NL;
    const int bb = xs ? IQ4_XS_BLOCK_BYTES : IQ4_NL_BLOCK_BYTES;
    const int n_blocks = cols / bs;
    float w[QK_K];
    float dot = 0.f;
    for (int ib = 0; ib < n_blocks; ++ib) {
        const device uchar *blk = row + size_t(ib) * bb;
        if (xs) iq4_xs_decode(blk, w);
        else    iq4_nl_decode(blk, w);
        const device float *xb = x + ib * bs;
        for (int j = 0; j < bs; ++j) {
            dot += w[j] * xb[j];
        }
    }
    return dot;
}

// `xs` is a specialization via the kernel name suffix (0 = IQ4_NL, 1 = IQ4_XS):
// we can't pass it as a buffer, so the two flavors are separate kernels.

inline int  xs_block_size()  { return QK_K; }
inline int  nl_block_size()  { return QK4_NL; }
inline int  xs_block_bytes() { return IQ4_XS_BLOCK_BYTES; }
inline int  nl_block_bytes() { return IQ4_NL_BLOCK_BYTES; }
inline bool xs_is_xs()       { return true; }
inline bool nl_is_xs()       { return false; }

// Templated matvec. Each (out_row, in_row) thread computes one full dot
// product and writes one output element.
#define IQ4_MATVEC(T, TAG, XSTAG)                                   \
kernel void matvec_##TAG##_##XSTAG(                                 \
    const device uchar  *packed   [[ buffer(0) ]],                  \
    const device float  *input    [[ buffer(1) ]],                  \
    device T            *output   [[ buffer(2) ]],                  \
    constant int        &input_rows                               \
                        [[ buffer(3) ]],                            \
    constant int        &output_rows                               \
                        [[ buffer(4) ]],                            \
    constant int        &cols                                      \
                        [[ buffer(5) ]],                            \
    uint2                gid      [[ thread_position_in_grid ]]) { \
    const int out_row = int(gid.x);                                 \
    const int in_row  = int(gid.y);                                 \
    if (out_row >= output_rows || in_row >= input_rows) return;     \
    const size_t row_bytes = size_t(cols /                          \
        (XSTAG##_block_size())) * (XSTAG##_block_bytes());          \
    const device uchar *row = packed + size_t(out_row) * row_bytes; \
    const device float *x = input + size_t(in_row) * cols;          \
    const float dot = iq4_row_dot(row, x, cols, XSTAG##_is_xs());   \
    output[size_t(in_row) * output_rows + out_row] = T(dot);        \
}

IQ4_MATVEC(float, f32, xs)
IQ4_MATVEC(float, f32, nl)
IQ4_MATVEC(half,  f16, xs)
IQ4_MATVEC(half,  f16, nl)
#if defined(__HAVE_BFLOAT__)
IQ4_MATVEC(bfloat, bf16, xs)
IQ4_MATVEC(bfloat, bf16, nl)
#endif

// Dequantize: one thread per (row, block) pair, writes `block_size` values.
#define IQ4_DEQUANT(T, TAG, XSTAG)                                          \
kernel void dequant_##TAG##_##XSTAG(                                        \
    const device uchar *packed  [[ buffer(0) ]],                            \
    device T           *output  [[ buffer(1) ]],                            \
    constant int       &n_rows                                              \
                       [[ buffer(2) ]],                                      \
    constant int       &cols                                                \
                       [[ buffer(3) ]],                                      \
    uint2               gid      [[ thread_position_in_grid ]]) {           \
    const int row = int(gid.x);                                             \
    const int ib  = int(gid.y);                                             \
    const int bs = XSTAG##_block_size();                                    \
    const int bb = XSTAG##_block_bytes();                                   \
    const int n_blocks = cols / bs;                                         \
    if (row >= n_rows || ib >= n_blocks) return;                            \
    const size_t row_bytes = size_t(n_blocks) * bb;                         \
    const device uchar *blk = packed + size_t(row) * row_bytes              \
                                    + size_t(ib)  * bb;                     \
    float w[QK_K];                                                          \
    if (XSTAG##_is_xs()) iq4_xs_decode(blk, w);                             \
    else                iq4_nl_decode(blk, w);                              \
    device T *out = output + size_t(row) * cols + size_t(ib) * bs;          \
    for (int j = 0; j < bs; ++j) {                                          \
        out[j] = T(w[j]);                                                   \
    }                                                                       \
}

IQ4_DEQUANT(float, f32, xs)
IQ4_DEQUANT(float, f32, nl)
IQ4_DEQUANT(half,  f16, xs)
IQ4_DEQUANT(half,  f16, nl)
#if defined(__HAVE_BFLOAT__)
IQ4_DEQUANT(bfloat, bf16, xs)
IQ4_DEQUANT(bfloat, bf16, nl)
#endif
