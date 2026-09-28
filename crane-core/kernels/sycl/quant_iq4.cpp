// llama.cpp IQ4_XS / IQ4_NL weight kernels for the Intel SYCL backend — the
// counterpart of `kernels/cuda/quant_iq4.cu` (see
// `crane-core/src/quantized/iquant.rs` for the block layouts). Built into
// `libcrane_gdn_sycl.so` by `crane-core/build.rs` (icpx, `--features sycl`
// only) and driven by `ops/quant_iq/sycl.rs`.
//
// Unlike the CUDA kernel, activations are not quantized to int8 for a `dp4a`
// dot product (no portable SYCL equivalent across Intel GPU generations) —
// both entry points here decode weights on the fly and accumulate a plain
// float dot product, one work-item per (output row, block) or (output row,
// input row) pair. Only F32 and F16 outputs are supported, matching the rest
// of the SYCL fused-op surface (`fused_ops.cpp`, `gdn.cpp`).
#include <sycl/sycl.hpp>

namespace {

constexpr int QK_K = 256;
constexpr int QK4_NL = 32;
constexpr int IQ4_XS_BLOCK_BYTES = 2 + 2 + QK_K / 64 + QK_K / 2; // 136
constexpr int IQ4_NL_BLOCK_BYTES = 2 + QK4_NL / 2;               // 18

// dtype tags match `ops/quant_iq/sycl.rs`.
enum { CRANE_IQ4_F32 = 0, CRANE_IQ4_F16 = 1 };

inline float iq4_kval(int nibble) {
  constexpr float kv[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f,
                             -22.f,  -10.f,  1.f,   13.f,  25.f,  38.f,
                             53.f,   69.f,   89.f,  113.f};
  return kv[nibble];
}

// `p` points at a little-endian f16 (2 bytes); no alignment assumed.
inline float iq4_half(const uint8_t *p) {
  sycl::half h;
  auto *hp = reinterpret_cast<uint8_t *>(&h);
  hp[0] = p[0];
  hp[1] = p[1];
  return static_cast<float>(h);
}

// Decodes IQ4_XS super-block `blk` (136 bytes, 256 values) into `out`.
inline void iq4_xs_decode_block(const uint8_t *blk, float *out) {
  const float d = iq4_half(blk);
  const uint16_t scales_h = uint16_t(blk[2]) | (uint16_t(blk[3]) << 8);
  const uint8_t *scales_l = blk + 4;
  const uint8_t *qs = blk + 4 + QK_K / 64;
  for (int sb = 0; sb < QK_K / 32; ++sb) {
    const int lo = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0xf;
    const int hi = (scales_h >> (2 * sb)) & 3;
    const float dl = d * float((lo | (hi << 4)) - 32);
    const uint8_t *q = qs + 16 * sb;
    float *y = out + 32 * sb;
    for (int j = 0; j < 16; ++j) {
      y[j] = dl * iq4_kval(q[j] & 0xf);
      y[j + 16] = dl * iq4_kval(q[j] >> 4);
    }
  }
}

// Decodes IQ4_NL block `blk` (18 bytes, 32 values) into `out`.
inline void iq4_nl_decode_block(const uint8_t *blk, float *out) {
  const float d = iq4_half(blk);
  const uint8_t *qs = blk + 2;
  for (int j = 0; j < 16; ++j) {
    out[j] = d * iq4_kval(qs[j] & 0xf);
    out[j + 16] = d * iq4_kval(qs[j] >> 4);
  }
}

inline int iq4_block_size(bool xs) { return xs ? QK_K : QK4_NL; }
inline int iq4_block_bytes(bool xs) { return xs ? IQ4_XS_BLOCK_BYTES : IQ4_NL_BLOCK_BYTES; }

// Full row dot product against `x` (`cols` f32 activations), decoding blocks
// on the fly. Used by the decode-sized matvec kernel.
inline float iq4_row_dot(const uint8_t *row, const float *x, int cols, bool xs) {
  const int bs = iq4_block_size(xs);
  const int bb = iq4_block_bytes(xs);
  const int n_blocks = cols / bs;
  float w[QK_K];
  float dot = 0.f;
  for (int ib = 0; ib < n_blocks; ++ib) {
    const uint8_t *blk = row + size_t(ib) * bb;
    if (xs) {
      iq4_xs_decode_block(blk, w);
    } else {
      iq4_nl_decode_block(blk, w);
    }
    const float *xb = x + ib * bs;
    for (int j = 0; j < bs; ++j) {
      dot += w[j] * xb[j];
    }
  }
  return dot;
}

template <typename T>
void iq4_matvec_launch(sycl::queue &q, const uint8_t *packed, const float *input,
                        T *output, int input_rows, int output_rows, int cols,
                        bool xs) {
  const size_t row_bytes = size_t(cols / iq4_block_size(xs)) * iq4_block_bytes(xs);
  q.parallel_for(sycl::range<2>(size_t(output_rows), size_t(input_rows)),
                 [=](sycl::id<2> idx) {
                   const int out_row = static_cast<int>(idx[0]);
                   const int in_row = static_cast<int>(idx[1]);
                   const uint8_t *row = packed + size_t(out_row) * row_bytes;
                   const float *x = input + size_t(in_row) * cols;
                   const float dot = iq4_row_dot(row, x, cols, xs);
                   output[size_t(in_row) * output_rows + out_row] =
                       static_cast<T>(dot);
                 });
}

template <typename T>
void iq4_dequant_launch(sycl::queue &q, const uint8_t *packed, T *output,
                        int n_rows, int cols, bool xs) {
  const int bs = iq4_block_size(xs);
  const int bb = iq4_block_bytes(xs);
  const int n_blocks = cols / bs;
  const size_t row_bytes = size_t(n_blocks) * bb;
  q.parallel_for(sycl::range<2>(size_t(n_rows), size_t(n_blocks)),
                 [=](sycl::id<2> idx) {
                   const int row = static_cast<int>(idx[0]);
                   const int ib = static_cast<int>(idx[1]);
                   const uint8_t *blk = packed + row * row_bytes + size_t(ib) * bb;
                   float w[QK_K];
                   if (xs) {
                     iq4_xs_decode_block(blk, w);
                   } else {
                     iq4_nl_decode_block(blk, w);
                   }
                   T *out = output + size_t(row) * cols + size_t(ib) * bs;
                   for (int j = 0; j < bs; ++j) {
                     out[j] = static_cast<T>(w[j]);
                   }
                 });
}

} // namespace

extern "C" int crane_iq4_matvec_sycl(void *queue, int is_xs, int dtype,
                                     const void *packed, const void *input,
                                     void *output, int input_rows,
                                     int output_rows, int cols) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    const auto *p = static_cast<const uint8_t *>(packed);
    const auto *x = static_cast<const float *>(input);
    switch (dtype) {
    case CRANE_IQ4_F32:
      iq4_matvec_launch<float>(sq, p, x, static_cast<float *>(output),
                               input_rows, output_rows, cols, is_xs != 0);
      return 0;
    case CRANE_IQ4_F16:
      iq4_matvec_launch<sycl::half>(sq, p, x, static_cast<sycl::half *>(output),
                                    input_rows, output_rows, cols, is_xs != 0);
      return 0;
    default:
      return 2; // unsupported dtype
    }
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}

extern "C" int crane_iq4_dequant_sycl(void *queue, int is_xs, int dtype,
                                      const void *packed, void *output,
                                      int n_rows, int cols) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    const auto *p = static_cast<const uint8_t *>(packed);
    switch (dtype) {
    case CRANE_IQ4_F32:
      iq4_dequant_launch<float>(sq, p, static_cast<float *>(output), n_rows,
                                cols, is_xs != 0);
      return 0;
    case CRANE_IQ4_F16:
      iq4_dequant_launch<sycl::half>(sq, p, static_cast<sycl::half *>(output),
                                     n_rows, cols, is_xs != 0);
      return 0;
    default:
      return 2; // unsupported dtype
    }
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}
