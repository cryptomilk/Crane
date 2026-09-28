// SPDX-License-Identifier: MIT

//! llama.cpp "i-quant" tensor types that Candle's `GgmlDType` does not know.
//!
//! imatrix quants (`IQ4_XS`, `IQ4_NL`, and the lower-bit IQ family used by
//! e.g. unsloth's dynamic quants) make Candle's GGUF parser fail with the
//! misleading `unknown dtype for tensor <ggml type id>`. The
//! [`extended_gguf`](super::extended_gguf) probe hides these tensors from
//! Candle; this module decodes them.
//!
//! On CUDA, linear layers keep the packed encoding and run through native
//! kernels ([`IQuantLinear`]). Everything else (other devices, embeddings,
//! packed MoE experts) is dequantized on the CPU and re-quantized at load
//! time to a Candle-native type (see [`requant_target`]), so every backend
//! can still run it through its existing `QMatMul` kernels. Reference: `ggml/src/ggml-quants.c` and
//! `ggml/src/ggml-common.h` in llama.cpp.

use std::borrow::Cow;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor, bail};
use half::f16;

/// Super-block size shared by the k-quants and `IQ4_XS`.
const QK_K: usize = 256;
/// Block size of `IQ4_NL`.
const QK4_NL: usize = 32;

/// The non-linear 4-bit codebook shared by `IQ4_NL` and `IQ4_XS`.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IQuantType {
    /// ggml type 20: 32-value blocks, `f16` scale + 16 bytes of 4-bit indices.
    Iq4Nl,
    /// ggml type 23: 256-value super-blocks, `f16` scale, eight 6-bit
    /// sub-block scales and 128 bytes of 4-bit indices.
    Iq4Xs,
}

impl IQuantType {
    /// Map a ggml type id to a decodable i-quant, if this module supports it.
    pub fn from_ggml_type_id(id: u32) -> Option<Self> {
        match id {
            20 => Some(Self::Iq4Nl),
            23 => Some(Self::Iq4Xs),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq4Xs => "IQ4_XS",
        }
    }

    /// Number of weights per block.
    pub fn block_size(self) -> usize {
        match self {
            Self::Iq4Nl => QK4_NL,
            Self::Iq4Xs => QK_K,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Iq4Nl => 2 + QK4_NL / 2,
            Self::Iq4Xs => 2 + 2 + QK_K / 64 + QK_K / 2,
        }
    }

    /// Decode `blocks` (a whole number of blocks) into `out`.
    pub fn dequantize(self, blocks: &[u8], out: &mut [f32]) {
        debug_assert_eq!(
            blocks.len() / self.block_bytes() * self.block_size(),
            out.len()
        );
        match self {
            Self::Iq4Nl => dequantize_iq4_nl(blocks, out),
            Self::Iq4Xs => dequantize_iq4_xs(blocks, out),
        }
    }
}

/// Human-readable name for ggml type ids Candle cannot parse, for error
/// messages. Mirrors `enum ggml_type` in `ggml.h`.
pub fn ggml_type_name(id: u32) -> Option<&'static str> {
    Some(match id {
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => return None,
    })
}

fn dequantize_iq4_nl(blocks: &[u8], out: &mut [f32]) {
    for (block, y) in blocks
        .chunks_exact(2 + QK4_NL / 2)
        .zip(out.chunks_exact_mut(QK4_NL))
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..];
        for j in 0..QK4_NL / 2 {
            y[j] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0xf)]);
            y[j + QK4_NL / 2] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)]);
        }
    }
}

fn dequantize_iq4_xs(blocks: &[u8], out: &mut [f32]) {
    let block_bytes = IQuantType::Iq4Xs.block_bytes();
    for (block, y) in blocks
        .chunks_exact(block_bytes)
        .zip(out.chunks_exact_mut(QK_K))
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..4 + QK_K / 64];
        let qs = &block[4 + QK_K / 64..];
        for ib in 0..QK_K / 32 {
            let lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
            let hi = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = i32::from(lo | (hi << 4));
            #[allow(clippy::cast_precision_loss)]
            let dl = d * (ls - 32) as f32;
            let q = &qs[16 * ib..16 * (ib + 1)];
            let y = &mut y[32 * ib..32 * (ib + 1)];
            for j in 0..16 {
                y[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[j] & 0xf)]);
                y[j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[j] >> 4)]);
            }
        }
    }
}

/// Whether linear layers keep i-quant weights packed and run them through
/// the native kernels (CUDA only). True unless `CRANE_IQ_REQUANT` names an
/// explicit target, which forces re-quantization everywhere (handy for A/B
/// comparisons).
pub fn native_enabled() -> bool {
    std::env::var("CRANE_IQ_REQUANT").map_or(true, |v| v.eq_ignore_ascii_case("native"))
}

/// Candle-native type to re-quantize i-quant tensors into when they are not
/// run natively: on non-CUDA devices, for embeddings and packed MoE experts,
/// or everywhere when `CRANE_IQ_REQUANT` is set.
///
/// `CRANE_IQ_REQUANT` picks it (`q4k`, `q5k`, `q6k`, `q8_0`); unset or
/// `native` means the default `q5k`, which keeps nearly all of the source
/// precision at ~30% more memory than `IQ4_XS`. `q4k` is about the same size
/// as the source but quantizes twice.
pub fn requant_target() -> Result<GgmlDType> {
    let Ok(value) = std::env::var("CRANE_IQ_REQUANT") else {
        return Ok(GgmlDType::Q5K);
    };
    Ok(match value.to_ascii_lowercase().as_str() {
        "native" | "q5k" | "q5_k" => GgmlDType::Q5K,
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q6k" | "q6_k" => GgmlDType::Q6K,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        other => bail!(
            "CRANE_IQ_REQUANT: unsupported target {other:?} (use native, q4k, q5k, q6k or q8_0)"
        ),
    })
}

/// Decode an i-quant tensor and re-quantize it to `target` on `device`.
///
/// `packed` holds the raw GGUF bytes of a tensor with `shape` (outermost
/// first). Rows are decoded and quantized in parallel on the CPU; only the
/// final re-quantized bytes are uploaded. Falls back to `Q8_0` when the row
/// width is not a multiple of `target`'s block size (e.g. `IQ4_NL` rows).
///
/// # Errors
///
/// Returns an error if `packed` does not match `shape` or quantization fails.
pub fn requantize(
    ty: IQuantType,
    packed: &[u8],
    shape: &[usize],
    target: GgmlDType,
    device: &Device,
) -> Result<QTensor> {
    let cols = *shape.last().unwrap_or(&0);
    let rows: usize = shape[..shape.len().saturating_sub(1)].iter().product();
    if cols == 0 || !cols.is_multiple_of(ty.block_size()) {
        bail!(
            "{} tensor row width {cols} is not a multiple of {}",
            ty.name(),
            ty.block_size()
        )
    }
    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    if packed.len() != rows * row_bytes {
        bail!(
            "{} tensor of shape {shape:?} should be {} bytes, got {}",
            ty.name(),
            rows * row_bytes,
            packed.len()
        )
    }
    let target = if cols.is_multiple_of(target.block_size()) {
        target
    } else {
        GgmlDType::Q8_0
    };

    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(rows.max(1));
    let rows_per_chunk = rows.div_ceil(threads);
    let chunks = std::thread::scope(|scope| -> Result<Vec<Vec<u8>>> {
        let handles: Vec<_> = packed
            .chunks(rows_per_chunk * row_bytes)
            .map(|chunk| {
                scope.spawn(move || -> Result<Vec<u8>> {
                    let chunk_rows = chunk.len() / row_bytes;
                    let mut values = vec![0f32; chunk_rows * cols];
                    ty.dequantize(chunk, &mut values);
                    let src = Tensor::from_vec(values, (chunk_rows, cols), &Device::Cpu)?;
                    let q = QTensor::quantize(&src, target)?;
                    Ok(q.data()?.into_owned())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().map_err(|_| {
                    candle_core::Error::Msg("i-quant requantize thread panicked".into())
                })?
            })
            .collect()
    })?;
    let bytes = chunks.concat();
    let storage = QStorage::from_data(Cow::Owned(bytes), device, target)?;
    QTensor::new(storage, shape.to_vec())
}

/// Largest input-row count served by the decode matvec kernel; bigger
/// batches dequantize weight chunks and use a regular matmul.
#[cfg_attr(not(any(feature = "cuda", feature = "sycl")), allow(dead_code))]
const MATVEC_MAX_ROWS: usize = 8;
/// Upper bound on the transient dense weight chunk built during prefill.
const PREFILL_CHUNK_BYTES: usize = 256 << 20;

/// A linear layer whose weight stays in its packed i-quant encoding on the
/// device (see `kernels/cuda/quant_iq4.cu`).
///
/// Only built for CUDA by [`Gguf`](super::gguf_file::Gguf); other devices
/// get a re-quantized `QMatMul` instead. The CPU path here dequantizes the
/// whole weight per call and exists for tests and device fallbacks.
#[derive(Clone, Debug)]
pub struct IQuantLinear {
    ty: IQuantType,
    /// Raw GGUF bytes, `[rows * row_bytes]` u8.
    packed: Tensor,
    rows: usize,
    cols: usize,
}

impl IQuantLinear {
    /// Wrap the raw GGUF bytes of a `[rows, cols]` weight.
    ///
    /// # Errors
    ///
    /// Returns an error if `packed` does not match the shape or the upload fails.
    pub fn new(
        ty: IQuantType,
        packed: Vec<u8>,
        rows: usize,
        cols: usize,
        device: &Device,
    ) -> Result<Self> {
        if cols == 0 || !cols.is_multiple_of(ty.block_size()) {
            bail!(
                "{} linear width {cols} is not a multiple of {}",
                ty.name(),
                ty.block_size()
            )
        }
        let expected = rows * (cols / ty.block_size()) * ty.block_bytes();
        if packed.len() != expected {
            bail!(
                "{} linear [{rows}, {cols}] should be {expected} bytes, got {}",
                ty.name(),
                packed.len()
            )
        }
        let len = packed.len();
        Ok(Self {
            ty,
            packed: Tensor::from_vec(packed, (len,), device)?,
            rows,
            cols,
        })
    }

    pub fn ty(&self) -> IQuantType {
        self.ty
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn device(&self) -> &Device {
        self.packed.device()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.packed.elem_count()
    }

    /// Copy the packed weight to `device`, keeping it encoded.
    ///
    /// # Errors
    ///
    /// Returns an error if the transfer fails.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            packed: self.packed.to_device(device)?,
            ..self.clone()
        })
    }

    /// The whole weight as a dense `[rows, cols]` tensor on its device.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.packed.device().is_cuda() {
            return crate::ops::quant_iq::cuda::dequantize(
                &self.packed,
                self.ty,
                0,
                self.rows,
                self.cols,
                dtype,
            );
        }
        #[cfg(feature = "sycl")]
        if self.packed.device().is_sycl() && matches!(dtype, DType::F32 | DType::F16) {
            return crate::ops::quant_iq::sycl::dequantize(
                &self.packed,
                self.ty,
                0,
                self.rows,
                self.cols,
                dtype,
            );
        }
        let bytes = self.packed.to_device(&Device::Cpu)?.to_vec1::<u8>()?;
        let mut values = vec![0f32; self.rows * self.cols];
        self.ty.dequantize(&bytes, &mut values);
        Tensor::from_vec(values, (self.rows, self.cols), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(self.packed.device())
    }

    /// Project `xs` (`[..., cols]`), returning `[..., rows]` in `xs`'s dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes disagree or a kernel fails.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_as(xs, xs.dtype())
    }

    /// Like [`Self::forward`] but always returns F32.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes disagree or a kernel fails.
    pub fn forward_f32(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_as(xs, DType::F32)
    }

    fn forward_as(&self, xs: &Tensor, out_dtype: DType) -> Result<Tensor> {
        self.forward_chunked(xs, out_dtype, PREFILL_CHUNK_BYTES)
    }

    fn forward_chunked(&self, xs: &Tensor, out_dtype: DType, chunk_bytes: usize) -> Result<Tensor> {
        let mut out_dims = xs.dims().to_vec();
        match out_dims.last_mut() {
            Some(last) if *last == self.cols => *last = self.rows,
            _ => bail!(
                "{} linear expects input width {}, got shape {:?}",
                self.ty.name(),
                self.cols,
                xs.dims()
            ),
        }
        let n = xs.elem_count() / self.cols;

        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() {
            use crate::ops::quant_iq::cuda;
            if n <= MATVEC_MAX_ROWS {
                let x = xs.reshape((n, self.cols))?;
                let y = cuda::matvec(&x, &self.packed, self.ty, self.rows, self.cols, out_dtype)?;
                return y.reshape(out_dims);
            }
            let x = xs
                .to_dtype(out_dtype)?
                .reshape((n, self.cols))?
                .contiguous()?;
            let chunk = (chunk_bytes / (self.cols * out_dtype.size_in_bytes())).max(1);
            let mut outs = Vec::with_capacity(self.rows.div_ceil(chunk));
            for start in (0..self.rows).step_by(chunk) {
                let n_rows = chunk.min(self.rows - start);
                let w =
                    cuda::dequantize(&self.packed, self.ty, start, n_rows, self.cols, out_dtype)?;
                outs.push(x.matmul(&w.t()?)?);
            }
            let y = if outs.len() == 1 {
                outs.pop().unwrap()
            } else {
                Tensor::cat(&outs, 1)?
            };
            return y.reshape(out_dims);
        }

        #[cfg(feature = "sycl")]
        if xs.device().is_sycl() && matches!(out_dtype, DType::F32 | DType::F16) {
            use crate::ops::quant_iq::sycl;
            if n <= MATVEC_MAX_ROWS {
                let x = xs.reshape((n, self.cols))?;
                let y = sycl::matvec(&x, &self.packed, self.ty, self.rows, self.cols, out_dtype)?;
                return y.reshape(out_dims);
            }
            let x = xs
                .to_dtype(out_dtype)?
                .reshape((n, self.cols))?
                .contiguous()?;
            let chunk = (chunk_bytes / (self.cols * out_dtype.size_in_bytes())).max(1);
            let mut outs = Vec::with_capacity(self.rows.div_ceil(chunk));
            for start in (0..self.rows).step_by(chunk) {
                let n_rows = chunk.min(self.rows - start);
                let w =
                    sycl::dequantize(&self.packed, self.ty, start, n_rows, self.cols, out_dtype)?;
                outs.push(x.matmul(&w.t()?)?);
            }
            let y = if outs.len() == 1 {
                outs.pop().unwrap()
            } else {
                Tensor::cat(&outs, 1)?
            };
            return y.reshape(out_dims);
        }

        let x = xs.to_dtype(DType::F32)?.reshape((n, self.cols))?;
        let w = self.dequantize(DType::F32)?;
        x.matmul(&w.t()?)?.to_dtype(out_dtype)?.reshape(out_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one `IQ4_XS` block from explicit fields.
    fn iq4_xs_block(d: f32, scales: [u8; 8], qs: [u8; 128]) -> Vec<u8> {
        let mut out = f16::from_f32(d).to_le_bytes().to_vec();
        let mut scales_h = 0u16;
        let mut scales_l = [0u8; 4];
        for (ib, &s) in scales.iter().enumerate() {
            scales_l[ib / 2] |= (s & 0xf) << (4 * (ib % 2));
            scales_h |= u16::from(s >> 4) << (2 * ib);
        }
        out.extend_from_slice(&scales_h.to_le_bytes());
        out.extend_from_slice(&scales_l);
        out.extend_from_slice(&qs);
        out
    }

    #[test]
    fn iq4_xs_decodes_scales_and_nibbles() {
        let scales = [0, 1, 31, 32, 33, 47, 62, 63];
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i % 16) as u8 | (((i + 5) % 16) as u8) << 4;
        }
        let block = iq4_xs_block(0.5, scales, qs);
        assert_eq!(block.len(), IQuantType::Iq4Xs.block_bytes());
        let mut out = [0f32; 256];
        IQuantType::Iq4Xs.dequantize(&block, &mut out);
        for ib in 0..8 {
            let dl = 0.5 * (f32::from(scales[ib]) - 32.0);
            for j in 0..16 {
                let q = qs[16 * ib + j];
                assert_eq!(
                    out[32 * ib + j],
                    dl * f32::from(KVALUES_IQ4NL[(q & 0xf) as usize])
                );
                assert_eq!(
                    out[32 * ib + j + 16],
                    dl * f32::from(KVALUES_IQ4NL[(q >> 4) as usize])
                );
            }
        }
    }

    #[test]
    fn iq4_nl_decodes_nibbles() {
        let mut block = f16::from_f32(2.0).to_le_bytes().to_vec();
        block.extend((0u8..16).map(|j| j | ((15 - j) << 4)));
        let mut out = [0f32; 32];
        IQuantType::Iq4Nl.dequantize(&block, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], 2.0 * f32::from(KVALUES_IQ4NL[j]));
            assert_eq!(out[j + 16], 2.0 * f32::from(KVALUES_IQ4NL[15 - j]));
        }
    }

    #[test]
    fn requantize_round_trips_close_to_source() -> Result<()> {
        let (rows, cols) = (8, 512);
        let mut packed = Vec::new();
        for r in 0..rows * cols / 256 {
            let scales = [(r % 64) as u8, 40, 20, 50, 33, 10, 60, 45];
            let qs: [u8; 128] = std::array::from_fn(|i| ((i * 7 + r) % 256) as u8);
            packed.extend(iq4_xs_block(0.01, scales, qs));
        }
        let mut reference = vec![0f32; rows * cols];
        IQuantType::Iq4Xs.dequantize(&packed, &mut reference);
        let q = requantize(
            IQuantType::Iq4Xs,
            &packed,
            &[rows, cols],
            GgmlDType::Q8_0,
            &Device::Cpu,
        )?;
        assert_eq!(q.dtype(), GgmlDType::Q8_0);
        let got = q
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_ref = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
        let max_err = reference
            .iter()
            .zip(&got)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_err <= max_ref / 100.0,
            "max_err {max_err} vs max_ref {max_ref}"
        );
        Ok(())
    }

    /// `n` pseudo-random blocks of `ty` with a sane `f16` scale.
    fn random_blocks(ty: IQuantType, n: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(n * ty.block_bytes());
        for i in 0..n {
            out.extend(f16::from_f32(0.002 + 0.0001 * (i % 7) as f32).to_le_bytes());
            for _ in 2..ty.block_bytes() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                out.push(state as u8);
            }
        }
        out
    }

    #[test]
    fn cpu_linear_matches_dequantized_matmul() -> Result<()> {
        let (rows, cols) = (5, 256);
        let packed = random_blocks(IQuantType::Iq4Xs, rows * cols / 256, 7);
        let mut w = vec![0f32; rows * cols];
        IQuantType::Iq4Xs.dequantize(&packed, &mut w);
        let layer = IQuantLinear::new(IQuantType::Iq4Xs, packed, rows, cols, &Device::Cpu)?;
        let x = Tensor::arange(0f32, (2 * cols) as f32, &Device::Cpu)?.reshape((2, cols))? / 100.0;
        let x = x?;
        let want = x.matmul(&Tensor::from_vec(w, (rows, cols), &Device::Cpu)?.t()?)?;
        let got = layer.forward(&x)?;
        let diff = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "diff {diff}");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_linear_matches_cpu_reference() -> Result<()> {
        if !candle_core::utils::cuda_is_available() {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        // IQ4_NL at 288 columns exercises a partial last 256-value group.
        for (ty, rows, cols) in [
            (IQuantType::Iq4Xs, 37, 512),
            (IQuantType::Iq4Nl, 37, 288),
            (IQuantType::Iq4Nl, 19, 512),
        ] {
            let packed = random_blocks(ty, rows * cols / ty.block_size(), rows as u32 * 31);
            let cpu = IQuantLinear::new(ty, packed.clone(), rows, cols, &Device::Cpu)?;
            let gpu = IQuantLinear::new(ty, packed, rows, cols, &cuda)?;
            for n in [1usize, 3, 4, 7, 9, 33] {
                let x = Tensor::randn(0f32, 1.0, (n, cols), &Device::Cpu)?;
                let want = cpu.forward(&x)?;
                let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
                let check = |got: Tensor, tol: f32, what: &str| -> Result<()> {
                    let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let diff = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(
                        diff / scale < tol,
                        "{} {what} n={n}: rel diff {}",
                        ty.name(),
                        diff / scale
                    );
                    Ok(())
                };
                let xg = x.to_device(&cuda)?;
                // IQ4_XS decode quantizes activations to int8 (as llama.cpp does).
                let f32_tol = if ty == IQuantType::Iq4Xs && n <= MATVEC_MAX_ROWS {
                    1e-2
                } else {
                    1e-4
                };
                check(gpu.forward(&xg)?, f32_tol, "f32")?;
                check(gpu.forward(&xg.to_dtype(DType::BF16)?)?, 2e-2, "bf16")?;
                // A tiny chunk forces the multi-chunk prefill path.
                check(
                    gpu.forward_chunked(&xg, DType::F32, 4 * cols * 4)?,
                    f32_tol,
                    "chunked",
                )?;
            }
            let dense = gpu.dequantize(DType::F32)?.to_device(&Device::Cpu)?;
            let diff = (dense - cpu.dequantize(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(diff, 0.0, "{} dequantize", ty.name());
        }
        Ok(())
    }

    #[cfg(feature = "sycl")]
    #[test]
    fn sycl_linear_matches_cpu_reference() -> Result<()> {
        if !candle_core::utils::sycl_is_available() {
            return Ok(());
        }
        let sycl = Device::new_sycl(0)?;
        // IQ4_NL at 288 columns exercises a partial last 256-value group.
        for (ty, rows, cols) in [
            (IQuantType::Iq4Xs, 37, 512),
            (IQuantType::Iq4Nl, 37, 288),
            (IQuantType::Iq4Nl, 19, 512),
        ] {
            let packed = random_blocks(ty, rows * cols / ty.block_size(), rows as u32 * 31);
            let cpu = IQuantLinear::new(ty, packed.clone(), rows, cols, &Device::Cpu)?;
            let gpu = IQuantLinear::new(ty, packed, rows, cols, &sycl)?;
            for n in [1usize, 3, 4, 7, 9, 33] {
                let x = Tensor::randn(0f32, 1.0, (n, cols), &Device::Cpu)?;
                let want = cpu.forward(&x)?;
                let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
                let check = |got: Tensor, tol: f32, what: &str| -> Result<()> {
                    let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let diff = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(
                        diff / scale < tol,
                        "{} {what} n={n}: rel diff {}",
                        ty.name(),
                        diff / scale
                    );
                    Ok(())
                };
                let xg = x.to_device(&sycl)?;
                let tol = 1e-4;
                check(gpu.forward(&xg)?, tol, "f32")?;
                check(gpu.forward(&xg.to_dtype(DType::F16)?)?, 2e-2, "f16")?;
                // A tiny chunk forces the multi-chunk prefill path.
                check(
                    gpu.forward_chunked(&xg, DType::F32, 4 * cols * 4)?,
                    tol,
                    "chunked",
                )?;
            }
            let dense = gpu.dequantize(DType::F32)?.to_device(&Device::Cpu)?;
            let diff = (dense - cpu.dequantize(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(diff, 0.0, "{} dequantize", ty.name());
        }
        Ok(())
    }

    /// Decode-shaped timing of the native matvec against Candle's `QMatMul`
    /// on the same weight re-quantized to Q5_K / Q4_K. Run with
    /// `cargo test --release --features cuda -p crane-core --lib bench_iq4_matvec -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "benchmark"]
    fn bench_iq4_matvec() -> Result<()> {
        use candle_core::Module;
        use candle_core::quantized::QMatMul;
        let cuda = Device::new_cuda(0)?;
        // Every IQ4_XS linear shape in a Qwen 3.8-27B IQ4_XS GGUF.
        for (rows, cols) in [
            (17408usize, 5120usize),
            (5120, 17408),
            (12288, 5120),
            (6144, 5120),
            (5120, 6144),
            (1024, 5120),
            (48, 5120),
        ] {
            let packed = random_blocks(IQuantType::Iq4Xs, rows * cols / 256, 99);
            let native = IQuantLinear::new(IQuantType::Iq4Xs, packed.clone(), rows, cols, &cuda)?;
            let x = Tensor::randn(0f32, 1.0, (1, cols), &cuda)?.to_dtype(DType::BF16)?;
            let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
                for _ in 0..5 {
                    f()?;
                }
                cuda.synchronize()?;
                let iters = 200;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    f()?;
                }
                cuda.synchronize()?;
                Ok(t.elapsed().as_secs_f64() / f64::from(iters))
            };
            let t_native = time(&|| native.forward(&x))?;
            let t_deq = time(&|| native.dequantize(DType::BF16))?;
            println!(
                "[{rows}x{cols}] IQ4_XS dequant->bf16 {:7.1} us  {:6.1} GB/s written",
                t_deq * 1e6,
                (rows * cols * 2) as f64 / t_deq / 1e9
            );
            let xp = Tensor::randn(0f32, 1.0, (512, cols), &cuda)?.to_dtype(DType::BF16)?;
            let t_pre = time(&|| native.forward(&xp))?;
            println!("[{rows}x{cols}] IQ4_XS prefill 512 {:7.1} us", t_pre * 1e6);
            let bytes = packed.len() as f64;
            println!(
                "[{rows}x{cols}] IQ4_XS native {:7.1} us  {:6.1} GB/s",
                t_native * 1e6,
                bytes / t_native / 1e9
            );
            for target in [GgmlDType::Q4K, GgmlDType::Q5K] {
                let q = requantize(IQuantType::Iq4Xs, &packed, &[rows, cols], target, &cuda)?;
                let q_bytes = q.storage_size_in_bytes() as f64;
                let qmm = QMatMul::from_arc(std::sync::Arc::new(q))?;
                let t = time(&|| qmm.forward(&x.to_dtype(DType::F32)?)?.to_dtype(DType::BF16))?;
                println!(
                    "[{rows}x{cols}] {target:?} candle   {:7.1} us  {:6.1} GB/s",
                    t * 1e6,
                    q_bytes / t / 1e9
                );
            }
        }
        Ok(())
    }
}
