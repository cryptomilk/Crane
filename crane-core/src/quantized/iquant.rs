// SPDX-License-Identifier: MIT

//! llama.cpp "i-quant" tensor types that Candle's `GgmlDType` does not know.
//!
//! imatrix quants (`IQ4_XS`, `IQ4_NL`, and the lower-bit IQ family used by
//! e.g. unsloth's dynamic quants) make Candle's GGUF parser fail with the
//! misleading `unknown dtype for tensor <ggml type id>`. The
//! [`extended_gguf`](super::extended_gguf) probe hides these tensors from
//! Candle; this module decodes them.
//!
//! There are no native kernels yet: each tensor is dequantized on the CPU and
//! re-quantized at load time to a Candle-native type (see
//! [`requant_target`]), so every backend runs it through its existing
//! `QMatMul` kernels. Reference: `ggml/src/ggml-quants.c` and
//! `ggml/src/ggml-common.h` in llama.cpp.

use std::borrow::Cow;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{Device, Result, Tensor, bail};
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

/// Candle-native type to re-quantize i-quant tensors into.
///
/// `CRANE_IQ_REQUANT` picks it (`q4k`, `q5k`, `q6k`, `q8_0`); the default
/// `q5k` keeps nearly all of the source precision at ~30% more memory than
/// `IQ4_XS`. `q4k` is about the same size as the source but quantizes twice.
pub fn requant_target() -> Result<GgmlDType> {
    let Ok(value) = std::env::var("CRANE_IQ_REQUANT") else {
        return Ok(GgmlDType::Q5K);
    };
    Ok(match value.to_ascii_lowercase().as_str() {
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q5k" | "q5_k" => GgmlDType::Q5K,
        "q6k" | "q6_k" => GgmlDType::Q6K,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        other => {
            bail!("CRANE_IQ_REQUANT: unsupported target {other:?} (use q4k, q5k, q6k or q8_0)")
        },
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
}
