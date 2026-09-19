//! Polymorphic linear layer shared by the safetensors, in-situ-quantized (ISQ)
//! and GGUF loading paths.
//!
//! [`LinearLayer`] lives here (rather than in a model module) so that shared
//! ops like [`crate::ops::gdn`] can use it without depending on a specific
//! model; `models::hunyuan_dense::modeling` re-exports it for its existing
//! users (hunyuan, qwen3).

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use std::sync::Arc;

/// A `QMatMul` with an optional bias. `QMatMul` itself has no bias — Qwen2's
/// Q/K/V projections need this wrapper to be quantizable at all.
#[derive(Clone)]
pub struct QuantizedLinear {
    pub matmul: QMatMul,
    pub bias: Option<Tensor>,
}

impl QuantizedLinear {
    pub fn new(qmm: QMatMul) -> Self {
        Self {
            matmul: qmm,
            bias: None,
        }
    }

    pub fn with_bias(qmm: QMatMul, bias: Tensor) -> Self {
        Self {
            matmul: qmm,
            bias: Some(bias),
        }
    }

    /// Apply the quantized matmul + optional bias. `QMatMul` dequantizes to
    /// F32 internally and requires F32 input; we round-trip the input dtype
    /// so BF16/F16 activation pipelines keep their dtype downstream
    /// (residual adds, etc.).
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let input_dtype = xs.dtype();
        let xs_f32 = if input_dtype != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.clone()
        };
        let out = self.matmul.forward(&xs_f32)?;
        let out = match &self.bias {
            Some(b) => out.broadcast_add(&b.to_dtype(DType::F32)?)?,
            None => out,
        };
        if input_dtype != DType::F32 {
            out.to_dtype(input_dtype)
        } else {
            Ok(out)
        }
    }
}

/// A linear layer that can be either a standard (f16/f32) Linear or a
/// quantized QMatMul. Both implement Module::forward identically from the
/// caller's perspective. This allows the same model code to serve both
/// safetensors and GGUF weights with zero duplication.
#[derive(Clone)]
pub enum LinearLayer {
    Standard(Linear),
    Quantized(QuantizedLinear),
}

impl LinearLayer {
    /// Construct a bias-free [`LinearLayer::Quantized`] (the common case —
    /// GGUF loaders for non-Qwen2 models, etc.).
    pub fn quantized(qmm: QMatMul) -> Self {
        Self::Quantized(QuantizedLinear::new(qmm))
    }

    /// Construct a [`LinearLayer::Quantized`] with a bias carried over
    /// unquantized. Used by Qwen2's Q/K/V projections — `QMatMul` has no
    /// bias of its own, so the bias lives alongside in the wrapper.
    pub fn quantized_with_bias(qmm: QMatMul, bias: Tensor) -> Self {
        Self::Quantized(QuantizedLinear::with_bias(qmm, bias))
    }

    /// Moves this layer's weights to `device`, in `dtype`.
    ///
    /// `Standard` moves its tensors directly, casting to `dtype`. `QTensor`
    /// has no device-transfer primitive, so `Quantized` dequantizes (via
    /// `QMatMul::dequantize_f16`, then casts to `dtype`) and returns a
    /// `Standard` layer on `device` — this loses the quantized memory
    /// footprint for the moved weight, a deliberate tradeoff for promoting
    /// an expert from CPU to GPU once real headroom is known (see
    /// `Qwen3Model::promote_experts_to_gpu`). `dtype` must match the
    /// model's compute dtype: candle's matmul requires both operands to
    /// share a dtype, so a promoted weight left in the wrong dtype fails
    /// on its very next forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the dequantization, dtype cast, or device
    /// transfer fails.
    pub fn to_device(&self, device: &Device, dtype: DType) -> Result<LinearLayer> {
        match self {
            Self::Standard(l) => {
                let weight = l.weight().to_device(device)?.to_dtype(dtype)?;
                let bias = l
                    .bias()
                    .map(|b| b.to_device(device)?.to_dtype(dtype))
                    .transpose()?;
                Ok(Self::Standard(Linear::new(weight, bias)))
            },
            Self::Quantized(q) => {
                let weight = q
                    .matmul
                    .dequantize_f16()?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                Ok(Self::Standard(Linear::new(weight, None)))
            },
        }
    }
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(xs),
            Self::Quantized(q) => q.forward(xs),
        }
    }
}

/// Parse a quantization level name as accepted by `--quant` / `CRANE_ISQ`
/// (e.g. `q4_0`, `q8_0`, `q4k` / `q4_k`, case-insensitive).
pub fn parse_ggml_dtype(name: &str) -> Result<GgmlDType> {
    let normalized = name.trim().to_lowercase().replace("_k", "k");
    let dt = match normalized.as_str() {
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        "q2k" => GgmlDType::Q2K,
        "q3k" => GgmlDType::Q3K,
        "q4k" => GgmlDType::Q4K,
        "q5k" => GgmlDType::Q5K,
        "q6k" => GgmlDType::Q6K,
        _ => candle_core::bail!(
            "unknown quantization level '{name}' (expected one of q4_0, q4_1, q5_0, q5_1, q8_0, q2k, q3k, q4k, q5k, q6k)"
        ),
    };
    Ok(dt)
}

/// Quantize a loaded linear's weight in place (ISQ), returning a
/// [`LinearLayer::Quantized`].
///
/// K-quants need the input dim to be a multiple of 256; when it isn't, fall
/// back to `Q8_0` (block size 32) so oddly-shaped projections still shrink
/// instead of erroring out. A bias, if present, is carried over unquantized
/// (see [`QuantizedLinear`]) — e.g. Qwen2's biased Q/K/V projections.
pub fn quantize_linear(linear: Linear, dtype: GgmlDType) -> Result<LinearLayer> {
    let bias = linear.bias().cloned();
    let weight = linear.weight();
    let in_dim = weight.dim(candle_core::D::Minus1)?;
    let dtype = if in_dim % dtype.block_size() == 0 {
        dtype
    } else {
        GgmlDType::Q8_0
    };
    if in_dim % dtype.block_size() != 0 {
        // Even Q8_0 can't represent this shape; keep it in full precision.
        return Ok(LinearLayer::Standard(linear));
    }
    let qt = QTensor::quantize(weight, dtype)?;
    let qmm = QMatMul::from_arc(Arc::new(qt))?;
    Ok(match bias {
        Some(b) => LinearLayer::quantized_with_bias(qmm, b),
        None => LinearLayer::quantized(qmm),
    })
}

/// Load a bias-free linear from `vb`, optionally quantizing it at load time.
///
/// With `quant: None` this is exactly `linear_no_bias` wrapped in
/// [`LinearLayer::Standard`]; with `Some(dtype)` the bf16/f16 weight is
/// quantized immediately and dropped, keeping peak memory near the quantized
/// size when loading from mmaped safetensors.
pub fn linear_layer(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    quant: Option<GgmlDType>,
) -> Result<LinearLayer> {
    let linear = linear_no_bias(in_dim, out_dim, vb)?;
    match quant {
        None => Ok(LinearLayer::Standard(linear)),
        Some(dt) => quantize_linear(linear, dt),
    }
}

/// Like [`linear_layer`], but `vb_cpu` must be scoped to [`Device::Cpu`]
/// and the result is quantized directly onto `target_device` via
/// [`QTensor::quantize_onto`] instead of [`QTensor::quantize`].
///
/// Reads from a CPU-scoped `vb_cpu` so the transient unquantized weight is
/// ordinary, promptly-freed heap memory that never touches the target
/// device — only the smaller quantized buffer does. Used by the
/// KugelAudio decoder's `new_with_quant` path: per-tensor GPU-side staging
/// wasn't being reclaimed between layers otherwise on Metal with an 18GB
/// unified-memory budget.
pub fn quantize_linear_onto(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb_cpu: VarBuilder,
    dtype: GgmlDType,
    target_device: &Device,
) -> Result<LinearLayer> {
    debug_assert!(
        vb_cpu.device().is_cpu(),
        "quantize_linear_onto: vb_cpu must be scoped to Device::Cpu"
    );
    let weight = vb_cpu.get((out_dim, in_dim), "weight")?;
    let bias = if bias {
        Some(vb_cpu.get(out_dim, "bias")?.to_device(target_device)?)
    } else {
        None
    };
    let dtype = if in_dim % dtype.block_size() == 0 {
        dtype
    } else {
        GgmlDType::Q8_0
    };
    if in_dim % dtype.block_size() != 0 {
        // Even Q8_0 can't represent this shape; keep it in full precision
        // (still only ever materialized on the target device once, here).
        let weight = weight.to_device(target_device)?;
        return Ok(LinearLayer::Standard(Linear::new(weight, bias)));
    }
    let qt = QTensor::quantize_onto(&weight, dtype, target_device)?;
    let qmm = QMatMul::from_arc(Arc::new(qt))?;
    Ok(match bias {
        Some(b) => LinearLayer::quantized_with_bias(qmm, b),
        None => LinearLayer::quantized(qmm),
    })
}
