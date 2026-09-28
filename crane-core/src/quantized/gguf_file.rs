// SPDX-License-Identifier: MIT

//! Shared GGUF file loading: memory-mapping and typed tensor access.
//!
//! Used by every model that supports GGUF checkpoints (`hunyuan_dense`,
//! `gemma4`, `qwen3`, `qwen3_5`, `minicpm5`, `minicpmo`).

use candle_core::quantized::{QTensor, gguf_file};
use candle_core::{DType, Device, Result};
use candle_nn::RmsNorm;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek};
use std::sync::Arc;

use super::extended_gguf::{ExtendedGgufInfo, IQuantTensorInfo};
use super::ternary::{GdnPermutation, HadamardMode, TernaryLinear, TernaryWeight};

/// Opens and memory-maps a GGUF file for zero-syscall tensor reads.
///
/// The returned `Mmap` can be wrapped in a `std::io::Cursor` and passed
/// anywhere a `Read + Seek` reader is expected (e.g. [`Gguf::new`]), letting
/// tensor loads page data in from disk on demand instead of going through
/// per-tensor `seek`/`read_exact` syscalls.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or memory-mapped.
pub fn mmap_gguf_file(path: impl AsRef<std::path::Path>) -> std::io::Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the caller must not truncate or replace this file on disk while
    // the returned mapping is alive. Doing so raises SIGBUS on a later page-in
    // (e.g. during tensor loading), which is unrecoverable and not something a
    // `Result` can catch. Crane itself never writes to model files it has
    // loaded; this only holds if external tooling (re-downloads, redeploys)
    // avoids replacing a model file path while a server process has it mapped.
    unsafe { memmap2::Mmap::map(&file) }
}

/// Wraps a parsed GGUF file + reader for convenient tensor loading.
pub struct Gguf<R: Read + Seek> {
    pub ct: gguf_file::Content,
    reader: R,
    device: Device,
    /// Target compute dtype. Dequantized tensors (norms, embeddings) are
    /// cast to this dtype so they match the activations flowing through the
    /// model (e.g. BF16 on CUDA). Quantized linear layers (`QMatMul`) handle
    /// their own internal dtype and the `LinearLayer` wrapper casts their
    /// output to the input's dtype.
    dtype: DType,
    ternary: Option<TernaryContext>,
    /// llama.cpp i-quant tensors, re-quantized on load (see [`super::iquant`]).
    iquant: HashMap<String, IQuantTensorInfo>,
    /// Tensors hidden from Candle's parser that nothing here can decode
    /// (ternary tensors without their Prism metadata).
    undecodable: HashSet<String>,
}

struct TernaryContext {
    tensors: ExtendedGgufInfo,
    block_size: usize,
    forward: HashSet<String>,
    inverse: HashSet<String>,
    signs: HashMap<usize, Arc<Vec<f32>>>,
    gdn_geometry: Option<(usize, usize)>,
}

impl<R: Read + Seek> Gguf<R> {
    /// Wrap a parsed GGUF file. The reader must cover the whole file: it is
    /// re-probed for tensors Crane decodes itself, so `ct` may come from
    /// [`read_content_lenient`](super::extended_gguf::read_content_lenient).
    pub fn new(ct: gguf_file::Content, mut reader: R, device: Device, dtype: DType) -> Self {
        let mut info = super::extended_gguf::probe_reader(&mut reader).unwrap_or_else(|e| {
            eprintln!("[gguf] warning: could not probe for i-quant/ternary tensors: {e}");
            ExtendedGgufInfo::default()
        });
        let iquant = std::mem::take(&mut info.iquant);
        let mut undecodable = HashSet::new();
        let ternary = if info.is_ternary() {
            let names: Vec<String> = info.tensors.keys().cloned().collect();
            TernaryContext::from_metadata(&ct, info)
                .inspect_err(|e| {
                    eprintln!("[gguf] warning: ternary tensors present but unusable: {e}");
                    undecodable.extend(names);
                })
                .ok()
        } else {
            None
        };
        log_iquant(&iquant, &device);
        Self {
            ct,
            reader,
            device,
            dtype,
            ternary,
            iquant,
            undecodable,
        }
    }

    /// Construct a GGUF reader for tensors Candle cannot parse itself:
    /// Prism PTQ1_0/PQ2_0 ternary weights and/or llama.cpp i-quants.
    /// Unlike [`Self::new`], missing Prism metadata is an error.
    pub fn new_extended(
        ct: gguf_file::Content,
        reader: R,
        device: Device,
        dtype: DType,
        mut tensors: ExtendedGgufInfo,
    ) -> Result<Self> {
        let iquant = std::mem::take(&mut tensors.iquant);
        super::iquant::requant_target()?;
        log_iquant(&iquant, &device);
        let ternary = if tensors.is_ternary() {
            Some(TernaryContext::from_metadata(&ct, tensors)?)
        } else {
            None
        };
        Ok(Self {
            ct,
            reader,
            device,
            dtype,
            ternary,
            iquant,
            undecodable: HashSet::new(),
        })
    }

    /// Load a tensor by name as a `QTensor`, decoding i-quants that Candle
    /// cannot represent (see [`super::iquant`]).
    fn load_qtensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        let Some((info, packed)) = self.iquant_bytes(name)? else {
            if self.undecodable.contains(name) {
                candle_core::bail!(
                    "GGUF tensor {name} is ternary but the file lacks Prism metadata"
                )
            }
            return self.ct.tensor(&mut self.reader, name, device);
        };
        let target = super::iquant::requant_target()?;
        super::iquant::requantize(info.ty, &packed, &info.shape, target, device)
    }

    /// Raw GGUF bytes of an i-quant tensor, or `None` for any other tensor.
    fn iquant_bytes(&mut self, name: &str) -> Result<Option<(IQuantTensorInfo, Vec<u8>)>> {
        let Some(info) = self.iquant.get(name).cloned() else {
            return Ok(None);
        };
        let elems: usize = info.shape.iter().product();
        let bytes = elems / info.ty.block_size() * info.ty.block_bytes();
        let absolute = self
            .ct
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| candle_core::Error::Msg(format!("tensor {name} offset overflow")))?;
        self.reader.seek(std::io::SeekFrom::Start(absolute))?;
        let mut packed = vec![0u8; bytes];
        self.reader.read_exact(&mut packed)?;
        Ok(Some((info, packed)))
    }

    /// Load a quantized tensor and wrap as a `LinearLayer` (`QMatMul`).
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn linear(&mut self, name: &str) -> Result<crate::ops::linear::LinearLayer> {
        if self
            .ternary
            .as_ref()
            .is_some_and(|ctx| ctx.tensors.tensors.contains_key(name))
        {
            let layer = self.ternary_linear(name)?;
            return Ok(crate::ops::linear::LinearLayer::Ternary(layer));
        }
        let device = self.device.clone();
        self.linear_on(name, &device)
    }

    /// Load a quantized tensor onto `device` and wrap as a `LinearLayer` (`QMatMul`).
    ///
    /// Identical to [`Self::linear`] but places the weight on a caller-chosen
    /// device instead of `self.device` — used for `MoE` expert offloading where
    /// experts may live on a different device (e.g. CPU) than the rest of the
    /// model.
    ///
    /// # Errors
    /// Returns an error if the tensor is missing from the GGUF file, the
    /// quantization type is unsupported, or the `QMatMul` construction fails.
    pub fn linear_on(
        &mut self,
        name: &str,
        device: &Device,
    ) -> Result<crate::ops::linear::LinearLayer> {
        if native_iquant(device)
            && let Some(info) = self.iquant.get(name)
            && info.shape.len() == 2
        {
            let (info, packed) = self.iquant_bytes(name)?.expect("i-quant tensor present");
            let layer = super::iquant::IQuantLinear::new(
                info.ty,
                packed,
                info.shape[0],
                info.shape[1],
                device,
            )?;
            return Ok(crate::ops::linear::LinearLayer::IQuant(layer));
        }
        let ws = self.load_qtensor(name, device)?;
        let qmm = candle_core::quantized::QMatMul::from_arc(Arc::new(ws))?;
        Ok(crate::ops::linear::LinearLayer::quantized(qmm))
    }

    fn ternary_linear(&mut self, name: &str) -> Result<TernaryLinear> {
        let (weight, signs, mode, block_size) = self.ternary_weight(name)?;
        let gdn_permutation = self
            .ternary
            .as_ref()
            .and_then(|ctx| ctx.gdn_geometry)
            .filter(|_| name.contains(".ssm_out."))
            .map(|(value_heads, key_heads)| GdnPermutation {
                head_dim: weight.cols() / value_heads,
                key_heads,
                repeats: value_heads / key_heads,
            });
        TernaryLinear::new(weight, signs, block_size, mode, gdn_permutation)
    }

    fn ternary_weight(
        &mut self,
        name: &str,
    ) -> Result<(Arc<TernaryWeight>, Arc<Vec<f32>>, HadamardMode, usize)> {
        let (info, mode, signs, block_size) = {
            let ctx = self
                .ternary
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("ternary GGUF context is missing".into()))?;
            let info = ctx
                .tensors
                .tensors
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing ternary tensor {name}")))?
                .clone();
            let mode = if ctx.forward.contains(name) {
                HadamardMode::Forward
            } else if ctx.inverse.contains(name) {
                HadamardMode::Inverse
            } else {
                HadamardMode::None
            };
            let cols = *info.shape.get(1).unwrap_or(&0);
            let signs = if mode == HadamardMode::None {
                Arc::new(vec![1.0; cols])
            } else {
                ctx.signs.get(&cols).cloned().ok_or_else(|| {
                    candle_core::Error::Msg(format!("missing Hadamard signs for width {cols}"))
                })?
            };
            (info, mode, signs, ctx.block_size)
        };
        if info.shape.len() != 2 {
            candle_core::bail!("ternary linear {name} must be rank 2, got {:?}", info.shape)
        }
        let (rows, cols) = (info.shape[0], info.shape[1]);
        let bytes = rows
            .checked_mul(cols / 128)
            .and_then(|n| n.checked_mul(info.encoding.block_bytes()))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("ternary tensor {name} size overflow"))
            })?;
        let absolute = self
            .ct
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("ternary tensor {name} offset overflow"))
            })?;
        self.reader.seek(std::io::SeekFrom::Start(absolute))?;
        let mut packed = vec![0u8; bytes];
        self.reader.read_exact(&mut packed)?;
        let weight = Arc::new(TernaryWeight::new(
            info.encoding,
            packed,
            rows,
            cols,
            &self.device,
        )?);
        Ok((weight, signs, mode, block_size))
    }

    /// Load a tensor, dequantize, and create an `RmsNorm`.
    /// The weight is cast to the target `dtype` so it matches activations.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.load_qtensor(name, &self.device.clone())?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(RmsNorm::new(weight, eps))
    }

    /// Load an embedding table that stays quantized when it can, dequantizing
    /// only the rows a forward pass gathers.
    ///
    /// Prefer this over [`Self::embedding`] for large vocabularies: a 248k-row
    /// table costs ~2.4 GiB dense in BF16 versus ~0.7 GiB as `Q4_K`.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn quantized_embedding(
        &mut self,
        name: &str,
        hidden_size: usize,
    ) -> Result<crate::models::modules::embedding::EmbeddingLayer> {
        if self
            .ternary
            .as_ref()
            .is_some_and(|ctx| ctx.tensors.tensors.contains_key(name))
        {
            let (weight, signs, mode, block_size) = self.ternary_weight(name)?;
            if mode != HadamardMode::Inverse {
                candle_core::bail!("ternary embedding {name} must use inverse Hadamard metadata")
            }
            return Ok(
                crate::models::modules::embedding::EmbeddingLayer::from_ternary(
                    weight, signs, block_size, self.dtype,
                ),
            );
        }
        let ws = self.load_qtensor(name, &self.device.clone())?;
        crate::models::modules::embedding::EmbeddingLayer::from_qtensor(ws, hidden_size, self.dtype)
    }

    /// Load a tensor, dequantize, and create an Embedding.
    /// The weight is cast to the target `dtype` so lookups produce
    /// tensors in the expected compute precision.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn embedding(&mut self, name: &str, hidden_size: usize) -> Result<candle_nn::Embedding> {
        let ws = self.load_qtensor(name, &self.device.clone())?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(candle_nn::Embedding::new(weight, hidden_size))
    }

    /// Load a raw `QTensor` by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing.
    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.load_qtensor(name, &self.device.clone())
    }

    /// Load a raw `QTensor` by name onto `device`.
    ///
    /// Identical to [`Self::tensor`] but places the weight on a
    /// caller-chosen device instead of `self.device` — used for `MoE`
    /// packed expert tensors, which must load directly onto
    /// `expert_device` rather than the main model device.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing.
    pub fn tensor_on(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        self.load_qtensor(name, device)
    }

    /// Load a tensor, dequantize, and cast to the target compute dtype.
    /// For small full-precision tensors (norm weights, biases, conv kernels).
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn dequant_tensor(&mut self, name: &str) -> Result<candle_core::Tensor> {
        let device = self.device.clone();
        self.dequant_tensor_on(name, &device)
    }

    /// Load a tensor onto `device`, dequantize, and cast to the target compute
    /// dtype.
    ///
    /// Identical to [`Self::dequant_tensor`] but places the result on a
    /// caller-chosen device instead of `self.device` — used for `MoE` packed
    /// expert tensors, which must be dequantized directly onto
    /// `expert_device` rather than the main model device.
    ///
    /// # Errors
    /// Returns an error if the tensor is missing from the GGUF file, the
    /// quantization type is unsupported, or dequantization fails.
    pub fn dequant_tensor_on(
        &mut self,
        name: &str,
        device: &Device,
    ) -> Result<candle_core::Tensor> {
        let ws = self.load_qtensor(name, device)?;
        ws.dequantize(device)?.to_dtype(self.dtype)
    }

    /// Whether the file contains a tensor with this exact name.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Access GGUF metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

/// Whether i-quant linear layers on `device` run through the native kernels.
fn native_iquant(device: &Device) -> bool {
    cfg!(feature = "cuda") && device.is_cuda() && super::iquant::native_enabled()
}

fn log_iquant(iquant: &HashMap<String, IQuantTensorInfo>, device: &Device) {
    if iquant.is_empty() {
        return;
    }
    let mut types: Vec<_> = iquant.values().map(|t| t.ty.name()).collect();
    types.sort_unstable();
    types.dedup();
    let target =
        super::iquant::requant_target().map_or_else(|e| e.to_string(), |t| format!("{t:?}"));
    if native_iquant(device) {
        eprintln!(
            "[gguf] {} tensors in {}: linear layers run natively; embeddings and packed MoE \
             experts are re-quantized to {target} (CRANE_IQ_REQUANT=<type> forces re-quantization)",
            iquant.len(),
            types.join("/"),
        );
    } else {
        eprintln!(
            "[gguf] {} tensors in {} have no native kernel on this device; re-quantizing to \
             {target} at load (CRANE_IQ_REQUANT to change)",
            iquant.len(),
            types.join("/"),
        );
    }
}

impl TernaryContext {
    fn from_metadata(ct: &gguf_file::Content, tensors: ExtendedGgufInfo) -> Result<Self> {
        let u32_value = |key: &str| -> Result<usize> {
            ct.metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata {key}")))?
                .to_u32()
                .map(|v| v as usize)
        };
        let string_value = |key: &str| -> Result<String> {
            ct.metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata {key}")))?
                .to_string()
                .cloned()
        };
        if u32_value("prism.hadamard.version")? != 1 {
            candle_core::bail!("unsupported prism.hadamard.version")
        }
        let block_size = u32_value("prism.hadamard.block_size")?;
        if !block_size.is_power_of_two()
            || string_value("prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
            || string_value("prism.hadamard.axis")? != "input-last-dimension"
        {
            candle_core::bail!("unsupported Prism Hadamard configuration")
        }
        let strings = |key: &str| -> Result<HashSet<String>> {
            let values = ct
                .metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata {key}")))?
                .to_vec()?;
            values
                .iter()
                .map(|v| v.to_string().cloned())
                .collect::<Result<HashSet<_>>>()
        };
        let forward = strings("prism.hadamard.weight_names")?;
        let inverse = strings("prism.hadamard.inverse_weight_names")?;
        let widths = ct
            .metadata
            .get("prism.hadamard.sign_widths")
            .ok_or_else(|| candle_core::Error::Msg("missing prism.hadamard.sign_widths".into()))?
            .to_vec()?;
        let values = ct
            .metadata
            .get("prism.hadamard.sign_values")
            .ok_or_else(|| candle_core::Error::Msg("missing prism.hadamard.sign_values".into()))?
            .to_vec()?;
        let mut offset = 0usize;
        let mut signs = HashMap::new();
        for width in widths {
            let width = width.to_i32()? as usize;
            let end = offset + width;
            if width % block_size != 0 || end > values.len() {
                candle_core::bail!("invalid Prism Hadamard sign table")
            }
            let row = values[offset..end]
                .iter()
                .map(|v| v.to_i32().map(|x| x as f32))
                .collect::<Result<Vec<_>>>()?;
            if row.iter().any(|&v| v != -1.0 && v != 1.0) {
                candle_core::bail!("Prism Hadamard signs must be +/-1")
            }
            signs.insert(width, Arc::new(row));
            offset = end;
        }
        if offset != values.len() {
            candle_core::bail!("Prism Hadamard sign table length mismatch")
        }
        let gdn_grouped = matches!(
            ct.metadata.get("prism.hadamard.gdn_v_grouped"),
            Some(gguf_file::Value::Bool(true))
        );
        let gdn_geometry = if gdn_grouped {
            let arch = ct
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok())
                .ok_or_else(|| candle_core::Error::Msg("missing GGUF architecture".into()))?;
            let metadata_usize = |suffix: &str| -> Result<usize> {
                ct.metadata
                    .get(&format!("{arch}.{suffix}"))
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!("missing GGUF metadata {arch}.{suffix}"))
                    })?
                    .to_u32()
                    .map(|v| v as usize)
            };
            let value_heads = metadata_usize("ssm.time_step_rank")?;
            let key_heads = metadata_usize("ssm.group_count")?;
            if key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0 {
                candle_core::bail!("invalid Prism GDN head geometry")
            }
            Some((value_heads, key_heads))
        } else {
            None
        };
        Ok(Self {
            tensors,
            block_size,
            forward,
            inverse,
            signs,
            gdn_geometry,
        })
    }
}

#[cfg(test)]
mod mmap_gguf_tests {
    use super::mmap_gguf_file;
    use candle_core::Tensor;
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};

    /// Writes a single quantized tensor to a real GGUF file on disk, for
    /// exercising the mmap read path against a real file.
    fn write_test_gguf(path: &std::path::Path) {
        let src = Tensor::arange(0f32, 4. * 32., &candle_core::Device::Cpu)
            .unwrap()
            .reshape((4, 32))
            .unwrap();
        let qtensor = QTensor::quantize(&src, GgmlDType::Q4_0).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &[], &[("weight", &qtensor)]).unwrap();
    }

    // Reading a GGUF tensor through the mmap'd path must return the exact
    // same bytes as reading it through a plain `File` reader, since the mmap
    // change (`mmap_gguf_file`) is only meant to swap the I/O mechanism, not
    // the data it produces.
    #[test]
    fn mmap_read_matches_direct_file_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        write_test_gguf(&path);

        let mmap = mmap_gguf_file(&path).unwrap();
        let mut mmap_cursor = std::io::Cursor::new(mmap.as_ref());
        let mmap_content = gguf_file::Content::read(&mut mmap_cursor).unwrap();
        let mmap_tensor = mmap_content
            .tensor(&mut mmap_cursor, "weight", &candle_core::Device::Cpu)
            .unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let file_content = gguf_file::Content::read(&mut file).unwrap();
        let file_tensor = file_content
            .tensor(&mut file, "weight", &candle_core::Device::Cpu)
            .unwrap();

        let mmap_data = mmap_tensor.data().unwrap();
        let file_data = file_tensor.data().unwrap();
        assert_eq!(mmap_data, file_data);
    }

    // A missing path must surface as an `Err`, not panic (e.g. on the
    // `unsafe` `Mmap::map` call).
    #[test]
    fn mmap_missing_file_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.gguf");
        assert!(mmap_gguf_file(&path).is_err());
    }
}

#[cfg(test)]
mod iquant_gguf_tests {
    use super::Gguf;
    use crate::quantized::extended_gguf::read_content_lenient;
    use crate::quantized::iquant::IQuantType;
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device, Module, Tensor};

    /// Hand-write a GGUF (Candle's writer cannot emit i-quant types).
    /// Each tensor is `(name, shape outermost-first, ggml type id, bytes)`.
    fn write_gguf(tensors: &[(&str, &[usize], u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend(0u64.to_le_bytes());
        let mut offset = 0u64;
        for (name, shape, ty, data) in tensors {
            out.extend((name.len() as u64).to_le_bytes());
            out.extend(name.as_bytes());
            out.extend((shape.len() as u32).to_le_bytes());
            for &d in shape.iter().rev() {
                out.extend((d as u64).to_le_bytes());
            }
            out.extend(ty.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offset += (data.len() as u64).div_ceil(32) * 32;
        }
        for (_, _, _, data) in tensors {
            out.resize(out.len().div_ceil(32) * 32, 0);
            out.extend(data);
        }
        out
    }

    /// `n` pseudo-random `IQ4_XS` blocks with a sane `f16` scale.
    fn iq4_xs_blocks(n: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let mut out = Vec::with_capacity(n * IQuantType::Iq4Xs.block_bytes());
        for _ in 0..n {
            out.extend(half::f16::from_f32(0.01).to_le_bytes());
            for _ in 2..IQuantType::Iq4Xs.block_bytes() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                out.push(state as u8);
            }
        }
        out
    }

    #[test]
    fn gguf_new_decodes_iquant_tensors_from_lenient_header() -> candle_core::Result<()> {
        let (rows, cols) = (4, 512);
        let packed = iq4_xs_blocks(rows * cols / 256);
        let norm: Vec<u8> = (0..cols).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let bytes = write_gguf(&[
            ("weight", &[rows, cols], 23, packed.clone()),
            ("norm", &[cols], 0, norm),
        ]);
        let ct = read_content_lenient(&bytes)?;
        let mut gg = Gguf::new(
            ct,
            std::io::Cursor::new(&bytes[..]),
            Device::Cpu,
            DType::F32,
        );

        let mut reference = vec![0f32; rows * cols];
        IQuantType::Iq4Xs.dequantize(&packed, &mut reference);
        let weight = gg.tensor("weight")?;
        assert_eq!(weight.dtype(), GgmlDType::Q5K);
        assert_eq!(weight.shape().dims(), &[rows, cols]);
        let got = weight
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_ref = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
        let max_err = reference
            .iter()
            .zip(&got)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_err <= max_ref / 20.0,
            "max_err {max_err} vs max_ref {max_ref}"
        );

        let xs = Tensor::ones((1, cols), DType::F32, &Device::Cpu)?;
        assert_eq!(gg.linear("weight")?.forward(&xs)?.dims(), &[1, rows]);

        let norm = gg.dequant_tensor("norm")?.to_vec1::<f32>()?;
        assert_eq!(norm[cols - 1], (cols - 1) as f32);
        Ok(())
    }

    #[test]
    fn lenient_header_names_unsupported_ggml_types() {
        let bytes = write_gguf(&[("blk.0.ffn_down.weight", &[1, 256], 16, vec![0; 66])]);
        let err = read_content_lenient(&bytes).unwrap_err().to_string();
        assert!(
            err.contains("blk.0.ffn_down.weight") && err.contains("IQ2_XXS"),
            "{err}"
        );
    }
}
