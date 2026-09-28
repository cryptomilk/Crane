// SPDX-License-Identifier: MIT

//! Launchers for `kernels/metal/quant_iq4.metal`.
//!
//! Mirrors [`super::sycl`]: activations are not quantized to int8 for a
//! `dp4a`-style dot product (Metal has no portable int4 dot across the whole
//! family) — both entry points decode weights on the fly and accumulate a
//! plain float dot product. F32 / F16 / BF16 outputs are supported; on older
//! Macs without `bfloat` (Metal < 3.0) the BF16 kernels are absent from the
//! compiled library and the BF16 path returns an error so the caller can
//! fall back to the CPU path.
//!
//! The MSL source is compiled to an `MTLLibrary` once per `MetalDevice` (lazy,
//! behind a process-wide `RwLock`) via
//! [`candle_metal_kernels::Device::new_library_with_source`] — the same
//! pattern candle uses for its own kernels, no `.metallib` to ship.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use candle_core::metal_backend::MetalStorage;
use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Device, Library};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};

use crate::quantized::iquant::IQuantType;

const MSL_SRC: &str = include_str!("../../../kernels/metal/quant_iq4.metal");

fn dtype_tag(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => candle_core::bail!("i-quant Metal kernels do not support {other:?} output"),
    }
}

fn xs_tag(ty: IQuantType) -> &'static str {
    match ty {
        IQuantType::Iq4Xs => "xs",
        IQuantType::Iq4Nl => "nl",
    }
}

/// Cached `MTLLibrary` + per-kernel `ComputePipeline` for one device.
struct Iq4Kernels {
    #[allow(dead_code)]
    library: Library,
    /// `bf16` kernels are only present when the device defines
    /// `__HAVE_BFLOAT__` (Metal 3.0+); see [`matvec`]/[`dequantize`] for the
    /// fallback when the lookup misses.
    pipelines: Mutex<HashMap<String, ComputePipeline>>,
}

impl Iq4Kernels {
    fn pipeline(&self, device: &Device, name: &str) -> Result<ComputePipeline> {
        let mut guard = self.pipelines.lock().unwrap();
        if let Some(p) = guard.get(name) {
            return Ok(p.clone());
        }
        let function = self
            .library
            .get_function(name, None)
            .map_err(|e| candle_core::Error::Msg(format!("iq4 Metal: {name}: {e}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| candle_core::Error::Msg(format!("iq4 Metal pipeline {name}: {e}")))?;
        guard.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}

fn kernels_for(device: &Device) -> Result<Arc<Iq4Kernels>> {
    let registry_id = device.registry_id();
    static CACHE: OnceLock<RwLock<HashMap<u64, Arc<Iq4Kernels>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    if let Some(k) = cache.read().unwrap().get(&registry_id) {
        return Ok(k.clone());
    }
    let mut write = cache.write().unwrap();
    if let Some(k) = write.get(&registry_id) {
        return Ok(k.clone());
    }
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    let library = device
        .new_library_with_source(MSL_SRC, Some(&opts))
        .map_err(|e| candle_core::Error::Msg(format!("iq4 Metal: compile MSL failed: {e}")))?;
    let k = Arc::new(Iq4Kernels {
        library,
        pipelines: Mutex::new(HashMap::new()),
    });
    write.insert(registry_id, k.clone());
    Ok(k)
}

/// Returns the inner `candle_metal_kernels::Buffer` of a `Storage::Metal`
/// and the byte offset (taking the candle layout's element offset into
/// account).
fn metal_buf_and_offset<'a>(
    storage: &'a Storage,
    layout: &candle_core::Layout,
    extra_bytes: usize,
    elem_size: usize,
    name: &str,
) -> Result<(&'a candle_metal_kernels::metal::Buffer, usize)> {
    let buf = match storage {
        Storage::Metal(st) => st.buffer(),
        _ => candle_core::bail!("i-quant Metal kernel: {name} must be a metal tensor"),
    };
    Ok((buf, layout.start_offset() * elem_size + extra_bytes))
}

/// `input` (`[rows, cols]`, Metal, F32-cast) times the transposed packed
/// weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// Meant for decode-sized `rows`; prefill should use [`dequantize`] and a
/// regular matmul instead (see `IQuantLinear::forward_chunked`).
///
/// # Errors
///
/// Returns an error if the tensors are not on a Metal device, `out_dtype`
/// isn't F32/F16/BF16, the BF16 kernel is missing on older Metal, or the
/// kernel launch fails.
pub fn matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let dev = input.device().as_metal_device()?.clone();
    let mtl_dev = dev.metal_device();
    let kernels = kernels_for(mtl_dev)?;
    let dtype = dtype_tag(out_dtype)?;
    let xs = xs_tag(ty);

    // Activations are F32 in the kernel (matches the SYCL path).
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let rows = input.elem_count() / cols;

    let (input_storage, input_layout) = input.storage_and_layout();
    let (input_buf, input_offset) = metal_buf_and_offset(
        &input_storage,
        &input_layout,
        0,
        DType::F32.size_in_bytes(),
        "matvec input",
    )?;
    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let (packed_buf, packed_offset) =
        metal_buf_and_offset(&packed_storage, &packed_layout, 0, 1, "matvec weight")?;

    let kernel_name = format!("matvec_{dtype}_{xs}");
    let pipeline = kernels.pipeline(mtl_dev, &kernel_name)?;

    let out_el = rows * output_rows;
    let out_buf = dev
        .new_buffer_builder()
        .with_size_for(out_el, out_dtype)
        .with_label("iq4_matvec_out")
        .build()?;

    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_input_buffer(0, Some(packed_buf), packed_offset);
        enc.set_input_buffer(1, Some(input_buf), input_offset);
        enc.set_output_buffer(2, Some(&out_buf), 0);
        let input_rows_i = rows as i32;
        let output_rows_i = output_rows as i32;
        let cols_i = cols as i32;
        enc.set_bytes(3, &input_rows_i);
        enc.set_bytes(4, &output_rows_i);
        enc.set_bytes(5, &cols_i);

        // 2D grid of (output_rows, input_rows) — one thread per (out, in) pair.
        // 32-thread threadgroups in x give the GPU something to schedule.
        let grid = MTLSize {
            width: output_rows,
            height: rows,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
    drop(input_storage);
    drop(packed_storage);

    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(out_buf, dev.clone(), out_el, out_dtype)),
        dims,
        BackpropOp::none(),
        false,
    ))
}

/// Expand weight rows `row_start..row_start + n_rows` of a packed `[_, cols]`
/// tensor into a dense `[n_rows, cols]` tensor of `dtype`.
///
/// # Errors
///
/// Returns an error if `packed` is not on a Metal device, `dtype` isn't
/// F32/F16/BF16, the BF16 kernel is missing on older Metal, or the kernel
/// launch fails.
pub fn dequantize(
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = packed.device().as_metal_device()?.clone();
    let mtl_dev = dev.metal_device();
    let kernels = kernels_for(mtl_dev)?;
    let dtype_tag = dtype_tag(dtype)?;
    let xs = xs_tag(ty);

    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let (packed_buf, packed_offset) = metal_buf_and_offset(
        &packed_storage,
        &packed_layout,
        row_start * row_bytes,
        1,
        "dequantize weight",
    )?;

    let kernel_name = format!("dequant_{dtype_tag}_{xs}");
    let pipeline = kernels.pipeline(mtl_dev, &kernel_name)?;

    let out_buf = dev
        .new_buffer_builder()
        .with_size_for(n_rows * cols, dtype)
        .with_label("iq4_dequant_out")
        .build()?;

    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_input_buffer(0, Some(packed_buf), packed_offset);
        enc.set_output_buffer(1, Some(&out_buf), 0);
        let n_rows_i = n_rows as i32;
        let cols_i = cols as i32;
        enc.set_bytes(2, &n_rows_i);
        enc.set_bytes(3, &cols_i);

        let grid = MTLSize {
            width: n_rows,
            height: cols / ty.block_size(),
            depth: 1,
        };
        let tg = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
    drop(packed_storage);

    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(
            out_buf,
            dev.clone(),
            n_rows * cols,
            dtype,
        )),
        (n_rows, cols),
        BackpropOp::none(),
        false,
    ))
}
