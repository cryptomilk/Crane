// SPDX-License-Identifier: MIT

//! Launchers for `kernels/cuda/quant_iq4.cu` on ROCm/HIP.
//!
//! The same source the CUDA build compiles to PTX; here candle compiles it
//! with `hipcc` on first use and caches the code object on disk (see
//! [`crate::ops::rocm`]). Every intrinsic the kernel needs (`__byte_perm`,
//! `__dp4a`, the `*_sync` warp shuffles, bf16 conversions) is either native
//! on ROCm or already bridged by candle-rocm's HIP shim, so this mirrors
//! [`super::cuda`]'s int8 fast path for IQ4_XS exactly, rather than falling
//! back to the float-only approach [`super::metal`]/[`super::sycl`] use.

use std::ffi::c_void;

use candle_core::rocm_backend::{RocmStorageSlice, rocm_rs};
use candle_core::{DType, Result, Tensor, bail};
use rocm_rs::hip::Dim3;

use crate::ops::rocm::{self, RocmElem, arg, wrap};
use crate::quantized::iquant::IQuantType;

const MODULE_NAME: &str = "crane_quant_iq4";
const SOURCE: &str = include_str!("../../../kernels/cuda/quant_iq4.cu");
/// Output rows per thread block in the matvec kernels (one warp each) —
/// must match `IQ4_WARPS` in `kernels/cuda/quant_iq4.cu`.
const WARPS: usize = 4;

/// Device pointer into a packed u8 weight tensor, plus the storage guard the
/// pointer borrows from.
///
/// Mirrors [`super::cuda`]'s `packed_slice`: the caller must keep the
/// returned guard alive until after the kernel launch (see
/// [`rocm::device_ptr`]'s safety note) — dropping it early is what this
/// split return prevents, unlike a helper that drops its own local guard on
/// return.
fn packed_u8_ptr(
    packed: &Tensor,
    byte_offset: usize,
) -> Result<(candle_core::StorageRef<'_>, *mut c_void)> {
    let (storage, layout) = packed.storage_and_layout();
    let slice = rocm::rocm_slice(&storage, "i-quant packed weight")?;
    if slice.dtype() != DType::U8 {
        bail!("i-quant packed weight must be a ROCm u8 tensor");
    }
    // SAFETY: `byte_offset` is within the tensor's own allocation — either
    // its layout offset (matvec) or that plus a caller-checked row range
    // (dequantize).
    let ptr = unsafe {
        match slice {
            RocmStorageSlice::U8(m) => m.ptr_at(layout.start_offset() + byte_offset),
            _ => unreachable!("dtype checked above"),
        }
    };
    Ok((storage, ptr))
}

/// Warps-per-output-row geometry shared by the int8 and float matvec paths.
fn matvec_geometry(rows: usize, output_rows: usize) -> (usize, Dim3, Dim3) {
    let nr = if rows >= 4 { 4 } else { 1 };
    // Grid/block extents come from tensor shapes (rows, output_rows), never
    // large enough on real models to approach u32::MAX.
    #[allow(clippy::cast_possible_truncation)]
    let (grid, block) = (
        Dim3::new_2d(output_rows.div_ceil(WARPS) as u32, rows.div_ceil(nr) as u32),
        Dim3::new_1d(32 * WARPS as u32),
    );
    (nr, grid, block)
}

/// `input` (`[rows, cols]`, ROCm f32/f16/bf16) times the transposed packed
/// weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// IQ4_XS quantizes the activations to int8 (one scale per 32 values) and
/// runs an integer `dp4a` dot product, like llama.cpp, writing the output
/// directly in `out_dtype`. IQ4_NL uses a float kernel. Meant for
/// decode-sized `rows`; prefill should use [`dequantize`] and a regular
/// matmul instead.
///
/// # Errors
///
/// Returns an error if the tensors are not on a ROCm device, `out_dtype`
/// isn't F32/F16/BF16, or a kernel launch fails.
pub fn matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    match ty {
        IQuantType::Iq4Nl => matvec_nl_f32(input, packed, output_rows, cols)?.to_dtype(out_dtype),
        IQuantType::Iq4Xs => matvec_xs(input, packed, output_rows, cols, out_dtype),
    }
}

fn quantize_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("iq4_quantize_q8_f32"),
        DType::F16 => Ok("iq4_quantize_q8_f16"),
        DType::BF16 => Ok("iq4_quantize_q8_bf16"),
        other => bail!("i-quant kernels do not support {other:?} activations"),
    }
}

fn matvec_xs(
    input: &Tensor,
    packed: &Tensor,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let rows = input.elem_count() / cols;
    let dev = input.device().as_rocm_device()?.clone();
    let n_blocks = rows * cols / 32;
    let (nr, grid, block) = matvec_geometry(rows, output_rows);

    // int8 activations followed by one f32 scale per 32 of them; `rows * cols`
    // is a multiple of 256, so the scales stay 4-byte aligned.
    let scratch = dev.alloc::<u8>(rows * cols + 4 * n_blocks)?;
    let xq_ptr = scratch.as_ptr();
    // SAFETY: `xq_ptr` addresses `rows * cols + 4 * n_blocks` bytes; offsetting
    // by the first `rows * cols` lands exactly on the scales region.
    let xd_ptr = unsafe { xq_ptr.cast::<u8>().add(rows * cols).cast::<c_void>() };

    {
        let quantize = quantize_kernel_name(input.dtype())?;
        let (input_s, input_l) = input.storage_and_layout();
        let input_ptr = rocm::device_ptr(&input_s, input_l, input.dtype(), "i-quant matvec input")?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n_blocks_i = n_blocks as i32;
        let mut args = [arg(&input_ptr), arg(&xq_ptr), arg(&xd_ptr), arg(&n_blocks_i)];
        // One warp per 32 activations, 8 warps per thread block.
        #[allow(clippy::cast_possible_truncation)]
        let quantize_grid = n_blocks.div_ceil(8) as u32;
        unsafe {
            rocm::launch(
                &dev,
                MODULE_NAME,
                quantize,
                SOURCE,
                quantize_grid,
                256,
                0,
                &mut args,
            )?;
        }
        drop(input_s);
    }

    let (packed_s, packed_ptr) = packed_u8_ptr(packed, 0)?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let (rows_i, output_rows_i, cols_i) = (rows as i32, output_rows as i32, cols as i32);

    macro_rules! run {
        ($kernel_suffix:literal, $elem:ty) => {{
            let kernel = if nr == 1 {
                concat!("iq4_xs_matvec_q8_1_", $kernel_suffix)
            } else {
                concat!("iq4_xs_matvec_q8_4_", $kernel_suffix)
            };
            let output_buf = dev.alloc::<$elem>(rows * output_rows)?;
            let output_ptr = output_buf.as_ptr();
            let mut args = [
                arg(&packed_ptr),
                arg(&xq_ptr),
                arg(&xd_ptr),
                arg(&output_ptr),
                arg(&rows_i),
                arg(&output_rows_i),
                arg(&cols_i),
            ];
            let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, SOURCE)?;
            func.launch(grid, block, 0, Some(dev.stream()), &mut args)
                .map_err(|e| candle_core::Error::Msg(format!("{kernel} launch failed: {e}")))?;
            wrap(<$elem as RocmElem>::wrap_slice(output_buf), &dev, (rows, output_rows))
        }};
    }

    let out = match out_dtype {
        DType::F32 => run!("f32", f32),
        DType::F16 => run!("f16", half::f16),
        DType::BF16 => run!("bf16", half::bf16),
        other => bail!("i-quant matvec: unsupported output dtype {other:?}"),
    };
    drop(packed_s);
    Ok(out)
}

fn matvec_nl_f32(input: &Tensor, packed: &Tensor, output_rows: usize, cols: usize) -> Result<Tensor> {
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let rows = input.elem_count() / cols;
    let dev = input.device().as_rocm_device()?.clone();
    let (nr, grid, block) = matvec_geometry(rows, output_rows);

    let (input_s, input_l) = input.storage_and_layout();
    let input_ptr = rocm::device_ptr(&input_s, input_l, DType::F32, "i-quant matvec input")?;
    let (packed_s, packed_ptr) = packed_u8_ptr(packed, 0)?;

    let output_buf = dev.alloc::<f32>(rows * output_rows)?;
    let output_ptr = output_buf.as_ptr();
    let kernel = if nr == 1 {
        "iq4_nl_matvec1_f32"
    } else {
        "iq4_nl_matvec4_f32"
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let (rows_i, output_rows_i, cols_i) = (rows as i32, output_rows as i32, cols as i32);
    let mut args = [
        arg(&packed_ptr),
        arg(&input_ptr),
        arg(&output_ptr),
        arg(&rows_i),
        arg(&output_rows_i),
        arg(&cols_i),
    ];
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, SOURCE)?;
    func.launch(grid, block, 0, Some(dev.stream()), &mut args)
        .map_err(|e| candle_core::Error::Msg(format!("{kernel} launch failed: {e}")))?;
    drop(input_s);
    drop(packed_s);

    Ok(rocm::wrap_f32(output_buf, &dev, (rows, output_rows)))
}

/// Expand weight rows `row_start..row_start + n_rows` of a packed
/// `[_, cols]` tensor into a dense `[n_rows, cols]` tensor of `dtype`.
///
/// # Errors
///
/// Returns an error if `packed` is not a ROCm u8 tensor, `dtype` isn't
/// F32/F16/BF16, or the kernel launch fails.
pub fn dequantize(
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = packed.device().as_rocm_device()?.clone();
    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    let (packed_s, packed_ptr) = packed_u8_ptr(packed, row_start * row_bytes)?;

    // One warp per 256 weights: an IQ4_XS block or eight IQ4_NL blocks.
    let n_blocks = n_rows * cols / ty.block_size();
    let warps = n_rows * cols / 256 + usize::from(!(n_rows * cols).is_multiple_of(256));
    #[allow(clippy::cast_possible_truncation)]
    let grid = warps.div_ceil(8) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let n_blocks_i = n_blocks as i32;

    macro_rules! run {
        ($kernel:literal, $elem:ty) => {{
            let dst = dev.alloc::<$elem>(n_rows * cols)?;
            let dst_ptr = dst.as_ptr();
            let mut args = [arg(&packed_ptr), arg(&dst_ptr), arg(&n_blocks_i)];
            unsafe {
                rocm::launch(&dev, MODULE_NAME, $kernel, SOURCE, grid, 256, 0, &mut args)?;
            }
            wrap(<$elem as RocmElem>::wrap_slice(dst), &dev, (n_rows, cols))
        }};
    }

    let out = match (ty, dtype) {
        (IQuantType::Iq4Xs, DType::F32) => run!("iq4_xs_dequant_f32", f32),
        (IQuantType::Iq4Xs, DType::F16) => run!("iq4_xs_dequant_f16", half::f16),
        (IQuantType::Iq4Xs, DType::BF16) => run!("iq4_xs_dequant_bf16", half::bf16),
        (IQuantType::Iq4Nl, DType::F32) => run!("iq4_nl_dequant_f32", f32),
        (IQuantType::Iq4Nl, DType::F16) => run!("iq4_nl_dequant_f16", half::f16),
        (IQuantType::Iq4Nl, DType::BF16) => run!("iq4_nl_dequant_bf16", half::bf16),
        (_, other) => bail!("i-quant dequantize: unsupported output dtype {other:?}"),
    };
    drop(packed_s);
    Ok(out)
}
