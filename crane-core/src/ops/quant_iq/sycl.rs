// SPDX-License-Identifier: MIT

//! Launchers for `kernels/sycl/quant_iq4.cpp`.
//!
//! Unlike the CUDA path ([`super::cuda`]), there is no int8-activation fast
//! path (no portable `dp4a` equivalent across Intel GPU generations) — both
//! entry points decode weights on the fly and accumulate a plain float dot
//! product, mirroring the CUDA IQ4_NL kernel's simpler design rather than
//! IQ4_XS's int8 one. Only F32/F16 outputs are supported, matching the rest
//! of the SYCL fused-op surface (`ops/fused_ops/sycl_impl.rs`); a BF16
//! request falls back to [`crate::quantized::iquant::IQuantLinear`]'s generic
//! CPU dequantize path.

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

use crate::quantized::iquant::IQuantType;

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_iq4_matvec_sycl(
        queue: *mut c_void,
        is_xs: i32,
        dtype: i32,
        packed: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        input_rows: i32,
        output_rows: i32,
        cols: i32,
    ) -> i32;

    fn crane_iq4_dequant_sycl(
        queue: *mut c_void,
        is_xs: i32,
        dtype: i32,
        packed: *const c_void,
        output: *mut c_void,
        n_rows: i32,
        cols: i32,
    ) -> i32;
}

fn dtype_tag(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        other => candle_core::bail!("i-quant SYCL kernels do not support {other:?} output"),
    }
}

fn byte_ptr(storage: &Storage, byte_offset: usize, name: &str) -> Result<*const c_void> {
    match storage {
        Storage::Sycl(st) => {
            Ok(unsafe { (st.buf().as_ptr() as *const u8).add(byte_offset) as *const c_void })
        },
        _ => candle_core::bail!("i-quant SYCL kernel: {name} must be a sycl tensor"),
    }
}

/// `input` (`[rows, cols]`, SYCL, any float dtype) times the transposed
/// packed weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// Meant for decode-sized `rows`; prefill should use [`dequantize`] and a
/// regular matmul instead (see `IQuantLinear::forward_chunked`).
///
/// # Errors
///
/// Returns an error if the tensors are not on a SYCL device, `out_dtype`
/// isn't F32/F16, or the kernel launch fails.
pub fn matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let dev = input.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();
    let dtype = dtype_tag(out_dtype)?;
    let is_xs = i32::from(matches!(ty, IQuantType::Iq4Xs));

    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let rows = input.elem_count() / cols;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_ptr = byte_ptr(
        &input_storage,
        input_layout.start_offset() * DType::F32.size_in_bytes(),
        "matvec input",
    )?;
    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let packed_ptr = byte_ptr(
        &packed_storage,
        packed_layout.start_offset(),
        "matvec weight",
    )?;

    let out_el = rows * output_rows;
    let out_buf = dev.alloc_bytes(out_el * out_dtype.size_in_bytes())?;

    let status = unsafe {
        crane_iq4_matvec_sycl(
            queue,
            is_xs,
            dtype,
            packed_ptr,
            input_ptr,
            out_buf.as_mut_ptr(),
            rows as i32,
            output_rows as i32,
            cols as i32,
        )
    };
    drop(input_storage);
    drop(packed_storage);
    if status != 0 {
        candle_core::bail!("crane_iq4_matvec_sycl failed (status {status})");
    }

    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    Ok(Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, out_dtype, out_el)),
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
/// Returns an error if `packed` is not on a SYCL device, `dtype` isn't
/// F32/F16, or the kernel launch fails.
pub fn dequantize(
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = packed.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();
    let dtype_i = dtype_tag(dtype)?;
    let is_xs = i32::from(matches!(ty, IQuantType::Iq4Xs));

    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let packed_ptr = byte_ptr(
        &packed_storage,
        packed_layout.start_offset() + row_start * row_bytes,
        "dequantize weight",
    )?;

    let out_el = n_rows * cols;
    let out_buf = dev.alloc_bytes(out_el * dtype.size_in_bytes())?;

    let status = unsafe {
        crane_iq4_dequant_sycl(
            queue,
            is_xs,
            dtype_i,
            packed_ptr,
            out_buf.as_mut_ptr(),
            n_rows as i32,
            cols as i32,
        )
    };
    drop(packed_storage);
    if status != 0 {
        candle_core::bail!("crane_iq4_dequant_sycl failed (status {status})");
    }

    Ok(Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, dtype, out_el)),
        (n_rows, cols),
        BackpropOp::none(),
        false,
    ))
}
