// SPDX-License-Identifier: MIT

//! Launchers for `kernels/cuda/quant_iq4.cu`.

use candle_core::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaDType, WrapErr};
use candle_core::op::BackpropOp;
use candle_core::{CudaDevice, CudaStorage, DType, Result, Storage, Tensor, bail};

use crate::quantized::iquant::IQuantType;

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_quant_iq4";
/// Output rows per thread block in the matvec kernels (one warp each).
const WARPS: usize = 4;

fn packed_slice(packed: &Tensor) -> Result<(std::sync::RwLockReadGuard<'_, Storage>, usize)> {
    let (storage, layout) = packed.storage_and_layout();
    if !matches!(&*storage, Storage::Cuda(_)) {
        bail!("i-quant packed weight must be a CUDA u8 tensor")
    }
    Ok((storage, layout.start_offset()))
}

/// `input` (`[rows, cols]`, CUDA f32/f16/bf16) times the transposed packed
/// weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// IQ4_XS quantizes the activations to int8 (one scale per 32 values) and
/// runs an integer `dp4a` dot product, like llama.cpp, writing the output
/// directly in `out_dtype`. IQ4_NL uses a float kernel. Meant for
/// decode-sized `rows`; prefill should use [`dequantize`] and a regular
/// matmul instead.
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
        IQuantType::Iq4Xs => {
            let dev = input.device().as_cuda_device()?.clone();
            match out_dtype {
                DType::F32 => matvec_xs::<f32>(&dev, input, packed, output_rows, cols, out_dtype),
                DType::F16 => {
                    matvec_xs::<half::f16>(&dev, input, packed, output_rows, cols, out_dtype)
                },
                DType::BF16 => {
                    matvec_xs::<half::bf16>(&dev, input, packed, output_rows, cols, out_dtype)
                },
                other => bail!("i-quant matvec: unsupported output dtype {other:?}"),
            }
        },
    }
}

fn launch_config(rows: usize, output_rows: usize) -> (usize, LaunchConfig) {
    let nr = if rows >= 4 { 4 } else { 1 };
    let config = LaunchConfig {
        grid_dim: (
            output_rows.div_ceil(WARPS) as u32,
            rows.div_ceil(nr) as u32,
            1,
        ),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    (nr, config)
}

fn output_tensor<T: CudaDType + DeviceRepr>(
    buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    dev: CudaDevice,
    input: &Tensor,
    output_rows: usize,
) -> Tensor {
    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev)),
        dims,
        BackpropOp::none(),
        false,
    )
}

fn matvec_xs<T: CudaDType + DeviceRepr>(
    dev: &CudaDevice,
    input: &Tensor,
    packed: &Tensor,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let rows = input.elem_count() / cols;
    let n_blocks = rows * cols / 32;
    let (nr, config) = launch_config(rows, output_rows);

    // int8 activations followed by one f32 scale per 32 of them; `rows * cols`
    // is a multiple of 256, so the scales stay 4-byte aligned.
    let scratch = unsafe { dev.alloc::<u8>(rows * cols + 4 * n_blocks) }?;
    let xq = scratch.slice(..rows * cols);
    let xd = scratch.slice(rows * cols..);
    {
        let (input_storage, input_layout) = input.storage_and_layout();
        let Storage::Cuda(input_cuda) = &*input_storage else {
            bail!("i-quant matvec input must be on CUDA")
        };
        let offset = input_layout.start_offset();
        let quantize = match input.dtype() {
            DType::F32 => "iq4_quantize_q8_f32",
            DType::F16 => "iq4_quantize_q8_f16",
            DType::BF16 => "iq4_quantize_q8_bf16",
            other => bail!("i-quant kernels do not support {other:?} activations"),
        };
        let func = dev.get_or_load_custom_func(quantize, MODULE_NAME, ptx::QUANT_IQ4)?;
        let n_blocks_i = n_blocks as i32;
        let (x_f32, x_f16, x_bf16);
        let mut builder = func.builder();
        match input.dtype() {
            DType::F32 => {
                x_f32 = input_cuda.as_cuda_slice::<f32>()?.slice(offset..);
                builder.arg(&x_f32);
            },
            DType::F16 => {
                x_f16 = input_cuda.as_cuda_slice::<half::f16>()?.slice(offset..);
                builder.arg(&x_f16);
            },
            _ => {
                x_bf16 = input_cuda.as_cuda_slice::<half::bf16>()?.slice(offset..);
                builder.arg(&x_bf16);
            },
        }
        builder.arg(&xq);
        builder.arg(&xd);
        builder.arg(&n_blocks_i);
        // 8 warps per thread block, one warp per 32 activations.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n_blocks.div_ceil(8) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }

    let kernel = match (nr, out_dtype) {
        (1, DType::F32) => "iq4_xs_matvec_q8_1_f32",
        (1, DType::F16) => "iq4_xs_matvec_q8_1_f16",
        (1, _) => "iq4_xs_matvec_q8_1_bf16",
        (_, DType::F32) => "iq4_xs_matvec_q8_4_f32",
        (_, DType::F16) => "iq4_xs_matvec_q8_4_f16",
        _ => "iq4_xs_matvec_q8_4_bf16",
    };
    let (packed_storage, packed_offset) = packed_slice(packed)?;
    let Storage::Cuda(packed_cuda) = &*packed_storage else {
        unreachable!()
    };
    let packed_slice = packed_cuda.as_cuda_slice::<u8>()?.slice(packed_offset..);
    let output_buf = unsafe { dev.alloc::<T>(rows * output_rows) }?;
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, ptx::QUANT_IQ4)?;
    let rows_i = rows as i32;
    let output_rows_i = output_rows as i32;
    let cols_i = cols as i32;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&xq);
    builder.arg(&xd);
    builder.arg(&output_buf);
    builder.arg(&rows_i);
    builder.arg(&output_rows_i);
    builder.arg(&cols_i);
    unsafe { builder.launch(config) }.w()?;
    drop(packed_storage);
    Ok(output_tensor(output_buf, dev.clone(), &input, output_rows))
}

fn matvec_nl_f32(
    input: &Tensor,
    packed: &Tensor,
    output_rows: usize,
    cols: usize,
) -> Result<Tensor> {
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let rows = input.elem_count() / cols;
    let dev = input.device().as_cuda_device()?.clone();
    let (nr, config) = launch_config(rows, output_rows);
    let (input_storage, input_layout) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => bail!("i-quant matvec input must be CUDA f32"),
    };
    let input_slice = input_slice.slice(input_layout.start_offset()..);
    let (packed_storage, packed_offset) = packed_slice(packed)?;
    let Storage::Cuda(packed_cuda) = &*packed_storage else {
        unreachable!()
    };
    let packed_slice = packed_cuda.as_cuda_slice::<u8>()?.slice(packed_offset..);
    let output_buf = unsafe { dev.alloc::<f32>(rows * output_rows) }?;
    let kernel = if nr == 1 {
        "iq4_nl_matvec1_f32"
    } else {
        "iq4_nl_matvec4_f32"
    };
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, ptx::QUANT_IQ4)?;
    let rows_i = rows as i32;
    let output_rows_i = output_rows as i32;
    let cols_i = cols as i32;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&input_slice);
    builder.arg(&output_buf);
    builder.arg(&rows_i);
    builder.arg(&output_rows_i);
    builder.arg(&cols_i);
    unsafe { builder.launch(config) }.w()?;
    drop(input_storage);
    drop(packed_storage);
    Ok(output_tensor(output_buf, dev, &input, output_rows))
}

/// Expand weight rows `row_start..row_start + n_rows` of a packed
/// `[_, cols]` tensor into a dense `[n_rows, cols]` tensor of `dtype`.
pub fn dequantize(
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = packed.device().as_cuda_device()?.clone();
    match dtype {
        DType::F32 => dequantize_as::<f32>(&dev, packed, ty, row_start, n_rows, cols, "f32"),
        DType::F16 => dequantize_as::<half::f16>(&dev, packed, ty, row_start, n_rows, cols, "f16"),
        DType::BF16 => {
            dequantize_as::<half::bf16>(&dev, packed, ty, row_start, n_rows, cols, "bf16")
        },
        other => bail!("i-quant dequantize: unsupported output dtype {other:?}"),
    }
}

fn dequantize_as<T: CudaDType + DeviceRepr>(
    dev: &CudaDevice,
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    suffix: &str,
) -> Result<Tensor> {
    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    let (packed_storage, packed_offset) = packed_slice(packed)?;
    let Storage::Cuda(packed_cuda) = &*packed_storage else {
        unreachable!()
    };
    let start = packed_offset + row_start * row_bytes;
    let packed_slice = packed_cuda
        .as_cuda_slice::<u8>()?
        .slice(start..start + n_rows * row_bytes);

    let output_buf = unsafe { dev.alloc::<T>(n_rows * cols) }?;
    // One warp per 256 weights: an IQ4_XS block or eight IQ4_NL blocks.
    let n_blocks = n_rows * cols / ty.block_size();
    let warps = n_rows * cols / 256 + usize::from(!(n_rows * cols).is_multiple_of(256));
    let kernel = match (ty, suffix) {
        (IQuantType::Iq4Xs, "f32") => "iq4_xs_dequant_f32",
        (IQuantType::Iq4Xs, "f16") => "iq4_xs_dequant_f16",
        (IQuantType::Iq4Xs, _) => "iq4_xs_dequant_bf16",
        (IQuantType::Iq4Nl, "f32") => "iq4_nl_dequant_f32",
        (IQuantType::Iq4Nl, "f16") => "iq4_nl_dequant_f16",
        (IQuantType::Iq4Nl, _) => "iq4_nl_dequant_bf16",
    };
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, ptx::QUANT_IQ4)?;
    let n_blocks_i = n_blocks as i32;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&output_buf);
    builder.arg(&n_blocks_i);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (warps.div_ceil(8) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    drop(packed_storage);

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output_buf, dev.clone())),
        (n_rows, cols),
        BackpropOp::none(),
        false,
    ))
}
