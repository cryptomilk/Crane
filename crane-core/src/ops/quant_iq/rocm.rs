// SPDX-License-Identifier: MIT

//! Launchers for `kernels/cuda/quant_iq4.cu` (IQ4_XS `dp4a` decode, IQ4
//! dequant) and `kernels/cuda/quant_iq.cu` (every other type's float matvec
//! and dequant, plus the by-expert-id routing packed `MoE` experts need) on
//! ROCm/HIP.
//!
//! The same sources the CUDA build compiles to PTX; here candle compiles
//! them with `hipcc` on first use and caches the code object on disk (see
//! [`crate::ops::rocm`]). Every intrinsic the kernels need (`__byte_perm`,
//! `__dp4a`, the `*_sync` warp shuffles, bf16 conversions, `__popc`) is
//! either native on `ROCm` or already bridged by candle-rocm's HIP shim, so
//! this mirrors [`super::cuda`]'s dispatch exactly, rather than falling back
//! to the float-only approach [`super::metal`]/[`super::sycl`] use.
//! `quant_iq.cu`'s `#include "../sycl/iq_grids.h"` only resolves through
//! CUDA's build-time include path; `ROCm` compiles from the source string
//! `hipcc` is handed, so [`generic_source`] splices the header's text in at
//! that `#include`'s position instead.

use std::ffi::c_void;
use std::sync::OnceLock;

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

/// Module of the type-generic float kernels (`kernels/cuda/quant_iq.cu`).
const GENERIC_MODULE: &str = "crane_quant_iq";
/// `ROWS_PER_BLOCK` in `quant_iq.cu`.
const ROWS_PER_BLOCK: usize = 4;
/// Pair count from which [`matvec_indexed`] switches from the by-id matvec
/// (ids stay on the device) to [`super::indexed_via_gemm`] (one host sync for
/// the ids, then each routed expert is decoded once and multiplied by a
/// batched GEMM). The crossover SYCL, Metal and CUDA use; not tuned on `ROCm`.
const GEMM_MIN_PAIRS: usize = 2048;

/// `kernels/cuda/quant_iq.cu`'s source with its `#include
/// "../sycl/iq_grids.h"` spliced in place of the directive.
///
/// `hipcc` compiles this module from the string `get_or_load_custom_func`
/// hands it rather than from a file on disk, so the project-relative
/// `#include` CUDA's build resolves via its include path would fail to find
/// the header here. Splicing the header's text in keeps the single string
/// self-contained; the `#define inline __device__` / `#undef inline`
/// already bracketing the `#include` in `quant_iq.cu` applies to the
/// spliced text exactly as it would to the real `#include`.
fn generic_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        include_str!("../../../kernels/cuda/quant_iq.cu").replace(
            "#include \"../sycl/iq_grids.h\"",
            include_str!("../../../kernels/sycl/iq_grids.h"),
        )
    })
}

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

/// Device pointer into a flattened `U32` expert-id tensor, plus the storage
/// guard the pointer borrows from (see [`packed_u8_ptr`]'s caller note).
fn ids_u32_ptr(ids: &Tensor) -> Result<(candle_core::StorageRef<'_>, *mut c_void)> {
    let (storage, layout) = ids.storage_and_layout();
    let slice = rocm::rocm_slice(&storage, "expert ids")?;
    if slice.dtype() != DType::U32 {
        bail!("expert ids must be a ROCm U32 tensor");
    }
    // SAFETY: `layout.start_offset()` is within the tensor's own allocation.
    let ptr = unsafe {
        match slice {
            RocmStorageSlice::U32(m) => m.ptr_at(layout.start_offset()),
            _ => unreachable!("dtype checked above"),
        }
    };
    Ok((storage, ptr))
}

/// The type part of the generic kernel names (`IQ_ALL_TYPES` in `quant_iq.cu`).
fn type_tag(ty: IQuantType) -> &'static str {
    match ty {
        IQuantType::Iq4Nl => "iq4_nl",
        IQuantType::Iq4Xs => "iq4_xs",
        IQuantType::Iq2S => "iq2_s",
        IQuantType::Iq3Xxs => "iq3_xxs",
        IQuantType::Iq3S => "iq3_s",
        IQuantType::Q2_0 => "q2_0",
    }
}

fn float_tag(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => bail!("i-quant kernels: unsupported dtype {other:?}"),
    }
}

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{what} {n} exceeds i32")))
}

fn to_u32(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{what} {n} exceeds u32")))
}

/// Warps-per-output-row geometry shared by the int8 and float matvec paths.
fn matvec_geometry(rows: usize, output_rows: usize) -> Result<(usize, Dim3, Dim3)> {
    let nr = if rows >= 4 { 4 } else { 1 };
    let grid = Dim3::new_2d(
        to_u32(output_rows.div_ceil(WARPS), "matvec output rows")?,
        to_u32(rows.div_ceil(nr), "matvec rows")?,
    );
    // WARPS is the compile-time constant 4; `32 * WARPS` never approaches u32::MAX.
    #[allow(clippy::cast_possible_truncation)]
    let block = Dim3::new_1d(32 * WARPS as u32);
    Ok((nr, grid, block))
}

/// `input` (`[rows, cols]`, ROCm f32/f16/bf16) times the transposed packed
/// weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// IQ4_XS quantizes the activations to int8 (one scale per 32 values) and
/// runs an integer `dp4a` dot product, like llama.cpp, writing the output
/// directly in `out_dtype`. Every other type runs `quant_iq.cu`'s
/// type-generic float kernel instead (see [`launch_generic_matvec`]). Meant
/// for decode-sized `rows`; prefill should use [`dequantize`] and a regular
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
        IQuantType::Iq4Xs => matvec_xs(input, packed, output_rows, cols, out_dtype),
        _ => {
            let rows = input.elem_count() / cols;
            let out = launch_generic_matvec(
                input,
                packed,
                ty,
                None,
                1,
                rows,
                output_rows,
                cols,
                out_dtype,
            )?;
            let mut dims = input.dims().to_vec();
            *dims.last_mut().unwrap() = output_rows;
            out.reshape(dims)
        },
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
    let (nr, grid, block) = matvec_geometry(rows, output_rows)?;

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
        let n_blocks_i = to_i32(n_blocks, "matvec n_blocks")?;
        let mut args = [
            arg(&input_ptr),
            arg(&xq_ptr),
            arg(&xd_ptr),
            arg(&n_blocks_i),
        ];
        // One warp per 32 activations, 8 warps per thread block.
        let quantize_grid = to_u32(n_blocks.div_ceil(8), "matvec quantize grid")?;
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
    let rows_i = to_i32(rows, "matvec rows")?;
    let output_rows_i = to_i32(output_rows, "matvec output rows")?;
    let cols_i = to_i32(cols, "matvec cols")?;

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
            // SAFETY: args match `iq4_xs_matvec_q8_{1,4}_<dtype>`'s
            // (packed, xq, xd, out, rows, output_rows, cols) signature; grid
            // covers every (output_row, row) pair via `matvec_geometry`.
            unsafe {
                rocm::launch_2d(&dev, MODULE_NAME, kernel, SOURCE, grid, block, 0, &mut args)?;
            }
            wrap(
                <$elem as RocmElem>::wrap_slice(output_buf),
                &dev,
                (rows, output_rows),
            )
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
    // Every type but IQ4_XS/IQ4_NL only has a `quant_iq.cu` generic kernel.
    if !matches!(ty, IQuantType::Iq4Nl | IQuantType::Iq4Xs) {
        let row_bytes = cols / ty.block_size() * ty.block_bytes();
        return launch_generic_dequant(
            packed,
            row_start * row_bytes,
            ty,
            None,
            1,
            n_rows,
            cols,
            dtype,
        )?
        .reshape((n_rows, cols));
    }
    let dev = packed.device().as_rocm_device()?.clone();
    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    let (packed_s, packed_ptr) = packed_u8_ptr(packed, row_start * row_bytes)?;

    // One warp per 256 weights: an IQ4_XS block or eight IQ4_NL blocks.
    let n_blocks = n_rows * cols / ty.block_size();
    let warps = n_rows * cols / 256 + usize::from(!(n_rows * cols).is_multiple_of(256));
    let grid = to_u32(warps.div_ceil(8), "dequantize grid")?;
    let n_blocks_i = to_i32(n_blocks, "dequantize n_blocks")?;

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

/// `output[p, r] = dot(W_e[r], input[p / x_div])` via `quant_iq.cu`'s
/// type-generic `matvec_{type}_{dtype}` kernels — every type but the IQ4_XS
/// dp4a fast path (see [`matvec`]), and the only `ROCm` path with expert-id
/// routing built in (`ids`/`has_ids`), which [`matvec_indexed`] uses.
#[allow(clippy::too_many_arguments)]
fn launch_generic_matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: Option<&Tensor>,
    x_div: usize,
    pairs: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let dev = input.device().as_rocm_device()?.clone();
    let source = generic_source();
    let name = format!("matvec_{}_{}", type_tag(ty), float_tag(out_dtype)?);
    let expert_stride = (output_rows * (cols / ty.block_size()) * ty.block_bytes()) as u64;
    let x_div_i = to_i32(x_div.max(1), "x_div")?;
    let out_rows_i = to_i32(output_rows, "matvec output rows")?;
    let cols_i = to_i32(cols, "matvec cols")?;
    let has_ids = i32::from(ids.is_some());

    // The generic kernel takes f32 activations (matches the SYCL / Metal
    // generic paths); only the IQ4_XS dp4a fast path quantizes to int8.
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let (input_s, input_l) = input.storage_and_layout();
    let input_ptr = rocm::device_ptr(&input_s, input_l, DType::F32, "i-quant matvec input")?;
    let (packed_s, packed_ptr) = packed_u8_ptr(packed, 0)?;

    let ids_guard = ids.map(ids_u32_ptr).transpose()?;
    let dummy = if ids.is_none() {
        Some(dev.alloc_zeros::<u32>(1)?)
    } else {
        None
    };
    let ids_ptr = match (&ids_guard, &dummy) {
        (Some((_, ptr)), _) => *ptr,
        (None, Some(d)) => d.as_ptr(),
        (None, None) => unreachable!("one of ids_guard/dummy is always Some"),
    };

    let grid = Dim3::new_2d(
        to_u32(output_rows.div_ceil(ROWS_PER_BLOCK), "matvec output rows")?,
        to_u32(pairs, "matvec pairs")?,
    );
    // ROWS_PER_BLOCK is the compile-time constant 4; `32 * ROWS_PER_BLOCK`
    // never approaches u32::MAX.
    #[allow(clippy::cast_possible_truncation)]
    let block = Dim3::new_1d(32 * ROWS_PER_BLOCK as u32);

    macro_rules! run {
        ($elem:ty) => {{
            let output_buf = dev.alloc::<$elem>(pairs * output_rows)?;
            let output_ptr = output_buf.as_ptr();
            let mut args = [
                arg(&packed_ptr),
                arg(&ids_ptr),
                arg(&input_ptr),
                arg(&output_ptr),
                arg(&expert_stride),
                arg(&x_div_i),
                arg(&out_rows_i),
                arg(&cols_i),
                arg(&has_ids),
            ];
            // SAFETY: args match `matvec_<type>_<dtype>`'s (packed, ids,
            // input, out, expert_stride, x_div, output_rows, cols, has_ids)
            // signature; grid covers every (output_row, pair) above.
            unsafe {
                rocm::launch_2d(
                    &dev,
                    GENERIC_MODULE,
                    &name,
                    source,
                    grid,
                    block,
                    0,
                    &mut args,
                )?;
            }
            wrap(
                <$elem as RocmElem>::wrap_slice(output_buf),
                &dev,
                (pairs, output_rows),
            )
        }};
    }

    let out = match out_dtype {
        DType::F32 => run!(f32),
        DType::F16 => run!(half::f16),
        DType::BF16 => run!(half::bf16),
        other => bail!("i-quant matvec: unsupported output dtype {other:?}"),
    };
    drop(input_s);
    drop(packed_s);
    drop(ids_guard);
    Ok(out)
}

/// Decode `n_mats` matrices of `n_rows` x `cols` from `packed` (optionally
/// one per entry of `ids`) into `[n_mats * n_rows, cols]`, via
/// `quant_iq.cu`'s type-generic `dequant_{type}_{dtype}` kernels — every
/// type but IQ4_XS/IQ4_NL (see [`dequantize`]), and the only `ROCm` path that
/// can decode a list of experts in one launch (see [`dequantize_experts`]).
#[allow(clippy::too_many_arguments)]
fn launch_generic_dequant(
    packed: &Tensor,
    byte_offset: usize,
    ty: IQuantType,
    ids: Option<&Tensor>,
    n_mats: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = packed.device().as_rocm_device()?.clone();
    let source = generic_source();
    let name = format!("dequant_{}_{}", type_tag(ty), float_tag(dtype)?);
    let expert_stride = (n_rows * (cols / ty.block_size()) * ty.block_bytes()) as u64;
    let n_mats_i = to_i32(n_mats, "dequantize matrices")?;
    let n_rows_i = to_i32(n_rows, "dequantize rows")?;
    let cols_i = to_i32(cols, "dequantize cols")?;
    let has_ids = i32::from(ids.is_some());

    let (packed_s, packed_ptr) = packed_u8_ptr(packed, byte_offset)?;

    // The flattened tensor, not just its storage guard, must outlive the
    // launch, so it is bound to a local instead of dropped as a temporary.
    let ids = ids.map(|t| t.flatten_all()?.contiguous()).transpose()?;
    let ids_guard = ids.as_ref().map(ids_u32_ptr).transpose()?;
    let dummy = if ids.is_none() {
        Some(dev.alloc_zeros::<u32>(1)?)
    } else {
        None
    };
    let ids_ptr = match (&ids_guard, &dummy) {
        (Some((_, ptr)), _) => *ptr,
        (None, Some(d)) => d.as_ptr(),
        (None, None) => unreachable!("one of ids_guard/dummy is always Some"),
    };

    let chunks = cols / 32;
    let grid = Dim3::new_2d(
        to_u32(chunks.div_ceil(256), "dequantize grid")?,
        to_u32(n_mats * n_rows, "dequantize matrices*rows")?,
    );
    let block = Dim3::new_1d(to_u32(chunks.min(256), "dequantize block")?);

    macro_rules! run {
        ($elem:ty) => {{
            let output_buf = dev.alloc::<$elem>(n_mats * n_rows * cols)?;
            let output_ptr = output_buf.as_ptr();
            let mut args = [
                arg(&packed_ptr),
                arg(&ids_ptr),
                arg(&output_ptr),
                arg(&expert_stride),
                arg(&n_mats_i),
                arg(&n_rows_i),
                arg(&cols_i),
                arg(&has_ids),
            ];
            // SAFETY: args match `dequant_<type>_<dtype>`'s (packed, ids,
            // out, expert_stride, n_mats, n_rows, cols, has_ids) signature;
            // grid covers every (chunk, matrix*row) pair above.
            unsafe {
                rocm::launch_2d(
                    &dev,
                    GENERIC_MODULE,
                    &name,
                    source,
                    grid,
                    block,
                    0,
                    &mut args,
                )?;
            }
            wrap(
                <$elem as RocmElem>::wrap_slice(output_buf),
                &dev,
                (n_mats * n_rows, cols),
            )
        }};
    }

    let out = match dtype {
        DType::F32 => run!(f32),
        DType::F16 => run!(half::f16),
        DType::BF16 => run!(half::bf16),
        other => bail!("i-quant dequantize: unsupported output dtype {other:?}"),
    };
    drop(packed_s);
    drop(ids_guard);
    Ok(out)
}

/// Decode experts `ids` (`U32`, on the device) of a packed `[experts, rows,
/// cols]` tensor into a dense `[ids.len(), rows, cols]` tensor of `dtype`,
/// in one launch.
///
/// # Errors
///
/// Returns an error if the tensors are not on a ROCm device, `dtype` isn't
/// F32/F16/BF16, or the kernel launch fails.
pub fn dequantize_experts(
    packed: &Tensor,
    ty: IQuantType,
    ids: &Tensor,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let n = ids.elem_count();
    launch_generic_dequant(packed, 0, ty, Some(ids), n, rows, cols, dtype)?.reshape((n, rows, cols))
}

/// `MoE` matmul by expert id: for each of the `ids.len()` pairs `p`, row `p`
/// of the result is expert `ids[p]`'s `[output_rows, cols]` matrix (from the
/// packed `[experts, output_rows, cols]` tensor) times activation row
/// `p / x_div` of `input` (`[_, cols]`). Returns `[ids.len(), output_rows]`.
///
/// Below [`GEMM_MIN_PAIRS`] pairs `ids` is never read on the host.
///
/// # Errors
///
/// Returns an error if the tensors are not on a ROCm device, `ids` is not
/// `U32`, `out_dtype` isn't F32/F16/BF16, or the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn matvec_indexed(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: &Tensor,
    x_div: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    if ids.dtype() != DType::U32 {
        bail!("expert ids must be U32, got {:?}", ids.dtype())
    }
    let ids = ids.flatten_all()?.contiguous()?;
    let pairs = ids.elem_count();
    if pairs >= GEMM_MIN_PAIRS {
        return super::indexed_via_gemm(input, &ids, x_div, output_rows, cols, out_dtype, |e| {
            dequantize_experts(packed, ty, e, output_rows, cols, DType::F16)
        });
    }
    launch_generic_matvec(
        input,
        packed,
        ty,
        Some(&ids),
        x_div,
        pairs,
        output_rows,
        cols,
        out_dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `generic_source` must splice `iq_grids.h`'s body in place of the
    /// `#include`, since hipcc compiles from a string and cannot resolve a
    /// project-relative path the way the CUDA build's include path does.
    #[test]
    fn grids_splice_into_source() {
        let src = generic_source();
        assert!(!src.contains("#include \"../sycl/iq_grids.h\""));
        assert!(src.contains("iq2s_grid"));
        assert!(src.contains("iq3xxs_grid"));
        assert!(src.contains("iq3s_grid"));
    }
}
