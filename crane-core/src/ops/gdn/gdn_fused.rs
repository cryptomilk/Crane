//! Launchers for the fused GDN stages in `kernels/cuda/gdn_fused.cu` (bf16
//! activations, or f32 for single-token decode; head dim 128, batch 1).
//!
//! Short passes through a GDN layer -- speculative-decoding verification
//! blocks, prefill chunks -- are dispatch-bound: the op-by-op conv / split /
//! norm / gates / layout chain is ~45 launches per layer. Three kernels replace
//! it. They reproduce the op chain's rounding points, so the layer output is
//! the same; `GatedDeltaNet::forward` falls back to the op chain whenever a
//! precondition ([`supported`]) does not hold.

use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use half::bf16;

use super::config::{GdnDims, VHeadOrder};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE: &str = "crane_gdn_fused";
const HEAD_DIM: usize = 128;
const CONV_K: usize = 4;
/// Warps (value heads) per block in the prep / post kernels.
const WARPS: usize = 4;

/// Whether the fused path can run for this layer and input: CUDA, batch 1,
/// 128-wide heads and a 4-tap conv; bf16 for any length, f32 (GGUF decode)
/// for single tokens only. Single-token decode remains opt-in.
pub fn supported(x: &Tensor, dims: &GdnDims) -> bool {
    x.device().is_cuda()
        && x.dim(0).is_ok_and(|b| b == 1)
        && x.dim(1).is_ok_and(|s| {
            let decode = s == 1 && std::env::var("CRANE_GDN_FUSED").is_ok_and(|v| v == "1");
            match x.dtype() {
                DType::BF16 => s > 1 || decode,
                DType::F32 => decode,
                _ => false,
            }
        })
        && dims.head_k_dim == HEAD_DIM
        && dims.head_v_dim == HEAD_DIM
        && dims.conv_kernel_size == CONV_K
        && std::env::var("CRANE_GDN_PORTABLE").is_err()
        && std::env::var("CRANE_GDN_FUSED").map_or(true, |v| v != "0")
}

/// Borrow a CUDA tensor's data as a device view starting at its layout offset.
macro_rules! cuda_view {
    ($name:ident, $guard:ident, $tensor:expr, $ty:ty) => {
        let ($guard, layout) = $tensor.storage_and_layout();
        let $name = match &*$guard {
            Storage::Cuda(c) => c
                .as_cuda_slice::<$ty>()
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "gdn fused: `{}` is {:?}, expected {}: {e}",
                        stringify!($tensor),
                        $tensor.dtype(),
                        stringify!($ty)
                    ))
                })?
                .slice(layout.start_offset()..),
            _ => candle_core::bail!("gdn fused: expected a CUDA tensor"),
        };
    };
}

fn wrap<T: candle_core::cuda_backend::CudaDType>(
    buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    dev: &candle_core::CudaDevice,
    shape: impl Into<candle_core::Shape>,
) -> Tensor {
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev.clone())),
        shape,
        BackpropOp::none(),
        false,
    )
}

/// Result of [`conv_silu`].
pub struct ConvOut {
    /// `[S, C]` in the input dtype, silu already applied.
    pub y: Tensor,
    /// `[1, C, 4]` conv state after all `S` tokens.
    pub state: Tensor,
    /// State after the first `ckpt_at` tokens, if one was requested.
    pub ckpt: Option<Tensor>,
}

/// Causal conv1d + silu over `x: [S, C]` (contiguous bf16) continuing from
/// `state: [1, C, 4]` (contiguous bf16). `weight: [C, 4]` is f32, already
/// rounded through bf16 as the op-by-op path does.
///
/// # Errors
///
/// Returns an error if a tensor is not a contiguous CUDA tensor of the
/// expected dtype or the launch fails.
pub fn conv_silu(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    ckpt_at: Option<usize>,
) -> Result<ConvOut> {
    conv_silu_with_numerics(x, state, weight, ckpt_at, false)
}

/// CUDA extension precision: accumulate and activate in F32, then round once.
pub fn conv_silu_with_numerics(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    ckpt_at: Option<usize>,
    fla_numerics: bool,
) -> Result<ConvOut> {
    let (s, c) = x.dims2()?;
    let dev = x.device().as_cuda_device()?.clone();
    cuda_view!(x_v, x_g, x, bf16);
    cuda_view!(state_v, state_g, state, bf16);
    cuda_view!(w_v, w_g, weight, f32);
    let y = unsafe { dev.alloc::<bf16>(s * c) }?;
    let new_state = unsafe { dev.alloc::<bf16>(c * CONV_K) }?;
    let want_ckpt = ckpt_at.filter(|&p| p > 0 && p < s);
    let ckpt_buf = match want_ckpt {
        Some(_) => Some(unsafe { dev.alloc::<bf16>(c * CONV_K) }?),
        None => None,
    };

    let func = dev.get_or_load_custom_func("gdn_conv_silu_bf16", MODULE, ptx::GDN_FUSED)?;
    let (s_i, c_i, xs_i, p_i) = (s as i32, c as i32, c as i32, want_ckpt.unwrap_or(0) as i32);
    let null_ckpt;
    let mut b = func.builder();
    b.arg(&x_v);
    b.arg(&state_v);
    b.arg(&w_v);
    b.arg(&y);
    b.arg(&new_state);
    match &ckpt_buf {
        Some(buf) => {
            b.arg(buf);
        },
        None => {
            null_ckpt = 0u64; // nullptr
            b.arg(&null_ckpt);
        },
    }
    b.arg(&s_i);
    b.arg(&c_i);
    b.arg(&xs_i);
    b.arg(&p_i);
    let modern = i32::from(fla_numerics);
    b.arg(&modern);
    let cfg = if fla_numerics {
        LaunchConfig {
            grid_dim: ((c as u32).div_ceil(256), s as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig::for_num_elems(c as u32)
    };
    unsafe { b.launch(cfg) }.w()?;

    Ok(ConvOut {
        y: wrap(y, &dev, (s, c)),
        state: wrap(new_state, &dev, (1, c, CONV_K)),
        ckpt: ckpt_buf.map(|buf| wrap(buf, &dev, (1, c, CONV_K))),
    })
}

/// Single-token convolution + SiLU. `x` and `state` are both BF16 (rounded
/// through BF16 before the FP32 SiLU) or both F32 (GGUF decode); `weight` is
/// F32 and `y`/`state` come back in the input dtype.
///
/// # Errors
///
/// Rejects invalid shapes, layouts, dtypes or devices before launching CUDA.
pub(super) fn decode_conv_silu(x: &Tensor, state: &Tensor, weight: &Tensor) -> Result<ConvOut> {
    let (s, c) = x.dims2()?;
    if s != 1 || c == 0 || c > i32::MAX as usize / CONV_K {
        candle_core::bail!("gdn decode conv: expected a nonempty [1, C] input");
    }
    if state.dims() != [1, c, CONV_K] || weight.dims() != [c, CONV_K] {
        candle_core::bail!("gdn decode conv: expected state [1, C, 4] and weight [C, 4]");
    }
    if !x.is_contiguous() || !state.is_contiguous() || !weight.is_contiguous() {
        candle_core::bail!("gdn decode conv: inputs must be contiguous");
    }
    if !x.device().same_device(state.device()) || !x.device().same_device(weight.device()) {
        candle_core::bail!("gdn decode conv: inputs must be on the same CUDA device");
    }
    if state.dtype() != x.dtype() {
        candle_core::bail!("gdn decode conv: x and state must share a dtype");
    }
    match x.dtype() {
        DType::BF16 => decode_conv_launch::<bf16>(x, state, weight, "gdn_decode_conv_silu_bf16"),
        DType::F32 => decode_conv_launch::<f32>(x, state, weight, "gdn_decode_conv_silu_f32"),
        d => candle_core::bail!("gdn decode conv: unsupported dtype {d:?}"),
    }
}

fn decode_conv_launch<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    kernel: &str,
) -> Result<ConvOut> {
    let c = x.dim(1)?;
    let dev = x.device().as_cuda_device()?.clone();
    cuda_view!(x_v, x_g, x, T);
    cuda_view!(state_v, state_g, state, T);
    cuda_view!(w_v, w_g, weight, f32);
    let y = unsafe { dev.alloc::<T>(c) }?;
    let new_state = unsafe { dev.alloc::<T>(c * CONV_K) }?;
    let func =
        dev.get_or_load_custom_func(kernel, "crane_gdn_decode_conv", ptx::GDN_DECODE_CONV)?;
    let c_i = c as i32;
    let mut launch = func.builder();
    launch.arg(&x_v);
    launch.arg(&state_v);
    launch.arg(&w_v);
    launch.arg(&y);
    launch.arg(&new_state);
    launch.arg(&c_i);
    unsafe { launch.launch(LaunchConfig::for_num_elems(c as u32)) }.w()?;
    Ok(ConvOut {
        y: wrap(y, &dev, (1, c)),
        state: wrap(new_state, &dev, (1, c, CONV_K)),
        ckpt: None,
    })
}

/// Tensors for the recurrence kernel, covering one run of tokens.
pub struct Prepped {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub g: Tensor,
    pub beta: Tensor,
}

/// Token stride (elements between consecutive tokens) of a `[1, S, H]` tensor
/// whose head dim is packed, or `None` if it is not.
fn token_stride(t: &Tensor) -> Option<usize> {
    let layout = t.layout();
    (layout.dims().len() == 3 && layout.stride()[2] == 1).then(|| layout.stride()[1])
}

/// Split Q|K|V of `conv: [S, C]`, L2-normalise Q and K, scale Q, derive the
/// gates from `a`/`b` (`[1, S, H]` bf16, any token stride) and lay everything
/// out as the recurrence kernel reads it, for tokens `t0 .. t0 + len`.
///
/// # Errors
///
/// Returns an error if an input is not a CUDA tensor with packed trailing
/// dims, or the launch fails.
#[allow(clippy::too_many_arguments)]
pub fn prep(
    conv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    neg_exp_a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    t0: usize,
    len: usize,
    scale: f32,
) -> Result<Prepped> {
    prep_with_numerics(
        conv,
        a,
        b,
        neg_exp_a_log,
        dt_bias,
        dims,
        t0,
        len,
        scale,
        false,
    )
}

/// Prepare current FLA-style FP32-square/BF16-output Q/K normalization.
#[allow(clippy::too_many_arguments)]
pub fn prep_with_numerics(
    conv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    neg_exp_a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    t0: usize,
    len: usize,
    scale: f32,
    fla_numerics: bool,
) -> Result<Prepped> {
    if a.dtype() != conv.dtype() || b.dtype() != conv.dtype() {
        candle_core::bail!("gdn fused prep: conv, a and b must share a dtype");
    }
    let (nea, dt) = (neg_exp_a_log, dt_bias);
    match conv.dtype() {
        DType::BF16 => prep_launch::<bf16>(
            conv,
            a,
            b,
            nea,
            dt,
            dims,
            t0,
            len,
            scale,
            fla_numerics,
            "gdn_prep_bf16",
        ),
        DType::F32 => prep_launch::<f32>(
            conv,
            a,
            b,
            nea,
            dt,
            dims,
            t0,
            len,
            scale,
            fla_numerics,
            "gdn_prep_f32",
        ),
        d => candle_core::bail!("gdn fused prep: unsupported dtype {d:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn prep_launch<T: candle_core::cuda_backend::CudaDType>(
    conv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    neg_exp_a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    t0: usize,
    len: usize,
    scale: f32,
    fla_numerics: bool,
    kernel: &str,
) -> Result<Prepped> {
    let c = conv.dim(1)?;
    let h = dims.num_v_heads;
    let dev = conv.device().as_cuda_device()?.clone();
    let (Some(as_), Some(bs_)) = (token_stride(a), token_stride(b)) else {
        candle_core::bail!("gdn fused prep: a/b must have packed head dims");
    };
    cuda_view!(conv_v, conv_g, conv, T);
    cuda_view!(a_v, a_g, a, T);
    cuda_view!(b_v, b_g, b, T);
    cuda_view!(nea_v, nea_g, neg_exp_a_log, f32);
    cuda_view!(dt_v, dt_g, dt_bias, f32);
    let q3 = unsafe { dev.alloc::<f32>(h * len * HEAD_DIM) }?;
    let k3 = unsafe { dev.alloc::<f32>(h * len * HEAD_DIM) }?;
    let v3 = unsafe { dev.alloc::<f32>(h * len * HEAD_DIM) }?;
    let g2 = unsafe { dev.alloc::<f32>(h * len) }?;
    let beta2 = unsafe { dev.alloc::<f32>(h * len) }?;

    let func = dev.get_or_load_custom_func(kernel, MODULE, ptx::GDN_FUSED)?;
    let order = i32::from(dims.v_head_order == VHeadOrder::Chunked);
    let ints = [
        t0 as i32,
        len as i32,
        h as i32,
        dims.num_k_heads as i32,
        dims.v_per_group as i32,
        order,
        dims.key_dim as i32,
        c as i32,
        as_ as i32,
        bs_ as i32,
    ];
    let mut bl = func.builder();
    bl.arg(&conv_v);
    bl.arg(&a_v);
    bl.arg(&b_v);
    bl.arg(&nea_v);
    bl.arg(&dt_v);
    bl.arg(&q3);
    bl.arg(&k3);
    bl.arg(&v3);
    bl.arg(&g2);
    bl.arg(&beta2);
    for i in &ints {
        bl.arg(i);
    }
    bl.arg(&scale);
    let modern = i32::from(fla_numerics);
    bl.arg(&modern);
    let decode = i32::from(conv.dim(0)? == 1);
    bl.arg(&decode);
    let cfg = LaunchConfig {
        grid_dim: (len as u32, h.div_ceil(WARPS) as u32, 1),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { bl.launch(cfg) }.w()?;

    Ok(Prepped {
        q: wrap(q3, &dev, (h, len, HEAD_DIM)),
        k: wrap(k3, &dev, (h, len, HEAD_DIM)),
        v: wrap(v3, &dev, (h, len, HEAD_DIM)),
        g: wrap(g2, &dev, (h, len)),
        beta: wrap(beta2, &dev, (h, len)),
    })
}

/// The `[S, H*128]` output of [`post_norm`], in the activation dtype.
pub enum PostBuf {
    Bf16(candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>),
    F32(candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>),
}

/// Allocate the output of [`post_norm`] in `like`'s dtype (bf16 or f32).
///
/// # Errors
///
/// Returns an error if the allocation fails or the dtype is unsupported.
pub fn alloc_post_out(like: &Tensor, seq_len: usize, h: usize) -> Result<PostBuf> {
    let dev = like.device().as_cuda_device()?;
    let n = seq_len * h * HEAD_DIM;
    match like.dtype() {
        DType::BF16 => Ok(PostBuf::Bf16(unsafe { dev.alloc::<bf16>(n) }?)),
        DType::F32 => Ok(PostBuf::F32(unsafe { dev.alloc::<f32>(n) }?)),
        d => candle_core::bail!("gdn fused post: unsupported dtype {d:?}"),
    }
}

/// Recurrence output `y: [H, len, 128]` f32 for tokens `t0 .. t0 + len` ->
/// activation dtype -> gated RMSNorm with `z: [1, S, H, 128]` -> rows of `out`.
///
/// # Errors
///
/// Returns an error if `z` is not packed per head or the launch fails.
#[allow(clippy::too_many_arguments)]
pub fn post_norm(
    out: &PostBuf,
    y: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    t0: usize,
    len: usize,
    h: usize,
) -> Result<()> {
    match (out, z.dtype()) {
        (PostBuf::Bf16(o), DType::BF16) => {
            post_norm_launch::<bf16>(o, y, z, weight, eps, t0, len, h, "gdn_post_norm_bf16")
        },
        (PostBuf::F32(o), DType::F32) => {
            post_norm_launch::<f32>(o, y, z, weight, eps, t0, len, h, "gdn_post_norm_f32")
        },
        (_, d) => candle_core::bail!("gdn fused post: z is {d:?}, which doesn't match the output"),
    }
}

#[allow(clippy::too_many_arguments)]
fn post_norm_launch<T: candle_core::cuda_backend::CudaDType>(
    out: &candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    y: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    t0: usize,
    len: usize,
    h: usize,
    kernel: &str,
) -> Result<()> {
    let dev = y.device().as_cuda_device()?.clone();
    let zs = z.layout().stride()[1];
    if z.layout().stride()[2] != HEAD_DIM || z.layout().stride()[3] != 1 {
        candle_core::bail!("gdn fused post: z must be packed per head");
    }
    cuda_view!(y_v, y_g, y, f32);
    cuda_view!(z_v, z_g, z, T);
    cuda_view!(w_v, w_g, weight, f32);
    let func = dev.get_or_load_custom_func(kernel, MODULE, ptx::GDN_FUSED)?;
    let (t0_i, len_i, h_i, zs_i) = (t0 as i32, len as i32, h as i32, zs as i32);
    let mut b = func.builder();
    b.arg(&y_v);
    b.arg(&z_v);
    b.arg(&w_v);
    b.arg(out);
    b.arg(&t0_i);
    b.arg(&len_i);
    b.arg(&h_i);
    b.arg(&zs_i);
    b.arg(&eps);
    let cfg = LaunchConfig {
        grid_dim: (len as u32, h.div_ceil(WARPS) as u32, 1),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.w()?;
    Ok(())
}

/// Wrap the buffer filled by [`post_norm`] as a `[S, H*128]` tensor.
pub fn post_out_tensor(buf: PostBuf, like: &Tensor, seq_len: usize, h: usize) -> Result<Tensor> {
    let dev = like.device().as_cuda_device()?.clone();
    let shape = (seq_len, h * HEAD_DIM);
    Ok(match buf {
        PostBuf::Bf16(b) => wrap(b, &dev, shape),
        PostBuf::F32(b) => wrap(b, &dev, shape),
    })
}

#[cfg(test)]
mod decode_conv_tests {
    use super::*;
    use candle_core::Device;

    fn values(t: &Tensor) -> Result<Vec<bf16>> {
        t.flatten_all()?.to_vec1::<bf16>()
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn decode_conv_matches_reference_and_preserves_history() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        // Pin both the BF16 halfway tie and the cancellation-sensitive sum tree.
        for (history, input, expected) in [
            ([0f32, 1., 1. / 256., 0.], 0f32, 0.73046875),
            ([0f32, 16777216., 1., -16777216.], 1f32, 1.7578125),
        ] {
            let x = Tensor::new(&[[input]], &dev)?.to_dtype(DType::BF16)?;
            let state = Tensor::new(&[[history]], &dev)?.to_dtype(DType::BF16)?;
            let weight = Tensor::ones((1, 4), DType::F32, &dev)?;
            let out = decode_conv_silu(&x, &state, &weight)?;
            assert_eq!(values(&out.y)?, vec![bf16::from_f32(expected)]);
        }
        for c in [1, 7, 6144] {
            // Nonzero storage offsets must be respected by all three inputs.
            let mut state = Tensor::arange(0f32, (c * CONV_K + 4) as f32, &dev)?
                .affine(0.003, -1.)?
                .to_dtype(DType::BF16)?
                .narrow(0, 4, c * CONV_K)?
                .reshape((1, c, CONV_K))?;
            let weight = Tensor::arange(0f32, (c * CONV_K + 4) as f32, &dev)?
                .affine(0.0001, -0.5)?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .narrow(0, 4, c * CONV_K)?
                .reshape((c, CONV_K))?;
            for step in 0..8 {
                let x = Tensor::arange(0f32, (c + 1) as f32, &dev)?
                    .affine(0.002, step as f64 * 0.01)?
                    .to_dtype(DType::BF16)?
                    .narrow(0, 1, c)?
                    .reshape((1, c))?;
                let original = values(&state)?;
                let window = Tensor::cat(&[&state.narrow(2, 1, 3)?, &x.reshape((1, c, 1))?], 2)?;
                let sum = window
                    .to_dtype(DType::F32)?
                    .broadcast_mul(&weight.unsqueeze(0)?)?
                    .sum_keepdim(2)?
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?;
                let expected = candle_nn::ops::silu(&sum)?
                    .to_dtype(DType::BF16)?
                    .reshape((1, c))?;
                let out = decode_conv_silu(&x, &state, &weight)?;
                assert_eq!(out.y.dims(), &[1, c]);
                assert_eq!(values(&out.y)?, values(&expected)?, "C={c}, step={step}");
                assert_eq!(values(&out.state)?, values(&window)?);
                assert_eq!(values(&state)?, original, "input state was modified");
                assert!(out.ckpt.is_none());
                state = out.state;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn decode_conv_rejects_invalid_buffers() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::zeros((1, 7), DType::BF16, &dev)?;
        let state = Tensor::zeros((1, 7, 4), DType::BF16, &dev)?;
        let weight = Tensor::ones((7, 4), DType::F32, &dev)?;
        assert!(decode_conv_silu(&x, &state.narrow(2, 0, 3)?, &weight).is_err());
        assert!(decode_conv_silu(&x, &state, &weight.transpose(0, 1)?).is_err());
        assert!(decode_conv_silu(&x.to_dtype(DType::F32)?, &state, &weight).is_err());
        assert!(decode_conv_silu(&x, &state, &weight.to_device(&Device::Cpu)?).is_err());
        assert!(decode_conv_silu(&x.narrow(1, 0, 0)?, &state, &weight).is_err());
        assert!(decode_conv_silu(&x.broadcast_as((2, 7))?, &state, &weight).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn decode_conv_f32_matches_reference_and_preserves_history() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let f32s = |t: &Tensor| t.flatten_all()?.to_vec1::<f32>();
        for c in [1, 7, 6144] {
            let mut state = Tensor::arange(0f32, (c * CONV_K + 4) as f32, &dev)?
                .affine(0.003, -1.)?
                .narrow(0, 4, c * CONV_K)?
                .reshape((1, c, CONV_K))?;
            let weight = Tensor::arange(0f32, (c * CONV_K + 4) as f32, &dev)?
                .affine(0.0001, -0.5)?
                .narrow(0, 4, c * CONV_K)?
                .reshape((c, CONV_K))?;
            for step in 0..8 {
                let x = Tensor::arange(0f32, (c + 1) as f32, &dev)?
                    .affine(0.002, step as f64 * 0.01)?
                    .narrow(0, 1, c)?
                    .reshape((1, c))?;
                let original = f32s(&state)?;
                let window = Tensor::cat(&[&state.narrow(2, 1, 3)?, &x.reshape((1, c, 1))?], 2)?;
                let expected = candle_nn::ops::silu(
                    &window
                        .broadcast_mul(&weight.unsqueeze(0)?)?
                        .sum_keepdim(2)?,
                )?
                .reshape((1, c))?;
                let out = decode_conv_silu(&x, &state, &weight)?;
                assert_eq!(out.y.dtype(), DType::F32);
                assert_eq!(out.y.dims(), &[1, c]);
                let max_abs = (&out.y - &expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(max_abs <= 1e-6, "C={c}, step={step}: max_abs {max_abs}");
                assert_eq!(f32s(&out.state)?, f32s(&window)?);
                assert_eq!(f32s(&state)?, original, "input state was modified");
                assert!(out.ckpt.is_none());
                state = out.state;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn decode_conv_rejects_mixed_dtypes() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::zeros((1, 7), DType::F32, &dev)?;
        let state = Tensor::zeros((1, 7, 4), DType::F32, &dev)?;
        let weight = Tensor::ones((7, 4), DType::F32, &dev)?;
        assert!(decode_conv_silu(&x, &state.to_dtype(DType::BF16)?, &weight).is_err());
        assert!(decode_conv_silu(&x.to_dtype(DType::BF16)?, &state, &weight).is_err());
        assert!(
            decode_conv_silu(
                &x.to_dtype(DType::F16)?,
                &state.to_dtype(DType::F16)?,
                &weight
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn prep_and_post_reject_mixed_dtypes() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let h = 1;
        let dims = GdnDims {
            hidden_size: HEAD_DIM,
            num_k_heads: h,
            num_v_heads: h,
            head_k_dim: HEAD_DIM,
            head_v_dim: HEAD_DIM,
            conv_kernel_size: CONV_K,
            key_dim: HEAD_DIM,
            value_dim: HEAD_DIM,
            conv_dim: 3 * HEAD_DIM,
            v_per_group: 1,
            v_head_order: VHeadOrder::Interleaved,
        };
        let conv = Tensor::zeros((1, dims.conv_dim), DType::F32, &dev)?;
        let ab = Tensor::zeros((1, 1, h), DType::F32, &dev)?;
        let gate = Tensor::zeros(h, DType::F32, &dev)?;
        let bf = |t: &Tensor| t.to_dtype(DType::BF16);
        let prep = |a: &Tensor, b: &Tensor| {
            prep_with_numerics(&conv, a, b, &gate, &gate, &dims, 0, 1, 1., false)
        };
        assert!(prep(&bf(&ab)?, &ab).is_err());
        assert!(prep(&ab, &bf(&ab)?).is_err());

        let y = Tensor::zeros((h, 1, HEAD_DIM), DType::F32, &dev)?;
        let z = Tensor::zeros((1, 1, h, HEAD_DIM), DType::F32, &dev)?;
        let w = Tensor::ones(HEAD_DIM, DType::F32, &dev)?;
        let f32_out = alloc_post_out(&y, 1, h)?;
        assert!(post_norm(&f32_out, &y, &bf(&z)?, &w, 1e-6, 0, 1, h).is_err());
        let bf16_out = alloc_post_out(&bf(&y)?, 1, h)?;
        assert!(post_norm(&bf16_out, &y, &z, &w, 1e-6, 0, 1, h).is_err());
        Ok(())
    }
}
