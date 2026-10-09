// SPDX-License-Identifier: MIT
//! Backend-agnostic matmul SDPA, usable as a fallback when `gpu_flash_attn`'s
//! fused kernels can't serve a call (CPU device, unsupported `head_dim`, or
//! no `cuda`/`rocm` feature compiled). Metal and SYCL always land here. Plain
//! `candle_core`/`candle_nn` ops only, with no `#[cfg(feature = ...)]` gate
//! anywhere in this file, so every function compiles and runs on every
//! backend. [`decode`] is wired into every model (`GqaAttention`, Qwen3's
//! `Attention`). [`causal_without_mask`]/[`causal_with_mask`]/[`full_matmul`]/
//! [`windowed_matmul`] aren't wired into any model's prefill path yet — they exist
//! as `gpu_flash_attn`'s fallback tier (unsupported `head_dim`, or no
//! `cuda`/`rocm` feature compiled) and as the eventual per-layer prefill
//! path once a model wires one up. `GqaAttention`'s prefill path hand-rolls
//! its own matmul SDPA independently — unlike Qwen3, it documents `None` as
//! meaning full non-causal attention, so neither `causal*` function applies
//! there without an explicit `causal` flag (a future commit).
//!
//! **[`causal_with_mask`] vs. [`causal_without_mask`]: pick based on whether
//! the mask is shared.** [`causal_without_mask`] builds a fresh
//! `O(q_len * kv_len)` mask (and, on GPU, re-uploads it) on every call —
//! correct for an occasional caller (`gpu_flash_attn`'s fallback, where no
//! shared mask exists), but wrong for a decoder stack calling it once per
//! layer per forward pass, where every layer shares the same
//! `q_len`/`kv_len`/`kv_offset` and so the exact same mask: rebuilding it
//! per layer would regress GPU prefill throughput measurably the same way
//! upcasting `attn_weights` did (see below) — build the mask once per
//! forward pass and call [`causal_with_mask`] with it instead, once a
//! caller needs this. [`full_matmul`]/[`windowed_matmul`] only have a
//! self-building form so far (reachable via `gpu_flash_attn`'s
//! `try_full`/`try_windowed` fallback, no direct per-layer model call site
//! yet) — add a `_with_mask` sibling if one ever gets a per-layer caller.
//!
//! [`causal_without_mask`], [`causal_with_mask`], [`full_matmul`], and
//! [`windowed_matmul`] mirror `gpu_flash_attn`'s `try_causal`/`try_full`/`try_windowed` in shape
//! and BHSD convention, but always succeed (no `Option`) and never require a
//! specific `head_dim` or backend. [`decode`] covers the `seq_len == 1`
//! case: CPU single-sequence decode delegates to the existing CPU
//! flash-attn dispatch (`super::flash_attn::dispatch_flash_attn`);
//! everything else (GPU, or CPU with more than one sequence) uses a
//! GQA-grouped reshape that avoids expanding K/V.
//!
//! **Softmax runs in native dtype, not F32.** An earlier version of
//! `prefill_sdpa` upcast the full `[B, H_q, q_len, kv_len]` score tensor to
//! F32 around softmax, matching `GqaAttention`'s existing prefill behavior.
//! That tensor is `H_q` times larger than the mask, and upcasting it
//! measurably regressed GPU prefill throughput for long contexts (confirmed
//! via `crane-serve`); seeing it caught on real hardware this early, before
//! more callers piled onto this fallback, is why `prefill_sdpa` now casts
//! only the mask (to `attn_weights`' native dtype — `NEG_INFINITY`/`0.0` are
//! exact in every float format, so this loses nothing) instead. [`decode`]'s
//! GQA-grouped-matmul path never upcast either, for the same reason: this
//! matches Qwen3's prior GPU decode path for every `n_rep`, and matches
//! `GqaAttention`'s prior GPU decode path only for `n_rep > 1`;
//! `GqaAttention`'s `n_rep == 1` decode previously fell through to its own
//! (separate, untouched) F32-upcasting prefill SDPA, so this consolidation
//! drops that upcast for the `n_rep == 1` decode case.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::attention::AttnMask;
use candle_nn::ops::softmax_last_dim;

use super::flash_attn::dispatch_flash_attn;
use crate::models::utils::{CausalMask, build_additive_causal_mask, repeat_kv};

/// Attention scale factor `1 / sqrt(head_dim)`, shared so callers don't each
/// repeat the `f64`-intermediate cast and its `#[allow]`s.
#[must_use]
pub fn attention_scale(head_dim: usize) -> f32 {
    // head_dim is a small positive integer (attention head sizes, typically
    // 64..256), exactly representable in f32.
    #[allow(clippy::cast_precision_loss)]
    {
        1.0 / (head_dim as f32).sqrt()
    }
}

/// Builds a `[1, 1, q_len, kv_len]` additive mask where position `(i, j)` is
/// `0` when `i + kv_offset - window_left <= j <= i + kv_offset + window_right`
/// and `f32::NEG_INFINITY` otherwise. Shares `kv_offset`'s meaning, and the
/// per-call allocation caveat, with [`build_additive_causal_mask`].
// Only reachable via `windowed`, not wired into any model yet either -
// see this module's doc comment.
#[allow(dead_code)]
fn build_windowed_mask(
    q_len: usize,
    kv_len: usize,
    kv_offset: usize,
    window_left: usize,
    window_right: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            // q_len/kv_len/window bounds are sequence lengths (far below
            // i64::MAX), so these never wrap; i64 is needed since `rel` can
            // be negative.
            #[allow(clippy::cast_possible_wrap)]
            let (center, j_i64, window_left_i64, window_right_i64) = (
                (i + kv_offset) as i64,
                j as i64,
                window_left as i64,
                window_right as i64,
            );
            let rel = j_i64 - center;
            let valid = rel >= -window_left_i64 && rel <= window_right_i64;
            if !valid {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_len), device)
}

/// Shared prefill SDPA core for [`causal_without_mask`], [`causal_with_mask`],
/// [`full`], and [`windowed`]: `repeat_kv` GQA expansion,
/// `Q @ K^T * scale [+ mask]`, native-dtype softmax, `@ V`. `q`/`k`/`v` are
/// BHSD; the mask (if any) is additive, built in F32 by this module's
/// mask-building functions, and broadcastable to `[B, H_q, q_len, kv_len]`.
/// Returns BHSD.
///
/// The mask is cast to `attn_weights`' native dtype rather than upcasting
/// `attn_weights` to F32: `NEG_INFINITY`/`0.0` are exactly representable in
/// every float format, so this is lossless, and it avoids upcasting the
/// `[B, H_q, q_len, kv_len]` score tensor — `H_q` times larger than the mask
/// — which measurably regressed GPU prefill throughput (confirmed via
/// `crane-serve`) for the long-context case this tier exists to serve.
fn prefill_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let n_rep = q.dim(1)? / k.dim(1)?;
    let k = repeat_kv(k.clone(), n_rep)?.contiguous()?;
    let v = repeat_kv(v.clone(), n_rep)?.contiguous()?;
    // Q may be non-contiguous after the caller's transpose(1, 2) into BHSD.
    let q = q.contiguous()?;

    let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * f64::from(scale))?;
    let attn_weights = match mask {
        Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
        None => attn_weights,
    };
    let attn_weights = softmax_last_dim(&attn_weights)?;

    attn_weights.matmul(&v)
}

/// Causal matmul SDPA, building its own mask internally on every call:
/// `j <= i + kv_offset`, where `kv_offset = k`'s sequence length minus
/// `q`'s. Mirrors
/// [`gpu_flash_attn::try_causal`](super::gpu_flash_attn::try_causal)'s mask
/// semantics and BHSD/GQA shape conventions, but always succeeds (no
/// `Option`) and never requires a specific `head_dim` or backend.
///
/// Named `_without_mask` (rather than a plain `causal`) deliberately, paired
/// with [`causal_with_mask`]: a per-layer caller reaching for the shorter
/// name by habit is exactly the mistake that regressed Qwen3's prefill
/// throughput once already (see this module's doc comment). Only reach for
/// this function when no shared mask exists for the call (e.g.
/// `gpu_flash_attn`'s fallback tier) — a decoder stack calling this once per
/// layer per forward pass should build the mask once itself and call
/// [`causal_with_mask`] instead.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible (GQA
/// head-count divisibility, matching `head_dim`) or if any tensor op fails.
pub fn causal_without_mask(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let q_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let kv_offset = kv_len.saturating_sub(q_len);
    let mask = build_additive_causal_mask(q_len, kv_len, kv_offset, DType::F32, q.device())?;
    prefill_sdpa(q, k, v, scale, Some(&mask))
}

/// Causal matmul SDPA using a caller-provided, already-built additive mask
/// (see [`build_additive_causal_mask`]) instead of building one internally. Exists
/// for callers that share one mask across multiple layers in a single
/// forward pass — every layer in a decoder stack sees the same
/// `q_len`/`kv_len`/`kv_offset`, so building the mask once per forward pass
/// and reusing it here is correct and avoids [`causal_without_mask`]'s
/// per-call `O(q_len * kv_len)` allocation (and, on GPU, a host->device
/// re-upload) repeated once per layer. Otherwise identical to
/// [`causal_without_mask`]: same math, same BHSD/GQA shape conventions.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible (GQA
/// head-count divisibility, matching `head_dim`), if `mask` isn't
/// broadcastable to `[B, H_q, q_len, kv_len]`, or if any tensor op fails.
// Not wired into any model's prefill path yet - see this module's doc
// comment.
#[allow(dead_code)]
pub(crate) fn causal_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: &CausalMask,
) -> Result<Tensor> {
    prefill_sdpa(q, k, v, scale, Some(mask.as_tensor()))
}

/// Non-causal (full, bidirectional) matmul-only SDPA: every query attends to
/// every key. See [`causal_without_mask`] for the error contract and shape
/// conventions. Internal to `modules/` — [`gpu_flash_attn`](super::gpu_flash_attn)'s
/// `try_full` fallback is the only caller; model code should call `full`
/// (the dispatcher) instead.
///
/// # Errors
///
/// See [`causal_without_mask`].
pub(super) fn full_matmul(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    prefill_sdpa(q, k, v, scale, None)
}

/// Sliding-window matmul-only SDPA: a query at `i` sees keys `j` with
/// `i + kv_offset - window_left <= j <= i + kv_offset + window_right`
/// (`kv_offset` computed the same way as [`causal_without_mask`]). See
/// [`causal_without_mask`] for the error contract and shape conventions.
/// Internal to `modules/` — [`gpu_flash_attn`](super::gpu_flash_attn)'s
/// `try_windowed` fallback is the only caller; model code should call
/// `windowed` (the dispatcher) instead.
///
/// # Errors
///
/// See [`causal_without_mask`].
// Not wired into any model's prefill path yet - see this module's doc
// comment.
#[allow(dead_code)]
pub(super) fn windowed_matmul(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window_left: usize,
    window_right: usize,
) -> Result<Tensor> {
    let q_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let kv_offset = kv_len.saturating_sub(q_len);
    let mask = build_windowed_mask(
        q_len,
        kv_len,
        kv_offset,
        window_left,
        window_right,
        q.device(),
    )?;
    prefill_sdpa(q, k, v, scale, Some(&mask))
}

/// CPU single-sequence decode: BHSD->BSHD, `dispatch_flash_attn`, cast the
/// kernel's F32 output back to `q`'s dtype, return BHSD. Mirrors the
/// CPU-flash branch `GqaAttention`/Qwen3's decode paths used to hand-roll.
fn decode_cpu_flash(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let q_bshd = q.transpose(1, 2)?;
    let k_bshd = k.transpose(1, 2)?;
    let v_bshd = v.transpose(1, 2)?;

    let attn_mask = match mask {
        Some(mask) => {
            debug_assert!(
                mask.dim(1).is_ok_and(|d| d == 1),
                "CPU flash-attn decode broadcasts one mask row across all heads; \
                 a mask with head dim != 1 would be silently misapplied"
            );
            // AttnMask::Mask takes ownership. Tensor is Arc-backed, so this
            // is a refcount bump, not a data copy.
            AttnMask::Mask(mask.clone())
        },
        None => AttnMask::None,
    };

    let out = dispatch_flash_attn(&q_bshd, &k_bshd, &v_bshd, scale, attn_mask)?;
    // dispatch_flash_attn always accumulates and returns F32 regardless of
    // input dtype.
    out.to_dtype(q.dtype())
}

/// GQA-grouped decode matmul: reshapes `q` to `[B, H_kv, n_rep, D]` and dots
/// against `k` without expanding K/V, avoiding the `repeat_kv` cost for a
/// single query position. No F32 softmax upcast — see this module's doc
/// comment for which prior GPU decode paths this does and doesn't match.
/// Returns BHSD `[B, H_q, 1, D]`.
fn decode_grouped_matmul(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b_sz, h_q, _one, head_dim) = q.dims4()?;
    let h_kv = k.dim(1)?;
    let n_rep = h_q / h_kv;

    let q_g = (q.reshape((b_sz, h_kv, n_rep, head_dim))? * f64::from(scale))?;
    let k_t = k.transpose(2, 3)?;
    let attn_weights = q_g.matmul(&k_t)?;
    let attn_weights = match mask {
        Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
        None => attn_weights,
    };
    let attn_weights = softmax_last_dim(&attn_weights)?;
    let attn_output = attn_weights.matmul(v)?;

    attn_output.reshape((b_sz, h_q, 1, head_dim))
}

/// Single-token decode SDPA. `q` is `[B, H_q, 1, D]`, `k`/`v` are
/// `[B, H_kv, kv_len, D]` (BHSD), `H_q` a multiple of `H_kv`. `attention_mask`
/// is an optional additive mask broadcastable to `[B, 1, 1, kv_len]` (e.g.
/// continuous-batching padding).
///
/// On a single-sequence CPU call (`q.dim(0) == 1 && q.device().is_cpu()`),
/// delegates to the existing CPU flash-attn dispatch. Every other case
/// (GPU, or CPU with more than one sequence) uses a GQA-grouped reshape that
/// dots `q` against `k` without expanding K/V. Returns BHSD `[B, H_q, 1, D]`.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible or if any
/// tensor op fails.
pub fn decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.dim(0)? == 1 && q.device().is_cpu() {
        return decode_cpu_flash(q, k, v, scale, attention_mask);
    }
    decode_grouped_matmul(q, k, v, scale, attention_mask)
}

/// Reshapes BHSD attention output `[B, H, S, D]` to `[B, S, H*D]` for the
/// output projection, the common step every [`causal`]/[`full`]/[`windowed`]/
/// [`decode`] caller performs afterward. Handles both prefill (`S > 1`,
/// needs a transpose) and decode (`S == 1`, a direct reshape since `H` and
/// `D` are already contiguous).
///
/// # Errors
///
/// Returns a candle error if `attn_output` isn't 4-dimensional or if the
/// reshape/transpose/contiguous ops fail.
pub fn merge_heads(attn_output: &Tensor) -> Result<Tensor> {
    let (b_sz, _h, s, _d) = attn_output.dims4()?;
    if s == 1 {
        attn_output.reshape((b_sz, 1, ()))
    } else {
        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, s, ()))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// `softmax(q @ k^T * scale [+ mask]) @ v` in F32 on the CPU, BHSD
    /// layout, with explicit GQA expansion. This is the ground truth every
    /// test in this module checks against.
    fn naive_attention_bhsd(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
        kv_offset: usize,
        window: Option<(usize, usize)>,
    ) -> Vec<f32> {
        let (b, hq, sq, d) = q.dims4().unwrap();
        let (_, hkv, skv, _) = k.dims4().unwrap();
        let n_rep = hq / hkv;

        let expand_kv = |t: &Tensor| {
            t.unsqueeze(2)
                .unwrap()
                .expand((b, hkv, n_rep, skv, d))
                .unwrap()
                .reshape((b, hq, skv, d))
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let k = expand_kv(k);
        let v = expand_kv(v);

        let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * f64::from(scale)).unwrap();
        let scores = if !causal && window.is_none() {
            scores
        } else {
            let mut mask_vals = vec![0f32; sq * skv];
            for i in 0..sq {
                for j in 0..skv {
                    let rel = j as i64 - (i as i64 + kv_offset as i64);
                    let valid = match window {
                        Some((left, right)) => rel >= -(left as i64) && rel <= right as i64,
                        None => rel <= 0,
                    };
                    mask_vals[i * skv + j] = if valid { 0.0 } else { f32::NEG_INFINITY };
                }
            }
            let mask = Tensor::from_vec(mask_vals, (1, 1, sq, skv), q.device()).unwrap();
            scores.broadcast_add(&mask).unwrap()
        };
        let probs = softmax_last_dim(&scores).unwrap();
        probs
            .matmul(&v)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn assert_close(got: &Tensor, want: &[f32], tol: f32) {
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "[{i}] got {g}, want {w}"
            );
        }
    }

    fn randn(shape: (usize, usize, usize, usize), device: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, device).unwrap()
    }

    // causal_without_mask() with q_len == kv_len must match a naive causal reference.
    #[test]
    fn causal_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 4, 2, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, true, 0, None);
        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_eq!(got.dims(), &[b, hq, s, d]);
        assert_close(&got, &want, 1e-4);
    }

    // causal_without_mask() with kv_len > q_len must apply the shifted diagonal, not a
    // naive j <= i mask, matching continuation prefill against a KV cache.
    #[test]
    fn causal_with_kv_offset_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, true, kv_offset, None);
        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // causal_with_mask() given the exact mask build_causal_mask() builds
    // must match causal_without_mask()'s own (self-built) output byte-for-byte — this is
    // the shared-mask path Qwen3's decode() relies on to avoid rebuilding
    // the mask once per layer, so it must compute exactly the same thing as
    // the per-call path it replaces there.
    #[test]
    fn causal_with_mask_matches_causal() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = causal_without_mask(&q, &k, &v, scale).unwrap();
        let mask = CausalMask::new(sq, skv, kv_offset, DType::F32, &device).unwrap();
        let got = causal_with_mask(&q, &k, &v, scale, &mask).unwrap();
        assert_close(
            &got,
            &want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            0.0,
        );
    }

    // full_matmul() applies no mask at all.
    #[test]
    fn full_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 2, 2, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = full_matmul(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // windowed_matmul() restricts each query to a band around its shifted position.
    #[test]
    fn windowed_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 4, 12, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, kv_offset, Some((3, 0)));
        let got = windowed_matmul(&q, &k, &v, scale, 3, 0).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // GQA ratio > 1 must be handled correctly by causal_without_mask()'s repeat_kv expansion.
    #[test]
    fn causal_gqa_expansion_correct() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 8, 2, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, true, 0, None);
        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // decode() on a single CPU sequence must match dispatch_flash_attn directly.
    #[test]
    fn decode_cpu_single_sequence_matches_flash_attn() {
        let device = Device::Cpu;
        let (b, hq, hkv, skv, d) = (1, 4, 2, 7, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, 1, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = decode(&q, &k, &v, scale, None).unwrap();
        assert_eq!(got.dims(), &[b, hq, 1, d]);
        assert_close(&got, &want, 1e-4);
    }

    // decode() with b_sz > 1 on CPU must take the grouped-matmul path (not
    // the single-sequence flash path, which only supports b_sz == 1) and
    // still produce correct output.
    #[test]
    fn decode_cpu_multi_sequence_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, skv, d) = (2, 8, 2, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, 1, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = decode(&q, &k, &v, scale, None).unwrap();
        assert_eq!(got.dims(), &[b, hq, 1, d]);
        assert_close(&got, &want, 1e-4);
    }

    // decode() with n_rep == 1 (no GQA) must still work through the
    // grouped-matmul path's reshape, which is a no-op fold in this case.
    #[test]
    fn decode_n_rep_one() {
        let device = Device::Cpu;
        let (b, h, skv, d) = (2, 4, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, h, 1, d), &device);
        let k = randn((b, h, skv, d), &device);
        let v = randn((b, h, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = decode(&q, &k, &v, scale, None).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // An explicit mask passed to decode() must change the output, on both
    // the CPU single-sequence path and the grouped-matmul path.
    #[test]
    fn decode_explicit_mask_changes_output() {
        let device = Device::Cpu;
        let (b, hq, hkv, skv, d) = (1, 4, 2, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, 1, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let y_no_mask = decode(&q, &k, &v, scale, None).unwrap();

        let mut mask_data = vec![-1e9f32; skv];
        mask_data[0] = 0.0;
        let mask = Tensor::from_vec(mask_data, (1, 1, 1, skv), &device).unwrap();
        let y_masked = decode(&q, &k, &v, scale, Some(&mask)).unwrap();

        let diff: f32 = (y_no_mask - y_masked)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 1e-4, "explicit mask must change decode output");
    }

    // Cross-check: for n_rep == 1 (no GQA complication), causal_without_mask()'s shifted
    // mask must agree with the CPU flash-attn path's AttnMask::Causal on the
    // exact same kv_offset semantics. This guards against the causal formula
    // drifting out of sync between the two independent implementations.
    #[test]
    fn causal_matches_cpu_flash_attn_causal_mask() {
        let device = Device::Cpu;
        let (b, h, sq, skv, d) = (1, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, h, sq, d), &device);
        let k = randn((b, h, skv, d), &device);
        let v = randn((b, h, skv, d), &device);

        let q_bshd = q.transpose(1, 2).unwrap();
        let k_bshd = k.transpose(1, 2).unwrap();
        let v_bshd = v.transpose(1, 2).unwrap();
        // dispatch_flash_attn takes BSHD and returns BHSD directly.
        let flash_out = dispatch_flash_attn(
            &q_bshd,
            &k_bshd,
            &v_bshd,
            scale,
            AttnMask::Causal { kv_offset },
        )
        .unwrap();
        let want = flash_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-3);
    }

    #[test]
    // Verifies merge_heads's decode path (S == 1): a direct reshape with no
    // transpose, since H and D are already contiguous.
    fn merge_heads_decode_single_token() {
        let device = Device::Cpu;
        let (b, h, d) = (2, 4, 8usize);
        let attn_output = randn((b, h, 1, d), &device);

        let got = merge_heads(&attn_output).unwrap();
        assert_eq!(got.dims(), &[b, 1, h * d]);

        let want = attn_output
            .reshape((b, 1, h * d))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_flat = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got_flat, want);
    }

    #[test]
    // Verifies merge_heads's prefill path (S > 1): BHSD -> BSHD transpose
    // before flattening heads, matching the hand-rolled boilerplate it
    // replaces.
    fn merge_heads_prefill_multi_token() {
        let device = Device::Cpu;
        let (b, h, s, d) = (2, 4, 3, 8usize);
        let attn_output = randn((b, h, s, d), &device);

        let got = merge_heads(&attn_output).unwrap();
        assert_eq!(got.dims(), &[b, s, h * d]);

        let want = attn_output
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, s, h * d))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_flat = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got_flat, want);
    }
}
