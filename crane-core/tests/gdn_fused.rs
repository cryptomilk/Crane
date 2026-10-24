//! The fused path of `GatedDeltaNet` (CUDA: `gdn_fused.cu`, bf16; Metal:
//! `gdn_fused.metal`, f16 and bf16) against the op-by-op chain it replaces:
//! same layer output, same conv / recurrent / checkpoint state, for several
//! lengths, head groupings and checkpoint splits. On Metal the reference is
//! the fully portable chain, per-timestep recurrence included.

#![cfg(any(feature = "cuda", feature = "metal"))]

use std::collections::HashMap;

use crane_core::candle_core::{DType, Device, Result, Tensor};
use crane_core::candle_nn::VarBuilder;
use crane_core::ops::gdn::{
    GateActivation, GatedDeltaNet, GdnConfig, GdnDims, GdnInputProjectionKind, GdnLayerCache,
};

struct Cfg {
    hidden: usize,
    k_heads: usize,
    v_heads: usize,
}

impl GdnConfig for Cfg {
    fn hidden_size(&self) -> usize {
        self.hidden
    }
    fn rms_norm_eps(&self) -> f64 {
        1e-6
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        4
    }
    fn linear_key_head_dim(&self) -> usize {
        128
    }
    fn linear_value_head_dim(&self) -> usize {
        128
    }
    fn linear_num_key_heads(&self) -> usize {
        self.k_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.v_heads
    }
    fn output_gate_activation(&self) -> Result<GateActivation> {
        Ok(GateActivation::Silu)
    }
}

/// The GPU this build targets and the activation dtypes its fused path takes.
fn gpu() -> (Device, &'static [DType]) {
    #[cfg(feature = "cuda")]
    return (Device::new_cuda(0).unwrap(), &[DType::BF16]);
    #[cfg(not(feature = "cuda"))]
    (Device::new_metal(0).unwrap(), &[DType::F16, DType::BF16])
}

fn rand(dev: &Device, shape: &[usize], std: f64) -> Tensor {
    Tensor::randn(0f32, std as f32, shape, dev).unwrap()
}

fn layer(dev: &Device, cfg: &Cfg, dt: DType) -> (GatedDeltaNet, GdnDims) {
    let dims = GdnDims::new(cfg);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    let mut put = |k: &str, t: Tensor| {
        w.insert(format!("linear_attn.{k}"), t);
    };
    put(
        "in_proj_qkv.weight",
        rand(dev, &[dims.conv_dim, cfg.hidden], 0.04),
    );
    put(
        "in_proj_z.weight",
        rand(dev, &[dims.value_dim, cfg.hidden], 0.04),
    );
    put(
        "in_proj_b.weight",
        rand(dev, &[dims.num_v_heads, cfg.hidden], 0.04),
    );
    put(
        "in_proj_a.weight",
        rand(dev, &[dims.num_v_heads, cfg.hidden], 0.04),
    );
    put("conv1d.weight", rand(dev, &[dims.conv_dim, 1, 4], 0.5));
    put("dt_bias.weight", Tensor::zeros(1, DType::F32, dev).unwrap());
    put("dt_bias", rand(dev, &[dims.num_v_heads], 1.0));
    put(
        "A_log",
        (rand(dev, &[dims.num_v_heads], 0.5).abs().unwrap() + 0.5).unwrap(),
    );
    put("norm.weight", (rand(dev, &[128], 0.1) + 1.0).unwrap());
    put(
        "out_proj.weight",
        rand(dev, &[cfg.hidden, dims.value_dim], 0.04),
    );
    let w = w
        .into_iter()
        .map(|(k, t)| (k, t.to_dtype(dt).unwrap()))
        .collect();
    let vb = VarBuilder::from_tensors(w, dt, dev);
    let gdn = GatedDeltaNet::load(&vb, cfg, GdnInputProjectionKind::Split, None).unwrap();
    (gdn, dims)
}

fn run(
    gdn: &GatedDeltaNet,
    dims: &GdnDims,
    x: &Tensor,
    cache: &GdnLayerCache,
    split: Option<usize>,
    fused: bool,
) -> (Tensor, GdnLayerCache) {
    run_with(gdn, dims, x, cache, split, fused, true)
}

fn run_with(
    gdn: &GatedDeltaNet,
    dims: &GdnDims,
    x: &Tensor,
    cache: &GdnLayerCache,
    split: Option<usize>,
    fused: bool,
    serial: bool,
) -> (Tensor, GdnLayerCache) {
    // SAFETY: single-threaded test body; nothing else reads the environment.
    unsafe { std::env::set_var("CRANE_GDN_FUSED", if fused { "1" } else { "0" }) };
    #[cfg(not(feature = "cuda"))]
    if fused {
        unsafe { std::env::remove_var("CRANE_GDN_PORTABLE") };
    } else {
        unsafe { std::env::set_var("CRANE_GDN_PORTABLE", "1") };
    }
    // Each side needs independent storage: recurrence backends may mutate state.
    let mut c = cache.clone();
    // These tests retain coverage of the legacy fused/portable implementations.
    c.fla_numerics = false;
    c.conv_state = cache.conv_state.copy().unwrap();
    c.recurrent_state = cache.recurrent_state.copy().unwrap();
    c.checkpoint = cache
        .checkpoint
        .as_ref()
        .map(|(conv, rec)| (conv.copy().unwrap(), rec.copy().unwrap()));
    c.checkpoint_at = split;
    c.prefer_serial = serial;
    let y = gdn.forward(x, dims, &mut c).unwrap();
    (y, c)
}

/// The production default must preserve nonzero state, speculative rollback
/// and decode when switching from chunk FLA to the single-token recurrence.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU; run with --test-threads=1"]
fn cuda_default_fla_state_and_decode() {
    for name in [
        "CRANE_GDN_FLA",
        "CRANE_GDN_FUSED",
        "CRANE_GDN_PORTABLE",
        "CRANE_GDN_SERIAL_PREFILL",
        "CRANE_GDN_SERIAL_MAX",
    ] {
        unsafe { std::env::remove_var(name) };
    }
    let (dev, _) = gpu();
    for (v_heads, order) in [
        (4, crane_core::ops::gdn::VHeadOrder::Interleaved),
        (8, crane_core::ops::gdn::VHeadOrder::Interleaved),
        (8, crane_core::ops::gdn::VHeadOrder::Chunked),
    ] {
        let cfg = Cfg {
            hidden: 512,
            k_heads: 4,
            v_heads,
        };
        let (gdn, mut dims) = layer(&dev, &cfg, DType::BF16);
        dims.v_head_order = order;
        let mut base = GdnLayerCache::new(&cfg, DType::BF16, &dev).unwrap();
        assert!(base.fla_numerics);
        assert!(
            !GdnLayerCache::new(&cfg, DType::F32, &dev)
                .unwrap()
                .fla_numerics
        );
        let prefix = rand(&dev, &[1, 17, cfg.hidden], 1.0)
            .to_dtype(DType::BF16)
            .unwrap();
        gdn.forward(&prefix, &dims, &mut base).unwrap();
        let x = rand(&dev, &[1, 130, cfg.hidden], 1.0)
            .to_dtype(DType::BF16)
            .unwrap();
        // Deep copies: the optimized recurrence may update storage in place.
        let copy = |cache: &GdnLayerCache| {
            let mut c = cache.clone();
            c.conv_state = cache.conv_state.copy().unwrap();
            c.recurrent_state = cache.recurrent_state.copy().unwrap();
            c.checkpoint_at = Some(65);
            c
        };
        let mut actual = copy(&base);
        let mut reference = copy(&base);
        let y = gdn.forward(&x, &dims, &mut actual).unwrap();
        unsafe { std::env::set_var("CRANE_GDN_PORTABLE", "1") };
        let expected = gdn.forward(&x, &dims, &mut reference).unwrap();
        unsafe { std::env::remove_var("CRANE_GDN_PORTABLE") };
        let (_, _, ymax) = diff(&y, &expected);
        let (_, _, rmax) = diff(&actual.recurrent_state, &reference.recurrent_state);
        println!("default FLA HV={v_heads} {order:?}: output max={ymax:.3e}, state max={rmax:.3e}");
        assert_eq!(diff(&actual.conv_state, &reference.conv_state).0, 0);
        // Different BF16 accumulation graphs; also tested against Torch FLA
        // by Soma's shared-tensor parity fixture and real backbone regression.
        assert!(ymax < 0.05 && rmax < 0.01);
        let (conv, state) = actual.checkpoint.as_ref().unwrap();
        let saved_conv = conv.copy().unwrap();
        let saved_state = state.copy().unwrap();
        actual.take_checkpoint();
        assert_eq!(diff(&actual.conv_state, &saved_conv).0, 0);
        assert_eq!(diff(&actual.recurrent_state, &saved_state).0, 0);
        // Compare decode algorithms from identical restored state, rather than
        // carrying the different prefill accumulation graphs into this check.
        reference = actual.clone();
        reference.conv_state = actual.conv_state.copy().unwrap();
        reference.recurrent_state = actual.recurrent_state.copy().unwrap();
        let token = x.narrow(1, 65, 1).unwrap();
        let decoded = gdn.forward(&token, &dims, &mut actual).unwrap();
        unsafe { std::env::set_var("CRANE_GDN_PORTABLE", "1") };
        let expected = gdn.forward(&token, &dims, &mut reference).unwrap();
        unsafe { std::env::remove_var("CRANE_GDN_PORTABLE") };
        assert!(diff(&decoded, &expected).2 < 0.05);
        assert_eq!(diff(&actual.conv_state, &reference.conv_state).0, 0);
        assert!(diff(&actual.recurrent_state, &reference.recurrent_state).2 < 1e-5);
    }
    unsafe { std::env::set_var("CRANE_GDN_FLA", "0") };
    let cfg = Cfg {
        hidden: 512,
        k_heads: 4,
        v_heads: 4,
    };
    assert!(
        !GdnLayerCache::new(&cfg, DType::BF16, &dev)
            .unwrap()
            .fla_numerics
    );
    unsafe { std::env::remove_var("CRANE_GDN_FLA") };
}

fn diff(a: &Tensor, b: &Tensor) -> (usize, usize, f32) {
    let a = a
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = b
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(&b).all(|v| v.is_finite()),
        "nonfinite tensor"
    );
    let mismatched = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    (mismatched, a.len(), max)
}

#[test]
#[ignore = "needs a CUDA or Metal GPU"]
fn fused_matches_op_chain() {
    let (dev, dtypes) = gpu();
    for &dt in dtypes {
        for cfg in [
            Cfg {
                hidden: 512,
                k_heads: 4,
                v_heads: 4,
            },
            Cfg {
                hidden: 512,
                k_heads: 4,
                v_heads: 8,
            },
        ] {
            let (gdn, dims) = layer(&dev, &cfg, dt);
            // A realistic non-zero state: run a prefix through the op chain first.
            let mut base = GdnLayerCache::new(&cfg, dt, &dev).unwrap();
            let prefix = rand(&dev, &[1, 11, cfg.hidden], 1.0).to_dtype(dt).unwrap();
            let (_, warmed) = run(&gdn, &dims, &prefix, &base, None, false);
            base = warmed;

            for (seq, split) in [
                (1usize, None),
                (2, None),
                (7, None),
                (20, None),
                (20, Some(1)),
                (20, Some(5)),
                (16, Some(15)),
                (3, Some(2)),
            ] {
                let x = rand(&dev, &[1, seq, cfg.hidden], 1.0).to_dtype(dt).unwrap();
                let (ya, ca) = run(&gdn, &dims, &x, &base, split, false);
                let (yb, cb) = run(&gdn, &dims, &x, &base, split, true);
                let (yd, yn, ymax) = diff(&ya, &yb);
                let (cd, cn, cmax) = diff(&ca.conv_state, &cb.conv_state);
                let (rd, rn, rmax) = diff(&ca.recurrent_state, &cb.recurrent_state);
                println!(
                    "{dt:?} v_heads {} S={seq:<2} split={split:?}: out {yd}/{yn} (max {ymax:.3e}) | conv {cd}/{cn} (max {cmax:.1e}) | state {rd}/{rn} (max {rmax:.1e})",
                    cfg.v_heads
                );
                if let (Some((cc_a, rc_a)), Some((cc_b, rc_b))) = (&ca.checkpoint, &cb.checkpoint) {
                    let (d1, n1, m1) = diff(cc_a, cc_b);
                    let (d2, n2, m2) = diff(rc_a, rc_b);
                    println!(
                        "      checkpoint: conv {d1}/{n1} (max {m1:.1e}) | state {d2}/{n2} (max {m2:.1e})"
                    );
                    assert_eq!(d1, 0, "checkpoint conv state differs");
                    assert!(m2 < 1e-3, "checkpoint recurrent state differs: {m2}");
                } else {
                    assert_eq!(
                        ca.checkpoint.is_some(),
                        cb.checkpoint.is_some(),
                        "checkpoint presence differs"
                    );
                }
                assert_eq!(cd, 0, "conv state differs");
                assert!(rmax < 1e-3, "recurrent state differs: {rmax}");
                assert!(ymax < 0.05, "layer output differs: {ymax}");
            }
        }
    }
}

/// Per-pass cost of one layer, fused vs op chain (prints; also proves the two
/// runs above really took different paths).
#[test]
#[ignore = "needs a CUDA or Metal GPU"]
fn fused_is_faster_per_pass() {
    let (dev, dtypes) = gpu();
    let dt = dtypes[0];
    let cfg = Cfg {
        hidden: 1024,
        k_heads: 16,
        v_heads: 16,
    };
    let (gdn, dims) = layer(&dev, &cfg, dt);
    let base = GdnLayerCache::new(&cfg, dt, &dev).unwrap();
    // CUDA's fused path starts at two tokens; Metal's takes decode steps too.
    let passes: &[(usize, Option<usize>)] = if cfg!(feature = "cuda") {
        &[(16, None), (20, Some(5))]
    } else {
        &[(1, None), (16, None), (20, Some(5))]
    };
    for &(seq, split) in passes {
        let x = rand(&dev, &[1, seq, cfg.hidden], 1.0).to_dtype(dt).unwrap();
        let mut ms = [0f64; 2];
        for (i, fused) in [false, true].into_iter().enumerate() {
            let warm = std::time::Instant::now();
            while warm.elapsed().as_secs_f64() < 0.5 {
                run(&gdn, &dims, &x, &base, split, fused);
                dev.synchronize().unwrap();
            }
            let t = std::time::Instant::now();
            for _ in 0..100 {
                run(&gdn, &dims, &x, &base, split, fused);
            }
            dev.synchronize().unwrap();
            ms[i] = t.elapsed().as_secs_f64() * 10.0;
        }
        println!(
            "S={seq} split={split:?}: op chain {:.3} ms/layer, fused {:.3} ms/layer ({:.2}x)",
            ms[0],
            ms[1],
            ms[0] / ms[1]
        );
        assert!(ms[1] < ms[0], "fused path is not faster");
    }
}

/// Long passes take the chunked recurrence (bf16 output on CUDA; opt-in on
/// Metal); the fused front and back ends must still match the op chain around
/// it.
#[test]
#[ignore = "needs a CUDA or Metal GPU"]
fn fused_matches_op_chain_with_chunked_recurrence() {
    // SAFETY: single-threaded test body; nothing else reads the environment.
    unsafe { std::env::set_var("CRANE_GDN_CHUNKED", "1") };
    let (dev, dtypes) = gpu();
    let dt = dtypes[0];
    let cfg = Cfg {
        hidden: 512,
        k_heads: 4,
        v_heads: 8,
    };
    let (gdn, dims) = layer(&dev, &cfg, dt);
    let base = GdnLayerCache::new(&cfg, dt, &dev).unwrap();
    for (seq, split) in [(130usize, None), (200, Some(70))] {
        let x = rand(&dev, &[1, seq, cfg.hidden], 1.0).to_dtype(dt).unwrap();
        let (ya, ca) = run_with(&gdn, &dims, &x, &base, split, false, false);
        let (yb, cb) = run_with(&gdn, &dims, &x, &base, split, true, false);
        let (yd, yn, ymax) = diff(&ya, &yb);
        let (cd, _, _) = diff(&ca.conv_state, &cb.conv_state);
        let (_, _, rmax) = diff(&ca.recurrent_state, &cb.recurrent_state);
        println!(
            "chunked S={seq} split={split:?}: out {yd}/{yn} (max {ymax:.3e}) | conv mismatches {cd} | state max {rmax:.1e}"
        );
        assert_eq!(cd, 0);
        assert!(rmax < 1e-3 && ymax < 0.05);
    }
    unsafe { std::env::remove_var("CRANE_GDN_CHUNKED") };
}

/// Long-pass cost of one layer on the fused path, serial recurrence kernel vs
/// the chunked formulation (prints only; the default is picked from these).
#[test]
#[ignore = "needs a CUDA or Metal GPU"]
fn prefill_serial_vs_chunked() {
    let (dev, dtypes) = gpu();
    let dt = dtypes[0];
    let cfg = Cfg {
        hidden: 1024,
        k_heads: 16,
        v_heads: 16,
    };
    let (gdn, dims) = layer(&dev, &cfg, dt);
    let base = GdnLayerCache::new(&cfg, dt, &dev).unwrap();
    // SAFETY: single-threaded test body; nothing else reads the environment.
    unsafe { std::env::set_var("CRANE_GDN_CHUNKED", "1") };
    for seq in [64usize, 256, 1024, 4096] {
        let x = rand(&dev, &[1, seq, cfg.hidden], 1.0).to_dtype(dt).unwrap();
        let mut ms = [0f64; 2];
        for (i, serial) in [true, false].into_iter().enumerate() {
            for _ in 0..2 {
                run_with(&gdn, &dims, &x, &base, None, true, serial);
            }
            dev.synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run_with(&gdn, &dims, &x, &base, None, true, serial);
            }
            dev.synchronize().unwrap();
            ms[i] = t.elapsed().as_secs_f64() * 200.0;
        }
        println!(
            "S={seq:<5} serial {:.3} ms/layer, chunked {:.3} ms/layer",
            ms[0], ms[1]
        );
    }
    unsafe { std::env::remove_var("CRANE_GDN_CHUNKED") };
}

/// Decode must preserve state over time, not just on one isolated pass.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU"]
fn decode_stages_match_across_steps_and_checkpoints() {
    let _env = FlaTestEnv::new();
    use crane_core::ops::gdn::VHeadOrder;
    let (dev, _) = gpu();
    for v_heads in [4, 8] {
        let cfg = Cfg {
            hidden: 512,
            k_heads: 4,
            v_heads,
        };
        let (gdn, mut dims) = layer(&dev, &cfg, DType::BF16);
        for order in [VHeadOrder::Interleaved, VHeadOrder::Chunked] {
            dims.v_head_order = order;
            let base = GdnLayerCache::new(&cfg, DType::BF16, &dev).unwrap();
            let prefix = rand(&dev, &[1, 11, cfg.hidden], 1.)
                .to_dtype(DType::BF16)
                .unwrap();
            let (_, mut reference) = run(&gdn, &dims, &prefix, &base, None, false);
            // Simulate a cache loaded in F32 before the first BF16 decode.
            reference.conv_state = reference.conv_state.to_dtype(DType::F32).unwrap();
            let mut fused = reference.clone();
            for step in 0..8 {
                let x = rand(&dev, &[1, 1, cfg.hidden], 1.)
                    .to_dtype(DType::BF16)
                    .unwrap();
                let checkpoint = match step {
                    0 => Some(0),
                    1 => Some(1),
                    _ => None,
                };
                let (ya, ca) = run(&gdn, &dims, &x, &reference, checkpoint, false);
                let (yb, cb) = run(&gdn, &dims, &x, &fused, checkpoint, true);
                let (_, _, ymax) = diff(&ya, &yb);
                let (cd, _, _) = diff(&ca.conv_state, &cb.conv_state);
                let (_, _, rmax) = diff(&ca.recurrent_state, &cb.recurrent_state);
                assert_eq!(cd, 0, "conv step {step}, {order:?}");
                assert_eq!(rmax, 0., "state step {step}, {order:?}");
                assert_eq!(ymax, 0., "output step {step}, {order:?}");
                assert_eq!(ca.checkpoint.is_some(), cb.checkpoint.is_some());
                if let (Some((ac, ar)), Some((bc, br))) = (&ca.checkpoint, &cb.checkpoint) {
                    assert_eq!(diff(ac, bc).0, 0);
                    assert!(diff(ar, br).2 < 1e-3);
                }
                reference = ca;
                fused = cb;
            }
        }
    }
}

/// GGUF decode: F32 activations through the fused single-token kernels,
/// starting from the BF16 conv state that a BF16 prefill leaves behind.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU; run with --test-threads=1"]
fn fused_f32_decode_matches_op_chain() {
    let _env = FlaTestEnv::new();
    use crane_core::ops::gdn::VHeadOrder;
    let (dev, _) = gpu();
    for v_heads in [4, 8] {
        let cfg = Cfg {
            hidden: 512,
            k_heads: 4,
            v_heads,
        };
        let (gdn, mut dims) = layer(&dev, &cfg, DType::F32);
        for order in [VHeadOrder::Interleaved, VHeadOrder::Chunked] {
            dims.v_head_order = order;
            let base = GdnLayerCache::new(&cfg, DType::F32, &dev).unwrap();
            let prefix = rand(&dev, &[1, 11, cfg.hidden], 1.);
            let (_, mut reference) = run(&gdn, &dims, &prefix, &base, None, false);
            reference.conv_state = reference.conv_state.to_dtype(DType::BF16).unwrap();
            let mut fused = reference.clone();
            // Without this the "fused" run is the op chain and every check is vacuous.
            unsafe { std::env::set_var("CRANE_GDN_FUSED", "1") };
            let probe = rand(&dev, &[1, 1, cfg.hidden], 1.);
            assert!(crane_core::ops::gdn::gdn_fused::supported(&probe, &dims));
            for step in 0..8 {
                let x = rand(&dev, &[1, 1, cfg.hidden], 1.);
                let checkpoint = match step {
                    0 => Some(0),
                    1 => Some(1),
                    _ => None,
                };
                let (ya, ca) = run(&gdn, &dims, &x, &reference, checkpoint, false);
                let (yb, cb) = run(&gdn, &dims, &x, &fused, checkpoint, true);
                assert_eq!(yb.dtype(), DType::F32);
                let (_, _, ymax) = diff(&ya, &yb);
                let (cd, _, _) = diff(&ca.conv_state, &cb.conv_state);
                let (_, _, rmax) = diff(&ca.recurrent_state, &cb.recurrent_state);
                println!("v_heads={v_heads} {order:?} step={step}: y {ymax:e}, state {rmax:e}");
                assert_eq!(cd, 0, "conv step {step}, {order:?}");
                assert!(rmax <= 1e-5, "state step {step}, {order:?}: {rmax}");
                assert!(ymax <= 1e-4, "output step {step}, {order:?}: {ymax}");
                assert_eq!(ca.checkpoint.is_some(), cb.checkpoint.is_some());
                if let (Some((ac, ar)), Some((bc, br))) = (&ca.checkpoint, &cb.checkpoint) {
                    assert_eq!(diff(ac, bc).0, 0);
                    assert!(diff(ar, br).2 <= 1e-5);
                }
                reference = ca;
                fused = cb;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU"]
fn fla_prefill_then_fused_decode_matches_reference() -> Result<()> {
    use crane_core::ops::gdn::VHeadOrder;
    let _env = FlaTestEnv::new();
    let (dev, _) = gpu();
    dev.set_seed(194)?;
    let mut mismatch = 0;
    for (v_heads, order) in [
        (4, VHeadOrder::Interleaved),
        (8, VHeadOrder::Interleaved),
        (8, VHeadOrder::Chunked),
    ] {
        let cfg = Cfg {
            hidden: 512,
            k_heads: 4,
            v_heads,
        };
        let (gdn, mut dims) = layer(&dev, &cfg, DType::BF16);
        dims.v_head_order = order;
        let inputs = rand(&dev, &[1, 386, cfg.hidden], 1.).to_dtype(DType::BF16)?;
        for prefix_len in [1, 63, 64, 65, 130] {
            let mut reference = GdnLayerCache::new(&cfg, DType::BF16, &dev)?;
            assert!(reference.fla_numerics);
            unsafe {
                std::env::set_var("CRANE_GDN_FUSED", if prefix_len == 1 { "0" } else { "1" });
            }
            let prefix = inputs.narrow(1, 0, prefix_len)?;
            let mut actual = GdnLayerCache::new(&cfg, DType::BF16, &dev)?;
            let expected = gdn.forward(&prefix, &dims, &mut reference)?;
            let y = gdn.forward(&prefix, &dims, &mut actual)?;
            for (a, b) in [
                (&y, &expected),
                (&actual.conv_state, &reference.conv_state),
                (&actual.recurrent_state, &reference.recurrent_state),
            ] {
                assert_eq!(diff(a, b).0, 0, "common prefix must be identical");
            }
            for step in 0..256 {
                let x = inputs.narrow(1, prefix_len + step, 1)?;
                unsafe {
                    std::env::set_var("CRANE_GDN_FUSED", "0");
                }
                let expected = gdn.forward(&x, &dims, &mut reference)?;
                unsafe {
                    std::env::set_var("CRANE_GDN_FUSED", "1");
                }
                let y = gdn.forward(&x, &dims, &mut actual)?;
                for (name, a, b) in [
                    ("output", &y, &expected),
                    ("conv", &actual.conv_state, &reference.conv_state),
                    ("state", &actual.recurrent_state, &reference.recurrent_state),
                ] {
                    let (n, _, max) = diff(a, b);
                    mismatch += n;
                    if n > 0 && (mismatch < 1000 || [0, 1, 63, 127, 255].contains(&step)) {
                        println!(
                            "FLA decode prefix={prefix_len} HV={v_heads} order={order:?} step={step} {name}: unequal={n}, max={max:.9e}"
                        );
                    }
                }
                if step == 63 {
                    actual.save_checkpoint()?;
                    let saved_conv = actual.conv_state.copy()?;
                    let saved_state = actual.recurrent_state.copy()?;
                    let next = inputs.narrow(1, prefix_len + step + 1, 1)?;
                    let first = gdn.forward(&next, &dims, &mut actual)?;
                    actual.take_checkpoint();
                    assert_eq!(diff(&actual.conv_state, &saved_conv).0, 0);
                    assert_eq!(diff(&actual.recurrent_state, &saved_state).0, 0);
                    actual.save_checkpoint()?;
                    let replay = gdn.forward(&next, &dims, &mut actual)?;
                    assert_eq!(diff(&first, &replay).0, 0, "checkpoint replay");
                    actual.take_checkpoint();
                }
            }
        }
    }
    assert_eq!(
        mismatch, 0,
        "added decode error exceeds initial zero-error gate"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
struct FlaTestEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

#[cfg(feature = "cuda")]
impl FlaTestEnv {
    fn new() -> Self {
        let keys = [
            "CRANE_GDN_FLA",
            "CRANE_GDN_FUSED",
            "CRANE_GDN_PORTABLE",
            "CRANE_GDN_SERIAL_PREFILL",
            "CRANE_GDN_SERIAL_MAX",
        ];
        let saved = keys.into_iter().map(|k| (k, std::env::var_os(k))).collect();
        for k in keys {
            unsafe {
                std::env::remove_var(k);
            }
        }
        Self(saved)
    }
}

#[cfg(feature = "cuda")]
impl Drop for FlaTestEnv {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            unsafe {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU"]
fn fla_decode_preserves_fallbacks() -> Result<()> {
    let _env = FlaTestEnv::new();
    let (dev, _) = gpu();
    let cfg = Cfg {
        hidden: 128,
        k_heads: 1,
        v_heads: 1,
    };
    for dtype in [DType::F16, DType::BF16] {
        let (gdn, dims) = layer(&dev, &cfg, dtype);
        let x = rand(&dev, &[1, 1, cfg.hidden], 1.).to_dtype(dtype)?;
        if dtype == DType::BF16 {
            unsafe {
                std::env::set_var("CRANE_GDN_PORTABLE", "1");
            }
        }
        let mut reference = GdnLayerCache::new(&cfg, dtype, &dev)?;
        let mut actual = GdnLayerCache::new(&cfg, dtype, &dev)?;
        unsafe {
            std::env::set_var("CRANE_GDN_FUSED", "0");
        }
        let expected = gdn.forward(&x, &dims, &mut reference)?;
        unsafe {
            std::env::set_var("CRANE_GDN_FUSED", "1");
        }
        assert!(!crane_core::ops::gdn::gdn_fused::supported(&x, &dims));
        let y = gdn.forward(&x, &dims, &mut actual)?;
        for (a, b) in [
            (&y, &expected),
            (&actual.conv_state, &reference.conv_state),
            (&actual.recurrent_state, &reference.recurrent_state),
        ] {
            assert_eq!(diff(a, b).0, 0);
        }
        unsafe {
            std::env::remove_var("CRANE_GDN_PORTABLE");
        }
    }
    let x = Tensor::zeros((1, 1, cfg.hidden), DType::BF16, &dev)?;
    let mut dims = GdnDims::new(&cfg);
    dims.head_k_dim = 64;
    assert!(!crane_core::ops::gdn::gdn_fused::supported(&x, &dims));
    dims.head_k_dim = 128;
    dims.conv_kernel_size = 3;
    assert!(!crane_core::ops::gdn::gdn_fused::supported(&x, &dims));
    Ok(())
}
