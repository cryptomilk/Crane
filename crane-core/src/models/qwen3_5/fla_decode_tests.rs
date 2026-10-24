//! Floating-point validation against current main's single-token path.
use super::*;
use std::io::Write;

#[derive(Debug, Default)]
struct ErrorStats {
    max_abs: f64,
    mean_abs: f64,
    relative_l2: f64,
}

fn error_stats(actual: &Tensor, reference: &Tensor) -> Result<ErrorStats> {
    anyhow::ensure!(actual.dims() == reference.dims(), "tensor shape mismatch");
    let a = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = reference
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    anyhow::ensure!(!a.is_empty(), "empty comparison");
    let (mut max, mut sum, mut squared, mut norm) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in a.into_iter().zip(&b) {
        anyhow::ensure!(a.is_finite() && b.is_finite(), "nonfinite tensor");
        let d = f64::from(a) - f64::from(*b);
        max = max.max(d.abs());
        sum += d.abs();
        squared += d * d;
        norm += f64::from(*b).powi(2);
    }
    Ok(ErrorStats {
        max_abs: max,
        mean_abs: sum / b.len() as f64,
        relative_l2: squared.sqrt() / norm.sqrt().max(1e-12),
    })
}

#[test]
fn tensor_error_metrics() -> Result<()> {
    let zero = Tensor::zeros(3, DType::F32, &Device::Cpu)?;
    let stats = error_stats(&zero, &zero)?;
    assert_eq!(
        (stats.max_abs, stats.mean_abs, stats.relative_l2),
        (0., 0., 0.)
    );
    let a = Tensor::new(&[1.25f32, 0., -2.], &Device::Cpu)?;
    let b = Tensor::new(&[1f32, 0., -2.], &Device::Cpu)?;
    let stats = error_stats(&a, &b)?;
    assert_eq!(stats.max_abs, 0.25);
    assert!((stats.mean_abs - 0.25 / 3.).abs() < 1e-12);
    assert!((stats.relative_l2 - 0.25 / 5f64.sqrt()).abs() < 1e-12);
    assert!(error_stats(&a.reshape((1, 3))?, &b).is_err());
    let bad = Tensor::new(&[f32::NAN, f32::INFINITY, 0.], &Device::Cpu)?;
    assert!(error_stats(&bad, &b).is_err());
    let stats = error_stats(&a, &zero)?;
    assert!(stats.relative_l2 > 1e12);
    Ok(())
}

struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl EnvRestore {
    fn new() -> Self {
        let keys = [
            "CRANE_GDN_FLA",
            "CRANE_GDN_FUSED",
            "CRANE_GDN_PORTABLE",
            "CRANE_GDN_SERIAL_PREFILL",
            "CRANE_GDN_SERIAL_MAX",
            "CRANE_ISQ",
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
impl Drop for EnvRestore {
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

#[test]
#[ignore = "needs CUDA and CRANE_QWEN35_MODEL pointing to a local checkpoint"]
fn real_qwen35_fla_decode_parity() -> Result<()> {
    let _env = EnvRestore::new();
    let path = std::env::var("CRANE_QWEN35_MODEL").context("set CRANE_QWEN35_MODEL")?;
    let report_path = std::env::var("CRANE_FLA_REPORT")
        .context("set CRANE_FLA_REPORT for per-step CSV evidence")?;
    let dev = Device::new_cuda(0)?;
    let mut reference = Model::new(&path, &dev, &DType::BF16)?;
    let mut actual = Model::new(&path, &dev, &DType::BF16)?;
    let tokenizer =
        Tokenizer::from_file(std::path::Path::new(&path).join("tokenizer.json")).map_err(E::msg)?;
    let text = "Explain how a computer processes a sentence, stores intermediate results, and generates the next word. Use a concrete example and describe each calculation carefully. ".repeat(40);
    let encoding = tokenizer.encode(text, false).map_err(E::msg)?;
    let tokens = encoding.get_ids();
    anyhow::ensure!(
        tokens.len() >= 386,
        "need at least 386 teacher-forced tokens"
    );
    let mut report = std::io::BufWriter::new(std::fs::File::create(report_path)?);
    writeln!(
        report,
        "prefix,step,tensor,layer,max_abs,mean_abs,relative_l2"
    )?;
    let mut errors = 0;
    let mut worst: std::collections::BTreeMap<String, (usize, usize, ErrorStats)> =
        Default::default();
    for prefix in [64usize, 130] {
        reference.inner.reset_gdn_caches()?;
        actual.inner.reset_gdn_caches()?;
        unsafe {
            std::env::set_var("CRANE_GDN_FUSED", "1");
        }
        assert!(
            reference
                .inner
                .gdn_caches
                .iter()
                .flatten()
                .all(|c| c.fla_numerics)
        );
        assert!(
            actual
                .inner
                .gdn_caches
                .iter()
                .flatten()
                .all(|c| c.fla_numerics)
        );
        let ids = Tensor::new(&tokens[..prefix], &dev)?.unsqueeze(0)?;
        let mask =
            super::super::prefill::causal_mask(prefix, 0, &dev, reference.inner.attention_dtype())?;
        let expected = reference.inner.forward_layers(&ids, 0, Some(&mask))?;
        let y = actual.inner.forward_layers(&ids, 0, Some(&mask))?;
        let stats = error_stats(&y, &expected)?;
        assert_eq!(stats.max_abs, 0., "all prefill hidden positions");
        writeln!(
            report,
            "{prefix},prefill,hidden,all,{},{},{}",
            stats.max_abs, stats.mean_abs, stats.relative_l2
        )?;
        for step in 0..256 {
            let pos = prefix + step;
            let ids = Tensor::new(&[[tokens[pos]]], &dev)?;
            unsafe {
                std::env::set_var("CRANE_GDN_FUSED", "0");
            }
            let expected = reference.inner.forward_layers(&ids, pos, None)?;
            let expected_logits = reference.inner.head(&expected)?;
            unsafe {
                std::env::set_var("CRANE_GDN_FUSED", "1");
            }
            let y = actual.inner.forward_layers(&ids, pos, None)?;
            let logits = actual.inner.head(&y)?;
            let mut compare = |name: &str, layer: usize, a: &Tensor, b: &Tensor| -> Result<()> {
                let stats = error_stats(a, b)?;
                writeln!(
                    report,
                    "{prefix},{step},{name},{layer},{:.12e},{:.12e},{:.12e}",
                    stats.max_abs, stats.mean_abs, stats.relative_l2
                )?;
                errors += usize::from(stats.max_abs != 0.);
                if worst
                    .get(name)
                    .is_none_or(|(_, _, old)| stats.max_abs > old.max_abs)
                {
                    worst.insert(name.to_owned(), (prefix, step, stats));
                }
                Ok(())
            };
            compare("hidden", 0, &y, &expected)?;
            compare("logits", 0, &logits, &expected_logits)?;
            for (layer, (a, b)) in actual
                .inner
                .gdn_caches
                .iter()
                .zip(&reference.inner.gdn_caches)
                .enumerate()
            {
                if let (Some(a), Some(b)) = (a, b) {
                    compare("conv", layer, &a.conv_state, &b.conv_state)?;
                    compare("state", layer, &a.recurrent_state, &b.recurrent_state)?;
                }
            }
            if step % 64 == 0 {
                println!("real FLA prefix={prefix}, step={step}");
                report.flush()?;
            }
        }
    }
    report.flush()?;
    for (name, (prefix, step, stats)) in worst {
        println!("worst {name} prefix={prefix} step={step}: {stats:?}");
    }
    assert_eq!(
        errors, 0,
        "added decode error exceeds initial zero-error gate; retain CSV for review"
    );
    Ok(())
}

#[test]
#[ignore = "needs CUDA and CRANE_QWEN35_GGUF pointing to a local .gguf"]
fn real_qwen35_gguf_fused_decode_parity() -> Result<()> {
    let _env = EnvRestore::new();
    let path = std::env::var("CRANE_QWEN35_GGUF").context("set CRANE_QWEN35_GGUF")?;
    let report_path = std::env::var("CRANE_GGUF_REPORT")
        .context("set CRANE_GGUF_REPORT for per-step CSV evidence")?;
    let dev = Device::new_cuda(0)?;
    let mut reference = Model::new(&path, &dev, &DType::F32)?;
    let mut actual = Model::new(&path, &dev, &DType::F32)?;
    let text = "Explain how a computer processes a sentence, stores intermediate results, and generates the next word. Use a concrete example and describe each calculation carefully. ".repeat(40);
    let encoding = actual
        .tokenizer
        .tokenizer()
        .encode(text, false)
        .map_err(E::msg)?;
    let tokens = encoding.get_ids();
    anyhow::ensure!(
        tokens.len() >= 258,
        "need at least 258 teacher-forced tokens"
    );
    let mut report = std::io::BufWriter::new(std::fs::File::create(report_path)?);
    writeln!(
        report,
        "prefix,step,tensor,layer,max_abs,mean_abs,relative_l2"
    )?;
    let mut failures = Vec::new();
    let mut first_flip = Vec::new();
    let mut worst: std::collections::BTreeMap<String, (usize, usize, ErrorStats)> =
        Default::default();
    for prefix in [64usize, 130] {
        reference.inner.reset_gdn_caches()?;
        actual.inner.reset_gdn_caches()?;
        let ids = Tensor::new(&tokens[..prefix], &dev)?.unsqueeze(0)?;
        let mask =
            super::super::prefill::causal_mask(prefix, 0, &dev, reference.inner.attention_dtype())?;
        let expected = reference.inner.forward_layers(&ids, 0, Some(&mask))?;
        let y = actual.inner.forward_layers(&ids, 0, Some(&mask))?;
        let stats = error_stats(&y, &expected)?;
        assert_eq!(stats.max_abs, 0., "all prefill hidden positions");
        let mut bitwise = true;
        for step in 0..128 {
            let pos = prefix + step;
            let ids = Tensor::new(&[[tokens[pos]]], &dev)?;
            unsafe {
                std::env::set_var("CRANE_GDN_FUSED", "0");
            }
            let expected = reference.inner.forward_layers(&ids, pos, None)?;
            let expected_logits = reference.inner.head(&expected)?;
            unsafe {
                std::env::set_var("CRANE_GDN_FUSED", "1");
            }
            let y = actual.inner.forward_layers(&ids, pos, None)?;
            let logits = actual.inner.head(&y)?;
            anyhow::ensure!(y.dtype() == DType::F32, "GGUF decode must stay F32");
            // GGUF attention runs in BF16, so once an ulp-level GDN difference
            // flips one BF16 rounding the hidden states diverge by ~3%, the same
            // as main's portable GDN path does against main. Until then the GDN
            // caches must agree to rounding error; afterwards the error must stay
            // bounded and the greedy token must not change.
            if bitwise && error_stats(&y, &expected)?.max_abs != 0. {
                bitwise = false;
                first_flip.push((prefix, step));
            }
            let mut compare = |name: &str, layer: usize, a: &Tensor, b: &Tensor| -> Result<()> {
                let stats = error_stats(a, b)?;
                writeln!(
                    report,
                    "{prefix},{step},{name},{layer},{:.12e},{:.12e},{:.12e}",
                    stats.max_abs, stats.mean_abs, stats.relative_l2
                )?;
                let ok = match name {
                    "conv" => !bitwise || stats.max_abs == 0.,
                    "state" => !bitwise || stats.max_abs <= 1e-6,
                    _ => stats.relative_l2 <= 0.05,
                };
                if !ok {
                    failures.push(format!("{name} layer={layer} prefix={prefix} step={step}"));
                }
                if worst
                    .get(name)
                    .is_none_or(|(_, _, old)| stats.max_abs > old.max_abs)
                {
                    worst.insert(name.to_owned(), (prefix, step, stats));
                }
                Ok(())
            };
            compare("hidden", 0, &y, &expected)?;
            compare("logits", 0, &logits, &expected_logits)?;
            for (layer, (a, b)) in actual
                .inner
                .gdn_caches
                .iter()
                .zip(&reference.inner.gdn_caches)
                .enumerate()
            {
                if let (Some(a), Some(b)) = (a, b) {
                    compare("conv", layer, &a.conv_state, &b.conv_state)?;
                    compare("state", layer, &a.recurrent_state, &b.recurrent_state)?;
                }
            }
            let top = |l: &Tensor| l.flatten_all()?.argmax(0)?.to_scalar::<u32>();
            if top(&logits)? != top(&expected_logits)? {
                failures.push(format!("argmax prefix={prefix} step={step}"));
            }
        }
    }
    report.flush()?;
    println!("first hidden difference (prefix, step): {first_flip:?}");
    for (name, (prefix, step, stats)) in worst {
        println!("worst {name} prefix={prefix} step={step}: {stats:?}");
    }
    // A silent fallback to the op chain would match bit for bit at every step.
    assert_eq!(first_flip.len(), 2, "fused decode never ran");
    assert!(failures.is_empty(), "parity gates failed: {failures:?}");
    Ok(())
}
