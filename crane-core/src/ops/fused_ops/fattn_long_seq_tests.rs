// SPDX-License-Identifier: MIT
//! Long-sequence regression tests for the MMA flash-attention kernels at
//! Qwen3.5's full-attention shape (8 Q heads, 2 KV heads, `head_dim=256`).
//!
//! Every row of every head is checked against a GPU F32 matmul reference,
//! and sampled rows against a CPU F64 one. Each input runs twice and must
//! give bitwise-identical output. Run on a GPU, in release mode:
//!
//! ```text
//! cargo test -p crane-core --no-default-features --features "rocm,onnx" \
//!     --release --lib long_seq -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `CRANE_LONG_SEQ_SEQS` and `CRANE_LONG_SEQ_SEEDS` narrow the sweep, for
//! example for a `compute-sanitizer` run on the test binary:
//!
//! ```text
//! CRANE_LONG_SEQ_SEQS=1024 CRANE_LONG_SEQ_SEEDS=2 compute-sanitizer \
//!     --tool racecheck <test binary> long_seq_bf16 --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use candle_core::{DType, Device, Tensor};

use super::super::fattn::test_support::{
    LONG_SEQ_D as D, LONG_SEQ_HKV as HKV, LONG_SEQ_HQ as HQ, LongSeqDist, MaskMode,
    attention_rows_f64, long_seq_inputs, naive_attention, splitmix64, test_gpu_device,
};
use super::super::fattn::{
    build_additive_mask_f16, is_amd_backend, mma_config, mma_config_bf16, select_mma_ncols,
    select_mma_ncols_bf16,
};
use super::{
    fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask, fattn_mma_causal,
    fattn_mma_causal_with_mask, padded_seq_kv_for_gqa_batching,
    padded_seq_kv_for_gqa_batching_bf16,
};
use crate::models::modules::gpu_flash_attn::try_causal_prefill;

/// Unaligned lengths take the K/V padding path, aligned ones do not.
const SEQS: [usize; 5] = [231, 238, 1024, 2048, 4096];
const DISTS: [LongSeqDist; 3] = [LongSeqDist::Normal, LongSeqDist::Scaled, LongSeqDist::Spike];
const SEEDS_PER_DIST: u64 = 10;
/// Rows per block in the F64 check, which samples each block's first and
/// last row.
const F64_BLOCK: usize = 64;
const F64_RANDOM_ROWS: usize = 16;
const WORST_ROWS: usize = 20;

///// Tests ///////////////////////////////////////////////////////////////

// The native-bf16 kernel matches both references at every length and seed,
// and gives the same bits on every run.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_bf16() {
    run_long_seq(DType::BF16);
}

// The f16 kernel matches both references at every length and seed, and
// gives the same bits on every run.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_f16() {
    run_long_seq(DType::F16);
}

// `compare` reports the failing row and head, and keeps a NaN visible
// even when later elements are finite.
#[test]
fn compare_flags_nan_and_locates_the_row() {
    let want = vec![1.0f32; 2 * HQ * D];
    let mut got = want.clone();
    let tol = Tolerance::for_dtype(DType::F16);
    assert!(compare(&got, &want, tol).failures.is_empty());

    // Row 1, head 2: a NaN followed by finite elements.
    got[(HQ + 2) * D] = f32::NAN;
    let cmp = compare(&got, &want, tol);
    assert!(cmp.max_err.is_nan());
    let [f] = cmp.failures[..] else {
        panic!("expected one failing row, got {:?}", cmp.failures);
    };
    assert_eq!((f.row, f.head, f.elems), (1, 2, 1));
    assert!(f.max_err.is_nan());
}

// `compare` names the element furthest outside tolerance, and the values
// it compared there.
#[test]
fn compare_reports_the_worst_element() {
    let want = vec![1.0f32; HQ * D];
    let mut got = want.clone();
    got[5] += 0.5;
    got[9] += 2.0;
    got[11] -= 1.0;
    let cmp = compare(&got, &want, Tolerance::for_dtype(DType::BF16));
    let [f] = cmp.failures[..] else {
        panic!("expected one failing row, got {:?}", cmp.failures);
    };
    assert_eq!((f.row, f.head, f.elems, f.elem), (0, 0, 3, 9));
    assert!((f.got - 3.0).abs() < 1e-6 && (f.want - 1.0).abs() < 1e-6);
}

// The sequence-length override accepts spaces and rejects junk.
#[test]
fn parse_seq_list_accepts_spaces_and_rejects_junk() {
    assert_eq!(
        parse_seq_list("1024, 2048 ,4096").unwrap(),
        [1024, 2048, 4096]
    );
    assert!(parse_seq_list("1024,abc").is_err());
    assert!(parse_seq_list("").is_err());
}

///// Runner //////////////////////////////////////////////////////////////

/// Checks every length, distribution and seed, prints one summary line per
/// length, and panics at the end if anything failed. [`sweep_from_env`]
/// narrows the sweep.
fn run_long_seq(dtype: DType) {
    let gpu = test_gpu_device();
    let (seqs, seeds_per_dist) = sweep_from_env();
    let mut report = Report::default();
    for seq in seqs {
        let geom = Geometry::new(dtype, seq);
        let mask = build_additive_mask_f16(seq, seq, 0, None, 0, &gpu).unwrap();
        let mut summary = Summary::default();
        for (d, &dist) in DISTS.iter().enumerate() {
            for idx in 0..seeds_per_dist {
                let case = Case {
                    seq,
                    dist,
                    dist_idx: d,
                    seed: d as u64 * SEEDS_PER_DIST + idx,
                    first_in_dist: idx == 0,
                    dtype,
                    geom,
                };
                check_case(&case, &mask, &gpu, &mut summary, &mut report);
            }
        }
        println!("{dtype:?} seq={seq} {geom}: {summary}");
        report.failed |= summary.failed();
    }
    report.finish(dtype);
}

/// The lengths and seeds per distribution to run. The defaults are [`SEQS`]
/// and [`SEEDS_PER_DIST`]. A sanitizer run is far too slow for the full
/// sweep, so two environment variables narrow it:
/// `CRANE_LONG_SEQ_SEQS=1024,2048` and `CRANE_LONG_SEQ_SEEDS=2`. Seed
/// numbers do not depend on the override, so a narrowed run replays inputs
/// of the full one.
///
/// # Panics
///
/// Panics on a value that does not parse, or on a seed count outside
/// `1..=SEEDS_PER_DIST`.
fn sweep_from_env() -> (Vec<usize>, u64) {
    let seqs = std::env::var("CRANE_LONG_SEQ_SEQS").map_or_else(
        |_| SEQS.to_vec(),
        |s| parse_seq_list(&s).unwrap_or_else(|e| panic!("CRANE_LONG_SEQ_SEQS={s:?}: {e}")),
    );
    let seeds = std::env::var("CRANE_LONG_SEQ_SEEDS").map_or(SEEDS_PER_DIST, |s| {
        s.trim()
            .parse()
            .unwrap_or_else(|e| panic!("CRANE_LONG_SEQ_SEEDS={s:?}: {e}"))
    });
    assert!(
        (1..=SEEDS_PER_DIST).contains(&seeds),
        "CRANE_LONG_SEQ_SEEDS must be in 1..={SEEDS_PER_DIST}, got {seeds}"
    );
    (seqs, seeds)
}

/// Parses `1024, 2048` into `[1024, 2048]`.
fn parse_seq_list(s: &str) -> Result<Vec<usize>, std::num::ParseIntError> {
    s.split(',').map(|part| part.trim().parse()).collect()
}

/// One input of the sweep.
#[derive(Debug, Clone, Copy)]
struct Case {
    seq: usize,
    dist: LongSeqDist,
    /// Position of `dist` in [`DISTS`].
    dist_idx: usize,
    seed: u64,
    /// Whether this is the distribution's first seed, the one the slow F64
    /// check runs on.
    first_in_dist: bool,
    dtype: DType,
    geom: Geometry,
}

/// Runs both kernel entries and `try_causal_prefill` on one input and
/// folds the results into `summary` and `report`.
fn check_case(
    case: &Case,
    mask: &Tensor,
    gpu: &Device,
    summary: &mut Summary,
    report: &mut Report,
) {
    let (seq, dtype) = (case.seq, case.dtype);
    // D is 256, exact in an f32.
    #[allow(clippy::cast_precision_loss)]
    let scale = (D as f32).sqrt().recip();
    let rounded = long_seq_inputs(seq, case.dist, case.seed).map(|h| round_to(h, dtype));
    let [q_host, k_host, v_host] = &rounded;
    let bshd = |h: &[f32], heads| Tensor::from_slice(h, (1, seq, heads, D), gpu).unwrap();
    // The kernels read q as F32 and k/v in the input dtype.
    let q = bshd(q_host, HQ);
    let k = bshd(k_host, HKV).to_dtype(dtype).unwrap();
    let v = bshd(v_host, HKV).to_dtype(dtype).unwrap();

    let want = naive_attention(
        &q,
        &bshd(k_host, HKV),
        &bshd(v_host, HKV),
        scale,
        MaskMode::Causal,
        0,
        0,
        0,
    );

    let (got, again, with_mask) = if dtype == DType::BF16 {
        (
            fattn_mma_bf16_causal(&q, &k, &v, scale, 0),
            fattn_mma_bf16_causal(&q, &k, &v, scale, 0),
            fattn_mma_bf16_causal_with_mask(&q, &k, &v, scale, 0, mask),
        )
    } else {
        (
            fattn_mma_causal(&q, &k, &v, scale, 0),
            fattn_mma_causal(&q, &k, &v, scale, 0),
            fattn_mma_causal_with_mask(&q, &k, &v, scale, 0, mask),
        )
    };
    let got = to_host(&got.unwrap());
    let nondeterministic = !bitwise_eq(&got, &to_host(&again.unwrap()));
    summary.cases += 1;
    summary.nondeterministic += usize::from(nondeterministic);
    summary.nondeterministic_by_dist[case.dist_idx] += usize::from(nondeterministic);
    summary.with_mask_mismatch += usize::from(!bitwise_eq(&got, &to_host(&with_mask.unwrap())));
    let tol = Tolerance::for_dtype(dtype);
    let cmp = compare(&got, &want, tol);
    summary.add_max(&cmp);
    summary.kernel_failed += usize::from(!cmp.failures.is_empty());
    summary.kernel_failed_by_dist[case.dist_idx] += usize::from(!cmp.failures.is_empty());
    report.add(case, "kernel", &cmp.failures);

    // The exact path models take: BHSD in the input dtype.
    let bhsd = |t: &Tensor| {
        t.to_dtype(dtype)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
    };
    let prefill = try_causal_prefill(
        &bhsd(&q).unwrap(),
        &bhsd(&k).unwrap(),
        &bhsd(&v).unwrap(),
        scale,
        None,
    )
    .unwrap()
    .expect("try_causal_prefill declined a supported shape");
    let prefill = to_host(&prefill);
    summary.prefill_mismatch += usize::from(!bitwise_eq(&got, &prefill));
    let cmp = compare(&prefill, &want, tol);
    summary.add_max(&cmp);
    summary.prefill_failed += usize::from(!cmp.failures.is_empty());
    report.add(case, "try_causal_prefill", &cmp.failures);

    // CPU F64 is slow, so it checks the first seed of each distribution.
    if case.first_in_dist {
        let rows = sampled_rows(seq, case.seed);
        let exact = attention_rows_f64(q_host, k_host, v_host, scale, &rows);
        let exact: Vec<f32> = exact.iter().map(|&x| narrow(x)).collect();
        let cmp = compare(&gather_rows(&got, &rows), &exact, tol);
        summary.f64_checked += 1;
        summary.f64_failed += usize::from(!cmp.failures.is_empty());
        let ref_cmp = compare(&gather_rows(&want, &rows), &exact, tol);
        summary.ref_vs_f64 = summary.ref_vs_f64.max(ref_cmp.max_err);
        // Failure rows index into `rows`, so map them back.
        let failures: Vec<_> = cmp
            .failures
            .iter()
            .map(|f| RowFailure {
                row: rows[f.row],
                ..*f
            })
            .collect();
        report.add(case, "kernel vs f64", &failures);
    }
}

///// Comparison //////////////////////////////////////////////////////////

/// Mixed absolute/relative tolerance. It sits well above output rounding
/// in the dtype, and well below the errors the maintainer reported.
#[derive(Debug, Clone, Copy)]
struct Tolerance {
    atol: f32,
    rtol: f32,
}

impl Tolerance {
    /// BF16 and AMD F16 accumulate in F32. The NVIDIA F16 kernel keeps its
    /// output accumulator in `half2` as llama.cpp does, which costs about
    /// 1% on peaked softmax rows, so it gets a looser limit than BF16
    /// despite the longer mantissa. Real bugs have been off by whole
    /// units, far outside either.
    fn for_dtype(dtype: DType) -> Self {
        if dtype == DType::BF16 {
            Self {
                atol: 1e-2,
                rtol: 2e-2,
            }
        } else if !is_amd_backend() {
            Self {
                atol: 1e-2,
                rtol: 4e-2,
            }
        } else {
            Self {
                atol: 2e-3,
                rtol: 1e-2,
            }
        }
    }

    fn limit(self, want: f32) -> f32 {
        self.atol + self.rtol * want.abs()
    }
}

/// Result of [`compare`].
#[derive(Debug, Default)]
struct Comparison {
    max_err: f32,
    /// Largest `err / limit`. Above `1.0` means failure.
    max_ratio: f32,
    failures: Vec<RowFailure>,
}

/// One `(row, head)` with at least one element out of tolerance.
#[derive(Debug, Clone, Copy)]
struct RowFailure {
    row: usize,
    head: usize,
    max_err: f32,
    /// Number of elements out of tolerance.
    elems: usize,
    /// Index within `head_dim` of the element furthest outside tolerance
    /// (largest error over its limit), and the two values compared there.
    /// A NaN counts as the furthest.
    elem: usize,
    got: f32,
    want: f32,
}

/// Checks `got` against `want`, both flat `[rows, HQ, D]`. A NaN in `got`
/// counts as out of tolerance.
fn compare(got: &[f32], want: &[f32], tol: Tolerance) -> Comparison {
    assert_eq!(got.len(), want.len());
    let mut cmp = Comparison::default();
    for (c, (g, w)) in got
        .as_chunks::<D>()
        .0
        .iter()
        .zip(want.as_chunks::<D>().0)
        .enumerate()
    {
        let mut row_err = 0f32;
        let mut worst = (0, 0f32, 0f32);
        let mut worst_ratio = 0f32;
        let mut elems = 0;
        for (e, (&g, &w)) in g.iter().zip(w).enumerate() {
            let err = (g - w).abs();
            let ratio = err / tol.limit(w);
            if err.is_nan() || ratio > 1.0 {
                elems += 1;
            }
            if (ratio.is_nan() && !worst_ratio.is_nan()) || ratio > worst_ratio {
                worst = (e, g, w);
                worst_ratio = ratio;
            }
            row_err = nan_max(row_err, err);
            cmp.max_ratio = nan_max(cmp.max_ratio, ratio);
        }
        cmp.max_err = nan_max(cmp.max_err, row_err);
        if elems > 0 {
            cmp.failures.push(RowFailure {
                row: c / HQ,
                head: c % HQ,
                max_err: row_err,
                elems,
                elem: worst.0,
                got: worst.1,
                want: worst.2,
            });
        }
    }
    cmp
}

/// `a.max(b)`, except that a NaN in either stays NaN. `f32::max` would drop
/// it and hide the failure.
fn nan_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

///// Reporting ///////////////////////////////////////////////////////////

/// Kernel tiling at one length, for locating failures relative to tile
/// boundaries.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    ncols1: usize,
    ncols2: usize,
    nbatch_fa: usize,
}

impl Geometry {
    /// The tiling the causal entries pick for `seq`, after their K/V
    /// padding.
    fn new(dtype: DType, seq: usize) -> Self {
        let is_amd = is_amd_backend();
        let gqa = HQ / HKV;
        let (ncols1, ncols2, nbatch_fa) = if dtype == DType::BF16 {
            let kv = padded_seq_kv_for_gqa_batching_bf16(D, gqa, seq, seq, is_amd).unwrap_or(seq);
            let (n1, n2) = select_mma_ncols_bf16(D, gqa, seq, kv, is_amd).unwrap();
            (
                n1,
                n2,
                mma_config_bf16(D, n1 * n2, is_amd).unwrap().nbatch_fa,
            )
        } else {
            let kv = padded_seq_kv_for_gqa_batching(D, gqa, seq, seq, is_amd).unwrap_or(seq);
            let (n1, n2) = select_mma_ncols(D, gqa, seq, kv, is_amd).unwrap();
            (n1, n2, mma_config(D, n1 * n2, is_amd).unwrap().nbatch_fa)
        };
        Self {
            ncols1,
            ncols2,
            nbatch_fa,
        }
    }
}

impl std::fmt::Display for Geometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ncols1={} ncols2={} nbatch_fa={}",
            self.ncols1, self.ncols2, self.nbatch_fa
        )
    }
}

/// Per-length counters. The `*_failed` counts are seeds with at least one
/// element out of tolerance, the maintainer's "x/30 checks" unit.
#[derive(Debug, Default)]
struct Summary {
    /// Inputs checked, which is the denominator of the failure counts.
    cases: usize,
    kernel_failed: usize,
    /// `kernel_failed` and `nondeterministic` split by [`DISTS`] entry.
    kernel_failed_by_dist: [usize; DISTS.len()],
    nondeterministic_by_dist: [usize; DISTS.len()],
    prefill_failed: usize,
    nondeterministic: usize,
    with_mask_mismatch: usize,
    prefill_mismatch: usize,
    f64_checked: usize,
    f64_failed: usize,
    max_err: f32,
    max_ratio: f32,
    ref_vs_f64: f32,
}

impl Summary {
    fn add_max(&mut self, cmp: &Comparison) {
        self.max_err = self.max_err.max(cmp.max_err);
        self.max_ratio = self.max_ratio.max(cmp.max_ratio);
    }

    fn failed(&self) -> bool {
        self.kernel_failed
            + self.prefill_failed
            + self.nondeterministic
            + self.with_mask_mismatch
            + self.prefill_mismatch
            + self.f64_failed
            > 0
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let seeds = self.cases;
        write!(
            f,
            "kernel {}/{seeds} ({}), try_causal_prefill {}/{seeds}, \
             nondeterministic {} ({}), with_mask mismatch {}, \
             try_causal_prefill mismatch {}, kernel vs f64 {}/{}, \
             max err {:.2e} ({:.0}% of tol), f32 ref vs f64 {:.1e}",
            self.kernel_failed,
            by_dist(&self.kernel_failed_by_dist),
            self.prefill_failed,
            self.nondeterministic,
            by_dist(&self.nondeterministic_by_dist),
            self.with_mask_mismatch,
            self.prefill_mismatch,
            self.f64_failed,
            self.f64_checked,
            self.max_err,
            self.max_ratio * 100.0,
            self.ref_vs_f64,
        )
    }
}

/// `Normal 0, Scaled 6, Spike 10`.
fn by_dist(counts: &[usize; DISTS.len()]) -> String {
    DISTS
        .iter()
        .zip(counts)
        .map(|(d, n)| format!("{d:?} {n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The worst failing rows across the whole sweep, plus failure counts by
/// position in the Q tile and by head.
#[derive(Debug, Default)]
struct Report {
    failed: bool,
    worst: Vec<(Case, &'static str, RowFailure)>,
    by_row_in_tile: BTreeMap<usize, usize>,
    by_head: BTreeMap<usize, usize>,
}

impl Report {
    fn add(&mut self, case: &Case, entry: &'static str, failures: &[RowFailure]) {
        for f in failures {
            *self
                .by_row_in_tile
                .entry(f.row % case.geom.ncols1)
                .or_default() += 1;
            *self.by_head.entry(f.head).or_default() += 1;
            self.worst.push((*case, entry, *f));
        }
        // NaN sorts as the largest error under `total_cmp`.
        self.worst
            .sort_by(|a, b| b.2.max_err.total_cmp(&a.2.max_err));
        self.worst.truncate(WORST_ROWS);
    }

    /// Prints the failure details and panics, if anything failed.
    fn finish(&self, dtype: DType) {
        if !self.failed {
            return;
        }
        println!("{dtype:?}: worst {} failing rows:", self.worst.len());
        for (case, entry, f) in &self.worst {
            println!(
                "  {entry}: seq={} {:?} seed={} head={} row={} \
                 row%ncols1={} row/nbatch_fa={} max err {:.3e} ({} elems), \
                 worst elem {}: got {:.5} want {:.5}",
                case.seq,
                case.dist,
                case.seed,
                f.head,
                f.row,
                f.row % case.geom.ncols1,
                f.row / case.geom.nbatch_fa,
                f.max_err,
                f.elems,
                f.elem,
                f.got,
                f.want,
            );
        }
        println!(
            "{dtype:?}: failing rows by row % ncols1: {:?}",
            self.by_row_in_tile
        );
        println!("{dtype:?}: failing rows by head: {:?}", self.by_head);
        panic!("{dtype:?}: long-sequence check failed, see the report above");
    }
}

///// Helpers /////////////////////////////////////////////////////////////

/// Rounds host values to `dtype` and back, so every path sees the same
/// inputs.
fn round_to(host: Vec<f32>, dtype: DType) -> Vec<f32> {
    let n = host.len();
    Tensor::from_vec(host, n, &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Copies a GPU result of any float dtype to the host as flat F32.
fn to_host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn bitwise_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Rows the F64 check covers: the first and last row of every
/// [`F64_BLOCK`]-row block, which includes rows 0 and `seq - 1`, plus
/// [`F64_RANDOM_ROWS`] seeded random rows.
#[allow(clippy::cast_possible_truncation)] // the modulus keeps it below seq
fn sampled_rows(seq: usize, seed: u64) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..seq)
        .step_by(F64_BLOCK)
        .flat_map(|s| [s, (s + F64_BLOCK - 1).min(seq - 1)])
        .collect();
    let mut state = seed;
    rows.extend((0..F64_RANDOM_ROWS).map(|_| (splitmix64(&mut state) % seq as u64) as usize));
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The `[HQ, D]` slices of flat BSHD `t` at `rows`, concatenated.
fn gather_rows(t: &[f32], rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| &t[r * HQ * D..][..HQ * D])
        .copied()
        .collect()
}

// Attention outputs are convex combinations of v, a few units at most.
#[allow(clippy::cast_possible_truncation)]
fn narrow(x: f64) -> f32 {
    x as f32
}
