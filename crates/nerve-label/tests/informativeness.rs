//! Stability is half the question. A constant label is perfectly stable and
//! perfectly useless, so each candidate is also measured for whether it carries
//! pairwise entanglement information at all.
//!
//! **Not tested here, because it is a theorem rather than an experiment:** writhe
//! is computed from differences of bead positions, so it is invariant under
//! translating a whole chain. Any two configurations differing only in the
//! relative placement of rigid chains therefore have identical writhe, exactly.
//! Asserting that in a test would restate the definition — the same mistake that
//! got the round-2 `sum_decomposable_energy` test struck. The empirical question,
//! which a theorem cannot settle, is whether writhe separates threaded from
//! unthreaded once the chains' own shapes vary too. That is what is measured.

use nerve_core::Melt;
use nerve_label::*;

const BOX: f64 = 30.0;
const N: usize = 60;
const R: f64 = 3.0;
const EPS: f64 = 0.25;
const K: u64 = 24;

/// Two ensembles at fixed topology, with genuine per-bead shape noise so that no
/// member of one is a rigid motion of a member of the other.
fn ensembles() -> (Vec<Melt>, Vec<Melt>) {
    let threaded = ring_pair(N, R, R, BOX);
    let unthreaded = ring_pair(N, R, 3.0 * R, BOX);
    (
        (0..K).map(|k| perturb(&threaded, EPS, 7000 + k)).collect(),
        // Different seed stream, so the two ensembles are not paired bead for bead.
        (0..K).map(|k| perturb(&unthreaded, EPS, 9000 + k)).collect(),
    )
}

fn stats<L: Fn(&Melt) -> f64>(e: &[Melt], label: L) -> (f64, f64) {
    let v: Vec<f64> = e.iter().map(label).collect();
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    (mean, sd)
}

/// Separation of two ensemble means in units of pooled within-ensemble sd — the
/// signal-to-noise form the standards require, never a raw distance.
fn separation<L: Fn(&Melt) -> f64 + Copy>(a: &[Melt], b: &[Melt], label: L) -> (f64, f64, f64, f64) {
    let (ma, sa) = stats(a, label);
    let (mb, sb) = stats(b, label);
    let pooled = (0.5 * (sa * sa + sb * sb)).sqrt();
    let z = if pooled > 0.0 {
        (ma - mb).abs() / pooled
    } else if (ma - mb).abs() > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    (ma, mb, pooled, z)
}

/// **Falsified if** `|Lk|` fails to separate the ensembles. This is the positive
/// control that makes the writhe result below non-vacuous: it proves the two
/// ensembles *are* distinguishable, so a label that cannot tell them apart is
/// failing rather than being handed an impossible task.
#[test]
fn abs_lk_closed_raw_separates_the_ensembles_decisively() {
    let (a, b) = ensembles();
    let (ma, mb, pooled, z) = separation(&a, &b, abs_lk_closed_raw);
    println!("|Lk| threaded {ma:.9} +/- , unthreaded {mb:.9}, pooled sd {pooled:e}, z {z:e}");
    assert!((ma - 1.0).abs() < 1e-6, "threaded ensemble mean |Lk| should be 1, got {ma}");
    assert!(mb < 1e-6, "unthreaded ensemble mean |Lk| should be 0, got {mb}");
    assert!(z > 10.0, "|Lk| separation only {z:e} pooled sd; the control has failed");
}

/// The question the coordinator called the leading candidate. **Falsified if**
/// writhe separates the ensembles by more than a couple of pooled sd, which would
/// make it both stable and informative — the ideal label.
///
/// Paired with the `|Lk|` control above: the ensembles are separable, so a small
/// `z` here is writhe's failure, not the ensembles' fault.
#[test]
fn writhe_does_not_separate_threaded_from_unthreaded_ensembles() {
    let (a, b) = ensembles();
    let (ma, mb, pooled, z) = separation(&a, &b, total_writhe);
    println!("writhe threaded {ma:.9}, unthreaded {mb:.9}, pooled sd {pooled:.9}, z {z:.6}");
    assert!(pooled > 1e-6, "writhe has no within-ensemble spread; z is meaningless");
    assert!(
        z < 2.0,
        "writhe separated the ensembles at z {z:.4}; it carries pairwise information after \
         all and is both stable and informative"
    );
}

/// Cameron's dominance result on my own configurations: the closure ambiguity of
/// `Lk(Open)` measured against the signal it is supposed to carry.
///
/// **Falsified if** the spread is small relative to the signal, which would make
/// `Lk(Open)` usable and contradict her finding.
#[test]
fn lk_open_closure_spread_dominates_its_own_signal() {
    let threaded = ring_pair(N, R, R, BOX);
    let s = nerve_topo::closure_spread(&threaded, 0, 1, 64);
    let signal = lk_open(&threaded).abs();
    let ratio = (s.max - s.min) / signal.max(f64::MIN_POSITIVE);
    println!(
        "Lk(Open) {signal:.6e}; closure spread min {:.6} max {:.6} mean {:.6} sd {:.6}; \
         spread/signal {ratio:.4e}",
        s.min, s.max, s.mean, s.sd
    );
    assert!(
        ratio > 1.0,
        "closure spread {:.4e} does not dominate the signal {signal:.4e}; \
         Lk(Open) may be usable and the ranking must be rewritten",
        s.max - s.min
    );
}

// Writhe's estimator ambiguity is not measured here because it does not exist:
// `nerve_topo::writhe(m, i)` takes no `Closure` argument and no second chain, so
// there is no closure to sweep and no image to choose. That is a property of the
// signature, not a measurement, and it is stated rather than tested — asserting
// invariance under a whole-box chain translation would only restate that writhe is
// built from position differences.
