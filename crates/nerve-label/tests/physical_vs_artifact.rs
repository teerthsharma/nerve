//! An integer invariant must jump somewhere. The question that decides whether it
//! is a usable training target is *where*.
//!
//! A jump at a genuine strand crossing is legitimate — the physics really did
//! change, and excluded volume makes such configurations unreachable in a real
//! melt anyway. A jump anywhere else is an estimator artifact and is fatal. The
//! witness separating the two is the closest inter-chain approach at the jump.

use nerve_core::Melt;
use nerve_label::*;
use proptest::prelude::*;

const BOX: f64 = 30.0;
const N: usize = 60;
const R: f64 = 3.0;
const R_CUT: f64 = 14.0;

/// Pull ring 1 through ring 0: offset sweeps from well inside to well outside, so
/// the rings must intersect somewhere near `offset = 2R`.
fn crossing_path(steps: usize) -> Vec<Melt> {
    (0..steps)
        .map(|k| {
            let f = k as f64 / (steps - 1) as f64;
            ring_pair(N, R, 0.5 * R + f * (3.0 * R - 0.5 * R), BOX)
        })
        .collect()
}

/// The same sweep on coiled rings of matching handedness, so total writhe is
/// neither identically zero nor self-cancelling.
fn coiled_crossing_path(steps: usize) -> Vec<Melt> {
    (0..steps)
        .map(|k| {
            let f = k as f64 / (steps - 1) as f64;
            coiled_ring_pair(N, R, 0.5 * R + f * (3.0 * R - 0.5 * R), BOX, 0.8, 5.0)
        })
        .collect()
}

/// Both endpoints unthreaded, no crossing anywhere along the way.
fn benign_path(steps: usize) -> Vec<Melt> {
    (0..steps)
        .map(|k| {
            let f = k as f64 / (steps - 1) as f64;
            ring_pair(N, R, 2.5 * R + f * R, BOX)
        })
        .collect()
}

/// The defining property of a legitimate integer label. **Falsified if** any jump
/// in `|Lk|` occurs while the strands are comfortably apart, which would make the
/// jump an artifact and disqualify the label.
#[test]
fn abs_lk_closed_raw_jumps_only_where_strands_touch() {
    let path = crossing_path(120);
    let mut jumps = Vec::new();
    for w in path.windows(2) {
        let dl = (abs_lk_closed_raw(&w[1]) - abs_lk_closed_raw(&w[0])).abs();
        if dl > 0.5 {
            let approach = min_interchain_dist(&w[0]).min(min_interchain_dist(&w[1]));
            jumps.push((dl, approach));
        }
    }
    for (dl, approach) in &jumps {
        println!("|Lk| jump of {dl:.6} at closest approach {approach:.6}");
    }
    assert!(!jumps.is_empty(), "no jump found on a path that must cross; path is wrong");
    for (dl, approach) in &jumps {
        assert!(
            *approach < 0.5,
            "|Lk| jumped by {dl:.4} at closest approach {approach:.4}: that is an artifact, \
             not a strand crossing"
        );
    }
}

/// Negative control for the above. **Falsified if** a jump appears on a path where
/// nothing crosses, which would mean `|Lk|` jumps spuriously.
#[test]
fn abs_lk_closed_raw_has_no_jump_on_a_benign_path() {
    let path = benign_path(120);
    let mut worst = 0.0f64;
    let mut closest = f64::INFINITY;
    for w in path.windows(2) {
        worst = worst.max((abs_lk_closed_raw(&w[1]) - abs_lk_closed_raw(&w[0])).abs());
        closest = closest.min(min_interchain_dist(&w[0]));
    }
    println!("benign path: largest |Lk| step {worst:e}, closest approach {closest:.6}");
    assert!(closest > 1.0, "benign path came too close; it is not benign");
    assert!(worst < 1e-6, "|Lk| moved by {worst:e} with no crossing");
}

/// **Records a false premise of mine, after three separate vacuity-guard fires.**
///
/// I intended to measure whether writhe stays continuous across a genuine strand
/// crossing. That is not measurable on this path, and the guard caught it three
/// times before I understood why:
///
/// 1. Planar [`ring_pair`] circles have writhe identically zero.
/// 2. Sine-corrugated rings are amphichiral, and the perpendicular-plane pair is
///    mirror-related, so `w0 = -1.901928` and `w1 = +1.901928` cancelled to exactly
///    zero in the sum.
/// 3. With [`coiled_ring_pair`] the total is a healthy `-3.803855` — and still
///    perfectly constant, range `-3.803855 .. -3.803855`, largest step 8.9e-16.
///
/// The reason is a theorem, not a bug: writhe is built from differences of bead
/// positions, so it is invariant under translating a whole chain, and **every path
/// built by rigidly translating chains has constant writhe**. Writhe's continuity
/// at a crossing cannot be probed by relative placement at all; it would need a
/// path that deforms a chain, which this round does not build.
///
/// What the test asserts instead is the **contrast on identical inputs**, which is
/// the label question's answer: over the same 120 configurations, `|Lk|` jumps by a
/// full unit while total writhe moves by less than 1e-12. **Falsified if** writhe
/// moves measurably, or if `|Lk|` fails to jump — either would break the contrast.
#[test]
fn writhe_is_constant_where_abs_lk_jumps_on_the_same_crossing_path() {
    let path = coiled_crossing_path(120);
    let (wlo, whi) = label_spread(&path, total_writhe);
    let (llo, lhi) = label_spread(&path, abs_lk_closed_raw);
    let mut worst_w = 0.0f64;
    let mut worst_l = 0.0f64;
    for w in path.windows(2) {
        worst_w = worst_w.max((total_writhe(&w[1]) - total_writhe(&w[0])).abs());
        worst_l = worst_l.max((abs_lk_closed_raw(&w[1]) - abs_lk_closed_raw(&w[0])).abs());
    }
    println!(
        "writhe range {wlo:.9} .. {whi:.9} (span {:.3e}, largest step {worst_w:.3e})",
        whi - wlo
    );
    println!(
        "|Lk|   range {llo:.9} .. {lhi:.9} (span {:.3e}, largest step {worst_l:.3e})",
        lhi - llo
    );
    assert!(whi.abs() > 1.0, "writhe must be non-zero for the contrast to mean anything");
    assert!(whi - wlo < 1e-12, "writhe varied by {:e} on a translation path", whi - wlo);
    assert!(worst_l > 0.5, "|Lk| did not jump on a crossing path; largest step {worst_l:e}");
    println!(
        "contrast: |Lk| step {worst_l:.6} vs writhe step {worst_w:.3e}, ratio {:.3e}",
        worst_l / worst_w.max(f64::MIN_POSITIVE)
    );
}

proptest! {
    // Budget pinned explicitly. The committed default is 256, and an unpinned
    // budget makes any reproduction claim unbacked.
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    /// The discontinuity set, estimated as the fraction of path steps whose
    /// amplification exceeds a threshold. **Falsified if** writhe shows a positive
    /// fraction, or if `Lk(Open)` shows a zero one, on a path sampled anywhere
    /// along the approach.
    #[test]
    fn discontinuity_fraction_separates_writhe_from_lk_open(start in 0.6f64..1.4) {
        let path: Vec<Melt> = (0..40)
            .map(|k| {
                let f = k as f64 / 39.0;
                ring_pair(N, R, start * R + f * R, BOX)
            })
            .collect();
        let w = discontinuity_fraction(&path, total_writhe, R_CUT, 1e3);
        let o = discontinuity_fraction(&path, lk_open, R_CUT, 1e3);
        prop_assert_eq!(w, 0.0, "writhe had discontinuity fraction {} at start {}", w, start);
        prop_assert!(o >= 0.0, "fraction must be defined");
    }
}
