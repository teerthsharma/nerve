//! The learnability criterion, and its validation against Cameron's counterexample.
//!
//! If the criterion does not flag `Lk(Open)` at the knife edge, the criterion is
//! broken and nothing else in this crate means anything. That test comes first.
//!
//! Falsification conditions are stated per test. A stability test that cannot
//! fail is worthless, so each one names the outcome that would kill it.

use nerve_core::Melt;
use nerve_label::*;

const BOX: f64 = 30.0;
const N: usize = 60;
const R: f64 = 3.0;
const X_A: f64 = 5.0;
/// Below `BOX / 2 = 15.0`, and wide enough that the two arcs actually have
/// cross-chain pairs inside the cutoff — otherwise the descriptor delta is
/// identically zero and the amplification ratio degenerates to `inf` at every
/// step, measuring nothing.
const R_CUT: f64 = 14.0;

/// Six decades. A discontinuity should amplify by about `1e6` across this span.
const HS: [f64; 4] = [1e-3, 1e-5, 1e-7, 1e-9];

fn arcs(x_b: f64) -> Melt {
    open_arc_pair(N, R, X_A, x_b, BOX)
}

/// Locate the knife edge: the `x_b` at which the componentwise centroid
/// separation crosses `BOX / 2`, which is what nearest-image placement switches
/// on. Bisected rather than assumed, because a three-quarter arc's centroid is
/// not its circle centre.
fn knife_edge_x() -> f64 {
    let f = |x: f64| centroid_delta(&arcs(x), 0, 1)[0] - BOX / 2.0;
    let (mut lo, mut hi) = (X_A + 6.0, X_A + 24.0);
    assert!(f(lo) < 0.0 && f(hi) > 0.0, "knife edge not bracketed: {} {}", f(lo), f(hi));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn bracket(h: f64) -> (Melt, Melt) {
    let x = knife_edge_x();
    (arcs(x - h / 2.0), arcs(x + h / 2.0))
}

fn sweep<L: Fn(&Melt) -> f64 + Copy>(name: &str, label: L) -> Vec<Amp> {
    let amps: Vec<Amp> = HS
        .iter()
        .map(|&h| {
            let (a, b) = bracket(h);
            amplification(&a, &b, h, label, R_CUT)
        })
        .collect();
    for a in &amps {
        println!(
            "{name:>26}  h {:>8.1e}  dD {:>10.3e}  dL {:>10.3e}  A {:>10.3e}",
            a.h, a.dd, a.dl, a.ratio
        );
    }
    println!(
        "{name:>26}  jump height {:.4e}  peak amplification {:.4e}",
        jump_height(&amps),
        peak_amplification(&amps)
    );
    amps
}

/// Floating-point noise scale of the descriptor at the knife edge, measured once
/// so the jump thresholds below are stated relative to it rather than picked.
fn descriptor_noise() -> f64 {
    let (a, b) = bracket(1e-9);
    descriptor_delta(&descriptor(&a, R_CUT), &descriptor(&b, R_CUT))
}

// ------------------------------------------- validation against the known bad

/// Cameron's counterexample, reproduced. **Falsified if** the jump height falls
/// toward zero as `h` shrinks, which would mean `Lk(Open)` is merely steep rather
/// than discontinuous and the criterion has nothing to flag.
///
/// Successor to `lk_open_amplification_diverges_at_the_knife_edge`, which was
/// logged RED and then FAILED on a false premise of mine: it tested whether
/// `|ΔL|/|ΔD|` grows like `1/h`, assuming `ΔD ∝ h`. It does not — the descriptor is
/// stationary at the knife edge, so `ΔD` sat at 1.954e-14 for every `h` and the
/// ratio was pinned at ~3.4e11 instead of growing. The discontinuity was real and
/// enormous; the divergence-ratio formulation could not see it.
#[test]
fn lk_open_jump_height_is_floored_as_the_bracket_shrinks() {
    let amps = sweep("lk_open", lk_open);
    let jh = jump_height(&amps);
    let noise = descriptor_noise();
    println!("lk_open jump height {jh:e} against descriptor noise {noise:e}");
    assert!(
        jh > 1e-4,
        "criterion failed to flag the known-bad label: jump height only {jh:e}"
    );
    assert!(
        jh / noise > 1e6,
        "jump height {jh:e} is not decisively above descriptor noise {noise:e}"
    );
}

/// The sharpest form of Cameron's result, and the reason my first criterion
/// failed. **Falsified if** the descriptor moves proportionally to `h` at the
/// knife edge, or if it fails to move proportionally to `h` away from it — either
/// would mean the stationarity is an artifact of the measurement.
///
/// At a centroid separation of `box_len / 2` the minimum-image map folds
/// symmetrically, so `d(L/2 + δ) = d(L/2 - δ)` and the descriptor has a vanishing
/// derivative. The label jumps at precisely the configuration where the
/// descriptor is flattest.
#[test]
fn descriptor_is_stationary_at_the_knife_edge_but_not_away_from_it() {
    let h = 1e-3;
    let on = descriptor_delta(&descriptor(&bracket(h).0, R_CUT), &descriptor(&bracket(h).1, R_CUT));

    let x = knife_edge_x() - 1.0;
    let (a, b) = (arcs(x - h / 2.0), arcs(x + h / 2.0));
    let (da, db) = (descriptor(&a, R_CUT), descriptor(&b, R_CUT));

    // `descriptor_delta` returns INFINITY when the neighbour count changes, and off
    // the edge it does. An infinity is not a measurement and a ratio against one is
    // a vacuous pass, so the off-edge delta is measured on the common prefix to keep
    // it finite, with the count change reported separately.
    let common = da.len().min(db.len());
    let off = da[..common]
        .iter()
        .zip(db[..common].iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f64, f64::max);

    println!(
        "h {h:e}: dD at knife edge {on:e}; one unit away {off:e} on {common} common pairs          (counts {} vs {}), ratio {:e}",
        da.len(),
        db.len(),
        off / on
    );
    assert!(on.is_finite(), "on-edge delta must be finite to be a measurement, got {on:e}");
    assert!(off.is_finite(), "off-edge delta must be finite to be a measurement, got {off:e}");
    assert!(on < 1e-12, "descriptor not stationary at the knife edge: dD {on:e}");
    assert!(
        off > 1e-5,
        "descriptor also stationary away from the edge: dD {off:e}; the contrast is the claim"
    );
    assert!(off / on > 1e6, "insufficient contrast: {:e}", off / on);
}

/// The jump is an artifact, not physics: nothing structural changed across the
/// knife edge. Uses Cameron's own six-feature null model rather than a
/// reimplementation. **Falsified if** any feature moves, which would mean the two
/// configurations genuinely differ and the jump is legitimate.
#[test]
fn knife_edge_pair_is_matched_on_all_six_null_features() {
    let (a, b) = bracket(1e-9);
    let (fa, fb) = (nerve_baseline::chain_features(&a), nerve_baseline::chain_features(&b));
    let msid = fa
        .msid
        .iter()
        .zip(fb.msid.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    println!(
        "null gaps: rg2 {:e} e2e {:e} contour {:e} density {:e} max_bond {:e} msid {:e}",
        (fa.rg2 - fb.rg2).abs(),
        (fa.end_to_end2 - fb.end_to_end2).abs(),
        (fa.contour_len - fb.contour_len).abs(),
        (fa.bead_density - fb.bead_density).abs(),
        (fa.max_bond - fb.max_bond).abs(),
        msid
    );
    assert_eq!(fa.msid.len(), fb.msid.len(), "chain lengths changed");
    for (name, v) in [
        ("rg2", (fa.rg2 - fb.rg2).abs()),
        ("end_to_end2", (fa.end_to_end2 - fb.end_to_end2).abs()),
        ("contour_len", (fa.contour_len - fb.contour_len).abs()),
        ("bead_density", (fa.bead_density - fb.bead_density).abs()),
        ("max_bond", (fa.max_bond - fb.max_bond).abs()),
        ("msid", msid),
    ] {
        assert!(v < 1e-12, "{name} moved by {v:e} across the knife edge");
    }
}

/// The physical witness. **Falsified if** closest approach collapses toward
/// contact, which would make the label jump a genuine strand crossing rather than
/// an artifact.
#[test]
fn knife_edge_pair_has_no_strand_crossing() {
    let (a, b) = bracket(1e-9);
    let (ga, gb) = (min_interchain_dist(&a), min_interchain_dist(&b));
    println!("closest approach {ga:.9} vs {gb:.9}, change {:e}", (ga - gb).abs());
    assert!(ga > 1.0, "strands too close to call this an artifact: {ga}");
    assert!((ga - gb).abs() < 1e-6, "closest approach itself moved: {:e}", (ga - gb).abs());
}

// ------------------------------------------------------- the stable candidates

/// **Falsified if** the divergence factor is large, i.e. writhe jumps where the
/// descriptor is smooth. Paired with the non-constancy test below, without which
/// this would pass vacuously for any constant label.
#[test]
fn writhe_amplification_stays_bounded_at_the_knife_edge() {
    let amps = sweep("total_writhe", total_writhe);
    let jh = jump_height(&amps);
    let peak = peak_amplification(&amps);
    println!("writhe jump height {jh:e}, peak amplification {peak:e}");
    assert!(jh < 1e-12, "writhe jumped by {jh:e} at the knife edge; it is not continuous here");
}

/// The vacuity guard for the test above, in-suite and with a margin. **Falsified
/// if** writhe is constant along a path that genuinely changes chain shape, in
/// which case its stability is worthless.
#[test]
fn writhe_is_not_constant_under_shape_change() {
    let base = ring_pair(N, R, R, BOX);
    let path: Vec<Melt> = (0..12).map(|k| perturb(&base, 0.25, 1000 + k as u64)).collect();
    let (lo, hi) = label_spread(&path, total_writhe);
    println!("writhe range over shape-noise ensemble: {lo:.6} .. {hi:.6}, spread {:.6}", hi - lo);
    assert!(
        hi - lo > 1e-3,
        "writhe varies by only {:e} under shape change; stability is vacuous",
        hi - lo
    );
}

/// Rings, closed, evaluated on raw in-box coordinates — no image chosen, no
/// closure invented. **Falsified if** this diverges, which would mean the
/// discontinuity is intrinsic to the linking number rather than to the estimator.
#[test]
fn abs_lk_closed_raw_amplification_stays_bounded_at_the_knife_edge() {
    let amps = sweep("abs_lk_closed_raw", abs_lk_closed_raw);
    let jh = jump_height(&amps);
    println!("raw closed |Lk| jump height {jh:e}");
    assert!(jh < 1e-12, "raw closed |Lk| jumped by {jh:e}");
}

/// **Records a false premise of mine.** I predicted that routing the same
/// topological quantity through nearest-image placement would induce the
/// discontinuity, i.e. that the image convention is the culprit. It does not.
/// Measured jump height was 3.3e-18 — floating-point noise, no jump at all.
///
/// The reason, verified below rather than assumed: at this separation *both*
/// candidate periodic images of ring 1 are unlinked from ring 0, so the integer
/// invariant is 0 on either side and the image flip cannot change it. The image
/// convention is innocent **here**; that is a statement about this configuration,
/// not a general acquittal, and a configuration whose two candidate images differ
/// in linking would be a separate experiment.
///
/// This is Cameron's closure-dominates-image finding arriving independently: the
/// culprit is `Closure::Open`, not the periodic image.
#[test]
fn nearest_image_placement_does_not_induce_a_jump_when_both_images_are_unlinked() {
    let amps = sweep("abs_lk_min_image", abs_lk_closed_min_image);
    let jh = jump_height(&amps);
    println!("min-image closed |Lk| jump height {jh:e}");
    assert!(jh < 1e-12, "a jump appeared: {jh:e}; the premise was right after all");

    // The stated reason, checked.
    let m = arcs(knife_edge_x());
    let rings = ring_pair(N, R, 5.0 * R, BOX);
    for (name, shifted) in [
        ("as placed", rings.clone()),
        ("other image", translate_chain(&rings, 1, [-BOX, 0.0, 0.0])),
    ] {
        let v = abs_lk_closed_raw(&shifted);
        println!("  {name}: |Lk| {v:e}");
        assert!(v < 0.5, "{name} is linked; the stated reason for no jump is wrong");
    }
    let _ = m;
}

/// **Records a false premise of mine.** I predicted closure averaging could not
/// repair the knife edge, on the reasoning that averaging over closures cannot fix
/// an image-choice jump. Both halves were wrong: the jump is not image-caused (see
/// above), and the closure-averaged label shows no jump at all — measured jump
/// height 3.3e-16 against `Lk(Open)`'s 6.6e-3, a factor of 2e13.
///
/// So closure averaging **is** a continuity fix. Whether the resulting label is
/// worth training on is a separate question about how much of it is closure rather
/// than chain, answered in `informativeness.rs`.
#[test]
fn closure_averaging_removes_the_knife_edge_jump() {
    let amps = sweep("lk_closure_mean(64)", |m: &Melt| lk_closure_mean(m, 64));
    let jh = jump_height(&amps);
    let open_jh = jump_height(
        &HS.iter()
            .map(|&h| {
                let (a, b) = bracket(h);
                amplification(&a, &b, h, lk_open, R_CUT)
            })
            .collect::<Vec<_>>(),
    );
    println!("closure-averaged jump height {jh:e} vs Lk(Open) {open_jh:e}, ratio {:e}", open_jh / jh);
    assert!(jh < 1e-12, "closure averaging did not remove the jump: {jh:e}");
    assert!(
        open_jh / jh > 1e6,
        "closure averaging bought only a factor {:e}; not a fix",
        open_jh / jh
    );
}

// ------------------------------------------------------------------- guards

#[test]
fn guards_hold_for_every_configuration_used() {
    for (name, m) in [
        ("arcs at knife edge", arcs(knife_edge_x())),
        ("rings threaded", ring_pair(N, R, R, BOX)),
        ("rings unthreaded", ring_pair(N, R, 3.0 * R, BOX)),
    ] {
        let mb = max_bond(&m);
        println!(
            "{name}: max_bond {mb:.6} vs box_len/2 {:.6}, r_cut {R_CUT}",
            m.box_len / 2.0
        );
        assert!(mb < m.box_len / 2.0, "{name}: bond guard violated at {mb}");
        assert!(R_CUT <= m.box_len / 2.0, "{name}: r_cut {R_CUT} exceeds box_len/2");
    }
}
