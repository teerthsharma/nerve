//! Melt-density tests for both blindness mechanisms.
//!
//! Every test here was observed RED before its implementation existed. Where a
//! test is definitional — true by the definition of a distance-only descriptor
//! and therefore incapable of failing — it says so in its own doc comment and is
//! labelled a regression guard, not evidence. That distinction is the whole
//! reason the round-1 result held and Foreman's did not.

use nerve_core::{Chain, Melt, Vec3};
use nerve_melt::*;
use nerve_topo::{closure_spread, image_spread, linking_number, Closure};

/// Melt used for the density-independent claims: 20 chains x 40 beads = 800
/// beads. At rho = 0.85 that is a box of 9.83 sigma.
fn melt(seed: u64) -> Melt {
    ideal_melt(&MeltSpec::kg(20, 40, seed))
}

/// Smaller melt for the O(27 * C^2 * M^2) image-spread sweeps.
fn small(seed: u64, density: f64) -> Melt {
    ideal_melt(&MeltSpec::kg(10, 30, seed).at_density(density))
}

// ------------------------------------------------------------------- fixtures

#[test]
fn ideal_melt_hits_target_density_with_valid_bonds() {
    let spec = MeltSpec::kg(20, 40, 1);
    let m = melt(1);
    assert_eq!(m.n_beads(), 800);
    assert!((m.box_len - spec.box_len()).abs() < 1e-12);
    assert!((m.n_beads() as f64 / m.box_len.powi(3) - 0.85).abs() < 1e-12);

    // Guard: chain features and nerve_topo::unwrap_chain are both meaningless
    // unless every bond is shorter than half the box.
    let f = nerve_baseline::chain_features(&m);
    assert!(f.max_bond < m.box_len / 2.0, "max_bond {} vs box/2 {}", f.max_bond, m.box_len / 2.0);
    assert!((f.max_bond - 0.97).abs() < 1e-9, "bond length drifted: {}", f.max_bond);
    assert!((f.contour_len - 39.0 * 0.97).abs() < 1e-9, "contour {}", f.contour_len);
}

#[test]
fn angular_pair_count_documents_the_on2_ceiling() {
    // The ponytail: marker on nerve-baseline's neighbour search. At rho = 0.85
    // the angular basis cost grows as r_cut^6, so "just widen the cutoff" is
    // expensive as well as ineffective. Volume ratio from 2.0 to 4.0 is 8x, so
    // pair count should grow far faster than that.
    let m = melt(1);
    let (lo, hi) = (angular_pair_count(&m, 2.0), angular_pair_count(&m, 4.0));
    println!("  angular pairs at rho=0.85, 800 beads: r_cut 2.0 -> {lo}, 4.0 -> {hi}");
    assert!(lo > 0);
    assert!(hi > 20 * lo, "expected ~r^6 growth, got {hi}/{lo} = {:.1}x", hi as f64 / lo as f64);
}

// ------------------------------------- mechanism 1: parity, definitional part

/// REGRESSION GUARD, NOT EVIDENCE.
///
/// This cannot fail for a distance-only descriptor: reflection is an exact
/// isometry, the descriptor is a function of minimum-image distances and angles
/// alone, so equality is true by definition. It is here so that a future change
/// making the descriptor parity-aware is noticed, and for no other reason. The
/// falsifiable content of mechanism 1 lives in `parity_pair_is_vacuous_*` below.
#[test]
fn reflection_leaves_the_distance_only_null_model_bitwise_unchanged() {
    let m = melt(1);
    let r = reflect(&m, 0);
    assert_eq!(min_image_max_diff(&m, &r), 0.0, "reflection perturbed a distance");
    assert_eq!(null_model(&m, 2.5), null_model(&r, 2.5));
    // And it is a genuine improper transformation, not a relabelling: the
    // coordinates really did move.
    assert_ne!(m.chains[0].beads[0], r.chains[0].beads[0]);
}

#[test]
fn reflection_flips_the_sign_of_pairwise_linking() {
    // The parity-odd component, and only that. |Lk| is preserved; the sign is
    // not. Open closure: the melt has open chains and Closure::Open is the only
    // scheme with no closure ambiguity.
    let m = melt(1);
    let r = reflect(&m, 0);
    let mut checked = 0;
    for (i, j) in [(0usize, 1usize), (2, 3), (4, 5), (6, 7)] {
        let a = linking_number(&m, i, j, Closure::Open);
        let b = linking_number(&r, i, j, Closure::Open);
        if a.abs() > 1e-6 {
            assert!((a + b).abs() < 1e-9 * a.abs().max(1.0), "pair ({i},{j}): {a} vs {b}");
            checked += 1;
        }
    }
    assert!(checked >= 3, "only {checked} pairs had a sign to flip");
}

// ---------------------------- mechanism 1: the part that can actually fail

/// # STRUCK. This is not a parity result and does not support one.
///
/// The predicate is `2|Lk| > image_spread.max - image_spread.min`, and this
/// crate's own measurements put that denominator at ~1e-16 at every density. So
/// the predicate degenerates to *"is `|Lk|` nonzero in f64"* and the rising
/// fraction below measures how often `Lk` is nonzero — driven by the bead overlap
/// that [`ideal_melt`] documents having, since it has no excluded volume. `2|Lk|`
/// also discards the sign, which is the only thing parity flips. Two further
/// limits now on the record: rho = 0.05 is 17x below melt density and is not a
/// melt, and 30-40 beads is at or below the Kremer-Grest entanglement length.
///
/// Retained under its true name rather than deleted, with the four fractions
/// pinned so they cannot drift inside a correct-shaped inequality — which was the
/// third defect found. The replacement instrument, with a measured nonzero floor,
/// is `writhe_sign_is_resolvable_against_a_measured_seed_noise_floor`.
///
/// The original claim was: a parity matched pair hides the sign of Lk, and is
/// non-vacuous only where
///
///     2 * |Lk|  >  image_spread.max - image_spread.min
///
/// # Falsified prediction, recorded rather than erased
///
/// The hypothesis committed before running was: *the ambiguity grows with density
/// faster than |Lk| does, so the non-vacuous fraction at rho = 0.85 is strictly
/// below the fraction at rho = 0.05.* It was logged RED, implemented, and then
/// **failed on real numbers**: 0.067, 0.185, 0.385, 0.511 across
/// rho = 0.05, 0.20, 0.45, 0.85 — monotonically *rising*.
///
/// The assertion below is the inverted claim, and the inversion is a
/// falsification, not a rewrite-to-fit: the original prediction and its numbers
/// are above, and the board log carries the same record. The cause is now
/// understood — `image_spread` returns ~1e-16 at every density, so this margin is
/// really only asking whether `|Lk|` is nonzero, and bead overlap makes that more
/// common as density rises. The denominator that actually binds is the closure
/// one, tested separately below.
///
#[test]
fn struck_fraction_of_pairs_with_nonzero_lk_rises_with_density_in_an_ev_free_melt() {
    let densities = [0.05, 0.2, 0.45, 0.85];
    println!("\n  [STRUCK instrument] rho   pairs with |Lk| resolvable above ~1e-16");
    let mut fracs = Vec::new();
    for d in densities {
        let mut ok = 0;
        let mut tot = 0;
        for seed in 1..=3u64 {
            let m = small(seed, d);
            // Direct, not Open: image_spread always evaluates the closed
            // integral, so the value and its own bound must be the same
            // integral or the margin compares two different quantities.
            let (a, b) = non_vacuous_fraction(&m, Closure::Direct);
            ok += a;
            tot += b;
        }
        let f = ok as f64 / tot as f64;
        fracs.push(f);
        println!("  {d:4.2}    {ok}/{tot} = {:.3}", f);
    }

    // Pinned to the exact measured values. A correct-shaped inequality let these
    // drift arbitrarily, which is the third defect the audit found.
    let expected = [9.0 / 135.0, 25.0 / 135.0, 52.0 / 135.0, 69.0 / 135.0];
    for (got, want) in fracs.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-12, "struck measurement drifted: {fracs:?}");
    }
}

/// The denominator that actually binds.
///
/// With the periodic-image ambiguity measured at ~1e-16, the non-vacuous test
/// above is really only asking whether `|Lk|` is nonzero. The closure ambiguity
/// is 15 orders of magnitude larger, so it is the one that decides whether a
/// melt pair has a quotable linking sign at all.
///
/// # Second falsified prediction, recorded rather than erased
///
/// Committed before running: *almost no pair survives this denominator at melt
/// density — under 20% — because `closure_spread.sd` is comparable to a full
/// linking unit while `|Lk|` is a fraction of one.* Logged RED, implemented, and
/// **failed: 37/135 = 0.274.**
///
/// So the closure ambiguity does not wipe out the signal. Roughly a quarter of
/// open-chain pairs at rho = 0.85 have a linking sign that survives their own
/// closure spread, and for exactly those pairs a parity matched pair is a
/// genuine, non-vacuous, exact blindness pair. Both of my committed predictions
/// this round were wrong in the project's favour; that is the finding.
///
/// The assertion is now two-sided and both bounds can fail: above 0.20 (the
/// signal is not wiped out) and below 0.60 (most pairs still have no quotable
/// sign, so this is a minority regime, not the default).
#[test]
fn quotable_linking_sign_survives_for_a_minority_of_melt_pairs() {
    let mut ok = 0;
    let mut tot = 0;
    for seed in 1..=3u64 {
        let (a, b) = closure_vacuous_fraction(&small(seed, 0.85), 16);
        ok += a;
        tot += b;
    }
    let f = ok as f64 / tot as f64;
    println!("\n  pairs with 2|Lk| > closure sd at rho=0.85: {ok}/{tot} = {f:.3}");
    assert!(f > 0.20, "closure ambiguity wiped out the signal after all: {f:.3}");
    assert!(f < 0.60, "quotable linking is the majority regime, not a minority: {f:.3}");
}

#[test]
fn closure_spread_is_reported_alongside_any_melt_linking_number() {
    // Second ambiguity, independent of the image choice: for open chains the
    // closure invents curve that is not in the data. Any Lk quoted for a melt
    // pair without this number beside it is unquotable.
    let m = small(1, 0.85);
    println!("\n  pair   Lk(Open)      image ambiguity   closure sd");
    let mut n_dominated = 0;
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 3), (3, 4)] {
        let lk = linking_number(&m, i, j, Closure::Open);
        let im = image_spread(&m, i, j, Closure::Direct);
        let cs = closure_spread(&m, i, j, 16);
        println!("  {i},{j}    {lk:+.4e}   {:.4e}        {:.4e}", im.max - im.min, cs.sd);
        if cs.sd > lk.abs() {
            n_dominated += 1;
        }
    }
    assert!(n_dominated > 0, "closure spread never dominated - unexpectedly clean");
}

// ------------------------------------------- mechanism 2: the PBC-image case

/// The PBC-image mechanism, constructed explicitly.
///
/// Two configurations an epsilon apart in every minimum-image distance, so any
/// distance-only descriptor differs by O(epsilon), while `linking_number` differs
/// by O(1) because the centroid separation crosses `box_len/2` and the nearest
/// image flips. If the chains never approach each other across the move, no
/// strand crossing occurred, so no genuine topological change is possible and
/// the jump is the estimator's.
#[test]
fn image_choice_jumps_linking_discontinuously_without_a_strand_crossing() {
    let m = small(1, 0.85);
    let l = m.box_len;
    let eps = 1e-9;

    // Put chain 1's centroid exactly box_len/2 from chain 0's along x, then step
    // an epsilon to either side of that knife edge.
    let ca = centroid_of(&m, 0);
    let cb = centroid_of(&m, 1);
    let to_edge = l / 2.0 - (cb[0] - ca[0]);
    let lo = translate_chain(&m, 1, [to_edge - eps, 0.0, 0.0]);
    let hi = translate_chain(&m, 1, [to_edge + eps, 0.0, 0.0]);

    // 1. Distance-only descriptors cannot tell these apart beyond O(eps).
    let dd = min_image_max_diff(&lo, &hi);
    assert!(dd < 1e-7, "minimum-image distances moved by {dd:.3e}, not O(eps)");

    // 2. No strand crossing: the two chains stay well separated in both.
    let (sep_lo, sep_hi) = (min_pair_distance(&lo, 0, 1), min_pair_distance(&hi, 0, 1));
    assert!(sep_lo > 0.1 && sep_hi > 0.1, "chains nearly touch: {sep_lo:.3} / {sep_hi:.3}");

    // 3. And yet the linking number jumps.
    let (lk_lo, lk_hi) = (
        linking_number(&lo, 0, 1, Closure::Open),
        linking_number(&hi, 0, 1, Closure::Open),
    );
    let jump = (lk_hi - lk_lo).abs();
    println!(
        "\n  knife edge: distances moved {dd:.3e}, closest approach {sep_lo:.3}/{sep_hi:.3}, \
         Lk {lk_lo:+.6e} -> {lk_hi:+.6e}, jump {jump:.3e}"
    );
    assert!(
        jump > 1e4 * dd,
        "Lk jumped {jump:.3e} for a distance change of {dd:.3e} - no discontinuity found"
    );
}

#[test]
fn image_spread_flags_the_knife_edge_pair_as_ambiguous() {
    // The instrument has to catch what the construction exploits, or it is not a
    // usable guard. Ambiguity must be at least as large as the value itself.
    let m = small(1, 0.85);
    let l = m.box_len;
    let ca = centroid_of(&m, 0);
    let cb = centroid_of(&m, 1);
    let edge = translate_chain(&m, 1, [l / 2.0 - (cb[0] - ca[0]) - 1e-9, 0.0, 0.0]);

    let lk = linking_number(&edge, 0, 1, Closure::Direct).abs();
    let sp = image_spread(&edge, 0, 1, Closure::Direct);
    println!("\n  knife-edge pair: |Lk| {lk:.4e}, image ambiguity {:.4e}", sp.max - sp.min);
    assert!(
        sp.max - sp.min >= lk,
        "image_spread reported ambiguity {:.3e} below the value {lk:.3e} it is meant to bound",
        sp.max - sp.min
    );
}

/// The control, applied to myself: an ambiguity that is large everywhere is not
/// evidence of a knife edge.
///
/// If a randomly chosen melt pair has the same image ambiguity as the
/// deliberately constructed knife-edge pair, the construction demonstrates
/// nothing about image choice — it demonstrates that periodic linking is
/// ill-defined for every pair. Committed hypothesis: the knife-edge pair is NOT
/// special, so this test asserts the opposite of what the mechanism needs and
/// fails if the mechanism is real.
#[test]
fn knife_edge_ambiguity_is_not_worse_than_a_typical_melt_pair() {
    let m = small(1, 0.85);
    let l = m.box_len;
    let ca = centroid_of(&m, 0);
    let cb = centroid_of(&m, 1);
    let edge = translate_chain(&m, 1, [l / 2.0 - (cb[0] - ca[0]) - 1e-9, 0.0, 0.0]);

    let sp = image_spread(&edge, 0, 1, Closure::Direct);
    let knife = sp.max - sp.min;

    let mut typical = Vec::new();
    for i in 0..m.chains.len() {
        for j in (i + 1)..m.chains.len() {
            let s = image_spread(&m, i, j, Closure::Direct);
            typical.push(s.max - s.min);
        }
    }
    typical.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = typical[typical.len() / 2];
    println!("\n  image ambiguity: knife-edge pair {knife:.4e}, median melt pair {median:.4e}");
    assert!(
        knife <= 10.0 * median,
        "the knife-edge pair is {:.1}x more ambiguous than a typical pair - image choice IS a \
         localised defect and the mechanism is real",
        knife / median
    );
}

// -------------------------------------------------- same-seed signal/noise

#[test]
fn parity_signal_is_exactly_zero_against_a_nonzero_seed_noise_floor() {
    // Round-1 control, applied here. Signal is the null model's separation of a
    // melt from its own mirror; noise is its separation of two melts that differ
    // only by seed. Signal/noise = 0 exactly is the strongest form this can take
    // and needs no averaging - but the noise floor must be shown nonzero, or the
    // ratio is 0/0 and means nothing.
    let a = melt(1);
    let signal = cosine_distance(&null_model(&a, 2.5), &null_model(&reflect(&a, 0), 2.5));
    let noise: Vec<f64> = (2..=4u64)
        .map(|s| cosine_distance(&null_model(&a, 2.5), &null_model(&melt(s), 2.5)))
        .collect();
    let mean_noise = noise.iter().sum::<f64>() / noise.len() as f64;
    println!("\n  parity signal {signal:.3e}, same-arch seed noise {mean_noise:.3e}");
    assert_eq!(signal, 0.0);
    assert!(mean_noise > 1e-9, "noise floor is zero, ratio is meaningless: {noise:?}");
}

// ---------------------------------------------------------------- guard rails

#[test]
fn null_model_rejects_a_cutoff_above_half_the_box() {
    let m = small(1, 0.85);
    let too_big = m.box_len / 2.0 + 0.01;
    assert!(std::panic::catch_unwind(|| null_model(&m, too_big)).is_err());
}

#[test]
fn reconnection_mechanism_is_inadmissible_at_melt_density() {
    // Round 1's exact mechanism - same point set, different connectivity - does
    // not transfer to a dense melt. The reconnected melt has bonds longer than
    // box_len/2, so it is not a physically realisable polymer configuration and
    // nerve_topo::unwrap_chain silently picks the wrong image. Reporting the
    // mechanism as surviving at melt density would be false.
    let m = melt(1);
    let all: Vec<_> = m.chains.iter().flat_map(|c| c.beads.iter().copied()).collect();
    let n = m.chains.len();
    let mut ch = vec![Vec::new(); n];
    for (k, b) in all.iter().enumerate() {
        ch[k % n].push(*b);
    }
    let recon = Melt::new(ch.into_iter().map(Chain::new).collect(), m.box_len);
    let f = nerve_baseline::chain_features(&recon);
    assert!(
        f.max_bond > m.box_len / 2.0,
        "reconnected melt passed the bond guard ({} vs {}), so the mechanism does transfer",
        f.max_bond,
        m.box_len / 2.0
    );
}

// ============================================================ round 3: writhe
//
// Scope restated at the top of the section because it applies to every test in
// it: open-curve writhe is NOT a topological invariant (nerve-topo's own
// `writhe_is_not_integral` holds) and it is a SINGLE-CURVE quantity, not a link
// invariant. It says nothing about entanglement between two chains. Its three
// useful properties are that it is parity-odd, reversal-invariant, and
// closure-free.

/// Chains long enough to be above the Kremer-Grest entanglement length
/// (N_e ~ 30-85 for KG), closing a limit left open in round 2.
fn long_melt(seed: u64) -> Melt {
    ideal_melt(&MeltSpec::kg(10, 100, seed))
}

/// INHERITED-PROPERTY CHECK, NOT EVIDENCE.
///
/// That reflection negates writhe is a property of `nerve-topo`, already covered
/// by its own suite. Asserted here only because every round-3 claim rests on it,
/// so a change there should break this crate loudly rather than quietly.
#[test]
fn writhe_flips_sign_under_reflection_per_chain() {
    let m = long_melt(1);
    let w = writhe_per_chain(&m);
    let wr = writhe_per_chain(&reflect(&m, 0));
    assert_eq!(w.len(), 10);
    for (a, b) in w.iter().zip(&wr) {
        assert!((a + b).abs() < 1e-9 * a.abs().max(1.0), "{a} vs {b}");
        assert!(a.abs() > 1e-6, "writhe is numerically zero, nothing to flip: {a}");
    }
}

/// The direct comparison against round 2's surviving knife-edge measurement.
///
/// `Lk(Open)` jumped 4.925e-2 for a 2.0e-9 coordinate move because the nearest
/// periodic image flipped. Writhe involves no image choice and no closure, so it
/// should not jump at all.
#[test]
fn writhe_does_not_jump_at_the_knife_edge_where_linking_did() {
    let m = small(1, 0.85);
    let l = m.box_len;
    let ca = centroid_of(&m, 0);
    let cb = centroid_of(&m, 1);
    let to_edge = l / 2.0 - (cb[0] - ca[0]);
    let lo = translate_chain(&m, 1, [to_edge - 1e-9, 0.0, 0.0]);
    let hi = translate_chain(&m, 1, [to_edge + 1e-9, 0.0, 0.0]);

    let lk_jump = (linking_number(&hi, 0, 1, Closure::Open)
        - linking_number(&lo, 0, 1, Closure::Open))
    .abs();
    let w_jump = writhe_per_chain(&hi)
        .iter()
        .zip(writhe_per_chain(&lo))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("\n  knife edge: Lk jump {lk_jump:.4e}, worst writhe jump {w_jump:.4e}");
    assert!(lk_jump > 1e-3, "the knife edge stopped biting Lk: {lk_jump:.3e}");
    assert!(w_jump < 1e-9, "writhe jumped too: {w_jump:.3e}");
}

/// Why the excluded-volume work below is not optional.
///
/// `ideal_melt` with `min_sep = 0` grows the same walk regardless of density —
/// only the box changes, and every bond is far shorter than `box_len/2`, so
/// unwrapping recovers that same walk exactly. A single-chain quantity is
/// therefore *bitwise identical* across densities, and any density sweep of it is
/// vacuous by construction rather than physically flat.
///
/// This is the sharpened form of the strike against my round-2 sweep: without
/// excluded volume there is no mechanism by which density could reach the
/// observable at all.
/// The first form of this test asserted *bitwise* equality and was RED: the two
/// writhe vectors agree to about 13 significant figures, not to the last bit
/// (e.g. -1.5421029354145026 vs -1.5421029354144982). The residual is wrapping
/// arithmetic — `rem_euclid` rounds when placing a bead in the box and
/// `min_image` recovers the bond to within an ULP — and it scales with box size,
/// which is the only thing density changes here. The claim is unchanged; the
/// tolerance was wrong. 1e-12 relative is fifteen orders below any physical
/// effect.
#[test]
fn writhe_is_density_independent_without_excluded_volume() {
    let base = MeltSpec::kg(10, 100, 7);
    let a = writhe_per_chain(&ideal_melt(&base.at_density(0.20)));
    let b = writhe_per_chain(&ideal_melt(&base.at_density(0.85)));
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0f64, f64::max);
    println!("\n  writhe across rho 0.20 vs 0.85, no excluded volume: worst rel diff {worst:.3e}");
    assert!(worst < 1e-12, "writhe moved with density in an EV-free melt: {worst:.3e}");
    assert!(worst > 0.0, "if it were exactly bitwise, wrapping arithmetic would be exact");
}

/// The density ceiling of rejection growth, as a measurement rather than a guess.
///
/// Committed hypothesis: hard-core rejection at `min_sep = 0.9` succeeds at
/// rho = 0.20 and fails at rho = 0.85, so the excluded-volume melt cannot reach
/// true Kremer-Grest density by this method and no density-dependent fraction may
/// be quoted there.
#[test]
fn self_avoiding_growth_has_a_density_ceiling_below_melt_density() {
    let base = MeltSpec::kg(10, 100, 3).with_excluded_volume(0.9);
    let mut reached = Vec::new();
    for d in [0.10, 0.20, 0.30, 0.45, 0.60, 0.70, 0.85] {
        let ok = try_melt(&base.at_density(d)).is_some();
        println!("  rho {d:4.2}  self-avoiding growth: {}", if ok { "placed" } else { "FAILED" });
        if ok {
            reached.push(d);
        }
    }
    assert!(reached.contains(&0.20), "rejection failed even at rho=0.20: {reached:?}");
    assert!(!reached.contains(&0.85), "rejection reached melt density; the ceiling claim is wrong");
}

/// The replacement for the struck predicate.
///
/// Floor is the same-architecture different-seed spread of writhe itself —
/// measured, and nonzero, unlike the ~1e-16 image spread that got round 2 struck.
/// Per chain, never pooled.
///
/// Committed hypothesis: over half of chains have a writhe sign resolvable
/// against that floor, at every density rejection growth can reach.
#[test]
fn writhe_sign_is_resolvable_against_a_measured_seed_noise_floor() {
    let base = MeltSpec::kg(10, 100, 0).with_excluded_volume(0.9);
    println!("\n  rho    seed-noise floor sd(Wr)   resolvable chains");
    let mut fracs = Vec::new();
    for d in [0.10, 0.20, 0.30, 0.45, 0.60] {
        // Floor: spread of writhe across same-architecture melts differing only
        // by seed. This is the round-1 control, transplanted.
        let ensemble: Vec<f64> = (1..=6u64)
            .flat_map(|s| writhe_per_chain(&ideal_melt(&MeltSpec { seed: s, ..base }.at_density(d))))
            .collect();
        let floor = sd(&ensemble);
        let (ok, tot) = resolvable_fraction(&ensemble, floor);
        let f = ok as f64 / tot as f64;
        fracs.push(f);
        println!("  {d:4.2}   {floor:.4e}                {ok}/{tot} = {f:.3}");
    }
    for f in &fracs {
        assert!(*f > 0.5, "writhe sign not resolvable above its own noise floor: {fracs:?}");
    }
}

/// Apples-to-apples against the `Lk` witness on the same melt: same gap
/// definition, same style of measured floor.
///
/// Committed hypothesis: writhe's median margin beats `Lk(Open)`'s, because
/// writhe carries no closure ambiguity while `Lk(Open)`'s closure spread was
/// measured at 0.331-0.559 against `|Lk|` of 2.3e-3 to 1.8e-1.
#[test]
fn writhe_margin_beats_the_linking_margin_on_the_same_melt() {
    let m = long_melt(1);
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    let w = writhe_per_chain(&m);
    let w_floor = sd(&(1..=6u64).flat_map(|s| writhe_per_chain(&long_melt(s))).collect::<Vec<_>>());
    let w_margin = median(w.iter().map(|x| 2.0 * x.abs() / w_floor).collect());

    let mut lks = Vec::new();
    for i in 0..m.chains.len() {
        for j in (i + 1)..m.chains.len() {
            lks.push(linking_number(&m, i, j, Closure::Open));
        }
    }
    let lk_floor = sd(&lks);
    let lk_margin = median(lks.iter().map(|x| 2.0 * x.abs() / lk_floor).collect());

    println!("\n  median margin: writhe {w_margin:.3}, Lk(Open) {lk_margin:.3}");
    assert!(
        w_margin > lk_margin,
        "writhe margin {w_margin:.3} did not beat Lk {lk_margin:.3}"
    );
}

/// Foreman's pooled-vs-per-bead warning, inherited rather than rediscovered.
///
/// Committed hypothesis: summing writhe over chains cancels most of the per-chain
/// signal, because a melt has no net chirality — so the pooled magnitude is a
/// small fraction of the summed magnitudes.
#[test]
fn pooling_writhe_over_chains_cancels_the_per_chain_signal() {
    let w = writhe_per_chain(&long_melt(1));
    let pooled = w.iter().sum::<f64>().abs();
    let per_chain = w.iter().map(|x| x.abs()).sum::<f64>();
    println!("\n  |sum Wr| {pooled:.4e} vs sum|Wr| {per_chain:.4e} = {:.3}", pooled / per_chain);
    assert!(
        pooled < 0.5 * per_chain,
        "pooling did not cancel: {pooled:.3e} vs {per_chain:.3e}"
    );
}

/// Independent replication of `nerve-order`'s pooled-vs-per-bead contrast, on a
/// second implementation of the same invariant and on a melt rather than rings.
///
/// This is the reach question. If a chirality-sensitive descriptor separates a
/// melt from its mirror, then the sign writhe carries is *reachable* by that
/// descriptor class, and parity blindness does not extend to it.
///
/// # Third falsified prediction, and it caveats someone else's surviving result
///
/// Committed before running: *per-bead separates (> 1e-3) and globally pooled does
/// not (< 1e-9), reproducing `nerve-order`'s 1.16e-1 vs 1.11e-16.* **Falsified.**
/// On an ideal melt at rho = 0.85, pooling is not blind at all, and it separates
/// the mirror *better* than per-bead does — the reverse ordering, stable across
/// four seeds:
///
/// | seed | per-bead | pooled | pooled/per-bead |
/// |---|---|---|---|
/// | 1 | 2.0781e-2 | 1.4685e-1 | 7.07 |
/// | 2 | 1.7493e-2 | 2.9869e-2 | 1.71 |
/// | 3 | 3.0220e-2 | 1.5507e-1 | 5.13 |
/// | 4 | 2.8899e-2 | 1.3401e-1 | 4.64 |
///
/// This does not contradict `nerve-order`; it bounds it. The global sum of
/// per-bead signed volumes is itself a pseudoscalar, so it vanishes only when the
/// configuration has no net chirality. Wavy rings are net-achiral by
/// construction, so pooling cancels there. A random melt has a net chiral
/// fluctuation that does not average away at 300 beads, so pooling retains and
/// even amplifies the signal. **The pooled-vs-per-bead contrast is a property of
/// the configuration, not of the pooling.** Anyone quoting 1.11e-16 as evidence
/// that pooling destroys chirality needs the net-achirality precondition attached.
///
/// The reach conclusion is unaffected and is what this test is for: both channels
/// see the mirror, so the sign writhe carries is reachable by a parity-odd
/// descriptor.
#[test]
fn per_bead_and_pooled_signed_volume_both_detect_the_mirror_on_a_melt() {
    let m = small(1, 0.85);
    let r = reflect(&m, 0);
    let (a, b) = (per_bead_signed_volumes(&m, 12), per_bead_signed_volumes(&r, 12));

    let per_bead = cosine_distance(&a, &b);
    let pooled_a = a.iter().sum::<f64>();
    let pooled_b = b.iter().sum::<f64>();
    let pooled = (pooled_a - pooled_b).abs() / a.iter().map(|x| x.abs()).sum::<f64>();
    println!("\n  seed 1 mirror separation: per-bead {per_bead:.4e}, globally pooled {pooled:.4e}");
    assert!(per_bead > 1e-3, "per-bead chiral descriptor is blind to the mirror: {per_bead:.3e}");
    assert!(pooled > 1e-3, "pooled volume was blind here: {pooled:.3e}");

    // The inversion, across four seeds, so it is not one draw.
    for s in 1..=4u64 {
        let mm = small(s, 0.85);
        let rr = reflect(&mm, 0);
        let (x, y) = (per_bead_signed_volumes(&mm, 12), per_bead_signed_volumes(&rr, 12));
        let pb = cosine_distance(&x, &y);
        let pl = (x.iter().sum::<f64>() - y.iter().sum::<f64>()).abs()
            / x.iter().map(|v| v.abs()).sum::<f64>();
        println!("  seed {s}: per-bead {pb:.4e}, pooled {pl:.4e}, ratio {:.2}", pl / pb);
        assert!(pb > 1e-3 && pl > 1e-3, "seed {s}: a channel went blind: {pb:.3e}, {pl:.3e}");
        assert!(pl > pb, "seed {s}: pooling did not beat per-bead: {pl:.3e} vs {pb:.3e}");
    }
}

/// REGRESSION GUARD, NOT EVIDENCE.
///
/// A distance-only descriptor cannot see a reflection; that is definitional and
/// cannot fail. Present so that making the null model chirality-aware breaks
/// loudly. The evidence about reach is the test above, which shows the sign IS
/// visible to a descriptor that carries a parity-odd channel.
#[test]
fn distance_only_null_model_stays_blind_while_writhe_flips() {
    let m = long_melt(1);
    let r = reflect(&m, 0);
    assert_eq!(null_model(&m, 2.5), null_model(&r, 2.5));
    let (w, wr) = (writhe_per_chain(&m), writhe_per_chain(&r));
    assert!(w.iter().zip(&wr).all(|(a, b)| (a + b).abs() < 1e-9 * a.abs().max(1.0)));
}

// ================================ label-free bounds, ported from `branchcut`
//
// `branchcut` (C:\Users\seal\Documents\seal\branchcut, Python, numpy-only, MIT,
// 31 tests passing). Theorem numbers are its own and are cited, not re-derived.
// These replace the struck ratio: Theorem 1 needs no noise-floor denominator and
// no ground-truth labels, so it is immune to the failure that struck round 2.
//
// VERIFICATION TESTS, NOT RED-FIRST EVIDENCE.
//
// The five tests in this section were written *after* the port they exercise, and
// all five passed on first run. I logged RED for them without observing it; those
// log entries are retracted on the board. They verify the port behaves as
// `branchcut` documents, and they are not admissible as RED-first evidence. The
// twenty-two tests above this line were genuine RED-first, several of them
// falsifying the hypothesis they were written to defend.

/// A set of configurations on which writhe is injective, and the descriptor's
/// collision partition over it.
fn mirror_set(k: u64) -> (Vec<Melt>, Vec<Vec<f64>>, Vec<f64>) {
    let mut cfgs = Vec::new();
    for s in 1..=k {
        let m = small(s, 0.85);
        cfgs.push(reflect(&m, 0));
        cfgs.push(m);
    }
    let desc: Vec<Vec<f64>> = cfgs.iter().map(|m| null_model(m, 2.5)).collect();
    // Ground relation: the per-chain writhe vector, reduced to a scalar key only
    // for the injectivity check below.
    let key: Vec<f64> = cfgs
        .iter()
        .map(|m| writhe_per_chain(m).iter().enumerate().map(|(i, w)| w * (i as f64 + 1.7)).sum())
        .collect();
    (cfgs, desc, key)
}

/// Theorem 1 (`branchcut/partition.py`), applied to the distance-only null model.
///
/// Over `n` configurations on which writhe is injective, if the descriptor emits
/// only `m` distinct values it is provably wrong on at least `n - m` of them —
/// **with no labelled set and no noise-floor denominator.** That is the property
/// the struck predicate lacked: nothing here degenerates to a machine-epsilon
/// comparison, because nothing here is a comparison against a magnitude.
///
/// Committed hypothesis: mirror pairs collide and nothing else does, so
/// `m = n/2`, the certified error floor is 0.5, and Theorem 2 caps any downstream
/// recovery at 0.5 as well.
///
/// Stated plainly so it is not oversold: the 0.5 follows from building blocks of
/// size two. The non-trivial content is that the injectivity precondition is
/// *verified* rather than assumed, that no accidental collisions inflate the
/// count, and that the consequence is a cap on **any** downstream head rather
/// than a statement about one descriptor.
#[test]
fn collision_error_floor_certifies_a_blindness_rate_without_labels() {
    let (_, desc, key) = mirror_set(6);
    let n = desc.len();

    // Precondition, checked and not assumed: writhe must separate all n.
    let inj = collides(&key.iter().map(|k| vec![*k]).collect::<Vec<_>>(), 1e-9).is_empty();
    assert!(inj, "writhe is not injective on this set; the bound would certify nothing");

    let p = components(n, &collides(&desc, 1e-12), inj);
    println!(
        "\n  Theorem 1: n={} configs -> m={} distinct descriptor values, largest block {}",
        p.n(),
        p.m(),
        p.largest()
    );
    println!(
        "  certified error floor {:.4} (>= this fraction provably wrong, no labels)",
        p.certified_error_rate()
    );
    println!("  Theorem 2 recovery ceiling {:.4} for ANY downstream head", p.recovery_ceiling());

    assert_eq!(p.m(), n / 2, "expected exactly the mirror pairs to collide");
    assert_eq!(p.largest(), 2);
    assert!((p.certified_error_rate() - 0.5).abs() < 1e-12);
    assert!((p.recovery_ceiling() - 0.5).abs() < 1e-12);
}

/// The ported guard, which is the whole reason to port rather than reinvent.
///
/// `branchcut`: *"If R is genuinely many-to-one, distinct elements should collide
/// and `n - m` bounds nothing."* Traversal reversal preserves writhe, so a set
/// containing a configuration and its reversal has a non-injective ground
/// relation. The bound must then return its vacuous value rather than a wrong
/// number — the failure mode that struck [`parity_margin`], which returned a
/// number where it should have declared itself vacuous.
#[test]
fn bounds_declare_themselves_vacuous_when_writhe_is_not_injective() {
    let m = small(1, 0.85);
    let rev = reverse_chains(&m);
    let (w, wr) = (writhe_per_chain(&m), writhe_per_chain(&rev));
    let worst = w
        .iter()
        .zip(&wr)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0f64, f64::max);
    println!("\n  reversal changes writhe by at most {worst:.3e} - so it is not injective here");
    assert!(worst < 1e-9, "reversal changed writhe; the premise of this test is gone");
    assert_ne!(m.chains[0].beads, rev.chains[0].beads, "reversal was a no-op");

    let desc = vec![null_model(&m, 2.5), null_model(&rev, 2.5)];
    let honest = components(2, &collides(&desc, 1e-12), false);
    let overclaimed = components(2, &collides(&desc, 1e-12), true);
    assert_eq!(honest.certified_errors(), 0, "a non-injective relation must certify nothing");
    assert!(
        overclaimed.certified_errors() >= honest.certified_errors(),
        "the injective=true path is the one that would have overclaimed"
    );
}

/// The plateau diagnostic (`branchcut/plateau.py`) applied to my own threshold.
///
/// The collision grouping above used `atol = 1e-12`. A grouping obtained by
/// thresholding inherits the threshold, so the number is only defensible if the
/// count holds steady across a range of tolerances. Committed hypothesis: there
/// is a plateau, and it sits at `n/2`.
#[test]
fn the_collision_threshold_sits_inside_a_plateau() {
    let (_, desc, _) = mirror_set(6);
    let sweep = collision_tolerance_sweep(&desc, 1e-15, 1e-4, 12);
    println!("\n  atol -> groups (max block)");
    for r in &sweep {
        println!("  {:.2e} -> {} ({})", r.atol, r.n_groups, r.max_multiplicity);
    }
    let (flat, count) = has_plateau(&sweep, 3);
    println!("  plateau: {flat} at {count} groups");
    assert!(flat, "no plateau: the threshold is reporting itself, not the data");
    assert_eq!(count, desc.len() / 2, "plateau is not at the mirror-pair count");
}

/// Negative control for the instrument, in `branchcut`'s own spirit: the sweep
/// must fail to find a plateau when there is no true collision to find.
///
/// Same melts, but perturbed by a lattice translation of one chain instead of
/// mirrored. These configurations are genuinely distinct to the descriptor, so
/// the grouping should slide from singletons to one blob with no flat stretch.
#[test]
fn tolerance_sweep_finds_no_plateau_without_true_collisions() {
    let desc: Vec<Vec<f64>> = (1..=6u64)
        .flat_map(|s| {
            let m = small(s, 0.85);
            let t = translate_chain(&m, 0, [0.37, -0.21, 0.11]);
            [null_model(&m, 2.5), null_model(&t, 2.5)]
        })
        .collect();
    let sweep = collision_tolerance_sweep(&desc, 1e-15, 1e-4, 12);
    let (flat, count) = has_plateau(&sweep, 3);
    println!("\n  no-collision control: plateau {flat} at {count}");
    assert!(!flat, "found a plateau where there are no true collisions: {count} groups");
}

/// Control on Theorem 1's input: the collisions must be exactly the mirror pairs.
///
/// If the descriptor accidentally merged two unrelated configurations, `m` would
/// drop and the floor would rise for a reason that has nothing to do with parity.
#[test]
fn the_only_descriptor_collisions_are_the_constructed_mirror_pairs() {
    let (_, desc, _) = mirror_set(6);
    let found = collides(&desc, 1e-12);
    // mirror_set pushes each mirror immediately before its melt, so the pairs are
    // (0,1), (2,3), ...
    let expected: Vec<(usize, usize)> = (0..desc.len() / 2).map(|i| (2 * i, 2 * i + 1)).collect();
    println!("\n  collisions found: {found:?}");
    assert_eq!(found, expected, "accidental collisions would inflate the certified floor");
}

// ================================================ real MD-equilibrated melts
//
// Reader for the LAMMPS data format, so hand-grown melts can be replaced by
// Svaneborg & Everaers, Zenodo 7319837 (CC-BY-4.0): equilibrated Kremer-Grest
// melts, 823-8408 beads/chain, 414-583 chains per file (M is chain count),
// 2M-8M beads, kappa -2.0 to
// 6.0. That dataset removes three caveats at once - no excluded volume, not
// MD-equilibrated, below entanglement length.
//
// THE DATA IS NOT PRESENT ON THIS MACHINE AND I HAVE NOT FETCHED IT. A ~1.4 GB
// download is not something an agent instruction authorises; that needs the
// user's own go-ahead. So the reader is tested against inline fixtures that
// exercise the format's real hazards, and is ready for the archive the moment
// someone puts it on disk. Files are gzipped; decompression is the caller's job
// (`gunzip`), which is why the entry point takes a &str and no gzip dependency
// is added.

/// Minimal molecular-style LAMMPS data file: two 4-bead chains in a cubic box.
///
/// Deliberately hostile in the ways the real files are: atom IDs are out of
/// order, bead order along each chain is given only by the Bonds section, image
/// flags are present and nonzero, and section headers carry trailing comments.
const FIXTURE: &str = "\
LAMMPS data file via write_data

    8 atoms
    6 bonds
    2 atom types
    1 bond types

  0.0 10.0 xlo xhi
  0.0 10.0 ylo yhi
  0.0 10.0 zlo zhi

Atoms # molecular

3 1 1 2.0 1.0 1.0 0 0 0
1 1 1 0.5 1.0 1.0 -1 0 0
4 1 1 2.5 1.0 1.0 0 0 0
2 1 1 1.2 1.0 1.0 0 0 0
7 2 1 6.0 5.0 5.0 0 0 0
5 2 1 4.5 5.0 5.0 0 0 0
8 2 1 6.6 5.0 5.0 0 0 0
6 2 1 5.2 5.0 5.0 0 0 0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 5 6
5 1 6 7
6 1 7 8
";

#[test]
fn lammps_reader_recovers_chains_ordered_by_bonds_not_by_id() {
    let m = from_lammps_str(FIXTURE).expect("fixture should parse");
    assert_eq!(m.chains.len(), 2, "one chain per molecule ID");
    assert_eq!(m.box_len, 10.0);
    for c in &m.chains {
        assert_eq!(c.len(), 4);
    }
    // Chain 1 must come back in bond order 1-2-3-4, i.e. ascending x, even though
    // the Atoms section listed them 3,1,4,2. Reading in file order would give
    // 2.0, 0.5, 2.5, 1.2 and every bond length would be wrong.
    let xs: Vec<f64> = m.chains[0].beads.iter().map(|b| b[0]).collect();
    assert_eq!(xs, vec![0.5, 1.2, 2.0, 2.5], "chain not ordered by bonds: {xs:?}");

    // And the guard every downstream quantity needs.
    let f = nerve_baseline::chain_features(&m);
    assert!(f.max_bond < m.box_len / 2.0, "max_bond {} vs box/2 {}", f.max_bond, m.box_len / 2.0);
}

#[test]
fn lammps_reader_rejects_a_non_cubic_box() {
    // nerve_core::Melt carries a single box_len, so a non-cubic cell cannot be
    // represented and must be refused rather than silently squashed.
    let bad = FIXTURE.replace("0.0 10.0 ylo yhi", "0.0 12.0 ylo yhi");
    let err = from_lammps_str(&bad).expect_err("non-cubic box must be rejected");
    assert!(err.contains("cubic"), "unhelpful error: {err}");
}

#[test]
fn lammps_reader_rejects_a_chain_that_is_not_a_simple_path() {
    // A ring or a branch point is not a linear chain; unwrap_chain and every
    // chain feature assume a path with two ends.
    let ring = format!("{FIXTURE}7 1 4 1\n").replace("    6 bonds", "    7 bonds");
    let err = from_lammps_str(&ring).expect_err("a ring must be rejected");
    assert!(err.contains("path") || err.contains("degree"), "unhelpful error: {err}");
}

#[test]
fn a_loaded_melt_supports_every_instrument_this_crate_reports() {
    // End-to-end: the reader's output must work with the null model, writhe, and
    // the parity construction, or the archive route is not actually wired up.
    let m = from_lammps_str(FIXTURE).unwrap();
    let r = reflect(&m, 0);
    assert_eq!(null_model(&m, 2.0), null_model(&r, 2.0), "parity blindness should hold here too");
    let w = writhe_per_chain(&m);
    assert_eq!(w.len(), 2);
    // Straight 4-bead chains have zero writhe; this fixture is a degenerate
    // geometry, not a melt, and it is here to test parsing rather than physics.
    assert!(w.iter().all(|x| x.abs() < 1e-12), "expected zero writhe on collinear beads: {w:?}");
}

// ========================================================= Alexander witness
//
// Ground-truth determinants are this crate's equivalent of a Hopf-link test and
// are the most valuable tests in it. Classical values: unknot 1, trefoil 3,
// figure-eight 5, 5_1 5, 5_2 7, 6_1 9, 6_2 11, 6_3 13. The (2,q) torus family has
// determinant exactly q, which is why three of the checks below come from one
// parametrisation.

#[test]
fn alexander_determinant_matches_the_classical_table() {
    let cases: [(&str, Vec<Vec3>, u128); 5] = [
        ("unknot", circle(120), 1),
        ("trefoil 3_1 = (2,3)", torus_knot(2, 3, 240), 3),
        ("figure-eight 4_1", figure_eight(240), 5),
        ("5_1 = (2,5)", torus_knot(2, 5, 240), 5),
        ("7_1 = (2,7)", torus_knot(2, 7, 240), 7),
    ];
    for (name, k, want) in cases {
        let got = alexander_det_minus_one(&k);
        println!("  {name:22} |Delta(-1)| = {got:?}, expected {want}");
        assert_eq!(got, Some(want), "{name}: determinant is wrong");
    }
}

/// The accuracy ceiling, measured rather than cited.
///
/// `|Delta(-1)|` is a genuine invariant but not a complete one. 4_1 and 5_1 are
/// distinct knots with the same determinant, so any label built from this witness
/// cannot separate them — and KymoKnot's t = -1 with t = -2 pair still fails on
/// some prime knots at 8+ crossings. This is the limit stated in the doc.
#[test]
fn determinant_does_not_separate_four_one_from_five_one() {
    let a = alexander_det_minus_one(&figure_eight(240)).unwrap();
    let b = alexander_det_minus_one(&torus_knot(2, 5, 240)).unwrap();
    println!("\n  4_1 -> {a}, 5_1 -> {b}: distinct knots, identical determinant");
    assert_eq!(a, b, "if these ever differ the invariant got stronger, not weaker");
}

/// Determinant must not depend on the projection direction, which is the one
/// thing a hand-rolled diagram pipeline gets wrong silently.
#[test]
fn determinant_is_invariant_under_rotation_of_the_knot() {
    let k = torus_knot(2, 3, 240);
    for (ax, ang) in [(0usize, 0.7), (1, 1.9), (2, 2.6), (0, 3.3)] {
        let mut r = k.clone();
        for p in &mut r {
            let (s, c) = f64::sin_cos(ang);
            let (i, j) = ((ax + 1) % 3, (ax + 2) % 3);
            let (u, v) = (p[i], p[j]);
            p[i] = c * u - s * v;
            p[j] = s * u + c * v;
        }
        assert_eq!(alexander_det_minus_one(&r), Some(3), "axis {ax} angle {ang}");
    }
}

/// # The registered prediction, run
///
/// Registered before running, from round 4's structural argument: Alexander is
/// mirror-invariant, so a mirror pair gives **identical** Alexander values while
/// the per-bead pseudoscalar fingerprint **separates** them — the exact inverse of
/// the writhe result, where the fingerprint separated at 1.7e-2 to 1.5e-1 and
/// writhe flipped sign.
///
/// Note this is not tautological. Reflecting through the projection plane leaves
/// the 2D diagram identical and swaps over/under at **every** crossing, so the
/// Alexander matrix is rebuilt from completely different combinatorics. Equal
/// determinants are a theorem being confirmed, not an identity being restated.
#[test]
fn alexander_is_mirror_blind_where_the_pseudoscalar_fingerprint_is_not() {
    let k = torus_knot(2, 3, 240);
    let mirror: Vec<Vec3> = k.iter().map(|p| [p[0], p[1], -p[2]]).collect();

    let (a, b) = (alexander_det_minus_one(&k), alexander_det_minus_one(&mirror));
    println!("\n  Alexander: knot {a:?} vs mirror {b:?}");
    assert_eq!(a, b, "Alexander must be mirror-invariant; that is the whole point");

    // Same two configurations, chirality-sensitive descriptor. Box is large so the
    // knot does not interact with its own periodic images.
    let as_melt = |v: &Vec<Vec3>| Melt::new(vec![Chain::new(v.clone())], 200.0);
    let fa = per_bead_signed_volumes(&as_melt(&k), 12);
    let fb = per_bead_signed_volumes(&as_melt(&mirror), 12);
    let sep = cosine_distance(&fa, &fb);
    println!("  per-bead signed volumes separate the same pair at {sep:.4e}");
    assert!(sep > 1e-3, "fingerprint failed to separate the mirror pair: {sep:.3e}");
}

// ------------------------------------------------- closure control, same commit

#[test]
fn stochastic_closure_gives_a_modal_label_with_its_probability() {
    // A knot that is genuinely tied stays tied under almost any closure, so the
    // modal probability should be high and the label trustworthy. Open the trefoil
    // by dropping one bead.
    let mut open = torus_knot(2, 3, 240);
    open.pop();
    let l = knot_label(&open, 60, 11);
    println!(
        "\n  open trefoil: modal {} p={:.3}, {} distinct values over {} resolved closures",
        l.modal, l.prob, l.distinct, l.resolved
    );
    assert!(l.resolved > 40, "too many closures failed to resolve: {}", l.resolved);
    assert_eq!(l.modal, 3, "modal label is not the trefoil");
    assert!(l.prob > 0.8, "a tied knot should survive most closures: p={:.3}", l.prob);
}

#[test]
fn minimally_interfering_closure_agrees_with_the_stochastic_mode_on_a_tied_knot() {
    let mut open = torus_knot(2, 3, 240);
    open.pop();
    let det = alexander_det_minus_one(&close_minimally_interfering(&open));
    println!("\n  MIC-proxy closure on the open trefoil: {det:?}");
    assert_eq!(det, Some(3), "deterministic comparator disagrees with the mode");
}

/// The control that makes the label honest, and a registered hypothesis.
///
/// An unknotted open chain from a real melt has no knot to protect it, so the
/// closure should matter far more than it does for a tied knot. Registered
/// prediction: a melt chain shows a strictly lower modal probability than the
/// tied trefoil's, i.e. closure ambiguity is visible exactly where the literature
/// says it is.
#[test]
fn a_melt_chain_carries_more_closure_ambiguity_than_a_tied_knot() {
    let m = long_melt(1);
    let mut worst = 1.0f64;
    for i in 0..m.chains.len() {
        let l = knot_label(&nerve_topo::unwrap_chain(&m, &m.chains[i]), 40, 7 + i as u64);
        worst = worst.min(l.prob);
        if i < 3 {
            println!(
                "  melt chain {i}: modal {} p={:.3}, {} distinct over {} resolved",
                l.modal, l.prob, l.distinct, l.resolved
            );
        }
    }
    let mut open = torus_knot(2, 3, 240);
    open.pop();
    let tied = knot_label(&open, 40, 11).prob;
    println!("  worst melt-chain modal p {worst:.3} vs tied trefoil {tied:.3}");
    assert!(worst < tied, "melt chains showed no more closure ambiguity than a tied knot");
}

/// # The informativeness check, not just the stability check
///
/// Writhe was eliminated as a label because it is stable *for the same reason it
/// is blind*: translation-invariant, therefore constant on every rigid-translation
/// family, informativeness z = 0.174 and identically 0.0 on all nine ladder rungs.
/// A witness that is stable and blind is the trap, so `|Delta(-1)|` has to clear
/// the informativeness bar too, not only the stability bar.
///
/// **Methodological point that has to be said first:** a rigid-translation ladder
/// cannot test a topological invariant at all. Translation is an isometry, so
/// every invariant is exactly constant on it, and such a ladder returns
/// informativeness zero for an informative witness and a blind one alike. The
/// probe has to cross strands. Writhe's verdict stands regardless, because writhe
/// is also constant on non-isometric paths where topology does change.
///
/// **Registered before measuring:** `|Delta(-1)|` is constant on the no-crossing
/// stretch, jumps by a nonzero integer exactly where the lifted arc touches the
/// strand it passed under, and the trefoil becomes the unknot — 3 to 1. That is
/// `|Lk|`'s behaviour (jump 1.000000 only at closest approach 0.025210), not
/// writhe's.
#[test]
fn alexander_jumps_only_at_a_genuine_strand_passage() {
    let n = 240;
    let mut rows = Vec::new();
    println!("\n     t      clearance   |Delta(-1)|");
    for k in 0..=24 {
        let t = k as f64 * 0.25;
        let curve = lifted_trefoil(n, t);
        let clear = lifted_window_clearance(&curve, n);
        let det = alexander_det_minus_one(&curve);
        rows.push((t, clear, det));
        if k % 2 == 0 {
            println!("  {t:5.2}   {clear:9.5}   {det:?}");
        }
    }

    let dets: Vec<u128> = rows.iter().filter_map(|r| r.2).collect();
    assert_eq!(dets.len(), rows.len(), "some diagrams failed to resolve");

    // 1. It must actually change - otherwise it is as blind as writhe was.
    let first = dets[0];
    let last = *dets.last().unwrap();
    assert_eq!(first, 3, "path does not start at the trefoil");
    assert_ne!(last, first, "determinant never changed: stable and blind, the writhe trap");

    // 2. Every change must sit at a local minimum of clearance, and the smallest
    //    clearance on the path must be small. A jump anywhere else would be an
    //    estimator artifact rather than a topology change.
    let mut jumps = Vec::new();
    for w in rows.windows(2) {
        if w[0].2 != w[1].2 {
            jumps.push((w[1].0, w[0].1.min(w[1].1), w[0].2.unwrap(), w[1].2.unwrap()));
        }
    }
    println!("  jumps: {jumps:?}");
    assert!(!jumps.is_empty());
    let worst_clearance_at_a_jump = jumps.iter().map(|j| j.1).fold(0.0f64, f64::max);
    let typical = rows.iter().map(|r| r.1).fold(0.0f64, f64::max);
    println!(
        "  clearance at jumps <= {worst_clearance_at_a_jump:.5}, max clearance on path {typical:.5}"
    );
    assert!(
        worst_clearance_at_a_jump < 0.25 * typical,
        "a jump happened while the strands were still far apart: {worst_clearance_at_a_jump:.5} \
         vs max {typical:.5} - that is an estimator artifact, not a strand passage"
    );

    // 3. And it ends at the unknot, which is what flipping one trefoil crossing does.
    assert_eq!(last, 1, "flipping one crossing of a trefoil should give the unknot, got {last}");
}

// -------------------------------------------------------------------- helpers

fn centroid_of(m: &Melt, i: usize) -> [f64; 3] {
    let b = &m.chains[i].beads;
    let n = b.len() as f64;
    let mut c = [0.0; 3];
    for p in b {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    c
}
