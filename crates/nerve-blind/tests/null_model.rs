//! Does the matched pair survive the non-topological null model?
//!
//! A pair matched only on local environments proves nothing. Cheap single-chain
//! size features separate melts with no topological invariant anywhere in sight,
//! so a pair that differs in radius of gyration or in its mean-square internal
//! distance curve is a demonstration that the descriptor is bad at chain size —
//! which nobody disputes — not that it is blind to topology.
//!
//! A linking-number-style invariant is necessary only if the melts it separates
//! match on **all** of: local environments, `rg2`, MSID, end-to-end distance and
//! contour length, and differ **only** in whether the chains thread each other.
//! That is what these tests check, per tier, and where a tier fails the gap is
//! measured rather than argued away.

use nerve_blind::*;
use nerve_core::Melt;

const N: usize = 40;
const R: f64 = 3.0;
const BOX: f64 = 40.0;

fn linked() -> Melt {
    two_rings(N, R, R, BOX)
}
fn lk(m: &Melt) -> f64 {
    linking_number(&m.chains[0], &m.chains[1])
}
fn gap_of(a: &Melt, b: &Melt) -> FeatureGap {
    feature_gap(&chain_features(a), &chain_features(b))
}

/// Both matched pairs that claim an exact match. Each differs only in
/// interpenetration: the parity pair by an improper isometry, the locality pair
/// by a rigid translation of one whole ring.
fn parity_pair() -> (Melt, Melt) {
    let a = linked();
    (a.clone(), mirror(&a))
}
fn locality_pair() -> (Melt, Melt) {
    (two_rings(N, R, R, BOX), two_rings(N, R, 3.0 * R, BOX))
}

// ------------------------------------------------------------ validity guards

/// Neither side has a measurement outside this regime, so it is asserted before
/// anything else is believed.
#[test]
fn every_melt_used_satisfies_the_bond_length_validity_guard() {
    let (p, q) = parity_pair();
    let (a, b) = locality_pair();
    for (name, m) in [("parity.a", &p), ("parity.b", &q), ("locality.a", &a), ("locality.b", &b)] {
        let f = chain_features(m);
        println!(
            "{name}: max_bond {:.6} vs box_len/2 {:.6}, bead_density {:.6}",
            f.max_bond,
            m.box_len / 2.0,
            f.bead_density
        );
        assert!(
            f.max_bond < m.box_len / 2.0,
            "{name}: max_bond {} exceeds box_len/2 {}; unwrapping cannot recover the walk",
            f.max_bond,
            m.box_len / 2.0
        );
    }
}

// ---------------------------------------------------- tier: parity, exact

/// Reflection preserves every intra-chain distance, and all four null features
/// are functions of intra-chain distances alone. So the null model has literally
/// nothing to separate: the gap is exactly zero, not small.
#[test]
fn parity_pair_matches_every_non_topological_feature_exactly() {
    let (a, b) = parity_pair();
    let g = gap_of(&a, &b);
    println!("parity null gap: {g:?}");
    println!("parity |dLk|: {}", (lk(&a) - lk(&b)).abs());
    assert_eq!(g.worst(), 0.0, "null model separates the parity pair: {g:?}");
    assert!((lk(&a) - lk(&b)).abs() > 1.9);
}

/// The cutoff-free form of the parity claim, which needs no radius sweep and so
/// never touches the `r_cut <= box_len / 2` bound.
#[test]
fn parity_pair_distance_multiset_is_bit_identical() {
    let (a, b) = parity_pair();
    let (da, db) = (distance_multiset(&a), distance_multiset(&b));
    assert_eq!(da.len(), db.len());
    let worst = da
        .iter()
        .zip(db.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    println!("parity distance-multiset worst disagreement over {} pairs: {worst:e}", da.len());
    assert_eq!(worst, 0.0, "distance multisets differ");
}

// -------------------------------------------------- tier: locality, exact

/// The locality pair is the same two rings in both melts, one rigidly
/// translated. Every null feature is translation invariant per chain, so again
/// the gap is exactly zero while the rings go from threaded to unthreaded —
/// matched `rg2`, matched MSID, differing only in interpenetration.
/// Every feature but `rg2` is exactly zero. `rg2` comes out at 1.8e-15 rather
/// than a hard zero because the two melts place ring B at different absolute
/// coordinates, so the centre-of-mass sum rounds differently — the quantity is
/// translation invariant in exact arithmetic but not bitwise in f64. The gap is
/// reported relative to `rg2` itself so the number can be read against the
/// baseline crate's 7.6x `rg2` separation on its own melt pair.
#[test]
fn locality_pair_matches_every_non_topological_feature_to_floating_point() {
    let (a, b) = locality_pair();
    let g = gap_of(&a, &b);
    let fa = chain_features(&a);
    println!("locality null gap: {g:?}");
    println!(
        "locality rg2 {:.12}, absolute rg2 gap {:e}, relative rg2 gap {:e}",
        fa.rg2,
        g.rg2,
        g.rg2 / fa.rg2
    );
    println!(
        "locality |Lk| {} vs {}, inter-ring gap {:.12}",
        lk(&a).abs(),
        lk(&b).abs(),
        min_interchain_dist(&a).min(min_interchain_dist(&b))
    );
    for (name, v) in [
        ("end_to_end2", g.end_to_end2),
        ("contour_len", g.contour_len),
        ("bead_density", g.bead_density),
        ("max_bond", g.max_bond),
        ("msid", g.msid),
    ] {
        assert_eq!(v, 0.0, "{name} must be a hard zero, got {v}");
    }
    assert!(
        g.rg2 / fa.rg2 < 1e-14,
        "rg2 gap {:e} is larger than f64 rounding on the centre-of-mass sum",
        g.rg2
    );
    assert!((lk(&a).abs() - lk(&b).abs()).abs() > 0.5);
}

// --------------------------------------- tier: connectivity, gap measured

/// The cross-splice at `i == j` is the only splice that preserves chain length,
/// so it is the only one with any chance against the null model. Whether it also
/// preserves `rg2` and MSID is a question for measurement, and the answer is
/// printed either way.
#[test]
fn connectivity_pair_non_topological_gap_is_measured_at_equal_chain_lengths() {
    let m = linked();
    let l0 = lk(&m).abs();
    let mut best: Option<(usize, FeatureGap, f64, f64)> = None;
    for i in (2..N - 2).step_by(2) {
        let r = reconnect(&m, 0, i, 1, i);
        assert!(
            r.chains.iter().all(|c| c.len() == N),
            "splice at i == j must preserve chain length"
        );
        let l1 = lk(&r).abs();
        if (l1 - l0).abs() < 0.5 {
            continue;
        }
        let g = gap_of(&m, &r);
        let cand = (i, g, l1, max_bond_closed(&r));
        if best.as_ref().is_none_or(|b| g.worst() < b.1.worst()) {
            best = Some(cand);
        }
    }
    let (i, g, l1, bond) = best.expect("no equal-length splice changed the linking number");
    let fa = chain_features(&m);
    println!("connectivity: best equal-length splice i = j = {i}, |Lk| {l0} -> {l1}");
    println!("connectivity: null gap {g:?}");
    println!(
        "connectivity: rg2 {:.6} with gap {:.6} (relative {:.4}), msid gap {:.6}, longest bond {bond:.4}",
        fa.rg2,
        g.rg2,
        g.rg2 / fa.rg2,
        g.msid
    );
    // NEGATIVE RESULT for this tier, reported rather than argued away. The splice
    // keeps the point set and the chain length, but it rearranges which beads
    // belong to which chain, so the single-chain size features move by far more
    // than rounding. The null model separates this pair without any topology, so
    // the tier does not meet the standard: it is bounded to models that see
    // positions only AND it fails the cheap-feature control.
    assert!(
        g.worst() > 1e-6,
        "an exact null match here would upgrade this tier — re-derive before trusting it"
    );
    assert!(
        g.rg2 / fa.rg2 > 1e-3,
        "rg2 gap {:e} is claimed to be substantial, not rounding",
        g.rg2
    );
}

// ------------------------------------------------- same-seed / different-seed

/// The control Cameron proved is mandatory. Two independently sampled
/// configurations differ by finite-size noise whether or not their topology
/// differs, so a raw distance means nothing without a noise floor.
///
/// Here the noise floor is a same-melt, different-seed perturbation pair, and the
/// signal is the true matched pair. Signal over noise is exactly zero for both
/// exact tiers: the numerator is a hard zero while the denominator is not.
#[test]
fn descriptor_blindness_is_below_the_different_seed_noise_floor() {
    let d = Descriptor { radius: 2.0, bond_aware: true };
    let eps = 1e-3;
    for (name, (a, b)) in [("parity", parity_pair()), ("locality", locality_pair())] {
        let signal = residual(&d.signature(&a), &d.signature(&b));
        let noise = residual(
            &d.signature(&perturb(&a, eps, 31)),
            &d.signature(&perturb(&a, eps, 32)),
        );
        println!("{name}: signal {signal:e}, different-seed noise floor {noise:e}, S/N {}", signal / noise);
        assert!(noise > 0.0, "{name}: noise floor is zero, the ratio would be undefined");
        assert_eq!(signal, 0.0, "{name}: signal must be a hard zero");
        assert_eq!(signal / noise, 0.0, "{name}: S/N must be exactly zero");
    }
}
