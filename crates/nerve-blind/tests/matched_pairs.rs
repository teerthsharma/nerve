//! The instrument itself: topologically-matched pairs.
//!
//! Each tier attacks a different assumption of a cutoff MLIP, and each test
//! name states the regime in which the claim holds. Run with `-- --nocapture`
//! to see the measured residuals.

use nerve_blind::*;
use nerve_core::Melt;
use proptest::prelude::*;

const N: usize = 40;
const R: f64 = 3.0;
const BOX: f64 = 40.0;

/// Radii from a physical cutoff up to `box_len / 2`, which is the hard ceiling:
/// above it the minimum image is not the nearest image, so a wider cutoff sums
/// over periodic copies of the same bead rather than over more neighbours.
///
/// The largest entry answers "with enough layers the receptive field covers the
/// whole box" without exceeding that ceiling — the configurations here span
/// about 12 units, so at radius 20 every bead already sees every other bead. The
/// cutoff-free version of the same claim is
/// `parity_pair_distance_multiset_is_bit_identical` in `null_model.rs`.
const RADII: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 20.0];

fn linked() -> Melt {
    two_rings(N, R, R, BOX)
}

fn lk(m: &Melt) -> f64 {
    linking_number(&m.chains[0], &m.chains[1])
}

/// Bit-exact multiset of coordinates, so "identical point set" is a claim about
/// f64 bit patterns rather than about a tolerance.
fn point_bits(m: &Melt) -> Vec<[u64; 3]> {
    let mut v: Vec<[u64; 3]> = m
        .chains
        .iter()
        .flat_map(|c| c.beads.iter().map(|b| [b[0].to_bits(), b[1].to_bits(), b[2].to_bits()]))
        .collect();
    v.sort_unstable();
    v
}

// ============================================================== tier: parity
//
// Assumption attacked: the model is O(3)-invariant. Every production MLIP is,
// because energy is built from invariants of the distance matrix.
//
// Regime: none. No bound on cutoff radius, message-passing depth, body order,
// or system size is required, because reflection preserves the distance matrix
// exactly and the descriptor is a function of the distance matrix.

#[test]
fn parity_pair_residual_is_exactly_zero_at_every_radius_including_whole_system() {
    let a = linked();
    let b = mirror(&a);
    for &radius in &RADII {
        for &bond_aware in &[false, true] {
            let d = Descriptor { radius, bond_aware };
            let res = residual(&d.signature(&a), &d.signature(&b));
            println!("parity: radius {radius:>6} bond_aware {bond_aware:>5} residual {res:e}");
            assert_eq!(res, 0.0, "radius {radius}, bond_aware {bond_aware}");
        }
    }
}

#[test]
fn parity_pair_linking_number_differs_by_two() {
    let a = linked();
    let b = mirror(&a);
    let (la, lb) = (lk(&a), lk(&b));
    println!("parity: Lk {la} vs {lb}, delta {}", (la - lb).abs());
    assert!((la - lb).abs() > 1.9, "Lk {la} vs {lb}");
}

// ======================================================== tier: connectivity
//
// Assumption attacked: the model reads bead positions and species only, with no
// bond graph. This is what an MLIP is given.
//
// Regime: none, for the same reason as above -- the two configurations are the
// same point set, so any function of positions and species agrees bit for bit
// at any radius and any depth.

/// Cross-splice sites, coarsely swept. Returns every site whose reconnection
/// changes |Lk|, with the longest bond it costs.
fn splice_sites_that_change_linking() -> Vec<(usize, usize, f64, f64)> {
    let m = linked();
    let l0 = lk(&m).abs();
    let mut out = Vec::new();
    for i in (2..N - 2).step_by(2) {
        for j in (2..N - 2).step_by(2) {
            let r = reconnect(&m, 0, i, 1, j);
            if r.chains.iter().any(|c| c.len() < 4) {
                continue;
            }
            let l1 = lk(&r).abs();
            if (l1 - l0).abs() > 0.5 {
                out.push((i, j, l1, max_bond_closed(&r)));
            }
        }
    }
    out
}

#[test]
fn connectivity_pair_exists_changing_linking_at_fixed_point_set() {
    let sites = splice_sites_that_change_linking();
    println!("connectivity: {} of {} swept splice sites change |Lk|", sites.len(), (19 * 19));
    if let Some(best) = sites.iter().min_by(|a, b| a.3.total_cmp(&b.3)) {
        println!(
            "connectivity: cheapest site (i={}, j={}) |Lk| {} -> {}, longest bond {:.4} \
             (original bond {:.4})",
            best.0,
            best.1,
            lk(&linked()).abs(),
            best.2,
            best.3,
            max_bond_closed(&linked())
        );
    }
    assert!(!sites.is_empty(), "no splice site changed the linking number");
}

#[test]
fn connectivity_pair_point_set_is_bit_identical() {
    let m = linked();
    let sites = splice_sites_that_change_linking();
    let (i, j, ..) = sites[0];
    let r = reconnect(&m, 0, i, 1, j);
    assert_eq!(point_bits(&m), point_bits(&r), "reconnection moved a bead");
    assert_eq!(m.n_beads(), r.n_beads());
    assert_eq!(m.chains.len(), r.chains.len(), "chain count must be preserved");
}

#[test]
fn connectivity_pair_residual_is_exactly_zero_at_every_radius_for_position_only_model() {
    let m = linked();
    let sites = splice_sites_that_change_linking();
    let (i, j, ..) = sites[0];
    let r = reconnect(&m, 0, i, 1, j);
    for &radius in &RADII {
        let d = Descriptor { radius, bond_aware: false };
        let res = residual(&d.signature(&m), &d.signature(&r));
        println!("connectivity: radius {radius:>6} residual {res:e}");
        assert_eq!(res, 0.0, "radius {radius}");
    }
}

/// The honest limit of this tier: a model that *is* handed the bond graph sees
/// the splice. It must, or the tier would be claiming too much.
#[test]
fn connectivity_pair_is_visible_to_a_bond_aware_model() {
    let m = linked();
    let sites = splice_sites_that_change_linking();
    let (i, j, ..) = sites[0];
    let r = reconnect(&m, 0, i, 1, j);
    let d = Descriptor { radius: 5.0, bond_aware: true };
    let res = residual(&d.signature(&m), &d.signature(&r));
    println!("connectivity: bond-aware residual {res:e} (expected non-zero)");
    assert!(res > 0.0, "a bond-aware model should see the splice");
}

// ============================================================ tier: locality
//
// Assumption attacked: locality alone, with no help from parity or from the
// missing bond graph. The pair is two rings displaced by 2*delta across the
// offset == 2r transition where the linking number jumps.
//
// Regime: bounded residual, not zero residual. This is the weak tier and the
// tests say so.

/// The exact form of this tier, and the strongest single result in the crate.
///
/// A Hopf link and an unlink built from congruent rings whose closest inter-ring
/// approach is `gap` in *both* configurations. Below that cutoff every bead's
/// environment consists only of its own ring, and the two rings are congruent
/// circles in both melts, so the environment multisets are bit-identical while
/// the linking number differs by one.
///
/// Regime, stated explicitly: cutoff strictly below the measured inter-ring gap,
/// which for rings of radius `r` in this family is `r`. No bound on depth is
/// needed for the equality of the environment multiset itself, but a
/// message-passing model whose effective receptive field exceeds `gap` is
/// outside the claim.
#[test]
fn locality_pair_residual_is_exactly_zero_for_cutoff_strictly_below_the_measured_gap() {
    let (a, b) = (two_rings(N, R, R, BOX), two_rings(N, R, 3.0 * R, BOX));
    let gap = min_interchain_dist(&a).min(min_interchain_dist(&b));
    let (la, lb) = (lk(&a).abs(), lk(&b).abs());
    println!("locality-exact: measured inter-ring gap {gap:.12}, |Lk| {la} vs {lb}");
    assert!((gap - R).abs() < 1e-9, "gap {gap} should equal the ring radius {R}");
    assert!((la - lb).abs() > 0.5, "|Lk| {la} vs {lb}");

    for &radius in &[0.5, 1.0, 2.0, 2.9, R - 1e-9] {
        for &bond_aware in &[false, true] {
            let d = Descriptor { radius, bond_aware };
            let res = residual(&d.signature(&a), &d.signature(&b));
            println!("locality-exact: radius {radius:>10} bond_aware {bond_aware:>5} residual {res:e}");
            assert_eq!(res, 0.0, "radius {radius} below gap {gap}, bond_aware {bond_aware}");
        }
    }
}

fn locality_pair(delta: f64) -> (Melt, Melt) {
    (two_rings(N, R, 2.0 * R - delta, BOX), two_rings(N, R, 2.0 * R + delta, BOX))
}

#[test]
fn locality_pair_linking_number_differs_by_one_across_the_transition() {
    let (a, b) = locality_pair(0.01);
    let (la, lb) = (lk(&a).abs(), lk(&b).abs());
    println!("locality: |Lk| {la} vs {lb}");
    assert!((la - lb).abs() > 0.5, "|Lk| {la} vs {lb}");
}

/// Largest distance any single bead moves between two melts of identical shape.
fn max_bead_displacement(a: &Melt, b: &Melt) -> f64 {
    a.chains
        .iter()
        .zip(b.chains.iter())
        .flat_map(|(ca, cb)| {
            ca.beads.iter().zip(cb.beads.iter()).map(|(p, q)| {
                let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
        })
        .fold(0.0f64, f64::max)
}

proptest! {
    /// The geometric side of the trade-off, and it is exact: one whole ring is
    /// displaced rigidly by 2*delta and nothing else moves, so the linking
    /// number can be flipped by an arbitrarily small perturbation of the
    /// configuration.
    #[test]
    fn locality_transition_flips_linking_under_a_displacement_of_exactly_twice_delta(
        delta in 1e-3f64..1.0,
    ) {
        let (a, b) = locality_pair(delta);
        let disp = max_bead_displacement(&a, &b);
        prop_assert!((disp - 2.0 * delta).abs() < 1e-9, "displacement {disp} != {}", 2.0 * delta);
        prop_assert!((lk(&a).abs() - lk(&b).abs()).abs() > 0.5);
    }
}

/// Negative result, reported rather than massaged.
///
/// The obvious way to build a matched pair is to sit either side of the
/// `offset == 2r` transition, where the linking number flips under a rigid
/// displacement of `2*delta`. It does not work. Every interbead distance moves by
/// at most `2*delta`, but near contact some cross-ring pairs cross the hard
/// cutoff, the neighbour count changes, and the residual is infinite — a
/// hard-cutoff descriptor separates the pair no matter how small `delta` gets.
///
/// So driving the two strands together is the wrong route to exactness, and the
/// route that works is the opposite one: hold the strands apart at a gap wider
/// than the cutoff, which is what
/// `locality_pair_residual_is_exactly_zero_for_cutoff_strictly_below_the_measured_gap`
/// does.
#[test]
fn locality_transition_family_is_not_a_matched_pair_under_a_hard_cutoff() {
    let d = Descriptor { radius: 2.0, bond_aware: true };
    let mut any_finite = false;
    for &delta in &[1.0, 0.3, 0.1, 0.03, 0.01, 0.003] {
        let (a, b) = locality_pair(delta);
        let res = residual(&d.signature(&a), &d.signature(&b));
        let sep = min_interchain_dist(&a).min(min_interchain_dist(&b));
        println!(
            "locality-transition: delta {delta:>6} bead displacement {:.6} residual {res:>10.3e} \
             min interchain separation {sep:.4} |dLk| {:.6}",
            max_bead_displacement(&a, &b),
            (lk(&a).abs() - lk(&b).abs()).abs()
        );
        any_finite |= res.is_finite();
    }
    assert!(
        !any_finite,
        "a finite residual here would make this family a usable matched pair; \
         re-derive the bound rather than deleting this assertion"
    );
}

// ============================================================== determinism

#[test]
fn construction_and_descriptor_are_bitwise_deterministic() {
    let d = Descriptor { radius: 2.0, bond_aware: true };
    let a = d.signature(&perturb(&linked(), 1e-3, 7));
    let b = d.signature(&perturb(&linked(), 1e-3, 7));
    assert_eq!(a, b, "repeated construction differed");
    assert_eq!(lk(&linked()).to_bits(), lk(&linked()).to_bits());
    let c = d.signature(&perturb(&linked(), 1e-3, 8));
    assert_ne!(a, c, "different seeds gave identical output");
}
