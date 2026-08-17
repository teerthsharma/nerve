//! Mechanism (a): body-order truncation.
//!
//! The Gauss linking integrand is a **four-point** object — two segments, four
//! beads. The question is what body order is needed, and whether raising it buys
//! any reach.
//!
//! Prior art is Pozdnyakov et al., PRL 125, 166001 (2020), arXiv:2001.11696,
//! which proves 3-body descriptors do not uniquely specify an environment and
//! that the 4-body bispectrum does not either. Their footnote 24 records that
//! correlations above ν=3 are chirality sensitive when defined over proper
//! rotations only. Nothing here rediscovers that. What is measured here is the
//! *cutoff reach* of each order on a topological pair, which their paper does not
//! address because it contains no chains, knots, or linking number.

use nerve_order::*;

const N: usize = 48;
const R: f64 = 3.0;
const BOX: f64 = 40.0;
const AMP: f64 = 0.3;
const WAVES: f64 = 3.0;

fn linked() -> nerve_core::Melt {
    wavy_rings(N, R, R, BOX, AMP, WAVES)
}
fn unlinked() -> nerve_core::Melt {
    wavy_rings(N, R, 3.0 * R, BOX, AMP, WAVES)
}

/// Baseline: below the gap neither order sees anything, because no bead has a
/// neighbour on the other ring.
#[test]
fn two_body_and_three_body_correlations_are_identical_below_the_gap() {
    let (a, b) = (linked(), unlinked());
    for order in [2usize, 3] {
        let res = correlation_residual(
            &n_body_correlation(&a, order, 2.0),
            &n_body_correlation(&b, order, 2.0),
        );
        println!("order {order} correlation residual at cutoff 2.0: {res:e}");
        assert_eq!(res, 0.0, "order {order} separates the pair below the gap");
    }
}

/// The measurement, and a hypothesis that can come out either way: **raising the
/// body order from 2 to 3 buys no extra cutoff reach on this pair.** Both orders
/// are functions of distances within the cutoff, so both should turn on at the
/// same threshold — the gap — and neither earlier.
///
/// If 3-body separates the pair at a cutoff where 2-body does not, this test
/// fails and the hypothesis is wrong, which would be the more interesting
/// outcome. Either way the sweep is printed.
#[test]
fn three_body_buys_no_extra_cutoff_reach_over_two_body() {
    let (a, b) = (linked(), unlinked());
    let gap = min_interchain_dist(&a).min(min_interchain_dist(&b));
    println!("measured inter-ring gap: {gap:.6}");
    let mut rows = Vec::new();
    for &r_cut in &[1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0] {
        let two = correlation_residual(
            &n_body_correlation(&a, 2, r_cut),
            &n_body_correlation(&b, 2, r_cut),
        );
        let three = correlation_residual(
            &n_body_correlation(&a, 3, r_cut),
            &n_body_correlation(&b, 3, r_cut),
        );
        println!(
            "r_cut {r_cut:>5}: 2-body residual {two:>10.3e}  3-body residual {three:>10.3e}  \
             (gap {gap:.4})"
        );
        rows.push((r_cut, two, three));
    }
    for (r_cut, two, three) in rows {
        let two_sees = two != 0.0;
        let three_sees = three != 0.0;
        assert_eq!(
            two_sees, three_sees,
            "at r_cut {r_cut} the two orders disagree on whether the pair is separable: \
             2-body {two:e}, 3-body {three:e}"
        );
    }
}

/// Largest elementwise disagreement between two globally pooled multisets.
fn pooled_residual(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

fn per_bead_chiral_residual(a: &nerve_core::Melt, b: &nerve_core::Melt, r_cut: f64) -> f64 {
    let d = Local { r_cut, chiral: true };
    fingerprint_residual(&d.fingerprints(a), &d.fingerprints(b))
}

/// **Records a false premise of mine.** I expected the globally pooled
/// signed-volume multiset — a parity-odd 4-body invariant — to separate a mirror
/// pair. It does not, and the first version of this test passed *vacuously* at a
/// residual of 1.11e-16, which is f64 noise rather than a separation.
///
/// The cause is the pooling, not the invariant. Reflection negates every signed
/// volume, and the volume distribution over a whole closed ring is very nearly
/// symmetric under negation, so the *sorted* pooled multiset of `{-v}` coincides
/// with that of `{v}`. Pooling symmetrises away the parity signal that the
/// individual features carry.
///
/// This is asserted in the direction that is actually true, with a threshold well
/// below the 1.16e-1 that the per-bead instrument achieves on the same pair.
#[test]
fn global_signed_volume_pooling_destroys_the_parity_signal() {
    let m = linked();
    let mir = mirror(&m);
    let (va, vb) = (signed_volume_multiset(&m, 2.0), signed_volume_multiset(&mir, 2.0));
    assert_eq!(va.len(), vb.len(), "mirror must not change the number of triples");
    let pooled = pooled_residual(&va, &vb);
    println!(
        "pooled 4-body residual against mirror: {pooled:e} over {} triples",
        va.len()
    );
    assert!(
        pooled < 1e-12,
        "pooled signed volumes separated the mirror pair by {pooled:e}; the premise that \
         pooling destroys parity is wrong and the finding must be rewritten"
    );
}

/// The same invariant, grouped per bead instead of pooled, does retain parity.
/// So body order 4 is where chirality *can* first be seen — but only if the
/// features are not pooled first, which is exactly what an additive energy does.
///
/// **Necessity, not sufficiency.** Order 4 is not a licence to claim linking number
/// becomes representable there: Pozdnyakov et al. show the 4-body bispectrum does
/// not uniquely specify an environment either. An earlier revision described that
/// bispectrum as "achiral" and attributed it to Fig. 2e — retracted, since *achiral*
/// appears nowhere in the paper and Fig. 2e concerns a pair sharing identical
/// **chiral** features. For inversion-invariance of the power spectrum the source is
/// Bartók, Kondor & Csányi, *Phys. Rev. B* **87**, 184115 (2013), arXiv:1209.3140,
/// Eq. (17).
#[test]
fn per_bead_signed_volumes_retain_the_parity_signal_that_pooling_loses() {
    let m = linked();
    for order in [2usize, 3] {
        let res = correlation_residual(
            &n_body_correlation(&m, order, 2.0),
            &n_body_correlation(&mirror(&m), order, 2.0),
        );
        println!("order {order} residual against mirror: {res:e} (definitional zero)");
        assert_eq!(res, 0.0, "order {order} distance correlations should be parity even");
    }
    let per_bead = per_bead_chiral_residual(&m, &mirror(&m), 2.0);
    println!("per-bead 4-body residual against mirror: {per_bead:e}");
    assert!(
        per_bead > 1e-3,
        "per-bead signed volumes failed to detect the mirror at {per_bead:e}; \
         the chirality control for the whole crate rests on this"
    );
}

/// The sharp limit of mechanism (a), and the decisive ranking result.
///
/// **No body order separates this pair below the gap** — not 2-body, not 3-body,
/// and not the parity-odd 4-body invariant in its strong per-bead form, the one
/// that does detect a mirror image at 1.16e-1. Body-order truncation is therefore
/// *not* the binding constraint here. Locality plus additivity is.
#[test]
fn no_body_order_including_per_bead_parity_odd_four_body_separates_the_pair_below_the_gap() {
    let (a, b) = (linked(), unlinked());
    for order in [2usize, 3] {
        let res = correlation_residual(
            &n_body_correlation(&a, order, 2.0),
            &n_body_correlation(&b, order, 2.0),
        );
        assert_eq!(res, 0.0, "order {order} separates the pair below the gap");
    }
    let per_bead = per_bead_chiral_residual(&a, &b, 2.0);
    println!("per-bead parity-odd 4-body residual below the gap: {per_bead:e}");
    assert_eq!(per_bead, 0.0, "per-bead parity-odd 4-body separates the pair below the gap");
}
