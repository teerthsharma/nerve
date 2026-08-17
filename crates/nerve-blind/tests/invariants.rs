//! Invariant tests for the two instruments: the topological discriminator
//! (linking number, writhe) and the local descriptor.
//!
//! Both instruments must be trustworthy before any matched pair means anything.
//! A descriptor that is blind to *everything* would "prove" the premise for
//! free, so the negative controls here carry as much weight as the positives.

use nerve_blind::*;
use nerve_core::Melt;
use proptest::prelude::*;

// Shrunk counterexamples land in `tests/invariants.proptest-regressions` and are
// re-run ahead of novel cases. The seeds currently in that file are the ones
// proptest shrunk during the RED phase against the unimplemented stubs; they now
// pass, and they stay checked at no cost.

const N: usize = 40;
const R: f64 = 3.0;
const BOX: f64 = 40.0;

/// Linked: `0 < offset < 2r`. Unlinked: `offset > 2r`.
fn linked() -> Melt {
    two_rings(N, R, R, BOX)
}
fn unlinked() -> Melt {
    two_rings(N, R, 3.0 * R, BOX)
}

fn lk(m: &Melt) -> f64 {
    linking_number(&m.chains[0], &m.chains[1])
}

// ------------------------------------------- ground truth for the topology

#[test]
fn linking_number_of_hopf_link_is_plus_or_minus_one() {
    let v = lk(&linked());
    assert!((v.abs() - 1.0).abs() < 1e-9, "expected |Lk| = 1, got {v}");
}

/// Negative control. An instrument that reported a link everywhere would make
/// every matched pair below meaningless.
#[test]
fn linking_number_of_unlinked_rings_is_zero() {
    let v = lk(&unlinked());
    assert!(v.abs() < 1e-9, "expected Lk = 0, got {v}");
}

// ------------------------------------------------- invariants of the topology

proptest! {
    /// The Klenin-Langowski solid angle is exact for straight segments, so a
    /// polygon's linking number is an integer to floating point, not a value
    /// converging to one. A midpoint-rule Gauss integral fails this near
    /// contact, which is exactly where the matched pairs live.
    #[test]
    fn linking_number_is_integer_valued(
        offset in 0.2f64..8.0,
        r in 1.0f64..4.0,
        n in 12usize..60,
    ) {
        prop_assume!((offset - 2.0 * r).abs() > 0.05);
        let v = lk(&two_rings(n, r, offset, BOX));
        prop_assert!((v - v.round()).abs() < 1e-6, "Lk = {v} is not an integer");
    }

    #[test]
    fn linking_number_invariant_under_proper_isometry(
        ang in 0.0f64..std::f64::consts::TAU,
        ax in prop::array::uniform3(-1.0f64..1.0),
    ) {
        prop_assume!(ax.iter().map(|c| c * c).sum::<f64>() > 0.01);
        let m = linked();
        let before = lk(&m);
        let after = lk(&rotate(&m, ax, ang));
        prop_assert!((before - after).abs() < 1e-6, "{before} vs {after}");
    }

    #[test]
    fn linking_number_invariant_under_uniform_scale(c in 0.2f64..5.0) {
        let m = linked();
        prop_assert!((lk(&m) - lk(&scale(&m, c))).abs() < 1e-6);
    }
}

/// The engine of the parity tier: reflection is an isometry of every distance
/// and an anti-isometry of the topology.
#[test]
fn linking_number_changes_sign_under_reflection() {
    let m = linked();
    let s = lk(&m) + lk(&mirror(&m));
    assert!(s.abs() < 1e-9, "Lk did not negate under reflection: sum {s}");
}

#[test]
fn writhe_changes_sign_under_reflection() {
    let m = two_rings(N, R, R, BOX);
    let w = writhe(&m.chains[0]);
    let wm = writhe(&mirror(&m).chains[0]);
    assert!((w + wm).abs() < 1e-9, "writhe did not negate: {w} vs {wm}");
}

#[test]
fn linking_matrix_covers_each_unordered_pair_once() {
    let m = two_rings(N, R, R, BOX);
    let lm = linking_matrix(&m);
    assert_eq!(lm.len(), 1, "two chains give exactly one pair");
    assert_eq!(lm[0].0, (0, 1));
}

// ----------------------------------------------- invariants of the descriptor

fn desc() -> Descriptor {
    Descriptor { radius: 2.0, bond_aware: false }
}

proptest! {
    /// Chain order is bookkeeping, not physics. The melt-level signature is a
    /// multiset over beads and must not know how the chains were listed.
    #[test]
    fn descriptor_invariant_under_chain_permutation(swap in any::<bool>()) {
        let m = linked();
        let mut p = m.clone();
        if swap { p.chains.swap(0, 1); }
        prop_assert_eq!(residual(&desc().signature(&m), &desc().signature(&p)), 0.0);
    }

    #[test]
    fn descriptor_invariant_under_isometry(
        ang in 0.0f64..std::f64::consts::TAU,
        ax in prop::array::uniform3(-1.0f64..1.0),
    ) {
        prop_assume!(ax.iter().map(|c| c * c).sum::<f64>() > 0.01);
        let m = linked();
        let res = residual(&desc().signature(&m), &desc().signature(&rotate(&m, ax, ang)));
        prop_assert!(res < 1e-9, "residual under rotation was {res}");
    }

    /// Catches a hardcoded epsilon: scaling the configuration by `c` must scale
    /// every descriptor distance by exactly `c` once the cutoff scales too.
    #[test]
    fn descriptor_is_scale_equivariant(c in 0.3f64..3.0) {
        let m = linked();
        let base = desc().signature(&m);
        let scaled = Descriptor { radius: desc().radius * c, bond_aware: false }
            .signature(&scale(&m, c));
        let res = residual(&base.scaled(c), &scaled);
        prop_assert!(res < 1e-9 * c.max(1.0), "residual {res} at c = {c}");
    }

    /// Moving every bead by at most eps changes every pairwise distance by at
    /// most 2*eps, hence the descriptor by at most 2*eps. Assumed away: pairs
    /// sitting within 2*eps of the hard cutoff, where the neighbour count
    /// itself changes. A real MLIP tapers there; this descriptor does not, and
    /// the assumption records that honestly.
    #[test]
    fn descriptor_is_stable_under_perturbation(
        eps in 1e-5f64..1e-2,
        seed in any::<u64>(),
    ) {
        let m = linked();
        let d = desc();
        let p = perturb(&m, eps, seed);
        let res = residual(&d.signature(&m), &d.signature(&p));
        prop_assume!(res.is_finite());
        prop_assert!(res <= 2.0 * eps + 1e-12, "residual {res} exceeds 2*eps = {}", 2.0 * eps);
    }
}

/// Negative control for the descriptor. It must see a genuine *local*
/// difference, or its blindness in the matched-pair tests proves nothing at all.
///
/// The difference used here is intra-ring: changing the ring radius changes
/// distances between neighbouring beads of the same ring, well inside the
/// cutoff. Note what is deliberately *not* used as the control — linked versus
/// unlinked, which the descriptor cannot separate below the inter-ring gap.
/// That is the result, not a defect, and it has its own test.
#[test]
fn descriptor_distinguishes_rings_of_different_radius() {
    let d = desc();
    let a = two_rings(N, R, R, BOX);
    let b = two_rings(N, R * 1.02, R, BOX);
    let res = residual(&d.signature(&a), &d.signature(&b));
    assert!(res > 1e-3, "descriptor is trivially blind: residual {res}");
}

#[test]
fn bond_aware_descriptor_distinguishes_rings_of_different_radius() {
    let d = Descriptor { radius: 2.0, bond_aware: true };
    let a = two_rings(N, R, R, BOX);
    let b = two_rings(N, R * 1.02, R, BOX);
    let res = residual(&d.signature(&a), &d.signature(&b));
    assert!(res > 1e-3, "bond-aware descriptor is trivially blind: residual {res}");
}

/// The sharp edge of the locality tier, from the other side. Once the cutoff
/// reaches past the inter-ring gap, beads on one ring see the other ring and the
/// descriptor separates the pair. Blindness below the gap is therefore a
/// property of the cutoff and not of a weak descriptor.
#[test]
fn descriptor_separates_linked_from_unlinked_once_cutoff_exceeds_the_gap() {
    let (a, b) = (linked(), unlinked());
    let gap = min_interchain_dist(&a).min(min_interchain_dist(&b));
    let d = Descriptor { radius: gap * 1.6, bond_aware: false };
    let res = residual(&d.signature(&a), &d.signature(&b));
    println!("gap {gap:.6}, cutoff {:.6}, residual {res:e}", d.radius);
    assert!(res > 0.0, "cutoff above the gap should separate the pair");
}
