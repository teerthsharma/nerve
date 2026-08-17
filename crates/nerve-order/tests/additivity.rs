//! Mechanism (b): sum-decomposability, tested against a **chirality-sensitive**
//! descriptor.
//!
//! Why the chiral fingerprint is the whole point. Round 1's parity claim was
//! struck for being definitional: reflection preserves a distance matrix, so a
//! distance descriptor returning identical output is a restatement of its own
//! definition. Here the fingerprint carries signed volumes — a parity-odd 4-body
//! invariant. Whether two melts share a *chiral* fingerprint multiset is a
//! contingent fact about their geometry, and the negative control below proves the
//! instrument can tell chirality apart.
//!
//! An earlier revision called this feature "strictly stronger than the standard
//! bispectrum, which Pozdnyakov's Fig. 2e shows is achiral". That is retracted —
//! *achiral* appears nowhere in that paper, Fig. 2e concerns identical **chiral**
//! features, and a signed-tetrahedron-volume multiset is at most as discriminating
//! as the chiral bispectrum. See the note on [`nerve_order::Fingerprint`]. Nothing
//! in this file depends on a strength ordering; the tests measure separation
//! directly.
//!
//! What that buys, and it is the reason this tier is worth more than parity:
//! the result reaches descriptors that admit pseudoscalars, which is what NequIP,
//! MACE and Allegro actually have, and it separates melts differing in the
//! **magnitude** of the linking number, which parity provably cannot.

use nerve_core::Melt;
use nerve_order::*;
use proptest::prelude::*;

const N: usize = 48;
const R: f64 = 3.0;
const BOX: f64 = 40.0;
const AMP: f64 = 0.3;
const WAVES: f64 = 3.0;

/// Below the measured inter-ring gap. Justified by measurement in
/// `guards_hold_for_every_melt_used`, not assumed.
const R_IN: f64 = 2.0;
/// Above the gap, where the pair must become separable.
const R_OUT: f64 = 5.0;

fn linked() -> Melt {
    wavy_rings(N, R, R, BOX, AMP, WAVES)
}
fn unlinked() -> Melt {
    wavy_rings(N, R, 3.0 * R, BOX, AMP, WAVES)
}
fn chiral(r_cut: f64) -> Local {
    Local { r_cut, chiral: true }
}

// ---------------------------------------------------------- negative controls

/// The control the entire tier rests on. If the signed-volume multiset could not
/// tell a configuration from its mirror image, then "identical chiral
/// fingerprints" would mean nothing and this tier would collapse back into the
/// struck parity claim.
///
/// This can fail: it does if the corrugation amplitude is zero, or if sorting
/// discards the sign, or if the volumes are accumulated as magnitudes.
#[test]
fn chiral_fingerprint_separates_a_configuration_from_its_mirror() {
    let m = linked();
    let d = chiral(R_IN);
    let res = fingerprint_residual(&d.fingerprints(&m), &d.fingerprints(&mirror(&m)));
    println!("chiral fingerprint residual against mirror image: {res:e}");
    assert!(res > 0.0, "chiral fingerprint is parity blind; the tier's control is vacuous");
}

/// Records *why* the rings are corrugated, so the choice is evidence rather than
/// an unexplained constant. A planar circle has every neighbour triple coplanar,
/// so every signed volume vanishes and the control above would pass vacuously on
/// it.
#[test]
fn planar_rings_have_no_local_chirality() {
    let flat = wavy_rings(N, R, R, BOX, 0.0, WAVES);
    let vols = signed_volume_multiset(&flat, R_IN);
    let worst = vols.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    println!("planar rings: {} signed volumes, largest magnitude {worst:e}", vols.len());
    assert!(!vols.is_empty(), "no triples found; cutoff too small to test anything");
    assert!(worst < 1e-12, "planar rings should have vanishing signed volumes, got {worst:e}");
}

/// Sensitivity floor for the instrument, stated as its own falsifiable claim
/// rather than folded into a ratio. Round 1 asserted `signal == 0` and then
/// `signal/noise == 0`, where the second follows from the first and so tested
/// nothing. These are two independent facts and each is checked separately.
#[test]
fn chiral_fingerprint_separates_two_different_seed_configurations() {
    let m = linked();
    let d = chiral(R_IN);
    let eps = 1e-3;
    let noise = fingerprint_residual(
        &d.fingerprints(&perturb(&m, eps, 101)),
        &d.fingerprints(&perturb(&m, eps, 202)),
    );
    println!("different-seed noise floor at eps {eps:e}: {noise:e}");
    assert!(noise > 0.0, "instrument cannot resolve two different seeds; it measures nothing");
}

// --------------------------------------------------------------- the claim

/// The primary claim of this crate.
///
/// Two melts, one with the rings threaded and one not, whose chirality-sensitive
/// local fingerprint multisets are identical. Contingent, not definitional: it
/// holds because ring A is untouched and ring B is moved by a **proper** isometry,
/// and because no bead is within `R_IN` of the other ring. Any of those failing
/// makes this test fail.
#[test]
fn locality_pair_chiral_fingerprints_are_identical_below_the_gap() {
    let (a, b) = (linked(), unlinked());
    let d = chiral(R_IN);
    let res = fingerprint_residual(&d.fingerprints(&a), &d.fingerprints(&b));
    println!(
        "chiral fingerprint residual below gap: {res:e}; |Lk| {:.12} vs {:.12}",
        lk(&a, 0, 1).abs(),
        lk(&b, 0, 1).abs()
    );
    assert_eq!(res, 0.0, "chiral fingerprints differ below the gap: {res:e}");
}

/// What parity cannot do. Reflection preserves `|Lk|` exactly, so no parity
/// argument reaches the magnitude. This pair differs in the magnitude itself.
#[test]
fn locality_pair_differs_in_linking_magnitude_not_only_in_sign() {
    let (a, b) = (linked(), unlinked());
    let (la, lb) = (lk(&a, 0, 1).abs(), lk(&b, 0, 1).abs());
    println!("|Lk| {la:.12} vs {lb:.12}");
    assert!((la - 1.0).abs() < 1e-6, "expected |Lk| = 1 for the threaded pair, got {la}");
    assert!(lb < 1e-6, "expected |Lk| = 0 for the unthreaded pair, got {lb}");
}

proptest! {
    /// The quantifier made concrete. The claim is that **every** energy of the
    /// form `E = sum_i f(env_i)` is equal across the pair, for any `f` — so the
    /// test samples `f` instead of fixing one. Bitwise equality, because equal
    /// multisets summed in canonical order agree exactly.
    #[test]
    fn sum_decomposable_energy_is_bitwise_equal_over_random_f(seed in any::<u64>()) {
        let d = chiral(R_IN);
        let (fa, fb) = (d.fingerprints(&linked()), d.fingerprints(&unlinked()));
        let (ea, eb) = (sum_decomposable_energy(&fa, seed), sum_decomposable_energy(&fb, seed));
        prop_assert_eq!(
            ea.to_bits(), eb.to_bits(),
            "seed {}: energies differ, {} vs {}", seed, ea, eb
        );
    }

    /// And the same `f` family must actually be able to tell melts apart, or the
    /// equality above is an artifact of a degenerate `f`.
    #[test]
    fn sum_decomposable_energy_separates_a_genuinely_different_melt(seed in any::<u64>()) {
        let d = chiral(R_IN);
        let a = d.fingerprints(&linked());
        let c = d.fingerprints(&wavy_rings(N, R * 1.05, R, BOX, AMP, WAVES));
        prop_assert_ne!(
            sum_decomposable_energy(&a, seed).to_bits(),
            sum_decomposable_energy(&c, seed).to_bits(),
            "seed {}: f cannot separate rings of different radius", seed
        );
    }
}

/// The regime boundary, which is what makes this a locality result rather than an
/// unconditional one. Once the cutoff reaches past the gap, beads see the other
/// ring and the pair separates. Sum-decomposability blindness is therefore
/// **conditional on the cutoff**, unlike parity.
#[test]
fn chiral_fingerprints_differ_once_cutoff_exceeds_the_gap() {
    let (a, b) = (linked(), unlinked());
    let d = chiral(R_OUT);
    let res = fingerprint_residual(&d.fingerprints(&a), &d.fingerprints(&b));
    println!("chiral fingerprint residual at cutoff {R_OUT} above the gap: {res:e}");
    assert!(res > 0.0, "cutoff above the gap must separate the pair");
}

// ------------------------------------------------------- null model and guards

/// Cameron's requirement. A pair matched only on local environments proves
/// nothing, because cheap single-chain size features separate melts with no
/// topology involved at all.
#[test]
fn locality_pair_matches_all_six_null_features() {
    let (a, b) = (linked(), unlinked());
    let g = feature_gap(&chain_features(&a), &chain_features(&b));
    let fa = chain_features(&a);
    println!("null gap: {g:?}");
    println!("rg2 {:.12}, relative rg2 gap {:e}", fa.rg2, g.rg2 / fa.rg2);
    for (name, v) in [
        ("end_to_end2", g.end_to_end2),
        ("contour_len", g.contour_len),
        ("bead_density", g.bead_density),
        ("max_bond", g.max_bond),
        ("msid", g.msid),
    ] {
        assert_eq!(v, 0.0, "{name} must be a hard zero, got {v:e}");
    }
    assert!(
        g.rg2 / fa.rg2 < 1e-14,
        "rg2 relative gap {:e} exceeds f64 rounding on the centre-of-mass sum",
        g.rg2 / fa.rg2
    );
}

/// Both guards, and the measurement that justifies `R_IN`.
#[test]
fn guards_hold_for_every_melt_used() {
    for (name, m) in [("linked", linked()), ("unlinked", unlinked())] {
        let f = chain_features(&m);
        let gap = min_interchain_dist(&m);
        println!(
            "{name}: max_bond {:.6} vs box_len/2 {:.6}, inter-ring gap {:.6}",
            f.max_bond,
            m.box_len / 2.0,
            gap
        );
        assert!(f.max_bond < m.box_len / 2.0, "{name}: bond guard violated");
        assert!(R_IN <= m.box_len / 2.0, "{name}: r_cut exceeds box_len/2");
        assert!(R_OUT <= m.box_len / 2.0, "{name}: r_cut exceeds box_len/2");
        assert!(gap > R_IN, "{name}: measured gap {gap:.6} does not exceed R_IN {R_IN}");
    }
}
