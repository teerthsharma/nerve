//! Test-first suite for nerve-orient.
//!
//! Two blindness mechanisms are under test, and my going-in position is that
//! BOTH fail. The tests are written to establish that rigorously rather than to
//! flatter either one.
//!
//! * **(a) traversal reversal** — expected VACUOUS: reversing a chain's index
//!   order moves no bead and changes no undirected bond, so the two "systems"
//!   are one system. The descriptor residual is zero because there is nothing
//!   to see.
//! * **(b) bond relabeling** — I predicted DEAD because "the only reconnections
//!   that preserve all six null features are isometry-relabelings, and those
//!   cannot change `|Lk|`". **That reasoning was WRONG and the tests falsified
//!   it.** Feature-indistinguishable reconnections are common — 404 were found
//!   across 5 architectures, one within 7.9e-3 against a noise floor of 4.6e-1,
//!   58x below it. The six features are not the obstruction.
//!
//!   The tier is nonetheless dead, for a third reason neither of my predictions
//!   anticipated: among those 404 feature-invisible reconnections the largest
//!   `|Lk|` change was **0.0037**, against 1.0 for a Hopf link. Feature-
//!   invisibility and topology-change turn out to be mutually exclusive here —
//!   a reconnection stays invisible only by staying local, and local rewiring
//!   cannot change how two chains interpenetrate. Both of my predictions are
//!   preserved as `#[ignore]`d failing tests under the names that record them.
//!
//! The load-bearing quantity throughout is `|Lk|`, not `Lk`. For an unoriented
//! two-component link only `|Lk|` is an invariant; `sign(Lk)` depends on an
//! arbitrary traversal choice and is therefore gauge, not an observable.

use nerve_baseline::{chain_features, Acsf, ChainFeatures};
use nerve_core::{Chain, Melt, Vec3};
use nerve_orient::{
    bead_multiset, bond_graph, crossover, max_bond, permute, reflect_z, reverse_chain,
    reverse_segment, swap_beads,
};
use nerve_topo::{linking_number_closed, linking_number_open, writhe_open, Closure};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::TAU;

// ------------------------------------------------------------------ fixtures

/// Boundary curves of an annulus with `n_twist` full twists: the (2,2n) torus
/// link, linking number exactly `n`. Same generator validated in nerve-topo.
fn twisted_band(n_twist: i32, r: f64, a: f64, m: usize) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut ca = Vec::with_capacity(m);
    let mut cb = Vec::with_capacity(m);
    for k in 0..m {
        let t = TAU * (k as f64) / (m as f64);
        let nt = (n_twist as f64) * t;
        let (c, s) = (t.cos(), t.sin());
        let u = [nt.cos() * c, nt.cos() * s, nt.sin()];
        ca.push([r * c + a * u[0], r * s + a * u[1], a * u[2]]);
        cb.push([r * c - a * u[0], r * s - a * u[1], -a * u[2]]);
    }
    (ca, cb)
}

/// A Hopf link as a two-chain melt, in a box large enough to be free space.
const BIG_BOX: f64 = 1000.0;

fn hopf_melt(n_twist: i32) -> Melt {
    let (a, b) = twisted_band(n_twist, 1.0, 0.3, 120);
    Melt::new(vec![Chain::new(a), Chain::new(b)], BIG_BOX)
}

/// Seeded unit-bond random-walk melt. Bond length 1 keeps `max_bond` far below
/// `box_len / 2`, which every feature in `ChainFeatures` depends on.
fn walk_melt(seed: u64, n_chains: usize, n_beads: usize, box_len: f64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut chains = Vec::with_capacity(n_chains);
    for _ in 0..n_chains {
        let mut p = [
            rng.gen_range(0.0..box_len),
            rng.gen_range(0.0..box_len),
            rng.gen_range(0.0..box_len),
        ];
        let mut beads = vec![p];
        for _ in 1..n_beads {
            let z: f64 = rng.gen_range(-1.0..1.0);
            let phi: f64 = rng.gen_range(0.0..TAU);
            let r = (1.0 - z * z).sqrt();
            p = [p[0] + r * phi.cos(), p[1] + r * phi.sin(), p[2] + z];
            beads.push(p);
        }
        chains.push(Chain::new(beads));
    }
    Melt::new(chains, box_len)
}

// ----------------------------------------------------------------- residuals

/// Relative difference, zero when both sides are exactly equal.
fn rel(x: f64, y: f64) -> f64 {
    if x == y {
        return 0.0;
    }
    let scale = x.abs().max(y.abs());
    if scale == 0.0 {
        0.0
    } else {
        (x - y).abs() / scale
    }
}

/// Per-feature relative residuals for all six of Cameron's null features.
fn feature_residuals(a: &ChainFeatures, b: &ChainFeatures) -> Vec<(&'static str, f64)> {
    let msid = if a.msid.len() != b.msid.len() {
        f64::INFINITY
    } else {
        a.msid
            .iter()
            .zip(&b.msid)
            .map(|(x, y)| rel(*x, *y))
            .fold(0.0, f64::max)
    };
    vec![
        ("rg2", rel(a.rg2, b.rg2)),
        ("end_to_end2", rel(a.end_to_end2, b.end_to_end2)),
        ("contour_len", rel(a.contour_len, b.contour_len)),
        ("bead_density", rel(a.bead_density, b.bead_density)),
        ("max_bond", rel(a.max_bond, b.max_bond)),
        ("msid", msid),
    ]
}

fn worst_feature_residual(a: &Melt, b: &Melt) -> (&'static str, f64) {
    feature_residuals(&chain_features(a), &chain_features(b))
        .into_iter()
        .fold(("none", 0.0), |acc, x| if x.1 > acc.1 { x } else { acc })
}

/// Cosine distance between melt-level ACSF descriptors.
fn descriptor_residual(a: &Melt, b: &Melt, r_cut: f64) -> f64 {
    assert!(
        r_cut <= a.box_len / 2.0 && r_cut <= b.box_len / 2.0,
        "guard violated: r_cut {r_cut} exceeds box_len/2"
    );
    let acsf = Acsf::new(r_cut);
    nerve_baseline::cosine_distance(&acsf.melt_descriptor(a), &acsf.melt_descriptor(b))
}

fn abs_lk_open(m: &Melt) -> f64 {
    linking_number_open(&m.chains[0].beads, &m.chains[1].beads).abs()
}

// ============================================================ MECHANISM (a)
// Traversal-direction reversal.

#[test]
fn reversal_preserves_the_bead_multiset() {
    let m = hopf_melt(1);
    let r = reverse_chain(&m, 0);
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&r));
    x.sort();
    y.sort();
    assert_eq!(x, y, "reversal moved a bead");
}

/// The vacuity lemma. Reversing a chain's index order maps bond {i, i+1} to
/// bond {n-2-i, n-1-i} — the same unordered pairs. So the undirected bond graph
/// is preserved *exactly*, not approximately. Any descriptor whose input is
/// (bead positions, undirected bonds) therefore receives byte-identical input,
/// and its blindness is correct behaviour rather than a defect.
#[test]
fn reversal_preserves_the_undirected_bond_graph() {
    let m = hopf_melt(1);
    let r = reverse_chain(&m, 0);
    assert_eq!(
        bond_graph(&m),
        bond_graph(&r),
        "reversal changed the undirected bond graph"
    );
}

#[test]
fn reversing_one_chain_negates_linking_number() {
    let m = hopf_melt(1);
    let base = linking_number_closed(&m.chains[0].beads, &m.chains[1].beads);
    let r = reverse_chain(&m, 0);
    let got = linking_number_closed(&r.chains[0].beads, &r.chains[1].beads);
    assert!(base.abs() > 0.5, "degenerate fixture: Lk={base}");
    assert!(
        (base + got).abs() < 1e-9,
        "reversing one chain must negate Lk: {base} vs {got}"
    );
}

#[test]
fn reversing_both_chains_preserves_linking_number() {
    let m = hopf_melt(2);
    let base = linking_number_closed(&m.chains[0].beads, &m.chains[1].beads);
    let r = reverse_chain(&reverse_chain(&m, 0), 1);
    let got = linking_number_closed(&r.chains[0].beads, &r.chains[1].beads);
    assert!(
        (base - got).abs() < 1e-9,
        "reversing both chains must preserve Lk: {base} vs {got}"
    );
}

/// The kill shot on mechanism (a). `|Lk|` is what an unoriented link actually
/// has, and reversal does not touch it. So reversal-blindness demonstrates
/// blindness to nothing observable.
#[test]
fn absolute_linking_number_is_invariant_under_reversal() {
    let m = hopf_melt(2);
    let base = abs_lk_open(&m);
    for r in [reverse_chain(&m, 0), reverse_chain(&m, 1)] {
        assert!(
            (base - abs_lk_open(&r)).abs() < 1e-9,
            "|Lk| changed under reversal: {base} vs {}",
            abs_lk_open(&r)
        );
    }
}

/// Writhe is invariant under orientation reversal (both `dr` flip, the cross
/// product's sign cancels), so unlike `sign(Lk)` it is a genuine observable of
/// an *unoriented* curve. It is also parity-odd. That combination is what makes
/// it the correct witness for a parity argument, and this test is the evidence.
#[test]
fn reversal_preserves_writhe() {
    let m = hopf_melt(3);
    let base = writhe_open(&m.chains[0].beads);
    assert!(base.abs() > 0.1, "degenerate fixture: writhe={base}");
    let r = reverse_chain(&m, 0);
    assert!(
        (base - writhe_open(&r.chains[0].beads)).abs() < 1e-9,
        "writhe changed under reversal: {base} vs {}",
        writhe_open(&r.chains[0].beads)
    );
}

#[test]
fn reversal_preserves_all_six_null_features() {
    let m = walk_melt(11, 4, 24, 40.0);
    let r = reverse_chain(&m, 0);
    let residuals = feature_residuals(&chain_features(&m), &chain_features(&r));
    println!("REVERSAL feature residuals: {residuals:?}");
    for (name, v) in residuals {
        assert!(v < 1e-12, "reversal perturbed {name} by {v:.3e}");
    }
}

#[test]
fn reversal_preserves_the_acsf_descriptor() {
    let m = walk_melt(12, 4, 24, 40.0);
    let r = reverse_chain(&m, 0);
    let d = descriptor_residual(&m, &r, 6.0);
    println!("REVERSAL descriptor residual: {d:.3e}");
    assert!(d < 1e-12, "reversal perturbed the ACSF by {d:.3e}");
}

// ============================================================ MECHANISM (b)
// Bond relabeling.

#[test]
fn reconnection_preserves_the_bead_multiset() {
    let m = walk_melt(21, 2, 16, 40.0);
    let c = crossover(&m, 0, 1, 8);
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&c));
    x.sort();
    y.sort();
    assert_eq!(x, y, "crossover moved a bead");
}

#[test]
fn crossover_reconnection_changes_the_bond_graph() {
    let m = walk_melt(22, 2, 16, 40.0);
    let c = crossover(&m, 0, 1, 8);
    assert_ne!(
        bond_graph(&m),
        bond_graph(&c),
        "crossover did not actually reconnect anything"
    );
}

/// A random permutation of the pooled beads is the naive "different bond
/// assignment", and it is unusable: it produces bonds spanning the box, which
/// violates the `max_bond < box_len / 2` guard that every feature in
/// `ChainFeatures` depends on. Above that guard the unwrapping is wrong and the
/// features describe a random walk regardless of input.
#[test]
fn random_permutation_reconnection_violates_the_max_bond_guard() {
    let m = walk_melt(23, 2, 16, 40.0);
    assert!(
        max_bond(&m) < m.box_len / 2.0,
        "fixture already violates the guard"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut violations = 0;
    for _ in 0..32 {
        let n = m.n_beads();
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.gen_range(0..=i));
        }
        if max_bond(&permute(&m, &perm)) >= m.box_len / 2.0 {
            violations += 1;
        }
    }
    println!("RANDOM PERMUTATION: {violations}/32 violate max_bond < box_len/2");
    assert!(
        violations > 24,
        "expected random reconnection to be guard-invalid nearly always, got \
         {violations}/32"
    );
}

/// Reproduces the death of the connectivity tier: a genuine reconnection that
/// keeps the bead set moves Cameron's cheap features, so they separate the pair
/// and the topology proves nothing.
#[test]
fn crossover_reconnection_perturbs_the_null_features() {
    let m = walk_melt(24, 2, 20, 40.0);
    let c = crossover(&m, 0, 1, 10);
    assert!(max_bond(&c) < c.box_len / 2.0, "crossover broke the guard");
    let residuals = feature_residuals(&chain_features(&m), &chain_features(&c));
    println!("CROSSOVER feature residuals: {residuals:?}");
    let worst = residuals.iter().fold(0.0f64, |a, x| a.max(x.1));
    // Bar tightened after adversarial review: the original `worst > 1e-3` was the
    // same order as the different-seed noise floor, so clearing it showed only
    // that crossover's effect is comparable to changing the random seed. The
    // claim needs the effect to be LARGE relative to that floor.
    let (_, noise) =
        worst_feature_residual(&walk_melt(33, 4, 24, 40.0), &walk_melt(34, 4, 24, 40.0));
    println!("  (noise floor {noise:.3e}, ratio {:.1}x)", worst / noise);
    assert!(
        worst > 2.0 * noise,
        "crossover perturbed features by only {worst:.3e} against a different-seed \
         noise floor of {noise:.3e} — not large relative to noise, so this is weak \
         evidence that reconnection is detectable at all"
    );
}

/// The one isometry-relabeling the reviewer identified as possibly NOT inert:
/// a rigid translation that pushes chains across a periodic boundary. Every
/// feature is computed on `min_image`-unwrapped coordinates, so if wrapping
/// changed which image a bond selects, the features would shift even though
/// nothing physically moved. Untested until now.
///
/// Prediction: preserved exactly, because every bond is far below `box_len / 2`
/// and `min_image` therefore reconstructs the same bond vector regardless of
/// where the chain sits relative to the boundary. If that is wrong the guard is
/// not doing the job the whole crate assumes it does.
#[test]
fn boundary_crossing_translation_preserves_all_six_features() {
    let l = 40.0;
    let m = walk_melt(51, 3, 20, l);
    assert!(max_bond(&m) < l / 2.0, "fixture broke the guard");
    let wrap = |p: Vec3| [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2].rem_euclid(l)];
    let base = chain_features(&m);
    let mut worst_overall = 0.0f64;
    let mut cases_with_straddle = 0usize;
    // Deterministic sweep rather than hand-picked offsets. The first version of
    // this test used four eyeballed translations and every one produced ZERO
    // wrapped bonds, so it proved nothing while appearing to pass. A fine sweep
    // along one axis guarantees some offsets land a bond across the boundary.
    let offsets: Vec<[f64; 3]> = (0..64).map(|i| [l * (i as f64) / 64.0, 0.0, 0.0]).collect();
    for t in offsets {
        let shifted = Melt::new(
            m.chains
                .iter()
                .map(|c| {
                    Chain::new(
                        c.beads
                            .iter()
                            .map(|p| wrap([p[0] + t[0], p[1] + t[1], p[2] + t[2]]))
                            .collect(),
                    )
                })
                .collect(),
            l,
        );
        // Count bonds that actually wrap, on any axis. Without this the test
        // could pass while never exercising a boundary crossing at all.
        let straddling = shifted
            .chains
            .iter()
            .flat_map(|c| c.beads.windows(2))
            .filter(|w| (0..3).any(|d| (w[0][d] - w[1][d]).abs() > l / 2.0))
            .count();
        let worst = feature_residuals(&base, &chain_features(&shifted))
            .into_iter()
            .fold(0.0f64, |a, x| a.max(x.1));
        if straddling > 0 {
            cases_with_straddle += 1;
        }
        worst_overall = worst_overall.max(worst);
    }
    println!(
        "  {cases_with_straddle}/64 translation offsets produced at least one \
         wrapped bond; worst feature residual over all offsets = {worst_overall:.3e}"
    );
    assert!(
        cases_with_straddle > 0,
        "no offset in the sweep produced a wrapped bond — the test proves nothing \
         about boundary crossing"
    );
    assert!(
        worst_overall < 1e-9,
        "boundary-crossing translation moved a feature by {worst_overall:.3e}; the \
         max_bond guard is not sufficient to make the features translation invariant"
    );
}

#[test]
fn segment_reversal_perturbs_msid() {
    let m = walk_melt(25, 2, 20, 40.0);
    let s = reverse_segment(&m, 0, 4, 14);
    assert_ne!(
        bond_graph(&m),
        bond_graph(&s),
        "segment reversal was a no-op"
    );
    let residuals = feature_residuals(&chain_features(&m), &chain_features(&s));
    println!("SEGMENT-REVERSAL feature residuals: {residuals:?}");
    let msid = residuals.iter().find(|r| r.0 == "msid").unwrap().1;
    assert!(
        msid > 1e-6,
        "partial segment reversal left msid within {msid:.3e}; only WHOLE-chain \
         reversal should be msid-neutral"
    );
}

/// The best case for mechanism (b): a reflection is an isometry, so it preserves
/// every intra-chain distance and therefore all six features exactly.
#[test]
fn reflection_preserves_all_six_null_features_exactly() {
    let m = walk_melt(26, 4, 24, 40.0);
    let f = reflect_z(&m);
    let residuals = feature_residuals(&chain_features(&m), &chain_features(&f));
    println!("REFLECTION feature residuals: {residuals:?}");
    for (name, v) in residuals {
        assert!(v < 1e-12, "reflection perturbed {name} by {v:.3e}");
    }
}

#[test]
fn reflection_preserves_the_acsf_descriptor() {
    let m = walk_melt(27, 4, 24, 40.0);
    let d = descriptor_residual(&m, &reflect_z(&m), 6.0);
    println!("REFLECTION descriptor residual: {d:.3e}");
    assert!(d < 1e-12, "reflection perturbed the ACSF by {d:.3e}");
}

/// The kill shot on mechanism (b). Reflection is the one transformation that
/// preserves all six features exactly — and it changes only the SIGN of Lk,
/// never its magnitude. So even the best case cannot exhibit a change in the
/// only linking quantity an unoriented pair of chains actually has.
#[test]
fn reflection_negates_linking_number_but_preserves_its_magnitude() {
    let m = hopf_melt(2);
    let base = linking_number_closed(&m.chains[0].beads, &m.chains[1].beads);
    let f = reflect_z(&m);
    let got = linking_number_closed(&f.chains[0].beads, &f.chains[1].beads);
    assert!(base.abs() > 0.5, "degenerate fixture: Lk={base}");
    assert!(
        (base + got).abs() < 1e-9,
        "reflection must negate Lk: {base} vs {got}"
    );
    assert!(
        (base.abs() - got.abs()).abs() < 1e-9,
        "|Lk| must be reflection invariant: {} vs {}",
        base.abs(),
        got.abs()
    );
}

/// And the reason reflection does not rescue the tier: on a generic bead set it
/// is not a reconnection at all. It MOVES the beads. To make it a relabeling of
/// a fixed bead set the set must be reflection-symmetric — at which point the
/// two configurations are mirror images and the argument IS the parity argument,
/// contributing nothing independent.
#[test]
fn reflection_is_not_a_reconnection_on_a_generic_bead_set() {
    let m = walk_melt(28, 2, 16, 40.0);
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&reflect_z(&m)));
    x.sort();
    y.sort();
    assert_ne!(
        x, y,
        "generic walk melt happened to be reflection symmetric — pick another seed"
    );
}

// ================================================================== CONTROL

/// Same-architecture different-seed control. Without this every "residual is
/// tiny" claim above is unreadable, because tiny relative to what is unstated.
#[test]
fn different_seed_control_gives_nonzero_descriptor_noise() {
    let r_cut = 6.0;
    let a = walk_melt(31, 4, 24, 40.0);
    let b = walk_melt(32, 4, 24, 40.0);
    let noise = descriptor_residual(&a, &b, r_cut);
    let signal_reversal = descriptor_residual(&a, &reverse_chain(&a, 0), r_cut);
    let signal_reflection = descriptor_residual(&a, &reflect_z(&a), r_cut);
    println!(
        "CONTROL  descriptor noise (different seed, same architecture) = {noise:.3e}\n  \
         signal/noise reversal   = {:.3e}\n  \
         signal/noise reflection = {:.3e}",
        signal_reversal / noise,
        signal_reflection / noise
    );
    assert!(
        noise > 1e-6,
        "different-seed noise floor is {noise:.3e}; with a floor this low the \
         descriptor is not resolving the architecture and no signal/noise \
         statement about it is meaningful"
    );
    assert!(
        signal_reversal / noise < 1e-3,
        "reversal signal/noise {:.3e} is not negligible",
        signal_reversal / noise
    );
    assert!(
        signal_reflection / noise < 1e-3,
        "reflection signal/noise {:.3e} is not negligible",
        signal_reflection / noise
    );
}

#[test]
fn different_seed_control_gives_nonzero_feature_noise() {
    let a = walk_melt(33, 4, 24, 40.0);
    let b = walk_melt(34, 4, 24, 40.0);
    let (name, noise) = worst_feature_residual(&a, &b);
    println!("CONTROL  worst feature residual (different seed) = {name} {noise:.3e}");
    assert!(
        noise > 1e-3,
        "different-seed feature noise floor is only {noise:.3e} ({name}); the \
         six features cannot separate anything at that resolution"
    );
}

#[test]
fn guards_hold_for_every_melt_used_in_this_suite() {
    let r_cut = 6.0;
    for (label, m) in [
        ("hopf1", hopf_melt(1)),
        ("hopf2", hopf_melt(2)),
        ("walk", walk_melt(41, 4, 24, 40.0)),
        ("walk_small", walk_melt(42, 2, 16, 40.0)),
    ] {
        let mb = max_bond(&m);
        assert!(
            mb < m.box_len / 2.0,
            "{label}: max_bond {mb} >= box_len/2 {}",
            m.box_len / 2.0
        );
        assert!(
            r_cut <= m.box_len / 2.0,
            "{label}: r_cut {r_cut} > box_len/2"
        );
    }
}

// =================================================================== SEARCH

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

    /// The search that decides the verdict on mechanism (b). Over guard-valid
    /// reconnections drawn from THREE move families only (crossover, inter-chain
    /// bead swap, partial segment reversal) — NOT "the standard polymer-move
    /// families", which was an overclaim and was struck. Families this search does
    /// NOT cover: length-preserving selection of double bridges (round 3 shows
    /// these OVERTURN the conclusion below), cyclic multi-chain bridging,
    /// end-bridging, reptation, and any move that relocates beads. Read this
    /// test's result as scoped to its three families and see
    /// `the_overturn_survives_without_a_chosen_invisibility_bar`.
    /// Original wording retained for the record: reconnections drawn from (crossover,
    /// inter-chain bead swap, partial segment reversal), assert that NONE both
    /// preserves all six null features and changes |Lk|.
    ///
    /// A failure here is a positive result and would mean the tier is repairable.
    #[test]
    fn prop_no_guard_valid_reconnection_preserves_features_and_changes_abs_lk(
        seed in 0u64..500,
        k in 1usize..11,
        i in 0usize..12,
        j in 0usize..12,
        which in 0usize..3,
    ) {
        let m = walk_melt(seed, 2, 12, 40.0);
        let cand = match which {
            0 => crossover(&m, 0, 1, k),
            1 => swap_beads(&m, 0, i, 1, j),
            _ => reverse_segment(&m, 0, i.min(j), i.max(j)),
        };
        // Only guard-valid reconnections are admissible evidence.
        proptest::prop_assume!(max_bond(&cand) < cand.box_len / 2.0);
        proptest::prop_assume!(bond_graph(&cand) != bond_graph(&m));

        let worst = feature_residuals(&chain_features(&m), &chain_features(&cand))
            .into_iter()
            .fold(0.0f64, |a, x| a.max(x.1));
        // Threshold corrected. This originally read `worst < 1e-9`, which is an
        // EXACTNESS threshold and made the test nearly vacuous: only isometry-
        // relabelings preserve the features bitwise, and that was already known
        // analytically. The bar that matters is what the features can RESOLVE,
        // which is the measured different-seed noise floor (~4.6e-1) — eight
        // orders of magnitude looser. A tenth of it is used, conservatively.
        // The claim under test is unchanged; only its resolution improved.
        let bar = worst_feature_residual(&walk_melt(33, 4, 24, 40.0), &walk_melt(34, 4, 24, 40.0)).1
            / 10.0;
        let preserves_features = worst < bar;
        let changes_abs_lk = (abs_lk_open(&m) - abs_lk_open(&cand)).abs() > 0.1;

        proptest::prop_assert!(
            !(preserves_features && changes_abs_lk),
            "FOUND a repair for the connectivity tier: seed={seed} which={which} \
             k={k} i={i} j={j} worst_residual={worst:.3e} \
             |Lk| {} -> {}",
            abs_lk_open(&m), abs_lk_open(&cand)
        );
    }

    /// Whole-chain reversal is feature-neutral for every melt the sampler picks.
    #[test]
    fn prop_reversal_is_always_feature_neutral(seed in 0u64..500, c in 0usize..3) {
        let m = walk_melt(seed, 3, 16, 40.0);
        let r = reverse_chain(&m, c);
        let worst = feature_residuals(&chain_features(&m), &chain_features(&r))
            .into_iter()
            .fold(0.0f64, |a, x| a.max(x.1));
        proptest::prop_assert!(
            worst < 1e-12,
            "seed={seed} chain={c}: reversal perturbed a feature by {worst:.3e}"
        );
    }
}

// ============================================== how hard did I actually look ==

/// A pass/fail search says "no counterexample found" and stops. This reports the
/// NEAR MISS: over a wide sweep of guard-valid reconnections, how close did any
/// of them come to preserving all six features, measured against the
/// different-seed noise floor that the control tests establish.
///
/// My up-front claim: the best near-miss stays far above the noise floor, i.e.
/// no reconnection gets even close. If that is false this test fails and the
/// tier is closer to repairable than I think.
/// KEPT VERBATIM AND FAILING. Prediction falsified: over 12523 guard-valid
/// reconnections the closest came to 7.866e-3 worst-feature-residual against a
/// different-seed noise floor of 4.604e-1 — 58x BELOW the floor. So
/// feature-indistinguishable reconnections DO exist and the six features are not
/// the obstruction. Ignored to keep the suite green; remove to watch it fail.
#[ignore = "PREMISE FALSE: a reconnection reached 7.866e-3, 58x below the 4.604e-1 noise floor"]
#[test]
fn closest_near_miss_reconnection_stays_far_above_the_noise_floor() {
    let noise = {
        let (_, n) =
            worst_feature_residual(&walk_melt(33, 4, 24, 40.0), &walk_melt(34, 4, 24, 40.0));
        n
    };
    let mut best = f64::INFINITY;
    let mut best_desc = String::new();
    let mut examined = 0usize;
    for seed in 0..60u64 {
        let m = walk_melt(seed, 2, 14, 40.0);
        let n = m.chains[0].beads.len();
        let mut cands: Vec<(String, Melt)> = Vec::new();
        for k in 1..n {
            cands.push((format!("crossover k={k}"), crossover(&m, 0, 1, k)));
        }
        for i in 0..n {
            for j in 0..n {
                cands.push((format!("swap {i}<->{j}"), swap_beads(&m, 0, i, 1, j)));
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                cands.push((format!("segrev {i}..{j}"), reverse_segment(&m, 0, i, j)));
            }
        }
        for (label, c) in cands {
            if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                continue;
            }
            examined += 1;
            let worst = feature_residuals(&chain_features(&m), &chain_features(&c))
                .into_iter()
                .fold(0.0f64, |a, x| a.max(x.1));
            if worst < best {
                best = worst;
                best_desc = format!("seed={seed} {label}");
            }
        }
    }
    println!(
        "NEAR MISS: examined {examined} guard-valid reconnections; closest \
         worst-feature-residual = {best:.3e} ({best_desc}); different-seed noise \
         floor = {noise:.3e}; ratio = {:.1}x",
        best / noise
    );
    assert!(examined > 5000, "search too small to conclude: {examined}");
    assert!(
        best > noise,
        "a reconnection came within the noise floor (best={best:.3e} vs \
         noise={noise:.3e}, {best_desc}) — the connectivity tier may be repairable"
    );
}

/// Which of the six features does the killing? If one cheap feature catches
/// every reconnection, the tier is dead for a structural reason rather than a
/// delicate one, and that is the finding worth reporting.
#[test]
fn max_bond_alone_detects_every_guard_valid_reconnection() {
    let mut caught_by_max_bond = 0usize;
    let mut total = 0usize;
    let mut missed: Vec<String> = Vec::new();
    for seed in 0..40u64 {
        let m = walk_melt(seed, 2, 14, 40.0);
        let n = m.chains[0].beads.len();
        let base = chain_features(&m);
        for k in 1..n {
            let c = crossover(&m, 0, 1, k);
            if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                continue;
            }
            total += 1;
            let r = feature_residuals(&base, &chain_features(&c));
            let mb = r.iter().find(|x| x.0 == "max_bond").unwrap().1;
            if mb > 1e-3 {
                caught_by_max_bond += 1;
            } else {
                missed.push(format!("seed={seed} k={k} max_bond_resid={mb:.3e}"));
            }
        }
    }
    println!(
        "ASSASSIN: max_bond alone flags {caught_by_max_bond}/{total} guard-valid \
         crossovers at >1e-3 relative"
    );
    if !missed.is_empty() {
        println!(
            "  missed by max_bond ({}): {:?}",
            missed.len(),
            &missed[..missed.len().min(5)]
        );
    }
    assert!(total > 100, "search too small: {total}");
    assert!(
        caught_by_max_bond * 10 >= total * 9,
        "max_bond only caught {caught_by_max_bond}/{total}; the tier does not die \
         on one cheap feature and the verdict needs restating"
    );
}

// ==================================================== THE DECIDING EXPERIMENT

/// Does a reconnection exist that the six features cannot DETECT (not merely
/// cannot detect exactly) and that changes `|Lk|`?
///
/// `closest_near_miss_reconnection_stays_far_above_the_noise_floor` is RED: a
/// reconnection was found at 7.866e-3 worst-feature-residual against a
/// different-seed noise floor of 4.604e-1. So feature-indistinguishable
/// reconnections exist. The only remaining question is whether any of them moves
/// the topology.
///
/// Prediction recorded before running: they do, and the connectivity tier is
/// therefore REPAIRABLE. If this test fails, the tier is dead for a subtler
/// reason than the one I originally argued, and the failure stands under this
/// name.
///
/// "Indistinguishable" is defined as worst feature residual below one tenth of
/// the measured different-seed noise floor — a deliberately conservative margin,
/// so a hit cannot be dismissed as sitting at the edge of resolution.
/// KEPT VERBATIM AND FAILING. Second prediction, also falsified. Across 5
/// architectures and 404 feature-indistinguishable guard-valid reconnections the
/// largest |Lk| change was 0.0037 — negligible against 1.0 for a Hopf link.
/// Ignored to keep the suite green; remove to watch it fail.
#[ignore = "PREMISE FALSE: largest |Lk| change among 404 feature-invisible reconnections was 0.0037"]
#[test]
fn a_feature_indistinguishable_reconnection_changes_absolute_linking_number() {
    let (_, noise) =
        worst_feature_residual(&walk_melt(33, 4, 24, 40.0), &walk_melt(34, 4, 24, 40.0));
    let bar = noise / 10.0;

    let mut indistinguishable = 0usize;
    let mut best: Option<(f64, f64, String)> = None; // (dlk, residual, label)
                                                     // Architecture sweep, added after an adversarial review pointed out that a
                                                     // single hardcoded 2-chain 14-bead fixture cannot support a general claim.
                                                     // More chains make the six features LESS sensitive (they average over
                                                     // chains), which favours the adversary, so this widens the attack surface
                                                     // rather than narrowing it.
    for (nc, nb) in [(2usize, 14usize), (3, 14), (4, 14), (2, 20), (3, 20)] {
        for seed in 0..40u64 {
            let m = walk_melt(seed, nc, nb, 40.0);
            let n = m.chains[0].beads.len();
            let base_f = chain_features(&m);
            let base_lk = abs_lk_open(&m);

            let mut cands: Vec<(String, Melt)> = Vec::new();
            for k in 1..n {
                cands.push((format!("crossover k={k}"), crossover(&m, 0, 1, k)));
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    cands.push((format!("segrev {i}..{j}"), reverse_segment(&m, 0, i, j)));
                }
            }
            for i in 0..n {
                for j in 0..n {
                    cands.push((format!("swap {i}<->{j}"), swap_beads(&m, 0, i, 1, j)));
                }
            }

            for (label, c) in cands {
                if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                    continue;
                }
                let worst = feature_residuals(&base_f, &chain_features(&c))
                    .into_iter()
                    .fold(0.0f64, |a, x| a.max(x.1));
                if worst >= bar {
                    continue;
                }
                indistinguishable += 1;
                let dlk = (abs_lk_open(&c) - base_lk).abs();
                if best.as_ref().is_none_or(|b| dlk > b.0) {
                    best = Some((dlk, worst, format!("nc={nc} nb={nb} seed={seed} {label}")));
                }
            }
        }
    }

    let (dlk, resid, label) = best.expect("no feature-indistinguishable reconnection found at all");
    println!(
        "DECIDING EXPERIMENT: noise floor={noise:.3e}, bar={bar:.3e}\n  \
         {indistinguishable} feature-indistinguishable guard-valid reconnections found\n  \
         largest |Lk| change among them = {dlk:.4} (residual {resid:.3e}, {label})"
    );
    assert!(
        dlk > 0.1,
        "largest |Lk| change among {indistinguishable} feature-indistinguishable \
         reconnections was only {dlk:.4} ({label}) — the six features cannot SEE \
         these reconnections, but the reconnections do not move the topology \
         either, so the connectivity tier is dead for that reason"
    );
}

// ================================ evidence for the struck nerve-topo test ===
//
// The Inspector struck `directional_closure_disagrees_with_itself_across_
// directions` in nerve-topo for having a name that asserts the opposite of what
// its body checks. nerve-topo is frozen, so the rename is escalated rather than
// applied. What can be supplied without touching it is the missing evidence
// that the assertion has teeth.
//
// My up-front prediction, recorded before running: agreement between the
// far-field directional closure and the direct chord closure DEGRADES as the
// chains get more open, and falls below majority for a strongly-open pair. If
// that is false this test stays RED under this name.

fn directional_agreement(keep: usize) -> (f64, usize) {
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    let a: Vec<Vec3> = a.into_iter().take(keep).collect();
    let b: Vec<Vec3> = b.into_iter().take(keep).collect();
    let m = Melt::new(vec![Chain::new(a), Chain::new(b)], BIG_BOX);
    let direct = nerve_topo::linking_number(&m, 0, 1, Closure::Direct);
    let agree = nerve_topo::fibonacci_directions(64)
        .into_iter()
        .filter(|dir| {
            let lk = nerve_topo::linking_number(
                &m,
                0,
                1,
                Closure::Directional {
                    dir: *dir,
                    r_scale: 200.0,
                },
            );
            lk.round() == direct.round()
        })
        .count();
    (direct, agree)
}

/// KEPT VERBATIM AND FAILING. My prediction was wrong and this test records
/// that rather than being edited to agree with the measurement.
///
/// Measured agreement vs truncation: 54, 52, 45, 37, 40, 55 out of 64 for
/// keep = 40, 60, 80, 100, 140, 180. Agreement does NOT fall monotonically with
/// openness — it dips in the middle and recovers at both ends, and it never
/// drops below majority. The minimum is 37/64 at keep=100 against the
/// assertion's bar of 33/64, so the margin is real but thin.
///
/// Consequence I have to own: I set out to supply empirical evidence that the
/// struck nerve-topo assertion has teeth, and I FAILED to. Openness is not the
/// knob that falsifies it. Marked `ignore` so the suite stays green while the
/// false premise stays in the tree under its original name; remove the attribute
/// to watch it fail.
#[ignore = "PREMISE FALSE: agreement never falls below majority (min 37/64 at keep=100); openness is not the falsifying knob"]
#[test]
fn closure_direction_agreement_falls_below_majority_for_strongly_open_chains() {
    for keep in [40usize, 60, 80, 100, 140, 180] {
        let (direct, agree) = directional_agreement(keep);
        println!("  keep={keep:3}/200 direct={direct:+.0} agreement={agree}/64");
    }
    let (_, nearly_closed) = directional_agreement(180);
    let (_, strongly_open) = directional_agreement(40);
    println!(
        "FALSIFIABILITY: nearly-closed agreement={nearly_closed}/64, \
         strongly-open agreement={strongly_open}/64"
    );
    assert!(
        nearly_closed * 2 > 64,
        "nearly-closed pair should keep majority agreement, got {nearly_closed}/64"
    );
    assert!(
        strongly_open * 2 <= 64,
        "PREMISE FALSE: strongly-open pair kept majority agreement \
         ({strongly_open}/64), so closure agreement does NOT degrade with \
         openness the way I predicted"
    );
}

// ============================================================== ROUND 3
// Length-preserving connectivity-altering moves: the families round 2 left out
// and whose absence I flagged as the live attack on my own verdict.
//
// FAMILIES SEARCHED IN ROUND 2: equal-index crossover (== double bridging's
// rewiring), inter-chain bead swap, partial segment reversal.
// FAMILIES NOT SEARCHED IN ROUND 2, added here: length-preserving selection of
// double bridges, cyclic three-chain bridging, end-bridging, reptation.
// STILL NOT COVERED even after round 3, stated so it is auditable in code
// rather than only in prose: any move that RELOCATES beads (concerted rotation,
// pivot, kink-jump, classical reptation where the displaced bead takes a new
// position) is excluded by construction, because the connectivity tier fixes
// the bead point set. See `classical_double_bridging_is_not_a_fixed_bead_set_move`.
//
// PREDICTION, RECORDED BEFORE SEARCHING: the verdict is OVERTURNED. My round-2
// structural argument was "feature-invisibility requires locality, and local
// rewiring cannot change interpenetration". Double bridging refutes the premise
// directly: it is a LOCAL rewiring (two short bonds exchanged) with a GLOBAL
// topological effect (the two chains swap tails, so everything downstream of the
// bridge changes chain identity). That is why the equilibration literature uses
// it to decorrelate entanglement. I expect a length-preserving double bridge
// that is invisible to all six features at the noise-floor bar and moves |Lk|
// past 0.1.
//
// Round-2 numbers carried forward as the bar to beat: 12,523 guard-valid
// reconnections, closest worst-feature residual 7.866e-3 against a 4.604e-1
// noise floor; 404 feature-indistinguishable reconnections, largest |Lk| change
// 0.0037 versus 1.0 for a Hopf link.

use nerve_orient::{
    chain_length_profile, end_bridge, new_bond_lengths, relocate_bead, triple_bridge,
};

/// Dense periodic melt with FLUCTUATING bond lengths, wrapped into the box.
///
/// Two deliberate differences from `walk_melt`. Density: chains must actually
/// come close, or no length-preserving reconnection can exist at all. And
/// jitter: `walk_melt` emits bonds of length exactly 1.0, which makes `max_bond`
/// identical across every seed and therefore a zero-noise razor that flags any
/// new bond longer than 1.0. Real chains have bond fluctuation, and whether
/// `max_bond` kills these moves turns entirely on that.
fn dense_melt(seed: u64, n_chains: usize, n_beads: usize, box_len: f64, jitter: f64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let w = |p: [f64; 3]| -> Vec3 {
        [
            p[0].rem_euclid(box_len),
            p[1].rem_euclid(box_len),
            p[2].rem_euclid(box_len),
        ]
    };
    let mut chains = Vec::with_capacity(n_chains);
    for _ in 0..n_chains {
        let mut p = [
            rng.gen_range(0.0..box_len),
            rng.gen_range(0.0..box_len),
            rng.gen_range(0.0..box_len),
        ];
        let mut beads = vec![w(p)];
        for _ in 1..n_beads {
            let z: f64 = rng.gen_range(-1.0..1.0);
            let phi: f64 = rng.gen_range(0.0..TAU);
            let r = (1.0 - z * z).sqrt();
            let len = 1.0 + jitter * rng.gen_range(-1.0..1.0);
            p = [
                p[0] + len * r * phi.cos(),
                p[1] + len * r * phi.sin(),
                p[2] + len * z,
            ];
            beads.push(w(p));
        }
        chains.push(Chain::new(beads));
    }
    Melt::new(chains, box_len)
}

const DENSE_CHAINS: usize = 8;
const DENSE_BEADS: usize = 20;
const DENSE_BOX: f64 = 8.0;
const DENSE_JITTER: f64 = 0.1;

fn dense(seed: u64) -> Melt {
    dense_melt(seed, DENSE_CHAINS, DENSE_BEADS, DENSE_BOX, DENSE_JITTER)
}

/// Noise floor for the dense architecture. Round 2's 4.604e-1 was measured on
/// `walk_melt` and must not be reused here.
fn dense_noise_floor() -> f64 {
    worst_feature_residual(&dense(90_001), &dense(90_002)).1
}

fn abs_lk_pair(m: &Melt, a: usize, b: usize) -> f64 {
    nerve_topo::linking_number(m, a, b, Closure::Open).abs()
}

// ------------------------------------------------------------- preconditions

#[test]
fn dense_melt_satisfies_the_guards() {
    for seed in [1u64, 2, 3] {
        let m = dense(seed);
        let mb = max_bond(&m);
        assert!(
            mb < m.box_len / 2.0,
            "seed {seed}: max_bond {mb} >= box_len/2 {}",
            m.box_len / 2.0
        );
        assert!(
            mb > 0.85 && mb < 1.15,
            "seed {seed}: unexpected max_bond {mb}"
        );
    }
    let m = dense(1);
    let density = m.n_beads() as f64 / m.box_len.powi(3);
    println!(
        "DENSE MELT: {} beads, box {}, density {density:.3}, max_bond {:.4}",
        m.n_beads(),
        m.box_len,
        max_bond(&m)
    );
    assert!(density > 0.2, "melt too dilute for contacts: {density:.3}");
}

/// Without non-bonded beads at bond distance, no length-preserving reconnection
/// can exist on a fixed bead set at all, and every search below would be
/// vacuous. Assert the precondition instead of trusting a later pass.
#[test]
fn dense_melt_has_nonbonded_pairs_within_bond_length() {
    let m = dense(1);
    let mb = max_bond(&m);
    let mut close = 0usize;
    for (ci, c) in m.chains.iter().enumerate() {
        for (bi, p) in c.beads.iter().enumerate() {
            for (cj, d) in m.chains.iter().enumerate() {
                for (bj, q) in d.beads.iter().enumerate() {
                    if ci == cj && bi.abs_diff(bj) <= 1 {
                        continue;
                    }
                    if m.min_image_dist(*p, *q) <= mb {
                        close += 1;
                    }
                }
            }
        }
    }
    println!("CONTACTS: {close} ordered non-bonded bead pairs within max_bond {mb:.4}");
    assert!(
        close > 0,
        "no non-bonded contacts; length-preserving moves are impossible"
    );
}

#[test]
fn double_bridge_creates_two_new_bonds_and_preserves_chain_lengths() {
    let m = dense(1);
    let d = crossover(&m, 0, 1, 7);
    assert_eq!(
        chain_length_profile(&m),
        chain_length_profile(&d),
        "chain lengths moved"
    );
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&d));
    x.sort();
    y.sort();
    assert_eq!(x, y, "double bridge moved a bead");
    let nb = new_bond_lengths(&m, &d);
    println!("DOUBLE BRIDGE new bonds: {nb:?}");
    assert_eq!(nb.len(), 2, "expected 2 new bonds, got {}", nb.len());
}

#[test]
fn triple_bridge_creates_three_new_bonds_and_preserves_chain_lengths() {
    let m = dense(1);
    let t = triple_bridge(&m, 0, 1, 2, 7);
    assert_eq!(
        chain_length_profile(&m),
        chain_length_profile(&t),
        "chain lengths moved"
    );
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&t));
    x.sort();
    y.sort();
    assert_eq!(x, y, "triple bridge moved a bead");
    let nb = new_bond_lengths(&m, &t);
    println!("TRIPLE BRIDGE new bonds: {nb:?}");
    assert_eq!(nb.len(), 3, "expected 3 new bonds, got {}", nb.len());
}

/// End-bridging, and its reptation special case, are polydisperse by
/// construction: one chain lengthens and another shortens. That changes
/// `contour_len` and the length of the `msid` vector, so both are visible to the
/// six features without any topology being involved.
#[test]
fn end_bridge_and_reptation_change_the_chain_length_profile() {
    let m = dense(1);
    let e = end_bridge(&m, 0, 1, 12);
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&e));
    x.sort();
    y.sort();
    assert_eq!(x, y, "end bridge moved a bead");
    assert_eq!(
        new_bond_lengths(&m, &e).len(),
        1,
        "end bridge makes one bond"
    );
    assert_ne!(
        chain_length_profile(&m),
        chain_length_profile(&e),
        "end bridge must change the chain length profile"
    );
    let reptation = end_bridge(&m, 0, 1, DENSE_BEADS - 1);
    assert_ne!(chain_length_profile(&m), chain_length_profile(&reptation));
    println!(
        "END BRIDGE profile {:?} -> {:?}; reptation -> {:?}",
        chain_length_profile(&m),
        chain_length_profile(&e),
        chain_length_profile(&reptation)
    );
}

/// Classical double bridging and reptation RELOCATE beads: the bridging geometry
/// solves for new positions satisfying bond length and angle constraints, and
/// reptation's displaced bead is placed at a freshly sampled position. The
/// connectivity tier fixes the bead point set, so those forms are unavailable
/// here by construction and no search can cover them. Recorded as a test so the
/// limitation is auditable in code rather than only in prose.
#[test]
fn classical_double_bridging_is_not_a_fixed_bead_set_move() {
    let m = dense(1);
    let mb = max_bond(&m);
    // A fixed-bead-set bridge can only use bonds that already exist as
    // inter-bead distances. Count how many of the possible bridges are even
    // geometrically available; the shortfall is what classical DB buys by
    // moving beads.
    let n = DENSE_BEADS;
    let (mut available, mut total) = (0usize, 0usize);
    for a in 0..m.chains.len() {
        for b in (a + 1)..m.chains.len() {
            for k in 1..n {
                total += 1;
                let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                if d1 <= mb && d2 <= mb {
                    available += 1;
                }
            }
        }
    }
    println!(
        "FIXED-BEAD-SET FEASIBILITY: {available}/{total} candidate double bridges have \
         both new bonds within max_bond {mb:.4} ({:.3}%)",
        100.0 * available as f64 / total as f64
    );
    assert!(
        available < total,
        "every candidate bridge was length-feasible, which would mean the fixed \
         bead set imposes no constraint at all"
    );
}

/// The razor finding: whether `max_bond` can detect these moves at all depends
/// entirely on bond rigidity, which is a property of the model, not the topology.
#[test]
fn max_bond_noise_floor_is_zero_for_rigid_bonds_and_nonzero_with_jitter() {
    let rigid = |s: u64| dense_melt(s, DENSE_CHAINS, DENSE_BEADS, DENSE_BOX, 0.0);
    let r = rel(max_bond(&rigid(1)), max_bond(&rigid(2)));
    let j = rel(max_bond(&dense(1)), max_bond(&dense(2)));
    println!("MAX_BOND noise floor: rigid bonds = {r:.3e}, jittered bonds = {j:.3e}");
    assert!(
        r < 1e-15,
        "rigid-bond max_bond should be seed independent, got {r:.3e}"
    );
    assert!(
        j > 1e-4,
        "jittered max_bond should vary between seeds, got {j:.3e}"
    );
}

// ------------------------------------------------------- the deciding search

/// Sweep length-preserving double bridges across dense melts and report the
/// largest |Lk| change among those invisible to all six features.
///
/// "Length-preserving" means every new bond is no longer than the longest bond
/// already present, so `max_bond` is unchanged. "Invisible" means worst feature
/// residual below one tenth of the measured dense-architecture noise floor.
/// |Lk| threshold 0.1, against 1.0 for a Hopf link; `linking_number_open` is
/// real-valued, so the threshold is stated rather than assumed.
fn double_bridge_search(seeds: u64) -> (usize, usize, f64, String) {
    let bar = dense_noise_floor() / 10.0;
    let mut length_preserving = 0usize;
    let mut invisible = 0usize;
    let mut best = 0.0f64;
    let mut best_label = "none".to_string();
    for seed in 0..seeds {
        let m = dense(seed);
        let nc = m.chains.len();
        let mb = max_bond(&m);
        let base_f = chain_features(&m);
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..DENSE_BEADS {
                    // Cheap filter first: two distances before any O(N^2) work.
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 > mb || d2 > mb {
                        continue;
                    }
                    let c = crossover(&m, a, b, k);
                    let nb = new_bond_lengths(&m, &c);
                    if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                        continue;
                    }
                    length_preserving += 1;
                    let worst = feature_residuals(&base_f, &chain_features(&c))
                        .into_iter()
                        .fold(0.0f64, |x, y| x.max(y.1));
                    if worst >= bar {
                        continue;
                    }
                    invisible += 1;
                    let dlk = (abs_lk_pair(&m, a, b) - abs_lk_pair(&c, a, b)).abs();
                    if dlk > best {
                        best = dlk;
                        best_label = format!(
                            "seed={seed} chains=({a},{b}) k={k} resid={worst:.3e} newbonds={nb:?}"
                        );
                    }
                }
            }
        }
    }
    (length_preserving, invisible, best, best_label)
}

#[test]
fn length_preserving_double_bridges_exist_in_a_dense_melt() {
    let (lp, _, _, _) = double_bridge_search(60);
    println!("LENGTH-PRESERVING DOUBLE BRIDGES found across 60 dense melts: {lp}");
    assert!(
        lp > 0,
        "no length-preserving double bridge exists on a fixed bead set across 60 \
         dense melts; the deciding test below would be vacuous"
    );
}

// ===== WITHDRAWN (round 5). =====================================================
// This test measures a real effect inside a regime that does not survive
// hardening. `dense_melt` carries no excluded volume; its jitter is 0.1 against
// a measured real-melt FENE relative sd of 3.401%; and the |Lk| bar reached
// through `image_carriers` is hardcoded at 0.1, not measured. On MD-equilibrated
// Kremer-Grest melts the move family has no length-feasible instances at all.
// The conclusion this test was written to support is WITHDRAWN. The withdrawal
// is asserted by `withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening`.
// Kept, not deleted: the in-regime measurement is real and reproducible, and the
// regime boundary is the finding.
// ===============================================================================
/// THE DECIDING TEST for round 3. Prediction: passes, overturning my round-2
/// verdict. If it fails it stays RED under this name.
#[test]
fn a_length_preserving_double_bridge_is_feature_invisible_and_changes_linking() {
    let nf = dense_noise_floor();
    let (lp, inv, best, label) = double_bridge_search(300);
    println!(
        "DECIDING (round 3, double bridge): dense noise floor={nf:.3e} bar={:.3e}\n  \
         length-preserving candidates      = {lp}\n  \
         of those, feature-invisible       = {inv}\n  \
         largest |Lk| change among invisible = {best:.4}\n  \
         best = {label}",
        nf / 10.0
    );
    assert!(lp > 0, "search vacuous: no length-preserving candidates");
    assert!(
        best > 0.1,
        "largest |Lk| change among {inv} feature-invisible length-preserving double \
         bridges was only {best:.4} ({label}) — double bridging did NOT overturn the \
         verdict"
    );
}

// ------------------------------------------- attacking the overturning result
//
// The search says a length-preserving double bridge is feature-invisible and
// moves |Lk| by 2.2869. Before that stands, three confounds have to die.
//
// PREDICTIONS, RECORDED BEFORE RUNNING:
//  1. The ACSF residual is exactly zero, because the bead set is untouched.
//     High confidence: ACSF reads positions only.
//  2. The |Lk| change survives the periodic-image ambiguity. THIS IS THE ONE I
//     AM LEAST SURE OF. `linking_number` places chain b at the nearest image by
//     centroid separation, and a tail swap moves centroids a long way, so the
//     image choice can flip and shift |Lk| with no topology change at all. If
//     the image spread is comparable to 2.2869 the result is confounded and the
//     overturn is withdrawn.
//  3. The effect is not a single cherry-picked outlier: a substantial number of
//     the feature-invisible candidates clear the 0.1 threshold.

/// Rebuild the winning candidate from the deciding search.
fn overturning_case() -> (Melt, Melt, usize, usize) {
    let (a, b, k) = (0usize, 6usize, 4usize);
    let m = dense(298);
    let c = crossover(&m, a, b, k);
    (m, c, a, b)
}

// ===== WITHDRAWN (round 5). =====================================================
// This test measures a real effect inside a regime that does not survive
// hardening. `dense_melt` carries no excluded volume; its jitter is 0.1 against
// a measured real-melt FENE relative sd of 3.401%; and the |Lk| bar reached
// through `image_carriers` is hardcoded at 0.1, not measured. On MD-equilibrated
// Kremer-Grest melts the move family has no length-feasible instances at all.
// The conclusion this test was written to support is WITHDRAWN. The withdrawal
// is asserted by `withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening`.
// Kept, not deleted: the in-regime measurement is real and reproducible, and the
// regime boundary is the finding.
// ===============================================================================
#[test]
fn the_overturning_double_bridge_leaves_the_bead_set_and_descriptor_untouched() {
    let (m, c, _, _) = overturning_case();
    let (mut x, mut y) = (bead_multiset(&m), bead_multiset(&c));
    x.sort();
    y.sort();
    assert_eq!(x, y, "the winning candidate moved a bead");
    assert_eq!(
        chain_length_profile(&m),
        chain_length_profile(&c),
        "the winning candidate changed the chain length profile"
    );
    let nb = new_bond_lengths(&m, &c);
    let mb = max_bond(&m);
    println!(
        "OVERTURN CASE: new bonds {nb:?} vs pre-existing max_bond {mb:.4}; \
         max_bond after = {:.4}",
        max_bond(&c)
    );
    assert_eq!(nb.len(), 2, "expected a two-bond double bridge");
    assert!(
        nb.iter().all(|l| *l <= mb),
        "not length-preserving: {nb:?} exceeds {mb:.4}"
    );
    // r_cut <= box_len/2 guard: box is 8, so 4.0 is the ceiling.
    let d = descriptor_residual(&m, &c, 4.0);
    println!("OVERTURN CASE: ACSF descriptor residual = {d:.3e}");
    assert_eq!(d, 0.0, "ACSF should be identical on an untouched bead set");
}

/// KEPT VERBATIM AND FAILING. Prediction falsified, but by a BAD INSTRUMENT of
/// mine rather than a real confound, and the correction matters more than the
/// verdict. This test used `image_spread`'s (max - min) range as the ambiguity
/// measure; that range conflates linking with ambiguity, since round 1 measured a
/// clean isolated Hopf link as range 1.0 with exactly one carrying image. It also
/// tripped over a nerve-topo inconsistency: `image_spread` ignores
/// `Closure::Open` and silently returns the CLOSED integer linking number, which
/// is why it reported 3.0000 for a configuration whose open |Lk| is 2.3163.
/// Replaced by `image_carriers`, which counts carrying images and is calibrated
/// against a known Hopf link. Ignored to keep the suite green; remove to watch it
/// fail.
#[ignore = "PREMISE FALSE via a bad instrument: image_spread range conflates linking with ambiguity; superseded by image_carriers. (Its Closure::Open routing was a separate defect, since fixed in nerve-topo.)"]
#[test]
fn the_overturning_double_bridge_survives_the_periodic_image_ambiguity() {
    let (m, c, a, b) = overturning_case();
    let before = nerve_topo::linking_number(&m, a, b, Closure::Open);
    let after = nerve_topo::linking_number(&c, a, b, Closure::Open);
    let dlk = (before.abs() - after.abs()).abs();

    let sb = nerve_topo::image_spread(&m, a, b, Closure::Open);
    let sa = nerve_topo::image_spread(&c, a, b, Closure::Open);
    let ambiguity = (sb.max - sb.min).max(sa.max - sa.min);

    println!(
        "IMAGE CONFOUND: Lk before={before:+.4} after={after:+.4} d|Lk|={dlk:.4}\n  \
         image spread before: min={:+.4} max={:+.4} (range {:.4})\n  \
         image spread after:  min={:+.4} max={:+.4} (range {:.4})\n  \
         worst image ambiguity = {ambiguity:.4}",
        sb.min,
        sb.max,
        sb.max - sb.min,
        sa.min,
        sa.max,
        sa.max - sa.min
    );
    assert!(
        dlk > ambiguity,
        "d|Lk| {dlk:.4} does not exceed the periodic-image ambiguity \
         {ambiguity:.4} — the overturning result is CONFOUNDED by the nearest-image \
         choice and is withdrawn"
    );
}

// ===== WITHDRAWN (round 5). =====================================================
// This test measures a real effect inside a regime that does not survive
// hardening. `dense_melt` carries no excluded volume; its jitter is 0.1 against
// a measured real-melt FENE relative sd of 3.401%; and the |Lk| bar reached
// through `image_carriers` is hardcoded at 0.1, not measured. On MD-equilibrated
// Kremer-Grest melts the move family has no length-feasible instances at all.
// The conclusion this test was written to support is WITHDRAWN. The withdrawal
// is asserted by `withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening`.
// Kept, not deleted: the in-regime measurement is real and reproducible, and the
// regime boundary is the finding.
// ===============================================================================
/// Count, do not cherry-pick. Reports the whole distribution of |Lk| change over
/// the feature-invisible length-preserving double bridges.
#[test]
fn many_invisible_double_bridges_change_linking_not_just_one() {
    let bar = dense_noise_floor() / 10.0;
    let mut dlks: Vec<f64> = Vec::new();
    for seed in 0..300u64 {
        let m = dense(seed);
        let nc = m.chains.len();
        let mb = max_bond(&m);
        let base_f = chain_features(&m);
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..DENSE_BEADS {
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 > mb || d2 > mb {
                        continue;
                    }
                    let c = crossover(&m, a, b, k);
                    let nb = new_bond_lengths(&m, &c);
                    if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                        continue;
                    }
                    let worst = feature_residuals(&base_f, &chain_features(&c))
                        .into_iter()
                        .fold(0.0f64, |x, y| x.max(y.1));
                    if worst >= bar {
                        continue;
                    }
                    dlks.push((abs_lk_pair(&m, a, b) - abs_lk_pair(&c, a, b)).abs());
                }
            }
        }
    }
    dlks.sort_by(|p, q| q.partial_cmp(p).unwrap());
    let n = dlks.len();
    let over = dlks.iter().filter(|d| **d > 0.1).count();
    let median = dlks[n / 2];
    println!(
        "DISTRIBUTION over {n} feature-invisible length-preserving double bridges:\n  \
         d|Lk| > 0.1 in {over}/{n} ({:.1}%)\n  \
         top 5 = {:?}\n  median = {median:.4}",
        100.0 * over as f64 / n as f64,
        &dlks[..n.min(5)]
    );
    assert!(n > 20, "distribution too small to characterise: {n}");
    assert!(
        over >= 10,
        "only {over}/{n} feature-invisible double bridges clear 0.1 — the overturn \
         may rest on a handful of outliers"
    );
}

// --------------------------------- the unconfounded topology measurement
//
// `the_overturning_double_bridge_survives_the_periodic_image_ambiguity` is RED
// and stays RED: image-spread range 3.0000 exceeded the 2.2869 signal. But that
// test's own instrument was wrong, and the correction matters more than the
// verdict. `image_spread`'s (max - min) range is NOT a measure of ambiguity: for
// a clean isolated Hopf link, round 1 measured min=-1.5e-16, max=+1.0, range
// 1.0, with exactly one image carrying the link. Range therefore conflates the
// linking with the ambiguity about it.
//
// The right criterion is round 1's: HOW MANY of the 27 images carry appreciable
// linking. One carrier (or none) means the answer is the same whichever image the
// convention picks, so it is unambiguous. Two or more means the number is a
// convention artifact and must not be reported.
//
// PREDICTION, RECORDED BEFORE RUNNING: a substantial fraction of the 133
// feature-invisible candidates will FAIL the one-carrier filter, because
// swapping tails in a box of side 8 makes chains extended enough to link several
// images at once. I do not know whether any survivor clears 0.1. The assertion
// is that at least one does — i.e. the overturn stands on an unconfounded
// measurement. If none does, the overturn is withdrawn and this stays RED.

/// Number of the 27 nearest periodic images of chain `b` that carry appreciable
/// linking with chain `a`, and the largest such |Lk|.
///
/// Built from public nerve-topo API only: unwrap both chains, then shift one by
/// each box vector and integrate. Deliberately does NOT use the centroid
/// nearest-image convention, because the whole point is to find out whether that
/// convention's choice matters.
fn image_carriers(m: &Melt, a: usize, b: usize) -> (usize, f64) {
    let ua = nerve_topo::unwrap_chain(m, &m.chains[a]);
    let ub = nerve_topo::unwrap_chain(m, &m.chains[b]);
    let l = m.box_len;
    let mut carriers = 0usize;
    let mut biggest = 0.0f64;
    for kx in -1..=1i32 {
        for ky in -1..=1i32 {
            for kz in -1..=1i32 {
                let t = [kx as f64 * l, ky as f64 * l, kz as f64 * l];
                let shifted: Vec<Vec3> = ub
                    .iter()
                    .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
                    .collect();
                let lk = nerve_topo::linking_number_open(&ua, &shifted).abs();
                if lk > 0.1 {
                    carriers += 1;
                }
                if lk > biggest {
                    biggest = lk;
                }
            }
        }
    }
    (carriers, biggest)
}

#[test]
fn the_unambiguity_instrument_reports_one_carrier_for_a_clean_hopf_link() {
    // Calibration against known ground truth before trusting it on melt data:
    // an isolated Hopf link in a box far larger than the structure must show
    // exactly one carrier holding |Lk| = 1.
    let (a, b) = twisted_band(1, 1.0, 0.3, 120);
    let m = Melt::new(vec![Chain::new(a), Chain::new(b)], 60.0);
    let (n, biggest) = image_carriers(&m, 0, 1);
    println!("CALIBRATION: clean Hopf link -> {n} carrier(s), largest |Lk| = {biggest:.4}");
    assert_eq!(
        n, 1,
        "clean Hopf link should have exactly one carrying image"
    );
    assert!(
        (biggest - 1.0).abs() < 0.05,
        "carrier should hold |Lk| = 1, got {biggest:.4}"
    );
}

// ===== WITHDRAWN (round 5). =====================================================
// This test measures a real effect inside a regime that does not survive
// hardening. `dense_melt` carries no excluded volume; its jitter is 0.1 against
// a measured real-melt FENE relative sd of 3.401%; and the |Lk| bar reached
// through `image_carriers` is hardcoded at 0.1, not measured. On MD-equilibrated
// Kremer-Grest melts the move family has no length-feasible instances at all.
// The conclusion this test was written to support is WITHDRAWN. The withdrawal
// is asserted by `withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening`.
// Kept, not deleted: the in-regime measurement is real and reproducible, and the
// regime boundary is the finding.
// ===============================================================================
#[test]
fn an_unconfounded_length_preserving_double_bridge_changes_linking() {
    let bar = dense_noise_floor() / 10.0;
    let mut invisible = 0usize;
    let mut unambiguous = 0usize;
    let mut cleared = 0usize;
    let mut best = 0.0f64;
    let mut best_label = "none".to_string();
    let mut carrier_hist = std::collections::BTreeMap::<usize, usize>::new();

    for seed in 0..300u64 {
        let m = dense(seed);
        let nc = m.chains.len();
        let mb = max_bond(&m);
        let base_f = chain_features(&m);
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..DENSE_BEADS {
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 > mb || d2 > mb {
                        continue;
                    }
                    let c = crossover(&m, a, b, k);
                    let nb = new_bond_lengths(&m, &c);
                    if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                        continue;
                    }
                    let worst = feature_residuals(&base_f, &chain_features(&c))
                        .into_iter()
                        .fold(0.0f64, |x, y| x.max(y.1));
                    if worst >= bar {
                        continue;
                    }
                    invisible += 1;

                    let (nb_before, lk_before) = image_carriers(&m, a, b);
                    let (nb_after, lk_after) = image_carriers(&c, a, b);
                    *carrier_hist.entry(nb_before.max(nb_after)).or_default() += 1;
                    // Unambiguous means at most one image carries linking, in
                    // BOTH configurations. Otherwise the number depends on a
                    // convention and is not evidence of anything.
                    if nb_before > 1 || nb_after > 1 {
                        continue;
                    }
                    unambiguous += 1;
                    let before = if nb_before == 0 { 0.0 } else { lk_before };
                    let after = if nb_after == 0 { 0.0 } else { lk_after };
                    let dlk = (before - after).abs();
                    if dlk > 0.1 {
                        cleared += 1;
                    }
                    if dlk > best {
                        best = dlk;
                        best_label = format!(
                            "seed={seed} chains=({a},{b}) k={k} resid={worst:.3e} \
                             |Lk| {before:.4} -> {after:.4}"
                        );
                    }
                }
            }
        }
    }
    println!(
        "UNCONFOUNDED (round 3): bar={bar:.3e}\n  \
         feature-invisible length-preserving double bridges = {invisible}\n  \
         max carriers per candidate (histogram) = {carrier_hist:?}\n  \
         of those, periodic-image UNAMBIGUOUS = {unambiguous}\n  \
         of those, d|Lk| > 0.1 = {cleared}\n  \
         largest unconfounded d|Lk| = {best:.4}\n  \
         best = {best_label}"
    );
    assert!(
        invisible > 20,
        "too few feature-invisible candidates to conclude: {invisible}"
    );
    assert!(
        best > 0.1,
        "no unconfounded feature-invisible length-preserving double bridge changed \
         |Lk| by more than 0.1 (best {best:.4} over {unambiguous} unambiguous of \
         {invisible} invisible) — the overturn is WITHDRAWN and the verdict returns \
         to: not repairable by any move searched, with the sharper reason that the \
         topology measurement becomes convention-dependent exactly when the move \
         does something topologically interesting"
    );
}

// ======================================== branchcut instruments, applied
//
// DISCLOSURE, so nobody has to reverse-engineer it from timestamps: the four
// ported functions in lib.rs (collides, group_by_gap, group_entropy,
// tolerance_sweep, has_plateau, recovery_ceiling) were written BEFORE their
// tests, because they are transcriptions of a known-correct reference
// (`branchcut`, MIT, this user's repo) rather than new logic. They are therefore
// ported-then-calibrated, not RED-first, and I am not claiming a red-green
// transition for them. The tests that follow them ARE new claims with
// predictions recorded before running.

use nerve_orient::{collides, group_by_gap, has_plateau, recovery_ceiling, tolerance_sweep};

// ------------------------------------------------------ calibrate the port

#[test]
fn ported_group_by_gap_matches_branchcut_semantics() {
    // Descending groups, multiplicity counted, groups tile the line.
    let g = group_by_gap(&[1.0, 1.0, 1.0, 5.0, 5.0, 9.0], 0.5);
    assert_eq!(g.len(), 3, "expected 3 groups, got {g:?}");
    assert_eq!(g[0].mult, 1, "9.0 is alone");
    assert_eq!(g[1].mult, 2, "the two 5.0s group");
    assert_eq!(g[2].mult, 3, "the three 1.0s group");
    assert_eq!(g[2].low, f64::NEG_INFINITY, "last group must tile to -inf");
    // Chaining is against the LAST MEMBER, not the group mean: a slow drift
    // merges even though the span (1.2) exceeds tol (0.5).
    let drift = group_by_gap(&[0.0, 0.4, 0.8, 1.2], 0.5);
    assert_eq!(
        drift.len(),
        1,
        "branchcut chains on the last member, so a 0.4-step drift is one group: {drift:?}"
    );
}

#[test]
fn ported_has_plateau_rejects_both_degenerate_ends() {
    // A smooth slide with no gap structure: no plateau.
    let slide = tolerance_sweep(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let (flat, _) = has_plateau(&slide, 3);
    assert!(!flat, "an evenly spaced sequence must not report a plateau");
    // Genuine two-cluster structure with a wide gap: plateau at 2 groups.
    let clustered: Vec<f64> = vec![0.0, 0.01, 0.02, 10.0, 10.01, 10.02];
    let sweep = tolerance_sweep(&clustered);
    let (flat2, count) = has_plateau(&sweep, 3);
    println!("PORT CALIBRATION clustered sweep: plateau={flat2} at {count} groups");
    assert!(flat2, "two well-separated clusters must plateau: {sweep:?}");
    assert_eq!(count, 2, "plateau should sit at 2 groups, got {count}");
}

#[test]
fn ported_collides_is_exact_to_atol_and_one_sided() {
    let images = vec![
        vec![1.0, 2.0],
        vec![1.0, 2.0 + 1e-12],
        vec![5.0, 5.0],
        vec![1.0, 2.5],
    ];
    let c = collides(&images, 1e-9);
    assert_eq!(c, vec![(0, 1)], "only 0 and 1 agree within 1e-9: {c:?}");
    // rtol is zero in the original, so a large-magnitude pair with the same
    // relative error must NOT merge.
    let big = vec![vec![1e6], vec![1e6 + 1.0]];
    assert!(
        collides(&big, 1e-9).is_empty(),
        "rtol must be zero: large values with absolute gap 1.0 must not merge"
    );
    assert!(
        collides(&[vec![1.0]], 1e-9).is_empty(),
        "a single point cannot collide"
    );
    assert_eq!(recovery_ceiling(4), 0.25, "Theorem 2: k=4 gives 1/4");
}

// ------------------------------- the bar, justified by a plateau not a choice

/// Every length-preserving double bridge's worst-feature residual, pooled.
fn length_preserving_residuals(seeds: u64) -> Vec<f64> {
    let mut out = Vec::new();
    for seed in 0..seeds {
        let m = dense(seed);
        let nc = m.chains.len();
        let mb = max_bond(&m);
        let base_f = chain_features(&m);
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..DENSE_BEADS {
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 > mb || d2 > mb {
                        continue;
                    }
                    let c = crossover(&m, a, b, k);
                    let nb = new_bond_lengths(&m, &c);
                    if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                        continue;
                    }
                    out.push(
                        feature_residuals(&base_f, &chain_features(&c))
                            .into_iter()
                            .fold(0.0f64, |x, y| x.max(y.1)),
                    );
                }
            }
        }
    }
    out
}

/// PREDICTION, RECORDED BEFORE RUNNING: the residual population has genuine gap
/// structure, so a plateau exists and the invisibility bar can be justified by it
/// rather than picked. If there is no plateau then my `noise_floor / 10` bar is
/// as arbitrary as the `1e-9` that got struck, and this stays RED.
/// KEPT VERBATIM AND FAILING. Prediction falsified, and it cost me a headline
/// number. Group count slides 386, 384, 383, 375, 347, 274, 171, 52, 16, 6, 2, 1,
/// 1 across the 13-point sweep with no flat stretch — branchcut's exact signature
/// for "no structure". So the residual population has NO natural threshold, my
/// `noise_floor / 10` bar is a chosen number in the same class as the struck
/// `1e-9`, and any COUNT of feature-invisible reconnections is a function of bar
/// placement rather than a fact about the melt. The overturn is therefore restated
/// threshold-free in `the_overturn_survives_without_a_chosen_invisibility_bar`.
/// Ignored to keep the suite green; remove to watch it fail.
#[ignore = "PREMISE FALSE: no plateau, group count slides 386->1 with no flat stretch; the invisibility bar is chosen, not justified"]
#[test]
fn the_feature_invisibility_bar_sits_on_a_tolerance_plateau() {
    let residuals = length_preserving_residuals(300);
    assert!(
        residuals.len() > 100,
        "population too small to sweep: {}",
        residuals.len()
    );
    let sweep = tolerance_sweep(&residuals);
    println!(
        "TOLERANCE SWEEP over {} length-preserving residuals:",
        residuals.len()
    );
    for r in &sweep {
        println!(
            "  tol={:.3e} groups={:>4} max_mult={:>4} entropy={:.4}",
            r.tol, r.n_groups, r.max_multiplicity, r.entropy
        );
    }
    let (flat, count) = has_plateau(&sweep, 3);
    let nf = dense_noise_floor();
    println!(
        "PLATEAU: {flat} at {count} groups; my bar was noise_floor/10 = {:.3e} \
         (noise floor {nf:.3e})",
        nf / 10.0
    );
    assert!(
        flat,
        "no plateau in the residual population — the invisibility bar is a chosen \
         number, not a justified one, and every count derived from it inherits that"
    );
}

// ------------------------------ collides as the taxonomy-free detector

/// Componentwise ratio to the reference, so `collides`'s absolute `atol` reads as
/// a relative tolerance directly comparable to `feature_residuals`.
fn relative_feature_vector(base: &ChainFeatures, f: &ChainFeatures) -> Vec<f64> {
    let g = |x: f64, y: f64| if x == 0.0 { y } else { y / x };
    let mut v = vec![
        g(base.rg2, f.rg2),
        g(base.end_to_end2, f.end_to_end2),
        g(base.contour_len, f.contour_len),
        g(base.bead_density, f.bead_density),
        g(base.max_bond, f.max_bond),
    ];
    for (x, y) in base.msid.iter().zip(&f.msid) {
        v.push(g(*x, *y));
    }
    v
}

// ===== WITHDRAWN (round 5). =====================================================
// This test measures a real effect inside a regime that does not survive
// hardening. `dense_melt` carries no excluded volume; its jitter is 0.1 against
// a measured real-melt FENE relative sd of 3.401%; and the |Lk| bar reached
// through `image_carriers` is hardcoded at 0.1, not measured. On MD-equilibrated
// Kremer-Grest melts the move family has no length-feasible instances at all.
// The conclusion this test was written to support is WITHDRAWN. The withdrawal
// is asserted by `withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening`.
// Kept, not deleted: the in-regime measurement is real and reproducible, and the
// regime boundary is the finding.
// ===============================================================================
/// PREDICTION, RECORDED BEFORE RUNNING: `collides` recovers the same conclusion
/// as my hand-built move taxonomy — merged pairs exist whose |Lk| differs by more
/// than 0.1 — and additionally finds BRIDGE-vs-BRIDGE merges that my round-2 and
/// round-3 searches never looked at, because both only ever compared a candidate
/// against its original.
#[test]
fn collides_detects_invisible_reconnections_without_a_move_taxonomy() {
    let atol = dense_noise_floor() / 10.0;
    let mut merged_pairs = 0usize;
    let mut bridge_vs_bridge = 0usize;
    let mut best_span = 0.0f64;
    let mut best_label = "none".to_string();
    let mut worst_group = 1usize;

    for seed in 0..300u64 {
        let m = dense(seed);
        let nc = m.chains.len();
        let mb = max_bond(&m);
        let base_f = chain_features(&m);
        // Population for this bead set: the original, then every
        // length-preserving double bridge on it.
        let mut pop: Vec<Vec<f64>> = vec![relative_feature_vector(&base_f, &base_f)];
        let mut lks: Vec<f64> = vec![f64::NAN];
        let mut labels: Vec<String> = vec!["original".to_string()];
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..DENSE_BEADS {
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 > mb || d2 > mb {
                        continue;
                    }
                    let c = crossover(&m, a, b, k);
                    let nb = new_bond_lengths(&m, &c);
                    if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                        continue;
                    }
                    pop.push(relative_feature_vector(&base_f, &chain_features(&c)));
                    let (carriers, biggest) = image_carriers(&c, a, b);
                    lks.push(if carriers == 1 { biggest } else { f64::NAN });
                    labels.push(format!("bridge({a},{b})k={k}"));
                }
            }
        }
        if pop.len() < 2 {
            continue;
        }
        // Fill in the original's |Lk| per bridged pair lazily: use the first
        // bridge's chain pair as the reference pair for the original.
        let hits = collides(&pop, atol);
        merged_pairs += hits.len();
        for &(i, j) in &hits {
            if i != 0 && j != 0 {
                bridge_vs_bridge += 1;
            }
            let (li, lj) = (lks[i], lks[j]);
            if li.is_nan() || lj.is_nan() {
                continue;
            }
            let span = (li - lj).abs();
            if span > best_span {
                best_span = span;
                best_label = format!("seed={seed} {} vs {}", labels[i], labels[j]);
            }
        }
        // Largest set of mutually-merged members, as a lower bound on group size.
        let mut counts = vec![1usize; pop.len()];
        for &(i, j) in &hits {
            counts[i] += 1;
            counts[j] += 1;
        }
        worst_group = worst_group.max(*counts.iter().max().unwrap());
    }

    println!(
        "COLLIDES (atol={atol:.3e}): {merged_pairs} merged pairs over 300 bead sets, \
         of which {bridge_vs_bridge} are bridge-vs-bridge (never compared before)\n  \
         largest |Lk| span inside a merged pair = {best_span:.4} ({best_label})\n  \
         largest merged group size >= {worst_group}, so Theorem 2 recovery ceiling \
         <= {:.3}",
        recovery_ceiling(worst_group)
    );
    assert!(
        merged_pairs > 0,
        "collides found no merges at the plateau tolerance; the descriptor separates \
         every configuration and the blindness claim fails on this population"
    );
    assert!(
        best_span > 0.1,
        "largest |Lk| span inside a feature-merged pair was only {best_span:.4} \
         ({best_label}) — collides does not reproduce the overturn"
    );
}

/// Theorem 1's certified error floor is VACUOUS for `|Lk|` on this population and
/// is deliberately not computed. Recorded as an executable note so the omission is
/// auditable rather than silent.
///
/// `branchcut`'s precondition: the floor certifies nothing if the reference map is
/// itself many-to-one, and `select_by_floor` raises rather than optimise an
/// inverted score. Here many genuinely distinct reconnections share the same
/// `|Lk|` — most obviously the large family with `|Lk| ~ 0` — so `|Lk|` is not
/// injective over reconnections and `err >= n - m` would count non-errors.
///
/// What IS valid is the Theorem 2 direction: inside a collision group the
/// descriptor emits one value while `|Lk|` spans a range, so any downstream head
/// must incur at least half that span on the group.
#[test]
fn theorem_1_error_floor_is_vacuous_for_abs_lk_on_this_population() {
    // Demonstrate the non-injectivity that makes it vacuous.
    let mut zeros = 0usize;
    let mut total = 0usize;
    for seed in 0..60u64 {
        let m = dense(seed);
        for a in 0..m.chains.len() {
            for b in (a + 1)..m.chains.len() {
                let (carriers, biggest) = image_carriers(&m, a, b);
                total += 1;
                if carriers == 0 || biggest < 0.1 {
                    zeros += 1;
                }
            }
        }
    }
    println!(
        "THEOREM 1 VACUITY: {zeros}/{total} distinct chain pairs have |Lk| < 0.1, so \
         |Lk| is heavily many-to-one over this population and a certified error \
         floor against it would count non-errors. Floor deliberately not computed."
    );
    assert!(
        zeros * 2 > total,
        "|Lk| turned out to be nearly injective ({zeros}/{total} near zero); the \
         vacuity argument would not hold and Theorem 1 should be computed after all"
    );
}

// ============================== the overturn, restated without a chosen bar
//
// `the_feature_invisibility_bar_sits_on_a_tolerance_plateau` is RED and stays
// RED. The residual population slides 386 -> 384 -> 383 -> 375 -> 347 -> 274 ->
// 171 -> 52 -> 16 -> 6 -> 2 -> 1 -> 1 groups with no flat stretch anywhere, which
// is branchcut's signature for "no structure". Consequence I have to accept: any
// COUNT of "feature-invisible reconnections" is a function of where I put the
// bar, so "133" is not a fact about the melt and must not be reported as one.
//
// What survives is threshold-free, and it is the claim that actually matters
// because the overturn was only ever an EXISTENCE claim:
//
//   * the reference is the MEASURED different-seed noise floor, not a chosen
//     number — how much the six features move when the melt is merely reseeded;
//   * for each candidate, report residual as a RATIO to that floor;
//   * report max d|Lk| at a range of ratio cutoffs, so the whole dependence is
//     visible and no single cutoff is smuggled in.
//
// PREDICTION, RECORDED BEFORE RUNNING: even at the strictest cutoff — reconnection
// perturbs the features less than one hundredth of what reseeding does — some
// candidate still changes |Lk| by more than 0.1. If that fails the curve is still
// reported and this stays RED under this name.

// ===== WITHDRAWN (round 5). =====================================================
// This test measures a real effect inside a regime that does not survive
// hardening. `dense_melt` carries no excluded volume; its jitter is 0.1 against
// a measured real-melt FENE relative sd of 3.401%; and the |Lk| bar reached
// through `image_carriers` is hardcoded at 0.1, not measured. On MD-equilibrated
// Kremer-Grest melts the move family has no length-feasible instances at all.
// The conclusion this test was written to support is WITHDRAWN. The withdrawal
// is asserted by `withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening`.
// Kept, not deleted: the in-regime measurement is real and reproducible, and the
// regime boundary is the finding.
// ===============================================================================
#[test]
fn the_overturn_survives_without_a_chosen_invisibility_bar() {
    let nf = dense_noise_floor();
    // (residual / noise_floor, d|Lk|) for every length-preserving double bridge
    // whose topology measurement is periodic-image unambiguous in both states.
    let mut pts: Vec<(f64, f64, String)> = Vec::new();
    for seed in 0..300u64 {
        let m = dense(seed);
        let nc = m.chains.len();
        let mb = max_bond(&m);
        let base_f = chain_features(&m);
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..DENSE_BEADS {
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 > mb || d2 > mb {
                        continue;
                    }
                    let c = crossover(&m, a, b, k);
                    let nb = new_bond_lengths(&m, &c);
                    if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(&m) {
                        continue;
                    }
                    let (cb, lb) = image_carriers(&m, a, b);
                    let (ca, la) = image_carriers(&c, a, b);
                    if cb > 1 || ca > 1 {
                        continue;
                    }
                    let before = if cb == 0 { 0.0 } else { lb };
                    let after = if ca == 0 { 0.0 } else { la };
                    let resid = feature_residuals(&base_f, &chain_features(&c))
                        .into_iter()
                        .fold(0.0f64, |x, y| x.max(y.1));
                    pts.push((
                        resid / nf,
                        (before - after).abs(),
                        format!("seed={seed} ({a},{b}) k={k} |Lk| {before:.3}->{after:.3}"),
                    ));
                }
            }
        }
    }
    println!(
        "THRESHOLD-FREE (measured noise floor {nf:.3e}; {} unambiguous \
         length-preserving double bridges):",
        pts.len()
    );
    let mut strictest_hit = None;
    for cutoff in [1.0f64, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01] {
        let kept: Vec<&(f64, f64, String)> = pts.iter().filter(|p| p.0 < cutoff).collect();
        let best = kept
            .iter()
            .fold(None::<&&(f64, f64, String)>, |acc, p| match acc {
                Some(q) if q.1 >= p.1 => Some(q),
                _ => Some(p),
            });
        match best {
            Some(b) => {
                println!(
                    "  residual < {cutoff:>5} x noise floor: n={:<4} max d|Lk| = {:.4}  [{}]",
                    kept.len(),
                    b.1,
                    b.2
                );
                if b.1 > 0.1 {
                    strictest_hit = Some((cutoff, b.1, b.2.clone()));
                }
            }
            None => println!("  residual < {cutoff:>5} x noise floor: n=0"),
        }
    }
    assert!(!pts.is_empty(), "no unambiguous candidates at all");
    let (cutoff, dlk, label) = strictest_hit.expect(
        "no candidate at any cutoff changed |Lk| by more than 0.1 — the overturn is \
         withdrawn",
    );
    println!("STRICTEST SURVIVING CUTOFF: {cutoff} x noise floor, d|Lk| = {dlk:.4} [{label}]");
    assert!(
        cutoff <= 0.01,
        "the overturn only survives down to a cutoff of {cutoff} x the noise floor; \
         it does not reach the 0.01 x cutoff I predicted, so the effect depends more \
         on bar placement than I claimed"
    );
}

// ================================================================== ROUND 4
// Attacking my own overturn on the three axes I named as its weaknesses.
//
// Predictions are registered on the board, not only here. Restated for the
// in-code record:
//   P1 overturn survives realistic fluctuation but signal-to-noise degrades
//      monotonically with jitter.
//   P2 real FENE bond relative sd is 2-5 percent, so my j=0.1 was ~2x generous.
//   P3 bead relocation raises feasibility sharply but destroys the zero ACSF
//      residual.
//   P4 phenomenon survives at larger scale; O(C^2 M^2) cost binds and forces a cap.
//   P6 excluded volume sharply reduces the 0.752 percent length-feasibility but
//      does not eliminate it; the overturn weakens but survives.

/// `dense_melt` plus hard-core exclusion. `min_sep = 0.0` reproduces the round-3
/// fixture exactly, so the excluded-volume axis is a strict extension.
///
/// Growth is by rejection with a fixed budget, matching
/// `nerve_melt::try_melt`'s approach; returns `None` when the budget is exhausted
/// so a density ceiling is reported rather than silently violated.
fn dense_melt_ev(
    seed: u64,
    n_chains: usize,
    n_beads: usize,
    box_len: f64,
    jitter: f64,
    min_sep: f64,
) -> Option<Melt> {
    const MAX_DIRS: usize = 200;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let probe = Melt::new(vec![], box_len);
    let w = |p: [f64; 3]| -> Vec3 {
        [
            p[0].rem_euclid(box_len),
            p[1].rem_euclid(box_len),
            p[2].rem_euclid(box_len),
        ]
    };
    let mut placed: Vec<Vec3> = Vec::with_capacity(n_chains * n_beads);
    let mut chains = Vec::with_capacity(n_chains);
    for _ in 0..n_chains {
        let mut beads: Vec<Vec3> = Vec::with_capacity(n_beads);
        // BUG FIXED: the first bead of each chain was pushed with no clearance
        // check at all, so chain starts landed on top of existing beads and the
        // melt violated min_sep while claiming to honour it. Caught by
        // cross-validating achieved separation against nerve_melt::try_melt
        // (mine 0.3833 vs hers 0.8053 at min_sep=0.8), not by any test of mine
        // that only inspected my own generator.
        let mut p = [0.0; 3];
        let mut seeded = false;
        for _ in 0..MAX_DIRS {
            let q = w([
                rng.gen_range(0.0..box_len),
                rng.gen_range(0.0..box_len),
                rng.gen_range(0.0..box_len),
            ]);
            if min_sep <= 0.0
                || placed
                    .iter()
                    .all(|o| probe.min_image_dist(*o, q) >= min_sep)
            {
                p = q;
                seeded = true;
                break;
            }
        }
        if !seeded {
            return None;
        }
        beads.push(p);
        placed.push(p);
        for _ in 1..n_beads {
            let mut ok = None;
            for _ in 0..MAX_DIRS {
                let z: f64 = rng.gen_range(-1.0..1.0);
                let phi: f64 = rng.gen_range(0.0..TAU);
                let r = (1.0 - z * z).sqrt();
                let len = 1.0 + jitter * rng.gen_range(-1.0..1.0);
                let q = w([
                    p[0] + len * r * phi.cos(),
                    p[1] + len * r * phi.sin(),
                    p[2] + len * z,
                ]);
                // Exclude against everything already placed except the bonded
                // predecessor, which is by construction one bond away.
                let clear = min_sep <= 0.0
                    || placed
                        .iter()
                        .rev()
                        .skip(1)
                        .all(|o| probe.min_image_dist(*o, q) >= min_sep);
                if clear {
                    ok = Some(q);
                    break;
                }
            }
            p = ok?;
            beads.push(p);
            placed.push(p);
        }
        chains.push(Chain::new(beads));
    }
    Some(Melt::new(chains, box_len))
}

/// Every length-preserving double bridge on one melt, as
/// `(residual, d|Lk|, ambiguous)`.
fn bridges_on(m: &Melt) -> Vec<(f64, f64, bool)> {
    let nc = m.chains.len();
    let n = m.chains[0].beads.len();
    let mb = max_bond(m);
    let base_f = chain_features(m);
    let mut out = Vec::new();
    for a in 0..nc {
        for b in (a + 1)..nc {
            for k in 1..n {
                let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                if d1 > mb || d2 > mb {
                    continue;
                }
                let c = crossover(m, a, b, k);
                let nb = new_bond_lengths(m, &c);
                if nb.len() != 2 || nb.iter().any(|l| *l > mb) {
                    continue;
                }
                if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(m) {
                    continue;
                }
                let resid = feature_residuals(&base_f, &chain_features(&c))
                    .into_iter()
                    .fold(0.0f64, |x, y| x.max(y.1));
                let (cb, lb) = image_carriers(m, a, b);
                let (ca, la) = image_carriers(&c, a, b);
                let ambiguous = cb > 1 || ca > 1;
                let before = if cb == 0 { 0.0 } else { lb };
                let after = if ca == 0 { 0.0 } else { la };
                out.push((resid, (before - after).abs(), ambiguous));
            }
        }
    }
    out
}

/// Count of `(a,b,k)` triples whose two bridge bonds are both within `max_bond`,
/// and the total examined. The 0.752 percent figure, parameterised.
fn feasibility(m: &Melt) -> (usize, usize) {
    let nc = m.chains.len();
    let n = m.chains[0].beads.len();
    let mb = max_bond(m);
    let (mut ok, mut total) = (0usize, 0usize);
    for a in 0..nc {
        for b in (a + 1)..nc {
            for k in 1..n {
                total += 1;
                let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                if d1 <= mb && d2 <= mb {
                    ok += 1;
                }
            }
        }
    }
    (ok, total)
}

fn noise_floor_of(a: &Melt, b: &Melt) -> f64 {
    worst_feature_residual(a, b).1
}

// ---------------------------------------------- precondition: my generator

#[test]
fn my_excluded_volume_generator_agrees_with_nerve_melt_on_achieved_separation() {
    // Cross-validation against Cameron's generator. Exact agreement is
    // impossible (different RNG streams), so compare the properties that matter:
    // achieved minimum pair separation and bond length.
    let min_sep = 0.8;
    let mine = dense_melt_ev(7, 8, 20, 8.0, 0.0, min_sep).expect("mine failed to place");
    let spec = nerve_melt::MeltSpec {
        n_chains: 8,
        n_beads: 20,
        density: 160.0 / 512.0,
        bond: 1.0,
        seed: 7,
        min_sep,
    };
    let hers = nerve_melt::try_melt(&spec).expect("nerve-melt failed to place");
    assert!(
        (hers.box_len - 8.0).abs() < 1e-9,
        "box mismatch: {} vs 8.0",
        hers.box_len
    );
    // Achieved separation, measured through her instrument on both melts.
    let sep = |m: &Melt| {
        let mut best = f64::INFINITY;
        for i in 0..m.chains.len() {
            for j in (i + 1)..m.chains.len() {
                best = best.min(nerve_melt::min_pair_distance(m, i, j));
            }
        }
        best
    };
    let (sm, sh) = (sep(&mine), sep(&hers));
    println!(
        "GENERATOR CROSS-CHECK at min_sep={min_sep}: mine inter-chain min sep={sm:.4} \
         max_bond={:.4}; nerve-melt min sep={sh:.4} max_bond={:.4}",
        max_bond(&mine),
        max_bond(&hers)
    );
    assert!(sm >= min_sep - 1e-9, "my generator violated min_sep: {sm}");
    assert!(sh >= min_sep - 1e-9, "nerve-melt violated min_sep: {sh}");
    let ideal = dense_melt_ev(7, 8, 20, 8.0, 0.0, 0.0).expect("ideal failed");
    println!(
        "  ideal (min_sep=0) inter-chain min sep = {:.4}",
        sep(&ideal)
    );
    assert!(
        sep(&ideal) < min_sep,
        "the ideal fixture should allow closer approaches than {min_sep}; got {:.4}",
        sep(&ideal)
    );
}

// ------------------------------------------- ATTACK: excluded volume (P6)

#[test]
fn excluded_volume_reduces_length_feasibility() {
    let mut rows = Vec::new();
    for min_sep in [0.0f64, 0.5, 0.7, 0.8, 0.9] {
        let (mut ok, mut total, mut melts) = (0usize, 0usize, 0usize);
        for seed in 0..40u64 {
            if let Some(m) = dense_melt_ev(seed, 8, 20, 8.0, 0.1, min_sep) {
                let (o, t) = feasibility(&m);
                ok += o;
                total += t;
                melts += 1;
            }
        }
        let pct = if total == 0 {
            0.0
        } else {
            100.0 * ok as f64 / total as f64
        };
        println!(
            "EV FEASIBILITY min_sep={min_sep:.1}: {ok}/{total} = {pct:.3}% \
             ({melts}/40 melts placed)"
        );
        rows.push((min_sep, pct, melts));
    }
    let ideal = rows[0].1;
    let physical = rows.iter().find(|r| r.0 == 0.8).unwrap();
    println!(
        "EV VERDICT: ideal {ideal:.3}% -> min_sep=0.8 {:.3}% (ratio {:.2}x)",
        physical.1,
        if physical.1 > 0.0 {
            ideal / physical.1
        } else {
            f64::INFINITY
        }
    );
    assert!(physical.2 > 0, "no melt placed at min_sep=0.8; ceiling hit");
    assert!(
        physical.1 > 0.0,
        "excluded volume eliminated length-preserving bridges entirely \
         ({ideal:.3}% -> 0%) — the overturn is an artifact of the ideal-chain \
         fixture and is WITHDRAWN"
    );
}

// -------------------------------------- ATTACK: bond fluctuation sweep (P1)

#[test]
fn bond_fluctuation_sweep_reports_the_working_window() {
    println!(
        "FLUCTUATION SWEEP (8 chains x 20 beads, box 8; 60 seeds per level)\n  \
         jitter  bond_sd%  max_bond_nf  feat_nf   feasible%  n<1.0nf  maxdLk<1.0nf  \
         n<0.1nf  maxdLk<0.1nf  plateau"
    );
    for &min_sep in &[0.0f64, 0.8] {
        println!("  --- min_sep = {min_sep} ---");
        for &j in &[0.0f64, 0.02, 0.05, 0.10, 0.20, 0.30] {
            let melts: Vec<Melt> = (0..60u64)
                .filter_map(|s| dense_melt_ev(s, 8, 20, 8.0, j, min_sep))
                .collect();
            if melts.len() < 2 {
                println!("  {j:6.2}  (only {} melts placed)", melts.len());
                continue;
            }
            let mb_nf = rel(max_bond(&melts[0]), max_bond(&melts[1]));
            let feat_nf = noise_floor_of(&melts[0], &melts[1]);
            let (mut ok, mut tot) = (0usize, 0usize);
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for m in &melts {
                let (o, t) = feasibility(m);
                ok += o;
                tot += t;
                for (resid, dlk, amb) in bridges_on(m) {
                    if !amb {
                        pts.push((resid / feat_nf, dlk));
                    }
                }
            }
            let at = |cut: f64| -> (usize, f64) {
                let k: Vec<&(f64, f64)> = pts.iter().filter(|p| p.0 < cut).collect();
                (k.len(), k.iter().fold(0.0f64, |x, p| x.max(p.1)))
            };
            let (n1, d1) = at(1.0);
            let (n01, d01) = at(0.1);
            let plateau = if pts.len() > 10 {
                let resids: Vec<f64> = pts.iter().map(|p| p.0).collect();
                let (flat, cnt) = has_plateau(&tolerance_sweep(&resids), 3);
                format!("{flat}@{cnt}")
            } else {
                "n/a".to_string()
            };
            println!(
                "  {j:6.2}  {:7.2}  {mb_nf:11.3e}  {feat_nf:7.3e}  {:8.3}  {n1:7}  \
                 {d1:12.4}  {n01:7}  {d01:12.4}  {plateau}",
                100.0 * j / 3.0f64.sqrt(),
                100.0 * ok as f64 / tot as f64
            );
        }
    }
    // The sweep is the deliverable; the assertion only guards against the whole
    // thing being empty.
    let m = dense_melt_ev(0, 8, 20, 8.0, 0.1, 0.0).expect("baseline melt failed");
    assert!(!bridges_on(&m).is_empty() || feasibility(&m).0 == 0);
}

// -------------------------------------------- ATTACK: bead relocation (P3)

#[test]
fn bead_relocation_raises_feasibility_but_costs_the_zero_acsf_residual() {
    let (nc, nb, bx, j) = (8usize, 20usize, 8.0f64, 0.1f64);
    let bond_sd = j / 3.0f64.sqrt();
    println!(
        "RELOCATION SWEEP (bond sd = {bond_sd:.4}; delta in units of bond sd)\n  \
         delta/sd  feasible%  acsf_resid_max  acsf_noise_floor  feat_resid_max  maxdLk"
    );
    let acsf_nf = {
        let a = dense_melt_ev(9001, nc, nb, bx, j, 0.0).unwrap();
        let b = dense_melt_ev(9002, nc, nb, bx, j, 0.0).unwrap();
        descriptor_residual(&a, &b, 4.0)
    };
    for &mult in &[0.0f64, 0.5, 1.0, 2.0] {
        let delta = mult * bond_sd;
        let (mut ok, mut tot) = (0usize, 0usize);
        let (mut acsf_max, mut feat_max, mut dlk_max) = (0.0f64, 0.0f64, 0.0f64);
        for seed in 0..25u64 {
            let m = match dense_melt_ev(seed, nc, nb, bx, j, 0.0) {
                Some(m) => m,
                None => continue,
            };
            let mb = max_bond(&m);
            let base_f = chain_features(&m);
            for a in 0..nc {
                for b in (a + 1)..nc {
                    for k in 1..nb {
                        tot += 1;
                        // Relocate the two beads that will carry the new bonds,
                        // each toward its new partner, capped at delta.
                        let m2 = relocate_bead(&m, b, k, m.chains[a].beads[k - 1], delta);
                        let m2 = relocate_bead(&m2, a, k, m2.chains[b].beads[k - 1], delta);
                        let d1 =
                            m2.min_image_dist(m2.chains[a].beads[k - 1], m2.chains[b].beads[k]);
                        let d2 =
                            m2.min_image_dist(m2.chains[b].beads[k - 1], m2.chains[a].beads[k]);
                        if d1 > mb || d2 > mb {
                            continue;
                        }
                        let c = crossover(&m2, a, b, k);
                        if max_bond(&c) >= c.box_len / 2.0 {
                            continue;
                        }
                        ok += 1;
                        acsf_max = acsf_max.max(descriptor_residual(&m, &c, 4.0));
                        feat_max = feat_max.max(
                            feature_residuals(&base_f, &chain_features(&c))
                                .into_iter()
                                .fold(0.0f64, |x, y| x.max(y.1)),
                        );
                        let (cb, lb) = image_carriers(&m, a, b);
                        let (ca, la) = image_carriers(&c, a, b);
                        if cb <= 1 && ca <= 1 {
                            let before = if cb == 0 { 0.0 } else { lb };
                            let after = if ca == 0 { 0.0 } else { la };
                            dlk_max = dlk_max.max((before - after).abs());
                        }
                    }
                }
            }
        }
        println!(
            "  {mult:8.1}  {:9.3}  {acsf_max:14.3e}  {acsf_nf:16.3e}  {feat_max:14.3e}  \
             {dlk_max:6.4}",
            100.0 * ok as f64 / tot as f64
        );
    }
    assert!(
        acsf_nf > 0.0,
        "ACSF noise floor is zero; nothing to compare to"
    );
}

// --------------------------------------------------- BLOCKED: real melt (P2)

/// The decisive measurement for P2, written so it runs the instant the archive
/// lands. Zenodo 7319837 is NOT on disk this round — a nurse swept three roots
/// for nine filename patterns and found nothing — so this is ignored rather than
/// faked. Set NERVE_LAMMPS to a decompressed LAMMPS data file to run it.
#[test]
#[ignore = "BLOCKED: Zenodo 7319837 not on disk; set NERVE_LAMMPS=<path> to run"]
fn real_melt_bond_distribution_versus_my_jitter_parameter() {
    let path = std::env::var("NERVE_LAMMPS").expect("set NERVE_LAMMPS to a LAMMPS data file");
    let m = nerve_melt::from_lammps_path(std::path::Path::new(&path)).expect("parse failed");
    let mut lens: Vec<f64> = Vec::new();
    for c in &m.chains {
        for w in c.beads.windows(2) {
            let d = m.min_image(w[0], w[1]);
            lens.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    let n = lens.len() as f64;
    let mean = lens.iter().sum::<f64>() / n;
    let sd = (lens.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n).sqrt();
    let (lo, hi) = (
        lens.iter().cloned().fold(f64::INFINITY, f64::min),
        lens.iter().cloned().fold(0.0, f64::max),
    );
    // Uniform jitter j has sd = j/sqrt(3), so the equivalent j is sd*sqrt(3).
    let j_equiv = sd / mean * 3.0f64.sqrt();
    println!(
        "REAL MELT {path}\n  {} chains, {} beads, box {:.3}\n  \
         bond mean={mean:.5} sd={sd:.5} rel_sd={:.3}% min={lo:.5} max={hi:.5}\n  \
         EQUIVALENT UNIFORM JITTER j = {j_equiv:.4}  (my round-3 choice was 0.10)",
        m.chains.len(),
        m.n_beads(),
        m.box_len,
        100.0 * sd / mean
    );
    assert!(mean > 0.0 && sd > 0.0);
}

// ============================================ the overturn under hardened filters
//
// Three biases found by adversarial review, all fixed here, all of which were
// working in my favour:
//
//  1. EXCLUDED VOLUME. The round-3 fixture was an ideal walk; beads could
//     overlap, so bridge partners at bond distance were free. Now min_sep = 0.8.
//  2. LENGTH-PRESERVATION REFERENCE. `new_bonds <= max_bond(melt)` compares
//     against the melt-wide maximum, which for 152 bonds of jitter 0.1 sits
//     within ~0.0013 of the ceiling 1.1 — the extreme upper tail — and gets more
//     permissive as the melt grows. Replaced by a LOCAL, scale-invariant
//     reference: each new bond must be no longer than the longer of the two
//     bonds it replaces.
//  3. |Lk| SIGNIFICANCE. The 0.1 threshold was chosen by eye against "1.0 for a
//     Hopf link". Open-chain Gauss linking is real-valued and its magnitude grows
//     with contour length, so a fixed 0.1 silently discards more candidates as N
//     grows. Now MEASURED: the 95th percentile of |Lk| over non-bridged chain
//     pairs at the same architecture.
//
// P7, registered on the board before running: the overturn survives, with
// single-digit to low-double-digit counts rather than 133 and a maximum d|Lk|
// below 2.7260. If it survives on fewer than 3 candidates I report it as an
// existence proof on a knife edge, not a robust phenomenon.

/// 95th-percentile |Lk| over non-bridged chain pairs — the measured scale for
/// "this linking number is appreciable", replacing the hardcoded 0.1.
fn lk_significance(melts: &[Melt]) -> f64 {
    let mut v: Vec<f64> = Vec::new();
    for m in melts {
        for a in 0..m.chains.len() {
            for b in (a + 1)..m.chains.len() {
                v.push(
                    nerve_topo::linking_number_open(&m.chains[a].beads, &m.chains[b].beads).abs(),
                );
            }
        }
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    if v.is_empty() {
        return 0.1;
    }
    v[(0.95 * (v.len() - 1) as f64).round() as usize]
}

/// Carrier count using a MEASURED significance threshold rather than 0.1.
fn image_carriers_at(m: &Melt, a: usize, b: usize, sig: f64) -> (usize, f64) {
    let ua = nerve_topo::unwrap_chain(m, &m.chains[a]);
    let ub = nerve_topo::unwrap_chain(m, &m.chains[b]);
    let l = m.box_len;
    let (mut carriers, mut biggest) = (0usize, 0.0f64);
    for kx in -1..=1i32 {
        for ky in -1..=1i32 {
            for kz in -1..=1i32 {
                let t = [kx as f64 * l, ky as f64 * l, kz as f64 * l];
                let sh: Vec<Vec3> = ub
                    .iter()
                    .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
                    .collect();
                let lk = nerve_topo::linking_number_open(&ua, &sh).abs();
                if lk > sig {
                    carriers += 1;
                }
                biggest = biggest.max(lk);
            }
        }
    }
    (carriers, biggest)
}

/// KEPT VERBATIM AND FAILING. P7 falsified: I predicted the overturn would
/// survive hardening with single-digit counts. It survives with ZERO. Under
/// physical excluded volume (min_sep 0.8), a local length-preservation reference,
/// and a MEASURED |Lk| significance of 1.2327 (95th percentile over non-bridged
/// pairs, versus the 0.1 I had hardcoded — 12x too lenient), no candidate is both
/// below the feature noise floor and above |Lk| significance over 159,600 triples
/// in 300 melts. Largest raw unfloored effect is 0.5397, ratio 0.438 to
/// significance: real, but sub-significance. The round-3 overturn is WITHDRAWN.
/// Ignored to keep the suite green; remove to watch it fail.
#[ignore = "P7 FALSE: overturn does NOT survive hardening — 0 candidates clear both bars; raw max d|Lk| 0.5397 vs measured significance 1.2327"]
#[test]
fn the_overturn_under_hardened_filters() {
    let (nc, nb, bx, j, min_sep) = (8usize, 20usize, 8.0f64, 0.1f64, 0.8f64);
    let melts: Vec<Melt> = (0..300u64)
        .filter_map(|s| dense_melt_ev(s, nc, nb, bx, j, min_sep))
        .collect();
    assert!(melts.len() > 250, "only {} melts placed", melts.len());

    let sig = lk_significance(&melts);
    let feat_nf = noise_floor_of(&melts[0], &melts[1]);
    println!(
        "HARDENED SETUP: {} melts at min_sep={min_sep} jitter={j}\n  \
         measured |Lk| significance (95th pct over non-bridged pairs) = {sig:.4} \
         (was a hardcoded 0.1)\n  measured feature noise floor = {feat_nf:.3e}",
        melts.len()
    );

    let mut pts: Vec<(f64, f64, String)> = Vec::new();
    let mut raw: Vec<(f64, f64)> = Vec::new();
    let (mut loose_feasible, mut local_feasible, mut examined) = (0usize, 0usize, 0usize);
    for (si, m) in melts.iter().enumerate() {
        let mb = max_bond(m);
        let base_f = chain_features(m);
        for a in 0..nc {
            for b in (a + 1)..nc {
                for k in 1..nb {
                    examined += 1;
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    if d1 <= mb && d2 <= mb {
                        loose_feasible += 1;
                    }
                    // LOCAL reference: the two bonds actually being replaced.
                    let bond = |c: usize, i: usize| {
                        let d = m.min_image(m.chains[c].beads[i - 1], m.chains[c].beads[i]);
                        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                    };
                    let local = bond(a, k).max(bond(b, k));
                    if d1 > local || d2 > local {
                        continue;
                    }
                    local_feasible += 1;
                    let c = crossover(m, a, b, k);
                    let nbl = new_bond_lengths(m, &c);
                    if nbl.len() != 2 || nbl.iter().any(|l| *l > local) {
                        continue;
                    }
                    if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(m) {
                        continue;
                    }
                    let (cb, lb) = image_carriers_at(m, a, b, sig);
                    let (ca, la) = image_carriers_at(&c, a, b, sig);
                    if cb > 1 || ca > 1 {
                        continue;
                    }
                    let before = if cb == 0 { 0.0 } else { lb };
                    let after = if ca == 0 { 0.0 } else { la };
                    let resid = feature_residuals(&base_f, &chain_features(&c))
                        .into_iter()
                        .fold(0.0f64, |x, y| x.max(y.1));
                    // Report the RAW d|Lk| too. Flooring sub-significance values
                    // to zero would understate the effect just as badly as the old
                    // hardcoded 0.1 overstated it, and "0.0000" must not be allowed
                    // to hide a real-but-insignificant change.
                    raw.push(((lb - la).abs(), resid / feat_nf));
                    pts.push((
                        resid / feat_nf,
                        (before - after).abs(),
                        format!("seed={si} ({a},{b}) k={k} |Lk| {before:.3}->{after:.3}"),
                    ));
                }
            }
        }
    }
    println!(
        "  feasibility: loose (melt-max ref) {}/{} = {:.4}%; LOCAL (replaced-bond ref) \
         {}/{} = {:.4}%",
        loose_feasible,
        examined,
        100.0 * loose_feasible as f64 / examined as f64,
        local_feasible,
        examined,
        100.0 * local_feasible as f64 / examined as f64
    );
    println!("  survivors after all hardened filters: {}", pts.len());
    let mut best_at = Vec::new();
    for cutoff in [1.0f64, 0.5, 0.2, 0.1, 0.05] {
        let kept: Vec<&(f64, f64, String)> = pts.iter().filter(|p| p.0 < cutoff).collect();
        let best = kept.iter().fold(0.0f64, |x, p| x.max(p.1));
        let lbl = kept
            .iter()
            .find(|p| p.1 == best)
            .map(|p| p.2.clone())
            .unwrap_or_else(|| "-".into());
        println!(
            "  residual < {cutoff:>4} x noise floor: n={:<4} max d|Lk| = {best:.4}  [{lbl}]",
            kept.len()
        );
        best_at.push((cutoff, kept.len(), best));
    }
    let raw_best = raw
        .iter()
        .filter(|r| r.1 < 1.0)
        .fold(0.0f64, |x, r| x.max(r.0));
    println!(
        "  RAW (unfloored) max d|Lk| among survivors below the noise floor = {raw_best:.4},          against measured |Lk| significance {sig:.4} -> ratio {:.3}",
        raw_best / sig
    );
    let clears: Vec<&(f64, f64, String)> = pts.iter().filter(|p| p.0 < 1.0 && p.1 > sig).collect();
    println!(
        "  candidates below the noise floor AND above measured |Lk| significance: {}",
        clears.len()
    );
    for c in clears.iter().take(5) {
        println!("    resid={:.4}x nf  d|Lk|={:.4}  {}", c.0, c.1, c.2);
    }
    assert!(
        !clears.is_empty(),
        "under hardened filters (excluded volume, local length reference, measured \
         |Lk| significance {sig:.4}) NO candidate is both below the feature noise \
         floor and above |Lk| significance, over {examined} triples in {} melts — \
         the overturn is WITHDRAWN and the round-3 result was an artifact of the \
         ideal-chain fixture plus a melt-max length reference",
        melts.len()
    );
}

/// The dilution measurement that makes a naive scale-up test meaningless, so the
/// refusal to run one is backed by a number rather than an argument.
#[test]
fn feature_sensitivity_dilutes_as_chains_and_beads_grow() {
    println!("DILUTION: same 2-bond rewiring, growing architecture");
    println!("  chains beads bonds  bonds_rewired_frac  worst_feature_residual");
    for (nc, nb) in [(4usize, 20usize), (8, 20), (16, 20), (8, 40), (8, 80)] {
        let m = match dense_melt_ev(3, nc, nb, ((nc * nb) as f64 / 0.31).cbrt(), 0.1, 0.0) {
            Some(m) => m,
            None => continue,
        };
        let c = crossover(&m, 0, 1, nb / 2);
        let worst = feature_residuals(&chain_features(&m), &chain_features(&c))
            .into_iter()
            .fold(0.0f64, |x, y| x.max(y.1));
        let bonds = nc * (nb - 1);
        println!(
            "  {nc:6} {nb:5} {bonds:5}  {:18.5}  {worst:.5}",
            2.0 / bonds as f64
        );
    }
    println!(
        "  A double bridge always rewires 2 bonds. The six features are averages, so \
         detectability falls as ~1/(C*N) by arithmetic. Any scale-up reporting MORE \
         invisible reconnections is measuring averaging, not topology."
    );
}

// ========================= |Lk| significance recalibrated on REAL melt geometry
//
// The withdrawal rests on a measured |Lk| significance of 1.2327, calibrated at
// 8 chains x 20 beads. These melts are 414 chains x 823 beads. My own Attack 3
// numbers showed residuals RISING with N rather than diluting, so extrapolating
// from N=20 is unsafe in either direction and has to be measured.
//
// P8, registered on the board before running: real-melt significance at N=823 is
// MATERIALLY HIGHER than 1.2327, because open-chain Gauss linking is real-valued
// and grows with contour length, so the withdrawal HARDENS. If it comes out lower
// the withdrawal must be revisited and I say so.
//
// Chain subsets are the first N beads of real equilibrated chains — genuine MD
// geometry, not a synthetic walk. Stated because it is a subset, not the whole
// chain, and the truncation is mine.
#[test]
#[ignore = "needs NERVE_LAMMPS=<path to LAMMPS data file>"]
fn lk_significance_recalibrated_on_real_melt() {
    let path = std::env::var("NERVE_LAMMPS").expect("set NERVE_LAMMPS");
    let full = nerve_melt::from_lammps_path(std::path::Path::new(&path)).expect("parse failed");
    println!(
        "REAL MELT: {} chains x {} beads, box {:.3}",
        full.chains.len(),
        full.chains[0].beads.len(),
        full.box_len
    );
    let contour = 0.96401 * (full.chains[0].beads.len() - 1) as f64;
    println!(
        "  contour length {:.1} vs box {:.3} -> a chain spans ~{:.1} box lengths",
        contour,
        full.box_len,
        contour / full.box_len
    );
    println!(
        "  GUARD max_bond {:.5} < box_len/2 {:.3}: {}",
        max_bond(&full),
        full.box_len / 2.0,
        max_bond(&full) < full.box_len / 2.0
    );

    println!("\n  N     pairs  p50|Lk|  p95|Lk|  max|Lk|  mean_carriers  frac_ambiguous");
    let n_full = full.chains[0].beads.len();
    for &n in &[20usize, 50, 100, 200, 823] {
        if n > n_full {
            continue;
        }
        // Cost control: the 27-image carrier scan is O(27 N^2) per pair.
        let n_pairs = if n >= 800 {
            10
        } else if n >= 200 {
            20
        } else {
            60
        };
        let sub = Melt::new(
            full.chains
                .iter()
                .take(40)
                .map(|c| Chain::new(c.beads[..n].to_vec()))
                .collect(),
            full.box_len,
        );
        let mut lks: Vec<f64> = Vec::new();
        let mut carriers: Vec<usize> = Vec::new();
        let mut done = 0usize;
        'outer: for a in 0..sub.chains.len() {
            for b in (a + 1)..sub.chains.len() {
                if done >= n_pairs {
                    break 'outer;
                }
                // Two-pass: collect first, then recount carriers at the MEASURED
                // p95 for this N. The first version passed a near-zero threshold,
                // so every image with any linking at all counted and the answer
                // was trivially 27/27 — an artifact of my threshold, not a finding.
                let (cc, big) = (0usize, 0.0f64);
                // Significance must be measured on the value the convention
                // actually returns, so use the nearest-image value, and report
                // carrier counts separately as the ambiguity witness.
                let lk = nerve_topo::linking_number(&sub, a, b, Closure::Open).abs();
                lks.push(lk);
                let _ = big;
                carriers.push(cc);
                done += 1;
            }
        }
        let mut sorted = lks.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let pct = |p: f64| {
            sorted[((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
        };
        // Second pass: carriers at the measured p95 for this N.
        let sig_n = pct(0.95);
        carriers.clear();
        let mut done2 = 0usize;
        'second: for a in 0..sub.chains.len() {
            for b in (a + 1)..sub.chains.len() {
                if done2 >= n_pairs {
                    break 'second;
                }
                carriers.push(image_carriers_at(&sub, a, b, sig_n).0);
                done2 += 1;
            }
        }
        let lks = sorted.clone();
        let mean_c = carriers.iter().sum::<usize>() as f64 / carriers.len() as f64;
        let amb = carriers.iter().filter(|c| **c > 1).count() as f64 / carriers.len() as f64;
        println!(
            "  {n:<5} {:<6} {:8.4} {:8.4} {:8.4} {mean_c:14.2} {amb:15.3}",
            lks.len(),
            pct(0.50),
            pct(0.95),
            lks[lks.len() - 1]
        );
    }
    println!(
        "\n  Synthetic reference (8 chains x 20 beads, min_sep 0.8): p95|Lk| = 1.2327, \
         the bar the withdrawal used."
    );
    println!(
        "  Carrier counts use a near-zero significance threshold, so they count every \
         image with any linking at all — an upper bound on ambiguity, deliberately."
    );
}

// ===================================================== ROUND 5 — ATTACK 3, SCALE
//
// Run on REAL melt geometry, because the round-4 gap between real p95|Lk| 0.0191
// and synthetic 1.2327 at matched N=20 turned out to be BOX SIZE, not density: a
// 20-bead chain has extent ~4.3 in my synthetic box of 8, so it interlocks with
// its own periodic images, while the same segment in box 73.747 is isolated. That
// inflated my bar AND my signal together, so a synthetic d|Lk| cannot be compared
// against a real bar. The only clean experiment is the whole hardened search on
// real chains, with both references measured on the same ensemble.
//
// Density matching, resolved rather than restated: |Lk|(a,b) depends only on
// chains a and b, so subsampling chains is exactly density-neutral for pairwise
// linking. And a truncated real chain is a genuine segment whose conformation was
// produced by MD at rho=0.85 — truncation changes what is counted, not the physics
// that shaped it. The feature noise floor is measured between two DISJOINT chain
// subsets of the same melt, which is a genuinely independent same-architecture
// control at identical density.
//
// P9 on the board: I predict the withdrawal REVERSES on real geometry.

/// Hardened search on one sub-melt. Returns
/// `(examined, local_feasible, survivors, max_raw_dlk, best_label)`.
fn hardened_search(m: &Melt, sig: f64, feat_nf: f64) -> (usize, usize, usize, f64, String) {
    let nc = m.chains.len();
    let n = m.chains[0].beads.len();
    let base_f = chain_features(m);
    let (mut examined, mut local_feasible, mut survivors) = (0usize, 0usize, 0usize);
    let (mut best, mut label) = (0.0f64, "none".to_string());
    let bond = |c: usize, i: usize| {
        let d = m.min_image(m.chains[c].beads[i - 1], m.chains[c].beads[i]);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    for a in 0..nc {
        for b in (a + 1)..nc {
            for k in 1..n {
                examined += 1;
                let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                let local = bond(a, k).max(bond(b, k));
                if d1 > local || d2 > local {
                    continue;
                }
                local_feasible += 1;
                let c = crossover(m, a, b, k);
                let nbl = new_bond_lengths(m, &c);
                if nbl.len() != 2 || nbl.iter().any(|l| *l > local) {
                    continue;
                }
                if max_bond(&c) >= c.box_len / 2.0 || bond_graph(&c) == bond_graph(m) {
                    continue;
                }
                let resid = feature_residuals(&base_f, &chain_features(&c))
                    .into_iter()
                    .fold(0.0f64, |x, y| x.max(y.1));
                if resid >= feat_nf {
                    continue;
                }
                survivors += 1;
                let (cb, lb) = image_carriers_at(m, a, b, sig);
                let (ca, la) = image_carriers_at(&c, a, b, sig);
                if cb > 1 || ca > 1 {
                    continue;
                }
                let dlk = (lb - la).abs();
                if dlk > best {
                    best = dlk;
                    label = format!(
                        "({a},{b}) k={k} resid={:.3}xnf |Lk| {lb:.4}->{la:.4}",
                        resid / feat_nf
                    );
                }
            }
        }
    }
    (examined, local_feasible, survivors, best, label)
}

/// p95 |Lk| over up to `cap` chain pairs of `m`, using the nearest-image value.
fn sig_of(m: &Melt, cap: usize) -> f64 {
    let mut v = Vec::new();
    'o: for a in 0..m.chains.len() {
        for b in (a + 1)..m.chains.len() {
            if v.len() >= cap {
                break 'o;
            }
            v.push(nerve_topo::linking_number(m, a, b, Closure::Open).abs());
        }
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[((0.95 * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
}

fn subset(full: &Melt, skip: usize, c: usize, n: usize) -> Melt {
    Melt::new(
        full.chains
            .iter()
            .skip(skip)
            .take(c)
            .map(|ch| Chain::new(ch.beads[..n].to_vec()))
            .collect(),
        full.box_len,
    )
}

/// KEPT VERBATIM AND FAILING. P9 falsified: I predicted the withdrawal would
/// REVERSE on real geometry. It does not — and the reason is stronger than
/// "the features detected them". Across 4 architectures and 277,815 triples on
/// real Kremer-Grest geometry, LOCAL-FEASIBLE candidates = 0. Not one pair of
/// same-index bridge bonds is short enough to be length-preserving, so there was
/// never anything to test. This null is VACUOUS by construction and I am labelling
/// it as such rather than presenting it as a passed test of invisibility.
#[ignore = "P9 FALSE + VACUOUS NULL: 0/277,815 real-melt triples are length-feasible, so nothing was tested; needs NERVE_LAMMPS"]
#[test]
fn attack3_scale_sweep_on_real_melt() {
    let path = std::env::var("NERVE_LAMMPS").expect("set NERVE_LAMMPS");
    let t0 = std::time::Instant::now();
    let full = nerve_melt::from_lammps_path(std::path::Path::new(&path)).expect("parse failed");
    println!(
        "ATTACK 3 on {} chains x {} beads, box {:.3} (parsed in {:?})",
        full.chains.len(),
        full.chains[0].beads.len(),
        full.box_len,
        t0.elapsed()
    );
    println!(
        "  GUARD max_bond {:.5} < box_len/2 {:.3}\n",
        max_bond(&full),
        full.box_len / 2.0
    );
    println!(
        "   C    N   feat_nf   p95|Lk|  examined  local_feas  survivors  max_raw_dlk  \
         dlk/sig  secs"
    );
    let mut verdicts = Vec::new();
    for &(c, n) in &[(40usize, 50usize), (40, 100), (30, 200), (20, 400)] {
        let t = std::time::Instant::now();
        let a = subset(&full, 0, c, n);
        let b = subset(&full, c, c, n); // disjoint chains, same melt, same density
        let feat_nf = worst_feature_residual(&a, &b).1;
        let sig = sig_of(&a, 60);
        let (ex, lf, sv, best, label) = hardened_search(&a, sig, feat_nf);
        let ratio = if sig > 0.0 { best / sig } else { f64::NAN };
        println!(
            "  {c:3} {n:4}  {feat_nf:.3e}  {sig:8.4}  {ex:8}  {lf:10}  {sv:9}  {best:11.4}  \
             {ratio:7.3}  {:.1}",
            t.elapsed().as_secs_f64()
        );
        if !label.starts_with("none") {
            println!("        best: {label}");
        }
        verdicts.push((c, n, sig, best, ratio, sv));
    }
    let any_clears = verdicts.iter().any(|v| v.4 > 1.0);
    println!(
        "\n  ATTACK 3 VERDICT: {} architecture(s) produced a feature-invisible \
         length-preserving double bridge whose d|Lk| exceeds the same-ensemble |Lk| \
         significance scale.",
        verdicts.iter().filter(|v| v.4 > 1.0).count()
    );
    assert!(
        any_clears,
        "on real melt geometry, across {} architectures, NO feature-invisible \
         length-preserving double bridge produced a d|Lk| exceeding the \
         same-ensemble significance scale — the withdrawal HOLDS at scale and P9 \
         is falsified",
        verdicts.len()
    );
}

/// BINDING TEST for the withdrawal. The Inspector's point is fair: the withdrawal
/// lived in an `#[ignore]`d failing test, i.e. in prose. This one RUNS and asserts
/// the negative result, so a future change that resurrects the synthetic overturn
/// breaks the suite instead of passing quietly.
///
/// 60 melts rather than 300 for runtime; the round-4 result used 300 and found
/// zero, so a smaller sample finding zero is consistent and the assertion is the
/// same direction.
#[test]
fn withdrawal_is_bound_synthetic_overturn_does_not_survive_hardening() {
    let melts: Vec<Melt> = (0..60u64)
        .filter_map(|s| dense_melt_ev(s, 8, 20, 8.0, 0.0589, 0.8))
        .collect();
    assert!(melts.len() > 50, "only {} melts placed", melts.len());
    // Bond jitter 0.0589 is the MEASURED real-melt equivalent, not the 0.10 used
    // in round 3.
    let sig = lk_significance(&melts);
    let feat_nf = noise_floor_of(&melts[0], &melts[1]);
    let mut clears = 0usize;
    let mut best = 0.0f64;
    for m in &melts {
        let (_, _, _, dlk, _) = hardened_search(m, sig, feat_nf);
        best = best.max(dlk);
        if dlk > sig {
            clears += 1;
        }
    }
    println!(
        "WITHDRAWAL BOUND: {} synthetic melts at measured jitter 0.0589 + min_sep 0.8; \
         sig={sig:.4} feat_nf={feat_nf:.3e}; max raw d|Lk|={best:.4}; \
         architectures clearing sig = {clears}",
        melts.len()
    );
    assert_eq!(
        clears, 0,
        "a synthetic hardened candidate cleared the significance bar (max d|Lk| \
         {best:.4} vs sig {sig:.4}) — the round-4 withdrawal no longer holds and the \
         verdict must be restated"
    );
}

/// Turns the 0/277,815 into a quantified distance from feasibility, so "zero" is
/// a measurement rather than an absence. Reports the best near-miss ratio
/// `max(d1, d2) / local_bond` over real-melt triples: 1.0 would be feasible.
///
/// The structural reason, which this number is evidence for: preserving chain
/// length on a fixed bead set forces the crossover to happen at the SAME index in
/// both chains, and in a real melt bead k of chain a and bead k of chain b are at
/// entirely uncorrelated positions. Classical double bridging escapes this by
/// RELOCATING beads, which the connectivity tier forbids.
#[test]
#[ignore = "needs NERVE_LAMMPS=<path to LAMMPS data file>"]
fn real_melt_near_miss_quantifies_the_zero() {
    let path = std::env::var("NERVE_LAMMPS").expect("set NERVE_LAMMPS");
    let full = nerve_melt::from_lammps_path(std::path::Path::new(&path)).expect("parse failed");
    for &(c, n) in &[(40usize, 100usize), (20, 400)] {
        let m = subset(&full, 0, c, n);
        let bond = |ci: usize, i: usize| {
            let d = m.min_image(m.chains[ci].beads[i - 1], m.chains[ci].beads[i]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let (mut best, mut examined) = (f64::INFINITY, 0usize);
        for a in 0..c {
            for b in (a + 1)..c {
                for k in 1..n {
                    examined += 1;
                    let d1 = m.min_image_dist(m.chains[a].beads[k - 1], m.chains[b].beads[k]);
                    let d2 = m.min_image_dist(m.chains[b].beads[k - 1], m.chains[a].beads[k]);
                    let local = bond(a, k).max(bond(b, k));
                    best = best.min(d1.max(d2) / local);
                }
            }
        }
        println!(
            "NEAR MISS C={c} N={n}: {examined} triples, best max(d1,d2)/local_bond =              {best:.3} (1.0 would be feasible) -> shortest realisable bridge is              {:.1}x too long",
            best
        );
    }
}
