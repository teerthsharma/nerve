//! Invariants the null model must satisfy before any of its numbers are
//! admissible. Every test here was written and observed RED against a `todo!()`
//! stub before the implementation existed.
//!
//! Invariant set taken from `anthropic-skills:tda-tdd`: permutation invariance,
//! isometry invariance, lattice-translation invariance under PBC, smoothness at
//! the cutoff, a negative control on a random gas, and determinism.

use nerve_baseline::*;
use nerve_core::{Chain, Melt, Vec3};
use proptest::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ----------------------------------------------------------------- fixtures

const FREE_BOX: f64 = 1000.0;
const CTR: f64 = 500.0;

/// Random-walk chains in the free (no-wrap) limit: the box is far larger than
/// any cutoff used, so `min_image` degenerates to plain subtraction and
/// rotations are a genuine symmetry. Rotating a periodic box is not.
fn rw_melt_free(seed: u64, n_chains: usize, n: usize, bond: f64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let chains = (0..n_chains)
        .map(|_| {
            let mut p: Vec3 = [
                CTR + rng.gen_range(-10.0..10.0),
                CTR + rng.gen_range(-10.0..10.0),
                CTR + rng.gen_range(-10.0..10.0),
            ];
            let mut beads = vec![p];
            for _ in 1..n {
                let s = rand_unit(&mut rng);
                p = [p[0] + bond * s[0], p[1] + bond * s[1], p[2] + bond * s[2]];
                beads.push(p);
            }
            Chain::new(beads)
        })
        .collect();
    Melt::new(chains, FREE_BOX)
}

/// Random-walk chains wrapped into a dense periodic box.
fn rw_melt_pbc(seed: u64, n_chains: usize, n: usize, bond: f64, box_len: f64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let chains = (0..n_chains)
        .map(|_| {
            let mut p: Vec3 = [
                rng.gen_range(0.0..box_len),
                rng.gen_range(0.0..box_len),
                rng.gen_range(0.0..box_len),
            ];
            let mut beads = vec![wrap(p, box_len)];
            for _ in 1..n {
                let s = rand_unit(&mut rng);
                p = [p[0] + bond * s[0], p[1] + bond * s[1], p[2] + bond * s[2]];
                beads.push(wrap(p, box_len));
            }
            Chain::new(beads)
        })
        .collect();
    Melt::new(chains, box_len)
}

/// Uniform ideal gas, arbitrarily chopped into "chains". The negative control:
/// there is no chain structure here to find.
fn gas_melt(seed: u64, n_chains: usize, n: usize, box_len: f64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let chains = (0..n_chains)
        .map(|_| {
            Chain::new(
                (0..n)
                    .map(|_| {
                        [
                            rng.gen_range(0.0..box_len),
                            rng.gen_range(0.0..box_len),
                            rng.gen_range(0.0..box_len),
                        ]
                    })
                    .collect(),
            )
        })
        .collect();
    Melt::new(chains, box_len)
}

/// Simple-cubic crystal at the same bead density as `gas_melt` with the same
/// counts. The positive control the gas must be distinguishable from.
fn lattice_melt(m: usize, n: usize, box_len: f64) -> Melt {
    let a = box_len / m as f64;
    let pts: Vec<Vec3> = (0..m)
        .flat_map(|i| {
            (0..m).flat_map(move |j| {
                (0..m).map(move |k| [i as f64 * a, j as f64 * a, k as f64 * a])
            })
        })
        .collect();
    Melt::new(pts.chunks(n).map(|c| Chain::new(c.to_vec())).collect(), box_len)
}

fn rand_unit(rng: &mut ChaCha8Rng) -> Vec3 {
    loop {
        let v: Vec3 = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 1e-3 && n <= 1.0 {
            return [v[0] / n, v[1] / n, v[2] / n];
        }
    }
}

fn wrap(p: Vec3, l: f64) -> Vec3 {
    [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2].rem_euclid(l)]
}

fn map_beads(melt: &Melt, f: impl Fn(Vec3) -> Vec3) -> Melt {
    Melt::new(
        melt.chains
            .iter()
            .map(|c| Chain::new(c.beads.iter().copied().map(&f).collect()))
            .collect(),
        melt.box_len,
    )
}

/// Rodrigues rotation about a unit axis.
fn rotate(p: Vec3, axis: Vec3, ang: f64) -> Vec3 {
    let (s, c) = ang.sin_cos();
    let d = dot(axis, p);
    let cr = [
        axis[1] * p[2] - axis[2] * p[1],
        axis[2] * p[0] - axis[0] * p[2],
        axis[0] * p[1] - axis[1] * p[0],
    ];
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = p[i] * c + cr[i] * s + axis[i] * d * (1.0 - c);
    }
    out
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Same bead positions, different chain connectivity: bead `i` of the flattened
/// list goes to chain `i % n_chains`. Point set identical, topology different.
fn reconnect(melt: &Melt, n_chains: usize) -> Melt {
    let all = beads(melt);
    let mut chains = vec![Vec::new(); n_chains];
    for (i, b) in all.iter().enumerate() {
        chains[i % n_chains].push(*b);
    }
    Melt::new(chains.into_iter().map(Chain::new).collect(), melt.box_len)
}

// ------------------------------------------------------- 1. permutation inv.

#[test]
fn descriptor_is_permutation_invariant_over_beads() {
    let m = rw_melt_pbc(11, 4, 10, 1.0, 12.0);
    let acsf = Acsf::new(3.5);
    let d0 = acsf.melt_descriptor(&m);

    // Reverse every chain and reverse the chain order: the flattened bead list
    // is permuted, the point set is not.
    let mut chains: Vec<Chain> = m
        .chains
        .iter()
        .map(|c| Chain::new(c.beads.iter().rev().copied().collect()))
        .collect();
    chains.reverse();
    let d1 = acsf.melt_descriptor(&Melt::new(chains, m.box_len));

    assert!(l2(&d0, &d1) < 1e-12, "descriptor moved under bead permutation: {d0:?} vs {d1:?}");
}

// ---------------------------------------------------------- 2. isometry inv.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn descriptor_is_isometry_invariant(
        seed in 0u64..8,
        ang in 0.0f64..std::f64::consts::TAU,
        tx in -50.0f64..50.0,
        ty in -50.0f64..50.0,
        tz in -50.0f64..50.0,
    ) {
        let m = rw_melt_free(seed, 3, 12, 1.0);
        let acsf = Acsf::new(4.0);
        let d0 = acsf.melt_descriptor(&m);

        let axis = { let n = (1.0f64 + 4.0 + 9.0).sqrt(); [1.0/n, 2.0/n, 3.0/n] };
        let ctr = [CTR, CTR, CTR];
        let moved = map_beads(&m, |p| {
            let rel = [p[0]-ctr[0], p[1]-ctr[1], p[2]-ctr[2]];
            let r = rotate(rel, axis, ang);
            [r[0]+ctr[0]+tx, r[1]+ctr[1]+ty, r[2]+ctr[2]+tz]
        });
        let d1 = acsf.melt_descriptor(&moved);
        prop_assert!(l2(&d0, &d1) < 1e-9, "isometry changed descriptor by {}", l2(&d0, &d1));
    }

    // --------------------------------------- 3. lattice-translation inv. (PBC)

    #[test]
    fn descriptor_is_lattice_translation_invariant(
        seed in 0u64..8,
        nx in -2i32..3, ny in -2i32..3, nz in -2i32..3,
        sx in 0.0f64..12.0, sy in 0.0f64..12.0, sz in 0.0f64..12.0,
    ) {
        let l = 12.0;
        let m = rw_melt_pbc(seed, 5, 10, 1.0, l);
        let acsf = Acsf::new(3.5);
        let d0 = acsf.melt_descriptor(&m);

        // Uniform translation composed with a whole-lattice-vector shift, then
        // re-wrapped. Neither is allowed to change anything.
        let moved = map_beads(&m, |p| wrap([
            p[0] + sx + nx as f64 * l,
            p[1] + sy + ny as f64 * l,
            p[2] + sz + nz as f64 * l,
        ], l));
        let d1 = acsf.melt_descriptor(&moved);
        prop_assert!(l2(&d0, &d1) < 1e-9, "PBC translation changed descriptor by {}", l2(&d0, &d1));

        // Chain geometry must not measure the box either.
        let f0 = chain_features(&m);
        let f1 = chain_features(&moved);
        prop_assert!((f0.rg2 - f1.rg2).abs() < 1e-9, "rg2 {} vs {}", f0.rg2, f1.rg2);
    }
}

// ------------------------------------------------------- 4. cutoff smoothness

#[test]
fn descriptor_goes_to_zero_smoothly_at_cutoff() {
    // Three isolated beads in a huge box. A sits inside the cutoff of C; B is
    // walked from just inside the cutoff to outside it. A–B is always outside
    // the cutoff (|A-B| = sqrt(1.5^2 + r^2) > r_cut), so the only term crossing
    // the boundary is the C–B one, and the C-centred angle is 90 degrees so the
    // angular basis is exercised too.
    let r_cut = 4.0;
    let acsf = Acsf::new(r_cut);
    let build = |r: f64| {
        let c = [CTR, CTR, CTR];
        Melt::new(
            vec![
                Chain::new(vec![c]),
                Chain::new(vec![[c[0] + 1.5, c[1], c[2]]]),
                Chain::new(vec![[c[0], c[1] + r, c[2]]]),
            ],
            FREE_BOX,
        )
    };

    let outside = acsf.melt_descriptor(&build(r_cut * 1.5));
    for h in [1e-2, 1e-3, 1e-4] {
        let inside = acsf.melt_descriptor(&build(r_cut - h));
        let jump = l2(&inside, &outside);
        // A hard cutoff gives O(1) here; a linear taper gives O(h). Only a
        // cutoff function with fc(r_cut) = fc'(r_cut) = 0 gives O(h^2).
        assert!(
            jump < 10.0 * h * h,
            "descriptor jumps {jump:.3e} at h={h:.0e}; needs to be O(h^2) or forces are discontinuous"
        );
        assert!(jump > 0.0, "descriptor is not responding to the neighbour at all");
    }
}

// ----------------------------------------------------------- 5. determinism

#[test]
fn descriptor_is_bitwise_deterministic() {
    let m = rw_melt_pbc(3, 4, 10, 1.0, 12.0);
    let acsf = Acsf::new(3.0);
    assert_eq!(acsf.melt_descriptor(&m), acsf.melt_descriptor(&m));
    assert_eq!(chain_features(&m), chain_features(&m));
    // Same seed, rebuilt melt, same answer.
    assert_eq!(
        acsf.melt_descriptor(&m),
        acsf.melt_descriptor(&rw_melt_pbc(3, 4, 10, 1.0, 12.0))
    );
}

// ------------------------------------------------------- 6. negative control

#[test]
fn random_gas_produces_no_structure() {
    // Same bead count (512) and same box (8.0) => same density 1.0.
    let g1 = gas_melt(1, 64, 8, 8.0);
    let g2 = gas_melt(2, 64, 8, 8.0);
    let xtal = lattice_melt(8, 8, 8.0);
    let acsf = Acsf::new(3.0);

    let d_gg = cosine_distance(&acsf.melt_descriptor(&g1), &acsf.melt_descriptor(&g2));
    let d_gx = cosine_distance(&acsf.melt_descriptor(&g1), &acsf.melt_descriptor(&xtal));

    assert!(d_gx > 1e-3, "descriptor cannot see a crystal at all: d_gx={d_gx:.3e}");
    assert!(
        d_gg < d_gx / 5.0,
        "two independent gases look as different as gas vs crystal: d_gg={d_gg:.3e} d_gx={d_gx:.3e}"
    );
}

#[test]
fn pooled_descriptor_retains_environment_heterogeneity() {
    // Pooling environments by their mean alone would throw away the spread
    // across beads, and a real MLIP does not: it evaluates a per-bead function
    // and sums, so heterogeneity is available to it. To be the strongest
    // possible opponent the melt descriptor must carry both moments.
    //
    // Ground truth: a simple-cubic crystal has one environment repeated, so the
    // spread channel is ~0. A gas at the same density has a wide spread.
    let acsf = Acsf::new(3.0);
    assert_eq!(acsf.dim(), 2 * (acsf.n_radial + acsf.n_angular), "layout is [mean, spread]");

    let half = acsf.dim() / 2;
    let nrm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let xtal = acsf.melt_descriptor(&lattice_melt(8, 8, 8.0));
    let gas = acsf.melt_descriptor(&gas_melt(1, 64, 8, 8.0));

    assert!(
        nrm(&xtal[half..]) < nrm(&gas[half..]) / 5.0,
        "spread channel does not separate a crystal from a gas: {:.4e} vs {:.4e}",
        nrm(&xtal[half..]),
        nrm(&gas[half..])
    );
    assert!(nrm(&xtal[..half]) > 0.0, "mean channel is empty");
}

#[test]
fn msid_grows_linearly_for_a_random_walk() {
    let rw = chain_features(&rw_melt_pbc(5, 8, 20, 1.0, 40.0));
    // Ideal chain: <R^2(k)> = k b^2, so msid[18]/msid[0] is ~19.
    let ratio = rw.msid[18] / rw.msid[0];
    assert!(ratio > 5.0, "random walk MSID is not growing with separation: {ratio:.2}");
}

#[test]
fn max_bond_flags_a_gas_whose_chain_features_are_meaningless() {
    // The negative control, in the only form that is actually true.
    //
    // MSID cannot be the negative control on its own: unwrapping any bead
    // sequence by min-image bond vectors produces a random walk in unwrapped
    // coordinates, so an ideal gas chopped into fake chains reports MSID
    // growing linearly with separation exactly like a real chain does
    // (measured ratio 14.59 vs 19). The discriminator is the precondition, not
    // the feature: bond vectors must be short compared with box_len/2 for
    // unwrapping to recover the true walk at all.
    let l = 40.0;
    let rw = chain_features(&rw_melt_pbc(5, 8, 20, 1.0, l));
    let gas = chain_features(&gas_melt(6, 8, 20, l));

    assert!(rw.max_bond < l / 2.0, "real chain flagged as unresolvable: {}", rw.max_bond);
    assert!(
        gas.max_bond >= l / 2.0,
        "gas passed the unwrap-validity guard, so structure in noise goes unreported: {}",
        gas.max_bond
    );
}

// ------------------------------------------- 7. the rebuttal, made executable

#[test]
fn local_descriptor_is_blind_to_chain_reconnection_at_every_cutoff() {
    // Two melts with an identical bead point set and different connectivity.
    // Any descriptor that is a function of interbead distances alone cannot
    // separate them - at ANY cutoff. If this test ever fails, the descriptor
    // has become connectivity-aware and stops being the null model.
    let a = rw_melt_pbc(7, 4, 10, 1.0, 12.0);
    let b = reconnect(&a, 4);
    assert_ne!(a.chains, b.chains, "reconnect() did not change connectivity");

    for (r, d) in discriminability_sweep(&a, &b, &[2.0, 3.0, 4.0, 5.0, 6.0]) {
        assert!(d < 1e-15, "local descriptor separated reconnected melts at r_cut={r}: d={d:.3e}");
    }
}

#[test]
fn cheap_chain_features_do_separate_reconnected_melts() {
    // ...and this is why "widen the cutoff" is the wrong rebuttal but
    // "add five cheap scalars" might be the right one.
    let a = rw_melt_pbc(7, 4, 10, 1.0, 12.0);
    let b = reconnect(&a, 4);
    let (fa, fb) = (chain_features(&a), chain_features(&b));

    assert!(fb.rg2 > 2.0 * fa.rg2, "rg2 failed to separate: {} vs {}", fa.rg2, fb.rg2);
    assert!(
        (fa.bead_density - fb.bead_density).abs() < 1e-12,
        "density must be identical - same beads, same box"
    );
}

// ------------------------------------------------------------- 8. plumbing

#[test]
fn contour_length_recovers_the_bond_length_through_the_boundary() {
    let m = rw_melt_pbc(9, 6, 15, 1.0, 10.0);
    let f = chain_features(&m);
    // 14 bonds of length exactly 1.0. Anything else means unwrapping is broken.
    assert!((f.contour_len - 14.0).abs() < 1e-9, "contour_len = {}", f.contour_len);
    assert!((f.bead_density - 90.0 / 1000.0).abs() < 1e-12, "density = {}", f.bead_density);
}

#[test]
fn cutoff_sweep_returns_one_descriptor_per_radius() {
    let m = rw_melt_pbc(4, 3, 8, 1.0, 12.0);
    let rs = [2.0, 3.0, 4.0];
    let out = cutoff_sweep(&m, &rs);
    assert_eq!(out.len(), 3);
    for (i, (r, d)) in out.iter().enumerate() {
        assert_eq!(*r, rs[i]);
        assert_eq!(d.len(), Acsf::new(*r).dim());
    }
    // Distinct radii must give distinct descriptors, or the sweep is a no-op.
    assert!(l2(&out[0].1, &out[2].1) > 1e-6);
    assert!(cosine_distance(&out[0].1, &out[0].1) < 1e-15);
}

// ----------------------------------------------- 9. the sweep, characterised

#[test]
fn cutoff_above_half_the_box_is_rejected() {
    // "Just widen the cutoff" has a hard ceiling: past box_len/2 the minimum
    // image is no longer the nearest image, so the descriptor silently starts
    // double-counting periodic copies. Refuse rather than return garbage.
    let m = rw_melt_pbc(2, 2, 5, 1.0, 10.0);
    let r = std::panic::catch_unwind(|| Acsf::new(5.001).melt_descriptor(&m));
    assert!(r.is_err(), "descriptor accepted r_cut > box_len/2 and returned a number anyway");
    assert!(Acsf::new(5.0).melt_descriptor(&m).len() == Acsf::new(5.0).dim());
}

#[test]
fn widening_the_cutoff_does_not_separate_melts_that_differ_only_in_chain_length() {
    // The null-model experiment. Two melts at identical bead density and
    // identical bond length, differing only in how the beads are wired into
    // chains: 32 chains of 8 beads vs 4 chains of 64. Chain length is the
    // crudest topological quantity there is.
    //
    // Hypothesis under test: the local descriptor's separation of these two
    // does not improve materially as the cutoff grows, because the local
    // environment statistics of an ideal melt are chain-length independent.
    // A single melt pair cannot answer this. Two independent melts differ by
    // finite-size sampling noise whether or not their chain lengths differ, so
    // the quantity that means anything is
    //
    //     signal = <d(32x8 melt, 4x64 melt)>   over cross pairs
    //     noise  = <d(same architecture, different seed)>   the control
    //
    // and the question is whether signal/noise rises above 1 as r_cut grows.
    // Without the control, the curve measures the seed.
    let l = 6.35; // 256 beads => density ~1.0
    let radii = [1.5, 2.0, 2.5, 3.0, 3.175]; // ceiling is box_len/2 = 3.175
    let seeds = [21u64, 31, 41, 51];

    let shorts: Vec<_> = seeds.iter().map(|&s| rw_melt_pbc(s, 32, 8, 1.0, l)).collect();
    let longs: Vec<_> = seeds.iter().map(|&s| rw_melt_pbc(s + 1, 4, 64, 1.0, l)).collect();
    assert_eq!(shorts[0].n_beads(), longs[0].n_beads());

    let sw = |m: &Melt| cutoff_sweep(m, &radii);
    let ds: Vec<_> = shorts.iter().map(sw).collect();
    let dl: Vec<_> = longs.iter().map(sw).collect();

    println!("\n  r_cut    signal (32x8 vs 4x64)      noise (same arch, new seed)   signal/noise");
    let mut ratios = Vec::new();
    for (k, r) in radii.iter().enumerate() {
        let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let signal = mean(
            ds.iter()
                .flat_map(|a| dl.iter().map(move |b| cosine_distance(&a[k].1, &b[k].1)))
                .collect(),
        );
        let mut ctrl = Vec::new();
        for set in [&ds, &dl] {
            for i in 0..set.len() {
                for j in i + 1..set.len() {
                    ctrl.push(cosine_distance(&set[i][k].1, &set[j][k].1));
                }
            }
        }
        let noise = mean(ctrl);
        ratios.push(signal / noise);
        println!("  {r:5.3}    {signal:.6e}              {noise:.6e}                 {:.2}", signal / noise);
    }

    let (fs, fl) = (chain_features(&shorts[0]), chain_features(&longs[0]));
    println!(
        "  chain features: rg2 {:.3} vs {:.3} ({:.1}x); msid at separation 7: {:.3} vs {:.3}",
        fs.rg2, fl.rg2, fl.rg2 / fs.rg2, fs.msid[6], fl.msid[6]
    );

    // Hypothesis: the local descriptor carries no chain-length information at
    // any admissible cutoff, so signal never separates from its own control.
    let worst = ratios.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        worst < 3.0,
        "signal/noise reached {worst:.2} - widening the cutoff DOES recover chain length, \
         and the rebuttal has teeth: {ratios:?}"
    );
    // The cheap non-topological features, by contrast, separate them outright.
    assert!(fl.rg2 > 3.0 * fs.rg2, "rg2 {} vs {}", fs.rg2, fl.rg2);
}

#[test]
fn unwrapped_chain_matches_the_unwrapped_source_walk() {
    // A chain that straddles the boundary must unwrap back to a walk whose
    // bond vectors are the original ones.
    let l = 6.0;
    let m = Melt::new(
        vec![Chain::new(vec![
            [5.5, 1.0, 1.0],
            [0.2, 1.0, 1.0], // crosses the +x wall
            [0.9, 1.0, 1.0],
        ])],
        l,
    );
    let u = unwrap_chain(&m, 0);
    assert!((u[1][0] - u[0][0] - 0.7).abs() < 1e-12, "unwrapped x: {:?}", u);
    assert!((u[2][0] - u[0][0] - 1.4).abs() < 1e-12, "unwrapped x: {:?}", u);
}
