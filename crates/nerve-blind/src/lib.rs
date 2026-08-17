//! Is a cutoff neural interatomic potential blind to chain topology?
//!
//! The instrument is a **topologically-matched pair**: two [`Melt`]s whose
//! multiset of local environments agrees to within a measured residual, but
//! whose pairwise Gauss linking number differs.
//!
//! Three tiers, each attacking a *different* assumption of a cutoff MLIP.
//!
//! | tier | assumption attacked | regime | residual |
//! |------|--------------------|--------|----------|
//! | parity | model is O(3)-invariant | any radius, any depth | exactly 0 |
//! | connectivity | model reads positions + species only | any radius, any depth | exactly 0 |
//! | locality | model has a finite cutoff | bounded by 2*delta | measured, > 0 |
//!
//! ## Regime, stated before the claim
//!
//! A finite periodic box holding finite chains is a finite graph, so a model
//! with enough message-passing layers has a receptive field covering the whole
//! system and no locality-based separation can exist. That objection binds only
//! on the *locality* tier. The parity and connectivity tiers are invariance
//! properties of the model class, not of its receptive field: they hold at any
//! radius and any depth, which is why the radius sweep in the tests runs out to
//! a cutoff larger than the box.
//!
//! ## Coordinates
//!
//! Linking number is computed on **raw, unwrapped** coordinates. Minimum-image
//! displacement applied per segment silently destroys topology. Every geometry
//! in this crate is built wholly inside the box so periodicity never enters the
//! topological quantities; [`Melt::min_image`] is used for descriptor distances
//! only, where it is the physically correct distance.
//!
//! Knot type is deliberately absent. Linear chains in a Kremer-Grest melt are
//! essentially unknotted, so knot type carries no signal there; the
//! discriminators are pairwise linking number and writhe on closed rings.

use nerve_core::{Chain, Melt, Vec3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;

const TWO_PI: f64 = std::f64::consts::TAU;
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

// ---------------------------------------------------------------- vector ops

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: Vec3) -> Option<Vec3> {
    let n = norm(a);
    if n < 1e-300 {
        None
    } else {
        Some([a[0] / n, a[1] / n, a[2] / n])
    }
}

// ------------------------------------------------------------------ topology

/// Signed area on the unit sphere swept by the Gauss map of two segments,
/// `r1 -> r2` and `r3 -> r4`.
///
/// Exact for straight segments (Klenin & Langowski, *Comput. Theor. Polym.
/// Sci.* 2000), so a polygon's linking number comes out an integer to floating
/// point rather than converging to one. That matters here: the matched pairs put
/// strands arbitrarily close together, which is exactly where a midpoint-rule
/// Gauss integral loses its accuracy.
fn segment_pair_solid_angle(r1: Vec3, r2: Vec3, r3: Vec3, r4: Vec3) -> f64 {
    let r13 = sub(r3, r1);
    let r14 = sub(r4, r1);
    let r23 = sub(r3, r2);
    let r24 = sub(r4, r2);
    let r12 = sub(r2, r1);
    let r34 = sub(r4, r3);

    let (n1, n2, n3, n4) = match (
        unit(cross(r13, r14)),
        unit(cross(r14, r24)),
        unit(cross(r24, r23)),
        unit(cross(r23, r13)),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        // Coplanar or degenerate: the Gauss map sweeps no area.
        _ => return 0.0,
    };

    let asin_c = |x: f64| x.clamp(-1.0, 1.0).asin();
    let area = asin_c(dot(n1, n2)) + asin_c(dot(n2, n3)) + asin_c(dot(n3, n4)) + asin_c(dot(n4, n1));
    let sign = dot(cross(r34, r12), r13);
    if sign < 0.0 {
        -area
    } else {
        area
    }
}

/// Closed-polygon segments of a chain: bead `k` to bead `k+1`, plus the closure
/// segment from the last bead back to the first.
fn segments(c: &Chain) -> Vec<(Vec3, Vec3)> {
    let n = c.len();
    (0..n).map(|k| (c.beads[k], c.beads[(k + 1) % n])).collect()
}

/// Gauss linking number of two chains, each treated as a **closed** polygon
/// (a segment from the last bead back to the first is implied).
///
/// Raw coordinates, no minimum-image.
pub fn linking_number(a: &Chain, b: &Chain) -> f64 {
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    let (sa, sb) = (segments(a), segments(b));
    let mut acc = 0.0;
    for &(p, q) in &sa {
        for &(u, v) in &sb {
            acc += segment_pair_solid_angle(p, q, u, v);
        }
    }
    acc / FOUR_PI
}

/// Writhe of a single closed chain: the same double sum over ordered pairs of
/// distinct, non-adjacent segments of the chain with itself.
pub fn writhe(c: &Chain) -> f64 {
    let n = c.len();
    if n < 4 {
        return 0.0;
    }
    let s = segments(c);
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            let sep = (i as i64 - j as i64).abs();
            let cyc = sep.min(n as i64 - sep);
            if cyc < 2 {
                continue;
            }
            acc += segment_pair_solid_angle(s[i].0, s[i].1, s[j].0, s[j].1);
        }
    }
    acc / FOUR_PI
}

/// Every pairwise linking number in the melt, chain index order, `i < j`.
pub fn linking_matrix(m: &Melt) -> Vec<((usize, usize), f64)> {
    let n = m.chains.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            out.push(((i, j), linking_number(&m.chains[i], &m.chains[j])));
        }
    }
    out
}

// ---------------------------------------------------------------- descriptor

/// A local descriptor of the kind a cutoff MLIP actually has.
///
/// Each bead's environment is the sorted list of minimum-image distances to
/// beads inside `radius`. This is a *radial* descriptor, deliberately weaker
/// than SOAP or ACE — but the parity and connectivity matched pairs are proved
/// indistinguishable at the level of the underlying geometry (bit-identical
/// distance matrix, or bit-identical point set), so the conclusion transfers to
/// any descriptor in the stated class however strong it is. The descriptor is
/// the concrete instrument, not the argument.
///
/// `bond_aware` additionally labels each neighbour with its cyclic along-chain
/// separation when that is `<= 2`. This is information a bond-graph model gets
/// and a position-only model does not, and it is what separates the parity tier
/// from the connectivity tier.
#[derive(Clone, Copy, Debug)]
pub struct Descriptor {
    pub radius: f64,
    pub bond_aware: bool,
}

/// Sorted multiset of local environments. Each environment is a sorted list of
/// `(distance, along-chain label)`; the label is `0` when `bond_aware` is off.
#[derive(Clone, Debug, PartialEq)]
pub struct Signature(pub Vec<Vec<(f64, i8)>>);

fn cmp_entry(a: &(f64, i8), b: &(f64, i8)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

fn cmp_env(a: &[(f64, i8)], b: &[(f64, i8)]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let c = cmp_entry(x, y);
        if c != Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

impl Descriptor {
    pub fn signature(&self, m: &Melt) -> Signature {
        let mut envs = Vec::with_capacity(m.n_beads());
        for (ci, ca) in m.chains.iter().enumerate() {
            for (bi, &a) in ca.beads.iter().enumerate() {
                let mut env = Vec::new();
                for (cj, cb) in m.chains.iter().enumerate() {
                    for (bj, &b) in cb.beads.iter().enumerate() {
                        if ci == cj && bi == bj {
                            continue;
                        }
                        let d = m.min_image_dist(a, b);
                        if d >= self.radius {
                            continue;
                        }
                        let label = if self.bond_aware && ci == cj {
                            let n = cb.len() as i64;
                            let sep = (bi as i64 - bj as i64).abs();
                            let cyc = sep.min(n - sep);
                            if cyc <= 2 {
                                cyc as i8
                            } else {
                                0
                            }
                        } else {
                            0
                        };
                        env.push((d, label));
                    }
                }
                env.sort_by(cmp_entry);
                envs.push(env);
            }
        }
        envs.sort_by(|a, b| cmp_env(a, b));
        Signature(envs)
    }
}

impl Signature {
    /// Every distance multiplied by `c`, labels untouched — the exact
    /// prediction of scale equivariance.
    pub fn scaled(&self, c: f64) -> Signature {
        Signature(
            self.0
                .iter()
                .map(|e| e.iter().map(|&(d, l)| (d * c, l)).collect())
                .collect(),
        )
    }
}

/// Largest distance disagreement between two signatures after canonical
/// sorting; `f64::INFINITY` when they differ structurally (different bead
/// count, different neighbour count, or different along-chain label), because no
/// finite tolerance makes those two signatures the same object.
pub fn residual(a: &Signature, b: &Signature) -> f64 {
    if a.0.len() != b.0.len() {
        return f64::INFINITY;
    }
    let mut worst = 0.0f64;
    for (ea, eb) in a.0.iter().zip(b.0.iter()) {
        if ea.len() != eb.len() {
            return f64::INFINITY;
        }
        for (&(da, la), &(db, lb)) in ea.iter().zip(eb.iter()) {
            if la != lb {
                return f64::INFINITY;
            }
            worst = worst.max((da - db).abs());
        }
    }
    worst
}

// -------------------------------------------------------------- constructions

/// Two circles of radius `r`, `n` beads each, centred on the box centre `c`:
/// the first in the plane `z = c`, the second in the plane `y = c` with its
/// centre displaced by `offset` along `x`.
///
/// The second circle pierces the plane of the first at `x = c + offset ± r`.
/// Exactly one of those punctures lies inside the first disc when
/// `0 < offset < 2r`, giving `|Lk| = 1`; neither does when `offset > 2r`, giving
/// `Lk = 0`. The circles touch at `offset == 2r`. The whole matched-pair family
/// falls out of that single transition.
pub fn two_rings(n: usize, r: f64, offset: f64, box_len: f64) -> Melt {
    let c = box_len / 2.0;
    let ring = |f: &dyn Fn(f64) -> Vec3| -> Chain {
        Chain::new((0..n).map(|k| f(TWO_PI * k as f64 / n as f64)).collect())
    };
    let a = ring(&|t: f64| [c + r * t.cos(), c + r * t.sin(), c]);
    let b = ring(&|t: f64| [c + offset + r * t.cos(), c, c + r * t.sin()]);
    Melt::new(vec![a, b], box_len)
}

/// Reflect the melt by exchanging the `x` and `y` coordinates of every bead.
///
/// An improper isometry (determinant −1) chosen because it is exact in binary
/// floating point: it permutes coordinates rather than arithmetically negating
/// them, so the distance matrix is preserved *bit for bit*, not merely to
/// tolerance. Any O(3)-invariant model — which is every production MLIP, since
/// energy is assembled from invariants of the distance matrix — therefore
/// cannot distinguish the two melts by any margin whatsoever. Linking number
/// and writhe both change sign.
pub fn mirror(m: &Melt) -> Melt {
    Melt::new(
        m.chains
            .iter()
            .map(|c| Chain::new(c.beads.iter().map(|b| [b[1], b[0], b[2]]).collect()))
            .collect(),
        m.box_len,
    )
}

/// Cross-splice two closed chains: chain `c1` keeps beads `..=i` and continues
/// into chain `c2`'s beads `j+1..`, and vice versa.
///
/// The multiset of bead positions is bit-identical to the input, and so is the
/// chain count. Only which bead follows which changes — information an MLIP is
/// never given.
pub fn reconnect(m: &Melt, c1: usize, i: usize, c2: usize, j: usize) -> Melt {
    let (a, b) = (&m.chains[c1].beads, &m.chains[c2].beads);
    let mut n1 = a[..=i].to_vec();
    n1.extend_from_slice(&b[j + 1..]);
    let mut n2 = b[..=j].to_vec();
    n2.extend_from_slice(&a[i + 1..]);
    let mut chains = m.chains.clone();
    chains[c1] = Chain::new(n1);
    chains[c2] = Chain::new(n2);
    Melt::new(chains, m.box_len)
}

/// Longest bond in the melt, **ring-closure segment included**. Reconnection
/// preserves positions but not bond lengths; this is the price, and it is
/// reported rather than hidden.
///
/// Distinct from [`ChainFeatures::max_bond`], which follows the baseline crate's
/// definition (consecutive beads only, no closure) and is the validity guard.
pub fn max_bond_closed(m: &Melt) -> f64 {
    m.chains
        .iter()
        .flat_map(|c| segments(c).into_iter().map(|(p, q)| norm(sub(q, p))))
        .fold(0.0f64, f64::max)
}

/// Smallest minimum-image distance between beads of two different chains.
pub fn min_interchain_dist(m: &Melt) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..m.chains.len() {
        for j in i + 1..m.chains.len() {
            for &a in &m.chains[i].beads {
                for &b in &m.chains[j].beads {
                    best = best.min(m.min_image_dist(a, b));
                }
            }
        }
    }
    best
}

// ------------------------------------------------------- non-topological null
//
// A pair matched only on local environments is not enough. Cheap single-chain
// size features separate melts with no topology at all, so if the pair differs
// in radius of gyration or in its internal-distance curve, the demonstration
// shows the descriptor is bad at chain size — which nobody disputes — rather
// than blind to topology.
//
// These definitions deliberately mirror `nerve-baseline`'s `ChainFeatures` so
// the matching criteria are the same quantities the null model measures. The
// code is independent: this crate does not depend on that one.

/// Cheap non-topological chain features, averaged over chains.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainFeatures {
    /// Mean squared radius of gyration.
    pub rg2: f64,
    /// Mean squared end-to-end distance.
    pub end_to_end2: f64,
    /// Mean contour length (sum of bond lengths, consecutive beads only).
    pub contour_len: f64,
    /// Beads per unit volume.
    pub bead_density: f64,
    /// Longest consecutive-bead min-image bond, closure excluded.
    ///
    /// **Validity guard.** Every other field is computed on bond-unwrapped
    /// coordinates, and unwrapping only recovers the true walk when bonds are
    /// shorter than `box_len / 2`. Above that the min-image choice is wrong and
    /// these statistics describe a random walk whatever the input. Check this
    /// before believing anything else here.
    pub max_bond: f64,
    /// Mean-square internal distance: `msid[k]` is the mean of
    /// `|r_{i+k+1} - r_i|^2` over all `i` and all chains.
    pub msid: Vec<f64>,
}

/// Unwrap a chain across periodic boundaries by walking min-image bonds.
/// Chain-internal geometry must not measure the box.
pub fn unwrap_chain(m: &Melt, chain: usize) -> Vec<Vec3> {
    let b = &m.chains[chain].beads;
    let mut out = Vec::with_capacity(b.len());
    let mut p = *b.first().expect("empty chain");
    out.push(p);
    for w in b.windows(2) {
        let d = m.min_image(w[0], w[1]);
        p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
        out.push(p);
    }
    out
}

fn sq_dist(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

pub fn chain_features(m: &Melt) -> ChainFeatures {
    let n_chains = m.chains.len();
    assert!(n_chains > 0, "cannot featurise an empty melt");
    let max_len = m.chains.iter().map(|c| c.len()).max().unwrap();

    let mut rg2 = 0.0;
    let mut e2e = 0.0;
    let mut contour = 0.0;
    let mut max_bond = 0.0f64;
    let mut msid_sum = vec![0.0; max_len.saturating_sub(1)];
    let mut msid_n = vec![0u64; max_len.saturating_sub(1)];

    for c in 0..n_chains {
        let u = unwrap_chain(m, c);
        let n = u.len() as f64;
        let mut com = [0.0; 3];
        for p in &u {
            for i in 0..3 {
                com[i] += p[i] / n;
            }
        }
        rg2 += u.iter().map(|p| sq_dist(*p, com)).sum::<f64>() / n / n_chains as f64;
        e2e += sq_dist(u[0], *u.last().unwrap()) / n_chains as f64;
        for w in u.windows(2) {
            let b = sq_dist(w[0], w[1]).sqrt();
            contour += b / n_chains as f64;
            max_bond = max_bond.max(b);
        }
        for k in 0..u.len().saturating_sub(1) {
            for i in 0..u.len() - k - 1 {
                msid_sum[k] += sq_dist(u[i], u[i + k + 1]);
                msid_n[k] += 1;
            }
        }
    }

    ChainFeatures {
        rg2,
        end_to_end2: e2e,
        contour_len: contour,
        bead_density: m.n_beads() as f64 / m.box_len.powi(3),
        max_bond,
        msid: msid_sum
            .iter()
            .zip(&msid_n)
            .map(|(s, n)| if *n == 0 { 0.0 } else { s / *n as f64 })
            .collect(),
    }
}

/// Absolute disagreement in each non-topological feature. `msid` collapses to
/// the largest elementwise gap, or `INFINITY` if the curves have different
/// lengths — different chain lengths are themselves a non-topological
/// difference, and no tolerance hides one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureGap {
    pub rg2: f64,
    pub end_to_end2: f64,
    pub contour_len: f64,
    pub bead_density: f64,
    pub max_bond: f64,
    pub msid: f64,
}

impl FeatureGap {
    /// The single number that decides whether the null model separates the pair.
    pub fn worst(&self) -> f64 {
        self.rg2
            .max(self.end_to_end2)
            .max(self.contour_len)
            .max(self.bead_density)
            .max(self.max_bond)
            .max(self.msid)
    }
}

pub fn feature_gap(a: &ChainFeatures, b: &ChainFeatures) -> FeatureGap {
    FeatureGap {
        rg2: (a.rg2 - b.rg2).abs(),
        end_to_end2: (a.end_to_end2 - b.end_to_end2).abs(),
        contour_len: (a.contour_len - b.contour_len).abs(),
        bead_density: (a.bead_density - b.bead_density).abs(),
        max_bond: (a.max_bond - b.max_bond).abs(),
        msid: if a.msid.len() != b.msid.len() {
            f64::INFINITY
        } else {
            a.msid
                .iter()
                .zip(b.msid.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        },
    }
}

/// Sorted multiset of every interbead min-image distance in the melt.
///
/// The cutoff-free statement of the parity tier. Any model whose energy is a
/// function of the distance matrix — which is every O(3)-invariant MLIP — cannot
/// separate two melts whose sorted distance multisets are equal, at any cutoff,
/// any depth, any body order. This avoids a radius sweep, and with it the
/// `r_cut <= box_len / 2` bound above which the minimum image is not the
/// nearest image and a wider cutoff sums over periodic copies instead of over
/// more neighbours.
pub fn distance_multiset(m: &Melt) -> Vec<f64> {
    let all: Vec<Vec3> = m.chains.iter().flat_map(|c| c.beads.iter().copied()).collect();
    let mut out = Vec::with_capacity(all.len() * (all.len().saturating_sub(1)) / 2);
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            out.push(m.min_image_dist(all[i], all[j]));
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

// ------------------------------------------------------------------- helpers

/// Rotate about `axis` by `ang` radians around the box centre (Rodrigues).
pub fn rotate(m: &Melt, axis: Vec3, ang: f64) -> Melt {
    let k = unit(axis).unwrap_or([0.0, 0.0, 1.0]);
    let c = m.box_len / 2.0;
    let (s, co) = (ang.sin(), ang.cos());
    let map = |p: Vec3| -> Vec3 {
        let v = [p[0] - c, p[1] - c, p[2] - c];
        let kv = cross(k, v);
        let kd = dot(k, v);
        let mut out = [0.0; 3];
        for a in 0..3 {
            out[a] = v[a] * co + kv[a] * s + k[a] * kd * (1.0 - co) + c;
        }
        out
    };
    Melt::new(
        m.chains
            .iter()
            .map(|ch| Chain::new(ch.beads.iter().map(|&b| map(b)).collect()))
            .collect(),
        m.box_len,
    )
}

/// Scale about the origin, box included, so the configuration stays similar.
pub fn scale(m: &Melt, c: f64) -> Melt {
    Melt::new(
        m.chains
            .iter()
            .map(|ch| Chain::new(ch.beads.iter().map(|b| [b[0] * c, b[1] * c, b[2] * c]).collect()))
            .collect(),
        m.box_len * c,
    )
}

/// Displace every bead by at most `eps`, deterministically from `seed`.
/// Rejection-sampled inside the ball so the bound is hard, not distributional.
pub fn perturb(m: &Melt, eps: f64, seed: u64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ball = || loop {
        let v: Vec3 = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        if dot(v, v) <= 1.0 {
            return [v[0] * eps, v[1] * eps, v[2] * eps];
        }
    };
    Melt::new(
        m.chains
            .iter()
            .map(|ch| {
                Chain::new(
                    ch.beads
                        .iter()
                        .map(|b| {
                            let d = ball();
                            [b[0] + d[0], b[1] + d[1], b[2] + d[2]]
                        })
                        .collect(),
                )
            })
            .collect(),
        m.box_len,
    )
}
