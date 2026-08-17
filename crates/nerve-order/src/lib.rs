//! A hierarchy of reasons a local model cannot represent chain topology.
//!
//! Round 1 of this project produced a parity argument that was **struck**: the
//! test could not fail, because reflection preserves a distance matrix by
//! definition, and the mechanism is 43 years old. This crate does not re-argue
//! it. Instead it builds the two mechanisms that are not definitional, and it
//! states plainly which parts are prior art.
//!
//! ## Prior art, engaged up front rather than discovered late
//!
//! - **Havel, Kuntz & Crippen**, *Bull. Math. Biol.* 45 (1983); Crippen &
//!   Havel, *Distance Geometry and Molecular Conformation* (1988). A distance
//!   matrix determines a configuration only up to an **improper** transformation.
//!   Every "distance-only descriptors cannot see chirality" claim is this.
//! - **Pattanaik, Ganea, Coley et al.**, arXiv:2110.04383 (2021). The same, made
//!   explicit for modern 3D GNNs using only pairwise distances or bond angles.
//! - arXiv:2402.15864 (2024). Formal theorem: E(3)-equivariant models cannot
//!   learn chirality.
//! - **Pozdnyakov, Willatt, Bartók, Ortner, Csányi & Ceriotti**, "On the
//!   Completeness of Atomic Structure Representations", PRL 125, 166001 (2020),
//!   arXiv:2001.11696. Body-order truncation: 3-body descriptors do **not**
//!   uniquely specify an environment, and neither does the 4-body bispectrum.
//!   Their footnote records that correlations above ν=3 **are** chirality
//!   sensitive, and Fig. 2e exhibits a mirror pair sharing identical **chiral**
//!   `|X^(3)>` features. The word *achiral* does not appear in that paper; an
//!   earlier revision of this file attributed an achirality claim to Fig. 2e and
//!   that attribution is retracted. For inversion-invariance of the power spectrum
//!   the correct source is Bartók, Kondor & Csányi, *Phys. Rev. B* **87**, 184115
//!   (2013), arXiv:1209.3140, Eq. (17). Their abstract also already
//!   covers additivity: writing a global property as a sum of atom-centred
//!   contributions "mitigates, but does not eliminate, the impact of this
//!   fundamental deficiency".
//!
//! So **both** mechanisms below are prior art *as mechanisms*. Pozdnyakov's
//! paper contains no mention of polymers, chains, knots, linking number, or any
//! global topological invariant — that domain is where the contribution here
//! sits, and nowhere else.
//!
//! ## Scope, stated once and meant every time
//!
//! Claims here bind to a **named descriptor class**, never to "a cutoff MLIP".
//! NequIP, MACE and Allegro carry vector and tensor features and admit
//! parity-odd pseudoscalars, so they are *not* covered by a parity argument. The
//! point of [`Fingerprint`] carrying signed volumes is to reach them anyway.

use nerve_core::{Chain, Melt, Vec3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const TWO_PI: f64 = std::f64::consts::TAU;

fn det3(u: Vec3, v: Vec3, w: Vec3) -> f64 {
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}
fn sq_dist(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

// ---------------------------------------------------------------- fingerprints

/// A local environment as a descriptor sees it.
///
/// `dists` is the achiral part: sorted min-image neighbour distances. Any
/// function of these alone is invariant under improper transformations, which is
/// exactly the 1983 result.
///
/// `volumes` is the **parity-odd** part: the sorted multiset of signed volumes
/// `det[r_j - r_i, r_k - r_i, r_l - r_i]` over neighbour triples. This is a
/// 4-body chiral invariant. It is empty when the descriptor is configured achiral.
///
/// **Correction, and a claim withdrawn.** Earlier revisions of this file said this
/// is "strictly stronger than the standard bispectrum, which Pozdnyakov Fig. 2e
/// shows is achiral". Both halves were wrong and both are retracted:
///
/// * The word *achiral* does not appear in Pozdnyakov et al. at all. The Fig. 2e
///   caption states the pair shares identical **chiral** `|X^(3)>` features, and the
///   paper's footnote records that correlations above ν=3 **are** chirality
///   sensitive. Fig. 2e is therefore not a statement that the bispectrum is achiral.
/// * The correct source for inversion-invariance of the **power spectrum** is
///   Bartók, Kondor & Csányi, *Phys. Rev. B* **87**, 184115 (2013),
///   arXiv:1209.3140, Eq. (17) — not Pozdnyakov.
/// * "Strictly stronger" is withdrawn outright. A multiset of signed tetrahedron
///   volumes is recoverable from an unordered list of signed tetrahedra, so this
///   feature is **at most as discriminating as the chiral bispectrum**, and Fig. 2e
///   then reads as a counterexample *to* the claim rather than support for it.
///   Whether the cutoff, the radial weighting, or the use of an arbitrary bead as
///   apex breaks that containment is undetermined, so no strength ordering is
///   asserted here at all.
///
/// What survives is only what the tests measure: this feature **does** separate a
/// configuration from its mirror image (residual 1.16e-1 per bead), and that is
/// enough for the blindness results to escape the parity lineage.
///
/// A blindness result that survives non-empty `volumes` is therefore **not** a
/// parity result and is not covered by the 1983/2021/2024 lineage.
#[derive(Clone, Debug, PartialEq)]
pub struct Fingerprint {
    pub dists: Vec<f64>,
    pub volumes: Vec<f64>,
}

/// Descriptor configuration.
#[derive(Clone, Copy, Debug)]
pub struct Local {
    /// Must satisfy `r_cut <= box_len / 2`, or the minimum image is not the
    /// nearest image and a wider cutoff sums over periodic copies of one bead
    /// instead of over more neighbours.
    pub r_cut: f64,
    /// Include the parity-odd signed-volume multiset.
    pub chiral: bool,
}

/// All beads, flattened, with the chain each belongs to.
fn flat(m: &Melt) -> Vec<(usize, Vec3)> {
    m.chains
        .iter()
        .enumerate()
        .flat_map(|(ci, c)| c.beads.iter().map(move |&b| (ci, b)))
        .collect()
}

fn cmp_vec(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let c = x.total_cmp(y);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

impl Local {
    /// Fingerprint of every bead, in canonical sorted order so that any
    /// sum over beads is summed in a reproducible sequence.
    pub fn fingerprints(&self, m: &Melt) -> Vec<Fingerprint> {
        let all = flat(m);
        let mut out = Vec::with_capacity(all.len());
        for (i, &(_, a)) in all.iter().enumerate() {
            // Neighbour displacements, so distances and volumes see the same set.
            let mut nb: Vec<Vec3> = Vec::new();
            for (j, &(_, b)) in all.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = m.min_image(a, b);
                if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < self.r_cut {
                    nb.push(d);
                }
            }
            let mut dists: Vec<f64> = nb
                .iter()
                .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
                .collect();
            dists.sort_by(f64::total_cmp);

            let mut volumes = Vec::new();
            if self.chiral {
                for p in 0..nb.len() {
                    for q in p + 1..nb.len() {
                        for r in q + 1..nb.len() {
                            volumes.push(det3(nb[p], nb[q], nb[r]));
                        }
                    }
                }
                volumes.sort_by(f64::total_cmp);
            }
            out.push(Fingerprint { dists, volumes });
        }
        out.sort_by(|x, y| cmp_vec(&x.dists, &y.dists).then(cmp_vec(&x.volumes, &y.volumes)));
        out
    }
}

/// Largest disagreement between two fingerprint multisets, or `INFINITY` when
/// they differ structurally (different bead count, neighbour count, or number of
/// signed volumes). No finite tolerance makes those the same object.
pub fn fingerprint_residual(a: &[Fingerprint], b: &[Fingerprint]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.dists.len() != y.dists.len() || x.volumes.len() != y.volumes.len() {
            return f64::INFINITY;
        }
        for (p, q) in x.dists.iter().zip(y.dists.iter()) {
            worst = worst.max((p - q).abs());
        }
        for (p, q) in x.volumes.iter().zip(y.volumes.iter()) {
            worst = worst.max((p - q).abs());
        }
    }
    worst
}

/// `E = sum_i f(env_i)` for one pseudo-random nonlinear `f`, drawn from `seed`.
///
/// The point of sampling `f` rather than fixing one: the claim to be tested is
/// quantified over *all* `f`, so a test that fixes a single `f` tests almost
/// nothing. Random Fourier features over the fingerprint entries give a dense
/// enough family that a genuine difference in the environment multiset shows up.
///
/// Summed in canonical fingerprint order, so equal multisets give **bitwise**
/// equal energies rather than equal-to-tolerance.
//
// ponytail: random Fourier features over sorted entries, not a universal
// approximator. If a reviewer wants the stronger statement, replace f with a
// trained network and compare fitted energies.
pub fn sum_decomposable_energy(fps: &[Fingerprint], seed: u64) -> f64 {
    const W: usize = 16;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let w: Vec<f64> = (0..W).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let freq: Vec<f64> = (0..W).map(|_| rng.gen_range(0.1..4.0)).collect();
    let phase: Vec<f64> = (0..W).map(|_| rng.gen_range(0.0..TWO_PI)).collect();

    let f = |fp: &Fingerprint| -> f64 {
        let mut acc = 0.0;
        for (idx, v) in fp.dists.iter().chain(fp.volumes.iter()).enumerate() {
            let k = idx % W;
            acc += w[k] * (freq[k] * v + phase[k]).cos();
        }
        acc
    };
    fps.iter().map(f).sum()
}

// ------------------------------------------------------------- body order

/// Multiset of `order`-body distance correlations within `r_cut`.
///
/// `order == 2` gives sorted pair distances. `order == 3` gives sorted triangles
/// as sorted `(d_ij, d_ik, d_jk)`. Both are functions of distances only, hence
/// parity even at every order — the reason body order and parity are entangled
/// and must be reported separately.
pub fn n_body_correlation(m: &Melt, order: usize, r_cut: f64) -> Vec<Vec<f64>> {
    assert!(
        order == 2 || order == 3,
        "only 2-body and 3-body are implemented; order {order} would be combinatorially \
         explosive and is not needed for any claim here"
    );
    let all = flat(m);
    let mut out = Vec::new();
    for (i, &(_, a)) in all.iter().enumerate() {
        let nb: Vec<Vec3> = all
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &(_, b))| b)
            .filter(|&b| m.min_image_dist(a, b) < r_cut)
            .collect();
        if order == 2 {
            for &b in &nb {
                out.push(vec![m.min_image_dist(a, b)]);
            }
        } else {
            for p in 0..nb.len() {
                for q in p + 1..nb.len() {
                    let mut tri = [
                        m.min_image_dist(a, nb[p]),
                        m.min_image_dist(a, nb[q]),
                        m.min_image_dist(nb[p], nb[q]),
                    ];
                    tri.sort_by(f64::total_cmp);
                    out.push(tri.to_vec());
                }
            }
        }
    }
    out.sort_by(|x, y| cmp_vec(x, y));
    out
}

/// Largest disagreement between two correlation multisets, `INFINITY` if their
/// cardinalities differ.
pub fn correlation_residual(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.len() != y.len() {
            return f64::INFINITY;
        }
        for (p, q) in x.iter().zip(y.iter()) {
            worst = worst.max((p - q).abs());
        }
    }
    worst
}

/// Sorted multiset of signed volumes over neighbour triples within `r_cut` — the
/// lowest-order parity-odd invariant available. Same quantity the chiral
/// [`Fingerprint`] carries, flattened across beads.
pub fn signed_volume_multiset(m: &Melt, r_cut: f64) -> Vec<f64> {
    let mut v: Vec<f64> = Local { r_cut, chiral: true }
        .fingerprints(m)
        .into_iter()
        .flat_map(|f| f.volumes)
        .collect();
    v.sort_by(f64::total_cmp);
    v
}

// --------------------------------------------------------------- topology

/// Linking number of chains `i` and `j`, treated as closed rings, delegated to
/// the frozen `nerve-topo`. Not reimplemented here: round 1 had its own
/// Klenin-Langowski version, and two implementations of one quantity is how a
/// sign convention drifts unnoticed.
pub fn lk(m: &Melt, i: usize, j: usize) -> f64 {
    nerve_topo::linking_number_closed(&m.chains[i].beads, &m.chains[j].beads)
}

// --------------------------------------------------------------- geometry

/// Two corrugated rings: circles of radius `r` carrying an out-of-plane wave of
/// amplitude `amp` and `waves` periods, the second displaced by `offset` along
/// `x` and lying in a perpendicular plane.
///
/// The corrugation is not decoration. A planar circle has **all** neighbour
/// triples coplanar, so every signed volume vanishes and a chirality-sensitive
/// fingerprint is identically zero on it — the negative control would pass
/// vacuously. See `planar_rings_have_no_local_chirality`, which records this.
pub fn wavy_rings(n: usize, r: f64, offset: f64, box_len: f64, amp: f64, waves: f64) -> Melt {
    let c = box_len / 2.0;
    let ring = |f: &dyn Fn(f64) -> Vec3| -> Chain {
        Chain::new((0..n).map(|k| f(TWO_PI * k as f64 / n as f64)).collect())
    };
    let a = ring(&|t: f64| {
        [c + r * t.cos(), c + r * t.sin(), c + amp * (waves * t).sin()]
    });
    let b = ring(&|t: f64| {
        [c + offset + r * t.cos(), c + amp * (waves * t).sin(), c + r * t.sin()]
    });
    Melt::new(vec![a, b], box_len)
}

/// Reflect by exchanging `x` and `y`. Used **only** as a negative control to
/// prove the chiral fingerprint really is parity sensitive. It is not the basis
/// of any blindness claim in this crate.
pub fn mirror(m: &Melt) -> Melt {
    Melt::new(
        m.chains
            .iter()
            .map(|c| Chain::new(c.beads.iter().map(|b| [b[1], b[0], b[2]]).collect()))
            .collect(),
        m.box_len,
    )
}

/// Displace every bead by at most `eps`, deterministically from `seed`.
/// Rejection sampled inside the ball, so the bound is hard.
pub fn perturb(m: &Melt, eps: f64, seed: u64) -> Melt {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ball = || loop {
        let v: Vec3 = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] <= 1.0 {
            return [v[0] * eps, v[1] * eps, v[2] * eps];
        }
    };
    Melt::new(
        m.chains
            .iter()
            .map(|c| {
                Chain::new(
                    c.beads
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

/// Smallest min-image distance between beads of two different chains.
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

// -------------------------------------------------------------- null model

/// The six cheap non-topological features a matched pair must also match, or the
/// pair demonstrates only that the descriptor is bad at chain size.
///
/// Definitions follow `nerve-baseline`; the code is independent because this
/// crate does not depend on that one.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainFeatures {
    pub rg2: f64,
    pub end_to_end2: f64,
    pub contour_len: f64,
    pub bead_density: f64,
    /// Validity guard: every other field is computed on bond-unwrapped
    /// coordinates, and unwrapping only recovers the walk when bonds are shorter
    /// than `box_len / 2`.
    pub max_bond: f64,
    pub msid: Vec<f64>,
}

fn unwrap_chain(m: &Melt, chain: usize) -> Vec<Vec3> {
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
