//! nerve-baseline — the null model.
//!
//! This crate exists to try to kill the project's thesis. The thesis is that
//! topological invariants are necessary because local-cutoff descriptors are
//! blind to chain topology. The cheapest rebuttal is "widen the cutoff". This
//! crate makes that rebuttal executable: a standard local descriptor with the
//! cutoff radius as a free parameter, a sweep harness over that parameter, and
//! a set of cheap non-topological chain features that might capture the signal
//! for free.
//!
//! Nothing here trains a model. It produces descriptors and curves.

use nerve_core::{Melt, Vec3};

// ---------------------------------------------------------------- descriptor

/// Behler–Parrinello atom-centred symmetry functions (ACSF): a radial G2 basis
/// plus an angular G4 basis, both damped by the cosine cutoff function
/// `fc(r) = ½(1 + cos(π r / r_cut))`.
///
/// Why this is a fair stand-in for "standard methods": ACSFs are what ANI,
/// DeePMD and the original Behler–Parrinello networks actually feed their
/// networks, and the cosine cutoff is the same one SchNet/GAP-style models use
/// to keep forces continuous. It is strictly a function of the interatomic
/// distances and angles inside `r_cut` — which is precisely the class of
/// descriptor the project claims is topology-blind. Choosing SOAP instead would
/// change the basis but not the class: any body-ordered local expansion has the
/// same domain of dependence.
#[derive(Clone, Debug, PartialEq)]
pub struct Acsf {
    pub r_cut: f64,
    pub n_radial: usize,
    pub n_angular: usize,
}

impl Acsf {
    /// 8 radial shells, 4 angular exponents — enough resolution that the
    /// descriptor is not the bottleneck in any comparison.
    pub fn new(r_cut: f64) -> Self {
        assert!(r_cut > 0.0, "r_cut must be positive, got {r_cut}");
        Self { r_cut, n_radial: 8, n_angular: 4 }
    }

    /// Length of a per-bead environment vector.
    pub fn env_dim(&self) -> usize {
        self.n_radial + self.n_angular
    }

    /// Length of a melt-level descriptor: `[mean over beads, spread over beads]`.
    pub fn dim(&self) -> usize {
        2 * self.env_dim()
    }

    /// `fc(r) = ½(1 + cos(π r / r_cut))` for `r < r_cut`, else 0.
    ///
    /// `fc(r_cut) = fc'(r_cut) = 0`, which is what makes the descriptor — and
    /// therefore any force derived from it — continuous as a neighbour leaves
    /// the shell. A hard cutoff would be O(1) discontinuous.
    fn fc(&self, r: f64) -> f64 {
        if r >= self.r_cut {
            0.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * r / self.r_cut).cos())
        }
    }

    /// Descriptor of the environment of bead `center` in the flattened bead
    /// list `all`. Distances go through `Melt::min_image` only.
    pub fn environment(&self, melt: &Melt, all: &[Vec3], center: usize) -> Vec<f64> {
        // Hard ceiling on the sweep: above box_len/2 the minimum image is not
        // the nearest image, so a wider cutoff would sum over periodic copies
        // of the same bead instead of over more neighbours.
        assert!(
            self.r_cut <= melt.box_len / 2.0,
            "r_cut {} exceeds box_len/2 = {}: minimum-image convention no longer holds, \
             widen the box before widening the cutoff",
            self.r_cut,
            melt.box_len / 2.0
        );
        let mut out = vec![0.0; self.env_dim()];
        let c = all[center];

        // ponytail: O(N) neighbour scan per bead, O(N^2) per melt. A cell list
        // is the upgrade path if melts get past ~1e4 beads.
        let nb: Vec<(Vec3, f64)> = all
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != center)
            .filter_map(|(_, &p)| {
                let d = melt.min_image(c, p);
                let r = norm(d);
                (r > 0.0 && r < self.r_cut).then_some((d, r))
            })
            .collect();

        // G2: Gaussian shells at r_s, damped by fc.
        let eta = (self.n_radial as f64 / self.r_cut).powi(2);
        for (k, slot) in out[..self.n_radial].iter_mut().enumerate() {
            let r_s = k as f64 * self.r_cut / self.n_radial as f64;
            *slot = nb
                .iter()
                .map(|&(_, r)| (-eta * (r - r_s).powi(2)).exp() * self.fc(r))
                .sum();
        }

        // G4: angular, over unordered neighbour pairs. fc on all three legs so
        // the term vanishes smoothly however the triplet breaks up.
        for (m, slot) in out[self.n_radial..].iter_mut().enumerate() {
            let zeta = 1i32 << m; // 1, 2, 4, 8
            let mut acc = 0.0;
            for (a, &(da, ra)) in nb.iter().enumerate() {
                for &(db, rb) in &nb[a + 1..] {
                    let cos = dot(da, db) / (ra * rb);
                    let r_ab = norm([db[0] - da[0], db[1] - da[1], db[2] - da[2]]);
                    acc += (1.0 + cos).powi(zeta)
                        * self.fc(ra)
                        * self.fc(rb)
                        * self.fc(r_ab);
                }
            }
            *slot = acc * 2f64.powi(1 - zeta);
        }
        out
    }

    /// Melt-level descriptor: `[mean environment, per-bead spread]`, each of
    /// length `env_dim()`.
    ///
    /// The spread (population standard deviation across beads) is carried
    /// because a real MLIP evaluates a per-bead function and sums it, so the
    /// heterogeneity of environments is information the network has access to.
    /// Pooling by mean alone would build a weaker opponent than the real thing.
    pub fn melt_descriptor(&self, melt: &Melt) -> Vec<f64> {
        let all = beads(melt);
        let n = all.len();
        assert!(n > 0, "cannot describe an empty melt");
        let envs: Vec<Vec<f64>> = (0..n).map(|i| self.environment(melt, &all, i)).collect();

        let mut out = vec![0.0; self.dim()];
        let (mean, spread) = out.split_at_mut(self.env_dim());
        for e in &envs {
            for (m, v) in mean.iter_mut().zip(e) {
                *m += v / n as f64;
            }
        }
        for (k, s) in spread.iter_mut().enumerate() {
            let var = envs.iter().map(|e| (e[k] - mean[k]).powi(2)).sum::<f64>() / n as f64;
            *s = var.sqrt();
        }
        out
    }
}

fn norm(d: Vec3) -> f64 {
    dot(d, d).sqrt()
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Flatten a melt into a single bead list, chains in order.
pub fn beads(melt: &Melt) -> Vec<Vec3> {
    melt.chains.iter().flat_map(|c| c.beads.iter().copied()).collect()
}

// --------------------------------------------------------------------- sweep

/// The same descriptor at increasing cutoff radius. Answers "does more locality
/// substitute for topology?" with a curve.
pub fn cutoff_sweep(melt: &Melt, r_cuts: &[f64]) -> Vec<(f64, Vec<f64>)> {
    r_cuts
        .iter()
        .map(|&r| (r, Acsf::new(r).melt_descriptor(melt)))
        .collect()
}

/// Cosine distance in [0, 2]. 0 means the two descriptors are identical up to
/// scale, i.e. the descriptor cannot tell the two melts apart.
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "descriptor length mismatch");
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let c = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / (na * nb);
    1.0 - c.clamp(-1.0, 1.0)
}

/// How well the local descriptor separates two melts, as a function of cutoff.
/// This is the null-model experiment: if this curve rises with `r_cut` toward
/// the separation a topological invariant achieves, topology bought nothing.
pub fn discriminability_sweep(a: &Melt, b: &Melt, r_cuts: &[f64]) -> Vec<(f64, f64)> {
    cutoff_sweep(a, r_cuts)
        .into_iter()
        .zip(cutoff_sweep(b, r_cuts))
        .map(|((r, da), (_, db))| (r, cosine_distance(&da, &db)))
        .collect()
}

// ------------------------------------------------------------ chain features

/// Cheap non-topological chain descriptors, averaged over chains.
///
/// If these close the gap, the expensive topology is unnecessary.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainFeatures {
    /// Mean squared radius of gyration.
    pub rg2: f64,
    /// Mean squared end-to-end distance.
    pub end_to_end2: f64,
    /// Mean contour length (sum of bond lengths).
    pub contour_len: f64,
    /// Beads per unit volume.
    pub bead_density: f64,
    /// Longest min-image bond vector in the melt. **Validity guard**: every
    /// feature in this struct is computed on bond-unwrapped coordinates, and
    /// unwrapping only recovers the true walk when bonds are shorter than
    /// `box_len / 2`. Above that the min-image choice is wrong and the
    /// "chain" statistics describe a random walk regardless of the input — an
    /// ideal gas will report chain-like MSID growth. Check this before
    /// believing anything else here.
    pub max_bond: f64,
    /// Mean-square internal distance: `msid[k]` is <|r_{i+k+1} - r_i|²>,
    /// averaged over all `i` and all chains. Length = (max chain len) - 1.
    pub msid: Vec<f64>,
}

/// Unwrap a chain across periodic boundaries by walking bonds with
/// `Melt::min_image`. Chain-internal geometry must not measure the box.
pub fn unwrap_chain(melt: &Melt, chain: usize) -> Vec<Vec3> {
    let b = &melt.chains[chain].beads;
    let mut out = Vec::with_capacity(b.len());
    let mut p = *b.first().expect("empty chain");
    out.push(p);
    for w in b.windows(2) {
        let d = melt.min_image(w[0], w[1]);
        p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
        out.push(p);
    }
    out
}

pub fn chain_features(melt: &Melt) -> ChainFeatures {
    let n_chains = melt.chains.len();
    assert!(n_chains > 0, "cannot featurise an empty melt");
    let max_len = melt.chains.iter().map(|c| c.beads.len()).max().unwrap();

    let mut rg2 = 0.0;
    let mut e2e = 0.0;
    let mut contour = 0.0;
    let mut max_bond: f64 = 0.0;
    // msid_sum[k] / msid_n[k] = <|r_{i+k+1} - r_i|^2>
    let mut msid_sum = vec![0.0; max_len.saturating_sub(1)];
    let mut msid_n = vec![0u64; max_len.saturating_sub(1)];

    for c in 0..n_chains {
        let u = unwrap_chain(melt, c);
        let n = u.len() as f64;
        let mut com = [0.0; 3];
        for p in &u {
            for i in 0..3 {
                com[i] += p[i] / n;
            }
        }
        rg2 += u
            .iter()
            .map(|p| sq_dist(*p, com))
            .sum::<f64>()
            / n
            / n_chains as f64;
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
        bead_density: melt.n_beads() as f64 / melt.box_len.powi(3),
        max_bond,
        msid: msid_sum
            .iter()
            .zip(&msid_n)
            .map(|(s, n)| if *n == 0 { 0.0 } else { s / *n as f64 })
            .collect(),
    }
}

fn sq_dist(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}
