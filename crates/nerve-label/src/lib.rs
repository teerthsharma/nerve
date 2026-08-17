//! Which topological label is learnable from a local descriptor?
//!
//! Cameron showed `Lk(Closure::Open)` jumps where the descriptor is smooth, which
//! is a defect in the *training target*, not in the descriptor. This crate turns
//! "learnable" into an executable criterion and ranks the candidate labels
//! against it.
//!
//! ## The criterion
//!
//! A label `L` is learnable from a descriptor `D` only if `L` is a function of
//! `D` and is stable: where `D` varies smoothly, `L` must not jump.
//!
//! Made executable by **amplification scaling**. Bracket a configuration with a
//! step `h`, and measure
//!
//! ```text
//! A(h) = |ΔL| / |ΔD|
//! ```
//!
//! * `L` continuous at that point  ⟹  `ΔL → 0` as `h → 0`.
//! * `L` discontinuous there       ⟹  `ΔL` **stays pinned at the jump height**,
//!   independent of `h`.
//!
//! The discriminator is therefore [`jump_height`]: the infimum of `ΔL` over
//! shrinking `h`. A continuous label drives it to zero; a jump floors it.
//!
//! ## A false premise of mine, recorded
//!
//! The first version of this criterion used the ratio `A(h) = |ΔL| / |ΔD|` and
//! tested whether it **diverged like `1/h`**, on the assumption that `ΔD ∝ h`.
//! That assumption is false at exactly the point of interest. At a centroid
//! separation of `box_len / 2` the minimum-image map folds symmetrically, so
//! `d(L/2 + δ) = d(L/2 - δ)` and the descriptor is **stationary**: measured `ΔD`
//! was 1.954e-14 — floating-point noise — at every `h` from 1e-3 down to 1e-9.
//! `A` was consequently pinned at ~3.4e11 rather than diverging, and a
//! divergence-ratio test read that as "no discontinuity".
//!
//! The stationarity is not a nuisance; it is the sharpest form of Cameron's
//! result. The knife edge is a point where the descriptor's derivative
//! **vanishes** while the label jumps, which is the worst possible case for
//! learnability. See `descriptor_is_stationary_at_the_knife_edge_but_not_away_from_it`.
//!
//! ## Vacuity, which this criterion invites
//!
//! A label that is **constant** has `A(h) = 0` everywhere and passes the
//! stability test perfectly while being worthless. So every stability result here
//! is paired with a non-constancy measurement on the same path, and the ranking
//! reports stability and informativeness together. They trade off, and that
//! trade-off is the actual finding.
//!
//! ## The ranking, and the number that justifies it
//!
//! Measured in this crate. Every figure below is printed by a named test.
//!
//! | candidate | stable? | informative? | estimator freedom | verdict |
//! |---|---|---|---|---|
//! | `|Lk|` closed, raw coords | jump height 1.1e-18 | z = 8.9e14 | none | **train on this** |
//! | closure-averaged `Lk` | jump height 3.3e-16 | pairwise, but closure-dominated | closure sd 0.331-0.559 | usable with the spread reported as label uncertainty |
//! | `writhe` (open, per chain) | jump height 0e0 | **z = 0.174** | none | stable because blind |
//! | `Lk(Closure::Open)` | **jump height 6.6e-3** | pairwise | closure sd 0.331-0.559 | unusable |
//!
//! **The verdict: train on `|Lk|` of closed rings on a fixed unwrapping.** It is the
//! only candidate that is both stable and informative. It jumps by exactly
//! `1.000000` and only where the closest approach is `0.025210` — a genuine strand
//! crossing, which excluded volume makes unreachable in a real melt — and its
//! largest step on a no-crossing path is `1.34e-16`.
//!
//! ## The tension this ranking does not dissolve
//!
//! Stated rather than papered over, because the two best properties belong to
//! different candidates and cannot currently be had together.
//!
//! * **Writhe is closure-free.** [`nerve_topo::writhe`] takes no `Closure` and no
//!   second chain, so it carries none of the 0.331-0.559 closure ambiguity that
//!   dominates image ambiguity (~1e-16) by fifteen orders. That is its whole appeal.
//!   The price is measured here and it is total: z = 0.174 against a `|Lk|` control
//!   of z = 8.9e14 on the same ensembles, and identically `0.0` across all nine
//!   rungs of the linking ladder, which makes Theorem 1's precondition fail outright.
//!
//! * **Knot type avoids the chirality trap only under a mirror-invariant
//!   polynomial.** The **Alexander** polynomial is mirror-invariant and genuinely
//!   global, so it qualifies. **HOMFLY does not** — it separates a knot from its
//!   mirror, which makes it a chirality witness reachable by exactly the per-bead
//!   signed volumes that detected mirror pairs at 1.7e-2 to 1.5e-1 elsewhere in this
//!   workspace. Choosing HOMFLY rebuilds the wall it was meant to cross. So if knot
//!   type enters this list it must be Alexander, and for that reason.
//!
//! * **But knot type of an open chain in a periodic melt needs a closure**, and that
//!   puts it back on the 0.331-0.559 ambiguity that writhe escapes.
//!
//! So writhe's continuity advantage and knot type's non-chirality advantage are in
//! direct opposition: the closure-free candidate carries no pairwise information,
//! and the informative global candidates all require a closure. Nothing measured
//! here resolves that, and the honest output is to name it. Closed rings sidestep it
//! entirely by needing no closure at all, which is the real reason `|Lk|` closed
//! wins — not that it is a better invariant, but that ring topology is the one
//! regime where the closure question does not arise.
//!
//! **Not evaluated:** Z1 / primitive-path kink count, and the Alexander polynomial
//! itself. Neither exists in this workspace and neither is cheap to build — Z1 needs
//! topology-preserving geometric minimisation. They are named as gaps rather than
//! guessed at.
//!
//! ## Physical versus artifactual discontinuity
//!
//! An integer invariant *must* jump somewhere. A jump is legitimate when the
//! physics changed — two strands passed through each other — and fatal when it is
//! an estimator artifact. The witness is [`min_interchain_dist`] at the jump: near
//! zero means a genuine strand crossing, comfortably positive means an artifact.

use nerve_core::{Chain, Melt, Vec3};
use nerve_topo::Closure;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const TWO_PI: f64 = std::f64::consts::TAU;

// ------------------------------------------------------------------ descriptor

/// Sorted multiset of min-image interbead distances below `r_cut`.
///
/// This is the quantity Cameron measured moving by 2.0e-9 across the knife edge,
/// so it is the right `D` for reproducing her result.
///
/// `r_cut` must satisfy `r_cut <= box_len / 2`, or the minimum image is not the
/// nearest image.
pub fn descriptor(m: &Melt, r_cut: f64) -> Vec<f64> {
    assert!(
        r_cut <= m.box_len / 2.0,
        "r_cut {r_cut} exceeds box_len/2 {}; minimum image is not nearest image",
        m.box_len / 2.0
    );
    let all: Vec<Vec3> = m.chains.iter().flat_map(|c| c.beads.iter().copied()).collect();
    let mut out = Vec::new();
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            let d = m.min_image_dist(all[i], all[j]);
            if d < r_cut {
                out.push(d);
            }
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// Largest elementwise disagreement between two descriptors; `INFINITY` if their
/// cardinalities differ, since no finite tolerance identifies those.
pub fn descriptor_delta(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

// ---------------------------------------------------------------- the criterion

/// One amplification measurement.
#[derive(Clone, Copy, Debug)]
pub struct Amp {
    pub h: f64,
    pub dd: f64,
    pub dl: f64,
    /// `dl / dd`. Diverges like `1/h` at a discontinuity.
    pub ratio: f64,
}

/// Amplification of a label against the descriptor across one bracketing step.
pub fn amplification<L: Fn(&Melt) -> f64>(a: &Melt, b: &Melt, h: f64, label: L, r_cut: f64) -> Amp {
    let dd = descriptor_delta(&descriptor(a, r_cut), &descriptor(b, r_cut));
    let dl = (label(a) - label(b)).abs();
    let ratio = if dd > 0.0 {
        dl / dd
    } else if dl > 0.0 {
        // The descriptor cannot distinguish the two configurations at all, yet the
        // label differs: the label is not a function of the descriptor here.
        f64::INFINITY
    } else {
        0.0
    };
    Amp { h, dd, dl, ratio }
}

/// Height of the jump a label exhibits across a bracketing family: the smallest
/// `ΔL` seen as `h` shrinks.
///
/// A continuous label drives this to zero with `h`. A discontinuous one floors it
/// at the jump height, because shrinking the bracket around a jump does not
/// shrink the difference across it.
///
/// This replaced a divergence-ratio test whose premise (`ΔD ∝ h`) is false at the
/// knife edge — see the module docs.
pub fn jump_height(amps: &[Amp]) -> f64 {
    amps.iter().map(|a| a.dl).fold(f64::INFINITY, f64::min)
}

/// Largest amplification ratio over the family, reported alongside the jump
/// height because it is the quantity that says how badly the label outruns the
/// descriptor.
pub fn peak_amplification(amps: &[Amp]) -> f64 {
    amps.iter().map(|a| a.ratio).fold(0.0f64, f64::max)
}

/// Fraction of sampled path steps whose amplification exceeds `threshold` — a
/// discrete estimate of the measure of the discontinuity set.
pub fn discontinuity_fraction<L: Fn(&Melt) -> f64 + Copy>(
    path: &[Melt],
    label: L,
    r_cut: f64,
    threshold: f64,
) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    let n = path.len() - 1;
    let hits = path
        .windows(2)
        .filter(|w| amplification(&w[0], &w[1], 0.0, label, r_cut).ratio > threshold)
        .count();
    hits as f64 / n as f64
}

// -------------------------------------------------------------------- labels
//
// Every candidate has the same shape `Fn(&Melt) -> f64` so the criterion applies
// to all of them unchanged.

/// `|Lk|` of chains 0 and 1 as closed polygons, on **raw in-box coordinates**.
///
/// No periodic image is chosen and no closure is invented, so this label has no
/// estimator freedom at all. Requires rings that lie wholly inside the box.
pub fn abs_lk_closed_raw(m: &Melt) -> f64 {
    nerve_topo::linking_number_closed(&m.chains[0].beads, &m.chains[1].beads).abs()
}

/// `|Lk|` of chains 0 and 1 via the periodic machinery: unwrap, place chain 1 at
/// its nearest image of chain 0, close with a direct chord.
///
/// The same topological quantity as [`abs_lk_closed_raw`] but routed through
/// nearest-image placement, which is where the knife edge lives. The pair exists
/// to separate "the closure is the problem" from "the image convention is the
/// problem".
pub fn abs_lk_closed_min_image(m: &Melt) -> f64 {
    nerve_topo::linking_number(m, 0, 1, Closure::Direct).abs()
}

/// Total writhe: the sum over chains of the closure-free open writhe.
///
/// No closure and no image choice — writhe of one chain needs neither. Real
/// valued and continuous, so a priori the strongest candidate. The question this
/// crate has to answer is whether it carries any pairwise information at all.
pub fn total_writhe(m: &Melt) -> f64 {
    (0..m.chains.len()).map(|i| nerve_topo::writhe(m, i)).sum()
}

/// Cameron's counterexample, included as the known-bad control. If the criterion
/// does not flag this one, the criterion is broken.
pub fn lk_open(m: &Melt) -> f64 {
    nerve_topo::linking_number(m, 0, 1, Closure::Open)
}

/// Mean of [`nerve_topo::closure_spread`] over `n` Fibonacci closure directions.
///
/// Tests whether averaging over the closure restores continuity. It cannot fix an
/// image-choice discontinuity, and measuring that is the point.
pub fn lk_closure_mean(m: &Melt, n: usize) -> f64 {
    nerve_topo::closure_spread(m, 0, 1, n).mean
}

// ------------------------------------------------------------------- geometry

/// Two open three-quarter arcs of radius `r`, in perpendicular planes, chain 1
/// centred `x_b` along `x` from the box origin corner region.
///
/// Open chains, because `Lk(Open)` is only defined for those and the knife edge
/// is a claim about them.
pub fn open_arc_pair(n: usize, r: f64, x_a: f64, x_b: f64, box_len: f64) -> Melt {
    let c = box_len / 2.0;
    // Three quarters of a turn, so the chains are genuinely open.
    let span = 0.75 * TWO_PI;
    let arc = |f: &dyn Fn(f64) -> Vec3| -> Chain {
        Chain::new((0..n).map(|k| f(span * k as f64 / (n - 1) as f64)).collect())
    };
    let a = arc(&|t: f64| [x_a + r * t.cos(), c + r * t.sin(), c]);
    let b = arc(&|t: f64| [x_b + r * t.cos(), c, c + r * t.sin()]);
    Melt::new(vec![a, b], box_len)
}

/// Two closed rings of radius `r` in perpendicular planes, ring 1 displaced by
/// `offset` along `x`. `|Lk| = 1` for `0 < offset < 2r` and `0` beyond.
pub fn ring_pair(n: usize, r: f64, offset: f64, box_len: f64) -> Melt {
    let c = box_len / 2.0;
    let ring = |f: &dyn Fn(f64) -> Vec3| -> Chain {
        Chain::new((0..n).map(|k| f(TWO_PI * k as f64 / n as f64)).collect())
    };
    let a = ring(&|t: f64| [c + r * t.cos(), c + r * t.sin(), c]);
    let b = ring(&|t: f64| [c + offset + r * t.cos(), c, c + r * t.sin()]);
    Melt::new(vec![a, b], box_len)
}

/// Rigidly translate one chain.
pub fn translate_chain(m: &Melt, chain: usize, d: Vec3) -> Melt {
    let mut chains = m.chains.clone();
    chains[chain] = Chain::new(
        m.chains[chain]
            .beads
            .iter()
            .map(|b| [b[0] + d[0], b[1] + d[1], b[2] + d[2]])
            .collect(),
    );
    Melt::new(chains, m.box_len)
}

/// Componentwise centroid difference between two chains, on raw coordinates.
///
/// Nearest-image placement decides each component independently, so the knife
/// edge is where a *component* of this crosses `box_len / 2` — not where the
/// scalar separation does.
pub fn centroid_delta(m: &Melt, i: usize, j: usize) -> Vec3 {
    let com = |c: &Chain| -> Vec3 {
        let n = c.len() as f64;
        let mut s = [0.0; 3];
        for b in &c.beads {
            for k in 0..3 {
                s[k] += b[k] / n;
            }
        }
        s
    };
    let (a, b) = (com(&m.chains[i]), com(&m.chains[j]));
    [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
}

/// Smallest min-image distance between beads of two different chains — the
/// witness that decides physical versus artifactual.
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

/// Largest bond in the melt on min-image displacements. Validity guard: must stay
/// below `box_len / 2`.
pub fn max_bond(m: &Melt) -> f64 {
    m.chains
        .iter()
        .flat_map(|c| c.beads.windows(2).map(|w| m.min_image_dist(w[0], w[1])))
        .fold(0.0f64, f64::max)
}

/// Spread of a label over a path, for the non-constancy check that keeps every
/// stability result from being vacuous.
pub fn label_spread<L: Fn(&Melt) -> f64>(path: &[Melt], label: L) -> (f64, f64) {
    let v: Vec<f64> = path.iter().map(label).collect();
    (
        v.iter().copied().fold(f64::INFINITY, f64::min),
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Displace every bead by at most `eps`, deterministically from `seed`.
/// Rejection sampled inside the ball so the bound is hard, not distributional.
/// Used to give an ensemble genuine shape noise.
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

// ============================================ collision bounds, ported not derived
//
// Ported from `branchcut` (Teerth Sharma, MIT, https://github.com/teerthsharma/branchcut),
// `branchcut/partition.py`. The theorems and their proofs are that package's; this
// is a transcription into Rust so `nerve` can use them without taking a Python
// dependency. Nothing here is re-derived and nothing here is claimed as new.
//
// **THEOREM 1 — Collision Error Bound** (`partition.py:9-31`). Let `R : X -> Y` be
// injective on `|X| = n` elements, and let `f : X -> Y` be any map producing `m`
// distinct values. Then `f` disagrees with `R` on at least `n - m` elements.
// Proof: a block of size `s` contributes at least `s - 1` errors, and summing over
// the `m` blocks gives `n - m`. Tight when every block contains one correct value.
//
// **THEOREM 2 — Pooling Recovery Bound** (`partition.py:33-44`). If `f` maps a
// block of `k` elements to one value then for ANY function `h`,
// `Pr[h(f(x)) = x] <= 1/k` under a uniform prior on that block. Proof: `h . f` is
// constant on the block, so it agrees with the identity on at most one of the `k`.
//
// Theorem 2 subsumes and strictly generalises what an earlier crate of mine
// claimed: that for a matched pair every `sum_i f(env_i)` agrees. That is the
// `k = 2` existence case of a bound that holds quantitatively for every `k`, and
// branchcut's own gloss is the corollary — "once `k` elements pool, no downstream
// stage recovers them; not a larger model, not a re-ranker, not a second lookup".
// That is "widening the cutoff or adding layers buys nothing", already proved.
//
// **THE PRECONDITION IS NOT OPTIONAL** (`partition.py:28-30`): "If `R` is genuinely
// many-to-one, distinct elements *should* collide and `n - m` bounds nothing."
// [`Partition::injective`] carries it and every bound returns the vacuous value
// when it is false, exactly as the source package does.

/// Theorem 1. Minimum number of wrong values, given the partition alone.
pub fn min_errors(n: usize, m: usize) -> usize {
    assert!(n >= 1, "need at least one element");
    assert!(m >= 1 && m <= n, "m must lie in [1, {n}]; got {m}");
    n - m
}

/// Theorem 1 as a rate, `(n - m) / n`. A lower bound on the true error rate.
pub fn collision_error_floor(n: usize, m: usize) -> f64 {
    min_errors(n, m) as f64 / n as f64
}

/// Theorem 2. Cap on any downstream recovery of an element from its value.
pub fn recovery_ceiling(block_size: usize) -> f64 {
    assert!(block_size >= 1, "block size must be positive");
    1.0 / block_size as f64
}

/// The blocks of a configuration set under a descriptor, with the bounds they
/// certify.
///
/// `injective` records whether the *label* map is injective on this set. Every
/// bound is meaningless without it, so it is carried on the struct rather than
/// passed at each call site and forgotten at one of them.
#[derive(Clone, Debug)]
pub struct Partition {
    pub blocks: Vec<Vec<usize>>,
    pub injective: bool,
}

impl Partition {
    /// Number of elements.
    pub fn n(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }
    /// Number of distinct descriptor values, i.e. blocks.
    pub fn m(&self) -> usize {
        self.blocks.len()
    }
    /// Size of the largest block. 1 means fully separated.
    pub fn largest(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).max().unwrap_or(0)
    }
    /// Theorem 1. **Zero on a non-injective label map, where it proves nothing.**
    pub fn certified_errors(&self) -> usize {
        if self.injective && self.n() >= 1 {
            min_errors(self.n(), self.m())
        } else {
            0
        }
    }
    /// Theorem 1 as a rate.
    pub fn certified_error_rate(&self) -> f64 {
        if self.n() == 0 {
            0.0
        } else {
            self.certified_errors() as f64 / self.n() as f64
        }
    }
    /// Theorem 2 on the largest block: the cap on any downstream recovery.
    pub fn recovery_ceiling(&self) -> f64 {
        if self.largest() == 0 {
            1.0
        } else {
            recovery_ceiling(self.largest())
        }
    }
    /// At least one block holds more than one element, and that is provable loss.
    pub fn collapsed(&self) -> bool {
        self.injective && self.largest() > 1
    }
}

/// Group configurations into blocks by descriptor value, and record whether the
/// label map is injective on them.
///
/// Two configurations share a block when their descriptors agree elementwise to
/// `atol`. Labels are compared to `atol` as well; if any two configurations share
/// a label the precondition fails, `injective` is set false, and every bound goes
/// vacuous rather than returning a number that merely looks like a score.
pub fn partition_by_descriptor<L: Fn(&Melt) -> f64>(
    configs: &[Melt],
    label: L,
    r_cut: f64,
    atol: f64,
) -> Partition {
    let ds: Vec<Vec<f64>> = configs.iter().map(|m| descriptor(m, r_cut)).collect();
    let ls: Vec<f64> = configs.iter().map(label).collect();

    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for i in 0..configs.len() {
        match blocks
            .iter_mut()
            .find(|b| descriptor_delta(&ds[i], &ds[b[0]]) <= atol)
        {
            Some(b) => b.push(i),
            None => blocks.push(vec![i]),
        }
    }

    let mut injective = true;
    for i in 0..ls.len() {
        for j in i + 1..ls.len() {
            if (ls[i] - ls[j]).abs() <= atol {
                injective = false;
            }
        }
    }
    Partition { blocks, injective }
}

/// Pairs of distinct configurations the descriptor sends to the same value.
///
/// The systematic form of hand-building a matched pair. `O(n^2)` and exact to
/// `atol`. An empty result is evidence on these configurations and is **not** a
/// proof of injectivity anywhere else — the certificate is one-sided, as in the
/// source package.
pub fn collides(configs: &[Melt], r_cut: f64, atol: f64) -> Vec<(usize, usize)> {
    let ds: Vec<Vec<f64>> = configs.iter().map(|m| descriptor(m, r_cut)).collect();
    let mut out = Vec::new();
    for i in 0..configs.len() {
        for j in i + 1..configs.len() {
            if descriptor_delta(&ds[i], &ds[j]) <= atol {
                out.push((i, j));
            }
        }
    }
    out
}

/// `p` widely spaced ring pairs, the first `threaded` of them interlocked.
///
/// The label `sum |Lk|` equals `threaded`, so sweeping `threaded` over `0..=p`
/// gives `p + 1` configurations with pairwise distinct labels — Theorem 1's
/// injectivity precondition, satisfied by construction.
///
/// Every ring is a congruent circle in both states and pairs are spaced far beyond
/// the cutoff, so the descriptor cannot see which pairs are threaded.
pub fn linking_ladder(p: usize, threaded: usize, n: usize, r: f64, spacing: f64) -> Melt {
    assert!(threaded <= p, "threaded {threaded} exceeds pair count {p}");
    let box_len = spacing * (p as f64 + 1.0);
    let mut chains = Vec::with_capacity(2 * p);
    for k in 0..p {
        let cx = spacing * (k as f64 + 0.5);
        let cy = box_len / 2.0;
        let offset = if k < threaded { r } else { 3.0 * r };
        let ring = |f: &dyn Fn(f64) -> Vec3| -> Chain {
            Chain::new((0..n).map(|q| f(TWO_PI * q as f64 / n as f64)).collect())
        };
        chains.push(ring(&|t: f64| [cx + r * t.cos(), cy + r * t.sin(), cy]));
        chains.push(ring(&|t: f64| [cx + offset + r * t.cos(), cy, cy + r * t.sin()]));
    }
    Melt::new(chains, box_len)
}

/// Total absolute linking over consecutive chain pairs `(0,1), (2,3), ...`.
pub fn total_abs_linking_by_pairs(m: &Melt) -> f64 {
    (0..m.chains.len() / 2)
        .map(|k| {
            nerve_topo::linking_number_closed(&m.chains[2 * k].beads, &m.chains[2 * k + 1].beads)
                .abs()
        })
        .sum()
}

/// Two closed rings carrying an out-of-plane wave, so that writhe is non-zero.
///
/// A planar circle has zero writhe identically, which makes any writhe continuity
/// claim on [`ring_pair`] vacuous — measured as an exact `0.000000 .. 0.000000`
/// range, and caught by the vacuity guard in
/// `writhe_is_continuous_even_across_a_genuine_strand_crossing`.
pub fn wavy_ring_pair(
    n: usize,
    r: f64,
    offset: f64,
    box_len: f64,
    amp: f64,
    waves: f64,
) -> Melt {
    let c = box_len / 2.0;
    let ring = |f: &dyn Fn(f64) -> Vec3| -> Chain {
        Chain::new((0..n).map(|k| f(TWO_PI * k as f64 / n as f64)).collect())
    };
    let a = ring(&|t: f64| [c + r * t.cos(), c + r * t.sin(), c + amp * (waves * t).sin()]);
    let b = ring(&|t: f64| {
        [c + offset + r * t.cos(), c + amp * (waves * t).sin(), c + r * t.sin()]
    });
    Melt::new(vec![a, b], box_len)
}

/// Two closed rings coiled on a torus with **matching handedness**, so the pair's
/// total writhe does not cancel.
///
/// The naive perpendicular-plane pair is mirror-related: measured `w0 = -1.901928`
/// and `w1 = +1.901928`, summing to exactly zero, which made every total-writhe
/// claim on it vacuous. Swapping two coordinates for ring 1 is an odd permutation
/// and therefore a reflection; negating the coiled component compensates, leaving a
/// rotation. Measured total writhe with the compensation: `-3.803855` at `|Lk| = 1`.
pub fn coiled_ring_pair(
    n: usize,
    r: f64,
    offset: f64,
    box_len: f64,
    coil: f64,
    k: f64,
) -> Melt {
    let c = box_len / 2.0;
    let ring = |f: &dyn Fn(f64) -> Vec3| -> Chain {
        Chain::new((0..n).map(|q| f(TWO_PI * q as f64 / n as f64)).collect())
    };
    let a = ring(&|t: f64| {
        let rho = r + coil * (k * t).cos();
        [c + rho * t.cos(), c + rho * t.sin(), c + coil * (k * t).sin()]
    });
    let b = ring(&|t: f64| {
        let rho = r + coil * (k * t).cos();
        [c + offset + rho * t.cos(), c - coil * (k * t).sin(), c + rho * t.sin()]
    });
    Melt::new(vec![a, b], box_len)
}
