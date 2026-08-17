//! Melt relabelings and reconnections, for testing what a descriptor is blind to.
//!
//! This crate holds no science. It holds the smallest set of exact
//! transformations needed to ask two questions, and the answers live in
//! `tests/orient.rs`:
//!
//! * **(a) traversal reversal** — [`reverse_chain`]. Reverses one chain's bead
//!   index order. Moves no bead and changes no undirected bond.
//! * **(b) bond relabeling** — [`crossover`], [`swap_beads`],
//!   [`reverse_segment`], [`permute`]. Keeps the bead multiset and changes which
//!   beads are bonded to which.
//!
//! # The one thing to know before using any of this
//!
//! For an unoriented two-component link, only `|Lk|` is an invariant.
//! `sign(Lk)` depends on which way you chose to walk each chain, and
//! [`reverse_chain`] negates it while changing nothing physical. So a
//! demonstration that some descriptor cannot see `sign(Lk)` is not a
//! demonstration that it is blind to topology — it is a demonstration that
//! `sign(Lk)` was never an observable. Any blindness claim built on these
//! transformations must be stated in terms of `|Lk|`, or in terms of a
//! reversal-invariant quantity such as writhe.
//!
//! # Guard
//!
//! Every feature in `nerve_baseline::ChainFeatures` is computed on
//! bond-unwrapped coordinates, and unwrapping only recovers the true walk when
//! bonds are shorter than `box_len / 2`. [`max_bond`] exists so a caller can
//! check that before believing any comparison. Reconnections routinely break it
//! — a random [`permute`] almost always does — and a comparison made across a
//! broken guard is measuring the box, not the chain.

use nerve_core::{Chain, Melt, Vec3};

// ============================================ ported from `branchcut` (MIT)
//
// Ported, not depended on: `branchcut` is Python/numpy and this is Rust. Source
// is this user's own repo at `C:\Users\seal\Documents\seal\branchcut` —
// `branchcut/nogo.py::collides` and
// `branchcut/plateau.py::{group_by_gap, entropy, tolerance_sweep, has_plateau}`.
// Semantics are reproduced rather than re-derived; the reasoning below is quoted
// from there, not invented here.
//
// **Theorem 3 (branchcut, No Local Criterion Detects Non-Injectivity).** There is
// a smooth `F` with nonsingular Jacobian everywhere that is not injective:
// `F(x,y) = (e^x cos y, e^x sin y)`, `det DF = e^{2x} > 0` everywhere, yet
// `F(x,y) = F(x, y+2pi)`. The Jacobians at the two preimages differ by a rotation
// and share every spectral invariant, so no pointwise function of the Jacobian
// separates injective from many-to-one. The escape is global and cheap: compare
// two points instead of examining one. That is why the blindness searches in this
// crate are all pairwise population scans and never per-configuration diagnostics.

/// Pairs of distinct configurations whose descriptor images agree to `atol`.
///
/// Faithful port of `branchcut.nogo.collides`, specialised to pre-computed image
/// vectors because callers here already have them. `O(n^2)`, exact to `atol`,
/// with `rtol = 0` exactly as in the original.
///
/// One-sided certificate, in the original's words: an empty result is evidence on
/// these points and is not a proof of injectivity anywhere else.
pub fn collides(images: &[Vec<f64>], atol: f64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            if images[i].len() == images[j].len()
                && images[i]
                    .iter()
                    .zip(&images[j])
                    .all(|(a, b)| (a - b).abs() <= atol)
            {
                out.push((i, j));
            }
        }
    }
    out
}

/// One run of near-equal values: `(high, low, multiplicity)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GapGroup {
    pub high: f64,
    pub low: f64,
    pub mult: usize,
}

/// Group a sequence into runs separated by gaps larger than `tol`.
///
/// Port of `branchcut.plateau.group_by_gap`. Comparison is against the group's
/// **last member**, not its mean — the original notes that chaining on the mean
/// lets a slow drift merge into one group even where no two adjacent values are
/// within `tol`.
pub fn group_by_gap(values: &[f64], tol: f64) -> Vec<GapGroup> {
    assert!(!values.is_empty(), "sequence is empty");
    assert!(tol >= 0.0, "tol must be non-negative; got {tol}");
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).expect("NaN in sequence"));
    let mut groups: Vec<Vec<f64>> = vec![vec![v[0]]];
    for &x in &v[1..] {
        if (groups.last().unwrap().last().unwrap() - x).abs() <= tol {
            groups.last_mut().unwrap().push(x);
        } else {
            groups.push(vec![x]);
        }
    }
    let mean = |g: &Vec<f64>| g.iter().sum::<f64>() / g.len() as f64;
    (0..groups.len())
        .map(|i| GapGroup {
            high: mean(&groups[i]),
            low: if i + 1 < groups.len() {
                mean(&groups[i + 1])
            } else {
                f64::NEG_INFINITY
            },
            mult: groups[i].len(),
        })
        .collect()
}

/// Shannon entropy of the multiplicity distribution, in nats.
///
/// Port of `branchcut.plateau.entropy`. `0.0` means one group holds everything;
/// `ln(D)` means `D` singletons, i.e. the grouping bought nothing.
pub fn group_entropy(groups: &[GapGroup]) -> f64 {
    let total: f64 = groups.iter().map(|g| g.mult as f64).sum();
    assert!(total > 0.0, "grouping has no members");
    -groups
        .iter()
        .map(|g| {
            let p = g.mult as f64 / total;
            p * p.max(1e-300).ln()
        })
        .sum::<f64>()
}

/// One row of a tolerance sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepRow {
    pub tol: f64,
    pub n_groups: usize,
    pub max_multiplicity: usize,
    pub entropy: f64,
}

/// Group count and entropy across 13 log-spaced tolerances spanning the data's
/// own range, so the sweep is scale-free.
///
/// Port of `branchcut.plateau.tolerance_sweep` with its default `tols`.
pub fn tolerance_sweep(values: &[f64]) -> Vec<SweepRow> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in values {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let spread = if values.len() > 1 { hi - lo } else { 1.0 };
    let top = spread.max(1e-12);
    let bot = top * 1e-6;
    (0..13)
        .map(|i| {
            let tol = bot * (top / bot).powf(i as f64 / 12.0);
            let g = group_by_gap(values, tol);
            SweepRow {
                tol,
                n_groups: g.len(),
                max_multiplicity: g.iter().map(|x| x.mult).max().unwrap_or(0),
                entropy: group_entropy(&g),
            }
        })
        .collect()
}

/// Does the group count hold steady over at least `min_run` tolerances?
///
/// Port of `branchcut.plateau.has_plateau`. Both degenerate ends are rejected
/// however long they hold: one group means everything merged, and
/// `max_multiplicity == 1` means nothing did.
///
/// A tolerance justified by a plateau is defensible; one justified by tidiness is
/// how this crate previously arrived at `1e-9` and got it struck.
pub fn has_plateau(sweep: &[SweepRow], min_run: usize) -> (bool, usize) {
    assert!(min_run >= 2, "a plateau needs at least two tolerances");
    let (mut best_run, mut best_count) = (0usize, 0usize);
    let (mut run, mut current) = (0usize, usize::MAX);
    for row in sweep {
        run = if row.n_groups == current { run + 1 } else { 1 };
        current = row.n_groups;
        if current > 1 && row.max_multiplicity > 1 && run > best_run {
            best_run = run;
            best_count = current;
        }
    }
    (best_run >= min_run, best_count)
}

/// **Theorem 2 (branchcut, recovery ceiling).** If `k` inputs pool to one
/// descriptor value then for any downstream `h`, `Pr[h(f(x)) = x] <= 1/k`.
///
/// A collision is terminal: a feature-invisible reconnection is not merely hard
/// to detect downstream, it is provably unrecoverable.
pub fn recovery_ceiling(k: usize) -> f64 {
    assert!(k >= 1, "k must be at least 1");
    1.0 / k as f64
}

/// A bead position as exact bits, for identity comparisons.
///
/// Bitwise rather than tolerance-based on purpose: these transformations
/// *relabel*, they never arithmetically perturb, so two beads are either the
/// same bead or a different one. A tolerance here would let a genuine bead
/// movement pass as a relabeling.
pub type BeadKey = [u64; 3];

fn key(p: Vec3) -> BeadKey {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

/// Every bead in the melt, chain by chain.
pub fn pooled_beads(m: &Melt) -> Vec<Vec3> {
    m.chains
        .iter()
        .flat_map(|c| c.beads.iter().copied())
        .collect()
}

/// Re-split a flat bead list into chains of the original lengths.
pub fn rebuild(m: &Melt, pooled: &[Vec3]) -> Melt {
    assert_eq!(pooled.len(), m.n_beads(), "bead count must be preserved");
    let mut at = 0;
    let chains = m
        .chains
        .iter()
        .map(|c| {
            let s = &pooled[at..at + c.beads.len()];
            at += c.beads.len();
            Chain::new(s.to_vec())
        })
        .collect();
    Melt::new(chains, m.box_len)
}

/// Reverse chain `i`'s bead index order.
///
/// Maps bond `{k, k+1}` to bond `{n-2-k, n-1-k}` — the same unordered pairs. The
/// undirected bond graph and the bead multiset are preserved exactly, so this is
/// a pure change of traversal direction and, for a homopolymer, not a different
/// physical system at all.
pub fn reverse_chain(m: &Melt, i: usize) -> Melt {
    let mut chains = m.chains.clone();
    chains[i].beads.reverse();
    Melt::new(chains, m.box_len)
}

/// Swap the tails of chains `a` and `b` at index `k`.
///
/// The standard polymer-MC crossover. A genuine reconnection: the bead multiset
/// is preserved and the bond graph changes. `k` is clamped into range, and `k`
/// of 0 or the full length gives back the same graph with chain labels swapped.
///
/// ponytail: requires the two chains to be the same length, which every melt in
/// this crate's tests satisfies. Unequal lengths would change the chain length
/// profile and therefore the msid vector length, making the comparison
/// ill-posed rather than merely different.
pub fn crossover(m: &Melt, a: usize, b: usize, k: usize) -> Melt {
    let (la, lb) = (m.chains[a].beads.len(), m.chains[b].beads.len());
    assert_eq!(
        la, lb,
        "crossover needs equal chain lengths, got {la} and {lb}"
    );
    let k = k.min(la);
    let mut chains = m.chains.clone();
    let (ha, ta) = m.chains[a].beads.split_at(k);
    let (hb, tb) = m.chains[b].beads.split_at(k);
    chains[a] = Chain::new([ha, tb].concat());
    chains[b] = Chain::new([hb, ta].concat());
    Melt::new(chains, m.box_len)
}

/// Cyclic three-chain bridge: tails of `a`, `b`, `c` rotate one step round the
/// cycle at index `k`. Creates three new bonds and preserves every chain length.
///
/// The multi-chain generalisation of [`crossover`]. Requires equal chain lengths
/// for the same reason: an unequal split changes the chain-length profile, which
/// is itself observable via `contour_len` and the `msid` vector length.
pub fn triple_bridge(m: &Melt, a: usize, b: usize, c: usize, k: usize) -> Melt {
    let (la, lb, lc) = (
        m.chains[a].beads.len(),
        m.chains[b].beads.len(),
        m.chains[c].beads.len(),
    );
    assert!(
        la == lb && lb == lc,
        "triple_bridge needs equal chain lengths, got {la}, {lb}, {lc}"
    );
    assert!(
        a != b && b != c && a != c,
        "triple_bridge needs three distinct chains"
    );
    let k = k.min(la);
    let mut chains = m.chains.clone();
    let heads: Vec<Vec<Vec3>> = [a, b, c]
        .iter()
        .map(|&i| m.chains[i].beads[..k].to_vec())
        .collect();
    let tails: Vec<Vec<Vec3>> = [a, b, c]
        .iter()
        .map(|&i| m.chains[i].beads[k..].to_vec())
        .collect();
    // Tail of the next chain in the cycle attaches to this chain's head.
    for (slot, &i) in [a, b, c].iter().enumerate() {
        let next = (slot + 1) % 3;
        chains[i] = Chain::new([heads[slot].clone(), tails[next].clone()].concat());
    }
    Melt::new(chains, m.box_len)
}

/// End-bridging: chain `a`'s tail from index `j` is transferred onto the end of
/// chain `b`. One new bond. Reptation is the `j = len - 1` case.
///
/// Polydisperse by construction — `a` shortens and `b` lengthens — so unlike
/// [`crossover`] and [`triple_bridge`] it changes the chain-length profile and is
/// therefore visible to the null features without any topology being involved.
pub fn end_bridge(m: &Melt, a: usize, b: usize, j: usize) -> Melt {
    assert!(a != b, "end_bridge needs two distinct chains");
    let la = m.chains[a].beads.len();
    let j = j.clamp(1, la.saturating_sub(1).max(1));
    let mut chains = m.chains.clone();
    let (head, tail) = m.chains[a].beads.split_at(j);
    chains[a] = Chain::new(head.to_vec());
    chains[b] = Chain::new([m.chains[b].beads.clone(), tail.to_vec()].concat());
    Melt::new(chains, m.box_len)
}

/// Minimum-image lengths of bonds present in `after` but absent from `before`.
///
/// The precondition instrument for every reconnection: it says how many bonds a
/// move actually created and how long they are, so "length-preserving" can be
/// asserted rather than assumed.
///
/// ponytail: compares bonds as a set, not a multiset, so a duplicated bond
/// (two chains sharing an identical bead pair) collapses. Melts here have
/// distinct bead positions, so this cannot bite; upgrade to a counting multiset
/// if degenerate coincident beads ever appear.
pub fn new_bond_lengths(before: &Melt, after: &Melt) -> Vec<f64> {
    let old: std::collections::BTreeSet<(BeadKey, BeadKey)> =
        bond_graph(before).into_iter().collect();
    let mut out = Vec::new();
    for c in &after.chains {
        for w in c.beads.windows(2) {
            let (x, y) = (key(w[0]), key(w[1]));
            let e = if x <= y { (x, y) } else { (y, x) };
            if !old.contains(&e) {
                let d = after.min_image(w[0], w[1]);
                out.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
    }
    out
}

/// Beads per chain, in chain order.
///
/// Part of the observable signature, not a bookkeeping detail: `contour_len` sums
/// bond lengths per chain and the `msid` vector's length is `max_chain_len - 1`,
/// so any move that changes this profile is detectable by the null features
/// before topology is considered. Deliberately unsorted, so a move that merely
/// swaps two chains' lengths still registers as a change.
pub fn chain_length_profile(m: &Melt) -> Vec<usize> {
    m.chains.iter().map(|c| c.beads.len()).collect()
}

/// Move bead `idx` of chain `c` toward `target`, by at most `max_shift`.
///
/// The one move in this crate that RELOCATES a bead, so it leaves the fixed
/// bead set the connectivity tier is defined on. Anything measured through it
/// forfeits the exactly-zero ACSF residual and must report the descriptor
/// residual it actually incurs.
pub fn relocate_bead(m: &Melt, c: usize, idx: usize, target: Vec3, max_shift: f64) -> Melt {
    let mut chains = m.chains.clone();
    let p = chains[c].beads[idx];
    let d = m.min_image(p, target);
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n > 0.0 {
        let step = n.min(max_shift) / n;
        chains[c].beads[idx] = [p[0] + d[0] * step, p[1] + d[1] * step, p[2] + d[2] * step];
    }
    Melt::new(chains, m.box_len)
}

/// Exchange one bead between two chains.
pub fn swap_beads(m: &Melt, ca: usize, ia: usize, cb: usize, ib: usize) -> Melt {
    let mut chains = m.chains.clone();
    let ia = ia.min(chains[ca].beads.len() - 1);
    let ib = ib.min(chains[cb].beads.len() - 1);
    let (pa, pb) = (chains[ca].beads[ia], chains[cb].beads[ib]);
    chains[ca].beads[ia] = pb;
    chains[cb].beads[ib] = pa;
    Melt::new(chains, m.box_len)
}

/// Reverse the sub-path `i..=j` of chain `c` in place.
///
/// Unlike whole-chain [`reverse_chain`] this is *not* an isometry of the chain:
/// it preserves distances among the reversed beads and among the fixed beads,
/// but not between the two groups. That is why it moves msid while whole-chain
/// reversal does not.
pub fn reverse_segment(m: &Melt, c: usize, i: usize, j: usize) -> Melt {
    let mut chains = m.chains.clone();
    let n = chains[c].beads.len();
    let (i, j) = (i.min(n - 1), j.min(n - 1));
    if i < j {
        chains[c].beads[i..=j].reverse();
    }
    Melt::new(chains, m.box_len)
}

/// Reassign every bead by a permutation of the pooled bead list.
///
/// The naive "different bond assignment". Almost always violates
/// `max_bond < box_len / 2`, which is why it is not a usable reconnection —
/// see [`max_bond`].
pub fn permute(m: &Melt, perm: &[usize]) -> Melt {
    let pooled = pooled_beads(m);
    assert_eq!(perm.len(), pooled.len(), "permutation length mismatch");
    let moved: Vec<Vec3> = perm.iter().map(|&i| pooled[i]).collect();
    rebuild(m, &moved)
}

/// Reflect every bead through the `z = 0` plane.
///
/// An isometry, so it preserves every intra-chain distance and therefore all six
/// null features exactly, and it negates both linking number and writhe. Note
/// that on a generic bead set this is **not** a reconnection: it moves the beads.
/// It only becomes a relabeling of a fixed bead set when that set is
/// reflection-symmetric, and then the two configurations are mirror images and
/// the argument is the parity argument rather than a connectivity argument.
///
/// Exact in floating point: negating a sign bit loses nothing, so residuals from
/// this transformation are zero rather than small.
pub fn reflect_z(m: &Melt) -> Melt {
    let chains = m
        .chains
        .iter()
        .map(|c| Chain::new(c.beads.iter().map(|p| [p[0], p[1], -p[2]]).collect()))
        .collect();
    Melt::new(chains, m.box_len)
}

/// Longest minimum-image bond in the melt.
///
/// Matches `nerve_baseline::ChainFeatures::max_bond` by construction: consecutive
/// beads, minimum-image separation. Compare against `box_len / 2` before
/// believing any chain statistic.
pub fn max_bond(m: &Melt) -> f64 {
    m.chains
        .iter()
        .flat_map(|c| c.beads.windows(2))
        .map(|w| {
            let d = m.min_image(w[0], w[1]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Every bead as an exact key. Sort before comparing as a multiset.
pub fn bead_multiset(m: &Melt) -> Vec<BeadKey> {
    pooled_beads(m).into_iter().map(key).collect()
}

/// The undirected bond graph, canonicalised: each bond's endpoints sorted, then
/// the whole list sorted. Two melts have the same value iff they present the
/// same undirected bonds between the same bead positions.
pub fn bond_graph(m: &Melt) -> Vec<(BeadKey, BeadKey)> {
    let mut bonds: Vec<(BeadKey, BeadKey)> = m
        .chains
        .iter()
        .flat_map(|c| c.beads.windows(2))
        .map(|w| {
            let (a, b) = (key(w[0]), key(w[1]));
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        })
        .collect();
    bonds.sort_unstable();
    bonds
}
