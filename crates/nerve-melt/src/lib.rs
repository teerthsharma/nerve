//! nerve-melt — do the blindness mechanisms survive at real melt density?
//!
//! The tradeoff this crate refuses: *exact blindness pairs OR realistic melt
//! density, pick one*.
//!
//! # Mechanism 1: parity, and what it actually claims
//!
//! A reflected configuration has a bitwise-identical minimum-image distance
//! matrix, so any descriptor that is a function of those distances alone is
//! bitwise blind to it, while every signed topological quantity flips sign.
//!
//! Three limits are stated here because leaving them out overstates the result:
//!
//! - It binds to the **distance-only descriptor class** — ACSF, SOAP, and
//!   `nerve_baseline::Acsf`. It does **not** bind to NequIP, MACE, or Allegro,
//!   which carry vector and tensor features and admit parity-odd pseudoscalars,
//!   which is exactly what separates enantiomers. Never write "a cutoff MLIP".
//! - It proves blindness only to the **parity-odd component: the sign of the
//!   linking number.** Not `|Lk|`, not knot type.
//! - It is a **rediscovery**, not new. Distance matrices fix a configuration
//!   only up to an improper transformation: Havel, Kuntz & Crippen (1983);
//!   Pattanaik et al., arXiv:2110.04383 for GNNs; arXiv:2402.15864 for the
//!   formal statement.
//!
//! Reflection is an exact isometry, so descriptor blindness to it is true by the
//! definition of a distance-only descriptor and cannot fail — a test asserting
//! it is a regression guard, not evidence. The question that *can* fail is
//! whether such a pair is **non-vacuous**: there is only a sign to be blind to
//! if the sign itself is well defined. The instrument for that is
//! [`resolvable_fraction`], whose floor is a measured seed-noise spread;
//! [`parity_margin`] is STRUCK and must not be used for it.
//!
//! # Mechanism 2: periodic image choice
//!
//! A descriptor sees only minimum-image distances, so it cannot see which
//! periodic image carries the linking. `nerve_topo::linking_number` picks the
//! single nearest image by centroid separation and `image_spread` *bounds* the
//! resulting ambiguity rather than resolving it.
//!
//! No lattice sum is hand-rolled here. The periodic linking number is
//! an infinite image series for open chains (Panagiotou, J. Comput. Phys. 300,
//! 533 (2015), which proves it converges); a
//! naive truncation returns a cutoff-dependent number that looks fine and means
//! nothing. TEPPP's `periodic_lk` is the port to make, and is not made here.

use nerve_core::{Chain, Melt, Vec3};
use nerve_topo::Closure;

// -------------------------------------------------------------- melt fixtures

/// A melt to build. `density` is beads per unit volume; the box side follows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeltSpec {
    pub n_chains: usize,
    pub n_beads: usize,
    pub density: f64,
    pub bond: f64,
    pub seed: u64,
    /// Hard-core exclusion: no two beads closer than this. `0.0` disables it.
    ///
    /// Load-bearing, not cosmetic. Without it chain conformations do not depend
    /// on density at all — the walk is generated identically and only the box
    /// changes — so any density sweep of a single-chain quantity is vacuous by
    /// construction. See `writhe_is_density_independent_without_excluded_volume`.
    pub min_sep: f64,
}

impl MeltSpec {
    /// Kremer-Grest reference point: rho = 0.85 sigma^-3, bond ~0.97 sigma.
    pub fn kg(n_chains: usize, n_beads: usize, seed: u64) -> Self {
        Self { n_chains, n_beads, density: 0.85, bond: 0.97, seed, min_sep: 0.0 }
    }

    pub fn at_density(self, density: f64) -> Self {
        Self { density, ..self }
    }

    /// Turn on hard-core exclusion at `min_sep`. Growth is by rejection, so this
    /// has a density ceiling above which it cannot place beads at all; that
    /// ceiling is a measurement, not a constant — see [`try_melt`].
    pub fn with_excluded_volume(self, min_sep: f64) -> Self {
        Self { min_sep, ..self }
    }

    pub fn box_len(&self) -> f64 {
        ((self.n_chains * self.n_beads) as f64 / self.density).cbrt()
    }
}

/// Freely-rotating (ideal) chains at the requested density, wrapped into the box.
///
/// **This is not an MD-equilibrated Kremer-Grest melt.** Right bead density,
/// right bond length, no excluded volume: beads overlap. Equilibrating a real KG
/// melt needs molecular dynamics, which is out of scope, and inventing one would
/// be worse than saying so.
///
/// Irrelevant to the parity argument, which holds configuration-by-configuration
/// for any input. Relevant to every *statistic* measured over the melt, and
/// flagged at each place one is reported.
pub fn ideal_melt(spec: &MeltSpec) -> Melt {
    try_melt(spec).unwrap_or_else(|| {
        panic!(
            "could not place a self-avoiding melt at density {} with min_sep {}: \
             rejection growth has a ceiling below this density",
            spec.density, spec.min_sep
        )
    })
}

/// Grow the melt, returning `None` when hard-core rejection cannot place a bead.
///
/// The rejection budget is fixed (`MAX_DIRS` directions per bead, `MAX_RESTARTS`
/// chain restarts), so failure means "not reachable within this budget" rather
/// than "impossible". That is the honest form of the density ceiling: it is a
/// property of the method, and the method is stated.
///
/// ponytail: O(N) overlap check per candidate, O(N^2) per melt. A cell list is
/// the upgrade if this ever needs to run at 1e4 beads; at 1e3 it is ~1e6 checks
/// and not the bottleneck.
pub fn try_melt(spec: &MeltSpec) -> Option<Melt> {
    const MAX_DIRS: usize = 200;
    const MAX_RESTARTS: usize = 40;
    use rand::{Rng, SeedableRng};
    let l = spec.box_len();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(spec.seed);
    let wrap = |p: Vec3| [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2].rem_euclid(l)];
    // Distances go through the sanctioned function only.
    let probe = Melt::new(vec![], l);
    let clear = |placed: &[Vec3], q: Vec3| {
        spec.min_sep <= 0.0
            || placed
                .iter()
                .all(|&o| probe.min_image_dist(o, q) >= spec.min_sep)
    };
    let unit_dir = |rng: &mut rand_chacha::ChaCha8Rng| loop {
        let v: Vec3 = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        let n = norm(v);
        if n > 1e-3 && n <= 1.0 {
            return [v[0] / n, v[1] / n, v[2] / n];
        }
    };

    let mut placed: Vec<Vec3> = Vec::with_capacity(spec.n_chains * spec.n_beads);
    let mut chains = Vec::with_capacity(spec.n_chains);
    for _ in 0..spec.n_chains {
        let mut grown = None;
        for _ in 0..MAX_RESTARTS {
            let head = loop {
                let c: Vec3 = [
                    rng.gen_range(0.0..l),
                    rng.gen_range(0.0..l),
                    rng.gen_range(0.0..l),
                ];
                if clear(&placed, c) {
                    break c;
                }
                if spec.min_sep <= 0.0 {
                    break c;
                }
            };
            let mut beads = vec![head];
            let mut raw = head;
            let ok = (1..spec.n_beads).all(|_| {
                for _ in 0..MAX_DIRS {
                    let u = unit_dir(&mut rng);
                    let cand = [
                        raw[0] + spec.bond * u[0],
                        raw[1] + spec.bond * u[1],
                        raw[2] + spec.bond * u[2],
                    ];
                    let w = wrap(cand);
                    // Exclude against everything already placed and against this
                    // chain's own earlier beads, skipping the bonded neighbour.
                    if clear(&placed, w) && clear(&beads[..beads.len() - 1], w) {
                        raw = cand;
                        beads.push(w);
                        return true;
                    }
                }
                false
            });
            if ok {
                grown = Some(beads);
                break;
            }
        }
        let beads = grown?;
        placed.extend(beads.iter().copied());
        chains.push(Chain::new(beads));
    }
    Some(Melt::new(chains, l))
}

// ---------------------------------------------------- ported from `branchcut`
//
// Ported, not depended on: `branchcut` is Python (numpy-only, MIT) at
// C:\Users\seal\Documents\seal\branchcut, 31 closed-form tests passing. Theorem
// numbering below is theirs and is cited, not re-derived.
//
//   Theorem 1 (Collision Error Bound), branchcut/partition.py: if R is injective
//     on n elements and f produces m distinct values, f disagrees with R on at
//     least n - m elements. Label-free: computed from f's outputs alone.
//   Theorem 2 (Pooling Recovery Bound), same file: if f maps a block of k
//     elements to one value then for ANY h, Pr[h(f(x)) = x] <= 1/k.
//   The plateau diagnostic, branchcut/plateau.py: a grouping obtained by
//     thresholding inherits the threshold, so sweep it and require a flat run.
//   Theorem 3 (no-go), branchcut/nogo.py: no function of the local Jacobian
//     alone decides injectivity, so collisions must be found by comparison.
//     `collides` is the O(n^2) comparison form.

/// Blocks of a finite set under a map, with the bounds they certify.
///
/// `injective` records whether the *ground relation* is injective on this set.
/// Every bound is meaningless without it, so it rides on the struct rather than
/// being passed and forgotten — and when it is false the bounds return their
/// vacuous value instead of a wrong one. That is precisely the discipline the
/// STRUCK [`parity_margin`] lacked: it returned a number where it should have
/// declared itself vacuous.
#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    pub blocks: Vec<Vec<usize>>,
    pub injective: bool,
}

impl Partition {
    pub fn n(&self) -> usize {
        self.blocks.iter().map(Vec::len).sum()
    }
    pub fn m(&self) -> usize {
        self.blocks.len()
    }
    pub fn largest(&self) -> usize {
        self.blocks.iter().map(Vec::len).max().unwrap_or(0)
    }
    /// Theorem 1. Zero on a non-injective relation, where it proves nothing.
    pub fn certified_errors(&self) -> usize {
        if self.injective {
            self.n() - self.m()
        } else {
            0
        }
    }
    /// Theorem 1 as a rate — a lower bound on the true error rate.
    pub fn certified_error_rate(&self) -> f64 {
        if self.n() == 0 {
            0.0
        } else {
            self.certified_errors() as f64 / self.n() as f64
        }
    }
    /// Theorem 2 on the largest block: the cap on ANY downstream recovery.
    pub fn recovery_ceiling(&self) -> f64 {
        recovery_ceiling(self.largest().max(1))
    }
}

/// Theorem 2. Cap on recovering an element from its pooled value.
pub fn recovery_ceiling(block_size: usize) -> f64 {
    assert!(block_size >= 1, "block size must be positive");
    1.0 / block_size as f64
}

/// Pairs of descriptors the map merges, by direct comparison.
///
/// Theorem 3's consequence: injectivity is not decidable from local derivative
/// information, so collisions are found by comparing outputs. O(n^2), no
/// derivatives.
pub fn collides(vals: &[Vec<f64>], atol: f64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..vals.len() {
        for j in (i + 1)..vals.len() {
            let d = vals[i]
                .iter()
                .zip(&vals[j])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if d <= atol {
                out.push((i, j));
            }
        }
    }
    out
}

/// Connected components under `edges`, as a [`Partition`].
///
/// Blocks ordered by smallest member, members ascending, so the result is a
/// function of the edge set and not of arrival order — a partition whose bounds
/// depend on iteration order is a partition whose bounds are not auditable.
pub fn components(n: usize, edges: &[(usize, usize)], injective: bool) -> Partition {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &(a, b) in edges {
        assert!(a < n && b < n, "edge ({a},{b}) out of range for n = {n}");
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[rb] = ra;
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for v in 0..n {
        let r = find(&mut parent, v);
        groups.entry(r).or_default().push(v);
    }
    let mut blocks: Vec<Vec<usize>> = groups.into_values().collect();
    blocks.sort_by_key(|b| b[0]);
    Partition { blocks, injective }
}

/// One row of a collision-tolerance sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepRow {
    pub atol: f64,
    pub n_groups: usize,
    pub max_multiplicity: usize,
}

/// Group count across log-spaced collision tolerances.
///
/// Adapted from `branchcut`'s `tolerance_sweep`, which groups a scalar sequence
/// by gap. Here the objects are descriptor vectors, so the grouping is the
/// collision components of [`collides`] at each tolerance; the diagnostic is
/// unchanged.
pub fn collision_tolerance_sweep(vals: &[Vec<f64>], lo: f64, hi: f64, steps: usize) -> Vec<SweepRow> {
    assert!(lo > 0.0 && hi > lo && steps >= 2, "need a positive increasing range");
    (0..steps)
        .map(|k| {
            let t = lo * (hi / lo).powf(k as f64 / (steps - 1) as f64);
            let p = components(vals.len(), &collides(vals, t), true);
            SweepRow { atol: t, n_groups: p.m(), max_multiplicity: p.largest() }
        })
        .collect()
}

/// Does the group count hold steady over at least `min_run` tolerances?
///
/// Verbatim port of `branchcut`'s `has_plateau`, including both degenerate
/// rejections: a count of 1 means everything merged and a max multiplicity of 1
/// means nothing did, and neither is structure however long it holds.
pub fn has_plateau(sweep: &[SweepRow], min_run: usize) -> (bool, usize) {
    assert!(min_run >= 2, "a plateau needs at least two tolerances");
    let (mut best_run, mut best_count, mut run, mut current) = (0usize, 0usize, 0usize, usize::MAX);
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

/// Reverse the bead order of every chain — traversal reversal.
///
/// Writhe is reversal-invariant, so this produces configurations that are
/// genuinely distinct but carry equal writhe. That makes writhe non-injective on
/// a set containing both, which is the condition under which every bound above
/// must declare itself vacuous.
pub fn reverse_chains(m: &Melt) -> Melt {
    Melt::new(
        m.chains
            .iter()
            .map(|c| Chain::new(c.beads.iter().rev().copied().collect()))
            .collect(),
        m.box_len,
    )
}

// ------------------------------------------------------- LAMMPS data reader

/// Parse a molecular-style LAMMPS data file into a [`Melt`].
///
/// Written to consume Svaneborg & Everaers, Zenodo 7319837 (CC-BY-4.0):
/// MD-equilibrated Kremer-Grest melts at full density with real excluded volume,
/// 823-8408 beads/chain (M in the title is the CHAIN COUNT, 414-583 per file).
/// Loading those removes the three
/// standing caveats on this crate's statistics simultaneously.
///
/// Takes text, not a path, and does no decompression: the archive ships `.gz`,
/// and `gunzip` is one command, so no gzip dependency is added here.
///
/// Requirements, all enforced rather than assumed:
/// - the cell must be cubic, because [`Melt`] carries one `box_len`;
/// - each molecule must be a simple path, because `unwrap_chain` and every chain
///   feature assume two ends;
/// - bead order comes from the `Bonds` section, never from atom-ID order, which
///   in real files is not the order along the chain.
///
/// Image flags are parsed and ignored: wrapped coordinates plus `min_image` is
/// exactly what this workspace's unwrapping expects, and re-applying the flags
/// would double-count the shift.
pub fn from_lammps_str(s: &str) -> Result<Melt, String> {
    use std::collections::BTreeMap;

    // ---- header: cubic box only
    let mut ext = [None; 3];
    for line in s.lines() {
        for (k, tag) in ["xlo xhi", "ylo yhi", "zlo zhi"].iter().enumerate() {
            if let Some(nums) = line.strip_suffix(tag) {
                let v: Vec<f64> = nums
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                if v.len() == 2 {
                    ext[k] = Some(v[1] - v[0]);
                }
            }
        }
    }
    let dims: Vec<f64> = ext.iter().map(|e| e.ok_or("missing box bounds")).collect::<Result<_, _>>()?;
    let l = dims[0];
    if dims.iter().any(|d| (d - l).abs() > 1e-9 * l.max(1.0)) {
        return Err(format!(
            "cell is not cubic ({dims:?}); nerve_core::Melt carries a single box_len, \
             so a non-cubic cell cannot be represented"
        ));
    }

    // ---- sections
    let section = |name: &str| -> Vec<Vec<f64>> {
        let mut rows = Vec::new();
        let mut inside = false;
        for line in s.lines() {
            let t = line.trim();
            // Headers are bare words, optionally with a trailing `# comment`.
            let head = t.split('#').next().unwrap_or("").trim();
            if !head.is_empty() && head.chars().next().is_some_and(|c| c.is_alphabetic()) {
                inside = head == name;
                continue;
            }
            if inside && !t.is_empty() {
                let v: Vec<f64> = t.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if !v.is_empty() {
                    rows.push(v);
                }
            }
        }
        rows
    };

    // Atoms, molecular style: id mol type x y z [ix iy iz]
    let mut pos: BTreeMap<i64, Vec3> = BTreeMap::new();
    let mut mol: BTreeMap<i64, i64> = BTreeMap::new();
    for r in section("Atoms") {
        if r.len() < 6 {
            return Err(format!("Atoms row has {} fields, need at least 6", r.len()));
        }
        let id = r[0] as i64;
        pos.insert(id, [r[3], r[4], r[5]]);
        mol.insert(id, r[1] as i64);
    }
    if pos.is_empty() {
        return Err("no Atoms section".into());
    }

    // Bonds: id type a b. Adjacency per molecule.
    let mut adj: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for r in section("Bonds") {
        if r.len() < 4 {
            return Err(format!("Bonds row has {} fields, need 4", r.len()));
        }
        let (a, b) = (r[2] as i64, r[3] as i64);
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }

    // ---- one chain per molecule, ordered by walking the bond path
    let mut by_mol: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for (&id, &mi) in &mol {
        by_mol.entry(mi).or_default().push(id);
    }
    let mut chains = Vec::with_capacity(by_mol.len());
    for (mi, ids) in by_mol {
        let deg = |id: &i64| adj.get(id).map_or(0, |v| v.len());
        let ends: Vec<i64> = ids.iter().copied().filter(|id| deg(id) == 1).collect();
        if ids.len() == 1 {
            chains.push(Chain::new(vec![pos[&ids[0]]]));
            continue;
        }
        if ends.len() != 2 || ids.iter().any(|id| deg(id) > 2) {
            return Err(format!(
                "molecule {mi} is not a simple path: {} endpoints of degree 1, max degree {}. \
                 A ring or branch point has no two ends, and unwrap_chain needs them",
                ends.len(),
                ids.iter().map(deg).max().unwrap_or(0)
            ));
        }
        // Walk from the lower-numbered end so the order is deterministic.
        let mut order = vec![ends[0].min(ends[1])];
        let mut prev = None;
        while order.len() < ids.len() {
            let cur = *order.last().unwrap();
            let next = adj[&cur].iter().copied().find(|n| Some(*n) != prev);
            match next {
                Some(n) => {
                    prev = Some(cur);
                    order.push(n);
                }
                None => break,
            }
        }
        if order.len() != ids.len() {
            return Err(format!(
                "molecule {mi} bond path covers {} of {} atoms; the molecule is disconnected",
                order.len(),
                ids.len()
            ));
        }
        chains.push(Chain::new(order.iter().map(|id| pos[id]).collect()));
    }
    Ok(Melt::new(chains, l))
}

/// [`from_lammps_str`] on a file's contents. Decompress `.gz` first.
pub fn from_lammps_path(p: &std::path::Path) -> Result<Melt, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    from_lammps_str(&s)
}

// ============================================================ Alexander witness
//
// The escape route this crate identified in round 4: a witness that is BOTH
// non-chiral (so the per-bead pseudoscalar channel carries no information about
// it) and non-local (so no cutoff reaches it). Alexander qualifies; HOMFLY does
// not, because HOMFLY separates a knot from its mirror and is therefore a
// chirality witness reachable by the same signed volumes that separated mirror
// pairs at 1.7e-2 to 1.5e-1.
//
// # Why t = -1 specifically, and why it makes this tractable
//
// The Alexander matrix row for a crossing with over-arc `o`, incoming under-arc
// `a` and outgoing under-arc `b` is, for the two crossing signs:
//
//   positive:  a -> t,      b -> -1,   o -> 1 - t
//   negative:  a -> 1,      b -> -t,   o -> t - 1
//
// At t = -1 those become `(a: -1, b: -1, o: 2)` and `(a: 1, b: 1, o: -2)`, which
// are negatives of each other. A row scaled by -1 does not change `|det|`, so
// **at t = -1 the crossing signs cancel and need not be computed at all.** Only
// the over/under combinatorics matter. That removes the single most error-prone
// step in a knot pipeline, and it is why t = -1 is the tractable target.
//
// # The accuracy ceiling, stated up front rather than found by a reviewer
//
// `|Delta(-1)|` is the classical knot determinant. It is a genuine invariant but
// it is **not a complete one**: distinct prime knots share determinants. The
// figure-eight knot 4_1 and the (2,5) torus knot 5_1 both have determinant 5, and
// `determinant_does_not_separate_four_one_from_five_one` measures that collision
// rather than citing it. KymoKnot (BSD-3-Clause, github.com/luca-tubiana/KymoKnot,
// last commit 2018, read as reference and not depended on) detects with the pair
// t = -1 and t = -2 and that pair still fails to separate some prime knots at 8+
// crossings. Any label produced here inherits that limit.

/// One closed polygon's knot determinant `|Delta(-1)|`.
///
/// The polygon is read cyclically. Returns `None` when the projection cannot be
/// made generic — a vertex landing on a strand, or two strands parallel in
/// projection — after a bounded number of seeded reorientations, rather than
/// returning a number from a degenerate diagram.
///
/// ponytail: exact integer determinant in i128 via Bareiss. Entries are in
/// {-2,-1,0,1,2} so this is comfortable to a few tens of crossings; a knot with
/// enough crossings to overflow returns `None` rather than a wrapped value.
pub fn alexander_det_minus_one(loop_: &[Vec3]) -> Option<u128> {
    if loop_.len() < 3 {
        return Some(1);
    }
    // Fixed, seed-free reorientation ladder, so the function is deterministic.
    (0..8).find_map(|attempt| det_for_orientation(loop_, attempt))
}

fn det_for_orientation(pts: &[Vec3], attempt: usize) -> Option<u128> {
    const EPS: f64 = 1e-9;
    let n = pts.len();
    let a = attempt as f64;
    // Rotate about x then y by incommensurate angles; attempt 0 is the identity.
    let p: Vec<Vec3> = pts
        .iter()
        .map(|q| {
            let (s1, c1) = (0.7137 * a).sin_cos();
            let (y, z) = (c1 * q[1] - s1 * q[2], s1 * q[1] + c1 * q[2]);
            let (s2, c2) = (0.3110 * a).sin_cos();
            let (x, z) = (c2 * q[0] - s2 * z, s2 * q[0] + c2 * z);
            [x, y, z]
        })
        .collect();

    // (segment, parameter, crossing id, is_over)
    let mut passes: Vec<(usize, f64, usize, bool)> = Vec::new();
    let mut n_cross = 0usize;
    for i in 0..n {
        let (a0, a1) = (p[i], p[(i + 1) % n]);
        for j in (i + 1)..n {
            // Segments sharing a vertex cannot cross transversally.
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let (b0, b1) = (p[j], p[(j + 1) % n]);
            let d0 = [a1[0] - a0[0], a1[1] - a0[1]];
            let d1 = [b1[0] - b0[0], b1[1] - b0[1]];
            let den = d0[0] * d1[1] - d0[1] * d1[0];
            let scale = (d0[0].abs() + d0[1].abs()) * (d1[0].abs() + d1[1].abs());
            let r = [b0[0] - a0[0], b0[1] - a0[1]];
            if den.abs() <= EPS * scale.max(1e-30) {
                // Parallel in projection, so no transversal crossing is possible.
                // Only the collinear case is degenerate — a convex curve has many
                // anti-parallel pairs and rejecting those would reject the unknot.
                let off = r[0] * d0[1] - r[1] * d0[0];
                if off.abs() <= EPS * (d0[0].abs() + d0[1].abs()).max(1e-30) {
                    return None;
                }
                continue;
            }
            let s = (r[0] * d1[1] - r[1] * d1[0]) / den;
            let u = (r[0] * d0[1] - r[1] * d0[0]) / den;
            if !(EPS..=1.0 - EPS).contains(&s) || !(EPS..=1.0 - EPS).contains(&u) {
                // Outside, or a vertex sitting on a strand. The second case is
                // degenerate, so only reject when it is genuinely near an end.
                let near_end = |t: f64| (t > -EPS && t < EPS) || (t > 1.0 - EPS && t < 1.0 + EPS);
                if near_end(s) || near_end(u) {
                    return None;
                }
                continue;
            }
            let za = a0[2] + s * (a1[2] - a0[2]);
            let zb = b0[2] + u * (b1[2] - b0[2]);
            if (za - zb).abs() <= EPS {
                return None; // strands touch in projection
            }
            let c = n_cross;
            n_cross += 1;
            passes.push((i, s, c, za > zb));
            passes.push((j, u, c, zb > za));
        }
    }
    if n_cross == 0 {
        return Some(1); // no crossings: the unknot
    }

    // Walk the curve in order and label arcs. Arcs are the stretches between
    // consecutive under-passes, so there are exactly n_cross of them. Starting at
    // curve position 0 we are in the arc that follows the last under-pass.
    passes.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
    let mut arc = n_cross - 1;
    let (mut inn, mut out, mut over) = (vec![usize::MAX; n_cross], vec![usize::MAX; n_cross], vec![usize::MAX; n_cross]);
    for &(_, _, c, is_over) in &passes {
        if is_over {
            over[c] = arc;
        } else {
            inn[c] = arc;
            arc = (arc + 1) % n_cross;
            out[c] = arc;
        }
    }
    if inn.iter().chain(&out).chain(&over).any(|&v| v == usize::MAX) {
        return None; // a crossing missing a pass means the walk was inconsistent
    }

    // At t = -1 both crossing signs give the same row up to an overall -1, which
    // does not change |det|. So no crossing sign is computed anywhere above.
    let mut mat = vec![vec![0i128; n_cross]; n_cross];
    for c in 0..n_cross {
        mat[c][inn[c]] -= 1;
        mat[c][out[c]] -= 1;
        mat[c][over[c]] += 2;
    }
    // Delete one row and one column; any choice gives the same |det|.
    let k = n_cross - 1;
    let minor: Vec<Vec<i128>> = mat[..k].iter().map(|r| r[..k].to_vec()).collect();
    det_bareiss(minor).map(|d| d.unsigned_abs())
}

/// Fraction-free (Bareiss) integer determinant. `None` on i128 overflow rather
/// than a silently wrapped value.
fn det_bareiss(mut a: Vec<Vec<i128>>) -> Option<i128> {
    let n = a.len();
    if n == 0 {
        return Some(1);
    }
    let mut sign = 1i128;
    let mut prev = 1i128;
    for k in 0..n - 1 {
        if a[k][k] == 0 {
            let piv = (k + 1..n).find(|&r| a[r][k] != 0)?;
            a.swap(k, piv);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let t = a[i][j]
                    .checked_mul(a[k][k])?
                    .checked_sub(a[i][k].checked_mul(a[k][j])?)?;
                a[i][j] = t / prev; // exact in Bareiss
            }
        }
        prev = a[k][k];
    }
    Some(sign * a[n - 1][n - 1])
}

/// The `(p, q)` torus knot as an `n`-gon. `(2,3)` is the trefoil, `(2,5)` is 5_1,
/// `(2,7)` is 7_1; the `(2,q)` family has determinant exactly `q`.
pub fn torus_knot(p: usize, q: usize, n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            let r = 2.0 + (q as f64 * t).cos();
            [r * (p as f64 * t).cos(), r * (p as f64 * t).sin(), (q as f64 * t).sin()]
        })
        .collect()
}

/// The figure-eight knot 4_1 as an `n`-gon. Determinant 5.
pub fn figure_eight(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            let r = 2.0 + (2.0 * t).cos();
            [r * (3.0 * t).cos(), r * (3.0 * t).sin(), (4.0 * t).sin()]
        })
        .collect()
}

/// A planar circle as an `n`-gon — the unknot. Determinant 1.
pub fn circle(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            [t.cos(), t.sin(), 0.0]
        })
        .collect()
}

/// A trefoil with a contiguous arc lifted by `t` under a smooth cosine bump.
///
/// The informativeness probe. A rigid-translation family cannot test a
/// topological invariant at all — translation is an isometry, so *every*
/// invariant is exactly constant on it and such a ladder returns informativeness
/// zero for an informative witness and a blind one alike. Distinguishing the two
/// needs a path that actually crosses strands, which is what this is: as `t`
/// grows the lifted arc rises through the strand it was passing under, and at the
/// moment they touch the knot changes.
///
/// The bump is zero at both ends of the window, so the curve stays closed and no
/// artificial kink is introduced at the window boundary.
pub fn lifted_trefoil(n: usize, t: f64) -> Vec<Vec3> {
    let mut k = torus_knot(2, 3, n);
    let (a, b) = (n / 12, n / 12 + n / 6);
    // Push the arc radially through the torus hole, not up out of the surface.
    // Lifting along +z moves it away from every other strand and is an isotopy,
    // which is why the first version of this probe produced no passage at all: the
    // clearance stayed pinned at 0.54746 for every t.
    let (c, _) = centroid_extent(&k);
    let mid = k[(a + b) / 2];
    let dir = {
        let v = [c[0] - mid[0], c[1] - mid[1], 0.0];
        let n = norm(v).max(1e-12);
        [v[0] / n, v[1] / n, 0.0]
    };
    #[allow(clippy::needless_range_loop)]
    for i in a..b.min(n) {
        let u = (i - a) as f64 / (b - a) as f64;
        let bump = 0.5 * (1.0 - (std::f64::consts::TAU * u).cos());
        for d in 0..3 {
            k[i][d] += t * bump * dir[d];
        }
    }
    k
}

/// Closest approach between the lifted window and the rest of the curve.
///
/// The no-crossing witness for the probe above: a jump in a topological invariant
/// is only legitimate where this reaches zero.
pub fn lifted_window_clearance(k: &[Vec3], n: usize) -> f64 {
    let (a, b) = (n / 12, n / 12 + n / 6);
    let margin = n / 24;
    let mut best = f64::INFINITY;
    for i in a..b.min(n) {
        for (j, q) in k.iter().enumerate() {
            if j + margin >= a && j <= b + margin {
                continue;
            }
            let d = [k[i][0] - q[0], k[i][1] - q[1], k[i][2] - q[2]];
            best = best.min(norm(d));
        }
    }
    best
}

fn centroid_extent(v: &[Vec3]) -> (Vec3, f64) {
    let n = v.len() as f64;
    let mut c = [0.0; 3];
    for p in v {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    let e = v
        .iter()
        .map(|p| norm([p[0] - c[0], p[1] - c[1], p[2] - c[2]]))
        .fold(0.0f64, f64::max);
    (c, e)
}

/// A knot label for an open chain, with the closure ambiguity attached.
///
/// A single deterministic closure is not an honest label for an open chain: this
/// crate measured closure ambiguity at sd 0.331-0.559 against periodic-image
/// ambiguity of ~1e-16, fifteen orders larger. So the label is the **mode over
/// many stochastic closures together with its probability**, and a label whose
/// `prob` is near 0.5 is not a label.
#[derive(Clone, Debug, PartialEq)]
pub struct KnotLabel {
    /// Most frequent determinant across closures.
    pub modal: u128,
    /// Fraction of closures giving `modal`. Report it or do not report the label.
    pub prob: f64,
    /// How many distinct determinants the closures produced. 1 is unambiguous.
    pub distinct: usize,
    /// Closures that produced a usable diagram.
    pub resolved: usize,
}

/// Close an open chain by running both ends out to a random far point.
///
/// Stochastic closure: the standard treatment for open chains, and the only
/// honest one given the measured closure spread.
pub fn close_stochastic(open: &[Vec3], seed: u64) -> Vec<Vec3> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let d = loop {
        let v: Vec3 = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        let n = norm(v);
        if n > 1e-3 && n <= 1.0 {
            break [v[0] / n, v[1] / n, v[2] / n];
        }
    };
    let (c, e) = centroid_extent(open);
    // One far apex: the cyclic reading joins both ends to it, and at 1000x the
    // chain's own extent the two closing segments are effectively parallel rays,
    // which is the limit the stochastic-closure literature takes.
    let r = 1000.0 * e.max(1e-12);
    let mut out = open.to_vec();
    out.push([c[0] + r * d[0], c[1] + r * d[1], c[2] + r * d[2]]);
    out
}

/// Minimally-interfering closure — the deterministic comparator.
///
/// Rule (Micheletti group): if both ends are closer to the hull surface than to
/// each other, connect each end to its nearest hull point and circularise
/// outside the hull; otherwise bridge the ends directly.
///
/// ponytail: the hull is approximated by the chain's bounding sphere, so
/// "distance to the hull surface" is `extent - |p - centroid|`. This is a real
/// deviation from the published rule, which is hull geometry, and it is here only
/// to provide *a* deterministic comparator against the stochastic mode. Upgrade
/// path is a 3D convex hull; do not quote this as minimally-interfering closure
/// proper.
pub fn close_minimally_interfering(open: &[Vec3]) -> Vec<Vec3> {
    if open.len() < 3 {
        return open.to_vec();
    }
    let (c, e) = centroid_extent(open);
    let (e0, e1) = (open[0], open[open.len() - 1]);
    let d_ends = norm([e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]]);
    let d_surf = |p: Vec3| e - norm([p[0] - c[0], p[1] - c[1], p[2] - c[2]]);

    if d_surf(e0) >= d_ends || d_surf(e1) >= d_ends {
        // Ends are closer to each other than to the surface: bridge directly.
        return open.to_vec();
    }
    // Otherwise run each end out past the surface and join around the outside.
    let outward = |p: Vec3| {
        let v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let n = norm(v).max(1e-12);
        let k = 1.5 * e / n;
        [c[0] + v[0] * k, c[1] + v[1] * k, c[2] + v[2] * k]
    };
    let (o0, o1) = (outward(e0), outward(e1));
    let apex = [
        c[0] + 3.0 * (o0[0] + o1[0] - 2.0 * c[0]),
        c[1] + 3.0 * (o0[1] + o1[1] - 2.0 * c[1]),
        c[2] + 3.0 * (o0[2] + o1[2] - 2.0 * c[2]),
    ];
    let mut out = open.to_vec();
    out.extend([o1, apex, o0]);
    out
}

/// Modal determinant over `n_closures` stochastic closures.
pub fn knot_label(open: &[Vec3], n_closures: usize, seed: u64) -> KnotLabel {
    let mut tally: std::collections::BTreeMap<u128, usize> = Default::default();
    for k in 0..n_closures {
        if let Some(d) = alexander_det_minus_one(&close_stochastic(open, seed + k as u64)) {
            *tally.entry(d).or_default() += 1;
        }
    }
    let resolved: usize = tally.values().sum();
    let (modal, count) = tally
        .iter()
        .max_by_key(|&(_, c)| *c)
        .map(|(d, c)| (*d, *c))
        .unwrap_or((0, 0));
    KnotLabel {
        modal,
        prob: if resolved == 0 { 0.0 } else { count as f64 / resolved as f64 },
        distinct: tally.len(),
        resolved,
    }
}

/// Population standard deviation.
pub fn sd(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt()
}

/// Open-curve writhe of every chain, on unwrapped coordinates, via
/// `nerve_topo::writhe`.
///
/// **Scope, restated because it is easy to overclaim:** open-curve writhe is
/// **not** a topological invariant (`nerve-topo`'s own `writhe_is_not_integral`
/// holds), and it is a **single-curve** quantity — not a link invariant, so it
/// says nothing about entanglement between two chains. What it is: parity-odd,
/// reversal-invariant, closure-free, and continuous.
pub fn writhe_per_chain(m: &Melt) -> Vec<f64> {
    (0..m.chains.len()).map(|i| nerve_topo::writhe(m, i)).collect()
}

/// `(n_resolvable, n_total)` — how many mirror gaps `2|v|` clear a noise `floor`.
///
/// The gap between a value and its mirror is `v - (-v)`, so `2|v|` is the
/// separation between the two candidate signs, and comparing it against a real
/// noise floor is a sign-discrimination test rather than a magnitude test. The
/// floor must be measured and nonzero — that is the whole defect in the struck
/// [`parity_margin`], whose floor was ~1e-16.
pub fn resolvable_fraction(vals: &[f64], floor: f64) -> (usize, usize) {
    assert!(floor > 0.0, "noise floor must be positive and measured, got {floor}");
    (vals.iter().filter(|v| 2.0 * v.abs() > floor).count(), vals.len())
}

/// Per-bead signed-volume descriptor: for each bead, the sum of signed volumes
/// `det[b-a, c-a, d-a]` over triples of its `k` nearest neighbours, returned as a
/// sorted multiset over beads.
///
/// Parity-odd: reflection negates every determinant, so the sorted list reverses
/// and negates. Written here rather than imported from `nerve-order` on purpose
/// — the point is an independent replication of that crate's pooled-vs-per-bead
/// contrast, and a second implementation of the same invariant is the only way to
/// check it.
pub fn per_bead_signed_volumes(m: &Melt, k: usize) -> Vec<f64> {
    let p = flat(m);
    let mut out: Vec<f64> = p
        .iter()
        .map(|&a| {
            // k nearest by (distance, index) so ties are broken deterministically.
            let mut nb: Vec<(f64, usize)> = p
                .iter()
                .enumerate()
                .filter(|(_, &q)| q != a)
                .map(|(j, &q)| (m.min_image_dist(a, q), j))
                .collect();
            nb.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            let d: Vec<Vec3> = nb.iter().take(k).map(|&(_, j)| m.min_image(a, p[j])).collect();
            let mut s = 0.0;
            for i in 0..d.len() {
                for j in (i + 1)..d.len() {
                    for l in (j + 1)..d.len() {
                        s += det3(d[i], d[j], d[l]);
                    }
                }
            }
            s
        })
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

fn det3(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn norm(d: Vec3) -> f64 {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn map_beads(m: &Melt, f: impl Fn(Vec3) -> Vec3) -> Melt {
    Melt::new(
        m.chains
            .iter()
            .map(|c| Chain::new(c.beads.iter().copied().map(&f).collect()))
            .collect(),
        m.box_len,
    )
}

// ------------------------------------------------------------------ mechanism 1

/// Reflect the melt by negating coordinate `axis`.
///
/// Negation is exact in binary floating point (one sign bit) and `f64::round` is
/// odd, so `min_image` is exactly antisymmetric under it and every
/// minimum-image distance is preserved *bitwise*. Wrapping the result back into
/// `[0, L)` would introduce rounding and destroy that, so coordinates are left
/// outside the box; under PBC it is the same configuration.
pub fn reflect(m: &Melt, axis: usize) -> Melt {
    assert!(axis < 3, "axis must be 0, 1 or 2");
    map_beads(m, |mut p| {
        p[axis] = -p[axis];
        p
    })
}

/// **STRUCK. Do not build on this.**
///
/// The denominator here is `image_spread.max - image_spread.min`, which this
/// crate's own measurements record as ~1e-16 at every density. A predicate of
/// the form `2|Lk| > 1e-16` degenerates to *"is `|Lk|` nonzero in f64"*, so any
/// fraction built on it measures how often `Lk` is nonzero — driven, in an
/// [`ideal_melt`], by the bead overlap that generator openly has — and not
/// whether a parity pair is constructible. `2|Lk|` also discards the sign, which
/// is the only thing parity flips.
///
/// Retained, not deleted, so the struck round-2 measurement stays reproducible
/// and its provenance visible. A replacement needs a denominator that is not
/// machine noise: see [`resolvable_fraction`], whose floor is a measured
/// same-architecture seed-noise spread.
///
/// `(signed_gap, ambiguity)` for chain pair `(i, j)`.
///
/// `signed_gap` is `2 * |Lk|`, the distance between the pair and its mirror in
/// the only quantity parity blindness concerns: the sign. `ambiguity` is
/// `image_spread.max - image_spread.min`, the bound `nerve_topo` places on how
/// much of `Lk` is the periodic-image choice's doing.
///
/// The parity pair is **non-vacuous** for this chain pair only when
/// `signed_gap > ambiguity`. Otherwise the sign being hidden was never resolved
/// in the first place, and the "blindness pair" is blindness to noise.
pub fn parity_margin(m: &Melt, i: usize, j: usize, c: Closure) -> (f64, f64) {
    // Instrument consistency, not preference: `image_spread` routes
    // `Closure::Open` to the open integral and every other closure to the
    // closed one, so mixing them here would compare two different quantities
    // and the margin would be meaningless. `closure_spread` reports the
    // closure's own contribution separately.
    assert!(
        c != Closure::Open,
        "parity_margin needs a closed scheme so the value and its image bound are \
         the same integral; use Closure::Direct and read closure_spread beside it"
    );
    let gap = 2.0 * nerve_topo::linking_number(m, i, j, c).abs();
    let sp = nerve_topo::image_spread(m, i, j, c);
    (gap, sp.max - sp.min)
}

/// **STRUCK** — built on [`parity_margin`], whose denominator is machine noise.
/// See that function's note. Retained only to keep the struck round-2
/// measurement reproducible.
pub fn non_vacuous_fraction(m: &Melt, c: Closure) -> (usize, usize) {
    let n = m.chains.len();
    let mut ok = 0;
    let mut tot = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let (gap, amb) = parity_margin(m, i, j, c);
            tot += 1;
            if gap > amb {
                ok += 1;
            }
        }
    }
    (ok, tot)
}

/// `(n_non_vacuous, n_pairs)` with the **closure** ambiguity as the denominator
/// instead of the periodic-image ambiguity.
///
/// Measured at melt density, `image_spread` returns an ambiguity of ~1e-16 while
/// `closure_spread` returns ~5e-1 against `|Lk|` of ~1e-1. The image choice is
/// machine noise; the closure is the whole error budget. So this, not
/// [`non_vacuous_fraction`], is the honest denominator for a melt of *open*
/// chains, and a pair only has a quotable linking sign when
///
/// `2 * |Lk|  >  closure_spread.sd`
pub fn closure_vacuous_fraction(m: &Melt, n_dirs: usize) -> (usize, usize) {
    let n = m.chains.len();
    let mut ok = 0;
    let mut tot = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let gap = 2.0 * nerve_topo::linking_number(m, i, j, Closure::Open).abs();
            let sd = nerve_topo::closure_spread(m, i, j, n_dirs).sd;
            tot += 1;
            if gap > sd {
                ok += 1;
            }
        }
    }
    (ok, tot)
}

// ------------------------------------------------------------------ mechanism 2

/// Translate chain `i` by `v`, leaving every other chain alone.
pub fn translate_chain(m: &Melt, i: usize, v: Vec3) -> Melt {
    let mut chains = m.chains.clone();
    for p in &mut chains[i].beads {
        for k in 0..3 {
            p[k] += v[k];
        }
    }
    Melt::new(chains, m.box_len)
}

/// Largest absolute difference between the two melts' minimum-image distance
/// matrices. Zero means every distance-only descriptor is bitwise identical.
///
/// Requires identical bead layout in both melts.
pub fn min_image_max_diff(a: &Melt, b: &Melt) -> f64 {
    let (pa, pb) = (flat(a), flat(b));
    assert_eq!(pa.len(), pb.len(), "melts must have the same bead layout");
    // ponytail: O(N^2) over the full distance matrix. At 800 beads that is 320k
    // pairs, which is fine; a neighbour-list version is the upgrade if this ever
    // runs inside a sweep.
    let mut worst = 0.0f64;
    for i in 0..pa.len() {
        for j in (i + 1)..pa.len() {
            let d = (a.min_image_dist(pa[i], pa[j]) - b.min_image_dist(pb[i], pb[j])).abs();
            worst = worst.max(d);
        }
    }
    worst
}

fn flat(m: &Melt) -> Vec<Vec3> {
    m.chains.iter().flat_map(|c| c.beads.iter().copied()).collect()
}

/// Closest approach between chains `i` and `j`, minimum-image.
///
/// The no-crossing witness: a configuration that never passes through a strand
/// crossing cannot change its topology, so a discontinuous jump in a
/// "topological" number across such a move belongs to the estimator, not the
/// physics.
pub fn min_pair_distance(m: &Melt, i: usize, j: usize) -> f64 {
    let mut best = f64::INFINITY;
    for a in &m.chains[i].beads {
        for b in &m.chains[j].beads {
            best = best.min(m.min_image_dist(*a, *b));
        }
    }
    best
}

// -------------------------------------------------------------- the null model

/// The six-feature null model from `nerve-baseline`, concatenated with the
/// pooled ACSF descriptor at cutoff `r_cut`.
///
/// Layout: 24 ACSF channels (`[mean env, spread env]`) then the six chain
/// scalars — `rg2`, `end_to_end2`, `contour_len`, `bead_density`, mean MSID,
/// `max_bond`. This is the distance-only opponent both mechanisms must defeat.
pub fn null_model(m: &Melt, r_cut: f64) -> Vec<f64> {
    let mut v = nerve_baseline::Acsf::new(r_cut).melt_descriptor(m);
    let f = nerve_baseline::chain_features(m);
    let msid_mean = if f.msid.is_empty() {
        0.0
    } else {
        f.msid.iter().sum::<f64>() / f.msid.len() as f64
    };
    v.extend([f.rg2, f.end_to_end2, f.contour_len, f.bead_density, msid_mean, f.max_bond]);
    v
}

/// Number of unordered in-cutoff neighbour pairs summed over all beads — the
/// exact operation count of the ACSF angular basis.
///
/// Reported because the `ponytail:` O(N^2) marker on the neighbour search may
/// bind at melt density, and a ceiling should be stated with a number.
pub fn angular_pair_count(m: &Melt, r_cut: f64) -> u64 {
    let p = flat(m);
    p.iter()
        .map(|&c| {
            let n = p
                .iter()
                .filter(|&&q| {
                    let r = m.min_image_dist(c, q);
                    r > 0.0 && r < r_cut
                })
                .count() as u64;
            n * n.saturating_sub(1) / 2
        })
        .sum()
}

/// Cosine distance, same convention as `nerve_baseline::cosine_distance`.
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    nerve_baseline::cosine_distance(a, b)
}

#[allow(dead_code)]
fn keep_chain_import(b: Vec<Vec3>) -> Chain {
    Chain::new(b)
}
