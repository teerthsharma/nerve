//! Topological invariants for polymer chains: Gauss linking number between
//! chain pairs, and writhe of a single chain.
//!
//! # What is trustworthy here, and what is not
//!
//! * [`linking_number_closed`] / [`writhe_closed`] / [`writhe_open`] operate in
//!   free space on explicit coordinates. These are the load-bearing routines
//!   and they are the ones the ground-truth tests pin to exact integers.
//! * [`linking_number`] adds two things that are *not* settled science: a
//!   closure of the open chain, and a choice of periodic image. Both are
//!   reported as spreads ([`closure_spread`], [`image_spread`]) rather than
//!   hidden. Read "Periodic boundaries" below before consuming its output.
//!
//! # Discretization
//!
//! The Gauss double integral
//!
//! ```text
//! Lk = 1/(4*pi) * INT INT ( (r1 - r2) . (dr1 x dr2) ) / |r1 - r2|^3
//! ```
//!
//! is *not* evaluated by quadrature. A polymer chain is already a polygon, and
//! for two straight segments the double integral has a closed form: it equals
//! the signed solid angle of the spherical quadrilateral spanned by the four
//! endpoint-to-endpoint vectors `r13, r14, r24, r23`. That solid angle is
//! computed exactly by Van Oosterom-Strackee (see [`solid_angle`]).
//!
//! Consequences for error behaviour:
//!
//! * There is **no quadrature error**, so no O(1/M) or O(1/M^2) convergence
//!   term. The result is exact for the polygon up to floating-point roundoff.
//! * What remains is roundoff accumulated over the O(M^2) summed segment pairs,
//!   and it does **grow with segment count**. Measured distance from the exact
//!   integer on (2,2n) torus links (`closed_curve_linking_is_integral`):
//!
//!   ```text
//!   twists \ segments      64        128        256        512       1024
//!         0            0.00e0     0.00e0     0.00e0     0.00e0     0.00e0
//!         1          1.33e-15   2.00e-15   3.77e-15   2.69e-14   4.57e-14
//!         2          2.89e-15   5.33e-15   2.13e-14   5.42e-14   2.16e-13
//!         3          2.66e-15   4.44e-15   1.82e-14   4.13e-14   2.13e-13
//!         4          3.55e-15   2.04e-14   7.11e-15   2.36e-13   8.88e-14
//!   ```
//!
//!   From M=64 to M=1024 (16x the segments, 256x the pairs) the deviation grows
//!   ~75x. That is between M and M^2, consistent with partially cancelling
//!   roundoff over O(M^2) terms; five points is not enough to claim a power law
//!   and none is claimed. **The accuracy figure is ~2e-13 at M=1024, and it is
//!   M-dependent** — do not quote a single number for a different chain length
//!   without re-measuring. It is still ~12 orders below the unit being measured,
//!   so rounding to the nearest integer is safe for closed curves at these M.
//!   The unlinked case (0 twists) is exactly `0.0`, not merely small.
//! * The remaining error is the difference between the polygon and whatever
//!   smooth curve it is meant to represent. That is a modelling choice made by
//!   whoever produced the beads, not a numerical error of this crate.
//! * There is no length-scale constant anywhere in the integrator. Degeneracy
//!   is detected by exact-zero tests (a zero-length vector normalises to NaN,
//!   which is caught), never by comparison against an epsilon. This is what
//!   makes the linking number bitwise invariant under power-of-two rescaling.
//!
//! # Which function to call
//!
//! For **open** chains in a melt — the normal case — use [`Closure::Open`]
//! ([`linking_number_open`]). Measured reasons, not taste:
//!
//! * Any deterministic closure returns an *integer*, because the closed polygon
//!   really is closed. Chord-closing truncations of a Hopf link gave exact
//!   integers at every truncation (deviation 0.0 to 5.3e-15). So a closed value
//!   is not "the open chain's linking number", it is the linking number of a
//!   curve you invented.
//! * The far-field [`Closure::Directional`] scheme disagrees with itself across
//!   directions. On 90% of a Hopf link, 64 Fibonacci directions returned
//!   `{-1: 55, 0: 8, +1: 1}` — 8 directions unlink the pair outright and 1
//!   inverts it. Only 85.9% agree with the chord closure.
//! * [`Closure::Direct`] agreed with the closed ground truth on every case
//!   tested here and is the best of the closures, but it still fabricates a
//!   chord across the whole chain.
//!
//! Use [`closure_spread`] on real data before trusting any closed value.
//!
//! # Periodic boundaries
//!
//! `min_image` is correct for *distances* and is used here only for that: bond
//! reconstruction and the bounding-sphere prefilter. Applying it per segment
//! inside the Gauss integral would destroy the topology, so it is not done.
//! Instead each chain is unwrapped into continuous coordinates
//! ([`unwrap_chain`]) and the integral runs on those.
//!
//! For a *pair*, a periodic image of the second chain must still be chosen.
//! **This is genuinely unresolved.** The mathematically correct object is a
//! infinite image sum whose naive truncation depends on the
//! cutoff; the published treatment is Panagiotou, *J. Comput. Phys.* 300, 533
//! (2015), implemented in TEPPP as `periodic_lk`. That is
//! **not** implemented here. [`linking_number`] uses the single nearest image
//! by centroid separation, and [`image_spread`] exists so a caller can measure
//! how much that choice is worth on their own data instead of trusting it.
//!
//! # Open chains
//!
//! Gauss linking of open chains is real-valued and is not a topological
//! invariant: it varies continuously under deformation. Nothing here rounds
//! it, and the integer assertion in the test suite applies only to closed
//! curves.

use nerve_core::{Chain, Melt, Vec3};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

// --------------------------------------------------------------------- vectors

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: Vec3) -> Vec3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}
fn finite(a: Vec3) -> bool {
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite()
}

fn centroid(v: &[Vec3]) -> Vec3 {
    if v.is_empty() {
        return [0.0; 3];
    }
    let n = v.len() as f64;
    let mut s = [0.0; 3];
    for p in v {
        s = add(s, *p);
    }
    mul(s, 1.0 / n)
}

/// Largest distance from the centroid — the bounding-sphere radius.
fn extent(v: &[Vec3]) -> f64 {
    let c = centroid(v);
    v.iter().map(|p| norm(sub(*p, c))).fold(0.0, f64::max)
}

// ---------------------------------------------------------------- solid angle

/// Signed solid angle subtended at the origin by the spherical triangle with
/// vertices along `a`, `b`, `c`, via Van Oosterom-Strackee.
///
/// The inputs are normalised first. That is not cosmetic: with unit vectors the
/// numerator and denominator are bitwise unchanged when every coordinate is
/// scaled by a power of two, which is what makes the linking number *bitwise*
/// scale invariant rather than merely nearly so.
///
/// Degenerate input (a zero-length vertex vector) normalises to NaN and returns
/// exactly `0.0`. Coplanar vertices give a zero numerator and therefore exactly
/// `0.0`, which is the correct Gauss-integral contribution for coplanar
/// segments — no special case is needed.
pub fn solid_angle(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (a, b, c) = (unit(a), unit(b), unit(c));
    if !finite(a) || !finite(b) || !finite(c) {
        return 0.0;
    }
    let num = dot(a, cross(b, c));
    let den = 1.0 + dot(a, b) + dot(a, c) + dot(b, c);
    2.0 * num.atan2(den)
}

/// Exact contribution of one segment pair to `4*pi*Lk`.
///
/// Segment 1 runs `r1 -> r2`, segment 2 runs `r3 -> r4`. The value is the
/// signed area of the spherical quadrilateral `(r13, r14, r24, r23)`, fanned
/// into two triangles from `r13`.
///
/// The trailing negation fixes the orientation of that fan to the standard
/// Gauss convention, `Lk = 1/(4pi) INT INT (r1-r2).(dr1 x dr2)/|r1-r2|^3`. It
/// is not cosmetic and it is not guesswork: without it this function returns
/// `-Lk`. The ground-truth tests cannot detect that (they assert `|Lk|`) and
/// neither can the reflection test (antisymmetric either way). It was caught
/// only by `linking_number_matches_midpoint_quadrature`, which evaluates the
/// same integral by an independent midpoint rule. Do not "simplify" this sign
/// away without re-running that test.
///
/// Confidence note: this sign is pinned **empirically**, against an independent
/// implementation of the same integral. It has not been derived analytically
/// from the orientation correspondence between the Gauss integral and the
/// closed-form spherical quadrilateral. An independent review of this file was
/// able to confirm the Van Oosterom-Strackee numerator/denominator and that the
/// quadrilateral fan is orientation-consistent, but explicitly could not confirm
/// the overall sign from first principles. Treat the global sign as
/// convention-checked, not proven.
fn omega(r1: Vec3, r2: Vec3, r3: Vec3, r4: Vec3) -> f64 {
    let r13 = sub(r3, r1);
    let r14 = sub(r4, r1);
    let r24 = sub(r4, r2);
    let r23 = sub(r3, r2);
    -(solid_angle(r13, r14, r24) + solid_angle(r13, r24, r23))
}

// ------------------------------------------------------------- free space core

/// Gauss linking number of two **closed** polygons in free space.
///
/// Each slice is treated as a cyclic polygon: the segment from the last bead
/// back to the first is included. Returns `0.0` for degenerate input (fewer
/// than 3 beads).
///
/// For genuinely closed curves this is an integer to within ~1e-15.
pub fn linking_number_closed(a: &[Vec3], b: &[Vec3]) -> f64 {
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..a.len() {
        let (p1, p2) = (a[i], a[(i + 1) % a.len()]);
        for j in 0..b.len() {
            let (p3, p4) = (b[j], b[(j + 1) % b.len()]);
            s += omega(p1, p2, p3, p4);
        }
    }
    s / FOUR_PI
}

/// Gauss double integral over two **open** polygons — no closure, no parameter.
///
/// This is the closure-free pairwise linking signal, and for open chains in a
/// melt it is the one to prefer: [`Closure::Direct`] and
/// [`Closure::Directional`] both invent a piece of curve that is not in the
/// data, and the directional scheme demonstrably returns different integers for
/// different directions on the same configuration.
///
/// **Real-valued. Not an integer. Not a topological invariant.** It varies
/// continuously as the chains move, which is exactly why it is usable as a
/// descriptor and unusable as a label. Do not round it, and do not describe it
/// as "the topology".
///
/// It does agree with [`linking_number_closed`] in the limit where the chains
/// are nearly closed, and it is symmetric, isometry invariant, sign-antisymmetric
/// under reflection, and bitwise invariant under power-of-two rescaling, exactly
/// like the closed version.
pub fn linking_number_open(a: &[Vec3], b: &[Vec3]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..a.len() - 1 {
        for j in 0..b.len() - 1 {
            s += omega(a[i], a[i + 1], b[j], b[j + 1]);
        }
    }
    s / FOUR_PI
}

/// Writhe of a **closed** polygon (cyclic: last bead joins the first).
///
/// Writhe is real-valued and is not a topological invariant. Do not round it.
pub fn writhe_closed(a: &[Vec3]) -> f64 {
    let n = a.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            s += omega(a[i], a[(i + 1) % n], a[j], a[(j + 1) % n]);
        }
    }
    2.0 * s / FOUR_PI
}

/// Writhe of an **open** polygon — no closure, no free parameter.
///
/// The self-integral over an open curve converges, so writhe is the one signal
/// in this crate that carries no closure ambiguity at all. Prefer it over
/// closing a chain just to get a writhe.
pub fn writhe_open(a: &[Vec3]) -> f64 {
    let n = a.len();
    if n < 3 {
        return 0.0;
    }
    let nseg = n - 1;
    let mut s = 0.0;
    for i in 0..nseg {
        for j in (i + 1)..nseg {
            s += omega(a[i], a[i + 1], a[j], a[j + 1]);
        }
    }
    2.0 * s / FOUR_PI
}

// ------------------------------------------------------------------- closures

/// How an open chain is turned into a closed loop.
///
/// `Direct` is the default and the one to reach for. `Directional` exists to
/// answer "is my answer an artifact of the closure?", via [`closure_spread`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Closure {
    /// Do not close at all — integrate the un-closed Gauss double integral over
    /// the two open curves.
    ///
    /// The only scheme here with no closure ambiguity whatsoever, and therefore
    /// the one to prefer for open chains in a melt. The price is that the result
    /// is real-valued and is *not* a topological invariant: it varies
    /// continuously under deformation. That is a feature for a learned
    /// descriptor and a defect for a topological label. Never round it.
    Open,
    /// Join the last bead to the first with a single chord.
    ///
    /// Chosen as primary because it is the only scheme here that is
    /// parameter-free, scale-free, deterministic, and — decisively — keeps the
    /// closure inside the chain's own convex hull. That last property means two
    /// chains with separated hulls are *provably* unlinked, which is what makes
    /// the bounding-sphere prefilter in [`all_pairs_linking`] exact rather than
    /// heuristic. Far-field closures do not have it.
    ///
    /// Its weakness: for a strongly open chain the chord is a large arbitrary
    /// piece of curve, and the resulting number is closure-dominated.
    Direct,
    /// Run each end out to distance `r_scale * extent` along `dir`, join across,
    /// and come back. A deterministic stand-in for the stochastic-closure family
    /// (as in KymoKnot's minimally-interfering closure, which is best practice
    /// and is *not* implemented here). Sweeping `dir` over the sphere gives the
    /// closure spread.
    Directional { dir: Vec3, r_scale: f64 },
}

/// Apply a closure, returning a polygon to be read cyclically.
pub fn closed_form(beads: &[Vec3], c: Closure) -> Vec<Vec3> {
    match c {
        Closure::Open | Closure::Direct => beads.to_vec(),
        Closure::Directional { dir, r_scale } => {
            if beads.len() < 2 {
                return beads.to_vec();
            }
            let u = unit(dir);
            if !finite(u) {
                return beads.to_vec();
            }
            let r = extent(beads) * r_scale;
            let (start, end) = (beads[0], beads[beads.len() - 1]);
            let mut out = beads.to_vec();
            out.push(add(end, mul(u, r)));
            out.push(add(start, mul(u, r)));
            out
        }
    }
}

/// `n` directions spread over the unit sphere by the Fibonacci lattice.
///
/// Deterministic by construction, so a closure spread needs no seed and is
/// reproducible across runs and machines.
pub fn fibonacci_directions(n: usize) -> Vec<Vec3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|k| {
            let z = if n <= 1 {
                0.0
            } else {
                1.0 - 2.0 * (k as f64) / ((n - 1) as f64)
            };
            let r = (1.0 - z * z).max(0.0).sqrt();
            let t = golden * (k as f64);
            let v = [r * t.cos(), r * t.sin(), z];
            unit(v)
        })
        .collect()
}

// ------------------------------------------------------------------------ PBC

/// Rebuild a chain as a continuous curve, following minimum-image bonds.
///
/// This is the sanctioned use of `min_image`: reconstructing bond vectors. Bead
/// 0 is left where it is, so the result is the true chain shape translated by
/// some box vector. Pair placement removes that translation.
///
/// Assumes every bond is shorter than `box_len / 2`. A chain with a bond longer
/// than half the box cannot be unwrapped from wrapped coordinates by any means
/// — the information is gone — and this function will silently pick the wrong
/// image. Bond length is a property of the simulation, not of this crate, so it
/// is the caller's job to know it holds.
pub fn unwrap_chain(m: &Melt, c: &Chain) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(c.len());
    if c.beads.is_empty() {
        return out;
    }
    out.push(c.beads[0]);
    for k in 1..c.beads.len() {
        let d = m.min_image(c.beads[k - 1], c.beads[k]);
        let prev = out[k - 1];
        out.push(add(prev, d));
    }
    out
}

/// Translate `b` onto the periodic image nearest `a`, by centroid separation.
///
/// ponytail: single nearest image. The correct object is Panagiotou's periodic
/// linking number (an infinite image series for open chains); upgrade path is to
/// port TEPPP's `periodic_lk`. Until then [`image_spread`] bounds the error.
///
/// Knife edge: if the centroid separation along an axis is exactly `box_len/2`,
/// the choice of image is a coin flip decided by `f64::round`'s tie rule. The
/// result is still deterministic, but it is discontinuous in the input there.
fn place_near(m: &Melt, a: &[Vec3], b: &[Vec3]) -> Vec<Vec3> {
    let (ca, cb) = (centroid(a), centroid(b));
    let target = add(ca, m.min_image(ca, cb));
    let shift = sub(target, cb);
    b.iter().map(|p| add(*p, shift)).collect()
}

/// Linking number of chains `i` and `j` in a periodic melt.
///
/// Both chains are unwrapped, `j` is moved to its nearest image of `i`, both are
/// closed, and the free-space integral runs on the result.
///
/// **Real-valued, not an integer, not a topological invariant** unless the
/// chains were already closed. See the module docs on periodic boundaries
/// before trusting the image choice.
pub fn linking_number(m: &Melt, i: usize, j: usize, c: Closure) -> f64 {
    let a = unwrap_chain(m, &m.chains[i]);
    let b = unwrap_chain(m, &m.chains[j]);
    let b = place_near(m, &a, &b);
    match c {
        Closure::Open => linking_number_open(&a, &b),
        _ => linking_number_closed(&closed_form(&a, c), &closed_form(&b, c)),
    }
}

/// Writhe of chain `i` in a periodic melt, on unwrapped coordinates.
///
/// Takes no closure argument: open writhe needs none, and there is no periodic
/// image to choose for a single chain. This is the cleanest topological signal
/// this crate produces.
pub fn writhe(m: &Melt, i: usize) -> f64 {
    writhe_open(&unwrap_chain(m, &m.chains[i]))
}

// -------------------------------------------------------------------- spreads

/// min / max / mean / population sd of a set of samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub sd: f64,
}

fn stats(v: &[f64]) -> Spread {
    if v.is_empty() {
        return Spread {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            sd: 0.0,
        };
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    Spread {
        min: v.iter().copied().fold(f64::INFINITY, f64::min),
        max: v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean,
        sd: var.sqrt(),
    }
}

/// How much of the linking number is the closure's doing.
///
/// Sweeps `n` Fibonacci directions of [`Closure::Directional`]. A large `sd`
/// relative to `mean` means the number is a closure artifact and should not be
/// reported as a property of the chains.
pub fn closure_spread(m: &Melt, i: usize, j: usize, n: usize) -> Spread {
    // r_scale is dimensionless (multiplies the chain's own extent), so this
    // stays scale-free. 200 puts the detour far enough out that its own
    // contribution is ~1e-5 of a linking unit.
    let vals: Vec<f64> = fibonacci_directions(n)
        .into_iter()
        .map(|dir| {
            linking_number(
                m,
                i,
                j,
                Closure::Directional {
                    dir,
                    r_scale: 200.0,
                },
            )
        })
        .collect();
    stats(&vals)
}

/// How much of the linking number is the periodic image choice's doing.
///
/// Evaluates the pair against the 27 nearest images of chain `j`. This is a
/// *diagnostic bound on an unresolved ambiguity*, not a periodic linking
/// number — see the module docs. If `max - min` is comparable to the value
/// itself, the number is meaningless for that pair.
pub fn image_spread(m: &Melt, i: usize, j: usize, c: Closure) -> Spread {
    let a = unwrap_chain(m, &m.chains[i]);
    let b0 = place_near(m, &a, &unwrap_chain(m, &m.chains[j]));
    let ca = closed_form(&a, c);
    let l = m.box_len;
    let mut vals = Vec::with_capacity(27);
    for kx in -1..=1i32 {
        for ky in -1..=1i32 {
            for kz in -1..=1i32 {
                let t = [kx as f64 * l, ky as f64 * l, kz as f64 * l];
                let b: Vec<Vec3> = b0.iter().map(|p| add(*p, t)).collect();
                // Route Open to the open integral. Reading an unclosed chain
                // cyclically silently closes it: this returned the *closed*
                // integer for `Closure::Open` (3.0000 where open |Lk| = 2.3163).
                vals.push(match c {
                    Closure::Open => linking_number_open(&a, &b),
                    _ => linking_number_closed(&ca, &closed_form(&b, c)),
                });
            }
        }
    }
    stats(&vals)
}

// ------------------------------------------------------------------ all pairs

/// Linking number of every chain pair, with a bounding-sphere prefilter.
///
/// The double contour integral is O(C^2 M^2) — a 1000-chain x 500-bead melt is
/// ~1e11 segment pairs per snapshot — so skipping pairs is a requirement, not an
/// optimisation.
///
/// For [`Closure::Direct`] the prefilter is **exact, not heuristic**: the
/// closing chord lies inside the chain's convex hull, so two chains whose
/// bounding spheres do not overlap have separated hulls, admit a separating
/// plane, and are unlinked with linking number exactly zero. Such pairs are
/// reported as `0.0` without integrating.
///
/// For [`Closure::Directional`] the closure leaves the hull, the argument does
/// not hold, and no pair is skipped.
pub fn all_pairs_linking(m: &Melt, c: Closure) -> Vec<(usize, usize, f64)> {
    let unwrapped: Vec<Vec<Vec3>> = m.chains.iter().map(|ch| unwrap_chain(m, ch)).collect();
    let spheres: Vec<(Vec3, f64)> = unwrapped.iter().map(|v| (centroid(v), extent(v))).collect();
    let exact_prefilter = matches!(c, Closure::Direct);

    let mut out = Vec::new();
    for i in 0..m.chains.len() {
        for j in (i + 1)..m.chains.len() {
            if exact_prefilter {
                let d = m.min_image_dist(spheres[i].0, spheres[j].0);
                if d > spheres[i].1 + spheres[j].1 {
                    out.push((i, j, 0.0));
                    continue;
                }
            }
            out.push((i, j, linking_number(m, i, j, c)));
        }
    }
    out
}
