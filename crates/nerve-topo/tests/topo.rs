//! Test-first suite for nerve-topo.
//!
//! Every test in this file was observed RED against a `unimplemented!()` stub
//! before any implementation existed.
//!
//! Structure follows tda-tdd: ground truth first (most valuable), then the
//! invariants (isometry / scale / lattice), then the negative control, then
//! cross-implementation parity, then determinism.

use nerve_core::{Chain, Melt, Vec3};
use nerve_topo::{
    all_pairs_linking, closed_form, closure_spread, fibonacci_directions, image_spread,
    linking_number, linking_number_closed, linking_number_open, unwrap_chain, writhe,
    writhe_closed, writhe_open, Closure,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use std::f64::consts::{PI, TAU};

// ---------------------------------------------------------------- vector help

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

/// A box large enough that `min_image` is the identity for every configuration
/// used here — i.e. free space reached through the periodic API.
const FREE: f64 = 1.0e9;

fn free_melt(a: &[Vec3], b: &[Vec3]) -> Melt {
    Melt::new(vec![Chain::new(a.to_vec()), Chain::new(b.to_vec())], FREE)
}

// ------------------------------------------------------------- ground truth

/// Boundary curves of an annulus carrying `n_twist` full twists.
///
/// This is the (2, 2n) torus link, whose two components have linking number
/// exactly `n`. n=0 gives two coplanar concentric circles (unlinked, Lk=0),
/// n=1 the Hopf link (Lk=1), n=2 the (2,4) torus link (Lk=2).
///
/// One generator covers every ground-truth case in the crate, which is why the
/// expected values are cross-consistent rather than three independent guesses.
fn twisted_band(n_twist: i32, r: f64, a: f64, m: usize) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut ca = Vec::with_capacity(m);
    let mut cb = Vec::with_capacity(m);
    for k in 0..m {
        let t = TAU * (k as f64) / (m as f64);
        let core = [r * t.cos(), r * t.sin(), 0.0];
        let e_r = [t.cos(), t.sin(), 0.0];
        let nt = (n_twist as f64) * t;
        let u = add(mul(e_r, nt.cos()), [0.0, 0.0, nt.sin()]);
        ca.push(add(core, mul(u, a)));
        cb.push(sub(core, mul(u, a)));
    }
    (ca, cb)
}

fn circle(r: f64, m: usize) -> Vec<Vec3> {
    (0..m)
        .map(|k| {
            let t = TAU * (k as f64) / (m as f64);
            [r * t.cos(), r * t.sin(), 0.0]
        })
        .collect()
}

#[test]
fn unlinked_concentric_rings_are_zero() {
    let (a, b) = twisted_band(0, 1.0, 0.3, 200);
    let lk = linking_number_closed(&a, &b);
    assert!(lk.abs() < 1e-9, "unlinked rings gave Lk = {lk}");
}

#[test]
fn unlinked_separated_rings_are_zero() {
    let a = circle(1.0, 200);
    let b: Vec<Vec3> = circle(1.0, 200)
        .iter()
        .map(|p| add(*p, [50.0, 0.0, 0.0]))
        .collect();
    let lk = linking_number_closed(&a, &b);
    assert!(lk.abs() < 1e-9, "separated rings gave Lk = {lk}");
}

#[test]
fn hopf_link_is_one() {
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    let lk = linking_number_closed(&a, &b);
    assert!(
        (lk.abs() - 1.0).abs() < 1e-9,
        "Hopf link gave Lk = {lk}, expected magnitude 1"
    );
}

#[test]
fn torus_2_4_link_is_two() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 400);
    let lk = linking_number_closed(&a, &b);
    assert!(
        (lk.abs() - 2.0).abs() < 1e-9,
        "(2,4) torus link gave Lk = {lk}, expected magnitude 2"
    );
}

#[test]
fn torus_2_6_link_is_three() {
    let (a, b) = twisted_band(3, 1.0, 0.25, 600);
    let lk = linking_number_closed(&a, &b);
    assert!(
        (lk.abs() - 3.0).abs() < 1e-9,
        "(2,6) torus link gave Lk = {lk}, expected magnitude 3"
    );
}

/// The accuracy metric for the crate: how far from an integer does the Gauss
/// double sum land on genuinely closed curves.
#[test]
fn closed_curve_linking_is_integral() {
    let mut worst = 0.0f64;
    let mut worst_case = String::new();
    println!("INTEGER DEVIATION TABLE (rows = twists, cols = segments):");
    print!("  n\\m ");
    for m in [64usize, 128, 256, 512, 1024] {
        print!("{m:>12}");
    }
    println!();
    for n in 0..=4 {
        print!("  {n:>3} ");
        for m in [64usize, 128, 256, 512, 1024] {
            let (a, b) = twisted_band(n, 1.0, 0.2, m);
            let lk = linking_number_closed(&a, &b);
            print!("{:>12.2e}", (lk - lk.round()).abs());
        }
        println!();
    }
    for n in 0..=4 {
        for m in [64usize, 128, 256, 512] {
            let (a, b) = twisted_band(n, 1.0, 0.2, m);
            let lk = linking_number_closed(&a, &b);
            let dev = (lk - lk.round()).abs();
            if dev > worst {
                worst = dev;
                worst_case = format!("n_twist={n} m={m} lk={lk:.17}");
            }
            assert_eq!(
                lk.round().abs(),
                n as f64,
                "n_twist={n} m={m}: |Lk| should round to {n}, got {lk}"
            );
        }
    }
    println!("INTEGER DEVIATION: worst = {worst:.3e} ({worst_case})");
    assert!(
        worst < 1e-12,
        "integer deviation {worst:.3e} exceeds 1e-12 ({worst_case})"
    );
}

#[test]
fn linking_number_is_symmetric() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let ab = linking_number_closed(&a, &b);
    let ba = linking_number_closed(&b, &a);
    assert!((ab - ba).abs() < 1e-12, "Lk(A,B)={ab} but Lk(B,A)={ba}");
}

// ----------------------------------------------------- cross-implementation

/// Independent midpoint quadrature of the Gauss double integral. Slow, low
/// order, obviously-correct. Pins the sign convention and the 1/4pi
/// normalisation of the exact segment-pair formula against a completely
/// different evaluation route.
fn lk_midpoint_quadrature(a: &[Vec3], b: &[Vec3]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        let (a0, a1) = (a[i], a[(i + 1) % a.len()]);
        let ta = sub(a1, a0);
        let ma = mul(add(a0, a1), 0.5);
        for j in 0..b.len() {
            let (b0, b1) = (b[j], b[(j + 1) % b.len()]);
            let tb = sub(b1, b0);
            let mb = mul(add(b0, b1), 0.5);
            let r = sub(ma, mb);
            let d = norm(r);
            s += dot(r, cross(ta, tb)) / (d * d * d);
        }
    }
    s / (4.0 * PI)
}

#[test]
fn linking_number_matches_midpoint_quadrature() {
    for n in 0..=2 {
        let (a, b) = twisted_band(n, 1.0, 0.3, 500);
        let exact = linking_number_closed(&a, &b);
        let quad = lk_midpoint_quadrature(&a, &b);
        assert!(
            (exact - quad).abs() < 2e-2,
            "n_twist={n}: exact={exact} quadrature={quad}"
        );
    }
}

// ------------------------------------------------------------- isometry

fn rotate(p: Vec3, ax: f64, ay: f64, az: f64) -> Vec3 {
    let (s, c) = ax.sin_cos();
    let p = [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]];
    let (s, c) = ay.sin_cos();
    let p = [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
    let (s, c) = az.sin_cos();
    [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
}

fn map_all(v: &[Vec3], f: impl Fn(Vec3) -> Vec3) -> Vec<Vec3> {
    v.iter().map(|p| f(*p)).collect()
}

#[test]
fn linking_number_invariant_under_rotation() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = linking_number_closed(&a, &b);
    let (ax, ay, az) = (0.731, -1.219, 2.443);
    let ra = map_all(&a, |p| rotate(p, ax, ay, az));
    let rb = map_all(&b, |p| rotate(p, ax, ay, az));
    let rot = linking_number_closed(&ra, &rb);
    assert!((base - rot).abs() < 1e-10, "base={base} rotated={rot}");
}

#[test]
fn linking_number_invariant_under_translation() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = linking_number_closed(&a, &b);
    let t = [123.5, -7.25, 4096.0];
    let ta = map_all(&a, |p| add(p, t));
    let tb = map_all(&b, |p| add(p, t));
    let moved = linking_number_closed(&ta, &tb);
    assert!((base - moved).abs() < 1e-10, "base={base} moved={moved}");
}

#[test]
fn linking_number_flips_sign_under_reflection() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = linking_number_closed(&a, &b);
    let ra = map_all(&a, |p| [p[0], p[1], -p[2]]);
    let rb = map_all(&b, |p| [p[0], p[1], -p[2]]);
    let refl = linking_number_closed(&ra, &rb);
    assert!(
        (base + refl).abs() < 1e-10,
        "reflection must negate: base={base} reflected={refl}"
    );
    assert!(base.abs() > 0.5, "degenerate test: base={base}");
}

// ------------------------------------------------------------- scale

#[test]
fn linking_number_exactly_invariant_under_power_of_two_scaling() {
    // Powers of two scale f64 coordinates without any rounding, so the
    // linking number must be reproduced *bitwise*. A hardcoded absolute
    // epsilon anywhere in the integrator shows up here and nowhere else.
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = linking_number_closed(&a, &b);
    for c in [0.25f64, 0.5, 2.0, 4.0, 1024.0, 1.0 / 1024.0] {
        let sa = map_all(&a, |p| mul(p, c));
        let sb = map_all(&b, |p| mul(p, c));
        let got = linking_number_closed(&sa, &sb);
        assert_eq!(
            base.to_bits(),
            got.to_bits(),
            "scale {c}: expected bitwise {base}, got {got}"
        );
    }
}

#[test]
fn linking_number_invariant_under_general_scaling() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = linking_number_closed(&a, &b);
    for c in [1e-6f64, 0.037, 3.7, 1.9e5] {
        let sa = map_all(&a, |p| mul(p, c));
        let sb = map_all(&b, |p| mul(p, c));
        let got = linking_number_closed(&sa, &sb);
        assert!(
            (base - got).abs() < 1e-10,
            "scale {c}: base={base} got={got}"
        );
    }
}

// ------------------------------------------------------------- writhe

#[test]
fn writhe_of_planar_circle_is_zero() {
    let c = circle(1.0, 256);
    let w = writhe_closed(&c);
    assert!(w.abs() < 1e-12, "planar circle writhe = {w}");
}

#[test]
fn writhe_of_planar_open_arc_is_zero() {
    let c: Vec<Vec3> = circle(1.0, 256).into_iter().take(180).collect();
    let w = writhe_open(&c);
    assert!(w.abs() < 1e-12, "planar open arc writhe = {w}");
}

#[test]
fn writhe_flips_sign_under_reflection() {
    let (a, _) = twisted_band(3, 1.0, 0.3, 300);
    let base = writhe_closed(&a);
    let r = map_all(&a, |p| [p[0], p[1], -p[2]]);
    let refl = writhe_closed(&r);
    assert!(base.abs() > 0.1, "degenerate test: writhe={base}");
    assert!(
        (base + refl).abs() < 1e-10,
        "reflection must negate writhe: {base} vs {refl}"
    );
}

#[test]
fn writhe_exactly_invariant_under_power_of_two_scaling() {
    let (a, _) = twisted_band(3, 1.0, 0.3, 300);
    let base = writhe_closed(&a);
    for c in [0.5f64, 2.0, 4096.0] {
        let s = map_all(&a, |p| mul(p, c));
        assert_eq!(
            base.to_bits(),
            writhe_closed(&s).to_bits(),
            "writhe not bitwise scale-invariant at c={c}"
        );
    }
}

#[test]
fn writhe_invariant_under_rotation() {
    let (a, _) = twisted_band(3, 1.0, 0.3, 300);
    let base = writhe_closed(&a);
    let r = map_all(&a, |p| rotate(p, 0.4, 1.1, -2.2));
    assert!(
        (base - writhe_closed(&r)).abs() < 1e-10,
        "writhe changed under rotation"
    );
}

/// Writhe is NOT a topological invariant and NOT an integer. Guard against a
/// future "tidy-up" that rounds it.
#[test]
fn writhe_is_not_integral() {
    let (a, _) = twisted_band(3, 1.0, 0.3, 300);
    let w = writhe_closed(&a);
    assert!(
        (w - w.round()).abs() > 1e-6,
        "writhe {w} is suspiciously integral — is something rounding?"
    );
}

// ------------------------------------------------------- negative control

fn walk(rng: &mut ChaCha8Rng, n: usize, origin: Vec3) -> Vec<Vec3> {
    let mut p = origin;
    let mut out = vec![p];
    for _ in 1..n {
        let z: f64 = rng.gen_range(-1.0..1.0);
        let phi: f64 = rng.gen_range(0.0..TAU);
        let r = (1.0 - z * z).sqrt();
        p = add(p, [r * phi.cos(), r * phi.sin(), z]);
        out.push(p);
    }
    out
}

#[test]
fn separated_random_walks_are_unlinked() {
    // 60 unit steps cannot leave a ball of radius 60, so the two convex hulls
    // are provably separable and the exact answer is 0. This runs the full
    // double sum (no prefilter), so it is a real exercise of the integrator.
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    for _ in 0..8 {
        let a = walk(&mut rng, 60, [0.0, 0.0, 0.0]);
        let b = walk(&mut rng, 60, [500.0, 0.0, 0.0]);
        let ca = closed_form(&a, Closure::Direct);
        let cb = closed_form(&b, Closure::Direct);
        let lk = linking_number_closed(&ca, &cb);
        assert!(lk.abs() < 1e-9, "separated walks gave Lk = {lk}");
    }
}

// ------------------------------------------------------------- PBC

#[test]
fn unwrap_chain_reconstructs_continuous_coordinates() {
    let m = Melt::new(vec![Chain::new(vec![])], 10.0);
    let wrapped = Chain::new(vec![
        [8.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    let got = unwrap_chain(&m, &wrapped);
    let want: Vec<Vec3> = vec![
        [8.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
    ];
    for (g, w) in got.iter().zip(&want) {
        assert!(
            (g[0] - w[0]).abs() < 1e-12,
            "unwrap gave {got:?}, want {want:?}"
        );
    }
}

/// The load-bearing PBC test. Same physical configuration, expressed once as
/// continuous coordinates and once wrapped into the box, must give the same
/// linking number. If it does not, the number is measuring the box.
#[test]
fn wrapped_and_unwrapped_configurations_agree() {
    let l = 6.0;
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    // Shift so the pair straddles the x=0 face.
    let shift = [0.5, 3.0, 3.0];
    let a = map_all(&a, |p| add(p, shift));
    let b = map_all(&b, |p| add(p, shift));

    let unwrapped = free_melt(&a, &b);
    let lk_free = linking_number(&unwrapped, 0, 1, Closure::Direct);

    let wrap = |p: Vec3| [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2].rem_euclid(l)];
    let wa = map_all(&a, wrap);
    let wb = map_all(&b, wrap);
    assert!(
        wa.iter().any(|p| p[0] > l / 2.0) && wa.iter().any(|p| p[0] < l / 2.0),
        "test configuration does not actually straddle the boundary"
    );
    let wrapped = Melt::new(vec![Chain::new(wa), Chain::new(wb)], l);
    let lk_wrapped = linking_number(&wrapped, 0, 1, Closure::Direct);

    assert!(
        (lk_free - lk_wrapped).abs() < 1e-9,
        "wrapped {lk_wrapped} vs unwrapped {lk_free}"
    );
    assert!(
        (lk_free.abs() - 2.0).abs() < 1e-9,
        "sanity: expected |Lk|=2, got {lk_free}"
    );
}

#[test]
fn lattice_translation_leaves_linking_unchanged() {
    let l = 6.0;
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    let shift = [3.0, 3.0, 3.0];
    let a = map_all(&a, |p| add(p, shift));
    let b = map_all(&b, |p| add(p, shift));
    let base = linking_number(
        &Melt::new(vec![Chain::new(a.clone()), Chain::new(b.clone())], l),
        0,
        1,
        Closure::Direct,
    );

    for t in [[l, 0.0, 0.0], [0.0, -l, 0.0], [2.0 * l, l, -3.0 * l]] {
        let bt = map_all(&b, |p| add(p, t));
        let got = linking_number(
            &Melt::new(vec![Chain::new(a.clone()), Chain::new(bt)], l),
            0,
            1,
            Closure::Direct,
        );
        assert!(
            (base - got).abs() < 1e-9,
            "lattice translation {t:?}: base={base} got={got}"
        );
    }
}

#[test]
fn writhe_is_computed_on_unwrapped_coordinates() {
    let l = 6.0;
    let (a, _) = twisted_band(3, 1.0, 0.3, 300);
    let a = map_all(&a, |p| add(p, [0.5, 3.0, 3.0]));
    let free = writhe(&free_melt(&a, &a), 0);
    let wrap = |p: Vec3| [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2].rem_euclid(l)];
    let wrapped = Melt::new(vec![Chain::new(map_all(&a, wrap))], l);
    let got = writhe(&wrapped, 0);
    assert!(
        (free - got).abs() < 1e-9,
        "writhe under wrapping: free={free} wrapped={got}"
    );
}

/// The periodic image choice is NOT resolved. This test does not assert it is
/// right — it asserts the crate *reports* the ambiguity, so a caller cannot
/// consume the number without seeing its spread.
#[test]
fn image_spread_is_reported_and_is_zero_for_an_isolated_pair() {
    let l = 60.0;
    let (a, b) = twisted_band(1, 1.0, 0.3, 120);
    let a = map_all(&a, |p| add(p, [30.0, 30.0, 30.0]));
    let b = map_all(&b, |p| add(p, [30.0, 30.0, 30.0]));
    let m = Melt::new(vec![Chain::new(a), Chain::new(b)], l);
    let s = image_spread(&m, 0, 1, Closure::Direct);
    println!("IMAGE SPREAD (isolated pair, L=60, extent~2.6): {s:?}");
    // For an isolated pair exactly one of the 27 images may carry linking. The
    // extreme of larger magnitude is that carrier; the opposite extreme bounds
    // all the others. Since every sample lies in [min, max] and the samples sum
    // to 27*mean, `27*mean == carrier` proves the other 26 are all ~0 — which
    // is the statement "this pair has no periodic image ambiguity".
    let (carrier, other) = if s.max.abs() > s.min.abs() {
        (s.max, s.min)
    } else {
        (s.min, s.max)
    };
    assert!(
        (carrier.abs() - 1.0).abs() < 1e-9,
        "carrier image should hold the Hopf link, got {carrier}"
    );
    assert!(
        other.abs() < 1e-9,
        "a second image carries linking: {other}"
    );
    assert!(
        (27.0 * s.mean - carrier).abs() < 1e-9,
        "more than one image carries linking: 27*mean={} carrier={carrier}",
        27.0 * s.mean
    );
}

// ------------------------------------------------------------- closures

#[test]
fn fibonacci_directions_are_unit_and_deterministic() {
    let d = fibonacci_directions(64);
    assert_eq!(d.len(), 64);
    for v in &d {
        assert!((norm(*v) - 1.0).abs() < 1e-12, "non-unit direction {v:?}");
    }
    assert_eq!(d, fibonacci_directions(64), "direction set is not stable");
}

/// Any *deterministic closure* makes the linking number an integer, because the
/// closed polygon really is closed. This is not a bug and it is not rounding —
/// it is why "the linking number of an open chain" is not a well-posed quantity
/// until you say what you did to the ends.
///
/// The first version of this test asserted the opposite and was wrong.
#[test]
fn deterministic_closure_always_yields_an_integer() {
    let (a_full, b_full) = twisted_band(1, 1.0, 0.3, 200);
    println!("CLOSED-BY-CHORD Lk vs truncation (from a Hopf link):");
    for keep in [40usize, 60, 80, 100, 120, 140, 160, 180, 200] {
        let a: Vec<Vec3> = a_full.iter().copied().take(keep).collect();
        let b: Vec<Vec3> = b_full.iter().copied().take(keep).collect();
        let lk = linking_number(&free_melt(&a, &b), 0, 1, Closure::Direct);
        let dev = (lk - lk.round()).abs();
        println!("  keep={keep:3}/200  Lk={lk:+.9}  dist_to_int={dev:.3e}");
        assert!(
            dev < 1e-12,
            "keep={keep}: chord-closed Lk {lk} is not integral (dev {dev:.3e})"
        );
    }
}

/// The un-closed Gauss double integral over two open curves. THIS is the
/// real-valued, closure-free pairwise linking signal, and it is the one to feed
/// downstream for open chains in a melt.
#[test]
fn open_linking_without_closure_is_real_valued() {
    let (a_full, b_full) = twisted_band(1, 1.0, 0.3, 200);
    let mut fractional = 0;
    println!("UN-CLOSED Gauss pair integral vs truncation (from a Hopf link):");
    for keep in [40usize, 60, 80, 100, 120, 140, 160, 180, 200] {
        let a: Vec<Vec3> = a_full.iter().copied().take(keep).collect();
        let b: Vec<Vec3> = b_full.iter().copied().take(keep).collect();
        let lk = linking_number_open(&a, &b);
        let dev = (lk - lk.round()).abs();
        println!("  keep={keep:3}/200  Lk_open={lk:+.9}  dist_to_int={dev:.3e}");
        if dev > 1e-6 {
            fractional += 1;
        }
    }
    assert!(
        fractional >= 6,
        "un-closed pair integral should be generically non-integer, only \
         {fractional}/9 truncations were"
    );
}

#[test]
fn open_linking_converges_to_the_closed_value_as_the_gap_shrinks() {
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    let closed = linking_number_closed(&a, &b);
    let open = linking_number_open(&a, &b);
    println!("closed={closed:+.9} open(one segment missing)={open:+.9}");
    assert!(
        (closed - open).abs() < 0.05,
        "closed {closed} vs un-closed {open} — a single missing segment out of \
         200 should barely matter"
    );
}

#[test]
fn open_linking_is_symmetric_and_isometry_invariant() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 150);
    let a: Vec<Vec3> = a.into_iter().take(110).collect();
    let b: Vec<Vec3> = b.into_iter().take(110).collect();
    let base = linking_number_open(&a, &b);
    assert!(base.abs() > 0.1, "degenerate test: {base}");
    assert!(
        (base - linking_number_open(&b, &a)).abs() < 1e-12,
        "un-closed pair integral is not symmetric"
    );
    let ra = map_all(&a, |p| add(rotate(p, 0.9, -1.7, 0.3), [11.0, -3.0, 7.0]));
    let rb = map_all(&b, |p| add(rotate(p, 0.9, -1.7, 0.3), [11.0, -3.0, 7.0]));
    assert!(
        (base - linking_number_open(&ra, &rb)).abs() < 1e-10,
        "un-closed pair integral is not isometry invariant"
    );
    let refl_a = map_all(&a, |p| [p[0], p[1], -p[2]]);
    let refl_b = map_all(&b, |p| [p[0], p[1], -p[2]]);
    assert!(
        (base + linking_number_open(&refl_a, &refl_b)).abs() < 1e-10,
        "un-closed pair integral does not flip sign under reflection"
    );
}

#[test]
fn open_linking_is_exactly_invariant_under_power_of_two_scaling() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 150);
    let a: Vec<Vec3> = a.into_iter().take(110).collect();
    let b: Vec<Vec3> = b.into_iter().take(110).collect();
    let base = linking_number_open(&a, &b);
    for c in [0.5f64, 2.0, 1024.0] {
        assert_eq!(
            base.to_bits(),
            linking_number_open(&map_all(&a, |p| mul(p, c)), &map_all(&b, |p| mul(p, c))).to_bits(),
            "un-closed pair integral not bitwise scale invariant at c={c}"
        );
    }
}

#[test]
fn closure_open_variant_is_routed_through_the_melt_api() {
    let l = 6.0;
    let (a, b) = twisted_band(2, 1.0, 0.3, 150);
    let a: Vec<Vec3> = a.into_iter().take(110).collect();
    let b: Vec<Vec3> = b.into_iter().take(110).collect();
    let a = map_all(&a, |p| add(p, [0.5, 3.0, 3.0]));
    let b = map_all(&b, |p| add(p, [0.5, 3.0, 3.0]));
    let free = linking_number(&free_melt(&a, &b), 0, 1, Closure::Open);
    let wrap = |p: Vec3| [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2].rem_euclid(l)];
    let wrapped = Melt::new(
        vec![Chain::new(map_all(&a, wrap)), Chain::new(map_all(&b, wrap))],
        l,
    );
    let got = linking_number(&wrapped, 0, 1, Closure::Open);
    assert!(
        (free - got).abs() < 1e-9,
        "Closure::Open under wrapping: free={free} wrapped={got}"
    );
    assert!(
        (free - linking_number_open(&a, &b)).abs() < 1e-12,
        "melt API disagrees with the free-space un-closed integral"
    );
}

#[test]
fn closure_spread_is_reported_for_open_chains() {
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    let a: Vec<Vec3> = a.into_iter().take(180).collect();
    let b: Vec<Vec3> = b.into_iter().take(180).collect();
    let m = free_melt(&a, &b);
    let direct = linking_number(&m, 0, 1, Closure::Direct);
    let s = closure_spread(&m, 0, 1, 64);
    println!("CLOSURE SPREAD (90% of a Hopf link): direct={direct:.6} {s:?}");
    assert!(s.sd.is_finite() && s.sd >= 0.0);
    assert!(
        s.min <= s.mean && s.mean <= s.max,
        "incoherent spread {s:?}"
    );
    assert!(
        (direct.abs() - 1.0).abs() < 1e-9,
        "direct closure should recover the Hopf link, got {direct}"
    );
}

/// The measured closure artifact, recorded as a test so it cannot be forgotten.
///
/// Each direction closes the curve, so each value is an integer — but *which*
/// integer depends on the direction. On a 90%-closed Hopf link the far-field
/// directional scheme returns values in {-1, 0, +1}: some directions invert the
/// link, and at least one unlinks it entirely, because the two chains' far-field
/// detours wind around each other for those directions.
///
/// Direct chord closure has no such failure on this input. That is the empirical
/// reason it is the primary scheme, and this test is the evidence.
#[test]
fn directional_closure_majority_agrees_with_direct_closure() {
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    let a: Vec<Vec3> = a.into_iter().take(180).collect();
    let b: Vec<Vec3> = b.into_iter().take(180).collect();
    let m = free_melt(&a, &b);
    let direct = linking_number(&m, 0, 1, Closure::Direct);

    let dirs = fibonacci_directions(64);
    let mut hist = std::collections::BTreeMap::<i64, usize>::new();
    for dir in &dirs {
        let lk = linking_number(
            &m,
            0,
            1,
            Closure::Directional {
                dir: *dir,
                r_scale: 200.0,
            },
        );
        assert!(
            (lk - lk.round()).abs() < 1e-8,
            "direction {dir:?} gave a non-integer {lk} — every closure closes \
             the curve, so every value must be integral"
        );
        *hist.entry(lk.round() as i64).or_default() += 1;
    }
    println!(
        "DIRECTIONAL CLOSURE HISTOGRAM (64 Fibonacci dirs, 90% of a Hopf link, \
         direct={direct:+.0}): {hist:?}"
    );
    let agree = hist.get(&(direct.round() as i64)).copied().unwrap_or(0);
    println!(
        "  agreement with direct closure: {agree}/64 = {:.1}%",
        100.0 * agree as f64 / 64.0
    );
    assert!(
        agree * 2 > dirs.len(),
        "only {agree}/64 directions agree with direct closure — this pair's \
         linking number is closure-dominated and must not be reported"
    );
}

#[test]
fn closure_spread_vanishes_for_separated_chains() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let a = walk(&mut rng, 50, [0.0, 0.0, 0.0]);
    let b = walk(&mut rng, 50, [400.0, 0.0, 0.0]);
    let m = free_melt(&a, &b);
    let s = closure_spread(&m, 0, 1, 32);
    println!("CLOSURE SPREAD (separated walks): {s:?}");
    assert!(
        s.max.abs() < 1e-6 && s.min.abs() < 1e-6,
        "separated chains show closure-dependent linking: {s:?}"
    );
}

#[test]
fn directional_closure_adds_segments_outside_the_chain() {
    let a = circle(1.0, 8);
    let d = closed_form(
        &a,
        Closure::Directional {
            dir: [0.0, 0.0, 1.0],
            r_scale: 100.0,
        },
    );
    assert_eq!(d.len(), a.len() + 2);
    assert!(
        d.iter().any(|p| p[2] > 50.0),
        "directional closure did not leave the neighbourhood: {d:?}"
    );
}

// ------------------------------------------------------ degenerate inputs

#[test]
fn short_chains_give_zero_not_a_panic() {
    let cases: Vec<Vec<Vec3>> = vec![
        vec![],
        vec![[0.0, 0.0, 0.0]],
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ];
    let full = circle(1.0, 64);
    for c in &cases {
        assert_eq!(linking_number_closed(c, &full), 0.0, "len {}", c.len());
        assert_eq!(linking_number_closed(&full, c), 0.0, "len {}", c.len());
        assert_eq!(writhe_closed(c), 0.0, "len {}", c.len());
        assert_eq!(writhe_open(c), 0.0, "len {}", c.len());
    }
}

#[test]
fn collinear_and_coincident_segments_do_not_produce_nan() {
    // Two identical straight chains, and a chain lying on top of another.
    let a: Vec<Vec3> = (0..8).map(|i| [i as f64, 0.0, 0.0]).collect();
    let b = a.clone();
    let lk = linking_number_closed(&a, &b);
    assert!(lk.is_finite(), "degenerate pair gave {lk}");
    let w = writhe_open(&a);
    assert!(
        w.is_finite() && w.abs() < 1e-12,
        "straight chain writhe {w}"
    );
}

// ------------------------------------------------------------- determinism

#[test]
fn results_are_bitwise_deterministic() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let first = linking_number_closed(&a, &b);
    for _ in 0..5 {
        assert_eq!(
            first.to_bits(),
            linking_number_closed(&a, &b).to_bits(),
            "linking_number_closed is not bitwise deterministic"
        );
    }
    let w = writhe_closed(&a);
    for _ in 0..5 {
        assert_eq!(w.to_bits(), writhe_closed(&a).to_bits());
    }
}

// --------------------------------------------------------------- prefilter

#[test]
fn prefilter_is_exact_not_approximate() {
    // Four chains: two linked pairs, far apart. The bounding-sphere prefilter
    // must reproduce the unfiltered answer BITWISE for the pairs it keeps and
    // return exact zero for the pairs it drops.
    let (a0, b0) = twisted_band(1, 1.0, 0.3, 120);
    let far = |v: &[Vec3]| map_all(v, |p| add(p, [1000.0, 0.0, 0.0]));
    let (a1, b1) = (far(&a0), far(&b0));
    let m = Melt::new(
        vec![
            Chain::new(a0),
            Chain::new(b0),
            Chain::new(a1),
            Chain::new(b1),
        ],
        FREE,
    );
    let pairs = all_pairs_linking(&m, Closure::Direct);
    assert_eq!(pairs.len(), 6, "expected all 6 chain pairs reported");
    for &(i, j, lk) in &pairs {
        let unfiltered = linking_number(&m, i, j, Closure::Direct);
        if (i, j) == (0, 1) || (i, j) == (2, 3) {
            // Overlapping bounding spheres: the prefilter keeps the pair, so
            // the value must come from the same code path, bitwise.
            assert_eq!(
                lk.to_bits(),
                unfiltered.to_bits(),
                "kept pair ({i},{j}): prefiltered {lk} != unfiltered {unfiltered}"
            );
            assert!((lk.abs() - 1.0).abs() < 1e-9, "pair ({i},{j}) Lk={lk}");
        } else {
            // Separated hulls: the prefilter returns the mathematically exact
            // 0.0, while the unfiltered double sum returns 0 only to roundoff.
            // The prefilter is therefore *more* accurate than the integral here,
            // not an approximation of it.
            assert_eq!(lk, 0.0, "dropped pair ({i},{j}) gave {lk}, want exact 0");
            assert!(
                unfiltered.abs() < 1e-9,
                "pair ({i},{j}): unfiltered integral {unfiltered} should be ~0"
            );
            println!("PREFILTER: dropped pair ({i},{j}) unfiltered residual = {unfiltered:.3e}");
        }
    }
}

// -------------------------------------------------------------- properties

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(32))]

    /// Rotation invariance over the sampler's choice of angles.
    #[test]
    fn prop_rotation_invariance(
        ax in -PI..PI, ay in -PI..PI, az in -PI..PI,
        n in 0i32..3,
    ) {
        let (a, b) = twisted_band(n, 1.0, 0.3, 100);
        let base = linking_number_closed(&a, &b);
        let ra = map_all(&a, |p| rotate(p, ax, ay, az));
        let rb = map_all(&b, |p| rotate(p, ax, ay, az));
        let got = linking_number_closed(&ra, &rb);
        proptest::prop_assert!(
            (base - got).abs() < 1e-9,
            "angles ({ax},{ay},{az}) n={n}: base={base} got={got}"
        );
    }

    /// Scale invariance over the sampler's choice of positive factor.
    #[test]
    fn prop_scale_invariance(c in 1e-4f64..1e4, n in 1i32..4) {
        let (a, b) = twisted_band(n, 1.0, 0.3, 200);
        let base = linking_number_closed(&a, &b);
        let sa = map_all(&a, |p| mul(p, c));
        let sb = map_all(&b, |p| mul(p, c));
        let got = linking_number_closed(&sa, &sb);
        proptest::prop_assert!(
            (base - got).abs() < 1e-9,
            "c={c} n={n}: base={base} got={got}"
        );
    }

    /// Negative control under adversarial sampling: chains confined to
    /// disjoint half-spaces are unlinked, whatever the sampler picks.
    #[test]
    fn prop_disjoint_halfspaces_unlinked(seed in 0u64..10_000) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let a = walk(&mut rng, 40, [0.0, 0.0, 0.0]);
        let b = walk(&mut rng, 40, [400.0, 0.0, 0.0]);
        let lk = linking_number_closed(
            &closed_form(&a, Closure::Direct),
            &closed_form(&b, Closure::Direct),
        );
        proptest::prop_assert!(lk.abs() < 1e-9, "seed {seed}: Lk={lk}");
    }

    /// Lattice translation invariance over the sampler's choice of cell offset.
    #[test]
    fn prop_lattice_translation_invariance(kx in -3i32..3, ky in -3i32..3, kz in -3i32..3) {
        let l = 8.0;
        let (a, b) = twisted_band(1, 1.0, 0.3, 100);
        let a = map_all(&a, |p| add(p, [4.0, 4.0, 4.0]));
        let b = map_all(&b, |p| add(p, [4.0, 4.0, 4.0]));
        let base = linking_number(
            &Melt::new(vec![Chain::new(a.clone()), Chain::new(b.clone())], l),
            0, 1, Closure::Direct,
        );
        let t = [kx as f64 * l, ky as f64 * l, kz as f64 * l];
        let bt = map_all(&b, |p| add(p, t));
        let got = linking_number(
            &Melt::new(vec![Chain::new(a.clone()), Chain::new(bt)], l),
            0, 1, Closure::Direct,
        );
        proptest::prop_assert!(
            (base - got).abs() < 1e-9,
            "k=({kx},{ky},{kz}): base={base} got={got}"
        );
    }
}
