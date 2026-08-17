//! Settling the flagged cross-crate sign dispute from first principles, and
//! bounding what parity actually buys.
//!
//! `nerve-topo`'s global linking sign is convention-checked rather than derived.
//! Comparing it against my own round-1 Klenin-Langowski implementation would
//! only show two implementations agreeing, which is not a derivation. So the
//! reference here is an analytic hand calculation instead.

use nerve_core::Vec3;
use nerve_order::*;

/// Ring A: the unit circle in the plane `z = 0`, parameterised counterclockwise
/// as seen from `+z`.
fn ring_a(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            [t.cos(), t.sin(), 0.0]
        })
        .collect()
}

/// Ring B: the unit circle centred at `(1, 0, 0)` in the plane `y = 0`,
/// parameterised as `(1 + cos u, 0, sin u)` with `u` increasing.
fn ring_b(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let u = std::f64::consts::TAU * k as f64 / n as f64;
            [1.0 + u.cos(), 0.0, u.sin()]
        })
        .collect()
}

/// Derivation, by signed piercings of a Seifert surface.
///
/// Ring A bounds the flat unit disc `D` in `z = 0`. With A counterclockwise seen
/// from `+z`, the induced orientation gives `D` the normal `+z`, and
/// `Lk(A, B)` is the sum of `sign(B' · n)` over the points where B pierces `D`.
///
/// `B(u) = (1 + cos u, 0, sin u)` has `z = sin u`, which vanishes at `u = 0` and
/// `u = π`.
///
/// * `u = 0` gives the point `(2, 0, 0)`, at radius 2 from the origin, **outside**
///   the unit disc. Not counted.
/// * `u = π` gives the point `(0, 0, 0)`, the disc centre, **inside**. There
///   `B'(u) = (-sin u, 0, cos u) = (0, 0, -1)`, so `B' · n = -1`.
///
/// Exactly one piercing, of sign `-1`. Therefore **`Lk(A, B) = -1`**.
///
/// This test is a genuine coin flip against an implementation whose sign
/// convention is unproven: the opposite convention returns `+1` and fails here.
/// A failure is a *disagreement to settle*, not a bug in either crate.
#[test]
fn linking_sign_of_hopf_link_matches_the_seifert_disc_derivation() {
    let (a, b) = (ring_a(400), ring_b(400));
    let v = nerve_topo::linking_number_closed(&a, &b);
    println!("seifert derivation predicts -1, nerve-topo returns {v:.12}");
    assert!(
        (v - (-1.0)).abs() < 1e-6,
        "sign convention disagreement: analytic Seifert count gives -1, \
         nerve-topo::linking_number_closed gives {v}. Settle explicitly."
    );
}

/// The bound on parity, per the Inspector's ruling that parity kills the sign
/// and nothing else.
///
/// Reflection negates the linking number, so its **sign** is unrepresentable by
/// any function invariant under improper transformations. Its **magnitude** is
/// untouched, so parity says nothing whatever about `|Lk|`. Any hierarchy that
/// puts parity above the other mechanisms has to answer this.
#[test]
fn reflection_negates_linking_number_but_preserves_its_magnitude() {
    let m = wavy_rings(48, 3.0, 3.0, 40.0, 0.3, 3.0);
    let (before, after) = (lk(&m, 0, 1), lk(&mirror(&m), 0, 1));
    println!("Lk {before:.12} -> {after:.12} under reflection");
    assert!((before + after).abs() < 1e-6, "sign did not flip: {before} vs {after}");
    assert!(
        (before.abs() - after.abs()).abs() < 1e-6,
        "magnitude changed: {} vs {}",
        before.abs(),
        after.abs()
    );
    assert!(before.abs() > 0.5, "need a genuinely linked pair, got |Lk| {}", before.abs());
}
