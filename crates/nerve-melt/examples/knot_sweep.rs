//! Determinant and modal-label stability across polygon resolution.
//!
//! Exists so the robustness question — does `|Delta(-1)|` depend on how finely the
//! curve is discretised, or on how many stochastic closures are drawn? — is
//! answered by measurement and can be fanned out across many jobs.
//!
//! Usage: `cargo run --release -p nerve-melt --example knot_sweep -- <knot> <n_points> [n_closures] [seed]`
//! where `<knot>` is one of `unknot`, `trefoil`, `fig8`, `t25`, `t27`.

use nerve_melt::*;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: knot_sweep <unknot|trefoil|fig8|t25|t27> <n_points> [n_closures] [seed]");
        std::process::exit(2);
    }
    let n: usize = a[1].parse().expect("n_points");
    let (name, k, expect): (&str, Vec<[f64; 3]>, u128) = match a[0].as_str() {
        "unknot" => ("unknot", circle(n), 1),
        "trefoil" => ("trefoil 3_1", torus_knot(2, 3, n), 3),
        "fig8" => ("figure-eight 4_1", figure_eight(n), 5),
        "t25" => ("5_1 = (2,5)", torus_knot(2, 5, n), 5),
        "t27" => ("7_1 = (2,7)", torus_knot(2, 7, n), 7),
        other => panic!("unknown knot {other}"),
    };

    let closed = alexander_det_minus_one(&k);
    let ok = closed == Some(expect);
    println!("knot={name} n_points={n} closed_det={closed:?} expected={expect} match={ok}");

    if a.len() >= 3 {
        let n_cl: usize = a[2].parse().expect("n_closures");
        let seed: u64 = a.get(3).map(|s| s.parse().expect("seed")).unwrap_or(1);
        let mut open = k;
        open.pop();
        let l = knot_label(&open, n_cl, seed);
        println!(
            "open_modal={} prob={:.4} distinct={} resolved={}/{} modal_match={}",
            l.modal,
            l.prob,
            l.distinct,
            l.resolved,
            n_cl,
            l.modal == expect
        );
    }
}
