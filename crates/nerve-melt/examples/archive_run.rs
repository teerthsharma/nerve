//! Run the Alexander witness against a real MD-equilibrated Kremer-Grest melt.
//!
//! Svaneborg & Everaers, Zenodo 7319837 (CC-BY-4.0). The archive supplies real
//! excluded volume, MD equilibration, Z=100 entanglements per chain, and chains
//! 8-84x longer than the hand-grown melts every other number in this crate rests
//! on. Those three caveats retire here or they do not retire at all.
//!
//! Subsetting is explicit and printed, never silent: the closure ensemble is
//! O(n_closures * M^2) per chain, so the number of chains and closures actually
//! used is reported alongside every statistic.
//!
//! Usage: `archive_run <file> <n_chains> <n_closures> [seed]`

use nerve_melt::*;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("usage: archive_run <file> <n_chains> <n_closures> [seed]");
        std::process::exit(2);
    }
    let (path, want_chains, n_cl) = (&a[0], a[1].parse::<usize>().unwrap(), a[2].parse::<usize>().unwrap());
    let seed: u64 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);

    let t0 = Instant::now();
    let m = from_lammps_path(std::path::Path::new(path)).expect("parse");
    let parse_s = t0.elapsed().as_secs_f64();
    let n_chains = m.chains.len();
    let per = m.chains[0].len();
    println!(
        "file={path}\nparsed in {parse_s:.1}s: {n_chains} chains x {per} beads = {} atoms, box_len {:.4}",
        m.n_beads(),
        m.box_len
    );

    // Guard, not an assumption: every chain feature and every unwrap needs it.
    let f = nerve_baseline::chain_features(&m);
    println!(
        "max_bond {:.4} vs box_len/2 {:.4} -> guard {}",
        f.max_bond,
        m.box_len / 2.0,
        if f.max_bond < m.box_len / 2.0 { "PASS" } else { "FAIL" }
    );
    println!("rg2 {:.2}  end_to_end2 {:.2}  contour {:.2}  density {:.4}", f.rg2, f.end_to_end2, f.contour_len, f.bead_density);

    let used = want_chains.min(n_chains);
    println!("\nSUBSET (explicit): {used}/{n_chains} chains, {n_cl} stochastic closures each, seed {seed}");

    let t1 = Instant::now();
    let mut labels = Vec::new();
    let mut mic: Vec<Option<u128>> = Vec::new();
    for i in 0..used {
        // Coordinates are already unwrapped in the archive (xu yu zu), but route
        // through nerve_topo::unwrap_chain anyway so the path is identical to the
        // hand-grown case and any difference is not a code difference.
        let c = nerve_topo::unwrap_chain(&m, &m.chains[i]);
        let l = knot_label(&c, n_cl, seed + i as u64);
        if i < 8 {
            println!(
                "  chain {i}: modal {} p={:.3} distinct={} resolved={}/{}",
                l.modal, l.prob, l.distinct, l.resolved, n_cl
            );
        }
        // The bounding-sphere MIC proxy is a ponytail:-marked deviation from the
        // published convex-hull rule. On chains this long the proxy may degrade,
        // so it is checked against the stochastic mode rather than inherited.
        mic.push(alexander_det_minus_one(&close_minimally_interfering(&c)));
        labels.push(l);
    }
    let knot_s = t1.elapsed().as_secs_f64();

    let resolved_chains = labels.iter().filter(|l| l.resolved > 0).count();
    let unknot = labels.iter().filter(|l| l.resolved > 0 && l.modal == 1).count();
    let knotted = resolved_chains - unknot;
    let mean_p: f64 = if resolved_chains == 0 {
        0.0
    } else {
        labels.iter().filter(|l| l.resolved > 0).map(|l| l.prob).sum::<f64>() / resolved_chains as f64
    };
    // The incompleteness limit, measured where it matters: determinant 5 cannot
    // tell 4_1 from 5_1, and every determinant shared by two knots is ambiguous.
    let ambiguous = labels
        .iter()
        .filter(|l| l.resolved > 0 && matches!(l.modal, 5 | 7 | 9 | 11 | 13 | 15))
        .count();

    // Side-by-side with the hand-grown numbers: writhe sign resolvability against
    // a measured seed-noise floor. Hand-grown (rejection-EV, N=100) gave
    // 0.633/0.583/0.667/0.667/0.600 at rho 0.10-0.60 with sd(Wr) floors 1.08-1.49.
    let w = writhe_per_chain(&m);
    let floor = sd(&w);
    let (ok, tot) = resolvable_fraction(&w, floor);
    println!(
        "\nwrithe: sd floor {floor:.4}, sign-resolvable {ok}/{tot} = {:.4}  (hand-grown N=100 was 0.583-0.667)",
        ok as f64 / tot as f64
    );
    let pooled = w.iter().sum::<f64>().abs();
    let per_chain = w.iter().map(|x| x.abs()).sum::<f64>();
    println!(
        "writhe pooling cancellation |sum|/sum|.| = {:.4} (hand-grown 0.201), Theorem 2 ceiling 1/{} = {:.5}",
        pooled / per_chain,
        n_chains,
        recovery_ceiling(n_chains)
    );

    let mic_agree = mic.iter().zip(&labels).filter(|(m, l)| **m == Some(l.modal)).count();
    let mic_unres = mic.iter().filter(|m| m.is_none()).count();
    println!("MIC bounding-sphere proxy vs stochastic mode: agree {mic_agree}/{used}, unresolved {mic_unres}");

    println!("\n--- results ---");
    println!("knot labels in {knot_s:.1}s ({:.2}s/chain)", knot_s / used.max(1) as f64);
    println!("chains resolved: {resolved_chains}/{used}");
    println!("modal unknot: {unknot}  modal knotted: {knotted}");
    if resolved_chains > 0 {
        println!("knotted fraction: {:.4}", knotted as f64 / resolved_chains as f64);
        println!("mean modal probability: {mean_p:.4}");
        println!(
            "determinant-ambiguous (>=2 knots share the value): {ambiguous}/{resolved_chains} = {:.4}",
            ambiguous as f64 / resolved_chains as f64
        );
    }
}
