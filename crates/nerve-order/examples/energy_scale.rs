// Vacuity check: are the sampled energies actually large and seed-varying?
// A bitwise-equality claim over random f is worthless if f returns ~0.
use nerve_order::*;
fn main() {
    let d = Local { r_cut: 2.0, chiral: true };
    let a = d.fingerprints(&wavy_rings(48, 3.0, 3.0, 40.0, 0.3, 3.0));
    let b = d.fingerprints(&wavy_rings(48, 3.0, 9.0, 40.0, 0.3, 3.0));
    let c = d.fingerprints(&wavy_rings(48, 3.0 * 1.05, 3.0, 40.0, 0.3, 3.0));
    println!("beads {}, neighbours/bead {}, volumes/bead {}",
        a.len(), a[0].dists.len(), a[0].volumes.len());
    for seed in [1u64, 2, 3, 42, 12345] {
        let (ea, eb, ec) = (
            sum_decomposable_energy(&a, seed),
            sum_decomposable_energy(&b, seed),
            sum_decomposable_energy(&c, seed),
        );
        println!(
            "seed {seed:>6}: E(linked) {ea:>14.6}  E(unlinked) {eb:>14.6}  \
             identical {}  E(r*1.05) {ec:>14.6}  separated by {:.6}",
            ea.to_bits() == eb.to_bits(), (ea - ec).abs()
        );
    }
}
