// Probe: a two-ring pair whose total writhe does NOT cancel.
use nerve_core::{Chain, Melt};
fn main() {
    let n = 60usize;
    let tau = std::f64::consts::TAU;
    let (c, r, coil, k) = (15.0f64, 3.0f64, 0.8f64, 5.0f64);
    let ring = |f: &dyn Fn(f64) -> [f64; 3]| Chain::new((0..n).map(|q| f(tau * q as f64 / n as f64)).collect());
    for (name, flip) in [("plain swap", 1.0f64), ("sign-compensated", -1.0f64)] {
        let a = ring(&|t| {
            let rho = r + coil * (k * t).cos();
            [c + rho * t.cos(), c + rho * t.sin(), c + coil * (k * t).sin()]
        });
        let b = ring(&|t| {
            let rho = r + coil * (k * t).cos();
            [c + r + rho * t.cos(), c + flip * coil * (k * t).sin(), c + rho * t.sin()]
        });
        let m = Melt::new(vec![a, b], 30.0);
        let (w0, w1) = (nerve_topo::writhe(&m, 0), nerve_topo::writhe(&m, 1));
        println!(
            "{name:>18}: w0 {w0:+.6} w1 {w1:+.6} total {:+.6} |Lk| {:.6}",
            w0 + w1,
            nerve_topo::linking_number_closed(&m.chains[0].beads, &m.chains[1].beads).abs()
        );
    }
}
