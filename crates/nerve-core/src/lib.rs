//! Shared types for the nerve workspace.
//!
//! This crate is the contract three independent crates build against. It owns
//! the data model and nothing else — no generation, no invariants, no models.

pub type Vec3 = [f64; 3];

/// A single polymer chain: an ordered sequence of bead positions.
///
/// Order is meaningful. A chain is an *embedded curve*, so permuting beads is
/// not a symmetry — this is exactly the opposite of a point cloud, and it is
/// the whole reason chain topology exists as a signal.
#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    pub beads: Vec<Vec3>,
}

impl Chain {
    pub fn new(beads: Vec<Vec3>) -> Self {
        Self { beads }
    }
    pub fn len(&self) -> usize {
        self.beads.len()
    }
    pub fn is_empty(&self) -> bool {
        self.beads.is_empty()
    }
}

/// A melt: many chains in a cubic periodic box of side `box_len`.
#[derive(Clone, Debug, PartialEq)]
pub struct Melt {
    pub chains: Vec<Chain>,
    pub box_len: f64,
}

impl Melt {
    pub fn new(chains: Vec<Chain>, box_len: f64) -> Self {
        assert!(box_len > 0.0, "box_len must be positive, got {box_len}");
        Self { chains, box_len }
    }

    pub fn n_beads(&self) -> usize {
        self.chains.iter().map(Chain::len).sum()
    }

    /// Minimum-image displacement from `a` to `b` under cubic PBC.
    ///
    /// Every distance in this workspace goes through here. A chain that is
    /// unwrapped across the boundary and a chain that is wrapped must give the
    /// same answer, or every downstream invariant is measuring the box rather
    /// than the polymer.
    pub fn min_image(&self, a: Vec3, b: Vec3) -> Vec3 {
        let l = self.box_len;
        let mut d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        for x in &mut d {
            *x -= l * (*x / l).round();
        }
        d
    }

    pub fn min_image_dist(&self, a: Vec3, b: Vec3) -> f64 {
        let d = self.min_image(a, b);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_image_wraps_across_boundary() {
        let m = Melt::new(vec![], 10.0);
        // 0.5 and 9.5 are 1.0 apart through the wall, not 9.0 across the box.
        assert!((m.min_image_dist([0.5, 0.0, 0.0], [9.5, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn min_image_is_antisymmetric() {
        let m = Melt::new(vec![], 10.0);
        let (a, b) = ([1.0, 2.0, 3.0], [8.0, 1.0, 9.0]);
        let f = m.min_image(a, b);
        let r = m.min_image(b, a);
        for i in 0..3 {
            assert!((f[i] + r[i]).abs() < 1e-12, "axis {i}: {} vs {}", f[i], r[i]);
        }
    }
}
