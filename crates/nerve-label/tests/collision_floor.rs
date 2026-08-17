//! Theorem 1 applied to the label question: a per-candidate quality metric that
//! needs no training run and no labelled set.
//!
//! Given `n` configurations whose true label differs pairwise, and a descriptor
//! emitting `m` distinct values, any predictor built on that descriptor is wrong on
//! at least `n - m` of them. That converts the existence proofs of earlier rounds
//! — "here is one pair the descriptor cannot separate" — into a **rate**.
//!
//! Theorems, proofs and API are `branchcut`'s (Teerth Sharma, MIT,
//! <https://github.com/teerthsharma/branchcut>), `branchcut/partition.py:9-44`.
//! Ported, not re-derived. The contribution here is only where they are pointed.

use nerve_label::*;

const N: usize = 60;
const R: f64 = 3.0;
const SPACING: f64 = 24.0;
/// Well below the 3.0 intra-pair gap, so the descriptor cannot see threading, and
/// far below `box_len / 2` for every ladder built here.
const R_CUT: f64 = 2.0;
const ATOL: f64 = 1e-9;

fn ladder(p: usize, threaded: usize) -> nerve_core::Melt {
    linking_ladder(p, threaded, N, R, SPACING)
}

/// All `p + 1` rungs: `threaded = 0 ..= p`.
fn rungs(p: usize) -> Vec<nerve_core::Melt> {
    (0..=p).map(|j| ladder(p, j)).collect()
}

// ------------------------------------------------------- the port, checked

/// Closed-form cases from Theorem 1. **Falsified if** the port disagrees with the
/// theorem's arithmetic on any of them.
#[test]
fn min_errors_matches_theorem_one_closed_forms() {
    assert_eq!(min_errors(5, 1), 4, "total collapse must certify n-1 errors");
    assert_eq!(min_errors(5, 5), 0, "full separation must certify nothing");
    assert_eq!(min_errors(9, 3), 6);
    assert_eq!(min_errors(1, 1), 0);
    assert_eq!(collision_error_floor(9, 1), 8.0 / 9.0);
    assert_eq!(collision_error_floor(4, 4), 0.0);
}

/// Theorem 2. **Falsified if** the ceiling is not `1/k`.
#[test]
fn recovery_ceiling_matches_theorem_two() {
    assert_eq!(recovery_ceiling(1), 1.0);
    assert_eq!(recovery_ceiling(2), 0.5);
    assert_eq!(recovery_ceiling(10), 0.1);
}

// ------------------------------------------------- the precondition, both ways

/// Theorem 1's precondition, satisfied by construction. **Falsified if** two rungs
/// share a linking number, which would void every bound below.
#[test]
fn linking_ladder_labels_are_pairwise_distinct() {
    let p = 8;
    let labels: Vec<f64> = rungs(p).iter().map(total_abs_linking_by_pairs).collect();
    println!("ladder labels: {labels:?}");
    for (j, l) in labels.iter().enumerate() {
        assert!((l - j as f64).abs() < 1e-6, "rung {j} has label {l}, expected {j}");
    }
    let part = partition_by_descriptor(&rungs(p), total_abs_linking_by_pairs, R_CUT, ATOL);
    assert!(part.injective, "precondition failed: labels are not pairwise distinct");
}

/// The guard branchcut insists on, and the one this project kept missing.
/// **Falsified if** a non-injective label map still returns a non-zero floor —
/// a number that looks like a score but certifies nothing is worse than no number.
///
/// Writhe is the natural example: threading does not change either ring's own
/// shape, so every rung carries the same writhe and the label map is many-to-one.
#[test]
fn floor_certifies_nothing_when_the_label_map_is_many_to_one() {
    let p = 8;
    let part = partition_by_descriptor(&rungs(p), total_writhe, R_CUT, ATOL);
    let w: Vec<f64> = rungs(p).iter().map(total_writhe).collect();
    println!("writhe across rungs: {w:?}");
    println!(
        "writhe partition: injective {}, n {}, m {}, certified errors {}, rate {}",
        part.injective,
        part.n(),
        part.m(),
        part.certified_errors(),
        part.certified_error_rate()
    );
    assert!(!part.injective, "writhe should be constant across rungs, making R many-to-one");
    assert_eq!(part.certified_errors(), 0, "non-injective input must certify nothing");
    assert_eq!(part.certified_error_rate(), 0.0);
}

// ------------------------------------------------------------- the headline

/// The descriptor collapses every rung into a single block. **Falsified if** `m > 1`,
/// i.e. the descriptor can see threading after all.
#[test]
fn descriptor_collapses_the_whole_linking_ladder_into_one_block() {
    let p = 8;
    let part = partition_by_descriptor(&rungs(p), total_abs_linking_by_pairs, R_CUT, ATOL);
    println!(
        "n {}, m {}, largest block {}, recovery ceiling {}",
        part.n(),
        part.m(),
        part.largest(),
        part.recovery_ceiling()
    );
    assert_eq!(part.m(), 1, "descriptor separated the rungs into {} blocks", part.m());
    assert_eq!(part.largest(), p + 1);
    assert!(part.collapsed(), "provable loss should be flagged");
}

/// Theorem 1 as a blindness **rate**, and the number this crate contributes to the
/// label question. **Falsified if** the rate fails to rise toward 1 with ladder
/// length, which would mean the collapse is a small-`n` artifact.
#[test]
fn collision_error_floor_on_the_linking_ladder_rises_toward_one() {
    let mut rows = Vec::new();
    for p in [2usize, 4, 8, 16] {
        let part = partition_by_descriptor(&rungs(p), total_abs_linking_by_pairs, R_CUT, ATOL);
        let rate = part.certified_error_rate();
        println!(
            "p {p:>3}: n {} m {} certified errors {} rate {rate:.6} recovery ceiling {:.6}",
            part.n(),
            part.m(),
            part.certified_errors(),
            part.recovery_ceiling()
        );
        rows.push((p, part.certified_errors(), rate));
    }
    for (p, errs, rate) in &rows {
        assert_eq!(*errs, *p, "expected p certified errors at p = {p}, got {errs}");
        let expected = *p as f64 / (*p as f64 + 1.0);
        assert!((rate - expected).abs() < 1e-12, "rate {rate} != p/(p+1) = {expected}");
    }
    let last = rows.last().unwrap();
    assert!(last.2 > 0.94, "rate at p = {} only {:.6}", last.0, last.2);
}

/// The systematic form of matched-pair construction. **Falsified if** any rung pair
/// is left unmerged, which would mean the collapse is partial.
#[test]
fn collides_finds_every_rung_pair_as_a_matched_pair() {
    let p = 8;
    let cs = collides(&rungs(p), R_CUT, ATOL);
    let expected = (p + 1) * p / 2;
    println!("collides found {} merged pairs, expected all {expected}", cs.len());
    assert_eq!(cs.len(), expected, "descriptor merged only {} of {expected} pairs", cs.len());
}

/// Guards.
#[test]
fn ladder_guards_hold() {
    for p in [2usize, 4, 8, 16] {
        let m = ladder(p, p / 2);
        let mb = max_bond(&m);
        println!("p {p}: box_len {}, max_bond {mb:.6}, r_cut {R_CUT}", m.box_len);
        assert!(mb < m.box_len / 2.0, "p {p}: bond guard violated at {mb}");
        assert!(R_CUT <= m.box_len / 2.0, "p {p}: r_cut exceeds box_len/2");
        assert!(
            min_interchain_dist(&m) > R_CUT,
            "p {p}: chains within the cutoff of each other, so the collapse is not clean"
        );
    }
}
