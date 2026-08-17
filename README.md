<h1 align="center">Nerve</h1>

<p align="center">
  <strong>Topological witnesses for polymer chains in Rust — and the harness that falsified three of its own four hypotheses.</strong>
</p>

<p align="center">
  <strong>A local atomic descriptor is a many-to-one map. Nerve measures exactly which chain topology it merges, certifies an error floor with no labelled data, and withdraws every claim that fails its own control.</strong><br>
  <a href="https://github.com/teerthsharma/nerve">github.com/teerthsharma/nerve</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&color=00aaff" alt="MIT"></a>
  <a href="#results"><img src="https://img.shields.io/badge/tests-224%20passing-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="#limitations"><img src="https://img.shields.io/badge/hypotheses-3%20of%204%20withdrawn-critical?style=flat-square" alt="Withdrawn"></a>
  <a href="#3-alexander-determinant-at-t--1"><img src="https://img.shields.io/badge/Alexander-no%20crossing%20signs-blueviolet?style=flat-square" alt="Alexander"></a>
  <a href="#results"><img src="https://img.shields.io/badge/knotted%20fraction-15.2%25%20at%20N%3D823-ff6b35?style=flat-square" alt="Knotted"></a>
  <a href="#results"><img src="https://img.shields.io/badge/FENE%20bond%20sd-3.401%25%20measured-00aaff?style=flat-square" alt="FENE"></a>
  <a href="#2-collision-error-bound"><img src="https://img.shields.io/badge/error%20floor-label--free-yellow?style=flat-square" alt="Floor"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/rust-2021%20edition-orange?style=flat-square" alt="Rust"></a>
  <a href="#limitations"><img src="https://img.shields.io/badge/periodic%20linking-unsolved-lightgrey?style=flat-square" alt="Periodic"></a>
</p>

<p align="center">
  <sub>
  <code>topological-data-analysis</code> · <code>polymer-physics</code> · <code>knot-theory</code> ·
  <code>gauss-linking-number</code> · <code>alexander-polynomial</code> · <code>writhe</code><br>
  <code>kremer-grest</code> · <code>molecular-dynamics</code> · <code>machine-learning-potentials</code> ·
  <code>descriptor-incompleteness</code> · <code>rust</code> · <code>computational-topology</code>
  </sub>
</p>

---

## Abstract

We present Nerve, eight Rust crates that measure whether a local atomic descriptor
can represent the topology of a polymer chain. The core insight is that an
atom-centred descriptor is a many-to-one map, so blindness is not an opinion about
model capacity but a collision that can be exhibited and counted: where the map
merges two configurations of different topology, the error floor is `n − m` from
the map's own outputs, with no labelled set in existence. Nerve computes Gauss
linking numbers in closed form — a chain is already a polygon, so the segment-pair
double integral is a signed solid angle with no quadrature term and no length
constant — plus writhe, closure ensembles, and the Alexander determinant at
`t = −1`, where positive- and negative-crossing rows are exact negatives and
`|det|` therefore requires no crossing signs at all. Against MD-equilibrated
Kremer-Grest melts from Svaneborg & Everaers (Zenodo 7319837), the measured FENE
bond distribution is `sd/mean = 3.401%`, the knotted fraction is 15.2% at N=823
against a published 23.6% at N=1024, and the single-nearest-image convention
resolves to `frac_ambiguous = 0.000`. Four blindness hypotheses were tested and
three were withdrawn on measurements taken here. 224 tests pass; ten are kept
deliberately failing to record predictions that turned out false.

**Keywords:** Gauss linking number · Alexander determinant · writhe · Kremer-Grest
melts · descriptor incompleteness · many-to-one maps · persistent topology

---

## Background

### Why measure collisions instead of model capacity?

**A descriptor merges configurations, and no later stage undoes it.** If a map
sends `k` configurations to one value, then for *any* downstream `h`,
`Pr[h(f(x)) = x] ≤ 1/k`, because `h ∘ f` is constant on the block. Widening the
cutoff, adding layers, or adding a re-ranker cannot recover what the descriptor
already discarded. This converts "is the model expressive enough" into "which
inputs does this map merge", which is checkable.

**No local criterion can decide it.** There exists a smooth `F` with nonsingular
Jacobian everywhere that is not injective — `F(x,y) = (eˣ cos y, eˣ sin y)`, where
`det DF = e²ˣ > 0` yet `F(x,y) = F(x, y+2π)`. The two preimages' Jacobians differ
by a rotation and share every spectral invariant, so no pointwise function of the
Jacobian distinguishes injective from many-to-one. The escape is global and cheap:
evaluate the map on a population and return the pairs that merge.

**Chain topology is the interesting merged set.** Knot type and linking number are
global invariants of an embedded curve. A cutoff descriptor sees a bounded
neighbourhood; a polymer's entanglement does not live there. Whether that gap is
real, and at what density and chain length, is what this repository measures.

**The controls are the product.** Every claim here is stated against a named
alternative: a same-architecture different-seed noise floor rather than a raw
distance, cheap chain features (`Rg`, MSID, end-to-end, contour length) rather than
a strawman local descriptor, and a measured `|Lk|` significance scale rather than a
threshold chosen by eye. Three of four hypotheses died to those controls.

### Prior Art

| System | Language | Periodic linking | Knot type | Open-chain closure | Ground-truth tests | Notes |
|---|---|---|---|---|---|---|
| [TEPPP](https://github.com/TEPPP-software/TEPPP) 2023 | C++17 / MPI | **`periodic_lk`, `periodic_wr`** | Jones, no closure | — | not published | **Better than Nerve here.** The published periodic treatment; Nerve does not implement it |
| [KymoKnot](https://github.com/luca-tubiana/KymoKnot) 2018 | C / Python | — | Alexander at `t=−1, −2` | **minimally-interfering** | not published | **Better than Nerve here.** True convex-hull MIC; Nerve uses a bounding-sphere proxy |
| [Z1+](https://doi.org/10.17632/m425t6xtwr.1) 2023 | Fortran 90 | primitive paths | — | — | not published | Kink counts and `N_e`; different observable |
| [Topoly](https://academic.oup.com/bib/article/22/3/bbaa196/5906197) 2021 | Python | — | full polynomials | closure ensembles | not published | Broader invariant set, slower |
| [TopologyNet](https://doi.org/10.1371/journal.pcbi.1005690) 2017 | Python | — | persistent homology | — | not published | **Better than Nerve here.** Topology in, *free energy* out — nine years before this repo |
| **Nerve** | Rust 2021 | single nearest image, ambiguity bounded | `\|Δ(−1)\|` exact `i128` | stochastic + proxy MIC | **224, incl. classical table** | Falsification harness, not a general TDA library |

Nerve does not win the invariant-coverage comparison and does not try to. What it
has that these do not is a test suite that asserts the classical knot-determinant
table, the Hopf and `(2,4)` torus link values, wrapped-vs-unwrapped agreement, and
a bitwise scale-invariance property.

**On mechanism novelty, stated plainly.** Parity blindness of distance-only
descriptors is Havel, Kuntz & Crippen (*Bull. Math. Biol.*, 1983); for GNNs it is
Pattanaik et al. ([arXiv:2110.04383](https://arxiv.org/abs/2110.04383)). The
reflection identity `c_l → (−1)^l c_l` is Bartók, Kondor & Csányi (*PRB* **87**,
184115, 2013, Eq. 17). Body-order incompleteness and the additivity caveat are
Pozdnyakov et al. (*PRL* **125**, 166001, 2020) — whose full text contains **zero**
occurrences of *polymer*, *chain*, *knot*, *linking number*, *writhe*, or
*topology*. Persistent homology as an MLIP descriptor is Minamitani et al. (*JCP*
**159**, 084101, 2023); writhe as a polymer ML feature is Sleiman et al. (*Soft
Matter* **20**, 71, 2024). Nerve's contribution is the application to chain
topology and the measurements below, not the mechanisms.

---

## Theoretical Foundation

### 1) Gauss linking number in closed form

For two closed polygons the double integral reduces to a sum of signed solid
angles over segment pairs, evaluated by Van Oosterom–Strackee:

$$
\mathrm{Lk}(A,B) \;=\; \frac{1}{4\pi} \sum_{i}\sum_{j} \Omega\big(r_{13}, r_{14}, r_{24}, r_{23}\big)
$$

where $r_{ab}$ are endpoint difference vectors of segment pair $(i,j)$ and $\Omega$
is the signed spherical-quadrilateral area. There is **no quadrature error term and
no length constant** — degeneracy is caught by NaN propagation rather than an
epsilon, which is why bitwise scale invariance holds exactly.

Integer-valuedness for closed curves is the accuracy metric, and it is
chain-length dependent: deviation rises from `2.89e-15` at `M=64` segments to
`2.16e-13` at `M=1024`. Do not quote one figure across chain lengths.

### 2) Collision error bound

For $R$ injective on $|X| = n$ and any $f$ producing $m$ distinct values:

$$
\mathrm{err}(f) \;\geq\; n - m,
\qquad
\Pr\big[h(f(x)) = x\big] \;\leq\; \frac{1}{k}
$$

The first is computed from $f$'s own outputs — no labels. The second caps any
downstream stage once $k$ inputs pool. Both are ported from
[`branchcut`](https://github.com/teerthsharma/branchcut) and cited, not re-derived.

**The precondition is enforced, not assumed.** If $R$ is genuinely many-to-one the
bound certifies nothing, so the port carries the guard: `|Lk|` is heavily
many-to-one over reconnections (1294/1680 chain pairs below 0.1), and traversal
reversal preserves writhe to `3.66e-15`. In both cases the bound returns zero
rather than a plausible wrong number.

### 3) Alexander determinant at `t = −1`

For a knot diagram with $c$ crossings, the Alexander matrix at $t = -1$ has rows
in which the positive- and negative-crossing forms are exact negatives:

$$
\big|\Delta(-1)\big| \;=\; \big|\det M\big|,
\qquad
M^{+}_{\text{row}} = -\,M^{-}_{\text{row}}
$$

so $|\det|$ is unchanged by crossing sign and **no crossing sign is computed
anywhere**. This deletes the most error-prone step in a knot pipeline. The
determinant is exact integer Bareiss in `i128`, returning `None` on overflow rather
than a wrapped value.

Mirror-invariance is a theorem, not an artifact: reflecting through the projection
plane leaves the 2-D diagram identical while swapping over/under at *every*
crossing, so the matrix is rebuilt from different combinatorics and the
determinants still agree.

### 4) Unwrapping is a lift, minimum image is not

`min_image` returns a *displacement*. Applying it per segment inside the Gauss
integrand leaves consecutive segments not sharing endpoints — the chain stops being
a curve, and the integral stops being the degree of a Gauss map, so it is no longer
a linking number even though it still evaluates. Chains are instead unwrapped by
accumulating minimum-image **bonds**, a discrete lift through the covering map
$\mathbb{R}^3 \to \mathbb{R}^3/L\mathbb{Z}^3$, unique up to one box vector. This is
the load-bearing convention in the repository.

---

## Implementation

Eight crates. `nerve-core` is a frozen contract; the others depend on it and not on
each other, so a defect cannot silently propagate between them.

### nerve-core, nerve-topo — the measurement layer
`Chain`, `Melt`, minimum-image displacement; then linking (closed and open), writhe
(closure-free), `unwrap_chain`, closure schemes, and `image_spread` /
`closure_spread` as ambiguity diagnostics. Cost is `O(C²M²)` for overlapping chain
pairs; a bounding-sphere prefilter over centroids is a requirement rather than an
optimisation at melt scale.

### nerve-baseline — the null model
Behler-Parrinello ACSF: 8 radial G2 shells plus 4 angular G4 exponents
(`ζ = 1,2,4,8`), cosine-cutoff damped on all three legs of each triplet, pooled as
`[mean, spread]` over beads. Plus the cheap non-topological features that any
topological claim must beat: `rg2`, `end_to_end2`, `contour_len`, `bead_density`,
`msid`, `max_bond`.

### nerve-melt — generation, witness, reader
Ideal and rejection-grown melts, the Alexander witness with stochastic closure, and
a LAMMPS data reader that derives chain order from the `Bonds` section rather than
file order. Reading in file order on a real archive file gives bond lengths of
1.5/0.8/1.3 where the truth is 0.7/0.8/0.5, corrupting every chain feature — the
reader asserts a cubic cell, a simple path per molecule, and `max_bond < box_len/2`.

### nerve-blind, nerve-order, nerve-orient, nerve-label
The four hypothesis crates: matched-pair construction, the body-order and
sum-decomposability hierarchy, traversal reversal and double bridging, and label
ranking by measured discontinuity.

---

## Results

All numbers below were re-run independently of the process that produced them.

### Label selection — which target is learnable

| Candidate | Jump height | Informativeness `z` | Estimator freedom |
|---|---|---|---|
| **`\|Lk\|` closed, raw coords** | **1.1e-18** | **8.9e14** | none |
| closure-averaged `Lk` | 3.3e-16 | pairwise | `sd 0.331–0.559` |
| `writhe` | 0e0 | **0.174** | none |
| `Lk(Open)` | **6.6e-3** | pairwise | `sd 0.331–0.559` |

`|Lk|` jumps by exactly `1.000000`, only where closest approach is `0.025210` — a
real strand crossing. Largest step on a benign path: `1.34e-16`.

**`Lk(Open)` jumps at the knife edge where the descriptor does not move.** At
centroid separation `box_len/2` the minimum image folds symmetrically and `ΔD`
reads `1.954e-14` — the same value at every `h` from `1e-3` to `1e-9`, which is
machine epsilon on order-1 doubles rather than a small-but-finite response. The
descriptor is *exactly invariant* along this family, not merely flat, so any
amplification ratio formed from that denominator measures `1/eps` and is not
reported here. What survives is the qualitative fact and its repair: the label
moves by `6.6e-3` where the descriptor does not move at all, and closure
averaging removes the jump (`3.3e-16` against `6.6e-3`).

**Writhe is exactly constant on this ladder, which makes the ladder the wrong
instrument for it.** Writhe is translation-invariant, so it is constant on any
rigid-translation family — and so is *every* topological invariant, which means
the ladder cannot rank them against each other at all. Its `z = 0.174` is a
property of the probe, not a measurement of writhe: published classifiers reach
>95% knot-type accuracy from local writhe (Sleiman et al., *Soft Matter* **20**,
71, 2024). Writhe is not ranked here.

### The surviving witness

| Metric | Value | Notes |
|---|---|---|
| Classical determinants | 1, 3, 5, 5, 7 | unknot, 3₁, 4₁, 5₁, 7₁ — exact |
| Rotation invariance | exact | four rotations of the knot |
| Mirror pair | `Some(3)` vs `Some(3)` | **regression guard, not evidence** — `Δ_{K*} = Δ_K` holds for every knot |
| Pseudoscalar contrast | **7.1641e-1** | bound: the fingerprint separates the pair the determinant cannot |
| Informativeness | changes, ends at unknot | strand-crossing path; the crossing is not independently detected |

**Mirror-blindness is definitional and is reported as a regression guard, not as a
result.** `Δ_{K*} = Δ_K` up to units for every knot, so equal determinants on a
mirror pair have no counterexample and that assertion cannot fail. It is kept
because it exercises over/under bookkeeping through the diagram builder. The bound
half of that row is the pseudoscalar contrast: a per-bead signed-volume fingerprint
separates the same pair at `7.1641e-1`, so the two channels are demonstrably not
measuring the same thing.

| Closure convergence | `p = 0.967` | modal label over 60 resolved closures |
| Closure count needed | **~40** | `p = 0.900 → 0.9625` over 10/20/40/80/160 |
| Discretisation | independent | 25/25 match at 60/120/240/480/960 points |

### Real Kremer-Grest melts

Substrate: Svaneborg & Everaers, [Zenodo 7319837](https://zenodo.org/records/7319837),
CC-BY-4.0, MD-equilibrated, `Z = 100` entanglements per chain. **`M=500` in that
title is the chain count, not beads per chain.**

| Quantity | κ = 5.50 | κ = 0.00 |
|---|---|---|
| chains × beads | 414 × **823** | 517 × **8,408** |
| atoms | 340,722 | 4,346,936 |
| box length | 73.747 | 172.291 |

| Metric | Value | Notes |
|---|---|---|
| bead density | **0.8495 / 0.8504** | κ=5.50 / κ=4.00 — real KG density, genuine excluded volume |
| FENE bond mean | **0.96401** | literature `l_b = 0.965 σ` |
| FENE bond sd | 0.03279 | **relative sd 3.401%**, range 0.845–1.183 |
| max bond | 1.18336 | **6.7 sd** above mean over 340k bonds |
| `frac_ambiguous` | **0.000** | at each N's measured p95; `mean_carriers 0.05–0.20` |
| **knotted fraction, N=823** | **0.1522 / 0.1546 / 0.1522** | seeds 1/2/3 — 15.2% ± 0.2% |
| **knotted fraction, N=896** | **0.1789** | κ=4.00; monotone in N |
| determinant resolved | **414/414, 436/436** | zero `i128` overflow |
| modal closure probability | **0.9709 / 0.9616** | 20 stochastic closures per chain |
| MIC proxy agreement | **409/414 (98.8%)**, 427/436 (97.9%) | vs the stochastic mode |
| determinant collisions | **4.11% / 3.67%** | chains sharing `\|Δ(−1)\|` with another knot — lower bound |

The knotting fractions are the load-bearing external check: published Kremer-Grest
melts give 23.6% at N=1024 for stiff chains, and these 823- and 896-bead
measurements are monotone in N and sit below it. 85% of chains (351/414) are modal
unknot, which is the correct physics at this chain length and *contradicted* the
expectation this measurement was set up to confirm.

**The `i128` determinant does not overflow on real chains, and the reason is
structural:** the Alexander matrix carries exactly three nonzeros per row, so its
minors stay far below the dense-matrix bound. The "few tens of crossings" ceiling
stated elsewhere in this codebase is too pessimistic for this matrix family.

**Hand-grown melts were vindicated rather than corrected.** Writhe sign-resolvable
fraction was `0.583–0.667` on rejection-grown melts at N=100 and ρ ≤ 0.60; on the
archive it is `0.5676` and `0.6078`. The ranges overlap, so the excluded-volume,
equilibration, and entanglement-length caveats retire without changing any
conclusion. Pooling cancels *far more* on real melts — `|ΣWr|/Σ|Wr|` of `0.0012`
and `0.0564` against `0.201` hand-grown, with a Theorem 2 ceiling of `1/414 =
0.00242` — so the pooling result is strengthened.

**κ=0.00 was not swept, and the cost is stated rather than hidden:** Alexander is
`O(M²)` per closure at a measured `0.11 s/chain` for 823 beads, scaling to
`≈11.5 s/chain` at 8,408 beads and **≈1.6 hours** for all 517 chains. That file is
where ~75% knotting lives and it needs an explicit subset.

The second row of that last table is a result the archive alone could give: the
single-nearest-image convention shipped documented as *genuinely unresolved*, and
on real melts never more than one periodic image carries linking — despite chains
spanning ~10.7 box lengths.

`|Lk|` significance is **lower** on real melts than on synthetic fixtures, which
made one withdrawal thinner rather than firmer: p95 `|Lk|` is `0.0191` at N=20,
`0.1533` at N=100, `0.9341` at N=200, `0.6814` at N=823, against a synthetic
reference of `1.2327`.

### Test suite

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  nerve-core     2     nerve-order    17      224 passing, 0 failing
  nerve-topo    44     nerve-label    25      10 kept deliberately failing
  nerve-blind   32     nerve-orient   49      clippy -D warnings clean
  nerve-baseline 16    nerve-melt     39
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Reproduce with `cargo test --workspace`. The ten `#[ignore]`d tests are not
disabled failures — each carries a falsified prediction and its measured number in
the reason string, and `cargo test --workspace -- --ignored` shows them still
failing.

**One defect worth reporting because invariant tests could not catch it.** A sign
error survived permutation, isometry, scale, and ground-truth tests, and was found
only by cross-implementation parity: the closed form returned
`+1.0000000000000313` against an independent midpoint quadrature's
`−1.0000414542607363` — magnitudes agreeing to `4e-5`, sign inverted. Ground truth
asserts `|Lk|`; reflection is antisymmetric either way. The convention was
subsequently derived from a Seifert disc (one piercing of sign −1, so `Lk = −1`),
and the opposite convention now fails a test.

---

## Quick Start

```toml
[dependencies]
nerve-topo = { git = "https://github.com/teerthsharma/nerve" }
nerve-core = { git = "https://github.com/teerthsharma/nerve" }
```

```rust
use nerve_core::{Chain, Melt};
use nerve_topo::{linking_number, writhe, Closure};

// Two rings in a periodic box; linking is computed on unwrapped coordinates.
let melt = Melt::new(vec![Chain::new(ring_a), Chain::new(ring_b)], 20.0);

let lk = linking_number(&melt, 0, 1, Closure::Direct);   // ±1 for a Hopf link
let wr = writhe(&melt, 0);                               // closure-free

// How much of that number is the periodic image choice's doing?
let amb = nerve_topo::image_spread(&melt, 0, 1, Closure::Direct);
assert!(amb.max - amb.min < 1e-9, "image choice is not determined here");
```

The `image_spread` assertion is the intended usage pattern: the ambiguity is
reported rather than hidden, and a caller decides whether the number is admissible
on their data.

---

## Requirements

- **Rust 2021 edition.** No nightly features, no `unsafe`.
- **Dependencies:** `rand`, `rand_chacha`, `proptest` (dev only). No BLAS, no C or
  C++ dependency, no Python.
- **Architecture-independent.** No SIMD intrinsics or CPU feature detection; the
  linking kernel is scalar `f64` and the Alexander determinant is integer `i128`.
- **No CI, by choice.** Correctness is argued from ground-truth tests and kept-
  failing predictions rather than a badge.
- Real-melt sections need the archive, which is not vendored:

```bash
curl -L -o kg.gz "https://zenodo.org/records/7319837/files/kg.kappa5.50.TZ100.M414.n3.F2.final.input.gz?download=1"
gunzip -c kg.gz > kg.kappa5.50.data
```

---

## Limitations

**Three of the four hypotheses this repository was built to test were withdrawn,
and the withdrawals are the substance.**

*Parity blindness* was struck as unbound: the two tests carrying the claim were
green with no prior red, and the blindness test asserted `signal == 0.0` then
`signal/noise == 0.0` — the second implied by the first — with a mirror operation
that made a distance-based descriptor bitwise identical *by construction*. The
mechanism is also 43-year-old prior art.

*Connectivity blindness* was withdrawn by its own author on three compounding
biases, all favouring the result: no excluded volume (13.5×), a melt-wide-maximum
length reference (~2×), and a `|Lk|` threshold chosen by eye at `0.1` against a
measured significance of `1.2327` (12×). The headline `0.752%` feasibility was
`n = 1`, a single melt, against `0.254%` over 40. The real-melt follow-up did not
settle it either: the search enumerated `for k in 1..n` with a single crossover
index shared by both chains, so it can only ever emit same-index crossovers and
its count is the cardinality of that loop rather than a sample. That measurement
is withdrawn as circular, and the question of whether a non-same-index
length-preserving reconnection exists on real melt geometry is open.

**A published result cuts against the motivating premise, and it belongs here
rather than in a reviewer's first comment.** Sleiman, Conforto, Gutierrez Fosado &
Michieletto (*Soft Matter* **20**, 71, 2024) classify **all prime knots up to 10
crossings at >95% from local writhe**, and separate mutants and composites that
knot polynomials cannot; Zhang, Zhu & Dai
([arXiv:2501.12780](https://arxiv.org/abs/2501.12780)) reach >99% knot-type
accuracy. Local writhe is a sum of local pairwise terms — exactly the functional
form a cutoff descriptor can express. Any claim that local descriptors are
*structurally* blind to knot type has to survive that, and this repository does not
make it. What it does claim is narrower and theorem-backed: `|Δ(−1)|` is
mirror-blind by construction of the Alexander matrix, and that is not exposed to
this counter-evidence. Relatedly, Bupathy et al.
([arXiv:2511.23265](https://arxiv.org/abs/2511.23265), 2025) build an ML potential
for *knotted* solitons and report that "handed interactions emerge naturally and
can be fully captured even without explicitly chiral descriptors."

*Body-order truncation* buys **exactly zero** extra reach: 2-body and 3-body
residuals are identical at every cutoff, `0e0` below the strand gap and `inf`
above, same threshold. Reported as a null result.

*Sum-decomposability* survives but is **cutoff-bounded** — it needs strands farther
apart than the cutoff, which is false in a dense melt at ~1σ. The mechanism that
reaches the pseudoscalar-carrying model class is the one that does not survive melt
density.

**Periodic linking is not solved.** `linking_number` uses the single nearest image
by centroid separation. The published treatment is Panagiotou, *J. Comput. Phys.*
**300**, 533 (2015), implemented in TEPPP as `periodic_lk`, and is not ported here.
`image_spread` bounds the ambiguity; the real-melt `frac_ambiguous = 0.000` is
evidence, not a proof.

**`|Δ(−1)|` is not a complete invariant** — 4₁ and 5₁ both give 5, measured here
rather than cited. Determinants at `t = −1` with `t = −2` still fail to separate
some prime knots at 8+ crossings.

**The minimally-interfering closure is a bounding-sphere proxy** for the convex
hull and must not be quoted as MIC proper.

**A bond longer than `box_len/2` cannot be unwrapped by any means.** The
information is gone and `unwrap_chain` will silently pick the wrong image; the
guard is asserted at every entry point but is the caller's responsibility.

**Scale is under-tested.** Most hypothesis work ran at 8 chains × 20 beads, while
real melts are 414–517 chains of 823–8,408 beads — and residuals were measured
*rising* with N rather than diluting, so extrapolation from the small fixture is
unsafe in either direction.

**Not a general TDA library.** No persistent homology, no Vietoris-Rips, no Mapper.
For those, [`topological-ml-toolkit`](https://github.com/teerthsharma/topological-ml-toolkit)
has ripser and GUDHI parity that this repository does not attempt.

---

## License

MIT. See [LICENSE](LICENSE).

*Invented by [Teerth Sharma](https://teerthsharma.vercel.app)*
