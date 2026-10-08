# Closed-form whitened truncation: first-move results

Plan: `crosscut_whitening_novelty.md`. Code and raw outputs: `whiten/` (nothing outside it was edited).

## Verdict: PARK (the Kronecker-GN route is KILLED)

The closed-form whitened SVD keeps a real but partial share of spcf's gain, only with oracle-grade (expensive) ingredients; no cheap construction exists, and by counting one would not reach the planned 1-2x SVD cost.

- **Kronecker GN step (`kfac`, plan arm A4): dead.** With deployable Shampoo factors its median retention is 0.10 at best and diverges for small damping.
- **Whitened SVD (A5):**
  - Per-cut objective, pooled over 419 cuts: it keeps 0.71 of the objective drop that spcf achieves.
  - End-to-end log-gain relative to spcf: g = 0.39 / 0.54 / 0.75 at χ = 12 / 16 / 24 (Ising N=16, T=5).
  - Target-free ingredients (static rows, linearised at M): g = 0.35 / 0.48 / 0.70.
  - Success needed g ≥ 0.6 at CPU ≤ 2.5x SVD, and the kill was g < 0.3 at every χ, so neither is met.
- **Revival (only if wanted):** build the target-free static-string construction of the whitening factors L and R, and time it against `np.linalg.svd` at χ = 32 and 64.

## What was run

1. **Math validation** (`whiten/validate_math.py`, `results/validate_math.json`).
   - The whitened SVD is the global minimiser of the Kronecker-weighted rank-k problem. Closed form 0.0734150692 against theory 0.0734150692, best of 60 L-BFGS starts 0.0734151720, plain SVD 0.591.
   - Subspace form matches the weighted M_k to 3e-14, and μ=0 equals plain SVD to 2e-15.
   - Shampoo and Van Loan-Pitsianis factors are exact on product sets (4e-16). VLP on an exactly Kronecker matrix is exact (4e-16), and the structured Kronecker solve matches the dense one (1e-14).
   - One bug was found and fixed before any probe run: a spurious complex conjugate in the VLP right factor.
2. **Probe self-tests** (first 5 cuts per cell, stored in the records).
   - The float64 M-space Jacobian agrees with finite differences to 1e-10.
   - The adjoint dot-product test gives 1e-16.
   - It agrees with spcf's float32 jvp/vjp closures to 2e-7, and with `exact()` to 1.5e-7.
   - The spcf replica and the logged `f_c` agree to 1e-8.
3. **T0** (`prof_spcfast.py`, `results/t0_prof_spcfast_raw.txt`).
4. **T1 oracle probe** (`whiten/whiten_probe.py`, `wlib.py`; `results/t1_*`).
   - Cells: Ising N=12 T=4 χ=8 (all 189 fired cuts) and Ising N=16 T=5 χ=12 (every 2nd of 459, 230 cuts).
   - Driven by production spcf (float32, best configuration).
   - Every arm is scored on the same exact float64 objective.
5. **T3-lite** (`whiten/spcwhite.py`, `spcw_bench.py`; `results/t3lite_*`).
   - End-to-end TEBD against the cached dense reference, with SVD, spcf and the oracle-grade `SPCWhite`.
   - T2 was skipped because T1 did not pass.
   - The earlier run was killed and lost, so T1 was rerun.

## T0, time split

`cut.tm`, Ising N=16 T=5, under cProfile (inflates Python-heavy parts), literal command `... 2 0 4 0-1.0`:

| χ | svd | setup | solve | line | (setup+line)/total |
|---|---|---|---|---|---|
| 24 | 0.95 | 0.72 | 3.62 | 0.52 | 21% |
| 48 | 1.64 | 0.24 | 0.97 | 0.16 | 13% |

The same split with `taus=1.0` gives 22% and 12%.

Pre-registered threshold: setup + line > 40% means `kfac` is capped at about 2-2.5x cheaper. Not triggered; the iterative solve dominates the fired-cut cost (about 70% of non-SVD time). This is moot because `kfac` died in T1.

- **Deviation:** the plan's `0-1.0` makes `taus=[0,0,1.0]` (the static block is added automatically and gets duplicated). I ran both spellings and used `1.0` as the real configuration.
- `whiten/t0_timing.py` (interleaved per-fired-cut SVD ratio) was written but not run, for budget reasons. No CPU-ratio claim is made.

## T1 against the pre-registered criteria

Criteria (written before any T1 number):
- **PASS:** ρ(A3) ≥ 0.6 and ρ(A5) ≥ 0.5 and span-2 residual ≤ 0.6x SVD in ≥ 70% of cuts.
- **KILL:** ρ(A3) < 0.35.
- **AMBIGUOUS:** anything else.

Definitions:
- ρ = (f_SVD − f_arm)/(f_SVD − f_GN*), with f_GN* the exact dense-lstsq GN step at the best α in (1, .5, .25).
- ρ(A3) uses the best fixed damping δ on the pooled cuts. ρ(A5) is at α = 1 with no residual evaluation, at the best fixed μ among {1, …, 1e4}. The span-2 criterion is applied to A5 at that μ.

| | N=12 χ=8 (189 cuts) | N=16 χ=12 (230 cuts) | pooled (419) |
|---|---|---|---|
| median ρ(A3, oracle Kronecker GN) | 0.508 (δ=1e-4) | 0.417 (δ=1e-3) | **0.464** (δ=1e-3) |
| median ρ(A5, whitened SVD, μ=1e4) | 0.584 | 0.935 | **0.712** |
| A5 span-2 ≤ 0.6x SVD, fraction of cuts | 0.44 | 0.33 | **0.38** |
| spcf (A1p) own ρ / span-2 median | 0.86 / 0.36 | 1.36 / 0.30 | 0.91 / 0.35 |
| A5 retention relative to spcf's own drop | 0.71 | 0.71 | 0.71 |
| Verdict | AMBIGUOUS | AMBIGUOUS | **AMBIGUOUS** |

Ambiguous because ρ(A3) is in [0.35, 0.6) and the span-2 criterion fails (0.38 < 0.70), even though ρ(A5) passes. The pre-registered consequence was to go to `kfac` in T3.

Caveats and sensitivity:
- **A3 is fragile.** Cross-validated δ (chosen on N=12, tested on N=16) gives ρ = −1.43; chosen on N=16, tested on N=12 it gives 0.50. A3's own span-2 fraction is 0.24.
- **GN\* is a weak denominator at N=16.** The Jacobian has only about 1.4 residual rows per unknown, and the undamped GN step overfits (discarded weight 7x SVD's), so spcf scores ρ = 1.36 > 1. The retention "relative to spcf" row is the more informative one.
- **The metric has weak Kronecker structure.** Median fraction captured by the best rank-1 Kronecker fit is 0.66 in tangent coordinates and 0.43 in M-space. The relative Frobenius error of the fit is 0.58 and 0.75-0.77.
- **The A5 result is stable in μ.** It saturates for μ ≥ 1e2, so it does not depend on the damping scan.
- **Target-free ingredients work at oracle grade.** Using static rows only, linearised at the untruncated M, pooled retention relative to spcf is 0.63-0.68 (A5s0 and A5sm); all rows linearised at M gives 0.71, and all rows at the SVD point 0.67-0.71.
- **The oracle VLP factors (A5o) are slightly worse than Shampoo.** Pooled 0.65 against 0.71.

### Kronecker GN and the A6 pivot

The A6 pivot was not triggered by the pre-registered rule, because ρ(A3) is above 0.35. The data is there anyway. Pooled, relative to spcf's gain, median of 2 CG iterations:

| preconditioner | median vs spcf |
|---|---|
| oracle Kronecker (dense J) | 0.89 (1 iteration: 0.70) |
| Shampoo Kronecker (deployable) | 0.34 |

My pre-registered "matches spcf" bar was ≥ 0.9. The oracle version just misses it and is not deployable; the deployable one is far off. The Kronecker-GN and Kronecker-preconditioner routes are therefore dead.

## T3-lite: end-to-end, oracle-grade `SPCWhite`

Setup:
- `SPCWhite` builds Shampoo L and R from dense Γ_i. It is slow (about 20-30x spcf wall), so no timing claim is made.
- Two variants: `all` (all rows at the SVD point, μ=1e4) and `static` (static rows at M, μ=100).
- The accept check (`accept=1`) never changed an outcome in Ising.
- One deterministic trajectory per cell, dense Trotter reference. g = log(ratio_arm)/log(ratio_spcf) on the 2-site RDM error.

| cell | spcf rdm2/SVD | `all`: rdm2/SVD, g | `static`: rdm2/SVD, g |
|---|---|---|---|
| Ising N=12 T=4 χ=8 | 0.30 | 0.58, 0.46 | 0.62, 0.40 |
| Ising N=16 T=5 χ=12 | 0.23 | 0.56, 0.39 | 0.60, 0.35 |
| Ising N=16 T=5 χ=16 | 0.20 | 0.42, 0.54 | 0.47, 0.48 |
| Ising N=16 T=5 χ=24 | 0.21 | 0.31, 0.75 | 0.33, 0.70 |
| ising2 N=16 T=4 χ=12 | 0.37 | 0.61, 0.50 | 0.70, 0.37 |

- The nn-RMS g values are similar: 0.40 / 0.51 / 0.67 for `all`.
- Infidelity ratio to SVD is 0.73-0.84 for `all`, in line with or slightly better than spcf's 0.79-0.94.
- Success (g ≥ 0.6 at CPU ≤ 2.5x SVD) is not met: g < 0.6 at χ = 8, 12 and 16, and CPU is unproven. The kill (g < 0.3 at every χ) is not triggered.
- The Heisenberg control was not run (budget).

## Cost floor (back-of-envelope, not a measurement of the method)

`results/svd_vs_gemm.json` (single thread, shared machine, medians of 15 interleaved repeats): a complex SVD of a 2χ × 2χ matrix costs 21.7 / 22.8 / 16.5 / 10.8 complex matmuls of the same size at χ = 16 / 32 / 64 / 128.

- An eigh costs about 0.7 SVD, so a whitened SVD with eigh square roots is at least 2.4 SVD before L and R exist. A Cholesky variant would avoid that.
- Building L and R from transferred strings needs on the order of 10²-10³ products of size χ to 2χ for the 171 static strings of the 6-site window. That is a few to about 15 SVD-equivalents.
- That is the same range as spcf's measured 4-16x, not the planned 1-3x. Unmeasured.

## Deviations and honesty notes

- The first T1 run was killed (nothing saved), so T1 was rerun with checkpointing. N=16 was subsampled (every 2nd of 459), not 200 hand-picked.
- The T3-lite arm is a new, oracle-grade implementation, not the planned `kfac`/`core` modes. Those were not built: `kfac` died in T1, and `core` (T2) is gated on a T1 pass. `SPCWhite` as written has modes `all` and `static` only.
- No bench arm was added to `mfc_bench.py`; `whiten/spcw_bench.py` reuses `hp_bench`/`mpsenh` by import.
- CPU timings throughout were under load of 4-14 on 4 cores. Treat the T0 fractions as indicative only.
- `ising2` and Heisenberg were not scanned beyond the single ising2 χ=12 cell.
- The far-observable collateral (open question 1 of the plan) was not measured.

## If revived: the single next experiment

Build the target-free static-string construction of L and R (transfer matrices, no `Wof`, no region objects) and, on the captured T1 cuts at χ = 32 and 64, time it against `np.linalg.svd` (interleaved, single thread, medians).

- **Go only if:** the construction plus the whitened SVD costs ≤ 3x SVD and per-cut retention relative to spcf stays ≥ 0.5.
- **Stop if:** it costs more than about 5x SVD. Then spcf at 4-16x is already equivalent.
