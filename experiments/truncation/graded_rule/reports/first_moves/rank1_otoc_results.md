# Rank 1 first moves, results: OTOC and operator-weight fronts on a vectorized Heisenberg operator

Plan executed: `reports/first_moves/rank1_otoc.md`. All code and raw output are in `rank1/` (nothing outside `rank1/` was touched, nothing committed).

## Verdict

| test | pre-registered reading | result |
|---|---|---|
| 1. headroom and norm artifact | Continue / kill / ambiguous | **Continue**, with the "ambiguous" clause also triggered (the per-cut residual is 0.2-0.5% of the final marginal error, so expect a gain below 2x) |
| 2. static single-sweep ceiling | Continue / kill / ambiguous | **Ambiguous.** Best cell is t = 4, chi = 8: ratio 0.52 (1-site), 0.55 (2-site), infidelity x1.14, far x0.50. It misses the 0.5 bar by 0.02-0.05 and is 1.0 at t = 2, where SVD is already at 3e-5. Not a kill: the ratio is below 0.8 at chi = 4 and 8 |
| 3. dynamic head-to-head | success / kill / ambiguous | **Success criterion not met; kill criterion not strictly met either, so "ambiguous, leaning kill".** Best gain is 1.4-1.7x at the front, chi = 8 and 16 only, none at chi = 32, nothing behind the front (0.83-1.07). A one-line reweighting control (`rw:2.0`) matches or beats it behind the front at about 0.2x the cost of SVD, against 3.4-4.2x for the tilt |

Recommended next step: park Rank 1 as a tilt application. Keep the harness (`op_tebd.py`, dense reference, metrics) and the two cheap findings (renormalize the operator; a diagonal label reweighting is a strong free control). Details and the caveats on this recommendation are in the last section.

## What was built (`rank1/`)

| file | lines | what it is |
|---|---|---|
| `op_tebd.py` | 236 | Heisenberg-picture operator TEBD on a Pauli-fused MPS (d = 4, real float64). Pauli-basis gates from `pauli_prop.transfer`, reshaped to `G[q_b, q_{b+1}, p_b, p_{b+1}]` (new, old). A d-generic copy of `mpsenh.run_tebd`. Dense reference with the same gates and order. Arms: `svd` (renormalised), `svdraw` (see deviations), `rw:g` (whole MPS in the reweighted frame `diag(1, 1/g, 1/g, 1/g)` per site, gates `D2 R^T D2^-1`, back-transformed to the physical frame for scoring). Scorers: one-site marginals, k-site window marginals, pair marginals, `C_Z`, `C_X`, weight density. Models: repo `ising` (hx = 0.9045, hz = 0.8090), `tfim` (hz = 0), `heis` |
| `spcop.py` | 357 | `OpCut` (SVD cut plus optional per-cut diagnostics) and `SPCOp` (the tilt). Port of `spcfast.py`: SVD, Grassmann chart `Q = qr(U_k + U_perp C)`, matrix-free CG on `J^T J`, one retraction with line search over alpha in (1, 0.5, 0.25), accepted only if the residual drops. Diagonal targets: region tensor `W = Lm theta Rm` (4^6 x o*p for a = 2), `h = rowsum(W^2)/sum`, residual `r = Msp (h_trunc - h_theta)` with a fixed 0/1 summation map (360 rows for L = 6: windows k = 1, 2, 3, row weight `sqrt(w_k/n_windows)`). `dh = 2 rowsum(W0 * dW)`, adjoint `dW^T = 2 eta[:,None] W0`, so no 4^L x 4^L matrix exists. Options kept from spcfast: `fw`, `iters`, `rel_skip`, `passes`, benefit/cost `gate`. `tilt=False` runs the same code path and returns plain SVD |
| `op_bench.py` | 291 | arms x chi ladder, three-region metrics (ahead < 1e-2 <= front <= 0.5 < behind by the exact `C_Z`), contour lags, snapshot statistics, cached dense reference, per-cut CPU time |
| `op_static.py` | 116 | Test 2: exact MPS of the cached exact operator, one left-to-right sweep of cuts |
| `gate_fit.py` | 76 | refit of the benefit/cost gate constants on Ising |
| `test_op_a.py`, `test_op_b.py` | 124, 128 | correctness checks (outputs in `results/test_op_a.txt`, `results/test_op_b.txt`) |
| `an_test{1,2,3}.py`, `make_tables.py`, `plot_*.py`, `run_test3_*.sh` | | analysis, tables (`results/tables_test{1,2,3}.md`), figures |

Raw JSON for every run is in `rank1/results/`. Dense references are cached in `rank1/results/refcache/` (git-ignored, 145 MB). Figures (PNG) are in `rank1/figures/`.

## Correctness checks (all passed before any test was run)

| check | result |
|---|---|
| `h_bond`, gates equal `mpsenh` (ising, heis) | 0 difference |
| `R` orthogonal | 2.2e-16 |
| dense Pauli-space circuit equals explicit `U^dag Z_0 U` from kron matrices (N = 5, 3 steps; ising, heis) | 4.4e-16, 8.3e-16; norm 1 to 7e-16 and 8e-15 |
| `C_Z`, `C_X` from marginals equal the commutator norm `(1/2) Tr([W,O]^dag [W,O])/2^N` | 1.8e-15, 3.3e-15 |
| `pauli_prop.propagate` (eps = 0, Neel expectation) equals the matrix expectation | 1.7e-16, 5.4e-16 |
| MPS at chi = inf equals the dense reference (N = 8, 8 steps) | ising 3.0e-14, heis 3.5e-14, tfim 4.7e-14 |
| same in the rw:1.6 frame, back-transformed | 4.7e-13, 5.2e-13, 2.0e-12 |
| TFIM N = 10, T = 6: SVD at chi = 2 and 4 equals the dense reference | 7.6e-14 (bond dimension 2 is exact, as the Majorana argument predicts) |
| window marginals from the region tensor (all 360 rows plus the full 4^6 label distribution) equal dense marginals of the state with theta spliced in | 3.9e-15 |
| adjoint pair `<g, Jx> = <J^T g, x>` at all 75 recorded cuts (N = 8, chi = 4, 25 steps) | max rel err 1.4e-12; with a = 1, ks = (1,2), passes = 2 (39 cuts) 5.5e-14 |
| central finite-difference linearisation of the chart (step 1e-5) | 2.4e-8 (a = 1 variant 8.7e-9) |
| `Wadj` is the adjoint of `Wof` | 4e-16 |
| residual drops at every fired cut | 75/75; median `f_final/f_svd` 0.45, max 0.885 |
| `SPCOp(tilt=False)` and `OpCut(diag=True)` equal `svd` | bit for bit (0.0) |
| `SPCOp` at chi = inf equals dense; centre canonical form; unit norm | 4.2e-14; 2.0e-15; 1.1e-15 |

Two things in this table were not clean on the first attempt, and both were test or physics issues, not code bugs:

1. The first finite-difference test used a one-sided step and a tolerance of 1e-4 and failed (6e-4). The error scaled linearly with the step (1.3 at 1e-3, 1.3e-4 at 1e-7), i.e. it was the chart curvature, with a large coefficient at cuts where a random direction has a tiny Jacobian image. Switching to central differences gives 2.4e-8, so the Jacobian is right.
2. The "unnormalised cut equals svd times a scalar" check fails on Heisenberg (6.1e-4). It passes to 1e-15 on Ising and TFIM. SU(2) symmetry makes the singular values at the rank boundary exactly degenerate at 6 of 75 cuts (relative gap < 1e-10), so which vectors of the multiplet are kept depends on rounding. This is a property of the Heisenberg SVD cut, not of the harness.

## Test 1: headroom and the norm artifact (SVD arms, Ising and TFIM, N = 10, T = 6, dt = 0.1, O = Z_0)

Reference: dense, 4^10 amplitudes, 60 steps (6 s). Metrics per the plan; "behind/front/ahead" are set by the exact `C_Z(x,t)` at each (x,t), 600 points, of which 263 are behind, 115 front, 222 ahead. The exact `C_Z` reaches 1.64 (overshoot before relaxing toward 1).

| arm | chi | rms dC_Z ahead | front | **behind** | max behind | mean behind (signed) | rms dw behind | lag th=0.1 (max over sites) | **lag th=0.5** | final infid |
|---|---|---|---|---|---|---|---|---|---|---|
| svd | 4 | 5.0e-04 | 4.5e-02 | **1.98e-01** | 0.470 | -0.082 | 1.22e-01 | 0.228 | **1.532 (censored at 2 sites)** | 0.967 |
| svd | 8 | 2.5e-04 | 2.0e-02 | **6.70e-02** | 0.177 | -0.040 | 4.08e-02 | 0.116 | 0.210 | 0.945 |
| svd | 16 | 1.2e-04 | 9.5e-03 | **2.87e-02** | 0.110 | -0.015 | 1.48e-02 | 0.064 | 0.060 | 0.750 |
| svd | 32 | 2.9e-05 | 3.4e-03 | 7.4e-03 | 0.041 | -0.003 | 4.6e-03 | 0.024 | 0.013 | 0.362 |
| svdraw | 4 | 6.3e-04 | 8.2e-02 | 2.87e-01 | 0.596 | -0.217 | 2.16e-01 | 0.519 | 1.532 (cens. 2) | n/a |
| svdraw | 8 | 3.1e-04 | 4.6e-02 | 1.51e-01 | 0.345 | -0.113 | 1.12e-01 | 0.253 | 0.724 (cens. 1) | n/a |
| svdraw | 16 | 1.3e-04 | 2.0e-02 | 6.5e-02 | 0.189 | -0.040 | 4.4e-02 | 0.108 | 0.279 | n/a |
| svdraw | 32 | 3.1e-05 | 6.8e-03 | 1.8e-02 | 0.078 | -0.010 | 1.3e-02 | 0.033 | 0.038 | n/a |
| rw:1.6 | 4 | 7.3e-04 | 8.1e-02 | 1.97e-01 | 0.513 | -0.112 | 1.24e-01 | 0.519 | 1.532 (cens. 2) | 0.956 |
| rw:1.6 | 8 | 8.3e-05 | 8.2e-03 | 6.42e-02 | 0.207 | -0.032 | 3.68e-02 | 0.006 | 0.171 | 0.926 |
| rw:1.6 | 16 | 1.8e-05 | 7.4e-03 | 2.81e-02 | 0.086 | -0.015 | 1.41e-02 | 0.022 | 0.079 | 0.748 |
| rw:1.6 | 32 | 4.9e-05 | 1.5e-03 | 7.33e-03 | 0.036 | -0.004 | 4.75e-03 | 0.000 | 0.011 | 0.413 |

TFIM null control (hz = 0): the largest `|dC_Z|` over all (x,t) is 5.8e-14 for `svd` at every chi from 2 to 32 (final infidelity -4e-16), and 4.3e-12 for `rw:1.6` (round-off from the reweighted frame). The harness is validated against the 1e-6 requirement. Figure: `figures/test1_ising_CZ_maps_chi8.png` shows the exact, renormalised and unnormalised `C_Z(x,t)` maps with the 0.1 and 0.5 contours: the unnormalised operator reproduces the "halting" front of Hemery-Pollmann-Luitz at chi = 8, and the renormalised one still lags late.

Norm artifact (the plan's open question 1). `svdraw` is the same trajectory as `svd` times the scalar `prod(kept fraction)`; this was checked numerically (above) and holds exactly because rank truncation is scale-equivariant.

| chi | norm^2 of the unnormalised operator at T | svdraw / svd rms dC_Z behind | share of the unnormalised error removed by renormalising |
|---|---|---|---|
| 4 | 0.575 | 1.45 | 31% |
| 8 | 0.720 | 2.26 | 56% |
| 16 | 0.869 | 2.25 | 56% |
| 32 | 0.954 | 2.49 | 60% |

So the norm artifact is real but partial: it accounts for 31-60% of the rms error behind the front. A renormalised SVD operator still has rms 0.03-0.2 at chi = 4-16.

Per-cut residual against the final error (`svd` arm, per-cut span-2 rows of the SVD residual in probability units, rms over the cuts of the last 5 steps; final error is the rms one-site marginal error at T):

| chi | per-cut span-2 rms | final rms dp1(T) | ratio (pre-registered form) | L2 version | accumulated: sqrt(sum of per-cut L2^2) / final L2 |
|---|---|---|---|---|---|
| 4 | 2.9e-04 | 1.19e-01 | 0.0024 | 0.0036 | 0.08 |
| 8 | 1.6e-04 | 4.8e-02 | 0.0034 | 0.0050 | 0.10 |
| 16 | 7.1e-05 | 2.1e-02 | 0.0035 | 0.0056 | 0.09 |
| 32 | 4.1e-05 | 7.6e-03 | 0.0053 | 0.0079 | 0.10 |

(The "ratio" is my operationalisation of the plan's wording; the L2 and accumulated columns are sensitivity checks. Only a fully coherent sum of the per-cut L2 norms reaches 1.0-1.6 of the final error.)

**Reading against the pre-registered criteria**

| criterion | needed | observed | met |
|---|---|---|---|
| renormalised `svd` behind-front rms dC_Z >= 0.02 at some chi | 0.02 | 0.198, 0.067, 0.029 at chi = 4, 8, 16 | yes |
| or contour lag >= 0.3 at th = 0.5 | 0.3 | 1.532 at chi = 4 (censored: two sites never reach 0.5 within T) | yes |
| TFIM error <= 1e-6 | 1e-6 | 5.8e-14 | yes |
| kill: `svd` within 0.005 of exact everywhere | | 0.007-0.2 behind, 0.003-0.045 at the front, so no | not triggered |
| `rw:1.6` worse than `svd` behind the front (expected sign) | | 0.99, 0.96, 0.98, 1.00 of svd: **not worse, equal** (front: 1.79, 0.42, 0.77, 0.43) | the expected sign is not seen on Ising |
| ambiguous: per-cut residual under 10% of the error | | 0.2-0.5% | yes, triggered |

**Verdict: Continue**, with the ambiguity flag. The plan says to proceed to Test 2 and expect a gain below 2x in that case.

## Test 2: static ceiling at a single snapshot (Ising N = 10, exact |O(t)>>)

One left-to-right sweep from the exact MPS (bond dimensions up to 227 at t = 2), cuts with targets from the untruncated two-site tensor at each cut, `rel_skip = 0` (every truncating cut tilted), a = 2, ks = (1,2,3), fw = 0, 4 CG iterations. The registered times are t = 2 and 4; t = 3, 5, 6 are supplementary (flagged). Entries are the ratio spcop / svd at equal chi. "Far" is the rms error of the pair marginals at distance >= 3 (not targeted); "ZZ-OTOC" is the two-site OTOC with W = Z_x Z_{x+3}. C_Z columns are noisy because only 2-8 sites fall in a region.

| t | chi | svd 1-site rms | 1-site | 2-site | 3-site | far d>=3 | ZZ-OTOC d=3 | infid | C_Z behind (n sites) | C_Z front (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| **2** | 4 | 3.2e-03 | 0.72 | 0.76 | 0.89 | 0.69 | 1.16 | 1.16 | 0.94 (2) | 0.85 (3) |
| **2** | 8 | 3.0e-05 | 1.02 | 0.66 | 0.72 | 1.08 | 1.03 | 1.05 | 2.47 (2) | 1.01 (3) |
| **2** | 16 | 4.7e-09 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | no cut fires (SVD exact to 2e-8) | |
| **2** | 32 | 4.6e-12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | no cut fires | |
| **4** | 4 | 4.8e-02 | 0.52 | 0.49 | 0.54 | 0.47 | 0.62 | 1.20 | 0.55 (6) | 0.74 (2) |
| **4** | 8 | 1.7e-02 | **0.52** | **0.55** | 0.63 | 0.50 | 0.87 | 1.14 | 0.41 (6) | 0.76 (2) |
| **4** | 16 | 3.4e-03 | 0.93 | 0.91 | 0.92 | 0.91 | 0.95 | 1.03 | 0.86 (6) | 0.98 (2) |
| **4** | 32 | 1.7e-04 | 0.98 | 0.95 | 0.89 | 0.97 | 0.99 | 1.01 | 0.99 (6) | 1.01 (2) |
| 3 (suppl.) | 4 / 8 / 16 / 32 | | 0.65 / 0.86 / 0.99 / 1.14 | 0.68 / 0.84 / 0.80 / 0.41 | | 0.64 / 0.89 / 0.96 / 1.03 | | 1.18 / 1.11 / 1.01 / 1.00 | | |
| 5 (suppl.) | 4 / 8 / 16 / 32 | | 0.30 / 0.42 / 0.61 / 0.94 | 0.29 / 0.43 / 0.64 / 0.95 | | 0.30 / 0.40 / 0.55 / 0.90 | | 1.21 / 1.29 / 1.18 / 1.02 | | |
| 6 (suppl.) | 4 / 8 / 16 / 32 | | 0.34 / 0.37 / 0.60 / 0.81 | 0.29 / 0.33 / 0.60 / 0.80 | | 0.31 / 0.35 / 0.53 / 0.76 | | 1.18 / 1.24 / 1.27 / 1.12 | | |

Other arms at t = 4 (1-site / 2-site / infid / far, ratio to svd):

| chi | `spcop_k1` (one-site targets only) | `spcop_k12` | `spcop_p3` (3 Gauss-Newton passes, optional upper bound) |
|---|---|---|---|
| 4 | 0.43 / 0.46 / 1.16 / 0.40 | 0.51 / 0.47 / 1.19 / 0.46 | 0.34 / 0.37 / 1.26 / 0.28 |
| 8 | 0.44 / 0.63 / 1.16 / 0.49 | 0.50 / 0.54 / 1.15 / 0.51 | 0.42 / 0.44 / 1.27 / 0.40 |
| 16 | 0.72 / 0.85 / 1.26 / 0.81 | 0.91 / 0.90 / 1.03 / 0.89 | 0.77 / 0.76 / 1.16 / 0.78 |

Figure: `figures/test2_static_ratio.png`.

Observations. (i) The gain grows with SVD pressure: ratio 0.3-0.4 at t = 5-6 with chi = 4-8 (SVD error 0.05-0.2), 0.5 at t = 4, about 1 where SVD is already accurate. (ii) The far, untargeted marginals improve together with the targeted ones (0.40-0.50 at t = 4, chi = 4-8), so there is no collateral damage of the kind seen in the Heisenberg state case. (iii) The infidelity costs 1.14-1.29x at the cells where the tilt helps; the "<= 1.2" bar is broken at t = 5 (chi = 4, 8: 1.21, 1.29) and t = 6 (chi = 8, 16: 1.24, 1.27) and is on the line at t = 4, chi = 4 (1.20); at the registered best cell t = 4, chi = 8 it is 1.14. (iv) The one-site-only arm (open question 2) does as well as the full set on the one-site marginal (0.43-0.44 at t = 4, chi = 4-8), unlike in the state case; its 2-site ratio is 0.46-0.63.

**Reading against the pre-registered criteria**

| criterion | needed | observed at the registered cells | met |
|---|---|---|---|
| Continue: 1-site and 2-site ratio <= 0.5 at chi = 8 or 16; infid <= 1.2; far <= 1.5 | 0.5 | t = 4, chi = 8: 0.52 / 0.55 (infid 1.14, far 0.50); chi = 16: 0.93 / 0.91; t = 2, chi = 8: 1.02 / 0.66; chi = 16: nothing to fix | no (misses by 0.02-0.05 at one cell) |
| Kill: ratio >= 0.8 at all chi | 0.8 | below 0.8 at t = 4 for chi = 4, 8 and at t = 2 for chi = 4 | not triggered |
| Ambiguous: 0.5-0.8 | | best cells 0.52 / 0.55 (t = 4, chi = 8) | yes |

**Verdict: Ambiguous.** The plan's branch is "run Test 3 only at the best chi". Among the Test 3 ladder {8, 16, 32} that is chi = 8 (chi = 4, outside the Test 3 ladder, is nominally lower, 0.49-0.52 at t = 4 and 0.76 at t = 2).

## Test 3: dynamic head-to-head (Ising, N = 10, T = 6, dt = 0.1)

Order of execution: first the registered chi = 8 only, then the full ladder chi = 4, 16, 32 plus the controls. The extension goes beyond the ambiguous branch of the plan (one chi) because it cost minutes; the verdict below is stated both for the registered cell and for the extension. Arms: `svd`; `spcop` (a = 2, ks = 1-2-3, fw = 0, 4 CG iterations, `rel_skip = 1e-2` as in the state production setting); `spcop` gated; `rw:1.6`.

Ratios to `svd` at equal chi (rms of |dC_Z| and |dw| by region; < 1 is better than SVD):

| chi | arm | C_Z front | C_Z behind | w front | w behind | lag th=0.5 (svd) | infid | far d>=3 | c_Z0(t) rms | cut CPU vs svd |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | svd, absolute | 1.97e-02 | 6.70e-02 | 1.54e-02 | 4.08e-02 | 0.210 | 0.945 | 2.35e-02 | 1.85e-02 | 1.0 |
| **8** | **spcop** | **0.62** | **1.07** | **0.60** | **0.88** | 0.146 | 1.00 | 0.94 | 1.27 | 3.7 |
| 8 | spcop, gate = 1 (constants refit on Ising) | 0.62 | 1.07 | 0.60 | 0.88 | 0.146 | 1.00 | 0.94 | 1.27 | 3.9 |
| 8 | rw:1.6 | 0.42 | 0.96 | 0.63 | 0.90 | 0.171 | 0.98 | 0.88 | 0.50 | 0.2 |
| 16 | svd, absolute | 9.52e-03 | 2.87e-02 | 7.02e-03 | 1.48e-02 | 0.060 | 0.750 | 9.35e-03 | 2.59e-03 | 1.0 |
| **16** | **spcop** | **0.72** | **0.85** | **0.69** | **0.77** | 0.031 | 1.01 | 0.79 | 1.46 | 3.4 |
| 16 | spcop, gate = 1 | 0.72 | 0.85 | 0.69 | 0.77 | 0.031 | 1.01 | 0.79 | 1.46 | 4.5 |
| 16 | rw:1.6 | 0.77 | 0.98 | 0.90 | 0.95 | 0.079 | 1.00 | 0.88 | 0.51 | 0.2 |
| 32 | svd, absolute | 3.38e-03 | 7.36e-03 | 2.44e-03 | 4.56e-03 | 0.013 | 0.362 | 3.46e-03 | 6.10e-04 | 1.0 |
| **32** | **spcop** | **1.05** | **0.83** | **1.02** | **0.76** | 0.013 | 0.99 | 0.84 | 0.44 | 3.7 |
| 32 | spcop, gate = 1 | 1.05 | 0.83 | 1.02 | 0.76 | 0.013 | 0.99 | 0.84 | 0.44 | 4.2 |
| 32 | rw:1.6 | 0.43 | 1.00 | 0.53 | 1.04 | 0.011 | 1.14 | 0.90 | 0.65 | 0.4 |
| 4 (suppl.) | spcop | 0.72 | 0.93 | 0.73 | 0.88 | 1.532 (unchanged, censored) | 1.01 | 0.86 | 0.67 | 4.2 |

Cut CPU is `process_time` inside the cut only (the machine was at load average 12-22 on 4 cores for most of the session, so wall times are not usable; the 3.4-4.2x is a CPU ratio). Per fired cut the tilt reduces the sum-of-squares window residual to a median 0.62, 0.51, 0.35, 0.26 of SVD's at chi = 4, 8, 16, 32 and raises the discarded weight by a median factor 1.06, 1.06, 1.02, 1.02; 81-84% of truncating cuts fire (the rest are skipped by `rel_skip`). Figure: `figures/test3_ising_ratio_vs_chi.png`.

The gate. Spcfast's gate formula (fire only if the span-2 residual of the SVD cut exceeds `gate * c0 * (tail/1e-4)^exp`) was refit in-sample on Ising from the measured collateral change of the untargeted marginals (windows outside the fit region) between the tilted and plain cut, at 773 fired cuts (`gate_fit.py`): `c0 = 1.29e-6`, `exp = 0.92`, log scatter 4.8x. The benefit exceeds the collateral at 99.4% of cuts (median ratio 10), so at gate = 1 the gated arm skips nothing beyond the `rel_skip` skips and is identical to `spcop` at chi = 8, 16 and 32. Gate = 10 reduces the number of fired cuts from 365 / 227 / 181 to 282 / 144 / 137 (chi = 8 / 16 / 32) and gives the gain away (front 0.80 at chi = 8, 0.85 at chi = 16). Gate = 3 is identical at chi = 8 and 16 and fires 6 fewer cuts at chi = 32. There is nothing to gate on Ising, as expected: the gate exists to stop harm, and the harm is not there.

**Reading against the pre-registered criteria (spcop, ungated)**

| criterion | needed | observed | met |
|---|---|---|---|
| front or behind rms dC_Z **and** dw >= 2x lower than svd at >= 2 of 3 chi | ratio <= 0.5 at 2 of {8, 16, 32} | front 0.62/0.60, 0.72/0.69, 1.05/1.02; behind 1.07/0.88, 0.85/0.77, 0.83/0.76: **0 of 3** (best single ratio 0.60) | **no** |
| lag at th = 0.5 reduced by >= 30% | 30% | 31% (chi = 8), 48% (chi = 16), 3% (chi = 32); absolute lags 0.03-0.2 time units, one or two steps | yes at chi = 8, 16 |
| infidelity <= 1.2x | 1.2 | 1.00, 1.01, 0.99 (but svd infidelity is 0.75-0.97 at chi <= 16, i.e. saturated, so this criterion cannot fail there) | yes, weak test |
| non-targeted far marginals <= 1.5x | 1.5 | 0.94, 0.79, 0.84 (better than SVD) | yes |
| linear autocorrelation no worse than 1.5x | 1.5 | 1.27, 1.46, 0.44 | yes (1.46 at chi = 16 is borderline) |
| kill: ratio > 0.75 at all chi on Ising | 0.75 | front 0.62-1.05, behind 0.83-1.07. Taking the better region at each chi: 0.60, 0.69, 0.76 for (8, 16, 32), so not all above 0.75; taking behind-front only, all above 0.75 | **ambiguous reading: not triggered by the front, triggered by the behind region** |
| kill: gain only ahead of the front | | the gain is in the front region; ahead/front/behind ratios at chi = 8: 0.70/0.62/1.07 | not triggered |
| ambiguous: 1.3-2x gain with growing collateral | | 1.4-1.7x at the front for chi = 8, 16 with collateral flat or improving (far 0.8-0.9), linear autocorrelation up to 1.46 | yes |

**Verdict for Test 3: ambiguous, leaning kill.** The success bar is missed on every chi (0 of 3). The kill bar is not cleanly met because the front region shows a 1.4-1.7x gain at the two lower chi. The registered chi = 8 cell alone reads 1.6x at the front, 1.0x behind.

The "next step" the plan prescribes for the ambiguous branch is an `fw` damping scan and a diagonal whitened-SVD control. Both were run in a light form (supplementary: chosen after seeing the first result, not part of the verdict):

| chi | arm | C_Z front | C_Z behind | w front | w behind | far d>=3 | c_Z0(t) rms | cut CPU vs svd |
|---|---|---|---|---|---|---|---|---|
| 8 | spcop (reference) | 0.62 | 1.07 | 0.60 | 0.88 | 0.94 | 1.27 | 3.7 |
| 8 | fw = 0.03 / 0.1 | 0.79 / 0.92 | 0.96 / 0.97 | 0.76 / 0.92 | 0.89 / 0.94 | 0.90 / 0.94 | 1.08 / 1.03 | 4.0 / 4.2 |
| 8 | ks = (1,2) | **0.49** | 0.94 | **0.48** | 0.84 | 0.82 | 0.83 | 3.8 |
| 8 | ks = (1,) | 0.59 | 0.89 | 0.59 | 0.87 | 0.87 | 1.02 | 3.5 |
| 8 | 3 passes | 0.61 | 1.16 | 0.58 | 0.94 | 0.97 | 1.03 | 9.8 |
| 8 | rw:1.2 / 1.6 / 2.0 / 3.0 | 1.18 / 0.42 / 1.45 / 6.06 | 1.07 / 0.96 / **0.80** / 1.33 | 1.26 / 0.63 / 1.09 / 5.15 | 1.07 / 0.90 / **0.66** / 1.03 | 1.02 / 0.88 / 0.68 / 1.33 | 0.87 / 0.50 / 0.51 / 0.97 | 0.2-0.3 |
| 16 | spcop (reference) | 0.72 | 0.85 | 0.69 | 0.77 | 0.79 | 1.46 | 3.4 |
| 16 | fw = 0.03 / 0.1 | 0.89 / 0.99 | 0.90 / 0.99 | 0.86 / 0.99 | 0.85 / 0.99 | 0.86 / 0.99 | 1.13 / 0.96 | 3.3 / 3.4 |
| 16 | ks = (1,2) / (1,) | 0.67 / 0.80 | 0.84 / 0.89 | 0.63 / 0.77 | 0.77 / 0.85 | 0.80 / 0.85 | 1.38 / 1.01 | 2.9 / 2.8 |
| 16 | 3 passes | 0.73 | 0.85 | 0.70 | 0.76 | 0.78 | 1.36 | 7.3 |
| 16 | rw:1.2 / 1.6 / 2.0 / 3.0 | 1.17 / 0.77 / 0.85 / 7.72 | 1.12 / 0.98 / **0.78** / 1.75 | 1.25 / 0.90 / 0.79 / 7.59 | 1.12 / 0.95 / **0.58** / 1.94 | 1.08 / 0.88 / 0.56 / 2.37 | 0.61 / 0.51 / 0.14 / 2.88 | 0.2 |
| 32 | spcop (reference) | 1.05 | 0.83 | 1.02 | 0.76 | 0.84 | 0.44 | 3.7 |
| 32 | ks = (1,2) / (1,) | 0.98 / 0.91 | 0.84 / 0.88 | 0.95 / 0.89 | 0.79 / 0.88 | 0.86 / 0.90 | 0.78 / 0.92 | 3.1 / 3.5 |
| 32 | rw:1.2 / 1.6 / 2.0 / 3.0 | 1.00 / 0.43 / 1.92 / 8.56 | 1.08 / 1.00 / **0.77** / 2.15 | 1.06 / 0.53 / 1.68 / 8.27 | 1.09 / 1.04 / **0.76** / 1.79 | 0.99 / 0.90 / 0.81 / 2.83 | 0.96 / 0.65 / 0.11 / 3.60 | 0.4 |

Findings from the controls:
- fw damping only removes the gain, monotonically (fw = 0.1 leaves 28 of 274 cuts firing at chi = 16). This matches the state case, where fw = 0 was best.
- `ks = (1,2)` reaches the 2x level at the front at chi = 8 (0.49 / 0.48), but 0.67 / 0.63 at chi = 16 and 0.98 / 0.95 at chi = 32; chosen after the fact and still 1 of 3, so it does not change the verdict.
- Three Gauss-Newton passes cost 2-2.6x more CPU and gain nothing in dynamics.
- The reweighting control is erratic at the front. `rw:1.6` is 0.42 and 0.43 at chi = 8 and 32 but `rw:1.2` and `rw:2.0` are 1.2-1.9 there, so the front error is a small signed quantity and rw:1.6's number looks like a partial cancellation, not a mechanism. Behind the front `rw:2.0` is consistently 0.77-0.80 (C_Z) and 0.58-0.76 (w) at all three chi, better than the tilt there (0.83-1.07 and 0.76-0.88), and lowers the untargeted far marginals (0.56-0.81) and the one-site / two-site / three-site window errors (0.78 / 0.63 / 0.48 at chi = 8), at 0.2-0.4x the CPU of SVD. `rw:3.0` is 1.3-2.2x worse behind the front and 6-9x worse at the front. `rw:2.0` also never lags the exact contour (max lag 0.000) and cuts the error of the linear autocorrelation c_Z0(t) to 0.51, 0.14, 0.11 of SVD's, which is the opposite of the weakening of high-weight content that a weight-biased truncation is supposed to cause. The caveat is that the rw numbers are sensitive to gamma (1.2 does nothing, 2.0 helps, 3.0 hurts) and, on Heisenberg, rw:1.6 hurts. In short, a free diagonal reweighting is a competitive baseline that the tilt has to beat before any claim.

Informational runs.
- Heisenberg (N = 10, T = 6, `heis`; exact `C_Z` max 0.997; no failure was pre-registered). spcop / svd, C_Z front / behind: chi = 8: 0.96 / 0.99, chi = 16: 0.81 / 0.72, chi = 32: 0.76 / 0.77. The far marginals change by 1.09, 0.52, 0.69 (chi = 8, 16, 32), the linear autocorrelation is 1.24-1.57x worse (above the 1.5 bar at chi = 16, 32), infidelity 1.00-1.05x, cut CPU 3.9-4.9x. `rw:1.6` is *worse* than svd behind the front here (1.22, 1.07, 1.50), which is the sign the plan expected for a low-weight-biased cut but did not occur on Ising. So the Heisenberg OTOC is not a clean tilt failure either, in line with 2503.20327: the tilt helps by about 1.3x at chi >= 16.
- TFIM: no cut truncates at chi >= 2, so `spcop` equals `svd` bit for bit (max difference 0.0 in the one-site marginals). The `rw:1.6` entries in the TFIM table of `tables_test3.md` read 40-100x svd only because both errors are round-off (1e-12 against 1e-14).

## Deviations from the plan, and why

1. **`svdraw` is not a separate arm.** Rank truncation is homogeneous, so the unnormalised trajectory is the renormalised one times a scalar. I log `sum log(kept/total)` in the `svd` run and rescale the marginals by it. Checked against a genuinely unnormalised cut (`RawSVDCut`) to 1e-15 on Ising and TFIM; the Heisenberg check is spoiled only by exact degeneracies (see above). Consequence: the HPL norm artifact needs no extra run.
2. **Operationalisation choices the plan left open.**
   - Contour lag is the first linear-interpolated crossing, taken per site and reported as the maximum over the sites that the exact operator reaches. A site that the arm never reaches within T gets a lower bound (T minus the exact crossing time) and is counted as censored (it matters only at chi = 4).
   - "Per-cut span-2 residual / final marginal error" is the rms over the span-2 rows of the SVD point residual (probability units) over the cuts of the last 5 steps, divided by the rms one-site marginal error at T. This definition was fixed before looking at the other two, which are reported as sensitivity checks.
   - Targets are uniform per row with weight `sqrt(w_k/n_windows)`, `w_k = 1`; no region weighting (open question 3 untested).
   - Test 2 "ratio on 1- and 2-site marginals" uses all consecutive windows of the chain, not only those near a cut.
   - Test 3 success is evaluated per region (front, behind) with both `C_Z` and `w`; I did not allow mixing regions or metrics across chi.
3. **Test 3 run beyond the registered branch.** The plan's ambiguous branch is "only at the best chi". chi = 8 was run first and alone; the other chi were added afterwards because they cost minutes. The success criterion needs 2 of 3 chi and cannot be evaluated from one, so the extension is what makes it evaluable. Nothing was moved: the registered chi = 8 cell is shown on its own.
4. **Gate refit.** The state-case collateral measure was not available for operators, so I measured it on the dense state (rms change of the untargeted window marginals between tilted and plain cut). The fit is in-sample on the same runs it is applied to, as the plan says ("refit on Ising"). It is inert, so a bad fit would not have changed anything.
5. **Static sweep settings.** `rel_skip = 0` in Test 2 (every truncating cut tilted), `rel_skip = 1e-2` in Test 3 (spcfast production default). In Test 3, `rel_skip = 0` at chi = 8 changes nothing material (front 0.59 vs 0.62).
6. **Supplementary things not in the plan:** Test 2 at t = 3, 5, 6; Test 3 at chi = 4; the fw, ks, passes and rw-gamma scans; the Heisenberg and TFIM runs. They are labelled everywhere and none feeds a verdict.
7. **Size and operator.** N = 10 only, O = Z_0 only, one T = 6, one reference per model. No N = 12.
8. **Machine load.** The box was at load average 12-22 on 4 cores. All wall times are inflated and noisy; cost is quoted as CPU time inside the cut.
9. **Infidelity criterion is weak here.** The final operator infidelity at T = 6 is 0.75-0.97 for chi <= 16 (0.36 at chi = 32), so a ratio near 1.0 is forced by saturation at the lower chi.

## Things that could change the reading (not done)

- A second operator site, a second T, N = 12-14 (higher pressure at the same chi).
- The competitors the plan lists: time splitting (Xu-Swingle, the known accuracy fix for late-time OTOCs), actual rTEBD code, DAOE, and matched-memory comparisons. Only a diagonal-reweighting stand-in was run.
- Weighting targets by region (open question 3), and a targets-from-the-whitened-frame hybrid (reweight, then tilt).
- XXZ with the correct integrable control, and the HPL variant of the model with swapped fields.

## Recommended next step

1. **Park the tilt for OTOCs.** It does not meet the 2x bar (0 of 3 chi), the gain is confined to a thin region (the front) at two chi values, it gives 1.0-1.3x where SVD's error actually sits (behind the front), the final-error budget is dominated by what the tilt does not target (per-cut residual 0.2-0.5% of the final error), and a one-line reweighting does as well behind the front for a fifth of the cost. The static ceiling of Test 2 (0.5 at best at the registered times) already said this: a single-cut tilt cannot buy 2x here at moderate pressure.
2. **Keep two cheap, unrelated findings.** (a) Renormalise the operator MPO: the unnormalised truncation inflates the rms OTOC error by 1.45-2.5x, and the HPL-style "halting" front is reproduced by the unnormalised operator at chi = 8 while the renormalised one is much closer. (b) A plain diagonal label reweighting (`rw:2.0`) is a free 1.3-1.7x behind the front on Ising at chi = 8-32, but it is model-dependent (Heisenberg: worse by 1.1-1.5x; gamma = 3 is 2x worse on Ising), which is rTEBD applied to operators and not new.
3. **If the direction is revisited, the decisive extra experiment is cheap:** the same Test 3 on N = 14 at T = 8-9 with chi = 8-16 (SVD pressure comparable to the strongest static cells), against `rw:2.0` and time splitting at matched memory. The static test says the gain rises with SVD error (0.3-0.4 at t = 5-6), but the dynamic gain at chi = 4 (SVD error 0.2) was only 0.72-0.93, so the expectation should be low.
4. The harness (`op_tebd.py`, reference cache, metrics, the d = 4 diagonal-target `spcop.py`) is reusable for Rank 2 (purification) and any other d = 4 object.
