# Rank 2 first move: results, spcf on purification MPS

Plan executed: `reports/first_moves/rank2_purification.md` (items 1-2 of "Where to start", then the decision tree 0 -> 1 -> 2; item 3, the numpy DMT, was built because Test 1 succeeded).
All code, raw output and figures are in `rank2/` (nothing outside that folder was touched, nothing committed by me). Everything below is Ising (`mpsenh` "ising" model), N = 10 unless stated, dt = 0.1 Strang, one dense reference per cell.

## Verdict

| test | pre-registered rule | outcome | verdict |
|---|---|---|---|
| 0(a) adjoint / linearisation / W-form | adjoint < 1e-5, linearisation error linear in step | float64: adjoint 1e-13, W-form 2e-15 vs dense purification, linearisation linear. float32 (the production setting): adjoint 2e-5 to 3.5e-4, same order as the d = 2 code (5.9e-5) | pass (adjoint threshold met only in float64, see deviations) |
| 0(b) anchor to d = 2 spcf | SVD arms to 1e-8, spcf within about 10 percent | SVD: max difference 3.6e-14. spcf: identical at chi = 8, 16; within 5 percent on rdm2/nn/nnn/single at chi = 6, 12; \|dE\| differs 3.6x at chi = 12 (tiny absolute value) | pass |
| 1, step 1 (SVD scan) | kill if no usable cell at mu <= 1 | usable cells in 12/12 (family, T, gauge) combinations at each of mu = 1, 0.5, 0.25 | not killed |
| 1, step 2 (spcf vs SVD, mu dial) | success: some mu <= 1 with median SVD(better gauge)/spcf >= 2 on rdm2 and nn, far ZZ <= 1.5x, trace distance <= +10 percent | **success**. 15 of 18 (family, mu <= 1, T) cells pass in at least one gauge; best median ratios 4.67 (rdm2) / 4.29 (nn) at staggered mu = 0.5, T = 4. The 3 failing cells are all mu = 0.25 and fail on collateral, not on the ratio | **success** |
| 1 control: heis at mu = 1, back gauge | expected no gain | **control violated**: like-for-like gain 2.4 to 2.9x in the back gauge, and 2 cells pass against the better-gauge SVD. Metric verified independently. Pure-state d = 2 heis at the same N, T gives at most 1.6x | not as expected, see caveat |
| 2 head-to-head with DMT | success: spcf <= DMT/1.5 on rdm2 and nnn at >= 3 of 4 chi; kill: DMT <= spcf on rdm2 at every chi | primary cell (staggered, mu = 0.5, T = 4): spcf-purification beats DMT by 2.9 to 7.8x on rdm2 and 2.5 to 4.4x on nnn at 4 of 4 chi. 7 of 7 cells at mu = 0.25 to 1 pass (6 at 4/4, one at 3/4) | **success** |

Two outcomes that qualify the success, stated up front:
1. spcf gains over SVD on **heis in the purification setting**, which the plan said it should not. The pure-state heis does not show this, so part of the gain may be SVD's gauge inefficiency rather than the local-observable objective. A gauge-fair baseline (Hauschild disentangler) is the first thing to add.
2. The lead over DMT on the 3-site-range correlator **shrinks toward the infinite-temperature limit**: at mu = 0.1 (extension beyond the plan's dial) DMT/spcf on nnn is 1.1 to 2.2 and the 1.5x rule passes at only 1 or 2 of 4 chi. DMT also has the lower energy drift in 35 of 36 (cell, chi) entries, typically by 6 to 250x (median 71x), its home ground as the plan expected.

## What was built

| file | role |
|---|---|
| `rank2/spcfpur.py` | `SPCFPur(SPCFast)`: spcf for a fused (physical, ancilla) site, d = 4. Inherits `_region`, `_strmeta`, the F matrices and the gate from the d = 2 class unchanged. Overrides `_maps` (splits the environment maps into physical and ancilla factors and moves the ancilla factor to the traced column side), the cut (`Wof`, `vjp`, `retract`, returns all d-generic) and `_plain`. `beta` is not supported |
| `rank2/purlib.py` | tilted-Bell-pair initial states (`stag`, `dw`, mu including inf), gates `plain` (G x 1) and `back` (G x G*), d-generic `svd_cut_d` / `run_tebd_d` / `mps_to_dense_d`, dense purification reference with snapshots and cache, physical rho and all metrics |
| `rank2/pur_bench.py` | driver: arms `svd`, `spcf[:opts]`, `spcfg:<gate>[:opts]`, both gauges, JSON rows, resumable job lists, multiprocessing |
| `rank2/test_spcfpur.py`, `test0b_anchor.py` | Test 0(a), 0(b) |
| `rank2/scan_table.py`, `plan_step2.py`, `analyze_test1.py`, `mu_dial.py`, `make_tables.py` | Test 1 scan, job planning, pre-registered criteria, tables, figure |
| `rank2/dmt_np.py`, `test_dmt.py` | numpy DMT (Pauli MPDO, radius 1) and Frobenius MPDO; unit checks |
| `rank2/test2_dmt.py`, `analyze_test2.py`, `test2_summary.py`, `fig_test2.py` | Test 2 |
| `rank2/n12_check.py`, `n12_dmt.py`, `heis_pure_anchor.py`, `check_metric_independent.py` | N = 12 extension, heis controls |
| `rank2/results/` | raw JSON/log/txt of every run; `results/tables.md` and `tables_condensed.md` hold the full Test 1 tables |
| `rank2/figures/` | `mu_dial.png`, `test2_stag_mu0.5_T4.png` |

Metrics, all on the **physical** density matrix (ancillas traced), from the dense purification reference evolved with the same fused gates: rms of single, nn, nnn Pauli expectations; `rdm2`, `rdm3` (rms Frobenius error of all 2-site and 3-site marginals, as `hp_bench.marginal_errors`); `|dE|`; far collateral `zzfar` (rms of the error of <Z_i Z_{i+3}> and <Z_i Z_{i+4}>); trace distance 0.5 ||rho - rho_ex||_1; lambda_min; purification infidelity (gauge dependent, diagnostic only).

## Test 0: correctness and anchor

**0(a), `test_spcfpur.py`** (N = 8, chi = 4, T = 2.5, mu = 1 staggered, 350 cuts, about 200 fired):

| check | plain f32 | plain f64 | back f32 | back f64 |
|---|---|---|---|---|
| W W^dag vs RDM of the dense purification, max rel err | 2.0e-15 | 2.6e-15 | 1.5e-15 | 1.8e-15 |
| adjoint pair, max rel err | 3.5e-4 | 2.3e-14 | 2.4e-5 | 1.1e-13 |
| linearisation rel err, median at step 0.1 / 0.01 / 0.001 / 1e-4 | 0.081 / 0.0085 / 8.6e-4 / 9.7e-5 | 0.082 / 0.0086 / 9.0e-4 / 8.5e-5 | 0.090 / 0.0089 / 8.1e-4 / 1.1e-4 | 0.090 / 0.0089 / 8.1e-4 / 8.2e-5 |
| fired cuts, median f_c / f_svd (max) | 0.43 (0.70) | 0.43 (0.70) | 0.36 (0.71) | 0.36 (0.71) |

The existing d = 2 test (`test_spcfast.py`) gives adjoint 5.9e-5 (float32) and median linearisation error 0.128 / 0.0124 / 1.3e-3 / 1.2e-4 at the same steps. The F-form check against the direct window objective is unchanged (0.0434180 both ways).

**0(b), `test0b_anchor.py`.** Staggered mu = inf, plain gauge, against the existing d = 2 code, T = 3. Entries are d4/d2 ratios of the error:

| chi | arm | max abs diff | rdm2 | rdm3 | single | nn | nnn | E | infid |
|---|---|---|---|---|---|---|---|---|---|
| 6 | svd | 1.1e-14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | spcf | 6.9e-4 | 0.967 | 0.985 | 0.952 | 0.975 | 1.041 | 0.971 | 1.005 |
| 8 | svd | 3.6e-15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | spcf | 2.1e-9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | svd | 3.6e-14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | spcf | 1.0e-4 | 0.966 | 0.999 | 1.024 | 0.954 | 1.013 | 3.579 | 1.006 |
| 16 | svd | 4.0e-15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 | spcf | 3.0e-12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The same cuts fire in both codes (145/145, 56/56, 33/33, 1/1). The 3 to 5 percent differences at chi = 6 and 12 are float32 rounding amplified over hundreds of cuts. The E value at chi = 12 differs by 3.6x but both are small (4.0e-5 vs 1.4e-4, against SVD's 2.9e-4); energy is not in the objective. This test also validates the new metric code, because the physical rho of the mu = inf purification equals the d = 2 pure state.

Other checks, all passing: dense purification at mu = inf plain equals the cached d = 2 reference (overlap equal to 1 within 3e-15); physical rho identical in the two gauges (6e-17); MPS at chi = 1024 reproduces the reference (rdm2 1e-14); for heis, an independently evolved dense rho (G rho G^dag, no ancilla) equals the purification-derived reference (3e-16), and 2-site marginals from transfer matrices on a spcf purification MPS equal the dense route (5e-16) and give the same rdm2 as `pur_metrics` (3.092425e-02, both).

No bug was found in `spcfpur.py` or the purification driver, so no fix cycle was needed.

## Test 1: SVD pressure scan, then spcf vs SVD

### Step 1: the scan

Grid actually run: families staggered and domain wall; mu in {inf, 2, 1, 0.5, 0.25}; both gauges; T in {2, 3, 4}; SVD at chi in {4, 6, 8, 12, 16, 24, 32, 48}. 480 SVD runs (`results/scan_svd_*.json`, table by `python scan_table.py`).

Usable cell = at least 3 ladder values with SVD rdm2 in [1e-3, 1e-1] (SVD in that gauge); the front does not reach its floor inside the ladder for T >= 3 at mu <= 1 (plain gauge rdm2 3e-6 to 6e-4 at chi = 48, back gauge 2e-4 to 3e-2; the floor is 1e-14).

| mu | usable (family x T x gauge, of 12) | combinations with SVD rdm2 < 1e-4 at some chi >= 8 |
|---|---|---|
| inf | 9 | 8 |
| 2 | 11 | 8 |
| 1 | 12 | 6 |
| 0.5 | 12 | 6 |
| 0.25 | 12 | 6 |

So **kill criterion 1 (no usable cell, SVD < 1e-4 for every chi >= 8) does not apply at any mu <= 1**. SVD rdm2 at chi = 16, plain / back, staggered T = 3: mu = inf 7.0e-5 / 7.5e-2; mu = 1 6.5e-3 / 4.9e-2; mu = 0.5 8.2e-3 / 2.0e-2; mu = 0.25 6.2e-3 / 4.9e-3. The plain gauge is the better SVD gauge for most cells; the back gauge becomes competitive only at mu = 0.25, T >= 3 (the initial state is not thermal for H, so KBM's trivial evolution holds only as mu -> 0).

### Step 2: spcf and gated spcf

Defaults of `pareto_spcf.py`: a = 2, fw = 0, 4 CG iterations, tau = 1, ks = 1-2-3. Per cell, up to 4 chi values from the usable window, chi <= 32. Cells: mu in {1, 0.5, 0.25} at T = 2, 3, 4 (ungated, both gauges), T = 3, 4 also gated (gate 1.0, constants not recalibrated); controls mu in {2, inf} at T = 3, 4 (ungated and gated). 336 spcf runs (plus 24 for the heis control).

Rule applied (literal reading): per metric, comparator SVD = min over the two gauges at the same chi; ratio R = SVD_best / spcf; median over the chi values run inside the usable window of the spcf gauge. Pass if R(rdm2) >= 2, R(nn) >= 2, far-ZZ ratio <= 1.5, trace-distance ratio <= 1.10 (both against SVD_best).

Condensed result (ungated; entries R rdm2 / R nn / far-ZZ ratio / trace-distance ratio; T = 2 and all chi lists are in `results/tables.md`):

| family | mu | T | spcf in plain gauge | | spcf in back gauge | |
|---|---|---|---|---|---|---|
| staggered | 0.25 | 3 | 2.52 / 2.32 / 1.10 / 1.17 | no | 3.23 / 3.05 / 1.56 / 1.44 | no |
| staggered | 0.25 | 4 | 2.63 / 2.34 / 1.41 / 1.21 | no | 2.96 / 2.33 / 0.89 / 0.94 | PASS |
| staggered | 0.5 | 3 | 4.20 / 4.06 / 0.99 / 0.98 | PASS | 1.79 / 1.87 / 2.96 / 2.27 | no |
| staggered | 0.5 | 4 | 4.67 / 4.29 / 1.22 / 0.99 | PASS | 2.98 / 2.59 / 0.69 / 1.06 | PASS |
| staggered | 1.0 | 3 | 4.38 / 4.14 / 0.79 / 0.97 | PASS | 0.36 / 0.38 / 9.58 / 11.75 | no |
| staggered | 1.0 | 4 | 4.09 / 3.83 / 0.89 / 0.97 | PASS | 0.98 / 0.95 / 3.64 / 2.93 | no |
| staggered | 2.0 | 3 | 4.16 / 4.40 / 0.38 / 0.92 | PASS | 0.01 / 0.01 / 94.6 / 78.1 | no |
| staggered | 2.0 | 4 | 3.01 / 3.19 / 0.54 / 0.94 | PASS | 0.07 / 0.08 / 28.2 / 14.0 | no |
| staggered | inf | 3 | 3.16 / 3.26 / 0.52 / 0.97 | PASS | 0.00 / 0.00 / (gauge) | no |
| staggered | inf | 4 | 2.34 / 2.41 / 0.65 / 0.98 | PASS | 0.00 / 0.00 / (gauge) | no |
| domain wall | 0.25 | 3 | 2.81 / 2.54 / 1.44 / 1.14 | no | 3.42 / 2.60 / 1.14 / 0.94 | PASS |
| domain wall | 0.25 | 4 | 1.71 / 1.46 / 1.48 / 1.35 | no | 2.71 / 2.45 / 1.03 / 0.93 | PASS |
| domain wall | 0.5 | 3 | 3.71 / 3.80 / 1.09 / 0.99 | PASS | 1.28 / 1.34 / 2.87 / 2.61 | no |
| domain wall | 0.5 | 4 | 3.00 / 2.65 / 0.88 / 0.99 | PASS | 2.21 / 2.01 / 1.09 / 1.08 | PASS |
| domain wall | 1.0 | 3 | 3.03 / 2.94 / 1.01 / 0.98 | PASS | 0.17 / 0.19 / 29.6 / 14.6 | no |
| domain wall | 1.0 | 4 | 2.50 / 2.43 / 0.96 / 0.98 | PASS | 0.29 / 0.33 / 9.10 / 3.20 | no |
| domain wall | 2.0 | 3 | 2.21 / 2.01 / 0.86 / 0.97 | PASS | 0.01 / 0.01 / 322 / 92 | no |
| domain wall | 2.0 | 4 | 2.39 / 2.20 / 0.71 / 0.97 | PASS | 0.02 / 0.02 / 112 / 26 | no |
| domain wall | inf | 3 | 1.47 / 1.37 / 0.87 / 0.99 | no | 0.00 / 0.00 / (gauge) | no |
| domain wall | inf | 4 | 1.59 / 1.61 / 0.85 / 0.99 | no | 0.00 / 0.00 / (gauge) | no |

Where the back-gauge ratios are tiny, SVD in the plain gauge is simply far better than anything in the back gauge (the gauge dominates), so those rows say nothing about spcf. The like-for-like gain in the back gauge (spcf-back / SVD-back) is 2.1 to 5.7x in every ungated cell (column in `results/tables.md`). Figure `figures/mu_dial.png`: left panel the pre-registered comparison (plain-gauge spcf vs SVD in the better gauge), right panel the back-gauge like-for-like gain.

The mu dial, median R(rdm2) / R(nn), spcf in the plain gauge:

| mu | stag T=3 | stag T=4 | dw T=3 | dw T=4 |
|---|---|---|---|---|
| inf | 3.16 / 3.26 | 2.34 / 2.41 | 1.47 / 1.37 | 1.59 / 1.61 |
| 2 | 4.16 / 4.40 | 3.01 / 3.19 | 2.21 / 2.01 | 2.39 / 2.20 |
| 1 | 4.38 / 4.14 | 4.09 / 3.83 | 3.03 / 2.94 | 2.50 / 2.43 |
| 0.5 | 4.20 / 4.06 | 4.67 / 4.29 | 3.71 / 3.80 | 3.00 / 2.65 |
| 0.25 | 2.52 / 2.32 | 2.63 / 2.34 | 2.81 / 2.54 | 1.71 / 1.46 |

The pure-state endpoints reproduce the repo's earlier d = 2 numbers (Neel 2.3 to 3.2x, domain wall 1.5 to 1.6x). **The gain does not decay as mu goes down to 0.25: it is as large or larger at mu = 0.5 to 1 than in the pure limit.** At mu = 0.25 the plain-gauge ratio drops to 1.5 to 2.8x, and what fails the pre-registered rule there is the collateral comparison against a back-gauge SVD that has become competitive.

Counts at mu <= 1 (18 cells = 2 families x 3 mu x 3 T): the ratio criterion alone holds in 18/18 cells for some gauge; the full rule holds in 15/18. Failing: staggered mu = 0.25 at T = 2 and 3, domain wall mu = 0.25 at T = 2 (far-ZZ ratio 1.52 to 1.60 in the plain gauge or trace-distance ratio 1.17 to 1.80 in either gauge, against the better-gauge SVD). Like-for-like (same gauge) the trace distance is never worse than SVD's (ratio 0.81 to 0.99 over all ungated cells).

The three "ambiguous" conditions: (i) gain only at mu >= 2: no, it is present and larger at mu <= 1. (ii) spcf-plain beating SVD-plain but not SVD-back: no in all passing cells, because the ratio is against the better gauge. (iii) rdm2 gain with trace distance worse by > 10 percent: no (like-for-like <= 1.01). **Kill condition 2 (ratio <= 1.2 in every usable mu <= 1 cell): not met**, the smallest plain-gauge median ratio at mu <= 1 is 1.46 (domain wall, mu = 0.25, T = 4, nn) and most are 2.5 to 4.7.

Gated spcf (gate 1.0, constants fitted on d = 2 pure-state cells): gated/ungated rdm2 has median 1.001 (10th to 90th percentile 1.000 to 1.40, worst 3.7), fired fraction 0.33 vs 0.36. The gate almost never closes here and costs accuracy in some cells (staggered mu = 0.25 plain R 1.76 vs 2.63). It does nothing useful as calibrated.

Weaker quantities: R(nnn) in the plain-gauge cells at mu <= 1 is 0.79 to 2.09 (median 1.44), so the gain concentrates on the near-neighbour data the objective targets. Far collateral in those cells (ratio to better-gauge SVD) is 0.60 to 1.60, i.e. neutral to mildly worse.

### Control: heis, mu = 1

`heis` cells (N = 10, staggered mu = 1, T in {1, 1.5, 2}; domain wall T = 1.5), both gauges, `results/heis_*`, `results/heis_analysis.txt`:

| cell | spcf gauge | R vs better-gauge SVD (rdm2 / nn) | like-for-like (rdm2 / nn) | verdict by the rule |
|---|---|---|---|---|
| stag T=1 | plain | 1.71 / 1.18 | 1.71 / 1.18 | no |
| stag T=1 | back | 0.04 / 0.05 | 2.66 / 2.55 | no |
| stag T=1.5 | plain | 2.50 / 2.25 | 2.50 / 2.25 | PASS |
| stag T=1.5 | back | 0.59 / 0.55 | 2.85 / 2.59 | no |
| stag T=2 | plain | 1.91 / 2.08 | 1.91 / 2.08 | no |
| dw T=1.5 | plain | 0.98 / 0.91 | 2.25 / 2.33 | no |
| dw T=1.5 | back | 2.41 / 2.28 | 2.41 / 2.28 | PASS |

**The control did not behave as the plan predicted.** The plan said a gain in the back gauge would be suspicious and call for checking the metric. Checks done: the anchor (0b) shows the metric code equals the d = 2 code; an independently evolved dense rho and a transfer-matrix evaluation of the MPS marginals reproduce `pur_metrics` to machine precision (see Test 0). So the metric is not the cause. The pure-state d = 2 heis at the same N and T, with the existing code (`heis_pure_anchor.py`), gives spcf/SVD rdm2 of 0.62 to 1.10 at T = 1.5 (chi = 6 to 24: 0.79, 0.98, 0.72, 0.62, 0.85) and 0.68 to 1.11 at T = 1.0, i.e. gains of at most 1.6x. So the purification pipeline adds a gain on heis that is absent in the pure state. The back gauge is also a poor SVD gauge here (mu = 1 is not thermal for H, SVD-back rdm2 stays at 0.06 to 0.1 for every chi >= 16 up to 48 at T = 1.5), so one natural reading is that SVD wastes bond dimension on gauge-dependent structure that spcf's physical objective ignores. That reading is not tested here.

### Test 1 verdict

**Success under the pre-registered rule.** Best cells: staggered mu = 1, T = 3 plain (R 4.38 rdm2, 4.14 nn, far ZZ 0.79, trace distance 0.97) and staggered mu = 0.5, T = 4 plain (4.67, 4.29, 1.22, 0.99). Kill criteria not met, no ambiguity trigger. The heis control is violated, which weakens the interpretation (below).

## Test 2: head-to-head with DMT

Built `rank2/dmt_np.py` to the paper's definition (Sec. III, App. B of 1707.01506): Pauli-basis MPDO with real tensors, real 16 x 16 Pauli superoperators for the same gates, Frobenius-canonical bases from the SVD of the two-site tensor, trace vectors tr[x_alpha sigma^mu] via identity-component environments, Gram-Schmidt reserved bases (rank 4 each in the bulk), connected matrix, SVD of its lower-right block to chi' = chi - r_L - r_R (8 in the bulk), M' = M - Q_L,perp (D - D') Q_R,perp^T, second SVD. Total bond <= chi, so the stored parameters are (chi, 4, chi) = 4 chi^2 per bulk site, equal to the purification. Unit checks (`test_dmt.py`): superoperators orthogonal (2e-16) and reproduce G rho G^dag; at large chi the MPDO TEBD equals the dense physical rho to 4e-15; at all 51 truncating cuts of a N = 6 run **tr rho, rho_{0..b+1} and rho_{b..N-1} are unchanged to 9e-15** (the Frobenius cut changes them by 5e-5, so the check has teeth).

Same Trotter circuit, same product initial state, chi in {12, 16, 24, 32}, spcf-purification in the gauge that passed Test 1.

**Primary cell: staggered mu = 0.5, T = 4 (best median ratio of Test 1), spcf in the plain gauge.** All errors against the dense physical rho:

| quantity | arm | chi=12 | chi=16 | chi=24 | chi=32 |
|---|---|---|---|---|---|
| rdm2 | SVD purification, plain | 5.94e-2 | 3.83e-2 | 1.31e-2 | 6.10e-3 |
| | SVD purification, back | 4.63e-2 | 3.60e-2 | 2.14e-2 | 1.58e-2 |
| | **spcf purification, plain** | **1.42e-2** | **8.05e-3** | **2.16e-3** | **8.59e-4** |
| | spcf gated 1.0, plain | 1.42e-2 | 8.04e-3 | 3.56e-3 | 1.39e-3 |
| | DMT (l = 3) | 4.05e-2 | 2.34e-2 | 1.42e-2 | 6.73e-3 |
| | Frobenius MPDO | 6.18e-2 | 4.38e-2 | 2.93e-2 | 1.64e-2 |
| nnn rms | SVD plain | 2.07e-2 | 1.13e-2 | 5.02e-3 | 2.57e-3 |
| | spcf plain | 1.30e-2 | 7.01e-3 | 2.69e-3 | 1.43e-3 |
| | DMT | 3.19e-2 | 2.05e-2 | 1.18e-2 | 6.08e-3 |
| far ZZ rms | SVD plain | 1.88e-2 | 1.13e-2 | 3.11e-3 | 1.46e-3 |
| | spcf plain | 1.55e-2 | 1.01e-2 | 4.05e-3 | 1.66e-3 |
| | DMT | 1.91e-2 | 2.47e-2 | 1.06e-2 | 8.61e-3 |
| \|dE\| | SVD plain | 4.8e-2 | 1.9e-2 | 8.2e-3 | 2.1e-3 |
| | spcf plain | 3.7e-2 | 1.1e-2 | 8.6e-4 | 3.9e-4 |
| | **DMT** | **3.3e-4** | **2.5e-4** | **8.8e-5** | **6.4e-5** |
| trace distance | SVD plain / spcf plain | 0.579 / 0.573 | 0.433 / 0.423 | 0.231 / 0.228 | 0.118 / 0.117 |
| | DMT | 0.731 | 0.743 | 0.698 | 0.630 |
| lambda_min | purification arms | >= 0 | >= 0 | >= 0 | >= 0 |
| | DMT | -1.1e-2 | -7.8e-3 | -6.6e-3 | -5.2e-3 |
| tr rho before normalisation | DMT / Frobenius | 1 / 0.953 | 1 / 0.959 | 1 / 0.971 | 1 / 0.990 |

DMT/spcf: rdm2 2.86, 2.91, 6.55, 7.83; nnn 2.45, 2.92, 4.38, 4.25 (rule: >= 1.5 at >= 3 of 4 chi; **4 of 4**); far ZZ 1.23 to 5.18; DMT <= spcf on rdm2 at 0 of 4 chi (kill not met). DMT wins on energy drift by 6 to 110x in this cell, as the plan predicted, and it preserves the trace exactly while the Frobenius arm loses 1 to 5 percent. Sanity: DMT is close to SVD-plain-purification at chi = 24 to 32 (1.1x), consistent with the DMT paper's statement that purification SVD converges like DMT. Figure: `figures/test2_stag_mu0.5_T4.png`. The "DMT at chi + 8" bracket (same as DMT at chi + 8, 4(chi + 8)^2 parameters) gives rdm2 1.74e-2, 1.42e-2, 6.73e-3, 4.78e-3, still behind spcf at the smaller parameter count.

**Other cells** (DMT/spcf; spcf in the gauge named; `results/test2_*.json`, `.txt`):

| cell | spcf gauge | rdm2 at chi=12,16,24,32 | nnn at chi=12,16,24,32 | pass (>= 1.5, 4 chi) | far ZZ | DMT lambda_min (chi=24) |
|---|---|---|---|---|---|---|
| staggered mu=0.5 T=4 (primary) | plain | 2.86 2.91 6.55 7.83 | 2.45 2.92 4.38 4.25 | 4/4 | 1.23 to 5.18 | -6.6e-3 |
| staggered mu=0.5 T=3 | plain | 7.66 6.17 37.3 41.4 | 4.85 5.08 20.6 30.6 | 4/4 | 2.0 to 16 | -3.5e-3 |
| staggered mu=1 T=3 | plain | 15.7 25.0 91.9 126 | 13.2 19.4 71.9 143 | 4/4 | 6 to 161 | -2.2e-2 |
| staggered mu=0.25 T=4 | back | 2.88 2.10 2.16 1.95 | 2.33 1.99 1.70 1.31 | 3/4 | 1.2 to 2.4 | -1.4e-3 |
| domain wall mu=1 T=4 | plain | 9.85 18.0 26.6 47.7 | 7.40 13.0 20.1 32.8 | 4/4 | 5.9 to 29.6 | -2.6e-2 |
| domain wall mu=0.5 T=4 | plain | 3.66 4.49 6.45 11.7 | 2.09 2.44 4.14 5.97 | 4/4 | 1.9 to 5.2 | -4.6e-3 |
| domain wall mu=0.25 T=4 | back | 2.88 2.60 2.64 1.96 | 2.19 1.68 1.59 1.58 | 4/4 | 0.9 to 2.0 | -4.9e-4 |
| **extension, beyond the plan's dial:** staggered mu=0.1 T=4 | back | 2.99 2.56 2.04 2.06 | 2.15 1.87 1.33 1.07 | **2/4** | 1.2 to 2.3 | +1.0e-4 |
| **extension:** domain wall mu=0.1 T=4 | back | 2.87 2.20 2.07 2.03 | 1.99 1.29 1.32 1.25 | **1/4** | 0.5 to 2.0 | +3.6e-4 |

DMT gets worse relative to the purification arms as the state gets closer to pure (mu up), which matches the known DMT weakness for near-pure states (at mu = 1, T = 3 DMT is 3 to 60x worse than the plain-gauge SVD purification, rdm2 3.8e-3 vs 6.2e-5 at chi = 32). Toward mu -> 0 the lead on nnn closes: at mu = 0.1 spcf still has 2.0 to 3.0x on rdm2 but only 1.1 to 2.2x on nnn. Near-infinite temperature is DMT's design regime, so this is the expected direction, and the success should be read for mu >= 0.25.

**N = 12 check** (staggered mu = 0.5, T = 4, plain gauge; metrics from the dense state without trace distance; `results/n12_*.json`; no pre-registered requirement, run as a robustness check):

| chi | SVD plain rdm2 | SVD back rdm2 | spcf plain rdm2 | R(rdm2) vs better gauge | R(nn) | R(nnn) | DMT rdm2 | DMT/spcf rdm2 | DMT/spcf nnn |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4.42e-2 | 3.90e-2 | 9.13e-3 | 4.28 | 3.56 | 1.61 | 2.31e-2 | 2.5 | 2.7 |
| 24 | 1.50e-2 | 2.26e-2 | 2.37e-3 | 6.32 | 5.76 | 1.86 | 1.49e-2 | 6.3 | 4.0 |
| 32 | 7.40e-3 | 1.78e-2 | 1.05e-3 | 7.08 | 7.58 | 1.82 | 7.26e-3 | 6.9 | 3.7 |
| 48 | 7.77e-4 | 1.29e-2 | 1.42e-4 | 5.48 | 6.69 | 1.79 | 4.35e-3 | 30.7 | 10.8 |

Both Test 1 and Test 2 conclusions hold at N = 12 for this cell.

### Test 2 verdict

**Success on the pre-registered rule** at the primary cell and at 7 of 7 cells with mu = 0.25 to 1. The qualification that matters: the lead on nnn narrows at the high-temperature end (mu = 0.1: 1.1 to 2.2x), DMT has the lower energy drift in 35 of 36 entries (median 71x), and DMT is strongly non-positive (lambda_min up to -2e-2) while the purification arms are positive by construction.

## Cost

Measured wall for a whole N = 10 run on a shared and loaded machine, indicative only (README: wall is not a comparison axis): spcf-purification 2 to 21 s (median 2.2 s at chi = 4, 16 s at chi = 16, 21 s at chi = 32), about 2 to 8x the SVD purification run (SVD in d = 4 is itself heavier than in d = 2; the d = 2 spcf is 3 to 16x SVD in the repo notes), and 19 to 27x the d = 2 spcf run on the same cell (anchor, chi = 6 to 12). N = 12, chi = 48: 82 s. DMT and Frobenius: 0.1 to 4.6 s. The plan's cost estimate (region W of 2^6 x 16 chi^2) was low by 4x: with a = 2 the column space is (outer bond) x 2^6 ancilla indices x (outer bond), i.e. 2^6 x 2^6 chi^2. It was still affordable at N = 10 and 12; a = 1 would cut it 16x and was not tried.

## Deviations from the plan

1. New code is in `rank2/` (as instructed), not `rule/spcfpur.py` and not copies of `test_spcfast.py` etc.; `SPCFPur` subclasses `SPCFast` instead of copying the file, so the F matrices are the d = 2 ones by construction. `pur_bench.py` is the driver; `purlib.py` holds the d-generic copies.
2. Test 0(a): the pre-registered adjoint threshold 1e-5 is met only in float64 (`precision='f64'`, 1e-13). In the production float32 setting the adjoint error is 2e-5 to 3.5e-4, the same order as the d = 2 code's 5.9e-5, so the threshold was not reachable in float32 by any d = 2 port. Reported both.
3. Test 0(b): spcf agreement on the targeted metrics is within 5 percent; the plan said "about 10 percent". |dE| at chi = 12 differs 3.6x (small absolute value, not in the objective).
4. Test 1 grid pruned: step 2 uses at most 4 chi (chi <= 32) per cell; spcf run only at T = 2 (ungated), T = 3, 4 (gated and ungated) for mu <= 1, and at T = 3, 4 for the mu = 2, inf controls. Gate constants were not recalibrated. The heis control was a smaller scan (T in {1, 1.5, 2, 3}, mu = 1, ungated).
5. Success rule needs three choices the plan left open: the "usable window" is per spcf-gauge (SVD in that gauge in [1e-3, 1e-1]); "SVD in the better of the two gauges" is per metric and per chi; far-ZZ and trace-distance comparators are the same better-gauge SVD. Medians over 3 to 4 chi. Like-for-like numbers are given alongside.
6. Test 2 chi set {12, 16, 24, 32} was run in the best Test 1 cell chosen before looking at Test 2 (largest median R(rdm2) among passing cells whose window overlaps the chi set: staggered mu = 0.5, T = 4, plain), plus 6 supplementary cells and a mu = 0.1 extension that is outside the plan's dial. The plan's optional "gamma-graded rTEBD arm" was not built because the ambiguous branch was not triggered.
7. DMT is my own implementation from the paper; it is unit-checked for the preservation property and for exact evolution, and it reproduces the qualitative statement DMT ~ purification-SVD. It was not benchmarked against the paper's published numbers (different model size and chi). Radius 1 only.
8. Trace distance reported as 0.5 ||rho - rho_ex||_1; the Frobenius MPDO is trace-normalised before scoring. Dense N = 12 references and the 1.7 GB cache live in `rank2/_refcache/` (git-ignored by the existing `.gitignore`).

## What these results do and do not show

- Do show: at N = 10 and 12, at equal stored parameters, spcf on a purification beats plain SVD in the better of two ancilla gauges by 2.5 to 4.7x on rdm2 and nn in genuinely mixed cells (mu = 0.25 to 1), beats DMT by about 2 to 130x on rdm2, and leaves the far ZZ correlator and the physical trace distance neutral or mildly worse. The gain does not fade as mu falls to 0.25 and is bigger than in the pure limit. The pre-registered kill conditions did not trigger.
- Do not show, and the first thing to resolve: (a) **gauge fairness of the SVD baseline.** The heis control gains where the pure-state code does not, and SVD-back is a poor gauge for a non-thermal start. Only plain and backward gauges were used, with no disentangler (Hauschild) and no optimised ancilla unitary. Part of the gain may be spcf compensating a gauge-inefficient SVD. DMT has no gauge, so the DMT result is not subject to this. (b) Initial states are field-only product Gibbs states quenched by H; the thermal Gibbs state with an H-matched backward gauge, where KBM's trivial evolution holds, was not tested. (c) The scored quantities rdm2 and nn are the objective's own; nnn gains are only 1.0 to 2.1x and rdm3 was not scored in the criteria. (d) N = 10 and 12, T <= 4, dt = 0.1, one model with one field set. (e) The gate is useless as calibrated.

## Recommended next step

Add the gauge-fair baseline and one thermal cell before any larger run:
1. **SVD with a Hauschild-style disentangler** (norm- or entropy-based ancilla rotation, applied per cut before the SVD) as a third purification baseline, with spcf run after it (the plan's open question on combining them). This is the single experiment that can either remove the heis anomaly and the gauge caveat or show that spcf's gain is orthogonal to gauge. It reuses `SPCFPur` and the `rank2` driver; the new piece is a small unitary optimisation on the ancilla index of the two-site tensor.
2. **A thermal start**: a Gibbs state of H prepared in imaginary time with a small chi (so its truncation is controlled), then a local quench, backward gauge, one N = 12 cell with an energy-density profile as the score (the plan's first experiment). Include DMT and report |dE| and nnn, where DMT is strongest.
3. Re-fit the gate on purification cells, or drop it. Try a = 1 to cut cost 16x before N >= 16.
4. Park the heis control until step 1, then repeat it: if the disentangler removes the heis gain it is the gauge; if not, the premise that heis is a no-gain model needs to be revisited for purifications.

## Files

Report: `reports/first_moves/rank2_purification_results.md`. Code, raw outputs and figures: `rank2/` (see the table under "What was built"). Key raw outputs: `rank2/results/scan_svd_stag.json`, `scan_svd_dw.json`, `step2A_spcf.json`, `step2B.json`, `step2C.json`, `step2D.json`, `test1_summary.json`, `test1_analysis.txt`, `heis_*`, `test2_*.json`, `n12_*.json`, `test0a_*.json`, `test0b_anchor.json`, `test_dmt.json`, `check_metric_independent_heis.log`.
