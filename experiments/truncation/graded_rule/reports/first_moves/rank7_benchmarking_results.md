# Rank 7 first moves: results of T1, T2 and T3

Plan executed: `rank7_benchmarking.md` (same folder). Everything new lives in `rank7/`; no existing file was modified. Nothing was committed.
Cell: Ising (hx = 0.9045, hz = 0.8090), N = 20, T = 8, dt = 0.1, Néel start, dense Trotter reference with the same gates.
N = 20 was **not** reduced: every run finished well inside the 30-minute limit even with the machine loaded (longest: a chi = 96 cold-gate timing job, a few minutes of CPU).

## 1. Verdicts

| test | pre-registered success / kill | outcome | verdict |
|---|---|---|---|
| **T1** wall-clock lever | success: some gate has r >= 1.5 and equal-error CPU ratio <= 1.0; kill: r < 1.3 whenever f < 0.2; ambiguous: ratio in 1.0-1.5, re-measure at chi = 64 | r(nn) = 1.56-1.78 and r(1-site) = 1.51-1.59 for every gate at chi = 32 and 48, so **not killed**. Run **as specified**, one cell passes (rs = 0.6, chi = 48: ratio 0.8 / 0.9), but that number is an artifact of `pareto_spcf.py` (section 3.2). With the gate state of the error run, the best ratio is 2.34 (chi = 32), 1.48 (chi = 48), **1.16 nn / 1.33 1-site (chi = 64, the required re-measure)** | **Wall-clock not shown; ambiguous band at chi = 64-96, not <= 1.0.** Memory claim intact (r about 1.5-1.8). |
| **T2** vs free post-processing | success: (a) t*_spcf(chi) >= t*_SVD(1.5 chi) at both tolerances, (b) spcf below the best post-processed SVD on >= 70 % of growth-phase samples, (c) eps_c no worse; kill: post-processing within 20 % over most of the window, or eps_c worse by > 1.2x | Ungated spcf (rs = 1e-2): (a) 12/12, (b) >= 0.88 in all six cells, (c) median eps_c ratio 0.30-0.32. Gate picked by the literal T1 reading (rs = 0.6): (a) 11/12. Gate picked by the cold T1 reading (rs = 0.3, eps_min = 1e-5): (a) 6/12, (b) 0.62 in one cell. No kill condition triggered for any gate | **Pass for the ungated rule; strict pass fails for the cost-optimised gates** (they give away the early-time horizon). Not killed. Post-processing is a weak competitor here. |
| **T3** kicked-Ising transfer | success: r >= 1.5 at SVD's horizon in 2 of 3 ergodic settings with eps_c no worse; kill: r < 1.3 in all | median r(1-site) at the SVD horizon = 1.00, 0.99, 1.07 for settings A, B, C (values per chi_ref 0.96-1.04, 0.94-1.00, 0.80-1.10); eps_c ratio about 1.0 | **KILLED.** The gain does not reach circuit benchmarks with static window targets. |

Net: for rank 7, the supportable claim is **memory, on small-step continuous-time quenches** (r about 1.4-2.1 over the run, 1.4-1.7 at T = 8; r^2 about 2-3x). Wall-clock break-even is not reached at chi <= 96 and the circuit-benchmark transfer is dead as formulated.

T3 was run on a judgement call. The plan says "only if T2 passes", and T2 passes for the ungated rule but not strictly for the T1-selected gates (section 4). T3 is cheap and its outcome is a kill, so the call does not change the conclusion.

## 2. What was built and run

| file (all in `rank7/`) | what |
|---|---|
| `run_t1.sh` | T1 exactly as specified: `pareto_spcf.py ising 20 8.0 32 32,48 out 1 '2:0:4:1.0:1-2-3:0:1:<eps>:all:<rs>'`, rs in {1e-2, 0.1, 0.3, 0.6} at eps_min = 1e-7 and eps_min = 1e-5 at rs = 0.3, plus an SVD timing ladder (chi = 24-128, 3 reps) before and after |
| `t1_cold.py`, `run_t1_cold.sh`, `run_t1_ext.sh` | **added**: same cell, but a fresh `SPCFast` per timed run so that error, fired fraction and time describe the same gate behaviour; SVD is timed in the same process, interleaved, at the equal-error chi. Used for the cold-gate table and the chi = 64 / 96 re-measures |
| `rutil.py`, `t1_score.py` | SVD ladder of `results/high_pressure/matched_hp.json` (chi 24-192), log-log chi_eq, r, ratios |
| `ladder_t.py`, `build_ref.py`, `run_t2_svd.sh`, `run_t2_spcf.sh`, `run_t2_static.sh` | T2: TEBD loop copied from `M.run_tebd` (selftest: bit-identical MPS), samples every 0.5 with all Pauli expectations (1-site, nn, nnn), <Z>, all-pair <ZZ>, overlap with the dense reference, running sum log F of per-cut truncation fidelities (`FidCut`) |
| `t2_analysis.py`, `t2_report_tables.py` | all arms from the same runs: SVD, SVD x F_MPS, SVD x F_true (oracle), log F and 1/chi extrapolation, spcf, spcf x F_MPS; t*, win fractions, eps_c, r(t) |
| `floquet.py`, `run_t3_svd.sh`, `run_t3_spcf.sh`, `t3_analysis.py`, `t3_diag.py` | T3: `make_kicked_gates`, one-sweep-per-Floquet-step TEBD with alternating direction (selftest: a sweep equals the dense unitary prod ZZ prod RX to 1e-16), dense reference, `SPCFast(taus=())` |
| `results/` | every output (`t1_*`, `t1c*_*`, `t1x_*`, `t2_*`, `t3_*`), `t1_scores.txt`, `t2_analysis*.txt`, `t2_report_tables.md`, `t3_analysis_*.txt`; `_refcache/` (git-ignored, 1.5 GB) holds the dense reference trajectories |

## 3. T1: can the gate make spcf pay in wall-clock?

### 3.1 As specified (`pareto_spcf.py`, reps = 1)

r is the log-log interpolated SVD chi from `matched_hp.json` that reaches spcf's error, divided by spcf's chi. "CPU SVD@eq" is interpolated from my own SVD timing ladder. f = fired/calls as printed by the script (average over its two runs, see 3.2).  Every chi_eq lies inside the ladder.

| gate (rs, eps_min) | chi | r nn | r 1-site | f | nn error | CPU spcf | CPU SVD@eq nn | ratio nn | ratio 1-site |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2, 1e-7 | 32 | 1.68 | 1.57 | 0.225 | 4.65e-3 | 10.8 | 3.6 | 3.0 | 3.4 |
| 1e-2, 1e-7 | 48 | 1.78 | 1.59 | 0.188 | 1.68e-3 | 15.7 | 8.0 | 2.0 | 2.3 |
| 0.1, 1e-7 | 32 | 1.68 | 1.57 | 0.198 | 4.65e-3 | 8.2 | 3.6 | 2.2 | 2.6 |
| 0.1, 1e-7 | 48 | 1.77 | 1.59 | 0.148 | 1.69e-3 | 12.1 | 7.9 | 1.5 | 1.8 |
| 0.3, 1e-7 | 32 | 1.65 | 1.55 | 0.161 | 4.82e-3 | 5.8 | 3.5 | 1.7 | 1.9 |
| 0.3, 1e-7 | 48 | 1.77 | 1.58 | 0.112 | 1.71e-3 | 8.8 | 7.9 | 1.1 | 1.3 |
| 0.6, 1e-7 | 32 | 1.56 | 1.51 | 0.119 | 5.36e-3 | 4.1 | 3.1 | 1.3 | 1.4 |
| **0.6, 1e-7** | **48** | **1.65** | **1.52** | 0.083 | 2.01e-3 | 6.0 | 7.1 | **0.8** | **0.9** |
| 0.3, 1e-5 | 32 | 1.65 | 1.52 | 0.122 | 4.83e-3 | 6.0 | 3.5 | 1.7 | 2.0 |
| 0.3, 1e-5 | 48 | 1.71 | 1.51 | 0.073 | 1.86e-3 | 8.7 | 7.5 | 1.2 | 1.4 |

Read literally, the success criterion is met once (bold row). The kill criterion (r < 1.3 whenever f < 0.2) is not met: every setting with f between 0.07 and 0.2 has r(nn) >= 1.56.

### 3.2 Why the "as specified" times are not trustworthy

`pareto_spcf.py` builds one `SPCFast` object and calls it 1 + reps times. `rel_skip` skips a cut when its discarded weight is below `rel_skip * tmax`, where `tmax` is the running maximum seen **by that object**. In run 1 `tmax` grows with time. In every later run it starts at its final value, so early cuts are skipped that run 1 fired. The script reports **errors from run 1 and "steady" time from the later runs**, so for rel_skip above 0.01 it pairs the error of one gate behaviour with the cost of a cheaper one. Evidence from the new cold-gate bench (`t1_cold.py`, every run starts with `tmax = 0`):

| gate | chi | fired, first run (cold) | steady CPU as specified | steady CPU cold gate |
|---|---|---|---|---|
| 0.1, 1e-7 | 32 | 727 | 8.2 s | 12.5 s |
| 0.3, 1e-7 | 32 | 707 | 5.8 s | 12.1 s |
| 0.6, 1e-7 | 32 | 581 | 4.1 s | 9.7 s |
| 0.6, 1e-7 | 48 | 421 | 6.0 s | 14.4 s |

In a real simulation `tmax` is not known in advance, so the cold numbers are the relevant ones. `rel_skip` barely changes the fired count in a cold run (727 -> 707 -> 581 as rs goes 1e-2 -> 0.3 -> 0.6 at chi = 32). The stateless knob is `eps_min`. I did not modify `pareto_spcf.py`.

### 3.3 Cold-gate protocol (same gate state for error and timing; SVD timed at the equal-error chi, interleaved, 2 reps, min)

CPU time (`process_time`) is used. The SVD ladder reproduced to 1-5 % between the start and the end of the scan (chi = 64: 5.12 s vs 5.16 s) while the **wall** time of the same run moved from 5.4 s to 12.5 s (load average 4-23 on 4 cores, about 6 other agents). Rep-to-rep spread inside the interleaved runs was 1-8 %. Treat ratios as +-10 %.

| gate (rs, eps_min) | chi | r nn | r 1-site | f | nn error | CPU spcf | CPU SVD@eq nn | **ratio nn** | **ratio 1-site** |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2, 1e-7 | 32 | 1.68 | 1.57 | 0.239 | 4.65e-3 | 12.7 | 3.8 | 3.35 | 3.68 |
| 1e-2, 1e-7 | 48 | 1.78 | 1.59 | 0.202 | 1.68e-3 | 19.8 | 8.1 | 2.45 | 2.85 |
| 0.1, 1e-7 | 32 | 1.68 | 1.57 | 0.239 | 4.65e-3 | 12.5 | 3.8 | 3.31 | 3.85 |
| 0.1, 1e-7 | 48 | 1.77 | 1.59 | 0.196 | 1.69e-3 | 20.5 | 8.5 | 2.40 | 2.92 |
| 0.3, 1e-7 | 32 | 1.65 | 1.55 | 0.233 | 4.82e-3 | 12.1 | 3.7 | 3.31 | 3.83 |
| 0.3, 1e-7 | 48 | 1.77 | 1.58 | 0.163 | 1.71e-3 | 16.3 | 7.6 | 2.14 | 2.57 |
| 0.6, 1e-7 | 32 | 1.56 | 1.51 | 0.191 | 5.36e-3 | 9.7 | 3.0 | 3.22 | 3.32 |
| 0.6, 1e-7 | 48 | 1.65 | 1.52 | 0.138 | 2.01e-3 | 14.4 | 6.5 | 2.22 | 2.54 |
| **0.3, 1e-5** | 32 | 1.65 | 1.52 | 0.156 | 4.83e-3 | 7.7 | 3.3 | 2.34 | 2.65 |
| **0.3, 1e-5** | 48 | 1.71 | 1.51 | 0.086 | 1.86e-3 | 9.7 | 6.5 | 1.48 | 1.79 |
| **0.3, 1e-5** | **64** | 1.64 | 1.50 | 0.049 | 9.88e-4 | 11.7 | 10.1 | **1.16** | **1.33** |

The chi = 64 row is the re-measure the decision tree asks for when the ratio is in the 1.0-1.5 band. It is still in the band, so by the plan's own text T1 stays **ambiguous**, not success.

### 3.4 Extension beyond the plan's grid (flagged: not part of the pre-registered scan)

`eps_min` is an absolute threshold on the discarded weight. A threshold that is right at chi = 48 switches the rule off at chi = 96 (the tails are below 1e-5 there):

| gate (rs = 0.3) | chi | r nn | r 1-site | f | CPU spcf | CPU SVD@eq nn | ratio nn | ratio 1-site |
|---|---|---|---|---|---|---|---|---|
| eps_min 1e-5 | 96 | **1.13** | **1.09** | 0.007 | 11.3 | 10.4 | 1.09 | 1.15 (r lost, so not a win) |
| eps_min 1e-6 | 64 | 1.73 | 1.49 | 0.106 | 18.7 | 11.0 | 1.70 | 2.34 |
| eps_min 1e-6 | 96 | 1.72 | 1.51 | 0.053 | 24.7 | 20.9 | **1.18** | **1.45** |
| eps_min 1e-7 | 64 | 1.75 | 1.51 | 0.134 | 23.4 | 11.5 | 2.02 | 2.69 |
| eps_min 1e-7 | 96 | 1.70 | 1.48 | 0.075 | 33.2 | 21.5 | 1.54 | 1.91 |

The best gate that keeps r about 1.5-1.7 at chi = 96 has an equal-error ratio of 1.18 (nn) / 1.45 (1-site). The ratio falls with chi (eps_min 1e-6: 1.70 -> 1.18 from chi = 64 to 96), so break-even near chi = 110-130 is plausible, but that is an extrapolation of two points and I did not measure it.

### 3.5 The plan's cost model does not describe this regime

The plan predicts equal-error wall-clock = (1 + 15 f) / r^3. Measured ratios are about 3x **above** that (1.4-3.5x; the column `pred` in `results/t1_scores.txt` is 0.4-1.0 where the cold-gate measurement is 1.1-3.4). Two inputs are wrong at N = 20, chi <= 128:

1. **SVD-TEBD time does not scale as chi^3.** CPU of the SVD run (steady, N = 20, T = 8): chi = 24 / 32 / 48 / 64 / 96 / 128 gives 1.01 / 1.63 / 2.89 / 5.12 / 9.56 / 17.3 s, local exponents 1.7, 1.4, 2.0, 1.5, 2.1 (overall 1.70). A factor r in chi therefore buys r^1.7, not r^3, in time (the SVD cost at the interpolated chi is 3.8 s at chi_eq = 54, not 1.63 x 1.68^3 = 7.7 s).
2. **A fired cut costs about 25-38 SVD-TEBD cuts, not 15.** (cpu_spcf - cpu_svd_same) / fired against cpu_svd_same / calls: 24-27x at chi = 32, 26-30x at chi = 48, 30x at chi = 64, 38x at chi = 96 (the 15x of the plan was one fired cut against an isolated two-site SVD on random tensors).

The corrected model ratio = (1 + c f) / r^p reproduces the data (chi = 64, eps_min 1e-5: c = 30, p = 1.4 gives 1.25, measured 1.16; chi = 96, eps_min 1e-6: c = 38, p = 1.7 gives 1.19, measured 1.18). It says wall-clock needs f of order 5 % **and** r >= 1.5 at once, and it improves with chi only because p drifts towards 3.

**T1 verdict.** Memory: r(nn) 1.56-1.78 and r(1-site) 1.51-1.59 at chi = 32-48 for all gates, 1.64-1.75 / 1.49-1.51 at chi = 64, 1.70-1.72 / 1.48-1.51 at chi = 96 for the gates that keep firing. Wall-clock: **not shown**. Best honest numbers are 1.16 / 1.33 (chi = 64) and 1.18 / 1.45 (chi = 96), above 1.0, inside the plan's "ambiguous" band. The single literal pass depends on the warm-gate artifact.

## 4. T2: does spcf beat SVD plus free post-processing on the verification horizon?

Runs: SVD at chi in {16, 24, 32, **36**, 48, 64, 96} (36 added so that t*_SVD(1.5 x 24) is measured, not interpolated); spcf at chi in {16, 24, 32} for four gates; samples every 0.5. All arms come from the same runs. Definitions fixed in the docstring of `t2_analysis.py` before any spcf data existed:

- t*(arm, tol, metric): first sample with rms error above tol (1-site or nn); inf if never within T = 8.
- (a) is read strictly: all 12 cases (3 chi x 2 metrics x 2 tolerances) must satisfy t*_spcf(chi) >= t*_SVD(1.5 chi); ties and double-censored cases (inf >= inf) count as holding.
- Growth phase for a given chi: samples up to the maximum of the SVD(chi) nn error, restricted to SVD error >= 1e-4.
- "Best post-processed SVD" per sample = minimum over SVD, SVD x F_MPS, log F extrapolation and 1/chi extrapolation, the last two taking the (up to) three largest SVD chi <= chi. This is generous to the competitors (the extrapolations cost up to three SVD runs).
- (b) must hold at every chi for both metrics; the kill reading of "within 20 %" is the fraction of growth samples with spcf error > 0.8 x best post-processed.
- (c): median over growth-phase samples of eps_c(spcf)/eps_c(SVD) at equal chi. eps_c = sqrt(sum_{i<j}(c_ij - c~_ij)^2 / sum c~_ij^2) over all 190 pairs; plain <ZZ> and connected give the same numbers to two digits.

"Best T1 gate" is ambiguous because T1 has two answers. I ran the gates for both readings, plus the ungated rule and one control:

| gate | meaning | (a) holds | failing (a) cases | (b) min / median win fraction | (c) median eps_c ratio, chi 16 / 24 / 32 | peak r(nn), chi 16 / 24 / 32 | r(nn) at t = 8 |
|---|---|---|---|---|---|---|---|
| g001: rs 1e-2, eps 1e-7, taus [1] | ungated | **12/12** | none | **0.88** / 1.00 | 0.32 / 0.30 / 0.30 | 2.12 / 1.99 / 1.95 | 1.39 / 1.71 / 1.68 |
| g06: rs 0.6, eps 1e-7, taus [1] | best of the as-specified T1 | 11/12 | chi 24, 1-site, 1e-3: 5.0 vs 5.5 | 0.88 / 1.00 | 0.31 / 0.30 / 0.34 | 2.00 / 1.91 / 1.78 | 1.32 / 1.58 / 1.56 |
| g03e5: rs 0.3, eps 1e-5, taus [1] | best of the cold T1 | **6/12** | all six tolerance-1e-3 cases (e.g. chi 32 nn: 5 vs 6) | **0.62** / 0.73 | 0.35 / 0.40 / 0.34 | 2.09 / 1.95 / 1.83 | 1.40 / 1.69 / 1.65 |
| s001: rs 1e-2, eps 1e-7, taus none | control: static targets only | 7/12 | five tolerance-1e-3 cases | 0.88 / 1.00 | 0.34 / 0.29 / 0.35 | 2.01 / 1.88 / 1.81 | 1.31 / 1.61 / 1.62 |

### 4.1 Verdict

- **Ungated spcf passes (a), (b), (c).** The margin on (a) is thin: tolerance 1e-2 is censored for chi >= 24 (neither arm leaves the window), and at tolerance 1e-3 the tests are ties at the 0.5 sampling step. Interpolated crossing times (log-error interpolation) for g001 at tolerance 1e-3 are within -0.3 to +0.5 of SVD(1.5 chi): chi = 16: 4.20 vs 4.28 (nn), 4.15 vs 4.43 (1-site); chi = 24: 5.21 vs 5.05, 5.02 vs 5.21; chi = 32: 6.03 vs 5.57, 5.78 vs 5.74. So the horizon gain is about 1.5 in chi, no more.
- **The cost-optimised gates give up the early-time horizon.** g03e5 fires only when the discarded weight exceeds 1e-5, which skips the early cuts that seed the 1e-3 error. Its nn error equals SVD's up to t = 4.5 (chi = 32: 4.66e-4 for both) and only then separates, while the ungated rule has 1.57e-4 at the same time (3x lower). At late times the two gates agree (chi = 32, t = 8: 4.65e-3 vs 4.83e-3).
- **No kill condition triggers for any gate.** The fraction of growth samples where spcf is within 20 % of the best post-processed SVD is 0.00-0.43 (kill needs "most"). eps_c is better, not worse: median ratio 0.29-0.40, final sample 0.54-0.81.
- Overall T2 as pre-registered ("best T1 gate"): **strict success is not reached for either candidate gate (11/12 and 6/12 on (a)); it is reached by the ungated rule.** The honest statement: spcf at equal chi is better than every free post-processing arm tried, and its horizon is about 1.5x chi equivalent, but a gate tuned for wall-clock removes the horizon benefit. Memory and wall-clock want different gates.

### 4.2 The competitors are weak here

nn rms error at t = 4.5 / 6.5 / 8.0 (chi = 16 / 24 / 32 in the three columns; extrapolations use the three largest SVD chi <= chi):

| arm | chi = 16 | chi = 24 | chi = 32 |
|---|---|---|---|
| SVD | 1.0e-2 / 2.9e-2 / 2.3e-2 | 2.1e-3 / 1.4e-2 / 1.5e-2 | 4.7e-4 / 8.6e-3 / 1.1e-2 |
| SVD x F_MPS (Mandrà) | 1.0e-2 / 2.9e-2 / 2.6e-2 | 2.1e-3 / 1.4e-2 / 1.6e-2 | 4.7e-4 / 8.7e-3 / 1.2e-2 |
| SVD x F_true (oracle, not available) | 1.2e-2 / 6.0e-2 / 9.9e-2 | 2.2e-3 / 2.5e-2 / 5.9e-2 | 4.7e-4 / 1.3e-2 / 3.6e-2 |
| SVD log F extrapolation (Anand) | same as SVD (one point) | 1.2e-3 / 9.1e-3 / 1.7e-2 | 7.0e-4 / 6.9e-3 / 1.2e-2 |
| SVD 1/chi extrapolation | same as SVD | 1.5e-2 / 2.3e-2 / 2.9e-2 | 1.1e-2 / 1.6e-2 / 1.8e-2 |
| spcf g001 | **1.7e-3 / 7.6e-3 / 1.6e-2** | **4.3e-4 / 3.2e-3 / 7.6e-3** | **1.6e-4 / 1.5e-3 / 4.6e-3** |
| spcf g06 | 1.7e-3 / 8.6e-3 / 1.7e-2 | 4.7e-4 / 3.6e-3 / 8.7e-3 | 1.6e-4 / 2.0e-3 / 5.4e-3 |
| spcf g03e5 | 3.8e-3 / 7.9e-3 / 1.6e-2 | 2.1e-3 / 3.4e-3 / 7.8e-3 | 4.7e-4 / 1.9e-3 / 4.8e-3 |
| spcf g001 x F_MPS | 1.8e-3 / 8.4e-3 / 1.9e-2 | 4.3e-4 / 3.6e-3 / 9.0e-3 | 1.6e-4 / 1.7e-3 / 5.1e-3 |

- **F_MPS is a poor fidelity estimator for Hamiltonian TEBD.** The product of per-cut truncation fidelities gives 1 - F_MPS = 0.030 at chi = 32, t = 8, against a true infidelity of 0.204 (ratio 4.4-8.1 over all runs; the product assumes independent errors, but truncation errors of a Trotter evolution add coherently). Rescaling by it does nothing, and rescaling by the *true* overlap (oracle) makes the nn error worse (3.6e-2 vs 1.1e-2 at chi = 32, t = 8) because the discarded component is not noise-like for local observables. The Mandrà and Anand arms are therefore not zero-overhead rivals in this regime. This is a statement about this cell and about gamma = 1, not about the methods in general (the F^gamma generalisation was not fitted).
- Log F extrapolation helps in places (chi = 32, t = 6.5: 6.9e-3 vs 8.6e-3) and hurts in others (chi = 24, t = 8: 1.7e-2 vs 1.5e-2). The literal "three largest chi" version (48, 64, 96) reaches an nn error of 1.2e-3 at t = 8, the same as SVD at chi = 96 (1.3e-3), so it buys nothing. The 1/chi extrapolation is worse than plain SVD at these chi.
- **spcf x F_MPS is worse than spcf alone** (the compound does not help, though it does not break either).

### 4.3 r along the trajectory (g001, from the nn error; `-` = both arms exact to rounding)

| chi | 3.5 | 4 | 4.5 | 5 | 5.5 | 6 | 6.5 | 7 | 7.5 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.11 | 1.37 | 1.56 | 1.79 | 2.06 | 2.04 | 2.12 | 1.91 | 1.77 | 1.39 |
| 24 | 1.00 | 1.21 | 1.35 | 1.56 | 1.74 | 1.90 | 1.99 | 1.87 | 1.87 | 1.71 |
| 32 | 1.00 | 1.00 | 1.15 | 1.36 | 1.59 | 1.81 | 1.95 | 1.89 | 1.83 | 1.68 |

r is 1 until the SVD error becomes visible, climbs to about 2 in the growth phase and falls back to 1.4-1.7 at T = 8 (so the "r about 1.6-1.7 at T = 8" of the plan is right for chi = 24-32, lower for chi = 16). From eps_c instead of nn the numbers are lower: peak 1.7-1.85, T = 8: 1.21 / 1.36 / 1.46 (chi = 16 / 24 / 32). Memory saving r^2: about 2-4x in the growth phase, 1.9-2.9x at T = 8 (nn), 1.5-2.1x (eps_c).

## 5. T3: transfer to the 1D kicked Ising circuit

Setup as in the plan: N = 20, depth <= 20, start |up>^N, one sweep per Floquet step with alternating direction (right sweep gate_b = ZZ_{b,b+1}(thJ) RX_{b+1}(thh), RX_0 folded into b = 0; left sweep mirrored), dense reference with the identical gate list, spcf with `taus = ()` (static window targets). SVD ladder chi in {6, 8, 12, 16, 24, 32, 48, 64}; spcf chi in {8, 12, 16, 24}; two gates (g001 and g03e5), identical conclusions. Settings: A = (thJ -pi/2, thh 0.7), B = (-pi/2, 1.0), C = (-pi/4, 0.7), D = (-pi/2, pi/2) Clifford control.

Pre-registered reading (docstring of `t3_analysis.py`, written before any T3 data): a setting passes if the median over chi_ref in {8, 12, 16, 24} (SVD horizon of |dZ_{N/2}| inside depth <= 20) of r(1-site rms) at the horizon is >= 1.5 and the median eps_c ratio is <= 1.2. r is read from the SVD ladder on the 1-site rms because the single-observable |dZ_mid| is not monotone in chi and cannot be inverted (for example in A at depth 18: svd32 7.0e-3, svd48 1.1e-2).

Gate g001 (g03e5 gives the same r(1-site) to within 0.01 for A, B and C; the fired fraction is lower, 0.18-0.33):

| setting | chi_ref | SVD horizon d*(Z_mid, 1e-2) | r(1-site) at d* | r(nn) at d* | median eps_c ratio | fired fraction |
|---|---|---|---|---|---|---|
| A | 8 / 12 / 16 / 24 | 8 / 10 / 13 / 14 | 0.96 / 1.04 / 0.98 / 1.03 | 0.93 / 1.07 / 1.05 / 0.98 | 0.81 / 0.96 / 0.98 / 1.01 | 0.58 / 0.50 / 0.46 / 0.36 |
| B | 8 / 12 / 16 / 24 | 5 / 6 / 7 / 10 | 1.00 / 0.99 / 0.94 / 1.00 | 1.02 / 1.00 / 0.96 / 0.96 | 0.91 / 1.04 / 1.05 / 1.08 | 0.57 / 0.52 / 0.46 / 0.41 |
| C | 8 / 12 / 16 / 24 | 8 / 10 / 11 / 18 | 1.09 / 1.04 / 1.10 / 0.80 | 1.20 / 1.28 / 1.04 / 1.00 | 0.89 / 0.96 / 0.93 / 1.05 | 0.55 / 0.47 / 0.40 / 0.34 |
| D (control) | 8 / 12 / 16 / 24 | 5 / 4 / 10 / 6 | 0.90 / 1.00 / 0.76 / 0.88 | 1.59 / 1.00 / 0.98 / 1.11 | 0.98 / 0.91 / 1.03 / 0.80 | 0.25 / 0.32 / 0.13 / 0.17 |

Medians: A 1.00, B 0.99, C 1.07 (control D 0.89). Over all depths >= 4 and all four spcf chi, r(1-site) lies in 0.77-1.25 (A, median 1.03), 0.81-1.27 (B, median 0.99) and 0.78-1.43 (C, median 1.00); the largest values are at chi = 8 and late depth, where SVD itself is far from converged. The error ratio spcf/SVD of <Z_mid> at the horizon scatters between 0.04 and 1.7 with no direction. The infidelity and eps_c also coincide with SVD's.

**T3 verdict: killed** (median r < 1.3 in all three ergodic settings; the criterion for success, r >= 1.5 in two of three, is missed by about 0.5). The Clifford control D shows the expected flat spectrum: the SVD error is at rounding level (1e-16) until the Schmidt rank exceeds chi and then jumps to 1e-2-1e-1 (first non-zero depth: chi = 6: 3; 8: 4; 12: 4; 16: 5; 24: 5; 32: 6; 64: 7, i.e. one more depth per doubling of chi). spcf gains nothing there either, and its r values are noise.

### 5.1 Is the failure the missing lookahead, or the circuit? Two checks

1. **Control on the continuous quench, static targets only** (`s001`, `taus = ()`, N = 20, T = 8, same code path): r(nn) peaks at 2.01 / 1.88 / 1.81 and is 1.31 / 1.61 / 1.62 at T = 8 for chi = 16 / 24 / 32, against 2.12 / 1.99 / 1.95 and 1.39 / 1.71 / 1.68 with the lookahead tau = 1. Static targets keep most of the gain (but lose a part of the horizon: (a) 7/12 vs 12/12). So the kicked-circuit null result is **not** explained by `taus = ()` alone.
2. **Diagnostic of the tilt itself** (`t3_diag.py`, residual of the static window objective after the tilt divided by before, per fired cut):

| run | fired / calls | residual after / before (median) | discarded weight of SVD at the cut (median) | extra discarded weight / SVD tail (median) |
|---|---|---|---|---|
| kicked A, chi = 16, depth 14 | 110 / 266 | 0.40 | 3.7e-4 | 0.08 |
| kicked B, chi = 16, depth 14 | 110 / 266 | 0.37 | 4.0e-3 | 0.16 |
| kicked B, chi = 24, depth 14 | 102 / 266 | 0.38 | 1.8e-3 | 0.11 |
| quench, chi = 16, T = 5, static | 450 / 1900 | **0.15** | **1.1e-5** | 0.23 |
| quench, chi = 24, T = 5, static | 318 / 1900 | 0.14 | 2.9e-6 | 0.11 |

On the kicked circuit the tilt does fire (32-58 % of the cuts) and reduces its own objective by 2.5x, but the per-cut discarded weight is roughly 30-1000x larger than in the small-step quench (1e-4 to 4e-3 against 1e-5 to 1e-6), and the objective is only a first-order expansion around the SVD point. My reading, **not tested beyond this diagnostic**: the large per-cut truncation of a big-kick Floquet step is outside the regime where one Gauss-Newton step from the SVD point improves the observables, whereas in a small-step Trotter evolution each cut is a small perturbation. A direct test would be a Trotterised kicked chain with small angles (n sub-steps per period) that interpolates between the two regimes; it was not run.

## 6. Deviations from the plan and judgement calls

1. **T1**: run exactly as specified, then supplemented. Added: (i) an SVD timing ladder before and after the scan; (ii) the cold-gate bench `t1_cold.py`, because `pareto_spcf.py` reuses one `SPCFast` for the error run and the timing runs (section 3.2); (iii) the chi = 64 re-measure the decision tree asks for; (iv) chi = 96 and the eps_min 1e-6 / 1e-7 gates at rs = 0.3, which are **outside the pre-registered grid** and are reported as an extension only. The T1 verdict does not use them.
2. **T1 scoring choices (mine)**: r from both nn and 1-site; success requires both r >= 1.5 and both ratios <= 1.0. These were fixed after I had seen the as-specified table and before I had seen the cold-gate numbers.
3. **T2 "best T1 gate"**: the plan does not say which T1 metric defines "best". I used the lowest equal-error ratio among settings with r(nn, chi = 32) >= 1.5, which picks rs = 0.6 from the as-specified table and (rs 0.3, eps_min 1e-5) from the cold table. Both were run, plus the ungated rule and the `taus = none` control (which are additions).
4. **T2 operationalisation**: the growth-phase definition, the strict reading of (a), the "within 20 %" fraction and the generous competitor set (extrapolations at chi use the three largest SVD chi <= chi) are my choices, written into `t2_analysis.py` before the spcf data existed. SVD chi = 36 was added to the ladder. Two oracle/extra arms were added (SVD x true overlap; spcf x F_MPS). The extrapolation guard "use the largest-chi run if the F values are indistinguishable" only matters at times where all errors are at rounding level.
5. **T3 started although T2 is not strictly a pass for the T1-selected gates** (section 1). The chi ladders (6-64 SVD, 8-24 spcf) and the sampling at every depth are my choices; the plan gave none.
6. **N = 20 kept** everywhere. The dense references (16 states x 16.8 MB for T2, 20 x 16.8 MB per setting for T3) are cached under `rank7/_refcache/` (git-ignored, 1.5 GB in total).
7. Early drafts of the r(t) table printed meaningless values (0.5-0.7) at times where both errors are rounding noise; these entries are now masked (`-`) below 1e-7. The first selftest of `ladder_t.py` at T = 1 was vacuous (nothing truncated) and was rerun at T = 3, chi = 4 before use.

## 7. Caveats

- **Timing**: 4 cores shared with about 6 other agents; load average 3.5-23. CPU time is used, SVD and spcf are timed interleaved in one process, and the ladder reproduced within 1-5 % between start and end, but absolute inflation by memory/SMT contention is unknown. Equal-error ratios carry +-10 %. Ratios near 1.0-1.2 (chi = 64, 96) cannot be called against 1.0 with this noise.
- One cell, one model, one seed-free initial state, N = 20 only. r constancy at N > 20 is untested (the plan lists it as open).
- t* is quantised at 0.5; tolerance 1e-2 is censored for chi >= 24, so only the 1e-3 tolerance discriminates.
- The F_MPS and extrapolation arms are as in the plan (gamma = 1); a fitted gamma or a better fidelity model could make them stronger competitors.
- The interpretation of the T3 mechanism (section 5.1) is a hypothesis supported by one diagnostic and one control.

## 8. Recommended next step

1. **Do not pursue the circuit-benchmark transfer further in this form** (T3 killed, with a control that points at the circuit, not at the missing lookahead). Keep one cheap bridging experiment if the circuit question matters: a Trotterised kicked chain with angles of order 0.1-0.3 and many sub-steps per period, using `floquet.py` (gate lists only), to find the step size where r goes from about 1.7 to 1.0. About 30 minutes.
2. **Position rank 7 as a memory/verification claim for small-step continuous-time quenches**: r about 1.5-2.1 over the growth phase (memory 2-4x), 1.4-1.7 at T = 8, eps_c about 3x lower at equal chi, free post-processing does not match it. This part of the plan's thesis survives, with the caveat that the horizon gain is about 1.5x in chi.
3. **If wall-clock matters, the lever is not rel_skip and not an absolute eps_min.** The data say it needs (a) a gate that scales with chi (a relative threshold on the tail compared with the typical tail at that chi), and (b) a much cheaper fired cut: the whitened closed form of the main report, targeting roughly 2x SVD-TEBD cost instead of the measured 25-38x. With the corrected model, f about 5 % and r about 1.6 give a ratio about 1.2 already with the present cost; a fired cut at 2-3x would give roughly 0.5-0.6 at f = 0.05-0.1 (model, not measured). That test (chi = 64-128, cold gate, equal-error SVD timed directly) is the decisive wall-clock experiment.
4. Before quoting r in any note, state which of 1-site, nn or eps_c defines it (1.5, 1.7 and 1.2-1.4 at T = 8 respectively for chi = 32), and report timing only with the gate state of the error run.
