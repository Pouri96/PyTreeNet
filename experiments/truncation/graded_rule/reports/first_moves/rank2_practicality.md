# Rank 2: can the purification spcf cut be made practical?

STATUS: pre-registration written before any run of the new experiments (2026-10-08 20:45 UTC). Results are added below the line "RESULTS" later in this file, the pre-registration text is not edited after the runs.

Question: spcf on a purification MPS beats SVD on local error at equal chi (2.5 to 4.7x lower rdm2, window a = 2) but costs about 10x SVD in time at equal error and its cut needs far more peak memory than an SVD cut (the window state W has about 4096 chi^2 numbers at a = 2). Judge by PEAK memory during the run and by TIME to reach the final state, not by stored parameters. Two approaches, in this order: (1) window a = 1, (2) a chunked window state. Code for the new work is in `rank2/` only, no existing file is modified, nothing is committed.

## Pre-registration

### P1. Approach 1, window a = 1 (accuracy)

Cells: staggered and domain wall, mu in {1, 0.5, 0.25}, Ising, N = 10, dt = 0.1, the gauge Test 2 used for spcf (plain at mu = 1 and 0.5, backward at mu = 0.25), chi in {12, 16, 24}: 18 (cell, chi) entries. Primary time T = 4, secondary T = 3 (same trajectories, sampled at t = 3).

Arms: SVD in the same gauge as spcf (existing `eqerr_raw_*.json`, `step2*.json`, `scan_svd_*.json`, `test2_*.json`), SVD in the other gauge (existing), spcf a = 2 (existing, not recomputed), spcf a = 1 (new; option string `1:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2`, i.e. `pur_bench.DEFAULT_OPT` with the first field 1). Metrics at the final time: rdm2, nn_rms, nnn_rms, zzfar, trace distance, purification infidelity, |dE|.

Ratios: R_a = metric(spcf window a) / metric(SVD, same gauge), per entry. Lower is better, R < 1 means spcf wins.

"a = 1 keeps the gain" iff all three hold at T = 4 on the pooled median over the 18 entries:
- (C1) median R_1(rdm2) <= 1.5 x median R_2(rdm2) (a = 1 is at most 1.5x worse than a = 2 relative to SVD);
- (C2) median R_1(rdm2) <= 1/1.5 = 0.667 (a = 1 is at least 1.5x better than SVD);
- (C3) median over the 18 entries of zzfar(a = 1) / zzfar(SVD, same gauge) <= 1.5.

Supplementary, not decisive: the same three conditions per cell (median over its 3 chi), reported as a count out of 6; the same at T = 3; R for nn, nnn, trace distance, |dE|, infidelity.

Overfit check (a = 1 overfit in pure states: residual down, RDM error up): for every entry record the median over fired cuts of f_c / f_svd (region objective after / before) and the count of entries with R_1(rdm2) > 1. Overfit signature = median f_c/f_svd <= 0.5 together with R_1(rdm2) > 1.

Decision rule for the dependent work: if the pooled median R_1(rdm2) > 0.8 (gain smaller than 1.25x) then a = 1 is declared dead, and the a = 1 Pareto points and the chunked a = 1 are skipped. If C1 or C2 fails but the median R_1 <= 0.8 the outcome is "partial": the a = 1 Pareto is still run and the verdict says it does not meet the bar.

### P2. Approach 2, chunked window state

New subclass `rank2/spcfpur_chunk.py` (`SPCFPurChunk`), the cut never builds the full W: rho = W W^dag, the Jacobian-vector product and its adjoint are accumulated block by block over the outer index o' of the left environment map (block size `blk` is a parameter).

Validation (float64, `precision='f64'`) on random cells (random MPS, random two-site tensor, a in {1, 2}, several bonds including the chain edges, block sizes including ones that do not divide the outer dimension): relative difference chunked vs unchunked <= 1e-10 for the region RDM, the initial residual, jvp, vjp, and for the tensors returned by one full fired cut (<= 1e-9, a CG solve amplifies rounding); the adjoint identity <g, J x> = <J^T g, x> to <= 1e-10 for the chunked class. Float32 (the production precision): jvp/vjp relative difference <= 1e-4, cut tensors <= 1e-3, same accepted line-search step.

Cost measurements (cost_scaling setup: N = 16, bulk bond, random purification MPS with TEBD-like spectrum, truncation 4 chi -> chi, options of Tests 1 and 2 with every skip rule off): per-cut median time and per-cut transient peak memory (tracemalloc, with a process-level cross-check) for chi in {16, 32, 48, 64, 96, 128}, a in {1, 2}, several block sizes, against the SVD cut. Scaling exponents from a log-log least-squares fit over chi >= 32.

Criterion for "chunking works" (memory side only): at chi = 64, a = 2, the best block size has transient peak memory <= 3x the SVD cut's (unchunked: about 100x) with per-cut time <= 2x the unchunked time.

### P3. Pareto at the final state and the verdict

Same 6 cells, T = 4 only. Variants entering the plots: spcf a = 1 (if not dead), chunked a = 2 (if it passes P2), chunked a = 1 (if a = 1 not dead). Baselines: SVD plain gauge, SVD backward gauge (chi ladder 8 to 96), DMT (`dmt_np.py`, chi ladder 12 to 96). Error = final rdm2. Cost axes:
- (i) peak memory over the whole run = stored MPS + transient inside every cut + method-constant tables (spcf's state independent F matrices), measured as the process resident-memory high-water mark growth over the run (VmHWM reset just before the run, one fresh process per run), cross-checked with tracemalloc (decomposition into MPS / constants / transient) on all spcf points and a subset of baselines. N = 10, so the outer bonds of the a = 2 window are capped (at most 16) and W does not grow with chi in these runs; the growth with chi is therefore additionally measured per cut at N = 16 (P2) and used for a labelled projection;
- (ii) total CPU time to reach T (`time.process_time`, TEBD only, scoring excluded; 3 single-threaded workers, as in `eqerr_time.py`; the existing baseline times are reused).

A spcf point is ON the front for an axis iff no baseline point has cost <= and error <= its own (strict in one). The equal-error factor = cost of the spcf point / cost of the baseline lower envelope (non-dominated baseline points, log-log interpolation) at the spcf point's error; below 1 spcf is cheaper. A variant is "on the front in a cell" iff at least one of its points is on the front. Count over the 6 cells, separately for memory and for time.

"Practical" iff the best variant is on the front for both axes in at least 4 of 6 cells. "Memory-only" iff on the memory front in at least 4 of 6 cells and not on the time front in at least 4 of 6. Otherwise "not practical", stated as such.

Whether a = 1 or chunking changes "rank 2 is an accuracy-at-fixed-chi result only" is answered by the front counts and the equal-error factors, with ladder-edge cases flagged as bounds.

## RESULTS

(to be filled in)
