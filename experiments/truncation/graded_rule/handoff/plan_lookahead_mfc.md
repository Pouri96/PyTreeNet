# Lookahead marginal-fit compression: from a dense proof of concept to a decisive win

Supersedes the earlier "graded rule" programme (its Phase 0 rehoming is done; its Phases 1-3 are
replaced by this plan). Everything lives in
`PTN trunc_research/experiments/truncation/graded_rule/`. Nothing is committed.

## Context

Mandate: find a truncation method with a decisive win at high truncation pressure, ideally on every
model, at MATCHED BOND DIMENSION (final-state infidelity and observable error against stored
parameters, never by target, pooled arms with a stated margin). Method and truncation rule are free.

What is established (dense, N=12, high pressure, fit/SVD error ratio, lower is better):

| cell | chi | infid | 1-site | 2-site | E |
|---|---|---|---|---|---|
| Ising T=4, lookahead {0,5,10} | 8 | 1.02 | 0.09 | 0.13 | 0.05 |
| Ising T=4, lookahead 0..20 (every step) | 8 | 1.21 | **0.05** | **0.08** | 0.12 |
| ising2 T=4, {0,5,10} | 8 | 1.05 | 0.26 | 0.32 | 0.05 |
| domain wall T=4, {0,5,10} | 8 | 1.03 | 0.14 | 0.22 | 0.11 |
| domain wall T=5, {0,5,10} | 8 / 10 | 1.05 / 1.03 | 0.30 / 0.25 | 0.43 / 0.38 | 0.11 / 0.08 |
| Heisenberg T=1.5, {0,5,10} | 12 / 16 | 1.07 / 1.12 | 0.17 / 0.56 | 0.37 / 0.37 | 0.13 / 0.13 |

Static ceiling (best rank-chi MPS for all 1-3-site marginals, Ising N=12 T=4): 700x at chi=8, 2500x
at chi=10, so capacity is not what limits the dynamic gain.

Mechanism chain: window fitting at a single step is undone within ~10 steps (the "leak": the fitted
state matches marginals but the dynamics also need correlations outside the window). Fitting the
marginals of U^m v, m in a set, is the same as fitting the expectation of the Heisenberg-evolved
operators (U†)^m P U^m, which are state independent. The gain grows with the density and length of
the set (2-site: no lookahead ~3-5x, {0,5,10} 7.7x, 0..20 12.5x). Energy error falls 8-20x as a
by-product with no energy term (the energy claim itself belongs to `spc_mps`).

What is NOT established, and what this plan resolves:
1. Dense N<=14 only. The prototype forms the full state vector, so it cannot run beyond N~18.
2. Cost is ~100x SVD wall in the prototype. Not the headline (user rule), but it has to be bounded.
3. Ising N=14 T=6 result file (`out_sf10_ising14_T6.txt`) is empty, so there is no N-scaling point.
4. The gain on ising2 / domain wall / Heisenberg is 2-7x, on Ising 8-12x. "Decisive on every model"
   is not met.
5. Prior art has not been checked (Heisenberg-picture and operator-based compression exist in the
   literature, see Stage 4).

## Status log (2026-10-06, updated as stages complete)

- Stage 1 (dense): done at N=12 (N=14 T=6 chi=8/12 gave .44/.44 and .21/.27). Gain depends on SVD infidelity (pressure): Ising .09-.13 at infid .06, ising2 .14/.20 at chi=12 (infid .16) but .26/.32 at chi=8 (infid .36), DW .22/.30, Heis .37. Passes "3 of 4 models" only at moderate pressure.
- Stage 2 (native): `rule/lookahead.py` (MPO bank, validated 1e-10), `rule/region.py` (local-region lookahead, exact local dynamics, any horizon). Key finding: the lookahead propagator need not be the dynamics (3-layer brickwork matches it). Native N=12 chi=8 rdm2 ratios vs dsvd:2: Ising .18-.25, DW .31, ising2 .41, Heis(chi=12) .43. So native is ~1.5-2x weaker than dense {0,5,10}. Stage-2 gate (native within 10% of dense) NOT met.
- Dead ends recorded in memory 12g/12h: Taylor jets, DAOE damping, local-sweep optimiser, more L-BFGS iterations (250 -> 800 no change), larger working rank (no change).
- Open: N=20 native result (running), region-lookahead grid (richer windows/times), pooled table with margins, prior-art deepening.

## Definition of "decisive win" (fixed before running, used as the pass line)

At matched chi, N>=14, SVD front unconverged (infidelity 1e-1 .. 1e-3 across the chi ladder), on at
least three of {ising, ising2, isingdw, heis}: pooled 2-site marginal error <= 0.25x SVD (4x) at the
10% margin, 1-site <= 0.25x, final-state infidelity <= 1.15x SVD, and the same chi stored. Also
beating 1-site TDVP at the same chi. Anything less is reported as a modest result, not a win.

## Stage 1 - dense capacity wall (N=12-14, JAX, no new library code)

Reuse `probe_stepfit.py` (args model N T chis maxiter fw kmax pair_weight lookahead, env `LASET`).

1a. Re-run Ising N=14 T=6 chi=8,12 with `LASET=0,5,10` and with a denser set, logging per step so a
    dead job is visible. Gives the N-scaling point 12 -> 14.
1b. Lookahead-set ablation on Ising N=12 chi=8: sets {0,5,10}, {0,2,4,..,20}, 0..20, 0..40, and a
    horizon that shrinks near the final step (no lookahead beyond the remaining steps). Settles
    whether the optimum is "all steps up to the correlation time" and what the cheapest sufficient set is.
1c. Per-model high-pressure cells with the densest affordable set: ising2, isingdw, heis (T chosen so
    the SVD front spans a decade, N=12 and 14).
1d. Weights: fidelity weight fw in {0.01, 0.03, 0.1}, per-m weights (decay in m), kmax 2 vs 3, so the
    infidelity ratio stays <= 1.15.

Gate to Stage 2: pooled 2-site <= 0.25x on at least three models at N=12-14 under the margin rule.
If not, the wall is "gain is model limited at ~2-4x", and the result is written up as a modest
method with the leak mechanism as the finding.

## Stage 2 - MPS-native lookahead (removes the dense state vector)

2a. Measure first (cheap, decides feasibility): `rule/heis_ops.py` MPO bond dimensions for 1- and
    2-site Pauli strings, m up to 20, eps 1e-6..1e-9, N=14-24, all four models (it already
    reproduces dense to 1e-13). The explosion seen so far was at tight eps; report bond vs eps vs the
    resulting marginal error.
2b. Candidate-side objective by MPO expectation (transfer matrices, cost ~chi^2 D per site, JAX) in
    place of dense marginals; target side from the untruncated working-rank step result evolved
    exactly m steps with a larger auxiliary rank. Same objective, so the dense probe is the unit test:
    native and dense results must agree within 10% on N=12.
2c. If MPO ranks are too large, fall back to window-restricted coarse-Trotter propagators (brickwork,
    dt_c = 0.2-0.5, backward light cone of 1-2-site operators), test first in the dense probe that
    swapping U for U_c in the targets keeps the gain.
2d. Cost: analytic gradient or cached environments, one L-BFGS call per step, report wall as a
    property of the prototype.

Gate: native N=14 matches dense within 10% on two models.

## Stage 3 - scale and baselines (the paper-grade table)

`mfc_bench.py`-style harness (add arm `mfcl:f:fw:k:iters:set`), N=16, 20, 24, models ising, ising2,
isingdw, heis at high pressure. Arms: svd, dsvd (delayed SVD, same working-rank transient), 1-site
TDVP (`tdvp_bench.py`), mfc (no lookahead), mfcl (lookahead). Cached references for N<=20 (dense),
N>20 against a high-chi reference with a stated Trotter floor (`exact_ref.py`). Report ratios
vs matched-chi SVD, margin ledger 1/2/5/10%, pooled over arms; cells whose SVD front reaches its
floor inside the chi ladder are rejected.

## Stage 4 - prior art (read-only, before any claim)

arXiv searches: Heisenberg-picture / operator-evolution truncation (dissipation-assisted operator
evolution, Pauli propagation, hybrid Schroedinger-Heisenberg and light-cone tensor networks),
observable-aware or reduced-density-matrix-fitting MPS compression, "meet in the middle" local
observable evaluation. Read with `mcp__arxiv__download_paper` (text). Flag any claim not in a source
as my inference. If the objective is already published, the contribution narrows to the leak
diagnosis and the cross-model measurement, and the paper says so.

## Stage 5 - housekeeping

- Memory: add section 12g to `project_graded_trunc_window_form.md` (M=20, DW chi=8, energy by-product,
  the pass line) and rewrite the stale `MEMORY.md` index line (it still says "correlators up to 2.2x").
- Copy result JSON and PNG figures into the repo (`*.pdf` is gitignored). No commits without permission.
- JAX lives only in the scratchpad `pylibs` (env `PYLIBS`); never install into the project venv.

## Wall (when to stop and report)

Stop at the first of: (i) Stage 1 gate fails (capacity wall, ~2-4x on three of four models);
(ii) Stage 2 gate fails with both MPO and coarse-Trotter targets (scalability wall); (iii) Stage 3
misses the pass line at N>=16 (size wall); (iv) Stage 4 finds the same objective published (novelty
wall); or the pass line is met (then Stage 3 becomes the paper table).

## Verification

Unit equivalence: native vs dense objective values to 1e-8 on a random N=8 state. End-to-end: kappa=0
analogue is arm `dsvd` (reproduces plain delayed SVD exactly). Matched chi asserted from
`stored_params` for every row. Reference cache keyed by (model, N, T, dt), checked against the
continuous exact solution for the Trotter floor.
