---
name: project_graded_trunc_window_form
description: "The graded pure-state truncation rule, improved 2026-10-04: its objective IS a window reduced-density-matrix distance (grading = depolarising filter), rho replaced by a trust region kappa with a guarantee, and the Heisenberg failure shown to be mostly a REGIME artefact."
metadata:
  node_type: memory
  type: project
  originSessionId: 70010d88-054e-446c-80e5-5eb83ce6463b
  modified: 2026-10-07T01:06:42.306Z
---

**2026-10-04, continuation of [[project_pure_state_dmt_closed]].** Asked to investigate the graded
truncation rule further, starting from its Heisenberg failure. Four results, two of them structural.

## 1 The Heisenberg failure was mostly a REGIME artefact, not the model

The old Heisenberg grid (N=10, T=3, Neel) spans SVD infidelity **0.13 to 0.93** at chi=4..16, i.e.
the state is barely represented, while every Ising setting the rule wins on reaches 1e-6..1e-2. At
**T=0.8** the Heisenberg SVD front covers the same decades (0.11 down to 1.7e-6 at chi=6..24), so
the comparison becomes apples to apples. Same-chi paired error ratio vs plain SVD, median over 12
chi, `grH:3:4:100`:

| grid | infid | single | nn | nnn | energy |
|---|---|---|---|---|---|
| OLD Heis T=3 (6 chi) | 1.18 [0\|1\|5] | 1.17 | 0.89 | 1.30 | 0.00 |
| Heis T=0.8 (12 chi) | **1.03 [0\|6\|6]** | 1.11 | 0.82 | **0.51 [12\|0\|0]** | 0.00 |

So "fidelity 18% worse, 0 of 6 better" becomes near-tied fidelity with nnn correlators 49% better
in 12 of 12. **LESSON: always check that the incumbent's front covers comparable error decades
before declaring a model a failure.** The genuine residual weakness is SINGLE-SITE observables.

## 2 Three causes ruled out for the single-site weakness

- **U(1) symmetry: NOT the cause.** Measured weight outside the initial total-Sz sector after the
  full run = **7.5e-14**. The rule preserves the charge sector exactly, because `P_U`, `P_V`, `P_B`
  are charge-block structured so the Gram matrix `J` is charge graded and magnetisation-changing
  strings (target and residual both identically 0) get zero multiplier.
- **Target contamination: NOT the cause.** An ORACLE arm taking targets from the EXACT state
  instead of the pre-truncation approximate state barely helps (single 1.15 vs 1.22, nn 0.69 vs
  0.78, nnn 0.49 vs 0.54). So the limit is geometric (the price of the displacement), not
  informational. Exact targets cannot make single-site observables beat SVD here.
- **"SVD truncation is partly corrective": REFUTED.** Per-step error ratio across the truncation
  sweep is >1 in every class and every setting (`exp12_drift.py`), never corrective.

What the drift diagnostic DOES show, and it explains the class pattern: one sweep multiplies the
nn error by **1.4-4.9x** and the nnn error by 1.4-3.3x, but the single-site error only by
**1.01-2.0x** (1.010 at Ising chi=8). The rule is a drift SUPPRESSOR, so it helps exactly the
classes truncation damages most, while every class pays the displacement cost.

## 3 STRUCTURAL FIX: rho was the wrong knob, replace it by a trust region (`exp9_tr.py`)

Diagnostic (`exp8_diag.py`): with the fixed absolute weight rho=3 the overspend
`||M-M_svd||^2 / eps` ranges over **6e-7 .. 1.3e-1 inside one Ising run** and is 10x larger in
median in Heisenberg (9.6e-2 vs 9.5e-3). The price was uncontrolled and model dependent.
Reformulated with `eps = sum_{i>chi} s_i^2`:

    minimise sum_s w_s (target_s - <P_s>_M)^2   subject to   ||M - M_svd||^2 <= kappa * eps

so the extra fidelity cost is at most `kappa*eps` and the total truncation error at most
`(1+kappa)*eps`. The KKT system is `(2 J W J + lam J) mu = 2 J W r - lam c`, which is **exactly the
old system with lam = 1/rho**, so the old rule is this family at a fixed multiplier and the only
change is to solve for the multiplier that saturates the budget (bisection on lam, ~36 K-by-K
solves, negligible). **The weights now set only the DIRECTION and kappa alone sets the size.**
Verified: kappa=0 is bit-identical to plain SVD, achieved/(kappa*eps) = 1.0000 at every kappa with
no breach, residual monotone decreasing in kappa (0.94 -> 0.043 for kappa 1e-3 -> 1).
kappa behaves as a budget should -- Heis single 1.22 at kappa=0.1 and 2.30 at kappa=0.3 -- which is
how the SPEND was identified as the mechanism behind the single-site weakness.
MY BUG, fixed: the feasible/infeasible ends of the lam bracket were assigned backwards, which
silently collapsed every step below kappa=1 to zero. A verify harness caught it only after a
SECOND fix -- it was itself mis-specified, feeding a chi=10 state into a chi=6 cut so the nesting
`ran(L_b) inside ran(L_{b-1}) x C^2` did not hold on the input. Prepare test states with the SAME
chi and sweep order the rule runs in.

## 4 STRUCTURAL FIX: the objective IS a window reduced-density-matrix distance (the COST result)

Proved and verified to **1e-16 at gamma = 1, 2, 4, 16** (`verify_window.py`). Since
`Tr[P_s P_t] = 2^n delta_st`, for any Hermitian `A` on an n-site window
`sum_s (Tr[P_s A])^2 = 2^n ||A||_F^2`, and the grading `w_s = gamma^-(n_s-1)` is exactly a
single-site DEPOLARISING map `D_lam`, PTM `diag(1,lam,lam,lam)`, `lam = gamma^-1/2`, applied to
each site of the window. With the exp4 protected set being (strings on `[b-2,b]`) union (strings on
`[b-1,b+1]`), inclusion-exclusion over the overlap gives

    sum_s w_s (tau_s - <P_s>)^2 = gamma * sum_W sign_W 2^{n_W} || D_lam(rho_W - tau_W) ||_F^2

with signs +,+,- over `[b-2,b]`, `[b-1,b+1]`, `[b-1,b]`. Consequences:

- The 111-string enumeration is UNNECESSARY. One 4-site reduced density matrix per bond, `O(chi^3)`
  in an MPS, the same order as the SVD itself.
- **The Pauli-weight grading IS a depolarising filter on the window** -- the rTEBD weight gauge
  (arXiv:2412.08730) in closed local form on a PURE state, which is the open door
  [[project_pure_state_dmt_closed]] recorded. See [[project_rtebd_reweighting]].
- The gradient is ONE vector, `(O - <O>)|psi>` with
  `O = sum_W sign_W (-gamma 2^{n_W} D_{1/gamma}(rho_W - tau_W)) (x) I`, i.e. three small window
  operators applied to the state, not 111 global ones.
- A step confined to that single direction keeps **39-92% (median ~67%)** of the full
  111-gradient residual reduction at the same budget, over 7 settings (`exp10_subspace.py`); three
  window directions are NOT systematically better than one.

## 4b HEAD TO HEAD on the headline setting -- the trust region does NOT buy accuracy

Ising N=10 T=3, same-chi error ratio vs plain SVD, median over the 9 chi common to both grids
(`v2_main.json` for fixed rho, `v4_tr_ising.json` for the trust region, `v5_cheap_ising.json`):

| arm | infid | single | nn | nnn | energy | worse on infid |
|---|---|---|---|---|---|---|
| fixed rho `grH:3:4:100` | 0.990 | **0.565** | 0.489 | 0.617 | 0.000 | **0 of 9** |
| trust region `trH:0.1:4:100` | 1.001 | 0.622 | **0.465** | **0.612** | 0.000 | 2 of 9 |
| trust region `trH:0.3:4:100` | 1.066 | 0.545 | 0.465 | 0.715 | 0.000 | 6 of 9 |
| cheap `w:0.1:4:100` | 1.001 | 0.785 | 0.654 | 0.885 | 0.042 | **0 of 9** |

**So the trust region is an accuracy TIE with the tuned fixed rho on Ising, slightly worse on
fidelity, and kappa=0.3 is clearly worse.** Its value is not better numbers. It is (i) a
GUARANTEE, total truncation error <= (1+kappa)*eps by construction rather than an empirical
observation, (ii) a dimensionless knob that transfers across models where rho=3 produced a 10x
different spend in Heisenberg, and (iii) it is what made the Heisenberg diagnosis possible, since
the monotone kappa dependence identified the SPEND as the mechanism. Do not claim it improves
accuracy.

## 5 The cheap rule end to end (`exp11_cheap.py`), same-chi paired vs plain SVD

Window RDMs only, trust region, energy carried as a second direction. kappa=0 == plain SVD exactly,
objective matches the string form to 12 digits.

| setting | arm | infid | single | nn | nnn | energy |
|---|---|---|---|---|---|---|
| Ising N=10 T=3 (12 chi) | w:0.1:4:100 | **1.00 [2\|10\|0]** | 0.81 [8\|0\|4] | **0.66 [12\|0\|0]** | 0.83 [12\|0\|0] | 0.04 [12\|0\|0] |
| Ising, kappa=0.3 | w:0.3:4:100 | 1.00 [1\|11\|0] | 0.78 [8\|0\|4] | 0.59 [12\|0\|0] | 0.81 | 0.05 |
| Heis N=10 T=0.8 (12 chi) | w:0.1:4:100 | **1.00 [0\|9\|3]** | 1.32 | 1.09 | 1.02 | 0.04 [12\|0\|0] |
| **Ising N=12 T=3 (chi 4-7)** | w:0.1:4:100 | 0.91-1.00 | 0.49-0.72 | 0.48-0.71 | 0.72-0.92 | 0.007-0.11 |

N=12 replicates N=10 cleanly on the 4 chi scored (fidelity never worse, single and nn 30-50%
better, energy 10-140x better), which partly closes the larger-N gap.

**The cheap form's fidelity is NEVER worse on Ising (0 of 12), which the full rule cannot claim.**
Per-chi, the single-site gain lives at chi <= 10 (ratios 0.53-0.94) and REVERSES at chi >= 11
(1.47, 1.37, 3.95, 2.36) where plain SVD is already at 1e-5 -- an accuracy FLOOR of the rule, not
a win anywhere. nn and nnn stay better at every chi including 16 (0.45, 0.39).

- In Ising the cheap form keeps roughly 60-70% of the full rule's single/nn gain (full: 0.61/0.51)
  but much less of the nnn gain (0.92 vs 0.61), and holds the energy ~25x better than SVD rather
  than machine-exact. Its fidelity is never worse, which the full rule cannot claim.
- In Heisenberg the cheap form holds fidelity exactly TIED (better than the full rule's 1.05) and
  keeps the energy gain, but LOSES the nn/nnn gains (1.09, 1.02 vs the full rule's 0.78, 0.54).
- **Window choice, one cell each (Ising chi=6, Heis chi=8):** a single 4-site window `[b-2,b+1]`
  (`r2`) matches the radius-1 inclusion-exclusion pair (`r1`) on every class at **half the wall
  time** -- simpler AND cheaper, so prefer it. The 2-site straddling window alone (`nn`, 15
  strings) is cheapest but loses the nn gain entirely (9.58e-3 vs r1's 5.66e-3, SVD 9.72e-3),
  so the 3-4 site support is load bearing. NEEDS a grid, not one cell.
- A hard energy EQUALITY (`tq`) is indistinguishable from the energy at weight 100 in the weighted
  set, so it is not worth the extra multiplier.
- gamma still matters under the trust region (Heis nn 0.78 at gamma=4 vs 1.07 at gamma=1), so the
  grading is not just a disguised spend limiter.
- **TOLERANCE GUARD `otol`** (skip the correction when the residual at the SVD point is below
  `otol^2`, i.e. when this truncation moves the protected observables by less than the target
  accuracy). It is a clean ON/OFF switch, NOT a refinement: at `otol=1e-3` the Ising chi=12,14,16
  cells become bit-identical to SVD, which removes the single-site damage (3.95x -> 1.000) but also
  forfeits the nn/nnn/energy gains there (0.45/0.39/0.04 -> 1.000). chi=11 is the one partially
  active cell and is strictly better than unguarded (single 1.20 vs 1.47, nn 0.69, nnn 0.67).
  Worth having because it makes the rule never worse than SVD once the user names a tolerance.

## 6 MPS-NATIVE TEST, FINAL VERDICT (last trial, 2026-10-04): NOT PRACTICAL as a truncation rule

Real TEBD (Strang sweeps, mixed canonical form, cut at every two-site gate), Ising N=14/18, T=3, dt=0.1,
reference = the dense Trotter circuit with the SAME gates (isolates truncation). Scratch `exp3/mpsenh.py`,
`mps_bench.py`, `mps_cost.py`, `mps_costmatch.py` (session scratchpad, will vanish). Verified before any
benchmark: untruncated infid 4e-16; kappa=0 and a closed gate bit-identical to SVD; slope f'(0) = -4|g|^2 to
2e-8; polynomial objective = from-scratch to 1e-11; predicted objective = realised; rank = cap; canonical form
1e-16 in both sweep directions.

Cheap design: window RDM as W = A theta B (A left-iso, B right-iso, so no string sums); 1-2 step directions;
FACTORED retraction (U+tZ)(S V^dag + t U^dag d), exactly rank chi, orthonormalised by Cholesky (U^dag Z = 0
makes P^dag P = I + t^2 Z^dag Z well conditioned); right sweep uses it, left sweep the mirrored form.
numpy's QR was 34 of ~50 ms overhead at chi=128 and is not needed.

**Per-cut cost vs plain SVD** (interleaved, min of 15, chi=128): bond-only energy 1.20x; 4-site window one
direction 1.63x; two directions 2.05x (a fixed ~20 ms 2-D grid search, so the ratio falls with chi).
Earlier estimates (dense toy 30-50x, then ~4x) were wrong in both directions. Python overhead dominates below
chi~48 (ratios 20-40x there are NOT the cost law).

**Same-chi (N=18, 12 chi, median ratio vs SVD):**
| arm | infid | single | nn | nnn | energy |
|---|---|---|---|---|---|
| 1-dir window, wE=0 | 0.998 [4\|8\|0] | 0.97 [6 better, 6 worse] | **0.61** [12/12] | **0.74** [12/12] | 2.29 worse (9 of 12) |
| 2-dir window+energy | 0.991 [4\|8\|0] | 0.78 [9/3] | 0.77 [12/12] | 0.84 [12/12] | **0.33** [9\|0\|3] |
| 4-site window energy-only | 1.001 | 0.97 | 1.00 | 1.00 | **0.48** [11\|1\|0] |
| bond-only energy-only | 1.000 [0\|12\|0] | 0.99 | 1.00 | 1.00 | 0.90 |
Single-site REVERSES at chi >= 10-13 (up to 3.2x worse) in the window arms. Heavy truncation (chi 4-7) even
improves fidelity 3-8% (sequential sweeps are not globally optimal) but that vanishes by chi=9.

**Matched cost (modelled cost chi^3 x [1 + f(R-1)], SVD interpolated at chi_eq, 2% margin to the incumbent),
gain > 1 means the rule wins:**
- 2-dir at R=1.3 / 1.6 / 1.8: fidelity 0.64 / 0.41 / 0.34 (0 wins of 11 in every case); single 0.69/0.50/0.41;
  nn 0.96/0.66/0.53; nnn 0.94/0.79/0.58; **energy 2.64 / 1.73 / 1.62 (8, 6, 6 of 11 wins).**
- 4-site energy-only at R=1.4 / 1.65: energy **1.61 / 1.34** (10, 9 of 11 wins); fidelity 0.76 / 0.65;
  local observables 0.7-0.83. bond-only at R=1.2: everything 0.90-0.93, energy 0.99 (tie).
So ONLY the energy wins at matched cost, and it is paid for by 25-35% WORSE fidelity and local observables.

**Structural reason (not an implementation shortfall):** at equal chi fidelity can at best TIE (SVD is optimal at
fixed rank), so ANY extra cost R > 1 loses on fidelity at matched cost by the SVD slope. A fixed 0.5-0.8 factor
on observables cannot beat the exponential fall of the SVD error with chi. No cost engineering fixes this.

**Gating was a mirage.** A gate relative to the running MEAN of the discarded weight never fires selectively,
because the discarded weight GROWS with time so the newest cut is always above the mean (351 of 411 cuts fired
at gate_rel=3). A decayed running MAX also fails, since bulk cuts at the cap sit within 2x of each other late in
the run. The absolute gate 1e-5 that fired on 11% of cuts and kept the gain was a TIME gate that skipped the early
cuts of a short run (T=3). In a long run every bulk cut is saturated, fires, and the cost is the full R.

Closed. Recommendation: do not ship as an accuracy rule. If energy conservation is wanted, the existing spc_mps
correction is the right tool; the 4-site window energy-only variant buys 1.3-1.6x energy at matched cost for ~30%
fidelity, which is not a good trade.

## 7 RE-SCORED under the rank_allocation methodology (2026-10-04, user's instruction): REVERSES section 6

User: score ONLY final (stored params, final error), as `experiments/truncation/rank_allocation/dp_analysis.py`
(pooled front, `beats` with margin, ledger vs baseline, 1/2/5/10% margins). Cost/time/modelled cost are NOT scored.
Scratch `exp3/mps_front.py` is a port of that logic, applied to infidelity AND each observable error.
Section 6's "not practical" verdict was a COST verdict and does not hold under this scoring.

Five cells (Ising Neel N=18 and 14, ising2 N=14, domain wall N=14, Heisenberg N=14 T=0.8), 12 chi each.
Pooled over 5 cells at 2% (rule beats baseline / beaten by baseline, of ~60 runs):
- window+energy 2-dir (wE=100): infid 9/6, single 39/25, nn 52/6, nnn 54/5, E 47/21
- window only 1-dir: infid 13/18, nn 46/14, nnn 56/3, E 23/36  (bad: Heisenberg infid beaten 10/12)
- 4-site energy-only: E 50/9; bond-only: E 53/10, rest null
Weak spots: single-site at chi>=13 (Ising) and chi>=8 (domain wall) up to 2.2x worse; energy erratic across chi
(SVD's own energy error is non-monotone); ising2_14 covers only 1.6 decades (chi<=16 too small).

BEST ARM (Ising N=18 only, 11 of 12 chi; the other 4 cells were NOT run, global-energy arms are very slow):
`enhg` = graded 4-site window + GLOBAL energy through Hamiltonian environments (exp3/mpsenv.py, verified:
MPO energy = dense 4e-15, <theta|H_eff|theta> = chain energy 3e-14), wE=1e4, refine=4. At 2%: infid 3/0,
single 9/2, nn 11/0, nnn 11/0, E 9/0; margin-stable for nn/nnn/E/single (infid wins fall to 1/0 at 5%, 0 at 10%).
Final energy error 4-200x below SVD (not exact, errors accumulate over cuts). Global-energy-only (`enhG`):
E 9/0, single 11/0 but infid 0/3 and nn/nnn null. Per-cut energy error is 3e-3 of SVD's.
Unscored fact: per-cut cost 1.2-2x SVD plus env upkeep, wall time ~10x in my numpy.

## Still not established

Real MPS cost (all of this is dense toy N=10-12 with an exact propagator, so wall times are not
MPS costs -- the honest claim is the dominant per-bond cost falls from ~111 global operator
applications to ~1-3 window contractions plus a line search). Cost-matched comparison against
plain SVD at larger chi (would still win). N>12. The single-site weakness in Heisenberg is
unexplained beyond "the displacement costs more than the drift it suppresses there".

## 8. Tree port, spin-boson d=(64,32,16,8,4,2), 2026-10-05 (scratch code lost with session)
- Ported to the m=3 tree walk: at a child cut the centre tensor has orthonormal neighbours, so any observable is a quadratic form through the TTNO env cache (`SandwichCache.init_cache_but_one` + `contract_ket_ham_with_envs`). Rule tilts the kept child-leg subspace, Pi(t)=sum v_j v_j^dag, 3k H-applications per cut, trust region (1+kappa)*SVD weight. Reproduces the recorded baseline with kappa=0.
- Energy term only: final energy error 100-1000x below SVD at chi<=10, 3-50x at chi 12-16, no firing at chi>=20 (discarded weight <1e-14). Infidelity, sz, sx, n_k, zx_k UNCHANGED (ratio 1.00); x_k 11-22% worse at chi 4,6. kappa 0.03-1.0 identical. Energy is conserved, so this is a soft pin, same as spc_mps.
- Weighted one-/two-body observable set (sx, sz, n_k, x_k, sz x_k) + energy: NO gain, all observables within ~2% of SVD (n_k 0.78, x_k 0.89 at chi=8 only), energy error up to 6x WORSE. Per-cut preservation of an expectation does not survive the later evolution.
- Graded window (Pauli-weight RDM) was NOT ported: leaves-only physical sites, no natural weight grading on a tree.
**Why:** answers "does it carry over to the paper's tree models". **How to apply:** do not claim a correlator/observable gain on trees; only the energy pin works.

## 9. VERDICT 2026-10-05: DEAD END (user decision), no library code was ever added
- Chain: nn/nnn ~2x on Ising Neel only, fidelity tie, single-site and energy mixed, Heisenberg window-only loses, only tested up to chi 16; per-cut cost 1.2-2x SVD; global-energy arm tested on ONE cell. Tree: energy pin only (= spc_mps effect), no observable gain.
- Nothing was written into RAGE, GCG, YAQS or RAGE-BASE PTN library code (rule lived as scratch monkeypatches). Only artefacts left in the repo: four PNGs in `RAGE-BASE PTN/experiments/truncation/rank_allocation/figures/` (`frontiers_observables_chain`, `frontiers_all_arms_chain`, `six_wide_tree_rule`, `six_wide_tree_rule_obs`).
- Do NOT reopen without a new idea. Untried: global rank allocation (DP budget) with the rule on top on a chain; tolerance-per-cut allocation was tested and loses to a uniform cap at equal final params (early discards accumulate).

## 10. Speed-up of window + global energy, 2026-10-05 (scratch only, still a dead end)
- Profile of the original: 55% line search (625x256 @ 256x256 complex matmul per evaluation), 25% H_eff einsums with a fresh path each call, rest Python overhead.
- Fixes: (1) objective along the step = cW (phi^T Gam phi)/(1+q)^2 with a K x K Gram matrix (K = 1+m+m(m+1)/2 monomials) built once per cut, so each grid point costs O(K^2); (2) H_eff pieces L.W_b and W_{b+1}.R fixed per cut, all tensors applied in ONE batched tensordot chain, env updates as tensordot chains; (3) cached line-search grids.
- Result: 11.3x faster than the original (84.8 s -> 7.5 s, then 2.1 s serial after grid caching) on Ising N=14 chi=8, 15 steps, errors equal to 3e-5 relative (fired 81 vs 78 cuts, near-tie threshold). Env code matches to 2e-15.
- Cost vs plain SVD run, serial single BLAS thread, N=16 Ising T=3: chi 8 13x, chi 16 15x, chi 32 7.3x, chi 48 7.2x (still falling, includes gate application in both). The old 400-5000x numbers were parallel-contention artefacts plus un-optimised einsum.
- Verdict unchanged: ~7x cost for ~1.5x correlator gain loses to simply raising chi. No library code added.

## 10. Same chain rule as RAGE_MPS's final truncation cut, 2026-10-05 (scratch code lost with session)
- Hook: RAGE_MPS truncates with one leftward sweeping_per_bond_truncation; each cut is the SVD of theta=T_{b-1}T_b with left-iso T_{b-2}, right-iso T_{b+1}, i.e. exactly EnhCut's setting. Patched `sweeping_truncation.split_svd_contract_sv_to_neighbour`; plain path reproduces the library bit for bit (chi=6: 5.571e-4 both). Hamiltonians as MPO (Ising Neel / domain wall T=3, Heisenberg Neel T=1 since T=3 gives infidelity ~0.95 at chi<=16), dt=0.05, exact reference expm_multiply, chi 4..16 (12 values), rel_tol 1e-12. Heisenberg needs `max_aug_bond_dim=3*chi` (unbounded augmentation ran out of RAM).
- Ledger at 2% pooled over 3 cells (36 runs/metric), window+energy 2dir beats/beaten: infid 6/5, single 23/17, nn 32/6, nnn 30/6, E 19/17. Ising Neel alone: nnn 24/0 of 24 at 10% margin (with dw), nn 21/0. Heisenberg: no infidelity win (1-7% worse), energy 1.3-2.7x worse. Window-only: infid 7/14, E 17/19.
- Same pattern as TEBD: nn/nnn win, energy and single-site mixed, fidelity no gain. Does NOT overturn the section 9 dead-end verdict.

## 11. VERDICT REVISED 2026-10-05: narrow but REAL regime (memory-bound, high pressure)
Section 9's "dead end" was decided on an EASY cell. Re-measured on a high-pressure cell
(Ising N=20, T=8, dt=0.1 -- SVD still at 8e-4 infidelity at chi=160) with the 11x-faster code,
serial, one BLAS thread. Arm = window + global energy, kappa=0.1, wE=1e4, refine=4.

Converting matched-parameter ratios into "what SVD needs to reach the same error":

| for equal | rule memory advantage | rule time cost |
|---|---|---|
| nn correlator | **2.0x** (1.31 at chi=16, then 2.04/1.97/1.96 at chi=24/32/48 -- PLATEAU, no trend) | 2.9-3.2x |
| energy | 3.5-4.5x (declining: 4.37/4.45/3.87/3.46) | 1.5-1.8x |
| infidelity | 0.92-0.98x, i.e. NO gain | 6-8x |

- Fidelity now LOSES at matched params (1.02-1.06x worse), unlike the easy cell where it tied.
  The trust region spends fidelity it cannot recover once truncation is severe.
- The EASY cell (N=18, T=3) is useless for a cost test: SVD hits machine precision at chi=64 in
  0.76 s, so nothing at fixed chi can win on time there. Always check the cell is unconverged.
- The energy column is DOMINATED by existing work: Method D / spc_mps energy pin
  ([[project_conserved_observable_correction]]) costs ~2 H_eff actions per step and pins energy to
  machine precision. Do not claim the energy gain as this rule's contribution.
- So the rule's only unique claim is **2x memory on correlators at ~3x time**.
**Why:** the earlier dead-end call was made in the wrong regime and on slow code.
**How to apply:** if revisited, the target is the 3x time gap -- the gain is 2x memory on
correlators and it does not grow. Test only on unconverged cells. Still no library code.

### 11b. Framing correction (user, 2026-10-05)
Matched BOND DIMENSION is the method-vs-method comparison and the agreed methodology: both rules
cut to the same rank and store the same tensor, so only the choice of rank-chi subspace differs.
Wall-clock is a property of this prototype, not of the rule. Do NOT lead with a matched-cost
comparison. On the matched-bond-dim axis the rule's unduplicated claim is the CORRELATOR gain
(1.2x easy cell, up to 2.2x at high pressure). The memory/time numbers in Sec. 11 remain valid as
a secondary note only.

### 11c. chi=64 point (same high-pressure cell), 2026-10-05
rule chi=64: infid 3.359e-2, nn 1.231e-3, E 2.935e-3  vs SVD chi=64: 3.328e-2, 3.352e-3, 3.439e-2.
Matched-bond-dim ratios 1.009 / 0.367 / 0.085. The nn ratio keeps improving with chi:
0.85, 0.63, 0.52, 0.46, 0.367 at chi = 16, 24, 32, 48, 64 (i.e. 1.2x -> 2.7x better correlators),
and the fidelity loss closes back to 1%.
NOTE the two statements are both true and not contradictory: the matched-chi error ratio keeps
improving, while MEMORY efficiency stays ~2x (SVD needs 148.8k params for the rule's chi=64 nn
error vs the rule's 76.5k) -- SVD's error-vs-params curve is steep, so 2.7x error = ~2x params.
Wall 40.8 s vs SVD 5.4 s at chi=64, so the time ratio ROSE to 7.6x (was 6.0x at chi=48); it does
not fall monotonically in this cell.

## 12. NEW DIRECTION 2026-10-06: marginal-FIT compression (MFC) -- global, not a cut rule. In `PTN trunc_research/experiments/truncation/graded_rule/`
User asked for ANY truncation method with a decisive high-pressure win, across models.

**What was learned first (all matched bond dim, final state):**
- Decomposition of the old best arm (Ising N=20 T=8, chi 32/48): the WINDOW term is the driver (window only: rdm2 0.46-0.59, nn 0.47-0.60, single 0.45-0.59 of SVD). GLOBAL energy alone: energy 0.25-0.34x but marginals 1.00x, no correlator gain. Window+global adds little (0.43/0.46/0.40). EXACT energy pin makes locals WORSE (single 1.33x, nn 1.15x). So conserving energy is NOT the mechanism.
- spc_mps-style pin inside TEBD: pinning to the energy BEFORE each cut lets gate-induced drift accumulate (error halves when dt halves = per-gate effect); pin to E_0 once per full step restores energy to the Trotter band (2.9e-3 on Heisenberg) but gives only 5-10% on observables.
- Exact window-marginal matching (Gauss-Newton, no trust region) FAILS: 4-site window 5.6x-28x worse infidelity; 2-site window matches the cut exactly yet final rdm2 only 0.58. Reason: after a quench, truncating one bond perturbs marginals FAR from the cut (light cone covers the chain), so local windows cannot fix it.
- Schmidt-weight refit at one cut (keep subspace, refit k weights to ALL window marginals): only 1-2x at k=8-16, 3.6x at k=32. Not decisive.
- **STATIC CEILING (probe_ceiling.py, N=10 T=3, JAX autodiff L-BFGS, fidelity-free):** best rank-chi MPS for ALL 1-,2-,3-site marginals is 4.6x (chi=4) and 13.5x (chi=6) better than SVD compression, fidelity 0.946->0.855, 0.988->0.955; optimiser hit maxiter so conservative. HEADROOM EXISTS.

**Dynamic test (probe_stepfit.py): dense exact step then compress EVERY step, SVD vs global marginal fit (windows k<=3), fw = weight of (1-F) with the uncompressed step result:**
- fw=0 (pure marginals): Ising wins (N=10 T=3 chi=6: 1site 0.23, 2site 0.24, 3site 0.52, E 0.04, infid 1.41x) but Heisenberg chi=6 BREAKS (single 2.5x worse, infid 0.98) -- state drifts, no fidelity anchor.
- **fw=0.01-0.03 is the sweet spot**: Ising chi=6 fw=0.01: infid 1.02x, 1site 0.28, 2site 0.28, 3site 0.51, E 0.10. Heisenberg chi=6 (SVD infid 0.64, very high pressure): fw=0.01 gives 0.22/0.68/0.75/E 0.41, noisy and non-monotone in fw, NOT yet claimable. fw>=0.3 collapses to SVD.
- Ising N=12 T=4 chi=8, fw=0: 1site 0.22, 2site 0.32, 3site 0.54, E 0.05, infid 1.88x.

**MPS-native implementation (rule/mfc.py, mfc_bench.py):** gate sweep at working rank f*chi, then compress to chi by SVD ('dsvd') or marginal fit ('mfc'). MPS marginals via transfer matrices match dense to 3e-16; pipeline reproduces the dense probe (Ising N=10 T=3 chi=6: dsvd infid 2.318e-2 = dense SVD; mfc nn 0.29x, E 0.12x of dsvd; vs classic per-gate svd rdm2 3.9x better). JAX installed ISOLATED in scratchpad `pylibs` (PYLIBS env var), NOT in the project venv (disk was 4.6 GB free).
**Caveats to keep:** transient working rank f*chi must be reported as `peak` and the dsvd baseline uses the SAME transient; mfc costs ~100x SVD wall in this prototype; Heisenberg unproven.

### 12b. MFC design grid, 2026-10-06 (dense per-step compress, fit/SVD ratios; Ising N=12 T=4 chi=8/10 unless stated)
- **Iterations do not matter**: k<=3 fw=.01, maxiter 250/600/1500 -> 1site .26/.25/.25, 2site .28/.28/.27 (chi=8). Converged at 250; the gain is NOT optimiser-limited.
- **Window size saturates ~0.2-0.25 on 1-2 site marginals**: chi=10: k<=2 0.45/0.46, k<=3 0.27/0.29, k<=4 0.21/0.23, k<=5(600it) 0.19/0.23; 4-site marginal 0.83->0.71->0.56->0.45. Larger windows improve marginals OUTSIDE the fitted set too.
- **fw**: k<=4 chi=8: fw=.003 -> 0.20/0.22 but infid 1.22x; fw=.01 -> 0.22/0.24, infid 1.09x; fw=.03 -> 0.27/0.28, infid **0.97x** (fidelity at or below SVD). Pick fw~0.03 for a fidelity-neutral claim.
- **Other models, k<=4 fw=.01 chi=8/10**: domain wall 0.33/0.46/0.58 and 0.35/0.48/0.56 (1/2/3-site), E 0.04, infid 1.06-1.07. **Heisenberg N=12 T=2.5 FAILS on 1-site: 1.77x (chi=8) and 2.07x (chi=10) WORSE**, 2-4-site 0.7-1.05, E 0.57/0.20, infid 1.09-1.14 (SVD infid already 0.64-0.72). Integrable system: marginal fit discards the long-range coherent info the dynamics feeds back on.
- N=14 T=6 dense (k<=3 fw=.01): chi=8/12/16 -> 1site .66/.43/.38, 2site .67/.50/.44, E .05-.06, infid 1.07-1.09: gain GROWS with chi.
- **N=20 T=8 MPS-native vs SVD front (stored params)**: fit chi=24 rdm2=1.18e-2 @13.7k params; SVD reaches that at ~44k params => ~3x memory efficiency on 2-site marginals; chi=16 fit rdm2 2.72e-2 ~ SVD chi~27 => 2.6x. Dynamic gain is ~3x below the STATIC ceiling (13x at N=10 T=3 chi=6) => dynamics loses info that feeds back.
- Cost: ~1300 s per N=20 chi=24 run (80 steps x 200 L-BFGS evals x 45-80 ms, JAX single thread) vs 16 s for SVD.

### 12c. The LEAK, 2026-10-06 (probe_ceiling.py, probe_leak.py, probe_stepfit.py lookahead option)
- **Static ceiling is ENORMOUS**: Ising N=12 T=4, best rank-chi MPS for all 1-,2-,3-site marginals: **700x (chi=8) and 2500x (chi=10) better than SVD** (2-site 1.9e-5 vs 1.35e-2), fidelity 0.934 vs 0.970. Representation headroom is not the limit.
- **probe_leak.py**: compress the exact state at t=4 then evolve it EXACTLY: SVD-compressed 2-site marginal error is a FLAT offset (1.3e-2, never grows); the fitted state starts at 1.9e-5, is 37x better after one step (3.8e-4), then GROWS ballistically (3e-3 at step 3, 8.3e-3 at step 6) and equals SVD's error at step 10 (t=1.0). So fitted-state advantage lives ~10 steps; compress-every-step recovers only ~4x of the 700x. Mechanism: exact local marginals but wrong long-range/higher-weight degrees of freedom, which feed local marginals back within ~5-10 steps.
- **Lookahead (fit marginals of U^m v too, m<=M), N=12 T=4 chi=8 k<=3 fw=.01**: M=0/1/2/3 -> 2-site 0.28/0.24/0.21/0.19, 1-site 0.26/0.21/0.18/0.16, E 0.14/0.08/0.05/0.03, infid 1.04/1.12/1.16/1.20. Far pairs (d>=4) STUCK at 0.48-0.49 for every setting.
- **All-pairs term** (all two-site marginals at ANY distance), weight 0/0.5/2, chi=8/10: far-pair 0.49/0.70 -> 0.45/0.62 -> 0.42/0.57, slightly worse near marginals. Modest, not the missing piece.
- **TDVP-1 at fixed chi is NOT better than SVD-TEBD** (Ising N=14 T=6, vs CONTINUOUS exact): 1.05-1.15x WORSE at chi=8/12/16/24. So SVD is the right standard baseline. MFC (fw .03, k<=3) at the same cell: chi=8 nn 0.64x, chi=12 nn 0.48x of SVD.
- Trotter floor of the dt=0.1 dense reference vs continuous (Ising N=14 T=6): single 6.4e-4, nn 8.9e-4, E 2.5e-2 -> energy below ~2e-2 is NOT comparable across reference types; marginals above ~1e-2 are fine.

### 12d. LOOKAHEAD IS DECISIVE (dense, Ising N=12 T=4 chi=8, k<=3, fit/SVD), 2026-10-06
M=0/3/6/10 (fw .01): 2-site 0.28/0.19/0.13/**0.08**, 1-site 0.26/0.16/0.11/**0.05**, 3-site .42/.31/.24/.19, infid 1.04/1.20/1.28/1.33. **M=10, fw=.03: 1-site 0.06, 2-site 0.09, 3-site 0.21, pair<=2 0.16, infid 1.16, E 0.08.** Far pairs d>=4 and 4-site marginal stay ~0.45-0.47 (never fitted). => fitting marginals of the PROPAGATED state U^m v (m<=M, M=10 = one time unit = the leak time) closes most of the gap between dynamic (4x) and static ceiling (700x). This is "prediction-optimal compression": the compression metric is the error of local observables over a future window, d^2 = sum_m sum_P |<(U^dag)^m P U^m>_c - <...>_exact|^2. The Heisenberg-evolved operators are STATE-INDEPENDENT so they can be precomputed once as MPOs (DAOE-style truncation) and reused every step. OPEN: MPS-native version (dense probe only so far), Heisenberg / domain-wall with lookahead, sparse time sets, M>10.

### 12e. COST-EFFICIENT LOOKAHEAD, dense Ising N=12 T=4 chi=8 fw=.03 (fit/SVD), 2026-10-06
| fitted time set (steps ahead) | cost | infid | 1-site | 2-site | 3-site |
|---|---|---|---|---|---|
| {10} only | 1x | 0.94 | .33 | .37 | .47 |
| {0,10} | 2x | 0.99 | .14 | .18 | .31 |
| **{0,5,10}** | 3x | **1.02** | **.09** | **.13** | .26 |
| all 0..10 | 11x | 1.16 | .06 | .09 | .21 |
- **k<=2 (singles + nn only) with full M=10: 1-site 0.08, 2-site 0.12, infid 1.04, 3-site 0.39** -> the operator set can stay small.
- Heisenberg-evolved operators (U^dag)^m P U^m are exact as MPOs (1e-13 vs dense) but their bond dimension EXPLODES under the sequential Strang sweep (N=8, m=5: bonds 107-136 of max 256), so MPO storage of the lookahead operators does not scale past M~5. Scalable routes: sparse Pauli-path expansion (Begusic-Chan style, top-K strings), coarse-Trotter light-cone regions, or dense lookahead for N<=16-18.

### 12f. CROSS-MODEL, lookahead set {0,5,10}, k<=3, fw=.03, dense, fit/SVD (partial, 2026-10-06)
| cell | chi | infid | 1-site | 2-site | 3-site | note |
|---|---|---|---|---|---|---|
| Ising N=12 T=4 (reference cell) | 8 | 1.02 | .09 | **.13** | .26 | best |
| ising2 (hx=1.4,hz=0.4) N=12 T=4 | 8 | 1.05 | .26 | .32 | .47 | |
| domain wall N=12 T=4 | 6 | 0.93 | .46 | .55 | .69 | |
| domain wall N=12 T=5 | 8 | 1.05 | .30 | .43 | .60 | |
| Heisenberg N=12 T=1.0 | 8 / 10 | 1.27 / 1.17 | .24 / .63 | .38 / .56 | .57 / .71 | no-lookahead Heisenberg was 1-site 1.8-2.1x WORSE; lookahead fixes it |
| Heisenberg N=12 T=1.5 | 12 / 16 | 1.07 / 1.12 | .17 / .56 | .37 / .37 | .44 / .44 | |
Energy ratios on Heisenberg are SVD-luck dominated (SVD E non-monotone in chi); do not quote. VERDICT SO FAR: lookahead removes the Heisenberg failure; gain is 8x on the best Ising cell but only 2-3x on ising2, domain wall and Heisenberg => NOT decisive across models. Still pending: Ising N=14 T=6, M=20 saturation. MPS-native lookahead unbuilt.

### 12g. Saturation, k-ablation, native bank (2026-10-06, same cells; plan file `...generic-ember.md` is the live plan)
- **M=20 (every step 0..20), Ising N=12 T=4 chi=8 fw=.03 k<=3: 1-site 0.05, 2-site 0.08, 3-site .21, 4-site .48, E .12, infid 1.21** (needs 1611 s dense). {0,5,10}: .09/.13; so gain grows with set density AND length (no lookahead 3-5x -> 12x).
- Domain wall T=4 chi=8 {0,5,10}: 1-site **.14**, 2-site **.22**, E .11 (chi=6 was .46/.55). Ising14 T=6 result file is empty (job still running at 12:00).
- **Which windows are fitted at m>0 matters** ({0,3,6}, Ising chi=8, k<=3 at m=0): k=1 only .22/.29, k<=2 .17/.21, k<=3 **.13/.16** (E 0.00). Static wide windows k<=5 with NO lookahead: .25/.27/.32/.42. So the propagator picks combinations a wide static window wastes capacity on (lookahead 1-2 site ~2x better than k<=5 static).
- **Heisenberg-evolved operator MPO bonds (Ising N=12, relative eps)**: eps 1e-4: m=1..10 -> 4/6/9/11/17/29/40, err<=1e-4; eps 1e-5: 5/7/13/18/35/52/87, err<=1.2e-5; eps 1e-6 reaches 150 at m=10. Exponential in m (doubling every ~4-5 steps), so native lookahead is feasible only to m~6 at D~20-35.
- **Native MPS bank built and validated**: `rule/lookahead.py` (Bank/build_banks/bank_values/LookaheadFit/run_mfcl), MPO window-cut + padded vmapped scan, matches dense propagated marginals to 4e-10 (`test_lookahead.py`); bench arm `mfcl:f:fw:k:iters:ms:eps:kla[:lam]` in `mfc_bench.py`. Native no-lookahead MFC at N=12: rdm2 1.31e-2 vs dsvd 3.78e-2 (0.35x), matches the dense probe (0.28x).
- **12h (2026-10-06 later): the lookahead propagator need NOT be the dynamics.** Dense Ising N=12 T=4 chi=8, fit/SVD 1-site/2-site: fine Strang-sweep {0,5,10} .09/.13; COARSE brickwork Strang step (3 layers: even-half, odd-full, even-half; strict light cone +-3) of time tau, steps {0,1}: tau=.5 .16/.19, tau=.7 .16/.19, tau=1.0 .22/.24; {0,1,2} at tau=.5 (6 layers) .10/.14; {0,1,2,3} at tau=.3 .09/.13. Independent cheap propagators: {.5 even-first, .5 odd-first} .13/.17; {tau .3,.6,1.0} .12/.15. L-BFGS iterations 50/100/250: .20/.18/.16 (1-site) so 100 is enough. TAYLOR JET matching (d^j/dt^j of window marginals, j<=4, k=1,2) = .27/.30, same as no lookahead: DEAD. DAOE-style damping of the operator MPO: bonds unchanged and errors 1e-2..3e-1, DEAD.
- Sweep-Heisenberg bank (the dynamics itself) has support ~whole chain (median 9-13 sites at eps 1e-4, N=12-16) and bonds doubling every ~4 steps (D=18 at m=6, 40 at m=10); brickwork banks: 3 layers W=8 D<=8, 6 layers W=12-14 D=22. Native eval cost value+grad (complex64 contraction, ragged bonds): N=20 chi=16 1-2-site strings: 3 layers 0.14 s, 3-site strings 0.9 s, +6 layers 1.0 s. Cost is flops-bound; local-sweep optimisers do NOT beat 250 global L-BFGS evals.
- **Native MPS (no dense vector), Ising N=12 T=4 chi=8, vs dsvd:2 (same 2chi transient), mfc:2:.03:3:250 baseline rdm2 1.31e-2**: coarse tau=.5, 1 step, strings 1-2: rdm2 .25x, nn .27x, E .10x, infid .97; strings 1-3: .22x/.24x; tau=.7 same. So native ~4.5x vs dense-exact-lookahead 8-12x (fewer lookahead terms).
- Ising N=14 T=6 (very high pressure, SVD infid .38/.16), dense {0,5,10}: chi=8 1-site .44 2-site .44 E .02; chi=12 .21/.27 E .04. Gain drops at high pressure.
- Cheap single coarse step (tau=.5, {0,1}), dense N=12 cross-model: ising2 .45/.48, domain wall T=4 chi=8 .22/.30, Heisenberg T=1.5 chi=12 (tau=.3) .24/.32, infid 1.02-1.05.
- **12i (2026-10-07): LOCAL-REGION lookahead** (`rule/region.py`, bench arm `mfcr:f:fw:k:iters:L:a:taus:kr[:lam_r]`): window marginals after evolving the L-site region (margin a) EXACTLY under its own open-chain H for times tau; cost = one L-site marginal, any horizon. Native N=12 T=4 chi=8 ratio vs dsvd:2 (rdm2/nn/E/infid): Ising L=6 .21/.25/.23/1.09, L=7 kr=1-3 .18/.20/.13/1.08; DW L=6 .31/.37/.017/1.06, L=7 kr 1-3 .30/.34/.07/1.07, 8 taus .30/.34/.001/1.10; ising2 L=6 .41/.46/.10/1.06; Heis chi=12 T=1.5 (MPO coarse) .44/.42/.25/1.01. mfc without lookahead: Ising .35, DW .57, ising2 .65, Heis .84. So lookahead adds a further 1.6-2x over MFC and E error falls 10-1000x (spc_mps owns the energy claim). Dense {0,5,10} reference: Ising .13, DW .22, ising2 .32.
- **Gain depends on pressure (SVD final infidelity)**: ising2 chi=8 (infid .36) .26/.32, chi=12 (.16) .14/.20; Ising N=14 T=6 chi=8 (.38) .44, chi=12 (.16) .21/.27; Ising N=12 (.06) .09/.13. Marginal error after the fit ~ 1-3% of infidelity, SVD's is 6-38% of it, so the achievable gain is SVD ratio / fit ratio ~ 3-12x and shrinks as infidelity grows.
- Null results 2026-10-06/07: working rank f=2/4/8 identical; L-BFGS 250 vs 800 iterations identical (.16 vs .15 on 1-site); rank allocation by greedy discarded-weight minimisation at fixed params on the Ising final state: 916 params cost .0442 vs uniform 936 .0449 (spectra uniform in the bulk, NO allocation lever); Taylor jets; DAOE damping.
- **12j (2026-10-07): PEAK-MEMORY and COST questions (user challenge: truncation exists to avoid large dimension, the global fit stores a 2chi working state).** (1) Working rank of the global fit, Ising N=12 chi=8, region lookahead 4 taus, 2-site ratio vs delayed SVD at the SAME working rank: 2.0chi .21, 1.5chi .23, 1.25chi .36 (peak memory 4x, 2.25x, 1.56x of a plain chi run). So 1.5chi keeps nearly all of the gain. (2) CUT-LOCAL lookahead rule `rule/lookcut.py` (bench arm `lcut:a:fw:iters:taus:ks`): at every gate cut pick the chi-dim subspace of the two-site tensor (Q = qr(top-k + perp@C), L-BFGS from the SVD point) minimising the lookahead-window mismatch inside a region of a sites per side, + fw*discarded weight. Peak bond dimension = chi like classic TEBD, nothing global. Ratios vs CLASSIC per-gate SVD (same peak) 2-site / infid: Ising chi=8 fw=.03 .36 / .97, fw=.1 .40 / .89; DW chi=8 .60 / .93; ising2 chi=12 .47 / .97; Heis chi=12 T=1.5 .80 / .91. a=3 no better than a=2 and 4.5x slower; 20 taus no better than 4. Energy error NOT improved (cut-local cannot fix a global quantity). So greedy per-cut selection gets 1.2-2.8x at classic memory with equal or better fidelity, the joint (global) fit gets 4-8x at 1.5-2x working rank with +13-28% infidelity. Wall time of the prototype is ~100x an SVD run for both. (3) The four-model matrix of the GLOBAL fit with region lookahead (20 taus 0.1..2.0, L=7, windows 1-3) vs delayed SVD: Ising chi=8 2-site .129 infid 1.23; DW chi=8 .241 / 1.13; ising2 chi=12 .144 / 1.24; Heis chi=12 T=1.5 .364 / 1.28. N=20 T=8 chi=16 (SVD infid .53): MFC .58, MFC + coarse-MPO lookahead .58 (E .18), i.e. NO extra gain at that pressure.
- Machine note: the laptop enters Modern Standby, jobs stall for hours and wall-clock secs in bench logs include sleep (26568 s for a 15 min job). Judge progress by CPU seconds.
- Prior art, FIRST PASS ONLY (arXiv search, not read in depth): DAOE Rakovszky-von Keyserlingk-Pollmann 2004.05177 (Heisenberg-picture MPO, damps high-weight Pauli strings), Hartmann et al PRL 102 057202 (Heisenberg-picture DMRG, via 0907.5582), Frias-Perez-Banuls light cone TN 2201.08402, DMT White-Zaletel-Mong-Refael PRB 97 035127. NO hit for "fit the marginals of the propagated state (Heisenberg-evolved operators) as the compression objective of a Schrodinger MPS". Not exhaustive.
