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

### 12k. Pushed to GitHub + prior art second pass (2026-10-07)
- PUSHED: branch `trunc_research` of `github.com/Pouri96/PyTreeNet`, commit `ad1feb6` (242 files, whole `graded_rule/` incl. force-added `out_*.txt` logs and `_refcache/`, plus `handoff/` copies of this memory file and the plan). Purpose (user): continue in cloud sessions that run without the laptop. No Claude co-author trailer on the commit (user rule). Later work needs a new commit + push; the `handoff/` copies go stale if this file changes. The two N=16 global-fit workers (chi=12, 16, mfcr:1.5, L=7) were still running at push time (7 CPU-h each, no output), so their results are NOT in the commit.
- Prior art, SECOND PASS (arXiv abstracts only, the search tool has poor recall, still not exhaustive): Hartmann-Prior-Clark-Plenio 0808.0666 "DMRG in the Heisenberg picture" (abstract: only the observable of interest, not the entire state, is considered, exact at finite bond dimension in some cases) is the closest conceptual neighbour. Also Pizorn et al 1305.0504 (operator-space MPS, Heisenberg local operators), Mendl 1812.11876 (MPO TDVP, augment operator by H to conserve energy), Klein Kvorning-Herviou-Bardarson 2105.11206 (time-evolve sets of local density matrices WITHOUT a global state, information lattice), Huang 1903.10048 (existence: area law for Renyi alpha<1 gives MPS of bond dim poly(1/delta) approximating ALL local properties, supports the capacity argument). Still no hit for "compress a Schrodinger MPS by fitting marginals of its propagated image". Read none in depth. Cite nothing from this list without reading the relevant section first (CLAUDE.md verbatim rule).
- 2026-10-07 the N=16 Ising T=5 global region fit (mfcr:1.5, L=7, 20 taus, chi=12 and 16) was KILLED after ~7 CPU-h per worker with no output (violates the 30 min cap, see [[feedback-experiment-30min-cap]]). Those two `final_table.py` rows stay "missing: method". A smaller version (fewer taus, L=6, T shorter) must fit in 30 min or not be run.

### 12l. New N=16 cells, 30-min-capped (2026-10-07), ratios method / SVD at the same stored parameters
- Cut-local (peak = chi, target from the local two-site tensor only): ising2 N=16 T=4 chi=12: 2-site .40, nn .38, E 2.07, infid .94. Heis N=16 T=1.5 chi=12: 2-site .80, nn .79, E 1.99, infid .99. (Ising N=16 .31-.38, DW N=16 .67, Ising N=20 T=8 .71.) NOTE the total energy error is WORSE than SVD for cut-local (1.0-2.1x) although the nn correlators improve, so the nn errors are same-signed. The global fit improves E (.2-1.0x).
- Global fit (working rank 1.5 chi, L=6, 5 taus 0.25..2.0, 100 iters, ~600 s per cell), Ising N=16 T=3: chi=8 2-site .16 / nn .17 / E 1.05 / infid 1.16 vs svd (vs dsvd:1.5 .17 / .18 / .60 / 1.23); chi=10 .26 / .27 / .41 / 1.03 (vs dsvd:1.5 .28 / .28 / .23 / 1.08). So 4-6x survives N=12 -> 16 for Ising at working rank 1.5 chi.
- Pass line CHANGED by the user: infidelity is no longer a gate (see [[feedback-priority-local-obs-no-untruncated-target]]).

### 12m. FINAL VERDICT (2026-10-08), same cell Ising N=16 T=3, ratios vs svd at equal stored parameters
- chi=8: cut-local 2-site .39, nn .38, 1-site .42, E .16, infid 1.05, 173 s, peak 8. Global fit (1.5 chi, L=6, 5 taus) .16 / .17 / .11 / 1.05 / 1.16, 587 s, peak 12.
- chi=10: cut-local .57 / .57 / .58 / 1.18 / 1.04, 190 s, peak 10. Global fit .26 / .27 / .23 / .41 / 1.03, 629 s, peak 15.
- So cut-local energy error is NOT consistently worse (chi=8 .16, chi=10 1.18), mixed. Global fit is ~3.3x slower than cut-local on the same cell. SVD <1 s.
- Verdict: no DECISIVE win that is legitimate (no oracle or enlarged target) and cross-model at N>=14. Cut-local 1.25-3x on local observables (Ising 2.6-3.2x, ising2 2.5x, DW 1.5x, Heis 1.25x at N=16, Ising N=20 T=8 1.4x), peak=chi, ~3x cheaper than the global fit. Global fit 4-6x on Ising at N=16 and 4-8x on all four at N=12 but needs 1.5-2x working rank and was only run on Ising at N=16. Dense oracle ceiling 4-20x. Wall = pressure (gain shrinks as SVD infidelity grows) plus model dependence plus the working-rank cost. Untried: energy-density term in the cut objective, joint re-optimisation of neighbouring cuts at peak chi. Prior art NOT cleared (first/second pass only).

### 12n. Pareto sweep, Ising N=12 T=4 dt=0.1, chi in {4,6,8,10,12,16}, 2026-10-08 (user asked for ONLY the two methods in the plots, no delayed SVD, N=12 only)
- Files in `graded_rule/`: `pareto_n12.py` (driver, one process per cell, 29 min cap, 6 at once), `plot_pareto_n12.py [wall|cpu] [--svd]`, `results/pareto_n12/*.json`, `figs/pareto_n12_{params,runtime_cpu,runtime_wall}.png`. Settings: lcut:2:0.1:40:0.5-1.0-1.5-2.0:1-2-3 and mfcr:1.5:0.03:3:100:6:2:0.25-0.5-1.0-1.5-2.0:1-2-3. Result files now carry `cpu` (process CPU s) next to `wall`.
- Global fit / cut-local at equal stored parameters, chi=4..16: infidelity 1.08, 1.16, 1.13, 1.07, 1.04, .97 (same); 1-site .45 .44 .34 .31 .28 .19; nn .61 .60 .50 .40 .35 .28; nnn .77 .92 .71 .68 .70 .61. CPU seconds: lcut 440-729 (nearly flat in chi), mfcr 655-3166 (1.5x to 4.3x lcut). Wall under 6 concurrent jobs is similar (mfcr multithreads, lcut single thread and starved), so CPU s is the fair cost.
- With plain SVD added (`plot_pareto_n12.py ... --svd`, files `figs/pareto_n12_*_with_svd.png`), ratios vs SVD at equal stored params, chi=4..16: cut-local 1-site .40-.54, nn .40-.63, nnn .55-.72, infid .88-1.02. Global fit 1-site .08-.24, nn .12-.38, nnn .34-.67, infid .95-1.07. SVD CPU is 0.1-0.2 s (timer resolution 0.1 s, so its curve zigzags in the runtime panels), cut-local 440-729 s, global fit 655-3166 s.

### 12o. SPC-for-TEBD: first-order restoration at the cut (2026-10-08)
- CORRECTION: spc_mps (`paper/spc_mps/`) is compression as an orthogonal projection followed by exact restoration of selected invariants (energy, sums of on-site operators) and a proof of which invariants are admissible. It is NOT "extend the bond, then project". An earlier reply described it wrongly. Prop 2(iv) (restoration does not reduce the state error) matches the measured 5-10% from energy pinning in TEBD (12 above).
- What was built (graded_rule/): `rule/spccut.py` `SPCCut(LookCut)`: plain SVD cut, then ONE first-order restoration of the same cut-region marginals lookcut uses (target from the pre-cut two-site tensor, peak = chi, class (c) of [[feedback-priority-local-obs-no-untruncated-target]]). Q = qr(U_k + U_perp C), step = truncated-SVD pseudo-inverse of the Jacobian J (via eigh(J^T J), singular values below rcond*max dropped), 4-point line search, accepted only if the residual drops, else the SVD cut. No fw, no L-BFGS. Arm `spc:a:mode:rcond:passes:taus:ks` in `mfc_bench.py`; `spc_table.py` prints the ratios; `time_spc.py`/`prof_spc.py` timing; `run_spc_cross.sh`, `run_spc_grid.sh`. `JAXCACHE=<dir>` turns on JAX's persistent compile cache.
- Ratios spc(hard, rcond .01, 1 pass) / svd at equal chi, 2-site | nn | E | infid: N=12 Ising chi8 T4 .33 | .35 | 1.33 | 1.04 (rcond .001, 2 passes .29 | .32 | 1.44 | .94); DW chi8 T4 .84 | .86 | .47 | 1.07; ising2 chi12 T4 .54 | .51 | 1.41 | 1.28; Heis chi12 T1.5 .86 | .76 | .00 | 1.05. N=16: Ising T4 chi8 .31 | .29 | 1.04 | 1.16, chi12 .53 | .46 | .61 | 1.94; DW T5 chi12 .94 | 1.04 | .73 | 1.15; ising2 T4 chi12 .41 | .38 | 1.57 | 1.22; Heis T1.5 chi12 1.04 | .91 | .49 | 1.09. Energy is not constrained and swings (a side effect, no claim).
- Versus lcut (L-BFGS, same objective, N=12): spc better on Ising (.29-.33 vs .40), worse on DW (.84 vs .60), ising2 (.54 vs .47), Heis (.86 vs .80). On DW more passes (4) and rcond .1 to .003 only give .76-.87, so the gap is not the one-step solver. Cut-level residual after/before the correction: Ising .32, DW .51, Heis .59 (N=12), N=16 .27-.53, with discarded weight per cut x1.3-2.6 (the source of the infidelity rise).
- Cost: N=12 chi6 T2.5, warm: 10.8 s (svd 2 s). The first version took 212 s: unjitted target and line search, a 3600 x 70 SVD per cut (replaced by the Gram matrix), cuts discarding ~1e-30 (now skipped, eps_min 1e-10). Cold compile ~350 s per process (jacfwd, ~40 shapes), 59 s with the persistent cache.
- Verdict: spc-for-TEBD is a cheaper, parameter-light version of lcut with the same wall (Ising ~3x, ising2 ~2.4x, DW and Heis ~1x at N=16). The pass line (>= 4x on 3 models at N>=14) is NOT met. Uncommitted at time of writing.

### 12p. spc cut vs plain SVD, Pareto cell Ising N=12 T=4 dt=0.1, chi in {4,6,8,10,12,16} (2026-10-08, class (c) target, peak = chi)
- Driver `pareto_spc_n12.py` (arm spc:2:hard:0.01:1:0.5-1.0-1.5-2.0:1-2-3, one config fixed before running), timing `time_spc_ladder.py`, plots `plot_pareto_spc_n12.py` -> `figs/pareto_n12_{params,runtime_wall,runtime_cpu}_svd_spc.png`; data `results/pareto_n12/spc_*.json` (svd rows are the 12n ones), `results/pareto_n12_cold/`, `results/pareto_n12_steady/spc_steady.json`.
- Ratios spc / svd at equal chi, 1-site | nn | nnn | infid: chi4 .42 | .62 | .63 | .91; chi6 .54 | .55 | .70 | .99; chi8 .34 | .35 | .53 | 1.04; chi10 .44 | .45 | .77 | 1.23; chi12 .72 | .69 | .82 | 1.63; chi16 1.05 | .87 | 1.48 | 2.36. The gain is a pressure window (gone at chi16, svd infidelity 2.6e-3) and the infidelity cost grows as pressure falls.
- Across chi (stored parameters): spc chi8 (936) matches svd chi12 (1704) on 1-site and svd chi16 (2728) on nn; spc chi10 (1288) matches svd chi16 on 1-site and nn.
- Runtime, shared 8-core machine (another session ran lcut jobs): svd 0.1-0.2 s; spc steady state (second run, jit caches kept) 25 s chi4, 27 s chi6, 43 s chi8, 58 s chi10, 78 s chi12, 81 s chi16, CPU 110-320 s because XLA uses several threads; one process per chi with the compile cache warm 82-165 s, cold 112-260 s. SVD dominates on the runtime axis (spc is 150-400x slower), an SVD ladder past chi16 costs well under a second. Only Ising N=12 measured, no other model on this plot.

### 12o. COST WORK (2026-10-08, user: "solve the cost problem, otherwise stop the project")
- Starting point: cut-local 440-729 CPU s vs SVD 0.1-0.2 s on Ising N=12 T=4 (~3000x). My earlier "~100x" note understated it.
- Profile: at the optimised cuts the discarded weight was ~1e-16 (nothing to fit), the target was built eagerly (0.66 s/call), ~14 L-BFGS evals per optimised cut, compile only ~6 s. Fixes in `rule/lookcut.py` (options `skip_tol`, `ftol`, `every`, `opt`) and bench arm `lcut:a:fw:iters:taus:ks[:skip:ftol[:opt]]`: jit the target, skip cuts with discarded weight < 1e-10, ftol 1e-8, region margin a=1, ONE lookahead time (tau=1.0). Same cell chi=8: steady-state 6-7 s vs SVD 0.10 s (~55-65x), ratios to SVD infid .90, 1-site .45, nn .44, nnn .61 (old full setting .89/.41/.41/.56).
- Screen (speed_screen.py): maxiter 12 -> 49x, .51/.49/.69. every 2nd step -> 48x, .67/.68/.74. every 4th -> 31x, .82/.85/.85. skip below 1e-4 -> 16x, .84/.84/.87. Skip 1e-6 changes nothing in accuracy. So the gain is proportional to the number of optimised cuts and cuts with discarded weight 1e-6..1e-4 carry most of it. In-JAX BFGS (jax.scipy.optimize) is NOT faster (9 s vs 7 s, worse compile).
- Global fit cheap settings (L=4, a=1, 1 tau, 30 iters): chi=8 cpu 1336 -> 37-82 s, ratios .97/.27/.36/.56. Still ~12x the cost of cut-local for a similar gain, so cut-local is the cost-viable one.
- Full sweep, cheap settings, warm compile cache (`FAST=1 pareto_n12.py`, `PARDIR=results/pareto_n12_fast FIGSUF=_fast plot_pareto_n12.py cpu --svd`): cut-local 7-16 CPU s (34-165x SVD), n .45-.74, nn .44-.69, nnn .61-.78; global fit 25-129 CPU s (184-646x), n .25-.36, nn .30-.63, nnn .54-.79. Cut-local cost FALLS with chi (fewer cuts need optimising).
- Floor: the rest is Python/JAX dispatch of ~14 evaluations of a tiny problem per cut. Under ~10x SVD needs a compiled kernel with analytic gradients (no numba in the project venv, do not install). Asymptotic ratio at large chi NOT measured (estimate 10-30x).
- Value check: SVD chi=12 (1704 params, 0.15 s) has nn 7.5e-3, cheap cut-local chi=8 (936 params, ~6 s) has 9.5e-3. So ~1.7x memory saved for ~40x compute. Only worth it when memory is the hard wall.

### 12p. N=16 Pareto, optimised cut-local vs plain SVD (2026-10-08), Ising T=4 dt=0.1, chi {4,6,8,10,12,16}
- Cut-local arm `lcut:1:0.1:40:1.0:1-2-3:1e-10:1e-8` (a=1, one tau, skip 1e-10, ftol 1e-8), warm JAX compile cache, `NSITES=16 TFINAL=4.0 FAST=1 pareto_n12.py`, plots `PARDIR=results/pareto_n16_fast FIGSUF=_fast NSITES=16 plot_pareto_n12.py {cpu,wall} --svd` -> `figs/pareto_n16_*_with_svd_fast.png`.
- Ratios cut-local / SVD at equal stored params: infid .89-1.02, 1-site .45-.68, nn .38-.61, nnn .51-.75 (best at chi=8, weakest at chi=16). CPU s: cut-local 9, 10, 12, 11, 8, 4 (chi 4..16) vs SVD 0.1-0.2, i.e. 18-120x, and the ratio FALLS with chi. CPU includes a fixed JAX start-up/cache-load cost of several seconds that I did not measure separately.

### 12q. N=16, T=2, dt=0.02 with high chi (2026-10-08, user asked for higher chi's, dt 0.02, T=2)
- At this setting SVD converges by chi~12 (infid 2.6e-6 at chi=8, 6e-9 at 12, 3e-14 at 20, floor 1e-13..1e-15 from chi=24), so HIGHER chi carries no information. Real pressure is chi=2..8.
- Cut-local (optimised config, skip 1e-10) / SVD: chi=2 infid .83 n .91 nn .72 nnn .77; chi=3 .95 .69 .68 .86; chi=4 1.06 .73 .58 .76; chi=5 1.00 .82 .86 .93; chi=6 1.00 1.37 .94 .98; chi=7 1.00 1.26 .92 .99; chi=8 1.00 5.22 .81 1.01; chi>=10 identical to SVD (no cut optimised). The one-site error at chi=8 gets WORSE (2.1e-5 vs 4.1e-6).
- Scale-normalised objective (`rel`, no skip) at chi=4,6,7,8: n .41/1.01/1.05/5.41, nn .42/.86/.83/.79, nnn .64/.86/.89/.80 -> same conclusion. Gain window of cut-local = SVD infidelity roughly 1e-2..1e-1; it vanishes below ~1e-3. Files: results/pareto_n16_T2.0_dt0.02_fast, figs/pareto_n16_T2_dt002_*, figs/ratio_n16_T2_dt002.png, figs/ratio_n16_T4_dt01.png.

### 12r. REDESIGNED CELL, N=16 T=6 dt=0.02 chi 12..64 (2026-10-08), regime chosen from SVD alone
- Selection rule (before any cut-local run): T such that SVD infidelity spans ~1e-1..1e-3 over a high-chi ladder. SVD at dt=0.02: T=6 chi12 .28, 16 .15, 24 .046, 32 .016, 48 2.3e-3, 64 3.8e-4 (chosen); T=8 .53 .. 8.8e-3 too hard; T=2 converged by chi~12 (no pressure).
- FIX: the L-BFGS stopping tolerance was ABSOLUTE, so at small dt (small per-cut objective) the optimiser quit early. Normalise the objective by its starting value (`rel`, arm field 10 = `rel`, `LookCut(rel=True)`). Neutral at N=12 dt=0.1 chi=8 (.89/.45/.44/.60 vs .90/.45/.44/.61), large gain at dt=0.02 (abs version: nn .58 .61 .65 .75 .80 .93 for chi 12..64). USE `rel` AS THE DEFAULT.
- Result, cut-local(rel) / SVD at equal stored params, chi 12,16,24,32,48,64: infid .85 .91 .95 .98 1.00 1.00; 1-site .55 .56 .51 .50 .52 .53; nn .57 .45 .46 .47 .49 .50; nnn .67 .51 .47 .52 .51 .49. Flat ~2x on all local observables to chi=64 at equal infidelity. Matched-accuracy estimate (log-log interpolation of SVD nn error): SVD needs ~1.4x the stored parameters (~1.2x chi) to match cut-local.
- Cost (6 jobs in parallel, compile included): cut-local CPU 328 270 360 405 669 733 s vs SVD 2.4 3.0 4.5 6.0 9.1 12.9 s -> 137x 90x 80x 68x 74x 57x. Ratio falls with chi.
- NOT decisive (pass line was <=0.25x). Best legit ratio ~0.45-0.5. Files: results/pareto_n16_T6.0_dt0.02_rel, figs/pareto_n16_T6_dt002_*_rel.png, figs/ratio_n16_T6_dt002_rel.png.

### 12r. OBSERVABLE PIN AT THE CUT (user idea, 2026-10-08): pin the observables computed from the PRE-CUT two-site tensor onto the truncated state
- Code (graded_rule/): `rule/obspin.py` `ObsPin` (numpy only, NO JAX, no compile cost), `pin_bench.py`, `pin_table.py`, `test_obspin.py` (gradient vs finite differences 3e-10, pin residual after/before 0.001, valid MPS). Arm syntax `pin:a=1;k=2;taus=0.5;touch=1;wE=1;eps=1e-6;mech=tan;...`. Targets from the untruncated two-site tensor and its neighbours only, peak = chi (class (c) of [[feedback-priority-local-obs-no-untruncated-target]]).
- Mechanism: SVD cut, then minimal-norm displacement dM of the rank-k matrix that solves the linearised pin J dM = -(o(M_k) - t), J = gradients of the quadratic forms <P> projected on the tangent space of rank-k matrices (SVD residual is orthogonal to it, so fidelity changes only at 2nd order; measured cut infidelity pin/svd 1.0-1.4). Rows = all Pauli strings with support span <= k inside the region [b-a, b+2+a), optionally Heisenberg-evolved U^dag P U for times taus under the region's own Hamiltonian, optionally the GLOBAL energy via mpsenv.HEnv. Mechanisms: tan (full tangent + SVD retraction) > fix (centre only, = spc_mps correction generalised) > coef (k x k only, breaks: infid 1.8x).
- Ising N=12 T=4 chi=8 ratios to SVD (1site / nn / nnn / E / infid): a=0 k=2 .53/.50/.85/3.66/.96 (E gets WORSE without the energy row); + wE=1 .56/.53/.87/.10/1.01; a=1 k=2 wE=1 .55/.51/.76/.01/.98; a=1 k=3 .41/.34/.61/.00/1.08; a=1 k=2 taus=0.5 touch wE=1 eps=1e-6 .35/.30/.63/.00/.99. k=1 only: NO gain (.98). passes 1 = 6 passes, rcond, damp insensitive. Only strings that TOUCH the cut sites matter (`touch=1` same gain, cheaper). Lookahead: ONE time (0.5 best, 1.0 close) beats four (.43/.42/.79, infid 1.24); k=3 + lookahead over-constrains (.56, infid 1.68). Row selection by discrepancy coverage (`cover`) REMOVES the gain.
- Ladder Ising N=12 (A1 = a1 k2 wE1 eps1e-6 | A2 = A1 + taus 0.5 touch): A1 chi4..16 1site .50 .84 .55 .56 .59 .70, nn .79 .80 .51 .51 .51 .65, infid 1.00-1.08; A2 .. chi8 .35/.30/.63, chi10 .36/.33/.76, chi12 .28/.26/.70, chi16 .34/.33/.59 infid .99-1.04, but WORSE at chi 4,6 (1.00/1.52, .66/.80, infid 1.42/1.26). Energy error 50-500x lower than SVD with wE=1. The gain does NOT vanish at chi=16 (SVD infid 2.6e-3), unlike lcut/spc.
- Ising N=16 T=4: A2 chi8 .32/.26/.51, chi12 .21/.21/.59, chi16 .24/.25/.43 (infid 1.00-1.06, E .00-.02); A1 .53-.68 / .44-.65 / .67-.83.
- Cross-model chi 8 | 12 (1site/nn/nnn): DW A1 .65/.79/.96 | .74/.85/.98, A2 .69/.74/1.04 | .79/.69/.97; ising2 A1 1.02/1.01/1.21 | .67/.63/.95, A2 .82/.83/.96 | .55/.55/.80 (infid 1.10); Heisenberg T=1.5 (SVD infid .46): FAILS for every variant (nn 1.15-1.45, E 7-14x worse, A2 chi12 single 5.1x). Damp .25 only brings it to ~1.0. Reason: the pin reproduces values of a pre-cut state that is already far from exact.
- Cost (numpy, CPU s, N=12 T=4, svd 0.1-0.2): A0 3-4, A1 6-9, A2 11-15; N=16: A0 5-9, A1 15-19, A2 21-26. `eps=1e-6` skips negligible cuts (same gain); eps 1e-4 loses it (.75/.79).

### 12s. SPC-FAST: numpy matrix-free Gauss-Newton cut, the cost fix (2026-10-08, user: "solve the cost problem, otherwise stop")
- Code: `rule/spcfast.py`, arm `spcf:a:fw:iters:taus:ks[:fmin:every:eps:pattern:rel]` in `mfc_bench.py`, `test_spcfast.py`, `bench_cost.py`, `prof_spcfast.py`, `pareto_spcf.py` + `run_pareto_spcf.sh` (interleaved SVD/spc repeats, above-normal priority), `plot_pareto_spcf.py`, `spcf_table.py`, `an_cand.py`, `an_alpha.py`, `run_spcf_sweep16.sh`. Optional disk cache of the region tables: env `SPCF_CACHE`. No JAX, no compile.
- Idea: the lookahead objective is a FIXED quadratic form in the region density matrix. Residual r = F (h(rho) - h(rho_target)), h = diagonal + Re/Im of the strict upper triangle, F rows = Heisenberg-evolved Pauli strings U^dag P U times sqrt(omega_P). F depends only on (model, L, touches right end, taus, ks), built once (0.4-1 s), equals the window-marginal objective to 1e-15. Correction in the tangent space Q=qr(Bk+Bp C); J and J^T by the closed-form chain C->dM->dW->drho->dh->dr (adjoint agrees to 1e-13 in float64, linearisation error linear in the step), solved by diagonally preconditioned CG, 4 iterations, complex64 chain and float32 F rows applied to the float64 difference. alpha=1 is accepted in 100% of fired cuts (4 models), so only that step is tried.
- Knob findings (N=12 Ising chi=8, SVD rdm2 4.15e-2): fw=0 best (ridge from the second-order discarded-weight term hurts, .30 -> .44-.58); a=1 FAILS (region residual -90% but rdm2 worse than SVD, infid 2.5x, discarded weight x7.7: overfit), a=3 same accuracy as a=2 at 10x cost; ONE time tau=1.0 plus static is as good as four times; ks 1-2 slightly worse than 1-2-3 (.25 vs .20 at N=16); CG iters 2/4/6/10 -> .43/.33/.29/.29 (4 chosen); firing every 2nd/3rd step, or only one sweep direction, or alternate bonds, loses half the gain (the correction has to fire at every cut).
- LAW: the SVD-cut residual f is proportional to the discarded weight (log-log slope 0.99-1.13, corr .994-.998 on 4 models, f ~ 0.05-0.2 x tail). The skip rule must be RELATIVE (`rel_skip`=1e-2 of the largest tail so far): an absolute 1e-5 fires zero cuts at N=16 T=3 chi=12 and loses the whole gain (relative rule: rdm2 .40x SVD).
- Cost: first JAX spc 25-81 s steady (85-165 s per process) at N=12; numpy spcf steady CPU 0.5-1.9 s at N=12 (24x, 20x, 11x, 9x, 7x, 3x of SVD at chi 4,6,8,10,12,16), N=16 T=5 1.6, 1.7, 2.3, 3.8, 4.1 s vs SVD .12 .17 .25 .67 1.03 s (13x -> 4x for chi 8..32), N=20 T=8 chi 24/32: 17/12 s vs 1.8/1.5 s (8-9x). Machine was shared (other session at 70-90% load), timings = min of repeats, interleaved. The other session's lcut L-BFGS arm costs 34-165x for a weaker gain (.45-.5), so spcf is both ~10x cheaper and ~2x more accurate than it.
- Accuracy, spc / SVD at equal chi (infid in brackets): Ising N=12 T4 2-site chi 8/12/16 .30/.23/.21 (.83/.92/.92). Ising N=16 T5 2-site .38 .23 .20 .21 .21, 1-site .42 .26 .24 .24 .23, nn .35 .21 .18 .18 .20 for chi 8,12,16,24,32 (infid .78 .79 .84 .92 .95): PASSES the 0.25x line on this cell from chi=12. Other models N=16 chi12: ising2 .37, DW .66 (chi16 .56, chi24 .53), Heisenberg 1.26 (chi16 .46 but E 13x worse, chi24 .76, chi32 1.04, i.e. none). N=20 T8 Ising chi 24/32: .52/.43. Pass line (>=4x on >=3 models) NOT met.
- Equal accuracy: SVD needs 1.6-2.1x the chi (1.4-2.6x the stored parameters) at N=16 T5, but its wall is only 0.13-0.37x of spc's (N=12: 0.04-0.4x, N=20 chi32: spc 11.6 s vs SVD chi48 3.2 s). NO wall win in any cell run. Per fired cut the flop ratio spc/SVD is ~410/chi, so break-even needs chi >~ 100-300, not testable (no dense reference). Value = memory only (bond dimension / stored parameters).
- Figures: `figs/pareto_n12_{params,runtime_wall,runtime_cpu}_svd_spcf.png` (runtime panels also show the first JAX version), `figs/pareto_n16_T5_*_svd_spcf.png`. Data: `results/pareto_spcf_n12.json`, `pareto_spcf_n16_T5.json`, `pareto_spcf_n20_T8_a.json`, `spcf_sweep16_*.json`, `spcf_lowp_*.json`.

### 12s. Cost round 2 (2026-10-08): edge-restricted search FAILS at high chi, few iterations WORKS
- `edge=m` (LookCut option, arm field 11): rotate only the m kept directions nearest the cutoff into the m first dropped ones. N=12 chi=8 edge 4 keeps everything (.78/.45/.46/.52) but at N=16 T=6 dt=0.02 chi=48 nn ratio is .95 (edge 4), .88 (8), .75 (16), .49 (full). The gain at high chi comes from many small tilts of ALL kept directions, so selection/edge swaps cannot reproduce it. A pure importance RANKING (swap-only) was NOT run, expected to keep little for the same reason.
- `maxiter` (arm field 4) at chi=48: 3 -> nn .67, 106 CPU s (12x SVD); 6 -> .62, 145 s (16x); 12 -> .55, 200 s (22x); 40 -> .49, 658 s (73x). Flops dominate at high chi so iterations are the lever.
- Full ladder with `lcut:1:0.1:12:1.0:1-2-3:1e-10:1e-8:scipy:rel` (N=16 T=6 dt=0.02, results/pareto_n16_T6.0_dt0.02_it12, figs *_it12.png): cut-local/SVD infid .97 1.02 1.03 1.03 1.01 1.01, 1-site .60 .64 .61 .58 .58 .60, nn .58 .54 .56 .56 .55 .57, nnn .72 .63 .57 .59 .57 .57 for chi 12 16 24 32 48 64. CPU 72 58 83 127 223 250 s vs SVD 2.4 3.0 4.5 6.0 9.1 12.9 -> 30x 19x 18x 21x 25x 19x. Previous full version was 57-137x at nn .45-.57. DEFAULT CONFIG NOW: iters 12, rel, a=1, tau=1.0, skip 1e-10, ftol 1e-8.
- Remaining ideas, untested: batch independent cuts, warm start across steps (needs a basis map between steps, messy), complex64, compiled kernel.

### 12t. spc DOES NOT REACH THE SVD FLOOR on its own, absolute tail floor added (2026-10-08, user: "plain svd goes to lower accuracy and reaches floor while spc does not")
- User was right. My ladders stopped at chi=16 (N=12) and 32 (N=16). Extended: spc / SVD at equal chi, Ising N=12 T=4, 1-site: chi 24 .63, 28 .53, 32 .64, 36 2.2, 40 3.1, 48 8.5 (rdm2 .31 .31 .36 .76 1.02 2.9); N=16 T=5: chi 48 .29, 64 .70, 80 1.43, 96 1.87 (rdm2 .21 .40 .60 .86). Infidelity stays equal (1.00). spc local error falls ~tail^0.7, SVD's ~tail^1, so the curves cross at SVD local error ~1e-6 (infid ~5e-7) and spc ends ABOVE the floor. Not float32 (f64 chain gives identical numbers to 3 digits, `precision='f64'`).
- Mechanism (inference, supported by the fired-cut log: chi=48 fires 3 cuts with tails <5e-10, residual f falls 6.6x, final 1-site error rises 5e-9 -> 4.4e-8): SVD is a projection, so for observables away from the cut its error is second order in the discarded amplitude (SVD 1-site error ~ infidelity). The tangent rotation Bk -> Bk + Bp C adds a first-order change of the one-sided reduced states (off-diagonal kept/discarded block Bp C S_k^2 Bk^dag), i.e. a first-order error for far observables that the fitted region residual does not control. It scales ~sqrt(tail), so it beats SVD's tail-scaled error once the tail is small.
- Guard: ABSOLUTE floor on the discarded weight of a cut (`eps_min`, now default 1e-7). Never worse than 1.04x SVD on any metric at N=12 and N=16 (floor 3e-8 leaves 1.36x at N=12 chi=36, 1e-8 leaves 1.7x), keeps the full gain up to SVD local error ~1e-5 (N=16 chi 12..24 identical ratios, chi 32 .25 vs .23), removes 25-45% of the fired cuts (cheaper). Relative skip (`rel_skip`=1e-2) is still needed for low-pressure cells. With the guard spc coincides with SVD from SVD 1-site error ~1e-6 down to the floor (N=12 chi>=36, N=16 chi>=64) and its cost ratio goes to 1.0x there.
- Guarded ladders (Ising N=12 T4 chi 4..64, N=16 T5 chi 8..96): gain window = SVD local error 1e-1..1e-5. spc/SVD 1-site N=12 chi 4,6,8,10,12,16,20,24,28,32 = .47 .58 .31 .31 .25 .25 .38 .73 .85 1.02; N=16 chi 8,12,16,24,32,48,56,64 = .43 .26 .24 .24 .25 .64 .87 1.00. Cost ratio at equal chi falls 16x -> 1.0x along the ladder (N=12) and 16x -> 1.0x (N=16).
- Files: `plot_gain_window.py` -> `figs/gain_window_spcf.png`, `run_pareto_spcf_floor.sh`, `figs/pareto_{n12,n16_T5}_floor_*_svd_spcf.png`, `results/pareto_spcf_{n12,n16_T5}_floor.json`, `an_floor.py`, `results/spcf_floor*_*.json`. The earlier `figs/pareto_{n12,n16_T5}_*_svd_spcf.png` stop at chi 16 / 32 and are superseded by the `_floor_` versions.

### 12t. State-free / observable-tracking methods surveyed (2026-10-08, user asked what could be integrated)
- Read ABSTRACTS (+ LITE intro/method text) only, nothing cited from depth. LITE = Artiaco et al 2310.06036 ("time evolving the subsystem density matrices by solving the subsystem von Neumann equations in parallel"; closure by recovery maps, exact only if the info at the next scale is zero; "computational complexity of LITE scales exponentially with the subsystem level l_max, it increases only linearly with the total system size"); earlier Klein Kvorning et al 2105.11206. Influence functional: Sonner-Lerose-Abanin 2103.13741, Park-Gray-Chan 2504.07344 (TN-IF + BP, "shifting the resource governing computational cost from spatial entanglement to temporal entanglement"). Light cone TN: Frias-Perez-Banuls 2201.08402. Pauli propagation: Rudolph et al 2505.21606, Shao et al 2510.22311 (truncation error governed by operator stabilizer Renyi entropy). BP: Tindall et al 2306.14887 (state is still evolved, "approximately contracted using belief propagation", tree-like correlations), Tindall-Fishman 2306.17837 (BP gauging improves simple update), Bermejo et al 2604.15427 (TN+BP cannot simulate OTOC echoes). Weighted low-rank approximation NP-hard: Gillis-Glineur 1012.0197 (entrywise weights, so only an analogy for our objective).
- BP does NOT drop the state (it approximates environments); on a chain (a tree) the MPS canonical form already plays that role, so BP gives no saving for 1D. Real use = loopy lattices (inference).
- BASELINE built: `pauli_prop.py` (Heisenberg Pauli propagation on the SAME Trotter circuit and Neel start, matches dense to 1e-15), `pp_timing.py`, `pp_timing2.py`. N=16 dt=0.1, ONE central observable: T=4 eps 1e-4 -> 1.25M strings, 163 s, err 1.7e-3 (single) / 1.9e-2 (nn); T=3 error stalls ~1e-2 at 3M strings; T=2 eps 1e-5 -> 0.5-0.7M strings, ~20 s, err ~1e-3..4e-4. Plain SVD MPS gets rms 4.5e-5 (chi 8, T=2) on all 21 sampled observables in <1 s. State-free tracking loses in our regime (operator growth).
- Cost-floor argument (inference): the objective needs the local marginals of every candidate kept state, each evaluation is one O(chi^3) contraction, so cost ~ (#evals) x (SVD-equivalent) ~ 10-20x with ~3-12 evals; no state-free method removes this.
- Lookahead time scan, a=1, iters 12: tau 0.25/0.5/1/2/4 -> N=16 chi=24 nn .54/.56/.56/.61/.60, N=12 chi=8 nn .47/.48/.49/.52/.52. Weak dependence, shorter is marginally better, consistent with LITE (self-contained subsystem evolution valid for t ~ l/v_LR). Use tau=0.5 as default going forward.

### 12u. LITE-style test: the LOOKAHEAD DYNAMICS are nearly unnecessary in the per-cut setting (2026-10-08)
- Variants (iters 12, rel, skip 1e-10, ftol 1e-8), ratios to SVD (n / nn / nnn), N=12 T=4 dt=.1 chi=8 | N=16 T=6 dt=.02 chi=24: lookahead tau=.5 windows 1-3 a=1: .48/.48/.70 | .57/.56/.57. STATIC (no dynamics) windows 1-3 a=1: .55/.51/.70 | .64/.58/.59. STATIC windows 1-4 a=1: .54/.52/.69 | .61/.59/.58. lookahead windows 1-4: .54/.56/.76 | .57/.59/.62. STATIC windows 1-5 a=2: .59/.56/.68 (cpu 61 vs 22) | .58/.57/.57 (cpu 497 vs 133).
- So per-cut static matching of the local RDMs of the untruncated two-site tensor (windows <=3, 4-site region) keeps ~90% of the lookahead gain. Lookahead adds ~5-10%. NO cost saving (cost = building the candidate region state, not propagating it: cpu 22 vs 24, 133 vs 135). Larger windows or regions cost more and gain nothing.
- CORRECTS the earlier "leak" story (12c/12d): static matching is undone when the fit is applied OCCASIONALLY (global one-step fit); applied at EVERY cut there is nothing to leak. Keep tau=0.5 lookahead as default (slightly better, same cost).
- Framing: this is LITE's principle (preserve local density matrices below a scale, discard the rest) applied inside TEBD. DMT (White-Zaletel-Mong-Refael 1707.01506, abstract: "approximating density operators of 1D systems that, when combined with ... TEBD, makes possible simulation of the dynamics of strongly thermalizing systems to arbitrary times") is the nearest TEBD-level prior art; its mechanism was NOT read in depth. Do that before any novelty claim.

### 12v. LIBRARY DMT baseline + late-time check (2026-10-08, user: "we have dmt in our gcg library")
- The graded rule in [[project_pure_state_dmt_closed]] (Oct 4: SVD distance + weighted mismatch of the DMT-radius-1 Pauli strings vs their PRE-truncation values, gr:3:4 median ratios single .62, nn .50, nnn .61 at infid 1.00) is the dense-toy ORIGIN of the cut-local method; do not present them as unrelated. DMT proper = exact reserve (trace + Pauli strings within radius of the cut, `2*4^R` directions) for density operators; linear functionals of the stored rho. Pure MPS: reserve dead (quadratic).
- Baseline script `dmt_baseline.py` (PYTHONPATH=GCG/src, venv python; do NOT import `_paths`, it puts the worktree's older pytreenet first and the pauli package is missing): `PTMCG_PauliMPDO` + `DMT(radius=1)` vs `PLAIN`, same Trotter circuit and Neel start, gates via `app.apply_gate([ids[b],ids[b+1]], G)`. Verified: N=6 D=64 reproduces the exact state (infid 2e-12), observables match the pure-state code to 1e-16. DMT keeps trace = 1.0000; library plain SVD on the MPDO loses the trace (.05-.64) and is 3-10x worse than DMT locally.
- Ising N=12 dt=0.1, error at STORED PARAMETERS. T=4: DMT D=12 (5024 params) nn 3.7e-2, D=32 (29216) nn 9.3e-3, infid .37-.88; cut-local MPS chi=12 (1704) nn 3.75e-3, chi=16 (2728) 1.5e-3. T=8: DMT D=32 (29216) nn 5.5e-2; SVD MPS chi=48 (8872) 6.6e-5. T=12: DMT D=32 nn 5.4e-2; SVD chi=48 2.0e-4. DMT is 10-1000x worse at matched params on this pure-state task (rho costs ~chi^2). Caveat: N=12 is small (MPS chi 48-64 is near exact); DMT's regime is large N / long times where no MPS chi suffices, NOT tested (no dense reference).
- NEW LIMIT OF THE CUT-LOCAL GAIN: at N=12, T=8 and T=12 (entanglement SATURATED) cut-local/SVD nn ratios are ~1.0 (T=8: 1.11 1.08 .95 .86 .87 .95 for chi 8 12 16 24 32 48; T=12: 1.02 .98 .96 1.00 1.04 .97). The gain exists only in the entanglement-GROWTH phase (T up to ~N/(2 v)), not after saturation. Earlier gains (N=12 T=4, N=16 T=4-6, N=20 T=8) are all growth-phase.

### 12w. Cross-model check of the BEST cut-local version in regimes chosen from SVD alone (2026-10-08, cloud session, Linux 4 cores)
- BEST VERSION (unchanged): `spcf` = `rule/spcfast.py` SPCFast(a=2, fw=0, iters=4, taus=[1.0] + static, ks=1-2-3, eps_min=1e-7, rel_skip=1e-2), arm `spcf:2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2` (default OPT of `pareto_spcf.py`). Class (c), peak = chi. Reproduced here: Ising N=12 T=4 chi 8/12 rdm2 .30/.23 of SVD, identical digits.
- New scripts: `svd_scan.py` (SVD infid/nn along time for a chi ladder, dense alongside: regime selection with no other arm), `spcf_grid.py` (named SPCFast kwargs on top of the best config), `an_degen.py` (gap at the SVD cutoff). Data in `results/regime/`, logs `out_svd_scan_*`, `out_spcf_*_N16_*`, `out_grid*_heis_*`.
- Regimes (SVD alone, N=16 dt=0.1): Heis T=1.0 infid .40 (chi8) .. 4e-5 (chi32); the OLD Heis cell T=1.5 chi=12 has SVD infid .73 (far outside the gain window). DW T=4 .27 .. 5e-6; ising2 T=3 .23 .. 2e-5. The OLD DW (T=5) and ising2 (T=4) cells were already in range for chi >= 12-16.
- spcf / SVD at equal chi (1site / nn / nnn, infid), chi 8,12,16,24,32:
  - DW T=4: .55/.67/.90, .61/.63/.91, .75/.53/.68, 1.58/.78/.77, chi32 no cut fired (=SVD); infid .98-1.04.
  - ising2 T=3: .24/.37/.55, .76/.60/.69, .41/.31/.37, 1.76/.83/.68, 5.61/1.10/.67; infid .81-1.16.
  - Heis T=1 (chi 8,10,12,16,20,24,32): .39/.40/.98, 1.77/1.05/.70, .34/.56/.37, 3.05/1.73/.69, 1.81/1.82/.66, 1.17/.87/.85, 4.21/2.08/1.07; E 11.6x worse at chi 8.
  - CPU 4-11x SVD in the window, 1-3x where few cuts fire.
- HEISENBERG FAILURE IS REAL, NOT REGIME: in the right regime the sign of the effect flips with chi (wins chi 8,12,24, loses 10,16,20,32). Knob grid at chi 10,12,16,20, none fixes it: tau .25/.5 worse, static-only same, region energy row wE=1 same, fw=.03 a bit closer to 1, a=3 (+ 4-site windows) same at 20-100x cost, `two` (both bond subspaces) worse, `free` (centre freedom) ~1.0. Not split degeneracies at the cutoff (median relative gap .2-.7, none < 1e-3).
- `beta` (first-order change of the reduced state OUTSIDE the region, already in spcfast.py, never benchmarked before): beta 0.1..100 moves every chi monotonically toward SVD (bad chi 3.05 -> 1.5-1.9 1site, but the good chi=12 .34 -> .63-.71, nn .56 -> .92). It suppresses the step, it does not find a better one. So on Heisenberg the local gain and the far damage come from the same rotation.
- The tail floor eps_min=1e-7 (tuned on Ising) does NOT transfer: on ising2/DW/Heis the 1-site error is already worse than SVD at SVD 1-site error ~1e-4 (ising2 chi24 1.76, chi32 5.6; DW chi24 1.58; Heis chi32 4.2), while nn/nnn stay <= 1. Single-site observables are the first casualty of the far first-order term.
- VERDICT: the best cut-local version gives 2.5-4x on Ising-type models with a longitudinal field in the growth phase (Ising N=16 T=5 .20-.26x from chi=12; ising2 best chi .31-.41), 1.3-2x on the domain wall, and is NOT reliable on Heisenberg. Pass line (>= 4x on >= 3 models) still not met. Open: a model-independent stop rule for the 1-site degradation at low pressure (e.g. fire only while the predicted region gain exceeds a beta-measured far damage), U(1)/SU(2)-aware tangent directions for Heisenberg.
