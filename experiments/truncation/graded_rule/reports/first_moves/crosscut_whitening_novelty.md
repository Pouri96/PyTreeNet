# Cross-cutting enablers: closed-form whitened truncation, and the Paeckel [73]/[74] novelty check

## Summary verdict

**Novelty check: resolved, no threat. Whitening: conditional.**

Paeckel et al. [73] and [74] are not pure-state, local-RDM-targeted MPS truncations:
- [74] is DMT itself (White, Zaletel, Mong, Refael, PRB 97, 035127 = arXiv:1707.01506).
- [73] is Stoudenmire's tensornetwork.org note on the density-matrix algorithm for MPO-MPS products. That algorithm is a fidelity-optimal truncation.

The same review's outlook (Sec. 9) explicitly suggests "combining e.g. the 2TDVP method with a truncation scheme which also ensures energy conservation [74]". So the spcf idea was flagged as an open direction in 2019. Our searches found nobody who has carried it out for pure states.

The closed-form whitened SVD is conditional, for two reasons:
- **The mechanism is well precedented.** Evenbly 2018, Jeannin–Chessari–von Delft 2026 and SVD-LLM all whiten with a separable (Kronecker) metric and then take an ordinary SVD. The only open question is whether spcf's local-RDM Gauss-Newton metric is close enough to Kronecker form.
- **The report's "1-2x SVD" target is not reachable with spcf's current region formulation.** The logged time split of spcf (DW N=16) is: SVD 5%, setup 23%, CG solve 48%, line search 22%. Each evaluation of the 6-site region objective costs about one SVD of the two-site matrix. Removing the iterative solve alone takes spcf from roughly 10x to about 5x SVD.
- **Reaching 1-2x requires a target-free metric built only from the two-site core and transferred local operators.** "Target-free" means it needs no region state and no residual. We would also have to skip the acceptance check. Whether such a core-only metric keeps the gain is the decisive unknown. The earlier failure at region margin a=1 (overfitting) is the main warning sign.

The first test is cheap: an oracle Kronecker-factorization probe on captured cuts. If even the best Kronecker approximation of the exact Gauss-Newton metric keeps less than about 35% of the one-step objective drop, drop the closed form. In that case, use the Kronecker factors only as a CG preconditioner.

## Literature, verified from full text

Every paper below was downloaded with `read_arxiv_paper` (save path `/tmp/papers_crosscut_whitening_novelty`) and its extracted text searched. "Read" here means the targeted sections, not cover to cover.

**Paeckel, Köhler, Swoboda, Manmana, Schollwöck, Hubig, review 1901.05824 (full reference list read).**
- The passage is at the end of Sec. 2.8 (MPO-MPS products, "Our experience" paragraph): "Further interesting possibilities may be the truncation based on the left/right density matrix[73] or by optimizing the local density matrix[74] which aims at preserving local observables."
- References: [73] "E. M. Stoudenmire, MPO-MPS Multiplication: Density Matrix Algorithm, http://tensornetwork.org/mps/algorithms/denmat_mpo_mps/"; [74] "C. D. White, M. Zaletel, R. S. K. Mong, G. Refael, Quantum dynamics of thermalizing systems, PRB 97 (2018) 035127".
- Sec. 9 (Future developments): "Combining e.g. the 2TDVP method with a truncation scheme which also ensures energy conservation[74] may prove particularly fruitful."

**DMT, White et al. 1707.01506 (Secs. I-IV read).**
- The guarantee is stronger than "≤3-site". Each truncation at bond j preserves "the trace … the reduced density matrix ρ_{1···j+1} … and ρ_{j···L}" (Sec. III, Fig. 3). Only operators that reach both ≤ j−1 and ≥ j+2 can change, hence "no truncation will change the expectation of any operator on three contiguous sites".
- Positivity is only "ameliorated" (abstract and intro), not guaranteed. This answers one of the report's open questions.
- The pure-state benchmark (Sec. IV B) uses Eq. (30) with the near-σʸ initial state of Eqs. (32)-(33), g_j = ±0.1 in blocks of 4. It compares against MPS TEBD at χ = 2^{⌊L/2⌋}, with "errors … ≈ 10⁻³ for a wide range of bond dimensions".

**Ye, Machado, White, Mong, Yao 1902.01859 (main text and supplement).**
- Bond accounting is resolved: χ_preserve = 2·4ⁿ with preserved diameter ℓ = 2n+1, so χ_preserve = 2^ℓ (8 for ℓ = 3). Supplement: "to guarantee this requires the bond dimension χ ≥ χ_preserve = 2 × 4ⁿ".
- The leak sentence is verified verbatim in the supplement: "errors in longer ranged operators propagate down to the three-site density operators via the system's dynamics".

**rTEBD, Guha Roy & Slagle 2412.08730 (v. May 19 2026; Google Scholar lists SciPost Phys. 21, 015).**
- Authorship confirmed.
- Fermion benchmark: L = 128, δt = 0.08, γ = 1.5 ("found to work better than γ = 2 or higher").
- Interacting spin chain (Eq. 25-26): J = 1, h_x = 0.9045, h_z = 0.8090, the same as DMT. L = 64, with γ = 1.6 chosen (Fig. 13: 1.6-1.7 best).
- At χ = 128, rTEBD's energy drift is "nearly an order of magnitude" below MPS-TEBD at long times, but MPS-TEBD is better at short times.
- rTEBD is MPDO-only, with no pure-state variant. Comparison with DMT is explicitly left to future work.

**Evenbly 1801.05390 = PRB 98, 085155 (ID confirmed; Secs. II-V and App. C read).**
- The weighted-trace gauge whitens the bond environment into left/right boundary matrices (ρ̃_L ∝ I, ρ̃_R ∝ I), after which Schmidt truncation is near-optimal.
- When the environment is not separable, "full environment truncation" (FET) alternates generalized-eigenproblem updates of u·σ̃ and σ̃·v†, initialized from the truncated SVD.
- The objective is fidelity, not local observables. It is the closest structural precedent: whiten with separable factors, SVD, optionally refine by ALS.

**Jeannin, Chessari, von Delft 2610.10027 (Sec. on truncation read).**
- Eq. (42) states the truncation as a weighted low-rank problem in the two-site norm metric N.
- Sec. B: "When the two-site norm metric is sufficiently separable … the truncation can be approximately performed using an ordinary SVD in the transformed frame" (Eqs. 43-44).
- Sec. C refines by ALS. Their references [50]-[52] are Srebro–Jaakkola (non-convex, local minima), Markovsky–Usevich, and Gillis–Glineur (NP-hard).
- **Correction:** the report attributes this paper to a "metric-aware truncation reduces to SVD whenever the metric is separable" claim without authors. The claim is right, but it is stated as approximate ("sufficiently separable"), and the metric is the norm, not an observable metric.

**SVD-LLM, Wang et al. 2403.07378 (ICLR 2025; Sec. 3.1 read).**
- The whitening S is the Cholesky factor of XXᵀ. W is compressed as U·Trunc(Σ)·Vᵀ·S⁻¹, and Theorem 3.2 / Corollary 3.3 give loss = √(Σ truncated σ²).
- **Implicit requirement:** XXᵀ must be positive definite. The SVD-LLM text in our extraction says nothing about regularizing it.

**2502.02723 is Dobi-SVD (Wang, Ke, Tomizuka, Chen, Keutzer, Xu; ICLR 2025), not an "ASVD paper".**
- **Correction:** the quote "S⁻¹ often fails due to theoretical and numerical issues" is in its App. A.4, describing ASVD and SVD-LLM.

**2408.05104 is Carere & Lie, not Friedland–Torokhti.**
- **Correction:** it generalizes Friedland–Torokhti (SIAM J. Matrix Anal. Appl. 2007) and Sondermann (1986), solving min ‖M − BXC‖ with rank(X) ≤ r via Moore–Penrose inverses.
- This is the tool for a rank-deficient whitener: the closed form stays defined when L or R is singular.

**CorDA 2406.05223 (Yang et al., KAUST; intro and method grepped).**
- SVD(WC) with C the input covariance, then "the inverse of these covariance matrices is multiplied with the decomposed components". This confirms the report's description.

**Also read: Surace–Piani–Tagliacozzo** (ID 1810.01231 confirmed; Google Scholar gives PRB 99, 235115), **and Hémery–Pollmann–Luitz 1901.05793.** Both were searched for the unattributed sentence.

**Not read in full:** Srebro–Jaakkola and Manton–Mahony–Hua (not on arXiv; known only via citation in 2610.10027), the Stoudenmire web note [73] (described from the review's citation and prior knowledge), and Gillis–Glineur 1012.0197 (abstract seen in search results).

**The unattributed sentence** ("only local observables with support on up to two neighboring sites remain unaffected") does not appear in any of the six physics full texts above. An exact-phrase Google Scholar search returned nothing. It also contradicts DMT's verified three-site guarantee. Treat it as a garbled search snippet and do not cite it. If a statement is needed, cite Ye et al.'s verified "propagate down to the three-site density operators".

## Prior-art / novelty risk

1. **The core spcf claim stands.** No source read or found does a pure-state MPS truncation that preserves local RDMs:
   - Paeckel [73] is fidelity-optimal and [74] is the MPDO method.
   - DMT and rTEBD are MPDO-only.
   - Surace et al. trade entanglement for mixture.
   - The review's Sec. 9 sentence is a *suggestion* in a widely cited review, so reviewers will see the idea as anticipated. Cite it as motivation, not as a threat.
2. **The whitened-SVD route has low novelty as mechanism, moderate as application.** Kronecker-separable whitening, SVD, then optional ALS refinement is established in TN (Evenbly 2018; Jeannin et al. 2026, citing Vanhecke et al. SciPost Core 4, 004 and Sinha et al. PRB 106, 195105) and in LLM compression (SVD-LLM, CorDA). Every TN instance found whitens the norm/fidelity metric. Whitening with a local-observable Gauss-Newton metric, with a Frobenius (fidelity) anchor, was not found. Our claim would be "observable-metric whitening for pure MPS", and it must cite Evenbly and Jeannin et al. as the mechanism.
3. **Search coverage caveat.** `search_semantic` and `search_openalex` returned empty results throughout, and `search_arxiv` returned unrelated papers for every query. Prior-art searching rested on Google Scholar plus reference lists of full texts. A forward-citation trawl of 1707.01506 is still undone (Google Scholar "cites" queries returned nothing).

## Technical details for implementation

**What the Gauss-Newton metric is.**
- The residual is r = F·h(ρ_region(M_k)) − F·h(ρ_region(M)), with M the 2χ×2χ two-site matrix.
- Linearized about M, ‖r‖² ≈ vec(ΔM)† G vec(ΔM) with G = J†J and ΔM = M_k − M. Truncation is then weighted low-rank approximation: NP-hard in general (Gillis–Glineur), closed form only if G = L⊗R.
- Add the fidelity anchor, G_λ = λI + μG. Using whitening factors Lλ = λI + μL and Rλ = λI + μR, the closed form is X = SVD_k(Lλ^{1/2} M Rλ^{1/2}) and M_k = Lλ^{−1/2} X Rλ^{−1/2}.
- Keep only the column space Q of M_k, then recompute the centre by Galerkin projection, Q†M (as `retract` already does). This keeps the code's isometry + centre form, and only the subspace changes.
- λ > 0 is the remedy for the S⁻¹ failure Dobi-SVD describes (as with GPTQ's damping). Note that spcf's best setting is fw = 0 (any ridge hurt: .30 → .44-.58), so μ/λ must be large and its scan must reach the ill-conditioned end.
- The whitened SVD needs only the metric, not the residual r0. This is what makes a target-free, cheap version possible at all.

**Kronecker factors, three grades.**
- (a) Oracle, for small χ: build J densely, which costs nx jvps (nx = 2·nb·k real; 128 at χ=8, 288 at χ=12). Take the Van Loan–Pitsianis nearest Kronecker product. In tangent coordinates the split is (Re/Im × discarded index) ⊗ (kept index). In M-space it is row (l,σ_b) ⊗ column (σ_{b+1},r).
- (b) Shampoo-type factors: L = Σ_i Γ_i Γ_i† and R = Σ_i Γ_i† Γ_i, where Γ_i is residual row i reshaped to 2l×2r.
- (c) Target-free core version, for static strings only:
  - With canonical environments, a string A⊗B split at the cut has ⟨A⊗B⟩ = Tr(M† Ã M B̃ᵀ), with Ã = transfer of A through at most one left tensor (2l×2l) and B̃ the same on the right. So Γ_{A,B} = Ã M B̃ᵀ.
  - For a product target set, L = Σ_A w_A Ã M (Σ_B w_B B̃ᵀB̃*) M† Ã†. This costs O((n_A + n_B)·(2χ)³), roughly 1-3 SVD equivalents for 1-2-site strings on each side.
  - The time-evolved (τ = 1.0) rows break the product structure. Handoff §12u says static matching keeps about 90% of the gain, so drop them in grade (c).

**Cost model, from the code and logs.**
- `Wof` (building the region state) costs about 16χ³ + 64χ³ complex multiply-adds, comparable to one SVD of the 2χ×2χ matrix.
- A spcf fired cut does about 2-3 of these in setup, about 9 in CG (4 iterations, plus the initial vjp), and 1-3 in the line search. That is 13-16 SVD-equivalents, consistent with the logged 4-16x CPU and with "maxiter 3 → 12x, 6 → 16x" at χ = 48.
- The logged split is `{'svd': 0.41, 'setup': 1.9, 'solve': 3-5.6, 'line': 1.8}` (`out_spcf_sweep16_dw.txt`).

**Benchmarks and metrics to reuse.**
- Best spcf configuration (handoff §12w): `spcf:2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2`, i.e. a = 2, fw = 0, iters = 4, τ = {0, 1.0}, ks = 1-3, eps_min = 1e-7, rel_skip = 1e-2.
- Reference ratios to SVD (spcf, 2-site RDM error):
  - Ising N=12, T=4, χ = 8/12: .30/.23.
  - Ising N=16, T=5, χ = 8, 12, 16, 24, 32: .38, .23, .20, .21, .21.
  - Heisenberg is not a win.
- Per-cut diagnostics:
  - `_low_span_resid`: span-2 residual, which is the "repairable" error.
  - `diag_cut.py`: far ⟨Z⟩ collateral, the cost side. Its first-order scaling is ∝ tail^0.65.
- Known failure modes:
  - Margin a = 1 overfits: region residual −90% but RDM error worse, infidelity 2.5x.
  - Any ridge hurts.
  - Span-1 strings get 3.7-7x worse.
  - The gain lives only in the pressure window where SVD's local error is 1e-1 to 1e-5.

## Where to start

1. **`whiten_probe.py`** (new; about 200 lines, half a day). It subclasses `spcfast.SPCFast` and wraps `__call__` to stash (theta, dirn, U, s, Vh, k, Lm, Rm) next to the existing `debug` entry, which already holds jvp, vjp, exact, r0 and f_svd. Per captured cut it computes these arms:
   - A0: SVD.
   - A1: spcf as is (CG, 4 iterations).
   - A2: exact GN step (dense lstsq).
   - A3: oracle-Kronecker GN step.
   - A4: Shampoo-factor GN step.
   - A5: whitened SVD in M-space with Shampoo factors, μ/λ ∈ {1, 10, 10², 10³, 10⁴}.
   - A6: Kronecker-preconditioned CG at 1-2 iterations.

   Map a subspace Q to tangent coordinates via C = (Bp†Q)(Bk†Q)⁻¹, so `exact(x)` scores every arm on the same objective.
2. **`rule/spcwhite.py`, class `SPCWhite(SPCFast)`** (about 1 day). It overrides `__call__` with:
   - `mode='kfac'`: one vjp, Kronecker solve, one accept evaluation.
   - `mode='core'`: target-free grade (c), with no `_region` or `Wof`, and an optional accept check.
3. **Bench arm `spcw:a:mode:mu:lam[:accept]`** in `mfc_bench.py`, next to the `spcf` branch at line 75 (about 2 hours). Reuse `pareto_spcf.py` and `run_pareto_spcf_floor.sh` for interleaved timing.

Total about 2 days of coding plus at most 3 hours of compute for the tests below. No files outside this folder change.

## First tests (decision tree)

**T0. Time split at high χ (15 min).** Run `prof_spcfast.py ising 16 5 {24,48} 2 0 4 0-1.0` and record `cut.tm`.
- Pre-registered: if setup + line exceeds 40% of spcf time at χ = 48, then `mode='kfac'` (which keeps the region) is capped at about 2-2.5x cheaper than spcf, and only `core` can reach ≤ 2x SVD.
- This only sets which arm T2 must succeed with. Nothing is killed here.

**T1. Oracle factorization probe** (Ising N=12 T=4 χ=8 and N=16 T=5 χ=12; about 200 fired cuts each; under 30 min).
- Metric: per-cut retention ρ = (f_SVD − f_arm)/(f_SVD − f_GN*) on the exact objective. Also the span-2 residual ratio to SVD, and the discarded-weight ratio.
- **Success:** median ρ(A3) ≥ 0.6 and median ρ(A5, best μ) ≥ 0.5, with span-2 residual ≤ 0.6x SVD in ≥ 70% of cuts. For comparison, spcf's own span-2 residual is 0.28x SVD in Ising.
- **Kill (closed-form route):** median ρ(A3) < 0.35, meaning the metric has no Kronecker structure. Then pivot to A6: if Kronecker-preconditioned CG at 1-2 iterations matches A1, adopt it as a pure speed fix (expected about 1.5-2x faster than spcf).
- **Ambiguous:** ρ(A3) between 0.35 and 0.6, or A3 passes while A5 fails. Then only the GN-step form (`kfac`) is viable, not the target-free whitened SVD. Go to T3 with `kfac`.

**T2. Target-free core whitener** (only if T1 passes; about 1 hour). Time `mode='core'` per cut against `np.linalg.svd` on the same M at χ ∈ {16, 32, 64, 128}, single BLAS thread. Score its retention on the T1 cuts.
- **Success:** ≤ 2x SVD per cut at χ ≥ 32, and median ρ ≥ 0.5.
- **Kill:** ρ < 0.3. The region metric is then essential, and the honest cost floor is about 4-6x SVD, set by `kfac`. Report it as such and stop promising 1-2x.
- **Ambiguous:** ρ between 0.3 and 0.5. Test whether adding the one exact accept evaluation (+1 SVD-equivalent) fixes it, as a 3x variant.

**T3. End-to-end** (about 1-2 hours of CPU). Arm `spcw` against spcf and SVD on:
- Ising N=16 T=5 at χ = 12, 16, 24.
- ising2 N=16 at χ = 12.
- Heisenberg N=16 T=1 at χ = 12, 16, as a negative control (gate on).

Metrics are the 1-site / nn / 2-site RDM ratios to SVD, the infidelity ratio and the CPU ratio, against `results/pareto_spcf_n16_T5_floor.json`.
- **Success:** retained log-gain g = log(ratio_spcw)/log(ratio_spcf) ≥ 0.6 at CPU ≤ 2.5x SVD. Example: Ising χ = 16 at 2-site ≤ .38, where spcf gets .20.
- **Kill:** g < 0.3 at every χ, or CPU > 0.5x spcf.
- Also report break-even: the χ reduction r at equal error, compared with r³ > CPU ratio.

## Open questions

1. Does the far-observable collateral of a whitened SVD scale like spcf's tilt (first order, ∝ tail^0.65)? Because Q is chosen from a reweighted problem rather than as a tangent step, it might be smaller. This decides Heisenberg.
2. Does the target-free core metric (a ≤ 1 support) fall into the a = 1 overfitting trap, or does the λ anchor prevent it?
3. Can the time-evolved (τ = 1.0) rows be put into product form, for example through a short-time operator expansion, or is static-only good enough? Handoff §12u says static gives about 90%.
4. Is one ALS sweep after the whitened SVD (Evenbly FET / Jeannin Sec. C style) a cheaper refinement than CG?
5. A forward-citation trawl of 1707.01506 and of Paeckel Sec. 9 for pure-state DMT variants is still undone. The search tools above could not provide it.
6. A feasibility sanity run (importing `spcfast` and capturing a Jacobian at N = 10) could not complete. The machine had load about 16 on 4 cores, and the import alone exceeded 60 s. The sizes quoted above (nx = 128-288) come from the code, not from a run.
