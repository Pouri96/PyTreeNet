# Rank 2 first move: spcf on purification MPS for real-time finite-T dynamics

## Summary verdict: conditional

The direction holds up, but the full texts move the bar. Four findings matter most.

1. **The novelty threat is gone.** The Paeckel review's ref [74] is DMT itself, and ref [73] is Stoudenmire's density-matrix MPO-MPS algorithm. No full text or search turned up a purification-side truncation that targets local physical observables.
2. **The baseline is stronger than the report assumed.** DMT's own paper (1707.01506, Sec. IV C, App. D/E) finds that plain SVD truncation of a purification with backward-evolved ancillas (KBM) converges "with approximately the same bond dimension vs. accuracy tradeoff" as DMT, for both near- and far-from-equilibrium mixed initial states.
3. **What spcf has to beat.** It must beat SVD on the purification in its best gauge. That baseline already roughly equals DMT. If spcf clears it, the DMT head-to-head is likely, but not guaranteed, to follow.
4. **There may be nothing to repair at high temperature.** KBM (1111.4508) shows that backward ancilla evolution makes the evolution of a thermal state trivial. Near equilibrium at high T, SVD's local error may simply be too small for spcf to fix.

The first move is therefore cheap. A d = 4 spcf cut is roughly a day of work. Then a scan from pure to mixed (μ from ∞ to 0) settles whether a usable error window exists at genuinely mixed μ before any DMT code is written.

## Literature, verified from full text

All full texts were fetched with `read_arxiv_paper` and are cached in `/tmp/papers_rank2_purification`, with text dumps under `txt/`.

**DMT: White, Zaletel, Mong, Refael, [arXiv:1707.01506](https://arxiv.org/abs/1707.01506). Full text read.**
- *What is preserved is more than the report said.* Sec. III guarantees that a truncation at bond j leaves tr ρ, **ρ_{1…j+1} and ρ_{j…L}** unchanged, i.e. each half-chain plus one site. The "all operators on ≤3 contiguous sites" statement is a corollary of that, not the definition. The construction is: QR of tr[x_Lα σ^μ_j] (χ×4) and the right-hand analogue; leave the top 4 rows and left 4 columns of M untouched; SVD the lower-right block of the *connected* matrix M̃ = M − M_{α0}M_{0β}/M_00 to rank χ′.
- *Bond accounting.* "The matrix M′ has rank at most 8 + χ′" (App. B). The total bond is therefore χ′ + 8 for ℓ = 3.
- *Positivity is not guaranteed.* The paper says DMT "ameliorates" the positivity problem, and that "Z ≥ 1 for DMT does not imply positive semi-definiteness" (Sec. IV B, Fig. 5). The report's wording "ameliorates" is correct, and positivity is not exact.
- *Benchmarks.* The Hamiltonian is H = Σ S^z S^z + (h_x/2)ΣS^x + (h_z/2)ΣS^z with h_z = 0.8090 and h_x = 0.9045. **That is the repo's `ising` model divided by 4**, so one unit of time there equals 4 repo units. The runs use the boustrophedon Trotter decomposition with **δt = 1.0** (repo units 0.25) and a fixed χ_max cap.
  - Pure near-σ^y initial state, L = 16-64, χ up to 1024: errors in ε_{k=π/4} are about 10⁻³ (Fig. 8).
  - Mixed states: a near-equilibrium Gibbs state (Eq. 36) and a far-from-equilibrium Gibbs quench (App. D, from h = (0.5, 0.5) to (2.0, 0.5)), both at L = 128 with χ = 16, 128, 512.
  - Purification (KBM) and DMT give "essentially identical results … even to quite small bond dimensions" (Figs. 9-12). Frobenius MPDO "converges more slowly than either method" (App. E).
  - The observables are only ε_{k=π/4}(t) and S^z_{L/2}(t).
- *DMT does not converge for the pure state.* Accuracy stops improving above χ ≈ 2⁵-2⁶, attributed to the Frobenius choice of the χ′ block.
- **Prior art for the grading idea.** The Conclusion says: "if we truncate at bond j, we should weight errors along σ^z_{j−1}σ^z_{j+2} more heavily than errors along σ^z_{j−7}σ^z_{j+6}. Such controlled-metric truncation is a natural extension of this work." This must be cited for both the graded rule and spcf's locality-weighted objective.

**Ye, Machado, White, Mong, Yao, [arXiv:1902.01859](https://arxiv.org/abs/1902.01859). Full text read, including the supplement.**
- The main text gives χ = χ_preserve + χ_extra with "χ_preserve = 2^ℓ". The supplement (around Eq. 16 and Fig. 8) derives χ_preserve = 2·4ⁿ for preservation diameter ℓ = 2n + 1, which is 2^ℓ. **The report's 2^ℓ is right, and 4^ℓ is wrong.** For ℓ = 3 this gives 8, consistent with χ′ + 8.
- **χ_preserve is counted inside the cap χ.** At an equal cap, DMT therefore spends only χ − 8 singular vectors on the Frobenius-optimal part.
- "Tuning ℓ at fixed χ can also affect the accuracy" (Fig. 3b).
- The leak sentence is verbatim from the supplement: errors in longer-range operators "propagate down to the three-site density operators via the system's dynamics."
- Benchmarks: χ = 32-180, L up to 100, Trotter convergence at L = 20, χ = 128, ℓ = 3.

**Hauschild et al., [arXiv:1711.01288](https://arxiv.org/abs/1711.01288). Full text read.**
- The "macrostate vs. microstate" quote is real, but it describes **TDVP (Leviatan et al.) and DMT, not purification methods**. Hauschild's own method "attempts … to compute the exact micro-state". The report's attribution needs that nuance.
- Fig. 5a, for |S⁺_{L/2}(t, β = 0)⟩ in a near-integrable Ising chain (J_x = h_z = 1, J_z = 0.1, L = 40): the max-over-bonds entropy "grows roughly linear in all three cases, yet with very different prefactors" (no disentangler, backward, optimized). This is **verified**.
- Further facts:
  - An entropy-minimizing disentangler gives "a slightly longer tail of small Schmidt values", so it does not reduce χ.
  - A norm-based disentangler reduces χ slightly, but "the optimization itself is computationally more expensive than the speed-up gained".
  - For the clean Heisenberg chain with backward ancillas, S ∝ log t (Fig. 6a). That predicts **spcf failure on `heis`**, consistent with the repo's state-TEBD result.
  - Truncation in this paper is plain SVD at ε = 10⁻⁶ per step.

**Karrasch-Bardarson-Moore.** **The report's ID 1205.3756 is wrong.** That arXiv number is a polar-codes paper (Sutter, Renes, Dupuis, Renner). The correct IDs are **[arXiv:1111.4508](https://arxiv.org/abs/1111.4508)** (PRL 108, 227206; full text read) and 1303.3942 (NJP 15, 083031; not read).
- The auxiliaries are evolved with U_aux = e^{+iH̃t}, which "renders the time evolution of |Ψ_β⟩ trivial". Entanglement then builds up only around the local perturbation.
- For the XX chain at L = 100, χ reaches only about 250 by tJ ≈ 30 at a total discarded weight of 10⁻⁶. The traditional approach breaks down by tJ ≈ 7-9 even at χ = 60-200.
- Truncation is plain SVD with a discarded-weight threshold.

**Barthel, [arXiv:1301.2246](https://arxiv.org/abs/1301.2246). Full text read, Secs. I-IV.**
- A purification |X⟩ is isomorphic to an MPO X̂ with ρ = X̂X̂†, and the truncation weight is the Frobenius norm of X̂ (Eq. 8).
- The schemes evolve Heisenberg operators to reach about 2× longer times.
- Setup: XXZ at L = 128, 4th-order Trotter with Δβ = Δt = 1/8, ε_β = 10⁻¹², ε_t = 10⁻¹⁰.
- Consequence for us: **spcf on a purification is a truncation of X̂ whose target, Tr_out(X̂X̂†), is quadratic in X̂.** That is spcf's native class.

**Binder & Barthel, [arXiv:1411.3033](https://arxiv.org/abs/1411.3033). Full text read, abstract and Secs. I-IV.**
- The quote is verified verbatim: "For almost all considered cases … purifications yield more accurate results than METTS – often by orders of magnitude". METTS wins only well below the gap.

**Paeckel et al., [arXiv:1901.05824](https://arxiv.org/abs/1901.05824). Reference list and the relevant paragraph read.**
- The sentence is "truncation based on the left/right density matrix[73] or by optimizing the local density matrix[74] which aims at preserving local observables."
- [73] is E. M. Stoudenmire, *MPO-MPS Multiplication: Density Matrix Algorithm*. [74] is **White et al., PRB 97 (2018), i.e. DMT**.
- **The report's top open check is resolved.** Neither reference is a pure-state, local-RDM truncation.

**rTEBD, Guha Roy & Slagle, [arXiv:2412.08730](https://arxiv.org/abs/2412.08730). Full text read; now SciPost Phys. 21, 015 (2026).**
- The authorship is confirmed.
- It works on MPDOs only, with no purification or pure-state variant.
- **There is no DMT comparison** ("we plan to compare rTEBD with … DMT").
- Recommended γ is about 1.5 for fermions and 1.6 for spins. The free-fermion benchmark is at L = 128.
- They warn that reweighting "can suppress long-range fermionic coherence, even when local observables are well converged."

**Abstract or snippet level only (not read in full):**
- Mendl, 1812.11876: MPO-TDVP augmented by H to conserve tr[HO]. This is a linear target.
- Mc Keever & Szymańska, 2012.12233: iPEPO "optimal truncations … by optimising an objective function appropriate for open systems". Probably a fidelity-type objective; to be checked.
- Godinez-Ramirez, Milbradt & Mendl, 2409.08127: LPDO Riemannian compression.
- Müller, Ayral & Bertrand, 2403.00152: LPDO noisy circuits with an optimized ancilla unitary.
- Jaschke, Montangero & Carr, 1804.09796: a review comparing trajectories, MPDO and LPTN.

## Prior-art and novelty risk

**Overall risk: low-moderate.** This is unchanged in level, but now better grounded:
- No paper found truncates a **purification** with an objective built from **physical local RDMs**.
- Every purification work read (KBM, Barthel, Binder-Barthel, Hauschild) truncates by SVD discarded weight. Their innovations live in the ancilla gauge (backward evolution, disentanglers) or in the evaluation scheme, not in the truncation criterion.

**Residual risks:**
1. **DMT's "controlled-metric truncation" remark.** The idea of locality-weighted truncation is explicitly suggested in 1707.01506. spcf's claim must be the *mechanism*: a fixed-χ Gauss-Newton tilt on a pure or purified state, with quadratic targets. The *idea* of weighting short-range operators is not ours to claim.
2. **The near-linear regime at high T.** With backward ancillas at high T, X̂ ≈ c(1 + δ), so ρ_phys ≈ c²(1 + δ + δ†) is close to *linear* in δ. In this regime a DMT-like exact reserved-block construction applied directly to X̂ could nearly work. That would erode the linear-vs-quadratic dividing line exactly where finite-T transport is studied.
   - This needs a sentence in any write-up.
   - It is also an opportunity: a cheap linearized, closed-form variant.
3. **The unread LPDO and iPEPO compression objectives** (Mc Keever 2012.12233; Godinez-Ramirez 2409.08127) should be skimmed before writing anything up. Both are positivity-preserving compressions with an ancilla or Kraus leg, structurally the same object.
4. **The Ye et al. remark** that "carefully choosing the operators which are preserved" helps at fixed χ is adjacent motivation, not an implementation.

## Technical details for implementation

**Object.**
- Fused site plus ancilla with d = 4, index order (p, a).
- Initial states are products of tilted Bell pairs, with ψ_i = √p↑|↑↑⟩ + √p↓|↓↓⟩ and p↑ = e^{μ_i}/(2 cosh μ_i), so ρ₀ = ∏ e^{μ_i Z_i}/Z.
- Two families, each with a pure-state limit already measured in this repo:
  - **staggered**, μ_i = (−1)^i μ, which tends to the Néel `ising` cell (spcf wins 2.5-5×);
  - **domain wall**, μ_i = ±μ, which tends to `isingdw` (spcf wins about 1.6× on rdm2 at χ = 8-16, `out_spcf_dw_N16_T4.txt`).
- The dial μ ∈ {∞, 2, 1, 0.5, 0.25} interpolates continuously from a known win to the trivial ρ = 1 limit. That makes it the cleanest first experiment.

**Gauges.**
- Plain: the gate is G⊗1_anc.
- Backward (KBM): the gate is G⊗G*. The physical RDMs are identical in both gauges, so the same dense reference serves both.
- The Hauschild disentangler is deferred.

**Target set.**
- Physical Pauli strings on 1-3 sites in the window, with identity on every ancilla.
- In `spcfast` terms the F matrix is **unchanged**: rows are built from physical `h_bond` and physical Paulis, and the lookahead evolution exp(−iτH_region) acts on physical sites. The ancilla, or its backward evolution, is gauge for physical RDMs.
- Only the region state W must route the ancilla legs into the traced (column) side.

**Reference.**
- A dense purification vector with 4^N amplitudes, evolved by the same fused Trotter gates. N = 10 is 16 MB and N = 12 is 268 MB. This measures truncation error only, as in `mps_bench.py`.
- The physical ρ = ψψ† (2^N × 2^N after reshaping to (phys, anc)). At N = 10 that is a 1024² matrix, so the **full-state trace distance** ‖ρ − ρ_ex‖₁ and λ_min(ρ) are affordable for every arm, including DMT.

**Metrics.** Reuse the repo's definitions:
- single, nn and nnn rms (`mpsenh.local_obs`), `rdm2` and `rdm3` (`hp_bench.marginal_errors`), and |ΔE|;
- far collateral: ⟨Z_iZ_{i+3}⟩ and ⟨Z_iZ_{i+4}⟩ rms (DMT's documented weak spot per rTEBD);
- the physical trace distance;
- for purification arms, the purification infidelity. It is gauge-dependent, so it is diagnostic only.

**Parameters.**
- dt = 0.1 Strang, N = 10, with N = 12 to check.
- χ ladder {4, 6, 8, 12, 16, 24, 32, 48}. The maximum exact bond at N = 10 is 4⁵ = 1024.
- T chosen by the pressure scan, roughly 2-5 repo units, which is about 8-20 in DMT units.
- spcf defaults as in `pareto_spcf.py`: a = 2, fw = 0, iters = 4, taus = 1, ks = 1-2-3.

**Cost (measured with a tiny probe).**
- At a = 2 the region W grows from 2^6 × χ² to 2^6 × 16χ².
- At χ = 32 that is 64 × 16384 complex64, 8 MB, and one Gram product takes about 4 ms.
- At a = 1 it is 16 × 4096, about 0.35 ms.
- Estimate: about 0.5 s to 1 min per N = 10, T = 4 run (720 cuts). Feasible.

**DMT baseline.**
- **`dmt_baseline.py` cannot run here.** It imports `gcg` and `pytreenet.special_ttn.pauli`, and neither exists in this environment; `_mps_vendor/bug_util.py` has only stub `DMT` classes. The existing `out_dmt_T*.txt` came from another machine and used a *pure Néel* initial state.
- A standalone numpy DMT is needed.
  - Pauli-basis MPDO with d² = 4 per site. Gates are superoperators G⊗G* converted to the Pauli basis (16 × 16).
  - Mixed-canonical (Frobenius) form. At a cut: QR of the trace vectors to get the 4-row and 4-column reserved blocks, the connected matrix, SVD of the lower-right block to χ′ = χ − 8, recombination, and a second SVD.
  - Equal parameters: 4χ² per site, both for the purification (χ, 4, χ) and for the MPDO (χ, 4, χ).
- Unit-check per cut against dense: tr ρ, ρ_{1…j+1} and ρ_{j…L} unchanged to 10⁻¹².

**Known failure modes to watch.**
- The gain may vanish at small μ (the near-trivial evolution), and on `heis` (log entanglement growth).
- In the plain gauge, entanglement grows uniformly (KBM), so SVD-plain may be so bad that "spcf beats SVD" holds only against a strawman. **Always compare against SVD in the best gauge.**
- DMT gives up fidelity and can go non-positive; report λ_min.
- Note the χ′ = χ − 8 floor: DMT needs χ ≥ 9.

## Where to start: code changes and effort

1. **`rule/spcfpur.py`: a copy of `rule/spcfast.py`, generalized.** About 0.5-1 day. Add `dp = 2, da = 2` with d = dp·da.
   - `Mm = theta.reshape(l*d, d*r)`.
   - `_maps` unchanged, but `Lm` has shape (o, d^{aL}, l). Split the middle index into (phys, anc) and move anc next to o.
   - `Wof`: reshape to (no, pL, aL, d, d, pR, aR, np), permute physical (pL, p_b, p_{b+1}, pR) to rows and everything else to columns. Rows stay D = 2^L, so `_region`, the F matrices and `hvec` are untouched.
   - The `vjp` reshape `Gd.reshape(nX, 4, nZ, no, npp)` changes to the same split.
   - `Gth` becomes (l·d, d·r), and `retract` and the returns use `reshape(l, d, k)`.
   - Replace `M.EnhCut._plain` (it hard-codes 2) with a local d-generic `_plain`.
   - Port `test_spcfast.py` sections 2-4 (adjoint, finite-difference linearization, W form) as `test_spcfpur.py`.
2. **`pur_bench.py`: a driver.** About 0.5 day.
   - Product purification constructor (staggered and DW, μ).
   - Fused gates: plain `kron(G, I)` and backward `kron(G, G.conj())`, reordered to (p1, a1, p2, a2).
   - A d-generic copy of `mpsenh.run_tebd` and `mps_to_dense`; the current ones hard-code 2.
   - Dense 4^N reference with `_refcache`, physical ρ and the metrics above.
   - Arms `svd`, `spcf:<opts>` and `spcfg:<gate>`, each in gauge `plain` or `back`. JSON output matching `pareto_spcf.py` rows.
3. **`dmt_np.py`.** About 1 day, needed only for Test 2. Write it to the paper's definition (Sec. III, App. B), not to GCG's.

**Total before the first clue:** about 1-1.5 days (items 1-2). Before the head-to-head: about 2.5 days.

## First tests: a decision tree

**Test 0. Correctness and anchor** (minutes, after items 1-2).
- (a) `test_spcfpur.py`: adjoint relative error < 10⁻⁵ and linearization error shrinking linearly with the step, as in `test_spcfast.py`.
- (b) The μ = ∞ staggered purification (ancilla effectively unentangled) against the existing d = 2 spcf on `ising`, N = 10, T = 3, χ ∈ {6, 8, 12, 16}. The SVD arms must match to 10⁻⁸. Ancilla-excited tangent directions have zero Jacobian at first order, so spcf should match within about 10%.
- **Kill or fix:** any mismatch means a bug. Do not proceed.

**Test 1. SVD pressure scan, then spcf vs SVD on the μ dial** (about 30-60 min CPU). This is the core first clue.
- Setup: `ising`, N = 10, staggered and DW, μ ∈ {∞, 2, 1, 0.5, 0.25}, plain and backward gauges, T ∈ {2, 3, 4}.
- Step 1: an SVD-only χ scan to find usable cells. Usable means SVD rdm2 lies between 10⁻³ and 10⁻¹ and is not at its floor inside the ladder (README rule).
- Step 2: in usable cells, run spcf and gated spcf at 3-4 χ values.
- Pre-registered **success**: at some μ ≤ 1 (genuinely mixed; per-site purity ≤ 0.79 at μ = 1), the median SVD/spcf ratio on rdm2 and nn rms is **≥ 2** over the usable χ window, **against SVD in the better of the two gauges**. In addition, far collateral (ZZ at distance 3-4) must be ≤ 1.5× SVD's and the physical trace distance no worse than +10%.
- **Kill:**
  - no usable cell exists at μ ≤ 1 in either gauge, because SVD rdm2 is < 10⁻⁴ for every χ ≥ 8 and there is nothing to repair; or
  - the ratio is ≤ 1.2 in every usable μ ≤ 1 cell.
- **Ambiguous** (then run the N = 12 check before deciding):
  - the gain appears only at μ ≥ 2 (nearly pure, which is the pure-state result restated);
  - spcf beats SVD only within the plain gauge, while SVD-backward is better than spcf-plain at equal χ (then the gauge dominates, and spcf must be run in the backward gauge);
  - the gain is ≥ 2 in rdm2 but trace distance worsens by more than 10%.
- **Control:** `heis` at μ = 1 in the backward gauge is expected to show no gain. A gain there would be suspicious and calls for checking the metric.

**Test 2. Equal-parameter head-to-head with DMT** (about 30 min CPU, only if Test 1 succeeds).
- The best Test 1 cell (state, μ, T) at χ ∈ {12, 16, 24, 32}.
- Arms:
  - spcf-purification (best gauge);
  - SVD-purification (both gauges);
  - DMT-MPDO (`dmt_np.py`, ℓ = 3, χ′ = χ − 8);
  - Frobenius-MPDO.
- Same Trotter circuit, same physical initial ρ₀ (a product MPDO).
- Metrics as above, plus λ_min(ρ), energy drift and tr ρ.
- **Success:** spcf-purification has ≤ 1/1.5 of DMT's rdm2 and nnn error at ≥ 3 of the 4 χ values. DMT is expected to win on energy drift, which is its home ground and should be reported, not hidden.
- **Kill:** DMT ≤ spcf on rdm2 at every χ.
- **Ambiguous:** spcf wins on 2-site but loses on 3-site or far observables. That reproduces the near-vs-far trade-off; report it as such and add a γ-graded rTEBD arm.

Order: 0 → 1 → (2 only on success). Test 1 alone answers the direction question for the cost of about one day of coding.

## Open questions

- Is there a usable regime at all? KBM's result that thermal states evolve trivially in the backward gauge, and DMT's finding that purification converges at χ = 16, suggest local SVD errors on purifications may be small at high T. Test 1 Step 1 exists to answer exactly this.
- Should spcf also exploit the ancilla gauge? For example, it could tilt only within physical-relevant directions, or be combined with a Hauschild norm-disentangler. A disentangler would change the cut before spcf acts.
- At high T in the backward gauge the target is nearly linear in δX. Does a DMT-style exact reserved block on X̂ (with zero extra bond?) beat the Gauss-Newton tilt there?
- Gibbs initial states need imaginary-time preparation, which itself truncates. Should the first tests stay with product ρ₀ (field-only Gibbs states, quenched)? That is the current plan; a Gibbs-gradient state as in DMT Eq. 36 would come later.
- The purification fidelity is gauge-dependent. Should the physical trace distance be the headline global metric? We propose yes.
- The equal-parameter claim holds per site (4χ² each), but DMT's cap includes the 8 reserved vectors. Should a "DMT at χ + 8" arm also be shown to bracket it?
- Mc Keever (2012.12233) and Godinez-Ramirez (2409.08127) truncation objectives remain unread. Skim them before writing up.

**Tool notes.**
- `read_arxiv_paper` worked for every ID; large results were saved to files and parsed with jq.
- `search_semantic` and `search_openalex` returned empty results on every query.
- `search_arxiv` returned mostly irrelevant hits, and fielded (`abs:`) queries returned nothing.
- `search_google_scholar` worked partially.
- The ID 1205.3756 is a citation error in the source report, not a tool error.
