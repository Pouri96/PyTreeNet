# Rank 4 first moves: reference-state-conditioned Heisenberg MPO ("DMT with rho_ref")

## Summary verdict: conditional (cheap to test, moderate novelty, and it is not really spcf)

The full texts make this direction both simpler and narrower than the report described. There are four findings.

1. **DMT's construction carries over exactly, so no tilt or Gauss-Newton step is needed.**
   - DMT (1707.01506, Sec. III.B, Eq. 16) uses the identity in only one place: the χ×4 matrix built by contracting every site outside a two-site window with the identity covector. That matrix is then QR'd to choose the preserved block of the bond matrix.
   - Replace the identity covector with the reference state's per-site covector, (1, ⟨X⟩, ⟨Y⟩, ⟨Z⟩) in the Pauli basis, and the same algebra exactly preserves Tr[(X_{1..j+1} ⊗ σ_ref^{j+2..L}) O] and Tr[(σ_ref^{1..j−1} ⊗ X_{j..L}) O] for every X.
   - It costs 8 reserved bond dimensions, as in DMT: the new rank is at most χ′ + 8 (1707.01506, Eq. 24 ff.).
   - The report's "linear least-squares tilt" is therefore needed only if we refuse the 8-dimension reserve, or want windows larger than 3 sites.
2. **A static σ0 = Néel protects only part of the error.** By the exact error identity of 2609.18246 (App. A, Eq. 6), the error at time T is E − b = Σ_n Tr[ρ(T − n dt) d_n].
   - Here d_n is the truncation defect made at step n of the Heisenberg run.
   - DMT-σ0 zeroes only the dual-state component ρ(0) = σ0. That is exactly the dual of the *last* truncations, which are also the heaviest, because the operator is largest then. This alignment is the strongest argument for the direction.
   - The protection decays once T − n·dt exceeds the local-relaxation time, about 1/h_x ≈ 1 for the repo's `ising`.
   - The natural upgrade is a time-dependent product reference ⊗_i ρ_i(s). Its marginals can come from a cheap forward run.
3. **The closest prior art is not in tensor networks, and it is very recent.**
   - Pérez, Wójtowicz and Plenio (2609.12840) define a covariance-geometry projection π_m^{σ0} that folds high-body Pauli strings onto low-body "centered" representatives and preserves Tr[σ0 ·] exactly (Eq. 4). It is the Pauli-propagation twin of DMT-σ0.
   - They cite two unpublished manuscripts ([30], [31]) that generalize it. An MPO version from that group is the main novelty threat.
   - Kim, Jeong and Oh (2609.18246) do expectation-aware MPS/MPO compression, but with a single scalar functional per time slice and both directions stored for all slices.
4. **The report got the xSPD ID wrong.** arXiv:2506.13241 is ORQA (Broers, Sun, Yunoki), a parallel Pauli-algebra code that truncates by plain coefficient threshold. xSPD is Begušić & Chan, arXiv:2409.03097 (PRX Quantum 6, 020302).

**Verdict.** Worth one cheap, closed-form experiment of about 1.5 days of coding and under an hour of runs. Two outcomes would kill it:
- static DMT-σ0 does not beat plain SVD-MPO and DMT with the identity covector (DMT-I) by about 3x on ⟨Néel|Z_i(t)|Néel⟩;
- every MPO arm loses badly to ordinary Schrödinger MPS-TEBD at matched memory.

The connection to spcf proper is weak. spcf's tilt survives only as an optional fixed-χ, no-reserve or wider-window variant.

## Literature, verified from full text

PDFs and extracted text are in `/tmp/papers_rank4_refstate`. All reads used `read_arxiv_paper`.

| arXiv | What I read | Verified facts (location) | Report claim: status |
|---|---|---|---|
| 2609.18246 (Kim, Jeong, Oh, "New directions in dynamical expectation estimation") | Full text, including App. A | **Losses (Eq. 1):** ℓ_B(O; O⋆, ρ̄) = γ\|Tr[ρ̄(O − O⋆)]\|² + μ‖O − O⋆‖². The ρ̄ is the *full* forward-evolved state at that slice, and the γ-term is **one scalar per slice**.<br>**Algorithm 1:** needs initial ansätze ρ̃_t and Õ_t for **every** slice t, then one forward and one backward sweep. Each local update alternates two linear solves, two iterations.<br>**Error identity (Eq. 2, App. Eq. 6):** E − b = Σ_t Tr[ρ̃_t d_t] + Σ_t Tr[(ρ_t − ρ̃_t) d_t].<br>**Benchmark:** 30-qubit McClean barren-plateau circuits, 500 instances, L = 5…100 layers, ⟨Z1Z2⟩, γ/μ = 1000, χ ≤ 30/50. Capped relative error (Eq. 4) is 2-3 orders below variational state compression (VSC) (Fig. 3). Fig. 1 uses 20 instances at n = 20, depth 50, χ = 30. No Hamiltonian dynamics, no DMT comparison. Pauli propagation (PP) is weight-truncated at ≤5 or ≤6. | "2-3 orders vs variational compression": **correct**. "Needs the full back-propagated observable in memory": **correct but understated**. It stores the state and the observable at *all* L slices. "Thin numerics": **fair**: one circuit family, one observable, one figure. |
| 2609.12840 (Pérez, Wójtowicz, Plenio) | Main text and SI start (~90k chars) | **Covariance form (Eq. 3):** (A,B) = Tr[σ0{A,B}/2]. For product σ0, Eq. 4: π_m^{σ0}(⊗p_i) = Σ_{S, \|S\|≤m} (∏_{i∉S} μ_i) ⊗_{i∈S} r_i, with μ_i = Tr(σ_i p_i) and r_i = p_i − μ_i I.<br>**Properties:** preserves ⟨·⟩_{σ0} exactly and never enlarges support. For pure σ0 it is a seminorm whose null space is {A : A\|Ψ⟩ = 0} (Eq. 8 discusses aliasing).<br>**Dynamics:** used inside an adaptive Krylov "hierarchical basis" restricted flow (Eqs. 7-12).<br>**Benchmark:** 3D TFIM (J = 1, h = 0.75), 3×3×2 (N = 18) and 6×5×3 (N = 90), σ0 = \|↑⟩^⊗N, O = Z_0, tJ ≤ 1.215. ∆_max is about 5e-2 / 5e-3 / ~1e-3 at m = 1/2/3, versus about 1 / 1e-1 / 1e-2 for Hilbert-Schmidt (HS) weight truncation (Fig. 2). The TN comparison is TeNPy Heisenberg TDVP-MPO with plain SVD, χ = 8…128. | "Folds high-body content onto low-body representatives": **correct**. The report missed that this is *exactly* the σ0-conditioned partial expectation, applied in body-order space rather than across a bond. |
| 2409.03097 (Begušić & Chan, xSPD; PRX Quantum 6, 020302) | Full text (grep of methods and 2D sections) | **X-truncation:** drop strings with more than M X/Y factors when σ0 is a computational-basis state. It assumes "limited operator backflow from high X-weight Paulis to the manifold of Z-type Pauli operators".<br>**Error budget:** 11×11 TFIM at h_c, M = 5, X-truncation error ≈ 0.003. M = 3 is too large an error; M = 7 saves little. Error scales with δ/∆t. | Content **correct**; the **ID is wrong** in the report. |
| 2506.13241 | Abstract and grep | ORQA (Broers, Sun, Yunoki): parallel Pauli propagation, 127-qubit kicked Ising, truncation \|O_I\| ≤ ε0·max\|O\|. Not state-aware. | **Mis-cited** as xSPD. |
| 1707.01506 (White, Zaletel, Mong, Refael, DMT) | Sec. III in full | Preserves tr ρ, ρ_{1..j+1} and ρ_{j..L} (Sec. III, list).<br>QR of the χ×4 matrices [A^0…A^0 A^μ_j] and [B^μ_{j+1} B^0…B^0] (Eq. 16).<br>Connected matrix M̃ = M − M_{α0}M_{0β}/M_00 (Eq. 23). SVD of the α,β ≥ 4 block, then rank(M′) ≤ χ′ + 8 (Eq. 24, App. B). | "Exactly preserves ≤3-site expectations": **correct**. "χ_preserve + χ_extra": **correct**, with χ_preserve = 8 for ℓ = 3. |
| 1902.01859 (Ye, Machado, White, Mong, Yao) | Grep of methods and appendix | "χ = χ_preserve + χ_extra", with χ_preserve stated as 2^ℓ for preservation diameter ℓ; the text extraction is ambiguous and gives 8 at ℓ = 3 either way. "Errors in longer ranged operators propagate down to the three-site density operators via the system's dynamics." | **Correct.** |
| 2412.08730 (Guha Roy & Slagle, rTEBD) | Grep of intro and Sec. 4 | Schrödinger-picture MPDO only. It mentions MPOs for "a time evolved observable" but never conditions on an initial state. It quotes DMT failing on ⟨σ_iσ_{i+3}⟩. | Not a competitor for this direction. |
| 1009.4646 (Muth, Unanyan, Fleischhauer) | Full | Heisenberg-picture TEBD with Frobenius (Schmidt) truncation. Operator-space Rényi entropy S_α (α > 1) is bounded by the infinite-temperature autocorrelation (Eq. 9), so the operator entanglement of a conserved density grows at most logarithmically. | **New and important for controls:** Z_i in `heis` belongs to the conserved S^z and is cheap for SVD-MPO, so it is an expected null. Z_i in `ising` (h_z ≠ 0) is not conserved, so its OSEE grows linearly, which is the pressure regime we need. |
| 0808.0666 (Hartmann et al., Heisenberg DMRG), 2201.08402 and 2308.04291 (Frías-Pérez, Bañuls, Tagliacozzo) | **Not read**; abstracts or the description in 2609.18246 only | Hartmann et al.: HS/Schmidt truncation of the MPO. 2308.04291: Schrödinger-picture conversion of long-range entanglement into mixture, preserving local observables. | Adjacent, not state-conditioned operator truncation. |

**Tool notes.**
- `read_arxiv_paper` worked for every ID tried.
- `search_arxiv` relevance ranking was poor, and `search_semantic` and `search_openalex` returned empty results for several queries.
- Google Scholar searches gave the most useful hits.
- I could not find the arXiv ID of Xu, Zhao, Zhang, Zhu, Zhao (2026), "Classical simulation of noiseless quantum dynamics without randomness". It is the forward-MPS/backward-low-weight-Pauli meet-in-the-middle method cited as ref. 14 of 2609.18246.

## Prior-art / novelty risk: moderate, rising

- **Nobody found does DMT-type, bond-local, reference-conditioned truncation of a Heisenberg MPO.**
  - DMT, its follow-ups and rTEBD are all Schrödinger-picture MPDOs.
  - Heisenberg MPO work (Hartmann 2009, Muth 2010, Xu-Swingle, HPL) uses HS/Schmidt truncation.
  - 2609.12840's own TN baseline is plain TDVP-MPO.
- **The idea is a one-line generalization of DMT**: swap the covector. Any referee will see it as "DMT in the covariance geometry of σ0". The publishable content would be the dual-state analysis, the time-dependent product reference, and a demonstration.
- **2609.12840 is the conceptual twin.** The Plenio group's unpublished refs [30] ("Adaptive Max-Ent restricted evolutions") and [31] ("Generalized m-body mean-field projections … d-dimensional Heisenberg dynamics") may already contain a TN version. Their paper already runs TeNPy MPO baselines. Re-check arXiv for [31] before any write-up.
- **2609.18246 owns "expectation-aware compression" as a slogan.** It uses one global functional per slice with the full forward state; we would use a family of window functionals with a product state at fixed χ, and no stored slices. That distinction is clean but must be stated.
- **xSPD and π_m^{σ0} are the Pauli-propagation baselines.** `pauli_prop.py` implements neither. Adding both is about 30 lines each, and a reviewer will ask for them.

## Technical details for implementation

**Object.**
- O(t) = Σ_P c_P P, with c real because O is Hermitian and the Pauli-basis gate is real orthogonal.
- Store it as an MPS over N sites with d = 4, basis order (I, X, Y, Z). Start from a product, bond 1: e_0 everywhere except e_3 (Z) at site i0.
- Gates: `pauli_prop.transfer(G4)` returns R[c_in, c_out] with code c = p_b + 4p_{b+1}, for G†PG. Reshape it to a (4,4,4,4) superoperator in ('labr' row-major (p_b, p_{b+1})) order.
- Verify the reshape against `pauli_prop.propagate(..., eps=0)` at N = 6. That check already passes for PP against dense (`__main__`).
- The symmetric step [0…N−2, N−2…0] is a palindrome, so one Heisenberg run gives O(n·dt) for all n: O(n) = G_step† O(n−1) G_step. Readout at each n is ⟨Néel|O(n)|Néel⟩.
- **Do not renormalize** the MPO after a cut, unlike `mpsenh.svd_cut`. Report both raw and Frobenius-renormalized SVD arms, because the norm drift itself is an error source (cf. the rank-1 note on HPL).

**Reference covectors.**
- For |0⟩: v = (1, 0, 0, +1). For |1⟩: v = (1, 0, 0, −1). In general v_i = (1, ⟨X⟩, ⟨Y⟩, ⟨Z⟩)_{σ_i}, with unnormalized Paulis.
- Readout: ∏_i (v_i · A_i) contracted along the chain.
- Maintain left environments L_j = L_{j−1}·Σ_μ v_μ A_μ and the right ones analogously. They update in O(χ²) per cut during the sweep.

**DMT-ref cut** at bond (b, b+1), with two-site θ (l, 4, 4, r) in mixed canonical form, Frobenius-orthonormal:
1. SVD θ = U s V†. Form R_L[α, μ] = Σ L_{b−1}·U[:, μ, α] (χ×4) and R_R similarly. QR each and keep the full unitary Q_L and Q_R.
2. M = Q_Lᵀ s Q_R. Keep rows < 4 and columns < 4 exactly. Truncate the lower-right block (α, β ≥ 4) to rank χ − 8 by SVD. Recombine, then SVD again to rank ≤ χ (Eqs. 23-29).
3. **Failure mode: drop DMT's connected-correlator subtraction.** It divides by M_00 ∝ Tr[σ_ref O], which is 0 for any traceless Z_i(t) under σ_ref = I and crosses zero under Néel.
4. **Rank deficiency.** Néel covectors make R_L rank-deficient; for example, at t = 0 only the I/Z columns are populated. Use rank-revealing QR with tolerance, and reserve only the numerical rank (≤ 8).
5. **Unit test.** After each cut, |Tr[σ_ref δ]| and all window functionals should be at the 1e-13 level.

**Arms (equal maximum bond χ, so equal stored parameters Σ 4χ_lχ_{l+1}):**

| Arm | What it is |
|---|---|
| SVD-raw | Plain SVD cut, MPO not renormalized |
| SVD-renorm | Plain SVD cut, Frobenius norm restored |
| DMT-I | Identity covector (Heisenberg-picture DMT, no state information) |
| DMT-σ0 | Static Néel reference |
| DMT-σ(s) | Covector at step n is the product of single-site marginals of ρ(T − n·dt), one run per output time T. Oracle marginals from the dense state first, then from a χ = 8 forward MPS |
| CLSQ (optional, spcf-flavoured) | Keep SVD's U_k with no reserve. Solve the equality-constrained least squares min ‖θ − U_k R‖ s.t. window functionals(U_k R) = window functionals(θ) for R, minimum-norm via KKT/pseudo-inverse. This is linear because the targets are linear in θ |

**Baselines.**
- `pauli_prop.propagate` over an ε ladder; memory is 2 numbers per string (key + coefficient).
- Schrödinger MPS-TEBD (`mpsenh.run_tebd` with `svd_cut`) at χ_s = √2·χ, which stores the same 2χ_s² ≈ 4χ² floats per site.
- Time splitting ⟨ψ(T/2)|O(T/2)|ψ(T/2)⟩, if cheap.

**Models.**
- `ising`: h_x = 0.9045, h_z = 0.8090, the same as rTEBD and HPL's tilted Ising with h_x and h_z swapped.
- `ising2`: (1.4, 0.4).
- `heis`: the expected null by Muth et al.
- Initial state: Néel (`initial_mps`). Use dt = 0.1, as in the repo.

**Error metric.**
- Following 2609.12840 Fig. 2: ∆(t) = |⟨Z_i(t)⟩_approx − ⟨Z_i(t)⟩_exact| and ∆_max over t ∈ [0, T].
- Plot error against stored parameters, as a Pareto front.
- Exact reference: `M.dense_reference` plus `P.exact_values`. N = 16, T = 3 runs in about 1 s (checked).

**Pressure window.**
- At N = 16, T = 3, Z_7: PP with ε = 1e-3 holds 15.7k strings with error ≈ 0.015 (exact −0.170, PP −0.155). With ε = 1e-2, PP holds 229 strings and errs by about 0.2. Both were measured in a sub-second sanity check.
- An MPO at χ ≈ 16-32 has comparable memory, so χ ∈ {12, 16, 24, 32, 48} and T up to about 6 should cover the window where SVD-MPO error is between 1e-3 and 1e-1.
- The light cone reaches the edges of N = 16 around t ≈ 3-4. Use N = 20 for T > 4; its dense reference has 2^20 amplitudes and is cheap.

**Environment caveats found.**
- This sandbox has **no scipy**: `mpsenh` imports `scipy.linalg.expm`, and I used a numpy shim in the scratchpad for the sanity check.
- It also has no `gcg`, so `dmt_baseline.py` cannot run. DMT must be reimplemented locally anyway, because its reference covector has to be swappable.

## Where to start

**New file:** `heis_mpo.py`, about 250 lines, in the graded_rule folder. It does not touch `rule/spcfast.py`. It contains:
- `pauli_gates(model, N, dt)`, which wraps `pauli_prop.transfer` and `mpsenh.make_gates`;
- `run_heis(model, N, i0, chi, nsteps, cut, ref)`, the Strang sweep copied from `mpsenh.run_tebd` with d = 4 and no renormalization, returning ⟨σ0|O(n)⟩ for every n;
- `svd_cut_op`, `dmt_cut(theta, chi, dirn, Lenv, Renv, v_b, v_b1)`, `clsq_cut`, and environment updates;
- a CLI matching `mps_bench.py`: model, N, T, dt, arms, χ list, out.json.

**Small helper additions (no edits to existing files):**
- `pp_variants.py`: xSPD (X-weight cap M) and π_m^{σ0} folding layered on `pauli_prop.propagate`'s keep step. For Néel, π_m^{σ0} replaces Z_i with r_i + μ_i I and drops X/Y-containing strings beyond weight m.
- `heis_attrib.py`: the defect attribution needed for test 2.

**Effort.**

| Item | Estimate |
|---|---|
| `heis_mpo.py` with SVD and DMT-I/σ0 cuts and unit tests | 1-1.5 days |
| DMT-σ(s) oracle and cheap-forward variant | +0.5 day |
| Attribution script | +0.5 day |
| CLSQ | +0.5 day, only if test 1 passes |

## First tests (decision tree)

**Test 0: plumbing and exactness (about 10 min of runs).**
- N = 6-8, `ising`, χ = ∞: the MPO readout should match `pauli_prop` (ε = 0) and the dense value to 1e-12.
- At finite χ: DMT-σ0 should keep |Tr[σ0 δ_n]| and all four-site-window functionals below 1e-12 per cut, and DMT-I should do the same for its functionals.
- Fail means a bug, not a verdict.

**Test 1: does conditioning on the reference matter? (about 30-45 min.)**

| | |
|---|---|
| Setup | `ising`, N = 16, O = Z_7, Néel, dt = 0.1, T = 6 (also N = 20, T = 8 if fast); repeat on `ising2` |
| Arms | SVD-raw, SVD-renorm, DMT-I, DMT-σ0 at χ ∈ {12, 16, 24, 32, 48}; PP ε-ladder; MPS-TEBD at matched parameters |
| Metrics | ∆_max over [0, T], ∆(T), stored parameters, wall time |
| Success | DMT-σ0 has ≥3x lower ∆_max than **both** SVD-renorm and DMT-I at equal χ, for at least two χ values in the window where SVD's ∆_max is between 1e-3 and 1e-1. Also, the best MPO arm is within 3x of MPS-TEBD at matched parameters, or better |
| Kill | DMT-σ0 is within 1.5x of DMT-I or SVD-renorm at every χ, meaning the reference adds nothing. Or MPS-TEBD at matched memory is ≥10x better than every MPO arm on both models, meaning the Heisenberg route is pointless in 1D for this observable. Run the `heis` control alongside; it is expected cheap for all MPO arms (Muth et al.) and should show no ordering |
| Ambiguous | DMT-σ0 wins only for t ≲ 1-2 (the protection-decay prediction), or DMT-I wins nearly as much (then DMT locality, not the reference, is doing the work). Either way, go to test 2 before deciding |

**Test 2: where does the error come from, and does a moving reference fix it? (about 45-60 min.)** Run only if test 1 is a success or ambiguous.

| | |
|---|---|
| Setup | N = 12-14, one χ in the pressure window, T = 4 |
| Step 1 | Store each defect d_n (an MPO of bond ≤ 2χ) and compute c_n = Tr[ρ(T − n·dt) d_n] against the exact dense ρ(s) through an MPO-times-dense-state contraction. Σ_n c_n must equal E − b to round-off, by the identity of 2609.18246, Eq. 6. Plot \|c_n\| against s = T − n·dt for SVD, DMT-I and DMT-σ0 |
| Step 2 | Rerun with DMT-σ(s) using oracle marginals, one run for this T |
| Success | (i) Under SVD, \|c_n\| is dominated by s ≲ 1 (the heavy late truncations). (ii) DMT-σ(s) beats static DMT-σ0 by ≥3x in ∆(T). Then build the cheap-forward-marginal variant and the CLSQ fixed-χ variant |
| Kill | \|c_n\| mass sits at s ≫ 1 even under SVD, or the oracle product reference gains < 1.5x over static σ0. In either case connected correlations in ρ(s), which no product reference sees, dominate. The fix would then be 2609.18246's full-state scalar loss, which is their method, not ours |

**Test 3 (optional): the spcf-shaped variant.**
- Run only if test 2 passes.
- At χ ∈ {8, 12, 16}, where an 8-dimension reserve is expensive, compare:
  - CLSQ (no reserve, 3-site and 5-site windows);
  - DMT-σ(s) (χ′ = χ − 8);
  - DMT-ℓ = 5, which needs about 32 reserved.
- Success: CLSQ is ≥2x better than DMT at equal χ ≤ 16.
- Kill: it is never better. In that case drop spcf from this direction and keep "DMT-σ(s)" as a standalone result.

## Open questions

1. **Are [30]/[31] of 2609.12840 out?** If an MPO covariance-projection paper exists, the novelty reduces to the time-dependent product reference and the dual-state analysis.
2. **Metric for the χ_extra block.** With a reference, Frobenius (as in DMT) is not obviously right. The covariance seminorm of σ0 (2609.12840, Eq. 3) is natural, but it is rank-deficient for a pure Néel σ0. A mixed reference, for example σ0 smeared by a depolarizing p ≈ 0.05, would regularize it. That gives a closed-form, whitened-SVD arm, as the report's "whitening is exact for linear targets" suggests.
3. **One run versus all output times.** A static reference serves every T. A moving reference needs one run per T, or several reserved references, each costing +8 bond. Is a two-reference DMT, with σ0 plus σ(s ≈ 1), the practical compromise?
4. **When should the readout be truncation-free?** Time splitting, ⟨ψ(T/2)|O(T/2)|ψ(T/2)⟩, halves both entanglement loads. DMT-ref on the operator half composes with it. Does that combination dominate both pure pictures?
5. **Positivity and boundedness.** The Heisenberg operator should keep ‖O‖_∞ = 1; HS truncation can violate expectation bounds. Check |⟨Z⟩| ≤ 1 violations as a side metric, as in 2609.12840's "expectation-range violation" remark.
6. **Unresolved ID.** The arXiv ID of Xu et al. (2026) "without randomness", the meet-in-the-middle PP+MPS method, is still unknown. It is another baseline family.
