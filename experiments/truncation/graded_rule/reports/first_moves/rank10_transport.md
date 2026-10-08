# Rank 10 first moves: infinite-temperature transport in a Heisenberg Pauli MPO (control direction)

## Summary verdict: weak as an application, useful as a cheap control, conditional on one reframing

This direction should stay a **control**, but it can be made informative.

**Why the tilt is the wrong tool here.**
- Every transport quantity at infinite temperature is *linear* in the evolved operator O(t):
  - C(x,t) = Tr[q_x O(t)]/2^N;
  - the current autocorrelation;
  - D(t).
- Two baselines already handle linear targets exactly or near-exactly:
  - DMT exactly preserves all window partial traces up to diameter ℓ around the cut. The traceless Heisenberg variant is described in 2310.06886, App. D.
  - rTEBD down-weights high-Pauli-weight content in closed form, at roughly SVD cost.
- For linear targets the spcf objective is an *exactly quadratic* form in the bond matrix. A Gauss-Newton tilt is therefore the wrong tool. The natural object is a **closed-form, metric-weighted ("window-whitened") SVD**.

**What makes the direction worth a small budget.**
1. It is the cleanest place to test the report's "linear vs quadratic" dividing line. It is also the cheapest place to test the report's proposed overhead fix, a whitened SVD, because whitening is exact here, not approximate.
2. It has one concrete open niche. DMT at its default ℓ = 3 gives span-4/5 cross-cut strings (σ_iσ_{i+3}) no priority. Protecting them exactly costs χ_preserve = 32 (ℓ = 5). That eats most of a small bond budget.
   - A soft, graded window metric could protect span-4/5 content at fixed χ without reserved rank.
   - DMT's own paper proposes exactly this "controlled-metric truncation" as future work (1707.01506, Discussion). I found no implementation. So the *idea* is not ours; a working implementation plus benchmark would be.

**Prior expectation.** DMT and rTEBD will be hard to beat on D. The only plausible win is span-4/5 and early-transient accuracy at small χ (χ ≲ 48).

**Recommendation.** Spend about 2 days, shared with the rank 1 and rank 4 Pauli-MPO infrastructure, and kill early if the static single-cut test (Test 1) shows no Pareto gain over DMT-n1/n2.

## Literature, verified from full text

All of the following were read in full via `read_arxiv_paper` (PDF text in `/tmp/papers_rank10_transport`). The Discussion and Sections III-IV of 1707.01506 were read; its appendices were skimmed.

| arXiv | What I read | Verified facts (with location) | Report claim → status |
|---|---|---|---|
| **2412.08730** (rTEBD) | Full text, v4, 16 May 2026 | **Authors:** Sayak Guha Roy and Kevin Slagle (Rice), SciPost Phys. 21, 015 (2026). Authorship confirmed.<br>**Method:** reweighted Pauli basis σ̃^μ = γσ^μ for μ ≠ 0 (Eq. 6). The SVD then minimizes ‖R†(ρ−ρ_χ)R‖_F. An n-weight string is suppressed by γ^(−n).<br>**MPDO only.** There is **no pure-state/MPS variant**. The intro says it could apply to "a time evolved observable", but it is never benchmarked in the Heisenberg picture or at infinite T.<br>**Benchmarks:**<br>- Free fermions: L = 128, δt = 0.08, χ = 16/64/256, γ = 1.5 (Figs. 6-9).<br>- Mixed-field Ising: J = 1, h_x = 0.9045, h_z = 0.8090, L = 64, χ = 32/64/128, γ = 1.6 (Figs. 10-12).<br>- Optimal γ = 1.6-1.7 (Fig. 13; T_f = 100, χ = 8-64).<br>**Interacting-model metric:** only the energy-density drift is scored. No observable is compared to an exact answer.<br>**Fig. 12:** "At short times, MPS-TEBD has smaller error than rTEBD, while at long times rTEBD has smaller error."<br>**DMT:** **no DMT comparison**; it is listed as future work (Sec. 5). | **Corrected.** "Competitive with MPS-TEBD" holds only at *late* times, and only for conserved-quantity drift on the interacting model. "rTEBD competes directly with spcf on our Ising cells" is unproven: rTEBD has never been scored against exact local observables in an interacting model. The Ising fields do coincide with the repo's `ising` (see units below). |
| **2310.06886** (Yi-Thomas, Ware, Sau, White, PRB 110, 134308) | Full text including App. A-F | **Model:** H = Σ 4J S^zS^z + 2g_z S^z + 2g_x S^x, i.e. ZZ + g_z Z + g_x X, with g_x = 1.4 and g_z = 0.9045.<br>**MPO runs:** L = 256, 4th-order Trotter, τ = 0.05 early and 0.5 late, time doubling (App. C).<br>**DMT:** converged to 0.23% at χ = 256, D ≈ 1.446 (Fig. 5). DAOE quoted as ≈ 1.40 ± 0.01 (Fig. 2).<br>**Key sentence (Sec. IV B 1):** "At shorter times (t ≲ 20), DMT performs no better than TEBD. Since this region overwhelmingly determines the diffusion coefficient…"<br>**Heisenberg DMT (App. D 2):** same gauge, but **no connected-component subtraction**, because M_00 = Tr O = 0. It preserves "partial traces of the operator", not RDMs. | "DMT/OST agree to t = 60, D within 1%; TEBD converges to t ≈ 20, D within 1%": **correct**. The report's suggested niche, early transients, is exactly where DMT gives **no** gain over TEBD at large χ. A transient win for any cut must therefore come at small χ. |
| **1707.01506** (DMT) | Sec. II-IV, Discussion | **Model:** H = Σ S^zS^z + ½h_x ΣS^x + ½h_z ΣS^z with h_z = 0.8090, h_x = 0.9045 (Eqs. 30-31).<br>**Pure-state test** (Fig. 8): L = 16, 20, 24 against semi-exact MPS. DMT beats MPS-TEBD at fixed χ ≪ 2^{L/2} on the energy-density Fourier component, with error ≈ 10⁻³.<br>**Positivity** is not exact: "Z ≥ 1 for DMT does not imply positive semi-definiteness".<br>**Non-convergence:** accuracy stops improving above χ ≈ 2⁵-2⁶ for near-σ^y states.<br>**Discussion:** "weight errors along σ^z_{j−1}σ^z_{j+2} more heavily than errors along σ^z_{j−7}σ^z_{j+6}. Such controlled-metric truncation is a natural extension of this work." | Report claims **correct**. The controlled-metric remark is **prior art for the idea** of a window-weighted metric (see next section). |
| **1902.01859** (Ye, Machado, White, Mong, Yao, PRL 125, 030601) | Main text and the preservation appendix | **Budget split:** χ = χ_preserve + χ_extra, with the preserve block counted inside χ.<br>**Generalized DMT:** preserves all (2n+1)-site operators using χ_preserve = 2·4^n = 2^ℓ (App., Fig. 8). A (2n+2)-site operator straddling the cut midpoint is *not* preserved.<br>**Fig. 3b:** at fixed χ, tuning ℓ changes accuracy.<br>**Leakage:** "errors in longer ranged operators propagate down to the three-site density operators via the system's dynamics." | "2^ℓ" is **correct**. Note that **σ_iσ_{i+3} *is* preserved by DMT at ℓ = 5 (χ_pres = 32)**. The weakness exists only at default ℓ = 3, or when χ is too small for ℓ = 5. |
| **2004.05177** (DAOE letter) | Full text | **What is truncated:** strings with Pauli *weight* (count of non-identities) above ℓ*. The dissipator is an MPO of bond dimension ℓ*+1.<br>**Ising** (g_x = 1.4, g_z = 0.9045): L = 51, Trotter step 0.01, Δt = 0.25 or 1, χ ≤ 768. D ≈ 1.40, within ≈ 1% across ℓ* = 2, 3, 4.<br>**Exact benchmark:** typicality, L ≤ 21, t ≲ 10.<br>**"TEBD without dissipation starts deviating … around t ≈ 7−8"** at χ = 512.<br>**Exponential smallness** in ℓ* is only "suggested" by random-circuit calculations. | **Attribution fix:** the "error in D exponentially small in ℓ*" statement is Eq. 3 of **2111.09904**, not the DAOE letter. |
| **2111.09904** (von Keyserlingk, Pollmann, Rakovszky, PRB 105, 245101) | Intro, Sec. II, conclusions | D − D_DAOE = exp[−O(ℓ*)] (Eq. 3). Deterministic-model numerics "are not decisive". DAOE needs χ ∼ exp[O(log²(1/ε))], while brute-force TEBD needs exp[poly(1/ε)]. A backflow analysis for **support-based** (rather than weight-based) truncation is listed as future work. | Report's backflow claim is **correct but softer** than stated: it is proven-ish for random circuits only. |
| **2408.08249** (Srivatsa, Lunt, Rakovszky, von Keyserlingk) | Main text and SM skim | DAOEμ replaces high-weight strings by ensemble averages (BBGKY-like). XX ladder at varying filling. The ballistic-to-diffusive crossover follows D ∝ 1/ρ. **TEBD at χ = 768, L = 201 is fine early and "after t ≳ 6 truncation errors cause the TEBD estimate to become unstable"** (Fig. 1c). | It is about *filling-dependent* crossovers, not infinite-T Ising. As a source for an "early transient" niche it is **weak**: TEBD is fine early at large χ. |
| **2311.17148** (FDAOE, Kuo, Ware, Lunts, Hafezi, White) | Abstract, intro, grep | Uses DMT (χ = 256, dt = 0.125) as the reference. DAOE ℓ* dependence is inconsistent at weak interaction. | Confirms that DMT is the community reference. |

**Tool notes.**
- `search_arxiv` returned irrelevant results for multi-word queries.
- `search_semantic` and `search_openalex` returned empty lists.
- `search_google_scholar` worked.
- GitHub access to `guharoysayak/reweighted_TEBD` and `Jack-Kemp/dmt` was **denied**: the session allows only pouri96/pytreenet. Neither public code was inspected.
- The unattributed sentence about "two neighboring sites remain unaffected" does **not** occur in 2412.08730.

**Units.** The repo's `ising` is h = ZZ + 0.9045 X + 0.8090 Z in Pauli units. That is exactly **4× the DMT/rTEBD Hamiltonian** (S = σ/2 and ½h fields). So repo time t corresponds to DMT/rTEBD time 4t.
- The OST/DAOE transport model (ZZ + 1.4 X + 0.9045 Z) is already in Pauli units, so repo time = paper time.
- It is **not** in `mpsenh.FIELDS`: `ising2` is (1.4, 0.4). It needs a new entry, `isingT` = (1.4, 0.9045).

## Prior-art / novelty risk: high for the idea, moderate for a fixed-χ graded window metric

- **DMT** already does exact local-observable-preserving truncation for MPDOs (Schrödinger picture) and traceless Heisenberg MPOs (2310.06886, App. D). Any "spcf preserves window partial traces" claim in this setting is DMT.
- **rTEBD** is a *global, weight-graded, separable* metric: per-site diag(1, γ⁻², γ⁻², γ⁻²) for the coefficient error. It is a closed form, because it keeps the whole MPO in the reweighted basis.
- **DAOE/FDAOE/DAOEμ/OST** truncate by weight or diameter in operator space.
- **The idea of a cut-local, support-graded metric is stated in 1707.01506.** It is also implicit in 2111.09904's future-work line on support-based truncation. I found no implementation through Google Scholar on "density matrix truncation", "controlled-metric", "local observable preserving truncation MPO", or "weighted SVD tangent space MPS local RDM". Every 2024-2026 hit (rTEBD, Yi-Thomas, Kuo, Samajdar turnstiles, Müller et al. noisy circuits) uses DMT or rTEBD as is.
- **The tilt itself adds nothing new here.** For linear targets the weighted problem
  - min ‖L^{1/2}(M − M′)R^{1/2}‖ with rank M′ ≤ χ
  - has the closed-form Eckart-Young solution of the whitened matrix when the metric is separable (Friedland-Torokhti 2408.05104, cited in the parent report).
  - A Gauss-Newton tilt would only approximate this.
- **Defensible novelty is narrow:**
  1. a *graded, cut-local window metric* (support-based, nested n = 1, 2 blocks with weights λ₁, λ₂) applied as a closed-form whitened SVD at fixed χ, with no reserved rank;
  2. a measured head-to-head of SVD, rTEBD, DMT-ℓ3, DMT-ℓ5 and the whitened cut at equal χ, which nobody has published (rTEBD explicitly lists the DMT comparison as future work).
- **Risk statement.** If it works, reviewers will call it "DMT's controlled-metric suggestion, implemented". That is publishable only as a methods note or benchmark, and it is not a contribution of *spcf* as such.

## Technical details for implementation

**Object.** O(t) = q_c(t) as a Pauli-basis MPS with d = 4 and real coefficients for Hermitian O. The Frobenius norm is the Euclidean norm of the coefficients.
- Gates come from `pauli_prop.transfer(G4)`: a real orthogonal 16×16 matrix (checked here: ‖RRᵀ − I‖ = 2e-16).
- Heisenberg evolution applies Rᵀ in reversed circuit order. `pauli_prop.propagate` already fixes the convention: `seq = circuit_bonds(...)[::-1]`.

**Densities.** On repo-style bonds:
- q_x = ε_x = Z_xZ_{x+1} + (g_x/2)(X_x + X_{x+1}) + (g_z/2)(Z_x + Z_{x+1}), the symmetric split of 2310.06886 Eq. 2 in Pauli units;
- current j_x ∝ g_x(Y_xZ_{x+1} − Z_{x−1}Y_x) (Eq. 4).

Start with q = Z_c as well. It is not conserved in Ising, but it is the simplest test object; energy is the real transport quantity.

**Observables and error metrics** (from 2310.06886, 2004.05177 and rTEBD):
- C(x,t) = Tr[q_x O(t)]/2^N, i.e. one Pauli-coefficient overlap per x (linear).
- Sum rule Σ_x C(x,t). It is conserved exactly by DMT and by the dynamics, so it is a direct truncation-error probe.
- MSD d²(t) and D(t) = ½ ∂_t d²(t).
- ⟨J(t)J(0)⟩/L, relative error against the reference (2310.06886, Eq. 28).
- **Far collateral:** rms error of the cross-cut string coefficients grouped by span 4, 5 and ≥ 6. These are DMT's documented blind spot.
- Frobenius fidelity of O(t) against the exact O(t).

**References.**
- **N ≤ 12: exact Heisenberg operator** as a dense real 4^N vector (134 MB at N = 12). One 11-gate half-sweep timed at 6.2 s here with naive `moveaxis`. A Strang step takes about 12 s, so t = 2.5 at dt = 0.1 takes about 5 min. That is acceptable, and einsum would be faster.
- **N = 16-20: canonical typicality**, as in 2004.05177: C(x,t) ≈ ⟨r|q_x U†q_cU|r⟩ for 2-4 random states, with the same Trotter circuit, through `mpsenh.dense_gate` at 2^20 amplitudes. Statistical error is about 2^{−N/2}, roughly 10⁻³.

**Typical parameters in the literature.**
- χ = 256-768 for converged D.
- Repo-scale pressure is χ = 8-64.
- The light cone reaches the edge of N = 12 by repo t ≈ 2, which is 8 in DMT units. Restrict scoring to |x − c| < 5 and t ≤ 2.

**Cut arms**, all at the same cap χ:

| Arm | What it does |
|---|---|
| `svd` | Plain truncation. Renormalization is off for operators: the norm is not conserved under truncation, and C should be read unnormalized. |
| `rw:γ`, γ ∈ {1.5, 1.6, 2} | rTEBD. Rescale site tensors by diag(1, γ⁻¹, γ⁻¹, γ⁻¹) on input; gates become S R S⁻¹, which is no longer orthogonal; canonicalize in reweighted coordinates; undo at measurement. |
| `dmt:n`, n ∈ {1, 2} | Traceless DMT, 2310.06886 App. D 2: QR of the χ×4^n "identity-far / Pauli-near" covector matrices. Keep the first 4^n rows and columns of M; SVD only the rest. Rank ≤ χ′ + 2·4^n with χ′ = χ − 2·4^n. **No** M_00 division. |
| `wsvd:n:λ1:λ2` | New. In the same DMT gauge, take bond-space projectors Π^{(1)}_L ⊂ Π^{(2)}_L (span of Y_L for n = 1, 2) and similarly on the right. Metric L = I + λ₁Π^{(1)}_L + λ₂Π^{(2)}_L, separable L⊗R. Truncate SVD(L^{1/2} M R^{1/2}) to χ and un-whiten with L^{−1/2}, R^{−1/2}. λ → ∞ approaches DMT-n's hard constraint; λ = 0 is plain SVD. Cost is one extra small QR. |

The separable metric L⊗R weights strings by "local on the left" × "local on the right". It contains DMT's one-sided preserved sets as the cross terms λ(Π_L⊗I + I⊗Π_R). It is an upper surrogate of the exact DMT target set, not equal to it.

**Known failure modes.**
- DMT does not converge for near-pure or σ^y-adjacent states (1707.01506). That is irrelevant at infinite T.
- MPDO-TEBD trace decay (rTEBD Figs. 6, 10) appears for the Schrödinger picture. In the Heisenberg picture the analogue is the decay of Tr[q_x O]: watch the sum rule.
- Whitening with near-singular L. The metric is ≥ I, so this is safe; no inverse of a small eigenvalue occurs.
- 2310.06886 needed τ ≤ 0.5 and 4th-order Trotter. Our dt = 0.1 with second-order Strang is fine because the reference uses the same gates.
- Boundary reflections at N = 12.

## Where to start (code changes and effort)

**Do not touch** `rule/spcfast.py` or `dmt_baseline.py`.
- `dmt_baseline.py` cannot run here: `gcg` and `pytreenet.special_ttn.pauli` are absent, as already noted in rank2_purification.md.
- Its existing outputs (`out_dmt_T*.txt`) are Schrödinger-picture runs on a *pure Néel* state, not transport.
- What to reuse from it is the **arm/params bookkeeping and the JSON/CLI shape**. DMT itself must be reimplemented locally in the Heisenberg traceless form.

**Coordinate with ranks 1 and 4.** Rank 1 plans `op_tebd.py`, with `svd`, `svdraw` and `rw:γ` arms. Rank 4 plans `heis_mpo.py`, with `dmt_cut` and a swappable covector. Build **one** module and give rank 10 three additions.

1. **`heis_mpo.py` (shared, about 250 lines, 1 day if not yet built).**
   - `pauli_gates(model, N, dt)` wraps `pauli_prop.transfer` over `mpsenh.make_gates`. Add `isingT` = (1.4, 0.9045) through a local FIELDS override instead of editing `rule/mpsenh.py`.
   - A d = 4 Strang sweep copied from `mpsenh.run_tebd`. Only `initial_mps`, `svd_cut` and the `l*2` reshapes hard-code d = 2.
   - Arms `svd`, `rw:γ`, `dmt:n`.
   - `coeffs(T, strings)`, which contracts the identity covector (Pauli index 0) on the far sites.
2. **`rule/wsvd_op.py` (new, about 80 lines, half a day).** The `wsvd:n:λ1:λ2` cut, plus a unit test:
   - λ = 0 reproduces `svd` bit for bit;
   - λ → 1e8 matches `dmt:n` on preserved coefficients to 1e-8.
3. **`heis_ref.py` (about 60 lines, 2 h).**
   - The exact Pauli-vector reference for N ≤ 12. Validate against `pauli_prop.propagate(eps=0)` at N = 6, which is already validated against the dense state.
   - The typicality reference for N ≤ 20.
4. **`transport_bench.py` (about 100 lines, half a day).**
   - Arms × χ ladder → JSON with the metrics above.
   - A **static single-cut mode**: load the exact O(t), canonicalize it to an exact MPO, truncate one cut, and score the coefficient errors by span class.

**Effort:** about 2 days to Test 2 if the shared module exists, about 3 days if it does not. Test 1 needs items 1 (gauge only), 2 and 3.

## First tests (decision tree)

### Test 1: static single-cut ceiling (about 1 day of coding, under 5 min of runs)

**Setup.**
- Exact q_c(t) for `isingT` at N = 12, with c = 5, t ∈ {0.5, 1.0, 1.5, 2.0} (repo units).
- Build an exact MPO and truncate only the center cut, b = 5|6, to χ ∈ {8, 16, 24, 32, 48}.
- Repeat at b = 4 and b = 6 for robustness.

**Arms.** `svd`; `rw:1.6`; `dmt:1`; `dmt:2` (χ ≥ 40 only); `wsvd:1:λ`, `wsvd:2:λ` and graded `wsvd:(λ1, λ2)` on a log grid λ ∈ {1, 3, 10, 100}.

**Metrics.**
- Coefficient error rms per span class s ∈ {≤3, 4, 5, ≥6} over strings straddling the cut;
- Frobenius error;
- the error in the q_x overlaps (C(x) at fixed t).

**Success.** At χ ≤ 32, at least one wsvd setting shows:
- (a) span-4/5 error **at least 2x below** `dmt:1`;
- (b) span ≤ 3 error ≤ 10⁻³ relative, or within 2x of DMT's machine zero in C(x) terms;
- (c) better span-4/5 error than `rw:1.6` at equal span ≤ 3 error.

All three must hold at ≥ 2 of the 4 times.

**Kill.** Any one of the following:
- `dmt:1`, `dmt:2` or `rw:γ` Pareto-dominates every wsvd point in the (span ≤ 3, span 4/5) plane at every χ;
- wsvd's best gain over `dmt:1` on span-4/5 is < 1.3x.

Then close rank 10 with a one-paragraph note: "linear targets are fully served by DMT/rTEBD; no cut-local metric adds value". That is itself a useful sentence for the paper.

**Ambiguous.** The wsvd gain appears only at χ = 8-16, where all arms have O(1) error, or only at a single t. Proceed to Test 2 with one χ in the transition region.

### Test 2: dynamic early-transient head-to-head, exact reference (about 30-60 min of runs)

Run only if Test 1 is not killed.

**Setup.** Same model and N = 12. Two observables:
- q = ε_c (energy, conserved);
- Z_c (control).

Settings: dt = 0.1, T = 2.0 (8 in DMT units), χ ∈ {8, 12, 16, 24, 32}.

**Arms.** `svd`; `rw:γ` with γ picked by rTEBD's own rule (minimum energy-sum-rule drift); `dmt:1`; `dmt:2` where it fits; the best two wsvd settings from Test 1, frozen and not retuned.

**Metrics.**
- rms_x C(x,t) error for |x − c| < 5;
- sum-rule drift;
- d²(t) error;
- span-4/5 coefficient error;
- Frobenius fidelity;
- wall time.

**Pre-registered success.** Over t ∈ [1, 2], wsvd is at least **1.5x better than `dmt:1`** on the time-averaged rms C(x,t) error at ≥ 2 χ values. It must also be no worse than `rw:γ`, and its sum-rule drift must stay below the C(x,t) error itself.

**Kill.** wsvd is within ±20% of the best of {`dmt:1`, `dmt:2`, `rw:γ`} at every χ. Or its gain from Test 1 is erased by accumulation: that would be backflow-limited error, exactly the 1902.01859 leak.

**Sanity requirement.** Before the comparison is trusted:
- DMT must hold the energy sum rule to about 1e-12;
- `svd` must lose it.

If DMT does not beat `svd` on C(x,t) at small χ, the implementation is suspect, not the idea.

**Ambiguous.** wsvd wins on span-4/5 but not on C(x,t). This would mean the extra content does not feed back into ≤ 3-site transport on these times, consistent with 2310.06886's "DMT no better than TEBD for t ≲ 20". Record it as a far-observable result only.

### Test 3: hydrodynamic window at N = 20 with a typicality reference (about 1 h)

Run only if Test 2 succeeds.

**Setup.** `isingT`, N = 20, q = ε_c, T = 4 (repo units = paper units for this model), χ ∈ {16, 32, 64}. Reference: typicality with 3 random states.

**Arms.** The Test 2 winners plus `dmt:1` and `rw:γ`.

**Metric.** D(t) against the reference over t ∈ [1, 4]. Also report the χ needed to reach 1% in D(t) per arm, the direct analogue of Fig. 13 of 2310.06886.

**Success.** wsvd reaches 1% at **≤ 0.7×** DMT's χ.

**Kill.** wsvd needs the same or more χ. Then rank 10 stays a control in the paper.

## Open questions

1. **Exact vs separable metric.** DMT's preserved set is "one side local" (Π_L⊗I + I⊗Π_R), which is a sum of Kroneckers, not a single one. The separable surrogate (I + λΠ_L)⊗(I + λΠ_R) over-weights the both-sides-local block. Is a two-term generalized SVD (alternating, or a Sylvester solve) worth its cost if Test 1 shows the separable surrogate falls short of DMT on span ≤ 3?
2. **Does spcf's quadratic machinery have any role here?**
   - Only one way is visible: treat |O⟩⟩ as a normalized state and target the operator-weight marginals, which is rank 1's object.
   - For transport, weight marginals do not enter C(x,t). The answer is probably "no", and that is the expected control outcome.
3. **rTEBD in the Heisenberg picture** has never been benchmarked. The gates S R S⁻¹ are non-orthogonal, so the conditioning at γ = 1.6 over about 200 Trotter steps should be checked.
4. **Leakage.** The 1902.01859 leak (errors in longer operators flowing down into 3-site RDMs) is the mechanism by which span-4/5 protection could matter. Does protecting span-4/5 measurably slow the drift of span-3 content? That is the physics question this control can answer cheaply.
5. **Public code.** The rTEBD and DMT (Jack-Kemp/dmt) repositories could not be accessed from this session. If the user adds them, use them to cross-check `rw:γ` and `dmt:n` on one cell before trusting the local reimplementations.
6. **Positivity** is not an issue for traceless Heisenberg operators. If this extends to finite-T MPDOs, DMT's "ameliorated, not guaranteed" positivity (1707.01506 Sec. IV B) applies to wsvd as well.
