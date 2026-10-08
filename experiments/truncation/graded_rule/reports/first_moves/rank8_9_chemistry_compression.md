# Ranks 8 and 9 first moves: compressing a converged MPS before RDM evaluation, and post-hoc compression of ground states

## Summary verdict: weak for rank 8 as written, conditional for a reformulated rank 8, weak for rank 9

The full texts confirm the report's premise. Chemistry codes do compress a converged DMRG reference to a smaller bond dimension M′ before evaluating high-order RDMs, using a "reverse schedule" of sweeps. They also confirm two of the report's numbers. Three findings change the picture.

1. **The quantity chemistry cares about is not near the cut.** SC-NEVPT2 contracts the full 4-RDM (or 4-PDM) with all active-space integrals. A ±2-site window covers a vanishing fraction of these elements: about 4 of the b(N−b) cross-cut index pairs at the centre cut, or 4% at N = 20. Nothing in the literature suggests that orbital ordering moves the energy-relevant RDM error near the cuts. The locality obstacle is real.
2. **There is little headroom where the report placed the method.** DMRG-SC-NEVPT2 is very insensitive to M′:
   - Cr₂: M′ = 800 vs 1200 changes D_e by 0.003 eV.
   - PPV: M′ = 500 vs 750 changes excitation energies by < 0.6 meV.
   - Freitag et al.: "very insensitive to the number of renormalized block states m".

   The real damage from low M sits elsewhere:
   - **partially contracted NEVPT2 develops "false intruder states"** at low m, in one case giving a spin-state ordering wrong by 34 kcal/mol;
   - in the time-dependent formulation, the 3-GF contributions need about 2× the bond dimension of everything else.

   Both of these are global or spectral effects, not local-RDM effects.
3. **N-representability is not the selling point the report suggests.** A compressed pure MPS automatically gives N-representable RDMs, but so does plain SVD compression. The literature's failure mode, false intruders, comes from the RDMs being *inconsistent with an eigenstate of the active-space Hamiltonian*, not from non-representability. The diagnostic for it is the spectrum of the Koopmans matrices (Guo, Sivalingam, Neese 2021).

**Conditional rescue for rank 8.** Keep the compression sweep, but replace the ±2 Pauli window with **block-renormalized cross-cut targets**: every 1-RDM element, plus selected 2-RDM elements γ_pq/Γ_pqrs with indices on both sides of the cut. Express them through the left- and right-block operators that DMRG RDM codes already build. This is the only version that addresses the locality obstacle, and it is a real code change.

**Rank 9.** Plain post-hoc compression of ground states faces a strong, cheap competitor: running DMRG directly at the small χ, which is energy-optimal and therefore already optimizes a sum of local terms. Expect to drop rank 9 unless the first test surprises.

**Recommendation.**
1. Spend about one day on the diagnostic **T0**, which needs no optimizer changes. It decides whether a local-window spcf could ever matter here.
2. Spend at most a few more days on the oracle **T2**, and only if T0 shows the error is non-local.
3. Fold rank 9 into the T1 control arm and do not build a uMPS version.

## Literature, verified from full text

PDFs are saved in `/tmp/papers_rank8_9_chemistry_compression`.

| Paper | Read | Verified content (corrections in bold) |
|---|---|---|
| Guo, Watson, Hu, Sun, Chan, **1512.08137** (DMRG-SC-NEVPT2, JCTC 12, 1583) | Full text | §2.3, quoted: "it is generally necessary to evaluate the 4-RDM using a lower bond dimension than is used in the DMRG energy optimization … done in practice by carrying out a 'reverse schedule' set of sweeps in a DMRG code, where M is decreased". The 4-RDM costs O(k⁸M² + k⁵M³). RDMs are evaluated on a **one-site** MPS after a two-site → one-site final sweep (§2.1). Cr₂ (12e,22o): M = 4000 compressed to M′ = 800; a 0.2 mEh kink near 1.85 Å is attributed to M′. Table 1: M′ = 800 vs 1200 gives D_e 1.435/1.432 eV, R₀ 1.655/1.656 Å, ω_e 469/470 cm⁻¹. PPV (22e,22o): M = 3200 compressed to M′ = 500 and 750, differing by "less than 0.6 meV" (Table 3). Split-localized (Pipek-Mezey) orbitals were used. |
| Motta et al., **1705.01608** (H-chain benchmark) | NEVPT2 methods section (§d) | Verified: H₃₀, (30,30) DMRG-CASSCF at D = 1000 (error < 0.1 mEh), compressed to D = 500, "compression error in the total energy of less than 0.3 mEh except at the shortest geometry, R = 1.0 a_B, where it was 10 mEh". Split-localized orbitals. **The report's "D = 500 … 10 mEh at the shortest bond" is correct. It is a total-energy compression error at the most delocalized, most entangled geometry.** |
| Sokolov, Guo, Ronca, Chan, **1703.10830** (t-MPS-NEVPT2, sc-MPS-NEVPT2) | §IV A-C | **This paper is missing from the report.** It reports an sc-MPS-NEVPT2 "that avoids computation of the four-particle reduced density matrix", which weakens the 4-RDM-bottleneck motivation. It also gives the best benchmark template. N₂ (10e,10o)/cc-pVQZ: reference at M′ = 1000, compressed to M = 50/100/200, with errors < 0.1 mEh at M = 200 everywhere and slower convergence near dissociation (r ≥ 2.1 Å). The 1-/2-GF terms converge with M, but the 3-GF terms need about 2M ("we recommend … ∼2M"). Cr₂: M′ = 6000 compressed to M = 500…2000; the 3-GF terms move −1.9 mEh (500→800) and −1.2 mEh (800→1200). Polyenes: 500 compressed to 250. |
| Freitag, Knecht, Angeli, Reiher, **1608.02006** (CD-DMRG-NEVPT2) | Full text | Table I, heme model (14,18): SC-NEVPT2 spin gaps vary < 1 kcal/mol over m = 256…1024. **PC-NEVPT2 at m = 256 places the triplet 34.1 kcal/mol *below* the quintet (qualitatively wrong).** First-order norms reach 2230 (Fig. 2). These are "'false intruder states' … [that] originate from the approximations in the zeroth-order wave function". This is the one documented place where low-m reference error is chemically catastrophic. |
| Krompiec & Muñoz Ramo, **2210.05702** | Full text | **Source correction.** The report attributes "profound or even catastrophic" and "up to 2.5 Ha" to the ORCA 6.1 manual. They are from this paper. "Profound or even catastrophic" is its paraphrase of Guo-Sivalingam-Neese (JCP 154, 214111, 2021) on CU(4) in FIC-NEVPT2 for stretched N₂/Cr₂. The **2.5 Ha comes from their "CU(4)-RDM-filtered" scheme** (exact 4-RDM elements inserted wherever the cumulant 4-RDM is non-zero), N₂ CAS(10,8)/cc-pVDZ. It concerns element-wise RDM approximation, not MPS compression. The exact 4-RDM there is about 5% non-zero and the 4-PDM about 12%. |
| ORCA 6.0/6.1 NEVPT2 manual pages (WebFetch) | Pages fetched | Neither page contains the quotes the report attributed to it. They say CU "is prone to false intruder states and should thus be avoided", that PS (D4TPre = 1e-12) is the default, and that DMRG-NEVPT2 takes `nevpt2_MaxM` (example 2000). |
| Guo, Sivalingam, (Kollmar,) Neese, JCP 154, 214111 and 214113 (2021); Paper III, JCP 162, 144110 (2025) | **Abstract only** (Crossref) | CU "virtually always leads to intruder states". PS is stable with conservative cutoffs. EPS is accurate but not much cheaper. "A good indicator of intruder state problems … is the eigenspectra of the Koopmans matrices." FR-NEVPT2 is more robust to approximate references but needs the 5-RDM. |
| Sharma et al., **1808.06273** (DMRG-PDFT) | Full text | Needs only the 1-RDM and 2-RDM. DMRG-PDFT is *less* M-sensitive than DMRG itself (Tables 1-2: anthracene vertical gap 2.28→2.38 eV for M = 100→2000). The "compression" in its abstract means the DMRG bond dimension itself, not a reverse-schedule step. |
| Vanhecke, Van Damme, Haegeman, Vanderstraeten, Verstraete, **2001.11882** (SciPost Phys. Core 4, 004) | Full text | A *fixed-point iteration* maximizing the fidelity per site λ (Eqs. 5-17), with mixed-gauge tangent projector P_A (Eq. 10) and polar-decomposition retraction (Eqs. 15-16). Cost O(χ³Dd + χ²D²d²) vs O(χ³D³) for Schmidt truncation of MPO·MPS. Fig. 1: spin-4 Heisenberg vumps state, D_total ≈ 21600 → 700; fidelities per site 1−5.37e-5 vs 1−3.78e-5 (variational better; the sentence order is ambiguous). "Local truncation performs fairly well across the board". **No local observable is benchmarked. The report's "collateral identical at every cut and adds coherently" is our own inference, not the paper's.** |
| Phung, Wouters, Pierloot (JCTC 2016); Phung & Pierloot (JCTC 2019); Kurashige et al. (JCP 2014) | Abstract only | DMRG-cu(4)-CASPT2 with m up to 10000 on Fe porphyrin. Exact-4-RDM and cu(4) flavours "essentially converge to similar relative energies". |

Tool notes:
- `read_arxiv_paper` worked for every arXiv ID tried. The longest outputs went to tool-result files and were read with `jq`/`grep`.
- `search_semantic` and `search_openalex` often returned empty lists, and `search_google_scholar` once returned empty.
- arXiv keyword search returns much noise.
- The Guo-Neese papers have no arXiv versions and were not downloaded.

## Prior art and novelty risk

- **No prior art found** for compressing a quantum-chemistry MPS so as to preserve RDM elements, rather than fidelity or discarded weight. The searches covered RDM-preserving MPS compression, observable-aware truncation, constrained low-rank MPS, and N-representability with DMRG. Every chemistry compression read is fidelity-based: reverse-schedule sweeps or variational compression. The novelty risk for the chemistry application is **low**. That is consistent with the report, and partly a sign that the community sees no problem to solve (SC-NEVPT2 is insensitive to M′).
- Element-wise RDM repair is a different object and does not threaten novelty:
  - cumulant reconstruction, PS/EPS filtering, CU(4)-PDM-filtered (2210.05702);
  - projection onto approximate N-representability (Lanssens et al., 1707.01022).

  These repair the RDM, not the wavefunction, and they are the failure-prone alternatives that a wavefunction-level method would be pitched against.
- Rank 9 (post-hoc compression of ground states): the uMPS variational truncation (2001.11882) and the finite-chain variational fitting (Verstraete-Cirac, which `pytreenet/dmrg/variational_fitting.py` implements) are fidelity-targeted. Neither preserves observables. **The real threat to rank 9 is practical, not novelty**: DMRG run directly at the small χ.
- Rank 5's first-moves report (`rank5_dmrg.md`) found the Paeckel refs [73]/[74] to be Stoudenmire's note and DMT, so they add no prior art here either.

## Technical details for implementation

**Benchmarks others use.** Typical reference-to-compressed bond dimensions (M → M′):

| System | Active space | M → M′ |
|---|---|---|
| N₂ | (10e,10o) | 1000 → 50-200 |
| Cr₂ | (12e,22o) | 4000-6000 → 500-2000 |
| PPV | (22,22) | 3200 → 500/750 |
| H₃₀ | (30,30) | 1000 → 500 |
| polyenes | up to (24,24) | 500 → 250 |

- Metrics used in the literature are energies: total compression error in mEh, PT2 correction in mEh, and excitation energies or D_e in eV. Nobody reports RDM-element errors.
- Thresholds: 0.1 mEh counts as converged, 1 kcal/mol as "chemical accuracy".
- Orbitals are almost always **split-localized**, which is the friendliest case for locality.

**A dense-feasible chemistry analogue exists.** (10e,10o) has 20 spin orbitals, which maps to 20 qubits under Jordan-Wigner. That is the repo's 2^20 dense limit. pyscf is **not installed**, but `pip index` shows pyscf 2.14.0 is available. Without pyscf, use a model with analytic integrals:
- **PPP or extended-Hubbard chain** of L = 8 sites, giving 16 qubits.
- In the **site basis** it has local hopping plus long-range Ohno Coulomb n_i n_j, so its non-locality is diagonal.
- Rotate to the **Hückel-MO basis** to get fully non-local g_pqrs, as in chemistry with canonical orbitals.

This one knob separates "local orbitals" from "delocalized orbitals" and answers the orbital-choice question directly.

**Mapping onto spcfast.**
- Use interleaved Jordan-Wigner (↑,↓ per orbital, d = 2 per qubit). A fermionic element whose indices all fall inside the ±2-qubit window is a Pauli string inside that window, so `SPCFast` can target it unchanged. Elements spanning more than 4 qubits carry JW Z-strings and lie outside its target set.
- Particle number and S_z are conserved. The tilt C can leak out of the sector (`diag_charge.py` already measures this for S_z). **Leakage is a hard failure for chemistry**, because the RDM traces change. Report ⟨N̂⟩ and Var(N̂) of every compressed state.
- SVD can also break symmetry when singular values are degenerate. Cut on gaps or symmetrize.

**Compression sweep with the existing cut.**
- `SPCFast.__call__(theta, chi, dirn, A, B, b)` (`rule/spcfast.py`, line ~193) only needs `self.T` set by `start(T)` and the theta of the current bond.
- A left-to-right sweep over an exact right-canonical MPS (built by SVD from the dense vector), with no gates, is exactly the "reverse schedule" step.
- Use `taus=()` (static targets only). The `model` argument then only enters the region Hamiltonian, which is unused when wE = 0.
- **Feasibility check, run once on a TFIM N = 12 ground state at χ = 4:** both the SVD and spcf sweeps complete. The first spcf call costs about 15 s to build the cached region operator F; later calls are cheap. Nothing should be read into the two numbers.
- Memory at N = 20 qubits, 10 orbitals: a dense state is 16 MB (complex128). A 2-RDM over 20 spin orbitals has 1.6e5 elements, built from about 190 pair-annihilated vectors (3 GB at complex128). Block the pairs, or use N ≤ 16 for the first pass (trivial).

**Known failure modes to watch.**
- False intruders in PC/FIC-NEVPT2 at low M.
- Slow M convergence near dissociation (N₂ r ≥ 2.1 Å; H-chain R = 1.0 a_B).
- The two-site → one-site final-sweep convention (RDMs need a single well-defined MPS).
- spcf's own first-order collateral on non-targeted far observables.

## Where to start (code changes and effort)

All new code goes in a new script. Existing modules are unchanged except for one optional hook.

1. **`compress_bench.py`** (new, ~250 lines, about 1 day).
   - Builds the dense ground state with `scipy.sparse.linalg.eigsh`. Generalize `exact_ref.sparse_H` to accept a list of arbitrary (Pauli-string, coefficient) terms, so that long-range and JW-fermionic Hamiltonians fit.
   - Builds the exact MPS by successive SVD and runs one compression sweep with a pluggable cut: `mpsenh.svd_cut` or `spcfast.SPCFast(...)`, with `a=2, taus=(), fw=0.0` as in the feasibility check.
   - Computes metrics against the exact state:
     - fidelity;
     - energy error;
     - all 1- and 2-body correlators (or γ_pq and Γ_pqrs), **binned by index span** (span ≤ 4 qubits = "window-local", otherwise "far");
     - ⟨N̂⟩ and Var(N̂).
2. **Proxy Hamiltonians** (inside the same script, about 0.5 day):
   - (a) TFIM from `mpsenh.h_bond('ising')` (local control);
   - (b) long-range Ising, J_ij = |i−j|^(−α) Z_iZ_j + h Σ X_i, with α ∈ {0.5, 1.5, 3}, natural or random site order;
   - (c) PPP/Hubbard chain, L = 8, interleaved JW, in the site basis and in the Hückel-MO basis, with three orderings: natural, Fiedler (from exact mutual information), and random.
3. **Optional target hook in `rule/spcfast.py`** (about 0.5 day): let `__call__` accept an external target window RDM in place of the pre-cut theta's own. In a compression sweep the exact reference is known, so the cut can target the *original* state's RDMs and absorb earlier cuts' errors. This is the "error feedback" item from the applications report, and it comes for free here.
4. **Only if T0/T1 justify it: `rule/blockcut.py`** (new, 2-3 days). Same tangent parametrization Q = qr(U_k + U_⊥C), but the residual is r_O = ⟨ψ(Q)|O|ψ(Q)⟩ − ⟨O⟩_exact over block-renormalized cross-cut operators O = O_L ⊗ O_R, as in the RDM algorithm of 1512.08137 §2.2. Use a†_p (left, with JW string) ⊗ a_q (right) for γ_pq, and n_p ⊗ n_q, then pair operators, for Γ. There are O(b(N−b)) targets per cut, each with an O(χ³) Jacobian column. Prototype with dense-assisted Jacobians at N ≤ 16 first.
5. pyscf N₂ or H₁₀ (10e,10o) on 20 qubits, about 1 day after `pip install pyscf`, only if T2 passes.

## First tests (decision tree)

All tests use dense references at 16-20 qubits on a CPU laptop. Each run takes minutes; building the F cache costs about 15 s per region shape.

### T0: locality budget of compression error (SVD only, no new optimizer)

**Setup.**
- Exact ground states of proxies (a)-(c).
- One SVD compression sweep to χ ∈ {4, 8, 16}. Pick χ per state so that infidelity falls in 1e-4…1e-2, which matches chemistry's few-to-tens of mEh regime.

**Metrics.**
- f_loc = (squared 1+2-RDM error in window-local elements) / (total).
- f_loc^E = the same, with errors weighted by the Hamiltonian coefficient of each element. This is the share of the *energy* error that spcf's window could see.
- Cross-tabulate f_loc^E by basis (site vs MO) and ordering.

**Pre-registered outcomes.**

| Outcome | Condition | Action |
|---|---|---|
| Kill local-window rank 8 | f_loc^E < 0.25 for both MO-basis PPP and α ≤ 1.5 Ising, at every χ | Go to T2 |
| Proceed to T1 on the non-local proxies | f_loc^E ≥ 0.4 on at least one non-local proxy | Run T1 there |
| Ambiguous | 0.25-0.4 | Run T1 anyway, but only on the best ordering |
| Side result | Ordering raises f_loc^E by ≥ 2× | The "ordering rescue" is real; record it |

### T1: spcf (as is) vs SVD vs fidelity-optimal compression; also serves as the rank-9 test

**Arms at matched χ.**
1. SVD sweep.
2. spcf sweep (`a=2, taus=(), fw=0`, plus `fw = 0.01` as the Levenberg-Marquardt damped variant).
3. spcf with external exact targets (hook 3).
4. Variational fitting to the exact state: 2-3 sweeps, from `pytreenet/dmrg/variational_fitting.py` or a dense-projection fit.
5. **Rank 9 only:** DMRG at the small χ (`pytreenet/dmrg/dmrg.py`).

**States.**
- Non-local proxies that passed T0.
- Local controls for rank 9: TFIM (`ising` fields, and the critical point h_x = 1, h_z = 0, which needs a new `FIELDS` entry), plus `heis`.

**Metrics.** Window-local RDM error, far RDM error, energy error, infidelity, ⟨N̂⟩ leak, wall-clock vs SVD.

**Rank 8 criteria.**
- *Success:* the energy error and the H-weighted 2-RDM error both fall ≥ 1.5× vs the best of SVD and variational fitting on a non-local proxy, with the far-RDM error rising ≤ 10% and no sector leak above 1e-10.
- *Kill:* < 1.2× on non-local proxies.

**Rank 9 criteria.**
- *Success:* spcf beats both SVD and DMRG(χ) on **non-energy** local observables (⟨X_i⟩, ⟨Z_iZ_{i+1}⟩, ⟨Z_iZ_{i+3}⟩) by a median ≥ 1.5×, and its energy is no worse than SVD's.
- *Kill:* it loses to DMRG(χ) on all of them. This is the expected outcome. In that case drop rank 9 and do not build the uMPS version.

**Ambiguous.** Gains appear only on the local TFIM. That is a rank-9-type result: report it as such, but do not count it as chemistry evidence.

### T2: the block-target oracle, the real test of a rescued rank 8 (only if T0 kills the local window)

**Setup.**
- MO-basis PPP, L = 8 (16 qubits).
- At each cut, target all cross-cut 1-RDM elements γ_pq plus diagonal pair densities ⟨n_p n_q⟩, then the full cross-cut 2-RDM.
- Exact targets. Dense-assisted Jacobian (a brute-force prototype is acceptable).

**Arms.** SVD, variational fitting, block-target tilt at the same χ, and SVD at χ + Δχ (the equal-parameter control).

**Metrics.**
- Energy error.
- Error of an RDM functional *not* used as a target: the 2-RDM contracted with a random 4-index tensor that has chemistry-like symmetry. This guards against overfitting to H.
- The smallest eigenvalue of the 2-RDM-derived Koopmans-like matrix (the generalized Fock / IP-EA matrices), as an early false-intruder proxy.

**Criteria.**
- *Success:* energy error ≥ 2× lower than the best of SVD and variational fitting at matched χ, the non-target functional improves ≥ 1.3×, and an equivalent χ saving of at least 1.4× (the cost cap in the applications report).
- *Kill:* < 1.3× on energy, or the non-target functional gets worse.
- *Ambiguous:* the energy gain comes entirely from 1-RDM targets. That means the tilt is just doing natural-orbital-aware truncation, so compare against an SVD performed in the natural-orbital basis before concluding anything.

**If T2 succeeds:** move to pyscf N₂ (10e,10o), M′ ∈ {16, 32, 64}. Score the exact SC-NEVPT2 and, most importantly, the **PC-NEVPT2** energy, which is the false-intruder channel. This is the first point at which a chemistry claim becomes possible.

## Open questions

1. Is the energy-weighted compression error in a localized-orbital chemistry MPS concentrated near cuts? T0 answers this on proxies. Real localized molecular orbitals may sit between the site-basis and MO-basis PPP extremes.
2. Does PC-NEVPT2's false-intruder sensitivity at low m (Freitag Table I) come from error in specific RDM blocks that a targeted compression could protect? Or does it come from the zeroth-order inconsistency (reference not an eigenstate of H_D), which no RDM-matching can fix? Guo-Neese's Koopmans-spectrum diagnostic is the natural measurement. Their full text was not available here.
3. With exact targets available in post-hoc compression, how much of any gain comes from absorbing earlier cuts' errors (hook 3) rather than from the local objective? This matters for every rank, not just 8/9.
4. Does sc-MPS-NEVPT2 (1703.10830), which avoids the 4-RDM, make the use case moot in modern codes? Its 3-GF compressed perturbers need about 2M, so the bottleneck moves to compressing *perturber* states. That is arguably a better target for a tilt that preserves observables, and it is worth a separate look.
5. uMPS: the claim that a uniform tilt's collateral adds coherently is untested. Test it only if T1's rank-9 arm unexpectedly succeeds.
