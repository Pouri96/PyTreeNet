# Cut-local, local-RDM-preserving truncation inside DMRG machinery (scope: DMRG only)

Scope note: the coordinator narrowed the scope mid-task. TDVP is dropped entirely: no projection error, no 1TDVP vs 2TDVP, no global-Krylov time evolution, no CBE-TDVP, no BUG integrators. These notes cover ground-state and excited-state DMRG basis truncation and subspace choice, DMRG-specific truncation aimed at observables or RDMs, and MPS compression only where DMRG uses it. Non-DMRG prior art (DMT, rTEBD, LITE, LoMaC, tangent-space uMPS truncation) appears only in Q2, because the novelty assessment needs it.

Access caveat, which applies to everything below: arxiv.org, export.arxiv.org and api.semanticscholar.org were unreachable from this sandbox (DNS failure or proxy 403). Every finding therefore comes from search-engine abstracts and snippets of the cited pages. Treat every item as **abstract/snippet level** unless it says otherwise. No full text was read.

Background on our method, for the reader. At each two-site cut, keep rank chi but tilt the kept subspace one Gauss-Newton step in the tangent space, Q = qr(U_k + U_perp C), so that the 1-, 2- and 3-site RDMs within ±2 sites of the cut match those of the untruncated two-site tensor. Mechanism: it repairs SVD's first-order (∝ sqrt(discarded weight)) error in correlators across the cut, at the price of a first-order collateral change in far observables. It pays only when the cross-cut error is large, and a benefit/cost gate removes the losses.

## Q1. How do DMRG truncation and subspace-expansion steps choose the kept or added subspace?

### Takeaway
Every DMRG basis-selection scheme found picks or enlarges the kept space by an **energy/Hamiltonian-residual** criterion: White's density-matrix perturbation, 3S/AMEn subspace expansion, CBE, the McCulloch-Osborne RSVD variant, and zero-site DMRG. Kept states are then chosen by **largest eigenvalues of a (possibly mixed or perturbed) reduced density matrix**: plain SVD, multi-target weighted RDMs, and Legeza's entropy-based DBSS. None of them chooses the kept rank-chi subspace so that local RDMs or local observables near the cut are preserved.

### Cited Findings
**White's density-matrix perturbation (single-center-site DMRG)**
- White (2005) "develops a correction to the density matrix ... to take into account the incompleteness of the environment block", which allows a single center site instead of two and gives a "factor of two to four" speedup. Published as PRB 72, 180403. — [arXiv:cond-mat/0508709](https://arxiv.org/abs/cond-mat/0508709) (abstract level)
- Hubig et al. restate White's correction as ρ' = ρ + α Tr(L ρ L†), where α is a small heuristic "mixing factor". The kept space is still the leading eigenvectors of ρ', so expansion directions come from the left-block Hamiltonian terms acting on the state. — [arXiv:1501.05504](https://arxiv.org/abs/1501.05504) (snippet level; the exact operator form in White's original text was not verified)
- DMRG for the bosonic quantum Hall effect describes the same trick as "admixing information from the Hamiltonian to the density matrix", needed because the interactions are not short-range. — [arXiv:0909.3693](https://arxiv.org/pdf/0909.3693) (snippet)

**Strictly single-site DMRG with subspace expansion (DMRG3S), Hubig, McCulloch, Schollwöck, Wolf, PRB 91, 155115 (2015)**
- 3S is based on the subspace expansion of AMEn, the alternating minimal energy method from tensor-train numerics, where "the expansion term is built from the residual of the eigenproblem". 3S replaces that with "a numerically much more cheaply available expansion term". — [arXiv:1501.05504](https://arxiv.org/abs/1501.05504) (abstract/snippet)
- The enrichment changes the neighbouring tensors without changing the state; the newly added components start at zero. The method avoids local minima like density-matrix perturbation does, but more cheaply. — [arXiv:1501.05504](https://arxiv.org/abs/1501.05504)
- Reported performance: a speed-up of about (d+1)/2 per Hamiltonian application; time to convergence up to 2.5x faster than the previous single-site method and up to 3.9x faster than two-site DMRG on Fermi-Hubbard. Compatible with non-abelian symmetries. — [arXiv:1501.05504](https://arxiv.org/abs/1501.05504)

**Controlled bond expansion (CBE), Gleis, Li, von Delft, PRL 130, 246402 (2023)**
- CBE "identifies parts of the orthogonal space carrying significant weight in HΨ and expands bonds to include only these". It gives two-site accuracy and convergence per sweep at single-site cost. — [arXiv:2207.14712](https://arxiv.org/abs/2207.14712); [PRL DOI 10.1103/PhysRevLett.130.246402](https://link.aps.org/doi/10.1103/PhysRevLett.130.246402) (abstract/snippet)
- The truncation that selects expansion directions is a "shrewd selection" in two stages, preselection then final selection. The authors call it cheap and practical but "not strictly optimal", since optimal selection would cost as much as two-site DMRG. CBE uses no mixing parameter and is "fully variational": each update strictly lowers the energy. — [arXiv:2207.14712](https://arxiv.org/html/2207.14712v2) (snippet)
- The projector formalism behind CBE splits the space into kept and discarded parts, and the expansion targets the "discarded-discarded" (two-site tangent-orthogonal) component of HΨ. — [McCulloch & Osborne comment, arXiv:2403.00562](https://arxiv.org/abs/2403.00562) (snippet, their description of CBE)
- McCulloch & Osborne (2024) argue three points. Projecting onto the two-site tangent space "is unnecessary, and is generally not helpful". The 5 SVDs can be replaced by one QR plus a randomized SVD. Some variational claims are "incorrect". The same RSVD idea also applies to 3S. — [arXiv:2403.00562](https://arxiv.org/abs/2403.00562) (snippet)
- Reply by Gleis, Li, von Delft (Jan 2025). They welcome RSVD but reject dropping the projection. Without it, expansion may target directions already covered by the single-site update, which for DMRG is "computationally inefficient, but not compromise accuracy". They say the projection's cost is negligible. — [arXiv:2501.12291](https://arxiv.org/abs/2501.12291) (snippet)
- A TUM thesis that implemented CBE reports numerical fragility ("diverging SVD during the shrewd selection"), which it presents as fixable. The thesis concerns the TDVP variant, so it is out of scope except as an implementation warning. — [mediatum.ub.tum.de thesis](https://mediatum.ub.tum.de/doc/1741343/9n9qkqspat58j1idk0psojfv4.pdf) (snippet)

**Zero-site DMRG (DMRG0), Núñez-Fernández & Torroba, PRB 101 (2020)**
- DMRG0 optimizes the bond ("Schmidt-value") matrices rather than the site tensors, which separates optimization from decimation. It adds global perturbations from the "best low-rank update", chosen variationally as the MPS closest to one Lanczos or Jacobi-Davidson step. Convergence per sweep matches single-site enrichment methods at lower cost. — [arXiv:1908.10880](https://arxiv.org/abs/1908.10880v2) (abstract level)

**Multi-target / state-averaged truncation (excited states, dynamics in frequency space)**
- With several targets, the reduced density matrix is a weighted mixture ρ = Σ_f w_f ρ_f, and the kept basis is its leading eigenvectors. An equivalent route stacks √w_f-weighted matrices and takes one SVD, which is "often advantageous ... for multiple targeting in MPS DMRG". — [arXiv:1902.09621](https://arxiv.org/abs/1902.09621) (snippet)
- In correction-vector and dynamical DMRG, the basis targets the ground state plus the correction vector (H − E₀ − ω − iη)⁻¹ O|ψ₀⟩, so the kept space is chosen to represent a particular operator-applied state. DDMRG errors in spectra are O(ε²) rather than O(ε). Accuracy "depend[s] significantly and sometimes unpredictably on the specific states included as target". — [Jeckelmann, arXiv:cond-mat/0203500](https://arxiv.org/abs/cond-mat/0203500); [arXiv:0808.2620](https://arxiv.org/pdf/0808.2620) (snippet)

**Information-theoretic and other non-SVD state-count rules**
- Dynamic block state selection (DBSS) fixes the accuracy in advance and adapts the number of kept states to control truncation error. — [Legeza, Röder, Hess, arXiv:cond-mat/0204602](https://arxiv.org/abs/cond-mat/0204602) (abstract)
- Entropy-based criteria keep the "quantum information loss" χ = S(ρ) − p_typ S(ρ_typ) − (1−p_typ) S(ρ_atyp) below a threshold. This changes **how many** states are kept, not **which** ones; they remain eigenvectors of ρ. Works for two-site but not one-site DMRG. — [Legeza & Sólyom, arXiv:cond-mat/0305336](https://arxiv.org/abs/cond-mat/0305336); [arXiv:cond-mat/0401136](https://arxiv.org/abs/cond-mat/0401136); summarized in [Schollwöck RMP review, arXiv:cond-mat/0409292](https://arxiv.org/abs/cond-mat/0409292); [arXiv:1412.5829](https://arxiv.org/abs/1412.5829) (snippet)
- A 2025 "continuity-aware" truncation (Uhlmann-gauge, quantum-information-geometry framing) adds penalty terms so the retained subspace varies smoothly along sweeps or geometries. It notes that keeping the M largest singular values is "near-optimal locally for the ground state". — [Comput. Theor. Chem. 2025, ScienceDirect S2210271X25005596](https://www.sciencedirect.com/science/article/abs/pii/S2210271X25005596) (snippet; authors and arXiv ID not retrieved)

**Error structure of SVD truncation in DMRG (directly relevant to our mechanism)**
- Energy error is linear in the discarded weight ε, which is the basis of zero-truncation extrapolation. Other observables scale "theoretically" as √ε; an order parameter showed an empirical exponent near 0.5. — [Jeckelmann DMRG lecture slides](https://www.issp.u-tokyo.ac.jp/public/CQCP/26dmrg_issp1.pdf); [arXiv:0808.2620](https://arxiv.org/pdf/0808.2620); [arXiv:cond-mat/0510637](https://arxiv.org/pdf/cond-mat/0510637) (snippet)
- In momentum-space DMRG, the linear energy-vs-discarded-weight relation "often does not hold" even at small weight. — [arXiv:cond-mat/0110420](https://arxiv.org/abs/cond-mat/0110420) (snippet)
- Correlation-function work needs much smaller discarded weight (≤10⁻⁹ in one Bose-Hubbard study) than energies do. — [arXiv:cond-mat/9906019](https://arxiv.org/pdf/cond-mat/9906019) (snippet)
- Eckart-Young makes SVD truncation the best rank-D least-squares approximation, minimizing wavefunction error, **not** energy. The truncation makes two-site DMRG "not strictly variational", but the effect is said to be negligible. — [Density-matrix renormalization group: a pedagogical introduction, EPJB 2023 / arXiv:2304.13395](https://link.springer.com/article/10.1140/epjb/s10051-023-00575-2) (snippet)
- The Paeckel et al. review of MPS methods lists, as alternatives to SVD, "truncation based on the left or right density matrix" and "optimizing the local density matrix", citing refs. [73] and [74]. — [arXiv:1901.05824](https://arxiv.org/abs/1901.05824) (snippet; refs [73] and [74] could not be resolved)

### Inferences
- The DMRG schemes split cleanly into two layers. (i) *Expansion*: which new directions enter the enlarged bond space. These are always residual- or H-driven (H_L ρ H_L†, the AMEn residual, the HΨ discarded-discarded projection, a Lanczos/JD step). (ii) *Truncation*: which rank-chi subspace of the enlarged space is kept. This is always leading eigenvectors of a (perturbed or mixed) ρ. Our trick lives in layer (ii). It is orthogonal to (i) and composes with any of them: 3S, CBE and DMP all end in an SVD of an expanded tensor, and that SVD can be tilted.
- Our TEBD finding has a DMRG analogue: SVD leaves a first-order (√ε) error in non-energy observables, while energy error is only O(ε). In ground-state DMRG the energy is first-order stationary, so a first-order tilt of the kept subspace costs energy only at O(ε) too. The prefactor could go either way.
- Using a local-RDM criterion to **select expansion directions** in 3S or CBE looks weak for ground-state search. The target local RDMs have to come from a bigger state, and in single-site expansion no untruncated reference exists other than the two-site projected HΨ. Energy-residual selection already targets what the sweep optimizes. Any gain would come in the truncation layer, not in expansion.

### Gaps
- Exact operator form of White's 2005 perturbation (ρ' = ρ + α Σ_a H_L^a ρ H_L^a† vs. the Hubig restatement), and CBE's selection formulas, could not be verified from full text (arXiv was blocked).
- The Paeckel review's refs. [73] and [74] ("optimizing the local density matrix") were not identified. They may be the closest DMRG/MPS prior art and **must be checked**.
- I found no study that quantifies observable error (as opposed to energy) of 3S, CBE or DMP relative to two-site SVD truncation at equal chi.

## Q2. Prior art for choosing or tilting the kept subspace to preserve local RDMs or observables

### Takeaway
The closest prior art is **outside DMRG**: White-Zaletel-Mong-Refael **density matrix truncation (DMT)** for MPDOs. It exactly preserves all operators of up to 3-site diameter around the cut by an explicit basis change in operator space, using part of the bond budget. It is the conceptual parent of "local-RDM-preserving truncation", but it works on density operators, reserves rank for locality rather than tilting, and is a TEBD-type method. Nearby work includes reweighted TEBD (low-weight operators prioritized), LITE (local-information evolution), LoMaC (conservative orthogonal projection in low-rank kinetic solvers) and tangent-space variational truncation of uMPS (global fidelity objective). Inside DMRG, the nearest analogues are (a) multi-target RDM weighting toward operator-applied states and (b) DMRG0's "optimal low-rank correction" for energy. I found **no** paper that tilts a pure-state rank-chi SVD subspace by a tangent-space step toward a local-RDM objective, in DMRG or elsewhere. Search coverage is incomplete, so this is a weak negative.

### Cited Findings
**DMT (closest conceptual prior art; a TEBD/MPDO method, not DMRG)**
- DMT approximates 1D density operators and argues that the Frobenius norm on density matrices and the norm on purifications are "not the correct notions of error". Its truncation keeps the energy density and other local conserved quantities exact. — [White, Zaletel, Mong, Refael, arXiv:1707.01506, PRB 97, 035127 (2018)](https://arxiv.org/abs/1707.01506) (abstract/snippet)
- Construction: Schmidt-decompose the density matrix as an operator, then "cleverly choose basis changes on the spaces of left and right operators". Exactly preserves expectation values of all operators on regions of up to three sites in diameter. Part of the bond dimension (2^ℓ in operator language) is reserved to carry all ℓ-site information around the cut; the remaining budget keeps the largest remaining correlations. It does **not** preserve longer-range two-body correlators such as ⟨σ_i σ_{i+3}⟩. — [arXiv:1707.01506](https://arxiv.org/pdf/1707.01506); [Ye, Machado, White, Yao, arXiv:1902.01859](https://arxiv.org/html/1902.01859) (snippet)
- DMT is described as "difficult to analyze analytically". — [arXiv:2412.08730](https://arxiv.org/abs/2412.08730) (snippet)

**Reweighted TEBD (rTEBD), 2024**
- At each truncation of an MPDO, rTEBD down-weights high-weight operator components by γ^(−n), where n is the operator weight. This prioritizes low-weight expectation values and "preserves conserved quantities to high precision", whereas plain TEBD on MPDOs conserves them poorly, even the trace. An automated review calls the gain heuristic with limited error analysis. — [arXiv:2412.08730](https://arxiv.org/abs/2412.08730); [Pith review](https://pith.science/paper/2412.08730) (snippet)

**Local-information time evolution (LITE)**
- The information lattice yields a locally conserved von Neumann information current. LITE evolves local density matrices up to a maximum scale and discards information above a second scale, i.e. it removes "large-scale quantum information while preserving all local observables". Benchmarks: energy diffusion constant in the mixed-field Ising model. — [Kvorning, Herviou, Bardarson, arXiv:2105.11206 (SciPost 2022)](https://scipost.org/submissions/2105.11206v4); [Artiaco et al., arXiv:2310.06036](https://arxiv.org/abs/2310.06036) (abstract level)

**Conservative low-rank truncation in applied math (closest algorithmic analogue to a constrained or projected kept subspace)**
- LoMaC: after basis adding, the pre-compressed solution is "orthogonally projected onto the reference subspace defined by macroscopic densities". This gives exact local conservation of mass, momentum and energy, which plain SVD truncation destroys. Extended to hierarchical Tucker and to Vlasov-Maxwell (2026). — [Guo & Qiu, arXiv:2207.00518](https://arxiv.org/abs/2207.00518); [J. Sci. Comput. DOI 10.1007/s10915-024-02684-1](https://link.springer.com/article/10.1007/s10915-024-02684-1); [arXiv:2607.01381](https://arxiv.org/pdf/2607.01381) (snippet)
- An earlier conservative-SVD approach preserved mass and momentum but not energy. — [same LoMaC sources](https://arxiv.org/pdf/2607.01381) (snippet)

**Tangent-space or variational truncation with a non-SVD but global objective**
- Vanhecke, Van Damme, Haegeman, Vanderstraeten, Verstraete: a variational tangent-space algorithm for truncating uniform MPS to lower bond dimension, which matters for MPO×MPS. The objective is global fidelity. — [arXiv:2001.11882, SciPost Phys. Core 4, 004 (2021), DOI 10.21468/SciPostPhysCore.4.1.004](https://scipost.org/SciPostPhysCore.4.1.004) (abstract)
- Riemannian tensor-train optimization uses TT-SVD as the retraction. Alternative retractions (quasi-optimal, ST-HOSVD) are still judged by approximation quality. No objective-aware retraction was found. — [arXiv:2508.20928](https://arxiv.org/abs/2508.20928); [arXiv:2108.12163](https://arxiv.org/pdf/2108.12163) (snippet)
- AMEn-type enrichment of TT cores by (preconditioned) gradients "can equivalently be viewed as a subspace correction technique". — [EPFL MATHICSE report 05.2015](https://www.epfl.ch/labs/mathicse/wp-content/uploads/2018/10/05.2015_MS.pdf) (snippet)
- Weighted low-rank approximation, where a non-uniform weight replaces the Frobenius norm, has no closed-form SVD solution and needs local optimization with no global-optimality guarantee. This is the generic reason objective-targeted truncation is usually iterative, as in our one Gauss-Newton step. — [arXiv:1511.00649](https://arxiv.org/abs/1511.00649) (snippet)

**DMRG-internal analogues**
- Multi-target RDM weighting toward operator-applied states (correction vectors) chooses the basis to represent observables. It is still a weighted-eigenvector rule with no exact-match constraint. — [arXiv:1902.09621](https://arxiv.org/abs/1902.09621); [arXiv:cond-mat/0203500](https://arxiv.org/abs/cond-mat/0203500)
- DMRG0's "optimal low-rank correction" is a variational low-rank update of the state for **energy**. — [arXiv:1908.10880](https://arxiv.org/abs/1908.10880v2)
- "Correlation density matrices" for 1D chains based on DMRG study the RDM of two separated clusters as a diagnostic, not a truncation rule. — [arXiv:0910.0753](https://arxiv.org/pdf/0910.0753) (title/snippet only)

### Inferences
- **Novelty, by component.** (a) "Truncate so that local RDMs or local conserved densities are preserved" is **not novel as a goal**: DMT (2017) did it for MPDOs, LITE and rTEBD pursue it, and LoMaC does it in applied math. (b) Doing it for a **pure-state MPS at fixed chi by tilting the SVD kept subspace** (a tangent-space Gauss-Newton step from U_k with a QR retraction, target RDMs taken from the untruncated two-site tensor, no extra bond budget) has **no prior art found**. (c) Applying it inside **DMRG**, ground state or multi-target, has no prior art found. Main risks: the unresolved Paeckel refs [73]/[74] and pure-state DMT variants that may exist but were not surfaced.
- Structural difference from DMT worth stating in a write-up. DMT gets *exact* preservation by spending part of the rank and working in operator space. Our method keeps rank chi for the state and gets approximate preservation through a first-order tilt. Our measured collateral damage to far observables (~√ε) is the price of not reserving rank. DMT avoids that trade-off by giving up fidelity.

### Gaps
- No pure-state (wavefunction) version of DMT was identified. A search for "pure-state DMT" or "local-observable-preserving MPS truncation" returned nothing specific.
- Kloss, Bar Lev, Reichman: the ID given in the brief (1712.03996) did not resolve. The paper is **arXiv:1710.09378** (PRB 97, 024307). It is TDVP, so out of scope now; noted only as an ID correction. — [arXiv:1710.09378](https://arxiv.org/pdf/1710.09378)

## Q3. Concrete insertion points in DMRG, expected benefit, and novelty risk

### Takeaway
The most promising insertion points are where DMRG produces a **final compressed MPS whose local observables, not energy, are the deliverable**. The best is final-sweep or post-hoc compression from chi_big (or the two-site chi·d) down to chi. The second is multi-target or operator-targeted DMRG. Using it inside the converged ground-state sweep cannot beat the energy and is expected to trade energy for local observables. Using it in subspace-expansion **selection** (3S, CBE, DMP) is expected to give little.

### Cited Findings (anchors for each insertion point)
- In two-site DMRG and in 3S, CBE and DMP, every local step ends in a truncation (SVD or ρ-eigenvectors) of an enlarged two-site or expanded tensor. That truncation is the hook. — [arXiv:1501.05504](https://arxiv.org/abs/1501.05504); [arXiv:2207.14712](https://arxiv.org/abs/2207.14712); [arXiv:cond-mat/0508709](https://arxiv.org/abs/cond-mat/0508709)
- SVD truncation minimizes wavefunction error, not energy. Observables other than H carry √ε error while energy carries ε. — [EPJB 2023](https://link.springer.com/article/10.1140/epjb/s10051-023-00575-2); [Jeckelmann slides](https://www.issp.u-tokyo.ac.jp/public/CQCP/26dmrg_issp1.pdf)
- Energy extrapolation in discarded weight assumes the SVD relation E − E₀ ∝ ε. — [arXiv:0808.2620](https://arxiv.org/pdf/0808.2620)
- Multi-target DMRG chooses the kept space via weighted RDM mixtures of operator-applied states. — [arXiv:1902.09621](https://arxiv.org/abs/1902.09621); [arXiv:cond-mat/0203500](https://arxiv.org/abs/cond-mat/0203500)

### Candidate insertion points
Fields: (a) known, (b) gap, (c) insertion, (d) expected benefit, (e) novelty risk.

1. **Final-sweep or post-hoc compression of a converged DMRG state (chi_big → chi), e.g. for storage, measurement, or as the initial state of a quench.**
   - (a) Standard practice is an SVD sweep or variational fit, which is fidelity-optimal ([arXiv:1901.05824](https://arxiv.org/abs/1901.05824); [arXiv:2001.11882](https://arxiv.org/abs/2001.11882)).
   - (b) No observable-targeted pure-state compression found.
   - (c) Replace each SVD in the compression sweep with the tilted QR, with targets from the chi_big two-site tensor.
   - (d) Ground states have small ε and short correlation length, so cross-cut two-site errors are small. Our gate would usually choose SVD, giving a modest gain at best. It may pay for critical or 2D-cylinder states where ε at fixed chi is large (inference).
   - (e) Low to moderate risk. Conceptually adjacent to DMT ([arXiv:1707.01506](https://arxiv.org/abs/1707.01506)), but pure-state and tilt-based.

2. **Two-site DMRG (or the 3S/CBE truncation stage) in the final sweeps, with the tilted truncation in place of SVD.**
   - (a) The kept space is leading ρ-eigenvectors, with no observable criterion ([arXiv:1501.05504](https://arxiv.org/abs/1501.05504)).
   - (b) Untested.
   - (c) Same hook as in TEBD: the targets are RDMs of the optimized two-site tensor.
   - (d) Converged single-site DMRG is a variational energy minimum on the fixed-chi manifold, so at convergence any tilt can only **raise** the energy (inference from the variational principle). The tilt therefore *trades* energy for local non-energy observables (order parameters, short-range correlators across cuts), much as DMT trades fidelity for locality. It also breaks the SVD-based energy-vs-ε extrapolation ([arXiv:0808.2620](https://arxiv.org/pdf/0808.2620)). For nearest-neighbour H, the bond energies within ±2 sites of the cut are among the preserved 2-site RDMs, which may offset the energy loss locally (inference).
   - (e) No prior art found, but expected benefit is unclear or negative for energy. Worth one small experiment (TFIM near criticality, compare ⟨σ^z_i σ^z_{i+1}⟩ and ⟨σ^x⟩ at equal chi).

3. **Multi-target or operator-targeted DMRG (excited states, correction vector / DDMRG, Lanczos-vector spectra).**
   - (a) The kept space is from ρ = Σ w_f ρ_f; accuracy depends "sometimes unpredictably" on the chosen targets and weights ([arXiv:0808.2620](https://arxiv.org/pdf/0808.2620); [arXiv:1902.09621](https://arxiv.org/abs/1902.09621)).
   - (b) No constrained alternative to weight tuning.
   - (c) Keep ground-state SVD rank chi, then tilt so the local RDMs of the ground state and of the secondary targets (e.g. O_i|ψ₀⟩ for O near the cut) match their untruncated values. This replaces heuristic weights with explicit local constraints.
   - (d) Plausibly the best DMRG fit. Several states compete for one basis, so cross-cut errors of the secondary states are large under any single eigenvector rule, which is the regime where our gate says the method pays (inference).
   - (e) Moderate risk: "targeting" is the same idea in spirit, but no tilt or RDM-constraint variant was found.

4. **Expansion-direction selection in 3S, CBE or DMP (choosing added directions by local-RDM repair).**
   - (a) Selection is by HΨ weight or residual ([arXiv:2207.14712](https://arxiv.org/abs/2207.14712); [arXiv:2403.00562](https://arxiv.org/abs/2403.00562); [arXiv:2501.12291](https://arxiv.org/abs/2501.12291)).
   - (c) Score candidate directions in the discarded space by how much they would reduce the local-RDM mismatch.
   - (d) Low: no untruncated reference RDM exists in single-site expansion, and energy residual already matches the DMRG objective (inference).
   - (e) Novel but probably not useful.

5. **Infinite DMRG or uMPS truncation.**
   - (a) Tangent-space variational truncation exists, with a fidelity objective ([arXiv:2001.11882](https://arxiv.org/abs/2001.11882)).
   - (c) Swap the fidelity objective for the local-RDM objective in that tangent-space step.
   - (d) Translation invariance means every cut is identical, so the collateral far-observable damage is coherent rather than random. The gate decision becomes global (inference).
   - (e) Moderate risk, since the Vanhecke et al. machinery is already tangent-space based.

### Inferences
- Ranking for DMRG: (3) multi-target > (1) post-hoc compression of hard states > (5) iDMRG > (2) in-sweep ground-state truncation > (4) expansion selection.
- The benefit/cost gate carries over directly. In DMRG, the per-cut cross-cut two-site RDM error can be computed from the same two-site tensor, so the gate costs nothing extra in memory.

### Gaps
- No quantitative data found on how the local-observable error of converged DMRG at fixed chi splits by distance from the cut, i.e. whether the cross-cut 2-site error dominates as it does in TEBD.
- No full-text checks were possible (arXiv blocked). Quantitative claims above are abstract-level: the 3S speedups, the CBE "two-site accuracy at one-site cost", and DMRG0's convergence.
