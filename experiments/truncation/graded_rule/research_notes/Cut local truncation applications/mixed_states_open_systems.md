# Cut-local, local-observable-preserving truncation in mixed-state and open-system tensor-network simulations

**How these notes were made (applies to every section):** In this session, arXiv, ar5iv, APS and the Semantic Scholar API could not be reached. WebFetch failed with DNS errors, and the curl CONNECT to export.arxiv.org got a 403 from the proxy. So **no full text was read**. Every finding below comes from search-engine snippets and abstract-level summaries. Treat each one as **abstract-level** unless it is tagged otherwise. A few items are tagged **[prior knowledge, not re-verified this session]**. They come from my earlier reading of the paper, the URL is given, and they should be checked before the report relies on them.

---

## 1. DMT in depth: mechanism, conservation, follow-ups, failure modes, and is there a pure-state analog?

### Takeaway
DMT (White, Zaletel, Mong, Refael) is a truncation of a matrix-product density operator (MPDO). It keeps every expectation value on regions of up to 3 sites around each cut exactly, so energy density and other local conserved densities are conserved to machine precision. It also largely avoids the loss of positivity that plain Frobenius (SVD) truncation causes. It works because, for a vectorized density matrix, local expectation values are *linear* in the tensor, so they can be preserved by a linear projection. In a pure-state MPS they are *quadratic* in the tensor, and a Gauss-Newton step like ours is needed instead. No published pure-state MPS truncation that preserves local reduced density matrices (RDMs) at a cut turned up in the searches. The closest items are a suggestion in the Paeckel et al. review and the pure-to-mixed "entanglement into mixture" methods (Section 2).

### Cited Findings
- DMT is introduced in "Quantum dynamics of thermalizing systems", arXiv:1707.01506, published as PRB 97, 035127 (2018). The abstract says the truncation "exactly preserves the energy density of the system and other local conserved quantities" — [arXiv:1707.01506](https://arxiv.org/pdf/1707.01506); [APS accepted ms](https://link.aps.org/accepted/10.1103/PhysRevB.97.035127)
- DMT "ameliorates the positivity problem of Frobenius truncation and exactly preserves the expectation values of all operators on all regions of up to three sites in diameter". The paper argues that Frobenius (Hilbert-Schmidt) truncation "is not the correct notion of error for density matrices". It also reports that Frobenius truncation drives the evolution into "non-physical states with Z<1", which come from negative eigenvalues — [arXiv:1707.01506](https://arxiv.org/pdf/1707.01506) (snippet level)
- Motivation in the paper: the RDMs form a BBGKY-like hierarchy. Two-site RDMs depend on three-site RDMs, three-site on four-site, and so on. DMT truncates this hierarchy at a fixed length and keeps the three-site level exact — [arXiv:1707.01506](https://arxiv.org/pdf/1707.01506) (snippet level)
- **Mechanism [prior knowledge, not re-verified this session]:**
  - At a bond, ρ = Σ_ij M_ij x_i^L ⊗ y_j^R in orthonormal operator bases on the left and right.
  - The left and right bases are gauged so that their first elements are the identity on the far part times the d² single-site operators on the site next to the cut. These are the "reserved" directions. Contracting with them gives every operator supported on {site left of cut, site right of cut} plus one neighbour.
  - The rows and columns of M that carry reserved operators are kept exactly. The remaining block is turned into a connected-correlation form (roughly M − M_{·0} M_{0·}/M_{00}), SVD-truncated, and recombined.
  - So all expectation values that live only in the reserved sector, i.e. all ≤3-site operators across the cut, are unchanged. The trace (the identity-identity element) is preserved exactly.
  - Source to check: [arXiv:1707.01506](https://arxiv.org/pdf/1707.01506)
- An independent description, from the rTEBD paper: DMT "conserve[s] local conserved quantities nearly exactly, up to machine precision errors" by "altering the SVD truncation so that all nearest-neighbor 3-body expectation values are not changed". Stated limitation: DMT's truncation "fails to maintain longer-range two-body correlations" — [arXiv:2412.08730](https://arxiv.org/pdf/2412.08730) (snippet)
- The same source notes that plain TEBD on an MPDO "does a poor job at conserving conserved quantities, including the trace". It also notes that a gate on an MPDO changes the bond and inner dimensions together, which makes truncation harder than for a pure MPS — [arXiv:2412.08730](https://arxiv.org/html/2412.08730) (snippet)
- Failure mode (the hierarchy leaks):
  - The Floquet-DMT follow-up says DMT preserves the RDM of subsystems of size ≤ℓ at the moment of truncation. It "fails to accurately capture the ℓ-site entropy to the extent that errors in longer ranged operators propagate down to the three-site density operators via the system's dynamics."
  - This snippet is attributed to the Ye et al. line of work. The search returned both [arXiv:1902.01859](https://arxiv.org/html/1902.01859) and [PRB 99, 054307 accepted ms](https://link.aps.org/accepted/10.1103/PhysRevB.99.054307), and which paper it comes from is uncertain.
- Follow-up 1:
  - Ye, Machado, White, Mong, Yao, "Emergent hydrodynamics in non-equilibrium quantum systems", arXiv:1902.01859, PRL 125, 030601 (2020).
  - They use DMT on Floquet chains and find it captures prethermalization and late-time heating to infinite temperature. It also handles spatially inhomogeneous drives, where Floquet heating interacts with diffusion.
  - In static chains they simulate generic chains up to L=100 and extract energy diffusion coefficients.
  - [arXiv:1902.01859](https://arxiv.org/pdf/1902.01859); [Yao group page](https://yao.physics.harvard.edu/?p=2042)
- Follow-up 2: Ye, Machado, Kemp et al., "Universal Kardar-Parisi-Zhang dynamics in integrable quantum systems", PRL 129, 230602 (2022). It uses DMT for KPZ transport in integrable chains. Known only via the group page — [Yao group page](https://yao.physics.harvard.edu/?p=2042)
- Follow-up 3:
  - DMT was used for the simulations in the KPZ quantum-gas-microscope experiment (Wei et al.), arXiv:2107.00038.
  - That paper's stated rationale: "DMT preserves local observables rather than maximizing mutual information, which allows DMT to correctly capture late-time equilibration and hydrodynamics."
  - It also contains TEBD results giving z = 3/2.
  - [arXiv:2107.00038](https://arxiv.org/pdf/2107.00038) (snippet)
- Benchmark:
  - Yi-Thomas, Ware, Sau, White, "Comparing numerical methods for hydrodynamics in a 1D lattice spin model", arXiv:2310.06886, PRB 110, 134308.
  - It compares TEBD, TEBD+DMT, recursion with the universal operator growth hypothesis (R-UOG), and operator-size-truncated (OST) dynamics at infinite temperature.
  - "DMT and OST give consistent dynamical correlations to t = 60/J and diffusion coefficients agreeing within 1%." Plain TEBD "only converges for t ≲ 20" but still gives D within 1%. R-UOG fails to converge.
  - Neither DMT nor OST shows long-time tails in the current-current correlator.
  - The paper has an appendix on "DMT for traceless operators", which is DMT applied in the Heisenberg/operator picture.
  - [arXiv:2310.06886](https://arxiv.org/html/2310.06886v2); [UMD RQS](https://rqs.umd.edu/publications/comparing-numerical-methods-hydrodynamics-one-dimensional-lattice-spin-model)
- Follow-up 4: fermionic DAOE (FDAOE, Kuo et al.) uses DMT as its *reference*. DAOE gave "inconsistent diffusion coefficients for different cutoff lengths ℓ*, and none matched the DMT reference" on a weakly interacting Majorana chain. With FDAOE they find D ∝ (interaction)^-4, contrary to naive Fermi's Golden Rule — [arXiv:2311.17148](https://arxiv.org/pdf/2311.17148) (snippet)
- Public code: an ITensor-based DMT implementation (Jack Kemp) — [github.com/Jack-Kemp/dmt](https://github.com/Jack-Kemp/dmt). C. D. White's PhD thesis has a chapter on DMT — [Caltech thesis 2019](https://thesis.library.caltech.edu/11558/1/white_christopher_2019.pdf) (table of contents only)
- On pure-state analogs, the Paeckel et al. review lists as "further interesting possibilities" truncation "based on the left/right density matrix or by optimizing the local density matrix which aims at preserving local observables". These are pointers, not worked-out algorithms — [arXiv:1901.05824](https://arxiv.org/pdf/1901.05824) (snippet; the refs it points to, numbered 73/74 in the snippet, were not identified)
- The same review notes TDVP conserves energy and norm exactly in the pure-state setting. This is a global conservation, not local-RDM preservation — [arXiv:1901.05824](https://arxiv.org/pdf/1901.05824) (snippet)
- A search snippet states that if you truncate at every bond in a sweep, "only local observables with support on up to two neighboring sites remain unaffected", and that errors in larger-support observables feed back into the dynamics of two-site observables. The search engine did not identify which paper this comes from; it is likely 2412.08730 or 1901.05824 — (unattributed; verify)

### Inferences
- **Why DMT can be exact and we cannot.** For a vectorized MPDO |ρ⟩⟩, Tr(O ρ) = ⟨⟨O|ρ⟩⟩ is *linear* in the two-site tensor. Keeping a set of local expectation values fixed is therefore a linear constraint, and DMT satisfies it exactly by reserving rows and columns. For a pure MPS, ⟨ψ|O|ψ⟩ is *quadratic*, the rank-χ constraint is non-convex, and one Gauss-Newton tilt (our Q = qr(U_k + U_⊥C)) is the natural cheap approximation. Our method can reasonably be described as "a pure-state, approximate analog of DMT". The literature found only *suggests* such a thing (Paeckel review) or takes a different route, converting pure to mixed (Surace et al.; Frías-Pérez et al., Section 2).
- DMT's documented weakness is that long-range two-body correlations are not maintained and errors leak down the hierarchy through the dynamics (2412.08730; 1902.01859). This parallels our measured "collateral first-order change of far observables". In DMT the reserved block is protected exactly and the error is pushed into the non-reserved (longer-range) sector. That is a warning that any local-RDM-preserving scheme moves error to longer distances. Measuring far-observable drift should be a standard diagnostic in our paper.
- DMT is designed for the late-time, near-equilibrium (hydrodynamic) regime, where the state approaches local equilibrium and long-range operator content is irrelevant. Our method's measured benefit is in the *growth phase* of entanglement. The two regimes are complementary, which helps position our work.

### Gaps
- The exact formulas for DMT's basis change and its connected-matrix subtraction could not be checked against the full text (arXiv was unreachable). Also unverified: the numerical claims of 1707.01506 (chain lengths, χ values, convergence of D with χ) and whether DMT exactly keeps positivity or only reduces its violation (the abstract says "ameliorates").
- No 2024–2026 papers proposing DMT improvements, 2D or tree-tensor-network (TTN) extensions, or Lindblad-DMT were found (searched; nothing returned). This is an absence of search hits, not proof that none exist.
- No direct DMT vs TDVP vs purification head-to-head beyond 2310.06886 (which covers TEBD, DMT, OST, R-UOG) and 2311.17148 (DMT vs DAOE/FDAOE) was found.

---

## 2. LITE and other "local information" / "entanglement into mixture" schemes

### Takeaway
LITE (Artiaco, Fleckenstein, Aceituno Chávez, Klein Kvorning, Bardarson; PRX Quantum 5, 020352, 2024) does not use an MPS or SVD. It evolves overlapping subsystem density matrices on an "information lattice". Information at scales ≥ ℓ_min is removed in a way that keeps all lower-level RDMs and the information currents. The cost is exponential in ℓ_max and linear in system size, and accuracy is controlled by ℓ_min. The Tagliacozzo group's "entanglement into mixture" line also targets local RDMs specifically. It turns pure states into effective mixtures rather than truncating a pure state at fixed χ.

### Cited Findings
- LITE: "the algorithm removes large-scale quantum information in a way that preserves all local observables". The premise is that information flows to larger scales "without returning to local scales" — [arXiv:2310.06036](https://arxiv.org/abs/2310.06036); [PRX Quantum 5, 020352](https://doi.org/10.1103/PRXQuantum.5.020352)
- LITE decomposes the system into subsystems whose von Neumann equations are solved in parallel. ℓ_min is the scale at which information is removed, "while lower-level density matrices and information currents are preserved". ℓ_max is the largest scale on which time evolution is performed — [arXiv:2310.06036](https://arxiv.org/html/2310.06036) (snippet)
- LITE cost: "computational complexity scales exponentially with the subsystem level ℓ_max but increases only linearly with the total system size" — [arXiv:2310.06036](https://arxiv.org/pdf/2310.06036) (snippet)
- LITE on the mixed-field Ising model:
  - D(t) is ballistic at short times and plateaus for t ≳ 20. The plateau oscillations are algorithmic artifacts that shrink as ℓ_min grows.
  - The average D "does not depend largely on ℓ_max, while it strongly depends on ℓ_min". Convergence is reached at modest cost.
  - The paper also treats magnetization transport in an *open* XX chain with dephasing. Seminar abstracts say it reproduces NV-center spin-texture experiments.
  - [arXiv:2310.06036](https://arxiv.org/pdf/2310.06036) (snippet); [IPhT seminar](https://www.ipht.fr/en/events/the-time-evolution-of-local-information/); [LPTMS seminar](https://www.lptms.universite-paris-saclay.fr/?p=273848)
- The predecessor is Klein Kvorning, Herviou, Bardarson, "Time-evolution of local information: thermalization dynamics of local observables", SciPost Phys. 13, 080 (2022). A referee comment on it says the best literature D seemed ≈0.55, about 20% off from the D ≈ 0.45 the paper claimed, and the authors disputed this — [arXiv:2105.11206](https://arxiv.org/abs/2105.11206); [SciPost](https://scipost.org/SciPostPhys.13.4.080); [SciPost submission page](https://scipost.org/submissions/2105.11206v1/)
- There is a third-party LITE code for topological applications — [github.com/YuliyaBilinskaya/topology_with_LITE](https://github.com/YuliyaBilinskaya/topology_with_LITE). The information lattice is also used to witness short- and long-range nonstabilizerness — [arXiv:2510.26696](https://arxiv.org/html/2510.26696v1)
- Surace, Piani, Tagliacozzo, "Simulating the out-of-equilibrium dynamics of local observables by trading entanglement for mixture", PRB 99, 235115 (2019). It converts entanglement between distant regions into classical mixture while keeping the local RDMs unchanged. It targets long-time equilibrium values of local operators and is tested on quenches of quadratic fermionic Hamiltonians — [arXiv:1810.01231](https://arxiv.org/pdf/1810.01231); [Strathprints](https://strathprints.strath.ac.uk/68990)
- Frías-Pérez, Tagliacozzo, Bañuls, "Converting long-range entanglement into mixture: tensor-network approach to local equilibration", PRL 132, 100402 (2024). It identifies long-range entanglement and turns it into mixture, giving a density-matrix description that captures the long-time behaviour of local operators at finite cost — [arXiv:2308.04291](https://arxiv.org/pdf/2308.04291); [Benasque slides](https://www.benasque.org/2023scs/talks_contr/155_slidesMiguelFrias.pdf)
- Follow-up: Carignano, Lami, De Nardis, Tagliacozzo combine a TN local-equilibration algorithm with Monte Carlo and transverse contractions. Seen only in slides — [arXiv:2505.09714 per YITP slides](https://www2.yukawa.kyoto-u.ac.jp/~yitp-tns-2025/slide/Tagliacozzo.pdf)

### Inferences
- LITE and the mixture methods are *not* cut-level replacements for SVD. They change the representation itself: subsystem RDMs, or a mixed state built from a pure one. Our tilt keeps the pure-state MPS at fixed χ. It slots into existing TEBD codes and keeps fidelity/global structure, which LITE gives up by construction. This is a clean point of difference.
- LITE's ℓ_min plays a role close to our "region of 2 sites each side" (a 5-site window around the cut). The comparison between the two cut-local windows is natural to make. LITE preserves *all* RDMs below ℓ_min exactly. Ours preserves the 1-, 2- and 3-site RDMs approximately, to first order.
- LITE's statement that D depends mainly on ℓ_min suggests that local-RDM fidelity at small scales, not large-scale information, controls hydrodynamic accuracy. That supports the premise of local-RDM-targeted truncation. But it is a late-time claim, while our gains are in the growth phase.

### Gaps
- Whether LITE has been run in 2D was not confirmed (no hit).
- The specific D values and the cost tables of 2310.06036 could not be read.
- The algorithmic details of 2308.04291 (how it detects the long-range entanglement, its cost) are known only at abstract level.

---

## 3. Hydrodynamics and transport with MPO density matrices or operators: DAOE, TDVP, OST, and what truncation does to D

### Takeaway
Good diffusion constants come from methods that throw away long or large-scale operator content while protecting short operators or local RDMs: DMT, DAOE/FDAOE, OST and LITE. Fixed-χ TDVP can converge quickly but to a *wrong* D, so apparent convergence in χ is not enough evidence. Agreement within about 1% between DMT and OST has been reported (2310.06886). DAOE can depend on its cutoff (2311.17148).

### Cited Findings
- DAOE (Rakovszky, von Keyserlingk, Pollmann; PRB 105, 075131, 2022; arXiv:2004.05177):
  - It evolves the observable as an MPO in the Heisenberg picture and adds artificial dissipation that acts on long operators. This makes operator entanglement decay.
  - The error it induces in D "is exponentially small in ℓ*", though the numerical check of this is memory-limited.
  - [arXiv:2004.05177](https://www.arxiv.org/pdf/2004.05177); [INSPIRE](https://inspirehep.net/literature/2034009)
- FDAOE (Kuo et al., arXiv:2311.17148):
  - DAOE gives D values that depend on ℓ* and don't match DMT for weakly interacting Majorana chains. The fermionic version is more efficient at weak interaction and can be improved by lowering γ, at the cost of more SVD time.
  - Result: D ∝ (interaction strength)^-4.
  - [arXiv:2311.17148](https://arxiv.org/pdf/2311.17148); [JQI pdf](https://hub.jqi.umd.edu/sites/default/files/2023-12/2311.17148.pdf)
- DAOE extension (arXiv:2408.08249, PRB 2025):
  - Non-local operators are replaced by their ensemble averages instead of being discarded. This "greatly reduces the operator entanglement entropy".
  - It finds D ∝ 1/ρ at low density. Converged values come from extrapolation in the cutoff, which gets harder because operator entanglement saturates late.
  - [arXiv:2408.08249](https://arxiv.org/abs/2408.08249); [PRB](https://journals.aps.org/prb/abstract/10.1103/gz9n-v8ty)
- Open-system variant: "Open-system spin transport and operator weight dissipation in spin chains" — [arXiv:2210.06494](https://arxiv.org/pdf/2210.06494) (title only)
- TDVP hydrodynamics:
  - Leviatan et al. (arXiv:1702.08894) report "surprisingly fast convergence of the heat diffusion constant with respect to bond dimension" in a non-integrable chain.
  - Kloss, Bar Lev, Reichman (PRB 97, 024307, 2018; arXiv:1710.09378) show that fixed-χ TDVP doesn't guarantee accuracy. It fails strikingly on integrable models and fails to reproduce the Anderson-localization plateau at every χ studied. The problem appears milder in disordered non-integrable systems.
  - A later review summarizes that TDVP "can converge very quickly in bond dimension, but to dynamics with an unphysical diffusion coefficient".
  - [Kloss et al. accepted ms](https://link.aps.org/accepted/10.1103/PhysRevB.97.024307); [arXiv:1710.09378](https://ar5iv.labs.arxiv.org/html/1710.09378); [review: Emergent hydrodynamics…, arXiv:1902.01859 per search](https://arxiv.org/pdf/1902.01859)
- DMT/OST vs TEBD (2310.06886): plain TEBD converges only to t ≲ 20/J but still gives D within 1%. DMT and OST give consistent correlators to t = 60/J with D agreeing within 1% — [arXiv:2310.06886](https://arxiv.org/pdf/2310.06886)
- rTEBD (arXiv:2412.08730, preprint):
  - It reweights the SVD in the Pauli basis, deprioritizing high-weight Pauli strings by γ^(-n) during truncation. The bias is "not equivalent to minimizing the physical Frobenius norm".
  - The underlying evolution stays unitary up to Trotter and SVD errors.
  - It is built for density-matrix or operator evolution and is positioned against DMT's failure to keep longer-range two-body correlations.
  - [arXiv:2412.08730](https://arxiv.org/pdf/2412.08730) (snippet)

### Inferences
- All the successful transport methods act on *operator space* (an MPDO or a Heisenberg MPO), where local quantities are linear in the tensor. Reweighting (rTEBD), dissipating (DAOE) and reserving (DMT) are all forms of "weight short operators more in the truncation norm". For pure-state TEBD, our tilt is the analogous move. The difference is that our metric is applied after the fact as a Gauss-Newton correction, not built into the SVD norm.
- **Prior-art risk for "local-observable-weighted truncation in operator space" is high** (DMT, rTEBD, DAOE, OST). If we port our method to operator evolution, the right framing is: (i) a 2-sided-window RDM target rather than a Pauli-weight reweighting, and (ii) fixed χ with a subspace tilt rather than reserved directions (which DMT effectively adds on top of the SVD-kept rank).
- For D extraction specifically, plain TEBD is already within 1% (2310.06886). So the late-time hydrodynamic regime may not be where our growth-phase advantage shows up. Our benefit should be sought in *transient* quantities: early-time current autocorrelation, the ballistic-to-diffusive crossover, and the onset regime that DAOE crossover studies (2408.08249) target.

### Gaps
- No study found that quantifies D error as a function of truncation-error type (local vs global), apart from the method comparisons above.
- The Leviatan et al. D values and the DAOE ℓ* scans were not read in detail.
- No head-to-head DAOE vs TDVP was found.

---

## 4. Lindblad, MPDO, purification and noisy circuits: trace, positivity, local observables

### Takeaway
Open-system TN practice uses three representations. MPDOs (vectorized ρ) are cheap but truncation breaks positivity. Locally purified forms (LPTN/LPDO) are positive by construction. Quantum trajectories are pure MPSs. For noisy circuits, MPDO/LPDO with "optimal truncation of inner and bond dimension" beats pure-state MPS. Truncation errors in noisy dynamics *contract* exponentially, so global errors don't build up the way they do in unitary dynamics. Positivity-preserving and CPTP low-rank schemes exist. No scheme was found that preserves trace, positivity and local observables together for truncated MPDOs.

### Cited Findings
- MPDO positivity: "the MPDO formulation does not preserve the positivity of the density matrix, but a simple extension based on a local purification of the density matrix fixes this". Checking positivity is "generally an NP-hard problem, in system size". In the OSMPS comparison, the locally purified tensor network uses bond dimension more efficiently than MPDOs, and correlations are "among the observables most affected by a limited bond dimension" — [OSMPS, Quantum Sci. Technol.](https://iopscience.iop.org/article/10.1088/2058-9565/aae724/ampdf); [arXiv:1802.10052](https://arxiv.org/pdf/1802.10052); [ResearchGate OSMPS](https://www.researchgate.net/publication/324793417_OSMPS_Many-body_entangled_open_quantum_systems) (snippets)
- Locally purified tensor networks (Werner, Jaschke, Silvi, Kliesch, Calarco, Eisert, Montangero, PRL 116, 237201 (2016)): positivity by construction — [arXiv:1412.5746](https://arxiv.org/abs/1412.5746) [prior knowledge, not re-verified this session]; [Silvi LPTN talk](https://www.uni-ulm.de/fileadmin/website_uni_ulm/nawi.inst.125/heraeus624/presentations/Silvi_20160919_LPTN_Talk.pdf)
- Early MPDO/Lindblad TN work: Zwolak & Vidal [cond-mat/0406440](https://arxiv.org/abs/cond-mat/0406440); Verstraete, García-Ripoll, Cirac [cond-mat/0406426](https://arxiv.org/abs/cond-mat/0406426) [prior knowledge, not re-verified this session]. Review: Weimer, Kshetrimayum, Orús, "Simulation methods for open quantum many-body systems" [arXiv:1907.07079](https://arxiv.org/abs/1907.07079) [prior knowledge, not re-verified this session]
- Noh, Jiang, Fefferman, "Efficient classical simulation of noisy random quantum circuits in one dimension", Quantum 4, 318 (2020):
  - MPO simulation of 1D noisy random circuits, with "MPO entanglement entropy" as the cost metric.
  - [arXiv:2003.13163](https://arxiv.org/abs/2003.13163v1); [Quantum](https://quantum-journal.org/papers/q-2020-09-11-318/)
  - [prior knowledge, not re-verified this session]: the MPO entanglement rises and then falls with circuit depth once noise wins.
- Cheng, Cao, Zhang, Liu, Hou, Xu, Zeng, "Simulating noisy quantum circuits with matrix product density operators", PRResearch 3, 023005 (2021):
  - The pure-state MPS method "fails to approximate the noisy output quantum states for any of the noise models considered", while MPDO works well.
  - It proposes "a more effective tensor updates scheme with optimal truncations for both the inner and the bond dimensions" after each layer.
  - [arXiv:2004.02388](https://arxiv.org/pdf/2004.02388); [PRResearch](https://link.aps.org/doi/10.1103/PhysRevResearch.3.023005)
- LPDO for noisy circuits (Chin. Phys. Lett. 41, 120302 (2024)): "truncation in MPOs breaks the positivity condition", which LPDO fixes — [CPL](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/41/12/120302); [arXiv:2312.02854](https://fugumt.com/fugumt/paper_check/2312.02854v3_enmode)
- "Noise-induced contraction of MPO truncation errors in noisy random circuits and Lindbladian dynamics" (preprint, March 2026):
  - Truncation errors contract exponentially in system size N and time t, so the usual error bounds are loose for open systems.
  - It notes that controlling the L1 error with an RG-style truncation "requires computing the entanglement of purification, which is generally intractable".
  - [arXiv:2603.20400](https://arxiv.org/pdf/2603.20400); [EmergentMind](https://www.emergentmind.com/papers/2603.20400)
- Lee, Ghosh, Oh, Noh, Fefferman, Jiang, "Classical simulation of noisy random circuits from exponential decay of correlation" — [arXiv:2510.06328](https://arxiv.org/pdf/2510.06328) (title and authors only)
- CPTP low-rank schemes with tensor-train compression for the Lindblad equation (preprint): "a family of low-rank, completely positive and trace preserving schemes" — [arXiv:2605.01494](https://arxiv.org/pdf/2605.01494) (snippet)
- Quantum trajectories at scale: "Large-scale stochastic simulation of open quantum systems" — [arXiv:2501.17913](https://arxiv.org/html/2501.17913v1) (title only). Dissipative NV-center dynamics with TNs — [arXiv:2406.08108](https://arxiv.org/pdf/2406.08108) (title only)
- Operator entanglement under dephasing: Wellnitz, Preisser, Alba, Dubail, Schachenmayer, "Rise and fall, and slow rise again, of operator entanglement under dephasing", PRL 129, 170401 (2022) — [arXiv:2201.05099](https://arxiv.org/abs/2201.05099) [prior knowledge, not re-verified this session]
- Pure-state limits of quantum-computer simulation (fidelity per gate): Zhou, Stoudenmire, Waintal, PRX 10, 041038 (2020) — [arXiv:2002.07730](https://arxiv.org/abs/2002.07730) [prior knowledge, not re-verified this session]

### Inferences
- **Where local accuracy matters more than global fidelity:**
  - Noisy-circuit expectation values (error mitigation, VQE energies, few-body Pauli observables).
  - Lindblad transport, such as the open XX chain with dephasing in LITE.
  - Steady-state local currents and profiles.
  
  In all of these the global state fidelity is exponentially small anyway, and only few-body expectation values are used.
- **Insertion points and expected fit (our mechanism needs a large cross-cut 2-site error and an entanglement-growth phase):**
  1. *Vectorized MPDO with Lindblad TEBD.*
     - Untruncated target: the two-site superoperator-updated block of |ρ⟩⟩.
     - Local RDMs are linear in the tensor, so DMT-style reserved directions already preserve them *exactly*, and the Gauss-Newton tilt adds nothing on that constraint.
     - Prior-art risk: **high** (DMT is literally this).
     - A possible niche: a fixed-χ tilt that also penalizes negative eigenvalues of the 3-site RDM, i.e. local positivity. No source was found for "local-positivity-preserving MPDO truncation".
  2. *Locally purified LPTN/LPDO and purification MPS.*
     - The tensor network is pure in the doubled space, so physical RDMs (ancillas traced out) are *quadratic* in the tensor. That is exactly the setting our Gauss-Newton tilt handles.
     - Target: the physical-only 1-, 2- and 3-site RDMs of the untruncated two-site (site+ancilla) block.
     - The ancilla gauge freedom (Section 5) gives extra room: the tilt can use directions that leave physical RDMs unchanged.
     - Prior-art risk: **low to moderate**. No local-RDM-targeted truncation for LPTN/purification was found. Cheng et al.'s "optimal truncation of inner and bond dimension" is fidelity/Frobenius-optimal as far as the abstract shows.
  3. *Quantum-trajectory MPS (MCWF unraveling of Lindblad).*
     - Each trajectory is a pure MPS evolved with TEBD, so our method applies unchanged.
     - Target: the trajectory's local RDMs. The figure of merit is the trajectory average of local observables.
     - Jumps and dephasing reduce entanglement per trajectory, so the growth phase is shorter. The benefit is probably biggest at weak dissipation.
     - Prior-art risk: **low** (nothing found).
  4. *Noisy-circuit MPDO/LPDO.*
     - Error contraction (2603.20400) means a per-step local improvement is not compounded, but it is not amplified either. The relevant error is the per-step local bias at the observable's support.
     - Our gate (benefit/cost) fits naturally: in the early "rise" of operator entanglement (Noh et al.; Wellnitz et al.) the cross-cut error is large, and once noise dominates SVD is already enough.
     - Prior-art risk: **low to moderate**.
- Trace is preserved in our pure-state tilt automatically (the norm is set by the Q orthonormalization). Positivity is automatic for purification and trajectory representations. So taking our method to those representations avoids the positivity issue that DMT had to "ameliorate".

### Gaps
- The Cheng et al. truncation criterion, the Noh et al. truncation details and the methods of the 2603.20400 preprint could not be read in full text.
- No paper was found that benchmarks local-observable error, as opposed to fidelity, of MPDO vs LPTN vs trajectories at equal cost.
- No source was found on DMT applied directly to Lindblad dynamics.

---

## 5. Finite temperature: purification, METTS, ancilla disentanglers, and truncation aimed at local thermal expectation values

### Takeaway
The finite-temperature literature uses the *ancilla gauge freedom* to reduce entanglement: disentanglers, and minimal-entanglement purification (Hauschild et al.). It does not change the truncation criterion. One paper does argue explicitly that the usual truncation error "is measured against the exact state" and can be large while local quantities are accurate. No truncation designed to preserve local *thermal* expectation values was found. That is an open niche, and our method fits it directly because purifications are pure states.

### Cited Findings
- Hauschild, Leviatan, Bardarson, Altman, Zaletel, Pollmann, "Finding purifications with minimal entanglement", PRB 97, 235163 (2018):
  - A purification is invariant under any ancilla-only unitary U_anc, and this freedom is used to lower MPS entanglement. It cites Karrasch et al. (2012) and Barthel (2013) for the idea.
  - It notes dynamics methods "aim to simulate the correct macrostate rather than the exact microstate", so the usual DMRG truncation error "can therefore be large, since it is measured against the exact state".
  - Benchmarks: entropy grows roughly linearly with no disentangling, with backward time evolution, and with the optimized disentangler, but with very different prefactors. The optimized method confines entropy to a causal light cone.
  - [arXiv:1711.01288](https://arxiv.org/pdf/1711.01288); [ar5iv](https://ar5iv.labs.arxiv.org/html/1711.01288). Author list: [prior knowledge, not re-verified this session]
- The TeNPy purification-MPS implementation includes disentanglers, following Hauschild et al. — [TeNPy docs](https://tenpy.readthedocs.io/en/v1.0.6/reference/tenpy.networks.purification_mps.html)
- METTS expands the thermal trace in classical product states, each with zero entanglement. Sampling adds a statistical error scaling as β^-1/√(M−1) — [Binder & Barthel, arXiv:1411.3033](https://arxiv.org/abs/1411.3033v1); [PRB 92, 125119](https://link.aps.org/doi/10.1103/PhysRevB.92.125119)
- Binder & Barthel find that at fixed cost, "for almost all considered cases, purifications yield more accurate results than METTS, often by orders of magnitude". They highlight the interplay between statistical and truncation errors in METTS — [arXiv:1411.3033](https://arxiv.org/abs/1411.3033v1)
- Symmetric METTS — [arXiv:1506.03336](https://arxiv.org/pdf/1506.03336) (title only). METTS was originally proposed by White (PRL 102, 190601 (2009)), with the algorithm in Stoudenmire & White [arXiv:1002.1305](https://arxiv.org/abs/1002.1305) [prior knowledge, not re-verified this session]
- A paper combining METTS and purification for Matsubara Green's functions — [arXiv:2107.13941](https://arxiv.org/pdf/2107.13941v1) (snippet)
- The TDVP for mixed MPS in the thermodynamic limit exists as a separate approach — [arXiv:2007.15035](https://arxiv.org/pdf/2007.15035) (title only)

### Inferences
- **Imaginary-time purification and METTS (state preparation):** entanglement is area-law-like and the per-cut cross-cut error is small. Our mechanism predicts little benefit, and the measured 3–16× cost is unlikely to pay off. The gate would mostly fall back to SVD.
- **Real-time dynamics of finite-T purifications, e.g. T>0 correlation functions and spectral functions:**
  - Entanglement grows linearly (Hauschild et al.), and only physical local correlators are needed. This is the growth regime our method needs.
  - Target: physical-leg RDMs of the untruncated two-site site+ancilla block.
  - It combines naturally with a disentangler. The disentangler lowers the discarded weight, and our tilt fixes the local error of what is still discarded.
  - Prior-art risk: **low** (nothing found).
- **METTS real-time or typicality ensembles:** each sample is a pure MPS in a quench-like growth phase, so the method applies per sample. Lower local bias per sample reduces the systematic error that does not average out, while statistical error is unaffected.

### Gaps
- The original Karrasch–Bardarson–Moore disentangler paper and Barthel 2013 were not retrieved (search returned only citations to them).
- No paper was found that studies the *local* thermal-expectation error of purification truncation as a function of the truncation criterion.

---

## 6. Candidate-area map: (a) known, (b) gap, (c) insertion and target, (d) expected benefit, (e) prior-art risk

### Takeaway
Our method fits best where (i) the representation is a *pure* tensor network (purification/LPTN, trajectories, METTS samples), so local RDMs are quadratic and a Gauss-Newton tilt is needed; (ii) entanglement is growing; and (iii) only few-body observables matter. Operator-space settings (MPDO-Schrödinger, DAOE/Heisenberg MPO) are where DMT, rTEBD, DAOE and OST already work. There, exact linear preservation makes our approximate step unnecessary, and the prior-art risk is high.

### Cited Findings
- DMT preserves ≤3-site expectation values exactly in MPDOs — [arXiv:1707.01506](https://arxiv.org/pdf/1707.01506). rTEBD reweights the Pauli-basis SVD — [arXiv:2412.08730](https://arxiv.org/pdf/2412.08730). DAOE damps long operators — [arXiv:2004.05177](https://www.arxiv.org/pdf/2004.05177). OST truncates by operator size — [arXiv:2310.06886](https://arxiv.org/pdf/2310.06886). LITE preserves all RDMs below ℓ_min — [arXiv:2310.06036](https://arxiv.org/abs/2310.06036)
- Pure-to-mixed, local-RDM-preserving methods: [arXiv:1810.01231](https://arxiv.org/pdf/1810.01231); [arXiv:2308.04291](https://arxiv.org/pdf/2308.04291)
- The suggestion of local-density-matrix-optimizing truncation for MPS appears in the review — [arXiv:1901.05824](https://arxiv.org/pdf/1901.05824)
- Purification gauge freedom and the "macrostate not microstate" argument — [arXiv:1711.01288](https://arxiv.org/pdf/1711.01288)
- Error contraction in noisy MPO dynamics — [arXiv:2603.20400](https://arxiv.org/pdf/2603.20400)

### Inferences
| Area | (a) Known | (b) Gap | (c) Insertion, and the "untruncated tensor" target | (d) Expected benefit under our mechanism | (e) Prior-art risk |
|---|---|---|---|---|---|
| MPDO (vectorized ρ) TEBD, closed system, hydrodynamics | DMT exact ≤3-site; D within 1% of OST (2310.06886) | DMT drops long-range 2-body correlations; errors leak down the hierarchy; positivity only "ameliorated" | Two-site block of \|ρ⟩⟩; local RDMs linear, so exact linear constraint | Small: linear exactness is already there; our tilt is redundant | **High** (DMT) |
| Heisenberg MPO (DAOE/OST/rTEBD) | DAOE D error exp. small in ℓ*, but ℓ*-dependent in weak-coupling fermions | Transient/crossover regime; cutoff dependence | Two-site MPO block; target ⟨O(t) A_local⟩, linear | Small to moderate; the target is again linear | **High** (rTEBD, DAOE, OST, DMT-traceless appendix) |
| Purification MPS, finite-T real-time dynamics | Disentanglers reduce entanglement; truncation error measured vs exact microstate | No local-RDM-targeted truncation found | Two-site (site+ancilla) block; target = *physical-only* 1–3-site RDMs (trace out ancillas); quadratic, so Gauss-Newton tilt | **Moderate to high**: linear entanglement growth, only local correlators needed; stacks with a disentangler | **Low to moderate** |
| LPTN / LPDO (Lindblad, noisy circuits) | Positive by construction; uses χ better than MPDO | Truncation still Frobenius/fidelity-based as far as found | Two-site block with Kraus/inner leg; target = physical RDMs; quadratic | Moderate during the operator-entanglement "rise"; the gate turns it off once noise dominates | **Low to moderate** |
| Quantum-trajectory MPS (Lindblad unraveling) | Large-scale stochastic simulation exists (2501.17913) | No local-RDM truncation per trajectory found | Unchanged from our pure-state TEBD | Moderate at weak dissipation; reduces bias that sampling cannot remove | **Low** |
| METTS (imag. time) / purification cooling | Purification usually beats METTS at equal cost | — | Per-sample pure MPS | Low: little entanglement growth, small cross-cut error | Low, but little payoff |
| Noisy-circuit MPDO (Cheng et al.; Noh et al.) | MPDO ≫ MPS; optimal inner/bond truncation; errors contract | Local-observable-targeted truncation not found | MPDO: linear (DMT-like). LPDO: quadratic, our tilt | Moderate for few-body observables at intermediate depth | Low to moderate |
| LITE / information lattice | Exact RDMs below ℓ_min; linear-in-L cost | Not an SVD method; 2D unclear | Not applicable; serves as a benchmark | — | — |

- **Framing suggestion:** "DMT shows that, for density operators, preserving local RDMs at each cut is the right truncation criterion, and it can be imposed exactly because the constraint is linear. We supply the pure-state counterpart, where the constraint is quadratic and is imposed to first order by one tangent-space Gauss-Newton tilt at fixed χ. This makes the criterion available to purification, LPTN and trajectory simulations of mixed and open systems."
- **Caveat for the report:** our measured far-observable collateral (~√discarded weight) resembles DMT's documented leak of error into longer-range correlations (2412.08730; 1902.01859). Any claim of an advantage over DMT in operator space would need that far-observable drift measured explicitly.

### Gaps
- Not confirmed: whether the Paeckel et al. pointers (refs 73/74 in the snippet) are an existing pure-state local-RDM-preserving MPS truncation. This is **the single most important prior-art check left open**. Retrieve the 1901.05824 reference list when arXiv is reachable.
- Not confirmed: whether 2412.08730 (rTEBD) also has a pure-state variant.
- All numbers were taken from abstracts and snippets. None of the quantitative claims (1%, t = 60/J, D ∝ λ^-4, D ∝ 1/ρ, L = 100) were checked against figures or tables.
