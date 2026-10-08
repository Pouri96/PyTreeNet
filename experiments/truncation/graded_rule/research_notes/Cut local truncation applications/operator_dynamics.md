# Cut-local, local-observable-preserving truncation for OPERATOR dynamics (Heisenberg picture)

**How this was checked (read first).** In this session arxiv.org, export.arxiv.org, Semantic Scholar and alphaxiv could not be reached by either WebFetch or curl (proxy refused the connection / DNS failed). So **nothing below was read in full text**. Each item carries one of two tags:
- **[S]**: confirmed at abstract or search-snippet level in this session (title and arXiv ID matched; the claims come from the abstract or the snippet).
- **[PK]**: from prior knowledge, not re-checked this session. The arXiv ID and claim are given so the writer can verify them. Treat any number tagged [PK] as unconfirmed.

Notation. "Our method" means the cut-local Gauss-Newton tilt of the kept rank-chi SVD subspace. It matches the 1-, 2- and 3-site RDMs in a window of ±2 sites around the cut to those of the untruncated two-site tensor. Mechanism: it fixes SVD's first-order error in correlators across the cut, but adds a first-order collateral change of size about sqrt(discarded weight) in observables far from the cut.

---

## Q1. MPO Heisenberg-picture evolution: truncation criteria, known failures, and methods that beat SVD for autocorrelations and transport

### Takeaway
The "keep local information, not Frobenius norm" idea already has a well-developed operator/MPDO literature: DMT, DAOE/FDAOE, rTEBD, operator-size truncation, LITE, and the operator-backflow theory. The closest prior art to our tilt is **White–Zaletel–Mong–Refael DMT** (arXiv:1707.01506). At every cut, DMT rotates the Schmidt spaces so that all expectation values on regions up to 3 sites in diameter are preserved exactly; for a non-density-matrix operator these become partial traces. So a plain "cut-local, RDM-preserving MPO truncation" is **not new**. The open space is narrower:
- targets that DMT cannot hold exactly at fixed chi (longer-range two-body strings across the cut);
- quadratic targets on the vectorized operator (OTOC and operator-weight profiles, see Q3);
- reference-state-conditioned targets (Q4).

### Cited Findings
**Operator entanglement and failure modes of Frobenius-SVD**
- Operator-space entanglement entropy (OSEE) was introduced for the transverse Ising chain. For initial operators of finite Majorana index it saturates; for infinite index it grows logarithmically, with prefactor 1/6 (critical) or 1/3 (non-critical). This is the basis of "integrable means Heisenberg-picture MPO is cheap". — [Prosen & Pižorn, arXiv:0706.2480](https://arxiv.org/abs/0706.2480) [S]
- XY chains: the OSEE of local operators grows at most logarithmically. The prefactor changes from 1/3 to 2/3 at the NESS transition, and weak disorder makes it saturate. — [Pižorn & Prosen, arXiv:0903.2432](https://arxiv.org/abs/0903.2432) [S]
- Original argument that Heisenberg-picture efficiency is linked to integrability: OSEE grows logarithmically for integrable chains and linearly for chaotic ones. — [Prosen & Žnidarič, quant-ph/0608057, PRE 75, 015202 (2007)](https://arxiv.org/abs/quant-ph/0608057) [PK]
- How well an operator can be approximated by an MPO **in Hilbert–Schmidt norm** depends on an operator area law, and the OSEE can be computed in CFT. Talks describe an "entanglement barrier" in which the operator entanglement of a reduced density matrix rises and then falls. — [Dubail, arXiv:1612.08630](https://arxiv.org/abs/1612.08630) [S]; [Dubail AMP21 slides](https://www.phys.ens.fr/~jacobsen/AMP21_Dubail.pdf) [S]
- Coarse-grained ("membrane") theory of operator and state entanglement in a chaotic Ising chain. In integrable systems a spreading operator can entangle sublinearly or not at all. — [arXiv:1803.00089 (Jonay, Huse, Nahum, by PK; the snippet did not show authors)](https://arxiv.org/abs/1803.00089) [S title, PK authors]
- Under dephasing, operator entanglement can rise again (logarithmically at late times) after the first rise and fall. — [APS DAMOP 2022 abstract](https://archive.aps.org/damop/2022/q06/5) [S]
- Early Heisenberg-picture MPO/DMRG work: [Hartmann, Prior, Clark, Plenio, arXiv:0808.0666](https://arxiv.org/abs/0808.0666) [PK]; [Muth, Unanyan, Fleischhauer, arXiv:1009.4646](https://arxiv.org/abs/1009.4646) [PK]. Purification and backward-time gauge tricks that reduce operator entanglement: [Karrasch, Bardarson, Moore, arXiv:1205.3756](https://arxiv.org/abs/1205.3756) [PK]; [Barthel, arXiv:1301.2246](https://arxiv.org/abs/1301.2246) [PK]; [Hauschild et al., arXiv:1711.01288](https://arxiv.org/abs/1711.01288) [S title]; [Kennes & Karrasch, arXiv:1511.02205](https://arxiv.org/abs/1511.02205) [PK].

**DMT: direct prior art for "cut-local, local-information-preserving"**
- DMT, from "Quantum dynamics of thermalizing systems" (PRB 97, 035127, 2018), works on matrix-product density operators. It "preserves the trace of the density matrix and the expectation values of conserved quantities by rotating the Schmidt spaces at the bond at which it truncates". It exactly preserves the expectation values of all operators on all regions of up to three sites in diameter, including the energy density. — [White, Zaletel, Mong, Refael, arXiv:1707.01506](https://arxiv.org/abs/1707.01506) [S]
- DMT was extended to preserve ℓ-site operators for any ℓ. — [Ye, Machado, White, Mong, Yao, "Emergent hydrodynamics in non-equilibrium quantum systems", arXiv:1902.01859](https://arxiv.org/abs/1902.01859) [S title/claim; authors PK]
- Limitations of DMT (as described by the rTEBD paper):
  - it does not keep longer-range two-body correlations such as ⟨σ_i^μ σ_{i+3}^ν⟩;
  - for an operator that is not a density matrix, the preserved quantities are partial traces of that operator, not RDMs;
  - it keeps local conserved quantities "up to machine precision".
  — [rTEBD, arXiv:2412.08730](https://arxiv.org/abs/2412.08730) [S]
- Related idea: trading entanglement for mixture to simulate local observables. — [arXiv:1810.01231 (Surace, Piani, Tagliacozzo, by PK)](https://arxiv.org/abs/1810.01231) [S title]

**Pauli-weight-biased SVD (rTEBD)**
- rTEBD rewrites the MPDO in a reweighted Pauli basis, so that correlators of n Paulis are favoured by a factor γ^{-n} during the SVD truncation. The time evolution stays unitary; only the truncation step is biased.
- Reported results: "significantly more accurate" than MPDO-TEBD, competitive with or better than MPS-TEBD, and conserved quantities kept to high precision.
- Free-fermion benchmark: L=128, dt=0.08, γ=1.5 (better than γ≥2).
- rTEBD is presented as fixing DMT's loss of few-body correlations at longer range.
- Detailed comparison with DAOE and LITE is left for future work.
— [arXiv:2412.08730](https://arxiv.org/abs/2412.08730) [S; authors not verified]

**Dissipation and operator-size truncation**
- DAOE adds an artificial dissipator that damps Pauli strings with weight above ℓ*. The dissipator is an exact MPO with bond dimension ℓ*+1, and the method is designed for hydrodynamic transport. — [Rakovszky, von Keyserlingk, Pollmann, arXiv:2004.05177](https://arxiv.org/abs/2004.05177) [S]
- DAOE applied to hydrodynamic crossovers: dissipation lowers the bond dimension needed and reaches later times. — [arXiv:2408.08249, PRB DOI 10.1103/gz9n-v8ty](https://arxiv.org/abs/2408.08249) [S]
- Fermionic DAOE (FDAOE) for energy diffusion in weakly interacting chains. FDAOE and DMT both reach the relevant regime at χ ≈ 64–128. — [arXiv:2311.17148](https://arxiv.org/abs/2311.17148) [S title; χ claim from snippet]
- Operator backflow:
  - Truncating or dissipating n-point content above ℓ* causes "backflow" errors in transport coefficients, and these are exponentially suppressed in ℓ*.
  - Conjecture: transport coefficients to precision ε need memory exp[O(log² ε)].
  - Tested on random circuits and ergodic spin chains.
  — [von Keyserlingk, Pollmann, Rakovszky, arXiv:2111.09904, PRB 105, 245101](https://arxiv.org/abs/2111.09904) [S]
- Benchmark of TEBD, TEBD+DMT, recursion method + UOGH (R-UOG) and operator-size-truncated (OST) dynamics for infinite-temperature hydrodynamics:
  - DMT and OST agree to t=60/J, with diffusion coefficients within 1%.
  - Plain TEBD converges only for t≲20 but still gives D within 1%.
  - R-UOG fails to converge beyond short times.
  - DAOE (cited, not rerun) gives D ≈ 1.4±0.01.
  — [Yi-Thomas, Ware, Sau, White, arXiv:2310.06886, PRB 110, 134308](https://arxiv.org/abs/2310.06886) [S]
- In TEBD, the high-Pauli-weight subspaces collect most of the truncation error but contribute less and less to the observable. This supports weight-biased truncation. — [arXiv:2409.13603](https://arxiv.org/abs/2409.13603) [S, snippet]
- LITE:
  - built on the "information lattice";
  - evolves subsystem density matrices up to a maximum scale and discards long-range information at a second scale in a controlled way, keeping the information currents;
  - tested on mixed-field Ising energy transport and dephased XX;
  - cost is exponential in the subsystem scale.
  — [Artiaco et al., arXiv:2310.06036, PRX Quantum 5, 020352](https://arxiv.org/abs/2310.06036) [S]; predecessor: [Klein Kvorning et al., arXiv:2105.11206, SciPost Phys. 13, 080](https://arxiv.org/abs/2105.11206) [S]
- Local conserved charges of the Heisenberg chain have exact MPOs whose bond dimension grows linearly with locality. These are available as explicit "conserved density" targets. — [Yamada & Fukai, arXiv:2306.03431](https://arxiv.org/abs/2306.03431) [S]; two-parameter MPO families of integrals of motion: [arXiv:2602.19741](https://arxiv.org/abs/2602.19741) [S]
- MPO TEBD with heavy truncation appears to **stop** operator spreading (unphysical), whereas TDVP (unitary projection) keeps ballistic spreading qualitatively right. TDVP on operators is therefore the other main "better than SVD" option. — [arXiv:1901.05793 (Hémery, Pollmann, Luitz, by PK)](https://arxiv.org/abs/1901.05793) [S]

### Inferences
- **(a) Known.** DMT already does "cut-local + preserve ≤3-site local content + conserved densities" for MPDOs and operators, and does it exactly. rTEBD does "bias SVD toward low Pauli weight" globally through γ^{-n}. DAOE, OST and LITE change the dynamics or the representation instead.
- **(b) Gap.**
  - No method found that keeps the SVD rank fixed at chi and corrects the kept subspace by a tangent-space step toward low-weight targets that DMT cannot hold exactly: two-body strings spanning 4–5 sites across the cut, short Heisenberg-evolved strings, or quadratic functionals (Q3).
  - No head-to-head DMT vs rTEBD vs DAOE vs LITE comparison exists; rTEBD says it is future work.
- **(c) Insertion.**
  - Target: the untruncated two-site MPO tensor's **partial traces / Pauli coefficients of all strings supported in the ±2-site window** (equivalently its window-reduced operator Tr_out[O]/d^{n_out}).
  - Tilt Q = qr(U_k + U_⊥C) so the truncated MPO reproduces these coefficients.
  - This is "DMT done as a least-squares tilt instead of an exact block construction". The added value is that it can include strings DMT drops, such as σ_i σ_{i+3} across the cut.
- **(d) Expected benefit.**
  - Operator backflow (2111.09904) says the far, high-weight content our tilt disturbs (the collateral of about sqrt(discarded weight)) feeds back into transport only with exponential suppression. So operator transport is a regime where our collateral costs **less** than for pure states.
  - But DMT already captures the leading low-weight content exactly. Any gain is incremental: probably the σ_iσ_{i+3}-type correlators and the current-current correlator at intermediate times.
- **(e) Prior-art risk.** **High** for any "local-RDM-preserving MPO truncation" framing, because of DMT (1707.01506 / 1902.01859). Moderate for "Pauli-weight-aware SVD", because of rTEBD (2412.08730). A paper must position itself as a generalization of DMT (approximate, larger windows, arbitrary targets, rank fixed at chi) and compare directly against DMT and rTEBD.

### Gaps
- Could not read DMT's construction in full text to check whether it keeps the bond dimension at chi or uses chi plus the constrained block. This matters for an "equal-chi" comparison.
- rTEBD authors and transport numbers could not be checked (no full text).
- Found no published TDVP-on-MPO autocorrelation benchmark with errors against DMT at equal chi.

---

## Q2. Pauli propagation: truncation rules, errors, hybrids, and whether observable-aware truncation exists

### Takeaway
Pauli propagation truncates per string: by weight, by coefficient magnitude, by top-K, or (xSPD) by X/Y count. It has no bond or cut, so our cut-local tilt has no direct place to go. State- and observable-aware truncation has only just appeared (Sep 2026):
- state-adapted low-body projection (arXiv:2609.12840);
- jointly optimized state/observable compression (arXiv:2609.18246).

Our trick fits only in hybrid MPO/Pauli or DAOE-style MPS-in-Pauli-basis pipelines.

### Cited Findings
- Framework paper (PRX Quantum). The PauliPropagation.jl package exposes the truncations `min_abs_coeff` (coefficient) and `max_weight` (weight), plus custom truncation functions. — [Rudolph et al., arXiv:2505.21606, DOI 10.1103/6vd7-l9bn](https://arxiv.org/abs/2505.21606) [S]
- Error control is thin:
  - Rigorous bounds exist only for restricted settings, such as weight truncation on random or locally scrambling circuits with anticoncentration.
  - Coefficient thresholding on structured Hamiltonians "should be regarded as heuristic".
  - Monte Carlo estimation of the average Pauli-propagation error is proposed.
  — [Angrisani et al., arXiv:2409.01706](https://arxiv.org/abs/2409.01706) [S, snippet]; [Practical Guide to Pauli Path Simulators (ResearchGate)](https://www.researchgate.net/publication/393724127_A_Practical_Guide_to_using_Pauli_Path_Simulators_for_Utility-Scale_Quantum_Experiments) [S]
- Top-K truncation analysed through operator complexity: √Δ(K) ≤ ‖O−Ô‖ ≤ √(2Δ(K)) in the Pauli 2-norm, with Δ the discarded coefficient weight. The cost is set by operator complexity, not state entanglement, and is linked to Rényi-entropy bounds on MPS truncation. — [arXiv:2510.22311, "Characterizing Pauli Propagation via Operator Complexity"](https://arxiv.org/abs/2510.22311) [S]. Note: the brief calls this "Shao et al., stabilizer Rényi"; I could not confirm the authors or the stabilizer-Rényi content from snippets.
- Noisy circuits: noise damps non-local correlations exponentially relative to local ones, so tracking only local information gives a polynomial-time algorithm for expectation values. The error is small *on average over input-state ensembles*, the cost is exponential in 1/noise, and the method is impractical at current noise rates. — [Schuster, Yin, Gao, Yao, arXiv:2407.12768, PRX 15, 041018](https://arxiv.org/abs/2407.12768) [S]
- Sparse Pauli dynamics (coefficient-threshold Heisenberg evolution) and TN methods reproduce the IBM 127-qubit kicked-Ising utility experiment, converged beyond the experimental accuracy. — [Begušić & Chan, arXiv:2308.05077, Sci. Adv. 10.1126/sciadv.adk4321](https://arxiv.org/abs/2308.05077) [S; authors PK]
- SPD for real-time operator evolution in 2D/3D. — [arXiv:2409.03097](https://arxiv.org/abs/2409.03097) [S title]
- xSPD / "Or-represented quantum algebra" drops strings with many X/Y. This is a **state-aware** rule, because X/Y strings have zero expectation in computational-basis states. It tracks up to 10^12 strings on the 127-qubit benchmark. — [arXiv:2506.13241](https://arxiv.org/abs/2506.13241) [S]
- **State-adapted generalized mean-field projections:**
  - Projects the observable onto the ≤m-body subspace using the covariance geometry of the initial (product) state, so high-body content is folded into "state-relevant low-body representatives" instead of being discarded.
  - Error is split into a representational part and a dynamical part.
  - Bounds are conditional and depend on nonstabilizerness and on tail control.
  - Tested on large 3D lattices (TFIM according to an automated summary).
  — [Pérez, Wójtowicz, Plenio, arXiv:2609.12840](https://arxiv.org/abs/2609.12840) [S]
- **Joint state/observable-guided compression:**
  - A sweep with coupled losses for forward state and backward observable updates, with an exact error relation for ⟨Ō⟩.
  - 30-qubit random circuits: 2–3 orders of magnitude lower error than variational state compression at equal bond dimension.
  - Needs the full back-propagated observable at each time slice, so memory goes up.
  - The automated Pith review flags thin numerics: one circuit family, no error bars.
  — [Kim, Jeong, Oh, arXiv:2609.18246](https://arxiv.org/abs/2609.18246) [S]; [Pith review](https://pith.science/paper/2609.18246) [S, automated]
- The same paper names the two standard state-agnostic baselines: MPO Schmidt truncation (which minimizes Hilbert–Schmidt error) and Pauli-weight truncation. It also describes a hybrid of forward MPS with backward low-weight Pauli propagation, contracted at an intermediate time. — [arXiv:2609.18246](https://arxiv.org/abs/2609.18246) [S, snippet]
- Other extensions:
  - thermal states: [arXiv:2602.04878](https://arxiv.org/abs/2602.04878) [S title]
  - imaginary time: [arXiv:2601.14400](https://arxiv.org/abs/2601.14400) [S title]
  - quantum-enhanced Pauli propagation: [arXiv:2603.14485](https://arxiv.org/abs/2603.14485) [S title]
  - backpropagating Pauli propagation: [arXiv:2607.15184](https://arxiv.org/abs/2607.15184) [S title]
  - noise-canceling observables: [arXiv:2606.20441](https://arxiv.org/abs/2606.20441) [S title]
  - Majorana propagation: [arXiv:2503.18939](https://arxiv.org/abs/2503.18939) [PK]
  - IBM operator backpropagation: [arXiv:2502.01897](https://arxiv.org/abs/2502.01897) [PK]
  - Pauli paths beyond the average case: [González-García, Cirac, Trivedi, arXiv:2407.16068](https://arxiv.org/abs/2407.16068) [PK]
  - noisy VQA simulation: [Fontana et al., arXiv:2306.05400](https://arxiv.org/abs/2306.05400) [PK]
- DAOE is conceptually "Pauli-weight truncation in MPS form", i.e. the bridge between Pauli propagation and TN. — [arXiv:2609.18246 summary](https://arxiv.org/abs/2609.18246) [S]; [arXiv:2004.05177](https://arxiv.org/abs/2004.05177) [S]

### Inferences
- **(a) Known.** Weight, coefficient, top-K and X/Y truncations are all *per string*, not per cut. Observable- or state-aware variants first appeared in 2026 (2609.12840, 2609.18246, and xSPD in a limited form).
- **(b) Gap.**
  - No truncation for an MPO in the Pauli basis (DAOE-style) that, at a bond, preserves the **coefficients of all low-weight strings straddling the cut**.
  - No Pauli-propagation scheme with a memory-local correction of the kind the tilt gives.
- **(c) Insertion.**
  - Only in a hybrid. Store the Heisenberg operator as an MPS over Pauli indices (vectorized MPO, as in DAOE/rTEBD).
  - At each SVD cut, use as target the untruncated tensor's coefficients on strings of weight ≤w supported in the ±2 window (these are *linear* functionals, i.e. partial traces), or their reference-state-weighted version (linear functionals ⟨⟨ρ_ref^{window}| · ⟩⟩).
  - The Gauss-Newton tilt is then a linear least-squares problem in C. It is cheaper than the RDM version, because the RDM targets are quadratic in the state.
- **(d) Expected benefit.** Moderate. Pauli-weight truncation discards whole high-weight strings and never mixes them. The tilt keeps rank chi but can push lost low-weight coefficients back. This is the same "first-order cross-cut repair" that gives us 2.5–5x in non-integrable Ising, and it should matter most where truncation error is concentrated in strings straddling the cut. Collateral damage goes to far, high-weight coefficients, which backflow theory (2111.09904) says matter little for local observables.
- **(e) Prior-art risk.**
  - Moderate-high from rTEBD (Pauli-reweighted SVD, 2412.08730) and DMT.
  - Moderate from 2609.12840 and 2609.18246, which are state-aware but global (whole-operator projection or a global sweep). Our *local* character, with peak memory chi and no back-propagated observable, is the main differentiator against 2609.18246.

### Gaps
- No source quantifies weight-truncation vs coefficient-truncation errors on the *same* Hamiltonian-dynamics benchmark against MPO-SVD at equal memory.
- The authors and contents of 2510.22311 do not match the brief's description ("Shao et al., stabilizer Rényi"). Needs checking.
- No full-text check of the 2609.12840 projection formula, i.e. whether it amounts to preserving ⟨ψ|P|ψ⟩-weighted coefficients.

---

## Q3. OTOCs, Krylov complexity and the recursion method under MPO truncation

### Takeaway
OTOCs and operator-weight profiles are **quadratic** in the operator. So if the MPO is vectorized to a normalized "state" |O⟩⟩, they are exactly **local expectation values of |O⟩⟩**: for a single-site Pauli W_x, the OTOC is a 1-site observable of W_x⊗W_x^* on |O⟩⟩. Our RDM-matching tilt therefore carries over *verbatim* to OTOC and operator-front computations. This is the cleanest fit found, and no prior art for it was found: DMT only preserves linear partial traces.

Krylov and Lanczos coefficients are global norms and orthogonality conditions, which a cut-local tilt is poorly suited to protect.

### Cited Findings
- MPO computation of OTOCs and the butterfly front in chaotic chains. — [Xu & Swingle, "Accessing scrambling using matrix product operators", Nat. Phys. 16, 199 (2020); arXiv:1802.00801](https://arxiv.org/abs/1802.00801) [S journal ref from tutorial snippet; arXiv ID PK]
- MPS approaches to operator spreading. Truncated MPO-TEBD halts spreading (unphysical); TDVP (unitary) is "a significant improvement over the non-unitary truncation used in TEBD". — [arXiv:1901.05793](https://arxiv.org/abs/1901.05793) [S]
- Random-circuit OTOC: biased diffusion, butterfly velocity v_B, front broadening ∝ t^{1/2}. — [Nahum, Vijay, Haah, PRX 8, 021014 (arXiv:1705.08975)](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021014) [S]
- Symmetry-resolved OTOCs with projected MPOs. They compare non-symmetric, charge-conserving and projected MPOs, with χ up to 6000 and 1024, and track the cumulative discarded singular values. — [arXiv:2503.20327, Quantum (2025-10-01) 1871](https://arxiv.org/abs/2503.20327) [S]
- OTOC tutorial: the operator support is confined to |x|<v_B t with exponentially small tails that a small bond dimension can capture. — [arXiv:2202.07060](https://arxiv.org/abs/2202.07060) [S]
- Universal operator growth hypothesis: b_n ~ αn/log n + γ in 1D and αn + γ in d>1. The recursion method starts from the infinite-temperature (Hilbert–Schmidt) inner product. — [Parker, Cao, Avdoshkin, Scaffidi, Altman, arXiv:1812.08657, PRX 9, 041017](https://arxiv.org/abs/1812.08657) [S claims via snippet; authors/journal PK]
- R-UOG (recursion method + UOGH) fails to converge for hydrodynamics and matches other methods only at short times. — [arXiv:2310.06886](https://arxiv.org/abs/2310.06886) [S]
- MPS-Lanczos: truncation after each MPO×MPS step destroys orthogonality among Krylov vectors. Errors build up and convergence stalls at high error. Remedies are re-orthogonalization and "optimally accurate" Lanczos; thick restart fails under truncation. — [arXiv:2504.21786](https://arxiv.org/abs/2504.21786) [S]
- In finite precision, Lanczos b_n sequences become unstable without full re-orthogonalization. — [arXiv:2505.02670](https://arxiv.org/abs/2505.02670) [S]
- No paper was found that computes operator-growth b_n directly with compressed MPO Krylov vectors and studies how the truncation affects them (two separate searches).

### Inferences
- **(a) Known.** MPO-TEBD OTOCs with SVD truncation work at moderate times but under-spread when chi is too small (1901.05793). TDVP is the established fix.
- **(b) Gap.** No truncation preserves the **operator-weight density / OTOC profile** near the cut at fixed chi.
- **(c) Insertion (strong fit).**
  - Vectorize: |O⟩⟩ with local dimension d² (Pauli basis), normalized in Frobenius norm.
  - Then OTOC(x,t) = ⟨⟨O|W_x⊗W_x^*|O⟩⟩, and the right-weight / end-point density is a 1-site observable as well.
  - Our existing code (RDMs of |O⟩⟩ in a ±2 window) gives exactly these 1–3-site "operator RDMs". The fidelity criterion becomes Frobenius fidelity, which our method already leaves unchanged.
  - Optional extra target: "short Heisenberg-evolved Pauli strings", which become short superoperator strings.
- **(d) Expected benefit.**
  - The operator front is a regime of fast operator-entanglement growth in non-integrable chains. That is the analogue of our winning regime (Ising during entanglement growth), and the cross-cut 1–2-site error should be large there. Hence a plausible 2–5x improvement in OTOC profiles near the front at equal chi.
  - Risk: our collateral, about sqrt(ε), shifts far-from-cut quantities. Behind the front the OTOC is saturated and insensitive; ahead of it the operator is trivial (bond dimension 1). So the collateral may be less visible than in state TEBD.
  - Integrable XXZ/Heisenberg OTOCs: expect failure, as for our Heisenberg-chain state results.
- **Krylov (weak fit).** Lanczos coefficients depend on global norms ‖L O_n‖ and on orthogonality to O_{n-1}, O_{n-2}. A cut-local tilt that adds a sqrt(ε) collateral change to far content would *worsen* orthogonality.
  - Possible niche: preserve local Pauli coefficients of O_n (the low-weight "head" that dominates b_n at small n).
  - Expected benefit is low and prior-art risk low; UOGH-based methods already fail for other reasons (2310.06886).
- **(e) Prior-art risk.**
  - OTOC/weight-profile-preserving truncation: **low**. Nothing found, but a targeted search of the Swingle, Luitz and Pollmann groups is advisable.
  - Krylov: low, and also low value.

### Gaps
- Could not check whether Xu–Swingle or the projected-MPO OTOC paper use any truncation criterion other than plain SVD/threshold (no full text).
- No quantitative data on how SVD truncation error is distributed between the front and the bulk of an MPO.

---

## Q4. Expectation-value-targeted compression (known state) and temporal-entanglement / influence-functional methods

### Takeaway
For ⟨ψ|O(t)|ψ⟩ with a known state, recent work compresses the operator using the state, but only globally:
- 2609.12840: state-covariance projection in Pauli propagation;
- 2609.18246: coupled forward/backward losses;
- xSPD: drop X/Y strings for computational-basis states.

In spatio-temporal TN, the field has moved from SVD of the influence-matrix MPS to truncating the **reduced transition matrix (RTM)**, which directly fixes local expectation values and is gauge-invariant. That is the closest temporal analogue of our idea.

Gap: no cut-local tilt preserving reference-state-weighted local content of a Heisenberg MPO was found. In the influence-functional case the analogue of "local RDM at the cut" is the local-in-time RTM.

### Cited Findings
- Light-cone tensor network for time evolution; transverse contraction of the light cone. — [Frías-Pérez & Bañuls, arXiv:2201.08402, PRB 106, 115117](https://arxiv.org/abs/2201.08402) [S journal; content PK]
- Temporal entanglement of the influence functional grows logarithmically in some models and linearly in others. — [Bañuls et al., PRL 2009, arXiv:0904.1926](https://arxiv.org/abs/0904.1926) [PK ID]; [Frías-Pérez & Bañuls](https://arxiv.org/abs/2201.08402) [S via Frontiers review snippet]
- Influence-matrix approach: [Lerose, Sonner, Abanin, arXiv:2009.10105, PRX 11, 021040](https://arxiv.org/abs/2009.10105) [PK]; IF as MPS and temporal entanglement: [Sonner, Lerose, Abanin, arXiv:2103.13741](https://arxiv.org/abs/2103.13741) [PK]; near integrability: [Lerose, Sonner, Abanin, arXiv:2104.07607](https://arxiv.org/abs/2104.07607) [S title]; space-time duality and the entanglement barrier: [arXiv:2201.04150](https://arxiv.org/abs/2201.04150) [PK]; impurity problems: [Thoenniss, Lerose, Abanin, arXiv:2205.04995](https://arxiv.org/abs/2205.04995) [PK]
- TN influence functionals (Chan/Reichman):
  - Anderson impurity: [Ng, Park, Millis, Chan, Reichman, arXiv:2211.10272](https://arxiv.org/abs/2211.10272) [PK]
  - continuous-time limit, where the boundary IF-MPS has finite entanglement: [Park, Ng, Reichman, Chan, arXiv:2401.12460](https://arxiv.org/abs/2401.12460) [S]
  - **2D TN-IF belief propagation**: on the heavy-hex kicked Ising model, temporal entanglement grows only logarithmically, so the cost is polynomial; an SVD-based interaction-tensor representation avoids a zero-temporal-entanglement pathology. [Park, Gray, Chan, arXiv:2504.07344, PRB 112, 174310](https://arxiv.org/abs/2504.07344) [S]. This is probably the "Park–Gray–Chan" paper in the brief.
- Temporal entropy and the complexity of local expectation values after a quench. If operator entanglement grows more slowly than state entanglement, truncating from the operator side compresses better. — [Carignano, Ramos Marimón, Tagliacozzo, arXiv:2307.11649, PRR 6, 033021](https://arxiv.org/abs/2307.11649) [S]
- RTM truncation:
  - Gauge transformations that lower temporal entanglement can push the contributions that matter for the overlap into the truncated tail of the singular values. The RTM cancels these gauges, so it is the right object to truncate for local expectation values.
  - The RTM rank is bounded by the rank of the matrix encoding the operator-space entanglement.
  — [Frontiers review 10.3389/frqst.2025.1568471](https://www.frontiersin.org/journals/quantum-science-and-technology/articles/10.3389/frqst.2025.1568471/full) [S]; [arXiv:2509.03699](https://arxiv.org/abs/2509.03699) [S]; [strong simulation via RTM, arXiv:2610.02082](https://arxiv.org/abs/2610.02082) [S title]; [low-rank RTM, Pith 2605.12665](https://pith.science/paper/2605.12665) [S, automated summary]
- Sampled TN with RTM truncation and Metropolis sampling; an automated summary reports exponential convergence in bond dimension for the Ising chain (N=18, 24). — [Carignano, Lami, De Nardis, Tagliacozzo, arXiv:2505.09714](https://arxiv.org/abs/2505.09714) [S; numbers from automated review]
- Temporal entanglement in chaotic circuits / transitions: [Foligno, Zhou, Bertini, arXiv:2302.08502](https://arxiv.org/abs/2302.08502) [PK]; [arXiv:2511.03846](https://arxiv.org/abs/2511.03846) [S title]; [mesoscopic regimes, arXiv:2605.08356](https://arxiv.org/abs/2605.08356) [S title]
- State-aware operator compression: see Q2. [arXiv:2609.12840](https://arxiv.org/abs/2609.12840) [S]; [arXiv:2609.18246](https://arxiv.org/abs/2609.18246) [S]; [xSPD arXiv:2506.13241](https://arxiv.org/abs/2506.13241) [S]

### Inferences
- **Heisenberg MPO with a reference state.**
  - (c) Target: at each cut, use the untruncated tensor's *reference-conditioned window operator* Tr_out[(ρ_ref^{out}) O], i.e. the partial expectation over sites outside the window, for a product reference state. All window coefficients are linear functionals of O, so the tilt solves a linear least-squares problem.
  - This is "DMT with ρ_ref replacing identity/d". A further step: preserve ⟨ψ|O|ψ⟩ itself plus its first derivatives with respect to local perturbations of ψ.
  - (d) Benefit: probably large when ψ is a product or low-entangled state and O(t) has much weight on strings with zero expectation in ψ (the xSPD insight). An SVD in Frobenius norm wastes rank on such strings, while the tilt moves rank toward the ψ-relevant subspace without a global fit (memory chi). This is the main advantage over 2609.18246, which needs the full back-propagated observable.
  - (e) Prior-art risk: moderate. 2609.12840 does it for Pauli propagation; DMT does the infinite-temperature version. No MPO cut-local reference-state version was found.
- **Influence matrix / temporal MPS.**
  - (c) Target: at a temporal cut, the untruncated two-time-site tensor's local-in-time RTM elements (local time correlators of the impurity or boundary observable).
  - (d) RTM truncation already targets the right object globally. Our tilt would be a cheap local correction on top of standard IM-MPS SVD compression, in the spirit of RTM truncation but without building the RTM.
  - Benefit: speculative. It depends on whether IM-SVD error is dominated by correlators near the temporal cut. The collateral (far-in-time correlators) is a problem when long-time memory matters, e.g. near integrability.
  - (e) Prior-art risk: moderate. The RTM/temporal-entropy line (Tagliacozzo, Carignano) is active.

### Gaps
- Could not get the actual truncation algorithm of 2504.07344 (Park–Gray–Chan), i.e. whether it uses plain SVD of the IF-MPS or something observable-aware.
- No full-text check of Frías-Pérez & Bañuls 2201.08402 for any expectation-value-targeted compression. Its content here is from prior knowledge.
- No "Heisenberg-picture TEBD with reference-state projection" paper could be identified under that name.

---

## Q5. Per-candidate summary: what is known, the gap, how to insert our tilt, expected benefit, prior-art risk

### Takeaway
Ranking for our cut-local tilt:
1. **OTOC / operator-weight profiles on the vectorized MPO.** Exact transfer of our RDM machinery; no prior art found; the operator front is our winning regime.
2. **Reference-state-conditioned Heisenberg MPO truncation** ("DMT with ρ_ref"). Large potential where many strings have zero expectation; moderate prior-art risk (2609.12840, DMT).
3. **Infinite-temperature transport (DMT/rTEBD territory).** Backflow makes our collateral cheap, but the prior art is strong; only incremental, e.g. longer-range two-body strings DMT misses.
4. **Temporal IM-MPS.** Speculative.
5. **Krylov / Lanczos.** Poor fit.
6. **Pure Pauli propagation.** No cut, so no fit except through a DAOE-style MPS-in-Pauli-basis hybrid.

### Cited Findings
- DMT: preserves all ≤3-site-diameter expectation values exactly by rotating Schmidt spaces at the cut. — [arXiv:1707.01506](https://arxiv.org/abs/1707.01506) [S]
- rTEBD: γ^{-n} Pauli reweighting in the SVD, and identifies DMT's failure on ⟨σ_iσ_{i+3}⟩. — [arXiv:2412.08730](https://arxiv.org/abs/2412.08730) [S]
- DAOE/backflow: high-weight content matters with exponential suppression in ℓ*. — [arXiv:2004.05177](https://arxiv.org/abs/2004.05177) [S]; [arXiv:2111.09904](https://arxiv.org/abs/2111.09904) [S]
- Benchmarks: DMT and OST agree on D within 1% to t=60/J; TEBD converges to t≈20. — [arXiv:2310.06886](https://arxiv.org/abs/2310.06886) [S]
- TDVP beats truncated TEBD for operator spreading. — [arXiv:1901.05793](https://arxiv.org/abs/1901.05793) [S]
- State-aware Pauli projection and joint state/observable compression (2026). — [arXiv:2609.12840](https://arxiv.org/abs/2609.12840) [S]; [arXiv:2609.18246](https://arxiv.org/abs/2609.18246) [S]
- RTM truncation targets local expectation values in spatio-temporal TN. — [arXiv:2509.03699](https://arxiv.org/abs/2509.03699) [S]; [Frontiers review](https://www.frontiersin.org/journals/quantum-science-and-technology/articles/10.3389/frqst.2025.1568471/full) [S]

### Inferences

| Candidate | (a) Known | (b) Gap | (c) Untruncated target for our tilt | (d) Expected benefit | (e) Prior-art risk |
|---|---|---|---|---|---|
| OTOC / operator front (vectorized MPO) | SVD-TEBD under-spreads; TDVP better (1901.05793); Xu–Swingle MPO OTOC (1802.00801) | No truncation that preserves the weight/OTOC profile | 1–3-site RDMs of normalized \|O⟩⟩ (d²-dim sites) in ±2 window = OTOC(x) and weight densities near the cut | Plausibly 2–5x at the front in non-integrable chains, by analogy with our Ising result; likely fails in XXZ | Low (nothing found) |
| Reference-state Heisenberg MPO | xSPD (2506.13241), state-adapted projection (2609.12840), global coupled losses (2609.18246) | No cut-local, memory-chi version for MPO | Window partial expectations Tr_out[ρ_ref^out O] (linear) | Potentially large for product reference states | Moderate |
| ∞-T transport / autocorrelation MPO | DMT (exact ≤3-site), rTEBD, DAOE/FDAOE, OST, LITE | Approximate DMT that also covers longer cross-cut strings at fixed chi | Pauli coefficients of all strings in the ±2 window (partial traces), optionally local conserved-density MPOs (2306.03431) | Incremental over DMT; collateral cheap due to backflow | **High** (DMT) |
| Temporal IM-MPS | SVD of IM-MPS; RTM truncation (2509.03699, 2505.09714); TN-IF BP (2504.07344) | Local, RTM-free observable-aware correction | Local-in-time RTM / two-time correlators at the temporal cut | Uncertain | Moderate |
| Krylov / Lanczos MPO | MPS-Lanczos loses orthogonality (2504.21786); R-UOG fails (2310.06886) | b_n-preserving compression | Low-weight head of O_n | Low; collateral harms orthogonality | Low |
| Pure Pauli propagation | weight/coeff/top-K/xSPD rules (2505.21606, 2510.22311) | No cut | n/a unless hybridized to an MPS in the Pauli basis | n/a | n/a |

- Cost note: for operator targets that are linear in O (partial traces, reference-state expectations), the Gauss-Newton step becomes a single linear least-squares problem. That should cut our 3–16x per-cut overhead. For the quadratic OTOC targets the overhead is as for state RDMs, with local dimension d² instead of d.
- Recommended first experiment: TFIM with longitudinal field (our winning model). Evolve σ^z_0(t) as an MPO and compare OTOC(x,t) and the weight profile from SVD vs tilt at equal chi, against a large-chi reference. Also report the autocorrelation Tr[σ^z_0(t)σ^z_0]/2^L, which depends only on a linear functional and is DMT's domain, as a control.

### Gaps
- Cannot rule out an unpublished or overlooked "operator-RDM-preserving" truncation for OTOCs. Suggested follow-up searches: "Frobenius-fidelity-preserving" or "operator weight preserving truncation"; the citation trails of 1802.00801 and 1901.05793.
- DMT's exact bond-dimension accounting (chi vs chi plus the preserved block) remains unverified, and it matters for any equal-chi claim against DMT.
- Full-text verification of every [PK] item is pending; arXiv was unreachable this session.
