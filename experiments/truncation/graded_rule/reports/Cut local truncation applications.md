# Purifications and operator fronts reward the tilt

The cut-local tilt (spcf) fits best in a place a pure-state method would not obviously aim for: **pure representations of mixed or operator objects**. That means three things. The first is vectorized Heisenberg operators, scored by OTOC and operator-weight profiles. The second is purification MPS for real-time dynamics at finite temperature. The third is quantum-trajectory unravelings of weakly dissipative Lindblad dynamics. In all three, the local quantities that matter are *quadratic* in the tensor. So DMT's exact linear construction does not apply. Entanglement also grows linearly in these settings, and only few-body outputs are wanted, which matches the regime where spcf beats SVD by 2.5-5x.

Wherever the target is *linear* in the tensor, DMT and reweighted TEBD (rTEBD) already own the idea and the prior-art risk is high. That covers MPDO TEBD, infinite-temperature transport, and Pauli-basis operator truncation. There the tilt adds at most incremental coverage of longer-range strings across the cut.

DMRG has one plausible insertion point: multi-target and correction-vector truncation. It also has one negative control: in-sweep ground-state truncation, which can only raise the energy. Machine-learning compression and weighted low-rank theory are better as sources of fixes than as places to deploy the method. The most valuable fix is a closed-form whitened SVD that could remove most of the 3-16x overhead. Quantum chemistry's compression of the MPS before 4-RDM evaluation is an open niche, but it fits poorly because the RDM elements chemistry needs are mostly far from any given cut.

The novelty framing "pure-state, fixed-χ, approximate counterpart of DMT" survives the searches, with two caveats. First, **arxiv.org full texts were unreachable**, so every claim below rests on abstracts and search snippets. Second, the Paeckel et al. review's refs [73]/[74] on "optimizing the local density matrix" remain the one unresolved threat to that framing. One engineering step would open the top three applications and allow a head-to-head with DMT at equal parameter count: generalizing `rule/spcfast.py` to local dimension 4 with a configurable set of target strings.

## DMT owns the linear problem; the tilt owns the quadratic one

**DMT is the conceptual parent, and the report has to say so first.**
- DMT is White, Zaletel, Mong and Refael's density-matrix truncation for matrix-product density operators (MPDOs). Its truncation "exactly preserves the energy density of the system and other local conserved quantities". It "ameliorates the positivity problem of Frobenius truncation", and it **exactly preserves the expectation values of all operators on regions of up to three sites** ([arXiv:1707.01506](https://arxiv.org/pdf/1707.01506)).
- The paper argues that Frobenius truncation "is not the correct notion of error for density matrices" ([arXiv:1707.01506](https://arxiv.org/pdf/1707.01506)).
- The Ye et al. follow-up generalizes the preserved diameter to ℓ. It splits the bond into a χ^(preserve) block that stores all ℓ-site observables around the cut (stated as 2^ℓ in the snippet) and a χ^(extra) block that keeps the largest remaining correlations. It also says Frobenius-optimal SVD "is blind to the fact that some operators, especially local operators like energy density, are more important than others" ([arXiv:1902.01859](https://arxiv.org/html/1902.01859)).
- DMT now has a track record. It was used for KPZ dynamics in a quantum-gas-microscope experiment, explicitly because it "preserves local observables rather than maximizing mutual information" ([arXiv:2107.00038](https://arxiv.org/pdf/2107.00038)).
- In a four-method benchmark, DMT and operator-size truncation (OST) agree out to **t = 60/J, with diffusion constants within 1%**. Plain TEBD converges only to t ≈ 20/J, yet its D is still within 1% ([arXiv:2310.06886](https://arxiv.org/abs/2310.06886)).
- DMT has a documented weakness that mirrors ours. It does not keep longer-range two-body correlators such as ⟨σ_iσ_{i+3}⟩ ([arXiv:2412.08730](https://arxiv.org/pdf/2412.08730)). Errors in longer-range operators also "propagate down to the three-site density operators via the system's dynamics" ([arXiv:1902.01859](https://arxiv.org/html/1902.01859)).

**Why DMT can be exact and spcf cannot.** For a vectorized density matrix |ρ⟩⟩, a local expectation value Tr(Oρ) = ⟨⟨O|ρ⟩⟩ is *linear* in the two-site tensor. Holding a set of local expectations fixed is then a linear constraint, and DMT meets it exactly by reserving rows and columns of the bond matrix. For a pure MPS, ⟨ψ|O|ψ⟩ is *quadratic* in the tensor, and the rank-χ constraint is non-convex. One Gauss-Newton step in the tangent space, Q = qr(U_k + U_⊥C), is the natural cheap approximation.

This is the honest novelty statement. **spcf is the pure-state, fixed-χ, approximate counterpart of DMT.** It applies DMT's criterion, "preserve the ≤3-site RDMs around the cut", where that criterion cannot be imposed exactly, and it spends no extra bond dimension doing so.

The trade-offs fall in opposite places. DMT keeps locality exactly and gives up fidelity. spcf leaves fidelity unchanged and gives up exactness. Both push error to longer range: DMT through the lost ⟨σ_iσ_{i+3}⟩ and the leak down the hierarchy, spcf through its first-order collateral on far observables. Any comparison between them therefore has to report far-observable drift as well as the targeted RDMs.

**rTEBD is the second anchor.**
- It is attributed on its SciPost submission page to Guha Roy and Slagle. It rewrites the MPDO in a reweighted Pauli basis, so that an n-Pauli correlator is suppressed by γ^(-n) during the SVD. This "causes the SVD truncation step to more accurately preserve the expectation value of few-body operators" and needs only some matrices rescaled ([arXiv:2412.08730](https://arxiv.org/pdf/2412.08730); [SciPost submission](https://scipost.org/submissions/2412.08730v3)).
- It is reported as "significantly more accurate than standard TEBD time evolution of an MPDO, and competitive with, and sometimes better than, TEBD using MPS" ([arXiv:2412.08730](https://arxiv.org/pdf/2412.08730)).
- A free-fermion benchmark is quoted at L = 128 with γ = 1.5 ([arXiv:2412.08730](https://arxiv.org/abs/2412.08730)).

The last claim matters most for us. rTEBD applied to ρ = |ψ⟩⟨ψ| competes **directly with spcf on our own Ising cells**, at roughly SVD cost. Its code is public (github guharoysayak/reweighted_TEBD). Our own lineage also overlaps with rTEBD: the repo's earlier graded rule already weights its window distance by Pauli weight with γ = 4 (`README.md`). Any write-up must cite rTEBD as the source of that grading. It should also run rTEBD as a baseline next to the existing `dmt_baseline.py`.

The remaining neighbours are further away, either because they change the representation or because they use a global objective:

| Method | Object truncated | What is preserved | How it is imposed | Bond budget | Fidelity |
|---|---|---|---|---|---|
| DMT ([1707.01506](https://arxiv.org/pdf/1707.01506)) | MPDO / operator | all ≤3-site expectations, exactly | linear: reserved rows/columns + connected-matrix SVD | χ_preserve + χ_extra | given up |
| rTEBD ([2412.08730](https://arxiv.org/pdf/2412.08730)) | MPDO / operator | low-Pauli-weight content, preferentially | γ^(-n) reweighting before the SVD (closed form) | χ | not targeted |
| LITE ([2310.06036](https://arxiv.org/abs/2310.06036)) | subsystem RDMs, not an MPS | all RDMs below ℓ_min, plus information currents | evolves the information lattice; cost exponential in ℓ_max | n/a | abandoned |
| Mixture methods ([1810.01231](https://arxiv.org/pdf/1810.01231), [2308.04291](https://arxiv.org/pdf/2308.04291)) | pure → mixed | local RDMs | turns long-range entanglement into mixture | varies | abandoned |
| Tangent-space uMPS truncation ([SciPost Core 4, 004](https://scipost.org/SciPostPhysCore.4.1.004)) | uniform MPS | global fidelity | variational tangent-space step | χ | optimized |
| **spcf (ours)** | pure MPS at a TEBD cut | 1-3-site RDMs in ±2 window, to first order | one Gauss-Newton tilt + QR retraction | χ | unchanged to first order |

Two more neighbours sit in applied mathematics.
- **LoMaC** orthogonally projects a low-rank kinetic solution onto the subspace fixed by the macroscopic densities. This gives exact local conservation of mass, momentum and energy, which plain SVD destroys ([arXiv:2207.00518](https://arxiv.org/abs/2207.00518)). It is a linear, DMT-like analogue outside physics.
- **Weighted low-rank approximation** has no closed form except when the weights are separable ([Srebro & Jaakkola](https://home.ttic.edu/~nati/Publications/ICML03slides.pdf)). It reduces to a subspace optimization on the Grassmannian with only local minima ([Manton, Mahony & Hua](https://escholarship.org/content/qt08m9c4z6/qt08m9c4z6.pdf?t=lnotos)).

In that language, spcf is **one Riemannian Gauss-Newton iteration on the Grassmannian, started from the Frobenius optimum, with a QR retraction**. Truncated SVD itself is a valid retraction on the fixed-rank manifold ([arXiv:1209.3834](https://ar5iv.labs.arxiv.org/html/1209.3834)). This framing also suggests why fidelity is untouched while far observables move at first order: the Frobenius loss is stationary at the SVD point, so a tangent step costs fidelity only at second order but can shift any particular non-targeted observable at first order. That argument is our own derivation, not something taken from a source, and it should be checked before it is published.

## A ranked map of where the tilt plugs in

The ranking uses four filters, all taken from what is known about the mechanism:
1. The local targets must be *quadratic* in the truncated tensor, because otherwise DMT-type exact methods exist.
2. The cross-cut 2-site error under SVD must be large. This holds in the growth phase of non-integrable dynamics, and fails for Heisenberg-like and area-law states.
3. Few-body outputs must be the actual deliverable, because spcf does not improve fidelity.
4. A first experiment must be feasible with the repo's Python TEBD, dense references of about 2^20 amplitudes (or 4^10-4^12 for doubled spaces), and the existing benefit/cost gate.

**One shared step opens ranks 1, 2, 4 and 6.** `rule/spcfast.py` currently builds its residual from Pauli strings on d = 2 sites. It needs two changes: local dimension d = 4 (fused site+ancilla, or a Pauli-basis operator site), and a configurable set of target strings. The target set for purification is physical Paulis with identity on the ancilla. For OTOC it is the diagonal superoperators W⊗W*. For reference-state targets it is linear partial expectations.

| Rank | Application | Where it plugs in / untruncated target | Expected benefit under our mechanism | Novelty risk | First experiment in this repo |
|---|---|---|---|---|---|
| 1 | OTOC and operator-weight fronts on a vectorized Heisenberg MPO | Two-site block of normalized \|O⟩⟩ (d = 4); 1-3-site "operator RDMs" near the cut, which contain OTOC(x) and the weight density | High in non-integrable chains (front = fast operator-entanglement growth); failure expected in XXZ | Low (nothing found) | Mixed-field Ising, N = 12, O = Z_6(t); SVD vs spcf vs gated spcf vs DMT on OTOC(x,t), weight profile, front position |
| 2 | Purification MPS, real-time finite-T dynamics | Two-site site+ancilla block; target = *physical-only* 1-3-site RDMs | Moderate-high: entanglement grows linearly; only local correlators needed; stacks with disentanglers | Low-moderate | N = 10 fused sites, high-T magnetization domain wall; spcf-purification vs DMT-MPDO at equal parameters |
| 3 | Quantum-trajectory MPS, weak Lindblad dissipation | Unchanged pure-state cut per trajectory | Moderate at weak dissipation, decaying with rate γ; removes bias that sampling cannot | Low | Dephasing as random Z kicks applied identically to MPS and dense state, N = 16-20, γ scan |
| 4 | Reference-state-conditioned Heisenberg MPO ("DMT with ρ_ref") | Window partial expectations Tr_out[ρ_ref^out O] (linear, so the step becomes linear least squares) | Potentially large for product references; cheaper per cut | Moderate | ⟨Néel\|Z_6(t)\|Néel⟩ vs dense, SVD-MPO, and `pauli_prop.py` at equal memory |
| 5 | Multi-target / correction-vector DMRG | Final-sweep truncation of the two-site tensor; target = local RDMs of every target state | Plausible: competing targets make cross-cut errors large under one eigenvector rule | Moderate | `pytreenet/dmrg`, TFIM + longitudinal field N = 20; spcf vs weighted-ρ targeting; Lanczos reference |
| 6 | LPDO noisy circuits / Lindblad | Block with physical + Kraus legs; target = physical RDMs (quadratic) | Moderate during the operator-entanglement "rise"; the gate shuts off once noise dominates | Low-moderate | After rank 3: depolarizing kicked Ising, N = 8 |
| 7 | 1D quench benchmarking / error-mitigation verification windows | Existing cut | Turns a 2.5-5x error drop into a χ reduction r; wall-clock win only if r³ exceeds the 3-16x overhead | Low (no truncation-rule work found) | `matched.py` ladders on a kicked-Ising chain, N = 20 |
| 8 | DMRG quantum chemistry: "reverse-schedule" compression before 3/4-RDMs | Compression-sweep cuts; target = window RDMs | Low-moderate: chemistry RDM elements are mostly far from the cut; spcf's RDMs come from a real state, so they stay N-representable | Low | Proxy: compress exact N = 16-20 long-range-model states; score all 2-RDM elements |
| 9 | Post-hoc compression of converged states; iDMRG/uMPS | Compression sweep; tangent-space uMPS step with a local objective | Small for ground states (gate stays off); collateral coherent across cuts in uMPS | Low-moderate | Same sweep as rank 8 on gapped vs critical ground states |
| 10 | Infinite-T transport in MPDO / Pauli MPO | Pauli coefficients of window strings (linear) | Incremental over DMT (longer-range cross-cut strings only); collateral cheap because of operator backflow | **High** | Control only, reusing `dmt_baseline.py` |
| 11 | Krylov/Lanczos MPO, pure Pauli propagation, 3S/CBE expansion selection, METTS / imaginary time, temporal influence-matrix MPS | Various | Poor or speculative | Low to moderate | Not recommended as first experiments |

## Three pure representations of non-pure objects lead the ranking

### Rank 1: OTOCs and operator fronts are spcf's machinery with d = 4

**What is known.**
- MPO computation of OTOCs and the butterfly front in chaotic chains is established (Xu & Swingle, [arXiv:1802.00801](https://arxiv.org/abs/1802.00801)). The arXiv ID is from prior knowledge; the journal reference was confirmed.
- Truncated MPO-TEBD is known to **halt operator spreading unphysically** ([arXiv:1901.05793](https://arxiv.org/abs/1901.05793)).
- In random circuits the operator front drifts with the butterfly velocity and broadens as t^{1/2} ([Nahum, Vijay, Haah](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021014)).
- Operator support stays within |x| < v_B t, with exponentially small tails that a small bond dimension can capture ([arXiv:2202.07060](https://arxiv.org/abs/2202.07060)).
- Recent symmetry-resolved OTOC work runs χ up to 6000 and tracks only the cumulative discarded singular values ([arXiv:2503.20327](https://arxiv.org/abs/2503.20327)).

**The gap.** No truncation found preserves the OTOC or weight profile near the cut at fixed χ.

**Why spcf fits.** For a Frobenius-normalized |O⟩⟩, the OTOC with a single-site Pauli W_x is a one-site observable of the vectorized operator, ⟨⟨O|W_x⊗W_x*|O⟩⟩. The Pauli-weight density at site x is also a one-site observable. Both are **quadratic in |O⟩⟩**, exactly the class spcf handles and DMT does not, since DMT preserves only linear partial traces. The fidelity spcf leaves unchanged becomes Frobenius fidelity.

**Expected benefit.**
- The operator front is a region of fast operator-entanglement growth in non-integrable chains. That is the analogue of spcf's winning regime, so a gain comparable to the 2.5-5x seen in Ising state TEBD is plausible. This is an inference, not a measurement.
- The collateral may also be less visible here than in state TEBD. Behind the front the OTOC is saturated; ahead of it the operator is trivial.
- Operator-backflow theory adds a further reason. Errors in high-weight, far content feed back into transport coefficients only with suppression that is exponential in the cutoff ([arXiv:2111.09904](https://arxiv.org/abs/2111.09904)).
- Integrable XXZ should fail, for the same reason Heisenberg does. Operator entanglement grows only logarithmically in integrable chains ([arXiv:0706.2480](https://arxiv.org/abs/0706.2480)), so SVD's cross-cut error stays small.

**First experiment.**
- Model and size: the repo's mixed-field Ising, Pauli-fused chain of N = 12 (a dense vectorized operator of 4^12 ≈ 1.7×10^7 amplitudes, about 270 MB). Evolve O = Z_6 in the Heisenberg picture with the same Trotter gates.
- Score:
  - OTOC(x,t) for W = X and W = Z;
  - the Pauli-weight profile;
  - the front position;
  - Frobenius fidelity;
  - as a linear control, the autocorrelation Tr[Z_6(t)Z_6]/2^N, which is DMT's home ground.
- Arms: SVD, spcf, and gated spcf at equal χ ∈ {8…64}, with XXZ as the expected-failure control.
- Cheaper variant: restrict the targets to the four diagonal superoperators per site. This keeps the residual small even though d = 4.
- Success: at least a 2x lower OTOC-profile error near the front, with Frobenius fidelity unchanged. A visible reduction of SVD's under-spreading would be a separate physics result.

### Rank 2: purifications make finite-temperature dynamics a pure-state problem

**What is known.**
- Purification MPS are pure states in a doubled space. Any ancilla-only unitary is a gauge freedom, already used to lower entanglement ([Hauschild et al., arXiv:1711.01288](https://arxiv.org/pdf/1711.01288)).
- The same paper says dynamics methods aim at "the correct macrostate rather than the exact microstate". The usual DMRG truncation error "can therefore be large, since it is measured against the exact state" ([arXiv:1711.01288](https://arxiv.org/pdf/1711.01288)). That is the spcf thesis stated in another context.
- Entropy grows roughly linearly under real-time evolution with or without disentanglers, with very different prefactors ([arXiv:1711.01288](https://arxiv.org/pdf/1711.01288)).
- At fixed cost, purifications beat METTS "for almost all considered cases … often by orders of magnitude" ([Binder & Barthel, arXiv:1411.3033](https://arxiv.org/abs/1411.3033v1)).

**The gap.** No truncation designed to preserve local *thermal* expectation values was found.

**How spcf plugs in.** Fuse each site with its ancilla (d = 4). Target the physical-only 1-3-site RDMs of the untruncated two-site block, which in Pauli form means strings with identity on every ancilla. Physical RDMs are quadratic in the purification, so this is spcf's native setting. Positivity and trace come for free, so the problem DMT had to "ameliorate" does not arise.

**Expected benefit.**
- Moderate to high for real-time correlators and transport at T > 0.
- Low for imaginary-time preparation and METTS, where entanglement is area-law-like and the gate would mostly fall back to SVD.
- A disentangler lowers the discarded weight, and spcf repairs the local error of what is still discarded, so the two should combine.

**First experiment.**
- Initial state: a high-temperature magnetization domain wall, ρ ∝ ∏ exp(μ_i Z_i) with μ_i = +μ on the left and −μ on the right. Its purification is a product state of tilted Bell pairs.
- Evolution: mixed-field Ising H⊗I on N = 10 fused sites (a dense reference of 4^10 ≈ 10^6 amplitudes). Run once with plain ancillas and once with backward-evolving ancillas, H⊗I − I⊗H^T.
- Score: physical ⟨Z_i⟩, ⟨Z_iZ_{i+1}⟩ and energy-density profiles.
- Head-to-head with DMT: the repo's `dmt_baseline.py` already runs DMT on a fused d = 4 Pauli MPDO and counts stored parameters. A d = 4 purification and a d² = 4 MPDO store the same 4χ² parameters per site, so **spcf-on-purification vs DMT-on-MPDO is a natural equal-parameter comparison**. This is the cleanest experiment available for turning the novelty framing into a measured result.

### Rank 3: quantum trajectories need no new cut code

**What is known.**
- Each MCWF trajectory of a Lindblad equation is a pure MPS evolved by TEBD. Large-scale stochastic simulation exists ([arXiv:2501.17913](https://arxiv.org/html/2501.17913v1), title level only).
- Dissipation contracts truncation errors in noisy MPO dynamics exponentially in system size N and time t ([arXiv:2603.20400](https://arxiv.org/pdf/2603.20400)). Per-step local bias is therefore neither compounded nor amplified.

**The gap.** No local-RDM truncation per trajectory was found.

**Expected benefit.** spcf applies unchanged. The benefit should be largest at weak dissipation, where trajectories still pass through a long growth phase, and fade as the dissipation rate γ grows. Lower per-trajectory local bias is valuable because sampling over trajectories cannot average a bias away.

**First experiment.** This one isolates truncation error cleanly.
- Unravel dephasing as random single-site Z rotations. This is a standard construction, not from the notes. The noise is then independent of the state, so the identical noise realization can be applied to the MPS and to a dense 2^16-2^20 state vector.
- Scan γ ∈ {0, 0.01, 0.03, 0.1} on the Ising chains where spcf already wins. Average 50-100 realizations, and record the gain and the gate's on-fraction as functions of γ.
- Prediction: the gain falls from the measured γ = 0 value toward 1, and the gate switches off progressively.

This is the cheapest experiment in the report and should run first as a smoke test of the "growth-phase only" mechanism.

## Operator, DMRG and beyond-chain uses pay only conditionally

### Rank 4: reference-state conditioning

**What is known.** State-aware operator compression has just appeared, but only in global form:
- xSPD drops strings with many X/Y factors, which have zero expectation in computational-basis states ([arXiv:2506.13241](https://arxiv.org/abs/2506.13241)).
- A state-adapted projection folds high-body content onto low-body representatives using the covariance geometry of the initial state ([arXiv:2609.12840](https://arxiv.org/abs/2609.12840)).
- A coupled forward/backward compression reports **2-3 orders of magnitude lower error than variational state compression** on 30-qubit random circuits. It needs the full back-propagated observable in memory, and an automated review flags thin numerics ([arXiv:2609.18246](https://arxiv.org/abs/2609.18246); [Pith review](https://pith.science/paper/2609.18246)).

**How spcf plugs in.** A cut-local MPO version would target the window coefficients Tr_out[ρ_ref^out O], which amounts to DMT with ρ_ref in place of the identity. These targets are linear, so the Gauss-Newton step becomes **a single linear least-squares solve**. That should cut the per-cut overhead. The differentiator against 2609.18246 is peak memory of χ, with no back-propagated observable.

**First experiment.** Compute ⟨Néel|Z_6(t)|Néel⟩ for N = 12. The repo's `pauli_prop.py` already evaluates coefficient-thresholded Pauli propagation on the Néel state through the same circuit, so the memory-matched baseline exists. Prior-art risk is moderate.

### Rank 5: DMRG multi-target and correction-vector truncation

**What is known.** Every DMRG basis rule found expands by an energy residual and truncates by the leading eigenvectors of a (perturbed or mixed) ρ:
- White's density-matrix perturbation ([cond-mat/0508709](https://arxiv.org/abs/cond-mat/0508709));
- DMRG3S, from AMEn ([arXiv:1501.05504](https://arxiv.org/abs/1501.05504));
- CBE, which selects the HΨ weight in the discarded space ([arXiv:2207.14712](https://arxiv.org/abs/2207.14712)), together with the McCulloch-Osborne comment and the reply ([arXiv:2403.00562](https://arxiv.org/abs/2403.00562); [arXiv:2501.12291](https://arxiv.org/abs/2501.12291));
- zero-site DMRG ([arXiv:1908.10880](https://arxiv.org/abs/1908.10880v2)).

None chooses the kept subspace to preserve local RDMs. spcf belongs in the truncation layer and composes with any of these expansion schemes.

**Ground-state sweeps are the wrong target.** Energy error is linear in the discarded weight ε, while other observables carry √ε ([Jeckelmann slides](https://www.issp.u-tokyo.ac.jp/public/CQCP/26dmrg_issp1.pdf); [EPJB 2023](https://link.springer.com/article/10.1140/epjb/s10051-023-00575-2)). At a converged variational minimum any tilt can only raise the energy. It would also break the energy-vs-ε extrapolation. That makes it a negative control, not a product.

**Multi-target DMRG is the real opening.** It mixes targets with weights and keeps the leading eigenvectors of Σ w_f ρ_f. Its accuracy "depend[s] significantly and sometimes unpredictably on the specific states included as target" ([arXiv:0808.2620](https://arxiv.org/pdf/0808.2620); [arXiv:1902.09621](https://arxiv.org/abs/1902.09621)). Replacing the heuristic weights with explicit local-RDM constraints on each target is a clean idea.

**First experiment.** Use `pytreenet/dmrg` (or a small numpy two-site sweep) on a TFIM with a longitudinal field at N = 20 near criticality. Use two targets, the ground state and σ^x_{10}|ψ₀⟩ (or the first excited state). Compare spcf truncation against weighted-ρ targeting over a sweep of weights, with Lanczos as the reference. The gate's diagnostic, the cross-cut RDM error, comes from the same two-site tensor at no extra memory cost.

### Rank 6: noisy circuits

**What is known.**
- Pure-state MPS "fails to approximate the noisy output quantum states for any of the noise models considered", while MPDO with "optimal truncations for both the inner and the bond dimensions" works ([arXiv:2004.02388](https://arxiv.org/pdf/2004.02388)).
- MPO truncation breaks positivity, and locally purified LPDOs fix that ([CPL 41, 120302](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/41/12/120302)).

**Where spcf fits.** On an MPDO the targets are linear, so spcf is redundant with DMT. On an LPDO the physical RDMs are quadratic, and spcf applies during the early rise of operator entanglement. Once noise dominates, SVD is already adequate and the gate should switch off.

**A possible niche.** A tilt that also penalizes negative eigenvalues of the 3-site RDM, i.e. local positivity, for MPDOs. No source was found for this. It needs a Kraus leg in the code, so it should follow rank 3.

### Rank 7: simulator benchmarking in 1D, and spcf's value is memory unless r is large

**What is known.** The quantum-advantage literature is mostly 2D/3D and so outside this report's scope. Its lessons still transfer:
- Headline metrics are one- and two-site observables during growth.
- Classical hardness is often quoted in MPS cost. In the D-Wave rebuttal, the scaling analysis "applies to MPS — the only method with which we can match QPU quality for all considered quench times" ([arXiv:2503.05693](https://arxiv.org/pdf/2503.05693)).
- MPS error on the IBM kicked-Ising observables grows from about step 10 ([arXiv:2306.14887](https://arxiv.org/pdf/2306.14887)).

**What spcf would change.** For 1D chains, a 2.5-5x local-error drop at equal χ converts into a χ reduction factor r at equal error. Memory falls by r². Since TEBD time scales as χ³, wall-clock falls only if r³ exceeds the overhead: **r > 1.44 at 3x overhead and r > 2.5 at 16x**. This break-even is our inference, not a sourced figure.

**First experiment.** Run `matched.py` SVD and spcf ladders on a 1D kicked-Ising chain at N = 20 and report r directly. MPS checks of zero-noise extrapolation beyond the verifiable regime (Anand et al., [arXiv:2306.17839](https://arxiv.org/abs/2306.17839), title only) are the natural consumer of a wider classically checkable window.

### Ranks 8 and 9: quantum chemistry and post-hoc compression

**What is known.**
- DMRG-SC-NEVPT2 evaluates the 4-RDM "using a lower bond dimension than is used in the DMRG energy optimization", via a "reverse schedule" of compression sweeps ([arXiv:1512.08137](https://arxiv.org/abs/1512.08137)).
- Compressing a DMRG-CASSCF hydrogen chain to D = 500 cost under 0.3 mEh at most geometries but **10 mEh at the shortest bond length** ([arXiv:1705.01608](https://arxiv.org/pdf/1705.01608)).
- Cumulant approximations to the 3/4-RDM can be "profound or even catastrophic" for stretched N₂ or Cr₂, with SC-NEVPT2 errors up to 2.5 Ha in one filtering scheme ([ORCA 6.1 manual](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/NEVPT2.html)).

**Where spcf fits.** A tilted compression sweep is a direct transplant, and its RDMs come from a genuine pure state, so they are automatically consistent (N-representable). The obstacle is locality. In an orbital-ordered MPS, most Γ_pqrs elements are far from any given cut. A ±2 window covers only a small fraction of what NEVPT2 needs.

**First experiment (proxy).** Compress exact N = 16-20 states of a model with long-range couplings and score *all* 2-RDM elements, not just the targeted ones. Rank 9, plain post-hoc compression of ground states, is expected to give little: ground states have small ε, and the gate will mostly decline. A uMPS version could reuse the tangent-space truncation machinery of Vanhecke et al. ([SciPost Core 4, 004](https://scipost.org/SciPostPhysCore.4.1.004)), but there the collateral is identical at every cut and adds coherently.

### Rank 10: infinite-temperature transport is DMT's home and should be a control only

**What is known.** This is DMT, rTEBD, DAOE and OST territory. DAOE damps strings above a weight ℓ*, with an error in D "exponentially small in ℓ*" ([arXiv:2004.05177](https://www.arxiv.org/pdf/2004.05177)). Plain TEBD already gets D within 1% ([arXiv:2310.06886](https://arxiv.org/abs/2310.06886)).

**What spcf could add.** At most, cross-cut strings such as σ_iσ_{i+3} that DMT drops. If anything here benefits from spcf's growth-phase advantage, it would be transient quantities: the early current autocorrelation and the ballistic-to-diffusive crossover ([arXiv:2408.08249](https://arxiv.org/abs/2408.08249)).

### Poor fits

**Krylov and Lanczos.** Truncated MPS-Lanczos already loses orthogonality ([arXiv:2504.21786](https://arxiv.org/abs/2504.21786)), and a first-order collateral change in far content would make that worse.

**Pure Pauli propagation.** It truncates string by string, with no bond or cut to tilt ([arXiv:2505.21606](https://arxiv.org/abs/2505.21606)).

**Expansion selection in 3S/CBE.** In single-site expansion there is no untruncated reference RDM to match.

**Temporal influence-matrix MPS.** This is speculative. The reduced-transition-matrix line already truncates the object that fixes local expectation values ([arXiv:2509.03699](https://arxiv.org/abs/2509.03699)).

**Machine-learning layer compression.** It is not a destination. For a linear layer the output-reconstruction objective is quadratic and is already solved exactly by whitening ([arXiv:2403.07378](https://arxiv.org/abs/2403.07378)), so a Gauss-Newton tilt adds nothing there.

## Borrow back whitening, damping, feedback and conserved currents

**The largest available gain is a closed-form whitened SVD to replace the iterative step.**
- SVD-LLM builds a whitening matrix from the Cholesky factor of the calibration activation covariance and truncates the SVD of W·S. It proves a "direct mapping between singular values and model compression loss" ([arXiv:2403.07378](https://arxiv.org/abs/2403.07378)).
- CorDA does the same with the input covariance and folds C⁻¹ back in ([arXiv:2406.05223](https://arxiv.org/pdf/2406.05223)).
- Friedland-Torokhti generalize Eckart-Young to min ‖A − BXC‖ with a closed form ([arXiv:2408.05104](https://arxiv.org/pdf/2408.05104)).
- A 2026 tensor-ring paper reduces a metric-aware truncation to an ordinary SVD whenever the metric is separable ([arXiv:2610.10027](https://arxiv.org/html/2610.10027)).

The transplant has three steps. Linearize the RDM map at the SVD point. Approximate the Gauss-Newton metric JᵀJ by a Kronecker product L⊗R over the left and right bond indices. Then take one SVD of L^{1/2}θR^{1/2} and un-whiten. The measure of success is the fraction of spcf's gain retained at roughly 1-2x SVD cost.

There is a known failure mode. ASVD's inverse scaling "often fails due to theoretical and numerical issues" ([arXiv:2502.02723](https://arxiv.org/pdf/2502.02723)), so regularize the whitener. For the linear operator targets of ranks 4 and 10, the metric is exactly quadratic, so whitening is exact rather than approximate.

**Damping maps the near-vs-far trade-off.** GPTQ adds λI to its Hessian for stability ([arXiv:2210.17323](https://arxiv.org/pdf/2210.17323)). `spcfast.py` already carries the analogous knob: `fw` weights a second-order discarded-weight penalty on C, and the best configuration uses fw = 0. A Levenberg-Marquardt scan of `fw` should trace a Pareto front between targeted-RDM error and far-observable collateral. That is the right figure to show reviewers who worry about collateral.

**Error feedback replaces per-cut isolation.**
- GPTQ/OBS compensate each quantization error on weights not yet processed ([arXiv:2210.17323](https://arxiv.org/pdf/2210.17323)).
- SVD-LLM adds a closed-form update of the remaining weights ([arXiv:2403.07378](https://arxiv.org/abs/2403.07378)).

The TEBD analogue: within a Strang sweep, carry cut b's residual RDM mismatch forward as an extra target at cut b+1.

**Conserved densities and currents are the Heisenberg-chain hypothesis.**
- DMT's success in transport comes from preserving energy density, magnetization and their currents ([arXiv:1902.01859](https://arxiv.org/html/1902.01859)).
- Exact local conserved charges of the Heisenberg chain are available as MPOs whose bond dimension grows linearly with locality ([arXiv:2306.03431](https://arxiv.org/abs/2306.03431)).

Two changes follow, and both are speculative. Add the spin current on the cut bond to spcf's target set. And restrict C to U(1) symmetry blocks so that the tilt cannot leak out of the S^z sector; `diag_charge.py` already measures that leak. Neither addresses the identified mechanism directly, which is SVD's small cross-cut error, so they could reduce the loss but are unlikely to produce a win.

**A predictive gate attacks cost before the step is computed.** TFWSVD introduces a "Fisher information variance" metric that predicts when standard SVD will fail ([arXiv:2211.09718](https://arxiv.org/abs/2211.09718)). A spectral statistic, compression ratio × stable rank, predicts compression damage with r ≈ 0.84-0.89, where perplexity manages only 0.093 ([arXiv:2604.18085](https://arxiv.org/abs/2604.18085)). The spcf analogue is to estimate the cross-cut SVD error from the two-site tensor and skip the Gauss-Newton solve outright where the current gate would decline anyway.

**Two more borrowings.**
- *DMT's reserved rank as an equal-parameter control.* Spend a few bond dimensions on exact preservation and compare that against tilting at the same total parameter count.
- *The ML community's reporting discipline.* In one Mistral-7B study, perplexity stayed reasonable at 50% SVD rank while MMLU fell from 62.3% to 51.5% ([arXiv:2601.07197](https://arxiv.org/pdf/2601.07197)). That is the ML version of our near-vs-far collateral. Always report far observables and fidelity next to the targeted RDMs.

## Every claim here is abstract-level, and four prior-art checks stay open

**Evidence status.**
- During this research, arxiv.org, export.arxiv.org, ar5iv, the APS pages, Semantic Scholar and alphaxiv were all unreachable: DNS failures, proxy 403 errors on CONNECT, and refused WebFetch.
- **No full text was read.** Every finding above comes from abstracts, search-engine snippets or automated summaries. A few arXiv IDs come from the researchers' prior knowledge and are flagged as such where used.
- No quantitative claim (1%, t = 60/J, 10 mEh, the rTEBD benchmarks, the 2609.18246 orders of magnitude) has been checked against a figure or table.
- One sentence from search results could not be attributed to any paper: when truncating at every bond, "only local observables with support on up to two neighboring sites remain unaffected" while errors feed back into two-site dynamics. It is probably from 2412.08730 or 1901.05824, and must be traced before anyone cites it.

| Open check | Why it matters | How to resolve |
|---|---|---|
| Paeckel et al. review refs [73]/[74] ("optimizing the local density matrix … aims at preserving local observables") ([arXiv:1901.05824](https://arxiv.org/pdf/1901.05824)) | **The single most important check.** If either is a pure-state, local-RDM-targeted MPS truncation, the novelty claim narrows to "tangent-space tilt at fixed χ" | Read the reference list of 1901.05824; follow both refs forward in citations |
| DMT bond-dimension accounting: χ vs χ_preserve + χ_extra, and 2^ℓ vs 4^ℓ in operator space; whether positivity is exact or only "ameliorated" ([arXiv:1707.01506](https://arxiv.org/pdf/1707.01506); [arXiv:1902.01859](https://arxiv.org/html/1902.01859)) | Every equal-χ claim against DMT depends on it | Full text of both; also inspect the GCG implementation behind `dmt_baseline.py`, but the paper's definition governs |
| rTEBD details ([arXiv:2412.08730](https://arxiv.org/pdf/2412.08730)) | Authorship (one note unverified), whether a pure-state variant exists, its numbers against MPS-TEBD, and whether it compares to DMT | Full text; run the public code on our Ising cells at matched parameters |
| OTOC/weight-preserving truncation | Rank 1 rests on "nothing found" | Citation trails of [1802.00801](https://arxiv.org/abs/1802.00801) and [1901.05793](https://arxiv.org/abs/1901.05793); Swingle, Luitz and Pollmann groups; search "operator weight preserving truncation" |
| Truncation criteria in Cheng et al. ([2004.02388](https://arxiv.org/pdf/2004.02388)), noise-contraction methods ([2603.20400](https://arxiv.org/pdf/2603.20400)), and the projection formula of [2609.12840](https://arxiv.org/abs/2609.12840) | Prior-art risk for ranks 4 and 6 | Full texts |
| Purification disentangler papers (Karrasch-Bardarson-Moore 1205.3756; Barthel 1301.2246; prior-knowledge IDs) | Any local-thermal-observable truncation would hit rank 2 | Retrieve and read |
| Environment-weighted truncation (Evenbly, 1801.05390, prior-knowledge ID) | Likely the closest mathematical precedent for the whitened route | Verify ID and content |
| ID hygiene | Kloss-Bar Lev-Reichman is [arXiv:1710.09378](https://arxiv.org/pdf/1710.09378), not 1712.03996. Surace-Piani-Tagliacozzo is cited as [1810.01231](https://arxiv.org/pdf/1810.01231), but one note proposed 1902.09523. The authors of [2510.22311](https://arxiv.org/abs/2510.22311) do not match the brief | Confirm before citing |

The four rows that most threaten or shape the core claim are the first four: Paeckel refs [73]/[74], DMT's bond accounting, rTEBD's details, and whether an OTOC-preserving truncation already exists. The remaining rows are prior-art checks for individual ranked applications, or ID hygiene.

## Conclusion

The dividing line for spcf's future is **linear versus quadratic targets, not mixed versus pure states**. That inverts the obvious positioning. A method designed for pure-state TEBD finds its least contested and most physically useful applications in mixed-state physics (via purifications and trajectories) and in operator physics (via OTOCs). Those are exactly the places where the honest novelty statement, a pure-state, fixed-χ, approximate counterpart of DMT, becomes a capability DMT cannot offer, not merely a cheaper copy of it.

The d = 4 generalization is therefore the strategic investment. It opens three of the top four applications at once, and it turns the open novelty question into a measurement: spcf on a purification against DMT on an MPDO at identical parameter count. Finally, the 3-16x overhead caps spcf's value at memory unless the χ reduction at equal error exceeds about 1.4-2.5. A Kronecker-factored, whitened closed form is the most likely way past that cap, and it should be built before arguing for wall-clock advantage.
