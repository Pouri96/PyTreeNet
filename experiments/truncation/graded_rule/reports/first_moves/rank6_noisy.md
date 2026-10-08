# Rank 6 first move: spcf on LPDOs for noisy circuits and Lindblad dynamics

## Summary verdict: conditional, and weak as a standalone direction

The core mechanism transfers cleanly. In a locally purified density operator (LPDO), ρ = XX†, so the physical reduced density matrices (RDMs) are quadratic in the site tensors. The bond truncation of an LPDO is therefore the purification cut from rank 2 with a per-site ancilla dimension K_j, called the Kraus dimension. A tilted cut keeps positivity by construction. To first order it also keeps the Werner et al. trace-norm certificate, ‖ρ − σ‖₁ ≤ √2‖X − Y‖₂, because spcf leaves the Frobenius error of the purification unchanged to first order.

No prior art was found that truncates an LPDO, or any positive tensor-network density operator, to preserve local physical observables. Every LPDO truncation found is Frobenius-optimal on the purification, an entanglement-entropy criterion, or a global Hilbert-Schmidt fidelity.

Three findings from the full texts limit the value of this direction:
1. **The real LPDO pain points are elsewhere.** They are Kraus-dimension growth and a fidelity collapse at the quantum-to-classical crossover. In Guo-Yang this happens at depth d ≈ 0.17/ε. Müller-Ayral-Bertrand attribute it to poor compression rather than to the ansatz.
2. **The proposed "local positivity" niche for matrix-product operators is already covered by DMT.** DMT preserves every 3-site RDM at a cut exactly.
3. **The strongest competitor for local observables in open systems is DMT on a plain MPO, not SVD-LPDO.** SVD-LPDO is a weak baseline.

**First move.** Reuse rank 2's `spcfpur` d = d_p·d_a cut with a per-site Kraus dimension. Add a small numpy LPDO-TEBD driver with dephasing Lindblad noise and a dense Lindblad reference. Then run a logging-only audit that asks whether SVD's repairable cross-cut residual survives γ > 0.

This direction should run **after** rank 3 (trajectories, γ scan) and rank 2 (d = 4 cut). Both are prerequisites, and either can kill this direction early.

## Literature, verified from full text

All texts were fetched with `read_arxiv_paper` into `/tmp/papers_rank6_noisy`. Text dumps are in `/tmp/papers_rank6_noisy/txt/`. Large results were saved by the tool and parsed with `jq`; no fetch failed.

**Cheng, Cao, Zhang, Liu, Hou, Xu, Zeng, [arXiv:2004.02388](https://arxiv.org/abs/2004.02388) (PRR 3, 023005). Full text read.**
- *Naming.* Their "MPDO" is a locally purified form: M = Σ_a T^{s,a} (T^{s',a})*, with an "inner" (Kraus) index a (Eq. 3.2, Fig. 1). The report's line "on an MPDO the targets are linear … on an LPDO quadratic" mixes two naming conventions. In Cheng, Werner and Müller, "MPDO" means LPDO. The *linear*-target object is the plain or vectorized MPO used by DMT and rTEBD.
- *Truncation criteria, which the report asked to check.* The paper's "optimal truncations" are:
  - first, a **local SVD of each T over the inner index**, keeping κ, with no environment (Eq. 3.5–3.6);
  - then a left-to-right QR to canonical form and a right-to-left **bond SVD**, which they call "globally optimal" (Eq. 3.7–3.9).

  Both are Frobenius-optimal for the purification only. Nothing targets local observables.
- *Benchmarks.*
  - 1D random circuits (random single-qubit gates, CNOT or CZ) at N = 10 and depth D = 24, and at N = 15 and D = 24 for the Porter-Thomas test.
  - Noise on every two-qubit gate: dephasing, depolarizing and amplitude damping.
  - The main runs use χ = 32 and κ = 48. Sec. V uses κ = 2χ.
  - The metrics are global: Uhlmann fidelity F(ρ_e, ρ_d), the cumulative output distribution, and cross entropy against ibmq_16_melbourne (Table II gives CNOT errors of 0.017–0.051).
  - Sec. V finds that above a gate error of about 0.01, a small χ suffices (Fig. 7), and that with weak noise a small κ suffices (Fig. 8).
- The quote "MPS fails to approximate the noisy output … for any of the noise models" is correct (abstract, Fig. 3, Table I). It refers to bond-truncated pure MPS used as a proxy for noise, not to trajectories.

**Werner, Jaschke, Silvi, Kliesch, Calarco, Eisert, Montangero, [arXiv:1412.5746](https://arxiv.org/abs/1412.5746) (PRL 116, 237201). Full text read.**
- *Representation.* ρ = XX†, with tensors A^{s,r}_{m,m'} of physical dimension d, bond D and Kraus K (Eq. 3).
- *Evolution.* TEBD with a five-layer Strang split, e^{τHo/2} e^{τHe/2} e^{τD} e^{τHe/2} e^{τHo/2} (Eq. 5). Coherent layers act on X only. Dissipative layers contract Kraus operators B_{l,q} into the Kraus leg, joining K with the channel rank k ≤ d².
- *Truncation.* SVD compression of the bond and Kraus dimensions separately (Fig. 1c). The Frobenius error of X bounds the trace norm (Lemma 1, ‖ρ−σ‖₁ ≤ √2‖X−Y‖₂; Theorem 1, App. D). Cost is O(d⁵D³K) + O(d⁵D²K²).
- *Benchmarks.*
  - Two cavity-qubit pairs at D = K = 40, scored by infidelity and relative Hilbert-Schmidt distance.
  - XXZ steady state with edge driving at N = 100 and D, K ≈ 60, scored by σᶻ_j and the spin current.
  - Kitaev wire with two-site Lindblad operators at N = 6 and K = D = 30 (Fig. 4). The discarded weight serves as an upper-bound estimator of infidelity.
- No observable-aware truncation appears.

**Guo & Yang, [arXiv:2312.02854](https://arxiv.org/abs/2312.02854) (= CPL 41, 120302). Full text and supplement read.**
- *Positivity claim verified.* For truncated MPOs, the trace norm ‖ρ‖₁ exceeds 1 (Fig. 2c). The "negativity" peaks precede the crossover (Fig. S6b, critical Ising, ε = 0.001).
- *A finding the report missed: LPDO fails at the quantum-classical crossover.*
  - Setup: brickwall Haar circuits with two-qubit depolarizing noise, N = 8 and 16, ε = 0.01–0.03, compared at χ = 16, d_κ = 2 against MPO with D = 64–256.
  - The LPDO fidelity drops below 0.95 at d_Fid ≈ 0.171/ε^1.06 (Fig. 2d). This is universal across random gates and Trotterized Ising and Heisenberg (Fig. S5).
  - The authors read it as a representational limit: "a crucial trade-off between the accuracy of observables and the positivity of output states", which "cannot be alleviated only by increasing d_κ" (Fig. S4).
- *Their truncation (Supplement, "The truncation method of LPDO").* Gauges L_j and R_j are obtained by QR and LQ sweeps. Then Frobenius-SVD projectors are built for the Kraus legs (absorbing the environment gauges) and for the bonds, and all projectors are applied at once.
  - They explicitly criticize the Cheng and Werner sequence as "neglect of the environment when truncating d_κ".
  - Projection fidelity is evaluated by gradient descent on ‖ρ − ρ′‖²_F (Eq. 4).
  - None of this is local.

**Müller, Ayral, Bertrand, [arXiv:2403.00152](https://arxiv.org/abs/2403.00152). Full text read.**
- *Method (IPD).* DMRG-like sweeps of two-site ancilla-basis unitaries that minimize bond entanglement of the purified state, followed by SVD truncation. This also shrinks the purification (Kraus) dimension.
- *Benchmarks.*
  - N = 6 random MPDO with χ = 12, r = 8 and ε = 10⁻³ (Fig. 5).
  - N = 8 random circuits with depolarizing p = 2–10% (Figs. 7–8).
  - Fixed χ = 32 and r = 2, p = 5% (Fig. 9).
  - Metric: the "pseudo-fidelity" Tr(ρ†σ)/√(Tr ρ² Tr σ²).
- *Rebuttal of Guo-Yang.* "the barrier observed in the data of Ref. [43] is caused by the MPDO compression they perform"; the fidelity minimum (depth 10–14) is not co-located with the operator-entanglement maximum (depth ≈ 3).
- **Their Fig. 7 shows the purification entropy peaking at depths 6–10, well after the operator entanglement (OEE) peaks.** The purified state therefore keeps a growth phase after the MPO's has ended. This is spcf's regime, and it argues for a longer usable window than the report assumed.
- Their ref. [26] (Surace et al.) uses MPDOs "as an approximation for pure states, conserving local correlations only". This is the closest local-RDM idea in the positive-TN literature; see the next section.

**Wei, Rajakumar, Nelson, Malz, Gullans, Gorshkov, [arXiv:2603.20400](https://arxiv.org/abs/2603.20400). Full text read.**
- *Object.* A plain MPO, vectorized and SVD-truncated with a threshold δ_err = 10⁻⁶ (Eq. 14–16), not an LPDO.
- *Setups.*
  - Haar brickwall circuits with single-qubit depolarizing or amplitude-damping noise.
  - Trotterized Lindblad dynamics of H = −ΣZZ − gΣX − hΣZ:
    - (g, h, κ_dep) = (1, 1, 0.04);
    - (g, h, κ_damp) = (8, 1, 0.4), with a Strang split of noise and unitary (Eq. 10–11).
- *Result.* The single-step L₂ error contracts as 2^{−Γ_p N Δt} (Eq. 32). Purity relaxes on T_s ∼ 1/p. An empirical L₁ bound follows.
- **Correction to the report.** The report says "Per-step local bias is therefore neither compounded nor amplified". That is an over-reading. The contraction is in the *global* Hilbert-Schmidt norm, and the exponential-in-N part comes largely from ‖ρ‖₂ ∼ 2^{−λN} itself. The paper measures no local observable (no hit for "observable" in the text). Local errors relax only at the local dissipation rate, roughly κ.

**Noh, Jiang, Fefferman, [arXiv:2003.13163](https://arxiv.org/abs/2003.13163) (Quantum 4, 318). Read in part (setup and truncation).**
- Vidal-form MPO with plain SVD.
- Two-qubit depolarizing noise at p = 0.06–0.15, n up to 32.
- Tr ρ̂ is used as the truncation-error proxy, and N_s = 24 circuit realizations are averaged.
- Noise is strong, so the growth phase lasts only a few layers, which is a poor regime for spcf.

**Other texts read for the novelty check.**
- Godinez-Ramirez, Milbradt, Mendl, [arXiv:2409.08127](https://arxiv.org/abs/2409.08127). Riemannian trust-region optimization of the Kraus-gauge isometries of the *channel*, which reduces splitting error and compresses Choi rank. It does not target state observables.
- Mc Keever & Szymańska, [arXiv:2012.12233](https://arxiv.org/abs/2012.12233). iPEPO with Full Environment Truncation that maximizes Hilbert-Schmidt overlap. Global, and it is not an LPDO.
- Wille et al., [arXiv:2609.20802](https://arxiv.org/abs/2609.20802), Sept 2026, "LOMPS". A *pure-state* variational evolution that replaces fidelity with a Hilbert-Schmidt cost on local L-site RDMs. It is closed-system and uniform MPS. **It is not rank-6 prior art, but it is new prior art for the whole spcf programme** and should be cited in the core write-up.

**Abstract only.**
- Jaschke-Montangero-Carr, [1804.09796](https://arxiv.org/abs/1804.09796): review comparing trajectories, MPDO and LPTN.
- Surace-Piani-Tagliacozzo, [1810.01231](https://arxiv.org/abs/1810.01231), and Frías-Pérez-Tagliacozzo-Bañuls, [2308.04291](https://arxiv.org/abs/2308.04291): entanglement converted to mixture while preserving local RDMs.
- Cichy et al., "locally entanglement-optimal unravelings" (PRA 2026, Google Scholar snippet): relevant to rank 3.

## Prior art and novelty risk

**Risk: low for "observable-aware LPDO truncation".** These works cover the full LPDO and MPDO truncation literature found:
- Werner 2014 (SVD);
- Cheng 2020 (local Kraus SVD plus canonical bond SVD);
- Guo-Yang 2023 (environment-gauged Frobenius projectors);
- Müller 2024 (entropy-minimizing disentangling, then SVD);
- Godinez-Ramirez 2024 (channel compression);
- Mc Keever 2020 (iPEPO, HS fidelity).

None uses a local-observable criterion. Searches also turned up nothing: arXiv "truncation preserving local observables open quantum", "positivity preserving density matrix truncation", "Kraus dimension truncation"; Google Scholar "locally purified density operator truncation preserving local observables"; Semantic Scholar and OpenAlex returned empty results.

**Adjacent ideas that a reviewer will raise.**
- *DMT on the plain MPO* preserves ≤ 3-site RDMs exactly and already handles dissipative transport. It gives up positivity. The honest pitch is "positivity-preserving, approximate counterpart of DMT". It is not "better than DMT".
- *Surace et al. and Frías-Pérez et al.* convert discarded entanglement into mixture while holding local RDMs fixed. In an LPDO this has a natural home: discarded bond weight could be pushed into the Kraus leg instead of being deleted. That is a different, potentially stronger construction (an exact local-RDM-preserving move inside a positive ansatz). Keep it as an open idea and a competitor, not as part of the first test.
- *LOMPS (2609.20802)* uses a local-RDM cost on pure states. It is variational, global and closed-system, with no per-cut fixed-χ tilt.

**The proposed "local positivity" niche is weak and should be dropped.**
- DMT at bond j leaves ρ_{1…j+1} and ρ_{j…L} unchanged (rank-2 report, 1707.01506 Sec. III). Every 3-site region touching the cut lies inside one of them, so DMT already inherits 3-site positivity exactly.
- A 3-site negativity penalty therefore competes only with SVD-MPO, where it would be a weaker DMT.
- On an LPDO, positivity is automatic.
- What remains is either a penalty on ≥ 4-site RDMs spanning the cut, or a "DMT plus local PSD repair" step. Neither uses spcf's machinery.

## Technical details for implementation

**Object and cut.**
- LPDO site tensor A_j[l, p, k, r], with p = 2 and Kraus dimension k = K_j that varies by site. Fuse (p, k) as rank 2 fuses (p, a).
- Bond cut: θ = A_b A_{b+1} with the gate applied to the physical legs only, Mm = θ.reshape(l·2K_b, 2K_{b+1}·r), and the kept rank set to χ.
- Region map W: physical legs of the window to rows (D = 2^L, so `_region`, F, `hvec` and the gate statistic are unchanged); all Kraus legs plus the outer bonds to columns. Then ρ_region = WW†.
- To keep W small, LQ-compress the left environment's (o, K_lo…K_{b−1}) column index to rank ≤ 2^{aL}·l. Compress the right environment the same way. This is exact, because only WW† matters.
- At χ = 16, K = 4 and a = 1, W is about 16 × 16k complex64, under 1 MB. Use a = 1 first.

**Kraus truncation.**
- After a noise layer, K_j → K_j·k_ch. Truncate to κ at the orthogonality center by SVD of A_j reshaped to ((l p r), K) (Cheng; Werner).
- The Kraus leg is traced, so the local physical RDMs are **linear in the kept projector P = QQ†**. One Gauss-Newton step is therefore a plain linear least-squares solve in C (Q = qr(V_k + V_⊥C)).
- This is a second, cheap tilt site, "spcf-K", and the bond tilt does not cover it.
- Linearity in P also suggests a DMT-like exact variant (reserved Kraus directions). That is an open question, not a first test.

**Lookahead rows.**
- spcf's F rows are U^†PU for τ ∈ `taus`. With dissipation the exact analogue is e^{τ𝓛†}(P), the adjoint region Lindbladian, which is still linear in ρ.
- For the first test, use `taus=[1.0]` with the Hamiltonian region propagator (an approximation) and `taus=[]` (static only) as the control. Add Lindblad-adjoint rows only if Test 2 is borderline.

**Model and noise.**
- Repo `ising` (ZZ + 0.9045X + 0.8090Z; DMT's model ×4) with Néel start, because that is where spcf wins 2.5–5× at γ = 0.
- **Dephasing** L = √γ Z per site: Kraus rank 2, the cheapest, and the same noise rank 3 uses. Depolarizing (rank 4) comes second.
- Strang: noise e^{dt·D/2} on each site as the right and left sweeps pass it, with the identical sequence in the reference. Then only truncation error is measured, as in `mps_bench.py`.
- γ ∈ {0, 0.003, 0.01, 0.03, 0.1} in repo units. Wei et al.'s κ_dep = 0.04 with J = 1 is about γ = 0.04–0.16 in repo units, so this range brackets it.
- dt = 0.1, T = 3–6 (spcf's growth-phase window at γ = 0).
- **Note:** Guo-Yang's spin-model checks used critical TFIM and Heisenberg, both integrable, where spcf is expected to fail. Do not copy those models.

**Reference.**
- A dense vectorized ρ of 4^N entries (N = 8 is 1 MB; N = 10 is 16 MB), evolved by superoperator gates G⊗G* and local channels. Numpy only.
- Physical RDMs: copy `rho_obs` from `dmt_baseline.py` (do not import it; it imports `gcg`).
- Full trace distance ‖ρ−ρ_ex‖₁ by `eigh` of a 2^N matrix (trivial at N ≤ 10).

**Metrics.**
- single, nn and nnn rms; |ΔE|; far collateral ⟨Z_iZ_{i+3}⟩ and ⟨Z_iZ_{i+4}⟩.
- Trace distance.
- Purification Frobenius error (the certificate).
- Gate on-fraction and the logged (tail, B_res, cost) triples from `gate_log`.

**Parameters.**
- N = 8, with N = 10 as a check.
- χ ∈ {4, 6, 8, 12, 16}; κ ∈ {2, 4, 8}.
- The exact bond at N = 8 is larger than for an MPS because of the Kraus legs, so pick T from a pressure scan in which SVD-LPDO error is between 10⁻³ and 10⁻¹.

**Known failure modes.**
- Gain decays with γ, as rank 3 predicts.
- Kraus truncation, not bond truncation, may dominate the local error. The bond tilt cannot touch that.
- SVD-LPDO is a weak baseline (Guo-Yang and Müller both improve on it), so a win must also be checked against the Guo-Yang environment-gauged truncation.
- Non-unital noise (amplitude damping) drives the state to a pure product state, so it is a late-time trivial regime.

**Environment caveat.** This container has numpy but **no scipy**. The repo's runs were made on a Windows venv (`run_spcf_sweep16.sh`). `spcfast._region` and `mpsenh.make_gates` need scipy (`scipy.sparse`, `scipy.linalg.expm`), so run the tests on the laptop venv.

## Where to start (code changes and effort)

**Prerequisites.**
- Rank 2's `rule/spcfpur.py`: the d = d_p·d_a cut, about 1 day.
- Rank 3's γ scan: no new cut code.

**Changes specific to rank 6.**
1. **`rule/spcflpdo.py`.** Subclass or copy `spcfpur`. About 0.5 day.
   - Per-site ancilla dimension list `K[j]` instead of a scalar `da`.
   - LQ compression of the environment maps in `_maps`.
   - A d-generic `_plain`.
   - Invariants to test:
     - K ≡ 1 reproduces `spcfast` bit for bit;
     - K ≡ 2 reproduces `spcfpur`;
     - `kappa`/`fw` behave as before.
   - Port `test_spcfast.py` sections 2–4 (adjoint, finite-difference linearization) as `test_spcflpdo.py`.
2. **`lpdo_bench.py`: the driver.** About 1 day.
   - LPDO Néel product constructor (K = 1).
   - `run_tebd_lpdo`: a copy of `mpsenh.run_tebd` with gates on the physical legs and noise applied at the orthogonality center during each sweep.
   - Kraus truncation by `svd` or `env` (Guo-Yang-style, using the canonical form already present at the center).
   - Dense Lindblad reference cached in `_refcache/lindblad_*`.
   - Metrics as above; JSON rows compatible with `pareto_spcf.py`.
   - Arms: `svd`, `spcf:<opts>` (bond tilt), `spcfK` (Kraus tilt, step 3), and `log` (`gate_log=True`, `gate=1e9`, which records but never fires).
3. **`rule/kraustilt.py`.** About 0.5–1 day, only if Test 1 says Kraus truncation dominates. One linear least-squares Gauss-Newton step on the kept Kraus subspace, against window physical RDMs.
4. **Deferred.** Numpy DMT-MPO from rank 2 (`dmt_np.py`) with channel superoperators in the Pauli basis, for Test 3.

**Effort to the first clue:** about 1.5 days on top of rank 2 (about 2.5 days standalone). **To the confirmation:** about 2.5 days. **To the head-to-head:** about 3.5 days.

## First tests: a decision tree

**Test 0: correctness gates (minutes).**
- `test_spcflpdo.py`: adjoint relative error < 10⁻⁵ and linearization error linear in the step.
- K ≡ 1 run is bit-identical to `mps_bench` spcf on `ising` N = 8.
- γ = 0 LPDO-SVD equals MPS-SVD.
- The dense Lindblad reference at γ = 0 equals `mpsenh.dense_reference` (|ψ⟩⟨ψ|).

Any failure stops the tree.

**Test 1: first clue, an audit with no tilt (about 20–40 min of CPU).**
- *Setup.* LPDO-SVD on `ising`, N = 8, T = 4, dt = 0.1. Grid: γ ∈ {0, 0.003, 0.01, 0.03, 0.1}, χ ∈ {6, 8, 12}, κ ∈ {2, 4}.
- *Three runs per cell:*
  - (a) both truncations;
  - (b) κ = ∞ (bond truncation only);
  - (c) χ = ∞ (Kraus truncation only; affordable at N = 8).
- *Logging.* The spcf `gate_log` on every bond cut.
- *Metrics.*
  - Error budget: nn rms of (b) and (c) relative to (a).
  - Gate on-fraction: the fraction of cuts with B_res > c₀(tail/10⁻⁴)^0.65.
  - Median B_res/cost.
  - All tracked against γ.
- *Success.* Two conditions together:
  - at γ = 0.01, bond truncation carries ≥ 50% of the local error;
  - the gate on-fraction is ≥ 50% of its γ = 0 value, at at least two χ values where SVD's nn rms is ≥ 10⁻³.
- *Kill, either of:*
  - the on-fraction at γ = 0.01 is < 25% of its γ = 0 value at every χ, with no repairable residual left;
  - Kraus truncation carries > 80% of the error at all γ ≥ 0.01, *and* a one-cell Kraus-tilt probe (step 3) gains < 1.3×.
- *Ambiguous.* The residual survives only at γ = 0.003. That is essentially the pure regime, so ranks 2 and 3 subsume it; pause rank 6.
- *Decision.* On a bond-dominated success, go to Test 2a. On a Kraus-dominated result, build `kraustilt.py` and go to Test 2b.

**Test 2: confirmation at matched (χ, κ) (about 30–60 min of CPU).**
- *Arms.*
  - SVD-LPDO, in Cheng order and in Guo-Yang env-gauged order.
  - spcf bond tilt (a = 1, fw = 0, iters = 4, taus = [1.0], ks = 1-2-3).
  - Gated spcf.
  - (2b) Kraus tilt, and bond plus Kraus tilt.
- *Cells.* The two best (γ, χ) cells from Test 1, plus γ = 0 as the anchor.
- *Metrics.*
  - nn, nnn and single rms; |ΔE|.
  - Far ZZ at distance 3–4.
  - Trace distance and purification Frobenius error.
  - Wall time, for information only.
- *Success, both of:*
  - nn rms ≥ 2× lower than the best SVD-LPDO ordering at two adjacent χ in a γ ≥ 0.01 cell, with trace distance no more than 10% worse and far ZZ rms no more than 1.5× worse;
  - the γ = 0.01 gain is at least half the γ = 0 gain.
- *Kill, either of:*
  - gain < 1.3× in every γ ≥ 0.01 cell;
  - trace distance > 25% worse wherever the gain is ≥ 1.3×.
- *Ambiguous.* The gain holds against Cheng-order SVD but not against env-gauged SVD. In that case the "gain" is the missing environment, not the tilt. Re-run spcf on top of the env-gauged Kraus truncation before deciding.

**Test 3: positioning, only after Test 2 succeeds (about 1 h).**
- *(i) Against DMT-MPO.* Numpy DMT on a plain MPO at equal stored parameters (LPDO 2Kχ² per site vs MPO 4D² per site), on the same Lindblad cell. Also record λ_min(ρ) and the minimum eigenvalue of 3- and 4-site RDMs for DMT and SVD-MPO; this measurement settles the positivity niche.
- *(ii) The Guo-Yang barrier.* Brickwall Haar plus two-qubit depolarizing at N = 8, ε ∈ {0.01, 0.03}, χ = 16 and κ = 2. Track the depth at which F(ρ_ex, ρ_LPDO) < 0.95, which they report as ≈ 0.17/ε, and the nn rms along the way.
- *Success.* Either:
  - spcf-LPDO ties or beats DMT-MPO on nn rms at equal parameters while staying PSD;
  - or spcf delays the Guo-Yang barrier by ≥ 20% in depth.
- *Kill.* DMT-MPO is ≥ 2× better on nn rms at equal parameters and its measured λ_min is negligible (|λ_min| < 10⁻³). Positivity then buys nothing for local observables, so drop the direction.

## Open questions

1. Will the purification-entropy growth phase that Müller et al. see after the OEE peak (Fig. 7, depths 6–10 vs ≈ 3) also appear for Lindblad Ising? If so, the spcf window is longer for LPDOs than for MPOs, which would help this direction.
2. Kraus-leg targets are linear in the kept projector. Is there an exact, DMT-like reserved-Kraus construction that preserves window RDMs at the cost of a few Kraus directions? If so, that, rather than spcf, may be the natural rank-6 method.
3. The entanglement-to-mixture move (Surace; Frías-Pérez) can be implemented inside an LPDO by moving discarded bond weight into the Kraus leg. It could beat both SVD and spcf for local observables and should be a competitor in Test 3.
4. Do the Guo-Yang crossover failures come from the compression, as Müller claims? If so, does a local-observable criterion address them, or only entanglement-optimal gauging (IPD)? Test 3(ii) is designed to answer this.
5. Is the Lindblad-adjoint lookahead (e^{τ𝓛†}P rows) worth adding, or is the static target enough? It only matters if Test 2 is borderline.
6. Equal-parameter accounting: an LPDO stores 2Kχ² per site against 4D² for an MPO. Which (K, χ) pairing is fair for the DMT comparison? At K = 2, χ = D.
7. Can the GCG library behind `dmt_baseline.py` apply non-unitary channels in its Pauli frame, so that it serves as an external DMT check? It is not installed here, so this is unknown.
