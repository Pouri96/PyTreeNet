# Rank 7 first moves: 1D quench benchmarking and verification windows

## Summary verdict: weak-to-conditional (memory yes, wall-clock marginal)

The existing ladders already measure the χ reduction r, so the report's proposed first experiment is mostly done. The report gave **2.5-5x lower local error at equal χ**. Converted to r by interpolating the SVD ladders on the same metric, that is only **r ≈ 1.5-1.9**. At N = 20, T = 8 the value is roughly constant, **r ≈ 1.6-1.7**, over χ = 24-64. The reason is that SVD error falls steeply with χ (local log-log slope p ≈ 1-4), so an error ratio R buys only r = R^(1/p).

That conversion gives two numbers:
- Memory saving at equal local error is r² ≈ 2.5-3x. This is real and cheap to claim.
- Wall-clock saving needs the overhead to be below r³ ≈ 4-5.

The overhead does not fall with χ. A sanity check timed one fired spcf cut at 14-19x the cost of one SVD, flat from χ = 16 to 128, because both scale as χ³. The only lever is the fraction f of cuts the gate fires on: equal-error wall-clock is ≈ (1 + 15f)/r³. Measured equal-error CPU ratios are 3.0-8.6x *against* spcf at N = 20, and 1.2-17x across all cells. Break-even needs f ≲ 0.2 while r stays ≈ 1.6. At N = 20, χ = 32 we have f = 0.25 and r = 1.68, which is close.

For benchmarking, a constant factor r buys only an *additive* shift in log χ_Q. D-Wave's χ_Q grows exponentially in bipartition area (2403.00910, Fig. 4B), so r = 1.6 extends a verification window by a fixed amount, not by a scaling.

The scientifically interesting angle is different, and cheap. spcf separates local-correlator error from fidelity. D-Wave argues that a correlator-based χ_Q tracks the fidelity-based one (SM Fig. S19), and the fidelity-based extrapolations used to verify ZNE (Anand et al.; Mandrà et al.) assume the same coupling.

## Literature, verified from full text

PDFs and extracted text are in `/tmp/papers_rank7_benchmarking` (`txt/` holds the plain text).

| arXiv | Read | Verified content | Report claim: status |
|---|---|---|---|
| 2306.14887 (Tindall, Fishman, Stoudenmire, Sels, PRX Quantum 5, 010308) | Full text incl. App. A/B | The kicked Ising U(θh) = ∏exp(iπ/4 ZZ)∏exp(−iθh X/2) is on the **2D heavy-hex** lattice. Their own MPS (App. B) uses χ = 2500, a snaked 1D ordering, gates as ≤5 commuting MPOs of bond dimension 2, light-cone depth reduction, and an error estimate E_n = 1 − Π(1 − ε_i)^{1/N} (Eq. B1). Fig. 4f text: "the BP results agree with the new MPS results until about step 10 where the MPS error starts to grow" (θh = 1.0, ⟨Z62⟩). BP-TNS results are extrapolated linearly in 1/χ (Fig. 4c). | "MPS error grows from about step 10": **correct for θh = 1.0 only**, and for a 2D snake MPS with long-range MPO gates. It is **not** 1D nearest-neighbour TEBD, so spcf's ±2-site window does not map onto that truncation directly. |
| 2306.17839 (Anand, Temme, Kandala, Zaletel) | Full text | Heisenberg-MPO (vectorized operator, d = 4) on the heavy-hex snake: 13 commuting layers, each followed by two-site variational compression, cost DNχ³. Weight-10 observable exact at χ = 384; weight-17 accurate to < 1e-4. MPO fidelity F_D = Πf_t, with f_t = \|⟨O_t\|Õ_t⟩\|². Pure-state MPS ⟨Z62⟩ (D = 20) is extrapolated **linearly in log F_D** from the three largest χ (App. A4, Fig. 10). MPO results are "often non-monotonic in χ", which makes extrapolation unreliable (Fig. 11). Classical methods disagree by ~20% near θh ≈ π/4. At the Clifford point θh = π/2 the pure-state entanglement spectrum is exactly flat with rank 2^D, so "any truncation … will immediately lead to noticeable errors". OTOC support stays well inside the light cone (v_B < 1). | Report had title only. **New and important**: the verification consumers are (i) Heisenberg MPO (rank 1 territory, not rank 7) and (ii) fidelity-based extrapolation, which spcf decouples from local error. |
| 2503.05693 (Tindall, Mello, Fishman, Stoudenmire, Sels; v4, May 2026) | Main text + SM headings | BP-TNS with χ_BP = 32 during evolution, then truncation to χ and cylindrical MPS message passing (R = 2χ). Error metric ε_c = sqrt(Σ_{i>j}(c_ij − c̃_ij)²/Σ c̃_ij²) over **all** pairwise ⟨σ^zσ^z⟩ (Eq. 2). Ground truth comes from converged MPS/TDVP. | **Misattribution.** The quote "applies to MPS — the only method with which we can match QPU quality…" is **not** in 2503.05693. It is from the D-Wave paper itself (2403.00910, p. 7). |
| 2403.00910 (King et al., D-Wave, "Beyond-classical computation in quantum simulation") | Main text, SM §III.B.5, §VI | Defines the **QPU-equivalent bond dimension χ_Q**: the MPS χ whose ε_c matches the QPU (≈ 64 for one 6×6 example). χ_Q grows exponentially in the MPS bipartition area (Fig. 4B). log χ_Q is linear in S_max. Resources are extrapolated as Nχ_Q² memory and Nχ_Q³ time (Fig. 5D). SM: "χ_Q—derived from pairwise correlation errors—is closely linked to the more expressive fidelity measure" (Fig. S19). SM 1D section: the non-canonical iMPS truncation minimizes a cluster Frobenius error on 2 + 2M sites. Ground truth uses TDVP with χ grown by √2 steps. | This is the paper the report meant. The **χ_Q framework is exactly "r at equal error"**: spcf would lower χ_Q(ε_c) by r but not χ_Q(F). |
| 2310.08567 (Mitra, Albash, …, Deutsch, NJP 2025) | Full Secs. IV-VI | 1D TFIM and slanted-field Ising (θ = π/4), TEBD/TDVP, χ = 8-512, n = 50-81. "local expectation values are most easily approximated for chaotic systems whose exact many-body state is most intractable." Local RDMs of chaotic quenches approach maximally mixed, more so for zero-energy initial states (\|↑y⟩). | New context. In late-time chaotic regimes SVD's local error is *already* small, so spcf's window is the growth phase. This is consistent with the repo's gate. |
| 2609.20802 (Wille, …, Green, LOMPS, Sep 2026) | Full text | uMPS time evolution in which each Trotter step is projected by minimizing ½‖ρ_L[A′] − ρ_L^evol‖²_HS over an L-site patch. It has TDVP-like equations of motion and runs on IBM Heron and Quantinuum H2. D ≈ 2^L, with patch sizes L = 2-6. Framed explicitly as "intensive quantum advantage" for local observables. **There is no equal-D comparison against SVD-TEBD.** | **Highest novelty risk found** (see next section). |
| 2511.23438 (Mandrà, …, Kechedzhi, PR 2026) | Method sections | 2D TFIM 7×8 at χ = 4096 on one GPU. Heuristic: ⟨O⟩ ≈ F_MPS·⟨O⟩_MPS (Eq. 6), generalized to F^γ. F_MPS is the product of per-gate truncation fidelities. Richardson / zero-truncation extrapolation "fails" for their case. | New **zero-overhead competitor** for local observables at low χ. It must be a baseline. |
| 2002.07730 (Zhou, Stoudenmire, Waintal) | Abstract only | Per-gate fidelity model of MPS circuit simulation. | Source of the F_D estimate used by Anand. |
| 2610.02082 (Grassi, …, Tagliacozzo) | Snippet only | 1D brick-wall circuits via reduced transition matrices. | Possible 1D local-observable competitor; not read. |

**Corrections to the report.**
1. The D-Wave quote belongs to 2403.00910, not 2503.05693.
2. The IBM "step 10" statement is θh-specific and comes from a 2D snake MPS with χ = 2500, not 1D TEBD.
3. The benchmarks the report calls "1D-relevant" are all 2D. I found no 1D quantum-advantage claim that rests on MPS cost. In 1D, MPS is the ground truth itself, so the realistic 1D consumer is **ZNE/verification-window benchmarking on 1D chains**, not refuting an advantage claim.
4. "Novelty risk: low" is now **moderate** because of LOMPS and the fidelity-rescaling line.

## Prior art and novelty risk

- **LOMPS (2609.20802) shares the thesis**: trade global fidelity for local-RDM accuracy, as a route to local-observable advantage. It differs in four ways:
  - it is translation-invariant uMPS, not finite inhomogeneous TEBD;
  - it fully minimizes an HS cost at each step, not one Gauss-Newton tilt seeded from the SVD optimum;
  - it gives up fidelity entirely, whereas spcf keeps fidelity to first order;
  - it never measures r against SVD at equal D.

  Our clean differentiator is therefore the measured χ-reduction at fixed χ and fidelity, with a gate. LOMPS must be cited, and an "L = 6 HS-projection" arm is the natural ablation of our one-step tilt.
- **Fidelity-based post-processing** (Anand's log F extrapolation, Mandrà's F-rescaling, Tindall's 1/χ extrapolation) addresses the same deliverable, local observables beyond the converged χ, at zero per-cut cost. If SVD plus one of these matches spcf's local error, rank 7 is dead. These methods also assume the discarded component behaves like noise ("⟨ψ⊥\|O\|ψ⊥⟩ ≈ 0"). spcf deliberately correlates what it discards with local observables, so **spcf + rescaling may fail to compose**, and that has to be tested.
- **Already known, but not as truncation rules**: DMT, rTEBD and Surace-Piani-Tagliacozzo, covered in the main report. D-Wave's 1D cluster-Frobenius truncation is environment-weighted fidelity, not RDM targeting.
- **No paper found reports a χ_Q- or r-type equal-error χ reduction from a modified truncation rule.** The "r" framing remains open.

## Technical details for implementation

- **Error metrics actually used.**
  - IBM/Anand: absolute error of single observables (⟨Z62⟩, weight-10/17 stabilizers) versus depth.
  - D-Wave: ε_c over all pairs i > j. This **includes long-range ZZ**, which is collateral for spcf. The repo's `errors()` only scores 1-site, nn and nnn Paulis. Earlier `out_sf8_*` results did score "pair d ≥ 4" and saw a ratio of about 0.49, but those used the old fit rule.
  - Mitra/LOMPS: trace distance of ℓ-site RDMs, and a deviation time t_dev at threshold 1e-3.
- **Verification horizon.** Define t*(χ, tol) as the first time the local error exceeds tol (tol = 1e-2 for ZNE-style comparisons, whose error bars are ~1e-2; 1e-3 for LOMPS-style comparisons). In the χ_Q language, r = χ_SVD(t*)/χ_spcf(t*).
- **Typical sizes.**
  - 2D benchmarks: χ = 500-4096 (pure state) and χ = 1024 (MPO).
  - 1D TEBD studies: χ = 8-512.
  - Our cells: χ = 4-192. The N = 20, T = 8 SVD ladder runs to χ = 192 (`results/high_pressure/matched_hp.json`), and its dense reference is cached (`_refcache/ising_N20_T8.0_dt0.1.npy`, 16.8 MB).
- **Known failure modes.**
  - Exactly flat spectra at Clifford points. A tilt among degenerate singular vectors is ill-posed, and `an_degen.py` is the relevant diagnostic.
  - Non-monotonic convergence in χ, which breaks extrapolation.
  - Chaotic late times, where SVD's local error is already small.
  - Heisenberg/integrable dynamics, where spcf already fails in this repo.
- **Kicked Ising in this code base.** `SPCFast._region` builds its lookahead rows from `M.h_bond(model)` (lines 92-117). A Floquet model therefore needs either `taus=()` (static targets only) or a Floquet unitary in place of exp(−iτH). Static-only targets kept most of the gain in the old fit rule: 1-site ratio 0.26 at lookahead 0 versus 0.21 at lookahead 1 (`out_sf8_la0/la1.txt`). `run_tebd` applies the same `Gs` in both half-sweeps, so the kicked circuit needs its own sweep. Use one sweep per Floquet step, with direction alternating each step:
  - right sweep: gate_b = ZZ_{b,b+1}(θJ)·RX_{b+1}(θh), with RX_0 folded into b = 0;
  - left sweep: mirrored.

  Every site is then kicked before both of its ZZ gates. The dense reference applies the identical gate list with `M.dense_gate`.

## What the existing results already say about r

`matched.py` uses the **old `EnhCut`**, not spcf. Its N = 20 output is `results/high_pressure/matched_hp.json`. The spcf ladders come from `pareto_spcf.py`. Below, r is the log-log interpolated SVD χ that reaches spcf's error, divided by spcf's χ. "CPU" is spcf's steady wall time divided by SVD's wall time at that interpolated χ. The scratch script used is `r_est.py` in the session scratchpad.

| Cell | χ (spcf) | r (nn / 1-site) | r (infid) | equal-error wall ratio | fired fraction |
|---|---|---|---|---|---|
| Ising N12 T4 | 6-12 | 1.4-1.6 | 1.02-1.06 | 5.5-13x | 15-36% |
| Ising N12 T4 | 16-24 | 1.1-1.2 | 1.0 | 1.6-2.9x | 3-7% |
| Ising N16 T5 | 8-16 | 1.6-1.9 | 1.04-1.10 | 3.4-10x | 18-32% |
| Ising N16 T5 | 32-48 | 1.1-1.3 | 1.0 | 1.6-2.5x | 3-7% |
| Ising N20 T8 (spcf) | 24, 32 | 1.69/1.58, 1.68/1.57 | 1.09, 1.07 | 7.6x, 3.0x | 33%, 25% |
| Ising N20 T8 (old rule) | 24-64 | 1.47-1.52 / 1.50-1.59 | ≈1.0 | 3.1-4.1x | — |

On N = 12 and N = 16 r shrinks toward 1 as χ approaches convergence. At N = 20 it holds roughly constant at about 1.5-1.7, while the SVD slope p rises from 0.97 to 2.4. **r is the more invariant quantity, not the error ratio.**

The tiny cost check (`cutcost.py` in the scratchpad, random two-site tensors) gave per fired cut **19x, 16x, 14x and 16x SVD at χ = 16, 32, 64 and 128**. The model of equal-error wall-clock is therefore (1 + ~15f)/r³.

## Where to start (code changes and effort)

1. **No code** (≈ 0 h): T1 below runs with the existing `pareto_spcf.py`, which already exposes `rel_skip` and `eps_min` in its OPT string, against the cached N = 20 reference.
2. **`ladder_t.py`** (new, ≈ 2-3 h). It copies the step loop of `M.run_tebd` and adds three things:
   - every k steps it records `M.local_obs` of `M.mps_to_dense(T)` and the dense reference;
   - it records all-pairs ⟨Z_iZ_j⟩ for ε_c;
   - it records a per-cut truncation fidelity through a wrapper `fid_wrap(cut)` that contracts the returned (A, B) and takes \|⟨θ\|θ′⟩\|²/‖θ‖², so F_MPS is available for both arms.

   Post-processing then gives Mandrà rescaling, log F extrapolation and 1/χ extrapolation for free.
3. **Kicked model** (≈ 3-4 h, only if T1 and T2 pass):
   - `make_kicked_gates(N, θJ, θh, direction)` and a `run_floquet` loop, in a new `rule/floquet.py` so `mpsenh.py` stays untouched;
   - a placeholder `h_bond` entry, or `SPCFast(..., taus=())`, so `_region` does not need a Hamiltonian.

## First tests (decision tree)

**T0, done here (zero cost).** r ≈ 1.6 at N = 20 and the overhead model above. → Go to T1.

**T1: can the gate make spcf pay in wall-clock? (≈ 10-15 min, existing code)**
- *Setup*: Ising N = 20, T = 8, χ ∈ {32, 48}. Run `pareto_spcf.py ising 20 8.0 32 32,48 out.json 1 '2:0:4:1.0:1-2-3:0:1:1e-7:all:<rs>'` with rs ∈ {1e-2, 0.1, 0.3, 0.6}, and eps_min ∈ {1e-7, 1e-5} at rs = 0.3. Score r against the χ ≤ 192 SVD ladder in `matched_hp.json`, since it is the same cell (nn errors agree to 3 digits at χ = 24).
- *Metrics*: r(nn), r(1-site), fired fraction f, steady CPU, and the predicted versus measured (1 + 15f)/r³.
- *Success*: some setting gives r ≥ 1.5 with an equal-error CPU ratio ≤ 1.0.
- *Kill (for wall-clock)*: r < 1.3 whenever f < 0.2. Rank 7 is then a memory-only claim, and the direction depends on the whitened closed form, not on gating.
- *Ambiguous*: CPU ratio between 1.0 and 1.5. Re-measure at χ = 64 before deciding.

**T2: does spcf beat SVD plus free post-processing on the verification horizon? (≈ 25 min, needs `ladder_t.py`)** Run it regardless of T1's wall-clock outcome, because memory is still at stake.
- *Setup*: Ising N = 20, T = 8, samples every 0.5. SVD χ ∈ {16, 24, 32, 48, 64, 96}; spcf χ ∈ {16, 24, 32} with the best T1 gate.
- *Arms*:
  - SVD;
  - SVD × F_MPS (Mandrà);
  - SVD log F extrapolation over the three largest χ (Anand);
  - SVD 1/χ extrapolation;
  - spcf;
  - spcf × F_MPS.
- *Metrics*:
  - t*(χ, 1e-2) and t*(χ, 1e-3) on 1-site and nn RMS;
  - r(t) along the trajectory;
  - ε_c over all pairs, as collateral;
  - infidelity.
- *Success*: t*_spcf(χ) ≥ t*_SVD(1.5χ) for both tolerances, AND spcf's local error is below the best post-processed SVD at equal χ for ≥ 70% of the samples in the growth phase, AND ε_c is no worse than SVD.
- *Kill*: post-processed SVD matches or beats spcf (within 20%) over most of the window, or ε_c degrades by more than 1.2x.
- *Ambiguous*: spcf wins on nn but loses ε_c. That is a real but narrow result: report it as "local-window verification only".

**T3: transfer to the benchmark circuit (≈ 30 min, kicked model needed; only if T2 passes).**
- *Setup*: 1D kicked Ising, N = 20, depth ≤ 20, initial state \|↑⟩^N.
  - θJ = −π/2 with θh ∈ {0.7, 1.0}, which is IBM's ergodic regime;
  - θJ = −π/4, θh = 0.7, Anand's harder non-Clifford case;
  - θh = π/2 as a Clifford control, expected to fail through the flat spectrum.
  - spcf uses `taus=()`.
- *Metrics*:
  - error of ⟨Z_{N/2}⟩ and the average magnetization versus depth;
  - ε_c;
  - the depth at which SVD's ⟨Z⟩ error first exceeds 1e-2;
  - r at that depth.
- *Success*: r ≥ 1.5 at SVD's horizon in at least two of the three ergodic settings, with ε_c no worse.
- *Kill*: r < 1.3 in all of them. That would mean the gain is specific to the continuous-time tilted Ising and does not reach circuit benchmarks.

Order: T1 (wall-clock lever) → T2 (competitor check) → T3 (application). T2 is the true go/no-go. T1 only decides whether the claim is "memory" or "memory + time".

## Open questions

- Is r ≈ 1.6 constant at larger χ (128-512) and larger N? With a dense reference this can only be tested up to N ≈ 22. Beyond that, the reference has to be SVD at χ_max, the protocol used by D-Wave and IBM.
- Does spcf's decoupling of local error from fidelity break fidelity-based extrapolation (Anand) and rescaling (Mandrà), or improve it? Either answer is publishable as a caution about χ_Q-type arguments.
- At exactly flat spectra (Clifford points), is a local-objective choice among degenerate subspaces a large win, given that SVD's choice is arbitrary there? Or is the Heisenberg MPO always the better tool?
- An LOMPS-style full HS projection on a finite chain at fixed χ, seeded from SVD: how much of its gain does one Gauss-Newton step capture, and at what cost?
- Would the whitened closed-form cut from the main report move the per-fired-cut cost from 15x to about 2x? At r ≈ 1.6 that alone would make every cell a wall-clock win.
