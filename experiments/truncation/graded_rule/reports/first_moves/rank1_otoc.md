# Rank 1 first moves: OTOCs and operator-weight fronts on a vectorized Heisenberg operator

## Summary verdict: conditional

The direction is still worth a first move, but the full texts change where any gain would appear. They also make one cheap control decide the case. Three findings matter:

1. **SVD-truncated MPOs are already accurate at and ahead of the front.** All the error sits *behind* it, in the growth and saturation region (C ≳ 0.1). That is where spcf would have to win. The report instead pictured the gain at the front.
2. **The best-known failure may be mostly a norm artifact.** Hémery-Pollmann-Luitz (HPL) describe the "halting" of operator spreading and the "severe underestimation of the saturation value" under MPO-TEBD. They evaluate a trace formula on an MPO that is not renormalized, and their saturation value scales with ‖O‖²_F. Our TEBD renormalizes the center tensor anyway. So test 1 must first check whether renormalized SVD already removes most of the headline error.
3. **In the fused Pauli basis, every single-site OTOC and the weight density are diagonal quadratic forms.** They are linear functions of the Pauli-label marginal distributions p_W(α) = Σ|c_P|² over the strings P whose labels on window W are α. The target set is therefore tiny: diagonal entries only, 4^k per window. This makes a d = 4 port much cheaper than the report's "generalize spcfast to d = 4" estimate implied.

Novelty risk stays low: no OTOC- or weight-preserving truncation was found. The weight-reweighting methods (rTEBD, DAOE, OST) bias against high-weight content *by design*, which is exactly what an OTOC measures. Two things would make this direction drop to "weak":
- renormalized SVD already gets the saturation value and the front contours right;
- a static single-cut ceiling shows less than 2x headroom on the Pauli-label marginals.

## Literature, verified from full text

Full text was read with `read_arxiv_paper`; PDFs are in `/tmp/papers_rank1_otoc`.

| arXiv | What I read | Verified facts (with location) | Report claim: status |
|---|---|---|---|
| 1802.00801 (Xu & Swingle, *Accessing scrambling using MPOs*) | Full main text | Mixed-field Ising H = −(1/E0)(J ZZ + hx X + hz Z), with J = 1, hx = 1.05, hz = 0.5 (chaotic) or 0 (free), E0 = √(4J² + 2hx² + 2hz²). L = 201, χ = 32, dt = 0.005, t up to about 150, "DMRG-like sweeping" (Eq. 8). Fig. 3: "χ = 4 produces the same result as χ = 32 in the early growth region where C(t) ranges from machine precision to ∼0.1". Fit λ = 3.8, p = 0.67, v_B = 0.67. Text: "Truncation only leads to errors in the operator representation behind the wavefront… physics ahead of and up to the wavefront is captured accurately". Eq. 10 gives the OTOC as a one-site observable of the operator state, C = 2 − 2⟨X_r(t)\|X_r′⊗X_r′\|X_r(t)⟩. For hz = 0, S ≤ log 4 (Fig. 2b). | "Established MPO OTOC": **correct**. The implied picture that the front is where SVD struggles is **wrong**: the front is captured at χ = 4. |
| 1901.05793 (Hémery, Pollmann, Luitz, PRB 100, 104303) | Full text incl. App. A | Tilted Ising J = 1, hx = 0.8090, hz = 0.9045. Note these are **swapped relative to the repo's `ising`**, which has hx = 0.9045, hz = 0.8090. L = 21 ED (Krylov) vs TEBD-MPO with χ = 64, dt = 0.01, 2nd-order Trotter, operator σ^z_4 near the left edge. L = 50 runs use χ = 64/128. Error is |C_exact − C_MPO| at the farthest site (Fig. 4, up to about 0.15). Contours t(θ, j) for θ = 1e-5, 1e-3, 0.1, 0.2 (Fig. 6): they agree at low θ and deviate at θ = 0.1-0.2. Conclusion: "after significant truncation the spreading of information appears to halt… strong truncation effects in the MPO TEBD approach lead to a severe underestimation of the saturation value". Their C is computed from Tr(V W V W) on the truncated, unnormalized MPO. | "Truncated MPO-TEBD halts spreading": **correct, with the norm caveat above**. |
| 2202.07060 (Xu & Swingle tutorial, PRX Quantum 5, 010201) | Sec. VI.B (tensor networks) in full | MPO results "agree perfectly with the exact result up to C ∼ 1" at L = 20, χ = 32 (Fig. 9b). Evaluating C as the squared norm of the commutator MPO is "much more accurate" than the overlap. **Time splitting**, i.e. computing C from W(t/2) and V(−t/2), "significantly enhances the accuracy… in the late-time regime" at cost χ² per probe site. TDVP-MPO "does not significantly increase the accuracy of TEBD-MPO". | The report attributed the support/tail statement here: **acceptable**. The report **missed** that time splitting is the known accuracy fix and therefore a competitor baseline. |
| 2503.20327 (Gisti, Luitz, Debertolis, Quantum 2025) | Intro, methods, Fig. 7 caption, App. (truncation) | Isotropic Heisenberg chain, L = 50, χ = 6000 (projected) and 1024 (full symmetric), dt = 0.005, t ≤ 20. Reliability is tracked by cumulative discarded weight per bond, with threshold 1e-4 (hatched "computational light cone"). | "χ up to 6000, tracks only discarded weight": **correct**. **Implication the report missed**: the Heisenberg OTOC is *not* cheap for SVD, so XXZ is not a clean predicted failure. |
| 2412.08730 (Guha Roy & Slagle, rTEBD, SciPost Phys. 21, 015) | Abstract, intro, Sec. 4.2, conclusion | Authorship confirmed. The intro explicitly mentions MPOs encoding "a time evolved observable", but all benchmarks are MPDOs: free fermions, and spin model J = 1, hx = 0.9045, hz = 0.8090 (same as ours). Optimal γ = 1.6-1.7 (Fig. 13). No OTOC anywhere. | It is a relevant **contrast** baseline: γ^(−n) deliberately suppresses exactly the high-weight content the OTOC reads. |
| 2111.09904 (von Keyserlingk, Pollmann, Rakovszky, backflow) | Abstract, intro; grep of the body | Backflow corrections to **transport coefficients** are exponentially suppressed in the cutoff ℓ*. | **Misapplied by the report.** Backflow concerns linear, low-weight observables. The OTOC and weight profile *are* high-weight content, so this does not argue that the collateral is cheap here. |
| 0706.2480 | Full (5 pages) | Prosen & Pižorn, *transverse-field* Ising (free fermions). Operator entanglement saturates for finite-index operators and grows as (1/6) ln t or (1/3) ln t for infinite-index operators. | **Wrong citation for XXZ.** It supports a TFIM (hz = 0) null control, not an XXZ one. |
| 1705.08975 (Nahum-Vijay-Haah, PRX 8, 021014) | Not re-read (prior knowledge) | Random circuits: the front moves at v_B and broadens as √t. | Unchanged; low stakes. |
| 2004.05177 (DAOE), 2601.12446 (exact MPO operator "mass/length" distributions), 2603.05656 (Dowling, simulability from operator entanglement), Alba PRB 104, 094410 | Abstract or snippet only | DAOE damps weight above ℓ*. 2601.12446 computes the full Pauli-weight distribution exactly from an MPS of the operator: a useful scoring tool, not a truncation rule. Alba: logarithmic operator-entanglement bounds in integrable models. | Context only. |

## Prior art and novelty risk

- **No paper found truncates a Heisenberg MPO to preserve OTOCs or the Pauli-label (weight) distribution.** This held across Google Scholar queries on OTOC/MPO truncation, operator-weight preserving truncation, density-matrix truncation applied to OTOCs, and operator-size truncation, and across the citation trails visible in 1802.00801, 1901.05793 and 2202.07060.
- **Known improvements in this setting are not truncation rules.** They are:
  - time splitting (2202.07060);
  - TDVP in the Schrödinger picture with sampling (HPL);
  - U(1)-projected MPOs with huge χ (2503.20327);
  - Lightcone-aware χ (1802.00801).
- **The nearest truncation-rule neighbours bias the wrong way for this target.** DAOE, OST and rTEBD down-weight high-weight strings; DMT preserves *linear* ≤3-site Pauli coefficients. None preserves Σ|c_P|² marginals.
- **Residual risk: low-moderate.**
  - rTEBD's public code can be pointed at an operator in a few lines. A reviewer will ask for it, and for time splitting, as baselines.
  - 2601.12446's exact marginal machinery shows that the marginals themselves are standard MPS quantities. Only using them as a truncation target is new.
- **One framing correction.** In the Pauli basis the targets are *diagonal* (probability marginals). A natural cheaper competitor is therefore a **diagonal-metric whitened SVD**: rescale each site's Pauli label by a factor tied to its marginal, which is rTEBD with a data-driven γ. That should be run as a control arm, because it is the closed-form "whitening" route the report recommends anyway.

## Technical details for implementation

**Object and normalization.**
- Write O(t) = Σ_P c_P P/√(2^N), with Hermitian O so the c_P are real.
- Store it as an MPS over N fused sites, d = 4, with basis labels {I, X, Y, Z}. The Frobenius norm Σc² = 1 is exactly conserved by unitary dynamics, so renormalizing after a cut is legitimate.
- The superoperator gate in the Pauli basis is real orthogonal (16×16). `pauli_prop.transfer(G4)` already builds R[c_in, c_out] with code c = p_b + 4p_{b+1}. Reindex it to the row-major (p_b, p_{b+1}) order used in `th` ('labr'), and transpose correctly for Heisenberg order: O → G†OG.
- `pauli_prop.circuit_bonds(N, nsteps)[::-1]` is the correct reversed gate order for the dense reference.

**Observables. All are functions of the one-site marginals p_x(α).**
- OTOC with W = Z, using HPL's normalization that saturates at 1: C_Z(x,t) = 2[p_x(X) + p_x(Y)]. Similarly C_X = 2[p_x(Y) + p_x(Z)].
- Weight density: w(x) = 1 − p_x(I). Total size: Σ_x w(x).
- Front contours t(θ, x) for θ ∈ {1e-3, 0.1, 0.5}. Comparing contours follows HPL Fig. 6.
- Collateral / non-targeted quantities:
  - 2-site label joint distributions at distance ≥ 3;
  - OTOCs with two-site probes, e.g. W = Z_x Z_{x+3}, which needs the joint p over (x, x+3);
  - Frobenius infidelity 1 − ⟨⟨O_ex\|O⟩⟩²;
  - the *linear* autocorrelation c_{Z_0}(t), which is DMT's home ground.
- Dense side: reshape c to [4]*N and sum c² over the other axes. This costs seconds at N = 10.

**Setup others used.**
- O = σ^z near the **left edge** (HPL use site 4 of 21) to maximize front travel. Place O = Z_0 for N = 10-12.
- v_B ≈ 1.4 sites per 1/J in HPL's tilted Ising, so the front reaches the right end at t ≈ 6-7 for N = 10.
- Literature χ for "converged tails" is 4-32. The saturation region is never converged at practical χ.

**Sizes and costs (sanity checked here).**
- One dense real 16×16 gate on 4^10 amplitudes takes 0.044 s, and on 4^12 (134 MB) 0.62 s. These were measured with `einsum`; matmul would be faster.
- N = 10 at T = 6, dt = 0.1 is about 1 min of reference. N = 12 is about 15 min and should be cached in `_refcache/`.
- This sandbox has numpy but **no scipy**. The runs belong on the user's machine, where `mpsenh` (scipy `expm`) works.

**Region cost for the tilt.**
- spcfast's region W is D × (χ_lo χ_hi) with D = d^L. With d = 4: a = 1 gives L = 4, D = 256; a = 2 gives L = 6, D = 4096.
- spcfast forms the full region ρ = WW†. For d = 4, a = 2 that is a 4096² matrix, which is infeasible per cut.
- **Diagonal targets avoid it.** h = row-sums of |W|², dh = 2 Re(conj(W0) ∘ dW) row-summed, and the window marginals are a fixed 0/1 summation map M (size: 4^k entries per window).
- With k ≤ 3 and L = 6 there are 360 residual rows. The memory notes found a = 1 overfits for spcfast while a = 2 works, so keep a = 2 and χ ≤ 32 for first tests.

**Known failure modes to watch.**
- **The staircase Strang sweep.** `run_tebd` sweeps 0 → N−2 → 0, which gives O(dt^k) weight k sites away within one step. That pollutes 1e-10-level tails. It does not matter for θ ≥ 1e-3 contours, but use the identical circuit for MPO and dense.
- **Norm loss masquerading as physics** (above).
- **Gauge.** The center must be at the cut, as in the existing drivers.

## Where to start (code changes and effort)

1. **`op_tebd.py` (new, about 200 lines, half a day).**
   - Pauli-basis gates from `pauli_prop.transfer`, reshaped to (4,4,4,4) in 'labr' order.
   - A d-generic copy of `mpsenh.run_tebd`. Its einsum is already generic; only `initial_mps`, `svd_cut` and the reshape `l*2` hard-code d = 2.
   - Initial operator: a product MPS with Z on site 0 and I elsewhere.
   - Cut arms:
     - `svd` (renormalized, the repo default);
     - `svdraw` (no renormalization of the kept center, HPL-like);
     - `rw:γ` (rTEBD-style per-site label rescaling diag(1, γ⁻¹, γ⁻¹, γ⁻¹) before the SVD and undo after);
     - the new tilt.
   - A dense reference via `transfer` + `circuit_bonds` and a `p_x(α)` scorer for both sides.
   - A per-cut log of the **cross-cut residual**: the pre- vs post-truncation window marginals, i.e. spcf's r0 restricted to span-2 rows, mirroring `_low_span_resid` for the gate.
2. **`rule/spcop.py` (new, about 250 lines, 1-1.5 days).** Copy the skeleton of `SPCFast.__call__`:
   - SVD;
   - `_maps` (already d-generic);
   - a `Wof` with `d` in place of the literal 2;
   - the tangent chart Q = qr(U_k + U_⊥C);
   - CG on JᵀJ;
   - one α = 1 retraction, accepted only if the residual drops.

   Replace `hvec`/`Fapply`/`_region` with the diagonal map above. Keep `fw = 0` (best in the state case) and the benefit/cost gate hook. Port `test_spcfast.py`'s jvp/vjp adjoint check and its finite-difference linearization check: this is the bug net.

   Do **not** generalize `spcfast.py` in place. Its Hermitian off-diagonal machinery is exactly what d = 4 cannot afford, and the existing N = 12-16 results depend on it.
3. **`op_bench.py` (about 80 lines).** Arms × χ ladder → JSON, with the metrics above split into three regions by the exact C: ahead (< 1e-2), front (1e-2 to 0.5), behind (> 0.5).

**Total: about 2-3 working days to the decisive test 3.** Test 1 needs only step 1.

## First tests: a decision tree

All tests use N = 10 (a 4^10 reference takes seconds to a minute) and a χ ladder of 4, 8, 16, 32. Times are at dt = 0.1, T ≤ 6 (front at the far edge). Each job has a 1800 s timeout per the 30-minute rule.

### Test 1: headroom and the norm artifact (SVD arms only; about 1 h of coding after `op_tebd.py`, about 5 min of runs)

- **Setup.** Repo `ising` (hx = 0.9045, hz = 0.8090) and TFIM (hz = 0, the null control: Xu-Swingle show S ≤ log 4, so SVD should be exact at χ ≥ 4). O = Z_0, T = 6.
- **Arms.** `svd`, `svdraw`, and `rw:1.6` (rTEBD's optimal γ on this exact model). The last is the low-weight-biased contrast.
- **Metrics.**
  - Region-split rms and max |ΔC_Z(x,t)| and |Δw(x,t)|.
  - Contour lag Δt(θ) at θ = 0.1 and 0.5.
  - Infidelity.
  - Ratio of the per-cut span-2 residual to the final marginal error, which says whether the error is made locally at cuts or arrives by propagation.
- **Pre-registered reading.**
  - *Continue* if renormalized `svd` on Ising, at some χ in the ladder, has behind-front rms |ΔC| ≥ 0.02 or a contour lag ≥ 0.3 time units at θ = 0.5. The TFIM error must also be ≤ 1e-6, which validates the harness.
  - *Kill the direction* (or reframe it as "renormalize your MPO", a one-line note) if `svd` is within 0.005 of exact everywhere while `svdraw` reproduces HPL's underestimate.
  - Record whether `rw:1.6` is *worse* than `svd` on C behind the front. That is the expected sign, and it supports the novelty framing.
  - *Ambiguous:* the error is large but the per-cut residual is under 10% of it, meaning the error is mostly propagated rather than made locally. Proceed to test 2 anyway, but expect a gain under 2x.

### Test 2: static ceiling at a single snapshot (needs `spcop.py`; about 2 min of runs)

- **Setup.**
  - Take the exact |O(t)⟩⟩ for N = 10 at t ∈ {2, 4} (front mid-chain).
  - Compress by one left-to-right sweep of truncations to χ, with two arms: SVD, and SVD + spcop tilt at each cut (targets from the untruncated tensor at that cut, class (c)).
  - Optional upper bound: several Gauss-Newton passes (`passes = 3`).
- **Metrics.** The error ratio spcop/SVD of the targeted marginals (k = 1-3) and of C_Z behind the front. Also the non-targeted distance-≥3 joint marginals and the infidelity ratio.
- **Pre-registered reading.**
  - *Continue* if the ratio on 1-site and 2-site marginals is ≤ 0.5 at χ = 8 or 16, the infidelity ratio is ≤ 1.2, and far-marginal error is ≤ 1.5x SVD.
  - *Kill* if the ratio is ≥ 0.8 at all χ. A single-step tilt with no dynamics then cannot buy 2x, and the dynamic gain is historically 3-10x *below* the static ceiling (memory notes, state case).
  - *Ambiguous:* 0.5-0.8 → run test 3 only at the best χ.

### Test 3: dynamic head-to-head (about 15-30 min total)

- **Setup.** `op_tebd` on Ising, TFIM and Heisenberg (the Heisenberg run is informational: 2503.20327 shows its OTOC is not SVD-easy, so no failure is pre-registered).
- **Arms.** `svd`, `spcop`, `spcop` gated (spcfast's gate formula, refit on Ising), and the `rw:1.6` contrast. All at equal χ ∈ {8, 16, 32}.
- **Metrics.** Test 1's metrics, plus wall-time per cut relative to SVD.
- **Pre-registered success.**
  - On Ising, behind-front or front rms |ΔC_Z| and |Δw| are ≥ 2x lower than `svd` at ≥ 2 of 3 χ values.
  - Contour lag at θ = 0.5 is reduced by ≥ 30%.
  - Infidelity is ≤ 1.2x, non-targeted far marginals are ≤ 1.5x, and the linear autocorrelation is no worse than 1.5x.
- **Kill.** Ratio > 0.75 at all χ on Ising, or the gain only appears ahead of the front (where SVD is already exact).
- **Ambiguous.** A 1.3-2x gain with growing collateral. The next step would then be the `fw` damping scan and the diagonal whitened-SVD control (rescale labels by measured marginals), before any claim.

## Open questions

1. **How much of the HPL failure is norm loss?** Test 1 answers this directly. Xu-Swingle normalize the operator state and report "qualitatively correct" late-time behavior (Fig. 4a), which hints that it is substantial.
2. **Do 1- to 3-site Pauli-label marginals suffice?** The OTOC with single-site W needs only k = 1 marginals. The state-case finding that k = 1 targets alone give no gain (`obspin`, memory notes) suggests k = 2-3 cross-cut joints are what make the tilt work. Test 2 should include a k = 1-only arm.
3. **Should the targets be weighted by region?** Uniform window weights spend capacity on the trivial region ahead of the front (p(I) ≈ 1). A "behind-front only" weighting may matter.
4. **Competitors at equal cost.** Time splitting (2202.07060) costs χ² per probe site, and rTEBD has public code. Both must eventually be run at matched memory, not just matched χ.
5. **The right integrable control.** It is TFIM for "SVD exact". XXZ/Heisenberg is unresolved: logarithmic operator-entanglement bounds (Alba) versus χ = 6000 needed in practice (2503.20327).
6. **Gate order.** Should the operator benchmark switch from the staircase Strang sweep to brickwork, to keep exact light cones in the tails (θ ≤ 1e-6)? This is irrelevant for the first tests.

**Tool notes.** `read_arxiv_paper` worked for every ID tried, with large outputs saved to files. `search_semantic` and `search_openalex` returned empty lists. `search_arxiv` keyword search returned mostly irrelevant results. `search_google_scholar` worked and was the main discovery channel.
