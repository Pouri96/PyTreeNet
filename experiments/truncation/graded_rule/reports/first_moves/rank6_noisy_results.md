# Rank 6 first move, results: spcf on LPDOs under dephasing Lindblad noise

## Verdict: CONTINUE (conditional), to a single decisive test against DMT-MPO

**One-line reason.** The spcf bond tilt keeps its gain under dephasing noise: at matched (χ, κ) it lowers nn rms by 2.2-3.8× against environment-gauged SVD-LPDO (trace distance 0.78-0.96× of SVD's) wherever bond truncation, not Kraus truncation, sets the error. The pre-registered grid (κ ≤ 4) only narrowly missed its success bar (best 1.9-2.1×), and the κ = 8 cells that pass are a post-hoc extension.

What decides the verdict:
- Test 1 (audit, no tilt) **succeeds** on the pre-registered criteria, and neither kill criterion is hit.
- Test 2 on the pre-registered grid (κ = 4) is **neither success nor kill**. Gains are 1.9-2.1× at the best adjacent χ pair, against a bar of 2×. Gain is never below 1.3× at χ ≤ 8, and trace distance never gets worse.
- A post-hoc κ = 8 extension (κ = 8 is in the plan's parameter list, but not in the Test 1/2 grid) **meets every Test 2 success condition**. It was replicated at N = 10. Details are in section 4.
- The gain window ends where the Kraus truncation floor takes over. At κ = 4 that is χ ≳ 12 (gain 1.2-1.3×). At κ = 8 it is χ ≈ 24 at N = 10 (gain 1.27×).
- Open threat, not tested here (Test 3, out of scope): DMT on a plain MPO. This is the planner's own "weak standalone" caveat.

**Single next experiment.** Test 3(i), restricted to one cell. Take N = 10, T = 4, γ = 0.03, with spcf-LPDO at (κ = 8, χ = 12 and 16) as the arm. The opponent is numpy DMT-MPO at equal stored parameters (LPDO 2Kχ² per site against 4D² for the MPO), using the DMT that rank 2 built. Record nn/nnn rms, λ_min(ρ) and the minimum 3- and 4-site RDM eigenvalues. Pre-registered kill: DMT ≥ 2× better on nn rms with |λ_min| < 10⁻³. If spcf-LPDO ties or wins, or DMT is non-positive, continue to Test 3(ii).

## 1. What was built (all new files are in `rank6/`; no existing file was modified)

| file | content |
|---|---|
| `lpdo.py` | LPDO site tensors A[l,p,k,r]. Strang operation list `schedule` shared by every arm and by the reference. Dephasing channel p = (1-e^{-2γτ})/2 applied at the orthogonality centre. Kraus truncation by SVD of the centre tensor ((l p r), 2K) (environment-gauged, Guo-Yang style). Bond SVD cut (with K = 1 this is `mpsenh.svd_cut` bit for bit). Dense Lindblad reference. Dense global Kraus-only purification (χ = ∞). LPDO → dense ρ. Metrics: single/nn/nnn rms, \|ΔE\|, far ZZ at distance 3 and 4, trace norm, HS norm, λ_min. |
| `gatelog.py` | spcf gate quantities (tail, B_res, cost) at every LPDO bond cut, audit only. Kraus legs enter through Gram matrices of the environments, which is an exact reformulation of ρ_region = W W†. |
| `spcflpdo.py` | Bond-tilt cut for an LPDO. A read-only derivative of `rank2/spcfpur.py` / `rule/spcfast.py`, generalised to per-site K and Gram-compressed environments. |
| `lpdo_bench.py`, `pressure_scan.py` | Cell runner, SVD pressure scan. |
| `test0.py`, `test_spcflpdo.py` | Correctness gates. |
| `test1.py`, `analyze_test1.py` | The Test 1 grid (pre-registered definitions are in the docstring) and its evaluation. |
| `test2.py`, `analyze_test2.py`, `test2_kappa.py`, `test2_n10k8.py`, `analyze_kappa.py` | Test 2 grid, and the κ = 8 extension with its N = 10 replicate. |
| `results/` | Raw JSON and JSONL, logs, tables. |

The tilt in `spcflpdo.py` follows `spcfpur.py` line for line. I did not import the rank 2 module, because its ancilla dimension is uniform and an LPDO needs a different K on each cut site. The state is the repo `ising` model with a Néel start, N = 8 (N = 10 for checks), dt = 0.1, T = 4. The noise is dephasing L_j = √γ Z_j. The operation list per step is: noise on site 0; for b = 0..N-2 gate(b, R) then noise on b+1; noise on N-1; for b = N-2..0 gate(b, L) then noise on b. Every site is dephased for a total time dt per step, and the reference uses exactly this list.

## 2. Test 0: correctness gates. All 22 pass (`results/test0.json`, `results/test_spcflpdo.json`)

| check | result |
|---|---|
| dense Lindblad at γ = 0 equals \|ψ⟩⟨ψ| from `mpsenh.dense_reference`, N = 8, T = 2 and 4 | max abs diff 2×10⁻¹⁶ |
| dephasing channel equals the exact exponential of L(ρ) = γ(ZρZ-ρ) | 3×10⁻¹⁷ |
| Strang operation list against the exact full-model Lindbladian exponential, dt = 0.2, 0.1, 0.05 | error 3.0e-3, 7.5e-4, 1.9e-4, so order 2.00. This confirms the total noise time, the Hamiltonian and the channel convention. |
| dense Lindblad at γ = 0.01 and 0.1: trace 1, Hermitian, PSD | pass (λ_min ≥ 10⁻⁷ at N = 8, T = 4) |
| exact LPDO (χ = 200, κ = 64) equals dense Lindblad, N = 6, γ = 0 and 0.1 | 2×10⁻¹⁵ |
| LPDO(χ = ∞, κ = 1, 2, 3) equals the dense **global** Kraus truncation. This validates the centre-gauged truncation as the Frobenius-optimal one. | 4×10⁻¹⁴ to 7×10⁻¹⁵ |
| γ = 0 LPDO-SVD equals MPS-SVD, N = 8, T = 4, χ = 6, 8, 12 | **bitwise identical** tensors |
| gate quantities (tail, B_res, cost, b) at K = 1 against `SPCFast.gate_log`, a = 2 and a = 1, χ = 6, 8 | tail and cost exactly equal, B_res relative difference ≤ 4×10⁻¹², same cut sets (147 and 39 entries) |
| region ρ with the Kraus legs traced (Gram form) against the partial trace of the dense ρ, K up to 4, a = 1 and 2 | worst relative difference 7×10⁻¹⁵ |
| LPDO ρ: trace 1, Hermitian, λ_min ≥ -10⁻¹⁴ | pass |
| tilt adjoint pair ⟨g, J x⟩ = ⟨Jᵀ g, x⟩ at K = 3 cuts | float32: 4.1×10⁻⁵. float64: 2.9×10⁻¹³. |
| linearisation error against step size | linear in the step: 0.13, 0.012, 0.0012, 1.2×10⁻⁴ for steps 10⁻¹..10⁻⁴ |
| K = 1 equivalence of `SPCFLpdo` with `SPCFast`, χ = 6, 8 | overlap 1.0000000000, identical fired counts (147/560, 39/560), identical nn rms (1.1991e-2, 3.5318e-3) |

Deviations in Test 0:
- The plan asked for adjoint error < 10⁻⁵. In float32 this is not reachable even by the unmodified `SPCFast` (its own `test_spcfast.py` gives 5.9×10⁻⁵). The tilt passes it in float64 and matches the base code in float32.
- "K ≡ 1 bit-identical to `mps_bench` spcf" is replaced by equivalence at the 10⁻¹⁰ level. The Gram-compressed environment uses a different column basis, so bitwise equality is not expected.
- The plan's "K ≡ 2 reproduces `spcfpur`" was not run. I did not import `spcfpur`.

## 3. Test 1: audit, no tilt (`results/test1_raw.jsonl`, `results/test1_table.json`)

**Pre-registered definitions** (written in the `test1.py` docstring before the grid ran):
- (a) is both truncations, centre-gauged Kraus truncation.
- (b) is bond truncation only: κ = 16 stands in for ∞.
- (c) is Kraus truncation only: χ = 64 stands in for ∞.
- s_b = e_b/e_a and s_c = e_c/e_a are ratios of nn rms.
- The gate on-fraction f is the number of logged cuts with B_res ≥ c₀(tail/10⁻⁴)^0.65 over **all** 560 bond cuts of the run. The a = 2 window is primary, because the gate constants were calibrated there. a = 1 is a robustness check. ρ_f = f(γ)/f(0) at the same χ.
- SUCCESS: at γ = 0.01, for some κ, at ≥ 2 values of χ with e_a ≥ 10⁻³: s_b ≥ 0.5 and ρ_f ≥ 0.5.
- KILL: ρ_f(0.01) < 0.25 at every χ; or s_c > 0.8 in every usable cell at γ ≥ 0.01 together with a Kraus-tilt probe gain < 1.3×.

**Numbers at γ = 0.01** (N = 8, T = 4):

| κ | χ | e_a (SVD nn rms) | e_b | e_c | s_b | s_c | f (a=2) | ρ_f |
|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 1.96e-2 | 2.10e-2 | 9.83e-3 | 1.07 | 0.50 | 0.420 | 1.60 |
| 2 | 8 | 1.36e-2 | 1.11e-2 | 9.83e-3 | 0.82 | 0.72 | 0.312 | 4.49 |
| 2 | 12 | 9.88e-3 | 3.30e-3 | 9.83e-3 | 0.33 | 0.99 | 0.141 | 2.47 |
| 4 | 6 | 2.04e-2 | 2.10e-2 | 3.05e-3 | 1.03 | 0.15 | 0.559 | 2.13 |
| 4 | 8 | 1.12e-2 | 1.11e-2 | 3.05e-3 | 0.99 | 0.27 | 0.343 | 4.92 |
| 4 | 12 | 4.06e-3 | 3.30e-3 | 3.05e-3 | 0.81 | 0.75 | 0.239 | 4.19 |

The γ = 0 on-fractions, which ρ_f divides by, are 0.263 at χ = 6, 0.070 at χ = 8, and 0.057 at χ = 12.

**Verdict on the criteria.**
- **SUCCESS: met.** κ = 4 satisfies both conditions at χ = 6, 8 and 12. κ = 2 satisfies them at χ = 6 and 8. The a = 1 gate gives the same sets.
- **KILL: not hit.**
  - K1 is false, because ρ_f(0.01) is between 1.6 and 4.9.
  - K2 is false. 10 of the 18 usable cells at γ ≥ 0.01 have s_c ≤ 0.8, so the Kraus-tilt probe was not needed.

**Things the criteria do not show, which matter for reading the result.**
- γ = 0.01 is weak noise (e^{-2γT} = 0.92), so passing there is a mild test. The same quantities at stronger noise, for κ = 4:

| γ | s_b at χ = 6 / 8 / 12 | ρ_f (a = 2) at χ = 6 / 8 / 12 |
|---|---|---|
| 0.03 | 1.02 / 0.98 / 0.86 | 2.28 / 5.36 / 5.22 |
| 0.1 | 1.00 / 0.94 / 0.74 | 2.38 / 5.85 / 5.72 |

- The repairable residual does not fade with γ. The on-fraction rises, because noise makes the purification more entangled and more cuts discard enough weight for the gate to fire. The "gain decays with γ" failure mode predicted by rank 3 is not seen in this audit.
- Kraus truncation does not dominate the nn-rms error at χ ≤ 8, even at γ = 0.1 with κ = 4 (s_c = 0.55 at χ = 6, 0.66 at χ = 8). It does dominate at χ = 12 (s_c = 0.85 at γ = 0.1) and at κ = 2.
- Kraus truncation **does** dominate the energy error. At γ = 0.03, E_abs is about 0.28 at κ = 4 against about 0.04 at κ = 16.
- Audit quality:
  - (c) at χ = 64 has a total bond discarded weight ≤ 2×10⁻⁷ over all cuts. At κ = 2 it equals the dense χ = ∞ purification to 5 digits (for example 9.8257e-3 for both at γ = 0.01).
  - (b) at κ = 16 has a cumulative Kraus discarded weight of 2×10⁻⁴ (γ = 0.01), 2×10⁻³ (γ = 0.03) and 2×10⁻² (γ = 0.1). So "∞" is good to γ = 0.03 and approximate at γ = 0.1.

## 4. Test 2: spcf bond tilt against environment-gauged SVD-LPDO (`results/test2_*`)

Pre-registered criteria, from the plan:
- SUCCESS (both): nn rms ≥ 2× lower than SVD at two **adjacent** χ in a γ ≥ 0.01 cell, with trace norm ≤ 1.10× SVD's and far ZZ (distance 3 and 4) ≤ 1.5×; and the γ = 0.01 gain ≥ half of the γ = 0 gain.
- KILL (either): gain < 1.3 in every γ ≥ 0.01 cell; or trace norm > 1.25× wherever the gain is ≥ 1.3.

### 4a. Pre-registered grid: N = 8, T = 4, κ = 4, a = 2 window (iters = 4, taus = [1.0], ks = 1-2-3, fw = 0)

Gain is SVD nn rms divided by spcf nn rms. The trace ratio is spcf over SVD.

| γ | χ | nn SVD | nn spcf | **gain** | nnn gain | trace ratio | far3 / far4 ratio | fired |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 1.79e-2 | 1.20e-2 | 1.49 | 1.34 | 0.97 | 0.85 / 1.56 | 147/560 |
| 0 | 8 | 6.69e-3 | 3.53e-3 | 1.89 | 1.20 | 0.98 | 0.75 / 0.59 | 39/560 |
| 0 | 12 | 6.30e-4 | 3.51e-4 | 1.79 | 1.22 | 1.01 | 1.03 / 0.68 | 32/560 |
| 0.01 | 6 | 2.04e-2 | 1.10e-2 | **1.86** | 1.51 | 0.94 | 0.43 / 0.79 | 317/560 |
| 0.01 | 8 | 1.12e-2 | 5.25e-3 | **2.13** | 1.23 | 0.97 | 0.70 / 0.41 | 192/560 |
| 0.01 | 12 | 4.06e-3 | 3.36e-3 | 1.21 | 1.15 | 0.97 | 0.92 / 0.62 | 153/560 |
| 0.03 | 6 | 2.89e-2 | 1.39e-2 | **2.08** | 1.61 | 0.90 | 0.28 / 0.55 | 336/560 |
| 0.03 | 8 | 2.04e-2 | 1.05e-2 | **1.95** | 1.35 | 0.94 | 0.56 / 0.76 | 209/560 |
| 0.03 | 12 | 1.16e-2 | 8.99e-3 | 1.29 | 1.21 | 0.96 | 0.91 / 0.62 | 167/560 |
| 0.1 | 6 | 4.27e-2 | 2.18e-2 | 1.96 | 1.53 | 0.85 | 0.38 / 0.93 | 350/560 |
| 0.1 | 8 | 3.60e-2 | 2.34e-2 | 1.54 | 1.31 | 0.89 | 0.40 / 0.86 | 228/560 |
| 0.1 | 12 | 2.77e-2 | 2.39e-2 | 1.16 | 1.15 | 0.95 | 1.19 / 0.85 | 183/560 |

- **Success: not met.** No γ ≥ 0.01 cell has two adjacent χ at gain ≥ 2. The nearest are (1.86, 2.13) at γ = 0.01 and (2.08, 1.95) at γ = 0.03.
- The second condition holds: the γ = 0.01 gains (1.86, 2.13, 1.21) are at least half the γ = 0 gains (1.49, 1.89, 1.79) at χ = 6 and 8.
- **Kill: not hit.** The maximum gain is 2.13 and the trace norm is never worse. The ratio is ≤ 1.01 everywhere.
- Classification: ambiguous, a narrow miss at N = 8. The pure-state gain in this small cell is itself only 1.5-1.9×, so a 2× bar is above what even the pure-state spcf delivers at N = 8.
- The gated arm is indistinguishable from ungated. Its gain is identical to two decimals, and it fires on the same cuts except 133 against 153 at (0.01, 12). So the gate is not costing anything here.
- The energy error is **not** improved. E_abs is 0.46-0.97× at γ = 0.01 and about 1× at γ ≥ 0.03, as in the closed-system findings: energy is not in the objective.

### 4b. N = 10 check on the same κ = 4 grid (T = 4, a = 2)

| γ | χ = 8 / 12 / 16 gain |
|---|---|
| 0 | 2.41 / 2.56 / 2.41 |
| 0.03 | 2.40 / 1.63 / 1.20 |

Trace ratios at γ = 0.03 are 0.89, 0.95 and 0.97, and far3 ratios are 0.42, 0.82 and 1.13. At κ = 4 the gain is therefore 2.4× at χ = 8 and decays to 1.2× by χ = 16, as the Kraus floor takes over.

### 4c. Post-hoc extension: κ = 8 (decided after seeing 4a; κ = 8 is in the plan's parameter list but not in the Test 1/2 grid)

The motivation is Test 1: at κ = 4 the Kraus floor (e_c about 3e-3 at γ = 0.01, 8.5e-3 at γ = 0.03) caps the achievable bond-tilt gain at χ ≥ 12. A larger κ lowers that floor.

These runs use the cheaper a = 1 window, whose γ = 0 gain is lower than a = 2's: at N = 8, 1.54 and 1.45 at χ = 8 and 12, against 1.89 and 1.79.

N = 8, T = 4, κ = 8, a = 1:

| γ | χ | gain | nnn gain | trace ratio | far3 / far4 ratio | E ratio |
|---|---|---|---|---|---|---|
| 0 | 8 | 1.54 | 1.27 | 0.98 | 0.67 / 0.39 | 1.73 |
| 0 | 12 | 1.45 | 1.06 | 0.95 | 1.09 / 0.89 | 1.10 |
| 0.01 | 8 | **2.31** | 1.43 | 0.94 | 0.54 / 0.25 | 2.02 |
| 0.01 | 12 | **2.90** | 1.45 | 0.96 | 0.44 / 0.84 | 1.43 |
| 0.01 | 16 | 2.59 | 1.47 | 0.94 | 0.47 / **1.70** | 1.13 |
| 0.03 | 8 | **3.07** | 1.71 | 0.92 | 0.44 / 0.63 | 1.42 |
| 0.03 | 12 | **3.17** | 1.94 | 0.91 | 0.37 / 0.75 | 1.16 |
| 0.03 | 16 | 2.18 | 1.45 | 0.91 | 0.68 / 0.99 | 1.04 |
| 0.1 | 8 | **3.19** | 1.77 | 0.78 | 0.23 / 0.79 | 1.05 |
| 0.1 | 12 | **2.22** | 1.66 | 0.79 | 0.51 / 0.97 | 1.02 |
| 0.1 | 16 | 1.45 | 1.17 | 0.86 | **2.52** / 1.27 | 1.00 |

N = 10 replicate (κ = 8, γ = 0.03, a = 1):

| χ | gain | trace ratio | far3 / far4 ratio |
|---|---|---|---|
| 12 | 3.77 | 0.91 | 0.50 / 0.89 |
| 16 | 2.97 | 0.92 | 0.58 / 1.21 |
| 24 | 1.27 | 0.97 | 1.05 / 0.76 |

- **Success conditions, κ = 8:** met at γ = 0.01, 0.03 and 0.1 (N = 8, adjacent pair χ = 8 and 12, all gains ≥ 2.2), and at N = 10 (pair χ = 12 and 16). Trace norm is 0.78-0.96× SVD's. Far ZZ is ≤ 1.5× in every pair counted. The γ = 0.01 gain (2.3-2.9×) exceeds the γ = 0 gain (1.5×).
- The gain is **larger** with noise than without, which is consistent with the rise in ρ_f in section 3.
- Where the tilt does not help:
  - It does not help once the error is at or below the Kraus floor, or once the SVD cut is near exact. At χ = 16, N = 8, γ = 0.1 the gain is 1.45 and far3 gets worse by 2.5× (both errors are tiny: 3.0e-3 against 7.5e-3). At γ = 0.01 and χ = 16 the far4 ratio is 1.70. At N = 10 and χ = 24 the gain is 1.27.
  - Energy: E_abs is up to 2.0× worse with the tilt at low χ (E ratio 2.02 at γ = 0.01, χ = 8), converging to about 1× at larger χ and γ. Anyone who needs the energy must add it to the objective.

## 5. Deviations, with the reason for each

- **Compute pruning.** The machine ran at load 12-22 on 4 cores, and the first launch of Test 1 was lost to an interruption.
  - Test 1 ran the full planned grid (γ five values, χ ∈ {6, 8, 12}, κ ∈ {2, 4}) apart from the (c) change below.
  - Test 2 was pruned to γ ∈ {0, 0.01, 0.03, 0.1}, χ ∈ {6, 8, 12}, κ = 4 and arms SVD / spcf / gated, N = 8 only, plus N = 10 at γ ∈ {0, 0.03} and χ ∈ {8, 12, 16}.
  - Not run: Cheng-order SVD, the Kraus-tilt arm, γ = 0.003 in Test 2, T scans, N = 12.
- **(c) "χ = ∞" uses the LPDO at χ = 64** instead of a dense purification, because the dense κ = 4 runs were too slow on the loaded machine. It was validated against the dense global-Kraus χ = ∞ run at κ = 2 (identical to 5 digits), and the total bond discarded weight is ≤ 2×10⁻⁷.
- **(b) "κ = ∞" is κ = 16**, with discarded Kraus weight as listed in section 3.
- **Gate window.** Test 1 uses a = 2 as primary (the calibration config) and a = 1 as a check; the plan suggested a = 1 first. The two agree.
- **Tilt window.** The κ = 4 grid in Test 2 uses a = 2. The κ = 8 extension uses a = 1 for cost.
- **Purification Frobenius error** was not computed, because it is gauge-dependent. The gauge-free ‖ρ - ρ_ex‖_HS is stored as `hs_norm` in every row.
- **The κ = 8 extension was decided after seeing the κ = 4 result.** It is a single-model, single-time, deterministic result at N = 8 and N = 10. It has not been tested at N = 12 or on other models (`heis`, `isingdw`) or with other noise (depolarizing).
- **The plan's "known failure mode" of the baseline** (SVD-LPDO being a weak strawman) is covered, in that the baseline here is the environment-gauged Kraus truncation. The Cheng-order baseline was not run.

## 6. What is established and what is not

Established, for dephasing on `ising` at N = 8 and 10:
- The LPDO machinery, the dense Lindblad reference and the gate audit are correct (Test 0).
- The spcf-repairable cross-cut residual survives and grows with dephasing (Test 1).
- The bond tilt carries that gain through to physical-RDM errors in the bond-dominated window (Test 2c and 4b). Trace distance does not degrade.

Not established:
- Whether the gain survives at larger N, other models, and depolarizing noise.
- Whether an LPDO with spcf beats DMT-MPO at equal parameters, which is the real positioning question.
- The value of the Kraus-leg tilt, which Test 1 says matters for the energy and at the Kraus floor. It was not built.
