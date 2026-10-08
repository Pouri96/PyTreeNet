# Rank 5 first moves: multi-target and correction-vector DMRG truncation

## Summary verdict: conditional, leaning weak

The idea in the applications report survives the full-text reading. Every DMRG truncation rule we read keeps the leading eigenvectors of a weighted ρ = Σ w_f ρ_f, or equivalently the SVD of the stacked matrix [√w_f ψ_f]. The weights are set by hand: equal weights, 0.1/0.3/0.3/0.3, or 1/3,1/6,1/6,1/3 found by trial. Nobody chooses the kept subspace to preserve local RDMs. The open Paeckel refs [73]/[74] turn out to be Stoudenmire's density-matrix MPO-MPS note and DMT itself, so they add no new prior art.

The full texts also show a structural problem the report missed. **In DMRG, truncation chooses a basis; it does not produce the deliverable.** After each truncation the targets are re-solved in the new basis (eigensolver, correction-vector (CV) solve, or time step). Observables near the centre are measured on the untruncated superblock. Spectral functions G(ω) = ⟨A|x(z)⟩ come from that re-solve, and DDMRG converges as ε² (variational).

spcf preserves the local RDMs of the *projected* targets. In DMRG that is only a proxy for "a basis in which the re-solved targets have accurate local RDMs", and nothing guarantees the first-order gain survives the re-solve. A second confound: under a shared basis, a tilt from the stacked-SVD point changes individual target fidelities at first order. spcf can therefore win simply by learning a better target weighting, which a one-parameter weight scan would also find.

The direction is worth one day of dense, single-cut tests, A1 and A2 below, which settle both issues. If spcf beats the best hand-tuned weight on projected RDMs and keeps at least half that gain after re-solving, build the sweep harness. Otherwise drop rank 5, or fold it into rank 8/9 (post-hoc compression).

The best-fitting sub-case is new to the report: **time-step-targeting tDMRG** (Feiguin-White). It is multi-target, its weights are heuristic, its error metric is literally a local-magnetization RMS, and it runs in spcf's home regime (the growth phase of dynamics).

## Literature, verified from full text

Read in full or in the relevant sections (PDFs in `/tmp/papers_rank5_dmrg`):

| Paper | What was read | Verified content |
|---|---|---|
| Jeckelmann, **0808.2620** (Prog. Theor. Phys. Suppl. review) | Full text, §§1-2 | Eq. (1.12) ρ = Σ c_s ρ_s with Σ c_s = 1, "limited to a small number M of targets (of the order of ten) because DMRG truncation errors grow rapidly with the number of targeted states". Lanczos-vector DMRG spectra "can be quite inaccurate … often depends strongly on where the superblock is split". DDMRG error is ε² vs ε for the CV method (§2.3). Fig. 4: 100-site Heisenberg energy error vs m. |
| D'Azevedo, Elwasif, Patel, Alvarez, **1902.09621** | Full text | Multi-target via SVD of the stacked X_{f,α;β} = √w_f ψ^f (Eq. 2, Eq. 6). Weights used: **0.1 GS, 0.3 c_r\|gs⟩, 0.3 each for Re/Im CV** ("Other weight values could have been used; see the discussion in [Kühner-White]"). 32×2 Hubbard ladder, U = 4, n = 0.9375, m ≤ 2000. The paper is about *performance*: the DM step falls from 40% to 5% of runtime, about 1.25x overall. It says nothing about accuracy vs targets. |
| Kühner & White, **cond-mat/9812372** | §§III-VI | Lanczos-vector weights: 50% to the GS, the rest distributed by spectral weight w_n (Eq. 17). The CV method targets \|0⟩, \|A_q⟩, Re x, Im x. Spin-1/2 chain, L = 80, η = 0.1, m = 128/256. Table IV: discarded weight grows 9.4e-8 → 6.2e-6 as Lanczos targets go 4 → 8 at m = 256. Two CVs Δω = 0.2 apart are "appropriate" for η = 0.1. |
| Jeckelmann, **cond-mat/0203500** (DDMRG) | Error-analysis section | Errors of order ε in E0 or \|A⟩ give errors of the same order in I_A, so \|A⟩ must be targeted. A second CV with η ≈ (ω2-ω1)/4 is targeted "to make the procedure robust". |
| Feiguin & White, **cond-mat/0502475** (time-step targeting) | Fig. 1 and the weights discussion | Targets ψ(t), ψ(t+τ/3), ψ(t+2τ/3), ψ(t+τ). Fig. 1 scans weightings (1/2,1/6,1/6,1/6; 1/2,1/8,1/8,1/4; equal; 1/3,1/6,1/6,1/3) at τ = 0.05 and 0.1, Haldane chain L = 32, m ≤ 100. The error metric is E(t) = √(1/L Σ_x (S^z(x,t) − S^z_exact)²), a local observable. "The best weighting we have found is w1 = w4 = 1/3, w2 = w3 = 1/6." |
| White, **cond-mat/0508709** | Full (4 pp.) | The ID is correct: single-centre-site DMRG with ρ' = ρ + a Σ_α Â_α ρ Â_α†, a ≈ 1e-4 … 1e-2. |
| Hubig et al., **1501.05504** (DMRG3S) | §§IV-VI | Expansion with mixing α, then SVD truncation. The adaptive α keeps ΔE_T ≈ −0.3 ΔE_O. ΔE_T can be "very small or even negative". "If … the aim is to extrapolate in the truncation error … a fixed value for α is of course absolutely necessary." |
| Gleis, Li, von Delft, **2207.14712** (CBE) + comment **2403.00562** + reply **2501.12291** | Grep of truncation passages; comment and reply openings | Every truncation is SVD-based (preselection, final selection, then the usual SVD). The reply concedes the post-update truncation "may slightly increase the energy", so CBE step (iv) is not variational. The dispute is over the tangent-space projection and randomized SVD, not the truncation criterion. |
| Núñez-Fernández & Torroba, **1908.10880** (zero-site) | §III | Truncation keeps the m largest singular values of the enriched M̃. On the enrichment: "β … assigns the same weight to all the states used for enrichment; this is not necessarily the optimal choice". |
| Dorando, Hachmann, Chan, **0707.3121** | §II | State-averaged Γ = Σ w_i ψ^i ψ^i†, "typically we choose equal weights". |
| Nocera & Alvarez, **1607.03538**, **2204.03165** | Intro; App. A | Krylov-space CV in DMRG++ uses the same multi-target weighted ρ or stacked SVD (Eq. A4). Typical runs: η = 0.1 (0.075, 0.02), m = 100-3000, m_min = 64-200, truncation error ≤ 1e-7 … 1e-8, L = 48 chains and 50×2 ladders. |
| Paeckel et al., **1901.05824** | §2.8 and the reference list | Ref. [73] is Stoudenmire's "MPO-MPS multiplication: density-matrix algorithm" (tensornetwork.org). Ref. [74] is White-Zaletel-Mong-Refael PRB 97, 035127, i.e. DMT (1707.01506). **Neither is a pure-state local-RDM truncation.** |

Downloaded but not read beyond the title: none. Not retrieved: the Jeckelmann ISSP slides and EPJB 2023 (the "energy ∝ ε, observables ∝ √ε" claim remains unverified from those sources). 0203500 states the related textbook fact that energy error is O(ε²) in the wavefunction error.

**Corrections to the applications report.**

1. The quote "depend[s] significantly and sometimes unpredictably on the specific states included as target" appears in **neither** 0808.2620 nor 1902.09621 (both read in full). It is misattributed or comes from a snippet. Replace it with the verified 0808.2620 sentences above, plus Feiguin-White Fig. 1, which is direct evidence that weights are tuned by trial.
2. "At a converged variational minimum any tilt can only raise the energy" is too strong. SVD truncation is not energy-optimal: DMRG3S reports ΔE_T ≤ 0 cases, and the CBE reply concedes truncation is the non-variational step. `spcfast.py` even carries an energy row (`wE`). Ground-state sweeps remain the right negative control, but for a different reason. The eigensolver re-optimizes the deliverable right after truncation, so a truncation tilt can only change the *basis*, and at convergence the discarded weight is small enough that the gate should decline.
3. 1902.09621 concerns runtime, not accuracy. It supports "weights are heuristic" (0.1/0.3/0.3/0.3), not "accuracy is unpredictable".
4. In MPS-native codes, CV, Chebyshev and Lanczos calculations usually keep **separate MPS per vector** (Dargel et al. 1203.2523 explicitly avoids multi-targeting, from its abstract). There the shared-basis weighting problem disappears; what remains is a single-target CV compression whose deliverable is an *overlap* ⟨A|x⟩. The shared-basis problem lives in DMRG++/block-style codes, state-averaged chemistry DMRG and time-step targeting.

## Prior art and novelty risk

- **Closest linear analogue (moderate risk, and a mandatory baseline):** operator-dressed density matrices. White's ρ + a Σ Â ρ Â† and the old practice of targeting O|ψ⟩ (Kühner-White target A|0⟩ precisely so that ⟨A|x⟩ is accurate) both add operator-dressed states to the density matrix. If P|ψ⟩ lies in the kept space, the term ⟨ψ|(1−Π)P|ψ⟩ of the first-order error of ⟨P⟩ vanishes. So "dressed ρ with window Paulis" is a closed-form, SVD-cost competitor to spcf, and Test A1 must include it.
- **DMT (1707.01506)** is confirmed as Paeckel [74]. It is the linear, mixed-state parent, not a DMRG multi-target method.
- **Nothing found** that chooses a shared multi-target basis by local-RDM criteria, nor a tangent-space tilt in DMRG truncation. Searches covered arXiv, Google Scholar and Semantic Scholar with queries on multi-target weights, observable-preserving truncation and state-averaged excited states. Semantic Scholar and OpenAlex returned empty results (likely rate limits), and arXiv keyword search was noisy, so the negative result is moderately strong, not exhaustive.
- **Novelty is weakened if spcf merely reweights targets** (confound above). A closed-form "per-target-error-equalizing weight" heuristic would then capture the gain, and it is an obvious idea.

## Technical details for implementation

- **Shared-basis truncation (R sweep):** stack M_f (l·2 × 2·r) for the targets, take the shared left basis Q (rank χ), and project each target as Q Q†M_f. The tangent chart is Q = qr(B_k + B_⊥C) with *one shared C*. For each target, `spcfast`'s R-branch linearization applies with Q1 = B_k†M_f and Q2 = B_⊥†M_f: dM_f = B_⊥ C Q1_f + B_k C† Q2_f. The residual stacks over targets as r = ⊕_f √ω_f F(h(ρ_f(Q)) − h(ρ_f^exact)), with RDMs normalized per target (`hnorm` already divides by the trace, which handles the unnormalized CV vectors). The environments Lm/Rm are shared across targets because all targets share block bases. The multi-target generalization of `jvp`/`vjp` is a loop over f with a shared C, about 60 lines.
- **Static-only targets:** `SPCFast(model, N, taus=(), fw=0.0, ...)` keeps only τ = 0 rows. Checked in seconds in a scratch venv: an N = 8 random MPS, cut b = 3, χ = 4 fired in 0.26 s and returned the (8,2,4),(4,2,8) tensors. The run also showed the discarded weight rising 0.37 → 0.53 on a random state with fw = 0, so the "fidelity unchanged to first order" claim fails at large tails. Use fw > 0 or the gate there.
- **Transition RDMs:** with A local at site c, G(ω) = ⟨0|A†|x⟩ = Tr[A† ρ_c^{x,0}], where ρ^{x,0} = Tr_out |x⟩⟨0|. The deliverable is therefore a 1-site *transition* RDM, which is bilinear across targets and non-Hermitian. `hvec` handles Hermitian matrices only, so a transition variant needs full complex vectorization (about half a day).
- **Models:** the repo's `ising` (hx = 0.9045, hz = 0.809, chaotic, gapped) and `ising2` (1.4, 0.4) via `mpsenh.h_bond`, with `exact_ref.sparse_H` for dense ED. Add a near-critical TFIM (hx = 1.0, hz = 0.1) in the harness, not in `mpsenh.FIELDS`. Use `heis` as the known-bad model.
- **Target sets (literature-typical):** (S1) GS only, the negative control. (S2) the lowest 3 eigenstates, state-averaged with equal weights. (S3) CV {\|0⟩, σ^z_c\|0⟩, Re x, Im x}, η = 0.2 at N = 12 (literature uses η = 0.1 at L = 48-80; scale it with the level spacing), ω at the main peak and at half maximum, with weights 0.1/0.3/0.3/0.3. (S4) time-step targeting {ψ(t), ψ(t+τ/3), ψ(t+2τ/3), ψ(t+τ)}, τ = 0.1, weights 1/3,1/6,1/6,1/3, Néel quench of `ising` at t ∈ {1, 2, 3} (growth phase).
- **Error measures used by others:** E(t) RMS of S^z(x) (Feiguin-White), peak position and weight of S(q,ω) vs m (Kühner-White), and truncation error vs m. Add per-target 1-3-site trace distance (near and far from the cut), per-target fidelity, eigen-energies and G(ω).
- **Known failure modes:** too many targets inflate ε quickly. CV accuracy is limited by E0 and \|A⟩ accuracy, so target \|A⟩. Changing the truncation rule breaks ε-extrapolation (DMRG3S), so report results at fixed χ, not extrapolated. spcf fails on SU(2)/Heisenberg. Far-observable collateral appears at first order.

## Where to start

**Do not start in `pytreenet/dmrg`.** `DMRGAlgorithm._update_two_site` (`pytreenet/dmrg/dmrg.py`, lines 170-216) is single-target (`davidson(..., nroots=1)`) and truncates through `state.split_node_svd(..., svd_params)` (`core/ttn.py:1476`). A multi-target insertion needs three things:
1. a target leg on the centre tensor;
2. `nroots > 1` or a CV solve;
3. a pluggable `truncator(theta_list, ...)` replacing `split_node_svd`, plus TTN-leg-to-(l,2,2,r) permutations to reuse spcf.

That is 3-5 days, and only worth it after the tests below pass.

**First move (about 1 day of code, under 30 min of compute):** a self-contained dense harness, `experiments/truncation/graded_rule/rank5/mt_cut.py`.
1. ED for the target sets above, using `exact_ref.sparse_H` and `scipy.sparse.linalg.eigsh`. For CV, solve ((H−E0−ω)²+η²) x_Im = −η A\|0⟩ by CG, as in Kühner-White Eq. 24.
2. Single-cut setup: at cut b, build *exact* shared left/right bases from the stacked targets (no truncation elsewhere), giving θ_f (l,2,2,r) and exact window tensors.
3. Truncation arms at χ:
   - (i) stacked SVD with literature weights;
   - (ii) a weight grid (GS weight ∈ {0.05, 0.1, 0.25, 0.5}, rest equal; all four Feiguin-White patterns for S4) with an oracle pick;
   - (iii) dressed ρ, ρ + a Σ_P PρP† over 1-site window Paulis, a ∈ {1e-3, 1e-2, 1e-1};
   - (iv) **spcf-multi**, equal per-target residual weights, started from (i);
   - (v) per-target separate SVD (unshared, F× parameters), as a reference ceiling.
4. Implement (iv) first as a dense Gauss-Newton on C with a finite-difference Jacobian. At N = 12 C has at most about 900 real parameters, so this takes seconds. That doubles as the reference implementation. Then port it to an `SPCMulti` class in `rule/` (copy of `SPCFast` with the f-loop in `jvp`/`vjp` and a shared C) for the sweep stage.
5. Log `spcfast`'s gate quantities (Bres vs c0·(tail/1e-4)^0.65) per target set.

## First tests (decision tree)

Common setup: N = 12 (and 14 for confirmation); cuts b = N/2−1 and b = N/2−3; χ ∈ {4, 6, 8} (exact centre rank 64); models `ising`, near-critical TFIM, `heis` (negative). Primary metric E_near = max over targets of the max 1-3-site trace distance within ±2 sites of the cut. Secondary metrics: far drift (windows beyond ±2), per-target fidelity, eigen-energies, G(ω) relative error, and S^z RMS for S4.

**A0, gate pre-check (minutes; uses the A1 harness).** For each target set and χ, record the SVD cut's Bres/cost ratio.
- Kill: Bres < cost at more than 80% of (set, χ, cut) cells for S2-S4. This would mean multi-target tails are in the regime where spcf historically loses.
- Expected: S1 mostly gated off (small ε); S3/S4 open at χ ≤ 6.

**A1, single-cut projection test (about 10 min compute).** Compare arms (i)-(v) on the *projected* targets.
- Success (pre-registered): in ≥ 2 of S2-S4 and ≥ 2 of 3 χ values, spcf-multi has E_near ≤ 0.5× arm (i), ≤ 0.67× the oracle-best arm (ii), and ≤ 0.67× the best arm (iii). Per-target fidelity loss must stay ≤ 10% of the SVD discarded weight, and far drift must not exceed SVD's by more than 1.5x.
- Kill: the gain over the oracle weight is < 1.3x everywhere, or the dressed ρ matches spcf within 1.3x. Either outcome means the effect is reweighting or linear targeting, both cheap and not novel.
- Negative control: S1 (GS) gain ≥ S2-S4 gain means the multi-target framing adds nothing. Merge into rank 9.
- Ambiguous: spcf beats (i) but not the oracle (ii). Then check whether the spcf solution's *effective* per-target fidelities correspond to some weight vector. If so, write a closed-form error-equalizing weight rule instead and drop the tilt. Also ambiguous: per-target RDMs improve but G(ω) does not. Then add transition-RDM targets (half a day) and rerun S3 only.

**A2, re-solve survival test (about 10 min compute; run only if A1 passes).** Same cut and arms. Project H onto (kept left basis ⊗ full right space), with dimension ≤ χ·2^{N−b−1} ≤ 512. Re-solve the targets in that space (eigenstates for S2; E0, A\|0⟩ and CV for S3; one RK4/Krylov step of τ for S4) and score the *re-solved* targets against exact.
- Success: spcf keeps ≥ 50% of its A1 log-gain on E_near, while eigen-energy and G(ω) errors are no worse than 1.5x SVD.
- Kill: all arms agree within ±20% after re-solve (the truncation rule washes out). The direction then reduces to "final compression of a stored multi-target MPS", so stop and fold it into rank 8/9.
- Ambiguous: the gain survives for S4 (time stepping is not a re-optimization to a fixed point) but not for S2/S3. Pursue the time-step-targeting branch only.

**B, sweep test (2-3 days of code, about 1 h compute; only if A2 passes).**
- Harness: numpy two-site DMRG in the `mpsenh` conventions (list of (l,2,r) tensors, MPO environments, dense effective H for 4χ² ≤ 1024), state-averaged S2 and CV S3 (or TST S4 if only that passed), N = 16-20, χ = 8-16, 8 sweeps, with ED references.
- Arms: (i), the best (ii), (iii), and spcf-multi in every sweep vs final sweep only.
- Success: converged local-observable error per target is ≥ 1.5x better than the best weight arm, with energies/G(ω) no worse than 1.5x, and runtime overhead < 3x per sweep.
- Kill: the gain is < 1.2x at the fixed point.

**C, negative control (embedded in A1/A2/B as S1).** Single-target GS. Expected: no gain, and slightly higher energy. A clear GS gain would be a surprise worth re-examining, because it would reopen rank 9.

## Open questions

1. Does spcf-multi's advantage reduce to an implicit optimal reweighting? A1's oracle-weight arm answers this. If yes, a closed-form weight rule is the product and the tilt is not.
2. Which deliverable to target for CV: per-target RDMs, or the transition RDM ρ^{x,0} that G(ω) actually reads? The latter is bilinear across targets and needs a non-Hermitian residual.
3. Is time-step-targeting tDMRG better framed under the dynamics ranks? It is the cleanest DMRG-family fit (local-observable metric, growth regime), but it may duplicate TEBD results unless re-solve survival differs.
4. How do target-residual weights ω_f get chosen without reintroducing heuristics? Equal weights on normalized RDMs is the default, and its sensitivity must be reported.
5. Does the far-observable collateral add coherently over a full sweep, as feared for uMPS? This is measured only in B.
6. For MPS-native separate-vector CV codes, is a single-target spcf with a reference-state transition target (as in rank 4) the better insertion than shared-basis multi-targeting?

Tool notes: `read_arxiv_paper` worked for new-style IDs but returned empty for old-style IDs (cond-mat/…). Those were fetched directly from arxiv.org/pdf and converted with `pdftotext`. Long outputs overflowed the tool limit and were parsed with `jq`. Semantic Scholar and OpenAlex searches returned empty lists. arXiv keyword search returned mostly off-topic hits. The repo's system Python lacks scipy, so the feasibility check used a scratch venv.
