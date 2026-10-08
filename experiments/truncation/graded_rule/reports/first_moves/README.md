# First moves for the cut-local truncation (spcf) application directions

This folder holds the second-pass investigation of the directions ranked in
`../Cut local truncation applications.md`. Rank 3 (quantum trajectories) is out of scope by request.

The first pass rested on abstracts only. This pass used the paper-search MCP server, which reached full texts.
One agent per direction did four things:
- re-checked the report's claims against the papers;
- searched for prior art;
- named the code entry points in this repo;
- planned a first move and 1-3 cheap tests, with success and kill criteria fixed before any run.

No benchmarks were run in the planning step. Each direction's first tests are now being executed, and the results go to `<direction>_results.md` in this folder.

## Verdicts at a glance

| Direction | Notes | Verdict | What changed versus the report | First move | Decisive first test |
|---|---|---|---|---|---|
| Rank 1: OTOC and operator fronts | `rank1_otoc.md` | conditional | SVD-MPO is already exact at and ahead of the front (χ = 4 matches χ = 32 for C ≲ 0.1), so any gain lies behind the front. The "halting" failure in 1901.05793 may be lost norm on an unrenormalized MPO. The targets are diagonal Pauli-label marginals, so a d = 4 port is cheap. Time splitting (2202.07060) is a baseline the report missed | `rank1/op_tebd.py`, `spcop.py` (diagonal-target tilt) | Does renormalized SVD still err behind the front (rms ΔC ≥ 0.02)? If not, kill |
| Rank 2: purification at finite T | `rank2_purification.md` | conditional | The bar is SVD on a backward-gauge purification, which DMT's own paper finds about as good as DMT. At high T there may be nothing to repair. DMT's conclusion already proposes "controlled-metric truncation" and must be cited. Wrong ID: KBM is 1111.4508 | `rank2/spcfpur.py` (d = 4 site+ancilla), driver with a 4^N reference | μ scan from pure to mixed: ≥ 2x on rdm2 at μ ≤ 1 against SVD in its best gauge |
| Rank 4: reference-state MPO | `rank4_refstate.md` | conditional (cheap; not really spcf) | Exact as "DMT with a swapped covector", so no tilt is needed. The closest prior art is 2609.12840 (Plenio group, Pauli-propagation twin). Wrong ID: xSPD is 2409.03097, not 2506.13241 | `rank4/heis_mpo.py` with SVD, DMT-I and DMT-σ0 | DMT-σ0 ≥ 3x better than both SVD and DMT-I on ⟨Néel\|Z(t)\|Néel⟩; also compared with MPS-TEBD at matched memory |
| Rank 5: multi-target DMRG | `rank5_dmrg.md` | conditional, leaning weak | Truncation only picks a basis; targets are re-solved afterwards, so a gain may wash out. A tilt may only rediscover better target weights. The "unpredictably" quote is not in either cited paper. Best sub-case: time-step-targeting tDMRG | `rank5/mt_cut.py`, a dense single-cut harness | A1: beats the oracle-best weight by ≥ 1.5x; A2: keeps ≥ 50% of the gain after re-solving |
| Rank 6: noisy/LPDO | `rank6_noisy.md` | conditional, weak standalone | The LPDO pain points are Kraus-dimension growth and the fidelity collapse at depth ≈ 0.17/ε, not the bond cut. The local-positivity niche is covered by DMT. "MPDO" in Cheng et al. means LPDO | LPDO-TEBD driver plus an audit with no tilt | Does SVD's repairable residual survive γ = 0.01? |
| Rank 7: 1D benchmarking | `rank7_benchmarking.md` | weak to conditional | A 2.5-5x error drop converts to only r ≈ 1.6 in χ. Memory saving is about 2.5-3x; wall-clock is 3-9x worse today. The D-Wave quote is from 2403.00910. The verification consumers use fidelity-based extrapolation, which spcf decouples from local error | existing `pareto_spcf.py`, then `rank7/ladder_t.py` | T2: does spcf beat SVD plus free post-processing (F-rescaling, log F and 1/χ extrapolation)? |
| Ranks 8 and 9: chemistry and post-hoc compression | `rank8_9_chemistry_compression.md` | weak (8 as written, 9); conditional (8 reformulated) | NEVPT2 barely depends on M′. The real low-M damage is false intruder states, a global effect. N-representability is no advantage, since SVD has it too. Rescue: block-renormalized cross-cut targets | `rank8_9/compress_bench.py` | T0: what share of the energy-weighted RDM error lies in the window? Below 25%: kill the window version |
| Rank 10: infinite-T transport | `rank10_transport.md` | weak as an application, useful control | Targets are linear, so the right tool is a closed-form window-weighted SVD (wsvd), not a GN tilt. One niche: span-4/5 strings that DMT ℓ = 3 ignores | `rank10/` wsvd, DMT ℓ = 3/5 and rTEBD arms | Single cut: wsvd ≥ 2x better on span-4/5 than DMT ℓ = 3 at equal span ≤ 3 error |
| Cross-cutting: whitening and novelty | `crosscut_whitening_novelty.md` | novelty resolved; whitening conditional | Paeckel [73] is Stoudenmire's fidelity note and [74] is DMT, so there is no threat. The review's Sec. 9 suggests the idea, so cite it. 1-2x SVD cost is unreachable with region-based metrics; only a target-free core metric could get there | `whiten/whiten_probe.py`, `SPCWhite` | T1: does the oracle Kronecker GN step keep ≥ 60% of the exact GN drop? Below 35%: kill the closed form |

## Findings that cut across directions

1. **The core novelty claim survives full-text reading.** No pure-state MPS truncation that targets local RDMs was found. The Paeckel [73]/[74] threat is resolved. Two framing obligations remain:
   - cite DMT's "controlled-metric truncation" remark (1707.01506, Discussion) as the origin of locality-weighted truncation;
   - cite Paeckel Sec. 9 as an anticipated direction.
2. **The linear-versus-quadratic dividing line holds, and it sharpens.**
   - Where targets are linear (ranks 4 and 10, and rank 2 at high T in the backward gauge), the right object is a closed-form construction: a DMT reserve, a swapped covector, or a window-whitened SVD. A Gauss-Newton tilt is the wrong tool there.
   - spcf's tilt is native only where targets are quadratic (ranks 1, 2, 5 and 6).
3. **Baselines are stronger than the report assumed.** Each direction now has to beat a specific baseline:
   - rank 1: renormalized SVD and time splitting;
   - rank 2: SVD in the backward gauge;
   - rank 4: Schrödinger MPS-TEBD at matched memory;
   - rank 5: an oracle weight scan;
   - rank 6: DMT on a plain MPO;
   - rank 7: post-processed SVD.

   Every first test is designed against that stronger baseline.
4. **Cost is the binding constraint.**
   - The χ reduction at equal error is r ≈ 1.6 (rank 7), so wall-clock only pays if the per-cut overhead drops below about 4x.
   - The whitening study puts the floor at about 5x for any metric that needs the region state.
   - Getting to about 2x needs a target-free core metric. That is the deciding open question for any wall-clock claim.
5. **Report corrections to carry into any write-up:**
   - KBM is 1111.4508, not 1205.3756.
   - xSPD is 2409.03097, not 2506.13241.
   - 0706.2480 is about TFIM, not XXZ.
   - The D-Wave quote is from 2403.00910, not 2503.05693.
   - The multi-target "unpredictably" quote is unattributable.
   - DMT χ_preserve = 2^ℓ (8 for ℓ = 3), counted inside the cap.
   - DMT positivity is only ameliorated, not guaranteed.
   - DMT's Hamiltonian is the repo's `ising` model divided by 4.
6. **Environment notes.**
   - `dmt_baseline.py` cannot run in this environment (`gcg` and `pytreenet.special_ttn.pauli` are missing), so ranks 2, 4, 6 and 10 need a standalone numpy DMT.
   - scipy was missing and has been installed.
   - In the paper-search server, `read_arxiv_paper` worked for every ID tried, while `search_semantic` and `search_openalex` always returned empty results. Google Scholar search was the working discovery channel.

## Suggested order

The first tests are cheap, and each comes with its own kill criterion. Three runs carry the most weight:
- **Whitening T1** decides whether any wall-clock claim is possible.
- **Rank 1 Test 1** decides whether the OTOC direction has anything to fix.
- **Rank 2 Test 1** decides whether purifications have a usable error window.

Ranks 4 and 10 test the linear-target constructions. Ranks 5, 6, 7 and 8/9 are lower-priority probes, and each can be dropped on its first test.

## Status after the first execution round (agents pruned to the top 3)

Token use was high, so the execution agents were pruned to the three directions with the best partial evidence. The stopped agents left code and partial results in their folders. Each partial result below is my reading of their logs, not a finished test.

| Direction | Status | Partial evidence at stop |
|---|---|---|
| Rank 7 (`rank7/`) | **continuing** | T2 passed every pre-registered check (N = 20 Ising, `rank7/results/t2_analysis_g001.txt`):<br>• spcf beats the best post-processed SVD at equal χ in 88-100% of growth-phase samples;<br>• all-pairs ZZ error ε_c is 0.30-0.32x SVD (median);<br>• χ reduction r(t) peaks at 1.9-2.1.<br>T3 (kicked Ising) was running. |
| Rank 1 (`rank1/`) | **continuing** | Test 1 continued: renormalized SVD still errs behind the front (rms 0.07 at χ = 8).<br>Test 2 was ambiguous: static ratios 0.5-0.8 (0.23-0.37 with 3 GN passes).<br>Test 3 was neither success nor kill: C_Z and w ratios 0.60-0.85, contour lag cut by 31-48% at χ = 8-16, infidelity unchanged, about 3.7x SVD CPU. |
| Rank 2 (`rank2/`) | **continuing** | Test 0 done. The SVD μ-scan is done and found usable cells at μ ≤ 1 in the backward gauge (e.g. stag μ = 0.25, T = 4: rdm2 2e-3 to 5e-2). The spcf arms had not run yet. |
| Rank 4 (`rank4/`) | stopped | Test 1 (Ising N = 16, T = 6) is noisy and non-monotone in χ. DMT-σ0 beats SVD-renorm and DMT-I by roughly 1-3x at some χ, not ≥ 3x consistently. Leaning toward the kill or ambiguous branch. |
| Rank 5 (`rank5/`) | stopped | A1 (Ising N = 12): spcf-multi/SVD is 0.75-1.07, but the oracle weight arm (ii+) is as good or better. This matches the kill branch "the gain is just reweighting". |
| Rank 6 (`rank6/`) | stopped | Test 0 passed. Test 1 (no-tilt audit) was mid-run; no verdict. |
| Ranks 8-9 (`rank8_9/`) | stopped | T0 sent it to T1. In T1 on the PPP proxies, spcf is *worse* than SVD and variational fitting at matched χ (e.g. site-PPP χ = 12: hw2 0.043 vs 0.028). That points to kill. |
| Rank 10 (`rank10/`) | stopped | Infrastructure and unit tests only; no test result. |
| Whitening (`whiten/`) | stopped | Math validation and T0 profiling only; T1 was not finished. It is still the key open question for any wall-clock claim, and the first thing to restart if budget allows. |

## Final verdicts of the execution round

The full write-ups are in the `*_results.md` files. Directions not listed here stopped at the status in the table above.

| Direction | Verdict | Key numbers |
|---|---|---|
| **Rank 2: purification** | **CONTINUE (lead result)** | Test 1 passes: spcf beats SVD in its better gauge by a median 2.5-4.7x on rdm2/nn at μ = 1, 0.5 and 0.25, in 15/18 cells, and the gain does not fade toward mixed states. Test 2 passes: spcf-purification beats numpy DMT at equal parameters by 2.9-7.8x on rdm2 and 2.5-4.4x on nnn, in 7/7 cells; N = 12 agrees. DMT keeps the lower energy drift (35/36 entries). **Caveats:** (i) the Heisenberg control gains 2.4-2.9x in the back gauge, so part of the gain may be SVD gauge inefficiency; (ii) the nnn lead over DMT shrinks to 1.1-2.2x at μ = 0.1; (iii) the gate as calibrated is useless |
| **Rank 6: LPDO with dephasing** | CONTINUE (conditional) | Bond truncation carries 80-100% of the nn error at γ = 0.01. On the pre-registered κ = 4 grid the result is ambiguous: the best adjacent pair is 1.86/2.13x. A post-hoc κ = 8 extension gives 2.2-3.8x vs env-gauged SVD-LPDO at γ = 0.01-0.1, and replicates at N = 10. Energy is up to 2x worse at low χ, and far ZZ is worse at χ = 16 |
| Rank 7: benchmarking | memory claim only | r = 1.5-1.8 (memory 2.3-3x). T2 passes ungated. Wall-clock is not shown (best cold equal-error CPU ratio is 1.16-1.33 at χ = 64). The circuit transfer (kicked Ising) is **killed**, with r ≈ 1.0 |
| Rank 1: OTOC | PARK | Test 3 gives 0.6-1.07x and no 2x at any χ. Free label reweighting (rTEBD, γ = 2) nearly matches it. Lessons kept: renormalize the operator MPO, and run reweighting as a baseline |
| Whitening | PARK | Kronecker GN is dead (deployable ρ ≤ 0.1). The whitened SVD keeps 0.39-0.75 of the log-gain end-to-end, but only with oracle factors. No cost win is shown. Revive only via target-free factors at ≤ 3x SVD |

**Where this leaves the project.** The defensible result is spcf on purifications: a pure-state, fixed-χ tilt that beats both SVD and DMT at equal parameters on local correlators of mixed-state dynamics. The LPDO noise result is consistent with it. Before claiming it, the open items are:
- explain the Heisenberg-control gain with a disentangler-gauge SVD baseline;
- test a thermal Gibbs quench;
- run the rank 6 DMT head-to-head, which shares the same `dmt_np.py`.

No cheap closed form exists yet, so any claim stays at the level of accuracy and memory, not wall-clock.

## Final verdict (after the cost and Pareto checks)

**What spcf is.** Plain SVD keeps the χ components of a cut that best preserve the whole state. spcf keeps the same χ but tilts the kept subspace so the 1-3-site reduced density matrices around the cut are preserved.

**What holds up.** At equal χ, spcf gives lower local error than SVD:
- 2.5-4.7x lower on purifications at finite temperature (rank 2), and also lower than DMT there;
- 2.2-3.8x lower on noisy LPDO dynamics (rank 6, partly from a post-hoc setting);
- about 3x lower on pure-state Ising quenches.

**What does not hold up.**
- **Time.** At equal error, spcf is about 10x slower than SVD (rank 2, N = 10). On the time Pareto front SVD dominates, because a larger χ wins.
- **Peak memory.** The cut builds a window state of about 4096·χ² numbers with the a = 2 window, which is 1 GB at χ = 64. Its peak RAM is therefore *higher* than SVD's at equal error. The earlier "2-3x memory saving" holds only for the *stored* state, not for peak memory.
- **Cost scaling.** With a = 2 the per-cut overhead grows from 150x to 300x SVD as χ goes from 8 to 64. With a = 1 it stays flat at about 11x SVD, but the accuracy at a = 1 is untested.
- **The closed-form speed-up** (whitening) did not deliver: it was parked.

**Directions.**

| Direction | Verdict |
|---|---|
| Ranks 1 (OTOC), 4, 5, 8-9, 10 | Dropped or parked |
| Rank 7 (circuits) | Killed |
| Rank 7 (quenches) | Memory claim only, now weakened by peak memory |
| Rank 2 (purifications) | Best accuracy result |
| Rank 6 (LPDO) | Promising only alongside rank 2 |

**Bottom line.** spcf is a real accuracy-at-fixed-χ improvement for local observables, but **it is not yet practical**: SVD at a larger χ reaches the same accuracy faster and with less peak RAM. Two cheap tests decide whether that can change:
1. Does the a = 1 window keep the gain?
2. Does building the window state in chunks bring peak memory down to the SVD level?

If both fail, write spcf up as a diagnostic or accuracy result, not a practical method.
