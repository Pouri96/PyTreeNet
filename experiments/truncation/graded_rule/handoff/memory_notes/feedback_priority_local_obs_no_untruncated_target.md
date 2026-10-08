---
name: feedback-priority-local-obs-no-untruncated-target
description: For the truncation-method search, local-observable accuracy is the goal and infidelity is reported not gated. The method must not take its target from an expensive untruncated copy of the state.
metadata:
  type: feedback
---

Two standing rules from the user, 2026-10-07, for the lookahead / marginal-fit truncation search.

1. **Priority = local observables.** A loss of global state fidelity does not make a method bad as long as the local observables (1-, 2-, 3-site marginals, nn correlators) improve. Do not gate a verdict on "infidelity <= 1.15x SVD". Report infidelity as a side effect.
2. **Do not cheat with an untruncated copy.** The method itself must not use data from an expensive evolved untruncated (or exact dense) state as its fitting target. The user already pushed back once and asked for something smarter and cheaper (expansion with an importance-based stopping rule, which became the cut-local rule). The exact reference is allowed ONLY for scoring. Any target that comes from a larger working rank or a dense state must be named as such, with its memory cost, and not counted as a free win.

**Why:** the point of truncation is to avoid paying for the large dimension. A dense-state or 2x-chi target proves a ceiling but is not a method.

**How to apply:** label every result as one of (a) dense oracle target (ceiling only), (b) one-step working-rank target at f*chi (state the peak), (c) cut-local target from the two-site tensor already present in TEBD (peak = chi, legitimate). Judge the pass line on local-observable ratios at matched stored parameters and report infidelity next to them. See [[project_graded_trunc_window_form]], [[project_truncation_comparison_methodology]], [[feedback-experiment-30min-cap]].
