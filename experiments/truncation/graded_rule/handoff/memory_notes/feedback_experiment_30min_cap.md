---
name: feedback-experiment-30min-cap
description: No single experiment or background job may run longer than 30 minutes of wall time. Size the run to fit, or kill it.
metadata:
  type: feedback
---

No experiment (foreground or background) may run longer than 30 minutes. User said so directly on 2026-10-07 after two N=16 global-fit workers had been running for hours (about 7 CPU-hours each, empty log).

**Why:** hours-long runs on the user's laptop starve the machine, stall through Modern Standby, and return nothing usable. They also cannot be steered mid-run.

**How to apply:** before launching, estimate the cost from a smaller cell (CPU seconds per step times steps) and shrink N, T, taus, window size or iterations until it fits in 30 minutes. Launch with a `timeout` of 1800 s. If a job is past 30 minutes, kill it and its orphaned multiprocessing workers (check `Get-CimInstance Win32_Process` for children of the launcher, they survive a launcher kill). Report a cell as not run rather than letting it drift. Related: [[feedback_kill_stalled_token_burn]], [[project_graded_trunc_window_form]].
