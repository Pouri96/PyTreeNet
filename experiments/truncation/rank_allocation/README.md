# Experiment for the truncation paper's budget figure

The data and the code that draw `figures/budget_frontiers.pdf`: infidelity at the final time
against stored parameters, for the maximum bond dimension `chi` and the bound `P_max`, on six
systems. The folder needs only this repository. Run everything with this folder as the
working directory:

```
python run.py all four              # the bond-cap (per_bond) ladder for one cell
python dp_sweep.py four             # the DP allocation at 12 parameter targets
python dp_figure.py                 # figures/budget_frontiers.pdf and .png
```

`results/<cell>.json` and `results/dense/<cell>.json` hold the recorded rows the figure draws,
so `python dp_figure.py` alone redraws it.

## Files

| file | what it is |
|---|---|
| `cells.py` | the six systems with their time window, warm-up and `chi` grid |
| `models/` | the physics: `spin_boson`, `spread_models` (pyrazine, Holstein) |
| `run.py` | the driver; one `BUGConfig` for every arm, so arms differ only in their knob |
| `dp_sweep.py` | the DP allocation sweep; run after `run.py per_bond <cell>` |
| `dp_figure.py` | the figure |
| `_rage_vendor/` | the BUG integrator, copied from RAGE so this folder runs without it |
| `_pytreenet_root.py` | puts this repository's `pytreenet` ahead of any other installed copy |

## How runs are compared

A run is two numbers: its stored parameters and its infidelity at the final time. The parameter
target is never used to pair runs, since a run does not land on its target exactly.

## Settings shared by every arm

- Each start is widened by warm-up passes (BUG's basis update without the Galerkin step).
- `rel_tol = 1e-12` and `total_tol = 1e-14`. A cut keeps `s > max(rel_tol * s_max, total_tol)`,
  so these act as a common floor for every arm.
