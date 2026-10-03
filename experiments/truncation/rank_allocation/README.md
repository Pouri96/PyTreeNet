# Experiments for `paper/Truncation`

Every number and figure in `paper/Truncation/rank_allocation.tex` is produced here. The folder
needs only this repository. Run everything with this folder as the working directory:

```
python run.py all four              # the bond-cap (per_bond) ladder for one cell
python dp_sweep.py four             # the DP allocation at 12 parameter targets
python cut_comparison.py            # Table I
python dp_figure.py --install       # the figure, copied into the paper
python dp_analysis.py [--margins]   # the DP allocation against the bond cap
```

## What produces what

| paper object | script | data |
|---|---|---|
| Table I, three cuts at one bond cap | `cut_comparison.py` | writes `results/cut_comparison.json` |
| Figure, infidelity against stored parameters | `dp_figure.py` | reads `results/<cell>.json` and `results/dense/<cell>.json` |
| Sec. III, no DP run lies above the cap's curve | `dp_analysis.py` | reads the same rows |

## Files

| file | what it is |
|---|---|
| `cells.py` | the six systems with their time window, warm-up and `chi` grid |
| `models/` | the physics: `spin_boson`, `spread_models` (pyrazine, Holstein), `tfim_tree` (Table I) |
| `run.py` | the driver; one `BUGConfig` for every arm, so arms differ only in their knob |
| `dp_sweep.py` | the DP allocation sweep; run after `run.py per_bond <cell>` |
| `dp_analysis.py` | compares the DP allocation with the bond cap |
| `dp_figure.py` | the figure |
| `cut_comparison.py`, `reference_truncations.py` | Table I and its two reference truncations |
| `_rage_vendor/` | the BUG integrator, copied from RAGE so this folder runs without it |
| `_pytreenet_root.py` | puts this repository's `pytreenet` ahead of any other installed copy |

## How runs are compared

A run is two numbers: its stored parameters and its infidelity at the final time. A run beats
another when it is no larger and no less accurate, and better by a margin in one of the two. The
parameter target is never used to pair runs, since a run does not land on its target exactly.

## Settings shared by every arm

- Each start is widened by warm-up passes (BUG's basis update without the Galerkin step). Without
  them the first step leaves an error that dominates; `cut_comparison.py --warmup=0` shows it.
- `rel_tol = 1e-12` and `total_tol = 1e-14`. A cut keeps `s > max(rel_tol * s_max, total_tol)`,
  so these act as a common floor for every arm.
