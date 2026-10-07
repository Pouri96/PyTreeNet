# Experiments for the graded truncation rule

The rule keeps the bond dimension SVD would keep and changes only which rank-`k` subspace is kept. At
a cut it moves the kept subspace to reduce a Pauli-weight-graded distance between the reduced density
matrices of the untruncated and truncated state on a 3 to 4 site window, plus an optional global-energy
term, inside a trust region on the extra discarded weight. Run everything with this folder as the
working directory.

```
python mps_bench.py ising 18 3.0 0.1 svd,enh:0.1:4:100:0:2:0 4,6,8,10 out.json 4   # chain TEBD, one cell
python matched.py ising 20 8.0 0.1 out.json 24,32,48,64 24,32,48                    # SVD ladder then rule ladder
python rage_bench.py ising 14 3.0 0.05 lib,svd,enh:0.1:4:100:0:2:0 4,6,8 out.json 4  # inside RAGE_MPS
python tree_bench.py svd,rule:0.1 4,6,8,10 out.json 4                                # spin-boson tree, BUG
```

## Status

The rule is an investigation, not a result. At matched bond dimension its correlator error is below SVD's
on Ising at high truncation pressure and is not on Heisenberg at the pressures tried. See the table in
`results/` and do not quote a number from one vehicle without the same cell on another.

## Vehicles

| vehicle | file | what it is |
|---|---|---|
| chain TEBD | `mps_bench.py`, `matched.py`, `rule/mpsenh.py`, `rule/mpsenv.py` | Strang sweeps in mixed canonical form against a dense Trotter reference with the same gates, so only truncation error is measured |
| inside `RAGE_MPS` | `rage_bench.py`, `rule/rage_rule.py` | the same cut patched into the integrator's final truncation sweep, against `expm_multiply` |
| tree, BUG | `tree_bench.py`, `rule/tree_rule.py` | an energy-aware subspace choice in the recursive node-cut walk, spin-boson on the `m = 3` tree |

`rage_rule.py` and `tree_rule.py` replace a library function at import of the experiment. Neither is a
library feature yet.

## Files

| file | what it is |
|---|---|
| `rule/` | the rule, as flat modules that import each other by bare name |
| `_paths.py` | import first: this repository's `pytreenet` and `rule/` ahead of any other copy |
| `_pytreenet_root.py` | the same bootstrap `rank_allocation` uses |
| `_mps_vendor/` | `RAGE_MPS` and `BUG_MPS` with the modules they use, flattened, relative imports rewritten |
| `_chain.py` | the one helper the RAGE bench needs from the RAGE benchmarks |
| `mps_front.py` | the margin ledger, `beats` and `ownership` as in `rank_allocation/dp_analysis.py` |
| `results/` | one JSON per cell, grouped by vehicle |
| `figures/` | PNG only, since `*.pdf` is ignored repository-wide |

## Dependencies on other folders

`tree_bench.py` reuses `../rank_allocation` for the spin-boson cells, models and the vendored `BUG`. This
folder does not copy them.

`_mps_vendor/bug_util.py` has local stand-ins for the open-system truncation policies, because this
branch's `pytreenet` has no `special_ttn.pauli`. Closed-system chains use only the plain policy.

## How runs are compared

A run is two numbers per metric: its stored parameters and its final error. The rule and SVD cut to the
same bond dimension, so their stored parameters are identical and only the error differs. Compare at
matched bond dimension, never by parameter target, and report the margin ledger with every raw ratio.
Wall-clock belongs to the prototype and is not a comparison axis.

A cell is usable only if SVD's front does not reach its floor inside the bond-dimension ladder. A cell
where SVD is already exact cannot show any effect of the rule.

## Settings shared by every arm

- `kappa = 0.1` trust region, `gamma = 4` weight grading, window of 3 to 4 sites.
- `rel_tol = 1e-12` in the RAGE arm, `dt = 0.1` in the chain arm and `dt = 0.05` in the RAGE arm.
- `kappa = 0` must reproduce plain SVD bit for bit through the same code path.
