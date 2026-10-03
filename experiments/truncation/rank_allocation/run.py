"""Run BUG trajectories on the cells and record them in ``results/<cell>.json``.

Arms:
    per_bond  the bond cap: every bond cut at ``chi``, over the cell's ``chi`` grid. A sweep
              stops early once two consecutive rows agree within ``FLOOR_BAND``.
    prep      the cap fixed at the top of the grid, warm-up passes varied; picks ``warmup``.
    all       prep at the cell's warm-up, then per_bond.

The DP allocation arm is ``dp_sweep.py``, which reuses ``_config`` and ``trajectory``.

    python run.py prep four --warmup=0,1,2,3
    python run.py per_bond four [--chis=6,8,10]
    python run.py all four pyrazine holstein
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import _pytreenet_root  # noqa: F401  (side-effecting: puts the right pytreenet first)
from _rage_vendor import BUG, TimeEvoMode, BUGConfig

import cells as cells_mod
from cells import CELLS, Rig
from models.spin_boson import parameters, widest_bond

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

#: Solver settings shared by every arm. ``rel_tol`` applies to every cut, budgeted or not.
REL_TOL, TOTAL_TOL = 1e-12, 1e-14
KRYLOV = dict(kstep_evo_mode=TimeEvoMode.KRYLOV, solver_options={"krylov_tol": 1e-10})

#: Runaway guard on the budget arm's bond dimension; no recorded row came near it.
BUDGET_RAIL = 256

#: Two consecutive rows within this error ratio mean the curve has reached its floor.
FLOOR_BAND = 1.05

#: A trajectory slower than this is abandoned and not recorded.
MAX_SECONDS = 900.0


def _config(rig: Rig, *, chi=None, target=None, warmup=None) -> BUGConfig:
    """The config every arm uses; an arm sets only ``chi``, ``target`` or ``warmup``."""
    return BUGConfig(max_bond_dim=BUDGET_RAIL if chi is None else chi,
                     rel_tol=REL_TOL, total_tol=TOTAL_TOL,
                     parameter_budget=target,
                     warmup_sweeps=rig.warmup if warmup is None else warmup,
                     warmup_every_step=False, **KRYLOV)


def trajectory(rig: Rig, config: BUGConfig, max_seconds: float = MAX_SECONDS) -> Optional[dict]:
    """Evolve ``rig`` to ``t_final`` under ``config``; ``None`` if over ``max_seconds``.

    Returns final ``params``, the largest size held (``peak``), final infidelity (``err``),
    widest bond, truncation walks per step and wall time.
    """
    solver = BUG(rig.start, rig.ttno, rig.dt, rig.dt * rig.nsteps, [], config=config)
    peak, probes, start = 0, 0, time.perf_counter()
    for step in range(rig.nsteps):
        solver.run_one_time_step()
        peak = max(peak, parameters(solver.state))
        solve = solver.trunc_engine.last_budget_solve
        probes += solve.trials if solve is not None else 0
        if step % 25 == 24 and time.perf_counter() - start > max_seconds:
            return None
    return {"params": parameters(solver.state), "peak": peak,
            "err": rig.infidelity(solver.state), "bond": widest_bond(solver.state),
            "walks": probes / rig.nsteps + 1.0, "wall": time.perf_counter() - start}


# --------------------------------------------------------------------------------- the arms
def sweep(name: str, arm: str, *, override=None, max_seconds=MAX_SECONDS) -> None:
    """The per-bond ladder over one cell. Rows merge on ``chi``, re-runs overwrite theirs."""
    if arm != "per_bond":
        raise SystemExit(f"unknown arm {arm!r}; one of prep, per_bond, all. The rule under "
                         "test is measured by dp_sweep.py, not this arm.")
    cell = CELLS[name]
    rig = cell.rig()
    store = load(name)
    store.update(cell=name, label=cell.label, dt=cell.dt, nsteps=cell.nsteps,
                 warmup=rig.warmup)
    rows = {r["chi"]: r for r in store["arms"].get("per_bond", [])}

    print(f"\n{'=' * 96}\n{name}  {cell.label}   arm per_bond   warm-up {rig.warmup}   "
          f"{rig.nsteps} steps x dt={cell.dt}\n{'=' * 96}", flush=True)
    print(f"  {'knob':>8}{'params':>9}{'peak':>9}{'infidelity':>14}{'bond':>6}"
          f"{'walks':>7}{'secs':>8}", flush=True)

    previous = None
    for knob in (override or list(rig.chis)):
        config = _config(rig, chi=int(knob))
        row = trajectory(rig, config, max_seconds)
        if row is None:
            print(f"  {knob:>8,}  abandoned over {max_seconds:.0f}s", flush=True)
            break
        row["chi"] = int(knob)
        rows[int(knob)] = row
        flat = previous is not None and previous <= FLOOR_BAND * row["err"]
        print(f"  {knob:>8,}{row['params']:>9,}{row['peak']:>9,}{row['err']:>14.4e}"
              f"{row['bond']:>6}{row['walks']:>7.1f}{row['wall']:>8.1f}"
              + ("  floor reached" if flat else ""), flush=True)
        store["arms"]["per_bond"] = [rows[k] for k in sorted(rows)]
        save(name, store)
        if flat:
            break
        previous = row["err"]


def prep(name: str, warmups: Sequence[int], chi=None, max_seconds=MAX_SECONDS) -> None:
    """One run per warm-up pass count at a fixed cap. Take the cheapest count that stops
    improving the result."""
    cell = CELLS[name]
    rig = cell.rig()
    chi = int(chi or cell.rail_chi)
    store = load(name)
    store.update(cell=name, label=cell.label, dt=cell.dt, nsteps=cell.nsteps,
                 warmup=rig.warmup, prep_chi=chi)
    print(f"\n{'=' * 96}\n{name}  {cell.label}   arm prep at chi={chi}   "
          f"{rig.nsteps} steps x dt={cell.dt}   cell default warm-up {rig.warmup}"
          f"\n  the truncation is held fixed; only the initial basis moves"
          f"\n{'=' * 96}", flush=True)
    print(f"  {'warmup':>8}{'start':>9}{'params':>9}{'infidelity':>14}{'bond':>6}{'secs':>8}",
          flush=True)
    rows = {r["warmup"]: r for r in store.get("prep", [])}
    for k in warmups:
        config = _config(rig, chi=chi, warmup=k)
        prepared = _warmed_start(rig, config, k)
        row = trajectory(rig, config, max_seconds)
        if row is None:
            print(f"  {k:>8}  abandoned over {max_seconds:.0f}s", flush=True)
            break
        row["warmup"] = int(k)
        row["start_params"] = prepared
        rows[int(k)] = row
        print(f"  {k:>8}{prepared:>9,}{row['params']:>9,}{row['err']:>14.4e}"
              f"{row['bond']:>6}{row['wall']:>8.1f}", flush=True)
        store["prep"] = [rows[i] for i in sorted(rows)]
        save(name, store)


def _warmed_start(rig: Rig, config: BUGConfig, k: int) -> int:
    """Parameters the start holds after ``k`` warm-up passes."""
    solver = BUG(rig.start, rig.ttno, rig.dt, rig.dt, [], config=config)
    for _ in range(k):
        solver.recursive_update(galerkin=False)
    return parameters(solver.state)


# ------------------------------------------------------------------------------- the store
def path_for(name: str) -> Path:
    return RESULTS / f"{name}.json"


def load(name: str) -> dict:
    """The recorded rows for a cell, or an empty store."""
    path = path_for(name)
    if not path.exists():
        return {"cell": name, "arms": {}, "prep": []}
    store = json.loads(path.read_text())
    store.setdefault("arms", {})
    store.setdefault("prep", [])
    return store


def save(name: str, store: dict) -> None:
    """Write atomically, so an interrupted sweep never leaves a half-written file."""
    RESULTS.mkdir(exist_ok=True)
    tmp = path_for(name).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=1))
    os.replace(tmp, path_for(name))


# -------------------------------------------------------------------------------- the CLI
def main(argv: Sequence[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a.split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if not args:
        print(__doc__)
        print(cells_mod.table())
        return 0

    arm, names = args[0], args[1:] or list(CELLS)
    unknown = [n for n in names if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; known: {', '.join(CELLS)}")
    max_seconds = float(opts.get("--max-seconds", MAX_SECONDS))
    grid = [int(x) for x in opts["--chis"].split(",")] if "--chis" in opts else None
    warmups = [int(x) for x in opts.get("--warmup", "0,1,2,3").split(",")]

    chi = int(opts["--chi"]) if "--chi" in opts else None
    for name in names:
        if arm == "prep":
            prep(name, warmups, chi, max_seconds)
        elif arm == "all":
            prep(name, [CELLS[name].warmup], chi, max_seconds)
            sweep(name, "per_bond", override=grid, max_seconds=max_seconds)
        else:
            sweep(name, arm, override=grid, max_seconds=max_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
