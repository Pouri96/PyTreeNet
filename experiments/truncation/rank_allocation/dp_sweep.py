"""The DP allocation (``BUGConfig(parameter_budget=target)``) at 12 parameter targets.

Targets span the sizes in the cell's ``per_bond`` ladder, so run ``run.py per_bond`` first.
Writes ``results/dense/<cell>.json``, the rows the paper's figure draws.

    python dp_sweep.py four six_hard pyrazine holstein [--grid=12]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from cells import CELLS, targets_from
from run import MAX_SECONDS, _config, trajectory

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DENSE = RESULTS / "dense"

SOURCE = ("measured with the exact rank allocation of "
         "pytreenet.core.truncation.parameter_budget ({n} targets); produced by dp_sweep.py")


def run_cell(name: str, n_targets: int, max_seconds: float) -> dict:
    cell = CELLS[name]
    rig = cell.rig()
    store = json.loads((RESULTS / f"{name}.json").read_text())
    sizes = [r["params"] for r in store["arms"]["per_bond"]]
    targets = list(targets_from(sizes, count=n_targets))

    out = {"cell": name, "targets": targets, "source": SOURCE.format(n=len(targets)),
           "arms": {"dp": []}}
    print(f"\n{'=' * 92}\n{name}  {cell.label}   DP sweep   targets {targets}\n{'=' * 92}",
          flush=True)
    for target in targets:
        config = _config(rig, target=target)
        row = trajectory(rig, config, max_seconds)
        if row is None:
            print(f"    {target:>9,}  abandoned", flush=True)
            continue
        row["target"] = target
        out["arms"]["dp"].append(row)
        print(f"    {target:>9,}{row['params']:>9,}{row['peak']:>9,}"
              f"{row['err']:>14.4e}{row['bond']:>6}{row['walks']:>7.2f}"
              f"{row['wall']:>8.1f}", flush=True)
    return out


def main(names: Sequence[str], n_targets: int, max_seconds: float) -> None:
    DENSE.mkdir(exist_ok=True)
    for name in names:
        got = run_cell(name, n_targets, max_seconds)
        path = DENSE / f"{name}.json"
        path.write_text(json.dumps(got, indent=1))
        print(f"  written to {path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    opts = dict(a.split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    names = [a for a in argv if not a.startswith("--")] or list(CELLS)
    main(names, int(opts.get("--grid", "12")), float(opts.get("--max-seconds", MAX_SECONDS)))
