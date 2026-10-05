"""The six systems of the truncation paper (Fig. 5) and their run settings.

A cell is a physical system plus its time window, step, warm-up and ``chi`` grid. It holds no
truncation setting, so every arm evolves the same system. Each start is a bond-1 product state
widened by ``warmup`` basis-only BUG passes (``BUGConfig.warmup_sweeps``), an exact change of
basis that leaves the state itself unchanged.

    python -c "import cells; print(cells.table())"
    python run.py prep <cell> --warmup=0,1,2,3     # choose a new cell's warm-up
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from models import spin_boson as sb
from models import spread_models as sm

HERE = Path(__file__).resolve().parent
#: Cache of dense references, one per (system, t_final), shared by every arm.
REFCACHE = HERE / "_refcache"

DEFAULT_WARMUP = 2

#: One step size and final time for every cell.
DEFAULT_DT = 0.01
T_FINAL = 1.0


@dataclass(frozen=True)
class Rig:
    """One cell, resolved: everything a trajectory needs and nothing about truncation."""
    start: object                       #: the bond-1 product state, unwarmed
    ttno: object                        #: the Hamiltonian as a TTNO
    exact: np.ndarray                   #: the dense reference at ``t_final``
    vector: Callable[[object], np.ndarray]   #: a tree state as a vector aligned with ``exact``
    dt: float
    nsteps: int
    warmup: int
    chis: Tuple[int, ...]
    label: str

    def infidelity(self, state) -> float:
        """``1 - |<exact|psi>|^2``, both sides normalised."""
        return sb.infidelity(self.vector(state), self.exact)


@dataclass(frozen=True)
class Cell:
    """A cell's specification; :meth:`rig` builds it."""
    label: str
    kind: str                           #: "boson" or "spread"
    dt: float
    t_final: float
    chis: Tuple[int, ...]
    warmup: int = DEFAULT_WARMUP
    #: Recorded infidelity floor, for reference only.
    recorded_floor: float = float("nan")
    spec: dict = field(default_factory=dict)

    @property
    def nsteps(self) -> int:
        return int(round(self.t_final / self.dt))

    @property
    def rail_chi(self) -> int:
        """The cap the ``prep`` arm runs at, the top of the ``chi`` grid (uncapped BUG is too slow)."""
        return max(self.chis)

    def rig(self) -> Rig:
        return _RIGS[self.kind](self)


# ------------------------------------------------------------------ builders, one per kind
def _boson_rig(cell: Cell) -> Rig:
    dims, delta = cell.spec["dims"], cell.spec["delta"]
    w, g = sb.ohmic_modes(len(dims), cell.spec.get("alpha", 2.8))
    model = sb.SpinBoson(w, g, dims, delta=delta)
    edges, root = sb.mode_tree(sb.balanced_assignment(model))
    exact, _ = sb.exact_reference(model, edges, root, cell.t_final, cache_dir=str(REFCACHE))
    return Rig(sb.build_state(edges, root, model.site_dims),
               sb.make_ttno(model, edges, root), exact,
               lambda s, _o=model.site_order: sb.aligned_vector(s, _o),
               cell.dt, cell.nsteps, cell.warmup, cell.chis, cell.label)


def _spread_rig(cell: Cell) -> Rig:
    model = sm.MODELS[cell.spec["model"]][0]()
    edges, root = model.tree()
    exact, _ = sm.exact_reference(model, cell.t_final, cache_dir=str(REFCACHE))
    return Rig(model.initial_state(), sb.make_ttno(model, edges, root), exact,
               lambda s, _o=model.site_order: sb.aligned_vector(s, _o),
               cell.dt, cell.nsteps, cell.warmup, cell.chis, cell.label)


_RIGS = {"boson": _boson_rig, "spread": _spread_rig}


# ------------------------------------------------------------------------------- the table
def _boson(label, dims, delta, t_final, chis, *, floor=float("nan")) -> Cell:
    return Cell(label, "boson", DEFAULT_DT, t_final, tuple(chis), DEFAULT_WARMUP, floor,
                {"dims": list(dims), "delta": delta})


#: The cells, in the panel order of the paper's Fig. 5.
CELLS: Dict[str, Cell] = {
    # -- ohmic spin-boson on the m=3 mode-combination tree ---------------------------------
    "four": _boson(r"$d = [32,16,16,16]$", [32, 16, 16, 16], 1.0, T_FINAL,
                   (6, 8, 10, 12, 14, 16, 20, 24, 32), floor=1.755e-12),
    "six_mid": _boson(r"$d = [24,16,12,8,6,4]$", [24, 16, 12, 8, 6, 4], 1.0, T_FINAL,
                      (4, 6, 8, 10, 12, 14, 16, 20, 24), floor=1.673e-12),
    "six_hard": _boson(r"$d = [8,8,8,8,8,8]$", [8] * 6, 1.0, T_FINAL,
                       (4, 6, 8, 10, 12, 14, 16, 20, 24), floor=1.652e-12),
    "six_wide": _boson(r"$d = [64,32,16,8,4,2]$", [64, 32, 16, 8, 4, 2], 1.0, T_FINAL,
                       (4, 6, 8, 10, 12, 14, 16, 20, 24), floor=1.634e-12),
    # -- chemistry ------------------------------------------------------------------------
    "pyrazine": Cell(sm.MODELS["pyrazine"][4], "spread", DEFAULT_DT, T_FINAL,
                     (4, 6, 8, 10, 12, 16, 20, 24, 32), DEFAULT_WARMUP,
                     spec={"model": "pyrazine"}),
    "holstein": Cell(sm.MODELS["holstein"][4], "spread", DEFAULT_DT, T_FINAL,
                     (4, 6, 8, 10, 12, 16, 20, 24, 32), DEFAULT_WARMUP,
                     spec={"model": "holstein"}),
}


def targets_from(sizes: Sequence[int], count: int = 6) -> Tuple[int, ...]:
    """``count`` parameter targets, log-spaced over the sizes the per-bond ladder reached."""
    lo, hi = min(sizes), max(sizes)
    return tuple(sorted({int(round(x)) for x in np.geomspace(lo, hi, count)}))


def table() -> str:
    """The cell table as text."""
    out = [f"{'cell':<14}{'kind':<8}{'dt':>8}{'t_final':>9}{'steps':>7}{'warmup':>8}  chis"]
    for name, cell in CELLS.items():
        chis = list(cell.chis)
        span = f"{chis[0]}..{chis[-1]} ({len(chis)})"
        out.append(f"{name:<14}{cell.kind:<8}{cell.dt:>8}{cell.t_final:>9}"
                   f"{cell.nsteps:>7}{cell.warmup:>8}  {span}")
    return "\n".join(out)


if __name__ == "__main__":
    print(table())
