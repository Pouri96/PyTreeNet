"""Table I: three ways to cut the same tree to the same bond cap ``chi``.

    ceruti        Ref. [Ceruti2024]: root-to-leaves, orthogonality centre fixed at the root.
    grasedyck     Ref. [Grasedyck2010]: the same, carrying the environments.
    centre_moved  the shipped walk, ``recursive_node_cut_truncation``.

The first two live in :mod:`reference_truncations`. Each arm runs BUG with the engine's
``recursive_node_cut_truncation`` swapped for its cut. System: 16 qubits on the leaves of a
binary tree, ``H = -sum_i (X_i X_{i+1} + Z_i)``, Neel start, ``t = 1``, ``dt = 0.02``.

    python cut_comparison.py                            # Table I, warm-up 4
    python cut_comparison.py --chis=8,12,16 --warmup=0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _pytreenet_root  # noqa: E402,F401  (side-effecting: puts the right pytreenet first)

from pytreenet.ttno.ttno_class import TTNO
from pytreenet.util.tensor_splitting import SVDParameters as PtnSVDParameters

from pytreenet.core.truncation import engine as trunc_module
from pytreenet.core.truncation.node_truncation import recursive_node_cut_truncation

from _rage_vendor import BUG, BUGConfig

import reference_truncations as ref
from models.tfim_tree import (COUPLING, aligned_vector, infidelity, neel_state,
                              tfim_hamiltonian, tfim_sparse)
from run import KRYLOV, REL_TOL, TOTAL_TOL

RESULTS = HERE / "results"
N, DT, T_FINAL = 16, 0.02, 1.0
CHIS = (8, 12, 16)
#: Stored in the result file; new rows merge only into a file with the same values.
SYSTEM = "neel start, H = -sum_i (X_i X_i+1 + Z_i), binary tree, virtual interiors"
GAUGE = "root-canonical form restored after every vendored cut"
ARMS = ("ceruti", "grasedyck", "centre_moved")
LABEL = {"ceruti": "Ref. [Ceruti2024]", "grasedyck": "Ref. [Grasedyck2010]",
         "centre_moved": "centre moved"}


def _vendored(carry_environments: bool):
    """A stand-in for ``recursive_node_cut_truncation`` running a vendored walk."""
    def walk(ttn, svd_params, bond_params=None, spectra=None):
        if bond_params is not None or spectra is not None:
            raise ValueError("the vendored reference truncations take no bond_params or "
                             "spectra; Table I is a cap comparison.")
        ref.recursive_truncation(
            ttn,
            PtnSVDParameters(max_bond_dim=svd_params.max_bond_dim,
                             rel_tol=svd_params.rel_tol,
                             total_tol=svd_params.total_tol),
            carry_environments=carry_environments)
        # The walk leaves non-isometries below the root; restore canonical form for BUG.
        # The recorded centre must be cleared first, or canonical_form does nothing.
        ttn.orthogonality_center_id = None
        ttn.canonical_form(ttn.root_id)
    return walk


CUTS = {"ceruti": _vendored(False),
        "grasedyck": _vendored(True),
        "centre_moved": recursive_node_cut_truncation}


def reference():
    """``exp(-i T H) |psi_0>`` on the dense chain, cached; independent of every truncation.

    The cache name carries the start state and the coefficient as well as ``N`` and ``T``, so
    a changed system can never be scored against a stale reference.
    """
    path = RESULTS / f"ref_tfim{N}_neel_c{COUPLING:g}_t{T_FINAL:g}.npy"
    if path.exists():
        return np.load(path)
    vec = expm_multiply(-1j * T_FINAL * tfim_sparse(N), aligned_vector(neel_state(N), N))
    RESULTS.mkdir(exist_ok=True)
    np.save(path, vec)
    return np.asarray(vec)


def evolve(arm: str, chi: int, warmup: int, exact) -> dict:
    """One trajectory under one cut, at cap ``chi``."""
    start_state = neel_state(N)
    ttno = TTNO.from_hamiltonian(tfim_hamiltonian(N), start_state)
    config = BUGConfig(max_bond_dim=chi, rel_tol=REL_TOL, total_tol=TOTAL_TOL,
                       warmup_sweeps=warmup, warmup_every_step=False, **KRYLOV)
    trunc_module.recursive_node_cut_truncation = CUTS[arm]
    try:
        solver = BUG(start_state, ttno, DT, T_FINAL, [], config=config)
        began = time.perf_counter()
        for _ in range(round(T_FINAL / DT)):
            solver.run_one_time_step()
        wall = time.perf_counter() - began
    finally:
        trunc_module.recursive_node_cut_truncation = recursive_node_cut_truncation
    return {"err": infidelity(solver.state, exact, N),
            "params": int(sum(np.asarray(solver.state.tensors[nid]).size
                              for nid in solver.state.nodes)),
            "wall": wall}


def main(chis, warmup, arms=ARMS):
    exact = reference()
    print(f"\n  {N} qubits, virtual interiors, Neel start, H = -sum (X_i X_i+1 + Z_i), "
          f"t = {T_FINAL:g}, dt = {DT:g}, warm-up {warmup}\n")
    print(f"    {'chi':>4}" + "".join(f"{LABEL[a]:>22}" for a in arms) + f"{'entries':>10}")
    out = {"n": N, "dt": DT, "t_final": T_FINAL, "warmup": warmup, "system": SYSTEM,
           "gauge": GAUGE, "rows": {}}
    path = RESULTS / "cut_comparison.json"
    if path.exists():
        # Rows merge on chi, but only into a file measured on this exact setup.
        old = json.loads(path.read_text())
        if all(old.get(k) == out[k]
               for k in ("n", "dt", "t_final", "warmup", "system", "gauge")):
            out["rows"] = old["rows"]
    for chi in chis:
        row = {a: evolve(a, chi, warmup, exact) for a in arms}
        out["rows"][str(chi)] = row
        sizes = {row[a]["params"] for a in arms}
        print(f"    {chi:>4}" + "".join(f"{row[a]['err']:>22.4e}" for a in arms)
              + f"{(str(sizes.pop()) if len(sizes) == 1 else 'DIFFER'):>10}", flush=True)
        RESULTS.mkdir(exist_ok=True)
        out["rows"] = dict(sorted(out["rows"].items(), key=lambda kv: int(kv[0])))
        path.write_text(json.dumps(out, indent=1))
    print(f"\n  written to {path}")
    return out


if __name__ == "__main__":
    opts = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    main([int(x) for x in opts.get("--chis", ",".join(map(str, CHIS))).split(",")],
         int(opts.get("--warmup", 4)),
         tuple(opts.get("--arms", ",".join(ARMS)).split(",")))
