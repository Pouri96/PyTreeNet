"""Heis control, part 2: the pure-state d = 2 heis cell at the same N = 10 and T as the purification control, SVD against spcf (existing code),
to tell whether a spcf gain on heis in the purification cells is a property of heis at this pressure rather than of the purification pipeline.
    python heis_pure_anchor.py T chis out.json
"""
import sys
import _p2  # noqa: F401
import json
import numpy as np
import mpsenh as M
import hp_bench
import spcfast
T, chis, out = float(sys.argv[1]), [int(c) for c in sys.argv[2].split(',')], sys.argv[3]
model, N, dt = 'heis', 10, 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)
ex = M.dense_reference(model, N, n, dt, gates=G)
OPT = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
rows = []
for chi in chis:
    res = {}
    for arm in ('svd', 'spcf', 'spcfg'):
        cut = M.svd_cut if arm == 'svd' else spcfast.SPCFast(model, N, **OPT, gate=1.0 if arm == 'spcfg' else 0.0)
        Tm, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
        ap = M.mps_to_dense(Tm)
        e = M.errors(ex, ap, N, model)
        e.update(hp_bench.marginal_errors(ex, ap, N))
        res[arm] = e
        rows.append(dict(arm=arm, chi=chi, T=T, **e))
    print(f"heis N={N} T={T} chi={chi:2d}: svd rdm2 {res['svd']['rdm2']:.2e} nn {res['svd']['nn_rms']:.2e} | spcf/svd rdm2 {res['spcf']['rdm2'] / res['svd']['rdm2']:.2f} nn {res['spcf']['nn_rms'] / res['svd']['nn_rms']:.2f} "
          f"nnn {res['spcf']['nnn_rms'] / res['svd']['nnn_rms']:.2f} E {res['spcf']['E_abs'] / res['svd']['E_abs']:.2f} | gated rdm2 {res['spcfg']['rdm2'] / res['svd']['rdm2']:.2f}", flush=True)
json.dump(rows, open(out, 'w'))
