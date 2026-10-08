"""Pareto data for the numpy spc cut against plain SVD on one cell, one chi after the other in one process.

    python pareto_spcf.py model N T chi_list_svd chi_list_spc out.json [reps] [spcf options a:fw:iters:taus:ks:fmin:every:eps:pattern:rel]
Every row carries the errors against the dense reference, the stored parameters, the single-run wall and cpu (the spc run includes
the one-off region precompute) and wall_steady / cpu_steady, the minimum over `reps` repeated runs with every cache warm.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
import numpy as np
try:                                    # timing on a shared machine: scheduling priority above normal
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
except Exception:
    pass
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import spcfast

model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
chis_svd = [int(c) for c in sys.argv[4].split(',')]
chis_spc = [int(c) for c in sys.argv[5].split(',')] if sys.argv[5] != '-' else []
out = sys.argv[6]
reps = int(sys.argv[7]) if len(sys.argv) > 7 else 3
OPT = sys.argv[8] if len(sys.argv) > 8 else '2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2'
p = OPT.split(':')
dt = 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)
ex = hp_bench.reference(model, N, T, dt)


def make_cut():
    return spcfast.SPCFast(model, N, a=int(p[0]), fw=float(p[1]), iters=int(p[2]), taus=[float(x) for x in p[3].split('-')],
                           ks=tuple(int(x) for x in p[4].split('-')), f_min=float(p[5]), every=int(p[6]), eps_min=float(p[7]),
                           pattern=p[8], rel_skip=float(p[9]))


def timed(cut, chi):
    w, c = time.time(), time.process_time()
    Tm, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
    return Tm, time.time() - w, time.process_time() - c


rows = []


def one(kind, chi):
    cut = M.svd_cut if kind == 'svd' else make_cut()
    Tm, w1, c1 = timed(cut, chi)
    ap = M.mps_to_dense(Tm)
    e = M.errors(ex, ap, N, model)
    e.update(hp_bench.marginal_errors(ex, ap, N))
    return cut, Tm, w1, c1, e


def best(ts):
    return min(t[0] for t in ts), min(t[1] for t in ts)


for chi in sorted(set(chis_svd) | set(chis_spc)):
    pair = {}
    for kind in ('svd', 'spcf'):
        if (kind == 'svd' and chi in chis_svd) or (kind == 'spcf' and chi in chis_spc):
            pair[kind] = one(kind, chi)
    times = {k: [] for k in pair}
    for _ in range(reps):                          # interleaved repeats: both arms see the same machine load
        for kind, (cut, *_r) in pair.items():
            times[kind].append(timed(cut, chi)[1:])
    for kind, (cut, Tm, w1, c1, e) in pair.items():
        ws, cs = best(times[kind])
        row = dict(arm=kind, chi=chi, params=M.stored_params(Tm), wall=round(w1, 3), cpu=round(c1, 3), wall_steady=round(ws, 3),
                   cpu_steady=round(cs, 3), **e)
        if kind == 'spcf':
            row.update(fired=cut.fired // (reps + 1), calls=cut.calls // (reps + 1))
        rows.append(row)
        json.dump(rows, open(out, 'w'))
        print(f"{kind:4s} chi={chi:3d} params={row['params']:6d} infid={e['infid']:.3e} rdm2={e['rdm2']:.2e} 1site={e['single_rms']:.2e} "
              f"nn={e['nn_rms']:.2e} nnn={e['nnn_rms']:.2e} | single run cpu {c1:.2f}s  steady cpu {cs:.2f}s wall {ws:.2f}s"
              + (f"  fired {row['fired']}/{row['calls']}" if kind == 'spcf' else ''), flush=True)
