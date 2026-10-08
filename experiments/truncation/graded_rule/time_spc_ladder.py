"""Steady-state runtime of the spc cut on the Pareto cell (Ising N=12, T=4, dt=0.1) for a chi ladder, one process, one chi after the other.

    python time_spc_ladder.py 4,6,8,10,12,16 results/pareto_n12_steady/spc_steady.json
Each chi runs the cell twice with one cut object. The first run pays the JAX cache load and tracing, the second is the steady state
(jit caches kept). The second run's errors are compared with the stored Pareto row to confirm it is the same computation.
"""
import json
import os
import sys
import time
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import spccut

chis = [int(c) for c in sys.argv[1].split(',')]
out = sys.argv[2]
model, N, T, dt = 'ising', 12, 4.0, 0.1
ex = hp_bench.reference(model, N, T, dt)
G = M.make_gates(model, N, dt)
n = int(round(T / dt))
rows = []
os.makedirs(os.path.dirname(out), exist_ok=True)
for chi in chis:
    cut = spccut.SPCCut(model, N, a=2, taus=[0.5, 1.0, 1.5, 2.0], ks=(1, 2, 3), mode='hard', rcond=0.01, passes=1)
    t = []
    for rep in range(2):
        w0, c0 = time.time(), time.process_time()
        Tm, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
        t.append((time.time() - w0, time.process_time() - c0))
    e = M.errors(ex, M.mps_to_dense(Tm), N, model)
    e.update(hp_bench.marginal_errors(ex, M.mps_to_dense(Tm), N))
    stored = json.load(open(f'results/pareto_n12/spc_{chi}.json'))[0]
    same = abs(e['single_rms'] - stored['single_rms']) <= 1e-6 * stored['single_rms']
    rows.append(dict(arm='spc', chi=chi, params=M.stored_params(Tm), wall_first=round(t[0][0], 1), cpu_first=round(t[0][1], 1),
                     wall=round(t[1][0], 2), cpu=round(t[1][1], 2), single_rms=e['single_rms'], same_as_pareto_row=bool(same)))
    json.dump(rows, open(out, 'w'))
    print(f"chi={chi} first {t[0][0]:.1f}s steady {t[1][0]:.2f}s cpu {t[1][1]:.2f}s same={same}", flush=True)
