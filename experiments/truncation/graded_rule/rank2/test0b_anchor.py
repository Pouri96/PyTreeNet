"""Test 0(b): anchor to the existing d = 2 code.  The mu = inf staggered purification (plain gauge, ancilla = a product of up/down states)
against the d = 2 `ising` cell, N = 10, T = 3, chi in {6, 8, 12, 16}.

    python test0b_anchor.py [out.json]

Pass criteria (plan): SVD arms agree to 1e-8; spcf arms agree within about 10 percent.
"""
import sys
import _p2  # noqa: F401
import json, time
import numpy as np
import mpsenh as M
import hp_bench
import purlib as P
import spcfast
from spcfpur import SPCFPur

out = sys.argv[1] if len(sys.argv) > 1 else 'results/test0b_anchor.json'
model, N, T, dt = 'ising', 10, 3.0, 0.1
n = int(round(T / dt))
OPT = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
G2 = M.make_gates(model, N, dt)
ex2 = hp_bench.reference(model, N, T, dt)
ref = P.Reference(model, N, 'stag', np.inf, 'plain', T)
T0 = P.initial_pur_mps(N, 'stag', np.inf)
Gp = P.pur_gates(model, N, dt, 'plain')
KEYS = ['infid', 'single_rms', 'nn_rms', 'nnn_rms', 'E_abs', 'rdm2', 'rdm3']
rows = []
for chi in (6, 8, 12, 16):
    res = {}
    for arm in ('svd', 'spcf'):
        c2 = M.svd_cut if arm == 'svd' else spcfast.SPCFast(model, N, **OPT)
        t = time.time()
        Tm2, _ = M.run_tebd(model, N, chi, n, dt, c2, gates=G2)
        w2 = time.time() - t
        ap = M.mps_to_dense(Tm2)
        e2 = M.errors(ex2, ap, N, model)
        e2.update(hp_bench.marginal_errors(ex2, ap, N))
        c4 = P.svd_cut_d if arm == 'svd' else SPCFPur(model, N, **OPT)
        t = time.time()
        Tm4, _ = P.run_tebd_d(T0, Gp, chi, n, c4)
        w4 = time.time() - t
        e4 = P.pur_metrics(ref, Tm4)
        e4['infid'] = e4['infid_pur']
        row = dict(chi=chi, arm=arm, wall_d2=w2, wall_d4=w4, maxrank_d2=max(M.ranks(Tm2)), maxrank_d4=max(P.ranks(Tm4)),
                   d2={k: e2[k] for k in KEYS}, d4={k: e4[k] for k in KEYS})
        if arm == 'spcf':
            row.update(fired_d2=c2.fired, fired_d4=c4.fired, calls=c2.calls)
        rows.append(row)
        json.dump(rows, open(out, 'w'), indent=1)
        absd = max(abs(e2[k] - e4[k]) for k in KEYS)
        print(f"chi={chi:2d} {arm:4s} maxrank d2/d4 {row['maxrank_d2']}/{row['maxrank_d4']}  wall {w2:.1f}s / {w4:.1f}s  max|d2-d4| = {absd:.2e}"
              + (f"  fired {c2.fired}/{c4.fired} of {c2.calls}" if arm == 'spcf' else ''), flush=True)
        for k in KEYS:
            print(f"      {k:10s} d2 {e2[k]:.4e}  d4 {e4[k]:.4e}  ratio d4/d2 {e4[k] / e2[k] if e2[k] else float('nan'):.4f}")
