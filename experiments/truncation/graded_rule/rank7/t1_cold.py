"""T1 supplement: the same cells as pareto_spcf.py, but the timing runs use the SAME gate state as the error run.

    python rank7/t1_cold.py chis_spc out.json reps OPT

Why this exists.  pareto_spcf.py builds ONE SPCFast and runs it 1 + reps times.  SPCFast.rel_skip skips a cut when its discarded
weight is below rel_skip * tmax, tmax being the running maximum of the discarded weight seen so far BY THIS OBJECT.  In the first
run tmax grows with time; in every later run it starts at its final value, so early cuts are skipped that the first run fired.
pareto_spcf.py reports the errors of the first run and the steady time of the later runs, so for rel_skip > 0.01 it pairs the
error of one gate behaviour with the cost of a cheaper one.  Here every run uses a fresh SPCFast (tmax = 0, as in a real
simulation) that shares the precomputed region matrices of the first one, so error, fired fraction and steady time all describe
the same run.  SVD is timed in the same process, interleaved, at chi = chi_svd (load reference) and at the two chi at which the
SVD ladder of matched_hp.json reaches the error of the spcf run (nn and 1-site), i.e. the equal-error SVD cost is measured, not
interpolated.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import _paths  # noqa: F401,E402
import mpsenh as M  # noqa: E402
import hp_bench  # noqa: E402
import spcfast  # noqa: E402
import rutil  # noqa: E402

model, N, T = 'ising', 20, 8.0
chis_spc = [int(c) for c in sys.argv[1].split(',')]
out = sys.argv[2]
reps = int(sys.argv[3])
OPT = sys.argv[4]
p = OPT.split(':')
dt = 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)
ex = hp_bench.reference(model, N, T, dt)
lad = rutil.svd_ladder()
LC = [r['chi'] for r in lad]


def make_cut(base=None):
    c = spcfast.SPCFast(model, N, a=int(p[0]), fw=float(p[1]), iters=int(p[2]), taus=[float(x) for x in p[3].split('-')],
                        ks=tuple(int(x) for x in p[4].split('-')), f_min=float(p[5]), every=int(p[6]), eps_min=float(p[7]),
                        pattern=p[8], rel_skip=float(p[9]))
    if base is not None:                       # share the state-independent region matrices, nothing else
        c._shape, c._f32, c._meta = base._shape, base._f32, base._meta
    return c


def timed(cut, chi):
    c0, w0 = time.process_time(), time.time()
    Tm, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
    return Tm, time.process_time() - c0, time.time() - w0


rows = []
for chi in chis_spc:
    c1 = make_cut()
    Tm, cpu1, wall1 = timed(c1, chi)
    ap = M.mps_to_dense(Tm)
    e = M.errors(ex, ap, N, model)
    e.update(hp_bench.marginal_errors(ex, ap, N))
    fired1, calls1 = c1.fired, c1.calls
    ceq = {m: rutil.chi_eq(LC, [r[m] for r in lad], e[m]) for m in ('nn_rms', 'single_rms')}
    chi_nn, chi_1 = int(round(ceq['nn_rms'][0])), int(round(ceq['single_rms'][0]))
    tcs = {'spcf': [], 'svd_same': [], 'svd_eq_nn': [], 'svd_eq_1site': []}
    firedreps = []
    for _ in range(reps):                     # interleaved: every arm sees the same machine load in the same minute
        c = make_cut(c1)
        tcs['spcf'].append(timed(c, chi)[1:])
        firedreps.append(c.fired)
        tcs['svd_same'].append(timed(M.svd_cut, chi)[1:])
        tcs['svd_eq_nn'].append(timed(M.svd_cut, chi_nn)[1:])
        if chi_1 != chi_nn:
            tcs['svd_eq_1site'].append(timed(M.svd_cut, chi_1)[1:])
    if not tcs['svd_eq_1site']:
        tcs['svd_eq_1site'] = tcs['svd_eq_nn']
    best = {k: (min(x[0] for x in v), min(x[1] for x in v)) for k, v in tcs.items()}
    row = dict(arm='spcf', chi=chi, opt=OPT, params=M.stored_params(Tm), cpu_first=round(cpu1, 3), fired=fired1, calls=calls1,
               fired_reps=firedreps, chi_eq_nn=ceq['nn_rms'][0], chi_eq_nn_flag=ceq['nn_rms'][1], chi_eq_1site=ceq['single_rms'][0],
               chi_eq_1site_flag=ceq['single_rms'][1], chi_svd_nn=chi_nn, chi_svd_1site=chi_1,
               cpu_spcf=round(best['spcf'][0], 3), cpu_svd_same=round(best['svd_same'][0], 3), cpu_svd_eq_nn=round(best['svd_eq_nn'][0], 3),
               cpu_svd_eq_1site=round(best['svd_eq_1site'][0], 3), wall_spcf=round(best['spcf'][1], 3), wall_svd_eq_nn=round(best['svd_eq_nn'][1], 3),
               all_cpu={k: [round(x[0], 3) for x in v] for k, v in tcs.items()}, **e)
    rows.append(row)
    json.dump(rows, open(out, 'w'))
    print(f"spcf chi={chi} nn={e['nn_rms']:.2e} 1site={e['single_rms']:.2e} fired {fired1}/{calls1} (reps {firedreps}) | cpu spcf {row['cpu_spcf']:.2f}s "
          f"svd@{chi} {row['cpu_svd_same']:.2f}s svd@{chi_nn}(eq nn) {row['cpu_svd_eq_nn']:.2f}s svd@{chi_1}(eq 1site) {row['cpu_svd_eq_1site']:.2f}s "
          f"| ratio nn {row['cpu_spcf'] / row['cpu_svd_eq_nn']:.2f} 1site {row['cpu_spcf'] / row['cpu_svd_eq_1site']:.2f}", flush=True)
