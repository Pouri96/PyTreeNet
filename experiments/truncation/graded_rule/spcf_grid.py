"""Knob grid for the numpy spc cut on one cell: SVD and named SPCFast variants at several chi, errors only (no repeat timing).

    python spcf_grid.py model N T chi_list out.json 'name=json_kwargs' ['name=json_kwargs' ...]
kwargs go straight to spcfast.SPCFast on top of the best config (a=2, fw=0, iters=4, taus=[1.0], ks=(1,2,3), eps_min=1e-7,
rel_skip=1e-2), e.g. 'tau025={"taus":[0.25]}'  'static={"taus":[]}'  'E1={"wE":1.0}'.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, time
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import spcfast

model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
chis = [int(c) for c in sys.argv[4].split(',')]
out = sys.argv[5]
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
variants = [('svd', None)] + [(s.split('=', 1)[0], json.loads(s.split('=', 1)[1])) for s in sys.argv[6:]]
dt = 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)
ex = hp_bench.reference(model, N, T, dt)
rows = []
for chi in chis:
    base = None
    for name, kw in variants:
        cut = M.svd_cut if kw is None else spcfast.SPCFast(model, N, **{**BEST, **kw})
        w, c = time.time(), time.process_time()
        Tm, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
        cpu = time.process_time() - c
        ap = M.mps_to_dense(Tm)
        e = M.errors(ex, ap, N, model)
        e.update(hp_bench.marginal_errors(ex, ap, N))
        row = dict(variant=name, chi=chi, cpu=round(cpu, 2), params=M.stored_params(Tm), **e)
        if kw is not None:
            row.update(fired=cut.fired, calls=cut.calls)
        rows.append(row)
        json.dump(rows, open(out, 'w'))
        if base is None:
            base = e
        rat = ' '.join(f'{k}={e[k] / base[k]:.2f}' for k in ('single_rms', 'nn_rms', 'nnn_rms', 'rdm2', 'E_abs', 'infid'))
        print(f'chi={chi:3d} {name:10s} {rat}  (abs nn {e["nn_rms"]:.2e}, cpu {cpu:.1f}s)', flush=True)
