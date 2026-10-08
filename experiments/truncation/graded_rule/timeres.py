"""Time-resolved local errors of plain SVD and the best spcf cut, same Trotter circuit, dense reference alongside.

    python timeres.py model N T chi out.json [spcf kwargs as json]
After every full Strang step it records, for both arms, the rms error of the 1-site, nn, nnn Pauli expectations, the
absolute energy error and the infidelity, and the ratio spcf/svd, so the time at which spcf falls behind is visible.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast

model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
out = sys.argv[5]
kw = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
dt = 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)


def sweep(Tm, cut):
    for rng, d in ((range(N - 1), 'R'), (range(N - 2, -1, -1), 'L')):
        for b in rng:
            th = np.einsum('abst,lstr->labr', G[b], np.tensordot(Tm[b], Tm[b + 1], axes=([2], [0])))
            A = Tm[b - 1] if b >= 1 else None
            B = Tm[b + 2] if b + 2 <= N - 1 else None
            if cut is M.svd_cut:
                Tm[b], Tm[b + 1], _, _ = M.svd_cut(th, chi, d)
            else:
                Tm[b], Tm[b + 1], _, _ = cut(th, chi, d, A, B, b)


cut = spcfast.SPCFast(model, N, **{**BEST, **kw})
arms = {'svd': (M.initial_mps(model, N), M.svd_cut), 'spcf': (M.initial_mps(model, N), cut)}
cut.start(arms['spcf'][0])
v = M.mps_to_dense(M.initial_mps(model, N))
rows = []
print('  t    ' + '  '.join(f'{k:>6s}' for k in ('1site', 'nn', 'nnn', 'E', 'infid')) + '   (spcf/svd)  fired')
for s in range(1, n + 1):
    for b in range(N - 1):
        v = M.dense_gate(v, G[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, G[b], b, N)
    e = {}
    for name, (Tm, c) in arms.items():
        sweep(Tm, c)
        e[name] = M.errors(v, M.mps_to_dense(Tm), N, model)
    r = {k: e['spcf'][k] / max(e['svd'][k], 1e-300) for k in ('single_rms', 'nn_rms', 'nnn_rms', 'E_abs', 'infid')}
    rows.append(dict(t=round(s * dt, 6), svd=e['svd'], spcf=e['spcf'], fired=cut.fired))
    print(f'{s * dt:5.2f}  ' + '  '.join(f'{r[k]:6.2f}' for k in ('single_rms', 'nn_rms', 'nnn_rms', 'E_abs', 'infid')) + f'   {cut.fired}', flush=True)
json.dump(rows, open(out, 'w'))
