"""Does the cut correction leave the initial total-Sz sector?  (Heisenberg conserves Sz, the exact state has <X_i> = <Y_i> = 0.)

    python diag_charge.py model N T chi [spcf kwargs json]
Per Strang step, for SVD and spcf: weight outside the initial sector, rms error of <X>, <Y> and <Z> separately (single-site),
and the rms of the charge-violating two-site combinations <XX - YY>, <XY + YX> (exactly 0 in a sector state).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast

model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
dt = 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)
idx = np.arange(2 ** N)
ups = np.zeros(2 ** N, dtype=int)
for i in range(N):
    ups += 1 - ((idx >> (N - 1 - i)) & 1)                  # bit 0 = spin up (Z = +1), site 0 most significant
v0 = M.mps_to_dense(M.initial_mps(model, N))
sector = ups == int(ups[np.argmax(np.abs(v0))])


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


def stats(v, ex_o):
    v = v / np.linalg.norm(v)
    o = M.local_obs(v, N, model)
    s = (o['single'] - ex_o['single']).reshape(N, 3)
    leak = float(np.sum(np.abs(v[~sector]) ** 2))
    nn = []
    for i in range(N - 1):
        r = M.rdm2(v, N, i, i + 1)
        nn.append([np.trace(r @ (np.kron(M.X, M.X) - np.kron(M.Y, M.Y))).real, np.trace(r @ (np.kron(M.X, M.Y) + np.kron(M.Y, M.X))).real])
    return leak, np.sqrt(np.mean(s ** 2, axis=0)), np.sqrt(np.mean(np.array(nn) ** 2, axis=0))


cut = spcfast.SPCFast(model, N, **{**BEST, **kw})
arms = {'svd': (M.initial_mps(model, N), M.svd_cut), 'spcf': (M.initial_mps(model, N), cut)}
cut.start(arms['spcf'][0])
v = v0.copy()
print('   t | leak svd   leak spcf | 1site err X,Y,Z  svd | 1site err X,Y,Z  spcf | viol(XX-YY, XY+YX) spcf | fired')
for s in range(1, n + 1):
    for b in range(N - 1):
        v = M.dense_gate(v, G[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, G[b], b, N)
    ex_o = M.local_obs(v, N, model)
    out = {}
    for name, (Tm, c) in arms.items():
        sweep(Tm, c)
        out[name] = stats(M.mps_to_dense(Tm), ex_o)
    f = lambda a: ' '.join(f'{x:8.1e}' for x in a)
    print(f'{s * dt:5.2f} | {out["svd"][0]:8.1e} {out["spcf"][0]:9.1e} | {f(out["svd"][1])} | {f(out["spcf"][1])} | {f(out["spcf"][2])} | {cut.fired}', flush=True)
