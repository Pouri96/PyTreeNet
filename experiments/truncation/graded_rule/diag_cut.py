"""Per-cut damage map.  For every cut at which the spcf correction fires, build the full dense state before the cut (the untruncated
two-site tensor in place), after the plain SVD cut and after the spcf cut, and record the change of <Z_i> and of the local energy
<h_j> relative to the pre-cut state, site by site.  Aggregated by distance from the cut and by time phase.

    python diag_cut.py model N T chi out.json [spcf kwargs json]
Site distance from the cut between sites b and b+1: d = b - i for i <= b, d = i - (b+1) for i > b (0 = adjacent site).
Bond distance k = j - b for the bond (j, j+1) (0 = the cut bond).  The fitted region covers d <= a (a = 2 by default).
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
H = [M.h_bond(model, j, N) for j in range(N - 1)]


def dense_theta(Tm, b, th):
    """normalised full state with the two-site tensor th (l,2,2,r) on sites b, b+1 and the current tensors elsewhere"""
    l, r = th.shape[0], th.shape[3]
    if b > 0:
        Lm = Tm[0]
        for j in range(1, b):
            Lm = np.tensordot(Lm, Tm[j], axes=([Lm.ndim - 1], [0]))
        Lm = Lm.reshape(-1, l)
    else:
        Lm = np.ones((1, l))
    if b + 2 < N:
        Rm = Tm[N - 1]
        for j in range(N - 2, b + 1, -1):
            Rm = np.tensordot(Tm[j], Rm, axes=([Tm[j].ndim - 1], [0]))
        Rm = Rm.reshape(r, -1)
    else:
        Rm = np.ones((r, 1))
    v = np.einsum('xl,labr,ry->xaby', Lm, th, Rm).reshape(-1)
    return v / np.linalg.norm(v)


def obs(v):
    p = (np.abs(v) ** 2).reshape([2] * N)
    z = np.array([np.take(p, 0, axis=i).sum() - np.take(p, 1, axis=i).sum() for i in range(N)])
    e = np.array([np.trace(M.rdm2(v, N, j, j + 1) @ H[j]).real for j in range(N - 1)])
    return z, e


class Spy:
    def __init__(self, inner):
        self.inner, self.records, self.calls = inner, [], 0

    def start(self, Tm):
        self.Tm = Tm
        self.inner.start(Tm)

    def __call__(self, theta, chi_, dirn, A, B, b):
        fired0 = self.inner.fired
        res = self.inner(theta, chi_, dirn, A, B, b)
        self.calls += 1
        if self.inner.fired > fired0:
            Tm = self.Tm
            ts = M.svd_cut(theta, chi_, dirn)
            pre = obs(dense_theta(Tm, b, theta))
            svd = obs(dense_theta(Tm, b, np.tensordot(ts[0], ts[1], axes=([2], [0]))))
            sp = obs(dense_theta(Tm, b, np.tensordot(res[0], res[1], axes=([2], [0]))))
            f_svd, f_c, t_svd, t_c = self.inner.log[-1][:4]
            self.records.append(dict(step=(self.calls - 1) // (2 * (N - 1)), b=b, dirn=dirn, tail_svd=float(t_svd), tail_spcf=float(t_c),
                                     f_svd=float(f_svd), f_spcf=float(f_c),
                                     dz_svd=(svd[0] - pre[0]).tolist(), dz_spcf=(sp[0] - pre[0]).tolist(),
                                     de_svd=(svd[1] - pre[1]).tolist(), de_spcf=(sp[1] - pre[1]).tolist()))
        return res


cut = Spy(spcfast.SPCFast(model, N, **{**BEST, **kw}))
M.run_tebd(model, N, chi, n, dt, cut, gates=G)
R = cut.records
json.dump(R, open(out, 'w'))
s0 = min(r['step'] for r in R)
phases = [('early (first 3 firing steps)', [r for r in R if r['step'] <= s0 + 2]), ('late (rest)', [r for r in R if r['step'] > s0 + 2])]
print(f'{model} N={N} T={T} chi={chi}: {len(R)} fired cuts, first firing step {s0}')
for name, rs in phases:
    if not rs:
        continue
    print(f'\n--- {name}: {len(rs)} cuts, mean discarded weight {np.mean([r["tail_svd"] for r in rs]):.1e} (svd) '
          f'{np.mean([r["tail_spcf"] for r in rs]):.1e} (spcf), region residual after/before {np.mean([r["f_spcf"] / r["f_svd"] for r in rs]):.2f}')
    print('  rms change of <Z_i> vs pre-cut state, by site distance d from the cut (region reaches d=%d):' % BEST['a'])
    print('   d      svd        spcf     spcf/svd')
    for d in list(range(0, 8)):
        a, c = [], []
        for r in rs:
            b = r['b']
            for i in (b - d, b + 1 + d):
                if 0 <= i < N:
                    a.append(r['dz_svd'][i]); c.append(r['dz_spcf'][i])
        if a:
            ra, rc = np.sqrt(np.mean(np.square(a))), np.sqrt(np.mean(np.square(c)))
            print(f'  {d:2d}  {ra:9.2e}  {rc:9.2e}  {rc / max(ra, 1e-300):7.1f}x')
    print('  rms change of local energy <h_j> by bond offset k = j - b (0 = cut bond):')
    for k in range(-5, 6):
        a, c = [], []
        for r in rs:
            j = r['b'] + k
            if 0 <= j < N - 1:
                a.append(r['de_svd'][j]); c.append(r['de_spcf'][j])
        if a:
            ra, rc = np.sqrt(np.mean(np.square(a))), np.sqrt(np.mean(np.square(c)))
            print(f'  {k:+2d}  {ra:9.2e}  {rc:9.2e}  {rc / max(ra, 1e-300):7.1f}x')
    tot_s = np.array([np.sum(r['de_svd']) for r in rs]); tot_c = np.array([np.sum(r['de_spcf']) for r in rs])
    print(f'  total energy change per cut: svd mean {tot_s.mean():+.2e} (rms {np.sqrt(np.mean(tot_s**2)):.1e}), '
          f'spcf mean {tot_c.mean():+.2e} (rms {np.sqrt(np.mean(tot_c**2)):.1e})')
