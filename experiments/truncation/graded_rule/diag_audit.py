"""Objective audit at every fired cut: does the fit lower the quantities it targets, and what happens to everything else?

    python diag_audit.py model N T chi [spcf kwargs json]
For each fired cut the 6-site (a=2) region reduced state of the pre-cut state, the SVD-cut state and the spcf-cut state is expanded in
Pauli strings.  The residual of a cut state is its string-coefficient difference to the pre-cut state, reported as an rms over strings in
classes: span 1, span 2, span 3 (static, what the objective sees), the tau=1 evolved version of the same set, and strings of span 4-6
(NOT in the objective).  Also the change of the Pauli strings of the whole chain outside the region, summarised by <Z_i>.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, itertools
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
from spcfast import _pstring

model, N, T, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
A_ = {**BEST, **kw}['a']
dt = 0.1
n = int(round(T / dt))
G = M.make_gates(model, N, dt)
L = 2 * A_ + 2
allS = [s for s in itertools.product(range(4), repeat=L) if any(s)]
span = np.array([max(i for i, x in enumerate(s) if x) - min(i for i, x in enumerate(s) if x) + 1 for s in allS])
Pm = np.array([_pstring(s) for s in allS])
Hm = np.zeros((2 ** L, 2 ** L), dtype=complex)
for bb in range(L - 1):
    Hm += np.kron(np.kron(np.eye(2 ** bb), M.h_bond(model, 3, N)), np.eye(2 ** (L - bb - 2)))
ev, V = np.linalg.eigh(Hm)
U1 = (V * np.exp(-1j * 1.0 * ev)) @ V.conj().T


def dense_theta(Tm, b, th):
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


def region_rho(v, lo, hi):
    t = v.reshape(2 ** lo, 2 ** (hi - lo), -1)
    return np.einsum('asb,atb->st', t, t.conj())


def coeffs(rho):
    return np.einsum('sij,ji->s', Pm, rho).real


def zprof(v):
    p = (np.abs(v) ** 2).reshape([2] * N)
    return np.array([np.take(p, 0, axis=i).sum() - np.take(p, 1, axis=i).sum() for i in range(N)])


class Spy:
    def __init__(self, inner):
        self.inner, self.rec, self.calls = inner, [], 0

    def start(self, Tm):
        self.Tm = Tm
        self.inner.start(Tm)

    def __call__(self, theta, chi_, dirn, A, B, b):
        f0 = self.inner.fired
        res = self.inner(theta, chi_, dirn, A, B, b)
        self.calls += 1
        lo, hi = b - A_, b + 2 + A_
        if self.inner.fired > f0 and lo >= 0 and hi <= N:
            Tm = self.Tm
            ts = M.svd_cut(theta, chi_, dirn)
            vp = dense_theta(Tm, b, theta)
            vs = dense_theta(Tm, b, np.tensordot(ts[0], ts[1], axes=([2], [0])))
            vc = dense_theta(Tm, b, np.tensordot(res[0], res[1], axes=([2], [0])))
            rp, rs, rc = (region_rho(v, lo, hi) for v in (vp, vs, vc))
            cp = coeffs(rp)
            d = {}
            for name, rr in (('svd', rs), ('spcf', rc)):
                dc = coeffs(rr) - cp
                de = coeffs(U1 @ rr @ U1.conj().T) - coeffs(U1 @ rp @ U1.conj().T)
                d[name] = dict(s1=float(np.sqrt(np.mean(dc[span == 1] ** 2))), s2=float(np.sqrt(np.mean(dc[span == 2] ** 2))),
                               s3=float(np.sqrt(np.mean(dc[span == 3] ** 2))), s456=float(np.sqrt(np.mean(dc[span >= 4] ** 2))),
                               ev=float(np.sqrt(np.mean(de[span <= 3] ** 2))))
            zp = zprof(vp)
            outside = np.array([i for i in range(N) if i < lo or i >= hi])
            d['far_z'] = {nm: float(np.sqrt(np.mean((zprof(v)[outside] - zp[outside]) ** 2))) for nm, v in (('svd', vs), ('spcf', vc))} if len(outside) else {}
            d['step'] = (self.calls - 1) // (2 * (N - 1))
            lg = self.inner.log[-1]
            d['tail'] = float(lg[2]); d['f_svd'] = float(lg[0]); d['f_spcf'] = float(lg[1]); d['b'] = b
            self.rec.append(d)
        return res


cut = Spy(spcfast.SPCFast(model, N, **{**BEST, **kw}))
M.run_tebd(model, N, chi, n, dt, cut, gates=G)
R = cut.rec
json.dump(R, open(os.environ.get('AUDIT_OUT', 'audit_last.json'), 'w'))
print(f'{model} N={N} T={T} chi={chi}: {len(R)} audited cuts (region {L} sites)')
print('rms over strings of the change relative to the pre-cut state, mean over cuts;  ratio = spcf/svd')
print('class                 svd        spcf     ratio')
for key, label in (('s1', 'span 1 (static, in objective)'), ('s2', 'span 2 (static, in objective)'), ('s3', 'span 3 (static, in objective)'),
                   ('ev', 'tau=1 evolved, span<=3 (obj.)'), ('s456', 'span 4-6 (NOT in objective)')):
    a, c = np.mean([r['svd'][key] for r in R]), np.mean([r['spcf'][key] for r in R])
    print(f'{label:30s} {a:9.2e}  {c:9.2e}  {c / a:6.2f}x')
fz = [r for r in R if r['far_z']]
if fz:
    a, c = np.mean([r['far_z']['svd'] for r in fz]), np.mean([r['far_z']['spcf'] for r in fz])
    print(f'{"<Z_i> outside the region":30s} {a:9.2e}  {c:9.2e}  {c / a:6.2f}x')
fr = np.array([[r['svd'][k] ** 2 for k in ('s1', 's2', 's3')] for r in R]); fc = np.array([[r['spcf'][k] ** 2 for k in ('s1', 's2', 's3')] for r in R])
print('fraction of cuts where spcf improved span1 / span2 / span3 / evolved:',
      ' / '.join(f'{np.mean(np.array([r["spcf"][k] < r["svd"][k] for r in R])):.2f}' for k in ('s1', 's2', 's3', 'ev')))
