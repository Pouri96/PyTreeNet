"""Refit of the benefit/cost gate constants on Ising operator TEBD (in-sample, as in the state case).

For every FIRED cut the collateral damage is measured on the dense state (N = 10): rms difference of the k = 1-3 window label marginals that lie
entirely OUTSIDE the fit region, between the state with the tilted cut and the state with the plain SVD cut.  The benefit is the rms span-2 residual
of the SVD point (what the fit can repair), in the same probability units.  Fit  damage = c0 * (tail / 1e-4)^exp  by least squares in log-log.
    python rank1/gate_fit.py chi_list out.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import op_tebd as O
import op_bench as B
import spcop


class SPCOpProbe(spcop.SPCOp):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.probe = []

    def _truncate(self, theta, Mm, U, s, Vh, k, l, r, dirn, tail, nrm2, b):
        out = super()._truncate(theta, Mm, U, s, Vh, k, l, r, dirn, tail, nrm2, b)
        if not out[3]:
            return out
        N = self.N
        lo, hi = max(0, b - self.aL), min(N, b + 2 + self.aR)
        alt = spcop._plain(U, s, Vh, k, l, r, dirn)
        T1, T2 = list(self.T), list(self.T)
        T1[b], T1[b + 1] = out[0], out[1]
        T2[b], T2[b + 1] = alt[0], alt[1]
        c1, c2 = O.mps_to_dense(T1), O.mps_to_dense(T2)
        c1, c2 = c1 / np.linalg.norm(c1), c2 / np.linalg.norm(c2)
        d2, cnt = 0.0, 0
        a1, a2 = (c1 * c1).reshape([4] * N), (c2 * c2).reshape([4] * N)
        for kk in (1, 2, 3):
            for x in range(N - kk + 1):
                if x + kk - 1 < lo or x >= hi:
                    other = tuple(j for j in range(N) if not (x <= j < x + kk))
                    dd = a1.sum(axis=other) - a2.sum(axis=other)
                    d2 += float(np.sum(dd ** 2))
                    cnt += dd.size
        if cnt:
            self.probe.append((tail, self.log[-1][8], np.sqrt(d2 / cnt), b, 1.0 if dirn == 'R' else 0.0, self.log[-1][10], self.log[-1][11]))
        return out


if __name__ == '__main__':
    chis = [int(c) for c in sys.argv[1].split(',')]
    out = sys.argv[2]
    model, N, T, dt = 'ising', 10, 6.0, 0.1
    nsteps = int(round(T / dt))
    gops = O.pauli_gates(model, N, dt)
    allp = {}
    for chi in chis:
        cut = SPCOpProbe(N)
        O.run_op_tebd(N, chi, nsteps, gops, cut)
        allp[chi] = np.array(cut.probe)
        print(f'chi={chi}: fired {cut.fired}, probed {len(cut.probe)}', flush=True)
    P = np.concatenate([allp[c] for c in chis])
    tail, bres, dmg = P[:, 0], P[:, 1], P[:, 2]
    good = (dmg > 0) & (tail > 0)
    x, y = np.log(tail[good] / 1e-4), np.log(dmg[good])
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c0, ex = float(np.exp(coef[0])), float(coef[1])
    resid = y - A @ coef
    print(f'fit damage = c0 (tail/1e-4)^exp: c0 = {c0:.3e}, exp = {ex:.3f}, log-rms scatter {resid.std():.2f} ({np.exp(resid.std()):.1f}x), n = {good.sum()}')
    print('median benefit/damage ratio', float(np.median(bres[good] / dmg[good])), ' fraction with benefit >= damage:', float(np.mean(bres[good] >= dmg[good])))
    json.dump(dict(c0=c0, exp=ex, scatter=float(resid.std()), n=int(good.sum()), probe={str(c): allp[c].tolist() for c in chis},
                   cols=['tail', 'Bres_span2_rms', 'damage_far_rms', 'b', 'dirR', 'f_svd', 'f_final']), open(out, 'w'))
