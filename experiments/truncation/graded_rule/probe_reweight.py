"""Ceiling of Schmidt-weight refitting at ONE cut of a dense high-pressure state.

Keep the SVD subspace (k Schmidt pairs) and refit only the k amplitudes q_i so that the marginals of ALL
2-site and 3-site windows match the exact state's. Compare with plain (renormalised) SVD.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np
from scipy.optimize import least_squares
import _paths  # noqa: F401
import mpsenh as M
from hp_bench import rdmk

model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), 0.1
ks = [int(x) for x in sys.argv[4].split(',')]
nsteps = int(round(T / dt))
psi = M.dense_reference(model, N, nsteps, dt, gates=M.make_gates(model, N, dt))
psi = psi / np.linalg.norm(psi)
wins = [(i, k) for k in (1, 2, 3) for i in range(N - k + 1)]
exact = {w: rdmk(psi, N, *w) for w in wins}


def marg_err(v):
    v = v / np.linalg.norm(v)
    out = {}
    for kk in (1, 2, 3):
        e = [np.linalg.norm(rdmk(v, N, i, kk) - exact[(i, kk)]) ** 2 for i in range(N - kk + 1)]
        out[kk] = float(np.sqrt(np.mean(e)))
    return out


print(f"{model} N={N} T={T}: single-cut ceiling, error of ALL windows (rms Frobenius), middle bond")
for cut in (N // 2, N // 4):
    Mx = psi.reshape(2 ** cut, 2 ** (N - cut))
    U, s, Vh = np.linalg.svd(Mx, full_matrices=False)
    print(f"\nbond {cut}   entropy {-(s**2 * np.log(s**2 + 1e-300)).sum():.3f}   Schmidt rank>1e-8: {(s > 1e-8).sum()}")
    print(f"{'k':>4}{'discarded':>11}{'1-site':>22}{'2-site':>22}{'3-site':>22}   cost(1-F) svd/refit")
    print(f"{'':>15}{'svd':>11}{'refit':>11}{'svd':>11}{'refit':>11}{'svd':>11}{'refit':>11}")
    for k in ks:
        def build(q):
            return ((U[:, :k] * q) @ Vh[:k]).reshape(-1)
        q0 = s[:k] / np.linalg.norm(s[:k])
        v0 = build(q0)

        def resid(q):
            v = build(q); v = v / np.linalg.norm(v)
            r = []
            for kk in (1, 2, 3):
                for i in range(N - kk + 1):
                    r.append((rdmk(v, N, i, kk) - exact[(i, kk)]).reshape(-1))
            r = np.concatenate(r)
            return np.concatenate([r.real, r.imag])
        sol = least_squares(resid, q0, method='lm', xtol=1e-14, ftol=1e-14, max_nfev=200)
        q1 = np.abs(sol.x); v1 = build(q1)
        e0, e1 = marg_err(v0), marg_err(v1)
        c0 = 1 - abs(np.vdot(psi, v0 / np.linalg.norm(v0))) ** 2
        c1 = 1 - abs(np.vdot(psi, v1 / np.linalg.norm(v1))) ** 2
        print(f"{k:>4}{(s[k:]**2).sum():>11.2e}" + "".join(f"{e0[kk]:>11.2e}{e1[kk]:>11.2e}" for kk in (1, 2, 3)) + f"   {c0:.2e} / {c1:.2e}")
