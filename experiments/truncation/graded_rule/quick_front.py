"""SVD-compress-every-step front on a dense cell, no fit: to choose cells whose SVD infidelity is high but not lost."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
model, N = sys.argv[1], int(sys.argv[2]); Ts = [float(x) for x in sys.argv[3].split(',')]; chis = [int(c) for c in sys.argv[4].split(',')]
dt = 0.1; Gs = M.make_gates(model, N, dt)
def step(v):
    for b in list(range(N - 1)) + list(range(N - 2, -1, -1)):
        v = M.dense_gate(v, Gs[b], b, N)
    return v
def comp(v, chi):
    T_, rest, left = [], v.reshape(1, -1), 1
    for i in range(N - 1):
        U, s, Vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        k = min(chi, len(s)); T_.append(U[:, :k].reshape(left, 2, k)); rest = s[:k, None] * Vh[:k]; left = k
    T_.append(rest.reshape(left, 2, 1)); return M.mps_to_dense(T_)
v0 = M.mps_to_dense(M.initial_mps(model, N))
ex = v0.copy(); ap = {c: v0.copy() for c in chis}; t = 0.0
print(f"{model} N={N}: SVD compress-every-step infidelity")
print(f"{'T':>5}" + "".join(f"{'chi='+str(c):>10}" for c in chis))
for T in Ts:
    while t < T - 1e-9:
        ex = step(ex)
        for c in chis:
            v = step(ap[c]); v /= np.linalg.norm(v); ap[c] = comp(v, c); ap[c] /= np.linalg.norm(ap[c])
        t += dt
    e = ex / np.linalg.norm(ex)
    print(f"{T:>5.1f}" + "".join(f"{1 - abs(np.vdot(e, ap[c])) ** 2:>10.2e}" for c in chis), flush=True)
