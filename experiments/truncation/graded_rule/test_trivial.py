"""How narrow can the operator windows be? Window width W and error of the lookahead bank vs the trivial-tensor tolerance.

    PYLIBS=<dir with jax> python test_trivial.py ising 12 3,6 1e-4 1e-7,1e-5,1e-4,1e-3,1e-2
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import heis_ops as H
import lookahead as LA
import jax.numpy as jnp

model, N = sys.argv[1], int(sys.argv[2])
ms = tuple(int(x) for x in sys.argv[3].split(','))
eps = float(sys.argv[4])
tols = [float(x) for x in sys.argv[5].split(',')]
dt = 0.1
Gs = M.make_gates(model, N, dt)


def step(v):
    for b in range(N - 1):
        v = M.dense_gate(v, Gs[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, Gs[b], b, N)
    return v


def mps_exact(v):
    T_, rest, left = [], v.reshape(1, -1), 1
    for i in range(N - 1):
        m = rest.reshape(left * 2, -1)
        U, s, Vh = np.linalg.svd(m, full_matrices=False)
        T_.append(U.reshape(left, 2, -1))
        rest = s[:, None] * Vh
        left = U.shape[1]
    T_.append(rest.reshape(left, 2, 1))
    return T_


v = M.mps_to_dense(M.initial_mps(model, N))
for _ in range(20):
    v = step(v)
v = v / np.linalg.norm(v)
Ts = [jnp.asarray(t) for t in mps_exact(v)]
strings = [s for s in H.operator_set(N, kmax=2)]
truth = {}
w = v
for m in range(1, max(ms) + 1):
    w = step(w)
    if m in ms:
        t = w.reshape((2,) * N)
        vals = []
        for sites, labels in strings:
            out = t
            for s, lab in zip(sites, labels):
                out = np.moveaxis(np.tensordot(H.PAULI[lab], out, axes=([1], [s])), 0, s)
            vals.append(np.vdot(t, out).real)
        truth[m] = np.array(vals)
print(f"{model} N={N} eps={eps}")
print(f"{'tol':>8}" + "".join(f"  m={m}: W   Dp  maxerr    rmserr" for m in ms))
for tol in tols:
    LA.TRIVIAL_TOL = tol
    try:
        banks = LA.build_banks(model, N, dt, ms, ks=(1, 2), eps=eps)
    except AssertionError as e:
        print(f"{tol:>8.0e}  window assertion: {str(e)[:60]}")
        continue
    row = f"{tol:>8.0e}"
    for m in ms:
        b = banks[m]
        val = np.asarray(LA.bank_values(Ts, LA.bank_args(b), b.W, 60))
        d = val - truth[m]
        row += f"  {'':>4}{b.W:>3}{b.Dp:>5}{np.max(np.abs(d)):>9.1e}{np.sqrt(np.mean(d ** 2)):>10.1e}"
    print(row, flush=True)
