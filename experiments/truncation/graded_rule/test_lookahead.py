"""Unit test: MPO-bank expectation values against the dense propagated marginals (Ising/Heisenberg, small N).

    PYLIBS=<dir with jax> python test_lookahead.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import heis_ops as H
import lookahead as LA
import jax.numpy as jnp

rng = np.random.default_rng(1)
dt = 0.1
ok = True
for model, N in (('ising', 8), ('isingdw', 9)):
    chi = 5
    bonds = [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]
    Ts = [(rng.normal(size=(bonds[i], 2, bonds[i + 1])) + 1j * rng.normal(size=(bonds[i], 2, bonds[i + 1]))) for i in range(N)]
    Gs = M.make_gates(model, N, dt)
    v = M.mps_to_dense(Ts)
    v = v / np.linalg.norm(v)

    def step(v):
        for b in range(N - 1):
            v = M.dense_gate(v, Gs[b], b, N)
        for b in range(N - 2, -1, -1):
            v = M.dense_gate(v, Gs[b], b, N)
        return v

    ms = (2, 3)
    t0 = time.time()
    banks = LA.build_banks(model, N, dt, ms, ks=(1, 2), eps=1e-9, dmax=60)
    strings = [s for s in H.operator_set(N, kmax=2)]
    w = v.copy()
    truth = {}
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
    for m in ms:
        b = banks[m]
        val = np.asarray(LA.bank_values([jnp.asarray(t) for t in Ts], LA.bank_args(b), b.W, 60, single=False))
        val32 = np.asarray(LA.bank_values([jnp.asarray(t) for t in Ts], LA.bank_args(b), b.W, 60, single=True))
        err = np.max(np.abs(val - truth[m]))
        err32 = np.max(np.abs(val32 - truth[m]))
        print(f"{model} N={N} m={m}: window={b.W} Dp={b.Dp} ops={b.n} max|err|={err:.2e}  complex64 {err32:.2e}  ({time.time() - t0:.1f}s)")
        ok &= err < 1e-6 and err32 < 1e-4
print('PASS' if ok else 'FAIL')
