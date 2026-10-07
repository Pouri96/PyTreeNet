"""Stage 2a: how large are the Heisenberg-evolved operator MPOs (U^dag)^m P U^m, and how accurate are they at tolerance eps?

For 1- and 2-site Pauli strings P near the chain centre, m up to MMAX Strang steps of dt=0.1:
  bond   largest MPO bond dimension after the SVD truncation at relative tolerance eps
  err    max |<psi|W_eps|psi> - <U^m psi|P|U^m psi>| over the strings, psi a dense evolved state (N <= 16 only)

    python probe_heis_rank.py ising 12,16 1e-3,1e-5,1e-7 20
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import heis_ops as H

model = sys.argv[1]
Ns = [int(x) for x in sys.argv[2].split(',')]
epss = [float(x) for x in sys.argv[3].split(',')]
MMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 20
TSTATE = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
dt = 0.1
MS = [1, 2, 3, 4, 6, 8, 10]
MS = [m for m in MS if m <= MMAX]
DMAX = int(os.environ.get('DMAX', 150))
GAMMA = float(os.environ.get('GAMMA', 0.0))


def mps_exact(v, N):
    T_, rest, left = [], v.reshape(1, -1), 1
    for i in range(N - 1):
        m = rest.reshape(left * 2, -1)
        U, s, Vh = np.linalg.svd(m, full_matrices=False)
        T_.append(U.reshape(left, 2, -1))
        rest = s[:, None] * Vh
        left = U.shape[1]
    T_.append(rest.reshape(left, 2, 1))
    return T_


def mpo_expect(W, A):
    L = np.ones((1, 1, 1), dtype=complex)
    for w, a in zip(W, A):
        L = np.einsum('awb,aoc,woiv,bid->cvd', L, a.conj(), w, a, optimize=True)
    return L[0, 0, 0]


def pauli_expect(v, N, sites, labels):
    t = v.reshape((2,) * N).copy()
    out = t
    for s, lab in zip(sites, labels):
        out = np.moveaxis(np.tensordot(H.PAULI[lab], out, axes=([1], [s])), 0, s)
    return np.vdot(t, out)


def strings(N):
    c = N // 2 - 1
    ops = []
    for lab in 'XYZ':
        ops.append(((c,), lab))
    for lab in ('ZZ', 'XX', 'YY', 'XZ', 'ZX'):
        ops.append(((c, c + 1), lab))
    return ops


for N in Ns:
    Gs = M.make_gates(model, N, dt)
    ops = strings(N)
    dense = N <= 16
    if dense:
        v = M.mps_to_dense(M.initial_mps(model, N))

        def step(v):
            for b in range(N - 1):
                v = M.dense_gate(v, Gs[b], b, N)
            for b in range(N - 2, -1, -1):
                v = M.dense_gate(v, Gs[b], b, N)
            return v
        for _ in range(int(round(TSTATE / dt))):
            v = step(v)
        v = v / np.linalg.norm(v)
        A = mps_exact(v, N)
        truth = {}
        w = v
        for m in range(0, MMAX + 1):
            if m > 0:
                w = step(w)
            if m in MS:
                truth[m] = [pauli_expect(w, N, s, l).real for s, l in ops]
    print(f"\n{model} N={N}: Heisenberg-evolved operator MPOs, dmax={DMAX}")
    print(f"{'eps':>8}{'m':>4}{'bond':>6}{'secs':>7}" + (f"{'err':>11}" if dense else ''), flush=True)
    for eps in epss:
        bonds = {m: 0 for m in MS}
        errs = {m: 0.0 for m in MS}
        t0 = time.time()
        for (s, l), k in zip(ops, range(len(ops))):
            W0 = H.local_op(N, s, l)
            res = H.evolve(W0, Gs, N, MMAX, MS, eps=eps, dmax=DMAX, gamma=GAMMA)
            for m in MS:
                bonds[m] = max(bonds[m], max(w.shape[0] for w in res[m]))
                if dense:
                    errs[m] = max(errs[m], abs(mpo_expect(res[m], A).real - truth[m][k]))
        secs = time.time() - t0
        for m in MS:
            print(f"{eps:>8.0e}{m:>4}{bonds[m]:>6}{secs:>7.0f}" + (f"{errs[m]:>11.2e}" if dense else ''), flush=True)
