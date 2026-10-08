"""How faithful is the lookahead proxy?  No truncation involved.

The cut objective evolves the region's reduced state under the ISOLATED region Hamiltonian for a time tau and compares the 1-3 site
marginals.  Here the exact state at t0 is evolved exactly (dense, same Trotter gates) and the true reduced state of the same region
at t0 + tau is compared with that proxy:
    explained(tau) = 1 - || P3(rho_true(tau) - rho_proxy(tau)) || / || P3(rho_true(tau) - rho(t0)) ||
where P3 keeps the Pauli strings supported inside a window of at most 3 contiguous sites (what the objective sees).  1 = the proxy
reproduces the true change, 0 = no better than not evolving at all, < 0 = worse than not evolving.

    python proxy_check.py model N t0 a b1,b2,...        e.g.  python proxy_check.py heis 16 0.8 2 7
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, itertools
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
from spcfast import _pstring

model, N, t0, a = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
bonds = [int(x) for x in sys.argv[5].split(',')]
dt = 0.1
G = M.make_gates(model, N, dt)
Tex, _ = M.run_tebd(model, N, 4096, int(round(t0 / dt)), dt, M.svd_cut, gates=G)
v = M.mps_to_dense(Tex)
v = v / np.linalg.norm(v)


def dstep(v):
    for b in range(N - 1):
        v = M.dense_gate(v, G[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, G[b], b, N)
    return v


def rdm(v, lo, hi):
    t = v.reshape(2 ** lo, 2 ** (hi - lo), -1)
    return np.einsum('asb,atb->st', t, t.conj())


def strings(L, kmax=3):
    keys = set()
    for k in range(1, kmax + 1):
        for off in range(L - k + 1):
            for s in itertools.product(range(4), repeat=k):
                if any(s):
                    keys.add((0,) * off + s + (0,) * (L - off - k))
    return sorted(keys)


taus = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
print(f'{model} N={N} t0={t0} region a={a}  (explained fraction of the true change of the <=3-site marginals)')
print('bond | ' + ' '.join(f'tau={t:<5g}' for t in taus))
res = {}
for b in bonds:
    lo, hi = b - a, b + 2 + a
    L = hi - lo
    Pm = np.array([_pstring(s) for s in strings(L)])
    Hm = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for bb in range(L - 1):
        Hm += np.kron(np.kron(np.eye(2 ** bb), M.h_bond(model, 3, N)), np.eye(2 ** (L - bb - 2)))
    ev, V = np.linalg.eigh(Hm)
    rho0 = rdm(v, lo, hi)
    c = lambda rho: np.einsum('sij,ji->s', Pm, rho).real
    c0 = c(rho0)
    out = []
    vt = v.copy()
    done = 0
    for tau in taus:
        n = int(round(tau / dt))
        while done < n:
            vt = dstep(vt); done += 1
        U = (V * np.exp(-1j * tau * ev)) @ V.conj().T
        cp = c(U @ rho0 @ U.conj().T)
        ct = c(rdm(vt, lo, hi))
        out.append(1 - np.linalg.norm(ct - cp) / max(np.linalg.norm(ct - c0), 1e-300))
    res[b] = out
    print(f'{b:4d} | ' + ' '.join(f'{x:9.2f}' for x in out))
mean = np.mean([res[b] for b in bonds], axis=0)
print('mean | ' + ' '.join(f'{x:9.2f}' for x in mean))
