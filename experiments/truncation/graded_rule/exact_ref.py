"""Continuous-time exact reference by expm_multiply, cached, plus the Trotter floor of the dense Trotter reference."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import _paths  # noqa: F401
import mpsenh as M

REF = Path(__file__).resolve().parent / '_refcache'


def sparse_H(model, N):
    H = sp.csr_matrix((2 ** N, 2 ** N), dtype=complex)
    for b in range(N - 1):
        h = M.h_bond(model, b, N)
        H = H + sp.kron(sp.kron(sp.identity(2 ** b), sp.csr_matrix(h), format='csr'), sp.identity(2 ** (N - b - 2)), format='csr')
    return H.tocsr()


def continuous(model, N, T):
    path = REF / f'cont_{model}_N{N}_T{T}.npy'
    if path.exists():
        return np.load(path)
    REF.mkdir(exist_ok=True)
    v0 = M.mps_to_dense(M.initial_mps(model, N))
    v = expm_multiply(-1j * T * sparse_H(model, N), v0)
    np.save(path, v)
    return v


if __name__ == '__main__':
    model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    ex = continuous(model, N, T)
    print(f'{model} N={N} T={T}: Trotter reference vs continuous exact (error of local observables)')
    for dt in (0.1, 0.05, 0.025):
        tr = M.dense_reference(model, N, int(round(T / dt)), dt, gates=M.make_gates(model, N, dt))
        e = M.errors(ex, tr, N, model)
        print(f'  dt={dt:<6}' + '  '.join(f"{k}={e[k]:.2e}" for k in ('infid', 'single_rms', 'nn_rms', 'nnn_rms', 'E_abs')))
