"""Dense Trotter reference trajectories (same gates and gate order as the TEBD and Heisenberg runs), cached.

marg[n, i, k] = <psi(n dt)| sigma_k(i) |psi(n dt)>, k = 0,1,2 for X,Y,Z, for n = 0..nsteps, all sites i.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import _paths  # noqa: F401,E402
import mpsenh as M  # noqa: E402

CACHE = _HERE / 'results' / 'refcache'


def marginals(v, N):
    out = np.zeros((N, 3))
    t = v.reshape(1, -1)
    for i in range(N):
        w = v.reshape(2 ** i, 2, 2 ** (N - i - 1))
        rho = np.einsum('aib,ajb->ij', w.conj(), w)       # rho[i,j] = sum conj(w_i) w_j  (transposed rdm)
        nrm = np.trace(rho).real
        for k, P in enumerate((M.X, M.Y, M.Z)):
            # <P> = tr(rdm P) with rdm[j,i] = rho[i,j]  ->  sum_{ij} rho[i,j] P[i,j]... use rdm = rho.T
            out[i, k] = np.trace(rho.T @ P).real / nrm
    return out


def dense_traj(model, N, nsteps, dt):
    """Returns marg (nsteps+1, N, 3).  Cached under results/refcache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'traj_{model}_N{N}_n{nsteps}_dt{dt}.npy'
    if path.exists():
        return np.load(path)
    Gs = M.make_gates(model, N, dt)
    v = M.mps_to_dense(M.initial_mps(model, N))
    out = np.zeros((nsteps + 1, N, 3))
    out[0] = marginals(v, N)
    for n in range(1, nsteps + 1):
        for b in range(N - 1):
            v = M.dense_gate(v, Gs[b], b, N)
        for b in range(N - 2, -1, -1):
            v = M.dense_gate(v, Gs[b], b, N)
        out[n] = marginals(v, N)
    np.save(path, out)
    return out


def dense_state_traj(model, N, nsteps, dt):
    """Generator over (n, state vector) for n = 0..nsteps (not cached; used by the attribution script)."""
    Gs = M.make_gates(model, N, dt)
    v = M.mps_to_dense(M.initial_mps(model, N))
    yield 0, v
    for n in range(1, nsteps + 1):
        for b in range(N - 1):
            v = M.dense_gate(v, Gs[b], b, N)
        for b in range(N - 2, -1, -1):
            v = M.dense_gate(v, Gs[b], b, N)
        yield n, v


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    t0 = time.time()
    m = dense_traj(model, N, int(round(T / dt)), dt)
    print(model, N, T, 'done', f'{time.time() - t0:.1f}s', 'Z[N//2-1] at T:', m[-1, N // 2 - 1, 2])
