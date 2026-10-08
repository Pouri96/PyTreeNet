"""Exact references for the rank 10 transport benchmark.

(1) ``exact_heis``: the dense Heisenberg operator O(n dt) = U_step^dag ... O U_step^n as a flat real 4^N vector (N <= 12), with the
    same Trotter gates and gate order as the MPO sweeps, so only truncation error is measured.  Snapshots go to rank10/_refcache/.
(2) ``typicality_*`` (N = 16-20, Test 3 only): C(x,t) ~ <r| q_x U^dag q_c U |r> for a few random states, same circuit.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import time
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pmpo as P  # noqa: E402
import mpsenh as M  # noqa: E402
import pauli_prop as PP  # noqa: E402

CACHE = HERE / '_refcache'


def circuit16(model, N, dt):
    return [g.reshape(16, 16) for g in P.pauli_gates(model, N, dt)]


def exact_heis(model, N, nsteps, dt, c0, snap_steps=(), verbose=False):
    """Returns (c1 (nsteps+1, N, 4), c2 (nsteps+1, N-1, 4, 4), norm2 (nsteps+1), snaps {step: flat c})."""
    G16 = circuit16(model, N, dt)
    c = c0.copy()
    c1 = np.zeros((nsteps + 1, N, 4))
    c2 = np.zeros((nsteps + 1, N - 1, 4, 4))
    n2 = np.zeros(nsteps + 1)
    c1[0], c2[0] = P.local_coeffs_dense(c, N)
    n2[0] = float(c @ c)
    snaps = {}
    t0 = time.time()
    for n in range(1, nsteps + 1):
        for b in range(N - 1):
            c = P.dense_gate(c, G16[b], b, N)
        for b in range(N - 2, -1, -1):
            c = P.dense_gate(c, G16[b], b, N)
        c1[n], c2[n] = P.local_coeffs_dense(c, N)
        n2[n] = float(c @ c)
        if n in snap_steps:
            snaps[n] = c.copy()
        if verbose:
            print(f'  exact step {n}/{nsteps}  norm2={n2[n]:.12f}  {time.time() - t0:.0f}s', flush=True)
    return c1, c2, n2, snaps


def start_operator(model, N, c, kind):
    """Dense flat vector of the start operator: 'E' = eps_c (energy density on bond (c, c+1)), 'Z' = Z_c, 'X' = X_c."""
    if kind == 'E':
        return P.init_dense_local(N, c, P.eps_matrix(model))
    v = np.zeros(4)
    v[{'X': 1, 'Y': 2, 'Z': 3}[kind]] = 1.0
    return P.init_dense_local(N, c, v)


def start_mpo(model, N, c, kind):
    if kind == 'E':
        return P.init_local(N, c, P.eps_matrix(model))
    v = np.zeros(4)
    v[{'X': 1, 'Y': 2, 'Z': 3}[kind]] = 1.0
    return P.init_local(N, c, v)


def cached_reference(model, N, c, kind, nsteps, dt, snap_steps):
    """Cached exact reference.  Local coefficient histories in an .npz, snapshots as .npy (float64, 4^N)."""
    tag = f'{model}_N{N}_c{c}_{kind}_n{nsteps}_dt{dt}'
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f'ref_{tag}.npz'
    sn = {s: CACHE / f'snap_{tag}_s{s}.npy' for s in snap_steps}
    if f.exists() and all(p.exists() for p in sn.values()):
        z = np.load(f)
        return z['c1'], z['c2'], z['n2'], sn
    c1, c2, n2, snaps = exact_heis(model, N, nsteps, dt, start_operator(model, N, c, kind), snap_steps, verbose=True)
    for s, v in snaps.items():
        np.save(sn[s], v)
    np.savez(f, c1=c1, c2=c2, n2=n2)
    return c1, c2, n2, sn


def validate(N=6, nsteps=5, dt=0.1, models=('ising', 'isingT', 'heis')):
    """Dense Heisenberg operator vs Pauli propagation (eps = 0) and vs the dense state, on the Neel state; max abs deviation."""
    out = {}
    for model in models:
        G16 = circuit16(model, N, dt)
        worst_pp, worst_dense = 0.0, 0.0
        ex = M.dense_reference(model, N, nsteps, dt)
        obs = PP.sample_observables(N)
        exv = PP.exact_values(ex, N, obs)
        for k, (st, _) in enumerate(obs):
            c = np.zeros(4 ** N)
            key = 0
            for s, p in st.items():
                key += p * 4 ** (N - 1 - s)
            c[key] = 1.0
            for n in range(nsteps):
                for b in range(N - 1):
                    c = P.dense_gate(c, G16[b], b, N)
                for b in range(N - 2, -1, -1):
                    c = P.dense_gate(c, G16[b], b, N)
            # expectation on Neel: only I and Z contribute, Z = +1 on even sites, -1 on odd sites
            t = c.reshape([4] * N)
            for s in range(N - 1, -1, -1):
                v = np.array([1.0, 0, 0, 1.0 if s % 2 == 0 else -1.0])
                t = np.tensordot(t, v, axes=([s], [0]))
            val = float(t)
            ppv = PP.propagate(model, N, nsteps, dt, st, 0.0)[0]
            worst_pp = max(worst_pp, abs(val - ppv))
            worst_dense = max(worst_dense, abs(val - exv[k]))
        out[model] = dict(vs_pauli_prop=worst_pp, vs_dense_state=worst_dense)
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'validate':
        r = validate()
        print(json.dumps(r, indent=1))
        json.dump(r, open(HERE / 'results' / 'heis_ref_validate.json', 'w'), indent=1)
    elif sys.argv[1] == 'build':
        # python rank10/heis_ref.py build MODEL N C KIND NSTEPS DT SNAPS(comma)
        model, N, c, kind, ns, dt = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), float(sys.argv[7])
        snaps = [int(s) for s in sys.argv[8].split(',')]
        t0 = time.time()
        cached_reference(model, N, c, kind, ns, dt, snaps)
        print('done', time.time() - t0)
