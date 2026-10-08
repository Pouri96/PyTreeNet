"""Regime selection from plain SVD alone: SVD-TEBD infidelity and nn error along the time axis, for a chi ladder.

    python svd_scan.py model N dt Tmax chi_list every out.json
The dense Trotter state is evolved alongside, so one pass gives every checkpoint time (every `every` steps). Pick a cell
whose SVD infidelity spans ~1e-1..1e-3 over the ladder before running any other arm on it.
"""
import sys, json, time
import numpy as np
import _paths  # noqa: F401
import mpsenh as M

model, N, dt, Tmax = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
chis = [int(c) for c in sys.argv[5].split(',')]
every, out = int(sys.argv[6]), sys.argv[7]
n = int(round(Tmax / dt))
G = M.make_gates(model, N, dt)


def step_dense(v):
    for b in range(N - 1):
        v = M.dense_gate(v, G[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, G[b], b, N)
    return v


def step_mps(T, chi):
    for rng, d in ((range(N - 1), 'R'), (range(N - 2, -1, -1), 'L')):
        for b in rng:
            th = np.einsum('abst,lstr->labr', G[b], np.tensordot(T[b], T[b + 1], axes=([2], [0])))
            T[b], T[b + 1], _, _ = M.svd_cut(th, chi, d)
    return T


rows = []
t0 = time.time()
v = M.mps_to_dense(M.initial_mps(model, N))
Ts = {chi: M.initial_mps(model, N) for chi in chis}
for s in range(1, n + 1):
    v = step_dense(v)
    for chi in chis:
        Ts[chi] = step_mps(Ts[chi], chi)
    if s % every == 0:
        line = [f't={s * dt:5.2f}']
        for chi in chis:
            e = M.errors(v, M.mps_to_dense(Ts[chi]), N, model)
            rows.append(dict(t=round(s * dt, 6), chi=chi, **e))
            line.append(f'chi{chi}: {e["infid"]:.1e}/{e["nn_rms"]:.1e}')
        print('  '.join(line), flush=True)
json.dump(rows, open(out, 'w'))
print(f'done in {time.time() - t0:.0f} s (entries infid/nn_rms)')
