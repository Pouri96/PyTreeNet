"""DMT baseline from the GCG library (Pauli MPDO, d=4 fused sites) on the same Trotter circuit and Neel start.

    PYTHONPATH=<BUG>/GCG/src python dmt_baseline.py ising 12 4.0 0.1 "10,12,16,20,24" out.json
For each bond cap D it runs the circuit twice, with DMT(radius=1) and with the library's plain SVD, and scores single / nn / nnn
rms error and infidelity against the exact pure state, with the same observables as mps_bench. Stored parameters are the total tensor
size of the MPDO, so they can be matched against an MPS by parameters.
"""
import json
import sys
import time
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rule'))      # NOT _paths: it puts this worktree's older pytreenet first, which has no pauli package
import mpsenh as M
from gcg import DMT, PLAIN, PTMCG_PauliMPDO, PTMCGPauliMPDOConfig
from pytreenet.special_ttn.pauli import pauli_product_mps, pauli_site_ids, pauli_to_dense, pauli_trace

PAUL = M.PAUL


def rho_obs(rho, N, model):
    rho = rho / np.trace(rho).real

    def red(i, k):
        R = rho.reshape(2 ** i, 2 ** k, 2 ** (N - i - k), 2 ** i, 2 ** k, 2 ** (N - i - k))
        return np.einsum('asbatb->st', R)
    nn = [red(i, 2) for i in range(N - 1)]
    nnn = []
    for i in range(N - 2):
        R = rho.reshape(2 ** i, 2, 2, 2, 2 ** (N - i - 3), 2 ** i, 2, 2, 2, 2 ** (N - i - 3))
        nnn.append(np.einsum('apqrbaxqzb->prxz', R).reshape(4, 4))
    single = []
    for i in range(N):
        r = nn[i] if i < N - 1 else nn[N - 2]
        r = r.reshape(2, 2, 2, 2)
        rr = np.einsum('abcb->ac', r) if i < N - 1 else np.einsum('abad->bd', r)
        single += [np.trace(rr @ P).real for P in PAUL]
    nnv = [np.trace(r @ np.kron(P, Q)).real for r in nn for P in PAUL for Q in PAUL]
    nnnv = [np.trace(r @ np.kron(P, Q)).real for r in nnn for P in PAUL for Q in PAUL]
    return dict(single=np.array(single), nn=np.array(nnv), nnn=np.array(nnnv))


def run(model, N, D, policy_name, nsteps, dt):
    Gs = M.make_gates(model, N, dt)
    kets = [np.array([1, 0], dtype=complex) if i % 2 == 0 else np.array([0, 1], dtype=complex) for i in range(N)]
    state = pauli_product_mps(kets)
    pol = DMT(radius=1) if policy_name == 'dmt' else PLAIN
    app = PTMCG_PauliMPDO(state, PTMCGPauliMPDOConfig(max_bond_dim=D, truncation_policy=pol))
    ids = pauli_site_ids(app.state)
    t0, c0 = time.time(), time.process_time()
    for _ in range(nsteps):
        for b in range(N - 1):
            app.apply_gate([ids[b], ids[b + 1]], Gs[b].reshape(4, 4))
        for b in range(N - 2, -1, -1):
            app.apply_gate([ids[b], ids[b + 1]], Gs[b].reshape(4, 4))
    wall, cpu = time.time() - t0, time.process_time() - c0
    net = app.physical_state()
    params = int(sum(t.size for t in app.state.tensors.values()))
    rho = pauli_to_dense(net)
    return rho, params, wall, cpu


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    Ds = [int(x) for x in sys.argv[5].split(',')]
    out = sys.argv[6]
    ns = int(round(T / dt))
    refp = HERE / '_refcache' / f'{model}_N{N}_T{T}_dt{dt}.npy'
    ex = np.load(refp) if refp.exists() else M.dense_reference(model, N, ns, dt)
    ex = ex / np.linalg.norm(ex)
    a = M.local_obs(ex, N, model)
    res = []
    for D in Ds:
        for pol in ('dmt', 'plain'):
            rho, params, wall, cpu = run(model, N, D, pol, ns, dt)
            tr = np.trace(rho).real
            rho = rho / tr
            b = rho_obs(rho, N, model)
            e = {k + '_rms': float(np.sqrt(np.mean((a[k] - b[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
            e['infid'] = float(1 - np.real(np.vdot(ex, rho @ ex)))
            e.update(arm=f'{pol}:D{D}', D=D, params=params, wall=round(wall, 1), cpu=round(cpu, 1), trace=float(tr))
            res.append(e)
            json.dump(res, open(out, 'w'))
            print(f"D={D:>3} {pol:<5} params={params:>6} infid={e['infid']:.3e} single={e['single_rms']:.2e} nn={e['nn_rms']:.2e} "
                  f"nnn={e['nnn_rms']:.2e} cpu={cpu:.0f}s trace={tr:.4f}", flush=True)
