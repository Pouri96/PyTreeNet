"""N = 12 robustness check of Test 1 (reduced): SVD plain / SVD back / spcf (plain) at a few chi, one cell.  Metrics from the purification vector directly
(no 4096 x 4096 density matrix): rdm2, rdm3, nn rms, nnn rms, |dE|.   python n12_check.py family mu T chis out.json
"""
import sys
import _p2  # noqa: F401
import json, time
import multiprocessing as mp
import numpy as np
import mpsenh as M
import purlib as P
from pur_bench import make_cut

fam, mu, T = sys.argv[1], (np.inf if sys.argv[2] == 'inf' else float(sys.argv[2])), float(sys.argv[3])
chis, out = [int(c) for c in sys.argv[4].split(',')], sys.argv[5]
N, model, dt = 12, 'ising', 0.1
n = int(round(T / dt))


def rdm_psi(psi, i, k):
    t = psi.reshape([2, 2] * N)
    ax = [2 * j for j in range(i, i + k)]
    rest = [a for a in range(2 * N) if a not in ax]
    t = t.transpose(ax + rest).reshape(2 ** k, -1)
    return t @ t.conj().T


class RS:
    def __init__(self, psi):
        psi = psi / np.linalg.norm(psi)
        self.r = {k: [rdm_psi(psi, i, k) for i in range(N - k + 1)] for k in (1, 2, 3)}


def obs(S):
    nn = S.r[2]
    nnn = [np.einsum('abcdbf->acdf', S.r[3][i].reshape(2, 2, 2, 2, 2, 2)).reshape(4, 4) for i in range(N - 2)]
    ev = lambda r, op: float(np.trace(r @ op).real)
    single = [ev(S.r[1][i], Pa) for i in range(N) for Pa in M.PAUL]
    nnv = [ev(r, np.kron(Pa, Q)) for r in nn for Pa in M.PAUL for Q in M.PAUL]
    nnnv = [ev(r, np.kron(Pa, Q)) for r in nnn for Pa in M.PAUL for Q in M.PAUL]
    E = sum(ev(nn[b], M.h_bond(model, b, N)) for b in range(N - 1))
    return dict(single=np.array(single), nn=np.array(nnv), nnn=np.array(nnnv), E=E)


def ref():
    path = P.REF / f'pur_{model}_N{N}_{fam}_mu{P.mu_tag(mu)}_plain_T{T:g}_dt{dt}.npy'
    if path.exists():
        return np.load(path)
    psi = P.dense_snapshots(model, N, fam, mu, 'plain', dt, [T])[T]
    P.REF.mkdir(exist_ok=True)
    np.save(path, psi)
    return psi


_C = {}


def one(job):
    arm, g, chi = job
    if 'ref' not in _C:
        psi = ref()
        S = RS(psi)
        _C['ref'] = (psi / np.linalg.norm(psi), S, obs(S))
    pex, Sx, ox = _C['ref']
    cut = make_cut(arm, model, N)
    Tm, wall = P.run_tebd_d(P.initial_pur_mps(N, fam, mu), P.pur_gates(model, N, dt, g), chi, n, cut)
    psi = P.mps_to_dense_d(Tm)
    S = RS(psi)
    o = obs(S)
    e = {k + '_rms': float(np.sqrt(np.mean((ox[k] - o[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
    e['E_abs'] = abs(ox['E'] - o['E'])
    for k, key in ((2, 'rdm2'), (3, 'rdm3')):
        e[key] = float(np.sqrt(np.mean([np.linalg.norm(Sx.r[k][i] - S.r[k][i]) ** 2 for i in range(N - k + 1)])))
    e.update(arm=arm, gauge=g, chi=chi, family=fam, mu='inf' if np.isinf(mu) else mu, T=T, N=N, wall=round(wall, 1), params=P.stored_params(Tm))
    if arm != 'svd':
        e.update(fired=cut.fired, calls=cut.calls)
    return e


if __name__ == '__main__':
    ref()
    jobs = [(a, g, c) for c in sorted(chis, reverse=True) for a, g in (('spcf', 'plain'), ('svd', 'plain'), ('svd', 'back'))]
    rows, t0 = [], time.time()
    with mp.Pool(3) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            rows.append(r)
            json.dump(rows, open(out, 'w'))
            print(f"{len(rows)}/{len(jobs)} chi={r['chi']} {r['arm']}_{r['gauge']} rdm2={r['rdm2']:.2e} nn={r['nn_rms']:.2e} nnn={r['nnn_rms']:.2e} {r['wall']}s {int(time.time() - t0)}s", flush=True)
