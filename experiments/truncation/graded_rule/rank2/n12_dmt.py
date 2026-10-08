"""N = 12 extension of Test 2 (reduced): DMT MPDO and Frobenius MPDO against the same N = 12 purification reference as n12_check.py.
    python n12_dmt.py family mu T chis out.json
"""
import sys
_args = list(sys.argv)
sys.argv = [_args[0], _args[1], _args[2], _args[3], _args[4], 'unused.json']
import n12_check as C   # noqa: E402  (module-level argument parsing: family mu T chis)
import json, time
import multiprocessing as mp
import numpy as np
import purlib as P
import dmt_np as D

out = _args[5]
N, model, dt, fam, mu, T = C.N, C.model, C.dt, C.fam, C.mu, C.T
_R = {}


class RSrho:
    def __init__(self, rho):
        rho = rho / np.trace(rho).real
        self.r = {k: [P.rdm_k(rho, N, i, k) for i in range(N - k + 1)] for k in (1, 2, 3)}


def one(job):
    arm, chi = job
    if 'ref' not in _R:
        psi = C.ref()
        S = C.RS(psi)
        _R['ref'] = (S, C.obs(S))
    Sx, ox = _R['ref']
    cut = D.DMTCut(offset=8 if arm == 'dmt+8' else 0) if arm != 'frob' else D.FrobCut()
    Tm, wall = D.run_tebd_mpdo(D.initial_mpdo(N, fam, mu), D.mpdo_gates(model, N, dt), chi, C.n, cut)
    rho = D.vec_to_rho(D.mpdo_to_vec(Tm), N)
    rho = 0.5 * (rho + rho.conj().T)
    trace = float(np.trace(rho).real)
    S = RSrho(rho)
    del rho
    o = C.obs(S)
    e = {k + '_rms': float(np.sqrt(np.mean((ox[k] - o[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
    e['E_abs'] = abs(ox['E'] - o['E'])
    for k, key in ((2, 'rdm2'), (3, 'rdm3')):
        e[key] = float(np.sqrt(np.mean([np.linalg.norm(Sx.r[k][i] - S.r[k][i]) ** 2 for i in range(N - k + 1)])))
    e.update(arm=arm, chi=chi, family=fam, mu='inf' if np.isinf(mu) else mu, T=T, N=N, wall=round(wall, 1), trace=trace, params=P.stored_params(Tm))
    return e


if __name__ == '__main__':
    jobs = [(a, c) for c in sorted(C.chis, reverse=True) for a in ('dmt', 'frob')]
    rows, t0 = [], time.time()
    with mp.Pool(3) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            rows.append(r)
            json.dump(rows, open(out, 'w'))
            print(f"{len(rows)}/{len(jobs)} chi={r['chi']} {r['arm']} rdm2={r['rdm2']:.2e} nn={r['nn_rms']:.2e} nnn={r['nnn_rms']:.2e} tr={r['trace']:.4f} {r['wall']}s {int(time.time() - t0)}s", flush=True)
