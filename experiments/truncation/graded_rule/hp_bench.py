"""High-pressure chain TEBD bench: one dense reference per cell (cached), many arms, matched bond dimension.

    python hp_bench.py ising 20 8.0 0.1 svd,enh:0.1:4:0:0:1:0 16,32,48 out.json 6

Arms are ``svd``, ``enh:kappa:gamma:wE:eps_gate:ndir:gate_rel`` and the energy-only forms of
``mps_bench.py``. Besides the five usual metrics each row has ``rdm2`` and ``rdm3``, the root-mean-square
Frobenius error of every 2-site and every 3-site reduced density matrix, the quantities the rule targets.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
import numpy as np
import multiprocessing as mp
import _paths  # noqa: F401  (rule/ onto sys.path)
import mpsenh as M

HERE = Path(__file__).resolve().parent
REF = HERE / '_refcache'


def reference(model, N, T, dt):
    path = REF / f'{model}_N{N}_T{T}_dt{dt}.npy'
    if path.exists():
        return np.load(path)
    REF.mkdir(exist_ok=True)
    nsteps = int(round(T / dt))
    v = M.dense_reference(model, N, nsteps, dt, gates=M.make_gates(model, N, dt))
    np.save(path, v)
    return v


def rdmk(v, N, i, k):
    t = v.reshape(2 ** i, 2 ** k, -1)
    return np.einsum('asb,atb->st', t, t.conj())


def marginal_errors(ex, ap, N):
    ex, ap = ex / np.linalg.norm(ex), ap / np.linalg.norm(ap)
    out = {}
    for k, key in ((2, 'rdm2'), (3, 'rdm3')):
        e = [np.linalg.norm(rdmk(ex, N, i, k) - rdmk(ap, N, i, k)) ** 2 for i in range(N - k + 1)]
        out[key] = float(np.sqrt(np.mean(e)))
    return out


def make_cut(arm, model, N):
    if arm == 'svd':
        return M.svd_cut
    p = arm.split(':')
    if arm.startswith('match'):
        # match:window:gamma:kappa:iters   e.g. match:r2:1:inf:4
        import matchcut
        return matchcut.MatchCut(model, N, window=p[1], gamma=float(p[2]), kappa=float(p[3]), iters=int(p[4]))
    if arm.startswith('pinstep') or arm.startswith('pincut'):
        # inner window rule optional:  pinstep  |  pinstep:kappa:gamma:wE:ndir  (EnhCut without environments)
        inner = None
        if len(p) > 1:
            inner = M.EnhCut(model, N, kappa=float(p[1]), gamma=float(p[2]), wE=float(p[3]), ndir=int(p[4]))
        return M.PinCut(model, N, inner=inner, mode='step' if arm.startswith('pinstep') else 'cut')
    if arm.startswith('enhG') or arm.startswith('enhg'):
        return M.EnhCut(model, N, kappa=float(p[1]), gamma=4.0, wE=float(p[2]), energy='global',
                        energy_only=arm.startswith('enhG'), refine=4)
    if arm.startswith('enhb') or arm.startswith('enhe'):
        return M.EnhCut(model, N, kappa=float(p[1]), gamma=4.0, wE=float(p[2]),
                        window='bond' if arm.startswith('enhb') else 'r2', energy_only=True, ndir=1)
    return M.EnhCut(model, N, kappa=float(p[1]), gamma=float(p[2]), wE=float(p[3]),
                    eps_gate=float(p[4]) if len(p) > 4 else 0.0,
                    ndir=int(p[5]) if len(p) > 5 else 1,
                    gate_rel=float(p[6]) if len(p) > 6 else 0.0)


def one(args):
    model, N, T, dt, arm, chi = args
    nsteps = int(round(T / dt))
    ex = reference(model, N, T, dt)
    cut = make_cut(arm, model, N)
    Tm, wall = M.run_tebd(model, N, chi, nsteps, dt, cut, gates=M.make_gates(model, N, dt))
    ap = M.mps_to_dense(Tm)
    e = M.errors(ex, ap, N, model)
    e.update(marginal_errors(ex, ap, N))
    e.update(arm=arm, chi=chi, params=M.stored_params(Tm), wall=round(wall, 2))
    if arm != 'svd':
        e.update(fired=cut.fired, truncating=cut.truncating)
    return e


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    arms, chis = sys.argv[5].split(','), [int(c) for c in sys.argv[6].split(',')]
    out, nproc = sys.argv[7], int(sys.argv[8]) if len(sys.argv) > 8 else 4
    reference(model, N, T, dt)
    jobs = [(model, N, T, dt, a, c) for c in chis for a in arms]
    res, t0 = [], time.time()
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"{len(res)}/{len(jobs)} chi={r['chi']} {r['arm']} infid={r['infid']:.3e} rdm2={r['rdm2']:.2e} "
                  f"nn={r['nn_rms']:.2e} E={r['E_abs']:.2e} {int(time.time() - t0)}s", flush=True)
