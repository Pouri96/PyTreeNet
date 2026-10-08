"""Purification-MPS TEBD bench: svd and spcf arms, plain and backward ancilla gauge, product tilted-Bell-pair initial states.

    python pur_bench.py CELLS ARMS CHIS OUT.json [nproc] [N] [model]

CELLS  comma list of  family:mu:gauge:T   (family stag|dw, mu a float or inf, gauge plain|back), e.g. stag:1:back:3,dw:inf:plain:4
ARMS   comma list of  svd | spcf[:a:fw:iters:taus:ks:fmin:every:eps:pattern:rel] | spcfg:<gate>[:a:...]   (defaults as pareto_spcf.py: 2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2)
CHIS   comma list of bond-dimension caps
Every row has the errors against the dense purification reference: the physical single / nn / nnn rms, rdm2, rdm3, |dE|, far collateral
(ZZ at distance 3 and 4), the physical trace distance 0.5 ||rho - rho_ex||_1, lambda_min(rho), the purification infidelity (gauge dependent)
and the stored parameters.  Rows are appended to OUT.json as they finish.
"""
import sys
import _p2  # noqa: F401
import json, time
import multiprocessing as mp
import numpy as np
import purlib as P
from spcfpur import SPCFPur

DEFAULT_OPT = '2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2'
_REFS = {}


def parse_cell(s):
    f, mu, g, T = s.split(':')
    return f, (np.inf if mu == 'inf' else float(mu)), g, float(T)


def make_cut(arm, model, N):
    p = arm.split(':')
    if p[0] == 'svd':
        return P.svd_cut_d
    gate = 0.0
    if p[0] == 'spcfg':
        gate = float(p[1])
        p = ['spcf'] + p[2:]
    opt = DEFAULT_OPT.split(':')
    for i, x in enumerate(p[1:]):
        opt[i] = x
    return SPCFPur(model, N, a=int(opt[0]), fw=float(opt[1]), iters=int(opt[2]), taus=[float(x) for x in opt[3].split('-')],
                   ks=tuple(int(x) for x in opt[4].split('-')), f_min=float(opt[5]), every=int(opt[6]), eps_min=float(opt[7]),
                   pattern=opt[8], rel_skip=float(opt[9]), gate=gate)


def get_ref(model, N, f, mu, g, T):
    key = (model, N, f, mu, g, T)
    if key not in _REFS:
        if len(_REFS) >= 4:
            _REFS.pop(next(iter(_REFS)))
        _REFS[key] = P.Reference(model, N, f, mu, g, T)
    return _REFS[key]


def one(job):
    model, N, f, mu, g, T, arm, chi, dt = job
    ref = get_ref(model, N, f, mu, g, T)
    cut = make_cut(arm, model, N)
    T0 = P.initial_pur_mps(N, f, mu)
    Gs = P.pur_gates(model, N, dt, g)
    c0 = time.process_time()
    Tm, wall = P.run_tebd_d(T0, Gs, chi, int(round(T / dt)), cut)
    cpu = time.process_time() - c0
    e = P.pur_metrics(ref, Tm)
    e.update(model=model, N=N, family=f, mu=('inf' if np.isinf(mu) else mu), gauge=g, T=T, arm=arm, chi=chi, params=P.stored_params(Tm),
             maxrank=max(P.ranks(Tm)), wall=round(wall, 2), cpu=round(cpu, 2))
    if arm != 'svd':
        e.update(fired=cut.fired, calls=cut.calls, skipped=cut.skipped)
    return e


def run(cells, arms, chis, out, nproc=3, N=10, model='ising', dt=0.1, quiet=False):
    jobs = [(model, N, f, mu, g, T, a, c, dt) for (f, mu, g, T) in cells for c in chis for a in arms]
    # make sure every reference exists before forking (one process per trajectory)
    for key in sorted({(f, mu, g) for (f, mu, g, T) in cells}, key=str):
        P.reference_pur(model, N, key[0], key[1], key[2], cells[0][3], dt, Ts=tuple(sorted({c[3] for c in cells})))
    res, t0 = [], time.time()
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            res.append(r)
            json.dump(res, open(out, 'w'))
            if not quiet:
                print(f"{len(res)}/{len(jobs)} {r['family']}:{r['mu']}:{r['gauge']}:{r['T']:g} chi={r['chi']} {r['arm']} rdm2={r['rdm2']:.2e} nn={r['nn_rms']:.2e} "
                      f"tdist={r['tdist']:.2e} {r['wall']}s {int(time.time() - t0)}s", flush=True)
    return res


if __name__ == '__main__':
    cells = [parse_cell(s) for s in sys.argv[1].split(',')]
    arms = sys.argv[2].split(',')
    chis = [int(c) for c in sys.argv[3].split(',')]
    out = sys.argv[4]
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    N = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    model = sys.argv[7] if len(sys.argv) > 7 else 'ising'
    run(cells, arms, chis, out, nproc, N, model)
