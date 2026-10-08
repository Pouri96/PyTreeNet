"""Test-1 driver.  python bench.py MODEL N T dt OUT.json NPROC SPEC [SPEC ...]
SPEC:  mpo:ARM:CHI      ARM in svd_raw, svd_renorm, dmt_I, dmt_neel
       pp:EPS[:XCAP]    Pauli propagation, coefficient threshold EPS (optional X/Y-count cap)
       ppf:M[:EPS]      pi_m^{sigma0} folding to M-body after every gate (optional EPS)
       mps:CHI          Schroedinger MPS-TEBD
Error metric: Delta(n) = |<Z_i0>_approx(n dt) - <Z_i0>_dense(n dt)|, Delta_max over n = 0..T/dt, Delta(T)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
import multiprocessing as mp
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import heis_mpo as H
import baselines as B
import refs


def run_job(args):
    model, N, T, dt, spec = args
    nsteps = int(round(T / dt))
    i0 = int(os.environ.get('RANK4_I0', N // 2 - 1))
    ex = refs.dense_traj(model, N, nsteps, dt)[:, i0, 2]
    p = spec.split(':')
    rec = dict(model=model, N=N, T=T, dt=dt, spec=spec, kind=p[0], i0=i0)
    if p[0] == 'mpo':
        arm, chi = p[1], int(p[2])
        cutname = {'svd_raw': 'svd_raw', 'svd_renorm': 'svd_renorm', 'dmt_I': 'dmt', 'dmt_neel': 'dmt'}[arm]
        kind = {'dmt_I': 'I', 'dmt_neel': 'neel'}.get(arm)
        cut = H.make_cut(cutname)
        r = H.run_heis(model, N, i0, chi, nsteps, dt, cut, H.ref_provider(model, N, kind) if kind else None)
        err = np.abs(r['val'] - ex)
        rec.update(arm=arm, chi=chi, params=int(r['params'].max()), maxbond=int(r['maxbond'].max()),
                   norm_T=float(r['norm'][-1]), wall=r['wall'], disc_total=float(r['disc'].sum()))
        rec['val'] = r['val'].tolist()
    elif p[0] == 'pp':
        eps = float(p[1])
        xcap = int(p[2]) if len(p) > 2 else None
        r = B.pp_traj(model, N, nsteps, dt, i0, 3, eps, xcap=xcap)
        err = np.abs(r['val'] - ex)
        rec.update(arm='pp' if xcap is None else f'xspd{xcap}', eps=eps, xcap=xcap, params=r['params'],
                   peak_strings=r['peak_strings'], wall=r['wall'], truncated=r['truncated'], last_n=r['last_n'])
        rec['val'] = r['val'].tolist()
    elif p[0] == 'ppf':
        m = int(p[1])
        eps = float(p[2]) if len(p) > 2 else 0.0
        r = B.pp_traj(model, N, nsteps, dt, i0, 3, eps, fold=m)
        err = np.abs(r['val'] - ex)
        rec.update(arm=f'fold{m}', eps=eps, fold=m, params=r['params'], peak_strings=r['peak_strings'], wall=r['wall'],
                   truncated=r['truncated'], last_n=r['last_n'])
        rec['val'] = r['val'].tolist()
    elif p[0] == 'mps':
        chi = int(p[1])
        r = B.mps_traj(model, N, chi, nsteps, dt, i0)
        err = np.abs(r['val'] - ex)
        rec.update(arm='mps', chi=chi, params=r['params'], wall=r['wall'], maxbond=r['maxbond'])
        rec['val'] = r['val'].tolist()
    else:
        raise ValueError(spec)
    rec['err'] = err.tolist()
    rec['dmax'] = float(np.nanmax(err)) if not np.all(np.isnan(err)) else float('nan')
    rec['dT'] = float(err[-1])
    return rec


if __name__ == '__main__':
    model, N, T, dt, out, nproc = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
    specs = sys.argv[7:]
    jobs = [(model, N, T, dt, s) for s in specs]
    res = []
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"done {r['spec']:<22s} dmax={r['dmax']:.3e} dT={r['dT']:.3e} params={r['params']} wall={r['wall']:.1f}s "
                  f"({len(res)}/{len(jobs)}, {time.time() - t0:.0f}s)", flush=True)
