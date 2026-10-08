"""Test 2: equal-parameter head-to-head, spcf on the purification vs DMT on the Pauli MPDO (plus Frobenius MPDO, SVD purification in both gauges).

    python test2_dmt.py family mu T chis spcf_gauge out.json [N] [model]
e.g.  python test2_dmt.py stag 0.5 4 12,16,24,32 plain results/test2_stag_mu0.5_T4.json

Arms per chi (all with 4 chi^2 stored numbers per bulk site):
  svd_plain, svd_back            SVD purification (chi, 4, chi)
  spcf_plain, spcf_back          spcf purification, a = 2, fw = 0, 4 CG iterations, tau = 1, ks = 1-2-3 (as pareto_spcf.py)
  spcfg_plain, spcfg_back        gated spcf (gate 1.0)
  dmt                            DMT MPDO (chi, 4, chi), radius 1 (l = 3), chi' = chi - r_L - r_R (= chi - 8 in the bulk)
  dmt+8                          DMT with chi' = chi (total bond chi + 8): the bracket of the 'chi counts the 8 preserved vectors' accounting
  frob                           Frobenius (plain SVD) MPDO, trace-normalised at the end
"""
import sys
import _p2  # noqa: F401
import json, time
import multiprocessing as mp
import numpy as np
import purlib as P
import dmt_np as D
from spcfpur import SPCFPur
from pur_bench import make_cut

fam, mu, T = sys.argv[1], (np.inf if sys.argv[2] == 'inf' else float(sys.argv[2])), float(sys.argv[3])
chis = [int(c) for c in sys.argv[4].split(',')]
gsp = sys.argv[5]
out = sys.argv[6]
N = int(sys.argv[7]) if len(sys.argv) > 7 else 10
model = sys.argv[8] if len(sys.argv) > 8 else 'ising'
dt = 0.1
nsteps = int(round(T / dt))
ARMS = ['svd_plain', 'svd_back', 'spcf_plain', 'spcf_back', 'spcfg_plain', 'spcfg_back', 'dmt', 'dmt+8', 'frob']


def one(job):
    arm, chi = job
    ref = P.Reference(model, N, fam, mu, 'plain', T)
    t0 = time.time()
    if arm in ('dmt', 'dmt+8', 'frob'):
        cut = D.DMTCut(offset=8 if arm == 'dmt+8' else 0) if arm != 'frob' else D.FrobCut()
        Tm, wall = D.run_tebd_mpdo(D.initial_mpdo(N, fam, mu), D.mpdo_gates(model, N, dt), chi, nsteps, cut)
        e = D.mpdo_metrics(ref, Tm)
        e.update(params=P.stored_params(Tm), maxrank=max(t.shape[2] for t in Tm[:-1]), fired=cut.fired, calls=cut.calls)
    else:
        kind, g = arm.split('_')
        cut = make_cut({'svd': 'svd', 'spcf': 'spcf', 'spcfg': 'spcfg:1.0'}[kind], model, N)
        Tm, wall = P.run_tebd_d(P.initial_pur_mps(N, fam, mu), P.pur_gates(model, N, dt, g), chi, nsteps, cut)
        e = P.pur_metrics(ref, Tm)
        e.update(params=P.stored_params(Tm), maxrank=max(P.ranks(Tm)))
        if kind != 'svd':
            e.update(fired=cut.fired, calls=cut.calls)
    e.update(arm=arm, chi=chi, family=fam, mu=('inf' if np.isinf(mu) else mu), T=T, N=N, wall=round(wall, 2), spcf_gauge=gsp)
    return e


if __name__ == '__main__':
    for g in ('plain', 'back'):
        P.reference_pur(model, N, fam, mu, g, T, dt)
    jobs = [(a, c) for c in sorted(chis, reverse=True) for a in ARMS]
    rows, t0 = [], time.time()
    with mp.Pool(int(__import__('os').environ.get('NPROC', 3))) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            rows.append(r)
            json.dump(rows, open(out, 'w'))
            print(f"{len(rows)}/{len(jobs)} chi={r['chi']:2d} {r['arm']:12s} rdm2={r['rdm2']:.2e} nnn={r['nnn_rms']:.2e} nn={r['nn_rms']:.2e} E={r['E_abs']:.1e} "
                  f"zzfar={r['zzfar']:.1e} tdist={r['tdist']:.2e} lam_min={r['lam_min']:.1e} {r['wall']}s {int(time.time() - t0)}s", flush=True)
