"""Equal-error cost study, time series: spcf (purification MPS) vs SVD (purification, both ancilla gauges) vs DMT (Pauli MPDO) along the TEBD run.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python eqerr_time.py CELL [--nproc 3] [--noise] [--only arm1,arm2]
CELL  one of stag_mu1 stag_mu0.5 stag_mu0.25 dw_mu1 dw_mu0.5 dw_mu0.25     (Ising, N = 10, dt = 0.1, T = 4)

Each run (arm, chi) is ONE TEBD trajectory, paused every 0.5 time units (5 Strang steps) to score it against the dense reference at that same time.
Per sample: rdm2, nn_rms (+ nnn_rms, zzfar, E_abs as extras), cumulative CPU seconds of the TEBD only (time.process_time, metrics excluded),
stored parameters at that time (sum of tensor sizes) and the maximal bond.  The spcf rows also carry the cumulative CPU spent building the
state-independent F matrices (``cpu_setup``, included in ``cpu``).

Arms: spcf (same configuration and gauge as Test 2: a = 2, fw = 0, 4 CG iterations, tau = 1, ks = 1-2-3; plain gauge at mu = 1, 0.5, back at 0.25),
svd_plain and svd_back (SVD purification, both gauges; the analysis takes the better one per sample), dmt (the 'dmt' arm of test2_dmt).
Dense reference at all sample times: the plain-gauge purification, cached in _refcache/ (psi snapshots in the usual files, the marginals and
observables for the lean scoring in eqerr_lean_*.pkl).  The physical rho does not depend on the gauge, so one reference serves every arm.
"""
import sys
import os
import _p2  # noqa: F401  (sets single-threaded BLAS defaults and sys.path)
import json
import pickle
import time
import argparse
import multiprocessing as mp
import numpy as np
import purlib as P
import dmt_np as D
from pur_bench import make_cut

MODEL, N, DT, TEND = 'ising', 10, 0.1, 4.0
STEPS_PER_SAMPLE = 5                                       # 0.5 time units
TIMES = [0.5 * k for k in range(1, int(round(TEND / 0.5)) + 1)]
CHI_SPCF = [12, 16, 24]
CHI_SVD = [8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96]
CHI_DMT = [12, 16, 20, 24, 32, 40, 48, 64, 80, 96]
# cell -> (family, mu, gauge in which spcf was run in Test 2)
CELLS = {'stag_mu1': ('stag', 1.0, 'plain'), 'stag_mu0.5': ('stag', 0.5, 'plain'), 'stag_mu0.25': ('stag', 0.25, 'back'),
         'dw_mu1': ('dw', 1.0, 'plain'), 'dw_mu0.5': ('dw', 0.5, 'plain'), 'dw_mu0.25': ('dw', 0.25, 'back')}
RESULTS = P.HERE / 'results'


# ------------------------------------------------------------------ dense reference at every sample time
def _psi_path(fam, mu, t):
    return P.REF / f'pur_{MODEL}_N{N}_{fam}_mu{P.mu_tag(mu)}_plain_T{t:g}_dt{DT}.npy'


class LeanRef:
    """What the scoring needs from the dense reference at one time: the marginals and the observables."""
    def __init__(self, S, obs):
        self.S, self.obs, self.N, self.model = S, obs, N, MODEL


def lean_refs(fam, mu):
    """{t: LeanRef} for all sample times; builds and caches the dense snapshots (only the missing ones are written) on first use."""
    P.REF.mkdir(exist_ok=True)
    pk = P.REF / f'eqerr_lean_{MODEL}_N{N}_{fam}_mu{P.mu_tag(mu)}_dt{DT}.pkl'
    if pk.exists():
        return pickle.load(open(pk, 'rb'))
    missing = [t for t in TIMES if not _psi_path(fam, mu, t).exists()]
    if missing:
        snaps = P.dense_snapshots(MODEL, N, fam, mu, 'plain', DT, TIMES)
        for t in missing:
            np.save(_psi_path(fam, mu, t), snaps[t])
        for t in TIMES:                                     # consistency of the recomputed trajectory with the files that already existed
            if t not in missing:
                ov = abs(np.vdot(np.load(_psi_path(fam, mu, t)), snaps[t]))
                assert abs(ov - 1) < 1e-10, (fam, mu, t, ov)
    out = {}
    for t in TIMES:
        psi = np.load(_psi_path(fam, mu, t))
        psi = psi / np.linalg.norm(psi)
        S = P.RDMSet(P.psi_to_rho(psi, N), N)
        out[t] = LeanRef(S, P.obs_from_rdms(S, MODEL))
    pickle.dump(out, open(pk, 'wb'))
    return out


def score(ref, rho):
    """Errors of a physical density matrix against the lean reference (same definitions as purlib.pur_metrics / dmt_np.mpdo_metrics)."""
    S = P.RDMSet(rho, N)
    ob = P.obs_from_rdms(S, MODEL)
    e = {k + '_rms': P._rms(ref.obs[k], ob[k]) for k in ('nn', 'nnn')}
    e['rdm2'] = P.marg_err(ref.S, S, 2)
    e['zzfar'] = float(np.sqrt(np.mean(np.concatenate([(ref.obs['zz3'] - ob['zz3']) ** 2, (ref.obs['zz4'] - ob['zz4']) ** 2]))))
    e['E_abs'] = float(abs(ref.obs['E'] - ob['E']))
    return e


# ------------------------------------------------------------------ one trajectory
_REFS = {}


def one(job):
    cell, arm, chi, rep = job
    fam, mu, sgauge = CELLS[cell]
    if cell not in _REFS:
        _REFS.clear()
        _REFS[cell] = lean_refs(fam, mu)
    refs = _REFS[cell]
    setup = [0.0]
    if arm == 'dmt':
        cut = D.DMTCut()
        Ss = D.mpdo_gates(MODEL, N, DT)
        Tm = D.initial_mpdo(N, fam, mu)
    else:
        kind, gauge = ('spcf', sgauge) if arm == 'spcf' else arm.split('_')
        cut = make_cut(kind, MODEL, N)
        Gs = P.pur_gates(MODEL, N, DT, gauge)
        Tm = P.initial_pur_mps(N, fam, mu)
        if kind == 'spcf':                                  # time the (state independent) F-matrix construction separately
            orig = cut._region

            def timed_region(*a, **k):
                n0, c0 = len(cut._shape), time.process_time()
                out = orig(*a, **k)
                if len(cut._shape) > n0:
                    setup[0] += time.process_time() - c0
                return out
            cut._region = timed_region
    cpu, series = 0.0, []
    for k, t in enumerate(TIMES):
        c0 = time.process_time()
        if arm == 'dmt':
            Tm, _ = D.run_tebd_mpdo(Tm, Ss, chi, STEPS_PER_SAMPLE, cut)
        else:
            Tm, _ = P.run_tebd_d(Tm, Gs, chi, STEPS_PER_SAMPLE, cut)
        cpu += time.process_time() - c0                     # scoring below is NOT part of the cost
        if arm == 'dmt':
            rho = D.vec_to_rho(D.mpdo_to_vec(Tm), N)
            rho = 0.5 * (rho + rho.conj().T)
        else:
            psi = P.mps_to_dense_d(Tm)
            rho = P.psi_to_rho(psi / np.linalg.norm(psi), N)
        e = score(refs[t], rho)
        e.update(t=t, cpu=cpu, params=P.stored_params(Tm), maxbond=int(max(x.shape[2] for x in Tm[:-1])))
        if arm == 'spcf':
            e.update(cpu_setup=setup[0], fired=cut.fired, calls=cut.calls)
        series.append(e)
    return dict(cell=cell, arm=arm, chi=chi, rep=rep, series=series)


def job_list(cell, noise, only):
    jobs = [(cell, 'spcf', c, 0) for c in CHI_SPCF]
    jobs += [(cell, 'dmt', c, 0) for c in CHI_DMT]
    jobs += [(cell, f'svd_{g}', c, 0) for c in CHI_SVD for g in ('plain', 'back')]
    if noise:                                               # repeats of three representative runs, under the same machine load, for the timing-noise estimate
        for r in (1, 2):
            jobs += [(cell, 'spcf', 12, r), (cell, 'svd_plain', 24, r), (cell, 'dmt', 24, r)]
    if only:
        jobs = [j for j in jobs if j[1] in only]
    big = {'spcf': 3, 'dmt': 2, 'svd_plain': 1, 'svd_back': 1}
    return sorted(jobs, key=lambda j: (-big[j[1]], -j[2]))   # longest first for load balance


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cell', choices=sorted(CELLS))
    ap.add_argument('--nproc', type=int, default=3)
    ap.add_argument('--noise', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    assert a.nproc <= 3
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == '1', f'set {v}=1'
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'eqerr_raw_{a.cell}.json'
    fam, mu, sg = CELLS[a.cell]
    lean_refs(fam, mu)                                      # build the reference before forking
    rows = json.load(open(out))['rows'] if out.exists() else []
    done = {(r['arm'], r['chi'], r['rep']) for r in rows}
    jobs = [j for j in job_list(a.cell, a.noise, set(a.only.split(',')) - {''}) if (j[1], j[2], j[3]) not in done]
    print(f'{a.cell}: {len(rows)} rows present, {len(jobs)} jobs to run', flush=True)
    t0 = time.time()
    with mp.Pool(a.nproc) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            rows.append(r)
            json.dump(dict(cell=a.cell, family=fam, mu=mu, spcf_gauge=sg, model=MODEL, N=N, dt=DT, T=TEND, rows=rows), open(out, 'w'))
            s = r['series'][-1]
            print(f"{len(rows)} {r['arm']:9s} chi={r['chi']:3d} rep={r['rep']} rdm2(T)={s['rdm2']:.2e} nn(T)={s['nn_rms']:.2e} cpu={s['cpu']:.1f}s "
                  f"params={s['params']} maxbond={s['maxbond']}  [{int(time.time() - t0)}s]", flush=True)
