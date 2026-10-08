"""One TEBD trajectory for the practicality study: time/error series, full metrics at the end, and peak-memory measurement.

    from traj import run_traj;  run_traj(cell, arm, chi, mode)

cell   one of eqerr_time.CELLS (stag_mu1 ... dw_mu0.25); Ising, N = 10, dt = 0.1, T = 4; spcf runs in the gauge Test 2 used.
arm    svd_plain | svd_back | dmt | spcf-a<A> (unchunked SPCFPur, window A) | spcfc-a<A>-b<BLK> (SPCFPurChunk, block BLK of the outer index)
         optional suffixes: '-f64' selects precision float64 for the spcf cut (default float32, as in all earlier tests), '-wf' (chunked only) recomputes
         the reference state W0 in complex64 (SPCFPurChunk w0_fast)
mode   'time'  samples every 0.5 time units, scored against the dense reference (lean metrics), cumulative CPU (TEBD only), params;
               full metrics (trace distance, infidelity, ...) at the final time.      Used for accuracy and CPU time.
       'rss'   one segment 0 -> T, no scoring; peak extra resident memory = VmHWM(after) - VmRSS(before), VmHWM reset just before the run
               (/proc/self/clear_refs).  Must run in a fresh process (see run_fresh).
       'tmem'  the same run under tracemalloc started before the initial state is built: peak of the Python/numpy arrays allocated during the
               run (MPS + method-constant tables + transients).  LAPACK workspaces are not traced by tracemalloc, so this undercounts SVD
               more than spcf; 'rss' is the primary peak-memory measure, 'tmem' gives the decomposition.
"""
import os
import sys
import time
import json
import pickle
import _p2  # noqa: F401
import numpy as np
import purlib as P
import dmt_np as D
import eqerr_time as E
from pur_bench import make_cut

MODEL, N, DT, TEND = E.MODEL, E.N, E.DT, E.TEND
NSTEPS = int(round(TEND / DT))


def make_arm(arm, cell):
    """-> (cut, gauge or None, kind)"""
    fam, mu, sg = E.CELLS[cell]
    if arm == 'dmt':
        return D.DMTCut(), None, 'dmt'
    if arm.startswith('svd_'):
        return P.svd_cut_d, arm.split('_')[1], 'svd'
    parts = arm.split('-')
    kind = parts[0]
    a = int(parts[1][1:])
    prec = 'f64' if 'f64' in parts else 'f32'
    if kind == 'spcf':
        cut = make_cut(f'spcf:{a}', MODEL, N)
        if prec != 'f32':
            from spcfpur import SPCFPur
            cut = SPCFPur(MODEL, N, a=a, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), f_min=0.0, every=1, eps_min=1e-7, pattern='all', rel_skip=1e-2,
                          precision=prec)
    elif kind == 'spcfc':
        from spcfpur_chunk import SPCFPurChunk
        blk = int([x for x in parts if x.startswith('b')][0][1:])
        cut = SPCFPurChunk(MODEL, N, a=a, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), f_min=0.0, every=1, eps_min=1e-7, pattern='all', rel_skip=1e-2,
                           precision=prec, blk=blk, w0_fast=('wf' in parts))
    else:
        raise ValueError(arm)
    return cut, sg, 'spcf'


def table_bytes(cut):
    """bytes of the state independent F tables held by an spcf cut (float64 originals + float32 copies + sparse static rows)."""
    tot, seen = 0, set()
    for key, (Dm, (Fs, Fd), iu, dg) in getattr(cut, '_shape', {}).items():
        tot += Fs.data.nbytes + Fs.indices.nbytes + Fs.indptr.nbytes + iu[0].nbytes * 2 + dg.nbytes
        for arr in (Fd,):
            if id(arr) not in seen:
                seen.add(id(arr))
                tot += arr.nbytes
    for v in getattr(cut, '_f32', {}).values():
        if id(v) not in seen:
            seen.add(id(v))
            tot += v.nbytes
    return tot


def _init(cell, arm):
    fam, mu, sg = E.CELLS[cell]
    cut, gauge, kind = make_arm(arm, cell)
    if kind == 'dmt':
        Ss = D.mpdo_gates(MODEL, N, DT)
        Tm = D.initial_mpdo(N, fam, mu)
        return cut, kind, Ss, Tm
    Gs = P.pur_gates(MODEL, N, DT, gauge if gauge else sg)
    Tm = P.initial_pur_mps(N, fam, mu)
    return cut, kind, Gs, Tm


def _advance(kind, Tm, G, chi, nsteps, cut):
    if kind == 'dmt':
        return D.run_tebd_mpdo(Tm, G, chi, nsteps, cut)
    return P.run_tebd_d(Tm, G, chi, nsteps, cut)


def _vm(key):
    for l in open('/proc/self/status'):
        if l.startswith(key):
            return int(l.split()[1]) * 1024
    return 0


_REF = {}


def full_metrics(cell, kind, Tm, gauge):
    fam, mu, sg = E.CELLS[cell]
    if cell not in _REF:
        _REF.clear()
        _REF[cell] = P.Reference(MODEL, N, fam, mu, 'plain', TEND)
    ref = _REF[cell]
    if kind == 'dmt':
        return D.mpdo_metrics(ref, Tm)
    return P.pur_metrics(ref, Tm)


_LEAN = {}


class _Unpickler(pickle.Unpickler):
    """The lean-reference pickles written by ``python eqerr_time.py`` name the class as __main__.LeanRef; map it to eqerr_time.LeanRef."""
    def find_class(self, module, name):
        if name == 'LeanRef':
            return E.LeanRef
        return super().find_class(module, name)


def load_lean(fam, mu):
    pk = P.REF / f'eqerr_lean_{MODEL}_N{N}_{fam}_mu{P.mu_tag(mu)}_dt{DT}.pkl'
    if pk.exists():
        with open(pk, 'rb') as fh:
            return _Unpickler(fh).load()
    return E.lean_refs(fam, mu)


def prebuild_tables(cut):
    """Build the state independent F tables of every window shape the run will use (and their float32 copies), as the first cut would."""
    if not hasattr(cut, '_region'):
        return
    for b in range(N - 1):
        lo, hi = max(0, b - cut.aL), min(N, b + 2 + cut.aR)
        Dm, (Fs, Fd64), iu, dg = cut._region(hi - lo, hi == N, b - lo)
        if id(Fd64) not in cut._f32:
            cut._f32[id(Fd64)] = np.ascontiguousarray(Fd64.astype(cut.fdt))


def run_traj(cell, arm, chi, mode='time'):
    fam, mu, sg = E.CELLS[cell]
    asbuilt = mode.endswith('A')                       # 'rssA' / 'tmemA': the F tables are built during the run (their build transient counts)
    mkind = mode[:-1] if asbuilt else mode
    if mkind == 'tmem' and asbuilt:
        import tracemalloc
        tracemalloc.start()
    cut, kind, G, Tm = _init(cell, arm)
    setup = [0.0]
    if kind == 'spcf':
        orig = cut._region

        def timed_region(*a, **k):
            n0, c0 = len(cut._shape), time.process_time()
            out = orig(*a, **k)
            if len(cut._shape) > n0:
                setup[0] += time.process_time() - c0
            return out
        cut._region = timed_region
    if mkind in ('rss', 'tmem'):
        if kind == 'spcf' and not asbuilt:
            prebuild_tables(cut)                       # tables resident before the measurement window: reported separately (tables_bytes)
        if mkind == 'rss':
            import gc
            import ctypes
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)               # give freed heap pages back, so the RSS before the run is the live set
            except Exception:
                pass
            open('/proc/self/clear_refs', 'w').write('5')            # reset VmHWM to the current RSS
            rss0 = _vm('VmRSS')
        else:
            import tracemalloc
            if not asbuilt:
                tracemalloc.start()                    # tables were built before tracing started
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
        c0 = time.process_time()
        Tm, _ = _advance(kind, Tm, G, chi, NSTEPS, cut)
        cpu = time.process_time() - c0
        out = dict(cell=cell, arm=arm, chi=chi, mode=mode, cpu=cpu, params=P.stored_params(Tm), mps_bytes=P.stored_params(Tm) * 16,
                   maxbond=int(max(x.shape[2] for x in Tm[:-1])), tables_bytes=table_bytes(cut))
        if mkind == 'rss':
            out.update(rss_before=rss0, hwm_after=_vm('VmHWM'), peak_delta=_vm('VmHWM') - rss0)
        else:
            cur, peak = tracemalloc.get_traced_memory()
            out.update(tm_peak=peak, tm_base=base, tm_cur_end=cur)
            tracemalloc.stop()
        if kind == 'spcf':
            out.update(fired=cut.fired, calls=cut.calls, skipped=cut.skipped)
        return out
    # ---- mode 'time'
    if cell not in _LEAN:
        _LEAN.clear()
        _LEAN[cell] = load_lean(fam, mu)
    refs = _LEAN[cell]
    cpu, series = 0.0, []
    for t in E.TIMES:
        c0 = time.process_time()
        Tm, _ = _advance(kind, Tm, G, chi, E.STEPS_PER_SAMPLE, cut)
        cpu += time.process_time() - c0
        if kind == 'dmt':
            rho = D.vec_to_rho(D.mpdo_to_vec(Tm), N)
            rho = 0.5 * (rho + rho.conj().T)
            rho = rho / np.trace(rho).real
        else:
            psi = P.mps_to_dense_d(Tm)
            rho = P.psi_to_rho(psi / np.linalg.norm(psi), N)
        e = E.score(refs[t], rho)
        e.update(t=t, cpu=cpu, params=P.stored_params(Tm), maxbond=int(max(x.shape[2] for x in Tm[:-1])))
        if kind == 'spcf':
            lg = np.array(cut.log) if cut.log else np.zeros((0, 7))
            ratio = lg[:, 1] / lg[:, 0] if len(lg) else np.zeros(0)
            e.update(cpu_setup=setup[0], fired=cut.fired, calls=cut.calls, skipped=cut.skipped,
                     fc_over_fsvd_med=float(np.median(ratio)) if len(ratio) else None, fc_over_fsvd_mean=float(np.mean(ratio)) if len(ratio) else None,
                     tm={k: round(v, 3) for k, v in cut.tm.items()}, tables_bytes=table_bytes(cut))
        series.append(e)
    out = dict(cell=cell, arm=arm, chi=chi, mode='time', series=series)
    out['full'] = full_metrics(cell, kind, Tm, G)
    if kind == 'spcf':
        lg = np.array(cut.log)
        out['fc_over_fsvd'] = (lg[:, 1] / lg[:, 0]).tolist() if len(lg) else []
        out['achosen'] = {str(k): v for k, v in cut.achosen.items()}
    return out


def _job(j):
    return run_traj(*j)


def run_fresh(jobs, nproc=3, timeout=None):
    """Run memory jobs, each in a fresh process (spawn, one task per child), results in job order."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(nproc, maxtasksperchild=1) as pool:
        return pool.map(_job, jobs, chunksize=1)


def check_env():
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == '1', f'set {v}=1'
