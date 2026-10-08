"""Per-cut time and transient peak memory of the purification spcf cut, unchunked and chunked, against the SVD cut (setup of cost_scaling.py).

    OMP_NUM_THREADS=1 ... python cost_chunk.py --chis 16,32,48,64,96,128 --variants svd,a1,a1c1,a1c8,a1c64,a2c1,a2c8,a2c64 --what time,tmem,rss --out results/cost_chunk.json

One fired cut at the centre bond (N = 16, so that the outer bonds of the window are bulk bonds at chi) of a random purification MPS with a TEBD-like
spectrum, truncation 4 chi -> chi, options of Tests 1 and 2 (a window, iters = 4, tau = 1, ks = 1-2-3, every skip rule off), float32 working precision.
Variants: svd | a<A> (SPCFPur, unchunked) | a<A>c<BLK> (SPCFPurChunk with blocks of BLK values of the outer index o'); a trailing 'f' (a2c8f) sets w0_fast.
Every measurement runs in a fresh spawned process.
  time   median (and min) wall seconds of REPS calls after one warm-up call (the warm-up builds the state independent F tables)
  tmem   tracemalloc peak of the arrays allocated during one call (inputs excluded; LAPACK-internal workspaces are not traced)
  rss    VmHWM growth during one call (VmHWM reset just before; F tables built before; inputs excluded); includes untraced workspaces
"""
import os
import sys
import json
import time
import gc
import argparse
import multiprocessing as mp
import _p2  # noqa: F401
import numpy as np
import purlib as P
from cost_scaling import rand_mps, theta_with_spectrum

N, D4 = 16, 4
OPT = dict(fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), f_min=0.0, every=1, eps_min=0.0, pattern='all', rel_skip=0.0)


def parse_variant(v):
    if v == 'svd':
        return None, None
    a = int(v[1])
    blk = int(v.split('c')[1].rstrip('f')) if 'c' in v else None
    return a, blk


def build(v, chi):
    a, blk = parse_variant(v)
    rng = np.random.default_rng(chi)
    T, c = rand_mps(N, D4, chi, rng)
    l, r = T[c].shape[0], T[c + 1].shape[2]
    th = theta_with_spectrum(l, D4, r, rng)
    if a is None:
        return (lambda: P.svd_cut_d(th, chi, 'R')), None, dict(l=l, r=r, theta_MB=th.nbytes / 2 ** 20)
    from spcfpur import SPCFPur
    if blk is None:
        cut = SPCFPur('ising', N, a=a, **OPT)
    else:
        from spcfpur_chunk import SPCFPurChunk
        cut = SPCFPurChunk('ising', N, a=a, blk=blk, w0_fast=v.endswith('f'), **OPT)
    cut.start(T)
    return (lambda: cut(th, chi, 'R', None, None, c)), cut, dict(l=l, r=r, theta_MB=th.nbytes / 2 ** 20)


def _vm(key):
    for ln in open('/proc/self/status'):
        if ln.startswith(key):
            return int(ln.split()[1]) * 1024
    return 0


def measure(job):
    v, chi, what, reps = job
    f, cut, info = build(v, chi)
    out = dict(variant=v, chi=chi, what=what, **info)
    f0 = cut.fired if cut else 0
    if what == 'time':
        f()                                             # warm-up (builds F tables)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        out.update(t_med=float(np.median(ts)), t_min=float(np.min(ts)), reps=reps)
        if cut:
            out.update(fired=cut.fired - f0 - 1, tm={k: round(x, 3) for k, x in cut.tm.items()}, nmv=cut.nmv)
        return out
    if what == 'tmem':
        f()                                             # warm-up: F tables, BLAS threads, imports
        gc.collect()
        import tracemalloc
        tracemalloc.start()
        f()
        out['peak'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        # rss: no warm-up call (the freed heap of a warm-up would hide the transient).  The state independent F tables are built first, BLAS/LAPACK are
        # initialised with a small call, the freed heap is returned to the OS (malloc_trim) and the process runs with a fixed mmap threshold
        # (MALLOC_MMAP_THRESHOLD_, set by the driver) so large blocks are returned on free; then VmHWM is reset and one call is made.
        if cut is not None:
            from spcfpur import SPCFPur
            L = 2 + 2 * cut.a
            Dm, (Fs, Fd64), iu, dg = cut._region(L, False, 0)
            cut._f32[id(Fd64)] = np.ascontiguousarray(Fd64.astype(cut.fdt))
        small = np.random.default_rng(0).normal(size=(64, 64)) + 0j
        np.linalg.svd(small)
        small @ small
        del small
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass
        open('/proc/self/clear_refs', 'w').write('5')
        r0 = _vm('VmRSS')
        f()
        out['peak'] = _vm('VmHWM') - r0
        out['rss0'] = r0
    out['tables_bytes'] = (sum(Fd.nbytes + Fs.data.nbytes + Fs.indices.nbytes + Fs.indptr.nbytes for (_, (Fs, Fd), _, _) in cut._shape.values()) +
                           sum(x.nbytes for x in cut._f32.values())) if cut else 0
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--chis', default='16,32,48,64,96,128')
    ap.add_argument('--variants', default='svd,a1,a1c1,a1c8,a1c64,a2c1,a2c8,a2c64')
    ap.add_argument('--what', default='time,tmem,rss')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--nproc', type=int, default=3)
    ap.add_argument('--skip', default='', help='comma list of variant:chi to skip')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    for vv in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(vv) == '1', vv
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    if a.what == 'rss':                                       # inherited by the spawned workers; time measurements use the default allocator settings
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
    done = {(r['variant'], r['chi'], r['what']) for r in rows}
    skip = {tuple(s.split(':')) for s in a.skip.split(',') if s}
    jobs = []
    for chi in [int(x) for x in a.chis.split(',')]:
        for v in a.variants.split(','):
            if (v, str(chi)) in skip:
                continue
            for w in a.what.split(','):
                if (v, chi, w) not in done:
                    jobs.append((v, chi, w, a.reps if chi < 96 else max(1, min(a.reps, 2))))
    jobs.sort(key=lambda j: -(j[1] ** 3) * (6 if j[0].startswith('a2') else 1))
    print(f'{len(jobs)} jobs', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.nproc, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(measure, jobs, chunksize=1):
            rows.append(r)
            json.dump(rows, open(a.out, 'w'), indent=1)
            val = f"{r['t_med'] * 1e3:.1f} ms" if r['what'] == 'time' else f"{r['peak'] / 2**20:.1f} MB"
            print(f"{r['variant']:7s} chi={r['chi']:4d} {r['what']:5s} {val}  [{int(time.time() - t0)}s]", flush=True)
