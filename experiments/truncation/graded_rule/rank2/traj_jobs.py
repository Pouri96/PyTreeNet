"""Run a list of traj.run_traj jobs on up to 3 single-threaded workers, resumable, rows appended to a JSON.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python traj_jobs.py --cells all --arms spcf-a1 --chis 12,16,24 --mode time --out results/x.json

--mode time uses forked workers (pool reused), --mode rss / tmem uses spawned workers, one task per child process (fresh heap), so the resident-memory
high-water mark of a run is not contaminated by earlier runs.   --extra cell:arm:chi,... adds explicit jobs.
"""
import sys
import os
import json
import time
import argparse
import multiprocessing as mp
import _p2  # noqa: F401
import traj as T
import eqerr_time as E

ap = argparse.ArgumentParser()
ap.add_argument('--cells', default='all')
ap.add_argument('--arms', default='')
ap.add_argument('--chis', default='')
ap.add_argument('--extra', default='')
ap.add_argument('--mode', default='time', choices=['time', 'rss', 'rssA', 'tmem', 'tmemA'])
ap.add_argument('--out', required=True)
ap.add_argument('--nproc', type=int, default=3)
ap.add_argument('--order', default='big_first')


def key(r):
    return (r['cell'], r['arm'], r['chi'], r.get('mode', 'time'))


def cost_guess(j):
    cell, arm, chi, mode = j
    w = {'spcfc': 4, 'spcf': 3, 'dmt': 1.5}.get(arm.split('-')[0], 1)
    a2 = 3 if '-a2' in arm else 1
    return w * a2 * (chi ** 2) / 100.0


if __name__ == '__main__':
    a = ap.parse_args()
    assert a.nproc <= 3
    T.check_env()
    cells = sorted(E.CELLS) if a.cells == 'all' else a.cells.split(',')
    jobs = []
    if a.arms:
        for c in cells:
            for arm in a.arms.split(','):
                for chi in [int(x) for x in a.chis.split(',')]:
                    jobs.append((c, arm, chi, a.mode))
    for x in [s for s in a.extra.split(',') if s]:
        c, arm, chi = x.split(':')
        jobs.append((c, arm, int(chi), a.mode))
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {key(r) for r in rows}
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    if a.order == 'big_first':
        jobs.sort(key=lambda j: -cost_guess(j))
    print(f'{len(rows)} rows present, {len(jobs)} jobs to run', flush=True)
    t0 = time.time()
    if a.mode.startswith('rss'):
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'      # fixed mmap threshold: big blocks are returned to the OS on free, so VmHWM follows the live peak
    ctx = mp.get_context('fork' if a.mode == 'time' else 'spawn')
    with ctx.Pool(a.nproc, maxtasksperchild=None if a.mode == 'time' else 1) as pool:
        for r in pool.imap_unordered(T._job, jobs, chunksize=1):
            rows.append(r)
            json.dump(rows, open(a.out, 'w'))
            if a.mode == 'time':
                s = r['series'][-1]
                print(f"{len(rows)} {r['cell']} {r['arm']} chi={r['chi']} rdm2={s['rdm2']:.3e} nn={s['nn_rms']:.3e} cpu={s['cpu']:.1f}s [{int(time.time() - t0)}s]", flush=True)
            else:
                v = r.get('peak_delta', r.get('tm_peak'))
                print(f"{len(rows)} {r['cell']} {r['arm']} chi={r['chi']} {a.mode} peak={v / 2**20:.2f} MB cpu={r['cpu']:.1f}s [{int(time.time() - t0)}s]", flush=True)
