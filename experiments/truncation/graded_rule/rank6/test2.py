"""Test 2 (minimal): spcf bond tilt vs env-gauged SVD-LPDO at matched (chi, kappa).
Arms: svd (centre-gauged Kraus truncation, plain bond SVD), spcf (a=2, fw=0, iters=4, taus=[1.0], ks=(1,2,3)), gated (same + gate=1: fire only if B_res >= c0 (tail/1e-4)^0.65).
    python test2.py nproc  -> results/test2_raw.jsonl (resumable)
PRE-REGISTERED (plan, Test 2):
  SUCCESS (both): nn rms >= 2x lower than SVD at two ADJACENT chi in a gamma >= 0.01 cell, with trace norm <= 1.10x SVD's and far-ZZ rms (dist 3, 4) <= 1.5x SVD's;
                  and the gamma = 0.01 gain >= half of the gamma = 0 gain (gain = SVD nn rms / spcf nn rms).
  KILL (either):  gain < 1.3 in every gamma >= 0.01 cell;  OR trace norm > 1.25x SVD's wherever the gain is >= 1.3.
"""
import _p6  # noqa: F401
import json
import os
import sys
import time
import multiprocessing as mp
import numpy as np
import lpdo_bench as B
import spcflpdo as S
import mpsenh as M

MODEL, N, T, DT = 'ising', int(os.environ.get('T2_N', 8)), float(os.environ.get('T2_T', 4.0)), 0.1
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
GAMMAS = [0.0, 0.01, 0.03, 0.1]
CHIS = [int(x) for x in os.environ.get('T2_CHIS', '6,8,12').split(',')]
KAP = 4
OUT = 'results/test2_raw.jsonl' if N == 8 else f'results/test2_N{N}_T{T:g}_raw.jsonl'


def key(j):
    return f'{j[0]}_g{j[1]:g}_chi{j[2]}_k{j[3]}'


def run(j):
    arm, g, chi, kap = j
    t = time.time()
    if arm == 'svd':
        r = B.cell(MODEL, N, T, DT, g, chi, kap)
    else:
        cut = S.SPCFLpdo(MODEL, N, **BEST, gate=(1.0 if arm == 'gated' else 0.0))
        r = B.cell_cut(MODEL, N, T, DT, g, chi, kap, cut)
    r['arm'] = arm
    r['job'] = key(j)
    r['cpu'] = time.time() - t
    return r


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    gam = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else GAMMAS
    arms = sys.argv[3].split(',') if len(sys.argv) > 3 else ['svd', 'spcf', 'gated']
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['job'] for l in open(OUT)}
    for g in gam:
        B.reference(MODEL, N, T, DT, g)
    jobs = [(a, g, c, KAP) for g in gam for c in CHIS for a in arms if key((a, g, c, KAP)) not in done]
    print(len(jobs), 'jobs', flush=True)
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(run, jobs):
            with open(OUT, 'a') as f:
                f.write(json.dumps(r, default=B._tojson) + '\n')
            print(f"{r['job']:22s} nn={r['nn_rms']:.3e} nnn={r['nnn_rms']:.2e} E={r['E_abs']:.2e} tn={r['trace_norm']:.3e} far3={r['far3_rms']:.2e} fired={r.get('fired', '-')}/{r.get('calls', '-')} ({r['cpu']:.0f}s)", flush=True)
