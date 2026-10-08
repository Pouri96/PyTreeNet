"""Replicate of the kappa = 8 extension at N = 10 (T = 4, gamma = 0.03, window a = 1):  svd vs spcf at chi in {12, 16, 24}.  -> results/test2_n10k8_raw.jsonl"""
import _p6  # noqa: F401
import json, sys, time
import multiprocessing as mp
import lpdo_bench as B
import spcflpdo as S

MODEL, N, T, DT = 'ising', 10, 4.0, 0.1
BEST1 = dict(a=1, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)


def run(j):
    arm, g, chi, kap = j
    t = time.time()
    r = B.cell(MODEL, N, T, DT, g, chi, kap) if arm == 'svd' else B.cell_cut(MODEL, N, T, DT, g, chi, kap, S.SPCFLpdo(MODEL, N, **BEST1))
    r['arm'], r['job'], r['cpu'] = arm, f'{arm}_g{g:g}_chi{chi}_k{kap}', time.time() - t
    return r


if __name__ == '__main__':
    g = 0.03
    B.reference(MODEL, N, T, DT, g)
    jobs = [(a, g, c, 8) for c in (24, 16, 12) for a in ('spcf', 'svd')]
    with mp.Pool(int(sys.argv[1])) as pool:
        for r in pool.imap_unordered(run, jobs):
            open('results/test2_n10k8_raw.jsonl', 'a').write(json.dumps(r, default=B._tojson) + '\n')
            print(f"{r['job']:22s} nn={r['nn_rms']:.3e} nnn={r['nnn_rms']:.2e} E={r['E_abs']:.2e} tn={r['trace_norm']:.3e} far3={r['far3_rms']:.2e} fired={r.get('fired','-')}/{r.get('calls','-')} kdisc={r['kraus_disc']:.1e} ({r['cpu']:.0f}s)", flush=True)
