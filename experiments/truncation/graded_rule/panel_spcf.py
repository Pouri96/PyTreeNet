"""Panel of cells for judging a variant of the numpy spc cut: ratios variant / svd (2-site, 1-site, nn, infid) per cell.

    python panel_spcf.py "arm1" "arm2" ...        (arm strings as in mfc_bench, e.g. spcf:2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2:f32:passes=2)
Cells run in parallel, 3 worker processes. Cell list is PANEL below.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
import multiprocessing as mp
import _paths  # noqa: F401
import mfc_bench

PANEL = [('ising', 12, 4.0, 8), ('ising', 16, 5.0, 16), ('ising', 16, 5.0, 32), ('ising2', 16, 4.0, 12), ('isingdw', 16, 5.0, 12),
         ('isingdw', 16, 5.0, 24), ('heis', 16, 1.5, 24)]
if os.environ.get('PANEL'):
    PANEL = [(c.split(',')[0], int(c.split(',')[1]), float(c.split(',')[2]), int(c.split(',')[3])) for c in os.environ['PANEL'].split(';')]


def job(a):
    m, N, T, chi, arm = a
    return a, mfc_bench.one((m, N, T, 0.1, arm, chi))


if __name__ == '__main__':
    arms = sys.argv[1:]
    jobs = [(m, N, T, chi, arm) for (m, N, T, chi) in PANEL for arm in ['svd'] + arms]
    res = {}
    with mp.Pool(int(os.environ.get('NPROC', 3))) as pool:
        for a, r in pool.imap_unordered(job, jobs):
            res[a] = r
    print('cell                      ' + ' | '.join(f'{a[-30:]:60s}' for a in arms))
    for (m, N, T, chi) in PANEL:
        sv = res[(m, N, T, chi, 'svd')]
        cells = []
        for arm in arms:
            r = res[(m, N, T, chi, arm)]
            cells.append(f"2s {r['rdm2'] / sv['rdm2']:.2f} 1s {r['single_rms'] / sv['single_rms']:.2f} nn {r['nn_rms'] / sv['nn_rms']:.2f} "
                         f"| max 1s {r['single_max'] / sv['single_max']:.2f} nn {r['nn_max'] / sv['nn_max']:.2f} inf {r['infid'] / sv['infid']:.2f}")
        print(f'{m:8s} N={N} T={T} chi={chi:2d} ({sv["infid"]:.0e}) ' + ' | '.join(f'{c:60s}' for c in cells))
    json.dump({str(k): v for k, v in res.items()}, open(os.environ.get('PANEL_OUT', 'results/panel_last.json'), 'w'))
