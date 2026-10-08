"""Pareto plots for Ising N=12 T=4: error vs stored parameters and error vs runtime, one point per chi.

    python plot_pareto_n12.py [wall|cpu] [--svd]   (the two methods only unless --svd adds the plain SVD reference)
Errors: final-state infidelity and rms error of the one-site (n), nearest-neighbour (nn) and next-nearest-neighbour (nnn)
correlators against the exact solution. Rings mark points on the overall Pareto front (no other point is at least as
good in both coordinates). Lines connect the chi ladder of one method.
"""
import glob
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

TIME = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'wall'
SHOW_SVD = '--svd' in sys.argv
STYLE = {   # arm prefix -> (label, colour, marker)
    'svd': ('SVD per gate (reference, peak chi)', 'tab:gray', 'o'),
    'lcut': ('cut-local lookahead (peak chi)', 'tab:blue', '^'),
    'mfcr': ('global fit, working rank 1.5 chi', 'tab:red', 'D'),
}
METRICS = [('infid', 'final-state infidelity'), ('single_rms', 'one-site (n) rms error'),
           ('nn_rms', 'nn correlator rms error'), ('nnn_rms', 'nnn correlator rms error')]

rows = []
import os
DIR = os.environ.get('PARDIR', 'results/pareto_n12')
FIG = os.environ.get('FIGSUF', '')
for f in glob.glob(DIR + '/*.json'):
    rows += json.load(open(f))
by = {}
for r in rows:
    key = r['arm'].split(':')[0]
    by.setdefault(key, []).append(r)
for k in by:
    by[k].sort(key=lambda r: r['chi'])


def front(points):
    """indices of points not dominated in (x, y), both smaller is better"""
    keep = []
    for i, (x, y) in enumerate(points):
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for j, (x2, y2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def figure(xkey, xlabel, fname):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, (mk, ml) in zip(axs.ravel(), METRICS):
        allp, tags = [], []
        for k, (lab, col, mrk) in STYLE.items():
            if k not in by or (k == 'svd' and not SHOW_SVD):
                continue
            xs = [r[xkey] for r in by[k]]
            ys = [max(r[mk], 1e-15) for r in by[k]]
            ax.plot(xs, ys, '-', color=col, marker=mrk, ms=6, lw=1.2, label=lab)
            for r in by[k]:
                allp.append((r[xkey], max(r[mk], 1e-15)))
                tags.append((k, r['chi']))
        for i in front(allp):
            ax.plot(*allp[i], 'o', mfc='none', mec='k', ms=13, mew=1.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ml)
        ax.grid(alpha=0.3, which='both')
    axs[0, 0].legend(fontsize=8, loc='best')
    chis = sorted({r['chi'] for r in rows})
    fig.suptitle(f'Ising chain N={os.environ.get("NSITES", "12")}, T={float(os.environ.get("TFINAL", "4.0")):g}, dt={os.environ.get("DTSTEP", "0.1")}. '
                 f'Each point is one bond dimension chi in {chis}. Rings = overall Pareto front', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=160)
    plt.close(fig)


SUF = ('_with_svd' if SHOW_SVD else '') + FIG
TAG = os.environ.get('FIGTAG', '')
figure('params', 'stored parameters of the final MPS', f'figs/pareto_n{os.environ.get("NSITES", "12")}{TAG}_params{SUF}.png')
tl = {'wall': 'wall-clock seconds (6 jobs sharing 8 cores)', 'cpu': 'process CPU seconds'}[TIME]
figure(TIME, tl, f'figs/pareto_n{os.environ.get("NSITES", "12")}{TAG}_runtime_{TIME}{SUF}.png')
print({k: len(v) for k, v in by.items()})
