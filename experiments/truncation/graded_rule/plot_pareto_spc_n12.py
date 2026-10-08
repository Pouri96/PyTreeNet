"""Pareto plots of the spc cut against plain SVD for Ising N=12, T=4, dt=0.1: error vs stored parameters and error vs runtime.

    python plot_pareto_spc_n12.py
Same four panels, rings and ladder as plot_pareto_n12.py, restricted to the two methods. Reads results/pareto_n12/svd_*.json and
spc_*.json (one process per chi, JAX compile cache warm) and, for the runtime panels, results/pareto_n12_steady/spc_steady.json
(second run of the cell in one process, jit caches kept, from time_spc_ladder.py). Also prints the matched-chi ratios and, for every spc point, the smallest SVD chi
whose error is not larger (a cross-chi match, no interpolation), and the compile overhead of the cold pass.
"""
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

STYLE = {
    'svd': ('SVD per gate (peak chi)', 'tab:gray', 'o'),
    'spc': ('spc cut: SVD + first-order marginal restoration (peak chi)', 'tab:blue', '^'),
}
METRICS = [('infid', 'final-state infidelity'), ('single_rms', 'one-site (n) rms error'),
           ('nn_rms', 'nn correlator rms error'), ('nnn_rms', 'nnn correlator rms error')]
TIMES = {'wall': 'wall-clock seconds (shared 8-core machine)', 'cpu': 'process CPU seconds (all threads)'}
STEADY_LABEL = 'spc cut, steady state (second run in one process)'

by = {}
for k in STYLE:
    for f in glob.glob(f'results/pareto_n12/{k}_*.json'):
        by.setdefault(k, []).extend(json.load(open(f)))
    by[k].sort(key=lambda r: r['chi'])
steady = {r['chi']: r for r in json.load(open('results/pareto_n12_steady/spc_steady.json'))}
cold = {}
for f in glob.glob('results/pareto_n12_cold/spc_*.json'):
    for r in json.load(open(f)):
        cold[r['chi']] = r


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
        allp = []
        for k, (lab, col, mrk) in STYLE.items():
            xs = [r[xkey] for r in by[k]]
            ys = [r[mk] for r in by[k]]
            ax.plot(xs, ys, '-', color=col, marker=mrk, ms=6, lw=1.2, label=lab)
            allp += list(zip(xs, ys))
            if k == 'spc' and xkey in TIMES:
                xs = [steady[r['chi']][xkey] for r in by[k]]
                ax.plot(xs, ys, '--', color='tab:green', marker='v', ms=6, lw=1.2, label=STEADY_LABEL)
                allp += list(zip(xs, ys))
        for i in front(allp):
            ax.plot(*allp[i], 'o', mfc='none', mec='k', ms=13, mew=1.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ml)
        ax.grid(alpha=0.3, which='both')
    axs[0, 0].legend(fontsize=7.5, loc='best')
    chis = ', '.join(str(r['chi']) for r in by['svd'])
    fig.suptitle(f'Ising chain N=12, T=4, dt=0.1. Each point is one bond dimension chi in {{{chis}}}. Rings = overall Pareto front',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=160)
    plt.close(fig)


figure('params', 'stored parameters of the final MPS', 'figs/pareto_n12_params_svd_spc.png')
for t, lab in TIMES.items():
    figure(t, lab, f'figs/pareto_n12_runtime_{t}_svd_spc.png')

sv = {r['chi']: r for r in by['svd']}
sp = {r['chi']: r for r in by['spc']}
print('matched chi, spc / svd (lower is better), then smallest svd chi with error <= spc error')
print('chi params | ' + ' | '.join(f'{m:>22}' for m, _ in METRICS) + ' | wall svd/spc/steady/cold (s) | cpu spc/steady (s)')
for c in sorted(sp):
    if c not in sv:
        continue
    cells = []
    for m, _ in METRICS:
        match = [d for d in sorted(sv) if sv[d][m] <= sp[c][m]]
        cells.append(f'{sp[c][m] / sv[c][m]:5.2f} (svd chi {match[0] if match else ">16"})')
    cw = cold[c]['wall'] if c in cold else float('nan')
    print(f"{c:3d} {sp[c]['params']:5d} | " + ' | '.join(f'{x:>22}' for x in cells)
          + f" | {sv[c]['wall']:5.1f} {sp[c]['wall']:6.1f} {steady[c]['wall']:6.1f} {cw:6.1f} | {sp[c]['cpu']:6.1f} {steady[c]['cpu']:6.1f}")
