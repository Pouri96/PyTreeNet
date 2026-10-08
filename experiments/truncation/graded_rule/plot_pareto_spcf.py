"""Pareto plots of the numpy spc cut against plain SVD: error vs stored parameters and error vs runtime, one point per chi.

    python plot_pareto_spcf.py results/pareto_spcf_n12.json "Ising chain N=12, T=4, dt=0.1" n12 [--old]
    python plot_pareto_spcf.py results/pareto_spcf_n16_T5.json "Ising chain N=16, T=5, dt=0.1" n16_T5
Same four panels and rings as plot_pareto_n12.py. Runtime is the steady-state wall and cpu time (minimum over repeated runs in one
process, all caches warm). --old adds the first JAX version of the spc cut (steady-state times from results/pareto_n12_steady).
Also prints, for every spc point, the SVD bond dimension with the same error (log-linear interpolation in chi), the stored
parameters and the steady wall time of that SVD run.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

src, title, tag = sys.argv[1], sys.argv[2], sys.argv[3]
OLD = '--old' in sys.argv
rows = json.load(open(src))
sv = sorted([r for r in rows if r['arm'] == 'svd'], key=lambda r: r['chi'])
sp = sorted([r for r in rows if r['arm'] == 'spcf'], key=lambda r: r['chi'])
old = []
if OLD:
    steady = {r['chi']: r for r in json.load(open('results/pareto_n12_steady/spc_steady.json'))}
    for c in (4, 6, 8, 10, 12, 16):
        r = json.load(open(f'results/pareto_n12/spc_{c}.json'))[0]
        r.update(wall_steady=steady[c]['wall'], cpu_steady=steady[c]['cpu'])
        old.append(r)
METRICS = [('infid', 'final-state infidelity'), ('single_rms', 'one-site (n) rms error'),
           ('nn_rms', 'nn correlator rms error'), ('nnn_rms', 'nnn correlator rms error')]
SERIES = [('SVD per gate (peak chi)', 'tab:gray', 'o', '-', sv), ('spc cut, numpy matrix-free (peak chi)', 'tab:blue', '^', '-', sp)]
if OLD:
    SERIES.append(('spc cut, first version in JAX (peak chi)', 'tab:orange', 'v', '--', old))


def front(points):
    keep = []
    for i, (x, y) in enumerate(points):
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for j, (x2, y2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def figure(xkey, xlabel, fname, series):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, (mk, ml) in zip(axs.ravel(), METRICS):
        allp = []
        for lab, col, mrk, ls, rs in series:
            xs, ys = [r[xkey] for r in rs], [r[mk] for r in rs]
            ax.plot(xs, ys, ls, color=col, marker=mrk, ms=6, lw=1.2, label=lab)
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
    chis = ', '.join(str(r['chi']) for r in sp)
    fig.suptitle(f'{title}. spc points: chi in {{{chis}}}. SVD ladder to chi={sv[-1]["chi"]}. Rings = overall Pareto front', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=160)
    plt.close(fig)


figure('params', 'stored parameters of the final MPS', f'figs/pareto_{tag}_params_svd_spcf.png', SERIES[:2])
figure('wall_steady', 'wall-clock seconds, steady state (min of repeats)', f'figs/pareto_{tag}_runtime_wall_svd_spcf.png', SERIES)
figure('cpu_steady', 'process CPU seconds, steady state (min of repeats)', f'figs/pareto_{tag}_runtime_cpu_svd_spcf.png', SERIES)

# matched chi ratios and the SVD run with the same error
svd_by = {r['chi']: r for r in sv}
chis = np.array([r['chi'] for r in sv], dtype=float)


def equiv(metric, value):
    ys = np.log(np.array([r[metric] for r in sv]))
    y = np.log(value)
    for i in range(len(sv) - 1):
        lo, hi = ys[i], ys[i + 1]
        if lo >= y >= hi:
            f = (lo - y) / (lo - hi)
            ch = chis[i] + f * (chis[i + 1] - chis[i])
            prm = np.exp(np.interp(ch, chis, np.log([r['params'] for r in sv])))
            wl = np.exp(np.interp(ch, chis, np.log([r['wall_steady'] for r in sv])))
            return ch, prm, wl
    return None


print('matched chi: spc / svd error ratios, then the SVD chi with the same error, its params / spc params and its wall / spc wall')
for r in sp:
    s = svd_by.get(r['chi'])
    if not s:
        continue
    line = f"chi={r['chi']:3d} params={r['params']:6d} | infid {r['infid'] / s['infid']:.2f} 1site {r['single_rms'] / s['single_rms']:.2f} " \
           f"rdm2 {r['rdm2'] / s['rdm2']:.2f} nn {r['nn_rms'] / s['nn_rms']:.2f} | cpu spc {r['cpu_steady']:.2f}s svd {s['cpu_steady']:.2f}s (x{r['cpu_steady'] / s['cpu_steady']:.1f})"
    for m, nm in (('single_rms', '1site'), ('rdm2', 'rdm2')):
        e = equiv(m, r[m])
        line += f" | {nm}: svd chi {e[0]:.0f} params x{e[1] / r['params']:.2f} wall x{e[2] / r['wall_steady']:.2f}" if e else f' | {nm}: beyond ladder'
    print(line)
