"""Pareto plots at the final time (t = T = 4): final 2-site RDM error against stored parameters and against the CPU time to
reach t = T, for SVD (both ancilla gauges) and spcf, one panel per cell.  Data: results/eqerr_raw_<cell>.json."""
import _p2  # noqa: F401
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CELLS = ['stag_mu1', 'stag_mu0.5', 'stag_mu0.25', 'dw_mu1', 'dw_mu0.5', 'dw_mu0.25']
TITLE = {'stag': 'staggered', 'dw': 'domain wall'}
ARMS = [('svd_plain', 'SVD, plain gauge', '#2a78d6', 'o', '-'),
        ('svd_back', 'SVD, backward gauge', '#1baf7a', 's', '-'),
        ('spcf', 'spcf', '#eb6834', 'D', '-')]
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'


def final_points(rows, arm):
    pts = {}
    for r in rows:
        if r['arm'] != arm:
            continue
        s = r['series'][-1]
        pts.setdefault(r['chi'], []).append((s['params'], s['cpu'], s['rdm2']))
    out = []
    for chi in sorted(pts):
        p = np.array(pts[chi])
        out.append((chi, p[0, 0], float(np.median(p[:, 1])), p[0, 2]))   # params and error are deterministic; CPU = median of repeats
    return out


def style(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.7)
    ax.grid(True, which='minor', color=GRID, lw=0.3, alpha=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    fig, axs = plt.subplots(2, 6, figsize=(19, 7.2))
    for j, cell in enumerate(CELLS):
        d = json.load(open(os.path.join('results', f'eqerr_raw_{cell}.json')))
        fam, mu = cell.split('_mu')
        for i, (xi, xlab) in enumerate(((1, 'stored parameters at t = 4'), (2, 'CPU time to reach t = 4 (s)'))):
            ax = axs[i, j]
            style(ax)
            for arm, lab, col, mk, ls in ARMS:
                P = final_points(d['rows'], arm)
                if not P:
                    continue
                x = [p[xi] for p in P]
                y = [p[3] for p in P]
                ax.plot(x, y, ls=ls, color=col, lw=2, marker=mk, ms=7 if arm == 'spcf' else 5.5,
                        mec='white', mew=1.2, label=lab, zorder=3 if arm == 'spcf' else 2)
                if arm == 'spcf':
                    for (chi, *_), xx, yy in zip(P, x, y):
                        ax.annotate(f'χ={chi}', (xx, yy), textcoords='offset points', xytext=(5, 4), fontsize=7.5, color=INK)
            ax.set_xlabel(xlab, fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel('final 2-site RDM error (rdm2)', fontsize=8.5, color=INK)
            if i == 0:
                ax.set_title(f'{TITLE[fam]}, μ = {mu}', fontsize=10, color=INK, loc='left')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10)
    fig.suptitle('Pareto at the final state (N = 10, T = 4): accuracy vs memory (top) and accuracy vs time to reach it (bottom). '
                 'Lower-left is better.', fontsize=11.5, x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig('figures/pareto_final.png', dpi=140)
    print('ok')


if __name__ == '__main__':
    main()
