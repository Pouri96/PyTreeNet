"""Ratio plot: cut-local / SVD error against chi (and SVD infidelity on the top axis), one panel per observable.

    PARDIR=results/pareto_n16_T2.0_dt0.02_fast OUT=figs/ratio_n16_T2_dt002.png python plot_ratio.py
Points where the SVD error is below 1e-11 (numerical floor) are dropped.
"""
import glob, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = []
for f in glob.glob(os.environ['PARDIR'] + '/*.json'):
    rows += json.load(open(f))
S = {r['chi']: r for r in rows if r['arm'] == 'svd'}
L = sorted([r for r in rows if r['arm'].startswith('lcut')], key=lambda r: r['chi'])
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (k, lab) in zip(axs, [('infid', 'infidelity'), ('single_rms', 'one-site (n)'), ('nn_rms', 'nn correlator'), ('nnn_rms', 'nnn correlator')]):
    pts = [(r['chi'], r[k] / S[r['chi']][k], S[r['chi']]['infid']) for r in L if S[r['chi']][k] > 1e-11]
    ax.semilogx([p[2] for p in pts], [p[1] for p in pts], '^-', color='tab:blue')
    ax.axhline(1.0, color='tab:gray', lw=1)
    ax.set_xlabel('SVD infidelity at the same chi (high pressure on the right)')
    ax.invert_xaxis()
    ax.set_title(f'{lab}: cut-local / SVD')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, which='both')
    for c, y, x in pts:
        ax.annotate(str(c), (x, y), fontsize=7, xytext=(2, 4), textcoords='offset points')
fig.suptitle('Error ratio to plain SVD at equal stored parameters (below the grey line = better). Labels are chi. ' + os.environ.get('NOTE', ''), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.environ['OUT'], dpi=150)
