"""PNG figures for Tests 2 and 3."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import op_bench as B

T, dt, N = 6.0, 0.1, 10
# ---- Test 2
recs = json.load(open(HERE / 'results' / 'test2_ising.json'))
R = {(r['step'], r['chi'], r['arm']): r for r in recs}
chis = [4, 8, 16, 32]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (key, name) in zip(axs, (('win1_rms', '1-site marginals'), ('win2_rms', '2-site marginals'), ('far_ge3_rms', 'far pairs d>=3 (not targeted)'))):
    for st, ls in ((20, '-'), (40, '-'), (30, ':'), (50, ':'), (60, ':')):
        y = [R[(st, c, 'spcop')][key] / R[(st, c, 'svd')][key] for c in chis]
        ax.plot(chis, y, 'o' + ls, label=f't={st * dt:.0f}' + (' (registered)' if st in (20, 40) else ''), lw=2 if st in (20, 40) else 1)
    ax.axhline(0.5, color='g', lw=0.8)
    ax.axhline(0.8, color='r', lw=0.8)
    ax.set_xscale('log', base=2)
    ax.set_title(name)
    ax.set_xlabel('chi')
axs[0].set_ylabel('error ratio spcop / svd (one sweep)')
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / 'figures' / 'test2_static_ratio.png', dpi=130)

# ---- Test 3
ref = B.get_ref('ising', N, T, dt)
recs = []
for fn in ('test3_ising.json', 'test3_ising_gate.json', 'test3_ising_rw.json', 'test3_ising_var.json'):
    p = HERE / 'results' / fn
    if p.exists():
        recs += json.load(open(p))
M = {(r['arm'], r['chi']): B.metrics(r, ref) for r in recs}
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
arms = [('spcop', 'C0'), ('rw:1.6', 'C2')]
for ax, reg in zip(axs, ('front', 'behind')):
    for arm, col in arms:
        ch = sorted(c for (a, c) in M if a == arm)
        ax.plot(ch, [M[(arm, c)]['dC_Z'][reg]['rms'] / M[('svd', c)]['dC_Z'][reg]['rms'] for c in ch], 'o-', color=col, label=f'{arm}: dC_Z')
        ax.plot(ch, [M[(arm, c)]['dw'][reg]['rms'] / M[('svd', c)]['dw'][reg]['rms'] for c in ch], 's--', color=col, label=f'{arm}: dw', alpha=0.7)
    ax.axhline(0.5, color='g', lw=0.8)
    ax.axhline(0.75, color='r', lw=0.8)
    ax.axhline(1.0, color='k', lw=0.5)
    ax.set_xscale('log', base=2)
    ax.set_title(f'{reg} of the front, Ising N=10 T=6')
    ax.set_xlabel('chi')
axs[0].set_ylabel('rms error ratio to svd')
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / 'figures' / 'test3_ising_ratio_vs_chi.png', dpi=130)
print('saved')
