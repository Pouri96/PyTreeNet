"""PNG figures for Test 1 (and Test 3 if present)."""
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
(HERE / 'figures').mkdir(exist_ok=True)
ref = B.get_ref('ising', N, T, dt)
recs = json.load(open(HERE / 'results' / 'test1_ising.json'))
ms = []
for r in recs:
    ms.append(B.metrics(r, ref))
    if r['arm'] == 'svd':
        ms.append(B.metrics(r, ref, raw=True))
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
col = {'svd': 'C0', 'svdraw': 'C3', 'rw:1.6': 'C2'}
for ax, reg in zip(axs, ('ahead', 'front', 'behind')):
    for arm in ('svd', 'svdraw', 'rw:1.6'):
        pts = sorted([(m['chi'], m['dC_Z'][reg]['rms']) for m in ms if m['arm'] == arm])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=col[arm], label=arm)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title(f'{reg} of the front (by exact C_Z)')
    ax.set_xlabel('chi')
    ax.axhline(0.02, color='k', ls=':', lw=0.8)
axs[0].set_ylabel('rms |dC_Z|, Ising N=10, T=6')
axs[0].legend()
fig.tight_layout()
fig.savefig(HERE / 'figures' / 'test1_ising_dCZ_vs_chi.png', dpi=130)

# C_Z(x, t) at chi = 8: exact vs svd vs svdraw (contours at 0.1 and 0.5)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
chi = 8
rec = [r for r in recs if r['arm'] == 'svd' and r['chi'] == chi][0]
t = np.arange(ref['p1'].shape[0]) * dt
Ce = B.O.C_from_p1(ref['p1'])
Ca = B.O.C_from_p1(np.array(rec['p1']))
Cr = B.O.C_from_p1(np.array(rec['p1']) * np.exp(np.array(rec['logkept']))[:, None, None])
for ax, (name, C) in zip(axs, (('exact', Ce), ('svd (renormalised)', Ca), ('svdraw (unnormalised)', Cr))):
    im = ax.imshow(C.T, origin='lower', aspect='auto', extent=[0, T, -0.5, N - 0.5], vmin=0, vmax=1.6, cmap='viridis')
    for th, ls in ((0.1, '--'), (0.5, '-')):
        ax.contour(t, np.arange(N), Ce.T, levels=[th], colors='w', linestyles=ls, linewidths=0.8)
        ax.contour(t, np.arange(N), C.T, levels=[th], colors='r', linestyles=ls, linewidths=0.8)
    ax.set_title(f'{name}, chi={chi}')
    ax.set_xlabel('t')
axs[0].set_ylabel('site x')
fig.colorbar(im, ax=axs, shrink=0.9, label='C_Z(x,t)')
fig.savefig(HERE / 'figures' / 'test1_ising_CZ_maps_chi8.png', dpi=130, bbox_inches='tight')
print('saved')
