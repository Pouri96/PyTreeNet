"""Plot of results/cost_scaling_*.json: per-cut time and peak memory of the purification spcf cut vs the plain SVD cut."""
import _p2  # noqa: F401
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a2 = json.load(open('results/cost_scaling_a2_N16.json')) + json.load(open('results/cost_scaling_a2_N16_chi64.json'))
a1 = json.load(open('results/cost_scaling_a1_N16.json'))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for rows, lab, c in ((a2, 'spcf, window a = 2 (used in Tests 1-2)', '#c0392b'), (a1, 'spcf, window a = 1', '#2471a3')):
    x = [r['chi'] for r in rows]
    ax[0].loglog(x, [r['spcf_med'] * 1e3 for r in rows], 'o-', c=c, label=lab)
    ax[1].semilogx(x, [r['ratio_med'] for r in rows], 'o-', c=c, label=lab)
    ax[2].loglog(x, [r['mem_spcf_MB'] for r in rows], 'o-', c=c, label=lab)
x = [r['chi'] for r in a1]
ax[0].loglog(x, [r['svd_med'] * 1e3 for r in a1], 's--', c='0.3', label='plain SVD cut')
ax[2].loglog(x, [r['mem_svd_MB'] for r in a1], 's--', c='0.3', label='plain SVD cut')
ax[0].set(xlabel='bond dimension chi', ylabel='time per fired cut (ms)', title='Time per cut')
ax[1].set(xlabel='bond dimension chi', ylabel='spcf time / SVD time', title='Overhead per fired cut', yscale='log')
ax[1].axhline(1, c='0.5', lw=0.8)
ax[2].set(xlabel='bond dimension chi', ylabel='peak memory in the cut (MB)', title='Peak memory per cut')
for a in ax:
    a.grid(alpha=0.3, which='both')
ax[0].legend(fontsize=8)
fig.suptitle('Purification spcf cut vs SVD: bulk bonds (N = 16, d = 4), one thread, every cut forced to fire', fontsize=10)
fig.tight_layout()
fig.savefig('figures/cost_scaling.png', dpi=130)
print('ok')
