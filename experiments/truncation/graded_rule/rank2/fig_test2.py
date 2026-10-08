"""Figure for Test 2: error against chi at equal stored parameters.   python fig_test2.py results/test2_stag_mu0.5_T4.json figures/test2_stag_mu0.5_T4.png"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = json.load(open(sys.argv[1]))
d = {(r['arm'], r['chi']): r for r in rows}
chis = sorted({r['chi'] for r in rows})
g = rows[0]['spcf_gauge']
series = [(f'spcf_{g}', f'spcf, purification ({g})', '#2a78d6', 'o'), ('svd_plain', 'SVD, purification (plain)', '#eb6834', 's'),
          ('svd_back', 'SVD, purification (back)', '#1baf7a', '^'), ('dmt', 'DMT, MPDO (l = 3)', '#eda100', 'D'), ('frob', 'Frobenius, MPDO', '#e87ba4', 'v')]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7))
for ax, (m, lab) in zip(axs, (('rdm2', 'rdm2 error'), ('nnn_rms', 'next-nearest-neighbour correlator rms'), ('zzfar', 'far ZZ (distance 3, 4) rms'))):
    for a, name, c, mk in series:
        ax.plot(chis, [d[(a, x)][m] for x in chis], '-' + mk, color=c, lw=2, ms=6, mec='white', mew=1.2, label=name)
    ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xticks(chis); ax.set_xticklabels([str(x) for x in chis]); ax.minorticks_off()
    ax.set_xlabel('bond dimension chi (4 chi^2 numbers per site)'); ax.set_title(lab, fontsize=9, loc='left')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color='#e5e4e0', lw=0.6)
axs[0].legend(frameon=False, fontsize=7.5, loc='lower left')
r0 = rows[0]
fig.suptitle(f"{'staggered' if r0['family'] == 'stag' else 'domain wall'} tilt mu={r0['mu']}, Ising N={r0['N']}, T={r0['T']:g}", fontsize=10, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
