"""mu-dial table and figure from the Test 1 summary.   python mu_dial.py results/test1_summary.json"""
import sys, json
import numpy as np
S = json.load(open(sys.argv[1]))
mus = ['inf', '2.0', '1.0', '0.5', '0.25']
def get(fam, T, g, mu, arm='spcf'):
    for s in S:
        if s['family'] == fam and s['T'] == T and s['gauge'] == g and s['mu'] == mu and s['arm'] == arm:
            return s
print('median over the usable chi window of SVD_best / spcf (ungated); R_best = against the better of the two SVD gauges')
for g in ('plain', 'back'):
    print(f'\nspcf in the {g} gauge   (cells: family T)   rdm2 ratio / nn ratio  [window chis]')
    print('mu'.ljust(6) + ''.join(f'{f} T{T:g}'.rjust(22) for f in ('stag', 'dw') for T in (3.0, 4.0)))
    for mu in mus:
        line = mu.ljust(6)
        for f in ('stag', 'dw'):
            for T in (3.0, 4.0):
                s = get(f, T, g, mu)
                line += (f"{s['med']['R_rdm2']:.2f}/{s['med']['R_nn']:.2f}".rjust(22) if s else '-'.rjust(22))
        print(line)
if len(sys.argv) > 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {('stag', 3.0): '#2a78d6', ('stag', 4.0): '#eb6834', ('dw', 3.0): '#1baf7a', ('dw', 4.0): '#eda100'}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    x = np.arange(len(mus))
    for ax, g in zip(axs, ('plain', 'back')):
        key = 'R_rdm2' if g == 'plain' else 'L_rdm2'
        for (f, T), c in col.items():
            ys = [get(f, T, g, mu)['med'][key] if get(f, T, g, mu) else np.nan for mu in mus]
            ax.plot(x, ys, '-o', color=c, lw=2, ms=6, mec='white', mew=1.5, label=f"{'staggered' if f == 'stag' else 'domain wall'}, T={T:g}")
        ax.axhline(2, color='#52514e', lw=1, ls='--'); ax.text(-0.15, 2.08, 'success threshold 2', fontsize=8, color='#52514e', ha='left', va='bottom', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
        ax.axhline(1, color='#52514e', lw=1, ls=':')
        ax.set_xticks(x); ax.set_xticklabels([r'$\infty$', '2', '1', '0.5', '0.25'])
        ax.set_xlabel(r'tilt $\mu$ (pure $\leftarrow$ mixed)'); ax.set_yscale('log'); ax.set_ylim(0.8, 8); ax.set_yticks([1, 2, 3, 4, 6]); ax.set_yticklabels(['1', '2', '3', '4', '6']); ax.minorticks_off()
        ax.set_title('spcf in the plain gauge, against SVD in the better gauge' if g == 'plain' else 'spcf in the back gauge, against SVD in the back gauge (like for like)', fontsize=9, loc='left')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.grid(axis='y', color='#e5e4e0', lw=0.6)
    axs[0].set_ylabel('median rdm2 ratio  SVD / spcf')
    axs[0].legend(frameon=False, fontsize=8, loc='lower left', ncol=2)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)
    print('saved', sys.argv[2])
