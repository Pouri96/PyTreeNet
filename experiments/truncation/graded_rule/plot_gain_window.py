"""Gain window of the spc cut: ratio spc / SVD at equal chi against the SVD error, without and with the absolute tail floor.

    python plot_gain_window.py
Left to right the SVD error falls with chi. Without the floor the ratio rises through 1 once the SVD error is below about 1e-5
(1-site) and the spc cut ends above the SVD floor. With the floor 1e-7 on the discarded weight of a cut, the cut defers to SVD there.
"""
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = [('single_rms', 'one-site (n) rms error'), ('rdm2', 'two-site RDM error'), ('nn_rms', 'nn correlator rms error')]


def split(rows, spc_arm):
    sv = {r['chi']: r for r in rows if r['arm'] == 'svd'}
    sp = {r['chi']: r for r in rows if spc_arm(r['arm'])}
    return sv, sp


def load_n12_unguarded():
    rows = json.load(open('results/pareto_spcf_n12.json')) + json.load(open('results/pareto_spcf_n12_hi.json'))
    return split(rows, lambda a: a == 'spcf')


def load_n16_unguarded():
    rows = json.load(open('results/pareto_spcf_n16_T5.json'))
    for c in (48, 64, 80, 96):
        rows += [r for r in json.load(open(f'results/spcf_floor_n16_{c}.json')) if r['arm'] == 'svd' or r['arm'].split(':')[8] == '1e-10']
    return split(rows, lambda a: a == 'spcf' or a.startswith('spcf:'))


def load_guarded(f):
    return split(json.load(open(f)), lambda a: a == 'spcf')


CELLS = [('Ising N=12, T=4', load_n12_unguarded(), load_guarded('results/pareto_spcf_n12_floor.json')),
         ('Ising N=16, T=5', load_n16_unguarded(), load_guarded('results/pareto_spcf_n16_T5_floor.json'))]
fig, axs = plt.subplots(2, 3, figsize=(12.5, 7))
for ci, (name, (svu, spu), (svg, spg)) in enumerate(CELLS):
    for mi, (mk, ml) in enumerate(METRICS):
        ax = axs[ci, mi]
        for lab, col, mrk, sv, sp in (('no floor (every cut with discarded weight > 1e-10)', 'tab:red', 'o', svu, spu),
                                      ('floor 1e-7 on the discarded weight of a cut', 'tab:blue', '^', svg, spg)):
            cs = sorted(c for c in sp if c in sv)
            ax.plot([sv[c][mk] for c in cs], [sp[c][mk] / sv[c][mk] for c in cs], '-', color=col, marker=mrk, ms=5, lw=1.1, label=lab)
        ax.axhline(1.0, color='k', lw=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_xlabel(f'SVD {ml} at the same chi (falls with chi to the right)', fontsize=8)
        ax.set_ylabel('spc / SVD', fontsize=8)
        ax.set_title(f'{name}, {ml}', fontsize=9)
        ax.grid(alpha=0.3, which='both')
axs[0, 0].legend(fontsize=7, loc='upper left')
fig.suptitle('spc cut against plain SVD at equal bond dimension. Below 1 the spc cut is more accurate, above 1 SVD is.', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig('figs/gain_window_spcf.png', dpi=160)
print('saved figs/gain_window_spcf.png')
