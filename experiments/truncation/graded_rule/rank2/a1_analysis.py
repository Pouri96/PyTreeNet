"""Approach 1 analysis: spcf window a = 1 against a = 2 and SVD, with the pre-registered criteria of rank2_practicality.md (section P1).

    python a1_analysis.py       reads results/eqerr_raw_*.json (SVD both gauges, spcf a = 2), results/test2_*, step2*, scan_svd_* (full metrics of the
                                existing runs), results/a1_traj.json (spcf a = 1, new); writes results/a1_analysis.json, a1_analysis.txt, figures/a1_accuracy.png
"""
import os
import json
import glob
import numpy as np
import _p2  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
CELLS = {'stag_mu1': ('stag', 1.0, 'plain'), 'stag_mu0.5': ('stag', 0.5, 'plain'), 'stag_mu0.25': ('stag', 0.25, 'back'),
         'dw_mu1': ('dw', 1.0, 'plain'), 'dw_mu0.5': ('dw', 0.5, 'plain'), 'dw_mu0.25': ('dw', 0.25, 'back')}
CHIS = (12, 16, 24)
LEAN = ['rdm2', 'nn_rms', 'nnn_rms', 'zzfar', 'E_abs']
FULL = LEAN + ['tdist', 'infid_pur']


def load_existing_full():
    """{(family, mu, kind, gauge, T, chi): row} for the pur_bench / test2 style rows with the full metrics (T = 3, 4)."""
    tab = {}
    for f in sorted(glob.glob(os.path.join(RES, '*.json'))):
        b = os.path.basename(f)
        if not (b.startswith('test2_') or b.startswith('step2') or b.startswith('scan_svd')):
            continue
        d = json.load(open(f))
        if not isinstance(d, list):
            continue
        for r in d:
            if not isinstance(r, dict) or 'tdist' not in r:
                continue
            arm = r['arm']
            if arm.startswith('svd_'):
                kind, g = 'svd', arm.split('_')[1]
            elif arm.startswith('spcf_'):
                kind, g = 'spcf', arm.split('_')[1]
            elif arm in ('svd', 'spcf'):
                kind, g = arm, r['gauge']
            else:
                continue
            mu = float(r['mu']) if str(r['mu']) != 'inf' else np.inf
            tab[(r['family'], mu, kind, g, float(r['T']), int(r['chi']))] = r
    return tab


def load_eqerr(cell):
    d = json.load(open(os.path.join(RES, f'eqerr_raw_{cell}.json')))
    out = {}
    for r in d['rows']:
        if r['rep'] != 0:
            continue
        out[(r['arm'], r['chi'])] = {s['t']: s for s in r['series']}
    return out


def main():
    FT = load_existing_full()
    new = json.load(open(os.path.join(RES, 'a1_traj.json')))
    newd = {(r['cell'], r['arm'], r['chi']): r for r in new}
    ent = []                                            # one record per (cell, chi, T)
    checks = []
    for cell, (fam, mu, sg) in CELLS.items():
        og = 'back' if sg == 'plain' else 'plain'
        Ed = load_eqerr(cell)
        for chi in CHIS:
            for T in (4.0, 3.0):
                rec = dict(cell=cell, chi=chi, T=T, family=fam, mu=mu, gauge=sg)
                srcs = {'svd': Ed[(f'svd_{sg}', chi)][T], 'svd_other': Ed[(f'svd_{og}', chi)][T], 'spcf2': Ed[('spcf', chi)][T],
                        'spcf1': newd[(cell, 'spcf-a1', chi)]['series'][[s['t'] for s in newd[(cell, 'spcf-a1', chi)]['series']].index(T)]}
                for k, s in srcs.items():
                    rec[k] = {m: s[m] for m in LEAN}
                    rec[k]['cpu'] = s['cpu']
                if T == 4.0:
                    full = {'svd': FT.get((fam, mu, 'svd', sg, T, chi)), 'svd_other': FT.get((fam, mu, 'svd', og, T, chi)),
                            'spcf2': FT.get((fam, mu, 'spcf', sg, T, chi)), 'spcf1': newd[(cell, 'spcf-a1', chi)]['full']}
                    if full['spcf2'] is None and (cell, 'spcf-a2', chi) in newd:       # one missing entry, filled by a re-run of the existing a = 2 configuration
                        full['spcf2'] = newd[(cell, 'spcf-a2', chi)]['full']
                        rec['spcf2_full_from_rerun'] = True
                    for k, f in full.items():
                        if f is not None:
                            rec[k]['tdist'] = f['tdist']
                            rec[k]['infid_pur'] = f['infid_pur']
                            if abs(f['rdm2'] - rec[k]['rdm2']) > 1e-6 * rec[k]['rdm2']:
                                checks.append(f'{cell} chi={chi} {k}: rdm2 {f["rdm2"]:.6e} (full source) vs {rec[k]["rdm2"]:.6e} (eqerr_raw)')
                ent.append(rec)
    # reproduction check of the existing a = 2 configuration
    rep = []
    for (cell, arm, chi), r in newd.items():
        if arm == 'spcf-a2':
            old = load_eqerr(cell)[('spcf', chi)][4.0]['rdm2']
            rep.append(dict(cell=cell, chi=chi, new=r['series'][-1]['rdm2'], old=old, rel_diff=abs(r['series'][-1]['rdm2'] - old) / old))
    # ratios
    for e in ent:
        for a in ('spcf1', 'spcf2'):
            e['R_' + a] = {m: e[a][m] / e['svd'][m] for m in FULL if m in e[a] and m in e['svd']}
        e['R_svd_other'] = {m: e['svd_other'][m] / e['svd'][m] for m in FULL if m in e['svd_other'] and m in e['svd']}
    out = dict(entries=ent, reproduction=rep, source_mismatch=checks)
    txt = []

    def med(T, a, m, cells=None):
        v = [e['R_' + a][m] for e in ent if e['T'] == T and m in e['R_' + a] and (cells is None or e['cell'] in cells)]
        return float(np.median(v)) if v else float('nan'), len(v)

    summ = {}
    for T in (4.0, 3.0):
        ms = FULL if T == 4.0 else LEAN
        summ[T] = {a: {m: med(T, a, m)[0] for m in ms} for a in ('spcf1', 'spcf2')}
        summ[T]['n'] = med(T, 'spcf1', 'rdm2')[1]
        summ[T]['C1'] = summ[T]['spcf1']['rdm2'] <= 1.5 * summ[T]['spcf2']['rdm2']
        summ[T]['C2'] = summ[T]['spcf1']['rdm2'] <= 1 / 1.5
        summ[T]['C3'] = summ[T]['spcf1']['zzfar'] <= 1.5
        summ[T]['pass'] = bool(summ[T]['C1'] and summ[T]['C2'] and summ[T]['C3'])
        summ[T]['frac_of_gain_kept'] = float(np.log(summ[T]['spcf1']['rdm2']) / np.log(summ[T]['spcf2']['rdm2']))
    # per cell
    percell = {}
    for cell in CELLS:
        percell[cell] = {}
        for T in (4.0, 3.0):
            r1 = med(T, 'spcf1', 'rdm2', [cell])[0]
            r2 = med(T, 'spcf2', 'rdm2', [cell])[0]
            z1 = med(T, 'spcf1', 'zzfar', [cell])[0]
            percell[cell][T] = dict(R1=r1, R2=r2, Z1=z1, Z2=med(T, 'spcf2', 'zzfar', [cell])[0], C1=r1 <= 1.5 * r2, C2=r1 <= 1 / 1.5, C3=z1 <= 1.5,
                                    all=bool(r1 <= 1.5 * r2 and r1 <= 1 / 1.5 and z1 <= 1.5))
    out['pooled'] = {str(T): v for T, v in summ.items()}
    out['percell'] = {c: {str(T): v for T, v in d.items()} for c, d in percell.items()}
    # overfit check: fired-cut residual reduction and entries with R_1 > 1
    ov = []
    for e in ent:
        if e['T'] != 4.0:
            continue
        r = newd[(e['cell'], 'spcf-a1', e['chi'])]
        fr = np.array(r['fc_over_fsvd'])
        ov.append(dict(cell=e['cell'], chi=e['chi'], fc_over_fsvd_med=float(np.median(fr)), n_fired=len(fr), R1_rdm2=e['R_spcf1']['rdm2'], R2_rdm2=e['R_spcf2']['rdm2'],
                       overfit=bool(np.median(fr) <= 0.5 and e['R_spcf1']['rdm2'] > 1)))
    out['overfit'] = ov
    # worse-than-SVD counts per metric (a = 1, a = 2) at T = 4
    worse = {}
    for a in ('spcf1', 'spcf2'):
        worse[a] = {m: int(sum(1 for e in ent if e['T'] == 4.0 and m in e['R_' + a] and e['R_' + a][m] > 1)) for m in FULL}
    out['n_worse_than_svd_T4'] = worse
    # time-resolved R_1 over all sample times (is there an overfit that appears at some t?)
    tr = {}
    for (cell, arm, chi), r in newd.items():
        if arm != 'spcf-a1':
            continue
        Ed = load_eqerr(cell)
        sg = CELLS[cell][2]
        tr[f'{cell}|{chi}'] = [(s['t'], s['rdm2'] / Ed[(f'svd_{sg}', chi)][s['t']]['rdm2'], Ed[('spcf', chi)][s['t']]['rdm2'] / Ed[(f'svd_{sg}', chi)][s['t']]['rdm2'])
                               for s in r['series']]
    out['ratio_vs_time'] = tr
    json.dump(out, open(os.path.join(RES, 'a1_analysis.json'), 'w'), indent=1, default=float)

    # text
    L = txt.append
    L('Approach 1: spcf window a = 1 vs a = 2 vs SVD (same gauge), Ising N = 10, chi in {12, 16, 24}, 6 cells (18 entries)')
    L('')
    L('Reproduction of the existing a = 2 configuration (new run vs eqerr_raw): ' + '; '.join(f"{r['cell']} chi={r['chi']} rdm2 {r['new']:.4e} vs {r['old']:.4e} (rel {r['rel_diff']:.1e})" for r in rep))
    if checks:
        L('SOURCE MISMATCHES: ' + '; '.join(checks))
    L('')
    for T in (4.0, 3.0):
        s = summ[T]
        L(f'T = {T:g}: pooled medians over {s["n"]} entries of R = metric(spcf)/metric(SVD same gauge)  [a = 1 | a = 2]')
        for m in (FULL if T == 4.0 else LEAN):
            L(f'   {m:10s} {s["spcf1"][m]:7.3f} | {s["spcf2"][m]:7.3f}')
        L(f'   C1 R1 <= 1.5 R2: {s["C1"]} ({s["spcf1"]["rdm2"]:.3f} vs {1.5 * s["spcf2"]["rdm2"]:.3f});  C2 R1 <= 0.667: {s["C2"]};  C3 zzfar R1 <= 1.5: {s["C3"]} ({s["spcf1"]["zzfar"]:.3f})  ->  {"KEEPS THE GAIN" if s["pass"] else "does not meet the pre-registered bar"}')
        L(f'   fraction of the a = 2 log-gain kept by a = 1: {s["frac_of_gain_kept"]:.2f}')
        L('')
    L('Per cell (median over chi 12, 16, 24), T = 4:  R1(rdm2) R2(rdm2) | zzfar R1 R2 | C1 C2 C3')
    for c, d in percell.items():
        x = d[4.0]
        L(f'   {c:12s} {x["R1"]:6.3f} {x["R2"]:6.3f} | {x["Z1"]:6.3f} {x["Z2"]:6.3f} | {x["C1"]} {x["C2"]} {x["C3"]}  -> {"pass" if x["all"] else "fail"}')
    L(f'   cells passing all three: T = 4: {sum(percell[c][4.0]["all"] for c in percell)} of 6;  T = 3: {sum(percell[c][3.0]["all"] for c in percell)} of 6')
    L('')
    for T in (4.0, 3.0):
        rr = [e['R_spcf1']['rdm2'] / e['R_spcf2']['rdm2'] for e in ent if e['T'] == T]
        L(f'T = {T:g}: per-entry rdm2(a = 1)/rdm2(a = 2): median {np.median(rr):.2f}, min {min(rr):.2f}, max {max(rr):.2f}, entries above 1.5: {sum(x > 1.5 for x in rr)} of {len(rr)}')
        out['pooled'][str(T)]['ratio_a1_over_a2'] = dict(median=float(np.median(rr)), min=float(min(rr)), max=float(max(rr)), n_above_1p5=int(sum(x > 1.5 for x in rr)))
    L('Entries worse than SVD (R > 1) at T = 4, a = 1 | a = 2: ' + ', '.join(f'{m} {worse["spcf1"][m]}|{worse["spcf2"][m]}' for m in FULL))
    L('Overfit check (a = 1, T = 4): median f_c/f_svd over fired cuts, R1(rdm2), R2(rdm2) per entry')
    for o in ov:
        L(f'   {o["cell"]:12s} chi={o["chi"]:2d} f_c/f_svd {o["fc_over_fsvd_med"]:.3f} (n={o["n_fired"]}) R1 {o["R1_rdm2"]:.3f} R2 {o["R2_rdm2"]:.3f}  overfit={o["overfit"]}')
    L(f'   overfit entries: {sum(o["overfit"] for o in ov)} of {len(ov)}')
    L('')
    L('CPU to T = 4 (s), median over the 18 entries: SVD same gauge %.2f, a = 2 %.2f, a = 1 %.2f; ratios a1/SVD %.1f, a2/SVD %.1f, a2/a1 %.1f' % (
        np.median([e['svd']['cpu'] for e in ent if e['T'] == 4.0]), np.median([e['spcf2']['cpu'] for e in ent if e['T'] == 4.0]),
        np.median([e['spcf1']['cpu'] for e in ent if e['T'] == 4.0]),
        np.median([e['spcf1']['cpu'] / e['svd']['cpu'] for e in ent if e['T'] == 4.0]), np.median([e['spcf2']['cpu'] / e['svd']['cpu'] for e in ent if e['T'] == 4.0]),
        np.median([e['spcf2']['cpu'] / e['spcf1']['cpu'] for e in ent if e['T'] == 4.0])))
    open(os.path.join(RES, 'a1_analysis.txt'), 'w').write('\n'.join(txt) + '\n')
    print('\n'.join(txt))

    # figure
    INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    xs, labs = [], []
    k = 0
    for ci, cell in enumerate(CELLS):
        for chi in CHIS:
            e = [x for x in ent if x['cell'] == cell and x['chi'] == chi and x['T'] == 4.0][0]
            ax.plot([k], [e['R_spcf2']['rdm2']], marker='D', color='#eb6834', ms=7, mec='white', mew=1.1, ls='', label='spcf a = 2 (existing)' if k == 0 else None, zorder=3)
            ax.plot([k], [e['R_spcf1']['rdm2']], marker='o', color='#2a78d6', ms=7, mec='white', mew=1.1, ls='', label='spcf a = 1 (new)' if k == 0 else None, zorder=3)
            xs.append(k)
            labs.append(f'{chi}')
            k += 1
        k += 1
    ax.axhline(1.0, color=MUTED, lw=1)
    ax.axhline(1 / 1.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(k - 1.2, 1.04, 'SVD (same gauge)', ha='right', va='bottom', fontsize=8, color=MUTED)
    ax.text(k - 1.2, 0.69, '1.5x better than SVD', ha='right', va='bottom', fontsize=8, color=MUTED)
    ax.set_yscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7.5, color=MUTED)
    names = {'stag_mu1': 'stag mu=1', 'stag_mu0.5': 'stag mu=0.5', 'stag_mu0.25': 'stag mu=0.25', 'dw_mu1': 'dw mu=1', 'dw_mu0.5': 'dw mu=0.5', 'dw_mu0.25': 'dw mu=0.25'}
    for ci, cell in enumerate(CELLS):
        ax.text(ci * 4 + 1, -0.10, names[cell], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8, color=INK)
    ax.set_ylabel('rdm2 error / SVD rdm2 (same gauge), T = 4', fontsize=9, color=INK)
    ax.set_xlabel('chi within each cell', fontsize=9, color=INK, labelpad=20)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(0.02, 0.86), ncol=2)
    ax = axs[1]
    cell_order = list(CELLS)
    for ci, cell in enumerate(cell_order):
        for chi in CHIS:
            tt = out['ratio_vs_time'][f'{cell}|{chi}']
            ax.plot([t for t, _, _ in tt], [r1 / r2 for _, r1, r2 in tt], color='#2a78d6', lw=1, alpha=0.55)
    ax.axhline(1.0, color=MUTED, lw=1)
    ax.axhline(1.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yscale('log')
    ax.set_xlabel('time t', fontsize=9, color=INK)
    ax.set_ylabel('rdm2(a = 1) / rdm2(a = 2)', fontsize=9, color=INK)
    ax.text(0.2, 1.52, '1.5x worse than a = 2', fontsize=8, color=MUTED, va='bottom')
    for a_ in axs:
        for s in ('top', 'right'):
            a_.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            a_.spines[s].set_color(MUTED)
        a_.tick_params(colors=MUTED, labelsize=8)
        a_.grid(True, which='major', color=GRID, lw=0.7)
    ax.set_title('a = 1 relative to a = 2 along the run (18 lines)', fontsize=9.5, loc='left', color=INK)
    axs[0].set_title('Accuracy against same-gauge SVD at the final state (lower is better)', fontsize=9.5, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'a1_accuracy.png'), dpi=140)


if __name__ == '__main__':
    main()
