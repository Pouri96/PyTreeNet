"""Equal-error matching of spcf against SVD and against DMT, at every sample time.

    python eqerr_analysis.py          reads results/eqerr_raw_<cell>.json, writes results/eqerr_ratios.json and eqerr_summary.json

At sample time t and for each spcf chi_s: along the baseline B's chi ladder, err_B(chi) is interpolated (log error against log chi, piecewise linear)
and inverted at spcf's error -> chi_eq.  Then
    param_ratio = params_B(chi_eq) / params_spcf(chi_s)          (params_B interpolated in log-log along the ladder, same t)
    time_ratio  = cpu_spcf(t) / cpu_B(chi_eq, t)                 (cumulative CPU up to t, interpolated likewise)
    product     = time_ratio / param_ratio                       (= cpu*params of spcf over cpu*params of B; < 1 means spcf wins on the memory-time product)
No extrapolation: if chi_eq falls outside the ladder it is clamped to the ladder end and flagged
    'above'  (the baseline at its largest chi is still worse than spcf): param_ratio is a LOWER bound, time_ratio an UPPER bound, product an UPPER bound
    'below'  (the baseline at its smallest chi is already as good as spcf): param_ratio is an UPPER bound, time_ratio a LOWER bound, product a LOWER bound
    'exact'  spcf error below EXACT_FLOOR: nothing is being truncated yet, no matching is attempted.
The baseline error is made monotone first (err_env(chi) = min over chi' <= chi of err(chi'): a larger bond cap can always do what a smaller one does).
SVD baseline: the better of the two ancilla gauges at each (chi, t) and metric (the other gauge is the "svd_same" sensitivity: SVD only in spcf's gauge).
"""
import sys
import os
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
CELLS = ['stag_mu1', 'stag_mu0.5', 'stag_mu0.25', 'dw_mu1', 'dw_mu0.5', 'dw_mu0.25']
CHI_S = [12, 16, 24]
METRICS = ['rdm2', 'nn_rms']
EXACT_FLOOR = 1e-9
BASELINES = ['SVD', 'DMT', 'SVD_same']


def load(cell):
    d = json.load(open(os.path.join(RES, f'eqerr_raw_{cell}.json')))
    D = dict(meta={k: d[k] for k in d if k != 'rows'}, spcf={}, svd={'plain': {}, 'back': {}}, dmt={}, noise=[])
    for r in d['rows']:
        if r['rep'] != 0:
            D['noise'].append(r)
            continue
        if r['arm'] == 'spcf':
            D['spcf'][r['chi']] = r['series']
        elif r['arm'] == 'dmt':
            D['dmt'][r['chi']] = r['series']
        else:
            D['svd'][r['arm'].split('_')[1]][r['chi']] = r['series']
    return D


def ladder(D, kind, metric, i):
    """(chi, err, params, cpu) arrays of the baseline at sample index i; err already made monotone."""
    if kind == 'DMT':
        chis = sorted(D['dmt'])
        s = [D['dmt'][c][i] for c in chis]
    else:
        gs = ('plain', 'back') if kind == 'SVD' else (D['meta']['spcf_gauge'],)
        chis = sorted(set.intersection(*[set(D['svd'][g]) for g in gs]))
        s = [min((D['svd'][g][c][i] for g in gs), key=lambda x: x[metric]) for c in chis]
    chi = np.array(chis, float)
    err = np.maximum(np.array([x[metric] for x in s]), 1e-16)
    env = np.minimum.accumulate(err)
    # envelope: if a smaller chi was better, carry that run's cost along with its error
    src = [int(np.argmax(err[:k + 1] == env[k])) for k in range(len(err))]
    par = np.array([s[j]['params'] for j in src], float)
    cpu = np.array([s[j]['cpu'] for j in src], float)
    return chi, env, par, cpu


def match(chi, env, target):
    """chi_eq with env(chi_eq) = target by log-log interpolation; returns (chi_eq, flag)."""
    if env[0] <= target:
        return chi[0], 'below'
    if env[-1] > target:
        return chi[-1], 'above'
    i = int(np.argmax(env <= target))                      # first index at or below the target, i >= 1
    f = (np.log(target) - np.log(env[i - 1])) / (np.log(env[i]) - np.log(env[i - 1]))
    return float(np.exp(np.log(chi[i - 1]) + f * (np.log(chi[i]) - np.log(chi[i - 1])))), 'ok'


def loginterp(x, xs, ys):
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def compute(Ds):
    out = []
    for cell, D in Ds.items():
        for metric in METRICS:
            for kind in BASELINES:
                for chs in CHI_S:
                    sp = D['spcf'][chs]
                    for i, row in enumerate(sp):
                        rec = dict(cell=cell, metric=metric, baseline=kind, chi_s=chs, t=row['t'], err_s=row[metric], params_s=row['params'],
                                   cpu_s=row['cpu'], cpu_setup_s=row['cpu_setup'])
                        if row[metric] < EXACT_FLOOR:
                            rec.update(flag='exact')
                            out.append(rec)
                            continue
                        chi, env, par, cpu = ladder(D, kind, metric, i)
                        ce, flag = match(chi, env, row[metric])
                        pb = loginterp(ce, chi, par)
                        cb = loginterp(ce, chi, cpu)
                        rec.update(flag=flag, chi_eq=ce, params_b=pb, cpu_b=cb, param_ratio=pb / row['params'], time_ratio=row['cpu'] / cb)
                        rec['product'] = rec['time_ratio'] / rec['param_ratio']
                        out.append(rec)
    return out


def med(vals):
    return float(np.median(vals)) if len(vals) else float('nan')


def summarise(recs):
    """Medians across cells for each (metric, baseline, chi_s, t); plus across cells and chi_s."""
    S = {}
    times = sorted({r['t'] for r in recs})
    for metric in METRICS:
        for kind in BASELINES:
            for chs in CHI_S + ['all']:
                for t in times:
                    sel = [r for r in recs if r['metric'] == metric and r['baseline'] == kind and r['t'] == t and r['flag'] != 'exact'
                           and (chs == 'all' or r['chi_s'] == chs)]
                    key = f'{metric}|{kind}|{chs}|{t:g}'
                    S[key] = dict(n=len(sel), n_above=sum(r['flag'] == 'above' for r in sel), n_below=sum(r['flag'] == 'below' for r in sel),
                                  param_ratio=med([r['param_ratio'] for r in sel]), time_ratio=med([r['time_ratio'] for r in sel]),
                                  product=med([r['product'] for r in sel]))
    return S


def noise_report(Ds):
    out = {}
    for cell, D in Ds.items():
        if not D['noise']:
            continue
        for r in D['noise']:
            pass
        groups = {}
        base = {('spcf', 12): D['spcf'][12], ('svd_plain', 24): D['svd']['plain'][24], ('dmt', 24): D['dmt'][24]}
        for r in D['noise']:
            groups.setdefault((r['arm'], r['chi']), []).append(r['series'][-1]['cpu'])
        for k, v in groups.items():
            allv = [base[k][-1]['cpu']] + v
            out[f'{cell}|{k[0]}|{k[1]}'] = dict(cpu_runs=allv, rel_spread=float((max(allv) - min(allv)) / np.mean(allv)))
    return out




# ====================================================================== sentences and tables
def _fmt(x):
    return f'{x:.0f}' if x >= 9.95 else f'{x:.1f}'


def _factor(val, bound, word_hi, word_lo):
    """'about 12x slower' / 'about 1.4x faster'; bound is 'lo' (val is a lower bound), 'hi' (upper bound) or None (inverting val flips the bound)."""
    inv = val < 1
    x = 1 / val if inv else val
    sym = ''
    if bound:
        lower = (bound == 'lo') != inv
        sym = '≥' if lower else '≤'
    return f'about {sym}{_fmt(x)}× {word_lo if inv else word_hi}'


def sentence(r, name):
    """spcf (chi=12) is about X x slower but uses about Y x fewer parameters than SVD  (bounds marked >= / <=; <1 ratios are flipped to faster / more)."""
    if r['flag'] == 'exact':
        return f"spcf (χ={r['chi_s']}) has error {r['err_s']:.0e} (< 1e-9): nothing truncated yet, not matched"
    pb = {'above': 'lo', 'below': 'hi'}.get(r['flag'])
    tb = {'above': 'hi', 'below': 'lo'}.get(r['flag'])
    return (f"spcf (χ={r['chi_s']}) is {_factor(r['time_ratio'], tb, 'slower', 'faster')} but uses "
            f"{_factor(r['param_ratio'], pb, 'fewer', 'more')} parameters than {name}")


def get(recs, **kw):
    return [r for r in recs if all(r[k] == v for k, v in kw.items())]


def fm(v, flagged=0, n=0):
    return f'{v:.2f}' if v < 10 else f'{v:.1f}'


def tables(recs, Ds):
    L = []
    cname = {'stag': 'staggered', 'dw': 'domain wall'}
    # ---- per cell sentences (rdm2)
    for cell in CELLS:
        m = Ds[cell]['meta']
        L.append(f"#### {cname[m['family']]}, mu = {m['mu']:g} (spcf in the {m['spcf_gauge']} gauge)\n")
        L.append('| t | spcf vs SVD (better gauge), error = rdm2 | spcf vs DMT, error = rdm2 |')
        L.append('|---|---|---|')
        for t in (1.0, 2.0, 3.0, 4.0):
            for cs in CHI_S:
                a = get(recs, cell=cell, metric='rdm2', baseline='SVD', chi_s=cs, t=t)[0]
                b = get(recs, cell=cell, metric='rdm2', baseline='DMT', chi_s=cs, t=t)[0]
                L.append(f'| {t:g} | {sentence(a, "SVD")} | {sentence(b, "DMT")} |')
        L.append('')
    # ---- medians across the 6 cells
    for metric in METRICS:
        L.append(f'#### Medians across the 6 cells, error metric = {metric}\n')
        L.append('Entries are `param_ratio / time_ratio / product`; n = cells with a matched entry (not "exact"); `*k` = k of them are ladder-edge bounds.\n')
        L.append('| baseline | t | χ_s = 12 | χ_s = 16 | χ_s = 24 | all χ_s pooled |')
        L.append('|---|---|---|---|---|---|')
        for kind in ('SVD', 'DMT'):
            for t in (1.0, 2.0, 3.0, 4.0, 'pooled 2-4'):
                cells_ = []
                for cs in CHI_S + ['all']:
                    sel = [r for r in recs if r['metric'] == metric and r['baseline'] == kind and r['flag'] != 'exact'
                           and (cs == 'all' or r['chi_s'] == cs) and ((r['t'] == t) if t != 'pooled 2-4' else (r['t'] >= 2.0))]
                    if not sel:
                        cells_.append('n/a')
                        continue
                    nb = sum(r['flag'] != 'ok' for r in sel)
                    cells_.append(f"{fm(med([r['param_ratio'] for r in sel]))} / {fm(med([r['time_ratio'] for r in sel]))} / {fm(med([r['product'] for r in sel]))}"
                                  f" (n={len(sel)}{', *' + str(nb) if nb else ''})")
                L.append(f"| {kind} | {t if t == 'pooled 2-4' else format(t, 'g')} | " + ' | '.join(cells_) + ' |')
        L.append('')
    # ---- trend table over all sample times
    L.append('#### Median over cells and χ_s at every sample time (rdm2): `param_ratio / time_ratio / product`, n matched entries of 18\n')
    L.append('| t | vs SVD | vs DMT |')
    L.append('|---|---|---|')
    for t in sorted({r['t'] for r in recs}):
        row = []
        for kind in ('SVD', 'DMT'):
            sel = [r for r in recs if r['metric'] == 'rdm2' and r['baseline'] == kind and r['flag'] != 'exact' and r['t'] == t]
            row.append(f"{fm(med([r['param_ratio'] for r in sel]))} / {fm(med([r['time_ratio'] for r in sel]))} / {fm(med([r['product'] for r in sel]))} (n={len(sel)})" if sel else 'n/a')
        L.append(f'| {t:g} | ' + ' | '.join(row) + ' |')
    L.append('')
    return '\n'.join(L)


def trend_stats(recs):
    """Per (cell, chi_s, baseline, metric) series: ratio at t = 4 over ratio at t = 2.  Returns text lines."""
    out = {}
    for metric in METRICS:
        for kind in ('SVD', 'DMT', 'SVD_same'):
            g = {q: [] for q in ('param_ratio', 'time_ratio', 'product')}
            for cell in CELLS:
                for cs in CHI_S:
                    a = get(recs, cell=cell, metric=metric, baseline=kind, chi_s=cs, t=2.0)[0]
                    b = get(recs, cell=cell, metric=metric, baseline=kind, chi_s=cs, t=4.0)[0]
                    if a['flag'] == 'exact' or b['flag'] == 'exact':
                        continue
                    for q in g:
                        g[q].append(b[q] / a[q])
            out[f'{metric}|{kind}'] = {q: dict(median_growth=med(v), n_up=int(sum(x > 1 for x in v)), n=len(v)) for q, v in g.items()}
    return out


# ====================================================================== figures
COL = {12: '#2a78d6', 16: '#eb6834', 24: '#1baf7a'}          # fixed categorical slots 1-3 (validated all-pairs set), one colour per chi_s
MRK = {12: 'o', 16: 's', 24: 'D'}
INK, GRID = '#0b0b0b', '#d9d8d3'


def _style(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=20))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.axhline(1.0, color='#52514e', lw=1.0, ls='--', zorder=1)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)


def _pts(recs, cell, metric, kind, cs, q):
    rr = sorted(get(recs, cell=cell, metric=metric, baseline=kind, chi_s=cs), key=lambda r: r['t'])
    rr = [r for r in rr if r['flag'] != 'exact']
    return rr, [r['t'] for r in rr], [r[q] for r in rr]


def fig_cell(recs, Ds, cell, metric='rdm2'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = Ds[cell]['meta']
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey='row')
    for j, (kind, nm) in enumerate((('SVD', 'SVD (better ancilla gauge)'), ('DMT', 'DMT'))):
        for i, (q, lab) in enumerate((('param_ratio', 'parameter ratio  params_B(χ_eq) / params_spcf'), ('time_ratio', 'time ratio  CPU_spcf / CPU_B(χ_eq)'))):
            ax = axs[i, j]
            _style(ax)
            for cs in CHI_S:
                rr, t, y = _pts(recs, cell, metric, kind, cs, q)
                ax.plot(t, y, color=COL[cs], lw=1.8, marker=MRK[cs], ms=5.5, mec='white', mew=0.8, label=f'spcf χ_s = {cs}', zorder=3)
                for r in rr:                                  # ladder-edge bounds: open triangle pointing along the bound
                    if r['flag'] != 'ok':
                        up = (r['flag'] == 'above') == (q == 'param_ratio')
                        ax.plot([r['t']], [r[q]], marker='^' if up else 'v', ms=9, mfc='white', mec=COL[cs], mew=1.6, ls='', zorder=4)
            ax.set_title(f'{lab.split("  ")[0]} vs {nm}', fontsize=10.5, loc='left', color=INK)
            ax.set_ylabel(lab.split('  ')[1], fontsize=9.5)
            if i == 1:
                ax.set_xlabel('time t', fontsize=10)
    axs[0, 0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle(f"Ising N=10, {'staggered' if m['family'] == 'stag' else 'domain wall'}, μ = {m['mu']:g}: cost at equal {metric} error "
                 f"(dashed line = 1; spcf in the {m['spcf_gauge']} gauge)", fontsize=11.5, x=0.01, ha='left')
    fig.text(0.01, 0.005, 'Param ratio > 1: spcf stores fewer numbers.  Time ratio > 1: spcf is slower.  Open triangles: ladder-edge bound, no extrapolation.  '
             'Samples with spcf error < 1e-9 are omitted.', fontsize=8.5, color='#52514e')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fn = os.path.join(HERE, 'figures', f'eqerr_{cell}.png')
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return fn


def fig_summary(recs, metric='rdm2', kinds=('SVD',)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    nm = {'stag': 'stag', 'dw': 'dw'}
    xl = [f"{c.split('_')[0]}\nμ={c.split('mu')[1]}" for c in CELLS]
    for row, t in enumerate((2.0, 4.0)):
        for col, (q, lab) in enumerate((('param_ratio', 'parameter ratio  params_B(\u03c7_eq) / params_spcf  (>1: spcf smaller)'),
                                        ('time_ratio', 'time ratio  CPU_spcf / CPU_B(\u03c7_eq)  (>1: spcf slower)'),
                                        ('product', 'memory-time product  time ratio / param ratio  (>1: spcf worse)'))):
            ax = axs[row, col]
            _style(ax)
            for ci, cell in enumerate(CELLS):
                for k, kind in enumerate(kinds):
                    for l, cs in enumerate(CHI_S):
                        r = get(recs, cell=cell, metric=metric, baseline=kind, chi_s=cs, t=t)[0]
                        if r['flag'] == 'exact':
                            continue
                        x = ci + (0.12 * (l - 1) if len(kinds) == 1 else (-0.27 + 0.09 * l) + 0.30 * k)
                        bound = r['flag'] != 'ok'
                        up = (r['flag'] == 'above') == (q == 'param_ratio') if q != 'product' else r['flag'] == 'below'
                        mk = ('^' if up else 'v') if bound else ('o' if kind == 'SVD' else 's')
                        ax.plot([x], [r[q]], marker=mk, ms=7.5, mfc='white' if bound else COL[cs], mec=COL[cs], mew=1.5 if bound else 0.8, ls='', zorder=3)
            ax.set_xticks(range(len(CELLS)))
            ax.set_xticklabels(xl, fontsize=8.5)
            ax.set_title(f'{lab.split("  ")[0].capitalize()} at t = {t:g}', fontsize=10.5, loc='left')
            ax.set_ylabel(lab.split('  ', 1)[1].replace('  (', '\n('), fontsize=8.5)
            for ci in range(1, len(CELLS)):
                ax.axvline(ci - 0.5, color=GRID, lw=0.6)
    h = [Line2D([], [], marker=MRK[cs], color=COL[cs], ls='', ms=7, label=f'χ_s = {cs}') for cs in CHI_S]
    if len(kinds) > 1:
        h += [Line2D([], [], marker='o', color='#52514e', ls='', ms=7, label='vs SVD (left of each pair)'),
              Line2D([], [], marker='s', color='#52514e', ls='', ms=7, label='vs DMT (right)')]
    h += [Line2D([], [], marker='^', mfc='white', mec='#52514e', ls='', ms=8, label='ladder-edge bound')]
    fig.legend(handles=h, frameon=False, fontsize=9.5, loc='lower center', ncol=6)
    fig.suptitle(f'Equal-{metric}-error cost of spcf relative to {" and ".join(kinds)}, all six cells, t = 2 and t = 4 (dashed line = 1)', fontsize=11.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fn = os.path.join(HERE, 'figures', 'eqerr_summary.png' if len(kinds) == 1 else 'eqerr_summary_svd_dmt.png')
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return fn


if __name__ == '__main__':
    Ds = {c: load(c) for c in CELLS if os.path.exists(os.path.join(RES, f'eqerr_raw_{c}.json'))}
    recs = compute(Ds)
    S = summarise(recs)
    json.dump(recs, open(os.path.join(RES, 'eqerr_ratios.json'), 'w'))
    json.dump(dict(summary=S, noise=noise_report(Ds), trend=trend_stats(recs)), open(os.path.join(RES, 'eqerr_summary.json'), 'w'), indent=1)
    open(os.path.join(RES, 'eqerr_tables.md'), 'w').write(tables(recs, Ds))
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    for c in Ds:
        fig_cell(recs, Ds, c)
    fig_summary(recs)
    fig_summary(recs, kinds=('SVD', 'DMT'))
    print(f'{len(recs)} records; flags:', {f: sum(r['flag'] == f for r in recs) for f in ('ok', 'above', 'below', 'exact')})
