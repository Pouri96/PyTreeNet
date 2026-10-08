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


if __name__ == '__main__':
    Ds = {c: load(c) for c in CELLS if os.path.exists(os.path.join(RES, f'eqerr_raw_{c}.json'))}
    recs = compute(Ds)
    S = summarise(recs)
    json.dump(recs, open(os.path.join(RES, 'eqerr_ratios.json'), 'w'))
    json.dump(dict(summary=S, noise=noise_report(Ds)), open(os.path.join(RES, 'eqerr_summary.json'), 'w'), indent=1)
    print(f'{len(recs)} records; flags:', {f: sum(r['flag'] == f for r in recs) for f in ('ok', 'above', 'below', 'exact')})
