"""Compact tables for the report (T2): per-gate criteria summary, t* of the main arms, nn error of the main arms.  Reads the same npz as t2_analysis.py."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2_analysis as A  # noqa: E402
import rutil  # noqa: E402

R = A.R
TAGS = ['g001', 'g06', 'g03e5', 's001']
NAMES = {'g001': 'g001: rs=1e-2, eps=1e-7, taus=1 (ungated)', 'g06': 'g06: rs=0.6, eps=1e-7, taus=1', 'g03e5': 'g03e5: rs=0.3, eps=1e-5, taus=1', 's001': 's001: rs=1e-2, eps=1e-7, taus=none'}
svd = {c: A.load(R / f't2_svd_{c}.npz') for c in A.SVD_CHIS}
chis = sorted(svd)
spcf = {t: {c: A.load(R / f't2_spcf_{t}_{c}.npz') for c in A.SPCF_CHIS} for t in TAGS}
E = {}
EC = {}
for c in chis:
    d = svd[c]
    E[f'svd{c}'] = A.errs(d['obs'])
    E[f'svdF{c}'] = A.errs(d['F'][:, None] * d['obs'])
    E[f'svdFtrue{c}'] = A.errs((np.abs(d['ov']) ** 2)[:, None] * d['obs'])
    trip = [svd[x] for x in chis if x <= c][-3:]
    E[f'svdlogF{c}'] = A.errs(A.extrap(trip, 'logF'))
    E[f'svdinv{c}'] = A.errs(A.extrap(trip, 'inv'))
    EC[f'svd{c}'] = A.eps_c(d)
for t in TAGS:
    for c, d in spcf[t].items():
        E[f'{t}_{c}'] = A.errs(d['obs'])
        E[f'{t}F_{c}'] = A.errs(d['F'][:, None] * d['obs'])
        EC[f'{t}_{c}'] = A.eps_c(d)

print('### T2 criteria by gate')
print('| gate | (a) cases holding (first-sample t*) | (a) failing cases | (b) win fraction vs best post-processed SVD, min / median over 6 cells | (c) median eps_c ratio spcf/SVD (chi 16 / 24 / 32) | peak r(nn) (chi 16 / 24 / 32) | r(nn) at t=8 (chi 16 / 24 / 32) |')
print('|---|---|---|---|---|---|---|')
for t in TAGS:
    ok, tot, fails, wins, eps, peak, last = 0, 0, [], [], [], [], []
    for c in A.SPCF_CHIS:
        c15 = int(round(1.5 * c))
        for key, name in (('enn', 'nn'), ('e1', '1site')):
            for tol in (1e-2, 1e-3):
                a = A.tstar(E[f'{t}_{c}'][key], tol)
                b = A.tstar(E[f'svd{c15}'][key], tol)
                tot += 1
                if a >= b:
                    ok += 1
                else:
                    fails.append(f'chi{c} {name} {tol:g} ({a:g} vs {b:g})')
        tp = int(np.argmax(E[f'svd{c}']['enn']))
        gp = np.array([i <= tp and E[f'svd{c}']['enn'][i] >= 1e-4 for i in range(A.NS)])
        for key in ('enn', 'e1'):
            pp = np.minimum.reduce([E[f'svd{c}'][key], E[f'svdF{c}'][key], E[f'svdlogF{c}'][key], E[f'svdinv{c}'][key]])
            wins.append(float((E[f'{t}_{c}'][key] < pp)[gp].mean()))
        eps.append(float(np.median((EC[f'{t}_{c}'] / EC[f'svd{c}'])[gp])))
        rr = []
        for i in range(A.NS):
            e = E[f'{t}_{c}']['enn'][i]
            rr.append(np.nan if e < 1e-7 else rutil.chi_eq(chis, [E[f'svd{x}']['enn'][i] for x in chis], e)[0] / c)
        peak.append(np.nanmax(rr))
        last.append(rr[-1])
    print(f"| {NAMES[t]} | {ok}/{tot} | {'; '.join(fails) if fails else '-'} | {min(wins):.2f} / {np.median(wins):.2f} | {' / '.join(f'{x:.2f}' for x in eps)} | {' / '.join(f'{x:.2f}' for x in peak)} | {' / '.join(f'{x:.2f}' for x in last)} |")

print('\n### t*(1e-3) and t*(1e-2) (first sample time with error above tol; inf = never up to T = 8), nn / 1-site')
print('| arm | nn 1e-3 | 1-site 1e-3 | nn 1e-2 | 1-site 1e-2 |')
print('|---|---|---|---|---|')


def trow(name, k):
    return f"| {name} | {A.tstar(E[k]['enn'], 1e-3):g} | {A.tstar(E[k]['e1'], 1e-3):g} | {A.tstar(E[k]['enn'], 1e-2):g} | {A.tstar(E[k]['e1'], 1e-2):g} |"


for c in chis:
    print(trow(f'SVD chi={c}', f'svd{c}'))
for c in A.SPCF_CHIS:
    for t in TAGS:
        print(trow(f'spcf {t} chi={c}', f'{t}_{c}'))
    print(trow(f'SVD x F_MPS chi={c}', f'svdF{c}'))
    print(trow(f'SVD logF-extrap (<=3 largest chi<={c})', f'svdlogF{c}'))
    print(trow(f'SVD 1/chi-extrap (<=3 largest chi<={c})', f'svdinv{c}'))
top3 = [svd[x] for x in chis][-3:]
for nm, xk in (('SVD logF-extrap top3 (48,64,96)', 'logF'), ('SVD 1/chi-extrap top3 (48,64,96)', 'inv')):
    e = A.errs(A.extrap(top3, xk))
    print(f"| {nm} | {A.tstar(e['enn'], 1e-3):g} | {A.tstar(e['e1'], 1e-3):g} | {A.tstar(e['enn'], 1e-2):g} | {A.tstar(e['e1'], 1e-2):g} |")

print('\n### nn rms error at t = 4.5 / 6.5 / 8.0 (growth phase)')
print('| arm | chi=16 | chi=24 | chi=32 |')
print('|---|---|---|---|')
ti = [list(A.TS).index(x) for x in (4.5, 6.5, 8.0)]
for nm, pat in (('SVD', 'svd{c}'), ('SVD x F_MPS', 'svdF{c}'), ('SVD x F_true (oracle)', 'svdFtrue{c}'), ('SVD logF-extrap', 'svdlogF{c}'), ('SVD 1/chi-extrap', 'svdinv{c}')) + tuple((f'spcf {t}', t + '_{c}') for t in TAGS) + (('spcf g001 x F_MPS', 'g001F_{c}'),):
    print(f'| {nm} | ' + ' | '.join(' / '.join(f"{E[pat.format(c=c)]['enn'][i]:.1e}" for i in ti) for c in A.SPCF_CHIS) + ' |')
