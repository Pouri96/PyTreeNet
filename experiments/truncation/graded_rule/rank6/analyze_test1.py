"""Evaluate the pre-registered Test 1 criteria (see test1.py) on results/test1_raw.jsonl.   python analyze_test1.py [raw.jsonl]"""
import _p6  # noqa: F401
import json
import sys
import numpy as np

fn = sys.argv[1] if len(sys.argv) > 1 else 'results/test1_raw.jsonl'
R = {}
for line in open(fn):
    r = json.loads(line)
    R[r['job']] = r

GAMMAS = [0.0, 0.003, 0.01, 0.03, 0.1]
CHIS = [6, 8, 12]
KAPS = [2, 4]
KB = 16


def get(kind, g, chi, kap):
    return R.get(f'{kind}_g{g:g}_chi{chi}_k{kap}')


def fgate(r, a='2'):
    return r['gate'][a]['on_frac_all'] if r else float('nan')


rows = []
print('gamma  chi kap |  e_a      e_b      e_c    |  s_b   s_c  | f(a=2) rho_f  med B/cost | f(a=1) rho_f | bond_disc kraus_disc | E_a')
for g in GAMMAS:
    for chi in CHIS:
        for kap in (KAPS if g > 0 else [4]):
            a = get('a', g, chi, kap)
            if a is None:
                continue
            b = get('b', g, chi, KB) if g > 0 else a
            c = get('c', g, 0, kap) if g > 0 else None
            a0 = get('a', 0.0, chi, 4)
            ea = a['nn_rms']
            eb = b['nn_rms'] if b else float('nan')
            ec = c['nn_rms'] if c else (0.0 if g == 0 else float('nan'))
            f2, f20 = fgate(a), fgate(a0)
            f1, f10 = fgate(a, '1'), fgate(a0, '1')
            row = dict(gamma=g, chi=chi, kappa=kap, e_a=ea, e_b=eb, e_c=ec, s_b=eb / ea, s_c=ec / ea, f2=f2, rho_f2=f2 / f20 if f20 > 0 else float('nan'),
                       f1=f1, rho_f1=f1 / f10 if f10 > 0 else float('nan'), med_ratio2=a['gate']['2']['med_ratio'], bond_disc=a['bond_disc'], kraus_disc=a['kraus_disc'],
                       E_a=a['E_abs'], tn_a=a['trace_norm'], n_logged2=a['gate']['2']['n_logged'])
            rows.append(row)
            print(f"{g:<6g} {chi:3d} {kap:2d}  | {ea:.2e} {eb:.2e} {ec:.2e} | {row['s_b']:.2f} {row['s_c']:.2f} | {f2:.3f} {row['rho_f2']:.2f} {row['med_ratio2']:8.2f} | {f1:.3f} {row['rho_f1']:.2f} | {a['bond_disc']:.1e} {a['kraus_disc']:.1e} | {a['E_abs']:.1e}")
json.dump(rows, open('results/test1_table.json', 'w'), indent=1)

# ---- criteria
print('\nSUCCESS test (gamma = 0.01): per kappa, chi values with e_a >= 1e-3, s_b >= 0.5, rho_f >= 0.5')
for kap in KAPS:
    ok = [r['chi'] for r in rows if r['gamma'] == 0.01 and r['kappa'] == kap and r['e_a'] >= 1e-3 and r['s_b'] >= 0.5 and r['rho_f2'] >= 0.5]
    print(f'  kappa={kap}: chi values satisfying both (a=2 gate): {ok}')
    ok1 = [r['chi'] for r in rows if r['gamma'] == 0.01 and r['kappa'] == kap and r['e_a'] >= 1e-3 and r['s_b'] >= 0.5 and r['rho_f1'] >= 0.5]
    print(f'           same with the a=1 gate:                        {ok1}')
print('\nrho_f and s_b versus gamma (cells with e_a >= 1e-3):')
for kap in KAPS:
    for chi in CHIS:
        s = []
        for g in GAMMAS[1:]:
            r = [x for x in rows if x['gamma'] == g and x['chi'] == chi and x['kappa'] == kap]
            if r:
                r = r[0]
                s.append(f"g={g:g}: s_b={r['s_b']:.2f} rho_f={r['rho_f2']:.2f}{'' if r['e_a'] >= 1e-3 else ' (e_a<1e-3)'}")
        print(f'  kappa={kap} chi={chi:2d}: ' + ' | '.join(s))
print('\nKILL test: (K1) rho_f(0.01) < 0.25 at every chi?   (K2) s_c > 0.8 in every usable cell with gamma >= 0.01?')
k1 = all(r['rho_f2'] < 0.25 for r in rows if r['gamma'] == 0.01)
usable = [r for r in rows if r['gamma'] >= 0.01 and r['e_a'] >= 1e-3]
k2 = all(r['s_c'] > 0.8 for r in usable)
print(f'  K1 = {k1};  K2 = {k2}  (usable cells: {len(usable)}; cells with s_c <= 0.8: {[(r["gamma"], r["chi"], r["kappa"], round(r["s_c"], 2)) for r in usable if r["s_c"] <= 0.8]})')
